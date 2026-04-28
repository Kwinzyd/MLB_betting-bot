"""
Time-based scheduler for the MLB bot.

Runs as a long-lived `python main.py schedule` process. Every 60s it checks
the wall clock against a declarative schedule and dispatches due jobs.

Schedule (local time):
    08:00              `run`     — morning sync + first scan
    11:00 … 22:30      `run`     — every 30min during the MLB pregame/in-game window
    02:00 (next day)   `settle`  — post-game settlement

Design notes:
  - Single-process, sequential dispatch. A long-running job blocks other jobs
    until it finishes — we'd rather skip a tick than fight over state.
  - In-memory "last run" state. A scheduler restart loses it, meaning the
    same job can fire twice in a day if you restart at an inopportune
    minute. Acceptable: pipelines are idempotent (UNIQUE keys on inserts).
  - Crashes inside a job are caught and Telegram-alerted via the same
    _notify_crash pathway main.py uses. The loop keeps going.
"""

from __future__ import annotations

import asyncio
from datetime import datetime
from typing import Awaitable, Callable, List, Tuple

from src.utils.logging_utils import get_logger

logger = get_logger(__name__)

# Tick granularity — fine enough to hit scheduled minute slots, coarse
# enough to avoid CPU churn. Jobs never need sub-minute precision.
TICK_SECONDS = 60


def morning_run_minutes() -> List[Tuple[int, int]]:
    """(hour, minute) pairs that should trigger `run`."""
    slots: List[Tuple[int, int]] = [(8, 0)]
    for hour in range(11, 23):
        slots.append((hour, 0))
        slots.append((hour, 30))
    return slots


def settle_minutes() -> List[Tuple[int, int]]:
    return [(2, 0)]


def should_run_at(command: str, now: datetime) -> bool:
    """Pure time-based predicate. No side effects; easy to unit-test."""
    hm = (now.hour, now.minute)
    if command == "run":
        return hm in morning_run_minutes()
    if command == "settle":
        return hm in settle_minutes()
    return False


async def _dispatch(command: str) -> None:
    """Import pipelines lazily so a missing optional dep never blocks startup."""
    from src.pipelines.sync_events import sync_events
    from src.pipelines.sync_injuries import sync_injuries
    from src.pipelines.sync_lineups import sync_lineups
    from src.pipelines.sync_umpires import sync_umpires
    from src.pipelines.sync_stats import sync_stats
    from src.pipelines.scan_props import scan_props
    from src.pipelines.send_alerts import send_alerts
    from src.pipelines.find_sgp import find_and_alert_sgps
    from src.pipelines.settle_results import settle_results

    if command == "run":
        await sync_events()
        await sync_injuries()
        await sync_lineups()
        sync_umpires()
        await scan_props(force=False)
        await send_alerts()
        await find_and_alert_sgps()
    elif command == "settle":
        await sync_stats()
        settle_results()
    else:
        raise ValueError(f"Unknown scheduled command: {command}")


async def _tick(
    now: datetime,
    last_fire: dict,
    dispatch: Callable[[str], Awaitable[None]],
) -> None:
    """Fire any jobs whose schedule matches `now`, deduped per minute.

    `last_fire` tracks the last (hour, minute) a job fired to prevent
    double-firing within the same tick window if the tick happens to
    span the minute boundary.
    """
    hm = (now.hour, now.minute)
    for command in ("run", "settle"):
        if not should_run_at(command, now):
            continue
        if last_fire.get(command) == hm:
            continue
        last_fire[command] = hm
        logger.info(f"Scheduler: firing '{command}' at {now.isoformat(timespec='seconds')}")
        try:
            await dispatch(command)
            logger.info(f"Scheduler: '{command}' completed")
        except Exception as e:
            logger.error(f"Scheduler: '{command}' failed: {e}", exc_info=True)
            _notify_job_crash(command, e)


def _notify_job_crash(command: str, exception: Exception) -> None:
    """Best-effort Telegram alert when a scheduled job crashes. Scheduler
    keeps running regardless."""
    try:
        import traceback
        from src.clients.telegram_bot import TelegramClient
        tb = ''.join(traceback.format_exception(
            type(exception), exception, exception.__traceback__
        ))
        tail = '\n'.join(tb.strip().splitlines()[-16:])
        msg = (
            f"\U0001F6A8 <b>Scheduled Job Failed</b>\n"
            f"Job: <code>{command}</code>\n"
            f"<pre>{tail[:800]}</pre>"
        )
        TelegramClient().send_message_sync(msg)
    except Exception as notify_err:
        logger.error(f"Scheduler: crash alert failed: {notify_err}")


async def run_scheduler() -> None:
    """Main loop. Runs forever until the process is terminated."""
    logger.info("Scheduler started. Ticking every %ds.", TICK_SECONDS)
    logger.info(
        "Schedule: run @ 08:00 and every :00/:30 from 11:00–22:30; "
        "settle @ 02:00."
    )
    last_fire: dict = {}
    while True:
        now = datetime.now()
        await _tick(now, last_fire, _dispatch)
        await asyncio.sleep(TICK_SECONDS)
