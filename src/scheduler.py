"""
Time-based scheduler for the MLB bot.

Runs as a long-lived `python main.py schedule` process. Every 60s it checks
the wall clock against a declarative schedule and dispatches due jobs.

Schedule (local time):
    08:00              `run`     — morning sync + first scan
    11:00 … 22:30      `run`     — every 30min during the MLB pregame/in-game window
    11:15 … 22:15      `trigger` — every 30min (offset 15min from run) for ump/weather snipes
    02:00 (next day)   `settle`  — post-game settlement + stats sync
    03:00 (next day)   `train`   — nightly GLM retrain on settled results + dispersion refit

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


def trigger_minutes() -> List[Tuple[int, int]]:
    """(hour, minute) pairs for `trigger` — offset 15min from `run` slots.

    Staggering avoids both jobs hitting the Odds API simultaneously at the
    top/bottom of the hour, and gives `run` time to write new snapshots that
    trigger_watch can then compare against.
    """
    slots: List[Tuple[int, int]] = []
    for hour in range(11, 23):
        slots.append((hour, 15))
        slots.append((hour, 45))
    return slots


def settle_minutes() -> List[Tuple[int, int]]:
    return [(2, 0)]


def train_minutes() -> List[Tuple[int, int]]:
    """(hour, minute) for nightly GLM retrain.

    Runs at 03:00, one hour after settle (02:00), so the freshly settled
    bet_results rows are already committed before the model fits on them.
    """
    return [(3, 0)]


def statcast_minutes() -> List[Tuple[int, int]]:
    """Nightly Statcast sync at 03:30 — after train writes the new model, before
    calibrate runs at 04:00. Season leaderboards update daily during the season."""
    return [(3, 30)]


def calibrate_minutes() -> List[Tuple[int, int]]:
    """Calibration pipeline at 04:00 — after Statcast sync, fits Platt scaling
    on the calibration_log built from settled bets."""
    return [(4, 0)]


def fit_correlations_minutes() -> List[Tuple[int, int]]:
    """Correlation fitting at 04:30 — after calibration, fits empirical portfolio
    correlations from bet_pair_outcomes accumulated by settle_results."""
    return [(4, 30)]


def fit_bookmaker_bias_minutes(now: datetime) -> List[Tuple[int, int]]:
    """Saturday 05:00 — weekly bias fit. Only fires on Saturday (weekday==5)."""
    if now.weekday() == 5:
        return [(5, 0)]
    return []


def monitor_drift_minutes(now: datetime) -> List[Tuple[int, int]]:
    """Sunday 05:30 — weekly drift check and conditional retrain.
    Only fires on Sunday (weekday==6)."""
    if now.weekday() == 6:
        return [(5, 30)]
    return []


def should_run_at(command: str, now: datetime) -> bool:
    """Pure time-based predicate. No side effects; easy to unit-test."""
    hm = (now.hour, now.minute)
    if command == "run":
        return hm in morning_run_minutes()
    if command == "trigger":
        return hm in trigger_minutes()
    if command == "settle":
        return hm in settle_minutes()
    if command == "train":
        return hm in train_minutes()
    if command == "statcast":
        return hm in statcast_minutes()
    if command == "calibrate":
        return hm in calibrate_minutes()
    if command == "fit_correlations":
        return hm in fit_correlations_minutes()
    if command == "fit_bookmaker_bias":
        return hm in fit_bookmaker_bias_minutes(now)
    if command == "monitor_drift":
        return hm in monitor_drift_minutes(now)
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
    from src.pipelines.find_parlays import find_and_alert_parlays
    from src.pipelines.trigger_watch import run_trigger_watch
    from src.pipelines.settle_results import settle_results

    if command == "run":
        from src.pipelines.enrich_injuries import enrich_injuries
        await sync_events()
        await sync_injuries()
        await enrich_injuries()  # LLM injury signals (no-op if LLM off)
        await sync_lineups()
        sync_umpires()
        from src.pipelines.calculate_team_stats import calculate_team_stats
        calculate_team_stats()
        await scan_props(force=False)
        await send_alerts()
        await find_and_alert_sgps()
        await find_and_alert_parlays()
        # Game markets (no-op unless GAME_MARKETS_ENABLED).
        from src.pipelines.scan_game_markets import scan_game_markets
        from src.pipelines.send_game_alerts import send_game_alerts
        await scan_game_markets()
        await send_game_alerts()
    elif command == "trigger":
        await run_trigger_watch()
    elif command == "settle":
        await sync_stats()
        from src.pipelines.calculate_team_stats import calculate_team_stats
        calculate_team_stats()
        settle_results()
    elif command == "train":
        # Run in an executor so scipy/numpy-heavy fitting doesn't block the
        # event loop. train_all() is synchronous CPU-bound work.
        from src.pipelines.train_model import train_all
        loop = asyncio.get_event_loop()
        results = await loop.run_in_executor(None, train_all)
        for r in results:
            logger.info(f"Nightly train result: {r}")
    elif command == "statcast":
        from src.pipelines.sync_statcast import sync_statcast
        await sync_statcast()
    elif command == "calibrate":
        from src.pipelines.calibrate_model import calibrate_all
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, calibrate_all)
    elif command == "fit_correlations":
        from src.pipelines.fit_correlations import fit_correlations
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, fit_correlations)
    elif command == "fit_bookmaker_bias":
        from src.pipelines.fit_bookmaker_bias import fit_bookmaker_bias
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, fit_bookmaker_bias)
    elif command == "monitor_drift":
        from src.pipelines.monitor_drift import monitor_drift
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, monitor_drift)
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
    for command in ("run", "trigger", "settle", "train", "statcast", "calibrate",
                    "fit_correlations", "fit_bookmaker_bias", "monitor_drift"):
        if not should_run_at(command, now):
            continue
        if last_fire.get(command) == hm:
            continue
        last_fire[command] = hm
        logger.info(f"Scheduler: firing '{command}' at {now.isoformat(timespec='seconds')}")
        try:
            timeout = 1800 if command == "train" else 600
            await asyncio.wait_for(dispatch(command), timeout=timeout)
            logger.info(f"Scheduler: '{command}' completed")
        except asyncio.TimeoutError:
            logger.error(f"Scheduler: '{command}' timed out after {timeout}s")
            _notify_job_crash(command, TimeoutError(f"Job '{command}' exceeded {timeout}s timeout"))
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
        "trigger @ every :15/:45 from 11:15–22:45; "
        "settle @ 02:00; train @ 03:00."
    )
    last_fire: dict = {}
    while True:
        now = datetime.now()
        await _tick(now, last_fire, _dispatch)
        await asyncio.sleep(TICK_SECONDS)
