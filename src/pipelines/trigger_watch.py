"""
Weather / umpire edge-trigger watcher.

Polls active games for environmental inflection points that shift our
projections faster than soft books can re-price. When an extreme umpire
k_factor or severe weather condition is detected, fires a targeted
per-game odds pull + alert pass, deduped so the same signal doesn't
re-fire each cron tick.

Cron invocation: `python main.py trigger`  (every 10-15 min).
"""

import json
from datetime import datetime, timezone, timedelta

from src.config import (
    TRIGGER_UMP_THRESHOLD,
    TRIGGER_WEATHER_HR_THRESHOLD,
    TRIGGER_WEATHER_SO_THRESHOLD,
    TRIGGER_DEDUP_HOURS,
    UMP_MIN_GAMES,
)
from src.clients.weather import WeatherClient
from src.clients.telegram_bot import TelegramClient
from src.data.db import get_db_connection
from src.data.park_factors import get_stadium_meta, compute_weather_adjustments
from src.pipelines.scan_props import scan_props
from src.pipelines.send_alerts import send_alerts
from src.utils.logging_utils import get_logger

logger = get_logger(__name__)


def run_trigger_watch():
    """Entry point: scan active games for ump/weather triggers, fire if extreme."""
    logger.info("Executing pipeline: trigger_watch")

    with get_db_connection() as conn:
        games = conn.execute(
            "SELECT game_id, home_team, away_team, venue "
            "FROM games WHERE status != 'COMPLETED'"
        ).fetchall()

    if not games:
        logger.info("No active games; trigger watch idle.")
        return

    weather_client = WeatherClient()
    now_utc = datetime.now(timezone.utc)
    dedup_cutoff = (now_utc - timedelta(hours=TRIGGER_DEDUP_HOURS)).isoformat()

    fired = []  # [(game_id, matchup, trigger_type, detail_dict)]

    for game in games:
        game_id = game['game_id']
        matchup = f"{game['away_team']} @ {game['home_team']}"
        venue = game['venue']

        ump_hit = _check_umpire(game_id)
        if ump_hit and not _already_fired(game_id, 'umpire', dedup_cutoff):
            fired.append((game_id, matchup, 'umpire', ump_hit))

        weather_hit = _check_weather(venue, weather_client)
        if weather_hit and not _already_fired(game_id, 'weather', dedup_cutoff):
            fired.append((game_id, matchup, 'weather', weather_hit))

    if not fired:
        logger.info("Trigger watch: no extreme conditions detected.")
        return

    triggered_game_ids = sorted({f[0] for f in fired})
    logger.info(
        f"Trigger watch: {len(fired)} signal(s) fired across "
        f"{len(triggered_game_ids)} game(s)."
    )

    _persist_triggers(fired, now_utc)
    _alert_triggers(fired)

    scan_props(force=True, game_ids=triggered_game_ids)
    send_alerts()


def _check_umpire(game_id):
    """Return detail dict if the game's plate umpire has an extreme k_factor, else None."""
    with get_db_connection() as conn:
        row = conn.execute(
            """
            SELECT us.umpire_name, us.k_factor, us.games_called
            FROM umpire_game_assignments uga
            JOIN umpire_stats us ON us.umpire_id = uga.umpire_id
            WHERE uga.game_id = ?
            """,
            (game_id,),
        ).fetchone()

    if not row:
        return None
    if row['games_called'] < UMP_MIN_GAMES:
        return None

    deviation = abs(row['k_factor'] - 1.0)
    if deviation < TRIGGER_UMP_THRESHOLD:
        return None

    return {
        'umpire_name': row['umpire_name'],
        'k_factor': row['k_factor'],
        'games_called': row['games_called'],
        'deviation': deviation,
    }


def _check_weather(venue, weather_client):
    """Return detail dict if weather-driven park-factor shift is extreme, else None."""
    meta = get_stadium_meta(venue)
    if not meta or meta.get('roof') == 'dome':
        return None

    weather = weather_client.get_game_weather(meta['lat'], meta['lon'])
    if not weather:
        return None

    adjustments = compute_weather_adjustments(weather, meta)
    hr_dev = abs(adjustments['hr'] - 1.0)
    so_dev = abs(adjustments['so'] - 1.0)

    if hr_dev < TRIGGER_WEATHER_HR_THRESHOLD and so_dev < TRIGGER_WEATHER_SO_THRESHOLD:
        return None

    return {
        'venue': venue,
        'temp_f': weather.get('temp_f'),
        'wind_mph': weather.get('wind_mph'),
        'wind_deg': weather.get('wind_deg'),
        'hr_adjust': adjustments['hr'],
        'so_adjust': adjustments['so'],
    }


def _already_fired(game_id, trigger_type, dedup_cutoff_iso):
    """True if a trigger of this type fired for this game within the dedup window."""
    with get_db_connection() as conn:
        row = conn.execute(
            "SELECT 1 FROM trigger_events "
            "WHERE game_id = ? AND trigger_type = ? AND triggered_at >= ? "
            "LIMIT 1",
            (game_id, trigger_type, dedup_cutoff_iso),
        ).fetchone()
    if row:
        logger.info(
            f"Trigger dedup: {trigger_type} already fired for {game_id} within "
            f"{TRIGGER_DEDUP_HOURS}h window; skipping."
        )
        return True
    return False


def _persist_triggers(fired, now_utc):
    ts = now_utc.isoformat()
    with get_db_connection() as conn:
        for game_id, _matchup, trigger_type, detail in fired:
            conn.execute(
                "INSERT OR IGNORE INTO trigger_events "
                "(game_id, trigger_type, detail, triggered_at) "
                "VALUES (?, ?, ?, ?)",
                (game_id, trigger_type, json.dumps(detail), ts),
            )
        conn.commit()


def _alert_triggers(fired):
    try:
        client = TelegramClient()
        lines = ["<b>Edge trigger fired</b>"]
        for _game_id, matchup, trigger_type, detail in fired:
            lines.append(f"\n<b>{matchup}</b> — {trigger_type}")
            lines.append(_format_detail(trigger_type, detail))
        lines.append("\nForcing targeted odds pull...")
        client.send_message("\n".join(lines))
    except Exception as e:
        logger.warning(f"Trigger Telegram alert failed: {e}")


def _format_detail(trigger_type, detail):
    if trigger_type == 'umpire':
        return (
            f"Umpire {detail['umpire_name']} "
            f"k_factor={detail['k_factor']:.3f} "
            f"(n={detail['games_called']})"
        )
    if trigger_type == 'weather':
        return (
            f"{detail['venue']}: {detail['temp_f']:.0f}°F, "
            f"wind {detail['wind_mph']:.0f}mph @ {detail['wind_deg']}° — "
            f"hr×{detail['hr_adjust']:.3f}, so×{detail['so_adjust']:.3f}"
        )
    return str(detail)
