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
    TRIGGER_TOTAL_SHIFT_THRESHOLD,
    TRIGGER_TOTAL_MIN_HISTORY_MINUTES,
    UMP_MIN_GAMES,
)
from src.clients.odds_api import OddsAPIClient
from src.clients.weather import WeatherClient
from src.clients.telegram_bot import TelegramClient
from src.data.db import get_db_connection
from src.data.park_factors import get_stadium_meta, compute_weather_adjustments
from src.pipelines.scan_props import scan_props, _parse_game_total, _record_total_snapshot
from src.pipelines.send_alerts import send_alerts
from src.utils.logging_utils import get_logger

logger = get_logger(__name__)


async def run_trigger_watch():
    """Entry point: scan active games for ump/weather triggers, fire if extreme."""
    logger.info("Executing pipeline: trigger_watch")
    
    from src.data.cache import cache
    if cache.get("odds_api_quota_exhausted"):
        logger.error("Trigger pipeline aborted: API Quota Circuit Breaker is active.")
        return

    with get_db_connection() as conn:
        games = conn.execute(
            "SELECT game_id, home_team, away_team, venue "
            "FROM games WHERE status != 'COMPLETED'"
        ).fetchall()

    if not games:
        logger.info("No active games; trigger watch idle.")
        return

    weather_client = WeatherClient()
    odds_client = OddsAPIClient()
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

        platoon_hit = _check_matchup_volatility(game_id, game['home_team'], game['away_team'])
        if platoon_hit and not _already_fired(game_id, 'platoon', dedup_cutoff):
            fired.append((game_id, matchup, 'platoon', platoon_hit))

        if not _already_fired(game_id, 'total_shift', dedup_cutoff):
            total_hit = await _check_total_shift(game_id, odds_client)
            if total_hit:
                fired.append((game_id, matchup, 'total_shift', total_hit))

    if not fired:
        logger.info("Trigger watch: no extreme conditions detected.")
        return

    triggered_game_ids = sorted({f[0] for f in fired})
    logger.info(
        f"Trigger watch: {len(fired)} signal(s) fired across "
        f"{len(triggered_game_ids)} game(s)."
    )

    _persist_triggers(fired, now_utc)
    await _alert_triggers(fired)

    await scan_props(force=True, game_ids=triggered_game_ids)
    await send_alerts()


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


def _check_matchup_volatility(game_id, home_team, away_team):
    """
    Return detail dict if a probable pitcher averages < 3.5 IP recently (bullpen game)
    AND the opposing lineup features switch-hitters (who are highly volatile vs relievers).
    """
    with get_db_connection() as conn:
        pitchers = conn.execute(
            "SELECT player_name, team FROM probable_pitchers WHERE game_id = ?",
            (game_id,)
        ).fetchall()
        
        for pp in pitchers:
            pitcher_name = pp['player_name']
            pitcher_team = pp['team'] or ''
            
            # Identify opposing team for the lineup check
            if pitcher_team.lower() in home_team.lower() or home_team.lower() in pitcher_team.lower():
                opp_team = away_team
            else:
                opp_team = home_team
                
            logs = conn.execute(
                """
                SELECT pgl.innings_pitched 
                FROM pitcher_game_logs pgl
                JOIN players p ON p.player_id = pgl.player_id
                WHERE p.name = ? COLLATE NOCASE
                ORDER BY pgl.date DESC LIMIT 5
                """, (pitcher_name,)
            ).fetchall()
            
            if not logs:
                continue
                
            avg_ip = sum(l['innings_pitched'] or 0 for l in logs) / len(logs)
            if avg_ip < 3.5:
                # Potential bullpen game. Check opponent daily lineup for switch hitters
                hitters = conn.execute(
                    """
                    SELECT dl.player_name
                    FROM daily_lineups dl
                    JOIN players p ON dl.player_name = p.name COLLATE NOCASE
                    WHERE dl.game_id = ? AND dl.team LIKE ? COLLATE NOCASE
                      AND p.bats = 'S'
                    """, (game_id, f"%{opp_team}%")
                ).fetchall()
                
                if hitters:
                    switch_names = [h['player_name'] for h in hitters]
                    return {
                        'opener': pitcher_name,
                        'avg_ip': float(avg_ip),
                        'opp_team': opp_team,
                        'switch_hitters': switch_names
                    }
                    
    return None


async def _check_total_shift(game_id, odds_client):
    """Return detail dict if the consensus game total has moved by
    TRIGGER_TOTAL_SHIFT_THRESHOLD vs the latest history snapshot, else None.

    Always re-records the freshly polled total (even when below threshold) so
    future deltas measure from the most recent observation rather than a
    stale baseline.
    """
    with get_db_connection() as conn:
        prior = conn.execute(
            "SELECT total, timestamp FROM game_totals_history "
            "WHERE game_id = ? ORDER BY timestamp DESC LIMIT 1",
            (game_id,),
        ).fetchone()
    if not prior:
        return None
    try:
        prior_ts = datetime.fromisoformat(prior['timestamp'])
    except ValueError:
        return None
    age_min = (datetime.now(timezone.utc) - prior_ts).total_seconds() / 60
    if age_min < TRIGGER_TOTAL_MIN_HISTORY_MINUTES:
        return None

    try:
        event_odds = await odds_client.get_event_odds(
            game_id, ['totals'], bust_cache=True
        )
    except Exception as e:
        logger.warning(f"Total-shift fetch failed for {game_id}: {e}")
        return None
    fresh = _parse_game_total(event_odds) if event_odds else None
    if fresh is None:
        return None

    _record_total_snapshot(game_id, fresh, source='trigger')
    delta = fresh - float(prior['total'])
    if abs(delta) < TRIGGER_TOTAL_SHIFT_THRESHOLD:
        return None
    return {
        'prior_total': float(prior['total']),
        'current_total': float(fresh),
        'delta': delta,
        'prior_age_min': round(age_min, 1),
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


async def _alert_triggers(fired):
    try:
        client = TelegramClient()
        lines = ["<b>Edge trigger fired</b>"]
        for _game_id, matchup, trigger_type, detail in fired:
            lines.append(f"\n<b>{matchup}</b> — {trigger_type}")
            lines.append(_format_detail(trigger_type, detail))
        lines.append("\nForcing targeted odds pull...")
        await client.send_message("\n".join(lines))
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
    if trigger_type == 'total_shift':
        sign = '+' if detail['delta'] >= 0 else ''
        return (
            f"Total moved {detail['prior_total']:.1f} → {detail['current_total']:.1f} "
            f"(Δ={sign}{detail['delta']:.1f}, prior {detail['prior_age_min']}min ago)"
        )
    if trigger_type == 'platoon':
        hitters = ", ".join(detail['switch_hitters'])
        return (
            f"Bullpen Game (Opener: {detail['opener']}, avg {detail['avg_ip']:.1f} IP)\n"
            f"    └ Volatile Switch-Hitters: {hitters}"
        )
    return str(detail)
