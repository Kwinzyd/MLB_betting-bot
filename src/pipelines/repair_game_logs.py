"""One-off data repair for historical game-log corruption.

Two defects accumulated in pitcher_game_logs / batter_game_logs:

1. innings_pitched was stored in baseball notation ("5.2" = 5 innings + 2
   outs) but read everywhere as a plain decimal — every K/9 / ERA / WHIP
   computed on top was inflated. Repair converts X.1 -> X+1/3, X.2 -> X+2/3.
   Idempotent: converted values have fractional parts ~.333/.667 which no
   longer match the notation pattern.

2. `date` was empty on every log row (the BDL stat payload's date fields are
   blank), so recency windows / rest days / as-of filters were meaningless.
   Repair backfills date from games.date via the bdl_game_id join. Rows whose
   game is missing from `games` stay empty — re-run `python main.py backfill
   --seasons <years>` first to restore those game rows (stats are skipped for
   already-synced games, so it only costs /games calls), then run this again.

Usage:
    python main.py repair-logs
    python main.py repair-logs --seasons 2022,2023,2024,2025
        (also restores missing `games` rows from BDL /games — no stats calls —
         so the date backfill can cover historical log rows)
"""
import asyncio

from src.data.db import get_db_connection
from src.utils.logging_utils import get_logger

logger = get_logger(__name__)

_LOG_TABLES = ("pitcher_game_logs", "batter_game_logs")

_FINISHED_STATUSES = frozenset({'Final', 'final', 'COMPLETED', 'STATUS_FINAL'})


async def _restore_games_rows(seasons) -> int:
    """Re-insert `games` rows for historical seasons from BDL /games only.

    Mirrors sync_historical's games upsert (game_id = "bdl:{id}",
    historical=1) but never touches /stats — the log rows already exist;
    only the game date/team metadata is missing.
    """
    from src.clients.mlb_stats import MLBStatsClient
    client = MLBStatsClient(requests_per_second=2.0)
    restored = 0
    for season in seasons:
        games = await client.get_games(season=season)
        completed = [g for g in games if g.get('status') in _FINISHED_STATUSES]
        logger.info("season=%s: %d completed games from BDL", season, len(completed))
        with get_db_connection() as conn:
            for g in completed:
                gid = g.get('id')
                if not gid:
                    continue
                home = g.get('home_team') or {}
                away = g.get('visitor_team') or g.get('away_team') or {}
                conn.execute(
                    """
                    INSERT INTO games (game_id, bdl_game_id, date, home_team, away_team,
                                       home_team_id, away_team_id, venue, status, historical)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, 'COMPLETED', 1)
                    ON CONFLICT(game_id) DO UPDATE SET
                        bdl_game_id=excluded.bdl_game_id,
                        date=excluded.date,
                        historical=1
                    """,
                    (
                        f"bdl:{gid}",
                        gid,
                        g.get('date'),
                        (home.get('display_name') or home.get('full_name') or '') if isinstance(home, dict) else '',
                        (away.get('display_name') or away.get('full_name') or '') if isinstance(away, dict) else '',
                        home.get('id') if isinstance(home, dict) else None,
                        away.get('id') if isinstance(away, dict) else None,
                        g.get('venue') or g.get('location') or '',
                    ),
                )
                restored += 1
            conn.commit()
    return restored


def _repair_ip(conn) -> int:
    """Convert baseball-notation IP to true decimal innings. Returns rows changed."""
    cur = conn.execute("""
        UPDATE pitcher_game_logs
        SET innings_pitched =
            CAST(innings_pitched AS INT) +
            ROUND((innings_pitched - CAST(innings_pitched AS INT)) * 10) / 3.0
        WHERE ABS((innings_pitched - CAST(innings_pitched AS INT)) * 10 - 1) < 0.05
           OR ABS((innings_pitched - CAST(innings_pitched AS INT)) * 10 - 2) < 0.05
    """)
    return cur.rowcount


def _repair_dates(conn, table: str) -> tuple[int, int]:
    """Backfill empty log dates from games.date. Returns (fixed, still_empty)."""
    cur = conn.execute(f"""
        UPDATE {table}
        SET date = (
            SELECT substr(g.date, 1, 10) FROM games g
            WHERE g.bdl_game_id = {table}.game_id
              AND g.date IS NOT NULL AND g.date != ''
        )
        WHERE (date IS NULL OR date = '')
          AND EXISTS (
            SELECT 1 FROM games g
            WHERE g.bdl_game_id = {table}.game_id
              AND g.date IS NOT NULL AND g.date != ''
          )
    """)
    remaining = conn.execute(
        f"SELECT COUNT(*) FROM {table} WHERE date IS NULL OR date = ''"
    ).fetchone()[0]
    return cur.rowcount, remaining


def repair_game_logs(seasons=None) -> dict:
    logger.info("Executing pipeline: repair_game_logs")
    summary: dict = {}
    if seasons:
        restored = asyncio.run(_restore_games_rows(list(seasons)))
        summary["games_restored"] = restored
        logger.info("Restored %d historical games rows from BDL.", restored)
    with get_db_connection() as conn:
        ip_fixed = _repair_ip(conn)
        summary["ip_converted"] = ip_fixed
        logger.info("IP notation repair: %d pitcher rows converted to true decimals.", ip_fixed)

        for table in _LOG_TABLES:
            fixed, remaining = _repair_dates(conn, table)
            summary[f"{table}_dates_fixed"] = fixed
            summary[f"{table}_dates_empty"] = remaining
            logger.info(
                "%s: %d dates backfilled from games; %d still empty.",
                table, fixed, remaining,
            )
            if remaining:
                logger.warning(
                    "%s: %d rows have no matching games row — run "
                    "`python main.py backfill --seasons <years>` to restore game "
                    "rows, then re-run repair-logs.",
                    table, remaining,
                )
        conn.commit()
    logger.info("repair_game_logs complete: %s", summary)
    return summary
