"""One-shot backfill of multi-season BDL game logs for model training.

Pulls every completed game for the given seasons, batches `/stats` calls to
50 games per HTTP request, and writes into the existing pitcher_game_logs /
batter_game_logs tables. Rows written by this pipeline set `games.historical=1`
so the 90-day prune leaves them alone.
"""
from __future__ import annotations

import asyncio
from typing import Iterable, List
from src.utils.time_utils import utcnow

from src.clients.mlb_stats import MLBStatsClient
from src.data.db import get_db_connection
from src.pipelines.sync_stats import _upsert_player, _insert_pitcher_log, _insert_batter_log, _upsert_teams
from src.utils.logging_utils import get_logger

logger = get_logger(__name__)

_COMPLETED_STATUSES = frozenset({
    'Final', 'final', 'COMPLETED',
    'STATUS_FINAL',         # BDL MLB API actual value
    'STATUS_SCHEDULED',     # Not completed but listed for reference
})
# Only these indicate a finished game we can learn from:
_FINISHED_STATUSES = frozenset({'Final', 'final', 'COMPLETED', 'STATUS_FINAL'})


async def _sync_historical_async(seasons: List[int], chunk_size: int) -> dict:
    """
    All BDL I/O runs inside a single async context so the httpx.AsyncClient
    is created, used, and closed within one event loop — multiple asyncio.run()
    calls would close and re-open the loop, breaking the persistent httpx client.
    """
    client = MLBStatsClient()

    # Seed teams table
    all_teams = await client.get_teams()
    with get_db_connection() as conn:
        _upsert_teams(conn, all_teams)
        conn.commit()

    totals = {"games": 0, "pitcher_logs": 0, "batter_logs": 0}

    for season in seasons:
        logger.info(f"Fetching season {season} games from BDL ...")
        games = await client.get_games(season=season)
        completed = [g for g in games if g.get('status') in _FINISHED_STATUSES]
        logger.info(f"  season={season}: {len(completed)} completed games")

        # Insert game rows tagged historical=1
        with get_db_connection() as conn:
            for g in completed:
                gid = g.get('id')
                if not gid:
                    continue
                home = g.get('home_team') or {}
                away = g.get('visitor_team') or g.get('away_team') or {}
                # BDL MLB uses 'display_name'; fall back chain for robustness
                home_name = (home.get('display_name') or home.get('full_name')
                             or home.get('name') or '') if isinstance(home, dict) else ''
                away_name = (away.get('display_name') or away.get('full_name')
                             or away.get('name') or '') if isinstance(away, dict) else ''
                home_id = home.get('id') if isinstance(home, dict) else None
                away_id = away.get('id') if isinstance(away, dict) else None
                conn.execute(
                    """
                    INSERT INTO games (game_id, bdl_game_id, date, home_team, away_team,
                                       home_team_id, away_team_id, venue, status, historical)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, 'COMPLETED', 1)
                    ON CONFLICT(game_id) DO UPDATE SET
                        bdl_game_id=excluded.bdl_game_id,
                        date=excluded.date,
                        home_team=excluded.home_team,
                        away_team=excluded.away_team,
                        home_team_id=excluded.home_team_id,
                        away_team_id=excluded.away_team_id,
                        venue=COALESCE(excluded.venue, games.venue),
                        status='COMPLETED',
                        historical=1
                    """,
                    (
                        f"bdl:{gid}",
                        gid,
                        g.get('date'),
                        home_name,
                        away_name,
                        home_id,
                        away_id,
                        g.get('venue') or g.get('location') or '',
                    ),
                )
                totals["games"] += 1
            conn.commit()

        # Fetch ALL stats in one get_stats_batch call — its own semaphore
        # (max_concurrency=5) handles rate-limiting correctly. Chunking here
        # too would create thousands of concurrent HTTP requests.
        game_ids: List[int] = [g['id'] for g in completed if g.get('id')]
        if not game_ids:
            continue

        # Resume support: skip games whose stats were already written
        with get_db_connection() as conn:
            placeholders = ",".join("?" * len(game_ids))
            rows = conn.execute(
                f"SELECT bdl_game_id FROM games WHERE bdl_game_id IN ({placeholders}) AND last_synced_at IS NOT NULL",
                game_ids,
            ).fetchall()
        already_synced = {r[0] for r in rows}
        if already_synced:
            game_ids = [gid for gid in game_ids if gid not in already_synced]
            logger.info(f"  season={season}: {len(already_synced)} already synced, {len(game_ids)} remaining")
        if not game_ids:
            logger.info(f"  season={season}: all games already synced, skipping stats fetch")
            continue

        logger.info(f"  season={season}: fetching stats for {len(game_ids)} games ...")
        # Larger batches (200) reduce total requests; max_concurrency=3 keeps it polite
        stats = await client.get_stats_batch(game_ids, chunk_size=200, max_concurrency=3)
        logger.info(f"  season={season}: received {len(stats)} player-game stat rows")

        # Write to DB in chunks of 500 so we don't hold one giant transaction
        for i in range(0, len(stats), 500):
            batch = stats[i:i + 500]
            with get_db_connection() as conn:
                for player_stat in batch:
                    player_id, player_name, position, gid, game_date = _upsert_player(conn, player_stat)
                    if player_id is None:
                        continue
                    ip = player_stat.get('ip') or player_stat.get('innings_pitched')
                    is_pitcher = ip is not None and str(ip) not in ('0', '0.0', '')
                    if is_pitcher:
                        totals["pitcher_logs"] += _insert_pitcher_log(
                            conn, gid, player_id, game_date, player_stat, player_name
                        )
                    else:
                        totals["batter_logs"] += _insert_batter_log(
                            conn, gid, player_id, game_date, player_stat, player_name
                        )
                # Mark these games as synced so restarts skip them
                synced_ids = list({stat.get("game", {}).get("id") for stat in batch if stat.get("game", {}).get("id")})
                if synced_ids:
                    now = utcnow().isoformat()
                    conn.execute(
                        "UPDATE games SET last_synced_at = ? WHERE bdl_game_id IN ({})".format(
                            ",".join("?" * len(synced_ids))
                        ),
                        [now] + synced_ids,
                    )
                conn.commit()
            if i > 0 and i % 5000 == 0:
                logger.info(f"  season={season}: wrote {i} stat rows so far ...")


    logger.info(f"sync_historical complete: {totals}")
    return totals


def sync_historical(seasons: Iterable[int], chunk_size: int = 50) -> dict:
    """
    Backfill games + game logs for the given MLB seasons.

    Synchronous entry point called from main.py. All async I/O is wrapped
    inside a single asyncio.run() so the httpx client stays on one event loop.

    Returns a summary dict of rows written per table.
    """
    seasons = list(seasons)
    logger.info(f"Executing pipeline: sync_historical for seasons={seasons}")
    return asyncio.run(_sync_historical_async(seasons, chunk_size))
