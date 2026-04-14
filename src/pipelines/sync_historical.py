"""One-shot backfill of multi-season BDL game logs for model training.

Pulls every completed game for the given seasons, batches `/stats` calls to
/50 games per HTTP request, and writes into the existing pitcher_game_logs /
batter_game_logs tables. Rows written by this pipeline set `games.historical=1`
so the 90-day prune leaves them alone.
"""
from __future__ import annotations

from typing import Iterable, List

from src.clients.mlb_stats import MLBStatsClient
from src.data.db import get_db_connection
from src.pipelines.sync_stats import _upsert_player, _insert_pitcher_log, _insert_batter_log, _upsert_teams
from src.utils.logging_utils import get_logger

logger = get_logger(__name__)

_COMPLETED_STATUSES = frozenset({'Final', 'final', 'COMPLETED'})


def sync_historical(seasons: Iterable[int], chunk_size: int = 50) -> dict:
    """
    Backfill games + game logs for the given MLB seasons.

    Returns a summary dict of rows written per table.
    """
    seasons = list(seasons)
    logger.info(f"Executing pipeline: sync_historical for seasons={seasons}")
    client = MLBStatsClient()

    # Seed teams table (the stat inserts depend on it for FK-like semantics)
    all_teams = client.get_teams()
    with get_db_connection() as conn:
        _upsert_teams(conn, all_teams)
        conn.commit()

    totals = {"games": 0, "pitcher_logs": 0, "batter_logs": 0}

    for season in seasons:
        logger.info(f"Fetching season {season} games from BDL ...")
        games = client.get_games(season=season)
        completed = [g for g in games if g.get('status') in _COMPLETED_STATUSES]
        logger.info(f"  season={season}: {len(completed)} completed games")

        # Insert games rows tagged historical=1
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
                        (home.get('full_name') if isinstance(home, dict) else None) or '',
                        (away.get('full_name') if isinstance(away, dict) else None) or '',
                        home.get('id') if isinstance(home, dict) else None,
                        away.get('id') if isinstance(away, dict) else None,
                        g.get('venue') or '',
                    ),
                )
                totals["games"] += 1
            conn.commit()

        # Fetch stats in batches
        game_ids: List[int] = [g['id'] for g in completed if g.get('id')]
        if not game_ids:
            continue

        for i in range(0, len(game_ids), chunk_size):
            chunk = game_ids[i:i + chunk_size]
            stats = client.get_stats_batch(chunk)
            if not stats:
                continue

            with get_db_connection() as conn:
                for player_stat in stats:
                    player_id, player_name, position, gid, game_date = _upsert_player(conn, player_stat)
                    if player_id is None:
                        continue
                    ip = player_stat.get('innings_pitched') or player_stat.get('ip')
                    if ip is not None and (position == 'P' or str(ip) != '0'):
                        totals["pitcher_logs"] += _insert_pitcher_log(
                            conn, gid, player_id, game_date, player_stat, player_name
                        )
                    else:
                        totals["batter_logs"] += _insert_batter_log(
                            conn, gid, player_id, game_date, player_stat, player_name
                        )
                conn.commit()

            if (i // chunk_size) % 20 == 0 and i > 0:
                logger.info(f"  season={season}: processed {i}/{len(game_ids)} games")

    logger.info(f"sync_historical complete: {totals}")
    return totals
