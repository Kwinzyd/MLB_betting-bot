"""Sync Statcast season leaderboard data from Baseball Savant into the DB.

Runs nightly at 03:30 (after the GLM train job, before calibration). Downloads
pitcher and batter CSV leaderboards, matches on players.mlb_id (MLBAM player ID),
and upserts into statcast_pitcher_stats / statcast_batter_stats.

Player ID mapping:
  Baseball Savant uses the MLB MLBAM integer ID as its 'player_id'. BDL uses its
  own integer IDs internally. The mapping lives in players.mlb_id, populated when
  BDL returns a player with an 'mlb_player_id' field. Players without mlb_id are
  skipped silently — their Statcast features will default to 0 / league-average in
  feature_builder.py.

Column aliases:
  Savant occasionally renames columns between seasons. The alias tables below map
  every known variant to the canonical DB column name.
"""
from __future__ import annotations

from typing import Any, Optional

import pandas as pd

from src.clients.statcast import StatcastClient
from src.config import MLB_SEASON
from src.data.db import get_db_connection
from src.utils.logging_utils import get_logger
from src.utils.time_utils import get_utc_now_iso

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Column alias tables: Savant CSV name → our canonical DB column name
# ---------------------------------------------------------------------------

# All known variants for each pitcher metric across different seasons.
_PITCHER_ALIASES: dict[str, list[str]] = {
    "whiff_pct":                 ["whiff_percent", "whiff%", "whiff_pct"],
    "chase_rate":                ["o_swing_percent", "oz_swing%", "chase_rate", "o_swing%"],
    "barrel_pct_against":        ["barrel_batted_rate", "barrel%", "barrels_per_bbe_percent"],
    "hard_hit_pct_against":      ["hard_hit_percent", "hard_hit%", "hard_hit_rate"],
    "spin_rate_ff":              ["avg_spin", "spin_rate", "release_spin_rate", "avg_spin_rate"],
    "avg_exit_velocity_against": ["avg_best_speed", "exit_velocity_avg", "avg_launch_speed",
                                  "avg_exit_velocity"],
}

_BATTER_ALIASES: dict[str, list[str]] = {
    "exit_velocity_avg":  ["avg_best_speed", "exit_velocity_avg", "avg_launch_speed",
                           "avg_exit_velocity"],
    "launch_angle_avg":   ["la", "avg_launch_angle", "launch_angle_avg"],
    "barrel_pct":         ["barrel_batted_rate", "barrel%", "barrels_per_bbe_percent"],
    "xwoba":              ["est_woba", "xwoba", "xwoba_mean"],
    "sprint_speed":       ["sprint_speed", "r_sprint_speed_top50p"],
    "whiff_pct":          ["whiff_percent", "whiff%", "whiff_pct"],
    "hard_hit_pct":       ["hard_hit_percent", "hard_hit%", "hard_hit_rate"],
}


def _get_val(row: pd.Series, aliases: list[str]) -> Optional[float]:
    """Extract the first matching alias value from a DataFrame row, or None."""
    for alias in aliases:
        if alias in row.index:
            v = row[alias]
            if pd.notna(v):
                try:
                    return float(v)
                except (TypeError, ValueError):
                    pass
    return None


# ---------------------------------------------------------------------------
# Pipeline entry point
# ---------------------------------------------------------------------------

async def sync_statcast(season: int = None) -> None:
    season = season or MLB_SEASON
    logger.info("sync_statcast: fetching Statcast leaderboards for season=%d", season)

    client = StatcastClient()

    import asyncio
    pitcher_df, batter_df = await asyncio.gather(
        client.get_all_pitcher_stats(season),
        client.get_all_batter_stats(season),
    )

    if pitcher_df.empty and batter_df.empty:
        logger.warning("sync_statcast: no data returned from Baseball Savant — skipping")
        return

    now = get_utc_now_iso()

    with get_db_connection() as conn:
        # Build lookups: mlb_id -> bdl_player_id AND name -> bdl_player_id
        player_rows = conn.execute(
            "SELECT player_id, name, mlb_id FROM players"
        ).fetchall()
        mlb_to_bdl: dict[int, int] = {r['mlb_id']: r['player_id'] for r in player_rows if r['mlb_id']}
        name_to_bdl: dict[str, int] = {r['name'].lower(): r['player_id'] for r in player_rows if r['name']}

        pitcher_count = _upsert_pitcher_stats(conn, pitcher_df, mlb_to_bdl, name_to_bdl, season, now)
        batter_count = _upsert_batter_stats(conn, batter_df, mlb_to_bdl, name_to_bdl, season, now)
        conn.commit()

    logger.info(
        "sync_statcast: upserted %d pitcher rows, %d batter rows for season %d",
        pitcher_count, batter_count, season,
    )

    # Derive platoon splits from game logs (no additional API call required).
    with get_db_connection() as conn:
        split_count = _compute_and_upsert_platoon_splits(conn, season)
        conn.commit()
    logger.info("sync_statcast: upserted %d batter platoon split rows", split_count)


_SPLIT_MARKETS = {
    "batter_hits":        "hits",
    "batter_home_runs":   "home_runs",
    "batter_total_bases": "total_bases",
}


def _compute_and_upsert_platoon_splits(conn: Any, season: int) -> int:
    """Derive per-player batter vs-LHP/RHP rates from game logs and upsert.

    Uses the pitcher with the most IP in each game as the 'representative'
    pitcher for the platoon matchup — a reliable proxy for the starter's hand.
    Ties (rare) are broken by MIN(player_id) for determinism.
    """
    year_prefix = f"{season}-%"
    now = get_utc_now_iso()

    try:
        rows = conn.execute(
            """
            SELECT bgl.player_id,
                   p_pitcher.throws           AS vs_hand,
                   SUM(bgl.hits)              AS total_hits,
                   SUM(bgl.home_runs)         AS total_hr,
                   SUM(bgl.total_bases)       AS total_tb,
                   SUM(COALESCE(bgl.plate_appearances, bgl.at_bats, 0)) AS total_pa
            FROM batter_game_logs bgl
            JOIN (
                SELECT game_id, MIN(player_id) AS pid
                FROM pitcher_game_logs
                WHERE innings_pitched = (
                    SELECT MAX(innings_pitched) FROM pitcher_game_logs p2
                    WHERE p2.game_id = pitcher_game_logs.game_id
                )
                GROUP BY game_id
            ) starter ON starter.game_id = bgl.game_id
            JOIN players p_pitcher ON p_pitcher.player_id = starter.pid
            WHERE bgl.date LIKE ?
              AND p_pitcher.throws IN ('L', 'R')
              AND COALESCE(bgl.plate_appearances, bgl.at_bats, 0) > 0
            GROUP BY bgl.player_id, p_pitcher.throws
            """,
            (year_prefix,),
        ).fetchall()
    except Exception as e:
        logger.warning("platoon splits query failed: %s", e)
        return 0

    count = 0
    for row in rows:
        player_id = row["player_id"]
        vs_hand = row["vs_hand"]
        total_pa = int(row["total_pa"] or 0)
        if total_pa == 0:
            continue

        stat_totals = {
            "batter_hits":        int(row["total_hits"] or 0),
            "batter_home_runs":   int(row["total_hr"] or 0),
            "batter_total_bases": int(row["total_tb"] or 0),
        }
        for market, total in stat_totals.items():
            rate = total / total_pa
            conn.execute(
                """INSERT INTO batter_platoon_splits
                   (player_id, season, vs_hand, market, rate_per_pa, n_pa, updated_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?)
                   ON CONFLICT(player_id, season, vs_hand, market) DO UPDATE SET
                       rate_per_pa = excluded.rate_per_pa,
                       n_pa        = excluded.n_pa,
                       updated_at  = excluded.updated_at""",
                (player_id, season, vs_hand, market, round(rate, 6), total_pa, now),
            )
            count += 1

    return count


def _upsert_pitcher_stats(
    conn: Any, df: pd.DataFrame, mlb_to_bdl: dict, name_to_bdl: dict, season: int, now: str
) -> int:
    if df.empty or 'player_id' not in df.columns:
        return 0
    count = 0
    for _, row in df.iterrows():
        try:
            mlb_id = int(row['player_id'])
        except (TypeError, ValueError):
            continue
        bdl_id = mlb_to_bdl.get(mlb_id)
        if not bdl_id:
            # Fallback to name matching
            raw_name = row.get('last_name,_first_name')
            if pd.isna(raw_name):
                continue
            
            parts = str(raw_name).split(', ')
            if len(parts) == 2:
                parsed_name = f"{parts[1]} {parts[0]}".lower()
            else:
                parsed_name = str(raw_name).lower()
                
            bdl_id = name_to_bdl.get(parsed_name)
            if not bdl_id:
                continue
            
            # Backfill mlb_id in players table to speed up future runs
            conn.execute("UPDATE players SET mlb_id = ? WHERE player_id = ?", (mlb_id, bdl_id))
            mlb_to_bdl[mlb_id] = bdl_id

        conn.execute("""
            INSERT INTO statcast_pitcher_stats
                (player_id, mlb_player_id, season,
                 whiff_pct, chase_rate, barrel_pct_against,
                 hard_hit_pct_against, spin_rate_ff,
                 avg_exit_velocity_against, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(player_id, season) DO UPDATE SET
                mlb_player_id            = excluded.mlb_player_id,
                whiff_pct                = COALESCE(excluded.whiff_pct, whiff_pct),
                chase_rate               = COALESCE(excluded.chase_rate, chase_rate),
                barrel_pct_against       = COALESCE(excluded.barrel_pct_against, barrel_pct_against),
                hard_hit_pct_against     = COALESCE(excluded.hard_hit_pct_against, hard_hit_pct_against),
                spin_rate_ff             = COALESCE(excluded.spin_rate_ff, spin_rate_ff),
                avg_exit_velocity_against = COALESCE(excluded.avg_exit_velocity_against,
                                                     avg_exit_velocity_against),
                updated_at               = excluded.updated_at
        """, (
            bdl_id, mlb_id, season,
            _get_val(row, _PITCHER_ALIASES["whiff_pct"]),
            _get_val(row, _PITCHER_ALIASES["chase_rate"]),
            _get_val(row, _PITCHER_ALIASES["barrel_pct_against"]),
            _get_val(row, _PITCHER_ALIASES["hard_hit_pct_against"]),
            _get_val(row, _PITCHER_ALIASES["spin_rate_ff"]),
            _get_val(row, _PITCHER_ALIASES["avg_exit_velocity_against"]),
            now,
        ))
        count += 1
    return count


def _upsert_batter_stats(
    conn: Any, df: pd.DataFrame, mlb_to_bdl: dict, name_to_bdl: dict, season: int, now: str
) -> int:
    if df.empty or 'player_id' not in df.columns:
        return 0
    count = 0
    for _, row in df.iterrows():
        try:
            mlb_id = int(row['player_id'])
        except (TypeError, ValueError):
            continue
        bdl_id = mlb_to_bdl.get(mlb_id)
        if not bdl_id:
            # Fallback to name matching
            raw_name = row.get('last_name,_first_name')
            if pd.isna(raw_name):
                continue
            
            parts = str(raw_name).split(', ')
            if len(parts) == 2:
                parsed_name = f"{parts[1]} {parts[0]}".lower()
            else:
                parsed_name = str(raw_name).lower()
                
            bdl_id = name_to_bdl.get(parsed_name)
            if not bdl_id:
                continue
            
            # Backfill mlb_id in players table to speed up future runs
            conn.execute("UPDATE players SET mlb_id = ? WHERE player_id = ?", (mlb_id, bdl_id))
            mlb_to_bdl[mlb_id] = bdl_id

        conn.execute("""
            INSERT INTO statcast_batter_stats
                (player_id, mlb_player_id, season,
                 exit_velocity_avg, launch_angle_avg, barrel_pct,
                 xwoba, sprint_speed, whiff_pct, hard_hit_pct, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(player_id, season) DO UPDATE SET
                mlb_player_id     = excluded.mlb_player_id,
                exit_velocity_avg = COALESCE(excluded.exit_velocity_avg, exit_velocity_avg),
                launch_angle_avg  = COALESCE(excluded.launch_angle_avg, launch_angle_avg),
                barrel_pct        = COALESCE(excluded.barrel_pct, barrel_pct),
                xwoba             = COALESCE(excluded.xwoba, xwoba),
                sprint_speed      = COALESCE(excluded.sprint_speed, sprint_speed),
                whiff_pct         = COALESCE(excluded.whiff_pct, whiff_pct),
                hard_hit_pct      = COALESCE(excluded.hard_hit_pct, hard_hit_pct),
                updated_at        = excluded.updated_at
        """, (
            bdl_id, mlb_id, season,
            _get_val(row, _BATTER_ALIASES["exit_velocity_avg"]),
            _get_val(row, _BATTER_ALIASES["launch_angle_avg"]),
            _get_val(row, _BATTER_ALIASES["barrel_pct"]),
            _get_val(row, _BATTER_ALIASES["xwoba"]),
            _get_val(row, _BATTER_ALIASES["sprint_speed"]),
            _get_val(row, _BATTER_ALIASES["whiff_pct"]),
            _get_val(row, _BATTER_ALIASES["hard_hit_pct"]),
            now,
        ))
        count += 1
    return count
