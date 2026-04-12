"""
sync_umpires pipeline — two-phase run each day:

Phase 1 — update umpire stats (incremental, DB-deduped)
  • Look back UMP_STATS_LOOKBACK_DAYS days for completed games.
  • Skip any mlb_game_pk already stored in umpire_game_assignments (processed flag).
  • For each new completed game: fetch box score, accumulate K/BB into umpire_stats.
  • Recompute k_per_game, bb_per_game, k_factor for affected umpires.

Phase 2 — assign today's umpires
  • Fetch today's MLB schedule with officials.
  • Match each MLB game to our games table by home team name.
  • Upsert into umpire_game_assignments with our game_id so scan_props can look it up.

Why DB-deduplication instead of cache:
  The SimpleCache is in-memory and resets between process runs. Box score API
  calls (one per completed game) must therefore be tracked in the DB to avoid
  refetching on every `python main.py sync` invocation.
"""

from __future__ import annotations

from datetime import date, timedelta

from src.clients.mlb_official_stats import MLBOfficialStatsClient
from src.config import (
    LEAGUE_AVG_BB_PER_GAME,
    LEAGUE_AVG_K_PER_GAME,
    UMP_STATS_LOOKBACK_DAYS,
)
from src.data.db import get_db_connection
from src.utils.logging_utils import get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Pipeline entry point
# ---------------------------------------------------------------------------

def sync_umpires() -> None:
    """Sync umpire stats from recent games and assign today's umpires."""
    logger.info("Executing pipeline: sync_umpires")
    client = MLBOfficialStatsClient()
    _update_umpire_stats(client)
    _assign_todays_umpires(client)


# ---------------------------------------------------------------------------
# Phase 1 — incremental umpire stats
# ---------------------------------------------------------------------------

def _update_umpire_stats(client: MLBOfficialStatsClient) -> None:
    """
    Scan the last UMP_STATS_LOOKBACK_DAYS days for completed games not yet
    recorded. For each new game, fetch the box score and credit its K/BB
    totals to the home-plate umpire.
    """
    already_processed = _get_processed_pks()
    today = date.today()

    # {umpire_id: {name, new_k, new_bb, new_games}}
    delta: dict[int, dict] = {}

    for days_back in range(1, UMP_STATS_LOOKBACK_DAYS + 1):
        target = (today - timedelta(days=days_back)).isoformat()
        games = client.get_games_with_officials(target)

        for game in games:
            if game.get("status") != "Final":
                continue
            game_pk = game.get("gamePk")
            if not game_pk or game_pk in already_processed:
                continue

            hp = _home_plate_umpire(game)
            if not hp:
                continue

            box = client.get_game_ks_and_bbs(game_pk)
            if not box:
                continue

            uid = hp["id"]
            if uid not in delta:
                delta[uid] = {"name": hp["name"], "k": 0, "bb": 0, "games": 0}
            delta[uid]["k"]     += box["strikeouts"]
            delta[uid]["bb"]    += box["walks"]
            delta[uid]["games"] += 1

            # Record this gamePk so we never fetch its box score again.
            # game_id is NULL — this is a historical-only row used as a processing log.
            _record_historical_assignment(game_pk, hp, target)

    if not delta:
        logger.info("No new completed games to update umpire stats from.")
        return

    _apply_stat_delta(delta, today.isoformat())
    total_new_games = sum(d["games"] for d in delta.values())
    logger.info(
        f"Updated umpire stats: {len(delta)} umpires, {total_new_games} new games processed."
    )


def _record_historical_assignment(game_pk: int, hp: dict, date_str: str) -> None:
    """Store a historical game assignment so it is never re-fetched."""
    with get_db_connection() as conn:
        conn.execute(
            """
            INSERT OR IGNORE INTO umpire_game_assignments
                (mlb_game_pk, game_id, umpire_id, umpire_name, date)
            VALUES (?, NULL, ?, ?, ?)
            """,
            (game_pk, hp["id"], hp["name"], date_str),
        )
        conn.commit()


def _apply_stat_delta(delta: dict[int, dict], today: str) -> None:
    """Add new K/BB counts to umpire_stats and recompute derived columns."""
    with get_db_connection() as conn:
        for uid, d in delta.items():
            conn.execute(
                """
                INSERT INTO umpire_stats
                    (umpire_id, umpire_name, games_called,
                     total_strikeouts, total_walks, updated_date)
                VALUES (?, ?, ?, ?, ?, ?)
                ON CONFLICT(umpire_id) DO UPDATE SET
                    umpire_name      = excluded.umpire_name,
                    games_called     = games_called     + excluded.games_called,
                    total_strikeouts = total_strikeouts + excluded.total_strikeouts,
                    total_walks      = total_walks      + excluded.total_walks,
                    updated_date     = excluded.updated_date
                """,
                (uid, d["name"], d["games"], d["k"], d["bb"], today),
            )

        # Recompute per-game rates and k_factor for every umpire that was touched.
        uid_placeholders = ",".join("?" * len(delta))
        conn.execute(
            f"""
            UPDATE umpire_stats SET
                k_per_game  = CAST(total_strikeouts AS REAL) / NULLIF(games_called, 0),
                bb_per_game = CAST(total_walks      AS REAL) / NULLIF(games_called, 0),
                k_factor    = (CAST(total_strikeouts AS REAL) / NULLIF(games_called, 0))
                              / ?
            WHERE umpire_id IN ({uid_placeholders})
            """,
            [LEAGUE_AVG_K_PER_GAME] + list(delta.keys()),
        )
        conn.commit()


# ---------------------------------------------------------------------------
# Phase 2 — today's active assignments
# ---------------------------------------------------------------------------

def _assign_todays_umpires(client: MLBOfficialStatsClient) -> None:
    """
    Fetch today's MLB schedule and match each game to our games table by
    home team name. Write the umpire assignment so scan_props can look it up.
    """
    today = date.today().isoformat()
    mlb_games = client.get_games_with_officials(today)

    if not mlb_games:
        logger.info("No MLB games found for today's umpire assignment.")
        return

    our_game_by_home = _build_home_team_index(today)
    if not our_game_by_home:
        logger.info("No upcoming games in our DB for today — skipping umpire assignment.")
        return

    assigned = 0
    with get_db_connection() as conn:
        for game in mlb_games:
            hp = _home_plate_umpire(game)
            if not hp:
                continue

            game_id = _match_game(game.get("homeTeam", ""), our_game_by_home)
            if not game_id:
                logger.debug(
                    f"No game_id match for MLB home team '{game.get('homeTeam')}'"
                )
                continue

            conn.execute(
                """
                INSERT INTO umpire_game_assignments
                    (mlb_game_pk, game_id, umpire_id, umpire_name, date)
                VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(mlb_game_pk) DO UPDATE SET
                    game_id     = excluded.game_id,
                    umpire_id   = excluded.umpire_id,
                    umpire_name = excluded.umpire_name
                """,
                (game["gamePk"], game_id, hp["id"], hp["name"], today),
            )
            assigned += 1

        conn.commit()

    logger.info(f"Assigned umpires to {assigned}/{len(mlb_games)} of today's games.")


def _build_home_team_index(today: str) -> dict[str, str]:
    """Return {normalised_home_team_name: game_id} for today's upcoming games."""
    with get_db_connection() as conn:
        rows = conn.execute(
            "SELECT game_id, home_team FROM games WHERE date LIKE ?",
            (f"{today}%",),
        ).fetchall()
    return {_norm(r["home_team"]): r["game_id"] for r in rows}


def _match_game(mlb_home: str, index: dict[str, str]) -> str | None:
    """Exact then partial match on normalised team names."""
    key = _norm(mlb_home)
    if key in index:
        return index[key]
    # Partial: accept if one name is a substring of the other
    for our_key, gid in index.items():
        if key in our_key or our_key in key:
            return gid
    return None


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def _home_plate_umpire(game: dict) -> dict | None:
    """Extract {id, name} for the home-plate umpire, or None if not listed."""
    for official in game.get("officials", []):
        if official.get("officialType") == "Home Plate":
            ump = official.get("official", {})
            if ump.get("id") and ump.get("fullName"):
                return {"id": ump["id"], "name": ump["fullName"]}
    return None


def _get_processed_pks() -> set[int]:
    """Return all mlb_game_pks already stored in umpire_game_assignments."""
    with get_db_connection() as conn:
        rows = conn.execute(
            "SELECT mlb_game_pk FROM umpire_game_assignments"
        ).fetchall()
    return {r["mlb_game_pk"] for r in rows}


def _norm(name: str) -> str:
    """Lower-case, strip whitespace for fuzzy team name matching."""
    return name.lower().strip()
