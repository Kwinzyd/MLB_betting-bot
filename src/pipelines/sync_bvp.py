"""Sync batter-vs-pitcher head-to-head lines for today's matchups.

For each active game, fetches every rostered/lineup batter's career line vs the
opposing team (BDL /players/versus returns all opposing pitchers in one call),
and upserts into bvp_stats. Read at scan time by bvp.py to build a heavily-shrunk
matchup multiplier.

Scope: batters in daily_lineups for non-final games — the set we actually scan.
One call per (batter, opponent_team). Fail-safe: a per-batter error is skipped.
"""
from __future__ import annotations

from typing import Dict, Optional

from src.data.db import get_db_connection
from src.utils.logging_utils import get_logger
from src.utils.time_utils import get_utc_now_iso

logger = get_logger(__name__)


def _num(v) -> int:
    try:
        return int(v)
    except (TypeError, ValueError):
        return 0


def parse_versus_row(row: Dict) -> Optional[Dict]:
    """Extract (pitcher_id, stat totals) from one /players/versus row.

    TB = hits + doubles + 2*triples + 3*home_runs. PA ≈ ab + walks. Returns
    None when the opposing pitcher id or AB is missing.
    """
    opp = row.get("opponent_player") or {}
    pitcher_id = opp.get("id")
    if not pitcher_id:
        return None
    ab = _num(row.get("at_bats"))
    if ab <= 0:
        return None
    h = _num(row.get("hits"))
    d2 = _num(row.get("doubles"))
    d3 = _num(row.get("triples"))
    hr = _num(row.get("home_runs"))
    bb = _num(row.get("walks"))
    return {
        "pitcher_id": pitcher_id,
        "ab": ab,
        "pa": ab + bb,
        "hits": h,
        "total_bases": h + d2 + (2 * d3) + (3 * hr),
        "home_runs": hr,
        "strikeouts": _num(row.get("strikeouts")),
        "walks": bb,
    }


def _matchups_to_fetch(conn) -> list:
    """(batter_id, opponent_team_id) pairs for batters in today's lineups.

    Resolves each batter's team via players.team_id and pairs it with the
    game's other team. Skips rows we can't attribute to one side of the game.
    """
    rows = conn.execute(
        """SELECT DISTINCT dl.player_id AS batter_id, p.team_id AS team_id,
                  g.home_team_id, g.away_team_id
           FROM daily_lineups dl
           JOIN players p ON p.player_id = dl.player_id
           JOIN games   g ON g.game_id  = dl.game_id
           WHERE g.status NOT IN ('COMPLETED', 'POSTPONED')
             AND p.team_id IS NOT NULL
             AND g.home_team_id IS NOT NULL AND g.away_team_id IS NOT NULL"""
    ).fetchall()
    pairs = []
    for r in rows:
        if r["team_id"] == r["home_team_id"]:
            opp = r["away_team_id"]
        elif r["team_id"] == r["away_team_id"]:
            opp = r["home_team_id"]
        else:
            continue  # batter's team isn't in this game — skip rather than guess
        pairs.append((r["batter_id"], opp))
    return sorted(set(pairs))


async def sync_bvp(client=None) -> int:
    """Fetch and upsert BvP lines for today's matchups. Returns rows upserted."""
    from src.clients.mlb_stats import MLBStatsClient
    client = client or MLBStatsClient(requests_per_second=5.0)
    now = get_utc_now_iso()

    with get_db_connection() as conn:
        pairs = _matchups_to_fetch(conn)

    if not pairs:
        logger.info("sync_bvp: no active lineups to fetch BvP for.")
        return 0

    upserted = 0
    with get_db_connection() as conn:
        for batter_id, opp_team_id in pairs:
            try:
                rows = await client.get_player_versus(batter_id, opp_team_id)
            except Exception as e:  # noqa: BLE001 — one batter must not abort the batch
                logger.debug("BvP fetch failed for batter %s vs team %s: %s",
                             batter_id, opp_team_id, e)
                continue
            for raw in rows or []:
                parsed = parse_versus_row(raw)
                if not parsed:
                    continue
                conn.execute(
                    """INSERT INTO bvp_stats
                       (batter_id, pitcher_id, ab, pa, hits, total_bases,
                        home_runs, strikeouts, walks, updated_at)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                       ON CONFLICT(batter_id, pitcher_id) DO UPDATE SET
                           ab=excluded.ab, pa=excluded.pa, hits=excluded.hits,
                           total_bases=excluded.total_bases, home_runs=excluded.home_runs,
                           strikeouts=excluded.strikeouts, walks=excluded.walks,
                           updated_at=excluded.updated_at""",
                    (
                        batter_id, parsed["pitcher_id"], parsed["ab"], parsed["pa"],
                        parsed["hits"], parsed["total_bases"], parsed["home_runs"],
                        parsed["strikeouts"], parsed["walks"], now,
                    ),
                )
                upserted += 1
        conn.commit()

    logger.info("sync_bvp: upserted %d BvP rows across %d matchup(s).", upserted, len(pairs))
    return upserted
