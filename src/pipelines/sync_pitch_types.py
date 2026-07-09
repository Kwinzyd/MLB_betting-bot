"""Sync BDL season pitch-type stats into the pitch_type_stats table.

Two bulk pulls per run (all pitchers, all hitters) — the season endpoints
return one row per (player, pitch_type). Feeds pitch_matchup.py, which builds
arsenal-weighted matchup multipliers applied to pitcher-K and batter
projections at scan time.

Runs inside the nightly sync (after sync_stats populates players). Fail-safe:
a fetch failure logs and leaves the prior table intact.
"""
from __future__ import annotations

from typing import Optional

from src.config import MLB_SEASON
from src.data.db import get_db_connection
from src.utils.logging_utils import get_logger
from src.utils.time_utils import get_utc_now_iso

logger = get_logger(__name__)

# BDL percent fields are on a 0-100 scale (e.g. 37.5). Stored as-is.
_ROLES = ("pitcher", "hitter")


def _num(v) -> Optional[float]:
    if v in (None, ""):
        return None
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def _int(v) -> Optional[int]:
    n = _num(v)
    return int(n) if n is not None else None


def _upsert_rows(conn, rows, role: str, season: int, now: str) -> int:
    count = 0
    for r in rows:
        pid = r.get("player_id")
        pitch_type = r.get("pitch_type")
        if not pid or not pitch_type:
            continue
        conn.execute(
            """INSERT INTO pitch_type_stats
               (player_id, season, role, pitch_type, pitch_count, usage_pct,
                whiff_pct, contact_pct, xwoba, pa_count, strikeout_count, updated_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
               ON CONFLICT(player_id, season, role, pitch_type) DO UPDATE SET
                   pitch_count     = excluded.pitch_count,
                   usage_pct       = excluded.usage_pct,
                   whiff_pct       = excluded.whiff_pct,
                   contact_pct     = excluded.contact_pct,
                   xwoba           = excluded.xwoba,
                   pa_count        = excluded.pa_count,
                   strikeout_count = excluded.strikeout_count,
                   updated_at      = excluded.updated_at""",
            (
                pid, season, role, pitch_type,
                _int(r.get("pitch_count")),
                _num(r.get("pitch_usage_percent")),
                _num(r.get("whiff_percent")),
                _num(r.get("contact_percent")),
                _num(r.get("xwoba")),
                _int(r.get("pa_count")),
                _int(r.get("strikeout_count")),
                now,
            ),
        )
        count += 1
    return count


async def sync_pitch_types(season: int = None, client=None) -> int:
    """Bulk-pull pitcher and hitter pitch-type season stats; upsert. Returns
    total rows upserted. Fail-safe: on fetch error for a role, that role is
    skipped and the previous rows remain."""
    season = season or MLB_SEASON
    from src.clients.mlb_stats import MLBStatsClient
    # Higher rps for the bulk paginated pull; the default 0.07 rps (14s/page)
    # would take many minutes across ~150 pages. 5 rps is well within Goat's
    # 600 req/min and matches the other bulk sync pipelines.
    client = client or MLBStatsClient(requests_per_second=5.0)
    now = get_utc_now_iso()

    total = 0
    with get_db_connection() as conn:
        for role in _ROLES:
            try:
                rows = await client.get_pitch_type_season_stats(role, season)
            except Exception as e:  # noqa: BLE001 — one role's failure must not abort the other
                logger.warning("sync_pitch_types: %s fetch failed: %s", role, e)
                continue
            n = _upsert_rows(conn, rows or [], role, season, now)
            total += n
            logger.info("sync_pitch_types: upserted %d %s pitch-type rows", n, role)
        conn.commit()

    logger.info("sync_pitch_types: %d total rows for season %d", total, season)
    return total
