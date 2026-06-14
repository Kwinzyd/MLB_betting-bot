"""Consensus game-level odds from BallDontLie (free, unlimited on Goat tier).

BDL has GAME odds only — moneyline, run line, and total — across several books;
it has NO player props. This module condenses the per-vendor rows into one
consensus quote per game, used to (a) back up the game total when the Odds API
is unavailable and (b) supply moneylines for moneyline-tilted implied team
totals in PA scaling. It never sources player-prop prices.
"""
from __future__ import annotations

from statistics import median
from typing import Optional

from src.clients.mlb_stats import MLBStatsClient
from src.utils.logging_utils import get_logger

logger = get_logger(__name__)


def _median_or_none(values):
    nums = []
    for v in values:
        if v in (None, ""):
            continue
        try:
            nums.append(float(v))
        except (TypeError, ValueError):
            continue
    return median(nums) if nums else None


async def get_game_market(bdl_game_id, client: MLBStatsClient = None) -> Optional[dict]:
    """Return consensus {total, ml_home, ml_away} for a BDL game, or None.

    Medians across all vendors that quoted the game. Fail-safe: returns None on
    any error so callers fall back to whatever they had.
    """
    if not bdl_game_id:
        return None
    client = client or MLBStatsClient()
    try:
        records = await client.get_odds(game_ids=bdl_game_id)
    except Exception as e:  # noqa: BLE001 — backup source must never raise
        logger.debug("BDL odds fetch failed for game %s: %s", bdl_game_id, e)
        return None
    if not records:
        return None
    total = _median_or_none(r.get("total_value") for r in records)
    ml_home = _median_or_none(r.get("moneyline_home_odds") for r in records)
    ml_away = _median_or_none(r.get("moneyline_away_odds") for r in records)
    if total is None and ml_home is None and ml_away is None:
        return None
    return {"total": total, "ml_home": ml_home, "ml_away": ml_away}
