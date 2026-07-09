"""Batter-vs-pitcher (BvP) head-to-head matchup multiplier.

Turns a batter's career line against a specific pitcher (bvp_stats, filled by
sync_bvp from BDL /players/versus) into a bounded multiplier on the batter's
projected mean.

BvP is a weak, small-sample signal — most matchups have well under 15 AB and
little repeatable predictive value. So the factor is:
  - Heavily shrunk toward the batter's OWN baseline rate (BVP_PRIOR_PA pseudo-PA),
    so a handful of AB barely moves it.
  - Gated by a minimum AB floor (below it, neutral 1.0).
  - Tightly clamped (default ±5%), tighter than the pitch-type arsenal factor
    because the two overlap (both describe this batter vs this pitcher).
Fail-safe: any missing row, thin sample, or bad baseline returns 1.0.
"""
from __future__ import annotations

from typing import Dict, Optional

from src.config import BVP_PRIOR_PA, BVP_MIN_AB, BVP_MIN, BVP_MAX
from src.utils.logging_utils import get_logger

logger = get_logger(__name__)

# Market -> the numerator column in bvp_stats.
_MARKET_STAT = {
    "batter_hits": "hits",
    "batter_total_bases": "total_bases",
    "batter_home_runs": "home_runs",
}


def bvp_rate(stats: Dict, market: str) -> Optional[float]:
    """Per-PA rate for a market from a BvP line, or None when unusable.

    PA is the stored `pa` (ab + bb), falling back to ab. Returns None when
    there are no plate appearances or the market isn't a BvP-scored one.
    """
    col = _MARKET_STAT.get(market)
    if not col:
        return None
    pa = stats.get("pa") or stats.get("ab") or 0
    if pa <= 0:
        return None
    num = stats.get(col)
    if num is None:
        return None
    return float(num) / float(pa)


def bvp_factor(stats: Dict, baseline_rate: float, market: str,
               prior_pa: float = None, min_ab: int = None,
               lo: float = None, hi: float = None) -> float:
    """Shrunk, bounded multiplier for a batter's mean vs a specific pitcher.

    shrunk = (pa*bvp_rate + prior_pa*baseline) / (pa + prior_pa)
    factor = clamp(shrunk / baseline, [lo, hi])

    Neutral 1.0 when the sample is below min_ab, the baseline is non-positive,
    or the market carries no BvP rate.
    """
    prior_pa = BVP_PRIOR_PA if prior_pa is None else prior_pa
    min_ab = BVP_MIN_AB if min_ab is None else min_ab
    lo = BVP_MIN if lo is None else lo
    hi = BVP_MAX if hi is None else hi

    if not baseline_rate or baseline_rate <= 0:
        return 1.0
    if (stats.get("ab") or 0) < min_ab:
        return 1.0
    rate = bvp_rate(stats, market)
    if rate is None:
        return 1.0
    pa = stats.get("pa") or stats.get("ab") or 0
    shrunk = (pa * rate + prior_pa * baseline_rate) / (pa + prior_pa)
    factor = shrunk / baseline_rate
    return max(lo, min(hi, factor))


def bvp_factor_db(conn, batter_id: int, pitcher_id: int, market: str,
                  baseline_rate: float) -> float:
    """Load the BvP line and compute the factor. Neutral 1.0 on any gap."""
    if not conn or not batter_id or not pitcher_id:
        return 1.0
    try:
        row = conn.execute(
            "SELECT ab, pa, hits, total_bases, home_runs, strikeouts, walks "
            "FROM bvp_stats WHERE batter_id = ? AND pitcher_id = ?",
            (batter_id, pitcher_id),
        ).fetchone()
    except Exception as e:  # noqa: BLE001 — BvP is optional; never break a projection
        logger.debug("bvp_stats load failed (%s vs %s): %s", batter_id, pitcher_id, e)
        return 1.0
    if not row:
        return 1.0
    return bvp_factor(dict(row), baseline_rate, market)
