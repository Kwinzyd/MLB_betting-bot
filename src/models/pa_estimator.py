"""Dynamic plate-appearance distribution estimator.

Maps (lineup_position, implied_team_total) to a probability mass function over
PA counts. Calibrated so the mean matches the legacy LINEUP_PA_MAP lookup when
implied_team_total equals the league average (game_total / 2).
"""
from __future__ import annotations

from math import exp, factorial
from typing import Dict, Optional

from src.config import (
    LINEUP_PA_MAP,
    DEFAULT_PROJECTED_PA,
    LEAGUE_AVG_GAME_TOTAL,
    PA_ELASTICITY_TO_TOTAL,
    PA_DIST_SUPPORT,
    PA_DIST_ANCHOR,
    PA_MONEYLINE_TILT_K,
)


def _shifted_mean(anchor: float, implied_team_total: float) -> float:
    """Apply game-total elasticity to the anchor PA mean.

    Mirrors src/models/projections.py:_scale_pa_for_game_total so the new
    estimator collapses to legacy behavior when totals are average.
    """
    pct_deviation = (implied_team_total * 2 - LEAGUE_AVG_GAME_TOTAL) / LEAGUE_AVG_GAME_TOTAL
    scale_factor = 1.0 + PA_ELASTICITY_TO_TOTAL * pct_deviation
    scale_factor = max(0.90, min(1.15, scale_factor))
    return anchor * scale_factor


def _truncated_pmf(lam: float) -> Dict[int, float]:
    raw: Dict[int, float] = {}
    for k in PA_DIST_SUPPORT:
        offset = k - PA_DIST_ANCHOR
        if offset < 0:
            raw[k] = 0.0
            continue
        raw[k] = exp(-lam) * (lam ** offset) / factorial(offset)
    total = sum(raw.values())
    if total <= 0:
        return {k: 0.0 for k in PA_DIST_SUPPORT}
    return {k: v / total for k, v in raw.items()}


def _solve_lambda(target_mean: float) -> float:
    """Find lambda such that the truncated shifted-Poisson PMF has the target mean."""
    lo_k, hi_k = min(PA_DIST_SUPPORT), max(PA_DIST_SUPPORT)
    if target_mean <= lo_k:
        return 0.0
    if target_mean >= hi_k:
        return 50.0
    lo, hi = 0.0, 50.0
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        pmf = _truncated_pmf(mid)
        m = sum(k * p for k, p in pmf.items())
        if m < target_mean:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def estimate_pa_distribution(lineup_position: Optional[int],
                             implied_team_total: float) -> Dict[int, float]:
    """Return a PMF over PA counts for the given lineup slot and implied total.

    Uses a shifted-Poisson truncated to PA_DIST_SUPPORT. The Poisson rate is
    solved numerically so the resulting PMF mean matches the elasticity-shifted
    anchor exactly (avoids tail-truncation bias for high-slot anchors).
    """
    anchor = LINEUP_PA_MAP.get(lineup_position, DEFAULT_PROJECTED_PA)
    mean = _shifted_mean(anchor, implied_team_total)
    lo_k, hi_k = min(PA_DIST_SUPPORT), max(PA_DIST_SUPPORT)
    mean = max(float(lo_k), min(float(hi_k), mean))
    lam = _solve_lambda(mean)
    pmf = _truncated_pmf(lam)
    if sum(pmf.values()) <= 0:
        nearest = min(PA_DIST_SUPPORT, key=lambda k: abs(k - mean))
        return {k: (1.0 if k == nearest else 0.0) for k in PA_DIST_SUPPORT}
    return pmf


def expected_pa(dist: Dict[int, float]) -> float:
    return sum(k * p for k, p in dist.items())


def _american_to_prob(ml) -> Optional[float]:
    """American odds -> implied probability (with vig). None on bad input."""
    if ml in (None, ""):
        return None
    try:
        ml = float(ml)
    except (TypeError, ValueError):
        return None
    if ml < 0:
        return (-ml) / ((-ml) + 100.0)
    return 100.0 / (ml + 100.0)


def implied_team_total(game_total: Optional[float],
                       moneyline_home: Optional[int] = None,
                       moneyline_away: Optional[int] = None,
                       side: str = "home") -> float:
    """Translate a game total into the per-team implied run total for `side`.

    Without moneylines, splits the total evenly (the historical behavior, kept
    for callers that don't have prices). With both moneylines, tilts the split
    toward the favorite: the no-vig win probability nudges the team's share of
    the total around 0.5, bounded so a heavy favorite can't run away with it.
    """
    gt = LEAGUE_AVG_GAME_TOTAL if game_total is None else game_total

    ph = _american_to_prob(moneyline_home)
    pa = _american_to_prob(moneyline_away)
    if ph is None or pa is None or (ph + pa) <= 0:
        return gt / 2.0

    p_home = ph / (ph + pa)                       # no-vig home win prob
    p_side = p_home if side == "home" else (1.0 - p_home)
    share = 0.5 + PA_MONEYLINE_TILT_K * (p_side - 0.5)
    share = max(0.40, min(0.60, share))           # guardrail on extreme favorites
    return gt * share


def estimate_remaining_pa_distribution(
    current_inning: int,
    current_outs: int,
    current_batter_slot: int,
    target_batter_slot: int
) -> Dict[int, float]:
    """Calculate remaining PA distribution from live game state.
    
    Maps the remaining outs in the game to the expected number of remaining
    plate appearances for a specific batter slot. Uses a Poisson distribution
    around the continuous target mean to capture tail probabilities.
    """
    left_outs = 27 - ((current_inning - 1) * 3 + current_outs)
    if left_outs <= 0:
        return {0: 1.0}
        
    # Average team ~38 PAs per 27 outs
    team_pa_rem = left_outs * (38.0 / 27.0)
    slots_until = (target_batter_slot - current_batter_slot) % 9
    
    if slots_until == 0:
        # Batter is currently up (guaranteed 1 PA now, next one in 9 team PAs)
        continuous_mean = 1.0 + max(0.0, team_pa_rem - 9.0) / 9.0
    else:
        # Batter must wait slots_until team PAs
        continuous_mean = max(0.0, team_pa_rem - slots_until) / 9.0

    pmf = {}
    for k in range(0, 7):
        pmf[k] = exp(-continuous_mean) * (continuous_mean ** k) / factorial(k)
        
    total = sum(pmf.values())
    return {k: v / total for k, v in pmf.items()}

