"""Synthetic market maker.

Inverts three liquid anchor markets — P(hits >= 1), P(total_bases >= 2),
P(home_runs >= 1) — into a per-PA multinomial over {1B, 2B, 3B, HR}, then
re-prices illiquid derived markets (singles, doubles, hits >= 2, etc.) without
spending API calls on additional sharp lines.

Per-PA outcomes are assumed i.i.d. and mixed over the PA distribution from
src/models/pa_estimator.py for consistency with the rest of the projection
stack. Triples are pinned to LEAGUE_AVG_3B_RATE since their anchor would be
too noisy to back out.
"""
from __future__ import annotations

import math
from typing import Dict, Optional, Tuple

from src.config import LEAGUE_AVG_3B_RATE


def _expected_pa_zero_power(rate: float, pa_distribution: Dict[int, float]) -> float:
    """E_k[(1-rate)^k] under the PA distribution."""
    return sum(p_k * (1.0 - rate) ** k for k, p_k in pa_distribution.items())


def _solve_per_pa_rate_from_over(target_over: float,
                                 pa_distribution: Dict[int, float]) -> float:
    """Find per-PA rate r such that 1 - E_k[(1-r)^k] == target_over.

    Monotone in r over [0, 1]; bisection converges in ~50 iterations.
    """
    if target_over <= 0:
        return 0.0
    if target_over >= 1:
        return 1.0
    target_zero = 1.0 - target_over
    lo, hi = 0.0, 1.0
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        if _expected_pa_zero_power(mid, pa_distribution) > target_zero:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def invert_anchors(p_hits_over: float, p_tb_ge_2: float, p_hr_over: float,
                   pa_distribution: Dict[int, float]) -> Optional[Dict[str, float]]:
    """Back out per-PA {1B, 2B, 3B, HR} rates from the three anchor probabilities.

    Returns None when inputs are degenerate (not in (0, 1)).
    """
    for v in (p_hits_over, p_tb_ge_2, p_hr_over):
        if not (0.0 < v < 1.0):
            return None

    p_h = _solve_per_pa_rate_from_over(p_hits_over, pa_distribution)
    p_hr = _solve_per_pa_rate_from_over(p_hr_over, pa_distribution)

    # P(TB == 1) = E_k[k * p1 * (1-p_h)^(k-1)] when no 2B/3B/HR occurred.
    # Solving 1 - E[(1-p_h)^k] - E[k*p1*(1-p_h)^(k-1)] = p_tb_ge_2 for p1.
    e_zero = _expected_pa_zero_power(p_h, pa_distribution)
    e_k_zero_minus_one = sum(
        p_k * k * (1.0 - p_h) ** (k - 1)
        for k, p_k in pa_distribution.items() if k >= 1
    )
    if e_k_zero_minus_one <= 0:
        return None
    target_p_tb_le_1 = 1.0 - p_tb_ge_2
    p1_numerator = target_p_tb_le_1 - e_zero
    p_1b = p1_numerator / e_k_zero_minus_one

    p_3b = LEAGUE_AVG_3B_RATE
    p_2b = p_h - p_1b - p_3b - p_hr

    p_1b = max(0.0, min(p_h, p_1b))
    # Floor p_2b at 0.01 to avoid negatives from heavily juiced anchor lines
    p_2b = max(0.01, min(p_h, p_2b))
    p_3b = max(0.0, min(p_h, p_3b))
    p_hr = max(0.0, min(p_h, p_hr))

    total = p_1b + p_2b + p_3b + p_hr
    if total > 0 and abs(total - p_h) > 1e-9:
        scale = p_h / total
        p_1b *= scale
        p_2b *= scale
        p_3b *= scale
        p_hr *= scale

    return {
        "p_any_hit": p_h,
        "p_1b": p_1b,
        "p_2b": p_2b,
        "p_3b": p_3b,
        "p_hr": p_hr,
    }


def _binomial_cdf(k: int, n: int, p: float) -> float:
    """P(X <= k) where X ~ Binomial(n, p). No scipy dependency."""
    if p <= 0:
        return 1.0
    if p >= 1:
        return 0.0 if k < n else 1.0
    cdf = 0.0
    log_pmf = n * math.log(1.0 - p)  # term for j=0
    cdf += math.exp(log_pmf)
    for j in range(1, min(k, n) + 1):
        # PMF(j) = PMF(j-1) * (n-j+1)/j * p/(1-p)
        log_pmf += math.log((n - j + 1) / j) + math.log(p / (1.0 - p))
        cdf += math.exp(log_pmf)
    return min(1.0, cdf)


def _prob_at_least_n(rate: float, n: int,
                     pa_distribution: Dict[int, float]) -> float:
    """E_k[P(Bin(k, rate) >= n)] under the PA distribution."""
    if n <= 0:
        return 1.0
    if rate <= 0:
        return 0.0
    total = 0.0
    for k, p_k in pa_distribution.items():
        if p_k <= 0 or k < n:
            continue
        total += p_k * (1.0 - _binomial_cdf(n - 1, k, rate))
    return min(1.0, max(0.0, total))


_RATE_KEY_BY_MARKET = {
    "batter_singles": "p_1b",
    "batter_doubles": "p_2b",
    "batter_triples": "p_3b",
    "batter_home_runs": "p_hr",
    "batter_hits": "p_any_hit",
}


def price_market(per_pa_rates: Dict[str, float], line: float, market: str,
                 pa_distribution: Dict[int, float]) -> Optional[Tuple[float, float]]:
    """Return (prob_over, prob_under) for a derived count market at the given line."""
    rate_key = _RATE_KEY_BY_MARKET.get(market)
    if rate_key is None:
        return None
    rate = per_pa_rates.get(rate_key)
    if rate is None:
        return None
    threshold = math.floor(line) + 1
    prob_over = _prob_at_least_n(rate, threshold, pa_distribution)
    return prob_over, 1.0 - prob_over


def synthetic_price(p_hits_over: float, p_tb_ge_2: float, p_hr_over: float,
                    pa_distribution: Dict[int, float],
                    market: str, line: float) -> Optional[Tuple[float, float]]:
    """Convenience: invert anchors then price a derived market in one call."""
    rates = invert_anchors(p_hits_over, p_tb_ge_2, p_hr_over, pa_distribution)
    if rates is None:
        return None
    return price_market(rates, line, market, pa_distribution)
