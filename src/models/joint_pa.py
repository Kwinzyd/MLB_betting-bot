"""Joint plate-appearance distribution for same-team batters.

A team's total batters faced (TBF) deterministically pins every lineup slot's
PA count: slot s gets PA = (TBF + 9 - s) // 9 for s in 1..9. So same-team
slots are perfectly comonotone given TBF, which makes per-slot marginals
inadequate for SGP joint probabilities.

This module models TBF as a Poisson latent, calibrated so the slot marginal
recovered by mapping TBF -> slot_pa_given_tbf agrees with the existing
estimate_pa_distribution from src/models/pa_estimator.py. The joint PA PMF
then falls out by direct enumeration over TBF support.
"""
from __future__ import annotations

from math import exp, lgamma, log
from typing import Dict, Tuple

from src.config import (
    JOINT_PA_BOOST_CAP,
    PA_DIST_SUPPORT,
)
from src.models.distributions import get_probabilities
from src.models.pa_estimator import (
    estimate_pa_distribution,
    expected_pa,
)


# TBF support is bounded by PA_DIST_SUPPORT: slot s gets PA in [k_lo, k_hi]
# means TBF in [9*(k_lo-1)+1, 9*k_hi]. With PA_DIST_SUPPORT=(3..7) that's
# TBF in [19, 63] — wide enough to cover any realistic game.
_PA_LO = min(PA_DIST_SUPPORT)
_PA_HI = max(PA_DIST_SUPPORT)
_TBF_LO = 9 * (_PA_LO - 1) + 1   # 19
_TBF_HI = 9 * _PA_HI              # 63


def slot_pa_given_tbf(slot: int, tbf: int) -> int:
    """Deterministic PA count for a lineup slot given team TBF.

    Clamped to PA_DIST_SUPPORT so out-of-support TBF realizations land on the
    truncation endpoints — matches estimate_pa_distribution's truncation.
    """
    raw = (tbf + 9 - slot) // 9
    return max(_PA_LO, min(_PA_HI, raw))


def _poisson_pmf(lam: float, k: int) -> float:
    if lam <= 0:
        return 1.0 if k == 0 else 0.0
    return exp(k * log(lam) - lam - lgamma(k + 1))


def _truncated_tbf_pmf(lam: float) -> Dict[int, float]:
    raw: Dict[int, float] = {}
    for t in range(_TBF_LO, _TBF_HI + 1):
        raw[t] = _poisson_pmf(lam, t)
    z = sum(raw.values())
    if z <= 0:
        return {t: 0.0 for t in raw}
    return {t: v / z for t, v in raw.items()}


def _slot_marginal_from_tbf(tbf_pmf: Dict[int, float], slot: int) -> Dict[int, float]:
    out: Dict[int, float] = {k: 0.0 for k in PA_DIST_SUPPORT}
    for t, p in tbf_pmf.items():
        if p <= 0:
            continue
        k = slot_pa_given_tbf(slot, t)
        if k in out:
            out[k] += p
    return out


def _solve_lambda_for_total(implied_team_total: float) -> float:
    """Find λ_TBF such that the team's expected total PAs across all 9 slots
    recovered from _truncated_tbf_pmf matches the sum of per-slot means from
    estimate_pa_distribution.

    Per-slot Poissons can't be matched exactly by a single TBF latent (the
    slot-1-vs-slot-9 spacing under the deterministic floor mapping is 8/9,
    while LINEUP_PA_MAP spacing is 0.80). Calibrating on the team total
    minimizes total mean drift; per-slot drift is bounded by ~0.2 PA at the
    extremes which is well below the per-batter prop signal.
    """
    target = sum(expected_pa(estimate_pa_distribution(s, implied_team_total))
                 for s in range(1, 10))
    lo, hi = float(_TBF_LO), float(_TBF_HI)
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        pmf = _truncated_tbf_pmf(mid)
        team_mean = sum(
            sum(k * p for k, p in _slot_marginal_from_tbf(pmf, s).items())
            for s in range(1, 10)
        )
        if team_mean < target:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def tbf_distribution(implied_team_total: float) -> Dict[int, float]:
    """PMF over team TBF, calibrated to the slot-1 marginal mean."""
    lam = _solve_lambda_for_total(implied_team_total)
    return _truncated_tbf_pmf(lam)


def joint_pa_distribution(slot_a: int, slot_b: int,
                          implied_team_total: float) -> Dict[Tuple[int, int], float]:
    """P(PA_a=k_a, PA_b=k_b) over PA_DIST_SUPPORT × PA_DIST_SUPPORT."""
    tbf = tbf_distribution(implied_team_total)
    out: Dict[Tuple[int, int], float] = {}
    for t, p in tbf.items():
        if p <= 0:
            continue
        k_a = slot_pa_given_tbf(slot_a, t)
        k_b = slot_pa_given_tbf(slot_b, t)
        out[(k_a, k_b)] = out.get((k_a, k_b), 0.0) + p
    return out


def _under_prob(per_pa_rate: float, k: int, line: float, market: str) -> float:
    if k <= 0 or per_pa_rate <= 0:
        return 1.0
    mean = per_pa_rate * k
    _, under = get_probabilities(mean, line, market)
    return under


def pa_correlation_boost(slot_a: int, slot_b: int, implied_team_total: float,
                         line_a: float, mean_a: float, market_a: str,
                         line_b: float, mean_b: float, market_b: str) -> float:
    """Multiplicative correction over the independent UNDER × UNDER product.

    boost = P_joint(both UNDER) / (P_marg_a_under * P_marg_b_under)

    Per-PA rate r_x = mean_x / E[PA_x]; conditional on PA, the count outcome
    is treated as independent (the only correlation channel is PA itself,
    which is the leak this function fixes).
    """
    if slot_a is None or slot_b is None:
        return 1.0
    if mean_a <= 0 or mean_b <= 0:
        return 1.0

    # Per-PA rate is anchored to the original estimator's marginal mean —
    # that's the projection's calibration target. But the boost is a *ratio*
    # of joint-vs-independent, both computed over the SAME TBF-derived
    # marginals so the only term that survives is the PA covariance.
    epa_a = expected_pa(estimate_pa_distribution(slot_a, implied_team_total))
    epa_b = expected_pa(estimate_pa_distribution(slot_b, implied_team_total))
    if epa_a <= 0 or epa_b <= 0:
        return 1.0
    rate_a = mean_a / epa_a
    rate_b = mean_b / epa_b

    tbf_pmf = tbf_distribution(implied_team_total)
    marg_a = _slot_marginal_from_tbf(tbf_pmf, slot_a)
    marg_b = _slot_marginal_from_tbf(tbf_pmf, slot_b)
    p_marg_a = sum(p * _under_prob(rate_a, k, line_a, market_a)
                   for k, p in marg_a.items() if p > 0)
    p_marg_b = sum(p * _under_prob(rate_b, k, line_b, market_b)
                   for k, p in marg_b.items() if p > 0)
    independent = p_marg_a * p_marg_b
    if independent <= 0:
        return 1.0

    joint_pmf: Dict[Tuple[int, int], float] = {}
    for t, p in tbf_pmf.items():
        if p <= 0:
            continue
        k_a = slot_pa_given_tbf(slot_a, t)
        k_b = slot_pa_given_tbf(slot_b, t)
        joint_pmf[(k_a, k_b)] = joint_pmf.get((k_a, k_b), 0.0) + p
    p_joint = 0.0
    for (k_a, k_b), p in joint_pmf.items():
        if p <= 0:
            continue
        p_joint += p * _under_prob(rate_a, k_a, line_a, market_a) \
                     * _under_prob(rate_b, k_b, line_b, market_b)

    if p_joint <= 0:
        return 1.0
    boost = p_joint / independent
    return max(1.0, min(JOINT_PA_BOOST_CAP, boost))
