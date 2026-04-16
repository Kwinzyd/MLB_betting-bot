"""Negative-binomial dispersion and fitted-σ helpers.

Parametrization: NB(n, p) with n = 1/α, p = 1/(1 + α·μ).
Variance = μ + α·μ².  When α → 0 the NB degenerates to Poisson.

All functions are pure math — no I/O, no DB access.
"""
from __future__ import annotations

import math

import numpy as np
from scipy.stats import nbinom, poisson, norm


def fit_negbin_mom(counts: np.ndarray, means: np.ndarray) -> float:
    """Method-of-moments α from observed counts and predicted means.

    α = max(0, (Var(y - μ̂) - mean(μ̂)) / mean(μ̂)²)

    Returns 0.0 when the data looks equi- or under-dispersed.
    """
    if len(counts) < 2 or len(means) < 2:
        return 0.0
    residuals = counts.astype(float) - means.astype(float)
    var_resid = float(np.var(residuals, ddof=1))
    mean_pred = float(np.mean(means))
    if mean_pred <= 0:
        return 0.0
    alpha = (var_resid - mean_pred) / (mean_pred ** 2)
    return max(0.0, alpha)


def fit_normal_residual_std(values: np.ndarray, means: np.ndarray) -> float:
    """Population stdev of (y - μ̂) for Normal-distributed markets."""
    if len(values) < 2:
        return 0.0
    residuals = values.astype(float) - means.astype(float)
    return float(np.std(residuals, ddof=0))


def shrink(per_entity: float, pooled: float, n_obs: int, k: int = 30) -> float:
    """Empirical-Bayes shrinkage toward pooled value.

    At n_obs == k the weight is 50/50; as n_obs → ∞ the entity value dominates.
    """
    w = n_obs / (n_obs + k)
    return w * per_entity + (1.0 - w) * pooled


def negbin_prob_over(mean: float, line: float, alpha: float) -> tuple[float, float]:
    """P(over) and P(under) via negative binomial, with Poisson fallback."""
    if mean <= 0:
        return 0.0, 1.0
    if not math.isfinite(alpha) or alpha <= 0:
        p_over = float(1.0 - poisson.cdf(math.floor(line), mean))
        p_under = float(poisson.cdf(math.floor(line), mean))
        return p_over, p_under
    n = 1.0 / alpha
    p = 1.0 / (1.0 + alpha * mean)
    p_over = float(1.0 - nbinom.cdf(math.floor(line), n, p))
    p_under = float(nbinom.cdf(math.floor(line), n, p))
    return p_over, p_under


def normal_prob_over(mean: float, line: float, std: float) -> tuple[float, float]:
    """P(over) and P(under) via Normal distribution with given σ."""
    if mean <= 0:
        return 0.0, 1.0
    std = max(0.5, std)
    p_over = float(norm.sf(line, loc=mean, scale=std))
    p_under = float(norm.cdf(line, loc=mean, scale=std))
    return p_over, p_under
