"""Tests for negative-binomial dispersion and fitted-σ helpers."""
import math
import numpy as np
import pytest
from scipy.stats import poisson, norm

from src.models.dispersion import (
    fit_negbin_mom,
    fit_normal_residual_std,
    shrink,
    negbin_prob_over,
    normal_prob_over,
)


def test_mom_alpha_zero_when_variance_equals_mean():
    rng = np.random.default_rng(42)
    lam = 6.0
    counts = rng.poisson(lam, size=5000)
    means = np.full_like(counts, lam, dtype=float)
    alpha = fit_negbin_mom(counts, means)
    assert alpha == pytest.approx(0.0, abs=0.05)


def test_mom_alpha_positive_for_overdispersed_sample():
    rng = np.random.default_rng(42)
    true_alpha = 0.5
    lam = 6.0
    n = 1.0 / true_alpha
    p = 1.0 / (1.0 + true_alpha * lam)
    counts = rng.negative_binomial(n, p, size=5000)
    means = np.full_like(counts, lam, dtype=float)
    alpha = fit_negbin_mom(counts, means)
    assert alpha > 0.2
    assert alpha == pytest.approx(true_alpha, abs=0.15)


def test_mom_returns_zero_for_short_arrays():
    assert fit_negbin_mom(np.array([5]), np.array([5.0])) == 0.0
    assert fit_negbin_mom(np.array([]), np.array([])) == 0.0


def test_normal_residual_std_matches_numpy():
    rng = np.random.default_rng(99)
    means = rng.uniform(1, 4, size=200)
    values = means + rng.normal(0, 1.5, size=200)
    sigma = fit_normal_residual_std(values, means)
    assert sigma == pytest.approx(1.5, abs=0.2)


def test_shrink_collapses_to_pool_when_n_small():
    result = shrink(per_entity=2.0, pooled=0.5, n_obs=1, k=30)
    assert result == pytest.approx(0.5, abs=0.1)


def test_shrink_approaches_entity_when_n_large():
    result = shrink(per_entity=2.0, pooled=0.5, n_obs=1000, k=30)
    assert result == pytest.approx(2.0, abs=0.1)


def test_shrink_fifty_fifty_at_k():
    result = shrink(per_entity=2.0, pooled=0.5, n_obs=30, k=30)
    assert result == pytest.approx(1.25, abs=1e-9)


def test_negbin_prob_over_matches_poisson_when_alpha_zero():
    mean = 6.0
    line = 5.5
    p_over_nb, p_under_nb = negbin_prob_over(mean, line, alpha=0.0)
    p_over_pois = float(1.0 - poisson.cdf(5, mean))
    assert p_over_nb == pytest.approx(p_over_pois, abs=1e-6)


def test_negbin_prob_over_widens_tail_vs_poisson():
    mean = 6.0
    line = 10.5  # far tail
    p_over_nb, _ = negbin_prob_over(mean, line, alpha=0.5)
    p_over_pois = float(1.0 - poisson.cdf(10, mean))
    assert p_over_nb > p_over_pois


def test_normal_prob_over_matches_scipy_norm_sf():
    mean = 3.5
    line = 3.5
    std = 1.8
    p_over, p_under = normal_prob_over(mean, line, std)
    assert p_over == pytest.approx(float(norm.sf(line, mean, std)), abs=1e-6)
    assert p_under == pytest.approx(float(norm.cdf(line, mean, std)), abs=1e-6)


def test_negbin_prob_over_handles_nonfinite_alpha():
    mean = 6.0
    line = 5.5
    p_over_inf, _ = negbin_prob_over(mean, line, alpha=float('inf'))
    p_over_nan, _ = negbin_prob_over(mean, line, alpha=float('nan'))
    p_over_neg, _ = negbin_prob_over(mean, line, alpha=-1.0)
    p_over_pois = float(1.0 - poisson.cdf(5, mean))
    assert p_over_inf == pytest.approx(p_over_pois, abs=1e-6)
    assert p_over_nan == pytest.approx(p_over_pois, abs=1e-6)
    assert p_over_neg == pytest.approx(p_over_pois, abs=1e-6)


def test_negbin_prob_over_zero_mean():
    p_over, p_under = negbin_prob_over(0.0, 5.5, alpha=0.5)
    assert p_over == 0.0
    assert p_under == 1.0


def test_normal_prob_over_zero_mean():
    p_over, p_under = normal_prob_over(0.0, 3.5, std=1.5)
    assert p_over == 0.0
    assert p_under == 1.0
