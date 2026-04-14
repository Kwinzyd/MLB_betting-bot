"""Sanity tests for Monte Carlo prop probability estimation."""
import math
from scipy.stats import poisson

from src.models.monte_carlo import mc_prob_over


def test_poisson_over_matches_cdf_at_large_n():
    mean, line = 6.5, 5.5
    expected = 1.0 - poisson.cdf(math.floor(line), mean)
    prob_over, prob_under = mc_prob_over(mean, line, "pitcher_strikeouts", n_sims=20000)
    assert abs(prob_over - expected) < 0.02
    assert abs((prob_over + prob_under) - 1.0) < 1e-9


def test_zero_mean_returns_floor():
    prob_over, prob_under = mc_prob_over(0.0, 0.5, "batter_hits")
    assert prob_over == 0.0
    assert prob_under == 1.0


def test_total_bases_uses_normal():
    # For a high-variance market mean ~ 1.5 line 1.5, prob_over should sit near 0.5
    prob_over, _ = mc_prob_over(1.5, 1.5, "batter_total_bases", n_sims=20000)
    assert 0.40 <= prob_over <= 0.60


def test_unknown_market_falls_back_to_poisson():
    prob_over, _ = mc_prob_over(3.0, 2.5, "unknown_market", n_sims=20000)
    expected = 1.0 - poisson.cdf(2, 3.0)
    assert abs(prob_over - expected) < 0.03
