import pytest
from src.models.distributions import (
    poisson_prob_over, poisson_prob_under,
    normal_prob_over, normal_prob_under,
    get_probabilities,
)


def test_poisson_probs_sum_to_one():
    mean = 6.0
    line = 5.5
    p_over = poisson_prob_over(mean, line)
    p_under = poisson_prob_under(mean, line)
    assert abs(p_over + p_under - 1.0) < 0.001


def test_poisson_high_mean_favors_over():
    # Mean of 8 strikeouts, line at 5.5 -> should heavily favor over
    p_over = poisson_prob_over(8.0, 5.5)
    assert p_over > 0.75


def test_poisson_low_mean_favors_under():
    # Mean of 3 strikeouts, line at 5.5 -> should heavily favor under
    p_under = poisson_prob_under(3.0, 5.5)
    assert p_under > 0.85


def test_poisson_rare_event():
    # Home runs: mean of 0.15, line at 0.5
    p_over = poisson_prob_over(0.15, 0.5)
    p_under = poisson_prob_under(0.15, 0.5)
    assert p_under > 0.85  # Should heavily favor under
    assert abs(p_over + p_under - 1.0) < 0.001


def test_normal_probs_sum_to_one():
    mean = 2.5
    std = 1.5
    line = 2.5
    p_over = normal_prob_over(mean, std, line)
    p_under = normal_prob_under(mean, std, line)
    assert abs(p_over + p_under - 1.0) < 0.001


def test_normal_at_mean():
    # Line equals mean -> ~50/50
    p_over = normal_prob_over(2.5, 1.5, 2.5)
    assert abs(p_over - 0.5) < 0.01


def test_get_probabilities_pitcher_strikeouts():
    p_over, p_under = get_probabilities(6.5, 5.5, "pitcher_strikeouts")
    assert p_over > 0.5  # Mean above line
    assert abs(p_over + p_under - 1.0) < 0.001


def test_get_probabilities_batter_total_bases():
    p_over, p_under = get_probabilities(1.8, 1.5, "batter_total_bases")
    assert p_over > 0.4  # Mean above line, but wider distribution
    assert abs(p_over + p_under - 1.0) < 0.001


def test_get_probabilities_zero_mean():
    p_over, p_under = get_probabilities(0.0, 0.5, "batter_home_runs")
    assert p_over == 0.0
    assert p_under == 1.0
