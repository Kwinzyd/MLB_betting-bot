import pytest

from src.config import LINEUP_PA_MAP, LEAGUE_AVG_GAME_TOTAL
from src.models.pa_estimator import (
    estimate_pa_distribution,
    expected_pa,
    implied_team_total,
)
from src.models.distributions import (
    get_probabilities,
    get_probabilities_mixture,
)


@pytest.mark.parametrize("slot", list(range(1, 10)))
@pytest.mark.parametrize("total", [7.0, 8.5, 10.5])
def test_distribution_sums_to_one(slot, total):
    dist = estimate_pa_distribution(slot, total / 2)
    assert abs(sum(dist.values()) - 1.0) < 1e-9


@pytest.mark.parametrize("slot", list(range(1, 10)))
def test_expected_pa_matches_legacy_at_league_avg(slot):
    dist = estimate_pa_distribution(slot, LEAGUE_AVG_GAME_TOTAL / 2)
    assert abs(expected_pa(dist) - LINEUP_PA_MAP[slot]) < 0.05


def test_monotonic_in_team_total():
    low = expected_pa(estimate_pa_distribution(1, 3.5))
    mid = expected_pa(estimate_pa_distribution(1, 4.5))
    high = expected_pa(estimate_pa_distribution(1, 5.5))
    assert low < mid < high


def test_mixture_diverges_from_point_estimate():
    dist = estimate_pa_distribution(4, 5.5)
    per_pa_rate = 0.7
    line = 1.5
    market = "batter_hits"
    point_mean = per_pa_rate * expected_pa(dist)
    point_over, _ = get_probabilities(point_mean, line, market)
    mix_over, _ = get_probabilities_mixture(per_pa_rate, line, market, dist)
    assert abs(mix_over - point_over) >= 0.005


def test_implied_team_total_fallback():
    assert implied_team_total(None) == LEAGUE_AVG_GAME_TOTAL / 2
    assert implied_team_total(9.0) == 4.5
