import pytest

from src.models.pa_estimator import estimate_pa_distribution
from src.models.synthetic_pricer import (
    invert_anchors,
    price_market,
    synthetic_price,
    _prob_at_least_n,
    _binomial_cdf,
)


def _anchors_from_rates(p_1b, p_2b, p_3b, p_hr, pa_dist):
    """Compute the three anchor over-probabilities for a known per-PA multinomial."""
    p_h = p_1b + p_2b + p_3b + p_hr
    p_hits_over = 1.0 - sum(p_k * (1.0 - p_h) ** k for k, p_k in pa_dist.items())
    p_hr_over = 1.0 - sum(p_k * (1.0 - p_hr) ** k for k, p_k in pa_dist.items())
    p_tb_le_1 = sum(
        p_k * ((1 - p_h) ** k + k * p_1b * (1 - p_h) ** (k - 1))
        for k, p_k in pa_dist.items()
    )
    return p_hits_over, 1.0 - p_tb_le_1, p_hr_over


@pytest.fixture
def pa_dist():
    return estimate_pa_distribution(2, 4.5)


def test_inversion_round_trip(pa_dist):
    truth = dict(p_1b=0.16, p_2b=0.045, p_3b=0.005, p_hr=0.035)
    p_hits, p_tb_ge_2, p_hr = _anchors_from_rates(**truth, pa_dist=pa_dist)
    rates = invert_anchors(p_hits, p_tb_ge_2, p_hr, pa_dist)
    assert rates is not None
    assert pytest.approx(rates["p_1b"], abs=2e-4) == truth["p_1b"]
    assert pytest.approx(rates["p_2b"], abs=2e-4) == truth["p_2b"]
    assert pytest.approx(rates["p_hr"], abs=2e-4) == truth["p_hr"]
    assert pytest.approx(rates["p_any_hit"], abs=1e-4) == sum(truth.values())


def test_invert_anchors_rejects_degenerate_inputs(pa_dist):
    assert invert_anchors(0.0, 0.5, 0.1, pa_dist) is None
    assert invert_anchors(0.5, 1.0, 0.1, pa_dist) is None


def test_doubles_pricing_monotonic_in_tb_anchor(pa_dist):
    low = synthetic_price(0.55, 0.30, 0.10, pa_dist, "batter_doubles", 0.5)
    high = synthetic_price(0.55, 0.50, 0.10, pa_dist, "batter_doubles", 0.5)
    assert low is not None and high is not None
    assert high[0] > low[0]


def test_singles_pricing_close_to_hits_when_no_xb(pa_dist):
    p_hits, p_tb_ge_2, p_hr = _anchors_from_rates(
        p_1b=0.20, p_2b=0.005, p_3b=0.005, p_hr=0.005, pa_dist=pa_dist,
    )
    over_under = synthetic_price(p_hits, p_tb_ge_2, p_hr, pa_dist, "batter_singles", 0.5)
    assert over_under is not None
    assert over_under[0] > 0.55


def test_price_market_unknown_returns_none(pa_dist):
    rates = {"p_1b": 0.15, "p_2b": 0.04, "p_3b": 0.005, "p_hr": 0.03, "p_any_hit": 0.225}
    assert price_market(rates, 0.5, "batter_runs_scored", pa_dist) is None


def test_binomial_cdf_matches_known_values():
    # P(X <= 1 | n=4, p=0.25) = 0.7383 (analytic)
    assert pytest.approx(_binomial_cdf(1, 4, 0.25), abs=1e-4) == 0.738281
    assert _binomial_cdf(0, 4, 0.0) == 1.0
    assert _binomial_cdf(3, 4, 1.0) == 0.0


def test_prob_at_least_n_zero_rate(pa_dist):
    assert _prob_at_least_n(0.0, 1, pa_dist) == 0.0


def test_doubles_over_under_sums_to_one(pa_dist):
    out = synthetic_price(0.6, 0.4, 0.10, pa_dist, "batter_doubles", 0.5)
    assert out is not None
    assert pytest.approx(out[0] + out[1], abs=1e-9) == 1.0
