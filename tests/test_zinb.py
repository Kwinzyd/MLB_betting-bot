import pytest

from src.config import LEAGUE_AVG_HR9, LEAGUE_AVG_ISO, HR_PI0_MAX
from src.models.dispersion import zinb_prob_over, negbin_prob_over
from src.models.distributions import compute_hr_pi0, get_probabilities
from src.models.monte_carlo import mc_prob_over


def test_zinb_recovers_negbin_when_pi0_zero():
    nb = negbin_prob_over(0.4, 0.5, 0.5)
    z = zinb_prob_over(0.4, 0.5, 0.5, 0.0)
    assert pytest.approx(nb[0], abs=1e-9) == z[0]
    assert pytest.approx(nb[1], abs=1e-9) == z[1]


def test_zinb_increases_zero_mass():
    nb_over, nb_under = negbin_prob_over(0.4, 0.5, 0.5)
    zi_over, zi_under = zinb_prob_over(0.4, 0.5, 0.5, 0.30)
    assert zi_under > nb_under
    assert zi_over < nb_over


def test_zinb_marginal_mean_preserved_at_low_lines():
    over, under = zinb_prob_over(0.4, 0.5, 0.5, 0.30)
    assert pytest.approx(over + under, abs=1e-9) == 1.0


def test_compute_hr_pi0_neutral_inputs_return_zero():
    assert compute_hr_pi0() == 0.0
    assert compute_hr_pi0(
        pitcher_hr9=LEAGUE_AVG_HR9,
        batter_iso=LEAGUE_AVG_ISO,
        park_hr_factor=1.0,
        wind_in_mph=0.0,
    ) == 0.0


def test_compute_hr_pi0_groundball_pitcher_inflates():
    pi0 = compute_hr_pi0(
        pitcher_hr9=0.6,
        batter_iso=0.110,
        park_hr_factor=0.85,
        wind_in_mph=10.0,
    )
    assert 0.0 < pi0 <= HR_PI0_MAX


def test_compute_hr_pi0_homerun_friendly_returns_zero():
    pi0 = compute_hr_pi0(
        pitcher_hr9=2.1,
        batter_iso=0.260,
        park_hr_factor=1.20,
        wind_in_mph=-12.0,
    )
    assert pi0 == 0.0


def test_get_probabilities_routes_through_zinb_for_hr():
    base_over, base_under = get_probabilities(0.4, 0.5, "batter_home_runs", alpha=0.5)
    zi_over, zi_under = get_probabilities(0.4, 0.5, "batter_home_runs", alpha=0.5, pi0=0.30)
    assert zi_under > base_under
    assert zi_over < base_over


def test_mc_prob_over_zinb_matches_analytic():
    mean, line, alpha, pi0 = 0.45, 0.5, 0.6, 0.25
    analytic_over, _ = zinb_prob_over(mean, line, alpha, pi0)
    mc_over, _ = mc_prob_over(mean, line, "batter_home_runs",
                              n_sims=20000, nb_alpha=alpha, zinb_pi0=pi0)
    assert abs(mc_over - analytic_over) < 0.02
