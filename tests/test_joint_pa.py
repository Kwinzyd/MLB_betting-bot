import pytest

from src.config import PA_DIST_SUPPORT, JOINT_PA_BOOST_CAP
from src.models.joint_pa import (
    tbf_distribution,
    joint_pa_distribution,
    pa_correlation_boost,
    slot_pa_given_tbf,
    _slot_marginal_from_tbf,
)
from src.models.pa_estimator import estimate_pa_distribution, expected_pa


@pytest.mark.parametrize("total", [3.5, 4.5, 6.0])
def test_tbf_pmf_sums_to_one(total):
    pmf = tbf_distribution(total)
    assert abs(sum(pmf.values()) - 1.0) < 1e-9


@pytest.mark.parametrize("total", [3.5, 4.5, 6.0])
def test_team_total_pa_recovered(total):
    """λ_TBF is calibrated to the team-total PA — that recovery is exact."""
    pmf = tbf_distribution(total)
    target = sum(expected_pa(estimate_pa_distribution(s, total)) for s in range(1, 10))
    recov = sum(
        sum(k * p for k, p in _slot_marginal_from_tbf(pmf, s).items())
        for s in range(1, 10)
    )
    assert abs(recov - target) < 0.01


@pytest.mark.parametrize("slot", [1, 5, 9])
@pytest.mark.parametrize("total", [3.5, 4.5, 6.0])
def test_per_slot_marginal_drift_bounded(slot, total):
    """Single TBF latent can't match per-slot Poissons exactly; verify drift
    stays under 0.25 PA at the extreme slots."""
    pmf = tbf_distribution(total)
    marg = _slot_marginal_from_tbf(pmf, slot)
    target_mean = expected_pa(estimate_pa_distribution(slot, total))
    recov_mean = sum(k * p for k, p in marg.items())
    assert abs(recov_mean - target_mean) < 0.25


def test_slot_pa_given_tbf_deterministic():
    # TBF=27: every slot gets exactly 3 PAs.
    assert all(slot_pa_given_tbf(s, 27) == 3 for s in range(1, 10))
    # TBF=28: slot 1 gets 4, others 3.
    assert slot_pa_given_tbf(1, 28) == 4
    assert slot_pa_given_tbf(2, 28) == 3
    # TBF=36: slots 1..9 all get 4.
    assert all(slot_pa_given_tbf(s, 36) == 4 for s in range(1, 10))


def test_joint_pmf_sums_to_one():
    joint = joint_pa_distribution(1, 5, 4.5)
    assert abs(sum(joint.values()) - 1.0) < 1e-9


def test_joint_pmf_concentrated_on_diagonal_at_low_total():
    """At low totals every TBF realization gives small PA spread, so most
    mass lands on (k, k) or (k, k-1)."""
    joint = joint_pa_distribution(1, 9, 3.5)
    diag_or_adj = sum(p for (a, b), p in joint.items() if abs(a - b) <= 1)
    assert diag_or_adj > 0.95


def test_joint_pmf_marginalizes_to_slot_marginal():
    total = 4.5
    joint = joint_pa_distribution(2, 7, total)
    pmf = tbf_distribution(total)
    marg_a_from_joint = {k: 0.0 for k in PA_DIST_SUPPORT}
    for (a, _b), p in joint.items():
        marg_a_from_joint[a] += p
    marg_a_direct = _slot_marginal_from_tbf(pmf, 2)
    for k in PA_DIST_SUPPORT:
        assert abs(marg_a_from_joint[k] - marg_a_direct[k]) < 1e-9


def test_boost_at_least_one_and_capped():
    b = pa_correlation_boost(1, 3, 4.5, 1.5, 1.3, 'batter_hits',
                             1.5, 1.3, 'batter_hits')
    assert 1.0 <= b <= JOINT_PA_BOOST_CAP


def test_boost_strictly_positive_for_realistic_pair():
    """A typical 1-3 same-team batter-under pair has measurable PA-induced
    correlation (>=0.5%); the static 1.04 baseline overstates it slightly."""
    b = pa_correlation_boost(1, 3, 4.5, 1.5, 1.3, 'batter_hits',
                             1.5, 1.3, 'batter_hits')
    assert b > 1.005


def test_boost_stronger_when_mean_near_line():
    """When mean is close to the line, the PA-bin choice flips the under
    probability more, so PA covariance contributes more boost."""
    b_far = pa_correlation_boost(1, 3, 4.5, 1.5, 0.7, 'batter_hits',
                                 1.5, 0.7, 'batter_hits')
    b_near = pa_correlation_boost(1, 3, 4.5, 1.5, 1.4, 'batter_hits',
                                  1.5, 1.4, 'batter_hits')
    assert b_near > b_far


def test_boost_returns_one_for_missing_slot():
    assert pa_correlation_boost(None, 3, 4.5, 1.5, 1.3, 'batter_hits',
                                1.5, 1.3, 'batter_hits') == 1.0
    assert pa_correlation_boost(1, None, 4.5, 1.5, 1.3, 'batter_hits',
                                1.5, 1.3, 'batter_hits') == 1.0


def test_boost_returns_one_for_zero_mean():
    assert pa_correlation_boost(1, 3, 4.5, 1.5, 0.0, 'batter_hits',
                                1.5, 1.3, 'batter_hits') == 1.0
