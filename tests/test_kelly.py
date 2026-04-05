from src.models.kelly import fractional_kelly


def test_positive_edge():
    # Model says 55% chance, odds are 2.0 (implied 50%)
    result = fractional_kelly(0.55, 2.0, fraction=0.25, bankroll=1000.0)
    assert result['kelly_fraction'] > 0
    assert result['recommended_stake'] > 0
    assert result['full_kelly_pct'] > 0


def test_negative_edge():
    # Model says 40% chance, odds are 2.0 (implied 50%) -> negative edge
    result = fractional_kelly(0.40, 2.0, fraction=0.25, bankroll=1000.0)
    assert result['kelly_fraction'] == 0.0
    assert result['recommended_stake'] == 0.0
    assert result['full_kelly_pct'] < 0


def test_no_edge():
    # Model matches implied: 50% at 2.0
    result = fractional_kelly(0.50, 2.0, fraction=0.25, bankroll=1000.0)
    assert result['kelly_fraction'] == 0.0
    assert result['recommended_stake'] == 0.0


def test_fractional_reduces_stake():
    # Use a smaller edge so the cap doesn't clamp both to the same value
    full = fractional_kelly(0.54, 2.0, fraction=1.0, bankroll=1000.0)
    quarter = fractional_kelly(0.54, 2.0, fraction=0.25, bankroll=1000.0)
    assert quarter['recommended_stake'] < full['recommended_stake']


def test_cap_at_5_percent():
    # Very large edge should still cap at 5% of bankroll
    result = fractional_kelly(0.90, 2.0, fraction=1.0, bankroll=1000.0)
    assert result['recommended_stake'] <= 50.0  # 5% of 1000


def test_known_kelly_calculation():
    # p=0.25, b=4.0 (decimal 5.0) -> Full Kelly = (4*0.25 - 0.75)/4 = 0.0625
    result = fractional_kelly(0.25, 5.0, fraction=1.0, bankroll=1000.0)
    assert abs(result['full_kelly_pct'] - 0.0625) < 0.001
    # Capped at 5%, 0.0625 * 1000 = 50 which is exactly 5%
    assert result['recommended_stake'] <= 50.0


def test_bankroll_scaling():
    small = fractional_kelly(0.55, 2.0, fraction=0.25, bankroll=100.0)
    large = fractional_kelly(0.55, 2.0, fraction=0.25, bankroll=10000.0)
    assert large['recommended_stake'] > small['recommended_stake']
