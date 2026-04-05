import pytest
from src.models.devig import devig_multiplicative, devig_additive, decimal_to_implied


def test_decimal_to_implied():
    assert abs(decimal_to_implied(2.0) - 0.5) < 0.001
    assert abs(decimal_to_implied(1.5) - 0.6667) < 0.001
    assert abs(decimal_to_implied(4.0) - 0.25) < 0.001


def test_devig_multiplicative_fair_odds():
    # Fair odds (no vig): 2.0 / 2.0
    p1, p2 = devig_multiplicative(2.0, 2.0)
    assert abs(p1 - 0.5) < 0.001
    assert abs(p2 - 0.5) < 0.001


def test_devig_multiplicative_with_vig():
    # Typical sportsbook line: -110/-110 = 1.909/1.909
    p1, p2 = devig_multiplicative(1.909, 1.909)
    assert abs(p1 - 0.5) < 0.001
    assert abs(p2 - 0.5) < 0.001
    assert abs(p1 + p2 - 1.0) < 0.001


def test_devig_multiplicative_asymmetric():
    # Favorite/underdog: 1.5 / 3.0
    p1, p2 = devig_multiplicative(1.5, 3.0)
    # implied: 0.667 / 0.333 -> total 1.0 (no vig in this case)
    assert abs(p1 - 0.667) < 0.01
    assert abs(p2 - 0.333) < 0.01


def test_devig_additive_symmetry():
    p1, p2 = devig_additive(1.909, 1.909)
    assert abs(p1 - 0.5) < 0.01
    assert abs(p2 - 0.5) < 0.01


def test_devig_probabilities_sum_to_one():
    # Various odds pairs
    pairs = [(1.87, 2.00), (1.50, 2.80), (2.10, 1.80), (1.91, 1.91)]
    for o1, o2 in pairs:
        p1, p2 = devig_multiplicative(o1, o2)
        assert abs(p1 + p2 - 1.0) < 0.001, f"Failed for odds {o1}/{o2}: {p1}+{p2}"
