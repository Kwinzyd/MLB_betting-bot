"""
Correlation-adjusted joint probability for same-game parlays.

The boost table captures known pitcher-vs-batter anti-correlation in MLB:
when a pitcher is striking out more batters, the balls-in-play rate drops,
so opposing hits and total bases tend UNDER. Values are multiplicative
adjustments to the independent-leg product (1.0 = independent).

This is deliberately coarse — a ranking signal, not a calibrated price.
All boosts are env-overridable.
"""
import os


def _env_float(name, default):
    try:
        return float(os.getenv(name, str(default)))
    except (TypeError, ValueError):
        return default


CORR_BOOST = {
    ('pitcher_strikeouts', 'over', 'batter_hits', 'under'):
        _env_float('SGP_CORR_K_HITS', 1.12),
    ('pitcher_strikeouts', 'over', 'batter_total_bases', 'under'):
        _env_float('SGP_CORR_K_TB', 1.10),
    ('pitcher_strikeouts', 'over', 'batter_home_runs', 'under'):
        _env_float('SGP_CORR_K_HR', 1.06),
    ('batter_hits', 'under', 'batter_total_bases', 'under'):
        _env_float('SGP_CORR_HITS_TB', 1.05),
    ('batter_hits', 'under', 'batter_hits', 'under'):
        _env_float('SGP_CORR_HITS_HITS', 1.04),
    ('batter_total_bases', 'under', 'batter_total_bases', 'under'):
        _env_float('SGP_CORR_TB_TB', 1.04),
}


def _lookup_boost(market_a, side_a, market_b, side_b):
    key = (market_a, side_a, market_b, side_b)
    if key in CORR_BOOST:
        return CORR_BOOST[key]
    rev = (market_b, side_b, market_a, side_a)
    return CORR_BOOST.get(rev, 1.0)


def joint_probability(legs):
    """
    Approximate joint probability of every leg hitting.

    legs: list of {'market': str, 'side': str, 'prob': float, ...}
    Returns a probability in [0, 1].

    Model: joint = (prod p_i) * (prod pairwise_boost_ij), clamped to 1.
    """
    if not legs:
        return 0.0
    p = 1.0
    for leg in legs:
        p *= leg['prob']
    boost = 1.0
    n = len(legs)
    for i in range(n):
        for j in range(i + 1, n):
            boost *= _lookup_boost(
                legs[i]['market'], legs[i]['side'],
                legs[j]['market'], legs[j]['side'],
            )
    return min(p * boost, 1.0)


def parlay_decimal_odds(legs):
    """Naive independent-parlay decimal odds = product of each leg's odds."""
    odds = 1.0
    for leg in legs:
        odds *= leg['odds']
    return odds
