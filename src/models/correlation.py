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
import numpy as np


def _env_float(name, default):
    try:
        return float(os.getenv(name, str(default)))
    except (TypeError, ValueError):
        return default


BATTER_CORR_BOOST = {
    ('batter_hits', 'under', 'batter_total_bases', 'under'):
        _env_float('SGP_CORR_HITS_TB', 1.05),
    ('batter_hits', 'under', 'batter_hits', 'under'):
        _env_float('SGP_CORR_HITS_HITS', 1.04),
    ('batter_total_bases', 'under', 'batter_total_bases', 'under'):
        _env_float('SGP_CORR_TB_TB', 1.04),
}


def compute_pitcher_correlation(player_id: int, db_conn) -> dict:
    """
    Computes the empirical Pearson correlation between a pitcher's strikeouts
    and opposing hits_allowed and proxy total_bases allowed.
    """
    if not player_id or not db_conn:
        return {'hits': -0.2, 'tb': -0.15, 'hr': -0.1}

    rows = db_conn.execute(
        '''
        SELECT strikeouts, hits_allowed, home_runs_allowed 
        FROM pitcher_game_logs 
        WHERE player_id = ? 
        ORDER BY date DESC LIMIT 50
        ''', (player_id,)
    ).fetchall()
    
    if len(rows) < 5:
        return {'hits': -0.2, 'tb': -0.15, 'hr': -0.1}
        
    k = np.array([r['strikeouts'] or 0 for r in rows], dtype=float)
    hits = np.array([r['hits_allowed'] or 0 for r in rows], dtype=float)
    hrs = np.array([r['home_runs_allowed'] or 0 for r in rows], dtype=float)
    tbs = hits + (hrs * 3) # approximation: HR is 4 TB, but already counted as 1 hit

    # numpy corrcoef returns a 2x2 matrix, [0, 1] is the correlation
    def safe_corr(x, y):
        if np.std(x) == 0 or np.std(y) == 0:
            return 0.0
        return float(np.corrcoef(x, y)[0, 1])

    return {
        'hits': safe_corr(k, hits),
        'tb': safe_corr(k, tbs),
        'hr': safe_corr(k, hrs)
    }


def _lookup_boost(leg_a, leg_b, db_conn=None):
    market_a, side_a = leg_a['market'], leg_a['side']
    market_b, side_b = leg_b['market'], leg_b['side']
    
    # Base batter-batter correlations
    key = (market_a, side_a, market_b, side_b)
    rev = (market_b, side_b, market_a, side_a)
    if key in BATTER_CORR_BOOST:
        return BATTER_CORR_BOOST[key]
    if rev in BATTER_CORR_BOOST:
        return BATTER_CORR_BOOST[rev]
        
    # Pitcher-batter empirical correlations
    pitcher_leg = None
    batter_leg = None
    
    if market_a == 'pitcher_strikeouts' and market_b.startswith('batter_'):
        pitcher_leg, batter_leg = leg_a, leg_b
    elif market_b == 'pitcher_strikeouts' and market_a.startswith('batter_'):
        pitcher_leg, batter_leg = leg_b, leg_a
        
    if pitcher_leg and batter_leg and pitcher_leg['side'] == 'over' and batter_leg['side'] == 'under':
        player_id = pitcher_leg.get('player_id')
        correlations = compute_pitcher_correlation(player_id, db_conn)
        
        # Mapping r to a boost: boost = 1.0 + max(0, -r) * SCALE
        scale_factor = _env_float('SGP_EMPIRICAL_SCALE', 0.35)
        
        b_market = batter_leg['market']
        if b_market == 'batter_hits':
            r = correlations['hits']
        elif b_market == 'batter_total_bases':
            r = correlations['tb']
        elif b_market == 'batter_home_runs':
            r = correlations['hr']
        else:
            r = 0.0
            
        return 1.0 + max(0.0, -r) * scale_factor
        
    return 1.0


def joint_probability(legs, db_conn=None):
    """
    Approximate joint probability of every leg hitting.

    legs: list of {'market': str, 'side': str, 'prob': float, 'player_id': int...}
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
            boost *= _lookup_boost(legs[i], legs[j], db_conn)
    return min(p * boost, 1.0)


def parlay_decimal_odds(legs):
    """Naive independent-parlay decimal odds = product of each leg's odds."""
    odds = 1.0
    for leg in legs:
        odds *= leg['odds']
    return odds
