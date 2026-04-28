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

from src.config import JOINT_PA_ENABLED
from src.models.joint_pa import pa_correlation_boost


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


def _same_team_batter_pair(leg_a, leg_b) -> bool:
    if not (leg_a.get('market', '').startswith('batter_')
            and leg_b.get('market', '').startswith('batter_')):
        return False
    team_a = (leg_a.get('team') or '').strip().lower()
    team_b = (leg_b.get('team') or '').strip().lower()
    if not team_a or not team_b or team_a != team_b:
        return False
    return all(leg_a.get(k) is not None and leg_b.get(k) is not None
               for k in ('lineup_position', 'mean_count',
                         'line', 'implied_team_total'))


def _offensive_stack_boost(leg_a, leg_b) -> float:
    pos_a = leg_a['lineup_position']
    pos_b = leg_b['lineup_position']
    
    # Lineup is 1-indexed (1 through 9 typically).
    # Wrap around calculation. Distance from pos_a to pos_b in batting order.
    dist_a_to_b = (pos_b - pos_a) % 9
    dist_b_to_a = (pos_a - pos_b) % 9

    if dist_a_to_b <= 3:
        early_leg, late_leg = leg_a, leg_b
        dist = dist_a_to_b
    elif dist_b_to_a <= 3:
        early_leg, late_leg = leg_b, leg_a
        dist = dist_b_to_a
    else:
        # Distance > 3, too far away for primary offensive transitions
        return 1.0

    market_early = early_leg['market']
    market_late = late_leg['market']

    base_boost = 1.0
    
    # "Gets on base" -> "Driven in"
    early_on_base = market_early in (
        'batter_hits', 'batter_total_bases', 'batter_singles', 
        'batter_doubles', 'batter_triples'
    )
    if early_on_base and market_late == 'batter_runs_batted_in':
        base_boost = 1.25

    # "Scores" -> "Driven in by Hit/RBI"
    if market_early == 'batter_runs':
        if market_late in (
            'batter_hits', 'batter_total_bases', 'batter_singles', 
            'batter_doubles', 'batter_triples'
        ):
            base_boost = 1.20
        elif market_late == 'batter_runs_batted_in':
            base_boost = 1.30

    if base_boost > 1.0:
        # Exponential exponential decay by distance:
        decay_factor = [1.0, 0.5, 0.25]
        return 1.0 + (base_boost - 1.0) * decay_factor[dist - 1]

    return 1.0



def _lookup_boost(leg_a, leg_b, db_conn=None):
    market_a, side_a = leg_a['market'], leg_a['side']
    market_b, side_b = leg_b['market'], leg_b['side']

    # Same-team batter-under pair with full data: PA-correlation boost.
    if (JOINT_PA_ENABLED and side_a == 'under' and side_b == 'under'
            and _same_team_batter_pair(leg_a, leg_b)):
        # Slot-equality is degenerate (same player); fall through to static.
        if leg_a['lineup_position'] != leg_b['lineup_position']:
            return pa_correlation_boost(
                slot_a=leg_a['lineup_position'],
                slot_b=leg_b['lineup_position'],
                implied_team_total=leg_a['implied_team_total'],
                line_a=leg_a['line'], mean_a=leg_a['mean_count'], market_a=market_a,
                line_b=leg_b['line'], mean_b=leg_b['mean_count'], market_b=market_b,
            )

    # Same-team batter-over pair: Offensive stack boost (Transition Matrix)
    if side_a == 'over' and side_b == 'over' and _same_team_batter_pair(leg_a, leg_b):
        if leg_a['lineup_position'] != leg_b['lineup_position']:
            return _offensive_stack_boost(leg_a, leg_b)

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
