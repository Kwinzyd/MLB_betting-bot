from src.models.correlation import joint_probability, parlay_decimal_odds


def _leg(market, side, prob, odds=2.0):
    return {'market': market, 'side': side, 'prob': prob, 'odds': odds}


def test_empty_legs_returns_zero():
    assert joint_probability([]) == 0.0


def test_no_correlation_falls_back_to_product():
    # No boost key for (batter_home_runs over, batter_home_runs over).
    legs = [
        _leg('batter_home_runs', 'over', 0.5),
        _leg('batter_home_runs', 'over', 0.5),
    ]
    assert abs(joint_probability(legs) - 0.25) < 0.001


def test_k_over_with_batter_hits_under_boosts_joint_prob():
    legs = [
        _leg('pitcher_strikeouts', 'over', 0.60),
        _leg('batter_hits', 'under', 0.60),
    ]
    # Independent = 0.36; default boost 1.12 → ~0.403.
    p = joint_probability(legs)
    assert p > 0.36
    assert p < 1.0


def test_three_leg_pitcher_anchor_ticket():
    legs = [
        _leg('pitcher_strikeouts', 'over', 0.60),
        _leg('batter_hits', 'under', 0.58),
        _leg('batter_total_bases', 'under', 0.57),
    ]
    p = joint_probability(legs)
    independent = 0.60 * 0.58 * 0.57
    assert p > independent
    assert p <= 1.0


def test_joint_clamped_to_one():
    legs = [
        _leg('pitcher_strikeouts', 'over', 0.99),
        _leg('batter_hits', 'under', 0.99),
    ]
    assert joint_probability(legs) <= 1.0


def test_boost_lookup_is_symmetric():
    # Reversed leg order should give the same joint.
    legs_a = [
        _leg('pitcher_strikeouts', 'over', 0.60),
        _leg('batter_hits', 'under', 0.60),
    ]
    legs_b = list(reversed(legs_a))
    assert abs(joint_probability(legs_a) - joint_probability(legs_b)) < 1e-9


def test_parlay_decimal_odds_multiplies():
    legs = [
        _leg('m', 's', 0.5, odds=2.0),
        _leg('m', 's', 0.5, odds=1.5),
        _leg('m', 's', 0.5, odds=3.0),
    ]
    assert abs(parlay_decimal_odds(legs) - 9.0) < 1e-9


def test_single_leg_returns_own_prob():
    legs = [_leg('pitcher_strikeouts', 'over', 0.6)]
    assert abs(joint_probability(legs) - 0.6) < 1e-9


# --- Joint-PA same-team batter-batter boost ---------------------------------


def _enriched_leg(market, line, prob, slot, mean, team='Yankees', total=4.5, odds=2.0):
    return {
        'market': market, 'side': 'under', 'prob': prob, 'odds': odds,
        'line': line, 'team': team, 'lineup_position': slot,
        'mean_count': mean, 'implied_team_total': total,
    }


def test_same_team_batter_pair_uses_joint_pa_boost():
    """Two same-team batter-under legs with full data dispatch to
    pa_correlation_boost, which gives a slot- and mean-aware boost — not the
    flat 1.04 BATTER_CORR_BOOST entry."""
    legs = [
        _enriched_leg('batter_hits', 1.5, 0.60, slot=1, mean=1.4),
        _enriched_leg('batter_hits', 1.5, 0.60, slot=3, mean=1.4),
    ]
    p = joint_probability(legs)
    independent = 0.60 * 0.60
    boost = p / independent
    # Joint-PA boost is generally smaller than the static 1.04, but still > 1.
    assert boost > 1.0
    assert boost < 1.04


def test_cross_team_batter_pair_falls_back_to_static():
    """Different teams → no joint-PA path; static BATTER_CORR_BOOST applies."""
    legs = [
        _enriched_leg('batter_hits', 1.5, 0.60, slot=1, mean=1.4, team='Yankees'),
        _enriched_leg('batter_hits', 1.5, 0.60, slot=3, mean=1.4, team='Red Sox'),
    ]
    p = joint_probability(legs)
    boost = p / (0.60 * 0.60)
    # Static SGP_CORR_HITS_HITS = 1.04
    assert abs(boost - 1.04) < 1e-9


def test_missing_slot_falls_back_to_static():
    """Without lineup_position, joint-PA path is skipped → static boost."""
    legs = [
        {'market': 'batter_hits', 'side': 'under', 'prob': 0.60, 'odds': 2.0,
         'line': 1.5, 'team': 'Yankees'},
        {'market': 'batter_hits', 'side': 'under', 'prob': 0.60, 'odds': 2.0,
         'line': 1.5, 'team': 'Yankees'},
    ]
    p = joint_probability(legs)
    boost = p / (0.60 * 0.60)
    assert abs(boost - 1.04) < 1e-9


def test_pitcher_batter_path_unchanged_by_joint_pa():
    """Pitcher × batter empirical-Pearson path is independent of joint-PA."""
    legs = [
        _leg('pitcher_strikeouts', 'over', 0.60),
        _enriched_leg('batter_hits', 1.5, 0.60, slot=3, mean=1.3),
    ]
    p = joint_probability(legs)
    # Should still get the empirical-correlation boost, > independent.
    assert p > 0.60 * 0.60
