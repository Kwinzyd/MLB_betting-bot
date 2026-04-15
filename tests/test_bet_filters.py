from src.models.edge_ranker import rank_edge


def _proj(prob_over=0.60, sample=15):
    return {
        'prob_over': prob_over,
        'prob_under': 1.0 - prob_over,
        'projected_mean': 6.5,
        'injury_status': 'Healthy',
        'sample_size': sample,
    }


def test_rejects_low_odds():
    # Odds 1.65 (-154) is below MIN_ODDS=1.70 floor.
    proj = _proj(prob_over=0.70)
    result = rank_edge(proj, odds=1.65, side='over', sharp_prob=0.70)
    assert result['is_playable'] is False
    assert any('Odds too juicy' in r for r in result['reasons'])


def test_accepts_odds_at_floor():
    proj = _proj(prob_over=0.65)
    result = rank_edge(proj, odds=1.70, side='over', sharp_prob=0.65)
    assert all('Odds too juicy' not in r for r in result['reasons'])


def test_rejects_low_confidence():
    # Sharp itself says 0.52 — coin-flip market, no play even if model agrees.
    proj = _proj(prob_over=0.52)
    result = rank_edge(proj, odds=2.20, side='over', sharp_prob=0.52)
    assert result['is_playable'] is False
    assert any('Low sharp confidence' in r for r in result['reasons'])


def test_accepts_confidence_at_floor():
    proj = _proj(prob_over=0.55)
    result = rank_edge(proj, odds=2.00, side='over', sharp_prob=0.55)
    assert all('Low sharp confidence' not in r for r in result['reasons'])


def test_sample_size_floor_is_ten():
    proj = _proj(prob_over=0.65, sample=9)
    result = rank_edge(proj, odds=2.0, side='over', sharp_prob=0.65)
    assert result['is_playable'] is False
    assert any('Insufficient sample (9 < 10 games)' in r for r in result['reasons'])

    proj_ok = _proj(prob_over=0.65, sample=10)
    result_ok = rank_edge(proj_ok, odds=2.0, side='over', sharp_prob=0.65)
    assert all('Insufficient sample' not in r for r in result_ok['reasons'])
