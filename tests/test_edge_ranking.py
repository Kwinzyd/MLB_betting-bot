from src.models.edge_ranker import rank_edge


def _make_projection(prob_over=0.6, prob_under=0.4, injury='Healthy', sample=15):
    return {
        'prob_over': prob_over,
        'prob_under': prob_under,
        'projected_mean': 6.5,
        'injury_status': injury,
        'sample_size': sample,
    }


def test_positive_edge_is_playable():
    proj = _make_projection(prob_over=0.60)
    result = rank_edge(proj, odds=1.87, side='over', sharp_prob=0.60)
    assert result['is_playable'] is True
    assert result['edge_pct'] > 5.0
    assert result['ev'] > 0


def test_small_edge_not_playable():
    proj = _make_projection(prob_over=0.52)
    result = rank_edge(proj, odds=1.91, side='over', sharp_prob=0.52)
    assert result['is_playable'] is False
    assert any('Edge too small' in r or 'Low sharp confidence' in r
               for r in result['reasons'])


def test_injury_il_not_playable():
    proj = _make_projection(prob_over=0.65, injury='IL')
    result = rank_edge(proj, odds=1.87, side='over', sharp_prob=0.65)
    assert result['is_playable'] is False
    assert any('Injury' in r for r in result['reasons'])


def test_injury_out_not_playable():
    proj = _make_projection(prob_over=0.65, injury='Out')
    result = rank_edge(proj, odds=1.87, side='over', sharp_prob=0.65)
    assert result['is_playable'] is False


def test_day_to_day_is_playable():
    proj = _make_projection(prob_over=0.65, injury='Day-to-Day')
    result = rank_edge(proj, odds=1.87, side='over', sharp_prob=0.65)
    assert result['is_playable'] is True


def test_insufficient_sample():
    proj = _make_projection(prob_over=0.65, sample=3)
    result = rank_edge(proj, odds=1.87, side='over', sharp_prob=0.65)
    assert result['is_playable'] is False
    assert any('sample' in r.lower() for r in result['reasons'])


def test_kelly_included_in_result():
    proj = _make_projection(prob_over=0.60)
    result = rank_edge(proj, odds=1.87, side='over', sharp_prob=0.60)
    assert 'kelly' in result
    assert 'recommended_stake' in result['kelly']
    assert result['kelly']['recommended_stake'] > 0


def test_under_side():
    proj = _make_projection(prob_over=0.35, prob_under=0.65)
    result = rank_edge(proj, odds=1.87, side='under', sharp_prob=0.65)
    assert result['model_prob'] == 0.65
    assert result['is_playable'] is True


def test_ev_calculation():
    proj = _make_projection(prob_over=0.60)
    result = rank_edge(proj, odds=2.0, side='over', sharp_prob=0.60)
    # EV = sharp_prob * odds - 1 = 0.60 * 2.0 - 1.0 = 0.20
    assert abs(result['ev'] - 0.20) < 0.01
