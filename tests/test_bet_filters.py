import pytest
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
    # Odds 1.65 (-153) clears MIN_ODDS=1.70 floor, still profitable edge.
    proj = _proj(prob_over=0.70)
    result = rank_edge(proj, odds=1.65, side='over', devigged_prob=0.58)
    assert result['is_playable'] is False
    assert any('Odds too juicy' in r for r in result['reasons'])


def test_accepts_odds_at_floor():
    proj = _proj(prob_over=0.65)
    result = rank_edge(proj, odds=1.70, side='over', devigged_prob=0.55)
    # odds=1.70 meets the floor, edge from 0.65 vs 1/1.70=0.588 → ~6.2%
    assert all('Odds too juicy' not in r for r in result['reasons'])


def test_rejects_low_confidence():
    # 52% model prob against a 45% implied line — real edge but a coin-flip.
    proj = _proj(prob_over=0.52)
    result = rank_edge(proj, odds=2.20, side='over', devigged_prob=0.45)
    assert result['is_playable'] is False
    assert any('Low model confidence' in r for r in result['reasons'])


def test_accepts_confidence_at_floor():
    proj = _proj(prob_over=0.55)
    result = rank_edge(proj, odds=2.00, side='over', devigged_prob=0.48)
    assert all('Low model confidence' not in r for r in result['reasons'])


def test_sample_size_floor_is_ten():
    proj = _proj(prob_over=0.65, sample=9)
    result = rank_edge(proj, odds=2.0, side='over', devigged_prob=0.50)
    assert result['is_playable'] is False
    assert any('Insufficient sample (9 < 10 games)' in r for r in result['reasons'])

    proj_ok = _proj(prob_over=0.65, sample=10)
    result_ok = rank_edge(proj_ok, odds=2.0, side='over', devigged_prob=0.50)
    assert all('Insufficient sample' not in r for r in result_ok['reasons'])
