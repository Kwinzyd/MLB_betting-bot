import pytest
from src.models.edge_ranker import rank_edge


@pytest.fixture
def base_projection():
    """Provide a standard profitable base projection for testing."""
    return {
        'prob_over': 0.60,
        'prob_under': 0.40,
        'projected_mean': 6.5,
        'injury_status': 'Healthy',
        'sample_size': 15
    }


def test_rank_edge_profitable_over(base_projection):
    """Test a clear profitable edge on the over."""
    # Odds 2.0 (implied 50%), Model 60%. Edge = 10%. EV = 0.20
    result = rank_edge(base_projection, odds=2.0, side='over', devigged_prob=0.50)

    assert result['is_playable'] is True
    assert result['edge_pct'] == pytest.approx(10.0)
    assert result['ev'] == pytest.approx(0.20)
    assert result['model_prob'] == 0.60
    assert result['book_implied'] == 0.50


def test_rank_edge_profitable_under(base_projection):
    """Test a clear profitable edge on the under."""
    base_projection['prob_under'] = 0.60
    # Odds 2.0 (implied 50%), Model 60%. Edge = 10%.
    result = rank_edge(base_projection, odds=2.0, side='under', devigged_prob=0.50)

    assert result['is_playable'] is True
    assert result['edge_pct'] == pytest.approx(10.0)
    assert result['ev'] == pytest.approx(0.20)
    assert result['model_prob'] == 0.60


def test_rank_edge_unplayable_negative_ev(base_projection):
    """Test that negative EV bets are correctly flagged as unplayable."""
    # Odds 1.5 (implied 66.6%), Model 60%. Edge = -6.6%
    result = rank_edge(base_projection, odds=1.5, side='over', devigged_prob=0.666)

    assert result['is_playable'] is False
    assert result['edge_pct'] < 0
    assert result['ev'] < 0
    assert any("Negative Kelly" in r or "Edge too small" in r for r in result['reasons'])


def test_rank_edge_unplayable_due_to_injury(base_projection):
    """Test that injured players are filtered out regardless of edge."""
    base_projection['injury_status'] = 'IL'
    result = rank_edge(base_projection, odds=2.0, side='over', devigged_prob=0.50)

    assert result['is_playable'] is False
    assert "Injury status: IL" in result['reasons']


def test_rank_edge_unplayable_small_sample(base_projection):
    """Test that projections with small samples are filtered out."""
    base_projection['sample_size'] = 4
    result = rank_edge(base_projection, odds=2.0, side='over', devigged_prob=0.50)

    assert result['is_playable'] is False
    assert any("Insufficient sample" in r for r in result['reasons'])