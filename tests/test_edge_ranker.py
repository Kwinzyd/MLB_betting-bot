import pytest
from src.models.edge_ranker import rank_edge


@pytest.fixture
def base_projection():
    """Profitable base projection. model/sharp agree so agreement gate passes."""
    return {
        'prob_over': 0.60,
        'prob_under': 0.40,
        'projected_mean': 6.5,
        'injury_status': 'Healthy',
        'sample_size': 15
    }


def test_rank_edge_profitable_over(base_projection):
    """Sharp says 0.60, book implies 0.50 → 10% edge on the over."""
    result = rank_edge(base_projection, odds=2.0, side='over', sharp_prob=0.60)

    assert result['is_playable'] is True
    assert result['edge_pct'] == pytest.approx(10.0)
    assert result['ev'] == pytest.approx(0.20)
    assert result['model_prob'] == 0.60
    assert result['sharp_prob'] == 0.60
    assert result['book_implied'] == 0.50


def test_rank_edge_profitable_under(base_projection):
    """Sharp says 0.60 on the under; book implies 0.50."""
    base_projection['prob_under'] = 0.60
    result = rank_edge(base_projection, odds=2.0, side='under', sharp_prob=0.60)

    assert result['is_playable'] is True
    assert result['edge_pct'] == pytest.approx(10.0)
    assert result['ev'] == pytest.approx(0.20)
    assert result['model_prob'] == 0.60


def test_rank_edge_unplayable_negative_ev(base_projection):
    """Odds 1.5 (implied 66.6%) and sharp 0.60 → negative edge, unplayable."""
    result = rank_edge(base_projection, odds=1.5, side='over', sharp_prob=0.60)

    assert result['is_playable'] is False
    assert result['edge_pct'] < 0
    assert result['ev'] < 0
    assert any("Negative Kelly" in r or "Edge too small" in r or "Odds too juicy" in r
               for r in result['reasons'])


def test_rank_edge_unplayable_due_to_injury(base_projection):
    base_projection['injury_status'] = 'IL'
    result = rank_edge(base_projection, odds=2.0, side='over', sharp_prob=0.60)

    assert result['is_playable'] is False
    assert "Injury status: IL" in result['reasons']


def test_rank_edge_unplayable_small_sample(base_projection):
    base_projection['sample_size'] = 4
    result = rank_edge(base_projection, odds=2.0, side='over', sharp_prob=0.60)

    assert result['is_playable'] is False
    assert any("Insufficient sample" in r for r in result['reasons'])


def test_rank_edge_blocked_by_model_sharp_disagreement(base_projection):
    """Model says 0.60, sharp says 0.75 → gap of 0.15 exceeds tol 0.05. Skip."""
    result = rank_edge(base_projection, odds=2.0, side='over', sharp_prob=0.75)

    assert result['is_playable'] is False
    assert any("disagreement" in r.lower() for r in result['reasons'])
    assert result['disagreement'] == pytest.approx(0.15)


def test_rank_edge_agreement_within_tolerance(base_projection):
    """Model 0.60 vs sharp 0.64 (gap 0.04) is within the default 0.05 tolerance."""
    result = rank_edge(base_projection, odds=2.0, side='over', sharp_prob=0.64)

    assert result['is_playable'] is True
    assert not any("disagreement" in r.lower() for r in result['reasons'])
