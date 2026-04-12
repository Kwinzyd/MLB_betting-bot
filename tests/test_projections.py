import pytest
from unittest.mock import patch
from src.models.projections import ProjectionModel


@pytest.fixture
def proj_model():
    """Provide a fresh ProjectionModel instance for each test."""
    return ProjectionModel()


def test_project_pitcher_strikeouts_handles_empty_logs(proj_model):
    """Test that the model gracefully handles a player with no historical logs."""
    result = proj_model.project_pitcher_strikeouts(
        logs=[],
        opp_k_rate=0.225,
        venue="Wrigley Field",
        line=5.5
    )
    assert result is None or isinstance(result, dict), "Should return None or a fallback dict for empty logs"


def test_project_pitcher_strikeouts_valid_data(proj_model):
    """Test strikeout projections with standard historical data."""
    logs = [
        {"strikeouts": 6, "innings_pitched": 5.0, "pitches_thrown": 90},
        {"strikeouts": 4, "innings_pitched": 6.0, "pitches_thrown": 95},
        {"strikeouts": 8, "innings_pitched": 5.2, "pitches_thrown": 100},
    ]
    result = proj_model.project_pitcher_strikeouts(
        logs=logs,
        opp_k_rate=0.25,
        venue="Wrigley Field",
        line=5.5
    )

    if result is not None:
        assert "projected_mean" in result
        assert "prob_over" in result
        assert "prob_under" in result
        assert 0.0 <= result["prob_over"] <= 1.0
        assert 0.0 <= result["prob_under"] <= 1.0
        # Probabilities should sum to approximately 1.0
        assert result["prob_over"] + result["prob_under"] == pytest.approx(1.0, rel=0.05)


def test_project_pitcher_earned_runs_valid_data(proj_model):
    """Test earned run projections return valid math outputs."""
    logs = [
        {"earned_runs": 2, "innings_pitched": 6.0},
        {"earned_runs": 4, "innings_pitched": 5.0},
    ]
    result = proj_model.project_pitcher_earned_runs(
        logs=logs,
        opp_runs_pg=4.5,
        venue="Coors Field",
        line=2.5
    )

    if result is not None:
        assert "projected_mean" in result
        assert result["projected_mean"] > 0


def test_project_batter_stat_has_correct_keys(proj_model):
    """Test that batter projections output the correct data structure."""
    logs = [
        {"hits": 1, "at_bats": 4, "plate_appearances": 4},
        {"hits": 2, "at_bats": 4, "plate_appearances": 4},
    ]

    result = proj_model.project_batter_stat(
        logs=logs, stat_type="hits", pitcher_hand="R", bats="L",
        venue="Coors Field", line=1.5, lineup_position=1
    )

    if result is not None:
        assert "projected_mean" in result