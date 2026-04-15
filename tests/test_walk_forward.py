"""Tests for the walk-forward backtest fold generator and runner.

The dataset-assembly path (_build_*_dataset) hits the live DB and is covered
elsewhere; here we focus on the fold-generation logic + the per-market runner
with synthetic in-memory inputs.
"""
from datetime import datetime, timedelta
from unittest.mock import patch

import numpy as np

from src.pipelines.walk_forward import (
    generate_folds,
    walk_forward_market,
)


# ---------------------------------------------------------------------------
# Fold generator
# ---------------------------------------------------------------------------

def _date_range(start_iso, days):
    start = datetime.strptime(start_iso, "%Y-%m-%d")
    return [(start + timedelta(days=i)).strftime("%Y-%m-%d") for i in range(days)]


def test_sliding_window_advances_train_start():
    dates = _date_range("2025-04-01", 60)
    folds = list(generate_folds(dates, train_window_days=45, step_days=7, mode="sliding"))
    # First fold: train [04-01, 05-16), test [05-16, 05-23)
    assert folds[0] == ("2025-04-01", "2025-05-16", "2025-05-16", "2025-05-23")
    # Second fold: train_start advances by step_days
    assert folds[1] == ("2025-04-08", "2025-05-23", "2025-05-23", "2025-05-30")


def test_expanding_window_anchors_train_start():
    dates = _date_range("2025-04-01", 60)
    folds = list(generate_folds(dates, train_window_days=45, step_days=7, mode="expanding"))
    # train_start always = first date
    for tr_start, tr_end, _, _ in folds:
        assert tr_start == "2025-04-01"
    # train_end and test boundaries advance
    assert folds[1][1] == "2025-05-23"
    assert folds[2][1] == "2025-05-30"


def test_no_folds_when_data_shorter_than_window():
    dates = _date_range("2025-04-01", 30)  # < 45-day window
    folds = list(generate_folds(dates, train_window_days=45, step_days=7, mode="sliding"))
    assert folds == []


def test_invalid_mode_raises():
    import pytest
    with pytest.raises(ValueError):
        list(generate_folds(["2025-04-01"], 45, 7, mode="bogus"))


def test_step_size_controls_fold_density():
    dates = _date_range("2025-04-01", 100)
    weekly = list(generate_folds(dates, 45, 7, "sliding"))
    daily = list(generate_folds(dates, 45, 1, "sliding"))
    assert len(daily) > len(weekly) * 6  # roughly 7x more folds


def test_empty_dates_yields_nothing():
    assert list(generate_folds([], 45, 7, "sliding")) == []


# ---------------------------------------------------------------------------
# walk_forward_market
# ---------------------------------------------------------------------------

def _synthetic_dataset(n_days=120, rows_per_day=10, seed=0):
    """Build a synthetic dataset where target = 2*x0 + noise (Poisson)."""
    rng = np.random.default_rng(seed)
    start = datetime(2025, 4, 1)
    X_rows, y_rows, exp_rows, date_rows = [], [], [], []
    for d in range(n_days):
        date_str = (start + timedelta(days=d)).strftime("%Y-%m-%d")
        for _ in range(rows_per_day):
            x0 = rng.normal(0, 1)
            x1 = rng.normal(0, 1)
            mu = np.exp(0.5 + 0.3 * x0 + 0.1 * x1)  # baseline rate
            y = rng.poisson(mu)
            X_rows.append([x0, x1])
            y_rows.append(y)
            exp_rows.append(1.0)
            date_rows.append(date_str)
    return (
        np.array(X_rows, dtype=float),
        np.array(y_rows, dtype=int),
        np.array(exp_rows, dtype=float),
        np.array(date_rows),
        ["x0", "x1"],
    )


def test_walk_forward_market_runs_and_aggregates():
    """Patch the dataset builder to avoid hitting the DB. Verify folds + aggregates."""
    data = _synthetic_dataset(n_days=120, rows_per_day=10)

    with patch(
        "src.pipelines.walk_forward._build_pitcher_dataset",
        return_value=data,
    ):
        result = walk_forward_market(
            "pitcher_strikeouts",
            train_window_days=45,
            step_days=14,
            mode="sliding",
            min_train_rows=50,
        )

    assert result["ran"] is True
    assert result["mode"] == "sliding"
    assert result["n_folds"] >= 3
    assert "weighted_mae" in result
    # Each fold should have a non-empty test set and a date-stamped window
    for f in result["folds"]:
        assert f["n"] > 0
        assert f["test_start"] < f["test_end"]
    # Monthly aggregates should partition the test rows without loss
    total_fold_n = sum(f["n"] for f in result["folds"])
    total_month_n = sum(m["n"] for m in result["monthly"])
    assert total_fold_n == total_month_n


def test_walk_forward_market_skips_when_no_data():
    empty = (np.empty((0, 2)), np.array([]), np.array([]), np.array([]), ["x0", "x1"])
    with patch(
        "src.pipelines.walk_forward._build_batter_dataset",
        return_value=empty,
    ):
        result = walk_forward_market(
            "batter_hits", train_window_days=45, step_days=7, mode="sliding",
        )
    assert result["ran"] is False
    assert result["reason"] == "no_data"


def test_walk_forward_market_skips_when_min_train_rows_unmet():
    """Tiny dataset → every fold's training side is below min_train_rows → no folds."""
    data = _synthetic_dataset(n_days=60, rows_per_day=1)  # ~45 train rows max
    with patch(
        "src.pipelines.walk_forward._build_pitcher_dataset",
        return_value=data,
    ):
        result = walk_forward_market(
            "pitcher_strikeouts",
            train_window_days=45,
            step_days=7,
            mode="sliding",
            min_train_rows=500,
        )
    assert result["ran"] is False
    assert result["reason"] == "no_eligible_folds"


def test_expanding_mode_uses_more_train_rows_than_sliding():
    """Late folds in expanding mode have larger n_train than sliding mode."""
    data = _synthetic_dataset(n_days=120, rows_per_day=10)

    with patch("src.pipelines.walk_forward._build_pitcher_dataset", return_value=data):
        sliding = walk_forward_market(
            "pitcher_strikeouts", train_window_days=30, step_days=14,
            mode="sliding", min_train_rows=50,
        )
    with patch("src.pipelines.walk_forward._build_pitcher_dataset", return_value=data):
        expanding = walk_forward_market(
            "pitcher_strikeouts", train_window_days=30, step_days=14,
            mode="expanding", min_train_rows=50,
        )

    # Last fold of expanding mode trains on substantially more rows
    assert expanding["folds"][-1]["n_train"] > sliding["folds"][-1]["n_train"]
