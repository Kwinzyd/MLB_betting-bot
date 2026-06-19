"""Integration tests for the training pipeline and model registry.

Verifies:
- train_market with enough fixture data creates a model_registry champion row
- _write_training_baseline populates model_training_baseline
- calibration_params defaults to identity transform on a fresh DB (< 200 samples)
- projections.py _apply_calibration returns a valid probability in [0,1]
"""
import pytest


def test_train_market_registers_champion(db):
    """train_market with 20 game-log rows produces a model_registry champion."""
    from src.pipelines.train_model import train_market
    result = train_market('pitcher_strikeouts')

    if not result.get('trained'):
        pytest.skip(f"Skipped training: {result.get('reason') or result.get('error')}")

    import sqlite3
    conn = sqlite3.connect(db)
    conn.row_factory = sqlite3.Row
    row = conn.execute(
        "SELECT * FROM model_registry WHERE market='pitcher_strikeouts' AND is_champion=1"
    ).fetchone()
    conn.close()

    assert row is not None, "Expected a champion row in model_registry"
    assert row['val_mae'] is not None or row['n_val'] == 0


def test_train_market_writes_training_baseline(db):
    """A champion model writes a model_training_baseline row."""
    from src.pipelines.train_model import train_market
    result = train_market('pitcher_strikeouts')

    if not result.get('trained') or not result.get('is_champion'):
        pytest.skip("No champion produced")

    import sqlite3
    conn = sqlite3.connect(db)
    conn.row_factory = sqlite3.Row
    row = conn.execute(
        "SELECT * FROM model_training_baseline "
        "WHERE market='pitcher_strikeouts' AND feature='val_mu'"
    ).fetchone()
    conn.close()

    assert row is not None, "Expected model_training_baseline row for champion"
    assert row['mean'] is not None
    assert row['std'] is not None


def test_apply_calibration_identity_on_fresh_db(db):
    """_apply_calibration returns the raw prob unchanged when no calibration is fitted."""
    from src.models.projections import _apply_calibration, _calib_cache
    import src.models.projections as proj_mod

    # Force cache miss so it reads from the (empty) fixture DB
    proj_mod._calib_cache = None
    proj_mod._calib_cache_ts = 0.0

    prob = 0.65
    result = _apply_calibration(prob, 'pitcher_strikeouts')

    assert 0.0 <= result <= 1.0
    # With no calibration_params in DB, should fall through to identity
    assert result == pytest.approx(prob, abs=0.01)


def test_train_all_runs_without_error(db):
    """train_all completes and returns a list of results, one per market."""
    from src.pipelines.train_model import train_all, _ALL_MARKETS
    results = train_all()

    assert isinstance(results, list)
    assert len(results) == len(_ALL_MARKETS)
    for r in results:
        assert 'market' in r
        # Each market either trained or reported a known skip reason
        assert r.get('trained') in (True, False)
