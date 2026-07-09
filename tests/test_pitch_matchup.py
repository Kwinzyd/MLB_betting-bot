import pytest

from src.models import pitch_matchup as pm


def _p(pitch_type, count, usage=None, whiff=None, xwoba=None):
    return {"pitch_type": pitch_type, "pitch_count": count, "usage_pct": usage,
            "whiff_pct": whiff, "xwoba": xwoba, "contact_pct": None,
            "pa_count": count, "strikeout_count": 0}


# --- profile builders ---

def test_pitcher_usage_normalizes_and_floors():
    rows = [_p("FF", 100, usage=60.0), _p("SL", 40, usage=30.0),
            _p("CH", 5, usage=10.0)]  # CH below the 20-pitch floor
    usage = pm._pitcher_usage(rows, min_pitches=20)
    assert set(usage) == {"FF", "SL"}
    assert usage["FF"] == pytest.approx(60 / 90)
    assert usage["SL"] == pytest.approx(30 / 90)
    assert sum(usage.values()) == pytest.approx(1.0)


def test_metric_by_pitch_drops_null_and_thin():
    rows = [_p("FF", 100, whiff=25.0), _p("SL", 10, whiff=40.0), _p("CH", 100, whiff=None)]
    prof = pm._metric_by_pitch(rows, "whiff_pct", min_pitches=20)
    assert set(prof) == {"FF"}
    assert prof["FF"]["val"] == 25.0


def test_natural_rate_is_count_weighted():
    prof = {"FF": {"val": 20.0, "count": 300.0}, "SL": {"val": 40.0, "count": 100.0}}
    # (20*300 + 40*100) / 400 = 25
    assert pm._natural_rate(prof) == pytest.approx(25.0)


def test_arsenal_weighted_renormalizes_over_covered():
    usage = {"FF": 0.5, "SL": 0.3, "CU": 0.2}       # CU not in profile
    prof = {"FF": {"val": 10.0, "count": 1}, "SL": {"val": 30.0, "count": 1}}
    weighted, covered = pm._arsenal_weighted(usage, prof)
    # covered usage = 0.8; renormalized FF=0.625, SL=0.375 -> 10*.625+30*.375=17.5
    assert covered == pytest.approx(0.8)
    assert weighted == pytest.approx(17.5)


# --- factor computations ---

def test_batter_factor_above_one_when_arsenal_favors_batter(monkeypatch):
    monkeypatch.setattr(pm, "PITCH_MATCHUP_SENS", 1.0)
    monkeypatch.setattr(pm, "PITCH_MATCHUP_MIN", 0.5)
    monkeypatch.setattr(pm, "PITCH_MATCHUP_MAX", 2.0)
    # Pitcher throws mostly FF; batter crushes FF (high xwoba) vs their baseline.
    pitcher = [_p("FF", 500, usage=80.0), _p("SL", 100, usage=20.0)]
    batter = [_p("FF", 300, xwoba=0.450), _p("SL", 300, xwoba=0.250)]
    f = pm.batter_arsenal_factor(batter, pitcher, min_pitches=20, min_coverage=0.5)
    # arsenal xwoba ~ .410 (FF-heavy) vs natural .350 -> factor > 1
    assert f > 1.0


def test_pitcher_k_factor_below_one_when_lineup_handles_arsenal(monkeypatch):
    monkeypatch.setattr(pm, "PITCH_MATCHUP_SENS", 1.0)
    monkeypatch.setattr(pm, "PITCH_MATCHUP_MIN", 0.5)
    monkeypatch.setattr(pm, "PITCH_MATCHUP_MAX", 2.0)
    pitcher = [_p("FF", 500, usage=90.0), _p("SL", 60, usage=10.0)]
    # Both hitters whiff LESS on FF than their baseline -> factor < 1.
    lineup = [
        [_p("FF", 200, whiff=10.0), _p("SL", 200, whiff=40.0)],
        [_p("FF", 200, whiff=12.0), _p("SL", 200, whiff=44.0)],
    ]
    f = pm.pitcher_k_factor(pitcher, lineup, min_pitches=20, min_coverage=0.5)
    assert f < 1.0


def test_low_coverage_returns_neutral():
    pitcher = [_p("FF", 500, usage=90.0), _p("KN", 100, usage=10.0)]
    # Batter only has data on a pitch the pitcher barely throws -> coverage < 0.5
    batter = [_p("SL", 300, xwoba=0.500)]
    assert pm.batter_arsenal_factor(batter, pitcher, min_pitches=20, min_coverage=0.5) == 1.0


def test_empty_inputs_return_neutral():
    assert pm.batter_arsenal_factor([], [], min_coverage=0.5) == 1.0
    assert pm.pitcher_k_factor([], [], min_coverage=0.5) == 1.0
    assert pm.pitcher_k_factor([_p("FF", 500, usage=100.0)], [], min_coverage=0.5) == 1.0


def test_clamp_bounds(monkeypatch):
    monkeypatch.setattr(pm, "PITCH_MATCHUP_SENS", 1.0)
    monkeypatch.setattr(pm, "PITCH_MATCHUP_MIN", 0.90)
    monkeypatch.setattr(pm, "PITCH_MATCHUP_MAX", 1.10)
    assert pm._clamp_factor(5.0) == 1.10   # huge ratio clamps to max
    assert pm._clamp_factor(0.01) == 0.90  # tiny ratio clamps to min
    assert pm._clamp_factor(None) == 1.0
    assert pm._clamp_factor(0.0) == 1.0


def test_sensitivity_dampens(monkeypatch):
    monkeypatch.setattr(pm, "PITCH_MATCHUP_MIN", 0.5)
    monkeypatch.setattr(pm, "PITCH_MATCHUP_MAX", 2.0)
    monkeypatch.setattr(pm, "PITCH_MATCHUP_SENS", 0.5)
    # ratio 1.44 with sens 0.5 -> 1.2
    assert pm._clamp_factor(1.44) == pytest.approx(1.2)
