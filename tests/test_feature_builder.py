"""Tests for the pitch-mix and volatility feature additions.

These features derive entirely from existing game_logs, so the same helper
runs at train and serve time — no skew risk, no new API spend.
"""
import math
import numpy as np
import pytest

from src.data.feature_builder import (
    PITCHER_FEATURE_NAMES,
    BATTER_FEATURE_NAMES,
    build_pitcher_features,
    build_batter_features,
    _stdev,
    _rolling_pitcher_hr_per_9,
    _rolling_pitcher_k_bb_ratio,
    _rolling_pitcher_start_stdev,
    _rolling_batter_game_stdev,
)


# ---------------------------------------------------------------------------
# Low-level helpers
# ---------------------------------------------------------------------------

def test_stdev_empty_returns_zero():
    assert _stdev([]) == 0.0


def test_stdev_single_value_returns_zero():
    assert _stdev([5.0]) == 0.0


def test_stdev_matches_numpy_population():
    vals = [8.0, 2.0, 9.0, 1.0, 6.0]
    assert _stdev(vals) == pytest.approx(float(np.std(vals, ddof=0)))


def test_k_bb_ratio_safe_when_zero_walks():
    logs = [{"strikeouts": 10, "walks": 0}, {"strikeouts": 8, "walks": 0}]
    # No div-by-zero; BB floored at 1 → ratio = 18
    assert _rolling_pitcher_k_bb_ratio(logs, 10) == pytest.approx(18.0)


def test_k_bb_ratio_normal_case():
    logs = [{"strikeouts": 10, "walks": 2}, {"strikeouts": 8, "walks": 2}]
    assert _rolling_pitcher_k_bb_ratio(logs, 10) == pytest.approx(4.5)


def test_hr_per_9_normal_case():
    logs = [
        {"innings_pitched": 6.0, "home_runs_allowed": 1},
        {"innings_pitched": 7.0, "home_runs_allowed": 2},
    ]
    # 3 HR / 13 IP * 9 = 2.077
    assert _rolling_pitcher_hr_per_9(logs, 10) == pytest.approx(3 * 9.0 / 13.0)


def test_hr_per_9_zero_ip_returns_zero():
    assert _rolling_pitcher_hr_per_9([{"innings_pitched": 0, "home_runs_allowed": 1}], 10) == 0.0


def test_pitcher_start_stdev_volatile_vs_steady():
    volatile = [{"strikeouts": k} for k in [8, 2, 9, 1, 10, 0]]
    steady = [{"strikeouts": 5} for _ in range(6)]
    assert _rolling_pitcher_start_stdev(volatile, "strikeouts", 10) > 3.0
    assert _rolling_pitcher_start_stdev(steady, "strikeouts", 10) == 0.0


def test_batter_game_stdev():
    logs = [{"hits": h} for h in [0, 2, 0, 3, 1]]
    assert _rolling_batter_game_stdev(logs, "hits", 15) == pytest.approx(
        float(np.std([0, 2, 0, 3, 1], ddof=0))
    )


# ---------------------------------------------------------------------------
# Feature vector integration
# ---------------------------------------------------------------------------

def test_pitcher_feature_vector_length_matches_names():
    logs = [
        {"date": "2025-05-01", "innings_pitched": 6.0, "strikeouts": 8,
         "walks": 2, "earned_runs": 2, "hits_allowed": 5, "pitches_thrown": 95,
         "home_runs_allowed": 1},
    ] * 10
    vec = build_pitcher_features(
        pitcher_logs=logs, market="pitcher_strikeouts",
        opponent_rate=0.22, venue="Wrigley Field", weather=None,
        ump_k_factor=1.0, projected_ip=5.5,
    )
    assert vec.shape == (len(PITCHER_FEATURE_NAMES),)


def test_batter_feature_vector_length_matches_names():
    logs = [
        {"date": "2025-05-01", "hits": 1, "total_bases": 2, "home_runs": 0,
         "strikeouts": 1, "plate_appearances": 4, "at_bats": 4},
    ] * 15
    vec = build_batter_features(
        batter_logs=logs, market="batter_hits",
        pitcher_hand="R", batter_hand="L", venue="Wrigley Field",
        weather=None, lineup_position=3, projected_pa=4.0,
    )
    assert vec.shape == (len(BATTER_FEATURE_NAMES),)


def test_pitcher_new_features_populated_at_correct_indices():
    # 4 starts: 10K/2BB/6IP/1HR and 0K/0BB/0IP/0HR alternating
    logs = [
        {"date": "2025-05-10", "innings_pitched": 6.0, "strikeouts": 10,
         "walks": 2, "earned_runs": 2, "hits_allowed": 5, "pitches_thrown": 95,
         "home_runs_allowed": 1},
        {"date": "2025-05-05", "innings_pitched": 6.0, "strikeouts": 4,
         "walks": 4, "earned_runs": 3, "hits_allowed": 6, "pitches_thrown": 90,
         "home_runs_allowed": 2},
        {"date": "2025-04-30", "innings_pitched": 7.0, "strikeouts": 12,
         "walks": 1, "earned_runs": 1, "hits_allowed": 3, "pitches_thrown": 100,
         "home_runs_allowed": 0},
        {"date": "2025-04-25", "innings_pitched": 5.0, "strikeouts": 6,
         "walks": 3, "earned_runs": 4, "hits_allowed": 7, "pitches_thrown": 85,
         "home_runs_allowed": 1},
    ]
    vec = build_pitcher_features(
        pitcher_logs=logs, market="pitcher_strikeouts",
        opponent_rate=0.22, venue=None, weather=None,
        ump_k_factor=1.0, projected_ip=6.0,
    )

    idx = {name: i for i, name in enumerate(PITCHER_FEATURE_NAMES)}
    # K/BB: 32/10 = 3.2
    assert vec[idx["k_bb_ratio_l10"]] == pytest.approx(32.0 / 10.0)
    # HR/9: 4 HR over 24 IP * 9 = 1.5
    assert vec[idx["hr_per_9_l10"]] == pytest.approx(4 * 9.0 / 24.0)
    # Volatility non-zero (strikeouts vary 4,6,10,12)
    assert vec[idx["k_volatility_l10"]] > 0.0
    assert vec[idx["ip_volatility_l10"]] > 0.0


def test_batter_new_features_populated():
    logs = [
        {"date": "2025-05-10", "hits": 0, "total_bases": 0, "home_runs": 0,
         "strikeouts": 2, "plate_appearances": 4, "at_bats": 4},
        {"date": "2025-05-09", "hits": 3, "total_bases": 5, "home_runs": 1,
         "strikeouts": 0, "plate_appearances": 4, "at_bats": 4},
        {"date": "2025-05-08", "hits": 0, "total_bases": 0, "home_runs": 0,
         "strikeouts": 3, "plate_appearances": 4, "at_bats": 4},
        {"date": "2025-05-07", "hits": 2, "total_bases": 3, "home_runs": 0,
         "strikeouts": 1, "plate_appearances": 4, "at_bats": 4},
    ]
    vec = build_batter_features(
        batter_logs=logs, market="batter_hits",
        pitcher_hand="R", batter_hand="R", venue=None, weather=None,
        lineup_position=4, projected_pa=4.0,
    )

    idx = {name: i for i, name in enumerate(BATTER_FEATURE_NAMES)}
    # Hits vary (0, 3, 0, 2) — stdev > 0
    assert vec[idx["stat_volatility_l15"]] > 0.0
    # K rate: 6 K / 16 PA = 0.375
    assert vec[idx["k_rate_l15"]] == pytest.approx(6.0 / 16.0)


def test_steady_batter_has_lower_volatility_than_streaky():
    steady = [{"date": "d", "hits": 1, "total_bases": 1, "home_runs": 0,
               "strikeouts": 1, "plate_appearances": 4, "at_bats": 4}] * 10
    # Same total (10 hits / 40 PA = 0.25) but alternating pattern, not flat.
    streaky = [{"date": "d", "hits": h, "total_bases": h, "home_runs": 0,
                "strikeouts": 1, "plate_appearances": 4, "at_bats": 4}
               for h in [0, 2, 0, 2, 0, 2, 0, 2, 0, 2]]

    v_steady = build_batter_features(
        batter_logs=steady, market="batter_hits",
        pitcher_hand="R", batter_hand="R", venue=None, weather=None,
        lineup_position=4,
    )
    v_streaky = build_batter_features(
        batter_logs=streaky, market="batter_hits",
        pitcher_hand="R", batter_hand="R", venue=None, weather=None,
        lineup_position=4,
    )
    idx = BATTER_FEATURE_NAMES.index("stat_volatility_l15")
    assert v_streaky[idx] > v_steady[idx]
    # Same mean hit rate (0.25/PA) in both cases — proves volatility is
    # independent of the mean, which is the whole point of the feature.
    mean_idx = BATTER_FEATURE_NAMES.index("recent_rate_per_pa_l15")
    assert math.isclose(v_steady[mean_idx], v_streaky[mean_idx], abs_tol=1e-9)


def test_empty_logs_return_neutral_features():
    vec = build_pitcher_features(
        pitcher_logs=[], market="pitcher_strikeouts",
        opponent_rate=None, venue=None, weather=None,
        ump_k_factor=1.0, projected_ip=5.5,
    )
    idx = {name: i for i, name in enumerate(PITCHER_FEATURE_NAMES)}
    assert vec[idx["k_bb_ratio_l10"]] == 0.0
    assert vec[idx["hr_per_9_l10"]] == 0.0
    assert vec[idx["k_volatility_l10"]] == 0.0
    assert vec[idx["ip_volatility_l10"]] == 0.0