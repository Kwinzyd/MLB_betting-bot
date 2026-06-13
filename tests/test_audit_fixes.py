"""Regression tests for the 2026-06 quant audit fixes.

Each test pins one corrected behavior:
  - parse_baseball_ip: "5.2" means 5 2/3 innings, not 5.2
  - push-aware pricing at integer lines (counts + Normal)
  - calibration preserves P(over) + P(under) = 1
  - fractional_kelly honors an explicit fraction of 0.0; bankroll floors at 0
  - CLV compares devigged close to devigged open
  - steam + line-movement multipliers cap at 1.5x combined
  - bookmaker-bias boost only rescues bets killed solely by the edge bar
  - sync_stats marks Final games COMPLETED (the settle trigger)
"""
import asyncio
import logging
import sqlite3
from unittest.mock import patch, AsyncMock

import pytest

from src.utils.validators import parse_baseball_ip


# ---------------------------------------------------------------------------
# parse_baseball_ip
# ---------------------------------------------------------------------------

class TestParseBaseballIP:
    def test_notation_thirds(self):
        assert parse_baseball_ip("5.2") == pytest.approx(5 + 2 / 3)
        assert parse_baseball_ip(6.1) == pytest.approx(6 + 1 / 3)

    def test_whole_innings_unchanged(self):
        assert parse_baseball_ip("7.0") == 7.0
        assert parse_baseball_ip(0) == 0.0

    def test_true_decimals_pass_through(self):
        # Already-converted values must not be double-converted.
        assert parse_baseball_ip(5.667) == pytest.approx(5.667)
        assert parse_baseball_ip(6.333) == pytest.approx(6.333)

    def test_garbage_is_zero(self):
        assert parse_baseball_ip(None) == 0.0
        assert parse_baseball_ip("abc") == 0.0
        assert parse_baseball_ip(-1.2) == 0.0


# ---------------------------------------------------------------------------
# Push-aware pricing
# ---------------------------------------------------------------------------

class TestPushAwarePricing:
    def test_integer_line_excludes_push_from_under(self):
        from src.models.distributions import get_probabilities
        from scipy.stats import poisson
        mean, line = 6.0, 6.0
        p_over, p_under = get_probabilities(mean, line, "pitcher_strikeouts")
        # Conditional on no push: P(X>6)/(1-pmf(6)) and P(X<6)/(1-pmf(6))
        push = poisson.pmf(6, mean)
        raw_over = 1 - poisson.cdf(6, mean)
        raw_under = poisson.cdf(5, mean)
        assert p_over == pytest.approx(raw_over / (raw_over + raw_under))
        assert p_under == pytest.approx(raw_under / (raw_over + raw_under))
        assert p_over + p_under == pytest.approx(1.0)
        assert push > 0.1  # the push mass is material at the mean

    def test_half_line_unchanged(self):
        from src.models.distributions import get_probabilities
        from scipy.stats import poisson
        mean, line = 6.0, 6.5
        p_over, p_under = get_probabilities(mean, line, "pitcher_strikeouts")
        assert p_over == pytest.approx(1 - poisson.cdf(6, mean))
        assert p_under == pytest.approx(poisson.cdf(6, mean))

    def test_total_bases_integer_line_continuity_corrected(self):
        from src.models.distributions import get_probabilities
        p_over, p_under = get_probabilities(1.8, 2.0, "batter_total_bases")
        assert p_over + p_under == pytest.approx(1.0)
        # Push band (1.5, 2.5) straddles the mean: over prob must be lower
        # than the naive P(X > 2.0) split.
        from scipy.stats import norm
        import math
        std = max(0.5, math.sqrt(1.8 * 1.2))
        naive_over = norm.sf(2.0, loc=1.8, scale=std)
        assert p_over < naive_over

    def test_mixture_normalizes_once(self):
        from src.models.distributions import get_probabilities_mixture
        pa_dist = {3: 0.2, 4: 0.5, 5: 0.3}
        p_over, p_under = get_probabilities_mixture(
            0.35, 1.0, "batter_hits", pa_dist)
        assert p_over + p_under == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# Calibration axioms
# ---------------------------------------------------------------------------

class TestCalibrationComplement:
    def test_probs_sum_to_one_after_platt(self):
        from src.models import projections as proj_mod
        with patch.object(proj_mod, '_load_calibration', return_value={
            'pitcher_strikeouts': {'method': 'platt', 'params': {'a': 1.3, 'b': 0.4}},
        }):
            result = proj_mod._calibrate_result({
                'market': 'pitcher_strikeouts',
                'prob_over': 0.60, 'prob_under': 0.40,
            })
        assert result['prob_over'] != pytest.approx(0.60)  # transform applied
        assert result['prob_over'] + result['prob_under'] == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# Kelly
# ---------------------------------------------------------------------------

class TestKelly:
    def test_explicit_zero_fraction_means_zero_stake(self):
        from src.models.kelly import fractional_kelly
        out = fractional_kelly(0.60, 2.0, fraction=0.0, bankroll=1000.0)
        assert out['kelly_fraction'] == 0.0
        assert out['recommended_stake'] == 0.0

    def test_bankroll_floors_at_zero(self):
        from tests.fixtures.fixture_db import memory_conn
        conn = memory_conn()
        conn.execute(
            "INSERT INTO alerts_sent (player_name, market, line, bookmaker, game_id, timestamp) "
            "VALUES ('X', 'm', 1.5, 'dk', 'g', '2026-01-01T00:00:00')")
        alert_id = conn.execute("SELECT last_insert_rowid()").fetchone()[0]
        conn.execute(
            "INSERT INTO bet_results (alert_id, result, profit) VALUES (?, 'LOSS', -999999.0)",
            (alert_id,))
        conn.commit()
        with patch('src.models.kelly.get_db_connection') as mock_db:
            mock_db.return_value.__enter__.return_value = conn
            mock_db.return_value.__exit__.return_value = None
            from src.models.kelly import get_current_bankroll
            assert get_current_bankroll() == 0.0

    def test_sgp_profit_counts_toward_bankroll(self):
        from tests.fixtures.fixture_db import memory_conn
        conn = memory_conn()
        conn.execute(
            "INSERT INTO sgp_candidates (game_id, legs_json, joint_prob, timestamp) "
            "VALUES ('g', '[]', 0.3, '2026-01-01T00:00:00')")
        sgp_id = conn.execute("SELECT last_insert_rowid()").fetchone()[0]
        conn.execute(
            "INSERT INTO sgp_results (sgp_candidate_id, result, profit) VALUES (?, 'WIN', 120.0)",
            (sgp_id,))
        conn.commit()
        with patch('src.models.kelly.get_db_connection') as mock_db, \
             patch('src.models.kelly.BANKROLL', 1000.0):
            mock_db.return_value.__enter__.return_value = conn
            mock_db.return_value.__exit__.return_value = None
            from src.models.kelly import get_current_bankroll
            assert get_current_bankroll() == pytest.approx(1120.0)


# ---------------------------------------------------------------------------
# CLV basis
# ---------------------------------------------------------------------------

def test_clv_uses_devigged_open_when_available():
    conn = sqlite3.connect(':memory:')
    conn.row_factory = sqlite3.Row
    conn.executescript('''
        CREATE TABLE prop_snapshots (
            snapshot_id TEXT PRIMARY KEY, game_id TEXT, player_name TEXT,
            market TEXT, line REAL, over_odds REAL, under_odds REAL,
            bookmaker TEXT, timestamp TEXT, devigged_over REAL, devigged_under REAL
        );
    ''')
    # Closing sharp quote 1.95/1.95 -> devigged 0.50/0.50
    conn.execute(
        "INSERT INTO prop_snapshots VALUES ('s1', 'g1', 'Cole', "
        "'pitcher_strikeouts', 6.5, 1.95, 1.95, 'pinnacle', "
        "'2024-05-01T22:00:00', NULL, NULL)")
    conn.commit()

    with patch('src.pipelines.settle_results.get_db_connection') as mock_db, \
         patch('src.pipelines.settle_results.SHARP_BOOKMAKERS', ['pinnacle']):
        mock_db.return_value.__enter__.return_value = conn
        mock_db.return_value.__exit__.return_value = None
        from src.pipelines.settle_results import _calculate_clv
        # Devigged open of 0.46 vs devigged close 0.50 -> CLV = +0.04 exactly.
        # The raw odds (2.10 -> implied 0.476, vig included) must be ignored.
        clv = _calculate_clv('g1', 'Cole', 'pitcher_strikeouts', 6.5, 'over',
                             opening_odds=2.10, opening_prob=0.46)
    assert clv == pytest.approx(0.04, abs=1e-6)


# ---------------------------------------------------------------------------
# Multiplier stacking cap
# ---------------------------------------------------------------------------

def test_steam_and_line_move_multipliers_do_not_stack():
    from src.models.edge_ranker import rank_edge
    from src.config import KELLY_FRACTION
    projection = {
        'prob_over': 0.55, 'prob_under': 0.45,
        'injury_status': 'Healthy', 'sample_size': 30,
    }
    # sharp 0.55 vs opening 0.53 -> diff +0.02 -> line-move boost 1.5x.
    # steam_detected on top must NOT push the combined boost to 2.25x.
    result = rank_edge(projection, 2.0, 'over', sharp_prob=0.55,
                       opening_prob=0.53, steam_detected=True)
    full_kelly = (0.55 * 1.0 - 0.45) / 1.0  # 0.10
    expected = full_kelly * KELLY_FRACTION * 1.5
    assert result['kelly']['kelly_fraction'] == pytest.approx(expected, abs=1e-4)


# ---------------------------------------------------------------------------
# Bias boost: edge must be the sole kill reason
# ---------------------------------------------------------------------------

class TestBiasBoostGate:
    """Drive the scan loop with a candidate whose edge lands just under
    EDGE_MIN, so the bias credit pushes it over — and verify the rescue only
    happens when nothing else killed the bet."""

    def _run_scan(self, sample_size, caplog):
        from tests.test_line_shopping import (
            _payload, _patch_stack, _enter, _exit, _seed_schema,
        )
        conn = sqlite3.connect(':memory:')
        conn.row_factory = sqlite3.Row
        _seed_schema(conn)
        conn.execute(
            "INSERT INTO games (game_id, home_team, away_team, venue, status) "
            "VALUES ('g1', 'Yankees', 'Red Sox', 'Yankee Stadium', 'SCHEDULED')")
        conn.commit()

        # Sharp devig at 7.5 (1.74/2.20) ~ 0.559 over; soft over 1.95 implied
        # 0.513 -> edge ~4.6% (< EDGE_MIN 5.0). Bias credit +0.5 -> ~5.1%.
        payload = _payload('Cole', 'pitcher_strikeouts', {
            'pinnacle': [(7.5, 1.74, 2.20)],
            'draftkings': [(7.5, 1.95, 2.20)],
        })
        projection = {
            'player_name': 'Cole', 'market': 'pitcher_strikeouts', 'line': 7.5,
            'projected_mean': 6.5, 'prob_over': 0.56, 'prob_under': 0.44,
            'alpha': None, 'sigma': None, 'pi0': None,
            'injury_status': 'Healthy', 'sample_size': sample_size,
            'context': {'model': 'test'},
        }
        stack = _patch_stack(conn, payload, projection, {7.5: (0.56, 0.44)})
        stack.append(patch(
            'src.pipelines.scan_props._load_bookmaker_bias',
            return_value={('draftkings', 'pitcher_strikeouts', 'over'): 0.05},
        ))
        _enter(stack)
        try:
            from src.pipelines.scan_props import scan_props
            with caplog.at_level(logging.INFO):
                asyncio.run(scan_props(force=True))
        finally:
            _exit(stack)
        return caplog.text

    def test_boost_rescues_edge_only_kill(self, caplog):
        text = self._run_scan(sample_size=30, caplog=caplog)
        assert "EDGE FOUND" in text

    def test_boost_cannot_override_sample_size_kill(self, caplog):
        # sample_size=5 < MIN_SAMPLE_SIZE -> two kill reasons -> no rescue.
        text = self._run_scan(sample_size=5, caplog=caplog)
        assert "EDGE FOUND" not in text


# ---------------------------------------------------------------------------
# Game lifecycle: sync_stats marks Final games COMPLETED
# ---------------------------------------------------------------------------

@patch('src.pipelines.sync_stats.MLBStatsClient')
@patch('src.pipelines.sync_stats.get_db_connection')
def test_sync_stats_marks_completed(mock_get_db, mock_bdl_cls):
    from tests.fixtures.fixture_db import memory_conn
    conn = memory_conn()
    conn.execute(
        "INSERT INTO games (game_id, bdl_game_id, date, home_team, away_team, status) "
        "VALUES ('odds_g9', 999, '2026-06-11', 'Yankees', 'Red Sox', 'SCHEDULED')")
    conn.commit()

    mock_get_db.return_value.__enter__.return_value = conn
    mock_get_db.return_value.__exit__.return_value = None

    client = mock_bdl_cls.return_value
    client.get_teams = AsyncMock(return_value=[])
    client.get_games = AsyncMock(return_value=[
        {'id': 999, 'status': 'Final',
         'home_team': {'id': 1}, 'away_team': {'id': 2},
         'date': '2026-06-11'},
    ])
    client.get_stats_batch = AsyncMock(return_value=[])

    from src.pipelines.sync_stats import sync_stats
    asyncio.run(sync_stats())

    row = conn.execute(
        "SELECT status FROM games WHERE bdl_game_id = 999").fetchone()
    assert row['status'] == 'COMPLETED'
