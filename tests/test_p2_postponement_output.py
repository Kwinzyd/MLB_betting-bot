"""P2 audit-fix tests: postponed-game voiding and transparent alert output."""
from unittest.mock import patch

import pytest

from tests.fixtures.fixture_db import memory_conn
from src.clients.execution.telegram_venue import _format_projection


class _DBCtx:
    """Reusable single-connection context manager for patching get_db_connection."""
    def __init__(self, conn):
        self._conn = conn

    def __call__(self, *a, **k):
        return self

    def __enter__(self):
        return self._conn

    def __exit__(self, *a):
        return False


def _seed_postponed_alert(conn):
    conn.execute(
        "INSERT INTO games (game_id, home_team, away_team, venue, status) "
        "VALUES ('g1','Yankees','Red Sox','Yankee Stadium','POSTPONED')"
    )
    conn.execute(
        "INSERT INTO alerts_sent "
        "(player_name, market, line, side, odds, game_id, timestamp) "
        "VALUES ('Cole','pitcher_strikeouts',6.5,'over',1.91,'g1','2026-06-18T00:00:00')"
    )
    conn.commit()


class TestPostponementVoiding:
    def test_void_postponed_singles_refunds_open_bet(self):
        conn = memory_conn()
        _seed_postponed_alert(conn)
        with patch('src.pipelines.settle_results.get_db_connection', _DBCtx(conn)):
            from src.pipelines.settle_results import _void_postponed_singles
            n = _void_postponed_singles()
        assert n == 1
        row = conn.execute(
            "SELECT result, profit, settled_at FROM bet_results"
        ).fetchone()
        assert row['result'] == 'VOID'
        assert row['profit'] == 0.0
        assert row['settled_at'] is not None

    def test_void_is_idempotent(self):
        conn = memory_conn()
        _seed_postponed_alert(conn)
        with patch('src.pipelines.settle_results.get_db_connection', _DBCtx(conn)):
            from src.pipelines.settle_results import _void_postponed_singles
            assert _void_postponed_singles() == 1
            assert _void_postponed_singles() == 0  # already settled, not re-voided
        assert conn.execute("SELECT COUNT(*) c FROM bet_results").fetchone()['c'] == 1

    def test_grade_leg_voids_postponed_game(self):
        conn = memory_conn()
        _seed_postponed_alert(conn)
        with patch('src.pipelines.settle_results.get_db_connection', _DBCtx(conn)):
            from src.pipelines.settle_results import _grade_leg
            outcome = _grade_leg(
                {'player_name': 'Cole', 'market': 'pitcher_strikeouts',
                 'line': 6.5, 'side': 'over', 'odds': 1.91},
                'g1',
            )
        assert outcome == 'VOID'


class TestProjectionOutput:
    def test_count_band_uses_nb_alpha_when_present(self):
        # NB std = sqrt(mean + alpha*mean^2). mean=6, alpha=0.1 -> sqrt(6+3.6)=3.10
        out = _format_projection(
            "pitcher_strikeouts", 6.0, {"alpha": 0.1, "sample_size": 24})
        assert out.startswith("6.00 ± 3.10")
        assert "(n=24)" in out

    def test_count_band_defaults_to_poisson(self):
        # No alpha -> Poisson std = sqrt(mean). mean=4 -> 2.00
        out = _format_projection("batter_hits", 4.0, {"sample_size": 12})
        assert out.startswith("4.00 ± 2.00")
        assert "(n=12)" in out

    def test_total_bases_uses_sigma(self):
        out = _format_projection(
            "batter_total_bases", 1.5, {"sigma": 1.3, "sample_size": 30})
        assert out.startswith("1.50 ± 1.30")

    def test_missing_mean_is_safe(self):
        assert _format_projection("batter_hits", None, {}) == "n/a"
