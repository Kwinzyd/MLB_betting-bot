import sqlite3
from datetime import datetime, timezone, timedelta
from unittest.mock import AsyncMock, patch

import pytest


@pytest.fixture
def memory_db():
    conn = sqlite3.connect(':memory:')
    conn.row_factory = sqlite3.Row
    conn.executescript('''
        CREATE TABLE game_totals_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            game_id TEXT NOT NULL,
            total REAL NOT NULL,
            source TEXT NOT NULL,
            timestamp TEXT NOT NULL
        );
    ''')
    return conn


def _seed(conn, game_id='g1', total=7.5, age_minutes=15.0, source='scan'):
    ts = (datetime.now(timezone.utc) - timedelta(minutes=age_minutes)).isoformat()
    conn.execute(
        "INSERT INTO game_totals_history (game_id, total, source, timestamp) "
        "VALUES (?, ?, ?, ?)",
        (game_id, total, source, ts),
    )
    conn.commit()


def _totals_payload(point: float) -> dict:
    return {
        'bookmakers': [{
            'markets': [{
                'key': 'totals',
                'outcomes': [
                    {'name': 'Over', 'point': point},
                    {'name': 'Under', 'point': point},
                ],
            }],
        }],
    }


class TestCheckTotalShift:
    @patch('src.pipelines.trigger_watch._record_total_snapshot')
    @patch('src.pipelines.trigger_watch.get_db_connection')
    async def test_shift_above_threshold_fires(self, mock_get_db, _rec, memory_db):
        _seed(memory_db, total=7.5, age_minutes=15.0)
        mock_get_db.return_value.__enter__.return_value = memory_db
        mock_get_db.return_value.__exit__.return_value = None

        odds_client = AsyncMock()
        odds_client.get_event_odds = AsyncMock(return_value=_totals_payload(8.5))

        from src.pipelines.trigger_watch import _check_total_shift
        result = await _check_total_shift('g1', odds_client)

        assert result is not None
        assert result['prior_total'] == 7.5
        assert result['current_total'] == 8.5
        assert result['delta'] == pytest.approx(1.0)

    @patch('src.pipelines.trigger_watch._record_total_snapshot')
    @patch('src.pipelines.trigger_watch.get_db_connection')
    async def test_shift_below_threshold_skips(self, mock_get_db, mock_rec, memory_db):
        _seed(memory_db, total=7.5, age_minutes=15.0)
        mock_get_db.return_value.__enter__.return_value = memory_db
        mock_get_db.return_value.__exit__.return_value = None

        odds_client = AsyncMock()
        odds_client.get_event_odds = AsyncMock(return_value=_totals_payload(7.6))

        from src.pipelines.trigger_watch import _check_total_shift
        result = await _check_total_shift('g1', odds_client)

        assert result is None
        # Fresh datapoint still recorded so the next delta measures from now.
        mock_rec.assert_called_once_with('g1', 7.6, source='trigger')

    @patch('src.pipelines.trigger_watch.get_db_connection')
    async def test_no_history_skips(self, mock_get_db, memory_db):
        # No seed.
        mock_get_db.return_value.__enter__.return_value = memory_db
        mock_get_db.return_value.__exit__.return_value = None

        odds_client = AsyncMock()
        odds_client.get_event_odds = AsyncMock(return_value=_totals_payload(8.5))

        from src.pipelines.trigger_watch import _check_total_shift
        result = await _check_total_shift('g1', odds_client)

        assert result is None
        # No baseline = no fetch attempted.
        odds_client.get_event_odds.assert_not_called()

    @patch('src.pipelines.trigger_watch.get_db_connection')
    async def test_fresh_history_skips(self, mock_get_db, memory_db):
        _seed(memory_db, total=7.5, age_minutes=1.0)  # < 5 min default
        mock_get_db.return_value.__enter__.return_value = memory_db
        mock_get_db.return_value.__exit__.return_value = None

        odds_client = AsyncMock()
        odds_client.get_event_odds = AsyncMock(return_value=_totals_payload(8.5))

        from src.pipelines.trigger_watch import _check_total_shift
        result = await _check_total_shift('g1', odds_client)

        assert result is None
        odds_client.get_event_odds.assert_not_called()

    @patch('src.pipelines.trigger_watch._record_total_snapshot')
    @patch('src.pipelines.trigger_watch.get_db_connection')
    async def test_fetch_failure_returns_none(self, mock_get_db, _rec, memory_db):
        _seed(memory_db, total=7.5, age_minutes=15.0)
        mock_get_db.return_value.__enter__.return_value = memory_db
        mock_get_db.return_value.__exit__.return_value = None

        odds_client = AsyncMock()
        odds_client.get_event_odds = AsyncMock(side_effect=RuntimeError("api down"))

        from src.pipelines.trigger_watch import _check_total_shift
        result = await _check_total_shift('g1', odds_client)

        assert result is None  # swallowed, trigger watch keeps running


class TestFormatDetail:
    def test_renders_total_shift(self):
        from src.pipelines.trigger_watch import _format_detail
        msg = _format_detail('total_shift', {
            'prior_total': 7.5,
            'current_total': 8.5,
            'delta': 1.0,
            'prior_age_min': 12.4,
        })
        assert "7.5 → 8.5" in msg
        assert "Δ=+1.0" in msg
        assert "12.4min" in msg

    def test_renders_negative_delta(self):
        from src.pipelines.trigger_watch import _format_detail
        msg = _format_detail('total_shift', {
            'prior_total': 9.0,
            'current_total': 8.0,
            'delta': -1.0,
            'prior_age_min': 8.0,
        })
        assert "Δ=-1.0" in msg
