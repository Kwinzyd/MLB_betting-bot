import sqlite3
from unittest.mock import patch

import pytest


@pytest.fixture
def conn():
    c = sqlite3.connect(':memory:')
    c.row_factory = sqlite3.Row
    c.executescript('''
        CREATE TABLE prop_snapshots (
            snapshot_id TEXT PRIMARY KEY, game_id TEXT, player_name TEXT,
            market TEXT, line REAL, over_odds REAL, under_odds REAL,
            bookmaker TEXT, timestamp TEXT, devigged_over REAL, devigged_under REAL
        );
    ''')
    return c


def _snap(c, snap_id, book, over, under, ts, line=6.5):
    c.execute(
        "INSERT INTO prop_snapshots VALUES (?, 'g1', 'Gerrit Cole', "
        "'pitcher_strikeouts', ?, ?, ?, ?, ?, NULL, NULL)",
        (snap_id, line, over, under, book, ts),
    )
    c.commit()


def test_clv_uses_pinnacle_when_available(conn):
    # Soft book is more recent but sharp snapshot must win.
    _snap(conn, 's1', 'draftkings', 2.50, 1.55, '2024-05-01T22:00:00')
    _snap(conn, 's2', 'pinnacle',   1.95, 1.95, '2024-05-01T21:00:00')

    with patch('src.pipelines.settle_results.get_db_connection') as mock_db, \
         patch('src.pipelines.settle_results.SHARP_BOOKMAKERS', ['pinnacle', 'circasports']):
        mock_db.return_value.__enter__.return_value = conn
        mock_db.return_value.__exit__.return_value = None
        from src.pipelines.settle_results import _calculate_clv
        clv = _calculate_clv('g1', 'Gerrit Cole', 'pitcher_strikeouts',
                             6.5, 'over', opening_odds=2.10)

    # Pinnacle 1.95/1.95 devigs to ~0.5/0.5. Opening implied 1/2.10 ~ 0.476.
    # Expect positive CLV around +0.024.
    assert 0.015 < clv < 0.035


def test_clv_falls_back_to_circa_when_no_pinnacle(conn):
    _snap(conn, 's1', 'fanduel',    2.30, 1.65, '2024-05-01T22:00:00')
    _snap(conn, 's2', 'circasports', 1.90, 2.00, '2024-05-01T21:00:00')

    with patch('src.pipelines.settle_results.get_db_connection') as mock_db, \
         patch('src.pipelines.settle_results.SHARP_BOOKMAKERS', ['pinnacle', 'circasports']):
        mock_db.return_value.__enter__.return_value = conn
        mock_db.return_value.__exit__.return_value = None
        from src.pipelines.settle_results import _calculate_clv
        clv = _calculate_clv('g1', 'Gerrit Cole', 'pitcher_strikeouts',
                             6.5, 'over', opening_odds=2.10)

    # Circa 1.90/2.00 used — devigged over ~ 0.513. Opening implied ~ 0.476.
    assert 0.03 < clv < 0.05


def test_clv_falls_back_to_any_book_when_no_sharp(conn):
    _snap(conn, 's1', 'draftkings', 2.10, 1.80, '2024-05-01T22:00:00')

    with patch('src.pipelines.settle_results.get_db_connection') as mock_db, \
         patch('src.pipelines.settle_results.SHARP_BOOKMAKERS', ['pinnacle', 'circasports']):
        mock_db.return_value.__enter__.return_value = conn
        mock_db.return_value.__exit__.return_value = None
        from src.pipelines.settle_results import _calculate_clv
        clv = _calculate_clv('g1', 'Gerrit Cole', 'pitcher_strikeouts',
                             6.5, 'over', opening_odds=2.10)

    # Fallback path returns a non-zero CLV rather than failing.
    assert isinstance(clv, float)
    assert clv != 0.0


def test_clv_returns_zero_when_no_snapshot(conn):
    with patch('src.pipelines.settle_results.get_db_connection') as mock_db, \
         patch('src.pipelines.settle_results.SHARP_BOOKMAKERS', ['pinnacle']):
        mock_db.return_value.__enter__.return_value = conn
        mock_db.return_value.__exit__.return_value = None
        from src.pipelines.settle_results import _calculate_clv
        clv = _calculate_clv('g1', 'Gerrit Cole', 'pitcher_strikeouts',
                             6.5, 'over', opening_odds=2.10)
    assert clv == 0.0


def test_clv_prefers_newer_pinnacle_snapshot(conn):
    _snap(conn, 's1', 'pinnacle', 2.20, 1.75, '2024-05-01T19:00:00')  # early
    _snap(conn, 's2', 'pinnacle', 1.95, 1.95, '2024-05-01T22:30:00')  # late / closer to first pitch

    with patch('src.pipelines.settle_results.get_db_connection') as mock_db, \
         patch('src.pipelines.settle_results.SHARP_BOOKMAKERS', ['pinnacle']):
        mock_db.return_value.__enter__.return_value = conn
        mock_db.return_value.__exit__.return_value = None
        from src.pipelines.settle_results import _calculate_clv
        clv = _calculate_clv('g1', 'Gerrit Cole', 'pitcher_strikeouts',
                             6.5, 'over', opening_odds=2.10)

    # Latest close (1.95/1.95) is used, not the early 2.20/1.75.
    assert 0.015 < clv < 0.035
