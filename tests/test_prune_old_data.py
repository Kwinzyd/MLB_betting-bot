import pytest
import sqlite3
from datetime import datetime, timedelta
from unittest.mock import patch


def iso(days_ago: int) -> str:
    """Return an ISO timestamp string `days_ago` days in the past."""
    return (datetime.utcnow() - timedelta(days=days_ago)).isoformat()


@pytest.fixture
def memory_db():
    """Full schema needed by prune_old_data — all prunable tables plus reference tables."""
    conn = sqlite3.connect(':memory:')
    conn.row_factory = sqlite3.Row
    conn.executescript('''
        CREATE TABLE games (
            game_id TEXT PRIMARY KEY, date TEXT, status TEXT,
            bdl_game_id INTEGER, historical INTEGER DEFAULT 0
        );
        CREATE TABLE pitcher_game_logs (
            game_id INTEGER, player_id INTEGER, date TEXT,
            PRIMARY KEY (game_id, player_id)
        );
        CREATE TABLE batter_game_logs (
            game_id INTEGER, player_id INTEGER, date TEXT,
            PRIMARY KEY (game_id, player_id)
        );
        CREATE TABLE prop_snapshots (
            snapshot_id TEXT PRIMARY KEY, timestamp TEXT
        );
        CREATE TABLE projections (
            id INTEGER PRIMARY KEY AUTOINCREMENT, timestamp TEXT
        );
        CREATE TABLE daily_lineups (
            id INTEGER PRIMARY KEY AUTOINCREMENT, date TEXT
        );
        CREATE TABLE probable_pitchers (
            id INTEGER PRIMARY KEY AUTOINCREMENT, date TEXT
        );
        CREATE TABLE injury_reports (
            id INTEGER PRIMARY KEY AUTOINCREMENT, date TEXT, player_name TEXT,
            UNIQUE(date, player_name)
        );
        CREATE TABLE alerts_sent (
            alert_id INTEGER PRIMARY KEY AUTOINCREMENT,
            player_name TEXT, market TEXT, line REAL, bookmaker TEXT,
            timestamp TEXT,
            UNIQUE(player_name, market, line, bookmaker)
        );
        CREATE TABLE bet_results (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            alert_id INTEGER,
            FOREIGN KEY(alert_id) REFERENCES alerts_sent(alert_id)
        );
        CREATE TABLE teams   (team_id INTEGER PRIMARY KEY);
        CREATE TABLE players (player_id INTEGER PRIMARY KEY);
        CREATE TABLE park_factors (venue TEXT PRIMARY KEY);
    ''')
    conn.commit()
    return conn


@patch('src.pipelines.prune_old_data.get_db_connection')
def test_prune_deletes_old_game_logs(mock_get_db, memory_db):
    """Game logs older than game_logs_days are removed; recent logs survive."""
    memory_db.execute("INSERT INTO pitcher_game_logs VALUES (1, 10, ?)", (iso(100),))
    memory_db.execute("INSERT INTO pitcher_game_logs VALUES (2, 10, ?)", (iso(10),))
    memory_db.execute("INSERT INTO batter_game_logs  VALUES (1, 20, ?)", (iso(100),))
    memory_db.execute("INSERT INTO batter_game_logs  VALUES (2, 20, ?)", (iso(10),))
    memory_db.commit()

    mock_get_db.return_value.__enter__.return_value = memory_db
    mock_get_db.return_value.__exit__.return_value = None

    from src.pipelines.prune_old_data import prune_old_data
    deleted = prune_old_data(game_logs_days=90)

    assert deleted["pitcher_game_logs"] == 1
    assert deleted["batter_game_logs"] == 1
    assert memory_db.execute("SELECT COUNT(*) FROM pitcher_game_logs").fetchone()[0] == 1
    assert memory_db.execute("SELECT COUNT(*) FROM batter_game_logs").fetchone()[0] == 1


@patch('src.pipelines.prune_old_data.get_db_connection')
def test_prune_only_removes_completed_games(mock_get_db, memory_db):
    """Only COMPLETED games past the cutoff are deleted; SCHEDULED games are untouched."""
    memory_db.execute("INSERT INTO games (game_id, date, status) VALUES ('old_completed', ?, 'COMPLETED')", (iso(100),))
    memory_db.execute("INSERT INTO games (game_id, date, status) VALUES ('old_scheduled', ?, 'SCHEDULED')", (iso(100),))
    memory_db.execute("INSERT INTO games (game_id, date, status) VALUES ('new_completed', ?, 'COMPLETED')", (iso(10),))
    memory_db.commit()

    mock_get_db.return_value.__enter__.return_value = memory_db
    mock_get_db.return_value.__exit__.return_value = None

    from src.pipelines.prune_old_data import prune_old_data
    deleted = prune_old_data(completed_games_days=90)

    assert deleted["games"] == 1
    remaining = {r["game_id"] for r in memory_db.execute("SELECT game_id FROM games").fetchall()}
    assert "old_completed" not in remaining
    assert "old_scheduled" in remaining
    assert "new_completed" in remaining


@patch('src.pipelines.prune_old_data.get_db_connection')
def test_prune_cascades_bet_results_before_alerts(mock_get_db, memory_db):
    """bet_results rows are deleted first so the FK constraint is never violated."""
    memory_db.execute(
        "INSERT INTO alerts_sent (player_name, market, line, bookmaker, timestamp) "
        "VALUES ('Cole', 'pitcher_strikeouts', 6.5, 'dk', ?)", (iso(100),)
    )
    alert_id = memory_db.execute("SELECT last_insert_rowid()").fetchone()[0]
    memory_db.execute("INSERT INTO bet_results (alert_id) VALUES (?)", (alert_id,))
    memory_db.commit()

    mock_get_db.return_value.__enter__.return_value = memory_db
    mock_get_db.return_value.__exit__.return_value = None

    from src.pipelines.prune_old_data import prune_old_data
    deleted = prune_old_data(alerts_days=90)

    assert deleted["bet_results"] == 1
    assert deleted["alerts_sent"] == 1
    assert memory_db.execute("SELECT COUNT(*) FROM bet_results").fetchone()[0] == 0
    assert memory_db.execute("SELECT COUNT(*) FROM alerts_sent").fetchone()[0] == 0


@patch('src.pipelines.prune_old_data.get_db_connection')
def test_prune_retains_recent_alerts(mock_get_db, memory_db):
    """Recent alerts_sent rows are not removed."""
    memory_db.execute(
        "INSERT INTO alerts_sent (player_name, market, line, bookmaker, timestamp) "
        "VALUES ('Judge', 'batter_hits', 1.5, 'fanduel', ?)", (iso(10),)
    )
    memory_db.commit()

    mock_get_db.return_value.__enter__.return_value = memory_db
    mock_get_db.return_value.__exit__.return_value = None

    from src.pipelines.prune_old_data import prune_old_data
    deleted = prune_old_data(alerts_days=90)

    assert deleted["alerts_sent"] == 0
    assert memory_db.execute("SELECT COUNT(*) FROM alerts_sent").fetchone()[0] == 1


@patch('src.pipelines.prune_old_data.get_db_connection')
def test_prune_hot_data(mock_get_db, memory_db):
    """prop_snapshots, projections, daily_lineups, probable_pitchers, injury_reports
    are pruned at the hot_data_days threshold."""
    memory_db.execute("INSERT INTO prop_snapshots   VALUES ('s1', ?)", (iso(5),))
    memory_db.execute("INSERT INTO prop_snapshots   VALUES ('s2', ?)", (iso(1),))
    memory_db.execute("INSERT INTO projections (timestamp) VALUES (?)",  (iso(5),))
    memory_db.execute("INSERT INTO projections (timestamp) VALUES (?)",  (iso(1),))
    memory_db.execute("INSERT INTO daily_lineups    (date) VALUES (?)",  (iso(5),))
    memory_db.execute("INSERT INTO probable_pitchers(date) VALUES (?)",  (iso(5),))
    memory_db.execute("INSERT INTO injury_reports   (date, player_name) VALUES (?, 'X')", (iso(5),))
    memory_db.commit()

    mock_get_db.return_value.__enter__.return_value = memory_db
    mock_get_db.return_value.__exit__.return_value = None

    from src.pipelines.prune_old_data import prune_old_data
    deleted = prune_old_data(hot_data_days=3)

    assert deleted["prop_snapshots"] == 1      # s2 (1 day ago) survives
    assert deleted["projections"] == 1
    assert deleted["daily_lineups"] == 1
    assert deleted["probable_pitchers"] == 1
    assert deleted["injury_reports"] == 1
    assert memory_db.execute("SELECT COUNT(*) FROM prop_snapshots").fetchone()[0] == 1


@patch('src.pipelines.prune_old_data.get_db_connection')
def test_prune_returns_zero_when_nothing_to_delete(mock_get_db, memory_db):
    """Returns all-zero dict and does not error when the database is already clean."""
    mock_get_db.return_value.__enter__.return_value = memory_db
    mock_get_db.return_value.__exit__.return_value = None

    from src.pipelines.prune_old_data import prune_old_data
    deleted = prune_old_data()

    assert all(v == 0 for v in deleted.values())
