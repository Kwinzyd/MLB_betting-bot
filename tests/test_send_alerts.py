import pytest
import sqlite3
from unittest.mock import patch, MagicMock


@pytest.fixture
def memory_db():
    """In-memory DB seeded with a playable projection and matching prop snapshot."""
    conn = sqlite3.connect(':memory:')
    conn.row_factory = sqlite3.Row
    conn.executescript('''
        CREATE TABLE games (
            game_id TEXT PRIMARY KEY, home_team TEXT, away_team TEXT,
            venue TEXT, status TEXT, date TEXT
        );
        CREATE TABLE projections (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            game_id TEXT, player_name TEXT, market TEXT,
            projected_mean REAL, prob_over REAL, prob_under REAL,
            context_json TEXT, timestamp TEXT,
            UNIQUE(game_id, player_name, market)
        );
        CREATE TABLE prop_snapshots (
            snapshot_id TEXT PRIMARY KEY, game_id TEXT, player_name TEXT,
            market TEXT, line REAL, over_odds REAL, under_odds REAL,
            bookmaker TEXT, timestamp TEXT, devigged_over REAL, devigged_under REAL
        );
        CREATE TABLE alerts_sent (
            alert_id INTEGER PRIMARY KEY AUTOINCREMENT,
            player_name TEXT, market TEXT, line REAL, side TEXT,
            edge REAL, ev REAL, kelly_stake REAL, bookmaker TEXT,
            odds REAL, opening_odds REAL, game_id TEXT, timestamp TEXT,
            UNIQUE(player_name, market, line, bookmaker)
        );
    ''')
    conn.execute('''
        INSERT INTO games VALUES
        ('g1', 'Yankees', 'Red Sox', 'Yankee Stadium', 'SCHEDULED', '2024-05-15')
    ''')
    # prob_over=0.65, devigged_over=0.50 → edge=15% (playable, above 5% threshold)
    conn.execute('''
        INSERT INTO projections
        (game_id, player_name, market, projected_mean, prob_over, prob_under, context_json, timestamp)
        VALUES ('g1', 'Gerrit Cole', 'pitcher_strikeouts', 7.5, 0.65, 0.35, '{}', '2024-05-15T12:00:00')
    ''')
    conn.execute('''
        INSERT INTO prop_snapshots VALUES
        ('snap1', 'g1', 'Gerrit Cole', 'pitcher_strikeouts', 6.5, 2.0, 1.8,
         'draftkings', '2024-05-15T12:00:00', 0.50, 0.50)
    ''')
    conn.commit()
    return conn


@patch('src.pipelines.send_alerts.get_db_connection')
@patch('src.pipelines.send_alerts.TelegramClient')
def test_send_alerts_new_alert_inserted(mock_telegram_cls, mock_get_db, memory_db):
    """A playable edge triggers a Telegram message and is recorded in alerts_sent."""
    mock_bot = mock_telegram_cls.return_value
    mock_get_db.return_value.__enter__.return_value = memory_db
    mock_get_db.return_value.__exit__.return_value = None

    from src.pipelines.send_alerts import send_alerts
    send_alerts()

    mock_bot.send_message.assert_called_once()

    alert = memory_db.execute("SELECT * FROM alerts_sent").fetchone()
    assert alert is not None
    assert alert['player_name'] == 'Gerrit Cole'
    assert alert['market'] == 'pitcher_strikeouts'
    assert alert['side'] == 'over'
    assert alert['bookmaker'] == 'draftkings'


@patch('src.pipelines.send_alerts.get_db_connection')
@patch('src.pipelines.send_alerts.TelegramClient')
def test_send_alerts_deduplication(mock_telegram_cls, mock_get_db, memory_db):
    """An already-alerted prop is not sent again."""
    memory_db.execute('''
        INSERT INTO alerts_sent
        (player_name, market, line, side, edge, ev, kelly_stake,
         bookmaker, odds, opening_odds, game_id, timestamp)
        VALUES ('Gerrit Cole', 'pitcher_strikeouts', 6.5, 'over', 15.0, 0.30,
                10.0, 'draftkings', 2.0, 2.0, 'g1', '2024-05-15T11:00:00')
    ''')
    memory_db.commit()

    mock_bot = mock_telegram_cls.return_value
    mock_get_db.return_value.__enter__.return_value = memory_db
    mock_get_db.return_value.__exit__.return_value = None

    from src.pipelines.send_alerts import send_alerts
    send_alerts()

    mock_bot.send_message.assert_not_called()
    rows = memory_db.execute("SELECT * FROM alerts_sent").fetchall()
    assert len(rows) == 1


@patch('src.pipelines.send_alerts.get_db_connection')
@patch('src.pipelines.send_alerts.TelegramClient')
def test_send_alerts_no_edge_no_alert(mock_telegram_cls, mock_get_db, memory_db):
    """A projection below the edge threshold produces no alert."""
    # Overwrite projection with sub-threshold edge: prob_over=0.51, devigged=0.50 → 1% edge
    memory_db.execute(
        "UPDATE projections SET prob_over = 0.51, prob_under = 0.49 WHERE player_name = 'Gerrit Cole'"
    )
    memory_db.commit()

    mock_bot = mock_telegram_cls.return_value
    mock_get_db.return_value.__enter__.return_value = memory_db
    mock_get_db.return_value.__exit__.return_value = None

    from src.pipelines.send_alerts import send_alerts
    send_alerts()

    mock_bot.send_message.assert_not_called()
    alert = memory_db.execute("SELECT * FROM alerts_sent").fetchone()
    assert alert is None
