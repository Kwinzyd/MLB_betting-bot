import pytest
import sqlite3
import datetime
from unittest.mock import patch, MagicMock
from freezegun import freeze_time
from src.pipelines.sync_events import sync_events


@pytest.fixture
def memory_db():
    """Provides an in-memory database with the games table schema."""
    conn = sqlite3.connect(':memory:')
    conn.row_factory = sqlite3.Row
    conn.execute('''
        CREATE TABLE games (
            game_id TEXT PRIMARY KEY, bdl_game_id INTEGER, date TEXT,
            home_team TEXT, away_team TEXT, home_team_id INTEGER,
            away_team_id INTEGER, venue TEXT, status TEXT,
            home_score INTEGER, away_score INTEGER
        )
    ''')
    conn.commit()
    return conn


@patch('src.pipelines.sync_events.get_db_connection')
@patch('src.clients.odds_api.requests.Session.get')
def test_sync_events_inserts_games(mock_session_get, mock_get_db, memory_db):
    """Test that sync_events parses Odds API data and saves it to the DB."""
    
    # 1. Setup the in-memory database mock
    mock_get_db.return_value.__enter__.return_value = memory_db
    mock_get_db.return_value.__exit__.return_value = None

    # 2. Mock the external API response (requests.Session.get)
    mock_response = MagicMock()
    mock_response.status_code = 200
    # Provide a fake JSON payload mimicking the Odds API /events endpoint
    mock_response.json.return_value = [{
        "id": "mock_game_123",
        "sport_title": "MLB",
        "commence_time": "2024-05-15T23:05:00Z",
        "home_team": "New York Yankees",
        "away_team": "Boston Red Sox"
    }]
    mock_session_get.return_value = mock_response

    # 3. Execute the pipeline
    sync_events()

    # 4. Verify the database state
    games = memory_db.execute("SELECT * FROM games").fetchall()
    assert len(games) == 1
    assert games[0]['game_id'] == 'mock_game_123'
    assert games[0]['home_team'] == 'New York Yankees'
    assert games[0]['away_team'] == 'Boston Red Sox'
    assert games[0]['status'] == 'SCHEDULED'


@freeze_time("2024-07-04 12:00:00")
@patch('src.pipelines.sync_events.get_db_connection')
@patch('src.clients.odds_api.requests.Session.get')
def test_sync_events_with_dynamic_frozen_date(mock_session_get, mock_get_db, memory_db):
    """Test that mock API payloads can dynamically align with the freezegun date."""
    mock_get_db.return_value.__enter__.return_value = memory_db
    mock_get_db.return_value.__exit__.return_value = None

    # Because time is frozen, datetime.utcnow() returns exactly "2024-07-04 12:00:00"
    # We use this to dynamically build the commence_time for our fake Odds API response!
    frozen_now = datetime.datetime.utcnow()
    mock_commence_time = frozen_now.strftime("%Y-%m-%dT%H:%M:%SZ")

    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = [{
        "id": "july_4th_game",
        "sport_title": "MLB",
        "commence_time": mock_commence_time,  # Injects: "2024-07-04T12:00:00Z"
        "home_team": "Chicago Cubs",
        "away_team": "Philadelphia Phillies"
    }]
    mock_session_get.return_value = mock_response

    sync_events()

    games = memory_db.execute("SELECT * FROM games").fetchall()
    assert len(games) == 1
    assert games[0]['game_id'] == 'july_4th_game'