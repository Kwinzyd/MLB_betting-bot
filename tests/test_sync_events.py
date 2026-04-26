import pytest
import sqlite3
import datetime
import asyncio
from unittest.mock import patch, MagicMock, AsyncMock
from freezegun import freeze_time
from src.pipelines.sync_events import sync_events


@pytest.fixture(autouse=True)
def _clear_cache():
    """Isolate cache state between tests — sync_events checks cache before calling the API."""
    from src.data.cache import cache
    cache.clear()
    yield
    cache.clear()


@pytest.fixture
def memory_db():
    """Provides an in-memory database with the games table + related schema."""
    conn = sqlite3.connect(':memory:')
    conn.row_factory = sqlite3.Row
    conn.executescript('''
        CREATE TABLE games (
            game_id TEXT PRIMARY KEY, bdl_game_id INTEGER, date TEXT, game_time TEXT,
            home_team TEXT, away_team TEXT, home_team_id INTEGER,
            away_team_id INTEGER, venue TEXT, status TEXT,
            home_score INTEGER, away_score INTEGER
        );
        CREATE TABLE teams (
            team_id INTEGER PRIMARY KEY, abbreviation TEXT, name TEXT,
            league TEXT, division TEXT
        );
        CREATE TABLE prop_snapshots (
            snapshot_id TEXT PRIMARY KEY, timestamp TEXT
        );
        CREATE TABLE projections (
            id INTEGER PRIMARY KEY AUTOINCREMENT, timestamp TEXT
        );
        CREATE TABLE daily_lineups (id INTEGER PRIMARY KEY AUTOINCREMENT, date TEXT);
        CREATE TABLE probable_pitchers (id INTEGER PRIMARY KEY AUTOINCREMENT, date TEXT);
        CREATE TABLE injury_reports (id INTEGER PRIMARY KEY AUTOINCREMENT, date TEXT);
    ''')
    # Seed teams so sync_events skips the BDL team-sync branch
    conn.executemany(
        "INSERT INTO teams (team_id, abbreviation, name, league, division) VALUES (?, ?, ?, ?, ?)",
        [
            (1, 'NYY', 'New York Yankees', 'AL', 'ALE'),
            (2, 'BOS', 'Boston Red Sox', 'AL', 'ALE'),
            (3, 'CHC', 'Chicago Cubs', 'NL', 'NLC'),
            (4, 'PHI', 'Philadelphia Phillies', 'NL', 'NLE'),
        ]
    )
    conn.commit()
    return conn


@patch('src.pipelines.sync_events.MLBStatsClient')
@patch('src.pipelines.sync_events.get_db_connection')
@patch('src.clients.odds_api.httpx.AsyncClient.get', new_callable=AsyncMock)
def test_sync_events_inserts_games(mock_session_get, mock_get_db, mock_bdl_cls, memory_db):
    """Test that sync_events parses Odds API data and saves it to the DB."""

    # 1. Setup the in-memory database mock
    mock_get_db.return_value.__enter__.return_value = memory_db
    mock_get_db.return_value.__exit__.return_value = None

    # 2. Mock the external API response (Odds API)
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.raise_for_status = MagicMock()
    mock_response.json.return_value = [{
        "id": "mock_game_123",
        "sport_title": "MLB",
        "commence_time": "2024-05-15T23:05:00Z",
        "home_team": "New York Yankees",
        "away_team": "Boston Red Sox"
    }]
    mock_session_get.return_value = mock_response

    # 3. Mock the BDL client to skip BDL network calls
    mock_bdl_cls.return_value.get_teams = AsyncMock(return_value=[])
    mock_bdl_cls.return_value.get_games = AsyncMock(return_value=[])

    # 4. Execute the pipeline
    asyncio.run(sync_events())

    # 4. Verify the database state
    games = memory_db.execute("SELECT * FROM games").fetchall()
    assert len(games) == 1
    assert games[0]['game_id'] == 'mock_game_123'
    assert games[0]['home_team'] == 'New York Yankees'
    assert games[0]['away_team'] == 'Boston Red Sox'
    assert games[0]['status'] == 'SCHEDULED'


@freeze_time("2024-07-04 12:00:00")
@patch('src.clients.odds_api.asyncio.sleep', new_callable=AsyncMock)
@patch('src.pipelines.sync_events.MLBStatsClient')
@patch('src.pipelines.sync_events.get_db_connection')
@patch('src.clients.odds_api.httpx.AsyncClient.get', new_callable=AsyncMock)
def test_sync_events_with_dynamic_frozen_date(mock_session_get, mock_get_db, mock_bdl_cls, _mock_sleep, memory_db):
    """Test that mock API payloads can dynamically align with the freezegun date."""
    mock_get_db.return_value.__enter__.return_value = memory_db
    mock_get_db.return_value.__exit__.return_value = None

    frozen_now = datetime.datetime.utcnow()
    mock_commence_time = frozen_now.strftime("%Y-%m-%dT%H:%M:%SZ")

    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.raise_for_status = MagicMock()
    mock_response.json.return_value = [{
        "id": "july_4th_game",
        "sport_title": "MLB",
        "commence_time": mock_commence_time,
        "home_team": "Chicago Cubs",
        "away_team": "Philadelphia Phillies"
    }]
    mock_session_get.return_value = mock_response

    mock_bdl_cls.return_value.get_teams = AsyncMock(return_value=[])
    mock_bdl_cls.return_value.get_games = AsyncMock(return_value=[])

    asyncio.run(sync_events())

    games = memory_db.execute("SELECT * FROM games").fetchall()
    assert len(games) == 1
    assert games[0]['game_id'] == 'july_4th_game'