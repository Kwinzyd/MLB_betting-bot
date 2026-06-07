import pytest
import sqlite3
from unittest.mock import patch, MagicMock


@pytest.fixture
def memory_db():
    """Provide an in-memory DB with seeded data to test stats calculations."""
    conn = sqlite3.connect(':memory:')
    conn.row_factory = sqlite3.Row
    conn.executescript('''
        CREATE TABLE teams (team_id INTEGER PRIMARY KEY, name TEXT);
        CREATE TABLE players (player_id INTEGER PRIMARY KEY, team_id INTEGER);
        CREATE TABLE batter_game_logs (
            game_id TEXT, player_id INTEGER, strikeouts INTEGER,
            plate_appearances INTEGER, runs INTEGER, rbis INTEGER
        );
        CREATE TABLE team_stats (
            team_id INTEGER PRIMARY KEY, k_rate REAL, runs_per_game REAL, 
            last_updated TEXT, UNIQUE(team_id)
        );
    ''')
    
    # Seed a team, a player, and two games worth of logs
    conn.execute("INSERT INTO teams (team_id, name) VALUES (1, 'Yankees')")
    conn.execute("INSERT INTO players (player_id, team_id) VALUES (100, 1)")
    # Game 1: 2 K's, 4 PAs, 1 run scored, 1 RBI
    conn.execute("INSERT INTO batter_game_logs VALUES ('g1', 100, 2, 4, 1, 1)")
    # Game 2: 1 K, 4 PAs, 3 runs scored, 2 RBIs
    conn.execute("INSERT INTO batter_game_logs VALUES ('g2', 100, 1, 4, 3, 2)")
    conn.commit()
    return conn


@patch('src.pipelines.calculate_team_stats.get_db_connection')
@patch('src.pipelines.calculate_team_stats.utcnow')
def test_calculate_team_stats_mocks_datetime(mock_utcnow, mock_get_db, memory_db):
    """Test that team stats are calculated correctly and the timestamp is applied."""
    # 1. Intercept the database connection
    mock_get_db.return_value.__enter__.return_value = memory_db
    mock_get_db.return_value.__exit__.return_value = None

    # 2. Mock utcnow().isoformat()
    mock_now = MagicMock()
    mock_now.isoformat.return_value = '2025-01-01T12:00:00'
    mock_utcnow.return_value = mock_now

    # 3. Execute the pipeline
    from src.pipelines.calculate_team_stats import calculate_team_stats
    calculate_team_stats()

    # 4. Verify the database state and the mocked timestamp
    stats = memory_db.execute("SELECT * FROM team_stats WHERE team_id = 1").fetchone()
    assert stats['k_rate'] == 0.375       # 3 total Ks / 8 total PAs
    assert stats['runs_per_game'] == 2.0  # 4 total runs scored / 2 total games
    assert stats['last_updated'] == '2025-01-01T12:00:00'  # Exact mocked time!