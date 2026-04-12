import pytest
import sqlite3
import json
from unittest.mock import patch


@pytest.fixture
def memory_db():
    """
    Creates an in-memory SQLite database with the minimum schema 
    required to test the scan_props pipeline.
    """
    conn = sqlite3.connect(':memory:')
    conn.row_factory = sqlite3.Row
    
    # Create minimal schema required for scan_props reads and inserts
    conn.executescript('''
        CREATE TABLE games (
            game_id TEXT PRIMARY KEY, home_team TEXT, away_team TEXT, venue TEXT, status TEXT
        );
        CREATE TABLE prop_snapshots (
            snapshot_id TEXT PRIMARY KEY, game_id TEXT, player_name TEXT, market TEXT, 
            line REAL, over_odds REAL, under_odds REAL, bookmaker TEXT, timestamp TEXT, 
            devigged_over REAL, devigged_under REAL
        );
        CREATE TABLE projections (
            id INTEGER PRIMARY KEY AUTOINCREMENT, game_id TEXT, player_name TEXT, 
            market TEXT, projected_mean REAL, prob_over REAL, prob_under REAL, 
            context_json TEXT, timestamp TEXT, UNIQUE(game_id, player_name, market)
        );
    ''')
    
    # Seed the DB with a dummy active game so the pipeline has something to scan
    conn.execute('''
        INSERT INTO games (game_id, home_team, away_team, venue, status) 
        VALUES ('game_123', 'Yankees', 'Red Sox', 'Yankee Stadium', 'SCHEDULED')
    ''')
    conn.commit()
    
    return conn


@patch('src.pipelines.scan_props.get_db_connection')
@patch('src.pipelines.scan_props.OddsAPIClient')
@patch('src.pipelines.scan_props.WeatherClient')
@patch('src.pipelines.scan_props._build_projection')
def test_scan_props_database_inserts(
    mock_build_projection, mock_weather_client, mock_odds_client, 
    mock_get_db, memory_db
):
    """Test that scan_props correctly saves snapshots and playable projections to the DB."""
    
    # 1. Setup database mock to return our in-memory DB context manager
    # Since `with get_db_connection() as conn:` is used, we mock the __enter__ method
    mock_get_db.return_value.__enter__.return_value = memory_db
    mock_get_db.return_value.__exit__.return_value = None

    # 2. Mock Odds API response to return fake odds for Gerrit Cole
    mock_odds_instance = mock_odds_client.return_value
    mock_odds_instance.get_event_odds.return_value = {
        'bookmakers': [{
            'key': 'draftkings',
            'markets': [{
                'key': 'pitcher_strikeouts',
                'outcomes': [
                    {'name': 'Over', 'description': 'Gerrit Cole', 'point': 6.5, 'price': 2.0},
                    {'name': 'Under', 'description': 'Gerrit Cole', 'point': 6.5, 'price': 1.8}
                ]
            }]
        }]
    }

    # 3. Mock the projection model to return a profitable edge on the OVER
    # Odds of 2.0 imply a 50% chance. We project 60%, creating a playable edge.
    mock_build_projection.return_value = {
        'projected_mean': 7.5,
        'prob_over': 0.60,
        'prob_under': 0.40,
        'injury_status': 'Healthy',
        'sample_size': 20,
        'context': {'mock_data': True}
    }

    # 4. Run the pipeline
    from src.pipelines.scan_props import scan_props
    scan_props()

    # 5. Assert the data was saved successfully to the in-memory database
    snapshots = memory_db.execute("SELECT * FROM prop_snapshots").fetchall()
    assert len(snapshots) > 0
    
    snapshot = dict(snapshots[0])
    assert snapshot['game_id'] == 'game_123'
    assert snapshot['player_name'] == 'Gerrit Cole'
    assert snapshot['market'] == 'pitcher_strikeouts'
    assert snapshot['bookmaker'] == 'draftkings'
    assert snapshot['line'] == 6.5
    assert snapshot['over_odds'] == 2.0

    projections = memory_db.execute("SELECT * FROM projections").fetchall()
    assert len(projections) == 1
    
    proj = dict(projections[0])
    assert proj['game_id'] == 'game_123'
    assert proj['player_name'] == 'Gerrit Cole'
    assert proj['market'] == 'pitcher_strikeouts'
    assert proj['prob_over'] == 0.60
    assert json.loads(proj['context_json']) == {'mock_data': True}