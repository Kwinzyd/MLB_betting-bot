import pytest
from unittest.mock import patch, MagicMock
from src.clients.mlb_stats import MLBStatsClient


@pytest.fixture
def bdl_client():
    """Fixture to provide a fresh MLBStatsClient instance."""
    return MLBStatsClient()


# --- Approach 1: Mocking the internal _get method ---

@patch('src.clients.mlb_stats.MLBStatsClient._get')
def test_get_teams_uses_correct_params(mock_get, bdl_client):
    """Test that get_teams calls the internal _get with the correct endpoint and cache settings."""
    # Arrange: Tell the mock what to return when called
    mock_get.return_value = [
        {"id": 1, "abbreviation": "ATL", "full_name": "Atlanta Braves"},
        {"id": 2, "abbreviation": "PHI", "full_name": "Philadelphia Phillies"}
    ]

    # Act: Call the method
    teams = bdl_client.get_teams()

    # Assert: Verify the mock was called correctly and returned our fake data
    mock_get.assert_called_once_with("teams", cache_key="bdl_mlb_teams", cache_ttl=86400)
    assert len(teams) == 2
    assert teams[0]['abbreviation'] == 'ATL'


# --- Approach 2: Mocking the requests library directly ---

@patch('src.clients.mlb_stats.requests.Session.get')
@patch('src.clients.mlb_stats.cache.get', return_value=None)  # Bypass the cache for this test
def test_get_players_handles_pagination_and_http(mock_cache_get, mock_session_get, bdl_client):
    """Test that the client handles HTTP requests and extracts data correctly."""
    
    # Arrange: We need to mock the response object returned by requests.get()
    mock_response = MagicMock()
    
    # Mock the JSON body that the API would return
    mock_response.json.return_value = {
        "data": [{"id": 100, "first_name": "Aaron", "last_name": "Judge"}],
        "meta": {}  # Empty meta means no next_cursor, so it won't paginate infinitely
    }
    # Mock raise_for_status so it doesn't throw an error
    mock_response.raise_for_status.return_value = None
    
    mock_session_get.return_value = mock_response

    # Act
    players = bdl_client.get_players(team_id=20)

    # Assert
    assert len(players) == 1
    assert players[0]['first_name'] == 'Aaron'
    # Verify requests.get was called with the correct URL and params
    mock_session_get.assert_called_once()
    args, kwargs = mock_session_get.call_args
    assert kwargs['params']['team_ids[]'] == 20