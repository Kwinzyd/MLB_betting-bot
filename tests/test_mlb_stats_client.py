import pytest
from unittest.mock import patch, MagicMock, AsyncMock
from src.clients.mlb_stats import MLBStatsClient


@pytest.fixture
def bdl_client():
    """Fixture to provide a fresh MLBStatsClient instance."""
    return MLBStatsClient()


@pytest.fixture(autouse=True)
def _clear_cache():
    from src.data.cache import cache
    cache.clear()
    yield
    cache.clear()


@patch('src.clients.mlb_stats.MLBStatsClient._get', new_callable=AsyncMock)
async def test_get_teams_uses_correct_params(mock_get, bdl_client):
    """get_teams calls _get with the teams endpoint, cache key, and 24h TTL."""
    mock_get.return_value = [
        {"id": 1, "abbreviation": "ATL", "full_name": "Atlanta Braves"},
        {"id": 2, "abbreviation": "PHI", "full_name": "Philadelphia Phillies"},
    ]

    teams = await bdl_client.get_teams()

    mock_get.assert_called_once_with("teams", cache_key="bdl_mlb_teams", cache_ttl=86400)
    assert len(teams) == 2
    assert teams[0]['abbreviation'] == 'ATL'


@patch('src.clients.mlb_stats.httpx.AsyncClient.get', new_callable=AsyncMock)
async def test_get_players_handles_pagination_and_http(mock_http_get, bdl_client):
    """get_players issues the HTTP call and extracts data from the response body."""
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.raise_for_status = MagicMock()
    mock_response.json.return_value = {
        "data": [{"id": 100, "first_name": "Aaron", "last_name": "Judge"}],
        "meta": {},
    }
    mock_http_get.return_value = mock_response

    players = await bdl_client.get_players(team_id=20)

    assert len(players) == 1
    assert players[0]['first_name'] == 'Aaron'
    mock_http_get.assert_called_once()
    _, kwargs = mock_http_get.call_args
    assert kwargs['params']['team_ids[]'] == 20
