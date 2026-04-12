import pytest
import requests
from unittest.mock import patch, MagicMock
from src.clients.odds_api import OddsAPIClient


@pytest.fixture
def odds_client():
    """Provide a fresh OddsAPIClient instance for testing."""
    return OddsAPIClient()


@patch('src.clients.odds_api.requests.Session.get')
@patch('src.clients.odds_api.cache.get', return_value=None)  # Bypass the cache
def test_get_mlb_events_raises_on_429_quota_exceeded(mock_cache_get, mock_session_get, odds_client):
    """Test that a 429 status code results in an HTTPError being raised."""
    
    # Arrange: Mock the response to simulate a 429 Quota Exceeded error
    mock_response = MagicMock()
    mock_response.status_code = 429
    # Simulate requests.Response.raise_for_status() raising an HTTPError
    mock_response.raise_for_status.side_effect = requests.exceptions.HTTPError("429 Client Error: Too Many Requests")
    mock_session_get.return_value = mock_response

    # Act & Assert: The @retry_api decorator will retry a few times, but should ultimately raise
    with pytest.raises(requests.exceptions.HTTPError) as exc_info:
        odds_client.get_mlb_events()

    assert "429 Client Error" in str(exc_info.value)