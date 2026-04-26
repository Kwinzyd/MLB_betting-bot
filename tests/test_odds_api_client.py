import pytest
import httpx
from unittest.mock import patch, MagicMock, AsyncMock
from src.clients.odds_api import OddsAPIClient


@pytest.fixture
def odds_client():
    """Provide a fresh OddsAPIClient instance for testing."""
    return OddsAPIClient()


@pytest.fixture(autouse=True)
def _clear_cache():
    from src.data.cache import cache
    cache.clear()
    yield
    cache.clear()


@patch('src.clients.odds_api.asyncio.sleep', new_callable=AsyncMock)
@patch('src.clients.odds_api.httpx.AsyncClient.get', new_callable=AsyncMock)
async def test_get_mlb_events_raises_on_429_quota_exceeded(mock_http_get, _mock_sleep, odds_client):
    """A 429 response from the Odds API surfaces as an httpx.HTTPStatusError after retries."""
    mock_response = MagicMock()
    mock_response.status_code = 429
    mock_response.headers = {}
    mock_response.raise_for_status.side_effect = httpx.HTTPStatusError(
        "429 Client Error: Too Many Requests",
        request=MagicMock(),
        response=mock_response,
    )
    mock_http_get.return_value = mock_response

    with pytest.raises(httpx.HTTPStatusError) as exc_info:
        await odds_client.get_mlb_events()

    assert "429" in str(exc_info.value)
