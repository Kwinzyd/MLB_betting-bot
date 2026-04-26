import pytest
import httpx
import requests
from unittest.mock import patch, MagicMock, AsyncMock

from src.clients.odds_api import OddsAPIClient
from src.clients.weather import WeatherClient


@pytest.mark.asyncio
@patch("src.clients.odds_api.TelegramClient.send_message", new_callable=AsyncMock)
@patch("httpx.AsyncClient.get", new_callable=AsyncMock)
async def test_odds_api_rate_limit_alert(mock_get, mock_send_message):
    """Test that OddsAPIClient triggers an async Telegram alert on HTTP 429."""
    # Setup mock 429 response
    mock_response = MagicMock()
    mock_response.status_code = 429
    mock_response.headers = {}
    mock_response.raise_for_status.side_effect = httpx.HTTPStatusError(
        "429 Too Many Requests", request=MagicMock(), response=mock_response
    )
    mock_get.return_value = mock_response

    client = OddsAPIClient()
    
    # OddsAPIClient will exhaust its retries, so we catch the final exception
    with pytest.raises(httpx.HTTPStatusError):
        await client.get_mlb_events()

    # Verify the Telegram alert was dispatched
    assert mock_send_message.called
    call_args = mock_send_message.call_args[0][0]
    assert "Rate Limit Hit" in call_args
    assert "Odds API" in call_args


@patch("src.clients.weather.TelegramClient.send_message_sync")
@patch("requests.get")
def test_weather_api_rate_limit_alert(mock_get, mock_send_message_sync):
    """Test that WeatherClient triggers a synchronous Telegram alert on HTTP 429."""
    # Setup mock 429 response
    mock_response = MagicMock()
    mock_response.status_code = 429
    mock_response.raise_for_status.side_effect = requests.exceptions.HTTPError("429 Too Many Requests")
    mock_get.return_value = mock_response

    client = WeatherClient()
    client.api_key = "dummy_test_key"  # Ensure it doesn't skip the API call
    
    # WeatherClient catches the exception internally and returns None
    result = client.get_game_weather(lat=41.88, lon=-87.65)
    
    assert result is None
    assert mock_send_message_sync.called
    call_args = mock_send_message_sync.call_args[0][0]
    assert "Rate Limit Hit" in call_args
    assert "Weather API" in call_args