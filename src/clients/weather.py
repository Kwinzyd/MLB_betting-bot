import requests
from src.config import WEATHER_API_KEY
from src.utils.logging_utils import get_logger
from src.utils.retry import retry_api
from src.data.cache import cache

logger = get_logger(__name__)


class WeatherClient:
    """
    OpenWeatherMap API client (free tier: 1000 calls/day).
    Fetches current weather conditions at stadium coordinates.
    """

    def __init__(self):
        self.base_url = "https://api.openweathermap.org/data/2.5"
        self.api_key = WEATHER_API_KEY

    @retry_api(max_retries=2, delay=1.0)
    def get_game_weather(self, lat: float, lon: float) -> dict:
        """
        Fetch current weather for stadium coordinates.
        Returns dict with temp_f, wind_mph, wind_deg, humidity, description.
        """
        if not self.api_key:
            logger.debug("No WEATHER_API_KEY configured, skipping weather lookup.")
            return None

        cache_key = f"weather_{lat:.2f}_{lon:.2f}"
        cached = cache.get(cache_key)
        if cached is not None:
            return cached

        params = {
            "lat": lat,
            "lon": lon,
            "appid": self.api_key,
            "units": "imperial",
        }

        try:
            response = requests.get(
                f"{self.base_url}/weather", params=params, timeout=8
            )
            response.raise_for_status()
            data = response.json()

            weather = {
                "temp_f": data["main"]["temp"],
                "wind_mph": data["wind"]["speed"],
                "wind_deg": data["wind"].get("deg", 0),
                "humidity": data["main"]["humidity"],
                "description": data["weather"][0]["description"] if data.get("weather") else "",
            }

            # Cache for 2 hours - weather doesn't change drastically
            cache.set(cache_key, weather, ttl_seconds=7200)
            logger.info(
                f"Weather at ({lat:.2f}, {lon:.2f}): "
                f"{weather['temp_f']:.0f}°F, wind {weather['wind_mph']:.0f}mph "
                f"@ {weather['wind_deg']}°, {weather['description']}"
            )
            return weather

        except requests.exceptions.RequestException as e:
            logger.warning(f"Weather API error: {e}")
            return None
