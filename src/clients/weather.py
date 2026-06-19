import math
import time
import random
from datetime import datetime, timezone

import requests
from src.config import WEATHER_API_KEY
from src.utils.logging_utils import get_logger
from src.utils.retry import retry_api
from src.data.cache import cache
from src.clients.telegram_bot import TelegramClient

logger = get_logger(__name__)


def signed_wind_in_mph(wind_mph: float, wind_deg: float,
                       outfield_bearing: float) -> float:
    """Component of the wind blowing IN toward home plate, in mph (signed).

    `wind_deg` is the compass direction the wind comes FROM. `outfield_bearing`
    is the direction from home plate to center field. Wind coming FROM the
    outfield (wind_deg ≈ outfield_bearing) blows IN and suppresses carry, so it
    is POSITIVE here — matching `compute_hr_pi0`, where a positive `wind_in_mph`
    raises the structural-zero HR probability. Wind blowing OUT is negative.

    Returns 0.0 when inputs are missing/non-finite (the neutral case).
    """
    try:
        wind_mph = float(wind_mph)
        wind_deg = float(wind_deg)
        outfield_bearing = float(outfield_bearing)
    except (TypeError, ValueError):
        return 0.0
    if not (math.isfinite(wind_mph) and math.isfinite(wind_deg)
            and math.isfinite(outfield_bearing)):
        return 0.0
    # 0° between wind source and the outfield bearing = blowing straight in.
    angle = math.radians((wind_deg - outfield_bearing) % 360.0)
    return wind_mph * math.cos(angle)


class WeatherClient:
    """
    OpenWeatherMap API client (free tier: 1000 calls/day).

    Prefers the 5-day / 3-hour FORECAST so the conditions used are those at
    first pitch, not whatever they happen to be at scan time (games are scanned
    up to a few hours out — afternoon wind/temp is not night-game wind/temp).
    Falls back to current conditions when no game time is supplied or the
    forecast call fails.
    """

    def __init__(self):
        self.base_url = "https://api.openweathermap.org/data/2.5"
        self.api_key = WEATHER_API_KEY

    @retry_api(max_retries=2, delay=1.0)
    def get_game_weather(self, lat: float, lon: float,
                         outfield_bearing: float = None,
                         game_time: str = None) -> dict:
        """
        Fetch weather for stadium coordinates at (or nearest to) first pitch.

        Returns dict with temp_f, wind_mph, wind_deg, humidity, description, and
        wind_in_mph (signed component blowing in toward home plate; requires
        outfield_bearing). Returns None when no API key is configured or on error.
        """
        if not self.api_key:
            logger.debug("No WEATHER_API_KEY configured, skipping weather lookup.")
            return None

        target_dt = self._parse_game_time(game_time)
        cache_bucket = target_dt.strftime("%Y%m%d%H") if target_dt else "now"
        cache_key = f"weather_{lat:.2f}_{lon:.2f}_{cache_bucket}"
        cached = cache.get(cache_key)
        if cached is not None:
            return cached

        weather = None
        if target_dt is not None:
            weather = self._fetch_forecast(lat, lon, target_dt)
        if weather is None:
            weather = self._fetch_current(lat, lon)
        if weather is None:
            return None

        # Signed in/out wind component so the HR zero-inflation term and the
        # park-factor weather model see direction, not just speed.
        if outfield_bearing is not None:
            weather["wind_in_mph"] = round(
                signed_wind_in_mph(
                    weather.get("wind_mph", 0.0),
                    weather.get("wind_deg", 0.0),
                    outfield_bearing,
                ),
                2,
            )

        cache.set(cache_key, weather, ttl_seconds=7200)
        logger.info(
            "Weather at (%.2f, %.2f) for %s: %.0f°F, wind %.0fmph @ %s° "
            "(in=%s), %s",
            lat, lon, cache_bucket, weather["temp_f"], weather["wind_mph"],
            weather["wind_deg"], weather.get("wind_in_mph"), weather["description"],
        )
        return weather

    # ------------------------------------------------------------------
    # HTTP helpers
    # ------------------------------------------------------------------

    def _fetch_forecast(self, lat: float, lon: float, target_dt: datetime) -> dict:
        """Pull the 5-day/3-hour forecast and pick the slot nearest target_dt."""
        params = {
            "lat": lat, "lon": lon,
            "appid": self.api_key, "units": "imperial",
        }
        data = self._get("/forecast", params)
        if not data or not data.get("list"):
            return None
        target_ts = target_dt.timestamp()
        best = min(
            data["list"],
            key=lambda slot: abs(float(slot.get("dt", 0)) - target_ts),
            default=None,
        )
        if best is None:
            return None
        return self._normalize(best)

    def _fetch_current(self, lat: float, lon: float) -> dict:
        """Current-conditions fallback (legacy behavior)."""
        params = {
            "lat": lat, "lon": lon,
            "appid": self.api_key, "units": "imperial",
        }
        data = self._get("/weather", params)
        return self._normalize(data) if data else None

    def _get(self, path: str, params: dict) -> dict:
        try:
            delay = random.uniform(0.1, 0.3)
            time.sleep(delay)
            response = requests.get(f"{self.base_url}{path}", params=params, timeout=8)
            if response.status_code == 429:
                msg = ("🚨 <b>Rate Limit Hit</b>\nWeather API rate limit (HTTP 429) "
                       "reached. Request blocked.")
                logger.warning(msg)
                TelegramClient().send_message_sync(msg)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            logger.warning(f"Weather API error ({path}): {e}")
            return None

    @staticmethod
    def _normalize(data: dict) -> dict:
        """Map an OpenWeather payload (current or forecast slot) to our schema."""
        if not data:
            return None
        main = data.get("main", {})
        wind = data.get("wind", {})
        weather_arr = data.get("weather") or []
        return {
            "temp_f": main.get("temp"),
            "wind_mph": wind.get("speed", 0.0),
            "wind_deg": wind.get("deg", 0),
            "humidity": main.get("humidity"),
            "description": weather_arr[0]["description"] if weather_arr else "",
        }

    @staticmethod
    def _parse_game_time(game_time: str):
        """Parse an ISO game time to a UTC-aware datetime, or None."""
        if not game_time:
            return None
        try:
            dt = datetime.fromisoformat(str(game_time).replace("Z", "+00:00"))
        except (ValueError, AttributeError):
            return None
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt
