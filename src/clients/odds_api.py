import requests
from src.config import ODDS_API_KEY, ODDS_REGION, BOOKMAKERS, SHARP_BOOKMAKERS
from src.utils.logging_utils import get_logger
from src.utils.retry import retry_api
from src.data.cache import cache

logger = get_logger(__name__)


class OddsAPIClient:
    def __init__(self):
        self.base_url = "https://api.the-odds-api.com/v4"
        self.sport = "baseball_mlb"
        self.api_key = ODDS_API_KEY
        self.session = requests.Session()
        self.session.params = {"apiKey": self.api_key}

    def _log_quota(self, response):
        requests_used = response.headers.get("x-requests-used")
        requests_remaining = response.headers.get("x-requests-remaining")
        if requests_used is not None:
            logger.info(f"Odds API Quota - Used: {requests_used}, Remaining: {requests_remaining}")

    @retry_api(max_retries=3)
    def get_mlb_events(self):
        cache_key = "odds_api_mlb_events"
        cached = cache.get(cache_key)
        if cached:
            return cached

        url = f"{self.base_url}/sports/{self.sport}/events"
        logger.info("Fetching MLB events from Odds API")
        response = self.session.get(url, timeout=10)
        self._log_quota(response)
        response.raise_for_status()

        data = response.json()
        cache.set(cache_key, data, ttl_seconds=3600)
        return data

    @retry_api(max_retries=3)
    def get_event_odds(self, event_id: str, markets: list, bust_cache: bool = False):
        """Fetch odds for an event. Pass all markets at once to save quota.

        bust_cache=True skips the in-memory cache read (still writes to it),
        used by trigger-driven scans that need fresh data inside the 5-min TTL.
        """
        markets_str = ",".join(markets)
        # Merge BOOKMAKERS + SHARP_BOOKMAKERS (deduped, preserving order) so sharp
        # books come back in the same call. Per Odds API docs, the bookmakers param
        # overrides regions and does not multiply quota cost.
        seen = set()
        merged_books = []
        for b in list(BOOKMAKERS) + list(SHARP_BOOKMAKERS):
            b = b.strip()
            if b and b not in seen:
                seen.add(b)
                merged_books.append(b)
        bookmakers_str = ",".join(merged_books)

        cache_key = f"odds_api_event_{event_id}_{markets_str}"
        if bust_cache:
            cache.delete(cache_key)
        else:
            cached = cache.get(cache_key)
            if cached:
                return cached

        url = f"{self.base_url}/sports/{self.sport}/events/{event_id}/odds"
        params = {
            "regions": ODDS_REGION,
            "markets": markets_str,
            "bookmakers": bookmakers_str,
            "oddsFormat": "decimal",
        }

        logger.info(f"Fetching odds for event {event_id}, markets: {markets_str}")
        response = self.session.get(url, params=params, timeout=10)
        self._log_quota(response)
        response.raise_for_status()

        data = response.json()
        cache.set(cache_key, data, ttl_seconds=300)
        return data
