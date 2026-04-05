import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from src.config import BDL_API_KEY, MLB_SEASON
from src.utils.logging_utils import get_logger
from src.data.cache import cache

logger = get_logger(__name__)


def _get_retry_session(retries=3, backoff_factor=1, status_forcelist=(429, 500, 502, 503, 504)):
    session = requests.Session()
    retry = Retry(
        total=retries,
        read=retries,
        connect=retries,
        backoff_factor=backoff_factor,
        status_forcelist=status_forcelist,
    )
    adapter = HTTPAdapter(max_retries=retry)
    session.mount("http://", adapter)
    session.mount("https://", adapter)
    return session


class MLBStatsClient:
    """BallDontLie MLB API client (Goat Tier)."""

    def __init__(self):
        self.base_url = "https://api.balldontlie.io/mlb/v1"
        self.api_key = BDL_API_KEY
        self.session = _get_retry_session()
        self.session.headers.update({"Authorization": self.api_key})

    def _get(self, endpoint, params=None, cache_key=None, cache_ttl=3600):
        """Generic GET with optional caching and pagination handling."""
        if cache_key:
            cached = cache.get(cache_key)
            if cached is not None:
                return cached

        url = f"{self.base_url}/{endpoint}"
        all_data = []

        try:
            while url:
                response = self.session.get(url, params=params, timeout=15)
                response.raise_for_status()
                body = response.json()

                data = body.get("data", [])
                all_data.extend(data)

                # BDL uses cursor-based pagination
                meta = body.get("meta", {})
                next_cursor = meta.get("next_cursor")
                if next_cursor:
                    params = params or {}
                    params["cursor"] = next_cursor
                    url = f"{self.base_url}/{endpoint}"
                else:
                    url = None

        except requests.exceptions.RequestException as e:
            logger.error(f"BDL API error on {endpoint}: {e}")
            return []

        if cache_key and all_data:
            cache.set(cache_key, all_data, ttl_seconds=cache_ttl)

        return all_data

    def get_teams(self):
        """Fetch all MLB teams."""
        return self._get("teams", cache_key="bdl_mlb_teams", cache_ttl=86400)

    def get_games(self, season=None, dates=None):
        """Fetch games for a season and/or specific dates."""
        season = season or MLB_SEASON
        params = {"seasons[]": season}
        if dates:
            params["dates[]"] = dates
        cache_key = f"bdl_mlb_games_{season}_{dates}"
        return self._get("games", params=params, cache_key=cache_key, cache_ttl=3600)

    def get_game_stats(self, game_id):
        """Fetch box score stats for a specific game."""
        cache_key = f"bdl_mlb_stats_{game_id}"
        return self._get("stats", params={"game_ids[]": game_id}, cache_key=cache_key, cache_ttl=21600)

    def get_players(self, team_id=None, search=None):
        """Search or list players."""
        params = {}
        if team_id:
            params["team_ids[]"] = team_id
        if search:
            params["search"] = search
        cache_key = f"bdl_mlb_players_{team_id}_{search}"
        return self._get("players", params=params, cache_key=cache_key, cache_ttl=21600)

    def get_season_averages(self, player_ids, season=None):
        """Fetch season averages for one or more players."""
        season = season or MLB_SEASON
        params = {"season": season}
        if isinstance(player_ids, list):
            params["player_ids[]"] = player_ids
        else:
            params["player_ids[]"] = player_ids
        cache_key = f"bdl_mlb_avg_{player_ids}_{season}"
        return self._get("season_averages", params=params, cache_key=cache_key, cache_ttl=21600)
