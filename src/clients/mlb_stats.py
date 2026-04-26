import asyncio
import httpx
import random
from src.config import BDL_API_KEY, MLB_SEASON
from src.utils.logging_utils import get_logger
from src.data.cache import cache
from src.utils.circuit_breaker import AsyncCircuitBreaker, CircuitBreakerOpenException
from src.clients.telegram_bot import TelegramClient

logger = get_logger(__name__)

async_bdl_circuit_breaker = AsyncCircuitBreaker(
    failure_threshold=5, 
    recovery_timeout=120.0, 
    exceptions=(httpx.RequestError,)
)


class MLBStatsClient:
    """BallDontLie MLB API client (Goat Tier)."""

    def __init__(self):
        self.base_url = "https://api.balldontlie.io/mlb/v1"
        self.api_key = BDL_API_KEY
        # httpx's built-in retries are basic. We rely on our Circuit Breaker.
        self.client = httpx.AsyncClient(
            headers={"Authorization": self.api_key},
            timeout=(3.0, 15.0) # (connect, read)
        )

    @async_bdl_circuit_breaker
    async def _get(self, endpoint, params=None, cache_key=None, cache_ttl=3600):
        """Generic GET with optional caching and pagination handling."""
        if cache_key:
            cached = cache.get(cache_key)
            if cached is not None:
                return cached

        delay = random.uniform(0.1, 0.4)
        logger.debug(f"Applying jitter delay of {delay:.3f}s before BDL API GET /{endpoint}")
        await asyncio.sleep(delay)

        url = f"{self.base_url}/{endpoint}"
        all_data = []

        try:
            while url:
                response = await self.client.get(url, params=params)
                if response.status_code == 429:
                    msg = f"🚨 <b>Rate Limit Hit</b>\nBDL API rate limit (HTTP 429) blocked request to /{endpoint}."
                    logger.warning(msg)
                    asyncio.create_task(TelegramClient().send_message(msg))
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

        except httpx.RequestError as e:
            logger.error(f"BDL API error on {endpoint}: {e}")
            # Fail loudly so downstream pipelines don't cache or process partial data
            raise

        if cache_key and all_data:
            cache.set(cache_key, all_data, ttl_seconds=cache_ttl)

        return all_data

    async def get_teams(self):
        """Fetch all MLB teams."""
        return await self._get("teams", cache_key="bdl_mlb_teams", cache_ttl=86400)

    async def get_games(self, season=None, dates=None, team_ids=None):
        """Fetch games for a season and/or specific dates, optionally filtered by team."""
        season = season or MLB_SEASON
        params = {"seasons[]": season}
        if dates:
            params["dates[]"] = dates
        if team_ids:
            params["team_ids[]"] = team_ids
        cache_key = f"bdl_mlb_games_{season}_{dates}_{team_ids}"
        return await self._get("games", params=params, cache_key=cache_key, cache_ttl=3600)

    async def get_game_stats(self, game_id):
        """Fetch box score stats for a specific game."""
        cache_key = f"bdl_mlb_stats_{game_id}"
        return await self._get("stats", params={"game_ids[]": game_id}, cache_key=cache_key, cache_ttl=21600)

    async def get_stats_batch(self, game_ids, chunk_size=50, max_concurrency=5, min_delay=0.1, max_delay=0.7):
        """Fetch stats for multiple game IDs in batched requests.

        Checks per-game cache first; only fetches uncached IDs. Results are stored
        back into the per-game cache so future single-game lookups are also fast.
        """
        game_ids = list(game_ids)
        all_stats = []
        uncached_ids = []

        for gid in game_ids:
            cached = cache.get(f"bdl_mlb_stats_{gid}")
            if cached is not None:
                all_stats.extend(cached)
            else:
                uncached_ids.append(gid)

        if not uncached_ids:
            logger.debug(f"All {len(game_ids)} game stat sets served from cache.")
            return all_stats

        logger.info(f"Fetching stats for {len(uncached_ids)} uncached games in batches of {chunk_size}.")

        semaphore = asyncio.Semaphore(max_concurrency)

        @async_bdl_circuit_breaker
        async def fetch_and_paginate_chunk(chunk: list[int]) -> list[dict]:
            """Fetches and paginates a single chunk of game stats."""
            async with semaphore:
                # Add a randomized delay (jitter) to prevent thundering herd and rate limits
                delay = random.uniform(min_delay, max_delay)
                logger.debug(f"Applying jitter delay of {delay:.3f}s before BDL API batch stats chunk")
                await asyncio.sleep(delay)
                
                params = {"game_ids[]": chunk}
                chunk_stats = []
                url = f"{self.base_url}/stats"
                try:
                    while url:
                        response = await self.client.get(url, params=params)
                        if response.status_code == 429:
                            msg = "🚨 <b>Rate Limit Hit</b>\nBDL API rate limit (HTTP 429) blocked batch stats chunk request."
                            logger.warning(msg)
                            asyncio.create_task(TelegramClient().send_message(msg))
                        response.raise_for_status()
                        body = response.json()
                        chunk_stats.extend(body.get("data", []))
                        next_cursor = body.get("meta", {}).get("next_cursor")
                        if next_cursor:
                            # For subsequent pages, only the cursor is needed
                            params = {"cursor": next_cursor}
                        else:
                            url = None
                    return chunk_stats
                except httpx.RequestError as e:
                    logger.error(f"BDL API batch stats error (chunk {chunk}): {e}")
                    raise

        tasks = []
        for i in range(0, len(uncached_ids), chunk_size):
            chunk = uncached_ids[i:i + chunk_size]
            tasks.append(fetch_and_paginate_chunk(chunk))

        # Run all chunk fetches concurrently. return_exceptions=True makes the
        # operation more resilient, allowing successful chunks to be processed
        # even if others fail or the circuit breaker opens mid-batch.
        list_of_chunk_results = await asyncio.gather(*tasks, return_exceptions=True)

        newly_fetched_stats = []
        for result in list_of_chunk_results:
            if isinstance(result, Exception):
                if not isinstance(result, CircuitBreakerOpenException):
                    logger.error(f"A stats chunk failed to fetch: {result}")
                # Skip failed chunks and continue with the data we have.
            else:
                newly_fetched_stats.extend(result)

        # Cache per game ID so individual lookups are warm on the next run
        stats_by_game: dict[int, list] = {}
        for stat in newly_fetched_stats:
            gid = stat.get("game", {}).get("id")
            if gid:
                stats_by_game.setdefault(gid, []).append(stat)
        for gid, stats in stats_by_game.items():
            cache.set(f"bdl_mlb_stats_{gid}", stats, ttl_seconds=21600)

        all_stats.extend(newly_fetched_stats)
        return all_stats

    async def get_players(self, team_id=None, search=None):
        """Search or list players."""
        params = {}
        if team_id:
            params["team_ids[]"] = team_id
        if search:
            params["search"] = search
        cache_key = f"bdl_mlb_players_{team_id}_{search}"
        return await self._get("players", params=params, cache_key=cache_key, cache_ttl=21600)

    async def get_lineups(self, game_id):
        """
        Fetch pre-game batting lineups and probable pitchers for a game.
        Returns list of lineup entries with batting_order, position, is_probable_pitcher.
        Available from 2026 season; typically posted 1-2 hours before first pitch.
        """
        cache_key = f"bdl_mlb_lineups_{game_id}"
        return await self._get("lineups", params={"game_id": game_id}, cache_key=cache_key, cache_ttl=3600)

    async def get_player_injuries(self, team_ids=None, player_ids=None):
        """
        Fetch current player injuries from BDL /player_injuries endpoint.
        Returns list of injury records with status, type, detail, return_date, etc.
        """
        params = {}
        if team_ids:
            params["team_ids[]"] = team_ids
        if player_ids:
            params["player_ids[]"] = player_ids
        cache_key = f"bdl_mlb_injuries_{team_ids}_{player_ids}"
        return await self._get("player_injuries", params=params, cache_key=cache_key, cache_ttl=3600)

    async def get_season_averages(self, player_ids, season=None):
        """Fetch season averages for one or more players."""
        season = season or MLB_SEASON
        params = {"season": season}
        if isinstance(player_ids, list):
            params["player_ids[]"] = player_ids
        else:
            params["player_ids[]"] = [player_ids]
        cache_key = f"bdl_mlb_avg_{player_ids}_{season}"
        return await self._get("season_averages", params=params, cache_key=cache_key, cache_ttl=21600)
