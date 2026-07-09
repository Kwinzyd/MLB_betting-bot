import asyncio
import httpx
import random
from src.config import BDL_API_KEY, MLB_SEASON
from src.utils.logging_utils import get_logger
from src.data.cache import cache
from src.utils.circuit_breaker import AsyncCircuitBreaker

logger = get_logger(__name__)

async_bdl_circuit_breaker = AsyncCircuitBreaker(
    failure_threshold=5, 
    recovery_timeout=120.0, 
    exceptions=(httpx.ConnectError, httpx.TimeoutException) # Avoid tripping on 429s
)


class AsyncRateLimiter:
    """Simple token bucket rate limiter for asyncio."""
    def __init__(self, requests_per_second: float):
        self.delay = 1.0 / requests_per_second
        self.last_call = 0.0
        self.lock = asyncio.Lock()

    async def wait(self):
        async with self.lock:
            elapsed = asyncio.get_event_loop().time() - self.last_call
            wait_time = self.delay - elapsed
            if wait_time > 0:
                await asyncio.sleep(wait_time)
            self.last_call = asyncio.get_event_loop().time()


class MLBStatsClient:
    """BallDontLie MLB API client (Goat Tier)."""

    def __init__(self, requests_per_second: float = 0.07):
        self.base_url = "https://api.balldontlie.io/mlb/v1"
        self.api_key = BDL_API_KEY
        self.limiter = AsyncRateLimiter(requests_per_second)
        # httpx's built-in retries are basic. We rely on our Circuit Breaker.
        self.client = httpx.AsyncClient(
            headers={"Authorization": self.api_key},
            timeout=(5.0, 30.0) # (connect, read) - bumped for large stats pulls
        )

    @async_bdl_circuit_breaker
    async def _get(self, endpoint, params=None, cache_key=None, cache_ttl=3600):
        """Generic GET with optional caching, pagination, and 429 backoff."""
        if cache_key:
            cached = cache.get(cache_key)
            if cached is not None:
                return cached

        delay = random.uniform(0.1, 0.4)
        logger.debug(f"Applying jitter delay of {delay:.3f}s before BDL API GET /{endpoint}")
        await asyncio.sleep(delay)

        url = f"{self.base_url}/{endpoint}"
        all_data = []
        base_params = dict(params) if params else {}
        if "per_page" not in base_params:
            base_params["per_page"] = 100

        try:
            page_params = base_params
            while url:
                # 429 retry with exponential backoff — up to 6 attempts, max 120s
                for attempt in range(6):
                    await self.limiter.wait() # Proactive rate limiting
                    response = await self.client.get(url, params=page_params)
                    
                    if response.status_code == 429:
                        # Respect Retry-After if provided, otherwise exponential backoff
                        retry_after = response.headers.get("Retry-After")
                        backoff = float(retry_after) if retry_after and retry_after.isdigit() else min(2 ** attempt * 5.0, 60.0)
                        
                        logger.warning(
                            f"🚨 <b>Rate Limit Hit</b> (/{endpoint}). "
                            f"Waiting {backoff:.1f}s (attempt {attempt+1}/6)..."
                        )
                        await asyncio.sleep(backoff)
                        continue
                    
                    response.raise_for_status()
                    break  # non-429 response
                
                body = response.json()

                data = body.get("data", [])
                all_data.extend(data)

                # BDL uses cursor-based pagination; preserve original params on each page
                meta = body.get("meta", {})
                next_cursor = meta.get("next_cursor")
                if next_cursor:
                    page_params = {**base_params, "cursor": next_cursor}
                    url = f"{self.base_url}/{endpoint}"
                else:
                    url = None

        except httpx.HTTPStatusError as e:
            if e.response.status_code == 429:
                logger.error(f"BDL API 429 exhausted retries on {endpoint}")
            raise
        except httpx.RequestError as e:
            logger.error(f"BDL API network error on {endpoint}: {e}")
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

    async def get_stats_batch(self, game_ids, chunk_size=100, max_concurrency=1):
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

        async def fetch_chunk_with_semaphore(chunk: list[int], chunk_idx: int, total_chunks: int):
            async with semaphore:
                params = {"game_ids[]": chunk}
                try:
                    result = await self._get("stats", params=params)
                    logger.info(f"Batch progress: {chunk_idx}/{total_chunks} chunks completed.")
                    return result
                except Exception as e:
                    logger.error(f"Chunk {chunk_idx} failed: {e}")
                    return []

        tasks = []
        for i in range(0, len(uncached_ids), chunk_size):
            chunk = uncached_ids[i:i + chunk_size]
            tasks.append(chunk)

        # Process chunks with controlled concurrency.
        # _get already handles rate limiting and retries per chunk.
        results = await asyncio.gather(*[
            fetch_chunk_with_semaphore(chunk, i + 1, len(tasks))
            for i, chunk in enumerate(tasks)
        ])

        newly_fetched_stats = []
        for res in results:
            newly_fetched_stats.extend(res)

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
        return await self._get("lineups", params={"game_ids[]": game_id}, cache_key=cache_key, cache_ttl=3600)

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

    async def get_odds(self, game_ids=None, dates=None):
        """Fetch BDL game-level betting odds (moneyline / run line / total).

        Used as a free, unlimited backup/supplement for the game total and as
        the moneyline source for implied-team-total tilting. One of game_ids or
        dates is required by the API. Player props live on a separate endpoint
        (see get_player_props).
        """
        params = {}
        if game_ids is not None:
            params["game_ids[]"] = game_ids
        if dates is not None:
            params["dates[]"] = dates
        if not params:
            return []
        cache_key = f"bdl_mlb_odds_{game_ids}_{dates}"
        return await self._get("odds", params=params, cache_key=cache_key, cache_ttl=300)

    async def get_player_props(self, game_id, bust_cache=False):
        """Fetch live player-prop odds for one game from /odds/player_props.

        Returns raw prop records: {game_id, player_id, vendor, prop_type,
        line_value, market: {type, over_odds, under_odds, odds}, updated_at}.
        Vendors are US soft books (draftkings, fanduel, betmgm, betrivers,
        caesars, fanatics). The endpoint returns everything in one response
        (no pagination) and only carries props while books quote them — near
        game end it may be empty. Cached 5 min to match the odds TTL.
        """
        if not game_id:
            return []
        cache_key = f"bdl_mlb_player_props_{game_id}"
        if bust_cache:
            cache.delete(cache_key)
        return await self._get(
            "odds/player_props",
            params={"game_id": game_id},
            cache_key=cache_key,
            cache_ttl=300,
        )

    @async_bdl_circuit_breaker
    async def get_player_splits(self, player_id, season=None):
        """Fetch a player's season splits, grouped by split_category.

        Unlike other endpoints, /players/splits returns `data` as an OBJECT
        keyed by split_category ('split', 'byBreakdown', 'bySituation', ...),
        not a list — so the generic paginated _get (which flattens list data)
        can't be used. This does a single GET and returns the raw data dict.

        The vs-LHP/RHP platoon rows live under 'byBreakdown' as split_name
        'vs. Left' / 'vs. Right'. Returns {} on any error or empty payload.
        """
        if not player_id:
            return {}
        season = season or MLB_SEASON
        cache_key = f"bdl_mlb_splits_{player_id}_{season}"
        cached = cache.get(cache_key)
        if cached is not None:
            return cached

        await self.limiter.wait()
        url = f"{self.base_url}/players/splits"
        params = {"player_id": player_id, "season": season}
        for attempt in range(6):
            resp = await self.client.get(url, params=params)
            if resp.status_code == 429:
                retry_after = resp.headers.get("Retry-After")
                backoff = float(retry_after) if retry_after and retry_after.isdigit() else min(2 ** attempt * 5.0, 60.0)
                logger.warning(f"🚨 Rate Limit (/players/splits). Waiting {backoff:.1f}s ({attempt+1}/6)...")
                await asyncio.sleep(backoff)
                continue
            resp.raise_for_status()
            break
        else:
            return {}

        data = resp.json().get("data", {}) or {}
        if data:
            cache.set(cache_key, data, ttl_seconds=21600)  # 6h — season splits move slowly
        return data

    async def get_player_versus(self, player_id, opponent_team_id):
        """Head-to-head batting line for a batter vs every pitcher on a team.

        Returns one row per opposing pitcher: {opponent_player: {...}, at_bats,
        hits, doubles, triples, home_runs, walks, strikeouts, avg, ...}. `data`
        is a list, so the generic _get works. Cached 6h — career BvP lines only
        change by one game's worth per day. Both params are required by the API.
        """
        if not player_id or not opponent_team_id:
            return []
        cache_key = f"bdl_mlb_versus_{player_id}_{opponent_team_id}"
        return await self._get(
            "players/versus",
            params={"player_id": player_id, "opponent_team_id": opponent_team_id},
            cache_key=cache_key, cache_ttl=21600,
        )

    async def get_pitch_type_season_stats(self, role, season=None):
        """Bulk-fetch season pitch-type stats for all pitchers or hitters.

        role: 'pitcher' or 'hitter' — selects the endpoint. Returns one row per
        (player_id, pitch_type) with usage %, whiff/contact %, xwoba, and PA/K
        counts. `data` is a list here, so the generic paginated _get works.
        Cached 24h (season-level pitch mix moves slowly). Big-ish payload
        (~thousands of rows), pulled once per nightly sync.
        """
        season = season or MLB_SEASON
        endpoint = ("pitcher_pitch_type_season_stats" if role == "pitcher"
                    else "hitter_pitch_type_season_stats")
        cache_key = f"bdl_mlb_{endpoint}_{season}"
        return await self._get(
            endpoint, params={"season": season}, cache_key=cache_key, cache_ttl=86400
        )

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
