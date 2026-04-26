import asyncio
import random
import requests
from bs4 import BeautifulSoup
from typing import List, Dict
from src.clients.mlb_stats import MLBStatsClient
from src.utils.logging_utils import get_logger
from src.data.cache import cache
from src.clients.telegram_bot import TelegramClient

logger = get_logger(__name__)


class InjuryClient:
    """
    MLB injury data client.

    Primary source: BallDontLie /player_injuries endpoint (structured API, stable).
    Fallback: CBS Sports HTML scraper (fragile, wrapped in broad try/except).
    """

    def __init__(self):
        self.cbs_url = "https://www.cbssports.com/mlb/injuries/"

    async def get_injuries(self) -> List[Dict]:
        cache_key = "mlb_injuries_current"
        cached = cache.get(cache_key)
        if cached is not None:
            return cached

        # Primary: BallDontLie API
        injuries = await self._fetch_from_bdl()

        # Fallback: CBS Sports scraper if BDL returns nothing
        if not injuries:
            logger.info("BDL injuries empty, falling back to CBS Sports scraper.")
            injuries = await self._fetch_from_cbs()

        if injuries:
            cache.set(cache_key, injuries, ttl_seconds=3600)

        logger.info(f"Fetched {len(injuries)} MLB injury entries.")
        return injuries

    async def _fetch_from_bdl(self) -> List[Dict]:
        """Fetch injuries from BallDontLie API (stable, structured)."""
        try:
            bdl_client = MLBStatsClient()
            raw = await bdl_client.get_player_injuries()
            if not raw:
                return []

            injuries = []
            for entry in raw:
                player = entry.get('player', {})
                team = entry.get('team', {}) or player.get('team', {})

                player_name = f"{player.get('first_name', '')} {player.get('last_name', '')}".strip()
                team_name = team.get('full_name', '') if isinstance(team, dict) else ''
                status = entry.get('status', '')
                injury_type = entry.get('type', '')
                detail = entry.get('detail', '')

                if not player_name:
                    continue

                injuries.append({
                    "player_name": player_name,
                    "team": team_name,
                    "status": self._normalize_status(status),
                    "injury_type": f"{injury_type} - {detail}".strip(' -') if detail else injury_type,
                    "raw_status": status,
                })

            logger.info(f"Fetched {len(injuries)} injuries from BDL API.")
            return injuries

        except Exception as e:
            logger.warning(f"BDL injuries fetch failed: {e}")
            return []

    async def _fetch_from_cbs(self) -> List[Dict]:
        """
        Fallback: scrape CBS Sports injuries page.
        Wrapped in broad try/except so HTML structure changes don't crash the pipeline.
        """
        try:
            logger.info("Fetching MLB injuries from CBS Sports (fallback)")
            headers = {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"
            }
            delay = random.uniform(0.5, 1.5)
            logger.debug(f"Applying jitter delay of {delay:.3f}s before CBS Sports scraper request")
            await asyncio.sleep(delay)
            res = await asyncio.to_thread(requests.get, self.cbs_url, headers=headers, timeout=10)
            if res.status_code == 429:
                msg = "🚨 <b>Rate Limit Hit</b>\nCBS Sports scraper rate limit (HTTP 429) reached. Request blocked."
                logger.warning(msg)
                asyncio.create_task(TelegramClient().send_message(msg))
            res.raise_for_status()

            soup = BeautifulSoup(res.text, 'html.parser')
            injuries = []

            teams = soup.find_all('div', class_='TableBaseWrapper')
            for team_block in teams:
                team_name_el = team_block.find('span', class_='TeamName')
                if not team_name_el:
                    continue
                team_name = team_name_el.text.strip()

                rows = team_block.find_all('tr', class_='TableBase-bodyTr')
                for row in rows:
                    cols = row.find_all('td')
                    if len(cols) >= 4:
                        player = cols[0].text.strip()
                        status = cols[3].text.strip()
                        injury_type = cols[2].text.strip()

                        injuries.append({
                            "player_name": player,
                            "team": team_name,
                            "status": self._normalize_status(status),
                            "injury_type": injury_type,
                            "raw_status": status,
                        })

            logger.info(f"Fetched {len(injuries)} injuries from CBS Sports.")
            return injuries

        except Exception as e:
            logger.error(
                f"CBS Sports scraper failed (HTML structure may have changed): {e}. "
                "Injury data will be stale until next successful sync."
            )
            return []

    def _normalize_status(self, status: str) -> str:
        s = status.lower()
        if 'il' in s or '10-day' in s or '15-day' in s or '60-day' in s:
            return 'IL'
        elif 'out' in s:
            return 'Out'
        elif 'day-to-day' in s or 'dtd' in s:
            return 'Day-to-Day'
        elif 'questionable' in s:
            return 'Day-to-Day'
        elif 'probable' in s:
            return 'Probable'
        return 'Unknown'
