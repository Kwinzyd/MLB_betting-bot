import requests
from bs4 import BeautifulSoup
from typing import List, Dict
from src.utils.logging_utils import get_logger
from src.utils.retry import retry_api
from src.data.cache import cache

logger = get_logger(__name__)


class InjuryClient:
    def __init__(self):
        self.url = "https://www.cbssports.com/mlb/injuries/"

    @retry_api(max_retries=3)
    def get_injuries(self) -> List[Dict]:
        cache_key = "mlb_injuries_current"
        cached = cache.get(cache_key)
        if cached is not None:
            return cached

        logger.info("Fetching MLB injuries from CBS Sports")
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"
        }
        res = requests.get(self.url, headers=headers, timeout=10)
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

        cache.set(cache_key, injuries, ttl_seconds=3600)
        logger.info(f"Fetched {len(injuries)} MLB injury entries.")
        return injuries

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
