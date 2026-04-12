"""
Client for the MLB Stats API (statsapi.mlb.com).

Free, no API key required. Used exclusively for umpire assignments and
per-game K/BB totals — data that is not available through BallDontLie.

Two endpoints used:
  GET /schedule?sportId=1&date=YYYY-MM-DD&hydrate=officials
      → game list with home-plate umpire per game

  GET /game/{gamePk}/boxscore
      → pitching totals (strikeouts, walks) for a completed game
"""

from __future__ import annotations

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from src.utils.logging_utils import get_logger

logger = get_logger(__name__)

_BASE = "https://statsapi.mlb.com/api/v1"


def _make_session() -> requests.Session:
    session = requests.Session()
    retry = Retry(
        total=3,
        backoff_factor=1,
        status_forcelist=(429, 500, 502, 503, 504),
    )
    adapter = HTTPAdapter(max_retries=retry)
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    return session


class MLBOfficialStatsClient:
    """Thin wrapper around two MLB Stats API endpoints needed for umpire data."""

    def __init__(self):
        self.session = _make_session()

    def get_games_with_officials(self, date_str: str) -> list[dict]:
        """
        Return all games scheduled on date_str with their official assignments.

        Each returned dict has the shape:
            {
                "gamePk":   int,
                "status":   str,   # e.g. "Final", "In Progress", "Preview"
                "homeTeam": str,
                "awayTeam": str,
                "officials": [{"officialType": str, "official": {"id": int, "fullName": str}}]
            }
        Returns [] on any network or parse error.
        """
        try:
            resp = self.session.get(
                f"{_BASE}/schedule",
                params={"sportId": 1, "date": date_str, "hydrate": "officials"},
                timeout=15,
            )
            resp.raise_for_status()
            body = resp.json()
        except requests.RequestException as e:
            logger.error(f"MLB Stats API schedule error ({date_str}): {e}")
            return []

        games = []
        for date_block in body.get("dates", []):
            for game in date_block.get("games", []):
                home = game.get("teams", {}).get("home", {}).get("team", {})
                away = game.get("teams", {}).get("away", {}).get("team", {})
                games.append({
                    "gamePk":    game.get("gamePk"),
                    "status":    game.get("status", {}).get("abstractGameState", ""),
                    "homeTeam":  home.get("name", ""),
                    "awayTeam":  away.get("name", ""),
                    "officials": game.get("officials", []),
                })
        return games

    def get_game_ks_and_bbs(self, game_pk: int) -> dict[str, int]:
        """
        Return combined pitching strikeouts and walks for a completed game.

            {"strikeouts": int, "walks": int}

        Returns {} on error or if the game is not yet final.
        """
        try:
            resp = self.session.get(
                f"{_BASE}/game/{game_pk}/boxscore",
                timeout=15,
            )
            resp.raise_for_status()
            box = resp.json()
        except requests.RequestException as e:
            logger.error(f"MLB Stats API boxscore error (gamePk={game_pk}): {e}")
            return {}

        try:
            teams = box.get("teams", {})
            home_p = teams.get("home", {}).get("teamStats", {}).get("pitching", {})
            away_p = teams.get("away", {}).get("teamStats", {}).get("pitching", {})
            return {
                "strikeouts": int(home_p.get("strikeOuts", 0)) + int(away_p.get("strikeOuts", 0)),
                "walks":      int(home_p.get("baseOnBalls", 0)) + int(away_p.get("baseOnBalls", 0)),
            }
        except (KeyError, TypeError, ValueError) as e:
            logger.debug(f"Boxscore parse error (gamePk={game_pk}): {e}")
            return {}
