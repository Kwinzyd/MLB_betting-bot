import os
import asyncio
import aiohttp
from typing import Dict, Optional, List
from src.utils.logging_utils import get_logger

logger = get_logger(__name__)

class BDLLiveClient:
    """Client for polling live MLB game state from Ball Don't Lie API."""
    
    BASE_URL = "https://api.balldontlie.io/mlb/v1"
    
    def __init__(self):
        self.api_key = os.getenv("BDL_API_KEY")
        self.headers = {"Authorization": self.api_key} if self.api_key else {}
        if not self.api_key:
            logger.warning("BDL_API_KEY not found in environment. Defaulting to free tier limits or potential unauthorized errors.")

    async def get_live_game_state(self, game_id: str, session: aiohttp.ClientSession) -> Optional[Dict]:
        """Fetch the latest play-by-play/state for a specific game."""
        url = f"{self.BASE_URL}/games/{game_id}"
        try:
            async with session.get(url, headers=self.headers) as response:
                if response.status == 200:
                    data = await response.json()
                    # Parse the relevant game state
                    # Assuming BDL JSON structure for MLB includes inning, outs, and batter
                    game_data = data.get('data', {})
                    return {
                        'game_id': game_id,
                        'status': game_data.get('status'),
                        'inning': game_data.get('inning'),
                        'outs': game_data.get('outs'),
                        'men_on_base': game_data.get('men_on_base', 0),
                        'current_batter_slot': game_data.get('current_batter_lineup_slot', 1)
                    }
                elif response.status == 429:
                    logger.warning("BDL API Rate limit exceeded.")
                    return None
                else:
                    text = await response.text()
                    logger.error(f"BDL API Error {response.status}: {text}")
                    return None
        except Exception as e:
            logger.error(f"Error fetching live state for game {game_id}: {e}")
            return None

    async def poll_live_games(self, game_ids: List[str], interval_seconds: int = 15):
        """High-frequency polling loop for given game IDs."""
        async with aiohttp.ClientSession() as session:
            while True:
                tasks = [self.get_live_game_state(gid, session) for gid in game_ids]
                results = await asyncio.gather(*tasks)
                
                # Process active results
                for state in results:
                    if state and state.get('status') == 'In Progress':
                        # Yield the state or dispatch to an event queue
                        # For now, we will return the batch to the caller
                        yield state
                        
                await asyncio.sleep(interval_seconds)
