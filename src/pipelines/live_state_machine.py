import asyncio
import json
from typing import List, Dict

from src.clients.bdl_live import BDLLiveClient
from src.clients.odds_api import OddsAPIClient
from src.clients.telegram_bot import TelegramClient
from src.models.pa_estimator import estimate_remaining_pa_distribution
from src.utils.logging_utils import get_logger

logger = get_logger(__name__)

class LiveStateMachine:
    def __init__(self):
        self.bdl = BDLLiveClient()
        self.odds = OddsAPIClient()
        self.telegram = TelegramClient()

    async def _evaluate_live_edge(self, game_state: Dict, odds_data: Dict):
        """Re-project models for the rest-of-game utilizing updated PA PMFs."""
        # This is a conceptual pipeline mapping
        inning = game_state.get('inning', 1)
        outs = game_state.get('outs', 0)
        current_batter_slot = game_state.get('current_batter_slot', 1)

        # Iterate over odds_data (OddsAPI returns active player props)
        for prop in odds_data.get('bookmakers', []):
            for market in prop.get('markets', []):
                for outcome in market.get('outcomes', []):
                    player_name = outcome.get('description')
                    # Assume we have logic to map player_name to target_batter_slot
                    # For demonstration, mocking target_slot
                    target_slot = 1 
                    
                    # Generate remainder PA PMF
                    rem_pa_pmf = estimate_remaining_pa_distribution(
                        current_inning=inning,
                        current_outs=outs,
                        current_batter_slot=current_batter_slot,
                        target_batter_slot=target_slot
                    )
                    
                    # You would inject this rem_pa_pmf into your statistical model.
                    # E.g. proj_stats = run_glm(rem_pa_pmf)
                    
                    # If heavily discounted line (due to mid game 0 hits) yields +EV:
                    # if ev > MIN_EV_THRESHOLD:
                    #    await self.telegram.send_message(f"Live SNIPE! {player_name} Over...")
        pass

    async def watch_live_games(self, active_game_ids: List[str]):
        """Runs the continuous polling loop orchestrator."""
        logger.info(f"Starting Live State Machine watch for {len(active_game_ids)} games.")
        
        async for game_state in self.bdl.poll_live_games(active_game_ids, interval_seconds=20):
            game_id = game_state['game_id']
            
            # Fetch real-time odds from OddsAPI for the game
            # Note: Heavy pulling here can bust your OddsAPI rate limits if not cached
            try:
                odds_data = await self.odds.get_event_odds(game_id, markets=['batter_hits'])
                if odds_data:
                    await self._evaluate_live_edge(game_state, odds_data)
            except Exception as e:
                logger.error(f"Live State Machine Odds Error for {game_id}: {e}")

async def run_live_state_machine():
    # Typically, you'd pull `active_game_ids` from your DB where status == 'In Progress'
    active_games = ["mock_game_id_1"] 
    machine = LiveStateMachine()
    await machine.watch_live_games(active_games)
