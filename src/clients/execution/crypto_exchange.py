import asyncio
import uuid
from typing import List, Dict

from src.clients.execution.base import ExecutionVenue, empty_record, utc_now_iso
from src.utils.logging_utils import get_logger

logger = get_logger(__name__)

class CryptoExchangeVenue(ExecutionVenue):
    """
    Abstract Exchange Venue for P2P Limit Orderbook books (Polymarket, SX Bet).
    Features native Volume-Weighted Average Price (VWAP) sweeping and Maker-routing.
    """
    name: str = "crypto_p2p"
    supports_cancel: bool = True

    async def _fetch_orderbook(self, market_id: str) -> List[Dict]:
        """
        Mock network I/O: Fetches resting orders against us.
        Universal structure: [{'price': decimal_odds, 'size': fiat_in_dollars}]
        """
        await asyncio.sleep(0.1) # Simulating REST delay
        return [
            {'price': 2.50, 'size': 100.0},
            {'price': 2.35, 'size': 250.0},
            {'price': 2.10, 'size': 500.0}
        ]

    def _calculate_vwap(self, orderbook: List[Dict], target_stake: float) -> float:
        """
        Sweeps the orderbook to calculate the exact VWAP.
        Sorts the orderbook to pick the best odds first (highest decimal odds).
        """
        sorted_book = sorted(orderbook, key=lambda x: x['price'], reverse=True)
        
        filled_dollars = 0.0
        weighted_odds_sum = 0.0
        
        for chunk in sorted_book:
            available_size = float(chunk['size'])
            if available_size <= 0: 
                continue
                
            take_size = min(available_size, target_stake - filled_dollars)
            weighted_odds_sum += take_size * float(chunk['price'])
            filled_dollars += take_size
            
            if filled_dollars >= target_stake:
                break
                
        if filled_dollars == 0:
            return 0.0
            
        return weighted_odds_sum / filled_dollars

    async def place_order(self, edge: dict, context: dict) -> dict:
        """
        Executes a dynamic liquidity-routed bet.
        If the orderbook VWAP covers our required edge, fire FOK/IOC. 
        If it falls short, post a Maker limit order at our sharp odds minus margin.
        """
        target_stake = edge.get('recommended_stake', 100.0)
        if target_stake <= 0:
            return empty_record(self.name, status="skipped", notes="Stake is zero")

        model_prob = edge.get('prob', 0.5)
        # SGP_MIN_EDGE typically provided via context or environment variables
        min_edge = float(context.get('SGP_MIN_EDGE', 0.05))

        market_id = edge.get('market_id', 'mock_market')
        orderbook = await self._fetch_orderbook(market_id)
        
        vwap_odds = self._calculate_vwap(orderbook, target_stake)
        
        # Calculate resulting probability and EV
        vwap_prob = (1.0 / vwap_odds) if vwap_odds > 0 else 1.0
        executable_edge = model_prob - vwap_prob

        # Routing Logic
        if vwap_odds > 0 and executable_edge >= min_edge:
            # We can cross the spread safely!
            logger.info(f"VWAP cleared minimum edge ({executable_edge:.3f} >= {min_edge:.3f}). Sending IOC.")
            return {
                "venue": self.name,
                "status": "filled", # Assuming IOC sweeps
                "venue_order_id": str(uuid.uuid4()),
                "offered_odds": round(vwap_odds, 3),
                "fill_odds": round(vwap_odds, 3),
                "stake": target_stake,
                "placed_at": utc_now_iso(),
                "filled_at": utc_now_iso(),
                "notes": f"IOC Swept Spread. EV: {executable_edge:.3f}",
            }
        else:
            # Insufficient liquidity or poor VWAP. Revert to Maker.
            # Post limit resting order at Sharp Fair Probability minus Required Margin.
            maker_target_prob = max(0.01, min(0.99, model_prob - min_edge))
            maker_odds = 1.0 / maker_target_prob
            
            logger.info(f"Insufficient VWAP liquidity. Posting Maker Limit Order at {maker_odds:.2f}.")
            return {
                "venue": self.name,
                "status": "sent",  # Resting / Pending
                "venue_order_id": str(uuid.uuid4()),
                "offered_odds": round(maker_odds, 3),
                "fill_odds": None,
                "stake": target_stake,
                "placed_at": utc_now_iso(),
                "filled_at": None,
                "notes": f"Maker Limit Posted. Target Prob: {maker_target_prob:.3f}",
            }

    async def cancel_order(self, venue_order_id: str) -> bool:
        # Abstract cancel implementation
        return True

    async def get_open_orders(self) -> list[dict]:
        # Abstract open orders
        return []
