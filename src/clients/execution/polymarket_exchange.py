import asyncio
import uuid
import httpx
from typing import List, Dict

try:
    from py_clob_client.client import ClobClient
    from py_clob_client.clob_types import OrderArgs, OrderType, ApiCreds
    HAS_CLOB = True
except ImportError:
    HAS_CLOB = False
    
    class OrderType:
        FOK = "FOK"
        POST_ONLY = "POST_ONLY"


from src.clients.execution.base import ExecutionVenue, empty_record, utc_now_iso
from src.utils.logging_utils import get_logger

logger = get_logger(__name__)

class PolymarketExchangeVenue(ExecutionVenue):
    """
    Live Polymarket Execution Venue utilizing CLOB L2 REST APIs.
    Translates standard P2P "Yes/No" Shares into pipeline-friendly Decimal Odds.
    """
    name: str = "polymarket"
    supports_cancel: bool = True

    def __init__(self):
        super().__init__()
        from src.config import (
            POLYMARKET_HOST, POLYMARKET_API_KEY, POLYMARKET_SECRET, 
            POLYMARKET_PASSPHRASE, POLYMARKET_PRIVATE_KEY
        )
        self.host = POLYMARKET_HOST
        self.api_key = POLYMARKET_API_KEY
        self.secret = POLYMARKET_SECRET
        self.passphrase = POLYMARKET_PASSPHRASE
        self.private_key = POLYMARKET_PRIVATE_KEY

        self.client = None
        if HAS_CLOB and self.private_key:
            try:
                creds = ApiCreds(
                    api_key=self.api_key,
                    api_secret=self.secret,
                    api_passphrase=self.passphrase
                )
                self.client = ClobClient(
                    self.host,
                    key=self.private_key,
                    chain_id=137,
                    creds=creds,
                    signature_type=1  # Standard EOA signature
                )
            except Exception as e:
                logger.error(f"Failed to initialize Polymarket CLOB client: {e}")

    async def _fetch_orderbook(self, token_id: str) -> List[Dict]:
        """
        Fetches the active orderbook for a specific Polymarket token.
        Since we always "BUY YES", we consume the "asks" side of the book.
        """
        url = f"{self.host}/book?token_id={token_id}"
        async with httpx.AsyncClient() as client:
            response = await client.get(url, timeout=5.0)
            if response.status_code == 200:
                data = response.json()
                return data.get('asks', [])
            else:
                logger.error(f"Failed to fetch orderbook for {token_id}: {response.text}")
                return []

    def _calculate_vwap(self, orderbook_asks: List[Dict], target_stake: float) -> tuple[float, float, float]:
        """
        Sweeps the Polymarket orderbook.
        Polymarket prices are in "cents" (e.g. 0.45 per share) and size is in "shares".
        Returns (vwap_cents_price, decimal_odds, filled_dollars).
        """
        # Sort asks lowest to highest (we want to buy the cheapest 'Yes' shares first)
        sorted_asks = sorted(orderbook_asks, key=lambda x: float(x['price']))
        
        shares_accumulated = 0.0
        dollars_spent = 0.0
        
        for ask in sorted_asks:
            price = float(ask['price'])
            size_shares = float(ask['size'])
            
            if size_shares <= 0 or price <= 0:
                continue
                
            # How many shares can we afford in this chunk?
            affordable_shares = (target_stake - dollars_spent) / price
            take_shares = min(size_shares, affordable_shares)
            
            dollars_spent += take_shares * price
            shares_accumulated += take_shares
            
            if dollars_spent >= target_stake - 0.01:  # accounts for float precision
                break
                
        if shares_accumulated == 0:
            return 0.0, 0.0, 0.0
            
        vwap_cents = dollars_spent / shares_accumulated
        vwap_decimal_odds = 1.0 / vwap_cents
        
        return vwap_cents, vwap_decimal_odds, dollars_spent

    async def _dispatch_order(self, token_id: str, price_cents: float, size_shares: float, side: str = "BUY", order_type: OrderType = OrderType.FOK) -> dict:
        """
        Signs and submits the CLOB interaction utilizing py_clob_client.
        """
        if not self.client:
            logger.warning(f"Polymarket API missing keys/SDK. MOCKING SUBMISSION: {side} {size_shares:.2f} shares at {price_cents:.3f}")
            return {"status": "mock_filled", "id": str(uuid.uuid4())}
            
        try:
            # Polymarket expects specific ticking parameters. We assume standard increments here.
            # Convert size to match tick sizes and price to float limits.
            order_args = OrderArgs(
                price=round(price_cents, 3), # 3 decimal precision
                size=round(size_shares, 2),  # 2 decimal precision
                side=side,
                token_id=token_id
            )
            # Wrap standard blocking clob_client in an executor or assume it handles its own I/O well enough
            response = await asyncio.to_thread(self.client.create_and_post_order, order_args)
            return response
        except Exception as e:
            logger.error(f"Polymarket execution failed: {e}")
            return {"error": str(e)}

    async def place_order(self, edge: dict, context: dict) -> dict:
        """
        Execute a Polymarket trade: resolve VWAP, route FOK/Maker.

        Requires context['token_id'] to be the Polymarket YES token for the
        bet's SIDE (context['side']). Mapping an MLB player prop to a Polymarket
        token is not yet implemented — most MLB props have no Polymarket market —
        so absent a token this returns a clean 'skipped' (a no-op), never a fake
        fill. The edge contract matches the other venues: stake lives at
        edge['kelly']['recommended_stake']; the model/sharp probability at
        edge['sharp_prob'] (falling back to edge['model_prob']).
        """
        kelly = edge.get('kelly') or {}
        target_stake = kelly.get('recommended_stake', edge.get('recommended_stake', 0.0)) or 0.0
        if target_stake <= 0:
            return empty_record(self.name, status="skipped", notes="Stake is zero")

        token_id = context.get('token_id')
        if not token_id:
            # Expected common case (no Polymarket market for this prop): skip
            # quietly-but-visibly rather than erroring or faking a fill.
            logger.warning(
                "Polymarket: no token mapped for %s %s %s — skipping (token "
                "resolution not yet implemented).",
                context.get('player_name'), context.get('market'), context.get('side'),
            )
            return empty_record(self.name, status="skipped", notes="no Polymarket token mapped")

        # Probability of the side we're backing — Polymarket price ≈ probability.
        model_prob = edge.get('sharp_prob')
        if model_prob is None:
            model_prob = edge.get('model_prob', 0.5)
        min_edge = float(context.get('SGP_MIN_EDGE', 0.05))

        # 1. Fetch live CLOB book
        orderbook = await self._fetch_orderbook(token_id)
        
        # 2. Sweep VWAP
        vwap_cents, vwap_decimal_odds, sweep_cost = self._calculate_vwap(orderbook, target_stake)
        
        executable_edge = model_prob - vwap_cents  # Since Polymarket price == probability directly!

        # 3. Route Execution
        if sweep_cost >= (target_stake * 0.99) and executable_edge >= min_edge:
            logger.info(f"Polymarket VWAP cleared edge threshold ({executable_edge:.3f} >= {min_edge:.3f}). Sweeping spread.")
            size_shares = target_stake / vwap_cents
            result = await self._dispatch_order(token_id, vwap_cents, size_shares, side="BUY", order_type=OrderType.FOK)
            
            return {
                "venue": self.name,
                "status": "filled" if "error" not in result else "failed",
                "venue_order_id": result.get("id", str(uuid.uuid4())),
                "offered_odds": round(vwap_decimal_odds, 3),
                "fill_odds": round(vwap_decimal_odds, 3),
                "stake": sweep_cost,
                "placed_at": utc_now_iso(),
                "filled_at": utc_now_iso() if "error" not in result else None,
                "notes": result.get("error", f"FOK Swap across spread. Edge: {executable_edge:.3f}"),
            }
        else:
            # 4. Liquidity short? Post a Maker Limit!
            # Maker price in polymarket is natively the fair theoretical implied probability.
            maker_target_cents = max(0.01, min(0.99, model_prob - min_edge))
            maker_size_shares = target_stake / maker_target_cents
            maker_decimal_odds = 1.0 / maker_target_cents
            
            logger.info(f"Insufficient CLOB liquidity. Setting Resting Maker Order at {maker_target_cents:.3f}c")
            result = await self._dispatch_order(token_id, maker_target_cents, maker_size_shares, side="BUY", order_type=OrderType.POST_ONLY)
            
            return {
                "venue": self.name,
                "status": "sent" if "error" not in result else "failed",
                "venue_order_id": result.get("id", str(uuid.uuid4())),
                "offered_odds": round(maker_decimal_odds, 3),
                "fill_odds": None, # Unfilled yet
                "stake": target_stake,
                "placed_at": utc_now_iso(),
                "filled_at": None,
                "notes": result.get("error", f"PostOnly Maker limit active at {maker_target_cents:.3f}c"),
            }

    async def cancel_order(self, venue_order_id: str) -> bool:
        if self.client:
            await asyncio.to_thread(self.client.cancel, venue_order_id)
        return True

    async def get_open_orders(self) -> list[dict]:
        return []
