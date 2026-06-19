"""Paper-trading execution venue.

Simulates immediate fills at the offered sharp-devigged odds. No external
calls and no money at risk. Persists an order record so we can validate the
end-to-end placement pipeline (sizing, dedup, settlement hooks) before
pointing it at a real exchange.
"""
from __future__ import annotations

from uuid import uuid4

from src.clients.execution.base import ExecutionVenue, utc_now_iso
from src.utils.logging_utils import get_logger

logger = get_logger(__name__)


class PaperExchange(ExecutionVenue):
    name = "paper_exchange"
    supports_cancel = False

    async def place_order(self, edge: dict, context: dict) -> dict:
        offered_odds = context["odds"]
        stake = edge["kelly"]["recommended_stake"]
        if not edge.get("is_playable"):
            return {
                "venue": self.name, "status": "rejected",
                "venue_order_id": None, "offered_odds": offered_odds,
                "fill_odds": None, "stake": stake,
                "placed_at": utc_now_iso(), "filled_at": None,
                "notes": "edge not playable",
            }
        import random
        
        # 1. Rejection Rate: 5% to 8%
        rejection_chance = random.uniform(0.05, 0.08)
        if random.random() < rejection_chance:
            logger.warning(
                f"[PAPER] Rejected: Spinny wheel delay caused {context.get('player_name')} "
                f"{context.get('market')} to be locked out."
            )
            return {
                "venue": self.name, "status": "rejected",
                "venue_order_id": None, "offered_odds": offered_odds,
                "fill_odds": None, "stake": stake,
                "placed_at": utc_now_iso(), "filled_at": None,
                "notes": "simulated line-movement lockout",
            }

        fill_odds = offered_odds
        
        # 2. Synthetic Slippage: 20% chance of 1 to 3 cents penalty
        if random.random() < 0.20:
            # Randomly subtract 0.01 to 0.03 essentially acting as a 1 to 3 cent penalty
            slippage = random.choice([0.01, 0.02, 0.03])
            # Ensure it never drops below 1.01 (min possible dec odds)
            fill_odds = max(1.01, offered_odds - slippage)

        now = utc_now_iso()
        order_id = f"paper-{uuid4().hex[:8]}"
        
        if fill_odds < offered_odds:
            logger.info(
                f"[PAPER] Filled with SLIPPAGE: {context.get('player_name')} {context.get('market')} "
                f"{context.get('side', '').upper()} {context.get('line')} requested @ {offered_odds:.2f}, "
                f"filled @ {fill_odds:.2f} stake=${stake:.2f}"
            )
        else:
            logger.info(
                f"[PAPER] Filled: {context.get('player_name')} {context.get('market')} "
                f"{context.get('side', '').upper()} {context.get('line')} @ {fill_odds:.2f} "
                f"stake=${stake:.2f}"
            )
            
        return {
            "venue": self.name, "status": "filled",
            "venue_order_id": order_id, "offered_odds": offered_odds,
            "fill_odds": round(fill_odds, 2), "stake": stake,
            "placed_at": now, "filled_at": now,
            "notes": "synthetic slippage applied" if fill_odds < offered_odds else None,
        }
