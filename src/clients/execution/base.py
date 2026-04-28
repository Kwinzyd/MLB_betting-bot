"""Common interface for order execution venues.

Today's "execution" is sending a Telegram alert. Soon it will include posting
limit orders on peer-to-peer exchanges (Prophet, Sporttrade, SX Bet) and a
PaperExchange that simulates fills. Each venue speaks the same place_order /
cancel_order / get_open_orders contract so send_alerts.py doesn't need to know
which one it is talking to.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional


class ExecutionVenue(ABC):
    name: str = "unknown"
    supports_cancel: bool = False

    @abstractmethod
    async def place_order(self, edge: dict, context: dict) -> dict:
        """Submit an order based on a playable edge.

        Returns a record dict with keys:
            venue, status, venue_order_id, offered_odds, fill_odds, stake,
            placed_at, filled_at, notes
        Status is one of: pending | sent | filled | cancelled | rejected | failed.
        """

    async def cancel_order(self, venue_order_id: str) -> bool:
        return False

    async def get_open_orders(self) -> list[dict]:
        return []


def utc_now_iso() -> str:
    from datetime import datetime, timezone
    return datetime.now(timezone.utc).isoformat()


def empty_record(venue: str, status: str = "skipped",
                 notes: Optional[str] = None) -> dict:
    return {
        "venue": venue,
        "status": status,
        "venue_order_id": None,
        "offered_odds": None,
        "fill_odds": None,
        "stake": None,
        "placed_at": None,
        "filled_at": None,
        "notes": notes,
    }
