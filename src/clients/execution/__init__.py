"""Execution venue registry.

Reads config flags and returns the list of active venues for the current run.
Order matters: send_alerts iterates this list per playable edge, so list the
primary venue first.
"""
from __future__ import annotations

from src.clients.execution.base import ExecutionVenue
from src.clients.execution.paper_exchange import PaperExchange
from src.clients.execution.telegram_venue import TelegramVenue
from src.clients.execution.polymarket_exchange import PolymarketExchangeVenue
from src.config import PAPER_EXCHANGE_ENABLED, TELEGRAM_VENUE_ENABLED, POLYMARKET_ENABLED
from src.utils.logging_utils import get_logger

logger = get_logger(__name__)


def build_venue_registry() -> list[ExecutionVenue]:
    venues: list[ExecutionVenue] = []
    if TELEGRAM_VENUE_ENABLED:
        venues.append(TelegramVenue())
    if POLYMARKET_ENABLED:
        # Experimental: MLB-prop → Polymarket token mapping is not implemented,
        # so this venue skips every order until token resolution is built. Warn
        # loudly so an enabled flag is never mistaken for working execution.
        logger.warning(
            "Polymarket venue is ENABLED but token mapping is unimplemented — "
            "it will skip all orders. Disable POLYMARKET_ENABLED until wired."
        )
        venues.append(PolymarketExchangeVenue())
    if PAPER_EXCHANGE_ENABLED:
        venues.append(PaperExchange())
    return venues


__all__ = ["ExecutionVenue", "PaperExchange", "TelegramVenue", "PolymarketExchangeVenue", "build_venue_registry"]
