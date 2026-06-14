"""Telegram alerting wrapped behind the ExecutionVenue interface.

Encapsulates today's send_alerts behavior: format a rich message and post it
to the configured Telegram chat. Returns a uniform order record so the
send_alerts pipeline can persist orders the same way for every venue.
"""
from __future__ import annotations

from typing import Optional

from src.clients.execution.base import ExecutionVenue, utc_now_iso
from src.clients.telegram_bot import TelegramClient
from src.config import BETTING_ENABLED
from src.utils.logging_utils import get_logger

logger = get_logger(__name__)


class TelegramVenue(ExecutionVenue):
    name = "telegram"
    supports_cancel = False

    def __init__(self, client: Optional[TelegramClient] = None):
        self._client = client or TelegramClient()

    async def place_order(self, edge: dict, context: dict) -> dict:
        offered_odds = context["odds"]
        stake = edge["kelly"]["recommended_stake"]
        message = format_alert_message(
            player_name=context["player_name"],
            market=context["market"],
            side=context["side"],
            line=context["line"],
            odds=offered_odds,
            bookmaker=context["bookmaker"],
            edge=edge,
            projection=context["projection"],
            context=context.get("model_context", {}),
            home_team=context["home_team"],
            away_team=context["away_team"],
            venue=context.get("game_venue"),
            rationale=context.get("llm_rationale"),
        )

        placed_at = utc_now_iso()
        if not BETTING_ENABLED:
            logger.info(
                f"[SHADOW] Would-alert: {context['player_name']} {context['market']} "
                f"{context['side'].upper()} {context['line']} "
                f"@ {context['bookmaker']} (edge {edge['edge_pct']:.1f}%). "
                f"Set BETTING_ENABLED=true to deliver."
            )
            return {
                "venue": self.name, "status": "skipped",
                "venue_order_id": None, "offered_odds": offered_odds,
                "fill_odds": None, "stake": stake,
                "placed_at": placed_at, "filled_at": None,
                "notes": "BETTING_ENABLED=false",
            }

        try:
            await self._client.send_message(message)
        except Exception as e:
            logger.error(f"Failed to send Telegram alert: {e}")
            return {
                "venue": self.name, "status": "failed",
                "venue_order_id": None, "offered_odds": offered_odds,
                "fill_odds": None, "stake": stake,
                "placed_at": placed_at, "filled_at": None,
                "notes": str(e),
            }

        logger.info(
            f"Alert sent: {context['player_name']} {context['market']} "
            f"{context['side'].upper()} {context['line']}"
        )
        return {
            "venue": self.name, "status": "sent",
            "venue_order_id": None, "offered_odds": offered_odds,
            "fill_odds": None, "stake": stake,
            "placed_at": placed_at, "filled_at": None,
            "notes": None,
        }


def format_alert_message(player_name, market, side, line, odds, bookmaker,
                         edge, projection, context, home_team, away_team, venue,
                         rationale=None):
    """Format a rich Telegram alert message. Moved from send_alerts.py."""
    if odds >= 2.0:
        american = f"+{int((odds - 1) * 100)}"
    else:
        american = f"-{int(100 / (odds - 1))}"

    market_display = market.replace('_', ' ').title()
    venue_display = venue or 'Unknown'

    msg = (
        f"<b>MLB PROP ALERT</b>\n"
        f"{'=' * 30}\n"
        f"<b>{player_name}</b> - {market_display}\n"
        f"<b>{side.upper()} {line}</b>\n\n"
        f"Book: {bookmaker.title()} @ {odds:.2f} ({american})\n"
        f"Edge: <b>{edge['edge_pct']:.1f}%</b> | EV: {edge['ev']:+.3f}\n"
        f"Model: {edge['model_prob']:.1%} | Book: {edge['book_implied']:.1%}\n"
        f"Kelly Stake: <b>${edge['kelly']['recommended_stake']:.2f}</b>\n\n"
        f"Matchup: {away_team} @ {home_team}\n"
        f"Venue: {venue_display}\n"
    )

    if context:
        if 'opp_k_rate' in context:
            msg += f"Opp K%: {context['opp_k_rate']:.1%} (avg {0.225:.1%})\n"
        if 'park_adj' in context:
            msg += f"Park Adj: {context['park_adj']:.3f}\n"
        if 'platoon_adj' in context:
            msg += f"Platoon Adj: {context['platoon_adj']:.3f}\n"

    msg += f"\nProjected: {projection['projected_mean']:.2f}"
    if rationale:
        msg += f"\n\n<i>\U0001F9E0 {rationale}</i>"
    return msg
