"""
Live State Machine — in-game prop sniping daemon.

Polls BDL for live game state every ~20s. For each active player prop offered
by a soft book, it:
  1. Derives remaining-PA PMF from the current inning / out / batter slot.
  2. Re-prices the model projection using get_probabilities_mixture on the
     rest-of-game PMF.
  3. Deviggs the live soft-book odds to get a sharp-anchored implied prob.
  4. Fires a Telegram alert when model EV exceeds LIVE_MIN_EV.

Invocation:
    python main.py live

Quota note: live odds are fetched at most once per game per 60s (cached).
BDL live polling is read-only and free-tier safe.
"""
from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from typing import Dict, List, Optional

from src.clients.bdl_live import BDLLiveClient
from src.clients.odds_api import OddsAPIClient
from src.clients.telegram_bot import TelegramClient
from src.config import (
    MARKETS_MAPPING, SHARP_BOOKMAKERS, LIVE_MIN_EV, LIVE_POLL_INTERVAL_SECONDS,
    KELLY_FRACTION,
)
from src.data.db import get_db_connection
from src.models.devig import devig_multiplicative
from src.models.distributions import get_probabilities_mixture
from src.models.kelly import fractional_kelly, get_current_bankroll
from src.models.pa_estimator import estimate_remaining_pa_distribution, expected_pa
from src.models.projections import ProjectionModel
from src.utils.logging_utils import get_logger

logger = get_logger(__name__)

# Reduce Kelly for live bets: live lines move fast, model lag is higher.
_LIVE_KELLY_MULT = 0.35


def _get_active_game_ids() -> List[Dict]:
    """Fetch in-progress game rows from the DB."""
    with get_db_connection() as conn:
        rows = conn.execute(
            "SELECT game_id, home_team, away_team, venue "
            "FROM games WHERE status = 'IN_PROGRESS'"
        ).fetchall()
    return [dict(r) for r in rows]


def _get_player_logs(player_id: int, market: str) -> List[Dict]:
    """Fetch game logs for a player depending on market type."""
    table = (
        "pitcher_game_logs"
        if market in ("pitcher_strikeouts", "pitcher_earned_runs")
        else "batter_game_logs"
    )
    with get_db_connection() as conn:
        rows = conn.execute(
            f"SELECT * FROM {table} WHERE player_id = ? ORDER BY date DESC",
            (player_id,),
        ).fetchall()
    return [dict(r) for r in rows]


def _get_player(player_name: str) -> Optional[Dict]:
    """Look up player row by name (exact then LIKE fallback)."""
    with get_db_connection() as conn:
        row = conn.execute(
            "SELECT * FROM players WHERE name = ? COLLATE NOCASE", (player_name,)
        ).fetchone()
        if not row:
            row = conn.execute(
                "SELECT * FROM players WHERE name LIKE ? COLLATE NOCASE",
                (f"%{player_name}%",),
            ).fetchone()
    return dict(row) if row else None


def _get_lineup_slot(player_name: str, game_id: str) -> Optional[int]:
    with get_db_connection() as conn:
        row = conn.execute(
            "SELECT lineup_position FROM daily_lineups "
            "WHERE player_name = ? AND game_id = ? COLLATE NOCASE",
            (player_name, game_id),
        ).fetchone()
    return row["lineup_position"] if row else None


def _parse_live_props(event_odds: dict) -> Dict:
    """
    Parse Odds API event odds into:
        { (player_name, market_key, line): { book: { 'over': odds, 'under': odds } } }
    """
    result: Dict = {}
    for bookmaker in event_odds.get("bookmakers", []):
        book = bookmaker["key"]
        for mkt in bookmaker.get("markets", []):
            market_key = mkt["key"]
            if market_key not in MARKETS_MAPPING:
                continue
            for outcome in mkt.get("outcomes", []):
                if "point" not in outcome:
                    continue
                player = outcome.get("description", "Unknown")
                line = float(outcome["point"])
                price = float(outcome["price"])
                side = outcome["name"].lower()
                key = (player, market_key, line)
                result.setdefault(key, {}).setdefault(book, {})[side] = price
    return result


class LiveStateMachine:
    def __init__(self):
        self.bdl = BDLLiveClient()
        self.odds = OddsAPIClient()
        self.telegram = TelegramClient()
        self.proj_model = ProjectionModel()
        # (game_id, player_name, market, line, side) → last alert ts
        self._alerted: Dict[tuple, datetime] = {}

    async def _evaluate_live_edge(
        self,
        game_id: str,
        game_state: Dict,
        event_odds: dict,
    ) -> None:
        """Re-price all live props for a game using rest-of-game PA PMF."""
        inning: int = game_state.get("inning", 1)
        outs: int = game_state.get("outs", 0)
        current_slot: int = game_state.get("current_batter_slot", 1)

        live_props = _parse_live_props(event_odds)
        sharp_set = set(SHARP_BOOKMAKERS)

        for (player_name, market_key, line), book_data in live_props.items():
            # Need a sharp book with both sides to devig
            sharp_pair = None
            for book in SHARP_BOOKMAKERS:
                pair = book_data.get(book, {})
                o, u = pair.get("over"), pair.get("under")
                if o and u and o > 1.0 and u > 1.0:
                    sharp_pair = (o, u, book)
                    break
            if sharp_pair is None:
                continue

            sharp_over, sharp_under, sharp_book = sharp_pair
            true_over, true_under = devig_multiplicative(sharp_over, sharp_under)
            if true_over is None:
                continue

            # Player lookup
            player = _get_player(player_name)
            if not player:
                continue

            player_id = player["player_id"]
            lineup_slot = _get_lineup_slot(player_name, game_id)
            if lineup_slot is None:
                continue  # Can't derive rest-of-game PA without slot

            # Rest-of-game PA distribution
            rem_pa_pmf = estimate_remaining_pa_distribution(
                current_inning=inning,
                current_outs=outs,
                current_batter_slot=current_slot,
                target_batter_slot=lineup_slot,
            )
            if not rem_pa_pmf or expected_pa(rem_pa_pmf) < 0.5:
                continue  # Game essentially over for this player

            # Build projection (use per-PA rate from logs; skip GLM path for speed)
            logs = _get_player_logs(player_id, market_key)
            if not logs:
                continue

            # Derive per-PA rate from L15 blended with season
            stat_key = {
                "batter_hits": "hits",
                "batter_total_bases": "total_bases",
                "batter_home_runs": "home_runs",
            }.get(market_key)
            if stat_key is None:
                continue  # pitcher markets don't use PA-mixture live

            recent = logs[:15]
            total_stat = sum(l.get(stat_key, 0) or 0 for l in recent)
            total_pa = sum(
                l.get("plate_appearances", 0) or l.get("at_bats", 0) or 0
                for l in recent
            )
            if total_pa == 0:
                continue
            per_pa_rate = total_stat / total_pa

            # Re-price at this line using the rest-of-game PA PMF
            prob_over, prob_under = get_probabilities_mixture(
                per_pa_rate, line, market_key, rem_pa_pmf,
            )

            # Use soft-book best price for EV calc
            best_soft_over = best_soft_under = (None, None)
            for book, pair in book_data.items():
                if book in sharp_set:
                    continue
                o, u = pair.get("over"), pair.get("under")
                if o and o > 1.0 and (best_soft_over[0] is None or o > best_soft_over[0]):
                    best_soft_over = (o, book)
                if u and u > 1.0 and (best_soft_under[0] is None or u > best_soft_under[0]):
                    best_soft_under = (u, book)

            for side, model_prob, soft_odds_book in [
                ("over", prob_over, best_soft_over),
                ("under", prob_under, best_soft_under),
            ]:
                soft_odds, soft_book = soft_odds_book
                if not soft_odds:
                    continue
                ev = model_prob * soft_odds - 1.0
                if ev < LIVE_MIN_EV:
                    continue

                # Dedup: don't re-alert the same edge within 30 min
                dedup_key = (game_id, player_name, market_key, line, side)
                last = self._alerted.get(dedup_key)
                now = datetime.now(timezone.utc)
                if last and (now - last).total_seconds() < 1800:
                    continue
                self._alerted[dedup_key] = now

                kelly = fractional_kelly(
                    model_prob, soft_odds,
                    fraction=KELLY_FRACTION * _LIVE_KELLY_MULT,
                )
                stake = kelly["recommended_stake"]
                logger.info(
                    f"LIVE EDGE: {player_name} {market_key} {side.upper()} {line} "
                    f"@ {soft_book} | EV={ev:.3f} | model_p={model_prob:.3f} | "
                    f"rem_pa≈{expected_pa(rem_pa_pmf):.1f} | Kelly=${stake:.2f}"
                )
                try:
                    msg = (
                        f"⚡ <b>LIVE SNIPE</b>\n"
                        f"{'='*28}\n"
                        f"<b>{player_name}</b> — {market_key.replace('_',' ').title()}\n"
                        f"{side.upper()} {line} @ <b>{soft_odds:.2f}</b> ({soft_book})\n\n"
                        f"Model P: <b>{model_prob:.1%}</b>  |  EV: <b>{ev*100:.1f}%</b>\n"
                        f"Rest-of-game PA ≈ {expected_pa(rem_pa_pmf):.1f}\n"
                        f"Inning: {inning}, Outs: {outs}\n"
                        f"Stake: <b>${stake:.2f}</b>\n"
                        f"<i>Sharp anchor: {sharp_book}</i>"
                    )
                    await self.telegram.send_message(msg)
                except Exception as e:
                    logger.warning(f"Live alert failed: {e}")

    async def watch_live_games(self, poll_interval: int = None) -> None:
        """Continuous polling loop. Runs until process is killed."""
        interval = poll_interval or LIVE_POLL_INTERVAL_SECONDS
        logger.info("Live State Machine started (poll interval=%ds).", interval)
        markets = list(MARKETS_MAPPING.keys())

        while True:
            # 1. Sync BDL live state → DB (inning, outs, batter_slot, running stats)
            try:
                await self.bdl.sync_game_states()
            except Exception as e:
                logger.warning("BDL live sync failed: %s", e)

            # 2. Pull in-progress games from the now-updated DB
            games = _get_active_game_ids()
            if not games:
                logger.debug("No in-progress games; sleeping.")
                await asyncio.sleep(interval)
                continue

            for game in games:
                game_id = game["game_id"]
                try:
                    event_odds = await self.odds.get_event_odds(
                        game_id, markets, bust_cache=False
                    )
                except Exception as e:
                    logger.warning(f"Live odds fetch failed for {game_id}: {e}")
                    continue

                if not event_odds:
                    continue

                # Read live state that sync_game_states just wrote
                with get_db_connection() as conn:
                    row = conn.execute(
                        "SELECT inning, outs, current_batter_slot "
                        "FROM games WHERE game_id = ?",
                        (game_id,),
                    ).fetchone()
                game_state = dict(row) if row else {}

                await self._evaluate_live_edge(game_id, game_state, event_odds)

            await asyncio.sleep(interval)


async def run_live_state_machine() -> None:
    """Entry point called by `python main.py live`."""
    machine = LiveStateMachine()
    await machine.watch_live_games()
