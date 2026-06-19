"""Consensus game-level odds from BallDontLie (free, unlimited on Goat tier).

BDL has GAME odds only — moneyline, run line, and total — across several books;
it has NO player props. This module condenses the per-vendor rows into one
consensus quote per game, used to (a) back up the game total when the Odds API
is unavailable and (b) supply moneylines for moneyline-tilted implied team
totals in PA scaling. It never sources player-prop prices.
"""
from __future__ import annotations

from statistics import median
from typing import Optional

from src.clients.mlb_stats import MLBStatsClient
from src.models.devig import devig_multiplicative
from src.utils.logging_utils import get_logger

logger = get_logger(__name__)

# Standard -110 both ways -> decimal 1.909; devigs to 50/50. Used when a vendor
# (or the whole feed) doesn't expose the side juice for totals / run lines.
_ASSUMED_DECIMAL = 1.909


def american_to_decimal(american) -> Optional[float]:
    """American odds -> decimal (>1.0). None on bad/invalid input."""
    if american in (None, ""):
        return None
    try:
        a = float(american)
    except (TypeError, ValueError):
        return None
    if a == 0:
        return None
    return 1.0 + (a / 100.0 if a > 0 else 100.0 / (-a))


def _first(record: dict, *keys):
    """First present, non-empty value among alias keys (BDL renames fields)."""
    for k in keys:
        if k in record and record[k] not in (None, ""):
            return record[k]
    return None


def _median_or_none(values):
    nums = []
    for v in values:
        if v in (None, ""):
            continue
        try:
            nums.append(float(v))
        except (TypeError, ValueError):
            continue
    return median(nums) if nums else None


async def get_game_market(bdl_game_id, client: MLBStatsClient = None) -> Optional[dict]:
    """Return consensus {total, ml_home, ml_away} for a BDL game, or None.

    Medians across all vendors that quoted the game. Fail-safe: returns None on
    any error so callers fall back to whatever they had.
    """
    if not bdl_game_id:
        return None
    client = client or MLBStatsClient()
    try:
        records = await client.get_odds(game_ids=bdl_game_id)
    except Exception as e:  # noqa: BLE001 — backup source must never raise
        logger.debug("BDL odds fetch failed for game %s: %s", bdl_game_id, e)
        return None
    if not records:
        return None
    total = _median_or_none(r.get("total_value") for r in records)
    ml_home = _median_or_none(r.get("moneyline_home_odds") for r in records)
    ml_away = _median_or_none(r.get("moneyline_away_odds") for r in records)
    if total is None and ml_home is None and ml_away is None:
        return None
    return {"total": total, "ml_home": ml_home, "ml_away": ml_away}


def _best_decimal(records, *alias_keys):
    """Best (max) decimal price for a side across vendors, with the vendor.

    Returns (best_decimal_or_None, vendor_or_None). Only valid (>1.0) prices
    count, so a missing/garbage quote never wins the shop.
    """
    best, best_vendor = None, None
    for r in records:
        dec = american_to_decimal(_first(r, *alias_keys))
        if dec is None or dec <= 1.0:
            continue
        if best is None or dec > best:
            best, best_vendor = dec, r.get("vendor")
    return best, best_vendor


def _consensus_devig(records, home_keys, away_keys):
    """Devigged (fair) probs for a two-sided market from the median vendor quote.

    Falls back to 50/50 when either side's juice is unavailable (BDL gives only
    the line for some markets), which is the right neutral prior for a
    model-driven edge: the model's deviation from the market line is the signal.
    """
    home_dec = _median_or_none(american_to_decimal(_first(r, *home_keys)) for r in records)
    away_dec = _median_or_none(american_to_decimal(_first(r, *away_keys)) for r in records)
    if home_dec and away_dec:
        p_home, p_away = devig_multiplicative(home_dec, away_dec)
        if p_home is not None:
            return p_home, p_away
    return 0.5, 0.5


async def get_game_quotes(bdl_game_id, client: MLBStatsClient = None) -> Optional[dict]:
    """Structured per-market game quotes for the model-driven game-market scan.

    Returns, for moneyline / total / run line, the devigged consensus 'fair'
    probability per side and the best shoppable decimal price (with vendor).
    Totals/run lines whose side juice BDL doesn't expose fall back to a 50/50
    devig and an assumed -110 price. Fail-safe: None on any error.

        {
          'total_line': float|None,
          'run_line': float,            # magnitude, default 1.5
          'home_is_ml_favorite': bool,
          'markets': {
             'moneyline': {'home': {...}, 'away': {...}},
             'game_total': {'over': {...}, 'under': {...}},
             'run_line':   {'home': {...}, 'away': {...}},
          }
        }
    where each side is {'devig': p, 'best_odds': dec, 'best_vendor': v}.
    """
    if not bdl_game_id:
        return None
    client = client or MLBStatsClient()
    try:
        records = await client.get_odds(game_ids=bdl_game_id)
    except Exception as e:  # noqa: BLE001 — odds source must never raise
        logger.debug("BDL quotes fetch failed for game %s: %s", bdl_game_id, e)
        return None
    if not records:
        return None

    total_line = _median_or_none(r.get("total_value") for r in records)
    run_line = _median_or_none(
        abs(float(v)) for r in records
        if (v := _first(r, "run_line_value", "spread_value", "run_line")) not in (None, "")
    ) or 1.5

    ml_home = _median_or_none(r.get("moneyline_home_odds") for r in records)
    ml_away = _median_or_none(r.get("moneyline_away_odds") for r in records)
    home_is_fav = (ml_home is not None and ml_away is not None and ml_home < ml_away)

    def _side(devig_p, best_keys):
        best, vendor = _best_decimal(records, *best_keys)
        if best is None:
            best, vendor = _ASSUMED_DECIMAL, None
        return {"devig": devig_p, "best_odds": best, "best_vendor": vendor}

    ml_p_home, ml_p_away = _consensus_devig(
        records, ["moneyline_home_odds"], ["moneyline_away_odds"])
    tot_p_over, tot_p_under = _consensus_devig(
        records, ["total_over_odds", "over_odds"], ["total_under_odds", "under_odds"])
    rl_p_home, rl_p_away = _consensus_devig(
        records, ["run_line_home_odds", "spread_home_odds"],
        ["run_line_away_odds", "spread_away_odds"])

    markets = {
        "moneyline": {
            "home": _side(ml_p_home, ["moneyline_home_odds"]),
            "away": _side(ml_p_away, ["moneyline_away_odds"]),
        },
        "game_total": {
            "over": _side(tot_p_over, ["total_over_odds", "over_odds"]),
            "under": _side(tot_p_under, ["total_under_odds", "under_odds"]),
        },
        "run_line": {
            "home": _side(rl_p_home, ["run_line_home_odds", "spread_home_odds"]),
            "away": _side(rl_p_away, ["run_line_away_odds", "spread_away_odds"]),
        },
    }
    if total_line is None and ml_home is None and ml_away is None:
        return None
    return {
        "total_line": total_line,
        "run_line": run_line,
        "home_is_ml_favorite": home_is_fav,
        "markets": markets,
    }
