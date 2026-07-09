"""Betting odds from BallDontLie (free, unlimited on Goat tier).

Two feeds:

1. GAME odds (/odds) — moneyline, run line, total — condensed into one
   consensus quote per game, used to (a) back up the game total when the Odds
   API is unavailable and (b) supply moneylines for moneyline-tilted implied
   team totals in PA scaling.

2. PLAYER PROP odds (/odds/player_props) — live over/under quotes from six US
   soft books (draftkings, fanduel, betmgm, betrivers, caesars, fanatics).
   get_player_prop_odds() translates them into the Odds API event-odds shape
   so scan_props can merge them into its line shop. Vendor keys are prefixed
   'bdl_' to keep the two feeds attributable and collision-free; none of them
   are sharp books, so the sharp anchor still comes from the Odds API.
"""
from __future__ import annotations

from statistics import median
from typing import Optional

from src.clients.mlb_stats import MLBStatsClient
from src.models.devig import devig_multiplicative
from src.utils.logging_utils import get_logger

logger = get_logger(__name__)

# BDL prop_type -> our market key (only markets the model prices).
PROP_TYPE_TO_MARKET = {
    "hits": "batter_hits",
    "home_runs": "batter_home_runs",
    "total_bases": "batter_total_bases",
    "pitcher_strikeouts": "pitcher_strikeouts",
    "pitcher_earned_runs": "pitcher_earned_runs",
}

# Prefix for BDL-sourced bookmaker keys ('bdl_draftkings', ...). Keeps them
# out of SHARP_BOOKMAKERS and separates their CLV/bias attribution from the
# same books' Odds API quotes.
BDL_BOOK_PREFIX = "bdl_"

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


def _load_player_names(player_ids) -> dict:
    """Map BDL player_id -> canonical name from the local players table.

    players.player_id IS the BDL id (sync_stats upserts player.get('id')),
    so no crosswalk is needed. Unknown ids simply drop out of the shop —
    a player we've never synced has no stats to project anyway.
    """
    if not player_ids:
        return {}
    from src.data.db import get_db_connection
    ids = list(player_ids)
    placeholders = ",".join("?" for _ in ids)
    with get_db_connection() as conn:
        rows = conn.execute(
            f"SELECT player_id, name FROM players WHERE player_id IN ({placeholders})",
            tuple(ids),
        ).fetchall()
    return {r["player_id"]: r["name"] for r in rows if r["name"]}


async def get_player_prop_odds(
    bdl_game_id,
    client: MLBStatsClient = None,
    bust_cache: bool = False,
    name_map: dict = None,
) -> Optional[dict]:
    """Live BDL player props for a game, in Odds API event-odds shape.

    Returns {'bookmakers': [{'key': 'bdl_<vendor>', 'markets': [{'key':
    <market>, 'outcomes': [{'name': 'Over'|'Under', 'point': line, 'price':
    decimal, 'description': player_name}]}]}]} — directly mergeable into the
    Odds API response consumed by scan_props._parse_odds_by_player.

    Only over_under markets on the five modeled prop types are kept
    (milestone markets have no line to price). Fail-safe: returns None on any
    error or when nothing translates, so the scan proceeds on Odds API alone.

    name_map: optional {bdl_player_id: name} override for tests; by default
    names resolve from the local players table.
    """
    if not bdl_game_id:
        return None
    client = client or MLBStatsClient()
    try:
        records = await client.get_player_props(bdl_game_id, bust_cache=bust_cache)
    except Exception as e:  # noqa: BLE001 — supplemental source must never raise
        logger.debug("BDL player props fetch failed for game %s: %s", bdl_game_id, e)
        return None
    if not records:
        return None

    if name_map is None:
        try:
            name_map = _load_player_names({r.get("player_id") for r in records})
        except Exception as e:  # noqa: BLE001
            logger.debug("BDL player-name lookup failed: %s", e)
            return None

    # vendor -> market_key -> list of outcomes
    by_vendor: dict = {}
    for r in records:
        market_key = PROP_TYPE_TO_MARKET.get(r.get("prop_type"))
        if not market_key:
            continue
        market = r.get("market") or {}
        if market.get("type") != "over_under":
            continue  # milestone props have no over/under line
        player_name = name_map.get(r.get("player_id"))
        if not player_name:
            continue
        try:
            line = float(r.get("line_value"))
        except (TypeError, ValueError):
            continue
        vendor = r.get("vendor")
        if not vendor:
            continue

        outcomes = []
        for side, odds_key in (("Over", "over_odds"), ("Under", "under_odds")):
            dec = american_to_decimal(market.get(odds_key))
            if dec is not None and dec > 1.0:
                outcomes.append({
                    "name": side,
                    "point": line,
                    "price": dec,
                    "description": player_name,
                })
        if outcomes:
            by_vendor.setdefault(vendor, {}).setdefault(market_key, []).extend(outcomes)

    if not by_vendor:
        return None

    bookmakers = [
        {
            "key": f"{BDL_BOOK_PREFIX}{vendor}",
            "markets": [
                {"key": mk, "outcomes": outs} for mk, outs in markets_map.items()
            ],
        }
        for vendor, markets_map in by_vendor.items()
    ]
    n_quotes = sum(len(o) for mm in by_vendor.values() for o in mm.values())
    logger.debug(
        "BDL player props for game %s: %d quotes across %d book(s).",
        bdl_game_id, n_quotes, len(bookmakers),
    )
    return {"bookmakers": bookmakers}
