import sqlite3

from src.config import KELLY_FRACTION, BANKROLL
from src.data.db import get_db_connection
from src.utils.logging_utils import get_logger

logger = get_logger(__name__)


def get_current_bankroll() -> float:
    """Starting bankroll plus cumulative settled P&L (singles + SGPs + parlays).

    `bet_results.profit` is the per-bet ledger written by settle_results;
    `sgp_results.profit` is the same-game parlay ledger and `parlay_results.profit`
    the cross-game parlay ledger. Summing all three gives net realized P&L since
    inception. Adding the configured starting bankroll yields the amount Kelly
    should size against today, floored at 0 so a deep drawdown can never produce
    negative stakes.

    Falls back to the static `BANKROLL` on any DB error (fresh install, table
    missing, file locked) so the pipeline never sizes against zero.
    """
    try:
        with get_db_connection() as conn:
            row = conn.execute(
                "SELECT COALESCE(SUM(profit), 0.0) AS total FROM bet_results"
            ).fetchone()
            settled_pnl = float(row['total'] or 0.0) if row else 0.0
            # Parlay ledgers may not exist on old DBs; tolerate each independently.
            for table in ("sgp_results", "parlay_results"):
                try:
                    prow = conn.execute(
                        f"SELECT COALESCE(SUM(profit), 0.0) AS total FROM {table}"
                    ).fetchone()
                    settled_pnl += float(prow['total'] or 0.0) if prow else 0.0
                except sqlite3.Error:
                    pass
        return max(0.0, BANKROLL + settled_pnl)
    except sqlite3.Error as e:
        logger.debug(
            f"Bankroll ledger lookup failed ({e}); using starting bankroll {BANKROLL}"
        )
        return BANKROLL


def fractional_kelly(model_prob: float, decimal_odds: float,
                     fraction: float = None, bankroll: float = None,
                     kelly_fraction_override: float = 1.0) -> dict:
    """
    Calculate fractional Kelly stake.

    Full Kelly: f* = (b*p - q) / b
    where b = decimal_odds - 1, p = model_prob, q = 1 - p

    Returns dict with kelly_fraction, recommended_stake, full_kelly_pct.
    Negative Kelly = no bet (edge is negative).

    When `bankroll` is not provided, reads the current ledger-adjusted bankroll
    (starting roll + settled P&L) so stake sizes compound with results rather
    than anchoring forever to the day-one value.
    """
    # `is None` — a deliberate fraction of 0.0 (e.g. a line-movement kill)
    # must produce a zero stake, not silently fall back to the default.
    fraction = KELLY_FRACTION if fraction is None else fraction
    if bankroll is None:
        bankroll = get_current_bankroll()

    b = decimal_odds - 1.0
    p = model_prob
    q = 1.0 - p

    if b <= 0:
        return {"kelly_fraction": 0.0, "recommended_stake": 0.0, "full_kelly_pct": 0.0}

    full_kelly = (b * p - q) / b

    if full_kelly <= 0:
        return {"kelly_fraction": 0.0, "recommended_stake": 0.0, "full_kelly_pct": full_kelly}

    adjusted_kelly = full_kelly * fraction * kelly_fraction_override

    # Cap at 5% of bankroll max per single bet
    max_fraction = 0.05
    adjusted_kelly = min(adjusted_kelly, max_fraction)

    stake = bankroll * adjusted_kelly

    return {
        "kelly_fraction": round(adjusted_kelly, 4),
        "recommended_stake": round(stake, 2),
        "full_kelly_pct": round(full_kelly, 4),
    }
