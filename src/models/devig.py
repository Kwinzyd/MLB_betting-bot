import math
from src.utils.logging_utils import get_logger

logger = get_logger(__name__)

# Sanity bounds on the two-sided hold (sum of implied probabilities).
# Real sportsbook prop markets run ~1.04 (4% hold) to ~1.20 (20% hold).
# Values outside this window signal stale/suspended lines, crossed markets,
# or arbitrage bait — in every case, devig output is untrustworthy.
MIN_TOTAL_IMPLIED = 0.98   # < 1.00 means arbitrage; tolerance for float noise
MAX_TOTAL_IMPLIED = 1.30   # > 30% hold — almost certainly stale or bad data


def decimal_to_implied(odds: float) -> float:
    return 1.0 / odds


def _is_valid_pair(odds1, odds2) -> bool:
    """Screen out odds inputs that would make devig output nonsense.

    Rejects: None, non-finite, <=1.0 (impossible decimal odds), and hold
    outside [MIN_TOTAL_IMPLIED, MAX_TOTAL_IMPLIED]. Callers should treat
    an invalid pair as 'no sharp truth available' and skip the prop.
    """
    if odds1 is None or odds2 is None:
        return False
    try:
        o1 = float(odds1)
        o2 = float(odds2)
    except (TypeError, ValueError):
        return False
    if not (math.isfinite(o1) and math.isfinite(o2)):
        return False
    if o1 <= 1.0 or o2 <= 1.0:
        return False
    total_implied = 1.0 / o1 + 1.0 / o2
    if total_implied < MIN_TOTAL_IMPLIED or total_implied > MAX_TOTAL_IMPLIED:
        logger.warning(
            "Rejecting devig: implausible hold (%.4f) for odds (%s, %s). "
            "Line is likely stale, suspended, or crossed.",
            total_implied, odds1, odds2,
        )
        return False
    return True


def devig_additive(odds1: float, odds2: float) -> tuple:
    """Remove vig by evenly splitting the margin. Returns (None, None) on bad input."""
    if not _is_valid_pair(odds1, odds2):
        return None, None
    imp1 = decimal_to_implied(odds1)
    imp2 = decimal_to_implied(odds2)
    margin = imp1 + imp2 - 1.0
    true1 = imp1 - (margin / 2.0)
    true2 = imp2 - (margin / 2.0)
    return max(0.001, true1), max(0.001, true2)


def devig_multiplicative(odds1: float, odds2: float) -> tuple:
    """Remove vig by rescaling to sum 1.0 (preferred for prop markets).

    Returns (None, None) on invalid input so callers can skip untrustworthy lines
    instead of computing an edge against garbage.
    """
    if not _is_valid_pair(odds1, odds2):
        return None, None
    imp1 = decimal_to_implied(odds1)
    imp2 = decimal_to_implied(odds2)
    total_implied = imp1 + imp2
    return imp1 / total_implied, imp2 / total_implied
