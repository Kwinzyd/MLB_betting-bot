def is_valid_odds(odds) -> bool:
    if odds is None:
        return False
    try:
        val = float(odds)
        return val > 1.0
    except (ValueError, TypeError):
        return False

def validate_prop_market(market: str) -> bool:
    from src.config import MARKETS_MAPPING
    return market in MARKETS_MAPPING


def parse_baseball_ip(raw) -> float:
    """Convert baseball-notation innings pitched to true decimal innings.

    BDL (like every box score) reports IP as X.0/X.1/X.2 where the tenths
    digit counts OUTS, not tenths of an inning: "5.2" means 5⅔ innings.
    Treating it as a plain float understates IP by up to 0.37 innings per
    start and inflates every rate stat (K/9, ERA, WHIP) built on top.

    Values whose fractional part is not ~0.1/0.2 are passed through unchanged
    (already true decimals). Returns 0.0 for None/garbage.
    """
    if raw is None:
        return 0.0
    try:
        val = float(raw)
    except (ValueError, TypeError):
        return 0.0
    if val < 0:
        return 0.0
    whole = int(val)
    tenths = round((val - whole) * 10)
    if abs((val - whole) * 10 - tenths) < 0.05 and tenths in (1, 2):
        return whole + tenths / 3.0
    return val
