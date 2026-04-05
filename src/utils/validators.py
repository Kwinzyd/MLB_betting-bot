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
