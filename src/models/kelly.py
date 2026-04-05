from src.config import KELLY_FRACTION, BANKROLL


def fractional_kelly(model_prob: float, decimal_odds: float,
                     fraction: float = None, bankroll: float = None) -> dict:
    """
    Calculate fractional Kelly stake.

    Full Kelly: f* = (b*p - q) / b
    where b = decimal_odds - 1, p = model_prob, q = 1 - p

    Returns dict with kelly_fraction, recommended_stake, full_kelly_pct.
    Negative Kelly = no bet (edge is negative).
    """
    fraction = fraction or KELLY_FRACTION
    bankroll = bankroll or BANKROLL

    b = decimal_odds - 1.0
    p = model_prob
    q = 1.0 - p

    if b <= 0:
        return {"kelly_fraction": 0.0, "recommended_stake": 0.0, "full_kelly_pct": 0.0}

    full_kelly = (b * p - q) / b

    if full_kelly <= 0:
        return {"kelly_fraction": 0.0, "recommended_stake": 0.0, "full_kelly_pct": full_kelly}

    adjusted_kelly = full_kelly * fraction

    # Cap at 5% of bankroll max per single bet
    max_fraction = 0.05
    adjusted_kelly = min(adjusted_kelly, max_fraction)

    stake = bankroll * adjusted_kelly

    return {
        "kelly_fraction": round(adjusted_kelly, 4),
        "recommended_stake": round(stake, 2),
        "full_kelly_pct": round(full_kelly, 4),
    }
