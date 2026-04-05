def decimal_to_implied(odds: float) -> float:
    return 1.0 / odds


def devig_additive(odds1: float, odds2: float) -> tuple:
    """
    Removes vig using additive method.
    Returns (true_prob1, true_prob2).
    """
    imp1 = decimal_to_implied(odds1)
    imp2 = decimal_to_implied(odds2)
    margin = imp1 + imp2 - 1.0
    true1 = imp1 - (margin / 2.0)
    true2 = imp2 - (margin / 2.0)
    return max(0.001, true1), max(0.001, true2)


def devig_multiplicative(odds1: float, odds2: float) -> tuple:
    """
    Removes vig using multiplicative method (preferred for prop markets).
    """
    imp1 = decimal_to_implied(odds1)
    imp2 = decimal_to_implied(odds2)
    total_implied = imp1 + imp2

    true1 = imp1 / total_implied
    true2 = imp2 / total_implied
    return true1, true2
