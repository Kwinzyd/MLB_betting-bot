import math
from scipy.stats import poisson, norm


def poisson_prob_over(mean: float, line: float) -> float:
    """P(X > line) assuming Poisson distribution. Line is typically a half-number like 5.5."""
    return 1.0 - poisson.cdf(math.floor(line), mean)


def poisson_prob_under(mean: float, line: float) -> float:
    """P(X <= floor(line)) assuming Poisson distribution."""
    return poisson.cdf(math.floor(line), mean)


def normal_prob_over(mean: float, std_dev: float, line: float) -> float:
    """P(X > line) assuming Normal distribution."""
    return norm.sf(line, loc=mean, scale=std_dev)


def normal_prob_under(mean: float, std_dev: float, line: float) -> float:
    """P(X <= line) assuming Normal distribution."""
    return norm.cdf(line, loc=mean, scale=std_dev)


def get_probabilities(mean: float, line: float, market: str, variance_factor: float = 1.2) -> tuple:
    """
    Returns (prob_over, prob_under) for a given MLB market.

    Distribution choices:
    - pitcher_strikeouts: Poisson (discrete count, mean typically 4-10)
    - batter_hits: Poisson (discrete count, mean typically 0.5-2)
    - batter_home_runs: Poisson (rare discrete event, mean typically 0.05-0.3)
    - pitcher_earned_runs: Poisson (discrete count, mean typically 1-4)
    - batter_total_bases: Normal (semi-continuous aggregate, higher variance)
    """
    if mean <= 0:
        return 0.0, 1.0

    if market == "batter_total_bases":
        # Total bases have higher variance - use Normal approximation
        std_dev = max(0.5, math.sqrt(mean * variance_factor))
        prob_over = normal_prob_over(mean, std_dev, line)
        prob_under = normal_prob_under(mean, std_dev, line)
    else:
        # Strikeouts, hits, home runs, earned runs are all discrete counts
        prob_over = poisson_prob_over(mean, line)
        prob_under = poisson_prob_under(mean, line)

    return prob_over, prob_under
