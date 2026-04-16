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


def get_probabilities(mean: float, line: float, market: str,
                      variance_factor: float = 1.2, *,
                      alpha: float = None, sigma: float = None) -> tuple:
    """
    Returns (prob_over, prob_under) for a given MLB market.

    When *alpha* (NB dispersion) or *sigma* (Normal residual std) are supplied,
    the fitted overdispersed distribution is used instead of the default
    Poisson / global-variance Normal.  Existing callers that omit the new
    kwargs get the original behaviour.
    """
    if mean <= 0:
        return 0.0, 1.0

    if market == "batter_total_bases":
        if sigma is not None and sigma > 0:
            std_dev = max(0.5, sigma)
        else:
            std_dev = max(0.5, math.sqrt(mean * variance_factor))
        prob_over = normal_prob_over(mean, std_dev, line)
        prob_under = normal_prob_under(mean, std_dev, line)
    else:
        if alpha is not None and alpha > 0:
            from src.models.dispersion import negbin_prob_over as _nb
            return _nb(mean, line, alpha)
        prob_over = poisson_prob_over(mean, line)
        prob_under = poisson_prob_under(mean, line)

    return prob_over, prob_under
