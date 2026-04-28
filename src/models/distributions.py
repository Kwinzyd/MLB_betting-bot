import math
from typing import Optional

from scipy.stats import poisson, norm

from src.config import (
    LEAGUE_AVG_HR9, LEAGUE_AVG_ISO, HR_PI0_MAX,
    HR_PI0_BETA_PITCHER, HR_PI0_BETA_BATTER,
    HR_PI0_BETA_PARK, HR_PI0_BETA_WIND,
)


def compute_hr_pi0(pitcher_hr9: Optional[float] = None,
                   batter_iso: Optional[float] = None,
                   park_hr_factor: Optional[float] = None,
                   wind_in_mph: Optional[float] = None) -> float:
    """Heuristic structural-zero probability for batter_home_runs.

    Returns 0.0 when no inputs are supplied (recovers vanilla NB). Inputs that
    favor 0 HR (low HR/9 pitcher, low-ISO batter, suppressive park, wind blowing
    in) push pi0 upward, capped at HR_PI0_MAX.
    """
    z = 0.0
    if pitcher_hr9 is not None:
        z += HR_PI0_BETA_PITCHER * (LEAGUE_AVG_HR9 - pitcher_hr9)
    if batter_iso is not None:
        z += HR_PI0_BETA_BATTER * (LEAGUE_AVG_ISO - batter_iso)
    if park_hr_factor is not None:
        z += HR_PI0_BETA_PARK * (1.0 - park_hr_factor)
    if wind_in_mph is not None:
        z += HR_PI0_BETA_WIND * wind_in_mph
    if z <= 0:
        return 0.0
    sig = 1.0 / (1.0 + math.exp(-z))
    return max(0.0, min(HR_PI0_MAX, (sig - 0.5) * 2.0 * HR_PI0_MAX))


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
                      alpha: float = None, sigma: float = None,
                      pi0: float = None) -> tuple:
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
        if pi0 is not None and pi0 > 0:
            from src.models.dispersion import zinb_prob_over as _zinb
            return _zinb(mean, line, alpha if (alpha and alpha > 0) else 0.0, pi0)
        if alpha is not None and alpha > 0:
            from src.models.dispersion import negbin_prob_over as _nb
            return _nb(mean, line, alpha)
        prob_over = poisson_prob_over(mean, line)
        prob_under = poisson_prob_under(mean, line)

    return prob_over, prob_under


def get_probabilities_mixture(per_pa_rate: float, line: float, market: str,
                              pa_distribution: dict, adjustments: float = 1.0,
                              *, alpha: float = None, sigma: float = None,
                              pi0: float = None) -> tuple:
    """Mix per-PA-conditional CDFs across a PA probability distribution.

    For each PA count k with weight p_k, compute mean_k = per_pa_rate * k *
    adjustments and call get_probabilities. Mixing CDFs is exact for both
    Poisson/NB and Normal markets.
    """
    prob_over = 0.0
    prob_under = 0.0
    for k, p_k in pa_distribution.items():
        if p_k <= 0:
            continue
        mean_k = per_pa_rate * k * adjustments
        over_k, under_k = get_probabilities(
            mean_k, line, market, alpha=alpha, sigma=sigma, pi0=pi0,
        )
        prob_over += p_k * over_k
        prob_under += p_k * under_k
    return prob_over, prob_under
