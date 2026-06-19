import math
from typing import Optional

from scipy.stats import poisson, norm, nbinom

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


def _count_pmf(k: int, mean: float, alpha: float = None, pi0: float = None) -> float:
    """PMF of the count distribution (Poisson / NB / ZINB) at integer k.

    Mirrors the parametrization in src/models/dispersion.py:
    NB(n=1/alpha, p=1/(1+alpha*mean)); ZINB rescales the NB mean to
    mean/(1-pi0) so the marginal mean is preserved.
    """
    if k < 0:
        return 0.0
    if pi0 is not None and pi0 > 0 and math.isfinite(pi0):
        pi0 = min(pi0, 0.999)
        mu_nb = mean / (1.0 - pi0)
        if alpha is not None and alpha > 0 and math.isfinite(alpha):
            n = 1.0 / alpha
            p = 1.0 / (1.0 + alpha * mu_nb)
            base = float(nbinom.pmf(k, n, p))
        else:
            base = float(poisson.pmf(k, mu_nb))
        return (1.0 - pi0) * base + (pi0 if k == 0 else 0.0)
    if alpha is not None and alpha > 0 and math.isfinite(alpha):
        n = 1.0 / alpha
        p = 1.0 / (1.0 + alpha * mean)
        return float(nbinom.pmf(k, n, p))
    return float(poisson.pmf(k, mean))


def _raw_probs(mean: float, line: float, market: str,
               variance_factor: float = 1.2, *,
               alpha: float = None, sigma: float = None,
               pi0: float = None) -> tuple:
    """Return (p_win_over, p_win_under, p_push) — unnormalized outcome masses.

    Half-point lines have zero push mass. Integer lines push on an exact hit:
    counts use the PMF at the line; the Normal (total_bases) market uses a
    continuity-corrected ±0.5 band.
    """
    if mean <= 0:
        return 0.0, 1.0, 0.0

    is_integer_line = float(line).is_integer()

    if market == "batter_total_bases":
        if sigma is not None and sigma > 0:
            std_dev = max(0.5, sigma)
        else:
            std_dev = max(0.5, math.sqrt(mean * variance_factor))
        if is_integer_line:
            p_over = normal_prob_over(mean, std_dev, line + 0.5)
            p_under = normal_prob_under(mean, std_dev, line - 0.5)
            p_push = max(0.0, 1.0 - p_over - p_under)
        else:
            p_over = normal_prob_over(mean, std_dev, line)
            p_under = normal_prob_under(mean, std_dev, line)
            p_push = 0.0
        return p_over, p_under, p_push

    # Count markets (Poisson / NB / ZINB)
    if pi0 is not None and pi0 > 0:
        from src.models.dispersion import zinb_prob_over as _zinb
        p_over, p_under_incl = _zinb(
            mean, line, alpha if (alpha and alpha > 0) else 0.0, pi0)
    elif alpha is not None and alpha > 0:
        from src.models.dispersion import negbin_prob_over as _nb
        p_over, p_under_incl = _nb(mean, line, alpha)
    else:
        p_over = poisson_prob_over(mean, line)
        p_under_incl = poisson_prob_under(mean, line)

    if is_integer_line:
        p_push = _count_pmf(int(line), mean, alpha=alpha, pi0=pi0)
        p_under = max(0.0, p_under_incl - p_push)
    else:
        p_push = 0.0
        p_under = p_under_incl
    return p_over, p_under, p_push


def get_probabilities(mean: float, line: float, market: str,
                      variance_factor: float = 1.2, *,
                      alpha: float = None, sigma: float = None,
                      pi0: float = None) -> tuple:
    """
    Returns (prob_over, prob_under) for a given MLB market.

    When *alpha* (NB dispersion) or *sigma* (Normal residual std) are supplied,
    the fitted overdispersed distribution is used instead of the default
    Poisson / global-variance Normal.

    Integer lines are push-aware: books refund an exact hit, so the returned
    probabilities are conditional on no push. That matches a sharp book's
    two-way devig at the same line (also implicitly push-conditional), so the
    edge and Kelly math stay apples-to-apples. At half-point lines this is
    identical to the unconditional CDF split.
    """
    p_over, p_under, p_push = _raw_probs(
        mean, line, market, variance_factor,
        alpha=alpha, sigma=sigma, pi0=pi0,
    )
    if p_push <= 0:
        return p_over, p_under
    denom = p_over + p_under
    if denom <= 0:
        return 0.0, 1.0
    return p_over / denom, p_under / denom


def get_probabilities_mixture(per_pa_rate: float, line: float, market: str,
                              pa_distribution: dict, adjustments: float = 1.0,
                              *, alpha: float = None, sigma: float = None,
                              pi0: float = None) -> tuple:
    """Mix per-PA-conditional outcome masses across a PA distribution.

    For each PA count k with weight p_k, compute mean_k = per_pa_rate * k *
    adjustments and accumulate the raw (over, under, push) masses; the
    push-conditional normalization happens once on the mixture, which is
    exact (normalizing each component first would weight the bins wrongly).
    """
    p_over = 0.0
    p_under = 0.0
    p_push = 0.0
    for k, p_k in pa_distribution.items():
        if p_k <= 0:
            continue
        mean_k = per_pa_rate * k * adjustments
        over_k, under_k, push_k = _raw_probs(
            mean_k, line, market, alpha=alpha, sigma=sigma, pi0=pi0,
        )
        p_over += p_k * over_k
        p_under += p_k * under_k
        p_push += p_k * push_k
    if p_push <= 0:
        return p_over, p_under
    denom = p_over + p_under
    if denom <= 0:
        return 0.0, 1.0
    return p_over / denom, p_under / denom
