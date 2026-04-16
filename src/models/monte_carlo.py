"""Monte Carlo simulation for prop probability estimation.

Replaces the direct Poisson CDF in distributions.py with empirical probability
estimates from n=1000 draws, which gives a sanity layer (jointly verifies the
distribution choice + mean) and lets us swap in fitted variance for total_bases
without rewiring edge_ranker.
"""
import math
import numpy as np

_POISSON_MARKETS = frozenset({
    "pitcher_strikeouts",
    "pitcher_earned_runs",
    "batter_hits",
    "batter_home_runs",
})

_DEFAULT_TB_STD_VARIANCE_FACTOR = 1.2


def mc_prob_over(mean: float, line: float, market: str, n_sims: int = 1000,
                 tb_std: float = None, nb_alpha: float = None) -> tuple[float, float]:
    """
    Returns (prob_over, prob_under) via Monte Carlo simulation.

    - Count markets: NB sampling when nb_alpha > 0, else Poisson.
    - batter_total_bases: Normal sampling with tb_std (fitted) or sqrt(mean*1.2) default.
    """
    if mean is None or mean <= 0 or math.isnan(mean):
        return 0.0, 1.0

    rng = np.random.default_rng()

    if market in _POISSON_MARKETS:
        if nb_alpha is not None and nb_alpha > 0 and math.isfinite(nb_alpha):
            n = 1.0 / nb_alpha
            p = 1.0 / (1.0 + nb_alpha * mean)
            samples = rng.negative_binomial(n, p, size=n_sims)
        else:
            samples = rng.poisson(lam=mean, size=n_sims)
    elif market == "batter_total_bases":
        std = tb_std if tb_std is not None else max(0.5, math.sqrt(mean * _DEFAULT_TB_STD_VARIANCE_FACTOR))
        samples = rng.normal(loc=mean, scale=std, size=n_sims)
    else:
        samples = rng.poisson(lam=mean, size=n_sims)

    prob_over = float(np.mean(samples > line))
    prob_under = float(np.mean(samples <= line))
    return prob_over, prob_under
