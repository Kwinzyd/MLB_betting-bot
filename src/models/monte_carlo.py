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

# Empirical residual std for total bases; replaced by fitted value if the
# model training pipeline writes one to models/tb_residual_std.txt.
_DEFAULT_TB_STD_VARIANCE_FACTOR = 1.2


def mc_prob_over(mean: float, line: float, market: str, n_sims: int = 1000,
                 tb_std: float = None) -> tuple[float, float]:
    """
    Returns (prob_over, prob_under) via Monte Carlo simulation.

    - Count markets (K, H, HR, ER) → Poisson sampling.
    - batter_total_bases → Normal sampling with std from tb_std (or a sqrt(mean * 1.2) default).
    Over is strict: P(X > line). With .5 lines this coincides with P(X >= ceil(line)).
    """
    if mean is None or mean <= 0 or math.isnan(mean):
        return 0.0, 1.0

    rng = np.random.default_rng()

    if market in _POISSON_MARKETS:
        samples = rng.poisson(lam=mean, size=n_sims)
    elif market == "batter_total_bases":
        std = tb_std if tb_std is not None else max(0.5, math.sqrt(mean * _DEFAULT_TB_STD_VARIANCE_FACTOR))
        samples = rng.normal(loc=mean, scale=std, size=n_sims)
    else:
        # Unknown market: fall back to Poisson as the safest count-like default
        samples = rng.poisson(lam=mean, size=n_sims)

    prob_over = float(np.mean(samples > line))
    prob_under = float(np.mean(samples <= line))
    return prob_over, prob_under
