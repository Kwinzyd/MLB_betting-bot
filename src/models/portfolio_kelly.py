"""
Portfolio-level Kelly staking.

Solves the multi-bet Kelly problem: given a slate of N simultaneous bets,
find the stake vector f = [f1, ..., fN] that maximises the expected log-growth
of the bankroll, subject to individual and total exposure caps.

Why this matters
----------------
Per-bet fractional Kelly ignores correlations. If you have 3 bets on the same
game (pitcher K-over, batter-1 hits-under, batter-2 hits-under), all three
lose together when the starter gets knocked out early. The true bankroll
variance is much higher than the sum of independent variances, so per-bet
Kelly overstates how much you can safely stake.

Approach
--------
We use a first-order approximation to the log-growth maximization problem,
derived from a second-order Taylor expansion of E[log(1 + r)]:

    max  E[log(1 + f·r)]
    ≈    max  μ·f - 0.5 * f^T Σ f

where μ_i = edge_i / odds_i (expected return per unit staked) and
Σ is the return covariance matrix.

The unconstrained solution is f* = Σ^{-1} μ, which we then clip to enforce:
  - f_i ≥ 0  (no shorting)
  - f_i ≤ PER_BET_CAP (5% of bankroll per leg)
  - sum(f_i) ≤ TOTAL_EXPOSURE_CAP (configurable, default 20% of bankroll)

Covariance estimation
---------------------
The exact joint return distribution is unknown. We use a structured estimator
based on three signal types:

1. Same game, same team:       ρ = +0.55
   (e.g. batter hits + same batter total_bases — near-collinear)

2. Same game, same team, but PITCHER vs BATTER cross:  ρ = -0.25
   (high K-count by pitcher → fewer batter opportunities)

3. Same game, different teams:  ρ = +0.10
   (total runs affect both sides; correlation is weak but positive)

4. Different games:             ρ = 0.00
   (statistically independent)

These are empirical approximations. They're intentionally conservative
(biased toward positive correlations) to undersize stakes when uncertain.
You can tune them in config.py via PORTFOLIO_CORR_* constants.
"""
from __future__ import annotations

import math
import time
from typing import List

import numpy as np

from src.config import (
    KELLY_FRACTION,
    PORTFOLIO_CORR_SAME_TEAM,
    PORTFOLIO_CORR_PITCHER_BATTER,
    PORTFOLIO_CORR_SAME_GAME_OPP_TEAM,
    PORTFOLIO_TOTAL_EXPOSURE_CAP,
)
from src.utils.logging_utils import get_logger

logger = get_logger(__name__)

# Per-leg hard cap regardless of portfolio math (same as single-bet cap).
_PER_BET_CAP = 0.05

_corr_cache: dict | None = None
_corr_cache_ts: float = 0.0
_CORR_CACHE_TTL = 3600.0


def _load_corr_params() -> dict:
    """Load active correlation_params from DB with 1h TTL; fall back to config constants."""
    global _corr_cache, _corr_cache_ts
    now = time.monotonic()
    if _corr_cache is not None and (now - _corr_cache_ts) < _CORR_CACHE_TTL:
        return _corr_cache
    result = {
        "same_team_batters": PORTFOLIO_CORR_SAME_TEAM,
        "pitcher_batter": PORTFOLIO_CORR_PITCHER_BATTER,
        "same_game_opp": PORTFOLIO_CORR_SAME_GAME_OPP_TEAM,
    }
    try:
        from src.data.db import get_db_connection
        with get_db_connection() as conn:
            row = conn.execute(
                "SELECT same_team_batters, pitcher_batter, same_game_opp "
                "FROM correlation_params WHERE is_active=1 "
                "ORDER BY fitted_at DESC LIMIT 1"
            ).fetchone()
        if row:
            if row["same_team_batters"] is not None:
                result["same_team_batters"] = float(row["same_team_batters"])
            if row["pitcher_batter"] is not None:
                result["pitcher_batter"] = float(row["pitcher_batter"])
            if row["same_game_opp"] is not None:
                result["same_game_opp"] = float(row["same_game_opp"])
            logger.debug(
                "Loaded empirical correlations: same_team=%.3f pitcher_batter=%.3f opp=%.3f",
                result["same_team_batters"], result["pitcher_batter"], result["same_game_opp"],
            )
    except Exception:
        pass  # table missing on fresh DB — silently use config constants
    _corr_cache = result
    _corr_cache_ts = now
    return result


def _classify_market(market: str) -> str:
    """Return 'pitcher' or 'batter' for covariance sign logic."""
    return "pitcher" if market.startswith("pitcher_") else "batter"


def _build_correlation_matrix(bets: List[dict]) -> np.ndarray:
    """
    Build NxN correlation matrix from structured domain knowledge.

    Each bet dict must contain:
        game_id   : str
        team      : str   (team the player plays for)
        market    : str   (e.g. 'pitcher_strikeouts')
    """
    corr = _load_corr_params()
    n = len(bets)
    rho = np.eye(n)

    for i in range(n):
        for j in range(i + 1, n):
            a, b = bets[i], bets[j]

            if a["game_id"] != b["game_id"]:
                r = 0.0
            elif a["team"] == b["team"]:
                types = {_classify_market(a["market"]), _classify_market(b["market"])}
                if len(types) == 2:  # one pitcher, one batter on same team
                    r = corr["pitcher_batter"]
                else:
                    r = corr["same_team_batters"]
            else:
                r = corr["same_game_opp"]

            rho[i, j] = r
            rho[j, i] = r

    return rho


def _build_covariance_matrix(bets: List[dict], rho: np.ndarray) -> np.ndarray:
    """
    Convert correlation matrix to covariance matrix using return volatilities.

    Per-bet return volatility: σ_i ≈ odds_i * sqrt(p_i * (1-p_i))
    This is the standard deviation of the Bernoulli payout per unit staked.
    """
    sigma = np.array([
        b["decimal_odds"] * math.sqrt(max(b["prob"] * (1 - b["prob"]), 1e-8))
        for b in bets
    ])
    # Σ = diag(σ) · ρ · diag(σ)
    cov = np.outer(sigma, sigma) * rho
    return cov


def portfolio_kelly(
    bets: List[dict],
    bankroll: float,
    fraction: float = None,
) -> List[float]:
    """
    Compute portfolio Kelly stakes (in $ amount) for a slate of simultaneous bets.

    Parameters
    ----------
    bets : list of dicts, each with keys:
        game_id      : str
        team         : str
        market       : str
        prob         : float    model probability of the chosen side winning
        decimal_odds : float    offered odds (decimal)
        side         : str      'over' | 'under'
    bankroll : float
        Current ledger-adjusted bankroll.
    fraction : float, optional
        Fractional Kelly multiplier (default: KELLY_FRACTION from config).

    Returns
    -------
    List of dollar stakes, one per bet, in the same order as `bets`.
    Returns zeros for all bets if the problem is trivially empty or ill-conditioned.
    """
    fraction = KELLY_FRACTION if fraction is None else fraction
    n = len(bets)

    if n == 0:
        return []
    if n == 1:
        # Single bet — fall back to standard fractional Kelly (no portfolio math needed)
        b = bets[0]
        f = _single_kelly_fraction(b["prob"], b["decimal_odds"], fraction)
        return [round(bankroll * f, 2)]

    # --- Build expected return vector (μ) ---
    # μ_i = E[return per unit staked] = p_i * (odds_i - 1) - (1 - p_i)
    mu = np.array([
        b["prob"] * (b["decimal_odds"] - 1.0) - (1.0 - b["prob"])
        for b in bets
    ])

    # Drop negative-edge bets from the calculation; they'll get 0 stake.
    positive_mask = mu > 0
    if not positive_mask.any():
        logger.debug("Portfolio Kelly: no positive-edge bets in slate.")
        return [0.0] * n

    # --- Build covariance matrix (Σ) ---
    rho = _build_correlation_matrix(bets)
    cov = _build_covariance_matrix(bets, rho)

    # --- Solve f* = Σ^{-1} μ (unconstrained portfolio Kelly) ---
    try:
        # Add small regularisation (Tikhonov) so near-singular matrices don't blow up
        reg = 1e-6 * np.eye(n)
        f_star = np.linalg.solve(cov + reg, mu)
    except np.linalg.LinAlgError:
        logger.warning("Portfolio Kelly: covariance matrix singular; falling back to per-bet Kelly.")
        return [
            round(bankroll * _single_kelly_fraction(b["prob"], b["decimal_odds"], fraction), 2)
            for b in bets
        ]

    # --- Apply fractional Kelly scaling ---
    f_star = f_star * fraction

    # --- Enforce constraints ---
    # 1. No negative fractions (no shorting)
    f_star = np.maximum(f_star, 0.0)

    # 2. Zero out originally negative-edge bets
    f_star[~positive_mask] = 0.0

    # 3. Per-leg cap
    f_star = np.minimum(f_star, _PER_BET_CAP)

    # 4. Total exposure cap — scale down proportionally if needed
    total = f_star.sum()
    if total > PORTFOLIO_TOTAL_EXPOSURE_CAP:
        f_star = f_star * (PORTFOLIO_TOTAL_EXPOSURE_CAP / total)

    stakes = [round(bankroll * float(fi), 2) for fi in f_star]

    logger.debug(
        "Portfolio Kelly: n=%d bets | fractions=%s | total_exp=%.1f%%",
        n,
        [f"{fi:.3f}" for fi in f_star],
        f_star.sum() * 100,
    )
    return stakes


def _single_kelly_fraction(prob: float, decimal_odds: float, fraction: float) -> float:
    """Standard fractional Kelly fraction, capped at PER_BET_CAP."""
    b = decimal_odds - 1.0
    if b <= 0 or prob <= 0:
        return 0.0
    full_kelly = (b * prob - (1.0 - prob)) / b
    if full_kelly <= 0:
        return 0.0
    return min(full_kelly * fraction, _PER_BET_CAP)


def apply_portfolio_kelly(candidates: List[dict], bankroll: float) -> List[dict]:
    """
    High-level helper called by send_alerts.

    Takes the list of alert candidate dicts (already ranked, filtered by
    MAX_BETS_PER_GAME / MAX_BETS_PER_PLAYER), computes portfolio-aware stakes,
    and patches each candidate's `edge['kelly']['recommended_stake']` in place.

    Each candidate dict must have:
        row['game_id']   : str
        row['home_team'] : str   (used as proxy for pitcher's team)
        row['market']    : str
        side             : str   'over' | 'under'
        edge['kelly']    : dict  from fractional_kelly()

    The 'team' field is inferred: for pitcher markets we use the home team
    (starter is most often the home pitcher — good enough for correlation
    grouping; the exact team doesn't matter as long as it's consistent).
    """
    if not candidates:
        return candidates

    # Build bet descriptors for the solver
    bet_descs = []
    for c in candidates:
        row = c["row"]
        prob = c["projection"]["prob_over"] if c["side"] == "over" else c["projection"]["prob_under"]
        market = row["market"]
        # Approximate team: pitcher props belong to home team, batter props to their own team.
        # For correlation grouping, what matters is same-game + same-team accuracy.
        team = row.get("player_team") or row.get("home_team", "")
        bet_descs.append({
            "game_id":      row["game_id"],
            "team":         team,
            "market":       market,
            "prob":         float(prob or 0.5),
            "decimal_odds": float(c["odds"]),
            "side":         c["side"],
        })

    stakes = portfolio_kelly(bet_descs, bankroll)

    for cand, stake in zip(candidates, stakes):
        cand["edge"]["kelly"]["recommended_stake"] = stake
        cand["edge"]["kelly"]["portfolio_sized"] = True

    return candidates
