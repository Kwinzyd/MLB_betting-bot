"""Empirical portfolio correlation fitting.

Reads bet_pair_outcomes (populated by settle_results) and computes realized
Pearson correlations for the three pair types used by portfolio_kelly:

  same_team_batters  — batter props on the same team in the same game
  pitcher_batter     — one pitcher prop + one batter prop on the same team
  same_game_opp      — props on opposing teams in the same game

Formula (Pearson ρ from Bernoulli outcomes):
  ρ = (P(A∧B) − P(A)·P(B)) / √(P(A)(1−P(A)) · P(B)(1−P(B)))

Bootstrap 95% CI (1000 resamples) guards against updating on noise: the active
row is replaced only when the new estimate differs from the current one by more
than the existing CI half-width.

Minimum 100 resolved pairs per type before fitting; falls back to config
constants otherwise.

Usage:
    python main.py fit_correlations      (via CLI)
    fit_correlations()                   (called from scheduler at 04:30)
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Dict, List, Tuple

import numpy as np

from src.config import (
    PORTFOLIO_CORR_PITCHER_BATTER,
    PORTFOLIO_CORR_SAME_GAME_OPP_TEAM,
    PORTFOLIO_CORR_SAME_TEAM,
)
from src.data.db import get_db_connection
from src.utils.logging_utils import get_logger

logger = get_logger(__name__)

_MIN_PAIRS = 100
_N_BOOTSTRAP = 1000
_PAIR_TYPES = ("same_team_batters", "pitcher_batter", "same_game_opp")

# Config fallbacks indexed by pair_type
_FALLBACKS = {
    "same_team_batters": PORTFOLIO_CORR_SAME_TEAM,
    "pitcher_batter": PORTFOLIO_CORR_PITCHER_BATTER,
    "same_game_opp": PORTFOLIO_CORR_SAME_GAME_OPP_TEAM,
}


# ---------------------------------------------------------------------------
# Core maths
# ---------------------------------------------------------------------------

def _rho(p_a: float, p_b: float, p_ab: float) -> float:
    """Pearson ρ from marginal and joint Bernoulli probabilities."""
    denom = (
        (p_a * (1.0 - p_a)) ** 0.5
        * (p_b * (1.0 - p_b)) ** 0.5
    )
    if denom < 1e-10:
        return 0.0
    return (p_ab - p_a * p_b) / denom


def _compute_rho_from_arrays(
    a_won: np.ndarray, b_won: np.ndarray, both_won: np.ndarray
) -> float:
    return _rho(float(a_won.mean()), float(b_won.mean()), float(both_won.mean()))


def _compute_rho_with_ci(
    pairs: List[Dict], n_bootstrap: int = _N_BOOTSTRAP
) -> Tuple[float, float]:
    """Return (rho, ci_half_width) for a list of pair dicts."""
    a = np.array([p["a_won"] for p in pairs], dtype=np.float64)
    b = np.array([p["b_won"] for p in pairs], dtype=np.float64)
    ab = np.array([p["both_won"] for p in pairs], dtype=np.float64)

    point_rho = _compute_rho_from_arrays(a, b, ab)

    rng = np.random.default_rng(seed=42)
    n = len(pairs)
    boot_rhos = []
    for _ in range(n_bootstrap):
        idx = rng.integers(0, n, n)
        r = _compute_rho_from_arrays(a[idx], b[idx], ab[idx])
        if not np.isnan(r):
            boot_rhos.append(r)

    if boot_rhos:
        ci = (float(np.percentile(boot_rhos, 97.5)) - float(np.percentile(boot_rhos, 2.5))) / 2.0
    else:
        ci = 0.05

    return float(point_rho), float(ci)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def fit_correlations() -> Dict:
    """Fit and persist empirical correlation params for the three pair types."""
    logger.info("fit_correlations: reading bet_pair_outcomes")

    with get_db_connection() as conn:
        rows = [dict(r) for r in conn.execute(
            "SELECT pair_type, both_won, a_won, b_won FROM bet_pair_outcomes"
        ).fetchall()]

    if not rows:
        logger.info("fit_correlations: no pair outcomes yet — nothing to fit.")
        return {}

    # Group by pair_type
    groups: Dict[str, List[Dict]] = {pt: [] for pt in _PAIR_TYPES}
    for r in rows:
        if r["pair_type"] in groups:
            groups[r["pair_type"]].append(r)

    fitted: Dict[str, Dict] = {}
    for pt in _PAIR_TYPES:
        pairs = groups[pt]
        if len(pairs) < _MIN_PAIRS:
            logger.info("  %s: %d pairs (need %d) — using config fallback %.3f",
                        pt, len(pairs), _MIN_PAIRS, _FALLBACKS[pt])
            continue
        rho, ci = _compute_rho_with_ci(pairs)
        fitted[pt] = {"rho": rho, "ci": ci, "n": len(pairs)}
        logger.info("  %s: n=%d  rho=%.3f  ci=±%.3f", pt, len(pairs), rho, ci)

    if not fitted:
        logger.info("fit_correlations: insufficient data across all pair types.")
        return {}

    # Collect new values (fall back to config for types with insufficient data)
    new_vals = {pt: fitted[pt]["rho"] if pt in fitted else _FALLBACKS[pt] for pt in _PAIR_TYPES}
    new_ci = max((fitted[pt]["ci"] for pt in fitted), default=0.05)

    # Only update if meaningfully different from current active params
    with get_db_connection() as conn:
        existing = conn.execute(
            "SELECT same_team_batters, pitcher_batter, same_game_opp, ci_half_width "
            "FROM correlation_params WHERE is_active=1 "
            "ORDER BY fitted_at DESC LIMIT 1"
        ).fetchone()

    if existing:
        cur = {
            "same_team_batters": float(existing["same_team_batters"] or _FALLBACKS["same_team_batters"]),
            "pitcher_batter": float(existing["pitcher_batter"] or _FALLBACKS["pitcher_batter"]),
            "same_game_opp": float(existing["same_game_opp"] or _FALLBACKS["same_game_opp"]),
        }
        gate = float(existing["ci_half_width"] or 0.05)
        max_delta = max(abs(new_vals[pt] - cur[pt]) for pt in _PAIR_TYPES)
        if max_delta <= gate:
            logger.info(
                "fit_correlations: max delta %.3f ≤ CI gate %.3f — no update needed.",
                max_delta, gate,
            )
            return fitted

    now = datetime.now(timezone.utc).isoformat()
    with get_db_connection() as conn:
        conn.execute("UPDATE correlation_params SET is_active=0")
        conn.execute(
            """INSERT INTO correlation_params
               (same_team_batters, pitcher_batter, same_game_opp,
                n_same_team, n_pitcher_batter, n_same_game,
                ci_half_width, fitted_at, is_active)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, 1)""",
            (
                new_vals["same_team_batters"],
                new_vals["pitcher_batter"],
                new_vals["same_game_opp"],
                fitted.get("same_team_batters", {}).get("n", 0),
                fitted.get("pitcher_batter", {}).get("n", 0),
                fitted.get("same_game_opp", {}).get("n", 0),
                new_ci, now,
            ),
        )
        conn.commit()

    logger.info(
        "fit_correlations: updated — same_team=%.3f  pitcher_batter=%.3f  same_game_opp=%.3f",
        new_vals["same_team_batters"], new_vals["pitcher_batter"], new_vals["same_game_opp"],
    )
    return fitted
