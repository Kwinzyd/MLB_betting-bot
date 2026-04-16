"""Fit per-entity negative-binomial dispersion (α) and Normal residual σ.

Runs after the mean model trains.  For each market, loads the trained GLM,
predicts μ̂ for every historical game-log row, then computes:
  - Pooled market-wide α (or σ for total_bases) via method-of-moments.
  - Per-entity α/σ shrunk toward the pool via Empirical Bayes.

Results go to the ``dispersion_params`` table and are consumed at scan time
by projections.py.

Usage:
    python main.py fit-dispersion        (standalone, after train)
    Called automatically at the end of train_all().
"""
from __future__ import annotations

import os
from collections import defaultdict
from datetime import datetime
from typing import Dict, List, Tuple

import numpy as np

from src.config import DISPERSION_MIN_OBS, DISPERSION_PRIOR_K, DISPERSION_ALPHA_CAP
from src.data.db import get_db_connection
from src.data.feature_builder import (
    PITCHER_FEATURE_NAMES, BATTER_FEATURE_NAMES,
    build_pitcher_features, build_batter_features,
    compute_bullpen_factor, compute_rest_days,
)
from src.models.dispersion import fit_negbin_mom, fit_normal_residual_std, shrink
from src.models.ml_model import PoissonGLM
from src.utils.logging_utils import get_logger

logger = get_logger(__name__)

_MODELS_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "models"
)

_PITCHER_MARKETS = ("pitcher_strikeouts", "pitcher_earned_runs")
_BATTER_MARKETS = ("batter_hits", "batter_total_bases", "batter_home_runs")
_ALL_MARKETS = _PITCHER_MARKETS + _BATTER_MARKETS

_MARKET_TO_TARGET = {
    "pitcher_strikeouts": ("pitcher_game_logs", "strikeouts", "innings_pitched"),
    "pitcher_earned_runs": ("pitcher_game_logs", "earned_runs", "innings_pitched"),
    "batter_hits": ("batter_game_logs", "hits", "plate_appearances"),
    "batter_total_bases": ("batter_game_logs", "total_bases", "plate_appearances"),
    "batter_home_runs": ("batter_game_logs", "home_runs", "plate_appearances"),
}


def _load_glm(market: str) -> PoissonGLM | None:
    path = os.path.join(_MODELS_DIR, f"mlb_poisson_{market}.pkl")
    if not os.path.exists(path):
        return None
    try:
        return PoissonGLM.load(path)
    except Exception as e:
        logger.warning(f"Could not load GLM for {market}: {e}")
        return None


def _collect_pitcher_residuals(market: str, glm: PoissonGLM) -> Dict[int, List[Tuple[float, float]]]:
    """Returns {player_id: [(y, μ̂), ...]} for a pitcher market."""
    table, target_col, exposure_col = _MARKET_TO_TARGET[market]
    result: Dict[int, List[Tuple[float, float]]] = defaultdict(list)

    with get_db_connection() as conn:
        rows = conn.execute(
            f"SELECT * FROM {table} ORDER BY player_id, date ASC"
        ).fetchall()
        by_player: Dict[int, List[dict]] = defaultdict(list)
        for r in rows:
            by_player[r["player_id"]].append(dict(r))

        game_ctx_rows = conn.execute(
            "SELECT game_id, bdl_game_id, date, game_time, venue, home_team_id, away_team_id FROM games"
        ).fetchall()
        games_by_bdl = {}
        for r in game_ctx_rows:
            if r["bdl_game_id"] is not None:
                games_by_bdl[r["bdl_game_id"]] = dict(r)

        for player_id, logs in by_player.items():
            for i, row in enumerate(logs):
                if i < 3:
                    continue
                prior = list(reversed(logs[:i]))
                target = row.get(target_col) or 0
                exposure = row.get(exposure_col) or 0
                if not exposure or exposure <= 0:
                    continue

                game_ctx = games_by_bdl.get(row["game_id"], {}) or {}
                opp_team_id = game_ctx.get("home_team_id") if game_ctx else None
                extra = {
                    "game_date": row.get("date"),
                    "game_time": game_ctx.get("game_time"),
                    "is_home": 0,
                    "month_of_season": None,
                    "rest_days": compute_rest_days(prior, row.get("date")),
                    "opp_bullpen_era": compute_bullpen_factor(opp_team_id, row.get("date"), conn),
                }
                try:
                    X = build_pitcher_features(
                        pitcher_logs=prior, market=market, opponent_rate=None,
                        venue=game_ctx.get("venue"), weather=None,
                        ump_k_factor=1.0, projected_ip=float(exposure), extra=extra,
                    )
                    mu = glm.predict_mean(X, exposure=float(exposure))
                except Exception:
                    continue
                result[player_id].append((float(target), float(mu)))

    return result


def _collect_batter_residuals(market: str, glm: PoissonGLM) -> Dict[int, List[Tuple[float, float]]]:
    """Returns {player_id: [(y, μ̂), ...]} for a batter market."""
    table, target_col, exposure_col = _MARKET_TO_TARGET[market]
    result: Dict[int, List[Tuple[float, float]]] = defaultdict(list)

    with get_db_connection() as conn:
        rows = conn.execute(
            f"SELECT * FROM {table} ORDER BY player_id, date ASC"
        ).fetchall()
        by_player: Dict[int, List[dict]] = defaultdict(list)
        for r in rows:
            by_player[r["player_id"]].append(dict(r))

        player_hand = {}
        for r in conn.execute("SELECT player_id, bats FROM players").fetchall():
            player_hand[r["player_id"]] = r["bats"]

        game_ctx_rows = conn.execute(
            "SELECT game_id, bdl_game_id, date, game_time, venue FROM games"
        ).fetchall()
        games_by_bdl = {}
        for r in game_ctx_rows:
            if r["bdl_game_id"] is not None:
                games_by_bdl[r["bdl_game_id"]] = dict(r)

        for player_id, logs in by_player.items():
            batter_hand = player_hand.get(player_id, "R")
            for i, row in enumerate(logs):
                if i < 5:
                    continue
                prior = list(reversed(logs[:i]))
                target = row.get(target_col) or 0
                exposure = row.get(exposure_col) or 0
                if not exposure or exposure <= 0:
                    continue

                game_ctx = games_by_bdl.get(row["game_id"], {}) or {}
                try:
                    X = build_batter_features(
                        batter_logs=prior, market=market,
                        pitcher_hand="R", batter_hand=batter_hand or "R",
                        venue=game_ctx.get("venue"), weather=None,
                        lineup_position=5, projected_pa=float(exposure),
                    )
                    mu = glm.predict_mean(X, exposure=float(exposure))
                except Exception:
                    continue
                result[player_id].append((float(target), float(mu)))

    return result


def _fit_market(market: str) -> List[dict]:
    """Fit dispersion for a single market. Returns rows for DB insert."""
    glm = _load_glm(market)
    if glm is None:
        logger.info(f"  {market}: no trained GLM, skipping dispersion fit.")
        return []

    is_pitcher = market in _PITCHER_MARKETS
    if is_pitcher:
        residuals = _collect_pitcher_residuals(market, glm)
    else:
        residuals = _collect_batter_residuals(market, glm)

    if not residuals:
        logger.info(f"  {market}: no residual data, skipping.")
        return []

    is_tb = market == "batter_total_bases"
    entity_type = "pitcher" if is_pitcher else "batter"
    now = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")

    all_y = np.concatenate([np.array([t[0] for t in v]) for v in residuals.values()])
    all_mu = np.concatenate([np.array([t[1] for t in v]) for v in residuals.values()])

    if is_tb:
        pooled_val = fit_normal_residual_std(all_y, all_mu)
    else:
        pooled_val = fit_negbin_mom(all_y, all_mu)
        pooled_val = min(pooled_val, DISPERSION_ALPHA_CAP)

    rows = []
    rows.append({
        "entity_id": "__pool__",
        "entity_type": "pool",
        "market": market,
        "alpha": None if is_tb else pooled_val,
        "sigma": pooled_val if is_tb else None,
        "n_obs": int(len(all_y)),
        "fitted_at": now,
    })

    for player_id, pairs in residuals.items():
        n = len(pairs)
        if n < DISPERSION_MIN_OBS:
            continue
        y_arr = np.array([t[0] for t in pairs])
        mu_arr = np.array([t[1] for t in pairs])

        if is_tb:
            raw = fit_normal_residual_std(y_arr, mu_arr)
            val = shrink(raw, pooled_val, n, k=DISPERSION_PRIOR_K)
            rows.append({
                "entity_id": str(player_id),
                "entity_type": entity_type,
                "market": market,
                "alpha": None,
                "sigma": val,
                "n_obs": n,
                "fitted_at": now,
            })
        else:
            raw = fit_negbin_mom(y_arr, mu_arr)
            raw = min(raw, DISPERSION_ALPHA_CAP)
            val = shrink(raw, pooled_val, n, k=DISPERSION_PRIOR_K)
            val = min(val, DISPERSION_ALPHA_CAP)
            rows.append({
                "entity_id": str(player_id),
                "entity_type": entity_type,
                "market": market,
                "alpha": val,
                "sigma": None,
                "n_obs": n,
                "fitted_at": now,
            })

    logger.info(
        f"  {market}: pooled={'σ' if is_tb else 'α'}={pooled_val:.4f}, "
        f"{len(rows)-1} entity rows"
    )
    return rows


def fit_all_dispersion() -> None:
    """Fit dispersion params for all markets and write to DB."""
    logger.info("Fitting per-entity dispersion parameters...")

    all_rows = []
    for market in _ALL_MARKETS:
        try:
            all_rows.extend(_fit_market(market))
        except Exception as e:
            logger.error(f"Dispersion fit failed for {market}: {e}", exc_info=True)

    if not all_rows:
        logger.info("No dispersion rows to write.")
        return

    with get_db_connection() as conn:
        conn.execute(
            "CREATE TABLE IF NOT EXISTS dispersion_params ("
            "id INTEGER PRIMARY KEY AUTOINCREMENT, "
            "entity_id TEXT NOT NULL, entity_type TEXT NOT NULL, "
            "market TEXT NOT NULL, alpha REAL, sigma REAL, "
            "n_obs INTEGER NOT NULL, fitted_at TEXT NOT NULL, "
            "UNIQUE(entity_id, market))"
        )
        for r in all_rows:
            conn.execute(
                "INSERT OR REPLACE INTO dispersion_params "
                "(entity_id, entity_type, market, alpha, sigma, n_obs, fitted_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?)",
                (r["entity_id"], r["entity_type"], r["market"],
                 r["alpha"], r["sigma"], r["n_obs"], r["fitted_at"]),
            )
        conn.commit()

    logger.info(f"Wrote {len(all_rows)} dispersion_params rows.")
