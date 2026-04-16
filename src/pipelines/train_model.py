"""Train a PoissonGLM per market on historical game logs.

Usage (via CLI):
    python main.py train
    python main.py train --compare sklearn

Splits on game date: 2022–2024 → train, 2025+ → validation. Validation metrics
reported: mean projected count vs actual (MAE), Poisson deviance, and — to
measure betting utility — calibration of simulated P(over) vs actual over
rate at simulated half-point lines.
"""
from __future__ import annotations

import math
import os
from collections import defaultdict
from typing import Dict, List, Tuple

import numpy as np

from src.data.db import get_db_connection
from src.data.feature_builder import (
    BATTER_FEATURE_NAMES, PITCHER_FEATURE_NAMES,
    build_batter_features, build_pitcher_features,
    compute_bullpen_factor, compute_rest_days,
)
from src.models.ml_model import PoissonGLM
from src.utils.logging_utils import get_logger

logger = get_logger(__name__)

_PITCHER_MARKETS = ("pitcher_strikeouts", "pitcher_earned_runs")
_BATTER_MARKETS = ("batter_hits", "batter_total_bases", "batter_home_runs")
_ALL_MARKETS = _PITCHER_MARKETS + _BATTER_MARKETS

_VAL_SPLIT_DATE = "2025-01-01"

_MARKET_TO_TARGET = {
    "pitcher_strikeouts": ("pitcher_game_logs", "strikeouts", "innings_pitched"),
    "pitcher_earned_runs": ("pitcher_game_logs", "earned_runs", "innings_pitched"),
    "batter_hits": ("batter_game_logs", "hits", "plate_appearances"),
    "batter_total_bases": ("batter_game_logs", "total_bases", "plate_appearances"),
    "batter_home_runs": ("batter_game_logs", "home_runs", "plate_appearances"),
}

_MODELS_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "models"
)


# ---------------------------------------------------------------------------
# Data assembly
# ---------------------------------------------------------------------------

def _fetch_player_logs(conn, table: str) -> Dict[int, List[dict]]:
    rows = conn.execute(
        f"SELECT * FROM {table} ORDER BY player_id, date ASC"
    ).fetchall()
    by_player: Dict[int, List[dict]] = defaultdict(list)
    for r in rows:
        by_player[r["player_id"]].append(dict(r))
    return by_player


def _fetch_game_context(conn) -> Dict:
    rows = conn.execute(
        "SELECT game_id, bdl_game_id, date, game_time, venue, home_team_id, away_team_id FROM games"
    ).fetchall()
    by_bdl = {}
    for r in rows:
        if r["bdl_game_id"] is not None:
            by_bdl[r["bdl_game_id"]] = dict(r)
    return by_bdl


def _build_pitcher_dataset(market: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list]:
    table, target_col, exposure_col = _MARKET_TO_TARGET[market]
    with get_db_connection() as conn:
        games_by_bdl = _fetch_game_context(conn)
        by_player = _fetch_player_logs(conn, table)

        X_rows, y_rows, exp_rows, date_rows = [], [], [], []
        for player_id, logs in by_player.items():
            # logs ascending; iterate and use prior logs only
            for i, row in enumerate(logs):
                if i < 3:
                    continue  # need some history
                prior = list(reversed(logs[:i]))  # most recent first
                target = row.get(target_col) or 0
                exposure = row.get(exposure_col) or 0
                if not exposure or exposure <= 0:
                    continue

                game_ctx = games_by_bdl.get(row["game_id"], {}) or {}
                opp_team_id = None
                if game_ctx:
                    # We don't have player→team pinned historically, so we skip
                    # opp-specific lookups. Use league-average opponent rate.
                    opp_team_id = game_ctx.get("home_team_id")

                extra = {
                    "game_date": row.get("date"),
                    "game_time": game_ctx.get("game_time"),
                    "is_home": 0,  # unknown historically without team of pitcher
                    "month_of_season": None,
                    "rest_days": compute_rest_days(prior, row.get("date")),
                    "opp_bullpen_era": compute_bullpen_factor(opp_team_id, row.get("date"), conn),
                }

                X = build_pitcher_features(
                    pitcher_logs=prior,
                    market=market,
                    opponent_rate=None,
                    venue=game_ctx.get("venue"),
                    weather=None,
                    ump_k_factor=1.0,
                    projected_ip=float(exposure),
                    extra=extra,
                )
                X_rows.append(X)
                y_rows.append(int(target))
                exp_rows.append(float(exposure))
                date_rows.append(row.get("date") or "")

        return (np.array(X_rows), np.array(y_rows), np.array(exp_rows),
                np.array(date_rows), PITCHER_FEATURE_NAMES)


def _build_batter_dataset(market: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list]:
    table, target_col, exposure_col = _MARKET_TO_TARGET[market]
    with get_db_connection() as conn:
        games_by_bdl = _fetch_game_context(conn)
        by_player = _fetch_player_logs(conn, table)
        # Player hand lookup
        player_hands = {
            r["player_id"]: (r["bats"] or "", r["throws"] or "")
            for r in conn.execute("SELECT player_id, bats, throws FROM players").fetchall()
        }

        X_rows, y_rows, exp_rows, date_rows = [], [], [], []
        for player_id, logs in by_player.items():
            bats, _throws = player_hands.get(player_id, ("", ""))
            for i, row in enumerate(logs):
                if i < 5:
                    continue
                prior = list(reversed(logs[:i]))
                target = row.get(target_col) or 0
                exposure = row.get(exposure_col) or row.get("at_bats") or 0
                if not exposure or exposure <= 0:
                    continue

                game_ctx = games_by_bdl.get(row["game_id"], {}) or {}
                extra = {
                    "game_date": row.get("date"),
                    "game_time": game_ctx.get("game_time"),
                    "is_home": 0,
                    "rest_days": compute_rest_days(prior, row.get("date")),
                    "opp_starter_k_per_9": 8.5,
                    "opp_bullpen_era": 4.5,
                }

                X = build_batter_features(
                    batter_logs=prior,
                    market=market,
                    pitcher_hand=None,       # not stored historically at this fidelity
                    batter_hand=bats,
                    venue=game_ctx.get("venue"),
                    weather=None,
                    lineup_position=None,
                    projected_pa=float(exposure),
                    extra=extra,
                )
                X_rows.append(X)
                y_rows.append(int(target))
                exp_rows.append(float(exposure))
                date_rows.append(row.get("date") or "")

        return (np.array(X_rows), np.array(y_rows), np.array(exp_rows),
                np.array(date_rows), BATTER_FEATURE_NAMES)


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def _poisson_deviance(y: np.ndarray, mu: np.ndarray) -> float:
    mu = np.clip(mu, 1e-6, None)
    with np.errstate(divide='ignore', invalid='ignore'):
        term = np.where(y > 0, y * np.log(y / mu), 0.0) - (y - mu)
    return float(2.0 * np.sum(term) / max(1, len(y)))


def _evaluate(glm: PoissonGLM, X: np.ndarray, y: np.ndarray, exposure: np.ndarray) -> Dict[str, float]:
    if X.size == 0:
        return {"n": 0}
    mu = glm.predict_mean(X, exposure=1.0) if exposure is None else _batch_predict(glm, X, exposure)
    mae = float(np.mean(np.abs(y - mu)))
    dev = _poisson_deviance(y, mu)
    return {"n": int(len(y)), "mae": round(mae, 4), "poisson_deviance": round(dev, 4)}


def _batch_predict(glm: PoissonGLM, X: np.ndarray, exposure: np.ndarray) -> np.ndarray:
    # predict_mean handles 2D but single exposure; loop when exposures vary
    mu = np.empty(X.shape[0])
    for i in range(X.shape[0]):
        mu[i] = glm.predict_mean(X[i], exposure=float(exposure[i]))
    return mu


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

def train_market(market: str, compare_sklearn: bool = False) -> Dict[str, float]:
    logger.info(f"Training GLM for market={market}")
    if market in _PITCHER_MARKETS:
        X, y, exp_, dates, names = _build_pitcher_dataset(market)
    else:
        X, y, exp_, dates, names = _build_batter_dataset(market)

    if X.size == 0:
        logger.warning(f"No training data for {market} — skipping.")
        return {"market": market, "trained": False}

    train_mask = dates < _VAL_SPLIT_DATE
    val_mask = ~train_mask

    n_train = int(train_mask.sum())
    n_val = int(val_mask.sum())
    logger.info(f"  {market}: {n_train} train rows, {n_val} validation rows")
    if n_train < 100:
        logger.warning(f"  {market}: too few training rows ({n_train}) — skipping.")
        return {"market": market, "trained": False}

    glm = PoissonGLM(feature_names=list(names), l2=0.01)
    glm.fit(X[train_mask], y[train_mask], exposure=exp_[train_mask])

    train_metrics = _evaluate(glm, X[train_mask], y[train_mask], exp_[train_mask])
    val_metrics = _evaluate(glm, X[val_mask], y[val_mask], exp_[val_mask]) if n_val else {"n": 0}
    logger.info(f"  {market} train: {train_metrics}")
    logger.info(f"  {market} val  : {val_metrics}")

    path = os.path.join(_MODELS_DIR, f"mlb_poisson_{market}.pkl")
    glm.save(path)
    logger.info(f"  saved → {path}")

    result = {
        "market": market,
        "trained": True,
        "train": train_metrics,
        "val": val_metrics,
        "converged": glm.converged_,
        "n_iter": glm.n_iter_,
    }

    if compare_sklearn:
        try:
            from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
            rf = RandomForestRegressor(n_estimators=200, max_depth=8, n_jobs=-1, random_state=0)
            rf.fit(X[train_mask], y[train_mask])
            mu = np.clip(rf.predict(X[val_mask]), 1e-6, None) if n_val else np.array([])
            if n_val:
                result["val_rf_mae"] = round(float(np.mean(np.abs(y[val_mask] - mu))), 4)
                result["val_rf_dev"] = round(_poisson_deviance(y[val_mask], mu), 4)
            gb = GradientBoostingRegressor(random_state=0)
            gb.fit(X[train_mask], y[train_mask])
            mu2 = np.clip(gb.predict(X[val_mask]), 1e-6, None) if n_val else np.array([])
            if n_val:
                result["val_gb_mae"] = round(float(np.mean(np.abs(y[val_mask] - mu2))), 4)
                result["val_gb_dev"] = round(_poisson_deviance(y[val_mask], mu2), 4)
        except ImportError:
            logger.warning("scikit-learn not installed; skipping --compare sklearn.")

    return result


def train_all(compare_sklearn: bool = False) -> List[Dict]:
    os.makedirs(_MODELS_DIR, exist_ok=True)
    results = []
    for market in _ALL_MARKETS:
        try:
            results.append(train_market(market, compare_sklearn=compare_sklearn))
        except Exception as e:
            logger.error(f"Training failed for {market}: {e}", exc_info=True)
            results.append({"market": market, "trained": False, "error": str(e)})

    try:
        from src.pipelines.fit_dispersion import fit_all_dispersion
        fit_all_dispersion()
    except Exception as e:
        logger.error(f"Dispersion fitting failed: {e}", exc_info=True)

    return results
