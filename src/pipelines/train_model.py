"""Train a Poisson model per market on historical game logs.

Primary model: LGBMPoissonModel (LightGBM with Poisson objective + Optuna tuning).
Fallback: PoissonGLM when lightgbm is not installed or data is too sparse.

Champion/challenger: after training, the new model is only promoted to champion
if its val MAE beats the current champion's val MAE by ≥ 2%. This prevents a
noisy retraining run from replacing a good model.

Each training run writes a row to model_registry and SHAP importances to
model_feature_importance.

Usage:
    python main.py train
    python main.py train --compare sklearn
"""
from __future__ import annotations

import json
import os
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Tuple

import numpy as np

from src.data.db import get_db_connection
from src.data.feature_builder import (
    BATTER_FEATURE_NAMES, PITCHER_FEATURE_NAMES,
    build_batter_features, build_pitcher_features,
    compute_bullpen_factor, compute_rest_days,
)
from src.models.ml_model import LGBMPoissonModel, PoissonGLM
from src.utils.logging_utils import get_logger

logger = get_logger(__name__)

_PITCHER_MARKETS = ("pitcher_strikeouts", "pitcher_earned_runs")
_BATTER_MARKETS = ("batter_hits", "batter_total_bases", "batter_home_runs")
_ALL_MARKETS = _PITCHER_MARKETS + _BATTER_MARKETS

# Rolling validation: the most recent _VAL_WINDOW_DAYS of available data are
# held out for promotion. A static cutoff ("2025-01-01") silently grew the val
# set to >1.5 seasons and made champion-vs-challenger MAE incomparable across
# runs as new data landed. _INNER_VAL_WINDOW_DAYS carves a SEPARATE early-stop /
# Optuna slice from the training rows so tuning never touches the promotion set.
_VAL_WINDOW_DAYS = 45
_INNER_VAL_FRACTION = 0.15

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


def _rolling_split_date(dates: List[str], window_days: int) -> str | None:
    """Date D such that rows with date >= D are the most recent window_days
    of available data (the promotion holdout). None if no valid dates."""
    valid = [d for d in dates if d]
    if not valid:
        return None
    max_d = datetime.strptime(max(valid)[:10], "%Y-%m-%d")
    return (max_d - timedelta(days=window_days)).strftime("%Y-%m-%d")


def _season_from_date(game_date) -> int | None:
    """Year of a YYYY-MM-DD game date, used to key season-level Statcast.

    Returning the row's own season (not the current year) is what stops the
    feature builder from stamping the current season's Statcast leaderboard
    onto a years-old game — pure future leakage. When no Statcast row exists
    for that season the builder falls back to neutral defaults, so historical
    rows simply carry zeros rather than future stats.
    """
    if not game_date:
        return None
    try:
        return int(str(game_date)[:4])
    except (ValueError, TypeError):
        return None


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
                # Exposure = the ACTUAL innings/PAs the count was generated over.
                # This is the statistically correct Poisson offset: the model
                # fits a RATE (count / exposure) and we multiply by *projected*
                # exposure at serve time. It is NOT leakage — exposure is never a
                # feature (build_*_features ignores projected_ip/pa), so the
                # target can't leak into X. Do not "fix" this to projected
                # exposure; that mis-specifies the likelihood.
                exposure = row.get(exposure_col) or 0
                if not exposure or exposure <= 0:
                    continue

                game_ctx = games_by_bdl.get(row["game_id"], {}) or {}
                # NOTE: historical player→team isn't reliably stored, so the
                # opponent-rate feature can't be built leak-free at train time
                # (it stays at a league-average constant here while it varies at
                # inference). Fixing it needs a historical team-assignment table;
                # tracked as a known train/serve-skew limitation.
                opp_team_id = None
                if game_ctx:
                    opp_team_id = game_ctx.get("home_team_id")

                extra = {
                    "game_date": row.get("date"),
                    "game_time": game_ctx.get("game_time"),
                    "is_home": 0,  # unknown historically without team of pitcher
                    "month_of_season": None,
                    "season": _season_from_date(row.get("date")),
                    "rest_days": compute_rest_days(prior, row.get("date")),
                    "opp_bullpen_era": compute_bullpen_factor(opp_team_id, row.get("date"), conn),
                    "db": conn,
                    "pitcher_id": player_id,
                    "opp_team_id": opp_team_id,
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
                    "season": _season_from_date(row.get("date")),
                    "batter_id": player_id,
                    "db": conn,
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

# ---------------------------------------------------------------------------
# Optuna hyperparameter search (LightGBM)
# ---------------------------------------------------------------------------

def _tune_lgbm(X_tr: np.ndarray, y_tr: np.ndarray, exp_tr: np.ndarray,
               X_val: np.ndarray, y_val: np.ndarray, exp_val: np.ndarray,
               feature_names: List[str], n_trials: int = 50) -> Dict:
    """Return best LightGBM hyperparams via Optuna. Falls back to defaults on error."""
    try:
        import optuna
        optuna.logging.set_verbosity(optuna.logging.WARNING)
    except ImportError:
        logger.info("optuna not installed — using default LGBM hyperparams")
        return {}

    def objective(trial):
        params = {
            "n_estimators":    trial.suggest_int("n_estimators", 100, 500),
            "max_depth":       trial.suggest_int("max_depth", 3, 8),
            "num_leaves":      trial.suggest_int("num_leaves", 15, 63),
            "learning_rate":   trial.suggest_float("learning_rate", 0.01, 0.15, log=True),
            "min_child_samples": trial.suggest_int("min_child_samples", 10, 50),
            "reg_lambda":      trial.suggest_float("reg_lambda", 0.1, 10.0, log=True),
        }
        m = LGBMPoissonModel(feature_names=feature_names, **params)
        try:
            m.fit(X_tr, y_tr, exp_tr, X_val, y_val, exp_val)
            mu = m.predict_mean(X_val, exposure=exp_val)
            return float(np.mean(np.abs(y_val - mu)))
        except Exception:
            return 1e9

    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    return study.best_params


# ---------------------------------------------------------------------------
# Model registry helpers
# ---------------------------------------------------------------------------

def _get_champion_val_mae(market: str) -> float:
    """Return the val_mae of the current champion, or inf if none exists."""
    try:
        with get_db_connection() as conn:
            row = conn.execute(
                "SELECT val_mae FROM model_registry WHERE market=? AND is_champion=1 ORDER BY trained_at DESC LIMIT 1",
                (market,),
            ).fetchone()
        return float(row["val_mae"]) if row and row["val_mae"] is not None else float("inf")
    except Exception:
        return float("inf")


def _register_model(market: str, model_type: str, version: str,
                    train_metrics: Dict, val_metrics: Dict,
                    feature_names: List[str], is_champion: bool) -> None:
    """Write a row to model_registry; demote previous champion if promoting new one."""
    with get_db_connection() as conn:
        if is_champion:
            conn.execute(
                "UPDATE model_registry SET is_champion=0 WHERE market=?", (market,)
            )
        conn.execute(
            """INSERT INTO model_registry
               (market, model_type, version, train_mae, val_mae, poisson_deviance,
                n_train, n_val, feature_list, is_champion, trained_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                market, model_type, version,
                train_metrics.get("mae"), val_metrics.get("mae"),
                val_metrics.get("poisson_deviance"),
                train_metrics.get("n"), val_metrics.get("n"),
                json.dumps(feature_names),
                1 if is_champion else 0,
                version,
            ),
        )
        conn.commit()


def _write_training_baseline(market: str, version: str, model,
                              X_val: np.ndarray, exp_val: np.ndarray) -> None:
    """Store val-set predicted-mean distribution stats for drift monitoring."""
    if X_val.size == 0:
        return
    try:
        mu = np.empty(X_val.shape[0])
        for i in range(X_val.shape[0]):
            mu[i] = model.predict_mean(X_val[i], exposure=float(exp_val[i]))
        mu = np.clip(mu, 0.0, None)
        stats = (
            float(np.mean(mu)),
            float(np.std(mu)),
            float(np.percentile(mu, 10)),
            float(np.percentile(mu, 50)),
            float(np.percentile(mu, 90)),
        )
        with get_db_connection() as conn:
            conn.execute(
                """INSERT OR REPLACE INTO model_training_baseline
                   (market, model_version, feature, mean, std, p10, p50, p90)
                   VALUES (?, ?, 'val_mu', ?, ?, ?, ?, ?)""",
                (market, version) + stats,
            )
            conn.commit()
        logger.info("  %s: training baseline written (mu mean=%.3f std=%.3f)", market, stats[0], stats[1])
    except Exception as e:
        logger.warning("  %s: could not write training baseline: %s", market, e)


def _save_shap_importances(market: str, version: str, model, X_sample: np.ndarray,
                           feature_names: List[str]) -> None:
    """Compute SHAP mean |value| per feature and write to model_feature_importance."""
    try:
        shap_vals = model.shap_values(X_sample[:min(500, len(X_sample))])
        mean_abs = np.abs(shap_vals).mean(axis=0)
        ranked = sorted(enumerate(mean_abs), key=lambda x: x[1], reverse=True)
        now = datetime.now(timezone.utc).isoformat()
        with get_db_connection() as conn:
            for rank, (idx, importance) in enumerate(ranked, 1):
                fname = feature_names[idx] if idx < len(feature_names) else f"f{idx}"
                conn.execute(
                    """INSERT INTO model_feature_importance
                       (market, model_version, feature, shap_mean_abs, rank, computed_at)
                       VALUES (?, ?, ?, ?, ?, ?)""",
                    (market, version, fname, float(importance), rank, now),
                )
            conn.commit()
        logger.info("  %s: SHAP importances written (%d features)", market, len(ranked))
    except Exception as e:
        logger.warning("  %s: SHAP computation failed: %s", market, e)


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

def train_market(market: str, compare_sklearn: bool = False) -> Dict:
    logger.info("Training model for market=%s", market)
    if market in _PITCHER_MARKETS:
        X, y, exp_, dates, names = _build_pitcher_dataset(market)
    else:
        X, y, exp_, dates, names = _build_batter_dataset(market)

    if X.size == 0:
        logger.warning("  %s: no training data — skipping.", market)
        return {"market": market, "trained": False}

    # Drop rows without a usable date — the rolling split and any time-ordered
    # holdout are undefined for them.
    date_ok = dates != ''
    X, y, exp_, dates = X[date_ok], y[date_ok], exp_[date_ok], dates[date_ok]
    if X.size == 0:
        logger.warning("  %s: no dated training rows — skipping.", market)
        return {"market": market, "trained": False}

    split_date = _rolling_split_date(dates.tolist(), _VAL_WINDOW_DAYS)
    if split_date is None:
        logger.warning("  %s: cannot compute rolling split — skipping.", market)
        return {"market": market, "trained": False}

    train_mask = dates < split_date
    val_mask = ~train_mask
    n_train, n_val = int(train_mask.sum()), int(val_mask.sum())
    logger.info("  %s: %d train rows, %d val rows (val >= %s)",
                market, n_train, n_val, split_date)

    if n_train < 100:
        logger.warning("  %s: too few training rows (%d) — skipping.", market, n_train)
        return {"market": market, "trained": False}

    version = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    X_tr, y_tr, exp_tr = X[train_mask], y[train_mask], exp_[train_mask]
    X_val = X[val_mask] if n_val else X_tr[:1]
    y_val = y[val_mask] if n_val else y_tr[:1]
    exp_val = exp_[val_mask] if n_val else exp_tr[:1]

    # Inner early-stop / Optuna slice carved from the TAIL of the training rows
    # (most recent dates), kept strictly separate from the promotion holdout so
    # tuning and early stopping never see val_mask.
    train_idx = np.where(train_mask)[0]
    order = np.argsort(dates[train_idx], kind="stable")
    train_idx = train_idx[order]
    n_es = max(1, int(len(train_idx) * _INNER_VAL_FRACTION))
    fit_idx, es_idx = train_idx[:-n_es], train_idx[-n_es:]
    if len(fit_idx) < 50:   # degenerate slice — fall back to fitting on all train
        fit_idx, es_idx = train_idx, train_idx
    X_fit, y_fit, exp_fit = X[fit_idx], y[fit_idx], exp_[fit_idx]
    X_es, y_es, exp_es = X[es_idx], y[es_idx], exp_[es_idx]

    # ---- Try LightGBM (champion path) ----
    model = None
    model_type = "glm"
    if LGBMPoissonModel.available():
        try:
            best_params = _tune_lgbm(X_fit, y_fit, exp_fit, X_es, y_es, exp_es,
                                     list(names), n_trials=50)
            lgbm = LGBMPoissonModel(feature_names=list(names), **best_params)
            lgbm.fit(X_fit, y_fit, exp_fit, X_es, y_es, exp_es)
            model = lgbm
            model_type = "lgbm"
            logger.info("  %s: LightGBM trained (%d trees)", market, lgbm.n_iter_)
        except Exception as e:
            logger.warning("  %s: LightGBM training failed (%s) — falling back to GLM", market, e)

    # ---- Fallback: GLM (no early stopping → fit on all training rows) ----
    if model is None:
        glm = PoissonGLM(feature_names=list(names), l2=0.01)
        glm.fit(X_tr, y_tr, exposure=exp_tr)
        model = glm

    train_metrics = _evaluate(model, X_tr, y_tr, exp_tr)
    val_metrics = _evaluate(model, X_val, y_val, exp_val) if n_val else {"n": 0}
    logger.info("  %s %s train=%s val=%s", market, model_type, train_metrics, val_metrics)

    # ---- Champion/challenger ----
    current_best_mae = _get_champion_val_mae(market)
    new_mae = val_metrics.get("mae") or float("inf")
    # Promote if: no existing champion OR new model is ≥2% better
    is_champion = (new_mae < current_best_mae * 0.98) or (current_best_mae == float("inf"))
    if not is_champion:
        logger.info("  %s: new model (mae=%.4f) does not beat champion (mae=%.4f) by 2%% — not promoting",
                    market, new_mae, current_best_mae)

    # ---- Save model file ----
    suffix = ".pkl"
    path = os.path.join(_MODELS_DIR, f"mlb_poisson_{market}{suffix}")
    if is_champion:
        model.save(path)
        logger.info("  %s: champion saved → %s", market, path)
    else:
        challenger_path = os.path.join(_MODELS_DIR, f"mlb_poisson_{market}_challenger_{version}.pkl")
        model.save(challenger_path)
        logger.info("  %s: challenger saved → %s", market, challenger_path)

    # ---- Registry + SHAP + training baseline ----
    _register_model(market, model_type, version, train_metrics, val_metrics,
                    list(names), is_champion)
    if model_type == "lgbm" and is_champion:
        _save_shap_importances(market, version, model, X_tr, list(names))
    if is_champion and n_val:
        _write_training_baseline(market, version, model, X_val, exp_val)

    result = {
        "market": market,
        "trained": True,
        "model_type": model_type,
        "is_champion": is_champion,
        "train": train_metrics,
        "val": val_metrics,
        "converged": model.converged_,
        "n_iter": model.n_iter_,
        "version": version,
    }

    if compare_sklearn:
        try:
            from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
            rf = RandomForestRegressor(n_estimators=200, max_depth=8, n_jobs=-1, random_state=0)
            rf.fit(X_tr, y_tr)
            if n_val:
                mu = np.clip(rf.predict(X_val), 1e-6, None)
                result["val_rf_mae"] = round(float(np.mean(np.abs(y_val - mu))), 4)
            gb = GradientBoostingRegressor(random_state=0)
            gb.fit(X_tr, y_tr)
            if n_val:
                mu2 = np.clip(gb.predict(X_val), 1e-6, None)
                result["val_gb_mae"] = round(float(np.mean(np.abs(y_val - mu2))), 4)
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
            logger.error("Training failed for %s: %s", market, e, exc_info=True)
            results.append({"market": market, "trained": False, "error": str(e)})

    try:
        from src.pipelines.fit_dispersion import fit_all_dispersion
        fit_all_dispersion()
    except Exception as e:
        logger.error("Dispersion fitting failed: %s", e, exc_info=True)

    return results
