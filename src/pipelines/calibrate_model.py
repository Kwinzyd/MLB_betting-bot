"""Model calibration pipeline — fits Platt scaling or isotonic regression per market.

Runs nightly at 04:00, after Statcast sync. Reads the calibration_log table
(populated by settle_results after each bet resolves) and fits a post-hoc
calibration transform so that predicted probabilities match empirical win rates.

Why calibration matters:
  If our model says p=0.65 but those bets only win 58% of the time, Kelly sizing
  over-bets by 12%. The Platt/isotonic transform corrects this bias market by
  market so stake sizes are financially correct.

Method selection:
  Both Platt scaling (logistic regression on logit(p) → outcome) and isotonic
  regression are fit and evaluated on a held-out 20% split. Whichever produces
  the lower Brier score is selected and written to calibration_params with
  is_active=1.

  Minimum 200 resolved bets per market before fitting; identity transform used
  otherwise (no change to probabilities).

Usage:
    python main.py calibrate          (via CLI)
    calibrate_all()                   (called from scheduler at 04:00)
"""
from __future__ import annotations

import json
import math
from datetime import datetime, timezone
from typing import Dict, List, Tuple

import numpy as np

from src.data.db import get_db_connection
from src.utils.logging_utils import get_logger

logger = get_logger(__name__)

_MARKETS = (
    "pitcher_strikeouts",
    "pitcher_earned_runs",
    "batter_hits",
    "batter_total_bases",
    "batter_home_runs",
)
_MIN_SAMPLES = 200
_VAL_FRACTION = 0.20


# ---------------------------------------------------------------------------
# Brier score
# ---------------------------------------------------------------------------

def _brier_score(probs: np.ndarray, outcomes: np.ndarray) -> float:
    return float(np.mean((probs - outcomes) ** 2))


# ---------------------------------------------------------------------------
# Platt scaling
# ---------------------------------------------------------------------------

def _fit_platt(p_train: np.ndarray, y_train: np.ndarray) -> Tuple[float, float]:
    """Fit logistic regression on logit(predicted) → actual outcome.

    Returns (a, b) such that calibrated = sigmoid(a * logit(p) + b).
    Uses scipy to avoid adding a sklearn hard dependency for a single use case.
    """
    from scipy.optimize import minimize
    from scipy.special import expit, logit

    p_clipped = np.clip(p_train, 1e-6, 1 - 1e-6)
    logits = logit(p_clipped)

    def nll(params):
        a, b = params
        eta = a * logits + b
        cal = expit(eta)
        cal = np.clip(cal, 1e-9, 1 - 1e-9)
        return -float(np.sum(y_train * np.log(cal) + (1 - y_train) * np.log(1 - cal)))

    res = minimize(nll, [1.0, 0.0], method="L-BFGS-B",
                   options={"maxiter": 200})
    return float(res.x[0]), float(res.x[1])


def _apply_platt(p: np.ndarray, a: float, b: float) -> np.ndarray:
    from scipy.special import expit, logit
    return expit(a * logit(np.clip(p, 1e-6, 1 - 1e-6)) + b)


# ---------------------------------------------------------------------------
# Isotonic regression
# ---------------------------------------------------------------------------

def _fit_isotonic(p_train: np.ndarray, y_train: np.ndarray
                  ) -> Tuple[np.ndarray, np.ndarray]:
    """Fit isotonic regression mapping predicted prob → calibrated prob.

    Returns (X_thresholds, y_values) that define a step function.
    """
    from sklearn.isotonic import IsotonicRegression
    ir = IsotonicRegression(out_of_bounds="clip")
    ir.fit(p_train, y_train)
    return ir.X_thresholds_, ir.y_thresholds_


def _apply_isotonic(p: np.ndarray, x_thresh: np.ndarray, y_vals: np.ndarray) -> np.ndarray:
    return np.interp(p, x_thresh, y_vals)


# ---------------------------------------------------------------------------
# Per-market calibration
# ---------------------------------------------------------------------------

def _calibrate_market(market: str, rows: List[Dict]) -> Dict:
    """Fit calibration for one market. Returns a result dict."""
    probs = np.array([r["predicted_prob"] for r in rows], dtype=np.float64)
    outcomes = np.array([r["actual_outcome"] for r in rows], dtype=np.float64)
    n = len(probs)

    # Shuffle and split 80/20
    rng = np.random.default_rng(seed=42)
    idx = rng.permutation(n)
    split = max(1, int(n * (1 - _VAL_FRACTION)))
    tr_idx, val_idx = idx[:split], idx[split:]

    p_tr, y_tr = probs[tr_idx], outcomes[tr_idx]
    p_val, y_val = probs[val_idx], outcomes[val_idx]

    baseline_brier = _brier_score(p_val, y_val)

    # Fit Platt
    platt_brier = baseline_brier
    platt_params = {"a": 1.0, "b": 0.0}
    try:
        a, b = _fit_platt(p_tr, y_tr)
        p_cal = _apply_platt(p_val, a, b)
        platt_brier = _brier_score(p_cal, y_val)
        platt_params = {"a": a, "b": b}
    except Exception as e:
        logger.warning("  %s: Platt fit failed: %s", market, e)

    # Fit Isotonic (requires sklearn)
    iso_brier = baseline_brier
    iso_params: Dict = {}
    try:
        x_thresh, y_vals = _fit_isotonic(p_tr, y_tr)
        p_iso = _apply_isotonic(p_val, x_thresh, y_vals)
        iso_brier = _brier_score(p_iso, y_val)
        iso_params = {
            "x_thresholds": x_thresh.tolist(),
            "y_thresholds": y_vals.tolist(),
        }
    except Exception as e:
        logger.warning("  %s: Isotonic fit failed: %s", market, e)

    # Choose best method
    if platt_brier <= iso_brier:
        method = "platt"
        params_json = json.dumps(platt_params)
        calibrated_brier = platt_brier
    elif iso_params:
        method = "isotonic"
        params_json = json.dumps(iso_params)
        calibrated_brier = iso_brier
    else:
        method = "identity"
        params_json = json.dumps({"a": 1.0, "b": 0.0})
        calibrated_brier = baseline_brier

    logger.info(
        "  %s: n=%d baseline_brier=%.4f %s_brier=%.4f",
        market, n, baseline_brier, method, calibrated_brier,
    )

    return {
        "market": market,
        "method": method,
        "params_json": params_json,
        "n_samples": n,
        "brier_score": round(baseline_brier, 6),
        "brier_score_calibrated": round(calibrated_brier, 6),
    }


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def calibrate_all() -> List[Dict]:
    """Fit and persist calibration params for all markets."""
    logger.info("calibrate_model: fitting probability calibration per market")
    now = datetime.now(timezone.utc).isoformat()
    results = []

    with get_db_connection() as conn:
        for market in _MARKETS:
            rows = [dict(r) for r in conn.execute(
                """SELECT predicted_prob, actual_outcome
                   FROM calibration_log
                   WHERE market = ?
                   ORDER BY settled_at DESC""",
                (market,),
            ).fetchall()]

            if len(rows) < _MIN_SAMPLES:
                logger.info(
                    "  %s: only %d resolved bets (need %d) — using identity transform",
                    market, len(rows), _MIN_SAMPLES,
                )
                result = {
                    "market": market,
                    "method": "identity",
                    "params_json": json.dumps({"a": 1.0, "b": 0.0}),
                    "n_samples": len(rows),
                    "brier_score": None,
                    "brier_score_calibrated": None,
                }
            else:
                try:
                    result = _calibrate_market(market, rows)
                except Exception as e:
                    logger.error("  %s: calibration failed: %s", market, e, exc_info=True)
                    continue

            # Deactivate previous active row, insert new one
            conn.execute(
                "UPDATE calibration_params SET is_active=0 WHERE market=?", (market,)
            )
            conn.execute(
                """INSERT INTO calibration_params
                   (market, method, params_json, n_samples,
                    brier_score, brier_score_calibrated, fitted_at, is_active)
                   VALUES (?, ?, ?, ?, ?, ?, ?, 1)""",
                (
                    market, result["method"], result["params_json"],
                    result["n_samples"], result["brier_score"],
                    result["brier_score_calibrated"], now,
                ),
            )
            results.append(result)

        conn.commit()

    logger.info("calibrate_model: done (%d markets processed)", len(results))
    return results
