"""Weekly model health check: MAE drift and prediction distribution shift.

Runs Sunday 05:30 via scheduler. For each market:
1. Pulls last 30 days of calibration_log → rolling mean(|predicted_prob - actual|) = mae_window
2. Compares against champion val_mae from model_registry → drift_score
3. Pulls last 30 days of projections.projected_mean → builds a 10-bin histogram
4. Compares against training baseline histogram (model_training_baseline) → PSI
5. If drift_score > DRIFT_SCORE_THRESHOLD OR psi_score > PSI_THRESHOLD:
   - Triggers train_all() immediately
   - Sends Telegram alert

PSI formula (Population Stability Index):
   PSI = Σ (actual_pct - expected_pct) × ln(actual_pct / expected_pct)
   PSI < 0.10 → stable; 0.10–0.25 → moderate shift; > 0.25 → large shift (retrain)
"""
from __future__ import annotations

import math
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional, Tuple

from src.data.db import get_db_connection
from src.utils.logging_utils import get_logger
from src.utils.time_utils import get_utc_now_iso

logger = get_logger(__name__)

_MARKETS = (
    "pitcher_strikeouts", "pitcher_earned_runs",
    "batter_hits", "batter_total_bases", "batter_home_runs",
)
_DRIFT_SCORE_THRESHOLD = 0.15   # 15% MAE degradation vs baseline
_PSI_THRESHOLD = 0.20
_MIN_SAMPLES = 30               # minimum calibration rows required to evaluate
_N_HIST_BINS = 10


def _histogram(values: List[float], lo: float, hi: float) -> List[float]:
    """Return a 10-bin histogram (fractions) over [lo, hi]. Adds small epsilon so no bin is zero."""
    eps = 1e-4
    bin_width = (hi - lo) / _N_HIST_BINS if hi > lo else 1.0
    counts = [0] * _N_HIST_BINS
    for v in values:
        idx = int((v - lo) / bin_width)
        idx = max(0, min(_N_HIST_BINS - 1, idx))
        counts[idx] += 1
    total = sum(counts) + eps * _N_HIST_BINS
    return [(c + eps) / total for c in counts]


def _psi(expected: List[float], actual: List[float]) -> float:
    """Compute PSI between two normalised histograms of equal length."""
    psi = 0.0
    for e, a in zip(expected, actual):
        if e > 0 and a > 0:
            psi += (a - e) * math.log(a / e)
    return round(psi, 6)


def _get_champion(conn, market: str) -> Optional[dict]:
    row = conn.execute(
        "SELECT version, val_mae FROM model_registry "
        "WHERE market=? AND is_champion=1 ORDER BY trained_at DESC LIMIT 1",
        (market,),
    ).fetchone()
    return dict(row) if row else None


def _get_baseline_stats(conn, market: str, version: str) -> Optional[dict]:
    row = conn.execute(
        "SELECT mean, std, p10, p50, p90 FROM model_training_baseline "
        "WHERE market=? AND model_version=? AND feature='val_mu'",
        (market, version),
    ).fetchone()
    return dict(row) if row else None


def _rolling_mae(conn, market: str, window_start: str) -> Tuple[float, int]:
    """MAE = mean(|predicted_prob - actual_outcome|) from calibration_log."""
    rows = conn.execute(
        "SELECT predicted_prob, actual_outcome FROM calibration_log "
        "WHERE market=? AND settled_at >= ?",
        (market, window_start),
    ).fetchall()
    if not rows:
        return 0.0, 0
    total = sum(abs(r["predicted_prob"] - r["actual_outcome"]) for r in rows)
    return round(total / len(rows), 6), len(rows)


def _recent_projected_means(conn, market: str, window_start: str) -> List[float]:
    rows = conn.execute(
        "SELECT projected_mean FROM projections "
        "WHERE market=? AND timestamp >= ? AND projected_mean IS NOT NULL",
        (market, window_start),
    ).fetchall()
    return [float(r["projected_mean"]) for r in rows]


def monitor_drift() -> List[Dict]:
    """Run drift check for all markets. Returns a list of result dicts."""
    now = datetime.now(timezone.utc)
    window_start = (now - timedelta(days=30)).isoformat()
    window_end = now.isoformat()
    logged_at = get_utc_now_iso()

    results = []
    retrain_markets = []

    with get_db_connection() as conn:
        for market in _MARKETS:
            champion = _get_champion(conn, market)
            if not champion:
                logger.debug("monitor_drift: no champion for %s — skipping", market)
                continue

            version = champion["version"]
            mae_baseline = float(champion["val_mae"]) if champion["val_mae"] else None

            mae_window, n_samples = _rolling_mae(conn, market, window_start)

            if n_samples < _MIN_SAMPLES:
                logger.debug(
                    "monitor_drift: %s only %d samples in last 30 days — skipping",
                    market, n_samples,
                )
                continue

            drift_score = None
            if mae_baseline and mae_baseline > 0:
                drift_score = round((mae_window - mae_baseline) / mae_baseline, 6)

            # PSI on projected_mean distribution
            psi_score = None
            baseline = _get_baseline_stats(conn, market, version)
            if baseline:
                recent_mus = _recent_projected_means(conn, market, window_start)
                if len(recent_mus) >= _MIN_SAMPLES:
                    lo = float(baseline["p10"] or 0.0)
                    hi = float(baseline["p90"] or 10.0)
                    if hi <= lo:
                        hi = lo + 1.0
                    # Baseline histogram: derive from stored stats
                    # Simulate expected distribution from [p10, p90] uniform approximation
                    baseline_vals = _simulate_baseline_vals(baseline)
                    expected_hist = _histogram(baseline_vals, lo, hi)
                    actual_hist = _histogram(recent_mus, lo, hi)
                    psi_score = _psi(expected_hist, actual_hist)

            should_retrain = (
                (drift_score is not None and drift_score > _DRIFT_SCORE_THRESHOLD)
                or (psi_score is not None and psi_score > _PSI_THRESHOLD)
            )

            if should_retrain:
                retrain_markets.append(market)

            conn.execute(
                """INSERT INTO model_drift_log
                   (market, model_version, window_start, window_end,
                    mae_this_window, mae_baseline, drift_score, psi_score,
                    retrain_triggered, logged_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    market, version, window_start, window_end,
                    mae_window, mae_baseline, drift_score, psi_score,
                    1 if should_retrain else 0, logged_at,
                ),
            )

            result = {
                "market": market,
                "version": version,
                "n_samples": n_samples,
                "mae_window": mae_window,
                "mae_baseline": mae_baseline,
                "drift_score": drift_score,
                "psi_score": psi_score,
                "retrain_triggered": should_retrain,
            }
            results.append(result)
            logger.info(
                "monitor_drift: %s drift_score=%s psi=%s n=%d retrain=%s",
                market,
                f"{drift_score:.3f}" if drift_score is not None else "n/a",
                f"{psi_score:.3f}" if psi_score is not None else "n/a",
                n_samples,
                should_retrain,
            )

        conn.commit()

    if retrain_markets:
        _trigger_retrain(retrain_markets)

    return results


def _simulate_baseline_vals(baseline: dict, n: int = 200) -> List[float]:
    """Approximate a sample from the training val distribution using stored percentiles.
    Uses a piecewise-linear CDF from p10/p50/p90 to generate synthetic samples.
    """
    p10 = float(baseline["p10"] or 0.0)
    p50 = float(baseline["p50"] or p10)
    p90 = float(baseline["p90"] or p50 + 1.0)

    # Three segments: [min, p10], [p10, p50], [p50, p90], [p90, max]
    # Approximate by sampling uniformly within each segment weighted by density
    vals = []
    seg_n = n // 3
    for _ in range(seg_n):
        vals.append(p10 + (p50 - p10) * (_ / max(1, seg_n - 1)))
    for _ in range(seg_n):
        vals.append(p50 + (p90 - p50) * (_ / max(1, seg_n - 1)))
    while len(vals) < n:
        vals.append(p50)
    return vals[:n]


def _trigger_retrain(markets: List[str]) -> None:
    """Fire train_all() in a thread executor and send a Telegram alert."""
    logger.warning("monitor_drift: triggering retrain for markets: %s", markets)
    try:
        from src.clients.telegram_bot import TelegramClient
        market_str = ", ".join(markets)
        msg = (
            "⚠️ <b>Model Drift Detected — Retrain Triggered</b>\n"
            f"Markets: <code>{market_str}</code>\n"
            "Drift score &gt; 15% or PSI &gt; 0.20 threshold exceeded."
        )
        TelegramClient().send_message_sync(msg)
    except Exception as e:
        logger.warning("monitor_drift: Telegram alert failed: %s", e)

    try:
        from src.pipelines.train_model import train_all
        results = train_all()
        promoted = [r["market"] for r in results if r.get("is_champion")]
        logger.info("monitor_drift: retrain complete; promoted=%s", promoted)
    except Exception as e:
        logger.error("monitor_drift: retrain failed: %s", e, exc_info=True)
