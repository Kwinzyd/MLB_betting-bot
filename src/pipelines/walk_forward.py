"""Walk-forward backtest for the Poisson GLM models.

The static pre-2025/post-2025 split in train_model.py answers "does the model
generalize to a future season". It does NOT answer the question that matters
for live betting: can the model adapt to mid-season meta shifts (deadened
ball, weather warming through July, bullpen-usage trends)?

Walk-forward simulates production: pick a train window, fit, predict the next
chunk, slide forward, repeat. Per-fold metrics let us see drift over time —
e.g. if MAE creeps up in August, the model is failing to absorb a late-season
regime change and live bets in that window would be miscalibrated.

Usage (via CLI):
    python main.py walkforward
    python main.py walkforward --train-window 60 --step 7 --mode expanding
    python main.py walkforward --market pitcher_strikeouts --market batter_hits
"""
from __future__ import annotations

from collections import defaultdict
from datetime import datetime, timedelta, timezone

UTC = timezone.utc
from typing import Dict, Iterator, List, Tuple

import numpy as np

from src.data.db import get_db_connection
from src.models.ml_model import LGBMPoissonModel, PoissonGLM
from src.pipelines.train_model import (
    _ALL_MARKETS,
    _BATTER_MARKETS,
    _PITCHER_MARKETS,
    _batch_predict,
    _build_batter_dataset,
    _build_pitcher_dataset,
    _evaluate,
)
from src.utils.logging_utils import get_logger

logger = get_logger(__name__)

_DATE_FMT = "%Y-%m-%d"


def generate_folds(
    sorted_unique_dates: List[str],
    train_window_days: int,
    step_days: int,
    mode: str = "sliding",
) -> Iterator[Tuple[str, str, str, str]]:
    """Yield (train_start, train_end, test_start, test_end) date strings.

    Bounds are half-open: [train_start, train_end) and [test_start, test_end).
    For sliding, train_start = test_start - train_window_days.
    For expanding, train_start = sorted_unique_dates[0].
    """
    if mode not in ("sliding", "expanding"):
        raise ValueError(f"mode must be 'sliding' or 'expanding', got {mode!r}")
    if not sorted_unique_dates:
        return

    parsed = [datetime.strptime(d, _DATE_FMT) for d in sorted_unique_dates]
    start = parsed[0]
    end = parsed[-1]

    cursor = start + timedelta(days=train_window_days)
    while cursor <= end:
        test_start = cursor
        test_end = cursor + timedelta(days=step_days)
        train_start = start if mode == "expanding" else test_start - timedelta(days=train_window_days)
        train_end = test_start
        yield (
            train_start.strftime(_DATE_FMT),
            train_end.strftime(_DATE_FMT),
            test_start.strftime(_DATE_FMT),
            test_end.strftime(_DATE_FMT),
        )
        cursor += timedelta(days=step_days)


def _financial_metrics(folds: List[Dict]) -> Dict:
    """Sharpe ratio, Sortino ratio, and max drawdown from fold-level MAE improvements.

    Per-fold "return" = naive_mae - model_mae, where naive_mae = mean(y_test).
    Positive → model beats the null predictor in that fold.
    Sharpe  = mean(r) / std(r)  (higher is better; >0.5 is useful)
    Sortino = mean(r) / std(downside r)  (only penalises below-zero folds)
    Max drawdown = largest peak-to-trough drop in the cumulative return series.
    """
    returns = []
    for f in folds:
        if f.get("n", 0) == 0:
            continue
        # naive baseline: predicting the fold mean has MAE ≈ std(y), but we
        # approximate it as the fold MAE of a zero-information model.
        # A simple proxy: naive_mae = f.get("mean_y", f["mae"])
        # We store mean_y in the fold dict below; fall back to f["mae"] * 1.1.
        naive = f.get("naive_mae", f["mae"] * 1.1)
        returns.append(naive - f["mae"])

    if len(returns) < 2:
        return {"sharpe_ratio": None, "sortino_ratio": None, "max_drawdown": None}

    arr = np.array(returns)
    mean_r = float(np.mean(arr))
    std_r = float(np.std(arr, ddof=1))

    sharpe = round(mean_r / std_r, 4) if std_r > 0 else None

    downside = arr[arr < 0]
    std_d = float(np.std(downside, ddof=1)) if len(downside) > 1 else 0.0
    sortino = round(mean_r / std_d, 4) if std_d > 0 else None

    cum = np.cumsum(arr)
    running_max = np.maximum.accumulate(cum)
    drawdowns = running_max - cum
    max_dd = round(float(np.max(drawdowns)), 6) if len(drawdowns) else None

    return {
        "sharpe_ratio": sharpe,
        "sortino_ratio": sortino,
        "max_drawdown": max_dd,
    }


def walk_forward_market(
    market: str,
    train_window_days: int = 45,
    step_days: int = 7,
    mode: str = "sliding",
    min_train_rows: int = 200,
    model_type: str = "glm",
) -> Dict:
    """Run walk-forward backtest on a single market. Returns per-fold + aggregate metrics."""
    logger.info(
        f"Walk-forward {market}: window={train_window_days}d step={step_days}d mode={mode}"
    )
    if market in _PITCHER_MARKETS:
        X, y, exp_, dates, names = _build_pitcher_dataset(market)
    elif market in _BATTER_MARKETS:
        X, y, exp_, dates, names = _build_batter_dataset(market)
    else:
        raise ValueError(f"Unknown market: {market}")

    if X.size == 0:
        logger.warning(f"  {market}: no data, skipping.")
        return {"market": market, "ran": False, "reason": "no_data"}

    # Sort everything by date so date-mask slicing is consistent
    sort_idx = np.argsort(dates)
    X, y, exp_, dates = X[sort_idx], y[sort_idx], exp_[sort_idx], dates[sort_idx]

    valid_dates = [d for d in dates.tolist() if d]
    if not valid_dates:
        return {"market": market, "ran": False, "reason": "no_dates"}

    sorted_unique = sorted(set(valid_dates))

    folds = []
    for train_start, train_end, test_start, test_end in generate_folds(
        sorted_unique, train_window_days, step_days, mode
    ):
        train_mask = (dates >= train_start) & (dates < train_end)
        test_mask = (dates >= test_start) & (dates < test_end)
        n_train = int(train_mask.sum())
        n_test = int(test_mask.sum())

        if n_train < min_train_rows or n_test == 0:
            continue

        # Fit per-fold model
        try:
            if model_type == "lgbm" and LGBMPoissonModel.available():
                mdl = LGBMPoissonModel(feature_names=list(names))
                # Early stopping must NOT use the test fold — that selects the
                # number of trees on the very set we then score, leaking it.
                # Carve an early-stop slice from the TAIL of the training window
                # (its most recent dates) instead.
                tr_idx = np.where(train_mask)[0]
                tr_idx = tr_idx[np.argsort(dates[tr_idx], kind="stable")]
                n_es = max(1, int(len(tr_idx) * 0.15))
                fit_idx, es_idx = tr_idx[:-n_es], tr_idx[-n_es:]
                if len(fit_idx) < 50:
                    fit_idx, es_idx = tr_idx, tr_idx
                mdl.fit(X[fit_idx], y[fit_idx], exp_[fit_idx],
                        X[es_idx], y[es_idx], exp_[es_idx])
            else:
                mdl = PoissonGLM(feature_names=list(names), l2=0.01)
                mdl.fit(X[train_mask], y[train_mask], exposure=exp_[train_mask])
        except Exception as e:
            logger.warning(f"  fit failed for fold test_start={test_start}: {e}")
            continue

        metrics = _evaluate(mdl, X[test_mask], y[test_mask], exp_[test_mask])
        y_test = y[test_mask].astype(float)
        mu_test = _batch_predict(mdl, X[test_mask], exp_[test_mask])
        mean_mu = float(np.mean(mu_test)) if len(mu_test) > 0 else 1.0
        var_resid = float(np.var(y_test - mu_test, ddof=1)) if len(mu_test) > 1 else 0.0
        var_ratio = round(var_resid / max(mean_mu, 1e-6), 4)
        # naive_mae: predicting the training mean for every test row
        naive_pred = float(np.mean(y[train_mask]))
        naive_mae = round(float(np.mean(np.abs(y_test - naive_pred))), 4)
        metrics.update({
            "test_start": test_start,
            "test_end": test_end,
            "n_train": n_train,
            "converged": getattr(mdl, "converged_", True),
            "var_ratio": var_ratio,
            "naive_mae": naive_mae,
        })
        folds.append(metrics)

    if not folds:
        return {"market": market, "ran": False, "reason": "no_eligible_folds"}

    # Aggregate: weighted MAE by n, plus monthly drift bins
    total_n = sum(f["n"] for f in folds)
    weighted_mae = sum(f["mae"] * f["n"] for f in folds) / max(1, total_n)
    weighted_dev = sum(f["poisson_deviance"] * f["n"] for f in folds) / max(1, total_n)
    weighted_var_ratio = sum(f["var_ratio"] * f["n"] for f in folds) / max(1, total_n)

    by_month = defaultdict(lambda: {"n": 0, "mae_num": 0.0, "dev_num": 0.0, "vr_num": 0.0})
    for f in folds:
        ym = f["test_start"][:7]  # YYYY-MM
        by_month[ym]["n"] += f["n"]
        by_month[ym]["mae_num"] += f["mae"] * f["n"]
        by_month[ym]["dev_num"] += f["poisson_deviance"] * f["n"]
        by_month[ym]["vr_num"] += f["var_ratio"] * f["n"]
    monthly = []
    for ym in sorted(by_month):
        agg = by_month[ym]
        monthly.append({
            "month": ym,
            "n": agg["n"],
            "mae": round(agg["mae_num"] / max(1, agg["n"]), 4),
            "poisson_deviance": round(agg["dev_num"] / max(1, agg["n"]), 4),
            "var_ratio": round(agg["vr_num"] / max(1, agg["n"]), 4),
        })

    financial = _financial_metrics(folds)

    result = {
        "market": market,
        "ran": True,
        "mode": mode,
        "train_window_days": train_window_days,
        "step_days": step_days,
        "n_folds": len(folds),
        "weighted_mae": round(weighted_mae, 4),
        "weighted_poisson_deviance": round(weighted_dev, 4),
        "weighted_var_ratio": round(weighted_var_ratio, 4),
        **financial,
        "monthly": monthly,
        "folds": folds,
    }
    _persist_result(result, model_type)
    return result


def _persist_result(result: Dict, model_type: str) -> None:
    """Write aggregate walk-forward metrics to walk_forward_results for trend tracking."""
    try:
        run_at = datetime.now(UTC).isoformat(timespec="seconds")
        with get_db_connection() as conn:
            conn.execute(
                """
                INSERT INTO walk_forward_results
                    (market, model_type, mode, train_window_days, step_days, n_folds,
                     weighted_mae, weighted_poisson_deviance, weighted_var_ratio,
                     sharpe_ratio, sortino_ratio, max_drawdown, run_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    result["market"], model_type, result["mode"],
                    result["train_window_days"], result["step_days"], result.get("n_folds"),
                    result.get("weighted_mae"), result.get("weighted_poisson_deviance"),
                    result.get("weighted_var_ratio"), result.get("sharpe_ratio"),
                    result.get("sortino_ratio"), result.get("max_drawdown"),
                    run_at,
                ),
            )
            conn.commit()
    except Exception as e:
        logger.warning("walk_forward: failed to persist result for %s: %s", result["market"], e)


def walk_forward_all(
    markets: List[str] = None,
    train_window_days: int = 45,
    step_days: int = 7,
    mode: str = "sliding",
    model_type: str = "glm",
) -> List[Dict]:
    targets = list(markets) if markets else list(_ALL_MARKETS)
    results = []
    for m in targets:
        try:
            results.append(walk_forward_market(
                m, train_window_days=train_window_days,
                step_days=step_days, mode=mode, model_type=model_type,
            ))
        except Exception as e:
            logger.error(f"Walk-forward failed for {m}: {e}", exc_info=True)
            results.append({"market": m, "ran": False, "error": str(e)})
    return results


def compare_walk_forward_market(
    market: str,
    train_window_days: int = 45,
    step_days: int = 7,
    mode: str = "sliding",
) -> Dict:
    """Run walk-forward for both GLM and LGBM on the same market and return a side-by-side comparison.

    Returns a dict with keys 'glm' and 'lgbm', each holding the full result
    from walk_forward_market, plus a 'delta' section showing the MAE / Sharpe
    difference (lgbm - glm; negative delta_mae = LGBM is better).
    """
    glm_result = walk_forward_market(
        market, train_window_days=train_window_days,
        step_days=step_days, mode=mode, model_type="glm",
    )
    lgbm_result = walk_forward_market(
        market, train_window_days=train_window_days,
        step_days=step_days, mode=mode, model_type="lgbm",
    )

    delta: Dict = {}
    if glm_result.get("ran") and lgbm_result.get("ran"):
        for key in ("weighted_mae", "weighted_poisson_deviance", "sharpe_ratio", "max_drawdown"):
            g = glm_result.get(key)
            l = lgbm_result.get(key)
            if g is not None and l is not None:
                delta[f"delta_{key}"] = round(l - g, 4)

    return {"market": market, "glm": glm_result, "lgbm": lgbm_result, "delta": delta}


def print_walk_forward_report(results: List[Dict]) -> None:
    """Human-readable summary: per-market headline, financial metrics, monthly drift table."""
    print("\n=== Walk-Forward Backtest ===")
    for r in results:
        market = r.get("market")
        if not r.get("ran"):
            reason = r.get("reason") or r.get("error") or "unknown"
            print(f"\n[{market}]  SKIPPED ({reason})")
            continue
        print(
            f"\n[{market}]  mode={r['mode']}  window={r['train_window_days']}d  "
            f"step={r['step_days']}d  folds={r['n_folds']}"
        )
        print(f"  weighted MAE      : {r['weighted_mae']}")
        print(f"  weighted deviance : {r['weighted_poisson_deviance']}")
        print(f"  weighted var_ratio: {r.get('weighted_var_ratio', 'N/A')}")
        sharpe = r.get('sharpe_ratio')
        sortino = r.get('sortino_ratio')
        max_dd = r.get('max_drawdown')
        print(f"  Sharpe ratio      : {sharpe if sharpe is not None else 'N/A'}")
        print(f"  Sortino ratio     : {sortino if sortino is not None else 'N/A'}")
        print(f"  Max drawdown      : {max_dd if max_dd is not None else 'N/A'}")
        if r.get("monthly"):
            print(f"  monthly drift:")
            print(f"    {'month':<9} {'n':>6} {'mae':>8} {'deviance':>10} {'var_ratio':>10}")
            for m in r["monthly"]:
                print(
                    f"    {m['month']:<9} {m['n']:>6} {m['mae']:>8.4f} "
                    f"{m['poisson_deviance']:>10.4f} {m.get('var_ratio', 0):>10.4f}"
                )


def print_compare_report(comparison: Dict) -> None:
    """Print a side-by-side GLM vs LGBM walk-forward comparison."""
    market = comparison["market"]
    print(f"\n=== GLM vs LGBM Walk-Forward: {market} ===")
    for label, key in [("GLM", "glm"), ("LGBM", "lgbm")]:
        r = comparison[key]
        if not r.get("ran"):
            print(f"  {label}: SKIPPED ({r.get('reason') or r.get('error')})")
        else:
            print(
                f"  {label}: MAE={r['weighted_mae']}  "
                f"Sharpe={r.get('sharpe_ratio', 'N/A')}  "
                f"MaxDD={r.get('max_drawdown', 'N/A')}"
            )
    delta = comparison.get("delta", {})
    if delta:
        print("  Δ (LGBM - GLM):")
        for k, v in delta.items():
            direction = "↑" if v > 0 else "↓"
            print(f"    {k:<30}: {v:+.4f} {direction}")


def print_walk_forward_history(
    markets: List[str] = None,
    last_n: int = 5,
) -> None:
    """Print trend table from persisted walk_forward_results rows."""
    from src.data.db import get_db_connection

    market_clause = ""
    params: list = []
    if markets:
        placeholders = ",".join("?" * len(markets))
        market_clause = f"WHERE market IN ({placeholders})"
        params.extend(markets)

    sql = f"""
        SELECT market, model_type, mode, train_window_days, step_days,
               n_folds, weighted_mae, weighted_poisson_deviance,
               sharpe_ratio, max_drawdown, run_at
        FROM walk_forward_results
        {market_clause}
        ORDER BY market, run_at DESC
    """
    with get_db_connection() as conn:
        rows = conn.execute(sql, params).fetchall()

    if not rows:
        print("\n=== Walk-Forward History ===")
        print("  No results stored yet. Run `python main.py walkforward` first.")
        return

    # Group by market, keep last_n per market
    by_market: Dict[str, list] = {}
    for r in rows:
        m = r["market"]
        if m not in by_market:
            by_market[m] = []
        if len(by_market[m]) < last_n:
            by_market[m].append(r)

    print("\n=== Walk-Forward History ===")
    for market, runs in sorted(by_market.items()):
        first = runs[0]
        print(
            f"\n[{market}]  model={first['model_type']}  "
            f"window={first['train_window_days']}d  step={first['step_days']}d  mode={first['mode']}"
        )
        print(f"  {'run_at':<22} {'folds':>6} {'wMAE':>8} {'wDev':>8} {'Sharpe':>8} {'MaxDD':>9}")
        print("  " + "-" * 66)
        for r in runs:
            sharpe = f"{r['sharpe_ratio']:>8.4f}" if r["sharpe_ratio"] is not None else "     N/A"
            maxdd = f"{r['max_drawdown']:>9.6f}" if r["max_drawdown"] is not None else "      N/A"
            mae = f"{r['weighted_mae']:>8.4f}" if r["weighted_mae"] is not None else "     N/A"
            dev = f"{r['weighted_poisson_deviance']:>8.4f}" if r["weighted_poisson_deviance"] is not None else "     N/A"
            folds = r["n_folds"] if r["n_folds"] is not None else "?"
            print(f"  {r['run_at'][:22]:<22} {folds:>6} {mae} {dev} {sharpe} {maxdd}")
