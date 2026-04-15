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
from datetime import datetime, timedelta
from typing import Dict, Iterator, List, Tuple

import numpy as np

from src.models.ml_model import PoissonGLM
from src.pipelines.train_model import (
    _ALL_MARKETS,
    _BATTER_MARKETS,
    _PITCHER_MARKETS,
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


def walk_forward_market(
    market: str,
    train_window_days: int = 45,
    step_days: int = 7,
    mode: str = "sliding",
    min_train_rows: int = 200,
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

        glm = PoissonGLM(feature_names=list(names), l2=0.01)
        try:
            glm.fit(X[train_mask], y[train_mask], exposure=exp_[train_mask])
        except Exception as e:
            logger.warning(f"  fit failed for fold test_start={test_start}: {e}")
            continue

        metrics = _evaluate(glm, X[test_mask], y[test_mask], exp_[test_mask])
        metrics.update({
            "test_start": test_start,
            "test_end": test_end,
            "n_train": n_train,
            "converged": glm.converged_,
        })
        folds.append(metrics)

    if not folds:
        return {"market": market, "ran": False, "reason": "no_eligible_folds"}

    # Aggregate: weighted MAE by n, plus monthly drift bins
    total_n = sum(f["n"] for f in folds)
    weighted_mae = sum(f["mae"] * f["n"] for f in folds) / max(1, total_n)
    weighted_dev = sum(f["poisson_deviance"] * f["n"] for f in folds) / max(1, total_n)

    by_month = defaultdict(lambda: {"n": 0, "mae_num": 0.0, "dev_num": 0.0})
    for f in folds:
        ym = f["test_start"][:7]  # YYYY-MM
        by_month[ym]["n"] += f["n"]
        by_month[ym]["mae_num"] += f["mae"] * f["n"]
        by_month[ym]["dev_num"] += f["poisson_deviance"] * f["n"]
    monthly = []
    for ym in sorted(by_month):
        agg = by_month[ym]
        monthly.append({
            "month": ym,
            "n": agg["n"],
            "mae": round(agg["mae_num"] / max(1, agg["n"]), 4),
            "poisson_deviance": round(agg["dev_num"] / max(1, agg["n"]), 4),
        })

    return {
        "market": market,
        "ran": True,
        "mode": mode,
        "train_window_days": train_window_days,
        "step_days": step_days,
        "n_folds": len(folds),
        "weighted_mae": round(weighted_mae, 4),
        "weighted_poisson_deviance": round(weighted_dev, 4),
        "monthly": monthly,
        "folds": folds,
    }


def walk_forward_all(
    markets: List[str] = None,
    train_window_days: int = 45,
    step_days: int = 7,
    mode: str = "sliding",
) -> List[Dict]:
    targets = list(markets) if markets else list(_ALL_MARKETS)
    results = []
    for m in targets:
        try:
            results.append(walk_forward_market(
                m, train_window_days=train_window_days,
                step_days=step_days, mode=mode,
            ))
        except Exception as e:
            logger.error(f"Walk-forward failed for {m}: {e}", exc_info=True)
            results.append({"market": m, "ran": False, "error": str(e)})
    return results


def print_walk_forward_report(results: List[Dict]) -> None:
    """Human-readable summary: per-market headline, then monthly drift table."""
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
        if r.get("monthly"):
            print(f"  monthly drift:")
            print(f"    {'month':<9} {'n':>6} {'mae':>8} {'deviance':>10}")
            for m in r["monthly"]:
                print(
                    f"    {m['month']:<9} {m['n']:>6} {m['mae']:>8.4f} "
                    f"{m['poisson_deviance']:>10.4f}"
                )
