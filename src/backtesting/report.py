"""Terminal report for BacktestSummary."""

from __future__ import annotations

import math

from .metrics import BacktestSummary

_W = 64   # total report width


def print_report(summary: BacktestSummary) -> None:
    """Print a structured backtest report to stdout."""
    _hr()
    _title(f"BACKTEST  {summary.start_date} → {summary.end_date}")
    _hr()

    _kv("Min edge threshold", f"{summary.min_edge*100:.1f}%")
    _kv("Total projections", str(summary.n_total_records))
    _kv("  with known results", str(summary.n_with_result))
    _kv("  bets placed", str(summary.n_bets))
    print()

    # --- P&L ---
    if summary.n_bets:
        _kv("Win rate",
            f"{summary.win_rate*100:.1f}%  ({summary.n_wins}/{summary.n_bets})")
        _kv("Flat P&L", f"{summary.flat_pnl:+.2f} units")
        _kv("Flat ROI", f"{summary.flat_roi*100:+.2f}%")
        _kv("Kelly P&L", f"{summary.kelly_pnl:+.4f} units")
        _kv("Avg edge (bets)", f"{summary.avg_edge*100:.2f}%")
    else:
        print("  No bets placed (edge threshold not reached or no results).")

    brier_str = (
        f"{summary.avg_brier:.4f}  (baseline 0.25 = random)"
        if not math.isnan(summary.avg_brier)
        else "N/A"
    )
    _kv("Brier score", brier_str)

    # --- Per-market table ---
    if summary.by_market:
        print()
        _section("BY MARKET")
        hdr = f"  {'Market':<28} {'Bets':>5} {'Win%':>6} {'ROI':>8} {'AvgEdge':>9} {'Brier':>7}"
        print(hdr)
        print("  " + "-" * (_W - 2))
        for m in sorted(summary.by_market.values(), key=lambda x: x.market):
            brier = (
                f"{m.avg_brier:.4f}" if not math.isnan(m.avg_brier) else "  N/A "
            )
            win_pct = m.win_rate * 100
            print(
                f"  {m.market:<28} {m.n_bets:>5}"
                f"  {win_pct:>5.1f}%  {m.flat_roi*100:>+6.2f}%"
                f"  {m.avg_edge*100:>+7.2f}%  {brier:>7}"
            )

    # --- Calibration table ---
    populated = [b for b in summary.calibration if b.n >= 5]
    if populated:
        print()
        _section("CALIBRATION  (predicted prob vs actual hit rate)")
        print(f"  {'Bucket':<12} {'N':>5} {'Predicted':>11} {'Actual':>8} {'Δ':>8}")
        print("  " + "-" * 48)
        for b in populated:
            delta_str = f"{b.delta*100:>+6.1f}%"
            print(
                f"  {b.label:<12} {b.n:>5}"
                f"  {b.predicted_rate*100:>9.1f}%"
                f"  {b.actual_rate*100:>6.1f}%"
                f"  {delta_str}"
            )
        print()
        print(
            "  A well-calibrated model has Δ ≈ 0 in every row.\n"
            "  Δ > 0 means the model is systematically underconfident.\n"
            "  Δ < 0 means it is overconfident."
        )

    _hr()
    print()


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _hr() -> None:
    print("=" * _W)


def _title(text: str) -> None:
    print(f"  {text}")


def _section(text: str) -> None:
    print(f"  {text}")


def _kv(label: str, value: str) -> None:
    print(f"  {label:<26} {value}")
