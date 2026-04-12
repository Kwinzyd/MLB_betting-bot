"""
Performance metrics for backtest results.

Three lenses on model quality:
  1. P&L / ROI — does betting the model's edges make money?
  2. Brier score — are the model's probabilities accurate in absolute terms?
     (0.25 = random coin flip; lower is better)
  3. Calibration table — when the model says 60%, does it hit 60%?
     A well-calibrated model lies on the diagonal of this table.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List

from .engine import BacktestRecord


# ---------------------------------------------------------------------------
# Summary dataclasses
# ---------------------------------------------------------------------------

@dataclass
class MarketSummary:
    market: str
    n_bets: int
    n_wins: int
    win_rate: float
    flat_pnl: float
    flat_roi: float
    avg_edge: float
    avg_brier: float   # NaN when no scoreable records


@dataclass
class CalibrationBucket:
    label: str          # e.g. "55%-60%"
    n: int
    predicted_rate: float
    actual_rate: float

    @property
    def delta(self) -> float:
        """actual_rate - predicted_rate. Positive = model underestimates."""
        return self.actual_rate - self.predicted_rate


@dataclass
class BacktestSummary:
    # Run parameters
    start_date: str
    end_date: str
    min_edge: float

    # Volume
    n_total_records: int   # all records with a projection
    n_with_result: int     # subset where actual result is known
    n_bets: int            # subset where edge >= min_edge AND result is known

    # P&L (flat $1 per bet)
    n_wins: int
    win_rate: float
    flat_pnl: float
    flat_roi: float        # flat_pnl / n_bets

    # Kelly P&L (fractional-Kelly sizing)
    kelly_pnl: float

    # Model quality
    avg_edge: float        # mean edge among bets placed
    avg_brier: float       # mean Brier score over all records with known results

    # Breakdowns
    by_market: Dict[str, MarketSummary] = field(default_factory=dict)
    calibration: List[CalibrationBucket] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def compute_summary(
    records: List[BacktestRecord],
    start_date: str,
    end_date: str,
    min_edge: float,
) -> BacktestSummary:
    """Derive all metrics from a list of BacktestRecords."""
    bets = [r for r in records if r.flat_pnl is not None]
    scoreable = [r for r in records if r.actual_over is not None]

    n_bets = len(bets)
    n_wins = sum(1 for r in bets if r.flat_pnl > 0)
    flat_pnl = sum(r.flat_pnl for r in bets)
    kelly_pnl = sum(r.kelly_pnl for r in bets if r.kelly_pnl is not None)

    return BacktestSummary(
        start_date=start_date,
        end_date=end_date,
        min_edge=min_edge,
        n_total_records=len(records),
        n_with_result=len(scoreable),
        n_bets=n_bets,
        n_wins=n_wins,
        win_rate=n_wins / n_bets if n_bets else 0.0,
        flat_pnl=flat_pnl,
        flat_roi=flat_pnl / n_bets if n_bets else 0.0,
        kelly_pnl=kelly_pnl,
        avg_edge=sum(r.best_edge for r in bets) / n_bets if n_bets else 0.0,
        avg_brier=_mean_brier(scoreable),
        by_market=_by_market(bets, scoreable),
        calibration=_calibration_table(scoreable),
    )


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _brier(record: BacktestRecord) -> float:
    """
    Brier score for a single record.
    Uses the probability assigned to the side we considered "best",
    evaluated against whether that side actually won.
    """
    if record.best_side == 'over':
        prob = record.model_prob_over
        outcome = float(record.actual_over)
    else:
        prob = record.model_prob_under
        outcome = float(not record.actual_over)
    return (prob - outcome) ** 2


def _mean_brier(scoreable: List[BacktestRecord]) -> float:
    if not scoreable:
        return math.nan
    return sum(_brier(r) for r in scoreable) / len(scoreable)


def _by_market(
    bets: List[BacktestRecord],
    scoreable: List[BacktestRecord],
) -> Dict[str, MarketSummary]:
    markets = {r.market for r in bets}
    result: Dict[str, MarketSummary] = {}

    for market in markets:
        mkt_bets = [r for r in bets if r.market == market]
        mkt_scoreable = [r for r in scoreable if r.market == market]
        n = len(mkt_bets)
        wins = sum(1 for r in mkt_bets if r.flat_pnl > 0)
        pnl = sum(r.flat_pnl for r in mkt_bets)

        result[market] = MarketSummary(
            market=market,
            n_bets=n,
            n_wins=wins,
            win_rate=wins / n if n else 0.0,
            flat_pnl=pnl,
            flat_roi=pnl / n if n else 0.0,
            avg_edge=sum(r.best_edge for r in mkt_bets) / n if n else 0.0,
            avg_brier=_mean_brier(mkt_scoreable),
        )

    return result


def _calibration_table(scoreable: List[BacktestRecord]) -> List[CalibrationBucket]:
    """
    Bin model probabilities into 5-percentage-point buckets and compare to
    actual hit rates. A perfectly calibrated model has predicted_rate = actual_rate
    in every bucket (i.e. delta = 0).

    Only returns buckets in the range 50%–100% because we always bet the side
    with the higher model probability (so we never have edge on a sub-50% call).
    """
    Bucket = dict  # internal accumulator
    bins: Dict[int, Bucket] = {}

    for r in scoreable:
        prob = r.model_prob_over if r.best_side == 'over' else r.model_prob_under
        outcome = r.actual_over if r.best_side == 'over' else not r.actual_over

        # Map to 5pp bucket: 0.50–0.55 → key 50, 0.55–0.60 → key 55, …
        key = min(95, max(50, int(prob * 20) * 5))
        if key not in bins:
            bins[key] = {'sum_prob': 0.0, 'wins': 0, 'n': 0}
        bins[key]['sum_prob'] += prob
        bins[key]['wins'] += int(outcome)
        bins[key]['n'] += 1

    result: List[CalibrationBucket] = []
    for key in sorted(bins):
        b = bins[key]
        n = b['n']
        result.append(CalibrationBucket(
            label=f"{key}%–{key+5}%",
            n=n,
            predicted_rate=b['sum_prob'] / n,
            actual_rate=b['wins'] / n,
        ))
    return result
