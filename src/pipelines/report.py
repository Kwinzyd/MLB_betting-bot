"""
On-demand live-performance report.

Answers the three questions paper trading needs to confirm before we flip
`BETTING_ENABLED=true`:
  1. Are our probabilities calibrated?      — Brier + calibration buckets
  2. Are we beating the closing line?       — avg CLV, % positive CLV
  3. Is the bankroll growing?               — flat/Kelly P&L, ROI

Reuses the backtest engine's metric helpers by exposing a duck-typed
`LiveBetRecord` with the same attribute names the helpers read off of
`BacktestRecord`. No schema of the live DB is duplicated into the backtest
side; we just adapt one to the other at query time.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple
from src.utils.time_utils import utcnow

from src.backtesting.metrics import compute_summary
from src.backtesting.report import print_report
from src.config import BANKROLL
from src.data.db import get_db_connection
from src.models.kelly import get_current_bankroll
from src.utils.logging_utils import get_logger

logger = get_logger(__name__)


@dataclass
class LiveBetRecord:
    """Duck-typed against `BacktestRecord` — only the attributes that
    `compute_summary` / `_brier` / `_calibration_table` / `_by_market`
    actually read are populated."""
    market: str
    best_side: str              # 'over' | 'under'
    best_edge: float            # fraction, e.g. 0.075 for 7.5%
    model_prob_over: float
    model_prob_under: float
    actual_over: bool           # actual_value > line (side-independent)
    flat_pnl: float             # +1 / 0 / -1 unit
    kelly_pnl: float            # profit / starting bankroll (fractional)


def generate_report(since_days: Optional[int] = None,
                    markets: Optional[List[str]] = None) -> None:
    """Render the live-performance report to stdout."""
    records, fallback_count, unsettled_count = _fetch_settled_bets(since_days, markets)

    _print_header(since_days, markets, len(records), fallback_count, unsettled_count)

    if not records:
        print("  No settled bets to report yet.\n")
        print("=" * 64)
        _print_bankroll_section()
        return

    start_date, end_date = _date_range(since_days)
    summary = compute_summary(records, start_date, end_date, min_edge=0.0)
    print_report(summary)

    _print_clv_section(since_days, markets)
    _print_bankroll_section()


def _fetch_settled_bets(
    since_days: Optional[int],
    markets: Optional[List[str]],
) -> Tuple[List[LiveBetRecord], int, int]:
    """Return (records, fallback_prob_count, unsettled_count).

    fallback_prob_count = bets that used projections.prob_* because
    alerts_sent.model_prob_* was NULL (placed before the migration).
    """
    params: list = []
    where = ["br.result IS NOT NULL"]

    if since_days is not None:
        where.append("a.timestamp >= ?")
        cutoff = _cutoff_iso(since_days)
        params.append(cutoff)

    market_clause = ""
    if markets:
        placeholders = ",".join("?" * len(markets))
        market_clause = f" AND a.market IN ({placeholders})"
        params.extend(markets)

    sql = f"""
        SELECT
            a.market,
            a.side,
            a.line,
            a.edge,
            a.model_prob_over AS alert_prob_over,
            a.model_prob_under AS alert_prob_under,
            p.prob_over AS proj_prob_over,
            p.prob_under AS proj_prob_under,
            br.actual_value,
            br.result,
            br.profit
        FROM alerts_sent a
        JOIN bet_results br ON a.alert_id = br.alert_id
        LEFT JOIN projections p
               ON a.game_id = p.game_id
              AND a.player_name = p.player_name
              AND a.market = p.market
        WHERE {' AND '.join(where)}{market_clause}
        ORDER BY a.timestamp ASC
    """

    records: List[LiveBetRecord] = []
    fallback_count = 0

    with get_db_connection() as conn:
        rows = conn.execute(sql, params).fetchall()
        unsettled_count = _count_unsettled(conn, since_days, markets)

    for row in rows:
        prob_over = row['alert_prob_over']
        prob_under = row['alert_prob_under']
        if prob_over is None or prob_under is None:
            # Fallback: use the current projection. Stale-risk, but the only
            # signal available for bets placed before the migration.
            prob_over = row['proj_prob_over']
            prob_under = row['proj_prob_under']
            fallback_count += 1

        if prob_over is None or prob_under is None:
            # No probability anywhere — can't score. Skip silently; these are
            # ancient rows with broken joins.
            continue
        if row['actual_value'] is None or row['line'] is None:
            continue

        actual_over = row['actual_value'] > row['line']
        result = row['result']
        flat_pnl = 1.0 if result == 'WIN' else (-1.0 if result == 'LOSS' else 0.0)
        kelly_pnl = (row['profit'] or 0.0) / BANKROLL if BANKROLL else 0.0

        records.append(LiveBetRecord(
            market=row['market'],
            best_side=row['side'],
            best_edge=(row['edge'] or 0.0) / 100.0,
            model_prob_over=prob_over,
            model_prob_under=prob_under,
            actual_over=actual_over,
            flat_pnl=flat_pnl,
            kelly_pnl=kelly_pnl,
        ))

    return records, fallback_count, unsettled_count


def _count_unsettled(conn, since_days, markets) -> int:
    params: list = []
    where = ["br.alert_id IS NULL"]
    if since_days is not None:
        where.append("a.timestamp >= ?")
        params.append(_cutoff_iso(since_days))
    market_clause = ""
    if markets:
        placeholders = ",".join("?" * len(markets))
        market_clause = f" AND a.market IN ({placeholders})"
        params.extend(markets)
    sql = f"""
        SELECT COUNT(*) AS n
        FROM alerts_sent a
        LEFT JOIN bet_results br ON a.alert_id = br.alert_id
        WHERE {' AND '.join(where)}{market_clause}
    """
    row = conn.execute(sql, params).fetchone()
    return int(row['n']) if row else 0


def _cutoff_iso(since_days: int) -> str:
    from datetime import timedelta
    return (utcnow() - timedelta(days=since_days)).isoformat()


def _date_range(since_days: Optional[int]) -> Tuple[str, str]:
    # compute_summary just stores these as display strings; exact bounds
    # don't matter for metrics. Use ISO dates so the header reads cleanly.
    today = utcnow().date().isoformat()
    if since_days is None:
        return ("all-time", today)
    from datetime import timedelta
    start = (utcnow().date() - timedelta(days=since_days)).isoformat()
    return (start, today)


def _print_header(since_days, markets, n_records, fallback_count, unsettled_count) -> None:
    print()
    print("=" * 64)
    scope = "all-time" if since_days is None else f"last {since_days} days"
    market_filter = f" | markets: {','.join(markets)}" if markets else ""
    print(f"  LIVE BET REPORT  ({scope}{market_filter})")
    print("=" * 64)
    print(f"  Settled bets scored : {n_records}")
    print(f"  Unsettled bets      : {unsettled_count}")
    if fallback_count:
        print(
            f"  Note: {fallback_count} bet(s) used fallback prob from current\n"
            f"        projections table (no placement-time prob recorded).\n"
            f"        Calibration for those rows is stale-risk."
        )
    print()


def _print_clv_section(since_days: Optional[int], markets: Optional[List[str]]) -> None:
    """CLV = devigged sharp closing prob - opening implied prob. Positive
    means we beat the sharp close. % positive CLV is the durable sharpness
    signal — P&L wobbles with variance, CLV doesn't."""
    params: list = []
    where = ["br.clv IS NOT NULL", "br.clv != 0.0"]  # 0.0 is a sentinel for bad devig

    if since_days is not None:
        where.append("a.timestamp >= ?")
        params.append(_cutoff_iso(since_days))

    market_clause = ""
    if markets:
        placeholders = ",".join("?" * len(markets))
        market_clause = f" AND a.market IN ({placeholders})"
        params.extend(markets)

    sql = f"""
        SELECT br.clv
        FROM alerts_sent a
        JOIN bet_results br ON a.alert_id = br.alert_id
        WHERE {' AND '.join(where)}{market_clause}
    """

    missing_sql = f"""
        SELECT COUNT(*) AS n
        FROM alerts_sent a
        JOIN bet_results br ON a.alert_id = br.alert_id
        WHERE (br.clv IS NULL OR br.clv = 0.0)
          {'AND a.timestamp >= ?' if since_days is not None else ''}
          {market_clause}
    """

    with get_db_connection() as conn:
        clvs = [row['clv'] for row in conn.execute(sql, params).fetchall()]
        missing = conn.execute(missing_sql, params).fetchone()
        missing_count = int(missing['n']) if missing else 0

    print()
    print("  CLV  (closing line value — sharpness signal)")
    print("  " + "-" * 60)
    if not clvs:
        print(f"  No scoreable CLV data yet. ({missing_count} bet(s) missing sharp close.)")
        return

    n = len(clvs)
    avg = sum(clvs) / n
    positive = sum(1 for c in clvs if c > 0)
    pct_positive = (positive / n) * 100
    best = max(clvs)
    worst = min(clvs)

    print(f"  Scored bets         : {n}   (missing sharp close: {missing_count})")
    print(f"  Avg CLV             : {avg:+.4f}")
    print(f"  % positive CLV      : {pct_positive:.1f}%   ({positive}/{n})")
    print(f"  Best / Worst        : {best:+.4f}  /  {worst:+.4f}")
    print()
    print("  Positive avg CLV + >52% positive-CLV rate = you are sharp.")
    print("  Negative avg CLV means the market closes past your prices —")
    print("  you are the soft side, even if short-run P&L is positive.")


def _print_bankroll_section() -> None:
    """Starting roll + ledgered P&L, 7-day curve, and circuit breaker state."""
    current = get_current_bankroll()
    delta = current - BANKROLL
    roi_pct = (delta / BANKROLL) * 100 if BANKROLL else 0.0
    print()
    print("  BANKROLL")
    print("  " + "-" * 60)
    print(f"  Starting            : ${BANKROLL:,.2f}")
    print(f"  Current             : ${current:,.2f}")
    print(f"  P&L                 : ${delta:+,.2f}   ({roi_pct:+.2f}%)")

    _print_bankroll_curve()
    _print_circuit_breaker_state()
    _print_ece_section()
    _print_calibration_status()
    _print_drift_status()

    print("=" * 64)
    print()


def _print_bankroll_curve() -> None:
    """7-day rolling P&L mini-table from bankroll_snapshots."""
    sql = """
        SELECT snapshot_date, bankroll, daily_pnl, rolling_7d_pnl, rolling_7d_roi
        FROM bankroll_snapshots
        ORDER BY snapshot_date DESC
        LIMIT 14
    """
    with get_db_connection() as conn:
        rows = conn.execute(sql).fetchall()

    if not rows:
        return

    print()
    print("  7-day P&L curve (most recent first):")
    print(f"  {'Date':<12} {'Bankroll':>10} {'Daily P&L':>10} {'7d ROI':>8}")
    print("  " + "-" * 44)
    for r in rows[:7]:
        daily = r['daily_pnl'] or 0.0
        roi7 = r['rolling_7d_roi']
        roi7_str = f"{roi7*100:+.1f}%" if roi7 is not None else "  N/A"
        sign = "+" if daily >= 0 else ""
        print(
            f"  {r['snapshot_date']:<12} ${r['bankroll']:>9,.2f}"
            f"  {sign}${daily:>8,.2f}  {roi7_str:>8}"
        )


def _print_circuit_breaker_state() -> None:
    """Current kelly_fraction_override and full_stop flag from today's snapshot."""
    sql = """
        SELECT snapshot_date, kelly_fraction_override, full_stop, halved_at,
               rolling_7d_roi
        FROM bankroll_snapshots
        ORDER BY snapshot_date DESC
        LIMIT 1
    """
    with get_db_connection() as conn:
        row = conn.execute(sql).fetchone()

    if not row:
        return

    print()
    print("  CIRCUIT BREAKERS")
    print("  " + "-" * 60)
    full_stop = bool(row['full_stop'])
    override = row['kelly_fraction_override'] if row['kelly_fraction_override'] is not None else 1.0
    roi7 = row['rolling_7d_roi']
    roi7_str = f"{roi7*100:+.1f}%" if roi7 is not None else "N/A"

    status = "FULL STOP — no new bets" if full_stop else (
        f"Kelly x{override:.2f}" if override != 1.0 else "Normal (Kelly x1.00)"
    )
    print(f"  Status              : {status}")
    print(f"  7-day rolling ROI   : {roi7_str}")
    if row['halved_at']:
        print(f"  Halved at           : {row['halved_at']}")
    if full_stop:
        print()
        print("  ** FULL STOP is active. send_alerts will not fire. **")
    elif override < 1.0:
        print()
        print(f"  Kelly reduced to {override:.0%}. Stakes are capped until ROI recovers.")


def _print_ece_section() -> None:
    """Expected Calibration Error per market from calibration_log.

    ECE = Σ (n_bin / n_total) × |mean_predicted_bin − actual_win_rate_bin|

    Lower is better; 0 = perfect. Requires ≥50 settled bets per market.
    """
    sql = """
        SELECT market,
               prob_bin,
               COUNT(*)          AS n_bin,
               AVG(predicted_prob) AS mean_pred,
               AVG(actual_outcome) AS actual_rate
        FROM calibration_log
        GROUP BY market, prob_bin
        ORDER BY market, prob_bin
    """
    with get_db_connection() as conn:
        rows = conn.execute(sql).fetchall()

    if not rows:
        return

    from collections import defaultdict
    bins_by_market: dict = defaultdict(list)
    for r in rows:
        bins_by_market[r["market"]].append(r)

    print()
    print("  CALIBRATION ECE  (Expected Calibration Error — lower is better)")
    print(f"  {'Market':<30} {'N':>6} {'ECE':>8}")
    print("  " + "-" * 48)

    for market in sorted(bins_by_market):
        bins = bins_by_market[market]
        total_n = sum(b["n_bin"] for b in bins)
        if total_n < 50:
            print(f"  {market:<30} {total_n:>6}  {'N/A':>8}  (need ≥50)")
            continue
        ece = sum(
            (b["n_bin"] / total_n) * abs((b["mean_pred"] or 0.0) - (b["actual_rate"] or 0.0))
            for b in bins
        )
        print(f"  {market:<30} {total_n:>6}  {ece:>8.4f}")


def _print_calibration_status() -> None:
    """Active calibration_params per market — method, Brier before/after, sample count."""
    sql = """
        SELECT market, method, n_samples, brier_score, brier_score_calibrated, fitted_at
        FROM calibration_params
        WHERE is_active = 1
        ORDER BY market
    """
    with get_db_connection() as conn:
        rows = conn.execute(sql).fetchall()

    if not rows:
        print()
        print("  CALIBRATION  No active calibration params (need ≥200 settled bets per market).")
        return

    print()
    print("  CALIBRATION PARAMS  (active fits)")
    print(f"  {'Market':<30} {'Method':<10} {'N':>5} {'Brier raw':>10} {'Brier cal':>10} {'Fitted':<22}")
    print("  " + "-" * 92)
    for r in rows:
        b_raw = f"{r['brier_score']:.4f}" if r['brier_score'] is not None else "  N/A  "
        b_cal = f"{r['brier_score_calibrated']:.4f}" if r['brier_score_calibrated'] is not None else "  N/A  "
        print(
            f"  {r['market']:<30} {r['method']:<10} {r['n_samples']:>5}"
            f"  {b_raw:>10}  {b_cal:>10}  {r['fitted_at'][:19]:<22}"
        )


def _print_drift_status() -> None:
    """Most recent model_drift_log entry per market."""
    sql = """
        SELECT market, model_version, drift_score, psi_score,
               mae_this_window, mae_baseline, retrain_triggered, logged_at
        FROM model_drift_log
        WHERE id IN (
            SELECT MAX(id) FROM model_drift_log GROUP BY market
        )
        ORDER BY market
    """
    with get_db_connection() as conn:
        rows = conn.execute(sql).fetchall()

    if not rows:
        return

    print()
    print("  MODEL DRIFT  (last weekly check per market)")
    print(f"  {'Market':<30} {'Drift':>7} {'PSI':>7} {'MAE now':>8} {'MAE base':>9} {'Retrained':<10} {'Logged':<22}")
    print("  " + "-" * 100)
    for r in rows:
        ds = f"{r['drift_score']:+.3f}" if r['drift_score'] is not None else "  N/A "
        ps = f"{r['psi_score']:.3f}" if r['psi_score'] is not None else "  N/A"
        mae_w = f"{r['mae_this_window']:.4f}" if r['mae_this_window'] is not None else "  N/A "
        mae_b = f"{r['mae_baseline']:.4f}" if r['mae_baseline'] is not None else "  N/A  "
        retrained = "YES" if r['retrain_triggered'] else "no"
        print(
            f"  {r['market']:<30} {ds:>7}  {ps:>7}  {mae_w:>8}  {mae_b:>9}  {retrained:<10}  {r['logged_at'][:19]:<22}"
        )
