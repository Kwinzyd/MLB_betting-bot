import argparse
import asyncio
from src.data.db import init_db
from src.pipelines.sync_events import sync_events
from src.pipelines.sync_injuries import sync_injuries
from src.pipelines.sync_stats import sync_stats
from src.pipelines.scan_props import scan_props
from src.pipelines.send_alerts import send_alerts
from src.pipelines.find_sgp import find_and_alert_sgps
from src.pipelines.trigger_watch import run_trigger_watch
from src.pipelines.sync_lineups import sync_lineups
from src.pipelines.sync_umpires import sync_umpires
from src.pipelines.settle_results import settle_results
from src.pipelines.prune_old_data import prune_old_data
from src.utils.logging_utils import get_logger

logger = get_logger(__name__)


def main():
    parser = argparse.ArgumentParser(description="MLB Player Prop Betting Bot")
    subparsers = parser.add_subparsers(dest='command', required=True, help="Command to execute")

    # sync
    subparsers.add_parser('sync', help="Sync daily events, injuries, lineups, and umpires")

    # scan
    scan_parser = subparsers.add_parser('scan', help="Scan active games for prop edges")
    scan_parser.add_argument('--force', action='store_true',
                             help="Bypass the quota gate and scan every active game")

    # run
    run_parser = subparsers.add_parser('run', help="Run full pipeline (sync -> scan -> alerts)")
    run_parser.add_argument('--force', action='store_true',
                            help="Bypass the quota gate and scan every active game")

    # settle, prune, sgp, trigger, fit-dispersion
    subparsers.add_parser('settle', help="Sync stats and settle completed bets")
    subparsers.add_parser('prune', help="Prune old database records")
    subparsers.add_parser('sgp', help="Find and alert Same Game Parlays (SGPs)")
    subparsers.add_parser('trigger', help="Run trigger watch for weather and umpires")
    subparsers.add_parser('fit-dispersion', help="Fit dispersion models for projections")

    # schedule
    subparsers.add_parser(
        'schedule',
        help="Run the long-lived scheduler (sync/scan loop + nightly settle)"
    )

    # report
    report_parser = subparsers.add_parser(
        'report',
        help="Summarize live/shadow bet performance (CLV, P&L, calibration)"
    )
    report_parser.add_argument('--since', type=int, metavar='DAYS', default=None,
                               help="Only include bets from the last N days (default: all)")
    report_parser.add_argument('--market', action='append', dest='markets', metavar='MARKET',
                               help="Restrict to specific market(s); repeatable")

    # backtest-specific flags (ignored for other commands)
    backtest_parser = subparsers.add_parser('backtest', help="Run historical backtests")
    backtest_parser.add_argument('--start', metavar='YYYY-MM-DD', required=True,
                                 help="Backtest start date (inclusive)")
    backtest_parser.add_argument('--end', metavar='YYYY-MM-DD', required=True,
                                 help="Backtest end date (inclusive)")
    backtest_parser.add_argument('--min-edge', type=float, default=5.0, metavar='PCT',
                                 help="Minimum edge %% to count as a bet (default: 5.0)")
    backtest_parser.add_argument('--market', action='append', dest='markets', metavar='MARKET',
                                 help="Restrict to specific market(s); repeatable. "
                                      "E.g. --market pitcher_strikeouts --market batter_hits")

    # backfill + train flags
    backfill_parser = subparsers.add_parser('backfill', help="Backfill historical game logs")
    backfill_parser.add_argument('--seasons', metavar='YYYY[,YYYY,...]', required=True,
                                 help="Seasons for backfill, comma-separated")

    train_parser = subparsers.add_parser('train', help="Train projection models")
    train_parser.add_argument('--compare', choices=['sklearn'], default=None,
                              help="also fit sklearn RF/GB for comparison")

    # walkforward flags
    wf_parser = subparsers.add_parser('walkforward', help="Run walk-forward validation")
    wf_parser.add_argument('--train-window', type=int, default=45, metavar='DAYS',
                           help="training window length in days (default 45)")
    wf_parser.add_argument('--step', type=int, default=7, metavar='DAYS',
                           help="step size between folds in days (default 7)")
    wf_parser.add_argument('--mode', choices=['sliding', 'expanding'], default='sliding',
                           help="sliding (fixed window) or expanding training set")

    args = parser.parse_args()

    # Always ensure DB is initialized
    init_db()

    try:
        if args.command == 'sync':
            logger.info("Running SYNC mode...")
            asyncio.run(sync_events())
            asyncio.run(sync_injuries())
            asyncio.run(sync_stats())
            asyncio.run(sync_lineups())
            sync_umpires()

        elif args.command == 'scan':
            logger.info("Running SCAN mode...")
            asyncio.run(scan_props(force=args.force))

        elif args.command == 'run':
            logger.info("Running FULL pipeline (sync -> scan -> alerts)...")
            asyncio.run(sync_events())
            asyncio.run(sync_injuries())
            asyncio.run(sync_lineups())
            sync_umpires()
            # asyncio.run(sync_stats()) omitted by default (slow, run separately)
            asyncio.run(scan_props(force=args.force))
            asyncio.run(send_alerts())
            asyncio.run(find_and_alert_sgps())

        elif args.command == 'sgp':
            logger.info("Running SGP mode...")
            asyncio.run(find_and_alert_sgps())

        elif args.command == 'trigger':
            logger.info("Running TRIGGER WATCH mode...")
            asyncio.run(run_trigger_watch())

        elif args.command == 'walkforward':
            from src.pipelines.walk_forward import walk_forward_all, print_walk_forward_report
            logger.info(
                f"Running WALK-FORWARD backtest "
                f"(window={args.train_window}d, step={args.step}d, mode={args.mode})..."
            )
            results = walk_forward_all(
                markets=args.markets,
                train_window_days=args.train_window,
                step_days=args.step,
                mode=args.mode,
            )
            print_walk_forward_report(results)

        elif args.command == 'settle':
            logger.info("Running SETTLE mode...")
            asyncio.run(sync_stats())
            settle_results()

        elif args.command == 'report':
            from src.pipelines.report import generate_report
            generate_report(since_days=args.since, markets=args.markets)

        elif args.command == 'schedule':
            from src.scheduler import run_scheduler
            logger.info("Starting SCHEDULER (Ctrl-C to stop)...")
            asyncio.run(run_scheduler())

        elif args.command == 'prune':
            logger.info("Running PRUNE mode...")
            prune_old_data()

        elif args.command == 'backfill':
            from src.pipelines.sync_historical import sync_historical
            seasons = [int(s.strip()) for s in args.seasons.split(',') if s.strip()]
            logger.info(f"Running BACKFILL for seasons={seasons} ...")
            sync_historical(seasons)

        elif args.command == 'fit-dispersion':
            from src.pipelines.fit_dispersion import fit_all_dispersion
            logger.info("Running FIT-DISPERSION mode...")
            fit_all_dispersion()

        elif args.command == 'train':
            from src.pipelines.train_model import train_all
            logger.info("Running TRAIN mode ...")
            results = train_all(compare_sklearn=(args.compare == 'sklearn'))
            for r in results:
                logger.info(f"Training result: {r}")

        elif args.command == 'backtest':
            from src.backtesting.engine import BacktestEngine
            from src.backtesting.metrics import compute_summary
            from src.backtesting.report import print_report

            min_edge = args.min_edge / 100.0   # convert % to fraction
            engine = BacktestEngine(
                start_date=args.start,
                end_date=args.end,
                min_edge=min_edge,
                markets=args.markets,
            )
            records = engine.run()
            summary = compute_summary(records, args.start, args.end, min_edge)
            print_report(summary)

    except Exception as e:
        logger.error(f"Execution failed: {e}", exc_info=True)
        _notify_crash(args.command, e)
        raise


def _notify_crash(command: str, exception: Exception) -> None:
    """Best-effort Telegram alert on unhandled pipeline failure.

    Runs regardless of `BETTING_ENABLED` — crash alerts are operational,
    not trading signals; silencing them during shadow mode is how paper
    trading goes dark without anyone noticing. Catches its own errors so
    a Telegram outage can never mask the original exception.
    """
    try:
        import traceback
        from src.clients.telegram_bot import TelegramClient
        tail = ''.join(traceback.format_exception_only(type(exception), exception)).strip()
        msg = (
            f"\U0001F6A8 <b>Pipeline Crash</b>\n"
            f"Command: <code>{command}</code>\n"
            f"Error: <code>{tail[:400]}</code>"
        )
        TelegramClient().send_message_sync(msg)
    except Exception as notify_err:
        logger.error(f"Failed to send crash alert: {notify_err}")


if __name__ == "__main__":
    main()
