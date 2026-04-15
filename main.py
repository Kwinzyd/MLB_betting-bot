import argparse
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
    parser.add_argument('command', choices=['sync', 'scan', 'run', 'settle', 'prune',
                                             'backtest', 'backfill', 'train', 'sgp',
                                             'trigger', 'walkforward'],
                        help="Command to execute")

    # backtest-specific flags (ignored for other commands)
    parser.add_argument('--start', metavar='YYYY-MM-DD',
                        help="Backtest start date (inclusive)")
    parser.add_argument('--end', metavar='YYYY-MM-DD',
                        help="Backtest end date (inclusive)")
    parser.add_argument('--min-edge', type=float, default=5.0, metavar='PCT',
                        help="Minimum edge %% to count as a bet (default: 5.0)")
    parser.add_argument('--market', action='append', dest='markets', metavar='MARKET',
                        help="Restrict to specific market(s); repeatable. "
                             "E.g. --market pitcher_strikeouts --market batter_hits")

    # backfill + train flags
    parser.add_argument('--seasons', metavar='YYYY[,YYYY,...]',
                        help="Seasons for backfill, comma-separated")
    parser.add_argument('--compare', choices=['sklearn'], default=None,
                        help="train: also fit sklearn RF/GB for comparison")

    # walkforward flags
    parser.add_argument('--train-window', type=int, default=45, metavar='DAYS',
                        help="walkforward: training window length in days (default 45)")
    parser.add_argument('--step', type=int, default=7, metavar='DAYS',
                        help="walkforward: step size between folds in days (default 7)")
    parser.add_argument('--mode', choices=['sliding', 'expanding'], default='sliding',
                        help="walkforward: sliding (fixed window) or expanding training set")

    # scan flags
    parser.add_argument('--force', action='store_true',
                        help="scan: bypass the quota gate and scan every active game")

    args = parser.parse_args()

    # Always ensure DB is initialized
    init_db()

    try:
        if args.command == 'sync':
            logger.info("Running SYNC mode...")
            sync_events()
            sync_injuries()
            sync_stats()
            sync_lineups()
            sync_umpires()

        elif args.command == 'scan':
            logger.info("Running SCAN mode...")
            scan_props(force=args.force)

        elif args.command == 'run':
            logger.info("Running FULL pipeline (sync -> scan -> alerts)...")
            sync_events()
            sync_injuries()
            sync_lineups()
            sync_umpires()
            # sync_stats() omitted by default (slow, run separately)
            scan_props(force=args.force)
            send_alerts()
            find_and_alert_sgps()

        elif args.command == 'sgp':
            logger.info("Running SGP mode...")
            find_and_alert_sgps()

        elif args.command == 'trigger':
            logger.info("Running TRIGGER WATCH mode...")
            run_trigger_watch()

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
            sync_stats()
            settle_results()

        elif args.command == 'prune':
            logger.info("Running PRUNE mode...")
            prune_old_data()

        elif args.command == 'backfill':
            if not args.seasons:
                parser.error("backfill requires --seasons (e.g. --seasons 2022,2023,2024,2025)")
            from src.pipelines.sync_historical import sync_historical
            seasons = [int(s.strip()) for s in args.seasons.split(',') if s.strip()]
            logger.info(f"Running BACKFILL for seasons={seasons} ...")
            sync_historical(seasons)

        elif args.command == 'train':
            from src.pipelines.train_model import train_all
            logger.info("Running TRAIN mode ...")
            results = train_all(compare_sklearn=(args.compare == 'sklearn'))
            for r in results:
                logger.info(f"Training result: {r}")

        elif args.command == 'backtest':
            if not args.start or not args.end:
                parser.error("backtest requires --start and --end  (YYYY-MM-DD)")
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


if __name__ == "__main__":
    main()
