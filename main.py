import argparse
from src.data.db import init_db
from src.pipelines.sync_events import sync_events
from src.pipelines.sync_injuries import sync_injuries
from src.pipelines.sync_stats import sync_stats
from src.pipelines.scan_props import scan_props
from src.pipelines.send_alerts import send_alerts
from src.pipelines.sync_lineups import sync_lineups
from src.pipelines.sync_umpires import sync_umpires
from src.pipelines.settle_results import settle_results
from src.pipelines.prune_old_data import prune_old_data
from src.utils.logging_utils import get_logger

logger = get_logger(__name__)


def main():
    parser = argparse.ArgumentParser(description="MLB Player Prop Betting Bot")
    parser.add_argument('command', choices=['sync', 'scan', 'run', 'settle', 'prune', 'backtest'],
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
            scan_props()

        elif args.command == 'run':
            logger.info("Running FULL pipeline (sync -> scan -> alerts)...")
            sync_events()
            sync_injuries()
            sync_lineups()
            sync_umpires()
            # sync_stats() omitted by default (slow, run separately)
            scan_props()
            send_alerts()

        elif args.command == 'settle':
            logger.info("Running SETTLE mode...")
            sync_stats()
            settle_results()

        elif args.command == 'prune':
            logger.info("Running PRUNE mode...")
            prune_old_data()

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
