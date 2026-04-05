import argparse
from src.data.db import init_db
from src.pipelines.sync_events import sync_events
from src.pipelines.sync_injuries import sync_injuries
from src.pipelines.sync_stats import sync_stats
from src.pipelines.scan_props import scan_props
from src.pipelines.send_alerts import send_alerts
from src.pipelines.settle_results import settle_results
from src.utils.logging_utils import get_logger

logger = get_logger(__name__)


def main():
    parser = argparse.ArgumentParser(description="MLB Player Prop Betting Bot")
    parser.add_argument('command', choices=['sync', 'scan', 'run', 'settle'],
                        help="Command to execute")

    args = parser.parse_args()

    # Always ensure DB is initialized
    init_db()

    try:
        if args.command == 'sync':
            logger.info("Running SYNC mode...")
            sync_events()
            sync_injuries()
            sync_stats()

        elif args.command == 'scan':
            logger.info("Running SCAN mode...")
            scan_props()

        elif args.command == 'run':
            logger.info("Running FULL pipeline (sync -> scan -> alerts)...")
            sync_events()
            sync_injuries()
            # sync_stats() omitted by default (slow, run separately)
            scan_props()
            send_alerts()

        elif args.command == 'settle':
            logger.info("Running SETTLE mode...")
            sync_stats()
            settle_results()

    except Exception as e:
        logger.error(f"Execution failed: {e}", exc_info=True)


if __name__ == "__main__":
    main()
