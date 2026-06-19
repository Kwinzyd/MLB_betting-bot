import argparse
import asyncio
from datetime import datetime
from src.data.db import init_db
from src.pipelines.sync_events import sync_events
from src.pipelines.sync_injuries import sync_injuries
from src.pipelines.sync_stats import sync_stats
from src.pipelines.scan_props import scan_props
from src.pipelines.send_alerts import send_alerts
from src.pipelines.scan_game_markets import scan_game_markets
from src.pipelines.send_game_alerts import send_game_alerts
from src.pipelines.find_sgp import find_and_alert_sgps
from src.pipelines.find_parlays import find_and_alert_parlays
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

    # scan-games (game markets: moneyline / total / run line, off BDL odds)
    subparsers.add_parser('scan-games',
                          help="Scan game markets (moneyline/total/run line) and alert")

    # settle, prune, sgp, trigger, fit-dispersion
    subparsers.add_parser('settle', help="Sync stats and settle completed bets")
    subparsers.add_parser('prune', help="Prune old database records")
    repair_parser = subparsers.add_parser(
        'repair-logs',
        help="One-off repair: IP baseball-notation + empty game-log dates")
    repair_parser.add_argument('--seasons', metavar='YYYY[,YYYY,...]', default=None,
                               help="Also restore missing historical games rows from BDL "
                                    "/games (no stats calls) before the date backfill")
    subparsers.add_parser('sgp', help="Find and alert Same Game Parlays (SGPs)")
    subparsers.add_parser('parlay', help="Find and alert cross-game 2/4/8-leg parlays")
    subparsers.add_parser('trigger', help="Run trigger watch for weather and umpires")
    subparsers.add_parser('fit-dispersion', help="Fit dispersion models for projections")
    subparsers.add_parser('enrich', help="LLM-normalize today's injury reports into availability signals")
    subparsers.add_parser('reconcile', help="LLM-resolve unmatched prop player names into the resolution cache")
    subparsers.add_parser('live', help="Run the continuous Live State Machine daemon for in-game prop sniping")

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
    backtest_parser.add_argument('--use-trained-models', action='store_true',
                                 help="Use the current champion GLM + calibration (IN-SAMPLE; "
                                      "default is the out-of-sample-safe weighted-average model)")

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
    wf_parser.add_argument('--market', action='append', dest='markets', metavar='MARKET',
                           help="Restrict to specific market(s); repeatable")
    wf_parser.add_argument('--model', choices=['glm', 'lgbm'], default='glm',
                           help="Model type to use for walk-forward (default: glm)")
    wf_parser.add_argument('--compare-versions', action='store_true',
                           help="Run both GLM and LGBM and print a side-by-side comparison")
    wf_parser.add_argument('--history', action='store_true',
                           help="Show historical walk-forward results from DB instead of running a new backtest")
    wf_parser.add_argument('--last', type=int, default=5, metavar='N',
                           help="Number of past runs to show per market with --history (default 5)")

    # statcast / calibrate / fit-correlations / monitor-drift
    statcast_parser = subparsers.add_parser(
        'statcast', help="Sync Baseball Savant Statcast leaderboards"
    )
    statcast_parser.add_argument(
        '--season', type=int, default=None,
        help="Season year to sync (default: current year)"
    )
    ask_parser = subparsers.add_parser(
        'ask', help="Ask the LLM research agent a natural-language question over the BDL data")
    ask_parser.add_argument('question', nargs='+', help="The question, e.g. ask how has Cole trended")

    subparsers.add_parser('calibrate', help="Fit Platt/isotonic calibration params per market")
    subparsers.add_parser('fit-correlations',
                          help="Fit empirical portfolio correlations from settled bets")
    subparsers.add_parser('monitor-drift',
                          help="Check model drift; retrain if drift_score > 0.15 or PSI > 0.20")

    args = parser.parse_args()

    # Always ensure DB is initialized
    init_db()

    try:
        if args.command == 'sync':
            logger.info("Running SYNC mode...")
            from src.pipelines.enrich_injuries import enrich_injuries
            asyncio.run(sync_events())
            asyncio.run(sync_injuries())
            asyncio.run(enrich_injuries())  # LLM injury signals (no-op if LLM off)
            asyncio.run(sync_stats())
            asyncio.run(sync_lineups())
            sync_umpires()
            from src.pipelines.sync_statcast import sync_statcast
            asyncio.run(sync_statcast())

        elif args.command == 'scan':
            logger.info("Running SCAN mode...")
            asyncio.run(scan_props(force=args.force))

        elif args.command == 'run':
            logger.info("Running FULL pipeline (sync -> scan -> alerts)...")
            from src.pipelines.enrich_injuries import enrich_injuries
            asyncio.run(sync_events())
            asyncio.run(sync_injuries())
            asyncio.run(enrich_injuries())  # LLM injury signals (no-op if LLM off)
            asyncio.run(sync_lineups())
            sync_umpires()
            # asyncio.run(sync_stats()) omitted by default (slow, run separately)
            asyncio.run(scan_props(force=args.force))
            asyncio.run(send_alerts())
            asyncio.run(find_and_alert_sgps())
            asyncio.run(find_and_alert_parlays())
            # Game markets (no-op unless GAME_MARKETS_ENABLED).
            asyncio.run(scan_game_markets())
            asyncio.run(send_game_alerts())

        elif args.command == 'scan-games':
            logger.info("Running SCAN-GAMES mode (moneyline / total / run line)...")
            asyncio.run(scan_game_markets())
            asyncio.run(send_game_alerts())

        elif args.command == 'sgp':
            logger.info("Running SGP mode...")
            asyncio.run(find_and_alert_sgps())

        elif args.command == 'parlay':
            logger.info("Running PARLAY mode...")
            asyncio.run(find_and_alert_parlays())

        elif args.command == 'trigger':
            logger.info("Running TRIGGER WATCH mode...")
            asyncio.run(run_trigger_watch())

        elif args.command == 'live':
            from src.pipelines.live_state_machine import run_live_state_machine
            logger.info("Running LIVE STATE MACHINE daemon (Ctrl-C to stop)...")
            asyncio.run(run_live_state_machine())

        elif args.command == 'walkforward':
            from src.pipelines.walk_forward import (
                walk_forward_all, print_walk_forward_report,
                compare_walk_forward_market, print_compare_report,
                print_walk_forward_history,
            )
            if args.history:
                print_walk_forward_history(markets=args.markets, last_n=args.last)
            elif args.compare_versions:
                targets = list(args.markets) if args.markets else None
                from src.pipelines.train_model import _ALL_MARKETS
                targets = targets or list(_ALL_MARKETS)
                for m in targets:
                    comparison = compare_walk_forward_market(
                        m,
                        train_window_days=args.train_window,
                        step_days=args.step,
                        mode=args.mode,
                    )
                    print_compare_report(comparison)
            else:
                results = walk_forward_all(
                    markets=args.markets,
                    train_window_days=args.train_window,
                    step_days=args.step,
                    mode=args.mode,
                    model_type=args.model,
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

        elif args.command == 'repair-logs':
            from src.pipelines.repair_game_logs import repair_game_logs
            logger.info("Running REPAIR-LOGS mode...")
            seasons = ([int(s.strip()) for s in args.seasons.split(',') if s.strip()]
                       if args.seasons else None)
            repair_game_logs(seasons=seasons)

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

        elif args.command == 'statcast':
            from src.pipelines.sync_statcast import sync_statcast
            season = args.season or datetime.now().year
            logger.info(f"Running STATCAST sync for season {season}...")
            asyncio.run(sync_statcast(season=season))

        elif args.command == 'ask':
            from src.pipelines.research_agent import ask
            question = ' '.join(args.question)
            answer = asyncio.run(ask(question))
            print(f"\n{answer}\n")

        elif args.command == 'enrich':
            from src.pipelines.enrich_injuries import enrich_injuries
            logger.info("Running ENRICH (LLM injury signals)...")
            asyncio.run(enrich_injuries())

        elif args.command == 'reconcile':
            from src.pipelines.reconcile_names import reconcile_names
            logger.info("Running RECONCILE (LLM name resolution)...")
            asyncio.run(reconcile_names())

        elif args.command == 'calibrate':
            from src.pipelines.calibrate_model import calibrate_all
            logger.info("Running CALIBRATE...")
            calibrate_all()

        elif args.command == 'fit-correlations':
            from src.pipelines.fit_correlations import fit_correlations
            logger.info("Running FIT-CORRELATIONS...")
            fit_correlations()

        elif args.command == 'monitor-drift':
            from src.pipelines.monitor_drift import monitor_drift
            logger.info("Running MONITOR-DRIFT...")
            monitor_drift()

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
                use_trained_models=args.use_trained_models,
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
        tb = ''.join(traceback.format_exception(
            type(exception), exception, exception.__traceback__
        ))
        tail = '\n'.join(tb.strip().splitlines()[-16:])
        msg = (
            f"\U0001F6A8 <b>Pipeline Crash</b>\n"
            f"Command: <code>{command}</code>\n"
            f"<pre>{tail[:800]}</pre>"
        )
        TelegramClient().send_message_sync(msg)
    except Exception as notify_err:
        logger.error(f"Failed to send crash alert: {notify_err}")


if __name__ == "__main__":
    main()
