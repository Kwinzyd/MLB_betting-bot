from src.config import SHARP_BOOKMAKERS
from src.data.db import get_db_connection
from src.models.devig import devig_multiplicative
from src.utils.logging_utils import get_logger

logger = get_logger(__name__)


def settle_results():
    """
    Settle bet results by comparing alerts against actual player stats.
    Calculates profit/loss and CLV (Closing Line Value).
    """
    logger.info("Executing pipeline: settle_results")

    with get_db_connection() as conn:
        # Find unsettled alerts for completed games
        unsettled = conn.execute('''
            SELECT
                a.alert_id, a.player_name, a.market, a.line, a.side,
                a.odds, a.opening_odds, a.kelly_stake, a.game_id
            FROM alerts_sent a
            JOIN games g ON a.game_id = g.game_id
            WHERE g.status = 'COMPLETED'
            AND a.alert_id NOT IN (SELECT alert_id FROM bet_results WHERE alert_id IS NOT NULL)
        ''').fetchall()

    if not unsettled:
        logger.info("No unsettled bets found.")
        return

    settled_count = 0
    total_profit = 0.0

    for alert in unsettled:
        alert_id = alert['alert_id']
        player_name = alert['player_name']
        market = alert['market']
        line = alert['line']
        side = alert['side']
        odds = alert['odds']
        stake = alert['kelly_stake']

        # Get actual stat value
        actual = _get_actual_stat(player_name, market, alert['game_id'])
        if actual is None:
            logger.debug(f"No actual stat found for {player_name} in {market}, skipping.")
            continue

        # Determine result
        if side == 'over':
            if actual > line:
                result = 'WIN'
                profit = stake * (odds - 1)
            elif actual == line:
                result = 'PUSH'
                profit = 0.0
            else:
                result = 'LOSS'
                profit = -stake
        else:  # under
            if actual < line:
                result = 'WIN'
                profit = stake * (odds - 1)
            elif actual == line:
                result = 'PUSH'
                profit = 0.0
            else:
                result = 'LOSS'
                profit = -stake

        # Calculate CLV (simplified: compare opening odds to closing snapshot)
        clv = _calculate_clv(alert['game_id'], player_name, market, line, side, alert['opening_odds'])

        with get_db_connection() as conn:
            conn.execute('''
                INSERT INTO bet_results (alert_id, actual_value, result, profit, closing_odds, clv)
                VALUES (?, ?, ?, ?, ?, ?)
            ''', (alert_id, actual, result, round(profit, 2), odds, clv))
            conn.commit()

        total_profit += profit
        settled_count += 1
        logger.info(
            f"SETTLED: {player_name} {market} {side.upper()} {line} - "
            f"Actual: {actual} | {result} | P&L: ${profit:+.2f} | CLV: {clv:+.3f}"
        )

    # Log summary
    if settled_count > 0:
        _log_pnl_summary()

    logger.info(f"Settlement complete. Settled {settled_count} bets. Session P&L: ${total_profit:+.2f}")


def _get_actual_stat(player_name: str, market: str, game_id: str) -> float:
    """Look up the actual stat value from game logs."""
    with get_db_connection() as conn:
        # Find player
        player = conn.execute(
            "SELECT player_id FROM players WHERE name = ? COLLATE NOCASE",
            (player_name,)
        ).fetchone()

        if not player:
            player = conn.execute(
                "SELECT player_id FROM players WHERE name LIKE ? COLLATE NOCASE",
                (f"%{player_name}%",)
            ).fetchone()

        if not player:
            return None

        player_id = player['player_id']

        # Find the BDL game ID for this Odds API game
        game = conn.execute(
            "SELECT bdl_game_id FROM games WHERE game_id = ?", (game_id,)
        ).fetchone()

        if not game or not game['bdl_game_id']:
            return None

        bdl_game_id = game['bdl_game_id']

        if market == 'pitcher_strikeouts':
            row = conn.execute(
                "SELECT strikeouts FROM pitcher_game_logs WHERE player_id = ? AND game_id = ?",
                (player_id, bdl_game_id)
            ).fetchone()
            return float(row['strikeouts']) if row else None

        elif market == 'pitcher_earned_runs':
            row = conn.execute(
                "SELECT earned_runs FROM pitcher_game_logs WHERE player_id = ? AND game_id = ?",
                (player_id, bdl_game_id)
            ).fetchone()
            return float(row['earned_runs']) if row else None

        elif market == 'batter_hits':
            row = conn.execute(
                "SELECT hits FROM batter_game_logs WHERE player_id = ? AND game_id = ?",
                (player_id, bdl_game_id)
            ).fetchone()
            return float(row['hits']) if row else None

        elif market == 'batter_total_bases':
            row = conn.execute(
                "SELECT total_bases FROM batter_game_logs WHERE player_id = ? AND game_id = ?",
                (player_id, bdl_game_id)
            ).fetchone()
            return float(row['total_bases']) if row else None

        elif market == 'batter_home_runs':
            row = conn.execute(
                "SELECT home_runs FROM batter_game_logs WHERE player_id = ? AND game_id = ?",
                (player_id, bdl_game_id)
            ).fetchone()
            return float(row['home_runs']) if row else None

    return None


def _calculate_clv(game_id: str, player_name: str, market: str,
                   line: float, side: str, opening_odds: float) -> float:
    """
    Calculate Closing Line Value against a sharp book's devigged closing line.

    Source of truth preference: the most recent snapshot whose bookmaker matches
    SHARP_BOOKMAKERS (walked in priority order). Soft books (DK/FD/etc.) carry
    liability-adjusted closes and are only used when no sharp snapshot exists.

    CLV = devigged_closing_implied(sharp) - opening_implied
    Positive = we beat the sharp close.
    """
    with get_db_connection() as conn:
        closing = None
        for sharp_book in SHARP_BOOKMAKERS:
            row = conn.execute('''
                SELECT over_odds, under_odds FROM prop_snapshots
                WHERE game_id = ? AND player_name = ? AND market = ?
                  AND line = ? AND bookmaker = ?
                ORDER BY timestamp DESC LIMIT 1
            ''', (game_id, player_name, market, line, sharp_book)).fetchone()
            if row and row['over_odds'] and row['under_odds']:
                closing = row
                break

        if closing is None:
            # Fallback: latest snapshot regardless of book. Noisier — soft closes
            # include liability adjustments — but better than returning 0.
            logger.debug(
                f"No sharp-book snapshot for {player_name} {market} {line}; "
                f"falling back to latest any-book close."
            )
            closing = conn.execute('''
                SELECT over_odds, under_odds FROM prop_snapshots
                WHERE game_id = ? AND player_name = ? AND market = ? AND line = ?
                ORDER BY timestamp DESC LIMIT 1
            ''', (game_id, player_name, market, line)).fetchone()

        if not closing or not closing['over_odds'] or not closing['under_odds']:
            return 0.0

        closing_over, closing_under = devig_multiplicative(
            closing['over_odds'], closing['under_odds']
        )
        opening_implied = 1.0 / opening_odds
        closing_implied = closing_over if side == 'over' else closing_under
        return round(closing_implied - opening_implied, 4)


def _log_pnl_summary():
    """Log cumulative P&L summary."""
    with get_db_connection() as conn:
        summary = conn.execute('''
            SELECT
                COUNT(*) as total_bets,
                SUM(CASE WHEN result = 'WIN' THEN 1 ELSE 0 END) as wins,
                SUM(CASE WHEN result = 'LOSS' THEN 1 ELSE 0 END) as losses,
                SUM(CASE WHEN result = 'PUSH' THEN 1 ELSE 0 END) as pushes,
                SUM(profit) as total_profit,
                AVG(clv) as avg_clv
            FROM bet_results
        ''').fetchone()

        if summary and summary['total_bets'] > 0:
            win_rate = (summary['wins'] / summary['total_bets']) * 100
            logger.info(
                f"\n{'=' * 40}\n"
                f"CUMULATIVE P&L SUMMARY\n"
                f"Total Bets: {summary['total_bets']}\n"
                f"Record: {summary['wins']}W - {summary['losses']}L - {summary['pushes']}P "
                f"({win_rate:.1f}%)\n"
                f"Total Profit: ${summary['total_profit']:+.2f}\n"
                f"Avg CLV: {summary['avg_clv']:+.4f}\n"
                f"{'=' * 40}"
            )
