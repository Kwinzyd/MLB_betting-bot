import json

from src.config import BANKROLL, SHARP_BOOKMAKERS
from src.data.db import get_db_connection
from src.models.devig import devig_multiplicative
from src.models.kelly import get_current_bankroll
from src.utils.logging_utils import get_logger

# Markets that resolve from pitcher_game_logs vs. batter_game_logs.
_PITCHER_MARKETS = {'pitcher_strikeouts', 'pitcher_earned_runs'}
_BATTER_MARKETS = {'batter_hits', 'batter_total_bases', 'batter_home_runs'}

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

    settled_count = 0
    total_profit = 0.0

    if not unsettled:
        logger.info("No unsettled single-bet alerts.")

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
            # Distinguish "BDL hasn't synced this game's box score yet" (skip)
            # from "the box score is in but this player has no row" (DNP void).
            if _player_dnp(player_name, market, alert['game_id']):
                clv = _calculate_clv(
                    alert['game_id'], player_name, market, line, side,
                    alert['opening_odds']
                )
                with get_db_connection() as conn:
                    conn.execute('''
                        INSERT INTO bet_results
                        (alert_id, actual_value, result, profit, closing_odds, clv)
                        VALUES (?, NULL, 'VOID', 0.0, ?, ?)
                    ''', (alert_id, odds, clv))
                    conn.commit()
                settled_count += 1
                logger.info(
                    f"VOIDED (DNP): {player_name} {market} {side.upper()} {line} - "
                    f"stake ${stake:.2f} refunded."
                )
            else:
                logger.debug(
                    f"No actual stat for {player_name} in {market}, "
                    f"box score not yet synced - skipping."
                )
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

    # Settle SGP tickets (handles void recalculation across legs).
    sgp_settled, sgp_profit = _settle_sgps()
    settled_count += sgp_settled
    total_profit += sgp_profit

    # Log summary
    if settled_count > 0:
        _log_pnl_summary()

    logger.info(f"Settlement complete. Settled {settled_count} bets. Session P&L: ${total_profit:+.2f}")


def _player_dnp(player_name: str, market: str, game_id: str) -> bool:
    """
    True iff the box score for this game has been synced but the player has
    no row in the relevant log table. That's our late-scratch / DNP signal.

    Returning False when the box score isn't in yet keeps the settler from
    voiding bets just because BDL is slow.
    """
    if market in _PITCHER_MARKETS:
        log_table = 'pitcher_game_logs'
    elif market in _BATTER_MARKETS:
        log_table = 'batter_game_logs'
    else:
        return False

    with get_db_connection() as conn:
        game = conn.execute(
            "SELECT bdl_game_id FROM games WHERE game_id = ?", (game_id,)
        ).fetchone()
        if not game or not game['bdl_game_id']:
            return False
        bdl_game_id = game['bdl_game_id']

        # Box score must have at least one row for this game; otherwise BDL
        # likely hasn't synced and we shouldn't void prematurely.
        any_rows = conn.execute(
            f"SELECT 1 FROM {log_table} WHERE game_id = ? LIMIT 1",
            (bdl_game_id,)
        ).fetchone()
        if not any_rows:
            return False

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
            # Unknown player — can't confirm DNP, fall through to skip.
            return False

        player_row = conn.execute(
            f"SELECT 1 FROM {log_table} WHERE game_id = ? AND player_id = ? LIMIT 1",
            (bdl_game_id, player['player_id'])
        ).fetchone()
        return player_row is None


def _settle_sgps():
    """
    Grade SGP tickets, recalculating payout when legs void/push.

    Per-leg outcomes: WIN, LOSS, PUSH, VOID, PENDING (box score not in).
    Ticket logic:
      - any PENDING leg -> defer (don't write a row)
      - any LOSS leg    -> ticket loses, profit = -stake
      - all surviving legs WIN -> profit = stake * (recalc_odds - 1),
        where recalc_odds = product of surviving legs' decimal odds
        (PUSH/VOID legs drop out, mirroring how soft books reprice voided SGPs)
      - all legs VOID/PUSH -> ticket fully refunded, profit = 0
    """
    settled = 0
    total_profit = 0.0
    with get_db_connection() as conn:
        rows = conn.execute('''
            SELECT s.id, s.game_id, s.legs_json, s.kelly_stake,
                   s.naive_parlay_odds
            FROM sgp_candidates s
            JOIN games g ON s.game_id = g.game_id
            WHERE g.status = 'COMPLETED'
              AND s.kelly_stake IS NOT NULL
              AND s.kelly_stake > 0
              AND s.id NOT IN (
                  SELECT sgp_candidate_id FROM sgp_results
                  WHERE sgp_candidate_id IS NOT NULL
              )
        ''').fetchall()

    for row in rows:
        try:
            legs = json.loads(row['legs_json'])
        except (TypeError, ValueError):
            logger.warning(f"SGP {row['id']}: malformed legs_json, skipping.")
            continue

        leg_results = []
        defer = False
        for leg in legs:
            outcome = _grade_leg(leg, row['game_id'])
            if outcome == 'PENDING':
                defer = True
                break
            leg_results.append(outcome)

        if defer:
            logger.debug(f"SGP {row['id']}: legs pending, deferring settlement.")
            continue

        stake = float(row['kelly_stake'])
        voided_idx = [i for i, r in enumerate(leg_results) if r in ('VOID', 'PUSH')]
        survivors = [
            (legs[i], leg_results[i]) for i in range(len(legs))
            if i not in voided_idx
        ]

        if any(r == 'LOSS' for _, r in survivors):
            ticket_result = 'LOSS'
            recalc_odds = float(row['naive_parlay_odds'] or 0.0)
            profit = -stake
        elif not survivors:
            # All legs voided/pushed -> full refund.
            ticket_result = 'VOID'
            recalc_odds = 1.0
            profit = 0.0
        else:
            recalc_odds = 1.0
            for leg, _ in survivors:
                recalc_odds *= float(leg['odds'])
            ticket_result = 'WIN'
            profit = stake * (recalc_odds - 1.0)

        with get_db_connection() as conn:
            conn.execute('''
                INSERT INTO sgp_results
                (sgp_candidate_id, leg_results_json, voided_legs_json,
                 surviving_legs, recalc_odds, result, profit, settled_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, datetime('now'))
            ''', (
                row['id'], json.dumps(leg_results), json.dumps(voided_idx),
                len(survivors), round(recalc_odds, 4),
                ticket_result, round(profit, 2),
            ))
            conn.commit()

        settled += 1
        total_profit += profit
        logger.info(
            f"SGP SETTLED: id={row['id']} legs={leg_results} "
            f"voided={voided_idx} -> {ticket_result} @ {recalc_odds:.2f} "
            f"| P&L: ${profit:+.2f}"
        )

    return settled, total_profit


def _grade_leg(leg: dict, game_id: str) -> str:
    """Grade a single SGP leg. Returns WIN/LOSS/PUSH/VOID/PENDING."""
    player = leg.get('player_name')
    market = leg.get('market')
    line = float(leg.get('line'))
    side = leg.get('side')

    actual = _get_actual_stat(player, market, game_id)
    if actual is None:
        if _player_dnp(player, market, game_id):
            return 'VOID'
        return 'PENDING'

    if actual == line:
        return 'PUSH'
    if side == 'over':
        return 'WIN' if actual > line else 'LOSS'
    return 'WIN' if actual < line else 'LOSS'


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
        if closing_over is None:
            # Closing line failed sanity checks; CLV is undefined.
            return 0.0
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
            current_bankroll = get_current_bankroll()
            roi_pct = (summary['total_profit'] / BANKROLL) * 100 if BANKROLL else 0.0
            logger.info(
                f"\n{'=' * 40}\n"
                f"CUMULATIVE P&L SUMMARY\n"
                f"Total Bets: {summary['total_bets']}\n"
                f"Record: {summary['wins']}W - {summary['losses']}L - {summary['pushes']}P "
                f"({win_rate:.1f}%)\n"
                f"Total Profit: ${summary['total_profit']:+.2f}\n"
                f"Avg CLV: {summary['avg_clv']:+.4f}\n"
                f"Starting Bankroll: ${BANKROLL:.2f}\n"
                f"Current Bankroll:  ${current_bankroll:.2f} ({roi_pct:+.1f}% ROI)\n"
                f"{'=' * 40}"
            )
