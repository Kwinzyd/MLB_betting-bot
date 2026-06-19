import json
from datetime import datetime, timezone

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
                a.odds, a.opening_odds, a.kelly_stake, a.game_id,
                a.model_prob_over, a.model_prob_under,
                a.player_id, a.open_devig_prob
            FROM alerts_sent a
            JOIN games g ON a.game_id = g.game_id
            WHERE g.status = 'COMPLETED'
            AND a.alert_id NOT IN (SELECT alert_id FROM bet_results WHERE alert_id IS NOT NULL)
        ''').fetchall()

    settled_count = 0
    total_profit = 0.0
    settled_game_ids: set = set()

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
        player_id = alert['player_id']
        open_devig_prob = alert['open_devig_prob']

        # Get actual stat value — graded by player_id when the alert carries
        # one (name lookups can hit the wrong player on duplicate MLB names).
        actual = _get_actual_stat(player_name, market, alert['game_id'],
                                  player_id=player_id)
        if actual is None:
            # Distinguish "BDL hasn't synced this game's box score yet" (skip)
            # from "the box score is in but this player has no row" (DNP void).
            if _player_dnp(player_name, market, alert['game_id'],
                           player_id=player_id):
                clv = _calculate_clv(
                    alert['game_id'], player_name, market, line, side,
                    alert['opening_odds'], opening_prob=open_devig_prob
                )
                with get_db_connection() as conn:
                    conn.execute('''
                        INSERT INTO bet_results
                        (alert_id, actual_value, result, profit, closing_odds, clv, settled_at)
                        VALUES (?, NULL, 'VOID', 0.0, ?, ?, datetime('now'))
                    ''', (alert_id, odds, clv))
                    conn.commit()
                settled_count += 1
                settled_game_ids.add(alert['game_id'])
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

        # Calculate CLV: devigged close vs devigged open when available.
        clv = _calculate_clv(alert['game_id'], player_name, market, line, side,
                             alert['opening_odds'], opening_prob=open_devig_prob)

        with get_db_connection() as conn:
            conn.execute('''
                INSERT INTO bet_results
                (alert_id, actual_value, result, profit, closing_odds, clv, settled_at)
                VALUES (?, ?, ?, ?, ?, ?, datetime('now'))
            ''', (alert_id, actual, result, round(profit, 2), odds, clv))
            if result in ('WIN', 'LOSS'):
                _write_calibration_log(conn, alert_id, market, side, result,
                                       alert['model_prob_over'], alert['model_prob_under'])
            conn.commit()

        total_profit += profit
        settled_count += 1
        settled_game_ids.add(alert['game_id'])
        logger.info(
            f"SETTLED: {player_name} {market} {side.upper()} {line} - "
            f"Actual: {actual} | {result} | P&L: ${profit:+.2f} | CLV: {clv:+.3f}"
        )

    # Void (refund) open singles on postponed games before parlay settlement so
    # capital isn't left tied up in a bet that can never resolve.
    postponed_voided = _void_postponed_singles()
    settled_count += postponed_voided

    # Settle SGP tickets (handles void recalculation across legs).
    sgp_settled, sgp_profit = _settle_sgps()
    settled_count += sgp_settled
    total_profit += sgp_profit

    # Settle cross-game parlay tickets (each leg graded against its own game).
    parlay_settled, parlay_profit = _settle_parlays()
    settled_count += parlay_settled
    total_profit += parlay_profit

    # Log summary and update bankroll snapshot
    if settled_count > 0:
        _log_pnl_summary()

    _emit_pair_outcomes(settled_game_ids)
    _update_bankroll_snapshot(total_profit)
    logger.info(f"Settlement complete. Settled {settled_count} bets. Session P&L: ${total_profit:+.2f}")


def _player_dnp(player_name: str, market: str, game_id: str,
                player_id: int = None) -> bool:
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

        if player_id is None:
            player_id = _lookup_player_id(conn, player_name)
        if player_id is None:
            # Unknown player — can't confirm DNP, fall through to skip.
            return False

        player_row = conn.execute(
            f"SELECT 1 FROM {log_table} WHERE game_id = ? AND player_id = ? LIMIT 1",
            (bdl_game_id, player_id)
        ).fetchone()
        return player_row is None


def _lookup_player_id(conn, player_name: str):
    """Resolve a player_id from a name for settlement.

    Uses the deterministic resolver (cache -> exact -> accent/suffix-normalized),
    which also surfaces any LLM-reconciled mapping. Only when that fails does it
    fall back to a logged fuzzy LIKE — the path that can mis-grade duplicate
    names — so the fallback firing is a visible signal, not a silent risk.
    """
    from src.data.player_resolver import resolve_player_id
    pid, method = resolve_player_id(conn, player_name)
    if pid is not None:
        return pid
    if method == "ambiguous":
        logger.warning("Settlement: '%s' is ambiguous (multiple players) — not grading by guess.",
                       player_name)
        return None
    player = conn.execute(
        "SELECT player_id FROM players WHERE name LIKE ? COLLATE NOCASE",
        (f"%{player_name}%",)
    ).fetchone()
    if player:
        logger.warning(
            "Settlement resolved '%s' via fuzzy name match — verify identity.",
            player_name,
        )
        return player['player_id']
    return None


def _is_postponed(game_id: str) -> bool:
    """True iff this game is marked POSTPONED (set by sync_stats)."""
    with get_db_connection() as conn:
        row = conn.execute(
            "SELECT status FROM games WHERE game_id = ?", (game_id,)
        ).fetchone()
    return bool(row and row['status'] == 'POSTPONED')


def _void_postponed_singles() -> int:
    """Void (refund, profit 0) open single-bet alerts on postponed games.

    A postponed game never reaches COMPLETED, so its bets would otherwise sit
    PENDING forever and tie up modeled exposure. Books void unplayed props.
    """
    voided = 0
    with get_db_connection() as conn:
        rows = conn.execute('''
            SELECT a.alert_id, a.odds
            FROM alerts_sent a
            JOIN games g ON a.game_id = g.game_id
            WHERE g.status = 'POSTPONED'
              AND a.alert_id NOT IN (
                  SELECT alert_id FROM bet_results WHERE alert_id IS NOT NULL
              )
        ''').fetchall()
        for r in rows:
            conn.execute('''
                INSERT INTO bet_results
                (alert_id, actual_value, result, profit, closing_odds, clv, settled_at)
                VALUES (?, NULL, 'VOID', 0.0, ?, 0.0, datetime('now'))
            ''', (r['alert_id'], r['odds']))
            voided += 1
        if voided:
            conn.commit()
    if voided:
        logger.info("Voided %d single bet(s) on postponed games (refund).", voided)
    return voided


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
            WHERE g.status IN ('COMPLETED', 'POSTPONED')
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


def _settle_parlays():
    """
    Grade cross-game parlay tickets (find_parlays.py), recalculating payout
    when legs void/push. Same ticket logic as _settle_sgps, but each leg is
    graded against its OWN game_id (the legs span different games), so a ticket
    only settles once every one of its games' box scores is in.

    Per-leg outcomes: WIN, LOSS, PUSH, VOID, PENDING (box score not in).
    Ticket logic:
      - any PENDING leg (or a leg missing its game_id) -> defer
      - any surviving LOSS leg -> ticket loses, profit = -stake
      - all legs VOID/PUSH     -> full refund, profit = 0
      - all survivors WIN      -> profit = stake * (recalc_odds - 1),
        where recalc_odds = product of surviving legs' decimal odds
    """
    settled = 0
    total_profit = 0.0
    with get_db_connection() as conn:
        rows = conn.execute('''
            SELECT id, legs_json, kelly_stake, parlay_odds
            FROM parlays
            WHERE kelly_stake IS NOT NULL
              AND kelly_stake > 0
              AND id NOT IN (
                  SELECT parlay_id FROM parlay_results
                  WHERE parlay_id IS NOT NULL
              )
        ''').fetchall()

    for row in rows:
        try:
            legs = json.loads(row['legs_json'])
        except (TypeError, ValueError):
            logger.warning(f"Parlay {row['id']}: malformed legs_json, skipping.")
            continue

        leg_results = []
        defer = False
        for leg in legs:
            game_id = leg.get('game_id')
            if not game_id:
                # Without a game_id we can't grade this leg; defer rather than
                # silently mis-settle. (find_parlays always writes game_id.)
                logger.warning(f"Parlay {row['id']}: leg missing game_id, deferring.")
                defer = True
                break
            outcome = _grade_leg(leg, game_id)
            if outcome == 'PENDING':
                defer = True
                break
            leg_results.append(outcome)

        if defer:
            logger.debug(f"Parlay {row['id']}: legs pending, deferring settlement.")
            continue

        stake = float(row['kelly_stake'])
        voided_idx = [i for i, r in enumerate(leg_results) if r in ('VOID', 'PUSH')]
        survivors = [
            (legs[i], leg_results[i]) for i in range(len(legs))
            if i not in voided_idx
        ]

        if any(r == 'LOSS' for _, r in survivors):
            ticket_result = 'LOSS'
            recalc_odds = float(row['parlay_odds'] or 0.0)
            profit = -stake
        elif not survivors:
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
                INSERT INTO parlay_results
                (parlay_id, leg_results_json, voided_legs_json,
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
            f"PARLAY SETTLED: id={row['id']} legs={leg_results} "
            f"voided={voided_idx} -> {ticket_result} @ {recalc_odds:.2f} "
            f"| P&L: ${profit:+.2f}"
        )

    return settled, total_profit


def _grade_leg(leg: dict, game_id: str) -> str:
    """Grade a single SGP/parlay leg. Returns WIN/LOSS/PUSH/VOID/PENDING."""
    player = leg.get('player_name')
    market = leg.get('market')
    line = float(leg.get('line'))
    side = leg.get('side')

    # A postponed game's leg can never resolve — void it so the ticket reprices
    # on the surviving legs (cross-game parlay) or refunds (all-postponed SGP).
    if _is_postponed(game_id):
        return 'VOID'

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


def _get_actual_stat(player_name: str, market: str, game_id: str,
                     player_id: int = None) -> float:
    """Look up the actual stat value from game logs.

    Grades by player_id when provided (captured at scan time); name lookup is
    a legacy fallback only.
    """
    with get_db_connection() as conn:
        if player_id is None:
            player_id = _lookup_player_id(conn, player_name)
        if player_id is None:
            return None

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
                   line: float, side: str, opening_odds: float,
                   opening_prob: float = None) -> float:
    """
    Calculate Closing Line Value against a sharp book's devigged closing line.

    Source of truth preference: the most recent snapshot whose bookmaker matches
    SHARP_BOOKMAKERS (walked in priority order). Soft books (DK/FD/etc.) carry
    liability-adjusted closes and are only used when no sharp snapshot exists.

    CLV = devigged_closing - devigged_opening. Both ends must be vig-free:
    `opening_prob` is the devigged open stored on the alert. The raw
    1/opening_odds fallback (vig included) only fires for legacy alerts and
    biases CLV negative by ~half the hold.
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
        if opening_prob is not None and 0.0 < opening_prob < 1.0:
            opening_implied = opening_prob
        else:
            # Legacy fallback (vig included) — biased low, kept only so old
            # alerts without open_devig_prob still settle.
            opening_implied = 1.0 / opening_odds
        closing_implied = closing_over if side == 'over' else closing_under
        return round(closing_implied - opening_implied, 4)


def _classify_pair(a: dict, b: dict) -> str:
    """Return pair_type for two bets from the same game."""
    a_type = "pitcher" if a["market"].startswith("pitcher_") else "batter"
    b_type = "pitcher" if b["market"].startswith("pitcher_") else "batter"
    if a["player_team"] == b["player_team"]:
        return "pitcher_batter" if a_type != b_type else "same_team_batters"
    return "same_game_opp"


def _emit_pair_outcomes(game_ids=None) -> None:
    """Record pairwise joint outcomes for correlation learning (bet_pair_outcomes table).

    Scopes to the games settled in this run (`game_ids`) rather than to bets
    *placed* today: a bet placed before a night game routinely settles after the
    next UTC midnight, so the old `date(a.timestamp) = today` filter silently
    dropped exactly the same-game pairs correlation learning needs. INSERT OR
    IGNORE on the (alert_id_a, alert_id_b) key keeps re-runs idempotent.
    """
    from itertools import combinations
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")

    game_ids = list(game_ids) if game_ids else []
    if not game_ids:
        return

    placeholders = ",".join("?" for _ in game_ids)
    with get_db_connection() as conn:
        rows = [dict(r) for r in conn.execute(
            f"""SELECT br.alert_id, a.game_id, a.market, br.result,
                      COALESCE(
                          (SELECT t.abbreviation FROM players pl
                           JOIN teams t ON pl.team_id = t.team_id
                           WHERE pl.name = a.player_name LIMIT 1),
                          ''
                      ) AS player_team
               FROM bet_results br
               JOIN alerts_sent a ON br.alert_id = a.alert_id
               WHERE a.game_id IN ({placeholders})
                 AND br.result IN ('WIN', 'LOSS')""",
            game_ids,
        ).fetchall()]

        # Group by game_id and emit a row for every pair
        by_game: dict = {}
        for r in rows:
            by_game.setdefault(r["game_id"], []).append(r)

        inserted = 0
        for game_id, bets in by_game.items():
            if len(bets) < 2:
                continue
            for a, b in combinations(bets, 2):
                pair_type = _classify_pair(a, b)
                try:
                    conn.execute(
                        """INSERT OR IGNORE INTO bet_pair_outcomes
                           (alert_id_a, alert_id_b, pair_type,
                            both_won, a_won, b_won, game_id, settled_date)
                           VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                        (
                            a["alert_id"], b["alert_id"], pair_type,
                            int(a["result"] == "WIN" and b["result"] == "WIN"),
                            int(a["result"] == "WIN"),
                            int(b["result"] == "WIN"),
                            game_id, today,
                        ),
                    )
                    inserted += 1
                except Exception as e:
                    logger.debug("bet_pair_outcomes insert failed: %s", e)
        if inserted:
            conn.commit()
            logger.info("Emitted %d bet pair outcome records.", inserted)


def _write_calibration_log(conn, alert_id: int, market: str, side: str,
                           result: str, model_prob_over, model_prob_under) -> None:
    """Append a resolved bet to calibration_log for probability calibration fitting."""
    predicted_prob = model_prob_over if side == 'over' else model_prob_under
    if predicted_prob is None:
        return
    prob_bin = round(round(predicted_prob / 0.05) * 0.05, 2)
    actual_outcome = 1 if result == 'WIN' else 0
    settled_at = datetime.now(timezone.utc).isoformat()
    try:
        conn.execute(
            """INSERT OR IGNORE INTO calibration_log
               (alert_id, market, predicted_prob, prob_bin, actual_outcome, settled_at)
               VALUES (?, ?, ?, ?, ?, ?)""",
            (alert_id, market, predicted_prob, prob_bin, actual_outcome, settled_at),
        )
    except Exception as e:
        logger.warning("calibration_log insert failed: %s", e)


def _send_circuit_breaker_alert(message: str) -> None:
    """Best-effort Telegram alert for circuit breaker events."""
    logger.warning("Circuit breaker: %s", message)
    try:
        from src.clients.telegram_bot import TelegramClient
        TelegramClient().send_message_sync(f"⚠️ <b>Circuit Breaker</b>\n{message}")
    except Exception as e:
        logger.error("Circuit breaker Telegram alert failed: %s", e)


def _update_bankroll_snapshot(session_profit: float) -> None:
    """Write today's bankroll snapshot and evaluate circuit breakers."""
    today = datetime.now(timezone.utc).strftime('%Y-%m-%d')
    now_iso = datetime.now(timezone.utc).isoformat()

    with get_db_connection() as conn:
        try:
            # Daily stop-loss keys on SETTLEMENT date, not placement date: a bet
            # placed for a night game settles after the next UTC midnight, so a
            # placement-dated sum couldn't see the slate it was meant to protect
            # (and would charge the loss to the wrong day).
            daily_row = conn.execute(
                """SELECT COALESCE(SUM(profit), 0.0) AS daily_pnl
                   FROM bet_results
                   WHERE date(settled_at) = ?""",
                (today,),
            ).fetchone()
            daily_pnl = float(daily_row['daily_pnl']) if daily_row else 0.0
            # SGP P&L counts toward the same daily-loss breaker (by settlement date).
            sgp_daily = conn.execute(
                """SELECT COALESCE(SUM(profit), 0.0) AS pnl
                   FROM sgp_results
                   WHERE date(settled_at) = ?""",
                (today,),
            ).fetchone()
            daily_pnl += float(sgp_daily['pnl'] or 0.0) if sgp_daily else 0.0
            # Cross-game parlay P&L counts toward the same daily-loss breaker.
            parlay_daily = conn.execute(
                """SELECT COALESCE(SUM(profit), 0.0) AS pnl
                   FROM parlay_results
                   WHERE date(settled_at) = ?""",
                (today,),
            ).fetchone()
            daily_pnl += float(parlay_daily['pnl'] or 0.0) if parlay_daily else 0.0
        except Exception:
            daily_pnl = session_profit

        try:
            roll_row = conn.execute(
                """SELECT COALESCE(SUM(br.profit), 0.0) AS pnl,
                          COALESCE(SUM(a.kelly_stake), 0.0) AS staked
                   FROM bet_results br
                   JOIN alerts_sent a ON br.alert_id = a.alert_id
                   WHERE date(a.timestamp) >= date('now', '-7 days')""",
            ).fetchone()
            rolling_7d_pnl = float(roll_row['pnl'] or 0.0)
            staked_7d = float(roll_row['staked'] or 0.0)
            sgp_roll = conn.execute(
                """SELECT COALESCE(SUM(sr.profit), 0.0) AS pnl,
                          COALESCE(SUM(sc.kelly_stake), 0.0) AS staked
                   FROM sgp_results sr
                   JOIN sgp_candidates sc ON sr.sgp_candidate_id = sc.id
                   WHERE date(sc.timestamp) >= date('now', '-7 days')""",
            ).fetchone()
            if sgp_roll:
                rolling_7d_pnl += float(sgp_roll['pnl'] or 0.0)
                staked_7d += float(sgp_roll['staked'] or 0.0)
            parlay_roll = conn.execute(
                """SELECT COALESCE(SUM(pr.profit), 0.0) AS pnl,
                          COALESCE(SUM(p.kelly_stake), 0.0) AS staked
                   FROM parlay_results pr
                   JOIN parlays p ON pr.parlay_id = p.id
                   WHERE date(p.timestamp) >= date('now', '-7 days')""",
            ).fetchone()
            if parlay_roll:
                rolling_7d_pnl += float(parlay_roll['pnl'] or 0.0)
                staked_7d += float(parlay_roll['staked'] or 0.0)
            rolling_7d_roi = rolling_7d_pnl / staked_7d if staked_7d > 0 else 0.0
        except Exception:
            rolling_7d_pnl = 0.0
            rolling_7d_roi = 0.0

        try:
            stats = conn.execute(
                """SELECT COUNT(*) as total_bets,
                          SUM(CASE WHEN result='WIN' THEN 1 ELSE 0 END) as total_wins
                   FROM bet_results WHERE result IN ('WIN', 'LOSS')""",
            ).fetchone()
            total_bets = int(stats['total_bets'] or 0)
            total_wins = int(stats['total_wins'] or 0)
        except Exception:
            total_bets = total_wins = 0

        current_bankroll = get_current_bankroll()

        # Carry forward existing override state for today if already written
        kelly_override = 1.0
        full_stop = 0
        halved_at = None
        try:
            existing = conn.execute(
                "SELECT kelly_fraction_override, halved_at, full_stop "
                "FROM bankroll_snapshots WHERE snapshot_date=?",
                (today,),
            ).fetchone()
            if existing:
                kelly_override = float(existing['kelly_fraction_override'] or 1.0)
                halved_at = existing['halved_at']
                full_stop = int(existing['full_stop'] or 0)
        except Exception:
            pass

        # Breaker 1: 7-day ROI < -15% → halve Kelly
        if rolling_7d_roi < -0.15 and kelly_override == 1.0 and full_stop == 0:
            kelly_override = 0.5
            halved_at = now_iso
            _send_circuit_breaker_alert(
                f"7-day ROI {rolling_7d_roi:.1%} below -15%. Kelly halved to 0.5."
            )
        elif rolling_7d_roi > -0.05 and kelly_override < 1.0 and full_stop == 0:
            kelly_override = 1.0
            halved_at = None
            _send_circuit_breaker_alert("7-day ROI recovered above -5%. Kelly restored to 1.0.")

        # Breaker 2: daily P&L < -5% of starting bankroll → full stop
        if daily_pnl < -(BANKROLL * 0.05) and full_stop == 0:
            full_stop = 1
            _send_circuit_breaker_alert(
                f"Daily P&L ${daily_pnl:+.2f} exceeds -5% of bankroll. FULL STOP activated."
            )

        try:
            conn.execute(
                """INSERT INTO bankroll_snapshots
                   (snapshot_date, bankroll, daily_pnl, rolling_7d_pnl, rolling_7d_roi,
                    total_bets, total_wins, kelly_fraction_override, halved_at, full_stop)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                   ON CONFLICT(snapshot_date) DO UPDATE SET
                       bankroll=excluded.bankroll,
                       daily_pnl=excluded.daily_pnl,
                       rolling_7d_pnl=excluded.rolling_7d_pnl,
                       rolling_7d_roi=excluded.rolling_7d_roi,
                       total_bets=excluded.total_bets,
                       total_wins=excluded.total_wins,
                       kelly_fraction_override=excluded.kelly_fraction_override,
                       halved_at=COALESCE(excluded.halved_at, bankroll_snapshots.halved_at),
                       full_stop=excluded.full_stop""",
                (
                    today, round(current_bankroll, 2), round(daily_pnl, 2),
                    round(rolling_7d_pnl, 2), round(rolling_7d_roi, 4),
                    total_bets, total_wins, kelly_override, halved_at, full_stop,
                ),
            )
            conn.commit()
        except Exception as e:
            logger.error("Failed to write bankroll_snapshots: %s", e)

    logger.info(
        "Bankroll snapshot: $%.2f | daily=%+.2f | 7d_roi=%.1f%% | "
        "kelly_override=%.1f | full_stop=%d",
        current_bankroll, daily_pnl, rolling_7d_roi * 100, kelly_override, full_stop,
    )


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
