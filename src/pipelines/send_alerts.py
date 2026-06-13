"""Dispatch scan_props' winning bet candidates to execution venues.

This pipeline does NOT re-derive bets. scan_props makes the edge decision
(sharp anchor, best soft book, steam, bias) and persists it to bet_candidates;
send_alerts consumes only fresh candidate rows, applies portfolio-level Kelly
sizing AFTER exposure dedupe, and records every selected bet in alerts_sent —
including shadow-mode (BETTING_ENABLED=false) runs, flagged delivered=0, so
the paper-trading window produces real settlement/CLV/calibration data.
"""
import json
from datetime import timedelta

from src.clients.execution import build_venue_registry
from src.utils.time_utils import utcnow
from src.data.db import get_db_connection
from src.config import (
    MAX_BETS_PER_GAME, MAX_BETS_PER_PLAYER, BETTING_ENABLED,
    ALERT_CANDIDATE_MAX_AGE_MINUTES,
)
from src.models.kelly import get_current_bankroll
from src.models.portfolio_kelly import apply_portfolio_kelly
from src.utils.logging_utils import get_logger
from src.utils.stake_rounding import round_stake

logger = get_logger(__name__)


def _load_bankroll_snapshot() -> dict:
    """Return today's bankroll_snapshots row as a plain dict, or empty dict on any error."""
    try:
        from datetime import datetime, timezone
        today = datetime.now(timezone.utc).strftime('%Y-%m-%d')
        with get_db_connection() as conn:
            row = conn.execute(
                "SELECT full_stop, kelly_fraction_override FROM bankroll_snapshots "
                "WHERE snapshot_date=?", (today,)
            ).fetchone()
            return dict(row) if row else {}
    except Exception:
        return {}


def _load_fresh_candidates() -> list[dict]:
    """Fresh bet_candidates rows joined to game context and projection context.

    Freshness gate: candidates older than ALERT_CANDIDATE_MAX_AGE_MINUTES are
    stale quotes — the book has likely moved — so they are never alerted.
    """
    cutoff = (utcnow() - timedelta(minutes=ALERT_CANDIDATE_MAX_AGE_MINUTES)).isoformat()
    with get_db_connection() as conn:
        rows = [dict(r) for r in conn.execute('''
            SELECT
                bc.*,
                g.home_team, g.away_team, g.venue AS game_venue, g.date AS game_date,
                p.context_json, p.projected_mean,
                COALESCE(
                    (SELECT t.abbreviation FROM players pl
                     JOIN teams t ON pl.team_id = t.team_id
                     WHERE pl.player_id = bc.player_id),
                    (SELECT t.abbreviation FROM players pl
                     JOIN teams t ON pl.team_id = t.team_id
                     WHERE pl.name = bc.player_name LIMIT 1),
                    g.home_team
                ) AS player_team
            FROM bet_candidates bc
            JOIN games g ON bc.game_id = g.game_id
            LEFT JOIN projections p ON
                p.game_id = bc.game_id AND
                p.player_name = bc.player_name AND
                p.market = bc.market
            WHERE bc.created_at >= ?
              AND g.status NOT IN ('COMPLETED', 'IN_PROGRESS')
            ORDER BY bc.edge_pct DESC
        ''', (cutoff,)).fetchall()]
    return rows


def _build_candidate(row: dict) -> dict:
    """Rebuild the candidate structure venues and portfolio Kelly expect."""
    ctx_data = {}
    if row.get('context_json'):
        try:
            ctx_data = json.loads(row['context_json'])
        except json.JSONDecodeError:
            pass

    side = row['side']
    model_prob = row['model_prob']
    truth_prob = row['truth_prob']
    odds = float(row['odds'])

    edge = {
        'edge_pct': row['edge_pct'],
        'ev': row['ev'],
        'model_prob': model_prob,
        'sharp_prob': truth_prob,
        'book_implied': round(1.0 / odds, 4) if odds > 1.0 else None,
        # Only playable winners are ever persisted to bet_candidates.
        'is_playable': True,
        'steam_detected': bool(row.get('steam_detected')),
        'kelly': {
            'kelly_fraction': row['kelly_fraction'],
            'recommended_stake': row['recommended_stake'],
        },
    }
    # Probabilities keyed by side for portfolio Kelly (truth basis — the same
    # probability the edge and per-bet Kelly were computed from).
    prob_over = truth_prob if side == 'over' else (1.0 - truth_prob if truth_prob is not None else None)
    projection = {
        'prob_over': prob_over,
        'prob_under': (1.0 - prob_over) if prob_over is not None else None,
        'projected_mean': row.get('projected_mean'),
    }
    return {
        'row': row,
        'projection': projection,
        'side': side,
        'odds': odds,
        'edge': edge,
        'ctx_data': ctx_data,
    }


async def send_alerts():
    """Dispatch fresh scan candidates to every enabled execution venue."""
    logger.info("Executing pipeline: send_alerts")

    snapshot = _load_bankroll_snapshot()
    if snapshot.get('full_stop'):
        logger.warning("send_alerts: FULL STOP active — no alerts sent today.")
        return
    kelly_override = float(snapshot.get('kelly_fraction_override') or 1.0)

    venues = build_venue_registry()
    if not venues:
        logger.warning("No execution venues enabled; send_alerts is a no-op.")
        return

    rows = _load_fresh_candidates()
    if not rows:
        logger.info("No fresh bet candidates found for alerting.")
        return

    # ---- Exposure dedupe BEFORE sizing, so portfolio Kelly only sees the
    # bets that will actually be placed (a slate inflated with duplicates
    # would undersize the survivors). ----
    seen_prop_key: set = set()
    seen_players: dict = {}
    game_counts: dict = {}
    selected: list[dict] = []

    with get_db_connection() as conn:
        for row in rows:
            prop_key = (row['player_name'], row['market'], row['line'])
            if prop_key in seen_prop_key:
                continue
            if seen_players.get(row['player_name'], 0) >= MAX_BETS_PER_PLAYER:
                continue
            if game_counts.get(row['game_id'], 0) >= MAX_BETS_PER_GAME:
                continue
            existing = conn.execute(
                "SELECT 1 FROM alerts_sent WHERE player_name=? AND market=? AND line=? AND bookmaker=? AND game_id=?",
                (row['player_name'], row['market'], row['line'], row['bookmaker'], row['game_id'])
            ).fetchone()
            if existing:
                continue
            seen_prop_key.add(prop_key)
            seen_players[row['player_name']] = seen_players.get(row['player_name'], 0) + 1
            game_counts[row['game_id']] = game_counts.get(row['game_id'], 0) + 1
            selected.append(_build_candidate(row))

    if not selected:
        logger.info("All fresh candidates were duplicates or already alerted.")
        return

    # ---- Portfolio-level Kelly on the final slate ----
    bankroll = get_current_bankroll()
    selected = apply_portfolio_kelly(selected, bankroll)
    if kelly_override != 1.0:
        for c in selected:
            k = c['edge']['kelly']
            k['recommended_stake'] = round(k['recommended_stake'] * kelly_override, 2)
    logger.info(
        "Portfolio Kelly applied to %d candidates (bankroll=$%.2f, override=%.2f).",
        len(selected), bankroll, kelly_override,
    )

    alerts_sent_count = 0
    for cand in selected:
        row = cand['row']
        with get_db_connection() as conn:
            display_stake = round_stake(cand['edge']['kelly']['recommended_stake'])
            cand['edge']['kelly']['recommended_stake'] = display_stake

            execution_context = {
                'player_name': row['player_name'],
                'market': row['market'],
                'side': cand['side'],
                'line': row['line'],
                'odds': cand['odds'],
                'bookmaker': row['bookmaker'],
                'game_id': row['game_id'],
                'home_team': row['home_team'],
                'away_team': row['away_team'],
                'game_venue': row['game_venue'],
                'projection': cand['projection'],
                'model_context': cand['ctx_data'],
            }

            timestamp = utcnow().isoformat()
            order_records: list[dict] = []
            telegram_delivered = False

            for venue in venues:
                record = await venue.place_order(cand['edge'], execution_context)
                order_records.append(record)
                if venue.name == 'telegram' and record['status'] == 'sent':
                    telegram_delivered = True

            delivered = 1 if (telegram_delivered and BETTING_ENABLED) else 0

            # Record the bet unconditionally: shadow-mode (delivered=0) rows
            # are what settlement, CLV, and calibration learn from during the
            # paper-trading window.
            model_prob = cand['edge']['model_prob']
            mp_over = model_prob if cand['side'] == 'over' else (
                1.0 - model_prob if model_prob is not None else None)
            mp_under = (1.0 - mp_over) if mp_over is not None else None
            cur = conn.execute('''
                INSERT INTO alerts_sent
                (player_name, market, line, side, edge, ev, kelly_stake,
                 bookmaker, odds, opening_odds, model_prob_over, model_prob_under,
                 game_id, timestamp, delivered, player_id, open_devig_prob)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(player_name, market, line, bookmaker, game_id) DO UPDATE SET
                    edge=excluded.edge,
                    ev=excluded.ev,
                    timestamp=excluded.timestamp,
                    delivered=MAX(alerts_sent.delivered, excluded.delivered)
            ''', (
                row['player_name'], row['market'], row['line'], cand['side'],
                cand['edge']['edge_pct'], cand['edge']['ev'],
                display_stake, row['bookmaker'], cand['odds'], cand['odds'],
                mp_over, mp_under,
                row['game_id'], timestamp, delivered,
                row.get('player_id'), row.get('open_devig_prob'),
            ))
            alert_id = cur.lastrowid

            for record in order_records:
                _persist_order(conn, record, execution_context, alert_id)
            conn.commit()

            if delivered:
                alerts_sent_count += 1

    logger.info(
        f"Alerts pipeline complete. Recorded {len(selected)} bets "
        f"({alerts_sent_count} delivered, betting_enabled={BETTING_ENABLED})."
    )


def _persist_order(conn, record: dict, ctx: dict, alert_id):
    # Persist every venue outcome, including shadow-mode 'skipped' records —
    # they are the order trail for the paper-trading window.
    if record.get('status') is None:
        return
    conn.execute('''
        INSERT INTO orders
        (venue, alert_id, player_name, market, line, side, game_id, bookmaker,
         offered_odds, fill_odds, stake, status, venue_order_id,
         placed_at, filled_at, notes)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    ''', (
        record['venue'], alert_id, ctx['player_name'], ctx['market'],
        ctx['line'], ctx['side'], ctx['game_id'], ctx['bookmaker'],
        record.get('offered_odds'), record.get('fill_odds'),
        record.get('stake'), record['status'], record.get('venue_order_id'),
        record.get('placed_at'), record.get('filled_at'), record.get('notes'),
    ))
