import json
from datetime import datetime
from src.clients.execution import build_venue_registry
from src.data.db import get_db_connection
from src.config import MAX_BETS_PER_GAME, MAX_BETS_PER_PLAYER, BETTING_ENABLED
from src.models.edge_ranker import rank_edge
from src.utils.logging_utils import get_logger
from src.utils.stake_rounding import round_stake

logger = get_logger(__name__)


async def send_alerts():
    """Find high-edge projections and dispatch them to every enabled execution venue."""
    logger.info("Executing pipeline: send_alerts")
    venues = build_venue_registry()
    if not venues:
        logger.warning("No execution venues enabled; send_alerts is a no-op.")
        return

    with get_db_connection() as conn:
        rows = [dict(r) for r in conn.execute('''
            SELECT
                p.game_id, p.player_name, p.market, p.projected_mean,
                p.prob_over, p.prob_under, p.context_json,
                ps.line, ps.over_odds, ps.under_odds, ps.bookmaker,
                ps.devigged_over, ps.devigged_under,
                g.home_team, g.away_team, g.venue, g.date
            FROM projections p
            JOIN prop_snapshots ps ON
                p.game_id = ps.game_id AND
                p.player_name = ps.player_name AND
                p.market = ps.market
            JOIN games g ON p.game_id = g.game_id
            WHERE g.status != 'COMPLETED'
            ORDER BY p.prob_over DESC
        ''').fetchall()]

    if not rows:
        logger.info("No projections found for alerting.")
        return

    candidates = []
    for row in rows:
        projection = {
            'prob_over': row['prob_over'],
            'prob_under': row['prob_under'],
            'projected_mean': row['projected_mean'],
            'injury_status': 'Healthy',
            'sample_size': 10,
        }

        best_side = None
        best_edge = None
        best_odds = None

        for side, odds_val, dev_prob in [
            ('over', row['over_odds'], row['devigged_over']),
            ('under', row['under_odds'], row['devigged_under']),
        ]:
            if not odds_val or odds_val <= 1.0:
                continue
            edge_result = rank_edge(projection, odds_val, side, dev_prob)
            if edge_result['is_playable']:
                if best_edge is None or edge_result['edge_pct'] > best_edge['edge_pct']:
                    best_edge = edge_result
                    best_side = side
                    best_odds = odds_val

        if not best_edge or not best_side:
            continue

        candidates.append({
            'row': row,
            'projection': projection,
            'side': best_side,
            'odds': best_odds,
            'edge': best_edge,
        })

    candidates.sort(key=lambda c: c['edge']['edge_pct'], reverse=True)

    seen_players: dict = {}
    game_counts: dict = {}
    seen_prop_key: set = set()
    alerts_sent_count = 0

    for cand in candidates:
        row = cand['row']
        player_name = row['player_name']
        market = row['market']
        line = row['line']
        bookmaker = row['bookmaker']
        game_id = row['game_id']

        prop_key = (player_name, market, line)
        if prop_key in seen_prop_key:
            continue
        if seen_players.get(player_name, 0) >= MAX_BETS_PER_PLAYER:
            continue
        if game_counts.get(game_id, 0) >= MAX_BETS_PER_GAME:
            continue

        with get_db_connection() as conn:
            existing = conn.execute(
                "SELECT 1 FROM alerts_sent WHERE player_name=? AND market=? AND line=? AND bookmaker=?",
                (player_name, market, line, bookmaker)
            ).fetchone()
            if existing:
                continue

            model_context = {}
            if row['context_json']:
                try:
                    model_context = json.loads(row['context_json'])
                except json.JSONDecodeError:
                    pass

            display_stake = round_stake(cand['edge']['kelly']['recommended_stake'])
            cand['edge']['kelly']['recommended_stake'] = display_stake

            execution_context = {
                'player_name': player_name,
                'market': market,
                'side': cand['side'],
                'line': line,
                'odds': cand['odds'],
                'bookmaker': bookmaker,
                'game_id': game_id,
                'home_team': row['home_team'],
                'away_team': row['away_team'],
                'game_venue': row['venue'],
                'projection': cand['projection'],
                'model_context': model_context,
            }

            timestamp = datetime.utcnow().isoformat()
            alert_id = None
            telegram_succeeded = False
            order_records: list[dict] = []

            for venue in venues:
                record = await venue.place_order(cand['edge'], execution_context)
                order_records.append(record)
                if venue.name == 'telegram' and record['status'] in ('sent', 'skipped'):
                    telegram_succeeded = True

            should_record_alert = telegram_succeeded and BETTING_ENABLED and any(
                r['venue'] == 'telegram' and r['status'] == 'sent' for r in order_records
            )
            if should_record_alert:
                cur = conn.execute('''
                    INSERT INTO alerts_sent
                    (player_name, market, line, side, edge, ev, kelly_stake,
                     bookmaker, odds, opening_odds, model_prob_over, model_prob_under,
                     game_id, timestamp)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(player_name, market, line, bookmaker) DO NOTHING
                ''', (
                    player_name, market, line, cand['side'],
                    cand['edge']['edge_pct'], cand['edge']['ev'],
                    cand['edge']['kelly']['recommended_stake'],
                    bookmaker, cand['odds'], cand['odds'],
                    row['prob_over'], row['prob_under'],
                    game_id, timestamp,
                ))
                alert_id = cur.lastrowid

            for record in order_records:
                _persist_order(conn, record, execution_context, alert_id)
            conn.commit()

            seen_prop_key.add(prop_key)
            seen_players[player_name] = seen_players.get(player_name, 0) + 1
            game_counts[game_id] = game_counts.get(game_id, 0) + 1
            if telegram_succeeded:
                alerts_sent_count += 1

    logger.info(f"Alerts pipeline complete. Sent {alerts_sent_count} new alerts.")


def _persist_order(conn, record: dict, ctx: dict, alert_id):
    if record.get('status') in (None, 'skipped'):
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
