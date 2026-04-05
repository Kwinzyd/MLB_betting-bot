import json
from datetime import datetime
from src.clients.telegram_bot import TelegramClient
from src.data.db import get_db_connection
from src.config import EDGE_MIN
from src.models.devig import devig_multiplicative
from src.models.edge_ranker import rank_edge
from src.utils.logging_utils import get_logger

logger = get_logger(__name__)


def send_alerts():
    """Find high-edge projections and send Telegram alerts for unsent ones."""
    logger.info("Executing pipeline: send_alerts")
    bot = TelegramClient()

    with get_db_connection() as conn:
        # Find projections with matching prop snapshots that haven't been alerted yet
        rows = conn.execute('''
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
        ''').fetchall()

    if not rows:
        logger.info("No projections found for alerting.")
        return

    alerts_sent_count = 0
    for row in rows:
        player_name = row['player_name']
        market = row['market']
        line = row['line']
        bookmaker = row['bookmaker']
        over_odds = row['over_odds']
        under_odds = row['under_odds']

        # Determine best side
        projection = {
            'prob_over': row['prob_over'],
            'prob_under': row['prob_under'],
            'projected_mean': row['projected_mean'],
            'injury_status': 'Healthy',
            'sample_size': 10,
        }

        # Check both sides and pick the one with the better edge
        best_side = None
        best_edge = None
        best_odds = None

        for side, odds_val, dev_prob in [
            ('over', over_odds, row['devigged_over']),
            ('under', under_odds, row['devigged_under']),
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

        # Check if already alerted
        with get_db_connection() as conn:
            existing = conn.execute(
                "SELECT 1 FROM alerts_sent WHERE player_name=? AND market=? AND line=? AND bookmaker=?",
                (player_name, market, line, bookmaker)
            ).fetchone()

            if existing:
                continue

            # Parse context for alert message
            context = {}
            if row['context_json']:
                try:
                    context = json.loads(row['context_json'])
                except json.JSONDecodeError:
                    pass

            # Format and send alert
            message = _format_alert_message(
                player_name=player_name,
                market=market,
                side=best_side,
                line=line,
                odds=best_odds,
                bookmaker=bookmaker,
                edge=best_edge,
                projection=projection,
                context=context,
                home_team=row['home_team'],
                away_team=row['away_team'],
                venue=row['venue'],
            )

            try:
                bot.send_message(message)
                logger.info(f"Alert sent: {player_name} {market} {best_side.upper()} {line}")
            except Exception as e:
                logger.error(f"Failed to send Telegram alert: {e}")
                continue

            # Log the alert
            timestamp = datetime.utcnow().isoformat()
            conn.execute('''
                INSERT INTO alerts_sent
                (player_name, market, line, side, edge, ev, kelly_stake,
                 bookmaker, odds, opening_odds, game_id, timestamp)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(player_name, market, line, bookmaker) DO NOTHING
            ''', (
                player_name, market, line, best_side,
                best_edge['edge_pct'], best_edge['ev'],
                best_edge['kelly']['recommended_stake'],
                bookmaker, best_odds, best_odds,
                row['game_id'], timestamp,
            ))
            conn.commit()
            alerts_sent_count += 1

    logger.info(f"Alerts pipeline complete. Sent {alerts_sent_count} new alerts.")


def _format_alert_message(player_name, market, side, line, odds, bookmaker,
                          edge, projection, context, home_team, away_team, venue):
    """Format a rich Telegram alert message."""
    # Convert decimal odds to American
    if odds >= 2.0:
        american = f"+{int((odds - 1) * 100)}"
    else:
        american = f"-{int(100 / (odds - 1))}"

    market_display = market.replace('_', ' ').title()
    venue_display = venue or 'Unknown'

    msg = (
        f"<b>MLB PROP ALERT</b>\n"
        f"{'=' * 30}\n"
        f"<b>{player_name}</b> - {market_display}\n"
        f"<b>{side.upper()} {line}</b>\n\n"
        f"Book: {bookmaker.title()} @ {odds:.2f} ({american})\n"
        f"Edge: <b>{edge['edge_pct']:.1f}%</b> | EV: {edge['ev']:+.3f}\n"
        f"Model: {edge['model_prob']:.1%} | Book: {edge['book_implied']:.1%}\n"
        f"Kelly Stake: <b>${edge['kelly']['recommended_stake']:.2f}</b>\n\n"
        f"Matchup: {away_team} @ {home_team}\n"
        f"Venue: {venue_display}\n"
    )

    # Add context details
    if context:
        if 'opp_k_rate' in context:
            msg += f"Opp K%: {context['opp_k_rate']:.1%} (avg {0.225:.1%})\n"
        if 'park_adj' in context:
            msg += f"Park Adj: {context['park_adj']:.3f}\n"
        if 'platoon_adj' in context:
            msg += f"Platoon Adj: {context['platoon_adj']:.3f}\n"

    msg += f"\nProjected: {projection['projected_mean']:.2f}"

    return msg
