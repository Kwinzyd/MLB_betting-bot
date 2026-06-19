"""Dispatch game-market candidates (scan_game_markets) to Telegram + the ledger.

Consumes fresh game_bet_candidates, caps exposure per game, honors the same
full-stop / Kelly-override circuit-breaker state as props, and records every bet
into the SHARED alerts_sent table (market in {moneyline, game_total, run_line}),
so game P&L settles through bet_results and the existing bankroll + breakers with
no parallel ledger. Shadow-mode (BETTING_ENABLED=false) still records rows.
"""
from __future__ import annotations

from datetime import timedelta

from src.config import (
    GAME_MARKETS_ENABLED, GAME_MAX_BETS_PER_GAME, GAME_CANDIDATE_MAX_AGE_MINUTES,
    BETTING_ENABLED,
)
from src.clients.telegram_bot import TelegramClient
from src.clients.execution.telegram_venue import format_game_alert_message
from src.data.db import get_db_connection
from src.utils.logging_utils import get_logger
from src.utils.stake_rounding import round_stake
from src.utils.time_utils import utcnow

logger = get_logger(__name__)


def _label(side: str, market: str, line: float, home_team: str, away_team: str) -> str:
    """Side-encoding label (also keeps the alerts_sent UNIQUE key from colliding
    over/under or home/away)."""
    team = home_team if side == 'home' else away_team
    if market == 'moneyline':
        return f"{team} ML"
    if market == 'run_line':
        return f"{team} {line:+.1f}"
    return f"Total {side.upper()}"


async def send_game_alerts():
    if not GAME_MARKETS_ENABLED:
        logger.info("send_game_alerts: GAME_MARKETS_ENABLED=false — skipping.")
        return
    logger.info("Executing pipeline: send_game_alerts")

    from src.pipelines.send_alerts import _load_bankroll_snapshot
    snapshot = _load_bankroll_snapshot()
    if snapshot.get('full_stop'):
        logger.warning("send_game_alerts: FULL STOP active — no game alerts today.")
        return
    kelly_override = float(snapshot.get('kelly_fraction_override') or 1.0)

    cutoff = (utcnow() - timedelta(minutes=GAME_CANDIDATE_MAX_AGE_MINUTES)).isoformat()
    with get_db_connection() as conn:
        rows = [dict(r) for r in conn.execute(
            """SELECT c.*, g.home_team, g.away_team
               FROM game_bet_candidates c
               JOIN games g ON c.game_id = g.game_id
               WHERE c.created_at >= ?
                 AND g.status NOT IN ('COMPLETED', 'IN_PROGRESS', 'POSTPONED')
               ORDER BY c.ev DESC""",
            (cutoff,),
        ).fetchall()]

    if not rows:
        logger.info("send_game_alerts: no fresh game candidates.")
        return

    client = TelegramClient()
    per_game: dict = {}
    sent = 0
    recorded = 0
    with get_db_connection() as conn:
        for row in rows:
            gid = row['game_id']
            if per_game.get(gid, 0) >= GAME_MAX_BETS_PER_GAME:
                continue
            label = _label(row['side'], row['market'], row['line'],
                           row['home_team'], row['away_team'])
            existing = conn.execute(
                "SELECT 1 FROM alerts_sent WHERE player_name=? AND market=? AND line=? "
                "AND bookmaker=? AND game_id=?",
                (label, row['market'], row['line'], row['vendor'], gid),
            ).fetchone()
            if existing:
                continue
            per_game[gid] = per_game.get(gid, 0) + 1

            stake = round_stake(round((row['recommended_stake'] or 0.0) * kelly_override, 2))
            edge = {
                'edge_pct': row['edge_pct'], 'ev': row['ev'],
                'model_prob': row['model_prob'], 'book_implied': row['book_implied'],
                'kelly': {'recommended_stake': stake,
                          'kelly_fraction': row['kelly_fraction']},
            }
            projection = {'lam_home': row['lam_home'], 'lam_away': row['lam_away']}
            message = format_game_alert_message(
                label, row['market'], row['side'], row['line'], row['odds'],
                row['vendor'], edge, projection, row['home_team'], row['away_team'])

            delivered = 0
            if BETTING_ENABLED:
                try:
                    await client.send_message(message)
                    delivered = 1
                    sent += 1
                except Exception as e:
                    logger.error("Game alert Telegram send failed: %s", e)
            else:
                logger.info("[SHADOW] Would-alert game market: %s %s @ %.2f (edge %.1f%%)",
                            label, row['market'], row['odds'], row['edge_pct'])

            # model_prob_over carries the side's model prob; settlement reads
            # market/side/line, not these, but they keep calibration data uniform.
            conn.execute(
                """INSERT INTO alerts_sent
                   (player_name, market, line, side, edge, ev, kelly_stake,
                    bookmaker, odds, opening_odds, model_prob_over, model_prob_under,
                    game_id, timestamp, delivered, player_id, open_devig_prob)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, NULL, ?)
                   ON CONFLICT(player_name, market, line, bookmaker, game_id) DO UPDATE SET
                       edge=excluded.edge, ev=excluded.ev, timestamp=excluded.timestamp,
                       delivered=MAX(alerts_sent.delivered, excluded.delivered)""",
                (label, row['market'], row['line'], row['side'], row['edge_pct'],
                 row['ev'], stake, row['vendor'], row['odds'], row['odds'],
                 row['model_prob'], 1.0 - row['model_prob'] if row['model_prob'] is not None else None,
                 gid, utcnow().isoformat(), delivered, row['book_implied']),
            )
            recorded += 1
        conn.commit()

    logger.info("send_game_alerts complete. Recorded %d game bets (%d delivered, "
                "betting_enabled=%s).", recorded, sent, BETTING_ENABLED)
