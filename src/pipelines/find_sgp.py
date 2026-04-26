"""
Pitcher-anchored 3-leg SGP discovery.

For each active game:
  1. Find the starting pitcher's K-over candidate (sharp-devigged).
  2. Find opposing-team batters' hits-under / total_bases-under candidates.
  3. Enumerate 3-leg combos (anchor + 2 opposing unders).
  4. Rank by edge vs a naive independent-parlay price.
  5. Persist top SGP_MAX_PER_GAME per game; Telegram-alert the kept ones.

The user places manually in the book's SGP builder — we can't read the
book's actual SGP price from any feed, so we emit our fair correlated
price for the user to compare against.
"""
import json
import itertools
from datetime import datetime

from src.clients.telegram_bot import TelegramClient
from src.data.db import get_db_connection
from src.config import (
    SGP_MIN_EDGE, SGP_MAX_PER_GAME, SGP_MAX_PER_DAY, SHARP_BOOKMAKERS,
)
from src.models.correlation import joint_probability, parlay_decimal_odds
from src.utils.logging_utils import get_logger

logger = get_logger(__name__)


async def find_and_alert_sgps():
    """Entry point. Discovers, persists, and alerts top SGP candidates."""
    logger.info("Executing pipeline: find_sgp")
    candidates = _build_candidates()
    if not candidates:
        logger.info("No SGP candidates found.")
        return

    candidates.sort(key=lambda c: c['edge_vs_naive'], reverse=True)
    by_game = {}
    kept = []
    for c in candidates:
        if by_game.get(c['game_id'], 0) >= SGP_MAX_PER_GAME:
            continue
        kept.append(c)
        by_game[c['game_id']] = by_game.get(c['game_id'], 0) + 1
        if len(kept) >= SGP_MAX_PER_DAY:
            break

    await _persist_and_alert(kept)
    logger.info(f"SGP pipeline complete. Alerted {len(kept)} candidates.")


def _build_candidates():
    out = []
    with get_db_connection() as conn:
        games = conn.execute(
            "SELECT game_id, home_team, away_team FROM games "
            "WHERE status != 'COMPLETED'"
        ).fetchall()

    for game in games:
        game_id = game['game_id']
        home_team = game['home_team']
        away_team = game['away_team']

        for anchor, opposing_team in _find_pitcher_anchors(
            conn, game_id, home_team, away_team
        ):
            opp_legs = _find_opposing_under_legs(conn, game_id, opposing_team)
            if len(opp_legs) < 2:
                continue
            for a, b in itertools.combinations(opp_legs, 2):
                if a['player_name'] == b['player_name']:
                    continue
                legs = [anchor, a, b]
                p = joint_probability(legs, db_conn=conn)
                if p <= 0:
                    continue
                parlay_odds = parlay_decimal_odds(legs)
                fair_odds = 1.0 / p
                edge = p * parlay_odds - 1.0
                if edge < SGP_MIN_EDGE:
                    continue
                out.append({
                    'game_id': game_id,
                    'matchup': f"{away_team} @ {home_team}",
                    'legs': legs,
                    'joint_prob': p,
                    'naive_parlay_odds': parlay_odds,
                    'fair_odds': fair_odds,
                    'edge_vs_naive': edge,
                })
    return out


def _find_pitcher_anchors(conn, game_id, home_team, away_team):
    """Yield (anchor_leg, opposing_team_name) for each playable pitcher K-over
    candidate on this game."""
    sharp_set = set(SHARP_BOOKMAKERS)
    placeholders = ','.join('?' for _ in sharp_set) or "''"
    rows = conn.execute(f'''
        SELECT p.player_name, pl.player_id,
               ps.line, ps.over_odds, ps.bookmaker, ps.devigged_over
        FROM projections p
        JOIN players pl ON p.player_name = pl.name COLLATE NOCASE
        JOIN prop_snapshots ps
          ON p.game_id = ps.game_id
         AND p.player_name = ps.player_name
         AND p.market = ps.market
        WHERE p.game_id = ?
          AND p.market = 'pitcher_strikeouts'
          AND ps.bookmaker NOT IN ({placeholders})
          AND ps.over_odds > 1.0
          AND ps.devigged_over IS NOT NULL
    ''', (game_id, *sharp_set)).fetchall()

    for row in rows:
        pp = conn.execute(
            "SELECT team FROM probable_pitchers "
            "WHERE game_id = ? AND player_name = ? COLLATE NOCASE",
            (game_id, row['player_name'])
        ).fetchone()
        if not pp:
            continue
        pitcher_team = pp['team']
        opposing = (
            away_team if _team_matches(pitcher_team, home_team)
            else home_team
        )
        yield ({
            'player_name': row['player_name'],
            'player_id': row['player_id'],
            'market': 'pitcher_strikeouts',
            'side': 'over',
            'line': row['line'],
            'odds': row['over_odds'],
            'prob': row['devigged_over'],
            'bookmaker': row['bookmaker'],
        }, opposing)


def _find_opposing_under_legs(conn, game_id, opposing_team):
    """Playable hits-under / total_bases-under legs for batters on opposing_team."""
    sharp_set = set(SHARP_BOOKMAKERS)
    placeholders = ','.join('?' for _ in sharp_set) or "''"
    rows = conn.execute(f'''
        SELECT p.player_name, p.market,
               ps.line, ps.under_odds, ps.bookmaker, ps.devigged_under
        FROM projections p
        JOIN prop_snapshots ps
          ON p.game_id = ps.game_id
         AND p.player_name = ps.player_name
         AND p.market = ps.market
        JOIN daily_lineups dl
          ON dl.game_id = p.game_id
         AND dl.player_name = p.player_name COLLATE NOCASE
        WHERE p.game_id = ?
          AND dl.team LIKE ? COLLATE NOCASE
          AND p.market IN ('batter_hits', 'batter_total_bases')
          AND ps.bookmaker NOT IN ({placeholders})
          AND ps.under_odds > 1.0
          AND ps.devigged_under IS NOT NULL
    ''', (game_id, f"%{opposing_team}%", *sharp_set)).fetchall()

    return [{
        'player_name': r['player_name'],
        'market': r['market'],
        'side': 'under',
        'line': r['line'],
        'odds': r['under_odds'],
        'prob': r['devigged_under'],
        'bookmaker': r['bookmaker'],
    } for r in rows]


def _team_matches(a, b):
    a = (a or '').lower()
    b = (b or '').lower()
    if not a or not b:
        return False
    return a in b or b in a


async def _persist_and_alert(candidates):
    if not candidates:
        return
    bot = TelegramClient()
    with get_db_connection() as conn:
        for c in candidates:
            legs_json = json.dumps(c['legs'])
            books = ','.join(sorted({leg['bookmaker'] for leg in c['legs']}))
            conn.execute('''
                INSERT INTO sgp_candidates
                (game_id, legs_json, joint_prob, naive_parlay_odds,
                 fair_odds, edge_vs_naive, bookmakers, timestamp)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(game_id, legs_json) DO NOTHING
            ''', (
                c['game_id'], legs_json, c['joint_prob'],
                c['naive_parlay_odds'], c['fair_odds'], c['edge_vs_naive'],
                books, datetime.utcnow().isoformat(),
            ))
            try:
                await bot.send_message(_format_sgp_message(c))
                logger.info(
                    f"SGP alerted: {c['matchup']} | "
                    f"joint {c['joint_prob']:.3f} | "
                    f"fair {c['fair_odds']:.2f} vs naive {c['naive_parlay_odds']:.2f} | "
                    f"edge {c['edge_vs_naive']*100:.1f}%"
                )
            except Exception as e:
                logger.error(f"Failed to send SGP alert: {e}")
        conn.commit()


def _format_sgp_message(c):
    leg_lines = []
    for leg in c['legs']:
        market = leg['market'].replace('_', ' ').title()
        leg_lines.append(
            f"  • {leg['player_name']} {leg['side'].upper()} "
            f"{leg['line']} {market} @ {leg['odds']:.2f} "
            f"({leg['bookmaker']})"
        )
    return (
        f"<b>MLB SGP CANDIDATE</b>\n"
        f"{'=' * 30}\n"
        f"<b>{c['matchup']}</b>\n\n"
        + "\n".join(leg_lines) + "\n\n"
        f"Joint prob: <b>{c['joint_prob']:.1%}</b>\n"
        f"Naive parlay odds: {c['naive_parlay_odds']:.2f}\n"
        f"Fair (correlated) odds: {c['fair_odds']:.2f}\n"
        f"Edge vs naive: <b>{c['edge_vs_naive']*100:.1f}%</b>\n\n"
        f"<i>Compare book's SGP builder price to fair odds before placing.</i>"
    )
