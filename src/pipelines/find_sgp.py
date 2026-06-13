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
from src.utils.time_utils import utcnow

from src.clients.telegram_bot import TelegramClient
from src.data.db import get_db_connection
from src.config import (
    SGP_MIN_EDGE, SGP_MAX_PER_GAME, SGP_MAX_PER_DAY, SHARP_BOOKMAKERS,
    KELLY_FRACTION, SGP_KELLY_FRACTION_MULT, MAX_BETS_PER_GAME,
)
from src.models.correlation import joint_probability, parlay_decimal_odds
from src.models.kelly import fractional_kelly, get_current_bankroll
from src.models.pa_estimator import implied_team_total as _implied_team_total
from src.utils.stake_rounding import round_stake
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

    kept = _apply_game_exposure_cap(kept)
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
                    stake_info = fractional_kelly(
                        p, parlay_odds,
                        fraction=KELLY_FRACTION * SGP_KELLY_FRACTION_MULT,
                    )
                    if stake_info['recommended_stake'] <= 0:
                        continue
                    out.append({
                        'game_id': game_id,
                        'matchup': f"{away_team} @ {home_team}",
                        'legs': legs,
                        'joint_prob': p,
                        'naive_parlay_odds': parlay_odds,
                        'fair_odds': fair_odds,
                        'edge_vs_naive': edge,
                        'kelly_stake': stake_info['recommended_stake'],
                        'kelly_pct': stake_info['kelly_fraction'],
                    })

    return out


def _sharp_and_soft_quotes(conn, game_id, market, side):
    """Pair each (player, line) with its sharp-devigged probability and the
    best soft-book price for `side`.

    scan_props only writes devigged_* for sharp books, so the leg probability
    MUST come from a sharp snapshot while the leg odds come from a soft book —
    two different rows joined on (player, line). Only the latest snapshot per
    (player, line, bookmaker) counts; stale quotes are exactly what an SGP
    built minutes later would no longer get.

    Returns {(player_name, line): {'prob': float, 'odds': float, 'book': str}}.
    """
    odds_col = f"{side}_odds"
    devig_col = f"devigged_{side}"
    rows = conn.execute(f'''
        SELECT ps.player_name, ps.line, ps.bookmaker, ps.{odds_col} AS odds,
               ps.{devig_col} AS devig
        FROM prop_snapshots ps
        WHERE ps.game_id = ? AND ps.market = ?
          AND ps.timestamp = (
              SELECT MAX(s2.timestamp) FROM prop_snapshots s2
              WHERE s2.game_id = ps.game_id AND s2.player_name = ps.player_name
                AND s2.market = ps.market AND s2.line = ps.line
                AND s2.bookmaker = ps.bookmaker
          )
    ''', (game_id, market)).fetchall()

    sharp_rank = {b: i for i, b in enumerate(SHARP_BOOKMAKERS)}
    sharp_best: dict = {}   # key -> (rank, prob)
    soft_best: dict = {}    # key -> (odds, book)
    for r in rows:
        key = (r['player_name'], r['line'])
        if r['bookmaker'] in sharp_rank:
            if r['devig'] is not None:
                rank = sharp_rank[r['bookmaker']]
                if key not in sharp_best or rank < sharp_best[key][0]:
                    sharp_best[key] = (rank, float(r['devig']))
        else:
            odds = r['odds']
            if odds and odds > 1.0:
                if key not in soft_best or odds > soft_best[key][0]:
                    soft_best[key] = (float(odds), r['bookmaker'])

    out = {}
    for key, (_, prob) in sharp_best.items():
        if key in soft_best:
            odds, book = soft_best[key]
            out[key] = {'prob': prob, 'odds': odds, 'book': book}
    return out


def _find_pitcher_anchors(conn, game_id, home_team, away_team):
    """Yield (anchor_leg, opposing_team_name) for each playable pitcher K-over
    candidate on this game. Leg prob = sharp devig; leg odds = best soft book."""
    quotes = _sharp_and_soft_quotes(conn, game_id, 'pitcher_strikeouts', 'over')

    for (player_name, line), q in quotes.items():
        player = conn.execute(
            "SELECT player_id FROM players WHERE name = ? COLLATE NOCASE",
            (player_name,)
        ).fetchone()
        pp = conn.execute(
            "SELECT team FROM probable_pitchers "
            "WHERE game_id = ? AND player_name = ? COLLATE NOCASE",
            (game_id, player_name)
        ).fetchone()
        if not pp:
            continue
        pitcher_team = pp['team']
        opposing = (
            away_team if _team_matches(pitcher_team, home_team)
            else home_team
        )
        yield ({
            'player_name': player_name,
            'player_id': player['player_id'] if player else None,
            'market': 'pitcher_strikeouts',
            'side': 'over',
            'line': line,
            'odds': q['odds'],
            'prob': q['prob'],
            'bookmaker': q['book'],
        }, opposing)


def _find_opposing_under_legs(conn, game_id, opposing_team):
    """Playable hits-under / total_bases-under legs for batters on opposing_team.
    Leg prob = sharp devig; leg odds = best soft book at the same line."""
    rows = []
    for market in ('batter_hits', 'batter_total_bases'):
        quotes = _sharp_and_soft_quotes(conn, game_id, market, 'under')
        for (player_name, line), q in quotes.items():
            dl = conn.execute(
                "SELECT lineup_position FROM daily_lineups "
                "WHERE game_id = ? AND player_name = ? COLLATE NOCASE "
                "AND team LIKE ? COLLATE NOCASE",
                (game_id, player_name, f"%{opposing_team}%")
            ).fetchone()
            if not dl:
                continue
            proj = conn.execute(
                "SELECT projected_mean FROM projections "
                "WHERE game_id = ? AND player_name = ? AND market = ?",
                (game_id, player_name, market)
            ).fetchone()
            rows.append({
                'player_name': player_name, 'market': market,
                'projected_mean': proj['projected_mean'] if proj else None,
                'line': line, 'under_odds': q['odds'], 'bookmaker': q['book'],
                'devigged_under': q['prob'],
                'lineup_position': dl['lineup_position'],
            })

    # Latest sharp-book total for the opposing team's implied run total.
    total_row = conn.execute(
        "SELECT total FROM game_totals_history WHERE game_id = ? "
        "ORDER BY timestamp DESC LIMIT 1",
        (game_id,),
    ).fetchone()
    game_total = total_row['total'] if total_row else None
    itt = _implied_team_total(game_total, side="opposing")

    return [{
        'player_name': r['player_name'],
        'market': r['market'],
        'side': 'under',
        'line': r['line'],
        'odds': r['under_odds'],
        'prob': r['devigged_under'],
        'bookmaker': r['bookmaker'],
        'team': opposing_team,
        'lineup_position': r['lineup_position'],
        'mean_count': r['projected_mean'],
        'implied_team_total': itt,
    } for r in rows]


def _team_matches(a, b):
    a = (a or '').lower()
    b = (b or '').lower()
    if not a or not b:
        return False
    return a in b or b in a


def _apply_game_exposure_cap(kept):
    """Trim/drop SGP stakes so per-game exposure (singles + SGPs) stays
    within MAX_BETS_PER_GAME * fractional_kelly's 5% per-bet cap.

    Today's `alerts_sent` rows are summed because the SGP is placed in
    the same session — historical days don't share a bankroll allocation.
    """
    if not kept:
        return kept
    bankroll = get_current_bankroll()
    per_bet_cap = bankroll * 0.05  # mirrors fractional_kelly's max_fraction
    game_cap = per_bet_cap * MAX_BETS_PER_GAME
    out = []
    with get_db_connection() as conn:
        # Aggregate same-game SGP stakes too, so multiple SGPs on one game
        # don't all get sized against the same headroom.
        used_today = {}
        for c in kept:
            row = conn.execute(
                "SELECT COALESCE(SUM(kelly_stake), 0) AS used "
                "FROM alerts_sent WHERE game_id = ? "
                "AND date(timestamp) = date('now')",
                (c['game_id'],),
            ).fetchone()
            singles_used = float(row['used'] or 0.0) if row else 0.0
            already = singles_used + used_today.get(c['game_id'], 0.0)
            remaining = max(0.0, game_cap - already)
            stake = min(c['kelly_stake'], remaining)
            if stake <= 0:
                logger.info(
                    f"SGP dropped for game {c['game_id']}: exposure cap "
                    f"reached (used={already:.2f} of cap={game_cap:.2f})"
                )
                continue
            # Camouflage: snap to a round increment ($5 default) so the
            # persisted/displayed stake doesn't fingerprint as a bot.
            rounded_stake = round_stake(stake)
            c['kelly_stake'] = rounded_stake
            # Track the post-round value so a later SGP on the same game
            # sees the actual exposure (an upward snap eats real headroom).
            used_today[c['game_id']] = already + rounded_stake
            out.append(c)
    return out


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
                 fair_odds, edge_vs_naive, kelly_stake, bookmakers, timestamp)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(game_id, legs_json) DO NOTHING
            ''', (
                c['game_id'], legs_json, c['joint_prob'],
                c['naive_parlay_odds'], c['fair_odds'], c['edge_vs_naive'],
                c.get('kelly_stake'), books, utcnow().isoformat(),
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
    stake_line = ""
    stake = c.get('kelly_stake')
    if stake is not None and stake > 0:
        pct = c.get('kelly_pct', 0.0) * 100
        stake_line = f"Stake: <b>${stake:.2f}</b> ({pct:.2f}% bankroll)\n"
    return (
        f"<b>MLB SGP CANDIDATE</b>\n"
        f"{'=' * 30}\n"
        f"<b>{c['matchup']}</b>\n\n"
        + "\n".join(leg_lines) + "\n\n"
        f"Joint prob: <b>{c['joint_prob']:.1%}</b>\n"
        f"Naive parlay odds: {c['naive_parlay_odds']:.2f}\n"
        f"Fair (correlated) odds: {c['fair_odds']:.2f}\n"
        f"Edge vs naive: <b>{c['edge_vs_naive']*100:.1f}%</b>\n"
        f"{stake_line}\n"
        f"<i>Compare book's SGP builder price to fair odds before placing.</i>"
    )
