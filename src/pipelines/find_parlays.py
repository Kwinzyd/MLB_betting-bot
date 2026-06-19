"""
Cross-game multi-leg parlay discovery (2 / 4 / 8 legs) — NOT same-game.

find_sgp.py models the within-game correlation between a pitcher and the
opposing batters he faces. This pipeline is the complement: it builds parlays
from the strongest single-bet edges across DIFFERENT games. Enforcing one leg
per game makes the legs statistically independent, so the ticket price is an
exact product with no correlation guesswork:

    joint_prob  = prod(leg_prob)
    parlay_odds = prod(leg_odds)
    ev          = joint_prob * parlay_odds - 1 = prod(1 + leg_ev) - 1

Because every leg is already +EV, parlay EV compounds upward with each leg —
but so does variance, which is why a tightened fractional Kelly sizes long
parlays down to almost nothing. That self-limiting stake is the risk control.

Legs come from bet_candidates: the playable edges scan_props already vetted
(sharp-anchored, model/sharp agreement-gated, freshness-stamped). For each
target size we take the top-N legs by per-leg EV (optimal under independence),
price the ticket, size it, and Telegram-alert the menu. The sizes nest (the
8-leg contains the 2/4-leg legs) and are alternative tickets — the user picks
one and places it manually in the book's parlay builder.
"""
import json
from datetime import timedelta

from src.clients.telegram_bot import TelegramClient
from src.data.db import get_db_connection
from src.config import (
    PARLAY_SIZES, PARLAY_MIN_EV, PARLAY_MIN_LEG_EDGE, PARLAY_MAX_PER_DAY,
    PARLAY_KELLY_FRACTION_MULT, PARLAY_CANDIDATE_MAX_AGE_MINUTES, KELLY_FRACTION,
)
from src.models.kelly import fractional_kelly, get_current_bankroll
from src.utils.time_utils import utcnow
from src.utils.logging_utils import get_logger

logger = get_logger(__name__)


async def find_and_alert_parlays():
    """Entry point. Builds, persists, and alerts cross-game parlays."""
    logger.info("Executing pipeline: find_parlays")
    legs = _load_legs()
    pool = _select_pool(legs)
    if len(pool) < 2:
        logger.info(
            "Not enough independent legs for a parlay (have %d, need >= 2).",
            len(pool),
        )
        return

    parlays = _build_parlays(pool)
    if not parlays:
        logger.info("No parlays cleared the EV / Kelly bar.")
        return

    await _persist_and_alert(parlays)
    logger.info("Parlay pipeline complete. Alerted %d ticket(s).", len(parlays))


def _load_legs():
    """Fresh, playable single-bet legs from bet_candidates, best one per game.

    One leg per game is what makes the parlay's legs independent (different
    games share no player and no game environment), so we keep only the
    highest-EV candidate per game. Rows are ordered EV-desc, so the first row
    seen for a game is its best leg.
    """
    cutoff = (utcnow() - timedelta(minutes=PARLAY_CANDIDATE_MAX_AGE_MINUTES)).isoformat()
    with get_db_connection() as conn:
        rows = conn.execute(
            '''
            SELECT bc.game_id, bc.player_name, bc.player_id, bc.market, bc.line,
                   bc.side, bc.bookmaker, bc.odds, bc.truth_prob, bc.edge_pct, bc.ev,
                   g.home_team, g.away_team
            FROM bet_candidates bc
            JOIN games g ON bc.game_id = g.game_id
            WHERE bc.created_at >= ?
              AND g.status NOT IN ('COMPLETED', 'IN_PROGRESS')
              AND bc.truth_prob IS NOT NULL AND bc.truth_prob > 0.0
              AND bc.odds > 1.0
              AND bc.edge_pct >= ?
            ORDER BY bc.ev DESC
            ''',
            (cutoff, PARLAY_MIN_LEG_EDGE),
        ).fetchall()

    best_by_game = {}
    for r in rows:
        gid = r['game_id']
        if gid in best_by_game:
            continue  # rows are EV-desc; first seen per game is the best leg
        odds = float(r['odds'])
        prob = float(r['truth_prob'])
        ev = float(r['ev']) if r['ev'] is not None else (prob * odds - 1.0)
        best_by_game[gid] = {
            'game_id': gid,
            'matchup': f"{r['away_team']} @ {r['home_team']}",
            'player_name': r['player_name'],
            'player_id': r['player_id'],
            'market': r['market'],
            'side': r['side'],
            'line': r['line'],
            'odds': odds,
            'prob': prob,
            'edge_pct': r['edge_pct'],
            'ev': ev,
            'bookmaker': r['bookmaker'],
        }

    legs = list(best_by_game.values())
    legs.sort(key=lambda leg: leg['ev'], reverse=True)
    return legs


def _select_pool(legs):
    """Ordered pool of distinct-game, distinct-player legs (EV-desc).

    `_load_legs` already guarantees one leg per game; the distinct-player guard
    is belt-and-suspenders for the doubleheader case (same player, two game ids)
    where two legs would otherwise be the same player and thus correlated.
    """
    pool = []
    seen_players = set()
    for leg in legs:  # already EV-desc
        if leg['player_name'] in seen_players:
            continue
        seen_players.add(leg['player_name'])
        pool.append(leg)
    return pool


def _build_parlays(pool):
    """Price one ticket per requested size from the top-N legs in the pool."""
    bankroll = get_current_bankroll()
    out = []
    for size in sorted({s for s in PARLAY_SIZES if s >= 2}):
        if size > len(pool):
            logger.debug("Skipping %d-leg parlay: only %d legs available.", size, len(pool))
            continue
        legs = pool[:size]

        joint_prob = 1.0
        parlay_odds = 1.0
        for leg in legs:
            joint_prob *= leg['prob']
            parlay_odds *= leg['odds']
        ev = joint_prob * parlay_odds - 1.0
        if ev < PARLAY_MIN_EV:
            logger.debug("Skipping %d-leg parlay: EV %.3f < %.3f.", size, ev, PARLAY_MIN_EV)
            continue

        stake_info = fractional_kelly(
            joint_prob, parlay_odds,
            fraction=KELLY_FRACTION * PARLAY_KELLY_FRACTION_MULT,
            bankroll=bankroll,
        )
        if stake_info['recommended_stake'] <= 0:
            logger.debug("Skipping %d-leg parlay: non-positive Kelly stake.", size)
            continue

        out.append({
            'num_legs': size,
            'legs': legs,
            'joint_prob': joint_prob,
            'parlay_odds': parlay_odds,
            'fair_odds': (1.0 / joint_prob) if joint_prob > 0 else None,
            'ev': ev,
            # Raw fractional-Kelly stake (2 dp). Unlike singles/SGPs we do NOT
            # snap to the camouflage increment: long parlays legitimately size to
            # cents, and rounding sub-$5 stakes up to $5 would overstate the risk
            # of an 8-leg ticket by an order of magnitude. The user places these
            # manually anyway, so the soft-book fingerprinting concern is moot.
            'kelly_stake': stake_info['recommended_stake'],
            'kelly_pct': stake_info['kelly_fraction'],
        })
    return out


async def _persist_and_alert(parlays):
    bot = TelegramClient()
    alerted = 0
    with get_db_connection() as conn:
        for p in parlays:
            if alerted >= PARLAY_MAX_PER_DAY:
                break
            legs_json = json.dumps(p['legs'])
            books = ','.join(sorted({leg['bookmaker'] for leg in p['legs']}))
            cur = conn.execute(
                '''
                INSERT INTO parlays
                (num_legs, legs_json, joint_prob, parlay_odds, fair_odds, ev,
                 kelly_stake, bookmakers, timestamp)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(legs_json) DO NOTHING
                ''',
                (
                    p['num_legs'], legs_json, p['joint_prob'], p['parlay_odds'],
                    p['fair_odds'], p['ev'], p['kelly_stake'], books,
                    utcnow().isoformat(),
                ),
            )
            if cur.rowcount <= 0:
                logger.debug("Parlay skipped (already alerted): %d-leg.", p['num_legs'])
                continue
            try:
                await bot.send_message(_format_parlay_message(p))
                alerted += 1
                logger.info(
                    "PARLAY alerted: %d-leg | joint %.3f | odds %.2f | EV %.1f%% | stake $%.2f",
                    p['num_legs'], p['joint_prob'], p['parlay_odds'],
                    p['ev'] * 100, p['kelly_stake'],
                )
            except Exception as e:
                logger.error("Failed to send parlay alert: %s", e)
        conn.commit()


def _american(decimal_odds):
    """Decimal -> American odds string, for books that display American."""
    if decimal_odds <= 1.0:
        return "n/a"
    if decimal_odds >= 2.0:
        return f"+{round((decimal_odds - 1.0) * 100)}"
    return f"-{round(100.0 / (decimal_odds - 1.0))}"


def _format_parlay_message(p):
    leg_lines = []
    for leg in p['legs']:
        market = leg['market'].replace('_', ' ').title()
        leg_lines.append(
            f"  • {leg['player_name']} {leg['side'].upper()} {leg['line']} "
            f"{market} @ {leg['odds']:.2f} ({leg['bookmaker']})\n"
            f"      <i>{leg['matchup']} | leg edge {leg['edge_pct']:.1f}%</i>"
        )
    pct = p.get('kelly_pct', 0.0) * 100
    return (
        f"<b>MLB {p['num_legs']}-LEG PARLAY</b> <i>(cross-game)</i>\n"
        f"{'=' * 30}\n"
        + "\n".join(leg_lines) + "\n\n"
        f"Joint prob: <b>{p['joint_prob']:.1%}</b>\n"
        f"Parlay odds: <b>{p['parlay_odds']:.2f}</b> ({_american(p['parlay_odds'])})\n"
        f"Fair odds: {p['fair_odds']:.2f}\n"
        f"EV: <b>{p['ev'] * 100:+.1f}%</b>\n"
        f"Kelly stake: <b>${p['kelly_stake']:.2f}</b> ({pct:.2f}% bankroll)\n\n"
        f"<i>Legs are independent (one per game). 2/4/8-leg tickets are "
        f"alternatives — pick one. Longer = higher EV but far lower hit rate.</i>"
    )
