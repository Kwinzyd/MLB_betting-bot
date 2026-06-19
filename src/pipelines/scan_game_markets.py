"""Game-market scan: moneyline / total / run line, model-driven off BDL odds.

Unlike scan_props (which anchors on a sharp book), BDL game odds are soft-only,
so the EDGE here is the team run model's probability minus the devigged BDL
consensus. We shop the best vendor price for EV/Kelly. Conservative by design:
a higher edge bar (GAME_EDGE_MIN), a Kelly haircut (GAME_KELLY_FRACTION_MULT),
confirmed starters required, and a guard that keeps each team's projected runs
near the book's implied team total.

Writes winners to game_bet_candidates; send_game_alerts dispatches them and lands
them in the shared alerts_sent / bet_results ledger.
"""
from __future__ import annotations

from datetime import datetime, timezone

from src.config import (
    GAME_MARKETS_ENABLED, GAME_EDGE_MIN, GAME_KELLY_FRACTION_MULT,
    GAME_LAMBDA_GUARD_RUNS, GAME_REQUIRE_CONFIRMED_STARTERS,
    KELLY_FRACTION, LEAGUE_AVG_RUNS_PER_GAME,
)
from src.clients.bdl_odds import get_game_quotes
from src.data.db import get_db_connection
from src.data.feature_builder import compute_bullpen_factor
from src.data.park_factors import get_park_factor
from src.models.game_model import (
    project_team_runs, moneyline_probabilities, total_probabilities,
    run_line_cover_prob,
)
from src.models.kelly import fractional_kelly
from src.models.pa_estimator import implied_team_total
from src.models.projections import ProjectionModel
from src.utils.logging_utils import get_logger
from src.utils.time_utils import get_eastern_local_date

logger = get_logger(__name__)

_ER_TO_RUNS = 1.08  # earned runs -> total runs (≈ 1/0.92 league ER share)


async def scan_game_markets(force: bool = False, game_ids: list = None):
    """Project team runs, price BDL game markets, persist +EV game candidates."""
    if not GAME_MARKETS_ENABLED:
        logger.info("scan_game_markets: GAME_MARKETS_ENABLED=false — skipping.")
        return

    logger.info("Executing pipeline: scan_game_markets")
    proj_model = ProjectionModel()
    today = str(get_eastern_local_date())

    with get_db_connection() as conn:
        if game_ids:
            placeholders = ",".join("?" for _ in game_ids)
            games = conn.execute(
                f"SELECT * FROM games WHERE status NOT IN ('COMPLETED','POSTPONED') "
                f"AND game_id IN ({placeholders})",
                tuple(game_ids),
            ).fetchall()
        else:
            games = conn.execute(
                "SELECT * FROM games WHERE status NOT IN ('COMPLETED','POSTPONED')"
            ).fetchall()
    games = [dict(g) for g in games]

    if not games:
        logger.info("scan_game_markets: no active games.")
        return

    total_edges = 0
    for game in games:
        quotes = await get_game_quotes(game.get('bdl_game_id'))
        if not quotes:
            logger.debug("No BDL quotes for %s; skipping.", game['game_id'])
            continue

        lambdas = _build_team_lambdas(proj_model, game, quotes, today)
        if lambdas is None:
            continue
        lam_home, lam_away = lambdas

        edges = _price_markets(game, quotes, lam_home, lam_away)
        if edges:
            _persist(game['game_id'], lam_home, lam_away, edges)
            total_edges += len(edges)

    logger.info("scan_game_markets complete. Found %d game-market edges.", total_edges)


# ---------------------------------------------------------------------------
# Projection
# ---------------------------------------------------------------------------

def _build_team_lambdas(proj_model, game, quotes, today):
    """Return (lam_home, lam_away) expected runs, or None if inputs are missing."""
    game_id = game['game_id']
    home_id = game.get('home_team_id')
    away_id = game.get('away_team_id')
    venue = game.get('venue')
    if not home_id or not away_id:
        return None

    with get_db_connection() as conn:
        starters = {
            r['team_id']: r['player_id']
            for r in conn.execute(
                """SELECT pp.player_id, p.team_id
                   FROM probable_pitchers pp JOIN players p ON pp.player_id = p.player_id
                   WHERE pp.game_id = ?""",
                (game_id,),
            ).fetchall()
            if r['team_id'] is not None
        }
        home_starter = starters.get(home_id)
        away_starter = starters.get(away_id)
        if GAME_REQUIRE_CONFIRMED_STARTERS and (not home_starter or not away_starter):
            logger.debug("Skipping %s: starters not confirmed (home=%s away=%s).",
                         game_id, home_starter, away_starter)
            return None

        home_off = _team_rpg(conn, home_id)
        away_off = _team_rpg(conn, away_id)
        # Each team bats vs the OPPOSING starter + bullpen.
        away_sp_runs9, away_sp_ip = _starter_runs9_ip(proj_model, conn, away_starter)
        home_sp_runs9, home_sp_ip = _starter_runs9_ip(proj_model, conn, home_starter)
        home_bp = compute_bullpen_factor(away_id, today, db=conn)  # home bats vs away pen
        away_bp = compute_bullpen_factor(home_id, today, db=conn)

    park_runs = get_park_factor(venue).get('runs', 1.0) if venue else 1.0

    lam_home = project_team_runs(home_off, away_sp_runs9, away_sp_ip, home_bp, park_runs)
    lam_away = project_team_runs(away_off, home_sp_runs9, home_sp_ip, away_bp, park_runs)

    # Guard: keep each team's projected runs within GAME_LAMBDA_GUARD_RUNS of the
    # book's implied team total so a broken projection can't invent a huge edge.
    total_line = quotes.get('total_line')
    if total_line is not None:
        # Center the guard on the book's implied team totals (even split; the
        # band is wide enough that the moneyline tilt isn't needed here).
        itt_home = implied_team_total(total_line)
        itt_away = total_line - itt_home
        lam_home = _clamp(lam_home, itt_home - GAME_LAMBDA_GUARD_RUNS, itt_home + GAME_LAMBDA_GUARD_RUNS)
        lam_away = _clamp(lam_away, itt_away - GAME_LAMBDA_GUARD_RUNS, itt_away + GAME_LAMBDA_GUARD_RUNS)

    return lam_home, lam_away


def _team_rpg(conn, team_id):
    row = conn.execute(
        "SELECT runs_per_game FROM team_stats WHERE team_id = ?", (team_id,)
    ).fetchone()
    if row and row['runs_per_game']:
        return float(row['runs_per_game'])
    return LEAGUE_AVG_RUNS_PER_GAME


def _starter_runs9_ip(proj_model, conn, pitcher_id):
    """Intrinsic (opponent- and park-neutral) runs/9 and projected IP for a starter.

    Reuses project_pitcher_earned_runs with a league-average opponent and a
    neutral venue so we extract the pitcher's own quality; project_team_runs then
    applies the real team offense and park (no double counting).
    """
    if not pitcher_id:
        return LEAGUE_AVG_RUNS_PER_GAME, 5.5
    logs = [dict(r) for r in conn.execute(
        "SELECT * FROM pitcher_game_logs WHERE player_id = ? ORDER BY date DESC",
        (pitcher_id,),
    ).fetchall()]
    proj = proj_model.project_pitcher_earned_runs(
        logs, LEAGUE_AVG_RUNS_PER_GAME, venue='', line=2.5, player_id=pitcher_id,
    )
    if not proj:
        return LEAGUE_AVG_RUNS_PER_GAME, 5.5
    proj_ip = (proj.get('context', {}) or {}).get('proj_ip') or 5.5
    er = proj.get('projected_mean') or 0.0
    runs9 = (er / proj_ip) * 9.0 * _ER_TO_RUNS if proj_ip > 0 else LEAGUE_AVG_RUNS_PER_GAME
    return runs9, proj_ip


# ---------------------------------------------------------------------------
# Pricing
# ---------------------------------------------------------------------------

def _price_markets(game, quotes, lam_home, lam_away):
    """Build playable (model > book) game-market candidates for this game."""
    run_line = quotes.get('run_line', 1.5)
    total_line = quotes.get('total_line')
    home_fav = quotes.get('home_is_ml_favorite', lam_home >= lam_away)
    m = quotes['markets']

    ml = moneyline_probabilities(lam_home, lam_away)
    sides = [
        ('moneyline', 'home', 0.0, ml['home'], m['moneyline']['home']),
        ('moneyline', 'away', 0.0, ml['away'], m['moneyline']['away']),
    ]
    if total_line is not None:
        tot = total_probabilities(lam_home, lam_away, total_line)
        sides += [
            ('game_total', 'over', total_line, tot['over'], m['game_total']['over']),
            ('game_total', 'under', total_line, tot['under'], m['game_total']['under']),
        ]
        home_line = -run_line if home_fav else run_line
        away_line = run_line if home_fav else -run_line
        sides += [
            ('run_line', 'home', home_line,
             run_line_cover_prob(lam_home, lam_away, 'home', home_line), m['run_line']['home']),
            ('run_line', 'away', away_line,
             run_line_cover_prob(lam_home, lam_away, 'away', away_line), m['run_line']['away']),
        ]

    out = []
    frac = KELLY_FRACTION * GAME_KELLY_FRACTION_MULT
    for market, side, line, model_prob, quote in sides:
        book_implied = quote['devig']
        odds = quote['best_odds']
        if not odds or odds <= 1.0 or model_prob is None:
            continue
        edge_pct = (model_prob - book_implied) * 100.0
        if edge_pct < GAME_EDGE_MIN:
            continue
        kelly = fractional_kelly(model_prob, odds, fraction=frac)
        if kelly['kelly_fraction'] <= 0:
            continue
        ev = model_prob * odds - 1.0
        out.append({
            'market': market, 'side': side, 'line': line,
            'vendor': quote.get('best_vendor') or 'consensus',
            'odds': round(odds, 4), 'model_prob': round(model_prob, 4),
            'book_implied': round(book_implied, 4), 'edge_pct': round(edge_pct, 2),
            'ev': round(ev, 4), 'kelly_fraction': kelly['kelly_fraction'],
            'recommended_stake': kelly['recommended_stake'],
        })
        logger.info(
            "GAME EDGE: %s %s %s %s @ %.2f (%s) | model %.1f%% vs book %.1f%% | "
            "edge %.1f%% | EV %.3f | $%.2f",
            game['away_team'], game['home_team'], market, side, odds,
            quote.get('best_vendor') or 'consensus', model_prob * 100,
            book_implied * 100, edge_pct, ev, kelly['recommended_stake'],
        )
    return out


def _persist(game_id, lam_home, lam_away, edges):
    now = datetime.now(timezone.utc).isoformat()
    with get_db_connection() as conn:
        for e in edges:
            conn.execute(
                """INSERT INTO game_bet_candidates
                   (game_id, market, side, line, vendor, odds, model_prob,
                    book_implied, edge_pct, ev, kelly_fraction, recommended_stake,
                    lam_home, lam_away, created_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                   ON CONFLICT(game_id, market, side) DO UPDATE SET
                       line=excluded.line, vendor=excluded.vendor, odds=excluded.odds,
                       model_prob=excluded.model_prob, book_implied=excluded.book_implied,
                       edge_pct=excluded.edge_pct, ev=excluded.ev,
                       kelly_fraction=excluded.kelly_fraction,
                       recommended_stake=excluded.recommended_stake,
                       lam_home=excluded.lam_home, lam_away=excluded.lam_away,
                       created_at=excluded.created_at""",
                (game_id, e['market'], e['side'], e['line'], e['vendor'], e['odds'],
                 e['model_prob'], e['book_implied'], e['edge_pct'], e['ev'],
                 e['kelly_fraction'], e['recommended_stake'],
                 round(lam_home, 3), round(lam_away, 3), now),
            )
        conn.commit()


def _clamp(x, lo, hi):
    return max(lo, min(hi, x))
