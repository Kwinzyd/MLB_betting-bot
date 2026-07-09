import uuid
import json
import os
import time
from collections import Counter
from datetime import datetime, timezone, timedelta
from src.config import (
    MARKETS_MAPPING, UMP_MIN_GAMES, UMP_K_WEIGHT, PREGAME_WINDOW_MINUTES,
    PREGAME_RESCAN_MINUTES,
    SHARP_BOOKMAKERS, ALT_LINE_MAX_DISTANCE, MAX_BETS_PER_PLAYER,
    SHARP_MODEL_AGREEMENT_TOL, BOOKMAKER_BIAS_THRESHOLD, EDGE_MIN,
    REQUIRE_CONFIRMED_LINEUP, MODEL_AS_TRUTH_MODE, MODEL_ONLY_KELLY_MULT,
    MARKET_SIDE_BLACKLIST, ALT_LINE_MARKET_SHRINK,
    PITCH_MATCHUP_ENABLED, MLB_SEASON, BVP_ENABLED,
    SPORTSDATAIO_ENABLED, SPORTSDATAIO_ANCHOR_FALLBACK,
)
from src.models.pitch_matchup import batter_arsenal_factor_db, pitcher_k_factor_db
from src.models.bvp import bvp_factor_db
from src.models.distributions import get_probabilities
from src.clients.odds_api import OddsAPIClient
from src.clients.bdl_odds import get_game_market, get_player_prop_odds
from src.clients.weather import WeatherClient
from src.data.db import get_db_connection
from src.data.park_factors import get_stadium_meta
from src.models.devig import devig_multiplicative
from src.models.projections import ProjectionModel
from src.models.edge_ranker import rank_edge
from src.utils.logging_utils import get_logger
from src.data.feature_builder import (
    compute_bullpen_factor,
    compute_pitcher_h2h_vs_team,
    compute_platoon_split,
)
from src.utils.time_utils import get_eastern_local_date, utcnow

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Bookmaker bias cache — refreshed at most once per day
# ---------------------------------------------------------------------------
_bias_cache: dict = {}   # (bookmaker, market, side) → avg_bias float
_bias_cache_ts: float = 0.0
_BIAS_CACHE_TTL = 86400.0  # 24h


def _load_bookmaker_bias() -> dict:
    """Load active bookmaker_bias rows from DB with 24h TTL."""
    global _bias_cache, _bias_cache_ts
    now = time.monotonic()
    if _bias_cache and (now - _bias_cache_ts) < _BIAS_CACHE_TTL:
        return _bias_cache
    result: dict = {}
    try:
        with get_db_connection() as conn:
            rows = conn.execute(
                "SELECT bookmaker, market, side, avg_bias FROM bookmaker_bias "
                "WHERE avg_bias IS NOT NULL"
            ).fetchall()
        for r in rows:
            result[(r["bookmaker"], r["market"], r["side"])] = float(r["avg_bias"])
    except Exception as e:
        logger.debug("Could not load bookmaker_bias: %s", e)
    _bias_cache = result
    _bias_cache_ts = now
    return result


async def scan_props(force: bool = False, game_ids: list = None):
    """
    Core pipeline: fetch live odds, run projections, identify edges.
    Batches all 5 markets in a single API call per game to conserve quota.

    Quota gate: a game is polled only when either
      (a) its lineups were confirmed since its last scan, or
      (b) first pitch is within PREGAME_WINDOW_MINUTES.
    Pass force=True (or set ODDS_SCAN_FORCE=1) to bypass the gate for testing.

    game_ids: optional list restricting the scan to specific games. Implies
    force=True (targeted pulls always bypass the gate) and busts the odds
    cache for those events so the trigger path beats the 5-min TTL.
    """
    logger.info("Executing pipeline: scan_props")
    
    from src.data.cache import cache
    if cache.get("odds_api_quota_exhausted"):
        logger.error("Pipeline aborted: API Quota Circuit Breaker is active.")
        return

    odds_client = OddsAPIClient()
    weather_client = WeatherClient()
    proj_model = ProjectionModel()
    bias_map = _load_bookmaker_bias()

    # SportsDataIO consensus props (optional). One PlayerPropsByDate pull per day
    # is cached on the instance, so reuse a single client across all games.
    sdio_client = None
    if SPORTSDATAIO_ENABLED:
        from src.clients.sportsdataio import SportsDataIOClient
        sdio_client = SportsDataIOClient()

    force = force or os.getenv("ODDS_SCAN_FORCE", "").lower() in ("1", "true", "yes")
    targeted = bool(game_ids)
    if targeted:
        force = True
    now_utc = datetime.now(timezone.utc)

    # 1. Get active games from DB
    with get_db_connection() as conn:
        if targeted:
            placeholders = ",".join("?" for _ in game_ids)
            games = conn.execute(
                f"SELECT * FROM games WHERE status NOT IN ('COMPLETED', 'POSTPONED') "
                f"AND game_id IN ({placeholders})",
                tuple(game_ids),
            ).fetchall()
        else:
            games = conn.execute(
                "SELECT * FROM games WHERE status NOT IN ('COMPLETED', 'POSTPONED')"
            ).fetchall()

    if not games:
        logger.info("No active games found for odds scanning.")
        return

    # All markets in one call; include game totals for PA scaling (zero extra quota)
    markets = list(MARKETS_MAPPING.keys())
    api_markets = markets + ["totals"]
    total_edges = 0
    skip_reasons: Counter = Counter()

    for game in games:
        game_id = game['game_id']
        home_team = game['home_team']
        away_team = game['away_team']
        venue = game['venue']

        if not force:
            decision, reason = _should_scan_game(game, now_utc)
            if decision == "skip":
                skip_reasons[reason] += 1
                logger.debug(
                    f"Skipping {away_team} @ {home_team} ({game_id}): {reason}"
                )
                continue

        logger.info(f"Scanning: {away_team} @ {home_team}")

        # Fetch first-pitch weather for this stadium (forecast nearest game_time,
        # with a signed in/out wind component for the HR model).
        weather = None
        meta = get_stadium_meta(venue)
        if meta and meta.get("roof") != "dome":
            weather = weather_client.get_game_weather(
                meta["lat"], meta["lon"],
                outfield_bearing=meta.get("outfield_bearing"),
                game_time=dict(game).get("game_time"),
            )

        # Umpire K factor — looked up once per game, applied to all pitcher props
        ump_k_factor = _get_ump_k_factor(game_id)

        odds_event_id = dict(game).get('odds_api_event_id')
        if not odds_event_id:
            skip_reasons['no_odds_api_event_id'] += 1
            logger.debug(f"Skipping {away_team} @ {home_team} ({game_id}): no Odds API event linked (run sync first).")
            continue

        try:
            event_odds = await odds_client.get_event_odds(
                odds_event_id, api_markets, bust_cache=targeted
            )
        except Exception as e:
            logger.error(f"Failed to fetch odds for {game_id}: {e}")
            continue

        # Merge supplemental books into the Odds API payload before parsing:
        #   - BDL live player props (free; six US soft books, keys 'bdl_*').
        #   - SportsDataIO consensus props (key 'sportsdataio'), which can serve
        #     as a fallback truth anchor when no sharp book quotes a prop.
        # Neither is in SHARP_BOOKMAKERS, so the primary sharp anchor is unchanged.
        extra_books = []
        bdl_props = await get_player_prop_odds(
            dict(game).get('bdl_game_id'), bust_cache=targeted
        )
        if bdl_props:
            extra_books.extend(bdl_props['bookmakers'])

        if sdio_client is not None:
            try:
                sdio_odds = await sdio_client.get_event_odds(
                    game_id, markets, bust_cache=targeted,
                    home_team_full=home_team, away_team_full=away_team,
                )
            except Exception as e:  # noqa: BLE001 — supplemental source must never raise
                logger.debug("SportsDataIO odds fetch failed for %s: %s", game_id, e)
                sdio_odds = None
            if sdio_odds and sdio_odds.get('bookmakers'):
                extra_books.extend(sdio_odds['bookmakers'])

        if extra_books:
            # Copy before merging — event_odds may be the odds client's cached
            # object, and mutating it would re-append books every rescan inside
            # the cache TTL.
            event_odds = dict(event_odds or {})
            event_odds['bookmakers'] = (
                list(event_odds.get('bookmakers', [])) + extra_books
            )

        if not event_odds or not event_odds.get('bookmakers'):
            continue

        # 2. Parse odds and group by player+market(+line) for devigging
        player_lines = _parse_odds_by_player(event_odds)
        player_market_groups = _group_by_player_market(player_lines)
        game_total = _parse_game_total(event_odds)

        # BDL game-level odds (free, unlimited): back up a missing Odds API
        # total and supply moneylines for moneyline-tilted implied team totals.
        # (Player props come from get_player_prop_odds above; this is the
        # separate game-total/moneyline feed.)
        bdl_market = await get_game_market(dict(game).get('bdl_game_id'))
        if game_total is None and bdl_market and bdl_market.get('total') is not None:
            game_total = bdl_market['total']
            logger.info("Using BDL game total %.1f for %s (Odds API total unavailable).",
                        game_total, game_id)
            _record_total_snapshot(game_id, game_total, source='bdl_fallback')
        elif game_total is not None:
            _record_total_snapshot(game_id, game_total, source='scan')

        # 3. For each (player, market):
        #    (a) Pick a sharp-anchored line and devig → TRUE probability at anchor.
        #    (b) Build the projection ONCE; agreement-gate model vs sharp at anchor.
        #    (c) For each soft-quoted alt-line within ALT_LINE_MAX_DISTANCE,
        #        re-price model probability via get_probabilities and rank_edge
        #        against the best soft offer at that line.
        #    (d) Keep up to MAX_BETS_PER_PLAYER highest-EV winners per player.
        for (player_name, market_key), lines_for_market in player_market_groups.items():
            if MODEL_AS_TRUTH_MODE:
                if not lines_for_market:
                    continue
                anchor_line = list(lines_for_market.keys())[0]
                book_data = lines_for_market[anchor_line]
                if 'sportsdataio' in book_data:
                    sharp_book = 'sportsdataio'
                    sharp_over_odds = book_data['sportsdataio'].get('over')
                    sharp_under_odds = book_data['sportsdataio'].get('under')
                else:
                    sharp_book = list(book_data.keys())[0]
                    sharp_over_odds = book_data[sharp_book].get('over')
                    sharp_under_odds = book_data[sharp_book].get('under')
                sharp_prob_over_anchor = None
                sharp_prob_under_anchor = None
            else:
                anchor = _pick_anchor_line(lines_for_market, SHARP_BOOKMAKERS)
                consensus_anchor = False
                if anchor is None and SPORTSDATAIO_ANCHOR_FALLBACK:
                    # No sharp book quotes this prop — fall back to SDIO consensus
                    # so it isn't skipped outright. Softer truth, tagged below.
                    anchor = _pick_anchor_line(lines_for_market, ['sportsdataio'])
                    consensus_anchor = anchor is not None
                if anchor is None:
                    continue
                anchor_line, sharp_over_odds, sharp_under_odds, sharp_book = anchor
                sharp_prob_over_anchor, sharp_prob_under_anchor = devig_multiplicative(
                    sharp_over_odds, sharp_under_odds
                )
                if sharp_prob_over_anchor is None:
                    continue
                # Exclude the anchoring book from its own soft-line shop (a true
                # sharp book is already in SHARP_BOOKMAKERS; SDIO as fallback is not).
                soft_exclude = (SHARP_BOOKMAKERS if sharp_book in SHARP_BOOKMAKERS
                                else SHARP_BOOKMAKERS + [sharp_book])

            projection = _build_projection(
                proj_model, player_name, market_key, anchor_line,
                game_id, home_team, away_team, venue,
                weather=weather, ump_k_factor=ump_k_factor,
                game_total=game_total, bdl_market=bdl_market,
            )
            if not projection:
                continue
            projection['player_name'] = player_name

            # Anchor-level agreement gate. If model and sharp disagree at the
            # consensus line, we don't trust the model anywhere — bail on this
            # (player, market) before evaluating any alt-lines.
            if not MODEL_AS_TRUTH_MODE:
                disagreement = abs(projection['prob_over'] - sharp_prob_over_anchor)
                if disagreement > SHARP_MODEL_AGREEMENT_TOL:
                    logger.debug("Skipped %s %s: model_prob=%.3f, sharp_prob=%.3f (diff=%.3f)", 
                                 player_name, market_key, projection['prob_over'], sharp_prob_over_anchor, disagreement)
                    continue

            timestamp = utcnow().isoformat()

            # Raw model probability at the anchor line — the basis for shifting
            # the sharp truth onto alt-lines. Computed with the same distribution
            # family as the alt-line reprice so the over/under ratio is consistent.
            raw_anchor_over, _ = get_probabilities(
                projection['projected_mean'], anchor_line, market_key,
                alpha=projection.get('alpha'),
                sigma=projection.get('sigma'),
                pi0=projection.get('pi0'),
            )
            # Calibrate to the anchor projection's basis (it was calibrated in
            # _finalize); keeps the truth-shift ratio and agreement gate on one
            # consistent probability scale.
            raw_anchor_over = proj_model.calibrate_over(raw_anchor_over, market_key)

            # Walk every soft-quoted alt-line within range; collect playable edges.
            candidates = []  # (ev, line, side, soft_book, odds, edge_result, line_data)
            with get_db_connection() as conn:
                for line, line_data in lines_for_market.items():
                    if abs(line - anchor_line) > ALT_LINE_MAX_DISTANCE:
                        continue

                    if MODEL_AS_TRUTH_MODE:
                        soft_best = _pick_best_line(line_data)
                    else:
                        soft_best = _pick_best_soft_line(line_data, soft_exclude)
                    over_odds, over_book = soft_best['over']
                    under_odds, under_book = soft_best['under']

                    # Re-price model at this line. The projection mean,
                    # dispersion, and zero-inflation are line-independent, so
                    # we just recompute over/under probabilities at the
                    # alt-line with the same distribution family as the anchor.
                    prob_over_line, prob_under_line = get_probabilities(
                        projection['projected_mean'], line, market_key,
                        alpha=projection.get('alpha'),
                        sigma=projection.get('sigma'),
                        pi0=projection.get('pi0'),
                    )
                    # Same calibration the anchor projection carries, so the
                    # agreement gate compares like-for-like probabilities.
                    prob_over_line = proj_model.calibrate_over(prob_over_line, market_key)
                    prob_under_line = 1.0 - prob_over_line
                    line_proj = dict(projection)
                    line_proj['line'] = line
                    line_proj['prob_over'] = prob_over_line
                    line_proj['prob_under'] = prob_under_line

                    opening_row = conn.execute('''
                        SELECT devigged_over, devigged_under
                        FROM prop_snapshots
                        WHERE game_id = ? AND player_name = ? AND market = ? AND line = ?
                          AND devigged_over IS NOT NULL
                        ORDER BY timestamp ASC
                        LIMIT 1
                    ''', (game_id, player_name, market_key, line)).fetchone()

                    steam_over = _detect_steam_in_soft_books(conn, game_id, player_name, market_key, line, 'over', line_data, SHARP_BOOKMAKERS)
                    steam_under = _detect_steam_in_soft_books(conn, game_id, player_name, market_key, line, 'under', line_data, SHARP_BOOKMAKERS)

                    # Snapshot every book at this line — schema unchanged.
                    # Also collect every two-sided devig for the alt-line
                    # market-consensus shrink below.
                    alt_line_devigs = []
                    for book, pair in line_data.items():
                        o_odds = pair.get('over')
                        u_odds = pair.get('under')
                        if not o_odds or not u_odds or o_odds <= 1.0 or u_odds <= 1.0:
                            continue
                        dev_o, dev_u = devig_multiplicative(o_odds, u_odds)
                        if dev_o is not None:
                            alt_line_devigs.append(dev_o)
                        if book not in SHARP_BOOKMAKERS:
                            dev_o, dev_u = None, None
                        conn.execute('''
                            INSERT INTO prop_snapshots
                            (snapshot_id, game_id, player_name, market, line,
                             over_odds, under_odds, bookmaker, timestamp,
                             devigged_over, devigged_under)
                            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        ''', (
                            str(uuid.uuid4()), game_id, player_name, market_key,
                            line, o_odds, u_odds, book, timestamp,
                            dev_o, dev_u,
                        ))

                    # Truth source per line:
                    #   anchor line → devigged sharp (Pinnacle/Circa) probability.
                    #   alt-line    → the SHARP anchor probability shifted along
                    #                 the model CDF to this line (sharp stays the
                    #                 basis; the model only interpolates shape).
                    #                 Using the raw model prob as truth here would
                    #                 let the model grade itself and defeat the
                    #                 agreement gate — see _shift_anchor_truth.
                    if MODEL_AS_TRUTH_MODE:
                        truth_over = prob_over_line
                        truth_under = prob_under_line
                        edge_source = 'model_only'
                    elif line == anchor_line:
                        truth_over = sharp_prob_over_anchor
                        truth_under = sharp_prob_under_anchor
                        edge_source = 'consensus_anchor' if consensus_anchor else 'sharp_anchor'
                    else:
                        truth_over, truth_under = _shift_anchor_truth(
                            sharp_prob_over_anchor, sharp_prob_under_anchor,
                            raw_anchor_over, prob_over_line,
                        )
                        if truth_over is None:
                            continue  # degenerate anchor prob — can't shift safely
                        # Shrink toward the books' devigged consensus at THIS
                        # line: the CDF shift borrows the model's tail shape,
                        # which settled-bet calibration showed is overconfident
                        # exactly at alt-line tails.
                        if alt_line_devigs and ALT_LINE_MARKET_SHRINK > 0:
                            consensus_over = sum(alt_line_devigs) / len(alt_line_devigs)
                            w = min(1.0, max(0.0, ALT_LINE_MARKET_SHRINK))
                            truth_over = w * consensus_over + (1.0 - w) * truth_over
                            truth_under = 1.0 - truth_over
                        edge_source = ('model_altline_consensus' if consensus_anchor
                                       else 'model_altline')
                    for side, odds_val, side_book, sharp_prob, is_steam in [
                        ('over', over_odds, over_book, truth_over, steam_over),
                        ('under', under_odds, under_book, truth_under, steam_under),
                    ]:
                        # Skip blacklisted market+side combos
                        if (market_key, side) in MARKET_SIDE_BLACKLIST:
                            continue
                        if not odds_val or odds_val <= 1.0:
                            continue
                        opening_prob = opening_row[f'devigged_{side}'] if opening_row else None
                        edge_result = rank_edge(
                            line_proj, odds_val, side, sharp_prob,
                            opening_prob=opening_prob, steam_detected=is_steam,
                        )
                        if MODEL_AS_TRUTH_MODE and edge_result.get('kelly'):
                            edge_result['kelly']['kelly_fraction'] = round(edge_result['kelly']['kelly_fraction'] * MODEL_ONLY_KELLY_MULT, 4)
                            edge_result['kelly']['recommended_stake'] = round(edge_result['kelly']['recommended_stake'] * MODEL_ONLY_KELLY_MULT, 2)
                        # Bookmaker bias boost: if this book systematically
                        # underprices this (market, side) vs sharp, add a small
                        # edge credit and re-flag as playable if it crosses the bar.
                        bias = bias_map.get((side_book, market_key, side), 0.0)
                        if bias > BOOKMAKER_BIAS_THRESHOLD:
                            edge_result = dict(edge_result)
                            edge_result['edge_pct'] = round(
                                edge_result['edge_pct'] + 0.5, 4
                            )
                            edge_result['bias_boosted'] = True
                            # The boost may only rescue a bet whose SOLE kill
                            # reason was the edge bar. Injury, sample-size,
                            # steam-against, or negative-Kelly kills stand.
                            reasons = edge_result.get('reasons', [])
                            edge_killed_only = (
                                len(reasons) == 1
                                and reasons[0].startswith('Edge too small')
                            )
                            if not edge_result['is_playable'] and edge_killed_only:
                                edge_result['is_playable'] = edge_result['edge_pct'] >= EDGE_MIN
                                if edge_result['is_playable']:
                                    logger.debug(
                                        "Bias boost made playable: %s %s %s @ %s (bias=%.3f)",
                                        player_name, market_key, side, side_book, bias,
                                    )
                        if edge_result['is_playable']:
                            edge_result['edge_source'] = edge_source
                            candidates.append((
                                edge_result['ev'], line, side, side_book,
                                odds_val, edge_result, sharp_book,
                            ))

                # Pick top MAX_BETS_PER_PLAYER by EV; persist the winners so
                # send_alerts consumes exactly this decision instead of
                # re-deriving bets from raw snapshots.
                candidates.sort(key=lambda c: c[0], reverse=True)
                winners = candidates[:MAX_BETS_PER_PLAYER]
                for ev, line, side, side_book, odds_val, edge_result, sharp_book_used in winners:
                    total_edges += 1
                    logger.info(
                        f"EDGE FOUND: {player_name} {market_key} {side.upper()} {line} "
                        f"@ {side_book} (vs {sharp_book_used}, anchor {anchor_line}) | "
                        f"Edge: {edge_result['edge_pct']:.1f}% | EV: {ev:.3f} | "
                        f"Kelly: ${edge_result['kelly']['recommended_stake']}"
                    )

                    # CLV opening basis: devig the chosen book's own two-sided
                    # quote at this line; fall back to the truth prob when the
                    # book is one-sided.
                    open_devig = None
                    book_pair = lines_for_market.get(line, {}).get(side_book) or {}
                    dev_o, dev_u = devig_multiplicative(
                        book_pair.get('over'), book_pair.get('under')
                    )
                    if dev_o is not None:
                        open_devig = dev_o if side == 'over' else dev_u
                    if open_devig is None:
                        open_devig = edge_result['sharp_prob']

                    conn.execute('''
                        INSERT INTO bet_candidates
                        (game_id, player_id, player_name, market, line, side,
                         bookmaker, odds, sharp_book, anchor_line, truth_prob,
                         model_prob, open_devig_prob, edge_pct, ev,
                         kelly_fraction, recommended_stake, steam_detected,
                         edge_source, created_at)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        ON CONFLICT(game_id, player_name, market, line, side, bookmaker)
                        DO UPDATE SET
                            odds=excluded.odds,
                            truth_prob=excluded.truth_prob,
                            model_prob=excluded.model_prob,
                            open_devig_prob=excluded.open_devig_prob,
                            edge_pct=excluded.edge_pct,
                            ev=excluded.ev,
                            kelly_fraction=excluded.kelly_fraction,
                            recommended_stake=excluded.recommended_stake,
                            steam_detected=excluded.steam_detected,
                            edge_source=excluded.edge_source,
                            created_at=excluded.created_at
                    ''', (
                        game_id, projection.get('player_id'), player_name,
                        market_key, line, side, side_book, odds_val,
                        sharp_book_used, anchor_line,
                        edge_result['sharp_prob'], edge_result['model_prob'],
                        open_devig, edge_result['edge_pct'], ev,
                        edge_result['kelly']['kelly_fraction'],
                        edge_result['kelly']['recommended_stake'],
                        int(edge_result.get('steam_detected', False)),
                        edge_result.get('edge_source'),
                        timestamp,
                    ))

                if winners:
                    ctx = dict(projection.get('context', {}))
                    ctx['sample_size'] = projection.get('sample_size', 0)
                    # Carry dispersion params so the alert can show an honest
                    # ± band instead of a bare point estimate. Only when present
                    # (keeps the heuristic/mock context shape unchanged).
                    for k in ('alpha', 'sigma', 'pi0'):
                        if projection.get(k) is not None:
                            ctx[k] = projection[k]
                    context_json = json.dumps(ctx)
                    conn.execute('''
                        INSERT INTO projections
                        (game_id, player_name, market, projected_mean,
                         prob_over, prob_under, context_json, timestamp)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                        ON CONFLICT(game_id, player_name, market) DO UPDATE SET
                            projected_mean=excluded.projected_mean,
                            prob_over=excluded.prob_over,
                            prob_under=excluded.prob_under,
                            context_json=excluded.context_json,
                            timestamp=excluded.timestamp
                    ''', (
                        game_id, player_name, market_key,
                        projection['projected_mean'],
                        projection['prob_over'], projection['prob_under'],
                        context_json, timestamp,
                    ))

                conn.execute(
                    "UPDATE games SET last_scanned_at = ? WHERE game_id = ?",
                    (now_utc.isoformat(), game_id),
                )
                conn.commit()

    if sdio_client is not None:
        await sdio_client.close()

    skipped_total = sum(skip_reasons.values())
    if skip_reasons:
        breakdown = ", ".join(f"{k}={v}" for k, v in sorted(skip_reasons.items()))
        skip_str = f"skipped {skipped_total} games ({breakdown})"
    else:
        skip_str = f"skipped {skipped_total} games"
    logger.info(f"Scan complete. Found {total_edges} playable edges; {skip_str}.")

    # Surface silent model degradation: a loaded GLM that failed feature-build or
    # predict fell back to the crude heuristic (no Statcast). This is invisible
    # per-prop; aggregate it so a broken model/feature pipeline doesn't quietly
    # downgrade the whole slate.
    from src.models.projections import pop_glm_degradations
    degradations = pop_glm_degradations()
    if degradations:
        logger.warning(
            "GLM degraded to heuristic on %d projection(s) this scan: %s",
            sum(degradations.values()), degradations,
        )


def _shift_anchor_truth(sharp_over: float, sharp_under: float,
                        model_anchor_over: float, model_alt_over: float):
    """Translate the devigged SHARP probability from the anchor line to an
    alt-line along the model's CDF, keeping sharp as the basis.

    The old code used the raw model probability as 'truth' at alt-lines, which
    let the model grade its own homework: the edge became (model - book) and the
    model/sharp agreement gate compared the model against itself. Here the sharp
    anchor probability is scaled by the model's relative CDF movement
    (odds-ratio style) and renormalized, so the edge stays tethered to the sharp
    market while the model only supplies the *shape* between lines.

    Returns (truth_over, truth_under), or (None, None) when the anchor model
    probability is degenerate (can't form a ratio) — caller should skip the line.
    """
    if not (0.0 < model_anchor_over < 1.0):
        return None, None
    over = sharp_over * (model_alt_over / model_anchor_over)
    under = sharp_under * ((1.0 - model_alt_over) / (1.0 - model_anchor_over))
    total = over + under
    if total <= 0:
        return None, None
    return over / total, under / total


def _parse_iso(ts: str):
    """Parse an ISO timestamp to UTC-aware datetime. Returns None on bad input."""
    if not ts:
        return None
    try:
        dt = datetime.fromisoformat(ts.replace('Z', '+00:00'))
    except (ValueError, AttributeError):
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt


def _detect_steam_in_soft_books(conn, game_id, player_name, market_key, line, side, current_line_data, sharp_books):
    """
    Check if at least 3 soft books have shortened their odds on 'side' 
    compared to their most recent recorded snapshot.
    """
    steam_count = 0
    sharp_set = set(sharp_books)
    for book, pair in current_line_data.items():
        if book in sharp_set:
            continue
        curr_odds = pair.get(side)
        if not curr_odds:
            continue
            
        last_snap = conn.execute('''
            SELECT over_odds, under_odds
            FROM prop_snapshots
            WHERE game_id = ? AND player_name = ? AND market = ? AND line = ? AND bookmaker = ?
            ORDER BY timestamp DESC
            LIMIT 1
        ''', (game_id, player_name, market_key, line, book)).fetchone()
        
        if last_snap:
            last_odds = last_snap[f'{side}_odds']
            if last_odds and curr_odds < last_odds:  # Odds shrank -> Implied prob increased
                steam_count += 1
                
    return steam_count >= 3


def _should_scan_game(game, now_utc: datetime) -> tuple[str, str | None]:
    """
    Eligibility gate. Returns ("scan", None) if the game should be scanned,
    else ("skip", reason). Reasons:
      - "already_started":   first pitch has passed — pregame pipeline never
                             produces in-play alerts (the live daemon does)
      - "no_lineup_confirm": lineups not yet stamped, and outside pregame window
      - "already_scanned":   lineups stamped but already scanned since
      - "recently_scanned":  in the window but scanned within PREGAME_RESCAN_MINUTES
      - "outside_window":    first pitch is not within PREGAME_WINDOW_MINUTES

    Scan when either:
      (a) lineups were confirmed after the last scan (new info → re-price), or
      (b) first pitch is within PREGAME_WINDOW_MINUTES (high-volatility window),
          throttled to one scan per PREGAME_RESCAN_MINUTES to bound quota.
    """
    confirmed_at = _parse_iso(game['lineups_confirmed_at'])
    last_scanned = _parse_iso(game['last_scanned_at'])
    game_time = _parse_iso(game['game_time'])

    # Hard guard: a game whose first pitch has passed is never scanned by the
    # pregame pipeline — this is what keeps a late lineup confirmation (or a
    # forced run) from firing an in-play alert.
    if game_time is not None and game_time <= now_utc:
        return ("skip", "already_started")

    # (a) Lineup drop triggers one re-scan.
    if confirmed_at is not None:
        if last_scanned is None or last_scanned < confirmed_at:
            return ("scan", None)

    # (b) Pregame window: first pitch within PREGAME_WINDOW_MINUTES from now.
    if game_time is not None:
        minutes_until = (game_time - now_utc).total_seconds() / 60.0
        if 0 <= minutes_until <= PREGAME_WINDOW_MINUTES:
            if (last_scanned is None
                    or (now_utc - last_scanned).total_seconds() >= PREGAME_RESCAN_MINUTES * 60):
                return ("scan", None)
            return ("skip", "recently_scanned")

    if confirmed_at is None:
        return ("skip", "no_lineup_confirm")
    if last_scanned is not None and last_scanned >= confirmed_at:
        return ("skip", "already_scanned")
    return ("skip", "outside_window")


def _pick_best_line(line_data: dict) -> dict:
    """
    Given {book: {'over': odds, 'under': odds}}, pick the max odds per side
    across all books. Returns {'over': (odds|None, book|None),
    'under': (odds|None, book|None)}. Over and under may come from different
    books — that's the point of shopping.

    Kept for back-compat / tests; scan_props now uses _pick_best_soft_line
    to exclude sharp books from the hunt.
    """
    best_over = (None, None)
    best_under = (None, None)
    for book, pair in line_data.items():
        over = pair.get('over')
        under = pair.get('under')
        if over is not None and (best_over[0] is None or over > best_over[0]):
            best_over = (over, book)
        if under is not None and (best_under[0] is None or under > best_under[0]):
            best_under = (under, book)
    return {'over': best_over, 'under': best_under}


def _pick_sharp_pair(line_data: dict, sharp_books: list):
    """
    Walk sharp_books in priority order; return the first (over_odds, under_odds,
    book) triple where that sharp book quotes both sides with valid prices.
    Returns None if no sharp book has a full two-sided quote.
    """
    for book in sharp_books:
        pair = line_data.get(book)
        if not pair:
            continue
        over = pair.get('over')
        under = pair.get('under')
        if not over or not under:
            continue
        if over <= 1.0 or under <= 1.0:
            continue
        return (over, under, book)
    return None


def _pick_best_soft_line(line_data: dict, sharp_books: list) -> dict:
    """
    Same as _pick_best_line but restricted to non-sharp books. This is the
    "rogue line hunt": we compare soft-book offers against sharp consensus,
    so the sharp book's own price should never be the target we're betting.
    """
    sharp_set = set(sharp_books)
    best_over = (None, None)
    best_under = (None, None)
    for book, pair in line_data.items():
        if book in sharp_set:
            continue
        over = pair.get('over')
        under = pair.get('under')
        if over is not None and (best_over[0] is None or over > best_over[0]):
            best_over = (over, book)
        if under is not None and (best_under[0] is None or under > best_under[0]):
            best_under = (under, book)
    return {'over': best_over, 'under': best_under}


def _group_by_player_market(player_lines: dict) -> dict:
    """Re-bucket {(player, market, line): book_data} into
    {(player, market): {line: book_data}}. Lets the scan loop iterate one
    (player, market) at a time and consider every quoted line within range
    instead of only sharp-anchored lines."""
    grouped = {}
    for (player, market, line), book_data in player_lines.items():
        grouped.setdefault((player, market), {})[line] = book_data
    return grouped


def _pick_anchor_line(lines_for_market: dict, sharp_books: list):
    """Find the first line for a (player, market) where a sharp book quotes
    a valid two-sided pair. Returns (line, sharp_over, sharp_under, book) or
    None. The anchor's devigged probability is the source-of-truth used to
    re-price every alt-line in this market."""
    for line in sorted(lines_for_market.keys()):
        sharp_pair = _pick_sharp_pair(lines_for_market[line], sharp_books)
        if sharp_pair is None:
            continue
        sharp_over, sharp_under, book = sharp_pair
        return (line, sharp_over, sharp_under, book)
    return None


def _parse_odds_by_player(event_odds: dict) -> dict:
    """
    Parse the Odds API response into a structure grouped by (player, market, line).
    Returns: { (player, market, line): { bookmaker: { 'over': odds, 'under': odds } } }
    """
    result = {}

    for bookmaker in event_odds.get('bookmakers', []):
        book = bookmaker['key']
        for market_data in bookmaker.get('markets', []):
            market_key = market_data['key']
            if market_key not in MARKETS_MAPPING:
                continue

            for outcome in market_data.get('outcomes', []):
                if 'point' not in outcome:
                    continue

                player_name = outcome.get('description', 'Unknown')
                line = float(outcome['point'])
                price = float(outcome['price'])
                side = outcome['name'].lower()

                key = (player_name, market_key, line)
                if key not in result:
                    result[key] = {}
                if book not in result[key]:
                    result[key][book] = {}

                result[key][book][side] = price

    return result


def _record_total_snapshot(game_id: str, total: float, source: str) -> None:
    """Persist a game-total observation. Read by trigger_watch as the
    baseline for between-scan shift detection."""
    try:
        with get_db_connection() as conn:
            conn.execute(
                "INSERT INTO game_totals_history (game_id, total, source, timestamp) "
                "VALUES (?, ?, ?, ?)",
                (game_id, float(total), source, datetime.now(timezone.utc).isoformat()),
            )
            conn.commit()
    except Exception as e:
        logger.warning(f"Failed to record game total for {game_id}: {e}")


def _parse_game_total(event_odds: dict) -> float | None:
    """Extract the consensus game total (over/under) line from the API response.

    Returns the median line across bookmakers, or None if unavailable.
    The totals market has outcomes with name="Over"/"Under" and point=<line>,
    with no 'description' field (game-level market, not player-level).
    """
    lines = []
    for bookmaker in event_odds.get('bookmakers', []):
        for market_data in bookmaker.get('markets', []):
            if market_data['key'] != 'totals':
                continue
            for outcome in market_data.get('outcomes', []):
                if 'point' in outcome:
                    lines.append(float(outcome['point']))
                    break  # one line per bookmaker is enough
    if not lines:
        return None
    lines.sort()
    return lines[len(lines) // 2]


def _build_projection(proj_model: ProjectionModel, player_name: str,
                      market_key: str, line: float,
                      game_id: str, home_team: str, away_team: str,
                      venue: str, weather: dict = None,
                      ump_k_factor: float = 1.0,
                      game_total: float = None,
                      bdl_market: dict = None) -> dict:
    """Build a projection for a player+market by looking up their stats in the DB."""
    with get_db_connection() as conn:
        # Resolve the player deterministically (cache -> exact -> accent/suffix-
        # normalized unique). The old `name LIKE '%X%'` first-row match could pull
        # an entirely different player's logs on duplicate MLB names (two Will
        # Smiths, multiple José Ramírez). We never project a bet on a guess: an
        # ambiguous or unresolved name skips the prop.
        from src.data.player_resolver import resolve_player_id
        pid, method = resolve_player_id(conn, player_name)
        if pid is None:
            logger.info(
                "Skipping %s %s: name unresolved (%s) — refusing to project on a guess.",
                player_name, market_key, method,
            )
            return None
        player = conn.execute(
            "SELECT * FROM players WHERE player_id = ?", (pid,)
        ).fetchone()
        if not player:
            return None

        player_id = player['player_id']
        team_id = player['team_id']
        position = player['position'] or ''
        bats = player['bats'] or ''
        throws = player['throws'] or ''
        
        # Determine opposing team ID. Resolve via the exact Odds-API->abbrev map
        # rather than `name LIKE '%City%'`, which collides on "Chicago"
        # (Cubs/White Sox) and "Los Angeles" (Angels/Dodgers).
        home_id = _resolve_team_id(conn, home_team)
        away_id = _resolve_team_id(conn, away_team)
        opp_team_id = away_id if team_id == home_id else home_id

        # Check injury status
        today = str(get_eastern_local_date())
        injury = conn.execute(
            "SELECT status FROM injury_reports WHERE player_name = ? AND date = ?",
            (player_name, today)
        ).fetchone()
        injury_status = injury['status'] if injury else 'Healthy'

        if injury_status in ('IL', 'Out'):
            return None

        # Confirmed-starter gate (pitcher markets). Bet the listed probable
        # pitcher only — never a pitcher who isn't confirmed to start (an
        # unconfirmed/scratched arm would otherwise project off stale logs and
        # only void post-hoc). Batter lineup confirmation is enforced in the
        # batter branch where the lineup slot is looked up.
        if (REQUIRE_CONFIRMED_LINEUP
                and market_key in ('pitcher_strikeouts', 'pitcher_earned_runs')
                and not _is_confirmed_starter(conn, game_id, player_id)):
            logger.info(
                "Skipping %s %s: starting pitcher not confirmed for game %s.",
                player_name, market_key, game_id,
            )
            return None

        # LLM injury signal (optional): a tightening-only gate. When the player
        # already carries a non-clear status AND the model estimates a low
        # probability of appearing, skip the prop. The LLM never loosens a
        # decision and never changes a projection number; its impact_summary is
        # carried into the projection context for the alert rationale.
        injury_summary = None
        if injury_status != 'Healthy':
            from src.pipelines.enrich_injuries import get_injury_signal
            from src.config import LLM_INJURY_SKIP_PROBABILITY
            sig = get_injury_signal(conn, player_id, today)
            if sig:
                injury_summary = sig.get('impact_summary')
                prob = sig.get('play_probability')
                if (sig.get('play_status') == 'out'
                        or (prob is not None and prob < LLM_INJURY_SKIP_PROBABILITY)):
                    logger.info(
                        "LLM injury gate skipped %s (%s, p_play=%s): %s",
                        player_name, injury_status, prob, injury_summary,
                    )
                    return None
            
        pitcher_bullpen_era = compute_bullpen_factor(team_id, today, db=conn)
        opp_bullpen_era = compute_bullpen_factor(opp_team_id, today, db=conn)
        # Materialize DB-derived features here so the conn never escapes this
        # context manager. Downstream feature builders prefer these scalars
        # over their `extra["db"]` fallback path.
        h2h_k_delta = compute_pitcher_h2h_vs_team(player_id, opp_team_id, today, db=conn)
        extra_features = {
            'pitcher_bullpen_era': pitcher_bullpen_era,
            'opp_bullpen_era': opp_bullpen_era,
            'opp_team_id': opp_team_id,
            'pitcher_id': player_id,
            'h2h_k_delta': h2h_k_delta,
            'db': conn,
        }

        # Build projection based on market type
        if market_key == 'pitcher_strikeouts':
            logs = conn.execute(
                "SELECT * FROM pitcher_game_logs WHERE player_id = ? ORDER BY date DESC",
                (player_id,)
            ).fetchall()
            logs = [dict(l) for l in logs]

            # Opponent = the team the pitcher faces (opp_team_id), NOT always
            # the away team — for away starters that would be their own club.
            from src.config import LEAGUE_AVG_K_RATE
            opp_team_stats = _get_team_stats_by_id(conn, opp_team_id)
            opp_k_rate = opp_team_stats.get('k_rate') or LEAGUE_AVG_K_RATE

            # Arsenal matchup: this pitcher's pitch mix vs the opposing lineup's
            # per-pitch whiff. Bounded multiplier, neutral 1.0 when data is thin.
            matchup_factor = 1.0
            if PITCH_MATCHUP_ENABLED:
                opp_batter_ids = [
                    r['player_id'] for r in conn.execute(
                        "SELECT dl.player_id FROM daily_lineups dl "
                        "JOIN players p ON p.player_id = dl.player_id "
                        "WHERE dl.game_id = ? AND p.team_id = ?",
                        (game_id, opp_team_id),
                    ).fetchall() if r['player_id']
                ]
                matchup_factor = pitcher_k_factor_db(
                    conn, player_id, opp_batter_ids, MLB_SEASON
                )

            proj = proj_model.project_pitcher_strikeouts(
                logs, opp_k_rate, venue, line, weather=weather, ump_k_factor=ump_k_factor,
                extra_features=extra_features, player_id=player_id,
                matchup_factor=matchup_factor,
            )
            if proj:
                proj['injury_status'] = injury_status
                proj['player_id'] = player_id
                if injury_summary:
                    proj.setdefault('context', {})['llm_injury_summary'] = injury_summary
            return proj

        elif market_key == 'pitcher_earned_runs':
            logs = conn.execute(
                "SELECT * FROM pitcher_game_logs WHERE player_id = ? ORDER BY date DESC",
                (player_id,)
            ).fetchall()
            logs = [dict(l) for l in logs]

            from src.config import LEAGUE_AVG_RUNS_PER_GAME
            opp_team_stats = _get_team_stats_by_id(conn, opp_team_id)
            opp_runs_pg = opp_team_stats.get('runs_per_game') or LEAGUE_AVG_RUNS_PER_GAME

            proj = proj_model.project_pitcher_earned_runs(
                logs, opp_runs_pg, venue, line, weather=weather, ump_k_factor=ump_k_factor,
                extra_features=extra_features, player_id=player_id,
            )
            if proj:
                proj['injury_status'] = injury_status
                proj['player_id'] = player_id
                if injury_summary:
                    proj.setdefault('context', {})['llm_injury_summary'] = injury_summary
            return proj

        elif market_key in ('batter_hits', 'batter_total_bases', 'batter_home_runs'):
            logs = conn.execute(
                "SELECT * FROM batter_game_logs WHERE player_id = ? ORDER BY date DESC",
                (player_id,)
            ).fetchall()
            logs = [dict(l) for l in logs]

            stat_type = {
                'batter_hits': 'hits',
                'batter_total_bases': 'total_bases',
                'batter_home_runs': 'home_runs',
            }[market_key]

            # Get opposing pitcher's throwing hand from probable pitchers
            pitcher_hand = _get_opposing_pitcher_hand(conn, game_id, home_team, away_team, player)

            # Look up today's lineup position for PA projection (by id when the
            # name lookup misses — the resolver gave us a trustworthy player_id).
            lineup_position = _get_lineup_position(conn, player_name, game_id)
            if lineup_position is None and player_id:
                row = conn.execute(
                    "SELECT lineup_position FROM daily_lineups WHERE game_id = ? AND player_id = ?",
                    (game_id, player_id),
                ).fetchone()
                if row and row['lineup_position']:
                    lineup_position = int(row['lineup_position'])

            # Confirmed-lineup gate (batter markets): a batter not in today's
            # posted lineup would otherwise project off DEFAULT_PROJECTED_PA and
            # only void post-hoc if they sit. Skip rather than guess.
            if REQUIRE_CONFIRMED_LINEUP and lineup_position is None:
                logger.info(
                    "Skipping %s %s: batter not in confirmed lineup for game %s.",
                    player_name, market_key, game_id,
                )
                return None

            # Materialize platoon rate here while conn is alive so the feature
            # builder doesn't need to reach back into the DB after this block.
            batter_extra = dict(extra_features)
            batter_extra['batter_id'] = player_id
            batter_extra['platoon_rate_vs_hand'] = compute_platoon_split(
                batter_logs=logs, vs_hand=pitcher_hand or '', stat_key=stat_type,
                db=conn, batter_id=player_id,
            )

            # Moneyline-tilted implied team total when BDL supplied moneylines:
            # the batter's club's share of the game total leans to the favorite.
            itt_override = None
            if (bdl_market and bdl_market.get('ml_home') is not None
                    and bdl_market.get('ml_away') is not None):
                from src.models.pa_estimator import implied_team_total
                batter_side = 'home' if team_id == home_id else 'away'
                itt_override = implied_team_total(
                    game_total, bdl_market['ml_home'], bdl_market['ml_away'],
                    side=batter_side,
                )

            # Matchup multipliers vs the opposing starter (resolved once).
            # Both are bounded and neutral 1.0 when data is thin:
            #   - arsenal: batter's per-pitch xwoba vs the starter's pitch mix.
            #   - BvP: batter's shrunk career line vs this exact pitcher.
            matchup_factor = 1.0
            if PITCH_MATCHUP_ENABLED or BVP_ENABLED:
                opp_pitcher = conn.execute(
                    "SELECT pp.player_id FROM probable_pitchers pp "
                    "JOIN players p ON p.player_id = pp.player_id "
                    "WHERE pp.game_id = ? AND p.team_id = ? LIMIT 1",
                    (game_id, opp_team_id),
                ).fetchone()
                opp_pitcher_id = opp_pitcher['player_id'] if opp_pitcher else None
                if opp_pitcher_id:
                    if PITCH_MATCHUP_ENABLED:
                        matchup_factor *= batter_arsenal_factor_db(
                            conn, player_id, opp_pitcher_id, MLB_SEASON
                        )
                    if BVP_ENABLED:
                        total_stat = sum(l.get(stat_type, 0) or 0 for l in logs)
                        total_pa = sum(
                            (l.get('plate_appearances') or l.get('at_bats') or 0)
                            for l in logs
                        )
                        baseline = (total_stat / total_pa) if total_pa > 0 else 0.0
                        matchup_factor *= bvp_factor_db(
                            conn, player_id, opp_pitcher_id, market_key, baseline
                        )

            proj = proj_model.project_batter_stat(
                logs, stat_type, pitcher_hand, bats, venue, line,
                lineup_position=lineup_position,
                weather=weather,
                extra_features=batter_extra,
                player_id=player_id,
                game_total=game_total,
                implied_team_total_override=itt_override,
                matchup_factor=matchup_factor,
            )
            if proj:
                proj['injury_status'] = injury_status
                proj['player_id'] = player_id
                if injury_summary:
                    proj.setdefault('context', {})['llm_injury_summary'] = injury_summary
            return proj

    return None


def _is_confirmed_starter(conn, game_id: str, player_id: int) -> bool:
    """True iff this player is the confirmed probable starter for this game.

    probable_pitchers is populated by sync_lineups from BDL's /lineups feed; a
    row keyed to this (game_id, player_id) is our 'starter confirmed' signal.
    """
    if not player_id:
        return False
    row = conn.execute(
        "SELECT 1 FROM probable_pitchers WHERE game_id = ? AND player_id = ? LIMIT 1",
        (game_id, player_id),
    ).fetchone()
    return row is not None


def _resolve_team_id(conn, team_name: str):
    """Resolve an Odds-API team name to a BDL team_id without LIKE collisions.

    Priority: exact Odds-API->abbreviation map (config) joined on teams.abbreviation,
    then exact name match, then a logged fuzzy LIKE as a last resort. Returns the
    team_id or None. The abbreviation path is what disambiguates "Chicago" and
    "Los Angeles", which `name LIKE '%City%'` cannot.
    """
    if not team_name:
        return None
    from src.config import ODDS_API_TEAM_ABBREV
    abbrev = ODDS_API_TEAM_ABBREV.get(team_name)
    if abbrev:
        row = conn.execute(
            "SELECT team_id FROM teams WHERE abbreviation = ? COLLATE NOCASE",
            (abbrev,),
        ).fetchone()
        if row:
            return row['team_id']
    row = conn.execute(
        "SELECT team_id FROM teams WHERE name = ? COLLATE NOCASE", (team_name,)
    ).fetchone()
    if row:
        return row['team_id']
    row = conn.execute(
        "SELECT team_id FROM teams WHERE name LIKE ? COLLATE NOCASE",
        (f"%{team_name}%",),
    ).fetchone()
    if row:
        logger.debug("Team '%s' resolved via fuzzy LIKE — verify abbrev map.", team_name)
        return row['team_id']
    return None


def _get_team_stats_by_id(conn, team_id) -> dict:
    """Pre-calculated team offensive stats (calculate_team_stats pipeline),
    looked up directly by BDL team_id. Returns {} when unknown so callers
    fall back to league averages."""
    if not team_id:
        return {}

    stats = conn.execute(
        "SELECT k_rate, runs_per_game FROM team_stats WHERE team_id = ?",
        (team_id,)
    ).fetchone()

    if not stats:
        return {}

    return {'k_rate': stats['k_rate'], 'runs_per_game': stats['runs_per_game']}


def _get_team_stats(conn, team_name: str) -> dict:
    """
    Look up pre-calculated team offensive stats from the `team_stats` table.
    This table is populated by the `calculate_team_stats` pipeline.
    """
    # Find team ID
    team = conn.execute(
        "SELECT team_id FROM teams WHERE name LIKE ? COLLATE NOCASE",
        (f"%{team_name}%",)
    ).fetchone()

    if not team:
        return {}

    return _get_team_stats_by_id(conn, team['team_id'])


def _get_opposing_pitcher_hand(conn, game_id: str, home_team: str, away_team: str, batter_player) -> str:
    """
    Look up the opposing probable pitcher's throwing hand from the confirmed
    probable_pitchers table (BDL /lineups). Returns '' (neutral) when no
    confirmed starter exists.

    The previous "most recent pitcher on the opposing team" fallback guessed a
    hand from an unrelated pitcher's game log — that silently applied the WRONG
    platoon split. A wrong platoon adjustment is worse than none, so when the
    starter isn't confirmed we return '' and the platoon multiplier stays 1.0.
    """
    batter_team_id = batter_player['team_id']

    # Determine which team the batter is on to find the opposing team.
    home_id = _resolve_team_id(conn, home_team)
    opp_team_name = away_team if (home_id is not None and batter_team_id == home_id) else home_team

    pitcher = conn.execute(
        "SELECT throws FROM probable_pitchers WHERE game_id = ? AND team LIKE ? COLLATE NOCASE",
        (game_id, f"%{opp_team_name}%")
    ).fetchone()

    if pitcher and pitcher['throws']:
        return pitcher['throws']

    logger.debug(
        "No confirmed probable pitcher for %s in game %s — neutral platoon (no guess).",
        opp_team_name, game_id,
    )
    return ''


def _get_ump_k_factor(game_id: str) -> float:
    """
    Look up the home-plate umpire's K factor for a game and return a
    blended adjustment suitable for use in pitcher projections.

    Returns 1.0 (neutral) when:
      - No umpire assignment exists for this game
      - The umpire has fewer than UMP_MIN_GAMES games called (too small a sample)

    Otherwise returns:
      1 + (raw_k_factor - 1) * UMP_K_WEIGHT

    The partial-weight blend (default 0.5) prevents overconfidence in a single
    umpire's historical tendency and smooths regression to the mean.
    """
    with get_db_connection() as conn:
        row = conn.execute(
            """
            SELECT us.k_factor, us.games_called
            FROM umpire_game_assignments uga
            JOIN umpire_stats us ON uga.umpire_id = us.umpire_id
            WHERE uga.game_id = ?
            """,
            (game_id,),
        ).fetchone()

    if not row or row['k_factor'] is None:
        return 1.0
    if row['games_called'] < UMP_MIN_GAMES:
        return 1.0

    raw_factor = row['k_factor']
    return 1.0 + (raw_factor - 1.0) * UMP_K_WEIGHT


def _get_lineup_position(conn, player_name: str, game_id: str) -> int:
    """
    Look up a player's lineup position for today's game.
    Returns None if not found (projection will use DEFAULT_PROJECTED_PA).
    """
    row = conn.execute(
        "SELECT lineup_position FROM daily_lineups WHERE player_name = ? AND game_id = ? COLLATE NOCASE",
        (player_name, game_id)
    ).fetchone()

    if row and row['lineup_position']:
        return int(row['lineup_position'])

    # Try partial name match
    row = conn.execute(
        "SELECT lineup_position FROM daily_lineups WHERE player_name LIKE ? AND game_id = ? COLLATE NOCASE",
        (f"%{player_name}%", game_id)
    ).fetchone()

    if row and row['lineup_position']:
        return int(row['lineup_position'])

    return None
