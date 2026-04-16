import uuid
import json
import os
from datetime import datetime, timezone, timedelta
from src.config import (
    MARKETS_MAPPING, UMP_MIN_GAMES, UMP_K_WEIGHT, PREGAME_WINDOW_MINUTES,
    SHARP_BOOKMAKERS,
)
from src.clients.odds_api import OddsAPIClient
from src.clients.weather import WeatherClient
from src.data.db import get_db_connection
from src.data.park_factors import get_stadium_meta
from src.models.devig import devig_multiplicative
from src.models.projections import ProjectionModel
from src.models.edge_ranker import rank_edge
from src.utils.logging_utils import get_logger
from src.data.feature_builder import compute_bullpen_factor
from src.utils.time_utils import get_eastern_local_date

logger = get_logger(__name__)


def scan_props(force: bool = False, game_ids: list = None):
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
    odds_client = OddsAPIClient()
    weather_client = WeatherClient()
    proj_model = ProjectionModel()

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
                f"SELECT * FROM games WHERE status != 'COMPLETED' "
                f"AND game_id IN ({placeholders})",
                tuple(game_ids),
            ).fetchall()
        else:
            games = conn.execute("SELECT * FROM games WHERE status != 'COMPLETED'").fetchall()

    if not games:
        logger.info("No active games found for odds scanning.")
        return

    # All markets in one call; include game totals for PA scaling (zero extra quota)
    markets = list(MARKETS_MAPPING.keys())
    api_markets = markets + ["totals"]
    total_edges = 0
    skipped_quota = 0

    for game in games:
        game_id = game['game_id']
        home_team = game['home_team']
        away_team = game['away_team']
        venue = game['venue']

        if not force and not _should_scan_game(game, now_utc):
            skipped_quota += 1
            continue

        logger.info(f"Scanning: {away_team} @ {home_team}")

        # Fetch live weather for this stadium
        weather = None
        meta = get_stadium_meta(venue)
        if meta and meta.get("roof") != "dome":
            weather = weather_client.get_game_weather(meta["lat"], meta["lon"])

        # Umpire K factor — looked up once per game, applied to all pitcher props
        ump_k_factor = _get_ump_k_factor(game_id)

        try:
            event_odds = odds_client.get_event_odds(game_id, api_markets, bust_cache=targeted)
        except Exception as e:
            logger.error(f"Failed to fetch odds for {game_id}: {e}")
            continue

        if not event_odds:
            continue

        # 2. Parse odds and group by player+market+line for devigging
        player_lines = _parse_odds_by_player(event_odds)
        game_total = _parse_game_total(event_odds)

        # 3. For each player+market+line:
        #    (a) Devig the sharp book → TRUE probability (source of truth).
        #    (b) Hunt soft books for the best offer deviating from sharp consensus.
        #    (c) rank_edge compares sharp_prob vs the soft-book's implied price.
        #    Props with no sharp quote are skipped — we don't bet without truth.
        for key, line_data in player_lines.items():
            player_name, market_key, line = key

            sharp_pair = _pick_sharp_pair(line_data, SHARP_BOOKMAKERS)
            if sharp_pair is None:
                continue

            sharp_over_odds, sharp_under_odds, sharp_book = sharp_pair
            sharp_prob_over, sharp_prob_under = devig_multiplicative(
                sharp_over_odds, sharp_under_odds
            )

            soft_best = _pick_best_soft_line(line_data, SHARP_BOOKMAKERS)
            over_odds, over_book = soft_best['over']
            under_odds, under_book = soft_best['under']
            if not over_odds or not under_odds:
                continue
            if over_odds <= 1.0 or under_odds <= 1.0:
                continue

            projection = _build_projection(
                proj_model, player_name, market_key, line,
                game_id, home_team, away_team, venue,
                weather=weather, ump_k_factor=ump_k_factor,
                game_total=game_total,
            )

            if not projection:
                continue

            projection['player_name'] = player_name

            # Composite book label when over and under come from different soft books
            book_label = over_book if over_book == under_book else f"{over_book}/{under_book}"

            snapshot_id = str(uuid.uuid4())
            timestamp = datetime.utcnow().isoformat()

            with get_db_connection() as conn:
                # Fetch opening prob before inserting new snapshot
                opening_row = conn.execute('''
                    SELECT devigged_over, devigged_under 
                    FROM prop_snapshots 
                    WHERE game_id = ? AND player_name = ? AND market = ? AND line = ?
                      AND devigged_over IS NOT NULL
                    ORDER BY timestamp ASC 
                    LIMIT 1
                ''', (game_id, player_name, market_key, line)).fetchone()

                # Primary snapshot: best soft-book offer (what we'd actually bet),
                # tagged with sharp-devigged truth in devigged_over/under.
                conn.execute('''
                    INSERT INTO prop_snapshots
                    (snapshot_id, game_id, player_name, market, line,
                     over_odds, under_odds, bookmaker, timestamp,
                     devigged_over, devigged_under)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    snapshot_id, game_id, player_name, market_key,
                    line, over_odds, under_odds, book_label, timestamp,
                    sharp_prob_over, sharp_prob_under,
                ))

                # Separate snapshot per sharp book (source of truth for CLV).
                for sb in SHARP_BOOKMAKERS:
                    sp = line_data.get(sb)
                    if not sp:
                        continue
                    s_over = sp.get('over')
                    s_under = sp.get('under')
                    if not s_over or not s_under or s_over <= 1.0 or s_under <= 1.0:
                        continue
                    s_dev_over, s_dev_under = devig_multiplicative(s_over, s_under)
                    conn.execute('''
                        INSERT INTO prop_snapshots
                        (snapshot_id, game_id, player_name, market, line,
                         over_odds, under_odds, bookmaker, timestamp,
                         devigged_over, devigged_under)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ''', (
                        str(uuid.uuid4()), game_id, player_name, market_key,
                        line, s_over, s_under, sb, timestamp,
                        s_dev_over, s_dev_under,
                    ))

                playable_any = False
                for side, odds_val, sharp_prob, side_book in [
                    ('over', over_odds, sharp_prob_over, over_book),
                    ('under', under_odds, sharp_prob_under, under_book),
                ]:
                    opening_prob = opening_row[f'devigged_{side}'] if opening_row else None
                    edge_result = rank_edge(projection, odds_val, side, sharp_prob, opening_prob=opening_prob)

                    if edge_result['is_playable']:
                        total_edges += 1
                        playable_any = True
                        logger.info(
                            f"EDGE FOUND: {player_name} {market_key} {side.upper()} {line} "
                            f"@ {side_book} (vs {sharp_book}) | Edge: {edge_result['edge_pct']:.1f}% | "
                            f"EV: {edge_result['ev']:.3f} | "
                            f"Kelly: ${edge_result['kelly']['recommended_stake']}"
                        )

                if playable_any:
                    context_json = json.dumps(projection.get('context', {}))
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

                # Stamp last_scanned_at so subsequent scans respect the gate
                conn.execute(
                    "UPDATE games SET last_scanned_at = ? WHERE game_id = ?",
                    (now_utc.isoformat(), game_id),
                )
                conn.commit()

    logger.info(
        f"Scan complete. Found {total_edges} playable edges; "
        f"skipped {skipped_quota} games by quota gate."
    )


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


def _should_scan_game(game, now_utc: datetime) -> bool:
    """
    Quota gate. Scan a game only if either:
      (a) lineups were confirmed after the last scan (new info → re-price), or
      (b) first pitch is within PREGAME_WINDOW_MINUTES (high-volatility window).
    """
    confirmed_at = _parse_iso(game['lineups_confirmed_at'])
    last_scanned = _parse_iso(game['last_scanned_at'])
    game_time = _parse_iso(game['game_time'])

    # (a) Lineup drop triggers one re-scan.
    if confirmed_at is not None:
        if last_scanned is None or last_scanned < confirmed_at:
            return True

    # (b) Pregame window: first pitch within PREGAME_WINDOW_MINUTES from now.
    if game_time is not None:
        minutes_until = (game_time - now_utc).total_seconds() / 60.0
        if 0 <= minutes_until <= PREGAME_WINDOW_MINUTES:
            return True

    return False


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
                      game_total: float = None) -> dict:
    """Build a projection for a player+market by looking up their stats in the DB."""
    with get_db_connection() as conn:
        # Find the player
        player = conn.execute(
            "SELECT * FROM players WHERE name = ? COLLATE NOCASE", (player_name,)
        ).fetchone()

        if not player:
            # Try partial match
            player = conn.execute(
                "SELECT * FROM players WHERE name LIKE ? COLLATE NOCASE",
                (f"%{player_name}%",)
            ).fetchone()

        if not player:
            return None

        player_id = player['player_id']
        team_id = player['team_id']
        position = player['position'] or ''
        bats = player['bats'] or ''
        throws = player['throws'] or ''
        
        # Determine opposing team ID
        home = conn.execute("SELECT team_id FROM teams WHERE name LIKE ? COLLATE NOCASE", (f"%{home_team}%",)).fetchone()
        away = conn.execute("SELECT team_id FROM teams WHERE name LIKE ? COLLATE NOCASE", (f"%{away_team}%",)).fetchone()
        
        home_id = home['team_id'] if home else None
        away_id = away['team_id'] if away else None
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
            
        # Precompute bullpen factors
        pitcher_bullpen_era = compute_bullpen_factor(team_id, today, db=conn)
        opp_bullpen_era = compute_bullpen_factor(opp_team_id, today, db=conn)
        extra_features = {
            'pitcher_bullpen_era': pitcher_bullpen_era,
            'opp_bullpen_era': opp_bullpen_era
        }

        # Build projection based on market type
        if market_key == 'pitcher_strikeouts':
            logs = conn.execute(
                "SELECT * FROM pitcher_game_logs WHERE player_id = ? ORDER BY date DESC",
                (player_id,)
            ).fetchall()
            logs = [dict(l) for l in logs]

            # Get opponent team K rate.
            # NOTE: _get_team_stats performs a heavy aggregation. For better performance,
            # this value should be pre-calculated by a separate pipeline and stored in a
            # `team_stats` table. The call would then be a simple lookup.
            from src.config import LEAGUE_AVG_K_RATE
            opp_team_stats = _get_team_stats(conn, away_team)
            opp_k_rate = opp_team_stats.get('k_rate', LEAGUE_AVG_K_RATE)

            proj = proj_model.project_pitcher_strikeouts(
                logs, opp_k_rate, venue, line, weather=weather, ump_k_factor=ump_k_factor,
                extra_features=extra_features, player_id=player_id,
            )
            if proj:
                proj['injury_status'] = injury_status
            return proj

        elif market_key == 'pitcher_earned_runs':
            logs = conn.execute(
                "SELECT * FROM pitcher_game_logs WHERE player_id = ? ORDER BY date DESC",
                (player_id,)
            ).fetchall()
            logs = [dict(l) for l in logs]

            from src.config import LEAGUE_AVG_RUNS_PER_GAME
            opp_team_stats = _get_team_stats(conn, away_team)
            opp_runs_pg = opp_team_stats.get('runs_per_game', LEAGUE_AVG_RUNS_PER_GAME)

            proj = proj_model.project_pitcher_earned_runs(
                logs, opp_runs_pg, venue, line, weather=weather, ump_k_factor=ump_k_factor,
                extra_features=extra_features, player_id=player_id,
            )
            if proj:
                proj['injury_status'] = injury_status
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

            # Look up today's lineup position for PA projection
            lineup_position = _get_lineup_position(conn, player_name, game_id)

            proj = proj_model.project_batter_stat(
                logs, stat_type, pitcher_hand, bats, venue, line,
                lineup_position=lineup_position,
                weather=weather,
                extra_features=extra_features,
                player_id=player_id,
                game_total=game_total,
            )
            if proj:
                proj['injury_status'] = injury_status
            return proj

    return None


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
 
    # Look up stats from the pre-calculated table
    stats = conn.execute(
        "SELECT k_rate, runs_per_game FROM team_stats WHERE team_id = ?",
        (team['team_id'],)
    ).fetchone()
 
    if not stats:
        return {}
 
    return {'k_rate': stats['k_rate'], 'runs_per_game': stats['runs_per_game']}


def _get_opposing_pitcher_hand(conn, game_id: str, home_team: str, away_team: str, batter_player) -> str:
    """
    Look up the opposing probable pitcher's throwing hand.

    Priority:
    1. probable_pitchers table (from BDL /lineups endpoint) — the correct answer
    2. Fallback: most recent pitcher on opposing team from game logs (old heuristic)
    """
    batter_team_id = batter_player['team_id']

    # Determine which team the batter is on to find the opposing team
    home = conn.execute(
        "SELECT team_id FROM teams WHERE name LIKE ? COLLATE NOCASE",
        (f"%{home_team}%",)
    ).fetchone()

    if home and batter_team_id == home['team_id']:
        opp_team_name = away_team
    else:
        opp_team_name = home_team

    # --- Primary: look up probable pitcher from today's lineup sync ---
    pitcher = conn.execute(
        "SELECT throws FROM probable_pitchers WHERE game_id = ? AND team LIKE ? COLLATE NOCASE",
        (game_id, f"%{opp_team_name}%")
    ).fetchone()

    if pitcher and pitcher['throws']:
        return pitcher['throws']

    # --- Fallback: most recent pitcher on opposing team (old heuristic) ---
    logger.debug(f"No probable pitcher for {opp_team_name} in game {game_id}, using fallback")

    opp_team = conn.execute(
        "SELECT team_id FROM teams WHERE name LIKE ? COLLATE NOCASE",
        (f"%{opp_team_name}%",)
    ).fetchone()

    if not opp_team:
        return 'R'

    pitcher = conn.execute('''
        SELECT p.throws FROM players p
        JOIN pitcher_game_logs pgl ON p.player_id = pgl.player_id
        WHERE p.team_id = ? AND p.position = 'P'
        ORDER BY pgl.date DESC LIMIT 1
    ''', (opp_team['team_id'],)).fetchone()

    return pitcher['throws'] if pitcher and pitcher['throws'] else 'R'


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
