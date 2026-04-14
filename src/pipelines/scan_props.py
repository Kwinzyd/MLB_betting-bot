import uuid
import json
from datetime import datetime
from src.config import MARKETS_MAPPING, UMP_MIN_GAMES, UMP_K_WEIGHT
from src.clients.odds_api import OddsAPIClient
from src.clients.weather import WeatherClient
from src.data.db import get_db_connection
from src.data.park_factors import get_stadium_meta
from src.models.devig import devig_multiplicative
from src.models.projections import ProjectionModel
from src.models.edge_ranker import rank_edge
from src.utils.logging_utils import get_logger

logger = get_logger(__name__)


def scan_props():
    """
    Core pipeline: fetch live odds, run projections, identify edges.
    Batches all 5 markets in a single API call per game to conserve quota.
    """
    logger.info("Executing pipeline: scan_props")
    odds_client = OddsAPIClient()
    weather_client = WeatherClient()
    proj_model = ProjectionModel()

    # 1. Get active games from DB
    with get_db_connection() as conn:
        games = conn.execute("SELECT * FROM games WHERE status != 'COMPLETED'").fetchall()

    if not games:
        logger.info("No active games found for odds scanning.")
        return

    # All markets in one call
    markets = list(MARKETS_MAPPING.keys())
    total_edges = 0

    for game in games:
        game_id = game['game_id']
        home_team = game['home_team']
        away_team = game['away_team']
        venue = game['venue']
        logger.info(f"Scanning: {away_team} @ {home_team}")

        # Fetch live weather for this stadium
        weather = None
        meta = get_stadium_meta(venue)
        if meta and meta.get("roof") != "dome":
            weather = weather_client.get_game_weather(meta["lat"], meta["lon"])

        # Umpire K factor — looked up once per game, applied to all pitcher props
        ump_k_factor = _get_ump_k_factor(game_id)

        try:
            event_odds = odds_client.get_event_odds(game_id, markets)
        except Exception as e:
            logger.error(f"Failed to fetch odds for {game_id}: {e}")
            continue

        if not event_odds:
            continue

        # 2. Parse odds and group by player+market+line for devigging
        player_lines = _parse_odds_by_player(event_odds)

        # 3. For each player+market+line, pick best line across books, devig, project, rank
        for key, line_data in player_lines.items():
            player_name, market_key, line = key

            best = _pick_best_line(line_data)
            over_odds, over_book = best['over']
            under_odds, under_book = best['under']

            if not over_odds or not under_odds:
                continue
            if over_odds <= 1.0 or under_odds <= 1.0:
                continue

            # Devig using best-of-books pair
            devigged_over, devigged_under = devig_multiplicative(over_odds, under_odds)

            projection = _build_projection(
                proj_model, player_name, market_key, line,
                game_id, home_team, away_team, venue,
                weather=weather, ump_k_factor=ump_k_factor,
            )

            if not projection:
                continue

            projection['player_name'] = player_name

            # Composite book label when over and under come from different books
            book_label = over_book if over_book == under_book else f"{over_book}/{under_book}"

            snapshot_id = str(uuid.uuid4())
            timestamp = datetime.utcnow().isoformat()

            with get_db_connection() as conn:
                conn.execute('''
                    INSERT INTO prop_snapshots
                    (snapshot_id, game_id, player_name, market, line,
                     over_odds, under_odds, bookmaker, timestamp,
                     devigged_over, devigged_under)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    snapshot_id, game_id, player_name, market_key,
                    line, over_odds, under_odds, book_label, timestamp,
                    devigged_over, devigged_under,
                ))

                playable_any = False
                for side, odds_val, dev_prob, side_book in [
                    ('over', over_odds, devigged_over, over_book),
                    ('under', under_odds, devigged_under, under_book),
                ]:
                    edge_result = rank_edge(projection, odds_val, side, dev_prob)

                    if edge_result['is_playable']:
                        total_edges += 1
                        playable_any = True
                        logger.info(
                            f"EDGE FOUND: {player_name} {market_key} {side.upper()} {line} "
                            f"@ {side_book} | Edge: {edge_result['edge_pct']:.1f}% | "
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

                conn.commit()

    logger.info(f"Scan complete. Found {total_edges} playable edges.")


def _pick_best_line(line_data: dict) -> dict:
    """
    Given {book: {'over': odds, 'under': odds}}, pick the max odds per side.
    Returns {'over': (odds|None, book|None), 'under': (odds|None, book|None)}.
    Over and under may come from different books — that's the point of shopping.
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


def _build_projection(proj_model: ProjectionModel, player_name: str,
                      market_key: str, line: float,
                      game_id: str, home_team: str, away_team: str,
                      venue: str, weather: dict = None,
                      ump_k_factor: float = 1.0) -> dict:
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
        position = player['position'] or ''
        bats = player['bats'] or ''
        throws = player['throws'] or ''

        # Check injury status
        from src.utils.time_utils import get_eastern_local_date
        today = str(get_eastern_local_date())
        injury = conn.execute(
            "SELECT status FROM injury_reports WHERE player_name = ? AND date = ?",
            (player_name, today)
        ).fetchone()
        injury_status = injury['status'] if injury else 'Healthy'

        if injury_status in ('IL', 'Out'):
            return None

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
