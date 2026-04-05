import uuid
import json
from datetime import datetime
from src.config import MARKETS_MAPPING
from src.clients.odds_api import OddsAPIClient
from src.data.db import get_db_connection
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

        try:
            event_odds = odds_client.get_event_odds(game_id, markets)
        except Exception as e:
            logger.error(f"Failed to fetch odds for {game_id}: {e}")
            continue

        if not event_odds:
            continue

        # 2. Parse odds and group by player+market+line for devigging
        player_lines = _parse_odds_by_player(event_odds)

        # 3. For each player+market+line, devig, project, and rank
        for key, line_data in player_lines.items():
            player_name, market_key, line = key

            for book, odds_pair in line_data.items():
                over_odds = odds_pair.get('over')
                under_odds = odds_pair.get('under')

                if not over_odds or not under_odds:
                    continue
                if over_odds <= 1.0 or under_odds <= 1.0:
                    continue

                # Devig
                devigged_over, devigged_under = devig_multiplicative(over_odds, under_odds)

                # Look up player stats and build projection
                projection = _build_projection(
                    proj_model, player_name, market_key, line,
                    home_team, away_team, venue
                )

                if not projection:
                    continue

                projection['player_name'] = player_name

                # Rank edge for both over and under
                for side, odds_val, dev_prob in [
                    ('over', over_odds, devigged_over),
                    ('under', under_odds, devigged_under),
                ]:
                    edge_result = rank_edge(projection, odds_val, side, dev_prob)

                    # Save snapshot regardless
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
                            line, over_odds, under_odds, book, timestamp,
                            devigged_over, devigged_under,
                        ))

                        # Save projection if playable
                        if edge_result['is_playable']:
                            total_edges += 1
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

                            logger.info(
                                f"EDGE FOUND: {player_name} {market_key} {side.upper()} {line} "
                                f"@ {book} | Edge: {edge_result['edge_pct']:.1f}% | "
                                f"EV: {edge_result['ev']:.3f} | "
                                f"Kelly: ${edge_result['kelly']['recommended_stake']}"
                            )

                        conn.commit()

    logger.info(f"Scan complete. Found {total_edges} playable edges.")


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
                      home_team: str, away_team: str, venue: str) -> dict:
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

            # Get opponent team K rate (approximate from batter logs)
            opp_k_rate = _get_team_k_rate(conn, away_team)

            proj = proj_model.project_pitcher_strikeouts(logs, opp_k_rate, venue, line)
            if proj:
                proj['injury_status'] = injury_status
            return proj

        elif market_key == 'pitcher_earned_runs':
            logs = conn.execute(
                "SELECT * FROM pitcher_game_logs WHERE player_id = ? ORDER BY date DESC",
                (player_id,)
            ).fetchall()
            logs = [dict(l) for l in logs]

            opp_runs_pg = _get_team_runs_per_game(conn, away_team)

            proj = proj_model.project_pitcher_earned_runs(logs, opp_runs_pg, venue, line)
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

            # Get opposing pitcher's throwing hand (if available)
            pitcher_hand = _get_opposing_pitcher_hand(conn, home_team, away_team, player)

            proj = proj_model.project_batter_stat(
                logs, stat_type, pitcher_hand, bats, venue, line
            )
            if proj:
                proj['injury_status'] = injury_status
            return proj

    return None


def _get_team_k_rate(conn, team_name: str) -> float:
    """Estimate a team's strikeout rate from batter game logs."""
    from src.config import LEAGUE_AVG_K_RATE

    # Find team ID
    team = conn.execute(
        "SELECT team_id FROM teams WHERE name LIKE ? COLLATE NOCASE",
        (f"%{team_name}%",)
    ).fetchone()

    if not team:
        return LEAGUE_AVG_K_RATE

    # Get batters on this team and their K rates
    batters = conn.execute(
        "SELECT player_id FROM players WHERE team_id = ?", (team['team_id'],)
    ).fetchall()

    if not batters:
        return LEAGUE_AVG_K_RATE

    batter_ids = [b['player_id'] for b in batters]
    placeholders = ','.join('?' * len(batter_ids))

    totals = conn.execute(f'''
        SELECT SUM(strikeouts) as total_k, SUM(plate_appearances) as total_pa
        FROM batter_game_logs WHERE player_id IN ({placeholders})
    ''', batter_ids).fetchone()

    if totals and totals['total_pa'] and totals['total_pa'] > 0:
        return totals['total_k'] / totals['total_pa']

    return LEAGUE_AVG_K_RATE


def _get_team_runs_per_game(conn, team_name: str) -> float:
    """Estimate a team's runs per game."""
    from src.config import LEAGUE_AVG_RUNS_PER_GAME

    team = conn.execute(
        "SELECT team_id FROM teams WHERE name LIKE ? COLLATE NOCASE",
        (f"%{team_name}%",)
    ).fetchone()

    if not team:
        return LEAGUE_AVG_RUNS_PER_GAME

    batters = conn.execute(
        "SELECT player_id FROM players WHERE team_id = ?", (team['team_id'],)
    ).fetchall()

    if not batters:
        return LEAGUE_AVG_RUNS_PER_GAME

    batter_ids = [b['player_id'] for b in batters]
    placeholders = ','.join('?' * len(batter_ids))

    # Sum RBIs as a proxy for team run production per game
    totals = conn.execute(f'''
        SELECT SUM(rbis) as total_rbis, COUNT(DISTINCT game_id) as games
        FROM batter_game_logs WHERE player_id IN ({placeholders})
    ''', batter_ids).fetchone()

    if totals and totals['games'] and totals['games'] > 0:
        return totals['total_rbis'] / totals['games']

    return LEAGUE_AVG_RUNS_PER_GAME


def _get_opposing_pitcher_hand(conn, home_team: str, away_team: str, batter_player) -> str:
    """Try to find the opposing starting pitcher's throwing hand."""
    # If the batter is on the home team, the opposing pitcher is on the away team, and vice versa
    batter_team_id = batter_player['team_id']

    home = conn.execute(
        "SELECT team_id FROM teams WHERE name LIKE ? COLLATE NOCASE",
        (f"%{home_team}%",)
    ).fetchone()

    if home and batter_team_id == home['team_id']:
        opp_team_name = away_team
    else:
        opp_team_name = home_team

    opp_team = conn.execute(
        "SELECT team_id FROM teams WHERE name LIKE ? COLLATE NOCASE",
        (f"%{opp_team_name}%",)
    ).fetchone()

    if not opp_team:
        return 'R'  # default to right-handed

    # Find a pitcher on the opposing team (best guess: most recent starter)
    pitcher = conn.execute('''
        SELECT p.throws FROM players p
        JOIN pitcher_game_logs pgl ON p.player_id = pgl.player_id
        WHERE p.team_id = ? AND p.position = 'P'
        ORDER BY pgl.date DESC LIMIT 1
    ''', (opp_team['team_id'],)).fetchone()

    return pitcher['throws'] if pitcher and pitcher['throws'] else 'R'
