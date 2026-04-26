from src.clients.odds_api import OddsAPIClient
from src.clients.mlb_stats import MLBStatsClient
from src.config import ODDS_API_TEAM_ABBREV
from src.data.db import get_db_connection
from src.utils.logging_utils import get_logger

logger = get_logger(__name__)


async def sync_events():
    """
    Fetch upcoming MLB games from Odds API and cross-reference with BDL for game IDs.
    Uses ODDS_API_TEAM_ABBREV mapping for deterministic team matching (no fuzzy strings).
    """
    logger.info("Executing pipeline: sync_events")
    odds_client = OddsAPIClient()
    bdl_client = MLBStatsClient()

    # 1. Fetch events from Odds API
    events = await odds_client.get_mlb_events()
    if not events:
        logger.warning("No MLB events returned from Odds API.")
        return

    # 2. Build abbreviation → BDL team_id lookup from teams table
    abbrev_to_team_id = {}
    with get_db_connection() as conn:
        teams = conn.execute("SELECT team_id, abbreviation, name FROM teams").fetchall()
        for t in teams:
            if t['abbreviation']:
                abbrev_to_team_id[t['abbreviation'].upper()] = {
                    'team_id': t['team_id'],
                    'name': t['name'],
                }

    # If teams table is empty, sync teams first
    if not abbrev_to_team_id:
        logger.info("Teams table empty, fetching from BDL...")
        all_teams = await bdl_client.get_teams()
        with get_db_connection() as conn:
            for team in all_teams:
                conn.execute('''
                    INSERT INTO teams (team_id, abbreviation, name, league, division)
                    VALUES (?, ?, ?, ?, ?)
                    ON CONFLICT(team_id) DO UPDATE SET
                        abbreviation=excluded.abbreviation,
                        name=excluded.name
                ''', (
                    team.get('id'),
                    team.get('abbreviation', ''),
                    team.get('full_name', ''),
                    team.get('league', ''),
                    team.get('division', ''),
                ))
            conn.commit()
            teams = conn.execute("SELECT team_id, abbreviation, name FROM teams").fetchall()
            for t in teams:
                if t['abbreviation']:
                    abbrev_to_team_id[t['abbreviation'].upper()] = {
                        'team_id': t['team_id'],
                        'name': t['name'],
                    }
        logger.info(f"Synced {len(all_teams)} teams from BDL.")

    # 3. Fetch today's games from BDL for cross-referencing by team_id
    from src.utils.time_utils import get_eastern_local_date
    today = str(get_eastern_local_date())
    bdl_games = await bdl_client.get_games(dates=today)

    # Build BDL game lookup keyed by (home_team_id, away_team_id) for exact matching
    bdl_by_teams = {}
    for g in bdl_games:
        home = g.get('home_team', {})
        away = g.get('visitor_team', {}) or g.get('away_team', {})
        h_id = home.get('id') if isinstance(home, dict) else None
        a_id = away.get('id') if isinstance(away, dict) else None
        if h_id and a_id:
            bdl_by_teams[(h_id, a_id)] = g

    # 4. Process each Odds API event
    saved_count = 0
    unmatched = []

    with get_db_connection() as conn:
        for event in events:
            game_id = event['id']
            odds_home = event.get('home_team', '')
            odds_away = event.get('away_team', '')
            commence_time = event.get('commence_time', '')

            # Resolve team abbreviations via hardcoded mapping
            home_abbrev = ODDS_API_TEAM_ABBREV.get(odds_home)
            away_abbrev = ODDS_API_TEAM_ABBREV.get(odds_away)

            if not home_abbrev:
                logger.warning(f"Unknown Odds API team name: '{odds_home}' — add to ODDS_API_TEAM_ABBREV")
                unmatched.append(odds_home)
            if not away_abbrev:
                logger.warning(f"Unknown Odds API team name: '{odds_away}' — add to ODDS_API_TEAM_ABBREV")
                unmatched.append(odds_away)

            # Look up BDL team IDs and canonical names
            home_info = abbrev_to_team_id.get(home_abbrev) if home_abbrev else None
            away_info = abbrev_to_team_id.get(away_abbrev) if away_abbrev else None

            home_team_id = home_info['team_id'] if home_info else None
            away_team_id = away_info['team_id'] if away_info else None
            # Store canonical BDL team name for consistent downstream usage
            home_team = home_info['name'] if home_info else odds_home
            away_team = away_info['name'] if away_info else odds_away

            # Match to BDL game using team IDs (guaranteed correct)
            bdl_game_id = None
            if home_team_id and away_team_id:
                bdl_game = bdl_by_teams.get((home_team_id, away_team_id))
                if bdl_game:
                    bdl_game_id = bdl_game.get('id')

                    # Also grab venue from BDL if available
                    venue = bdl_game.get('venue', '')
                    if venue:
                        conn.execute('''
                            UPDATE games SET venue = ? WHERE game_id = ? AND (venue IS NULL OR venue = '')
                        ''', (venue, game_id))

            conn.execute('''
                INSERT INTO games (game_id, bdl_game_id, date, game_time, home_team, away_team,
                                   home_team_id, away_team_id, status)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(game_id) DO UPDATE SET
                    bdl_game_id=COALESCE(excluded.bdl_game_id, games.bdl_game_id),
                    date=excluded.date,
                    game_time=excluded.game_time,
                    home_team=excluded.home_team,
                    away_team=excluded.away_team,
                    home_team_id=COALESCE(excluded.home_team_id, games.home_team_id),
                    away_team_id=COALESCE(excluded.away_team_id, games.away_team_id),
                    status=excluded.status
            ''', (game_id, bdl_game_id, commence_time, commence_time, home_team, away_team,
                  home_team_id, away_team_id, 'SCHEDULED'))
            saved_count += 1
        conn.commit()

    if unmatched:
        logger.error(f"Unmatched Odds API team names: {set(unmatched)}. Update ODDS_API_TEAM_ABBREV in config.py.")

    logger.info(f"Synced {saved_count} MLB events ({len(bdl_by_teams)} matched to BDL).")

    # 5. Prune stale data older than 3 days to prevent SQLite bloat
    _cleanup_stale_data()


def _cleanup_stale_data(retention_days: int = 3):
    """
    Delete prop_snapshots, projections, daily_lineups, probable_pitchers,
    and injury_reports older than retention_days to keep SQLite fast.
    """
    from datetime import datetime, timedelta
    cutoff = (datetime.utcnow() - timedelta(days=retention_days)).isoformat()

    with get_db_connection() as conn:
        tables = {
            "prop_snapshots": "timestamp",
            "projections": "timestamp",
            "daily_lineups": "date",
            "probable_pitchers": "date",
            "injury_reports": "date",
        }
        total = 0
        for table, col in tables.items():
            cursor = conn.execute(f"DELETE FROM {table} WHERE {col} < ?", (cutoff,))
            total += cursor.rowcount

        conn.commit()

    if total > 0:
        logger.info(f"Cleanup: deleted {total} rows older than {retention_days} days.")
