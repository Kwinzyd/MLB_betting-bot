from src.clients.odds_api import OddsAPIClient
from src.clients.mlb_stats import MLBStatsClient
from src.data.db import get_db_connection
from src.utils.logging_utils import get_logger

logger = get_logger(__name__)


def sync_events():
    """Fetch upcoming MLB games from Odds API and cross-reference with BDL for game IDs."""
    logger.info("Executing pipeline: sync_events")
    odds_client = OddsAPIClient()
    bdl_client = MLBStatsClient()

    # 1. Fetch events from Odds API
    events = odds_client.get_mlb_events()
    if not events:
        logger.warning("No MLB events returned from Odds API.")
        return

    # 2. Fetch today's games from BDL for cross-referencing
    from src.utils.time_utils import get_eastern_local_date
    today = str(get_eastern_local_date())
    bdl_games = bdl_client.get_games(dates=today)

    # Build a lookup of BDL games by team names for matching
    bdl_lookup = {}
    for g in bdl_games:
        home = g.get('home_team', {})
        away = g.get('visitor_team', {}) or g.get('away_team', {})
        home_name = home.get('full_name', '') if isinstance(home, dict) else ''
        away_name = away.get('full_name', '') if isinstance(away, dict) else ''
        key = f"{home_name}|{away_name}".lower()
        bdl_lookup[key] = g

    saved_count = 0
    with get_db_connection() as conn:
        for event in events:
            game_id = event['id']
            home_team = event.get('home_team', '')
            away_team = event.get('away_team', '')
            commence_time = event.get('commence_time', '')

            # Try to find matching BDL game
            bdl_game_id = None
            bdl_key = f"{home_team}|{away_team}".lower()
            for bk, bg in bdl_lookup.items():
                if home_team.lower() in bk and away_team.lower() in bk:
                    bdl_game_id = bg.get('id')
                    break

            conn.execute('''
                INSERT INTO games (game_id, bdl_game_id, date, home_team, away_team, status)
                VALUES (?, ?, ?, ?, ?, ?)
                ON CONFLICT(game_id) DO UPDATE SET
                    bdl_game_id=COALESCE(excluded.bdl_game_id, games.bdl_game_id),
                    date=excluded.date,
                    status=excluded.status
            ''', (game_id, bdl_game_id, commence_time, home_team, away_team, 'SCHEDULED'))
            saved_count += 1
        conn.commit()

    logger.info(f"Successfully synced {saved_count} MLB events.")
