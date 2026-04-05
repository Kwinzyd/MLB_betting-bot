from src.clients.injuries import InjuryClient
from src.data.db import get_db_connection
from src.utils.time_utils import get_eastern_local_date
from src.utils.logging_utils import get_logger

logger = get_logger(__name__)


def sync_injuries():
    """Scrape current MLB injuries and upsert into database."""
    logger.info("Executing pipeline: sync_injuries")
    client = InjuryClient()
    injuries = client.get_injuries()
    today = str(get_eastern_local_date())

    saved_count = 0
    with get_db_connection() as conn:
        for inj in injuries:
            conn.execute('''
                INSERT INTO injury_reports (date, player_name, team, status, injury_type)
                VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(date, player_name) DO UPDATE SET
                    status=excluded.status,
                    injury_type=excluded.injury_type
            ''', (today, inj['player_name'], inj['team'], inj['status'], inj['injury_type']))
            saved_count += 1
        conn.commit()

    logger.info(f"Successfully synced {saved_count} MLB injury reports for {today}.")
