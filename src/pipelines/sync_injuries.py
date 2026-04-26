from src.clients.injuries import InjuryClient
from src.data.db import get_db_connection
from src.utils.time_utils import get_eastern_local_date
from src.utils.logging_utils import get_logger

logger = get_logger(__name__)


async def sync_injuries():
    """
    Fetch current MLB injuries and upsert into database.
    Fully resilient: any failure here logs a warning and returns gracefully
    so the rest of the pipeline (scan, alerts) continues with stale injury data.
    """
    logger.info("Executing pipeline: sync_injuries")
    try:
        client = InjuryClient()
        injuries = await client.get_injuries()
        today = str(get_eastern_local_date())

        if not injuries:
            logger.warning("No injury data returned. Pipeline will use stale injury data.")
            return

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

        logger.info(f"Synced {saved_count} MLB injury reports for {today}.")

    except Exception as e:
        logger.error(f"sync_injuries failed: {e}. Continuing with stale injury data.", exc_info=True)
