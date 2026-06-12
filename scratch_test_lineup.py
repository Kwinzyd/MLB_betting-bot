import asyncio
from src.clients.mlb_stats import MLBStatsClient
from src.data.db import get_db_connection
from src.utils.time_utils import get_eastern_local_date

async def test():
    client = MLBStatsClient()
    today = str(get_eastern_local_date())

    # Get a bdl_game_id from DB
    with get_db_connection() as conn:
        game = conn.execute(
            "SELECT bdl_game_id FROM games WHERE status != 'COMPLETED' AND date LIKE ?",
            (f"{today}%",)
        ).fetchone()
        
    if game:
        bdl_game_id = game['bdl_game_id']
        lineups = await client.get_lineups(bdl_game_id)
        if lineups:
            print("First lineup entry:")
            print(lineups[0])
            print("\nAnother entry (maybe batter):")
            for entry in lineups:
                if not entry.get('is_probable_pitcher'):
                    print(entry)
                    break
    
    await client.session.close()

asyncio.run(test())
