import asyncio
from src.clients.mlb_stats import MLBStatsClient
from src.data.db import get_db_connection
from src.utils.time_utils import get_eastern_local_date

async def test():
    client = MLBStatsClient()
    today = str(get_eastern_local_date())

    with get_db_connection() as conn:
        games = conn.execute(
            "SELECT bdl_game_id FROM games WHERE status != 'COMPLETED' AND date LIKE ? AND bdl_game_id IS NOT NULL AND bdl_game_id != ''",
            (f"{today}%",)
        ).fetchall()
        
    for game in games:
        bdl_game_id = game['bdl_game_id']
        print(f"Fetching lineups for {bdl_game_id}")
        lineups = await client.get_lineups(bdl_game_id)
        if lineups:
            print("Lineups received! Count:", len(lineups))
            print("First entry:", lineups[0])
            for entry in lineups:
                if not entry.get('is_probable_pitcher'):
                    print("\nFirst non-pitcher entry:", entry)
                    break
            break
        else:
            print("Empty lineups array.")
    
    await client.session.close()

asyncio.run(test())
