import asyncio
from src.clients.mlb_stats import MLBStatsClient

async def main():
    client = MLBStatsClient()
    teams = await client.get_teams()
    if teams:
        print(teams[0])
    else:
        print("No teams.")

if __name__ == "__main__":
    asyncio.run(main())
