import asyncio
from src.clients.mlb_stats import MLBStatsClient

async def main():
    client = MLBStatsClient()
    games = await client.get_games(dates=['2026-06-14'])
    for g in games:
        if g['id'] == 5058832:
            print("Found game 5058832:")
            print(f"Status: {g.get('status')}")
            break
    else:
        print("Game 5058832 not found in today's games!")

if __name__ == "__main__":
    asyncio.run(main())
