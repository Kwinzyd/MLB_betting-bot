import asyncio
from src.clients.odds_api import OddsAPIClient

async def main():
    client = OddsAPIClient()
    events = await client.get_mlb_events()
    print(f"Got {len(events)} events.")
    if events:
        print("First event:")
        print(events[0])

if __name__ == "__main__":
    asyncio.run(main())
