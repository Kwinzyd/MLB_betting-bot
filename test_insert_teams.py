import sqlite3
import asyncio
from src.clients.mlb_stats import MLBStatsClient

async def main():
    bdl_client = MLBStatsClient()
    all_teams = await bdl_client.get_teams()
    conn = sqlite3.connect('data.db')
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
            team.get('display_name', ''),
            team.get('league', ''),
            team.get('division', ''),
        ))
    conn.commit()
    
    cur = conn.cursor()
    cur.execute("SELECT * FROM teams LIMIT 5")
    for row in cur.fetchall():
        print(row)

if __name__ == "__main__":
    asyncio.run(main())
