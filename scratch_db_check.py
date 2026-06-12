import sqlite3
from src.utils.time_utils import get_eastern_local_date

def check():
    today = str(get_eastern_local_date())
    conn = sqlite3.connect('data.db')
    conn.row_factory = sqlite3.Row
    games = conn.execute("SELECT game_id, away_team, home_team, game_time, lineups_confirmed_at, last_scanned_at FROM games WHERE status != 'COMPLETED' AND date LIKE ?", (f"{today}%",)).fetchall()
    
    print(f"Total games for today: {len(games)}")
    for g in games:
        print(f"{g['away_team']} @ {g['home_team']} - Time: {g['game_time']} - Confirmed: {g['lineups_confirmed_at']} - Scanned: {g['last_scanned_at']}")
    
    # Let's check games from other days just in case
    other_games = conn.execute("SELECT game_id, date, away_team, home_team, game_time, lineups_confirmed_at, last_scanned_at FROM games WHERE status != 'COMPLETED' AND date NOT LIKE ?", (f"{today}%",)).fetchall()
    print(f"\nTotal games not today: {len(other_games)}")
    for g in other_games:
        print(f"[{g['date']}] {g['away_team']} @ {g['home_team']} - Time: {g['game_time']} - Confirmed: {g['lineups_confirmed_at']} - Scanned: {g['last_scanned_at']}")

check()
