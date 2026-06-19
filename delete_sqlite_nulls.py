import sqlite3

try:
    conn = sqlite3.connect('data.db')
    cur = conn.cursor()
    
    cur.execute("DELETE FROM pitcher_game_logs WHERE game_id IS NULL")
    print(f"Deleted {cur.rowcount} from pitcher_game_logs")
    
    cur.execute("DELETE FROM batter_game_logs WHERE game_id IS NULL")
    print(f"Deleted {cur.rowcount} from batter_game_logs")
    
    conn.commit()
    print("Done")
except Exception as e:
    print(f"Error: {e}")
