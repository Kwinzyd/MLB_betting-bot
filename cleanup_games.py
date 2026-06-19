import sqlite3
from datetime import datetime, timedelta

try:
    conn = sqlite3.connect('data.db')
    cur = conn.cursor()
    
    # 1. Delete games with missing team names
    cur.execute("DELETE FROM games WHERE away_team = '' OR home_team = '' OR away_team IS NULL OR home_team IS NULL")
    deleted_missing = cur.rowcount
    
    # 2. Mark old games as completed
    cutoff = datetime.utcnow() - timedelta(days=2)
    cutoff_str = cutoff.isoformat() + "Z"
    
    cur.execute("UPDATE games SET status = 'COMPLETED' WHERE status != 'COMPLETED' AND date < ?", (cutoff_str,))
    marked_completed = cur.rowcount
    
    conn.commit()
    print(f"Deleted {deleted_missing} games with missing team names.")
    print(f"Marked {marked_completed} old games as COMPLETED.")

except Exception as e:
    print(f"Error: {e}")
