import sqlite3
import pandas as pd

try:
    conn = sqlite3.connect('data.db')
    df = pd.read_sql_query("SELECT game_id, away_team, home_team, date, status FROM games WHERE status != 'COMPLETED'", conn)
    if df.empty:
        print("No active games in DB.")
    else:
        print(df)
except Exception as e:
    print(e)
