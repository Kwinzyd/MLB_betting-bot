import sqlite3
import pandas as pd

try:
    conn = sqlite3.connect('data.db')
    df = pd.read_sql_query("SELECT player_name, market, side, ev, edge_pct, created_at FROM bet_candidates ORDER BY created_at DESC LIMIT 5", conn)
    if df.empty:
        print("No bet candidates found.")
    else:
        print(df)
except Exception as e:
    print(e)
