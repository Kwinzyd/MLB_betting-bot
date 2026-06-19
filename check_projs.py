import sqlite3
import pandas as pd

try:
    conn = sqlite3.connect('data.db')
    df = pd.read_sql_query("SELECT player_name, market, projected_mean, prob_over, timestamp FROM projections ORDER BY timestamp DESC LIMIT 5", conn)
    if df.empty:
        print("No projections found.")
    else:
        print(df)
except Exception as e:
    print(e)
