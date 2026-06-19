import sqlite3
import pandas as pd

try:
    conn = sqlite3.connect('data.db')
    df = pd.read_sql_query("SELECT away_team, home_team FROM games WHERE status != 'COMPLETED'", conn)
    for index, row in df.iterrows():
        print(f"Game: '{row['away_team']}' @ '{row['home_team']}'")
except Exception as e:
    print(e)
