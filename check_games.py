import sqlite3
import pandas as pd

try:
    conn = sqlite3.connect('data.db')
    df = pd.read_sql_query("SELECT game_id, away_team, home_team, status, date FROM games WHERE away_team = '' OR home_team = '' OR away_team IS NULL LIMIT 20", conn)
    print("Games with missing team names:")
    print(df)
except Exception as e:
    print(e)
