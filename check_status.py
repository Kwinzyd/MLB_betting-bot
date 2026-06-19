import sqlite3

try:
    conn = sqlite3.connect('data.db')
    cur = conn.cursor()
    cur.execute("SELECT game_id, bdl_game_id, away_team, home_team, status FROM games WHERE away_team='San Diego Padres' AND home_team='Baltimore Orioles'")
    for row in cur.fetchall():
        print(row)
except Exception as e:
    print(e)
