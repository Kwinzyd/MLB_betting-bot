import sqlite3

try:
    conn = sqlite3.connect('data.db')
    cur = conn.cursor()
    cur.execute("SELECT * FROM teams LIMIT 5")
    for row in cur.fetchall():
        print(row)
except Exception as e:
    print(e)
