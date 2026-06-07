import sqlite3
db_path = r'c:\Users\louis\nba-props-in\nba-prop-bot\props.db'
conn = sqlite3.connect(db_path)
cur = conn.cursor()
cur.execute("SELECT name, sql FROM sqlite_master WHERE type='table';")
for row in cur.fetchall():
    print(f"Table: {row[0]}")
    print(f"Schema: {row[1]}\n")
conn.close()
