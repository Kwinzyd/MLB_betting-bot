import psycopg2

try:
    conn = psycopg2.connect('postgresql://postgres:Johndoe1$@localhost:5432/nba_props')
    cur = conn.cursor()
    cur.execute("SELECT table_name FROM information_schema.columns WHERE column_name = 'game_id'")
    tables = [r[0] for r in cur.fetchall()]
    
    total = 0
    for t in tables:
        cur.execute(f"SELECT count(*) FROM {t} WHERE game_id IS NULL")
        count = cur.fetchone()[0]
        if count > 0:
            print(f"{t}: {count} NULLs")
            total += count

    print(f"Total: {total}")
except Exception as e:
    print(f"Error: {e}")
