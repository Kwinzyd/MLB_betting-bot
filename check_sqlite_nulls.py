import sqlite3

try:
    conn = sqlite3.connect('data.db')
    cur = conn.cursor()
    cur.execute("SELECT name FROM sqlite_master WHERE type='table'")
    tables = [r[0] for r in cur.fetchall()]
    
    total = 0
    for t in tables:
        try:
            cur.execute(f"PRAGMA table_info({t})")
            columns = [c[1] for c in cur.fetchall()]
            if 'game_id' in columns:
                cur.execute(f"SELECT count(*) FROM {t} WHERE game_id IS NULL")
                count = cur.fetchone()[0]
                if count > 0:
                    print(f"{t} (IS NULL): {count} NULLs")
                    total += count
                    
                cur.execute(f"SELECT count(*) FROM {t} WHERE game_id = 'NULL' OR game_id = 'None'")
                count_str = cur.fetchone()[0]
                if count_str > 0:
                    print(f"{t} ('NULL' or 'None'): {count_str}")
                    total += count_str
        except Exception as e:
            print(f"Skipping {t} because: {e}")

    print(f"Total: {total}")
except Exception as e:
    print(f"Error: {e}")
