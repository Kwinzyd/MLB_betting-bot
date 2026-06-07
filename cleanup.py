"""One-shot cleanup: wipe today's poisoned lineup/pitcher rows and clear cache."""
from src.data.db import get_db_connection

DATE = "2026-04-28"

with get_db_connection() as conn:
    # 1. Identify cache table if it exists
    tables = [r[0] for r in conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table'"
    ).fetchall()]
    print("Tables:", tables)

    # 2. Wipe poisoned lineup rows
    pp = conn.execute(
        "DELETE FROM probable_pitchers WHERE date = ?", (DATE,)
    ).rowcount
    dl = conn.execute(
        "DELETE FROM daily_lineups WHERE date = ?", (DATE,)
    ).rowcount
    print(f"Deleted {pp} probable_pitchers rows and {dl} daily_lineups rows for {DATE}.")

    # 3. Clear cache rows related to lineups if a cache table exists
    if "cache" in tables:
        cc = conn.execute(
            "DELETE FROM cache WHERE key LIKE '%lineup%' OR key LIKE '%pitcher%'"
        ).rowcount
        print(f"Cleared {cc} cache rows.")
    else:
        print("No 'cache' table found in DB — checking for file-based cache...")

    conn.commit()

# 4. Check for file-based cache
import os, glob
cache_files = glob.glob("cache/**", recursive=True) + glob.glob(".cache/**", recursive=True)
if cache_files:
    print("File-based cache files found:", cache_files[:10])
else:
    print("No file-based cache directory found.")

print("Cleanup complete.")
