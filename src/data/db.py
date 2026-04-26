import sqlite3
import os
from contextlib import contextmanager
from src.config import DB_PATH
from src.utils.logging_utils import get_logger

logger = get_logger(__name__)

def init_db():
    logger.info(f"Initializing database at {DB_PATH}")
    schema_path = os.path.join(os.path.dirname(__file__), 'schema.sql')
    if not os.path.exists(schema_path):
        logger.error(f"Schema file not found at {schema_path}")
        return

    with open(schema_path, 'r') as f:
        schema_sql = f.read()

    with get_db_connection() as conn:
        conn.executescript(schema_sql)
        _migrate_games_columns(conn)
        _migrate_alerts_columns(conn)
        conn.commit()
    logger.info("Database initialized successfully.")


def _migrate_games_columns(conn):
    """Add columns that postdate the original schema on existing DBs."""
    existing = {row['name'] for row in conn.execute("PRAGMA table_info(games)").fetchall()}
    if 'game_time' not in existing:
        conn.execute("ALTER TABLE games ADD COLUMN game_time TEXT")
    if 'historical' not in existing:
        conn.execute("ALTER TABLE games ADD COLUMN historical INTEGER DEFAULT 0")
    if 'lineups_confirmed_at' not in existing:
        conn.execute("ALTER TABLE games ADD COLUMN lineups_confirmed_at TEXT")
    if 'last_scanned_at' not in existing:
        conn.execute("ALTER TABLE games ADD COLUMN last_scanned_at TEXT")


def _migrate_alerts_columns(conn):
    """Backfill placement-time probability columns on existing alerts_sent rows."""
    existing = {row['name'] for row in conn.execute("PRAGMA table_info(alerts_sent)").fetchall()}
    if 'model_prob_over' not in existing:
        conn.execute("ALTER TABLE alerts_sent ADD COLUMN model_prob_over REAL")
    if 'model_prob_under' not in existing:
        conn.execute("ALTER TABLE alerts_sent ADD COLUMN model_prob_under REAL")

@contextmanager
def get_db_connection():
    conn = sqlite3.connect(DB_PATH, timeout=30.0)
    conn.row_factory = sqlite3.Row
    # WAL allows concurrent readers while a writer holds the DB — critical when
    # scan_props (live) overlaps with sync_stats / backfill / train_model (heavy
    # writes). journal_mode is persisted at the DB-file level, so this is a no-op
    # after the first run, but safe to re-issue.
    # synchronous=NORMAL is the recommended pairing with WAL: durable across
    # crashes, faster than FULL. busy_timeout waits up to 30s on lock contention
    # instead of raising OperationalError immediately.
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    conn.execute("PRAGMA busy_timeout=30000")
    try:
        yield conn
    finally:
        conn.close()
