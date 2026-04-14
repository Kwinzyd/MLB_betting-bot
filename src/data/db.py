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
        conn.commit()
    logger.info("Database initialized successfully.")


def _migrate_games_columns(conn):
    """Add columns that postdate the original schema on existing DBs."""
    existing = {row['name'] for row in conn.execute("PRAGMA table_info(games)").fetchall()}
    if 'game_time' not in existing:
        conn.execute("ALTER TABLE games ADD COLUMN game_time TEXT")
    if 'historical' not in existing:
        conn.execute("ALTER TABLE games ADD COLUMN historical INTEGER DEFAULT 0")

@contextmanager
def get_db_connection():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
    finally:
        conn.close()
