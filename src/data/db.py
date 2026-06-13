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
        _migrate_sgp_candidates_columns(conn)
        _migrate_live_state_columns(conn)
        _migrate_live_batter_stats(conn)
        _migrate_alerts_sent_unique(conn)
        _migrate_alerts_delivery_cols(conn)
        _migrate_bet_candidates(conn)
        _migrate_missing_indexes(conn)
        _migrate_players_mlb_id(conn)
        _migrate_games_last_synced_at(conn)
        _migrate_team_stats(conn)
        _migrate_walk_forward_results(conn)
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
    if 'bdl_game_id' not in existing:
        # BDL integer game ID used to cross-reference live box scores with
        # the Odds API string game IDs stored in game_id.
        conn.execute("ALTER TABLE games ADD COLUMN bdl_game_id INTEGER")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_games_bdl_game_id ON games(bdl_game_id)")


def _migrate_alerts_columns(conn):
    """Backfill placement-time probability columns on existing alerts_sent rows."""
    existing = {row['name'] for row in conn.execute("PRAGMA table_info(alerts_sent)").fetchall()}
    if 'model_prob_over' not in existing:
        conn.execute("ALTER TABLE alerts_sent ADD COLUMN model_prob_over REAL")
    if 'model_prob_under' not in existing:
        conn.execute("ALTER TABLE alerts_sent ADD COLUMN model_prob_under REAL")


def _migrate_sgp_candidates_columns(conn):
    """Backfill stake column on existing sgp_candidates rows."""
    existing = {row['name'] for row in conn.execute("PRAGMA table_info(sgp_candidates)").fetchall()}
    if 'kelly_stake' not in existing:
        conn.execute("ALTER TABLE sgp_candidates ADD COLUMN kelly_stake REAL")


def _migrate_live_state_columns(conn):
    """Add live game-state columns written by BDL live sync.

    inning            — current inning (1-based)
    outs              — outs in the current half-inning (0-2)
    current_batter_slot — lineup slot of the batter currently at the plate (1-9)
    """
    existing = {row['name'] for row in conn.execute("PRAGMA table_info(games)").fetchall()}
    if 'inning' not in existing:
        conn.execute("ALTER TABLE games ADD COLUMN inning INTEGER DEFAULT 1")
    if 'outs' not in existing:
        conn.execute("ALTER TABLE games ADD COLUMN outs INTEGER DEFAULT 0")
    if 'current_batter_slot' not in existing:
        conn.execute("ALTER TABLE games ADD COLUMN current_batter_slot INTEGER DEFAULT 1")


def _migrate_live_batter_stats(conn):
    """Create live_batter_stats table if it doesn't exist.

    Stores running totals for batters in currently in-progress games.
    Written by BDLLiveClient.sync_game_states() and read by the Live
    State Machine to compute 'already accumulated' stat counts so the
    rest-of-game projection can subtract them from the full-game line.

    (game_id, bdl_player_id) is the composite PK — one row per player
    per game, upserted on each live sync tick.
    """
    conn.execute("""
        CREATE TABLE IF NOT EXISTS live_batter_stats (
            game_id          TEXT NOT NULL,
            bdl_player_id    INTEGER NOT NULL,
            hits             INTEGER DEFAULT 0,
            total_bases      INTEGER DEFAULT 0,
            home_runs        INTEGER DEFAULT 0,
            plate_appearances INTEGER DEFAULT 0,
            updated_at       TEXT,
            PRIMARY KEY (game_id, bdl_player_id)
        )
    """)

def _migrate_alerts_sent_unique(conn):
    """Broaden the alerts_sent UNIQUE constraint to include game_id.

    The original constraint (player_name, market, line, bookmaker) incorrectly
    suppressed alerts for the same player/line at a different game. SQLite
    requires a full table rebuild to change a UNIQUE constraint.
    """
    # Check current constraint by inspecting the CREATE TABLE statement
    row = conn.execute(
        "SELECT sql FROM sqlite_master WHERE type='table' AND name='alerts_sent'"
    ).fetchone()
    if not row:
        return  # table doesn't exist yet; schema.sql will create it correctly
    if 'game_id' in (row['sql'] or '').split('UNIQUE')[- 1]:
        return  # already has game_id in the constraint

    logger.info("Migrating alerts_sent UNIQUE constraint to include game_id …")
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS alerts_sent_new (
            alert_id INTEGER PRIMARY KEY AUTOINCREMENT,
            player_name TEXT,
            market TEXT,
            line REAL,
            side TEXT,
            edge REAL,
            ev REAL,
            kelly_stake REAL,
            bookmaker TEXT,
            odds REAL,
            opening_odds REAL,
            model_prob_over REAL,
            model_prob_under REAL,
            game_id TEXT,
            timestamp TEXT,
            UNIQUE(player_name, market, line, bookmaker, game_id)
        );
        INSERT OR IGNORE INTO alerts_sent_new
            SELECT alert_id, player_name, market, line, side, edge, ev, kelly_stake,
                   bookmaker, odds, opening_odds, model_prob_over, model_prob_under,
                   game_id, timestamp
            FROM alerts_sent;
        DROP TABLE alerts_sent;
        ALTER TABLE alerts_sent_new RENAME TO alerts_sent;
    """)
    logger.info("alerts_sent constraint migration complete.")


def _migrate_alerts_delivery_cols(conn):
    """Add shadow-mode / identity / CLV columns to alerts_sent.

    delivered       — 1 = alert reached Telegram with betting enabled;
                      0 = shadow (paper) record. Legacy rows default to 1
                      because the old code only inserted on delivery.
    player_id       — BDL id captured at scan time; settlement grades by id.
    open_devig_prob — devigged prob of our side at placement, so CLV compares
                      vig-free open vs vig-free close.
    """
    existing = {row['name'] for row in conn.execute("PRAGMA table_info(alerts_sent)").fetchall()}
    if 'delivered' not in existing:
        conn.execute("ALTER TABLE alerts_sent ADD COLUMN delivered INTEGER DEFAULT 1")
    if 'player_id' not in existing:
        conn.execute("ALTER TABLE alerts_sent ADD COLUMN player_id INTEGER")
    if 'open_devig_prob' not in existing:
        conn.execute("ALTER TABLE alerts_sent ADD COLUMN open_devig_prob REAL")


def _migrate_bet_candidates(conn):
    """Create bet_candidates on DBs that predate the scan->alert handoff table."""
    conn.execute("""
        CREATE TABLE IF NOT EXISTS bet_candidates (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            game_id TEXT NOT NULL,
            player_id INTEGER,
            player_name TEXT NOT NULL,
            market TEXT NOT NULL,
            line REAL NOT NULL,
            side TEXT NOT NULL,
            bookmaker TEXT NOT NULL,
            odds REAL NOT NULL,
            sharp_book TEXT,
            anchor_line REAL,
            truth_prob REAL,
            model_prob REAL,
            open_devig_prob REAL,
            edge_pct REAL,
            ev REAL,
            kelly_fraction REAL,
            recommended_stake REAL,
            steam_detected INTEGER DEFAULT 0,
            created_at TEXT NOT NULL,
            UNIQUE(game_id, player_name, market, line, side, bookmaker)
        )
    """)
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_bet_candidates_created ON bet_candidates(created_at)"
    )


def _migrate_missing_indexes(conn):
    """Add indexes that postdate the original schema."""
    conn.execute("CREATE INDEX IF NOT EXISTS idx_alerts_sent_timestamp ON alerts_sent(timestamp)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_games_date ON games(date)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_games_status ON games(status)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_orders_placed_at ON orders(placed_at)")
    # Game-log lookups by (player_id, date) drive every rolling-window query in
    # projections and the training feature builder (h2h, bullpen, recency).
    # Without these the queries full-SCAN the log tables once per training row.
    conn.execute("CREATE INDEX IF NOT EXISTS idx_pgl_player_date ON pitcher_game_logs(player_id, date)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_bgl_player_date ON batter_game_logs(player_id, date)")
    # Bullpen factor joins players by team_id.
    conn.execute("CREATE INDEX IF NOT EXISTS idx_players_team ON players(team_id)")
    # prop_snapshots hot path: steam/CLV/opening-line lookups filter on
    # (game_id, player_name, market, line, bookmaker).
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_prop_snapshots_lookup "
        "ON prop_snapshots(game_id, player_name, market, line, bookmaker)"
    )


def _migrate_players_mlb_id(conn):
    """Add mlb_id column to players table for Baseball Savant cross-reference."""
    existing = {row['name'] for row in conn.execute("PRAGMA table_info(players)").fetchall()}
    if 'mlb_id' not in existing:
        conn.execute("ALTER TABLE players ADD COLUMN mlb_id INTEGER")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_players_mlb_id ON players(mlb_id)")


def _migrate_team_stats(conn):
    """Create team_stats table on existing DBs that predate schema.sql addition."""
    conn.execute("""
        CREATE TABLE IF NOT EXISTS team_stats (
            team_id       INTEGER PRIMARY KEY,
            k_rate        REAL,
            runs_per_game REAL,
            last_updated  TEXT,
            UNIQUE(team_id)
        )
    """)


def _migrate_walk_forward_results(conn):
    """Create walk_forward_results table on existing DBs."""
    conn.execute("""
        CREATE TABLE IF NOT EXISTS walk_forward_results (
            id                        INTEGER PRIMARY KEY AUTOINCREMENT,
            market                    TEXT    NOT NULL,
            model_type                TEXT    NOT NULL,
            mode                      TEXT    NOT NULL,
            train_window_days         INTEGER NOT NULL,
            step_days                 INTEGER NOT NULL,
            n_folds                   INTEGER,
            weighted_mae              REAL,
            weighted_poisson_deviance REAL,
            weighted_var_ratio        REAL,
            sharpe_ratio              REAL,
            sortino_ratio             REAL,
            max_drawdown              REAL,
            run_at                    TEXT    NOT NULL
        )
    """)
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_wf_results_market "
        "ON walk_forward_results(market, run_at)"
    )


def _migrate_games_last_synced_at(conn):
    """Add last_synced_at to games for incremental stats sync gating."""
    existing = {row['name'] for row in conn.execute("PRAGMA table_info(games)").fetchall()}
    if 'last_synced_at' not in existing:
        conn.execute("ALTER TABLE games ADD COLUMN last_synced_at TEXT")


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
    conn.execute("PRAGMA foreign_keys=ON")
    try:
        yield conn
    finally:
        conn.close()
