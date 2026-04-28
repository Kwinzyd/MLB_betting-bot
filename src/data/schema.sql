CREATE TABLE IF NOT EXISTS games (
    game_id TEXT PRIMARY KEY,
    bdl_game_id INTEGER,
    date TEXT,
    game_time TEXT,
    home_team TEXT,
    away_team TEXT,
    home_team_id INTEGER,
    away_team_id INTEGER,
    venue TEXT,
    status TEXT DEFAULT 'SCHEDULED',
    home_score INTEGER,
    away_score INTEGER,
    historical INTEGER DEFAULT 0,
    lineups_confirmed_at TEXT,
    last_scanned_at TEXT
);

CREATE TABLE IF NOT EXISTS teams (
    team_id INTEGER PRIMARY KEY,
    abbreviation TEXT,
    name TEXT,
    league TEXT,
    division TEXT
);

CREATE TABLE IF NOT EXISTS players (
    player_id INTEGER PRIMARY KEY,
    name TEXT,
    team_id INTEGER,
    position TEXT,
    bats TEXT,
    throws TEXT,
    active BOOLEAN DEFAULT 1
);

CREATE TABLE IF NOT EXISTS injury_reports (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    date TEXT,
    player_name TEXT,
    team TEXT,
    status TEXT,
    injury_type TEXT,
    UNIQUE(date, player_name)
);

CREATE TABLE IF NOT EXISTS pitcher_game_logs (
    game_id INTEGER,
    player_id INTEGER,
    date TEXT,
    innings_pitched REAL,
    hits_allowed INTEGER,
    runs_allowed INTEGER,
    earned_runs INTEGER,
    walks INTEGER,
    strikeouts INTEGER,
    home_runs_allowed INTEGER,
    pitches_thrown INTEGER,
    PRIMARY KEY (game_id, player_id)
);

CREATE TABLE IF NOT EXISTS batter_game_logs (
    game_id INTEGER,
    player_id INTEGER,
    date TEXT,
    at_bats INTEGER,
    hits INTEGER,
    doubles INTEGER,
    triples INTEGER,
    home_runs INTEGER,
    runs INTEGER,
    rbis INTEGER,
    walks INTEGER,
    strikeouts INTEGER,
    total_bases INTEGER,
    plate_appearances INTEGER,
    PRIMARY KEY (game_id, player_id)
);

CREATE TABLE IF NOT EXISTS daily_lineups (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    game_id TEXT,
    team TEXT,
    player_name TEXT,
    player_id INTEGER,
    lineup_position INTEGER,
    date TEXT,
    UNIQUE(game_id, player_name)
);

CREATE TABLE IF NOT EXISTS probable_pitchers (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    game_id TEXT,
    team TEXT,
    player_name TEXT,
    player_id INTEGER,
    throws TEXT,
    date TEXT,
    UNIQUE(game_id, team)
);

CREATE TABLE IF NOT EXISTS prop_snapshots (
    snapshot_id TEXT PRIMARY KEY,
    game_id TEXT,
    player_name TEXT,
    market TEXT,
    line REAL,
    over_odds REAL,
    under_odds REAL,
    bookmaker TEXT,
    timestamp TEXT,
    devigged_over REAL,
    devigged_under REAL
);

CREATE INDEX IF NOT EXISTS idx_prop_snapshots_timestamp ON prop_snapshots(timestamp);

CREATE TABLE IF NOT EXISTS projections (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    game_id TEXT,
    player_name TEXT,
    market TEXT,
    projected_mean REAL,
    prob_over REAL,
    prob_under REAL,
    context_json TEXT,
    timestamp TEXT,
    UNIQUE(game_id, player_name, market)
);

CREATE TABLE IF NOT EXISTS alerts_sent (
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
    -- Model's P(over) / P(under) at placement time. Stored so calibration
    -- metrics use the prob we actually bet on, not whatever later re-scans
    -- overwrite in the projections table.
    model_prob_over REAL,
    model_prob_under REAL,
    game_id TEXT,
    timestamp TEXT,
    UNIQUE(player_name, market, line, bookmaker)
);

CREATE TABLE IF NOT EXISTS orders (
    order_id INTEGER PRIMARY KEY AUTOINCREMENT,
    venue TEXT NOT NULL,
    alert_id INTEGER,
    player_name TEXT NOT NULL,
    market TEXT NOT NULL,
    line REAL NOT NULL,
    side TEXT NOT NULL,
    game_id TEXT,
    bookmaker TEXT,
    offered_odds REAL NOT NULL,
    fill_odds REAL,
    stake REAL NOT NULL,
    status TEXT NOT NULL DEFAULT 'pending',
    venue_order_id TEXT,
    placed_at TEXT NOT NULL,
    filled_at TEXT,
    notes TEXT,
    FOREIGN KEY(alert_id) REFERENCES alerts_sent(alert_id)
);
CREATE INDEX IF NOT EXISTS idx_orders_status ON orders(status);
CREATE INDEX IF NOT EXISTS idx_orders_venue ON orders(venue);

CREATE TABLE IF NOT EXISTS bet_results (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    alert_id INTEGER,
    actual_value REAL,
    result TEXT,
    profit REAL,
    closing_odds REAL,
    clv REAL,
    FOREIGN KEY(alert_id) REFERENCES alerts_sent(alert_id)
);

CREATE TABLE IF NOT EXISTS park_factors (
    venue TEXT PRIMARY KEY,
    runs_factor REAL DEFAULT 1.0,
    hr_factor REAL DEFAULT 1.0,
    hits_factor REAL DEFAULT 1.0,
    strikeouts_factor REAL DEFAULT 1.0
);

-- Per-umpire historical K and BB rates, computed from completed game box scores.
-- k_factor = k_per_game / LEAGUE_AVG_K_PER_GAME; used as a multiplier in projections.
CREATE TABLE IF NOT EXISTS umpire_stats (
    umpire_id   INTEGER PRIMARY KEY,
    umpire_name TEXT    NOT NULL,
    games_called      INTEGER NOT NULL DEFAULT 0,
    total_strikeouts  INTEGER NOT NULL DEFAULT 0,
    total_walks       INTEGER NOT NULL DEFAULT 0,
    k_per_game  REAL,          -- rolling average: total_strikeouts / games_called
    bb_per_game REAL,          -- rolling average: total_walks / games_called
    k_factor    REAL NOT NULL DEFAULT 1.0,  -- k_per_game / LEAGUE_AVG_K_PER_GAME
    updated_date TEXT
);

-- One row per MLB game (keyed by mlb_game_pk).
-- Historical rows (game_id IS NULL) serve as a processing log so box scores are
-- never fetched twice. Active rows (game_id IS NOT NULL) link today's games to
-- their umpire for live projection lookups.
CREATE TABLE IF NOT EXISTS umpire_game_assignments (
    mlb_game_pk INTEGER PRIMARY KEY,
    game_id     TEXT,   -- Odds API game_id; NULL for historical-only rows
    umpire_id   INTEGER NOT NULL,
    umpire_name TEXT    NOT NULL,
    date        TEXT    NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_uga_game_id ON umpire_game_assignments(game_id)
    WHERE game_id IS NOT NULL;
CREATE INDEX IF NOT EXISTS idx_uga_date   ON umpire_game_assignments(date);

CREATE TABLE IF NOT EXISTS sgp_candidates (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    game_id TEXT,
    legs_json TEXT,
    joint_prob REAL,
    naive_parlay_odds REAL,
    fair_odds REAL,
    edge_vs_naive REAL,
    kelly_stake REAL,
    bookmakers TEXT,
    timestamp TEXT,
    UNIQUE(game_id, legs_json)
);
CREATE INDEX IF NOT EXISTS idx_sgp_candidates_timestamp ON sgp_candidates(timestamp);

-- Settled SGP tickets. recalc_odds is the payout odds after dropping any
-- VOID/PUSH legs (their stake is refunded, surviving legs reprice as a smaller
-- parlay). voided_legs is a JSON list of leg indices that were voided so the
-- recalculation is auditable.
CREATE TABLE IF NOT EXISTS sgp_results (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    sgp_candidate_id INTEGER UNIQUE,
    leg_results_json TEXT,
    voided_legs_json TEXT,
    surviving_legs INTEGER,
    recalc_odds REAL,
    result TEXT,
    profit REAL,
    settled_at TEXT,
    FOREIGN KEY(sgp_candidate_id) REFERENCES sgp_candidates(id)
);

-- Per-scan game-total snapshots. Trigger watch reads the latest row as the
-- baseline against which it compares fresh totals to detect sharp-book moves
-- between scan cycles.
CREATE TABLE IF NOT EXISTS game_totals_history (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    game_id TEXT NOT NULL,
    total REAL NOT NULL,
    source TEXT NOT NULL,
    timestamp TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_game_totals_history_game_ts
    ON game_totals_history(game_id, timestamp DESC);

-- Fired weather / umpire edge triggers. Used for dedup so we don't re-pull
-- the same game every cron tick once a threshold is crossed.
CREATE TABLE IF NOT EXISTS trigger_events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    game_id TEXT NOT NULL,
    trigger_type TEXT NOT NULL,
    detail TEXT,
    triggered_at TEXT NOT NULL,
    UNIQUE(game_id, trigger_type, triggered_at)
);
CREATE INDEX IF NOT EXISTS idx_trigger_events_game
    ON trigger_events(game_id, trigger_type, triggered_at);

-- Per-entity overdispersion parameters (NB alpha for counts, residual sigma
-- for total_bases). Fitted post-hoc after the mean model trains.
CREATE TABLE IF NOT EXISTS dispersion_params (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    entity_id TEXT NOT NULL,
    entity_type TEXT NOT NULL,
    market TEXT NOT NULL,
    alpha REAL,
    sigma REAL,
    n_obs INTEGER NOT NULL,
    fitted_at TEXT NOT NULL,
    UNIQUE(entity_id, market)
);
CREATE INDEX IF NOT EXISTS idx_dispersion_lookup
    ON dispersion_params(entity_id, market);
