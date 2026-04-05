CREATE TABLE IF NOT EXISTS games (
    game_id TEXT PRIMARY KEY,
    bdl_game_id INTEGER,
    date TEXT,
    home_team TEXT,
    away_team TEXT,
    home_team_id INTEGER,
    away_team_id INTEGER,
    venue TEXT,
    status TEXT DEFAULT 'SCHEDULED',
    home_score INTEGER,
    away_score INTEGER
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
    rbis INTEGER,
    walks INTEGER,
    strikeouts INTEGER,
    total_bases INTEGER,
    plate_appearances INTEGER,
    PRIMARY KEY (game_id, player_id)
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
    game_id TEXT,
    timestamp TEXT,
    UNIQUE(player_name, market, line, bookmaker)
);

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
