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
    last_scanned_at TEXT,
    -- Live game state (written by BDLLiveClient.sync_game_states each poll tick)
    inning INTEGER DEFAULT 1,
    outs   INTEGER DEFAULT 0,
    current_batter_slot INTEGER DEFAULT 1
);

CREATE INDEX IF NOT EXISTS idx_games_bdl_game_id ON games(bdl_game_id);
CREATE INDEX IF NOT EXISTS idx_games_date ON games(date);
CREATE INDEX IF NOT EXISTS idx_games_status ON games(status);

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
    -- delivered=1 means the alert actually reached Telegram with betting
    -- enabled; 0 = shadow-mode (paper) record. Both settle, so the paper
    -- window produces real CLV/calibration data.
    delivered INTEGER DEFAULT 1,
    -- BDL player id captured at scan time; settlement grades by id, not name.
    player_id INTEGER,
    -- Devigged probability of our side at placement (same basis as the
    -- devigged close) so CLV is computed vig-free on both ends.
    open_devig_prob REAL,
    UNIQUE(player_name, market, line, bookmaker, game_id)
);
CREATE INDEX IF NOT EXISTS idx_alerts_sent_timestamp ON alerts_sent(timestamp);

-- Winning candidates persisted by scan_props. send_alerts consumes only
-- fresh rows from here — it never re-derives bets from raw snapshots, so the
-- sharp-anchored edge decision made at scan time is exactly what gets sized,
-- alerted, and settled.
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
    truth_prob REAL,        -- edge basis: sharp devig at anchor, model at alt-lines
    model_prob REAL,        -- model probability for this side at this line
    open_devig_prob REAL,   -- devigged prob of our side at placement (CLV open)
    edge_pct REAL,
    ev REAL,
    kelly_fraction REAL,
    recommended_stake REAL,
    steam_detected INTEGER DEFAULT 0,
    created_at TEXT NOT NULL,
    UNIQUE(game_id, player_name, market, line, side, bookmaker)
);
CREATE INDEX IF NOT EXISTS idx_bet_candidates_created ON bet_candidates(created_at);

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
CREATE INDEX IF NOT EXISTS idx_orders_placed_at ON orders(placed_at);

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

-- Statcast pitcher season stats (Baseball Savant, no API key required).
-- Keyed by (bdl player_id, season); mlb_player_id is the MLBAM ID used to join
-- the Savant CSV download. Null columns mean Savant didn't have data for that
-- player in that season — feature_builder falls back to 0 / league-avg safely.
CREATE TABLE IF NOT EXISTS statcast_pitcher_stats (
    player_id               INTEGER NOT NULL,
    mlb_player_id           INTEGER,
    season                  INTEGER NOT NULL,
    whiff_pct               REAL,
    chase_rate              REAL,
    barrel_pct_against      REAL,
    hard_hit_pct_against    REAL,
    spin_rate_ff            REAL,
    avg_exit_velocity_against REAL,
    updated_at              TEXT,
    PRIMARY KEY (player_id, season)
);
CREATE INDEX IF NOT EXISTS idx_statcast_pitcher_mlb
    ON statcast_pitcher_stats(mlb_player_id, season);

-- Statcast batter season stats (Baseball Savant).
CREATE TABLE IF NOT EXISTS statcast_batter_stats (
    player_id           INTEGER NOT NULL,
    mlb_player_id       INTEGER,
    season              INTEGER NOT NULL,
    exit_velocity_avg   REAL,
    launch_angle_avg    REAL,
    barrel_pct          REAL,
    xwoba               REAL,
    sprint_speed        REAL,
    whiff_pct           REAL,
    hard_hit_pct        REAL,
    updated_at          TEXT,
    PRIMARY KEY (player_id, season)
);
CREATE INDEX IF NOT EXISTS idx_statcast_batter_mlb
    ON statcast_batter_stats(mlb_player_id, season);

-- Model registry: one row per training run per market.
-- is_champion=1 marks the model currently loaded by projections.py.
CREATE TABLE IF NOT EXISTS model_registry (
    id                  INTEGER PRIMARY KEY AUTOINCREMENT,
    market              TEXT NOT NULL,
    model_type          TEXT NOT NULL,   -- 'lgbm' | 'glm'
    version             TEXT NOT NULL,   -- ISO timestamp of training run
    train_mae           REAL,
    val_mae             REAL,
    poisson_deviance    REAL,
    n_train             INTEGER,
    n_val               INTEGER,
    feature_list        TEXT,            -- JSON array of feature names
    is_champion         INTEGER DEFAULT 0,
    trained_at          TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_model_registry_market
    ON model_registry(market, is_champion);

-- SHAP feature importance per training run.
CREATE TABLE IF NOT EXISTS model_feature_importance (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    market          TEXT,
    model_version   TEXT,
    feature         TEXT,
    shap_mean_abs   REAL,
    rank            INTEGER,
    computed_at     TEXT
);
CREATE INDEX IF NOT EXISTS idx_feature_importance_market
    ON model_feature_importance(market, model_version);

-- Per-bet calibration log. Written by settle_results after each resolution.
-- Used by calibrate_model.py to fit Platt/isotonic transforms per market.
CREATE TABLE IF NOT EXISTS calibration_log (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    alert_id        INTEGER REFERENCES alerts_sent(alert_id),
    market          TEXT NOT NULL,
    predicted_prob  REAL NOT NULL,
    prob_bin        REAL NOT NULL,    -- floor(predicted_prob / 0.05) * 0.05
    actual_outcome  INTEGER NOT NULL, -- 1 = won, 0 = lost
    settled_at      TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_calibration_log_market
    ON calibration_log(market, settled_at);

-- Active calibration parameters per market (Platt scaling coefficients).
-- Only one is_active=1 row per market at a time.
CREATE TABLE IF NOT EXISTS calibration_params (
    id                      INTEGER PRIMARY KEY AUTOINCREMENT,
    market                  TEXT NOT NULL,
    method                  TEXT NOT NULL,  -- 'platt' | 'isotonic' | 'identity'
    params_json             TEXT NOT NULL,  -- {"a": float, "b": float} for Platt
    n_samples               INTEGER NOT NULL,
    brier_score             REAL,
    brier_score_calibrated  REAL,
    fitted_at               TEXT NOT NULL,
    is_active               INTEGER DEFAULT 0,
    UNIQUE(market, fitted_at)
);
CREATE INDEX IF NOT EXISTS idx_calibration_params_active
    ON calibration_params(market, is_active);

-- Daily bankroll snapshots with circuit-breaker state.
-- Written by settle_results at end of each settlement run.
CREATE TABLE IF NOT EXISTS bankroll_snapshots (
    id                      INTEGER PRIMARY KEY AUTOINCREMENT,
    snapshot_date           TEXT NOT NULL UNIQUE,
    bankroll                REAL NOT NULL,
    daily_pnl               REAL NOT NULL,
    rolling_7d_pnl          REAL,
    rolling_7d_roi          REAL,
    total_bets              INTEGER,
    total_wins              INTEGER,
    brier_score             REAL,
    kelly_fraction_override REAL DEFAULT 1.0,
    halved_at               TEXT,
    full_stop               INTEGER DEFAULT 0
);

-- Pairwise joint-outcome records for empirical correlation learning.
-- Written by settle_results for every pair of bets on the same game.
CREATE TABLE IF NOT EXISTS bet_pair_outcomes (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    alert_id_a   INTEGER REFERENCES alerts_sent(alert_id),
    alert_id_b   INTEGER REFERENCES alerts_sent(alert_id),
    pair_type    TEXT NOT NULL,  -- 'same_team_batters' | 'pitcher_batter' | 'same_game_opp'
    both_won     INTEGER NOT NULL,
    a_won        INTEGER NOT NULL,
    b_won        INTEGER NOT NULL,
    game_id      TEXT,
    settled_date TEXT,
    UNIQUE(alert_id_a, alert_id_b)
);
CREATE INDEX IF NOT EXISTS idx_bet_pair_outcomes_type
    ON bet_pair_outcomes(pair_type, settled_date);

-- Per-player batter platoon splits (vs LHP / vs RHP) computed from game logs.
-- Written nightly by sync_statcast. Used by feature_builder.compute_platoon_split
-- with Bayesian shrinkage toward the overall season rate when n_pa < 50.
CREATE TABLE IF NOT EXISTS batter_platoon_splits (
    player_id  INTEGER NOT NULL,
    season     INTEGER NOT NULL,
    vs_hand    TEXT    NOT NULL,  -- 'L' or 'R'
    market     TEXT    NOT NULL,  -- 'batter_hits' | 'batter_home_runs' | 'batter_total_bases'
    rate_per_pa REAL,
    n_pa       INTEGER,
    updated_at TEXT,
    PRIMARY KEY (player_id, season, vs_hand, market)
);

-- Empirically fitted portfolio correlation parameters.
-- Written by fit_correlations.py after ≥100 pairs per type accumulate.
-- Falls back to config constants when no is_active=1 row exists.
CREATE TABLE IF NOT EXISTS correlation_params (
    id                  INTEGER PRIMARY KEY AUTOINCREMENT,
    same_team_batters   REAL,
    pitcher_batter      REAL,
    same_game_opp       REAL,
    n_same_team         INTEGER,
    n_pitcher_batter    INTEGER,
    n_same_game         INTEGER,
    ci_half_width       REAL,    -- bootstrap 95% CI half-width (trust indicator)
    fitted_at           TEXT NOT NULL,
    is_active           INTEGER DEFAULT 0
);

-- Distribution statistics of the val-set predicted means captured at training time.
-- One row per (market, model_version, feature='val_mu'). Used by monitor_drift.py to
-- compute PSI against recent projections.projected_mean values.
CREATE TABLE IF NOT EXISTS model_training_baseline (
    market          TEXT NOT NULL,
    model_version   TEXT NOT NULL,
    feature         TEXT NOT NULL,   -- 'val_mu' (val-set predicted mean distribution)
    mean            REAL,
    std             REAL,
    p10             REAL,
    p50             REAL,
    p90             REAL,
    PRIMARY KEY (market, model_version, feature)
);

-- Weekly model-health snapshots written by monitor_drift.py.
-- drift_score = (mae_this_window - mae_baseline) / mae_baseline.
-- psi_score   = Population Stability Index on projected_mean distribution.
CREATE TABLE IF NOT EXISTS model_drift_log (
    id                  INTEGER PRIMARY KEY AUTOINCREMENT,
    market              TEXT NOT NULL,
    model_version       TEXT NOT NULL,
    window_start        TEXT,
    window_end          TEXT,
    mae_this_window     REAL,
    mae_baseline        REAL,
    drift_score         REAL,
    psi_score           REAL,
    retrain_triggered   INTEGER DEFAULT 0,
    logged_at           TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_model_drift_log_market
    ON model_drift_log(market, logged_at);

-- Per-team offensive stats, recomputed nightly from batter_game_logs.
-- Used by scan_props to adjust pitcher projections for opponent quality.
-- One row per team; overwritten each run (no season column needed because
-- sync_stats only ever holds the current season's data).
CREATE TABLE IF NOT EXISTS team_stats (
    team_id       INTEGER PRIMARY KEY,
    k_rate        REAL,   -- team batter K rate (K / PA); used to adjust pitcher K projections
    runs_per_game REAL,   -- team average runs per game; used to adjust pitcher ER projections
    last_updated  TEXT,
    UNIQUE(team_id)
);

-- Per-bookmaker systematic pricing bias by (market, side).
-- Positive avg_bias = soft book prices this side CHEAPER than sharp (in devigged prob units).
-- Computed weekly from the last 90 days of prop_snapshots; requires >=50 observations.
-- Used by scan_props to apply a small edge boost when bias > BOOKMAKER_BIAS_THRESHOLD.
CREATE TABLE IF NOT EXISTS bookmaker_bias (
    bookmaker       TEXT NOT NULL,
    market          TEXT NOT NULL,
    side            TEXT NOT NULL,  -- 'over' | 'under'
    avg_bias        REAL,           -- mean(sharp_devigged_side - soft_devigged_side)
    n_observations  INTEGER,
    last_computed   TEXT,
    PRIMARY KEY (bookmaker, market, side)
);

-- Walk-forward backtest results persisted for trend analysis.
-- One row per (market, model_type, run_at) so you can compare model quality
-- week-over-week and spot degradation before it affects live bets.
CREATE TABLE IF NOT EXISTS walk_forward_results (
    id                       INTEGER PRIMARY KEY AUTOINCREMENT,
    market                   TEXT    NOT NULL,
    model_type               TEXT    NOT NULL,  -- 'glm' | 'lgbm'
    mode                     TEXT    NOT NULL,  -- 'sliding' | 'expanding'
    train_window_days        INTEGER NOT NULL,
    step_days                INTEGER NOT NULL,
    n_folds                  INTEGER,
    weighted_mae             REAL,
    weighted_poisson_deviance REAL,
    weighted_var_ratio       REAL,
    sharpe_ratio             REAL,
    sortino_ratio            REAL,
    max_drawdown             REAL,
    run_at                   TEXT    NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_wf_results_market
    ON walk_forward_results(market, run_at);

-- Running batter box-score totals for in-progress games.
-- Written by BDLLiveClient.sync_game_states() every ~20s.
-- Used by the Live State Machine to know how many hits/TBs etc. a batter
-- has ALREADY accumulated, so the rest-of-game projection can be
-- applied to the REMAINING line (full_line - already_hit).
CREATE TABLE IF NOT EXISTS live_batter_stats (
    game_id           TEXT    NOT NULL,
    bdl_player_id     INTEGER NOT NULL,
    hits              INTEGER DEFAULT 0,
    total_bases       INTEGER DEFAULT 0,
    home_runs         INTEGER DEFAULT 0,
    plate_appearances INTEGER DEFAULT 0,
    updated_at        TEXT,
    PRIMARY KEY (game_id, bdl_player_id)
);
