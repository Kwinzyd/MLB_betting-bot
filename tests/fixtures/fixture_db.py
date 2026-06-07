"""Shared fixture DB factory for integration tests.

Creates a real SQLite file at a given path using the production schema.sql,
runs all migrations (mlb_id, last_synced_at), and seeds with realistic data:
  2 teams, 5 players (2 pitchers + 3 batters), 2 games, 20 game logs each,
  prop snapshots, projections, alerts, bankroll snapshot.
"""
from __future__ import annotations

import os
import random
import sqlite3
import uuid
from datetime import datetime, timezone

_SCHEMA_PATH = os.path.join(
    os.path.dirname(__file__), '..', '..', 'src', 'data', 'schema.sql'
)

# Canonical date strings used by the seed data so tests can reference them.
COMPLETED_GAME_DATE = '2024-07-15'
SCHEDULED_GAME_DATE = '2024-07-16'
COMPLETED_GAME_ID = 'g1'
SCHEDULED_GAME_ID = 'g2'
TODAY = datetime.now(timezone.utc).strftime('%Y-%m-%d')


def create_schema_only(conn: sqlite3.Connection) -> None:
    """Apply the production schema.sql (and migrations) to `conn`. No seed data.

    Unit tests should build their tables with this instead of hand-rolling
    CREATE TABLE statements — that is how the old fixtures silently drifted from
    production (missing model_prob_over, bankroll_snapshots, last_synced_at, …)
    and turned routine schema changes into a wall of red tests. On a fresh load
    every column already exists, so _run_migrations is a no-op here; it only
    matters when migrating a pre-existing DB.
    """
    with open(_SCHEMA_PATH) as f:
        conn.executescript(f.read())
    _run_migrations(conn)


def memory_conn() -> sqlite3.Connection:
    """In-memory SQLite connection preloaded with the full production schema.

    Drop-in replacement for `sqlite3.connect(':memory:')` + a hand-rolled
    schema. Row factory is set so rows behave like dicts, matching production
    get_db_connection().
    """
    conn = sqlite3.connect(':memory:')
    conn.row_factory = sqlite3.Row
    create_schema_only(conn)
    return conn


def create_fixture_db_at_path(path: str) -> None:
    """Write schema + seed data to `path`. Pass ':memory:' for in-memory (not sharable)."""
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    create_schema_only(conn)
    _seed(conn)
    conn.commit()
    conn.close()


# ---------------------------------------------------------------------------
# Migrations (mirrors db.py _migrate_* helpers for fresh DBs)
# ---------------------------------------------------------------------------

def _run_migrations(conn: sqlite3.Connection) -> None:
    players_cols = {r['name'] for r in conn.execute("PRAGMA table_info(players)").fetchall()}
    if 'mlb_id' not in players_cols:
        conn.execute("ALTER TABLE players ADD COLUMN mlb_id INTEGER")

    games_cols = {r['name'] for r in conn.execute("PRAGMA table_info(games)").fetchall()}
    if 'last_synced_at' not in games_cols:
        conn.execute("ALTER TABLE games ADD COLUMN last_synced_at TEXT")


# ---------------------------------------------------------------------------
# Seed helpers
# ---------------------------------------------------------------------------

def _seed(conn: sqlite3.Connection) -> None:
    rng = random.Random(42)

    # Teams
    conn.executemany(
        "INSERT INTO teams (team_id, abbreviation, name, league, division) VALUES (?,?,?,?,?)",
        [
            (1, 'NYY', 'New York Yankees', 'AL', 'East'),
            (2, 'BOS', 'Boston Red Sox', 'AL', 'East'),
        ]
    )

    # Players
    conn.executemany(
        "INSERT INTO players (player_id, name, team_id, position, bats, throws) VALUES (?,?,?,?,?,?)",
        [
            (101, 'Gerrit Cole',    1, 'SP', 'R', 'R'),
            (201, 'Chris Sale',     2, 'SP', 'L', 'L'),
            (102, 'Aaron Judge',    1, 'RF', 'R', 'R'),
            (103, 'Gleyber Torres', 1, '2B', 'R', 'R'),
            (202, 'Rafael Devers',  2, '3B', 'L', 'R'),
        ]
    )

    # Park factors
    conn.execute(
        "INSERT INTO park_factors (venue, runs_factor, hr_factor, hits_factor, strikeouts_factor) "
        "VALUES ('Yankee Stadium', 1.05, 1.15, 1.02, 0.97)"
    )

    # Games
    conn.execute(
        "INSERT INTO games (game_id, bdl_game_id, date, game_time, home_team, away_team, "
        "home_team_id, away_team_id, venue, status, home_score, away_score) "
        "VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
        (COMPLETED_GAME_ID, 1001, COMPLETED_GAME_DATE,
         f'{COMPLETED_GAME_DATE}T19:05:00', 'New York Yankees', 'Boston Red Sox',
         1, 2, 'Yankee Stadium', 'COMPLETED', 5, 3)
    )
    conn.execute(
        "INSERT INTO games (game_id, bdl_game_id, date, game_time, home_team, away_team, "
        "home_team_id, away_team_id, venue, status, lineups_confirmed_at) "
        "VALUES (?,?,?,?,?,?,?,?,?,?,?)",
        (SCHEDULED_GAME_ID, 1002, SCHEDULED_GAME_DATE,
         f'{SCHEDULED_GAME_DATE}T19:05:00', 'New York Yankees', 'Boston Red Sox',
         1, 2, 'Yankee Stadium', 'SCHEDULED', f'{SCHEDULED_GAME_DATE}T16:00:00')
    )

    # Pitcher game logs: 20 starts each
    p_rows = []
    for pid in (101, 201):
        for i in range(20):
            m, d = 4 + i // 4, (i % 4) * 7 + 1
            p_rows.append((
                2000 + pid * 20 + i, pid, f'2024-{m:02d}-{d:02d}',
                rng.choice([5.0, 6.0, 6.1, 7.0]),
                rng.randint(3, 8), rng.randint(1, 4), rng.randint(1, 3),
                rng.randint(1, 3), rng.randint(5, 11), rng.randint(0, 2),
                rng.randint(75, 100),
            ))
    conn.executemany(
        "INSERT INTO pitcher_game_logs "
        "(game_id,player_id,date,innings_pitched,hits_allowed,runs_allowed,"
        "earned_runs,walks,strikeouts,home_runs_allowed,pitches_thrown) "
        "VALUES (?,?,?,?,?,?,?,?,?,?,?)",
        p_rows
    )

    # Batter game logs: 20 games each
    b_rows = []
    for pid in (102, 103, 202):
        for i in range(20):
            m, d = 4 + i // 4, (i % 4) * 7 + 1
            hits = rng.randint(0, 3)
            hr = 1 if hits >= 2 and rng.random() > 0.7 else 0
            tb = hits + hr
            b_rows.append((
                3000 + pid * 20 + i, pid, f'2024-{m:02d}-{d:02d}',
                rng.randint(3, 5), hits, 0, 0, hr,
                rng.randint(0, 2), rng.randint(0, 2),
                rng.randint(0, 1), rng.randint(0, 2), tb, rng.randint(3, 5),
            ))
    conn.executemany(
        "INSERT INTO batter_game_logs "
        "(game_id,player_id,date,at_bats,hits,doubles,triples,home_runs,"
        "runs,rbis,walks,strikeouts,total_bases,plate_appearances) "
        "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
        b_rows
    )

    # Probable pitchers for g2
    conn.executemany(
        "INSERT INTO probable_pitchers (game_id,team,player_name,player_id,throws,date) "
        "VALUES (?,?,?,?,?,?)",
        [
            (SCHEDULED_GAME_ID, 'New York Yankees', 'Gerrit Cole', 101, 'R', SCHEDULED_GAME_DATE),
            (SCHEDULED_GAME_ID, 'Boston Red Sox',   'Chris Sale',  201, 'L', SCHEDULED_GAME_DATE),
        ]
    )

    # Daily lineups for g2
    conn.executemany(
        "INSERT INTO daily_lineups (game_id,team,player_name,player_id,lineup_position,date) "
        "VALUES (?,?,?,?,?,?)",
        [
            (SCHEDULED_GAME_ID, 'New York Yankees', 'Aaron Judge',    102, 3, SCHEDULED_GAME_DATE),
            (SCHEDULED_GAME_ID, 'New York Yankees', 'Gleyber Torres', 103, 2, SCHEDULED_GAME_DATE),
            (SCHEDULED_GAME_ID, 'Boston Red Sox',   'Rafael Devers',  202, 4, SCHEDULED_GAME_DATE),
        ]
    )

    # Prop snapshots for g1 (CLV lookup) and g2 (betting)
    snaps = [
        # g1 completed — for CLV calculation in settle_results
        (str(uuid.uuid4()), COMPLETED_GAME_ID, 'Gerrit Cole', 'pitcher_strikeouts',
         6.5, 2.05, 1.8, 'pinnacle', f'{COMPLETED_GAME_DATE}T14:00:00', 0.56, 0.44),
        (str(uuid.uuid4()), COMPLETED_GAME_ID, 'Aaron Judge', 'batter_hits',
         0.5, 1.72, 2.1, 'pinnacle', f'{COMPLETED_GAME_DATE}T14:00:00', 0.58, 0.42),
        # g2 scheduled — sharp truth + soft betting line
        (str(uuid.uuid4()), SCHEDULED_GAME_ID, 'Gerrit Cole', 'pitcher_strikeouts',
         6.5, 2.05, 1.80, 'pinnacle',    f'{SCHEDULED_GAME_DATE}T14:00:00', 0.52, 0.48),
        (str(uuid.uuid4()), SCHEDULED_GAME_ID, 'Gerrit Cole', 'pitcher_strikeouts',
         6.5, 2.20, 1.72, 'draftkings', f'{SCHEDULED_GAME_DATE}T14:00:00', None, None),
    ]
    conn.executemany(
        "INSERT INTO prop_snapshots "
        "(snapshot_id,game_id,player_name,market,line,over_odds,under_odds,"
        "bookmaker,timestamp,devigged_over,devigged_under) VALUES (?,?,?,?,?,?,?,?,?,?,?)",
        snaps
    )

    # Projection for g2 Gerrit Cole — strong edge vs sharp (0.66 vs 0.52 = 14%).
    # sample_size clears rank_edge's MIN_SAMPLE_SIZE gate so the bet is playable.
    conn.execute(
        "INSERT INTO projections "
        "(game_id,player_name,market,projected_mean,prob_over,prob_under,context_json,timestamp) "
        "VALUES (?,?,?,?,?,?,?,?)",
        (SCHEDULED_GAME_ID, 'Gerrit Cole', 'pitcher_strikeouts',
         7.5, 0.66, 0.34, '{"sample_size": 20}', f'{SCHEDULED_GAME_DATE}T14:00:00')
    )

    # Alerts on g1 — needed by settle_results and calibration tests
    conn.executemany(
        "INSERT INTO alerts_sent "
        "(player_name,market,line,side,edge,ev,kelly_stake,bookmaker,"
        "odds,opening_odds,model_prob_over,model_prob_under,game_id,timestamp) "
        "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
        [
            ('Gerrit Cole', 'pitcher_strikeouts', 6.5, 'over', 0.08, 0.04, 50.0,
             'draftkings', 2.0, 2.0, 0.64, 0.36, COMPLETED_GAME_ID, f'{COMPLETED_GAME_DATE}T14:00:00'),
            ('Aaron Judge', 'batter_hits', 0.5, 'over', 0.06, 0.03, 30.0,
             'fanduel', 1.85, 1.85, 0.60, 0.40, COMPLETED_GAME_ID, f'{COMPLETED_GAME_DATE}T14:00:00'),
        ]
    )

    # Bankroll snapshot for TODAY (so circuit breaker queries find it)
    conn.execute(
        "INSERT INTO bankroll_snapshots "
        "(snapshot_date,bankroll,daily_pnl,rolling_7d_pnl,rolling_7d_roi,"
        "total_bets,total_wins,kelly_fraction_override,full_stop) "
        "VALUES (?,?,?,?,?,?,?,?,?)",
        (TODAY, 1075.5, 75.5, 75.5, 0.075, 2, 2, 1.0, 0)
    )
