import pytest
import sqlite3
from unittest.mock import patch, MagicMock, AsyncMock


@pytest.fixture
def memory_db():
    """In-memory DB with the tables sync_stats reads from and writes to."""
    conn = sqlite3.connect(':memory:')
    conn.row_factory = sqlite3.Row
    conn.executescript('''
        CREATE TABLE games (
            game_id TEXT PRIMARY KEY, bdl_game_id INTEGER,
            home_team TEXT, away_team TEXT, home_team_id INTEGER,
            away_team_id INTEGER, status TEXT
        );
        CREATE TABLE teams (
            team_id INTEGER PRIMARY KEY, abbreviation TEXT, name TEXT,
            league TEXT, division TEXT
        );
        CREATE TABLE players (
            player_id INTEGER PRIMARY KEY, name TEXT, team_id INTEGER,
            position TEXT, bats TEXT, throws TEXT, active BOOLEAN DEFAULT 1
        );
        CREATE TABLE pitcher_game_logs (
            game_id INTEGER, player_id INTEGER, date TEXT,
            innings_pitched REAL, hits_allowed INTEGER, runs_allowed INTEGER,
            earned_runs INTEGER, walks INTEGER, strikeouts INTEGER,
            home_runs_allowed INTEGER, pitches_thrown INTEGER,
            PRIMARY KEY (game_id, player_id)
        );
        CREATE TABLE batter_game_logs (
            game_id INTEGER, player_id INTEGER, date TEXT,
            at_bats INTEGER, hits INTEGER, doubles INTEGER, triples INTEGER,
            home_runs INTEGER, runs INTEGER, rbis INTEGER, walks INTEGER,
            strikeouts INTEGER, total_bases INTEGER, plate_appearances INTEGER,
            PRIMARY KEY (game_id, player_id)
        );
    ''')
    conn.execute('''
        INSERT INTO games (game_id, bdl_game_id, home_team, away_team,
                           home_team_id, away_team_id, status)
        VALUES ('odds_g1', 999, 'Yankees', 'Red Sox', 1, 2, 'SCHEDULED')
    ''')
    conn.commit()
    return conn


def _make_bdl_client(pitcher_stat, batter_stat):
    """Build a mock MLBStatsClient returning one completed game with two player stats."""
    client = MagicMock()
    client.get_teams = AsyncMock(return_value=[
        {'id': 1, 'abbreviation': 'NYY', 'full_name': 'Yankees', 'league': 'AL', 'division': 'ALE'},
        {'id': 2, 'abbreviation': 'BOS', 'full_name': 'Red Sox', 'league': 'AL', 'division': 'ALE'},
    ])
    client.get_games = AsyncMock(return_value=[
        {
            'id': 999, 'status': 'Final', 'date': '2024-05-01',
            'home_team': {'id': 1}, 'away_team': {'id': 2},
        },
    ])
    client.get_stats_batch = AsyncMock(return_value=[pitcher_stat, batter_stat])
    return client


PITCHER_STAT = {
    'player': {'id': 10, 'first_name': 'Gerrit', 'last_name': 'Cole',
               'position': 'P', 'bats': '', 'throws': 'R'},
    'team': {'id': 1},
    'game': {'id': 999, 'date': '2024-05-01'},
    'innings_pitched': 6.0,
    'hits_allowed': 4, 'runs_allowed': 2, 'earned_runs': 2,
    'walks': 1, 'strikeouts': 8, 'home_runs_allowed': 1, 'pitches_thrown': 95,
}

BATTER_STAT = {
    'player': {'id': 20, 'first_name': 'Aaron', 'last_name': 'Judge',
               'position': 'OF', 'bats': 'R', 'throws': 'R'},
    'team': {'id': 1},
    'game': {'id': 999, 'date': '2024-05-01'},
    'innings_pitched': None,
    'at_bats': 4, 'hits': 2, 'doubles': 1, 'triples': 0, 'home_runs': 1,
    'runs': 2, 'rbis': 2, 'walks': 1, 'strikeouts': 1, 'plate_appearances': 5,
}


@patch('src.pipelines.sync_stats.get_db_connection')
@patch('src.pipelines.sync_stats.MLBStatsClient')
async def test_sync_stats_pitcher_log(mock_client_cls, mock_get_db, memory_db):
    """Pitcher stats (innings_pitched present) are written to pitcher_game_logs."""
    mock_client_cls.return_value = _make_bdl_client(PITCHER_STAT, BATTER_STAT)
    mock_get_db.return_value.__enter__.return_value = memory_db
    mock_get_db.return_value.__exit__.return_value = None

    from src.pipelines.sync_stats import sync_stats
    await sync_stats()

    row = memory_db.execute(
        "SELECT * FROM pitcher_game_logs WHERE player_id = 10"
    ).fetchone()
    assert row is not None
    assert row['game_id'] == 999
    assert row['strikeouts'] == 8
    assert row['earned_runs'] == 2
    assert row['innings_pitched'] == 6.0


@patch('src.pipelines.sync_stats.get_db_connection')
@patch('src.pipelines.sync_stats.MLBStatsClient')
async def test_sync_stats_batter_log(mock_client_cls, mock_get_db, memory_db):
    """Batter stats are written to batter_game_logs with total_bases derived correctly."""
    mock_client_cls.return_value = _make_bdl_client(PITCHER_STAT, BATTER_STAT)
    mock_get_db.return_value.__enter__.return_value = memory_db
    mock_get_db.return_value.__exit__.return_value = None

    from src.pipelines.sync_stats import sync_stats
    await sync_stats()

    row = memory_db.execute(
        "SELECT * FROM batter_game_logs WHERE player_id = 20"
    ).fetchone()
    assert row is not None
    assert row['hits'] == 2
    assert row['home_runs'] == 1
    assert row['runs'] == 2
    # singles=0, doubles=1 (×2=2), triples=0, home_runs=1 (×4=4) → total_bases=6
    assert row['total_bases'] == 6


@patch('src.pipelines.sync_stats.get_db_connection')
@patch('src.pipelines.sync_stats.MLBStatsClient')
async def test_sync_stats_upserts_players(mock_client_cls, mock_get_db, memory_db):
    """Both pitcher and batter are inserted into the players table."""
    mock_client_cls.return_value = _make_bdl_client(PITCHER_STAT, BATTER_STAT)
    mock_get_db.return_value.__enter__.return_value = memory_db
    mock_get_db.return_value.__exit__.return_value = None

    from src.pipelines.sync_stats import sync_stats
    await sync_stats()

    players = {r['player_id']: r for r in memory_db.execute("SELECT * FROM players").fetchall()}
    assert 10 in players
    assert players[10]['name'] == 'Gerrit Cole'
    assert 20 in players
    assert players[20]['name'] == 'Aaron Judge'
