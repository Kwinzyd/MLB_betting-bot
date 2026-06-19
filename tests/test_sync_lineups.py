import datetime
import sqlite3
from unittest.mock import patch, MagicMock, AsyncMock


@patch('src.pipelines.sync_lineups.get_eastern_local_date',
       return_value=datetime.date(2024, 5, 15))
def _memory_db_for_date(mock_date):
    """Helper — not a fixture; call explicitly to build seeded in-memory DB."""
    pass


import pytest


@pytest.fixture
def memory_db():
    """In-memory DB with a today's game ready for lineup sync."""
    conn = sqlite3.connect(':memory:')
    conn.row_factory = sqlite3.Row
    conn.executescript('''
        CREATE TABLE games (
            game_id TEXT PRIMARY KEY, bdl_game_id INTEGER,
            home_team TEXT, away_team TEXT, status TEXT, date TEXT,
            lineups_confirmed_at TEXT
        );
        CREATE TABLE daily_lineups (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            game_id TEXT, team TEXT, player_name TEXT, player_id INTEGER,
            lineup_position INTEGER, date TEXT,
            UNIQUE(game_id, player_name)
        );
        CREATE TABLE probable_pitchers (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            game_id TEXT, team TEXT, player_name TEXT, player_id INTEGER,
            throws TEXT, date TEXT,
            UNIQUE(game_id, team)
        );
    ''')
    conn.execute('''
        INSERT INTO games VALUES
        ('g1', 999, 'Yankees', 'Red Sox', 'SCHEDULED', '2024-05-15T18:10:00Z', NULL)
    ''')
    conn.commit()
    return conn


LINEUP_ENTRIES = [
    {
        'player': {'id': 10, 'first_name': 'Gerrit', 'last_name': 'Cole', 'bats_throws': 'Right/R'},
        'team': {'display_name': 'Yankees'},
        'is_probable_pitcher': True,
        'batting_order': None,
    },
    {
        'player': {'id': 20, 'first_name': 'Aaron', 'last_name': 'Judge', 'bats_throws': 'Right/R'},
        'team': {'display_name': 'Yankees'},
        'is_probable_pitcher': False,
        'batting_order': 3,
    },
]


@patch('src.pipelines.sync_lineups.get_eastern_local_date',
       return_value=datetime.date(2024, 5, 15))
@patch('src.pipelines.sync_lineups.get_db_connection')
@patch('src.pipelines.sync_lineups.MLBStatsClient')
async def test_sync_lineups_probable_pitcher(mock_client_cls, mock_get_db, _mock_date, memory_db):
    """A probable pitcher entry is written to the probable_pitchers table."""
    mock_client_cls.return_value.get_lineups = AsyncMock(return_value=LINEUP_ENTRIES)
    mock_get_db.return_value.__enter__.return_value = memory_db
    mock_get_db.return_value.__exit__.return_value = None

    from src.pipelines.sync_lineups import sync_lineups
    await sync_lineups()

    pitcher = memory_db.execute(
        "SELECT * FROM probable_pitchers WHERE game_id = 'g1'"
    ).fetchone()
    assert pitcher is not None
    assert pitcher['player_name'] == 'Gerrit Cole'
    assert pitcher['team'] == 'Yankees'
    assert pitcher['throws'] == 'R'


@patch('src.pipelines.sync_lineups.get_eastern_local_date',
       return_value=datetime.date(2024, 5, 15))
@patch('src.pipelines.sync_lineups.get_db_connection')
@patch('src.pipelines.sync_lineups.MLBStatsClient')
async def test_sync_lineups_batting_order(mock_client_cls, mock_get_db, _mock_date, memory_db):
    """A batter with a batting_order is written to daily_lineups with the correct position."""
    mock_client_cls.return_value.get_lineups = AsyncMock(return_value=LINEUP_ENTRIES)
    mock_get_db.return_value.__enter__.return_value = memory_db
    mock_get_db.return_value.__exit__.return_value = None

    from src.pipelines.sync_lineups import sync_lineups
    await sync_lineups()

    batter = memory_db.execute(
        "SELECT * FROM daily_lineups WHERE player_name = 'Aaron Judge'"
    ).fetchone()
    assert batter is not None
    assert batter['lineup_position'] == 3
    assert batter['team'] == 'Yankees'


@patch('src.pipelines.sync_lineups.get_eastern_local_date',
       return_value=datetime.date(2024, 5, 15))
@patch('src.pipelines.sync_lineups.get_db_connection')
@patch('src.pipelines.sync_lineups.MLBStatsClient')
async def test_sync_lineups_pitcher_upsert(mock_client_cls, mock_get_db, _mock_date, memory_db):
    """Running sync_lineups twice with a changed pitcher name updates the existing row."""
    mock_client_cls.return_value.get_lineups = AsyncMock(return_value=LINEUP_ENTRIES)
    mock_get_db.return_value.__enter__.return_value = memory_db
    mock_get_db.return_value.__exit__.return_value = None

    from src.pipelines.sync_lineups import sync_lineups
    await sync_lineups()

    # Second run — different pitcher announced
    updated_entries = [
        {
            'player': {'id': 11, 'first_name': 'Nathan', 'last_name': 'Eovaldi', 'bats_throws': 'Right/R'},
            'team': {'display_name': 'Yankees'},
            'is_probable_pitcher': True,
            'batting_order': None,
        }
    ]
    mock_client_cls.return_value.get_lineups = AsyncMock(return_value=updated_entries)
    await sync_lineups()

    pitchers = memory_db.execute("SELECT * FROM probable_pitchers WHERE game_id = 'g1'").fetchall()
    assert len(pitchers) == 1
    assert pitchers[0]['player_name'] == 'Nathan Eovaldi'


@patch('src.pipelines.sync_lineups.get_eastern_local_date',
       return_value=datetime.date(2024, 5, 15))
@patch('src.pipelines.sync_lineups.get_db_connection')
@patch('src.pipelines.sync_lineups.MLBStatsClient')
async def test_sync_lineups_includes_late_night_game(mock_client_cls, mock_get_db, _mock_date):
    """A 10pm-ET game rolls to the next UTC day (02:00Z). The old `date LIKE
    '<eastern-date>%'` filter dropped it; the Eastern-day window must include it
    so its lineups confirm pregame."""
    conn = sqlite3.connect(':memory:')
    conn.row_factory = sqlite3.Row
    conn.executescript('''
        CREATE TABLE games (
            game_id TEXT PRIMARY KEY, bdl_game_id INTEGER,
            home_team TEXT, away_team TEXT, status TEXT, date TEXT,
            lineups_confirmed_at TEXT
        );
        CREATE TABLE daily_lineups (
            id INTEGER PRIMARY KEY AUTOINCREMENT, game_id TEXT, team TEXT,
            player_name TEXT, player_id INTEGER, lineup_position INTEGER, date TEXT,
            UNIQUE(game_id, player_name)
        );
        CREATE TABLE probable_pitchers (
            id INTEGER PRIMARY KEY AUTOINCREMENT, game_id TEXT, team TEXT,
            player_name TEXT, player_id INTEGER, throws TEXT, date TEXT,
            UNIQUE(game_id, team)
        );
    ''')
    # 10pm ET on 2024-05-15 == 02:00 UTC on 2024-05-16.
    conn.execute(
        "INSERT INTO games VALUES ('late', 888, 'Yankees', 'Red Sox', "
        "'SCHEDULED', '2024-05-16T02:00:00Z', NULL)")
    conn.commit()

    mock_client_cls.return_value.get_lineups = AsyncMock(return_value=LINEUP_ENTRIES)
    mock_get_db.return_value.__enter__.return_value = conn
    mock_get_db.return_value.__exit__.return_value = None

    from src.pipelines.sync_lineups import sync_lineups
    await sync_lineups()

    pitcher = conn.execute("SELECT * FROM probable_pitchers WHERE game_id = 'late'").fetchone()
    assert pitcher is not None  # would be None under the old UTC-string filter
    assert pitcher['player_name'] == 'Gerrit Cole'
