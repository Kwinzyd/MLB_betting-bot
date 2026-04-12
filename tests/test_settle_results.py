import pytest
import sqlite3
from unittest.mock import patch


@pytest.fixture
def memory_db():
    """In-memory DB seeded with a completed game, a player, and game logs."""
    conn = sqlite3.connect(':memory:')
    conn.row_factory = sqlite3.Row
    conn.executescript('''
        CREATE TABLE games (
            game_id TEXT PRIMARY KEY, bdl_game_id INTEGER, status TEXT
        );
        CREATE TABLE players (
            player_id INTEGER PRIMARY KEY, name TEXT
        );
        CREATE TABLE alerts_sent (
            alert_id INTEGER PRIMARY KEY AUTOINCREMENT,
            player_name TEXT, market TEXT, line REAL, side TEXT,
            odds REAL, opening_odds REAL, kelly_stake REAL, game_id TEXT,
            timestamp TEXT, edge REAL, ev REAL, bookmaker TEXT,
            UNIQUE(player_name, market, line, bookmaker)
        );
        CREATE TABLE bet_results (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            alert_id INTEGER, actual_value REAL, result TEXT,
            profit REAL, closing_odds REAL, clv REAL
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
        CREATE TABLE prop_snapshots (
            snapshot_id TEXT PRIMARY KEY, game_id TEXT, player_name TEXT,
            market TEXT, line REAL, over_odds REAL, under_odds REAL,
            bookmaker TEXT, timestamp TEXT, devigged_over REAL, devigged_under REAL
        );
    ''')
    conn.execute("INSERT INTO games VALUES ('g1', 999, 'COMPLETED')")
    conn.execute("INSERT INTO players VALUES (10, 'Gerrit Cole')")
    conn.execute(
        "INSERT INTO pitcher_game_logs (game_id, player_id, date, innings_pitched, "
        "hits_allowed, runs_allowed, earned_runs, walks, strikeouts, home_runs_allowed, pitches_thrown) "
        "VALUES (999, 10, '2024-05-01', 6.0, 4, 2, 2, 1, 8, 1, 95)"
    )
    # Prop snapshot used for CLV calculation
    conn.execute(
        "INSERT INTO prop_snapshots VALUES "
        "('snap1', 'g1', 'Gerrit Cole', 'pitcher_strikeouts', 6.5, 2.1, 1.75, "
        "'draftkings', '2024-05-01T20:00:00', NULL, NULL)"
    )
    conn.commit()
    return conn


def _insert_alert(conn, side='over', line=6.5, odds=1.9, stake=50.0, bookmaker='draftkings'):
    conn.execute('''
        INSERT INTO alerts_sent
        (player_name, market, line, side, odds, opening_odds, kelly_stake, game_id, bookmaker)
        VALUES ('Gerrit Cole', 'pitcher_strikeouts', ?, ?, ?, ?, ?, 'g1', ?)
    ''', (line, side, odds, odds, stake, bookmaker))
    conn.commit()


@patch('src.pipelines.settle_results.get_db_connection')
def test_settle_results_win(mock_get_db, memory_db):
    """Over bet wins when actual stat exceeds the line."""
    _insert_alert(memory_db, side='over', line=6.5, odds=1.9, stake=50.0)
    mock_get_db.return_value.__enter__.return_value = memory_db
    mock_get_db.return_value.__exit__.return_value = None

    from src.pipelines.settle_results import settle_results
    settle_results()

    result = memory_db.execute("SELECT * FROM bet_results").fetchone()
    assert result is not None
    assert result['result'] == 'WIN'
    assert result['actual_value'] == 8.0
    assert result['profit'] == pytest.approx(45.0)  # 50 * (1.9 - 1)


@patch('src.pipelines.settle_results.get_db_connection')
def test_settle_results_loss(mock_get_db, memory_db):
    """Over bet loses when actual stat falls below the line."""
    # Override strikeouts to 5 (below line of 6.5)
    memory_db.execute("UPDATE pitcher_game_logs SET strikeouts = 5 WHERE player_id = 10")
    memory_db.commit()
    _insert_alert(memory_db, side='over', line=6.5, odds=1.9, stake=50.0)
    mock_get_db.return_value.__enter__.return_value = memory_db
    mock_get_db.return_value.__exit__.return_value = None

    from src.pipelines.settle_results import settle_results
    settle_results()

    result = memory_db.execute("SELECT * FROM bet_results").fetchone()
    assert result['result'] == 'LOSS'
    assert result['profit'] == -50.0


@patch('src.pipelines.settle_results.get_db_connection')
def test_settle_results_push(mock_get_db, memory_db):
    """Bet pushes when actual stat exactly equals the line."""
    # Set strikeouts to exactly 6.5 is impossible for an integer stat, so use line=8.0
    _insert_alert(memory_db, side='over', line=8.0, odds=1.9, stake=50.0)
    mock_get_db.return_value.__enter__.return_value = memory_db
    mock_get_db.return_value.__exit__.return_value = None

    from src.pipelines.settle_results import settle_results
    settle_results()

    result = memory_db.execute("SELECT * FROM bet_results").fetchone()
    assert result['result'] == 'PUSH'
    assert result['profit'] == 0.0


@patch('src.pipelines.settle_results.get_db_connection')
def test_settle_results_clv(mock_get_db, memory_db):
    """CLV is a non-zero float when a closing prop_snapshot exists."""
    _insert_alert(memory_db, side='over', line=6.5, odds=1.95, stake=50.0)
    mock_get_db.return_value.__enter__.return_value = memory_db
    mock_get_db.return_value.__exit__.return_value = None

    from src.pipelines.settle_results import settle_results
    settle_results()

    result = memory_db.execute("SELECT clv FROM bet_results").fetchone()
    assert result is not None
    # CLV = devigged_closing_over - (1 / opening_odds); snapshot has over_odds=2.1, under=1.75
    assert isinstance(result['clv'], float)
    assert result['clv'] != 0.0


@patch('src.pipelines.settle_results.get_db_connection')
def test_settle_results_idempotent(mock_get_db, memory_db):
    """Running settle_results twice does not create duplicate bet_results rows."""
    _insert_alert(memory_db, side='over', line=6.5, odds=1.9, stake=50.0)
    mock_get_db.return_value.__enter__.return_value = memory_db
    mock_get_db.return_value.__exit__.return_value = None

    from src.pipelines.settle_results import settle_results
    settle_results()
    settle_results()

    rows = memory_db.execute("SELECT * FROM bet_results").fetchall()
    assert len(rows) == 1


@patch('src.pipelines.settle_results.get_db_connection')
def test_settle_results_batter_hits(mock_get_db, memory_db):
    """The batter_hits market resolves using the hits column from batter_game_logs."""
    memory_db.execute("INSERT INTO players VALUES (20, 'Aaron Judge')")
    memory_db.execute(
        "INSERT INTO batter_game_logs (game_id, player_id, date, at_bats, hits, "
        "doubles, triples, home_runs, runs, rbis, walks, strikeouts, total_bases, plate_appearances) "
        "VALUES (999, 20, '2024-05-01', 4, 2, 0, 0, 0, 1, 1, 0, 1, 2, 4)"
    )
    memory_db.execute('''
        INSERT INTO alerts_sent
        (player_name, market, line, side, odds, opening_odds, kelly_stake, game_id, bookmaker)
        VALUES ('Aaron Judge', 'batter_hits', 1.5, 'over', 2.0, 2.0, 25.0, 'g1', 'fanduel')
    ''')
    memory_db.commit()
    mock_get_db.return_value.__enter__.return_value = memory_db
    mock_get_db.return_value.__exit__.return_value = None

    from src.pipelines.settle_results import settle_results
    settle_results()

    result = memory_db.execute(
        "SELECT * FROM bet_results WHERE actual_value = 2.0"
    ).fetchone()
    assert result is not None
    assert result['result'] == 'WIN'
