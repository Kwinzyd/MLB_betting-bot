import pytest
from unittest.mock import patch

from tests.fixtures.fixture_db import memory_conn


@pytest.fixture
def memory_db():
    """In-memory DB (full production schema) seeded with a completed game,
    a player, and game logs. Built from schema.sql via memory_conn() so it can
    never drift from production columns the way a hand-rolled schema would."""
    conn = memory_conn()
    conn.execute("INSERT INTO games (game_id, bdl_game_id, status) VALUES ('g1', 999, 'COMPLETED')")
    conn.execute("INSERT INTO players (player_id, name) VALUES (10, 'Gerrit Cole')")
    conn.execute(
        "INSERT INTO pitcher_game_logs (game_id, player_id, date, innings_pitched, "
        "hits_allowed, runs_allowed, earned_runs, walks, strikeouts, home_runs_allowed, pitches_thrown) "
        "VALUES (999, 10, '2024-05-01', 6.0, 4, 2, 2, 1, 8, 1, 95)"
    )
    # Prop snapshot used for CLV calculation
    conn.execute(
        "INSERT INTO prop_snapshots "
        "(snapshot_id, game_id, player_name, market, line, over_odds, under_odds, "
        "bookmaker, timestamp, devigged_over, devigged_under) VALUES "
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
def test_single_voided_when_player_dnp(mock_get_db, memory_db):
    """Late scratch: box score is in for the game but the player has no row -> VOID + refund."""
    memory_db.execute("INSERT INTO players (player_id, name) VALUES (30, 'Mookie Betts')")
    # batter_game_logs has rows for the game (someone played) but not for Mookie.
    memory_db.execute(
        "INSERT INTO batter_game_logs (game_id, player_id, date, at_bats, hits, "
        "doubles, triples, home_runs, runs, rbis, walks, strikeouts, total_bases, plate_appearances) "
        "VALUES (999, 99, '2024-05-01', 4, 1, 0, 0, 0, 0, 0, 0, 1, 1, 4)"
    )
    memory_db.execute('''
        INSERT INTO alerts_sent
        (player_name, market, line, side, odds, opening_odds, kelly_stake, game_id, bookmaker)
        VALUES ('Mookie Betts', 'batter_hits', 0.5, 'over', 1.85, 1.85, 40.0, 'g1', 'fanduel')
    ''')
    memory_db.commit()
    mock_get_db.return_value.__enter__.return_value = memory_db
    mock_get_db.return_value.__exit__.return_value = None

    from src.pipelines.settle_results import settle_results
    settle_results()

    row = memory_db.execute(
        "SELECT * FROM bet_results WHERE alert_id = "
        "(SELECT alert_id FROM alerts_sent WHERE player_name = 'Mookie Betts')"
    ).fetchone()
    assert row is not None
    assert row['result'] == 'VOID'
    assert row['profit'] == 0.0
    assert row['actual_value'] is None


@patch('src.pipelines.settle_results.get_db_connection')
def test_single_pending_when_box_score_not_synced(mock_get_db, memory_db):
    """If the batter table has zero rows for the game, treat it as not-yet-synced (skip, don't void)."""
    memory_db.execute("INSERT INTO players (player_id, name) VALUES (31, 'Freddie Freeman')")
    memory_db.execute('''
        INSERT INTO alerts_sent
        (player_name, market, line, side, odds, opening_odds, kelly_stake, game_id, bookmaker)
        VALUES ('Freddie Freeman', 'batter_hits', 0.5, 'over', 1.85, 1.85, 40.0, 'g1', 'fanduel')
    ''')
    memory_db.commit()
    mock_get_db.return_value.__enter__.return_value = memory_db
    mock_get_db.return_value.__exit__.return_value = None

    from src.pipelines.settle_results import settle_results
    settle_results()

    row = memory_db.execute(
        "SELECT * FROM bet_results WHERE alert_id = "
        "(SELECT alert_id FROM alerts_sent WHERE player_name = 'Freddie Freeman')"
    ).fetchone()
    assert row is None  # deferred, not voided


@patch('src.pipelines.settle_results.get_db_connection')
def test_sgp_void_recalculates_payout(mock_get_db, memory_db):
    """3-leg SGP with one DNP leg pays out as a 2-leg parlay on remaining legs' odds."""
    import json as _json
    # Two batters with hits. Third batter has no row but team's box score is in -> DNP.
    memory_db.execute("INSERT INTO players (player_id, name) VALUES (40, 'A B')")
    memory_db.execute("INSERT INTO players (player_id, name) VALUES (41, 'C D')")
    memory_db.execute("INSERT INTO players (player_id, name) VALUES (42, 'E F')")
    for pid in (40, 41):
        memory_db.execute(
            "INSERT INTO batter_game_logs (game_id, player_id, date, at_bats, hits, "
            "doubles, triples, home_runs, runs, rbis, walks, strikeouts, total_bases, plate_appearances) "
            f"VALUES (999, {pid}, '2024-05-01', 4, 0, 0, 0, 0, 0, 0, 0, 1, 0, 4)"
        )
    legs = [
        {'player_name': 'Gerrit Cole', 'market': 'pitcher_strikeouts',
         'line': 6.5, 'side': 'over', 'odds': 1.9, 'bookmaker': 'fanduel'},
        {'player_name': 'A B', 'market': 'batter_hits',
         'line': 0.5, 'side': 'under', 'odds': 2.5, 'bookmaker': 'fanduel'},
        {'player_name': 'E F', 'market': 'batter_hits',  # DNP -> VOID
         'line': 0.5, 'side': 'under', 'odds': 2.0, 'bookmaker': 'fanduel'},
    ]
    memory_db.execute('''
        INSERT INTO sgp_candidates
        (game_id, legs_json, joint_prob, naive_parlay_odds, fair_odds,
         edge_vs_naive, kelly_stake, bookmakers, timestamp)
        VALUES ('g1', ?, 0.21, 9.5, 7.0, 0.05, 20.0, 'fanduel', '2024-05-01')
    ''', (_json.dumps(legs),))
    memory_db.commit()
    mock_get_db.return_value.__enter__.return_value = memory_db
    mock_get_db.return_value.__exit__.return_value = None

    from src.pipelines.settle_results import settle_results
    settle_results()

    row = memory_db.execute("SELECT * FROM sgp_results").fetchone()
    assert row is not None
    assert row['result'] == 'WIN'
    assert row['surviving_legs'] == 2
    # Recalc odds = 1.9 * 2.5 = 4.75; profit = 20 * (4.75 - 1) = 75.00
    assert row['recalc_odds'] == pytest.approx(4.75)
    assert row['profit'] == pytest.approx(75.0)


@patch('src.pipelines.settle_results.get_db_connection')
def test_sgp_full_refund_when_all_legs_void(mock_get_db, memory_db):
    """If every leg voids, SGP is fully refunded (profit = 0)."""
    import json as _json
    # Box score has a row for an unrelated player so DNP detection triggers for our two scratches.
    memory_db.execute("INSERT INTO players (player_id, name) VALUES (50, 'X Y')")
    memory_db.execute("INSERT INTO players (player_id, name) VALUES (51, 'Z W')")
    memory_db.execute(
        "INSERT INTO batter_game_logs (game_id, player_id, date, at_bats, hits, "
        "doubles, triples, home_runs, runs, rbis, walks, strikeouts, total_bases, plate_appearances) "
        "VALUES (999, 88, '2024-05-01', 4, 1, 0, 0, 0, 0, 0, 0, 1, 1, 4)"
    )
    legs = [
        {'player_name': 'X Y', 'market': 'batter_hits',
         'line': 0.5, 'side': 'over', 'odds': 1.8, 'bookmaker': 'fanduel'},
        {'player_name': 'Z W', 'market': 'batter_hits',
         'line': 0.5, 'side': 'over', 'odds': 1.7, 'bookmaker': 'fanduel'},
    ]
    memory_db.execute('''
        INSERT INTO sgp_candidates
        (game_id, legs_json, joint_prob, naive_parlay_odds, fair_odds,
         edge_vs_naive, kelly_stake, bookmakers, timestamp)
        VALUES ('g1', ?, 0.3, 3.06, 2.8, 0.05, 25.0, 'fanduel', '2024-05-01')
    ''', (_json.dumps(legs),))
    memory_db.commit()
    mock_get_db.return_value.__enter__.return_value = memory_db
    mock_get_db.return_value.__exit__.return_value = None

    from src.pipelines.settle_results import settle_results
    settle_results()

    row = memory_db.execute("SELECT * FROM sgp_results").fetchone()
    assert row is not None
    assert row['result'] == 'VOID'
    assert row['profit'] == 0.0
    assert row['surviving_legs'] == 0


@patch('src.pipelines.settle_results.get_db_connection')
def test_settle_results_batter_hits(mock_get_db, memory_db):
    """The batter_hits market resolves using the hits column from batter_game_logs."""
    memory_db.execute("INSERT INTO players (player_id, name) VALUES (20, 'Aaron Judge')")
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
