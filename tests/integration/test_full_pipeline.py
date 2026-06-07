"""Integration tests for the settle pipeline producing calibration_log rows.

These tests verify the NEW behaviours added in the quant upgrade:
- settle_results writes to calibration_log for WIN/LOSS bets
- settle_results emits bet_pair_outcomes for same-game pairs
- settle_results writes/updates bankroll_snapshots
"""
import sqlite3
from unittest.mock import patch

from tests.fixtures.fixture_db import COMPLETED_GAME_ID, TODAY


def _connect(db_path: str) -> sqlite3.Connection:
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    return conn


def _seed_game_logs_for_completed(db_path: str) -> None:
    """Insert box-score rows for the completed game so settle_results can resolve bets."""
    conn = _connect(db_path)
    conn.execute(
        "INSERT OR IGNORE INTO pitcher_game_logs "
        "(game_id,player_id,date,innings_pitched,hits_allowed,runs_allowed,"
        "earned_runs,walks,strikeouts,home_runs_allowed,pitches_thrown) "
        "VALUES (1001,101,'2024-07-15',7.0,4,2,2,1,8,0,95)"
    )
    conn.execute(
        "INSERT OR IGNORE INTO batter_game_logs "
        "(game_id,player_id,date,at_bats,hits,doubles,triples,home_runs,"
        "runs,rbis,walks,strikeouts,total_bases,plate_appearances) "
        "VALUES (1001,102,'2024-07-15',4,2,0,0,0,1,1,0,1,2,4)"
    )
    conn.commit()
    conn.close()


@patch('src.clients.telegram_bot.TelegramClient')
def test_settle_writes_calibration_log(mock_tg, db):
    """settle_results populates calibration_log for WIN/LOSS bets."""
    mock_tg.return_value.send_message_sync = lambda *a, **kw: None
    _seed_game_logs_for_completed(db)

    from src.pipelines.settle_results import settle_results
    settle_results()

    conn = _connect(db)
    rows = conn.execute("SELECT * FROM calibration_log").fetchall()
    conn.close()

    assert len(rows) >= 1
    row = rows[0]
    assert row['market'] in ('pitcher_strikeouts', 'batter_hits')
    assert row['predicted_prob'] > 0
    assert row['actual_outcome'] in (0, 1)
    assert row['prob_bin'] == round(round(row['predicted_prob'] / 0.05) * 0.05, 2)


@patch('src.clients.telegram_bot.TelegramClient')
def test_settle_emits_pair_outcomes(mock_tg, db):
    """settle_results writes bet_pair_outcomes for the same-game alert pair."""
    mock_tg.return_value.send_message_sync = lambda *a, **kw: None
    _seed_game_logs_for_completed(db)

    from src.pipelines.settle_results import settle_results
    settle_results()

    conn = _connect(db)
    rows = conn.execute("SELECT * FROM bet_pair_outcomes").fetchall()
    conn.close()

    # The fixture has 2 alerts on COMPLETED_GAME_ID → 1 pair
    assert len(rows) >= 1
    row = rows[0]
    assert row['pair_type'] in ('same_team_batters', 'pitcher_batter', 'same_game_opp')


@patch('src.clients.telegram_bot.TelegramClient')
def test_settle_updates_bankroll_snapshot(mock_tg, db):
    """settle_results upserts a bankroll_snapshots row after each run."""
    mock_tg.return_value.send_message_sync = lambda *a, **kw: None
    _seed_game_logs_for_completed(db)

    from src.pipelines.settle_results import settle_results
    settle_results()

    conn = _connect(db)
    snap = conn.execute(
        "SELECT * FROM bankroll_snapshots WHERE snapshot_date=?", (TODAY,)
    ).fetchone()
    conn.close()

    assert snap is not None
    assert snap['bankroll'] > 0


@patch('src.clients.telegram_bot.TelegramClient')
def test_settle_is_idempotent(mock_tg, db):
    """Running settle_results twice produces exactly one bet_result per alert."""
    mock_tg.return_value.send_message_sync = lambda *a, **kw: None
    _seed_game_logs_for_completed(db)

    from src.pipelines.settle_results import settle_results
    settle_results()
    settle_results()

    conn = _connect(db)
    n_results = conn.execute("SELECT COUNT(*) FROM bet_results").fetchone()[0]
    n_alerts = conn.execute(
        "SELECT COUNT(*) FROM alerts_sent WHERE game_id=?", (COMPLETED_GAME_ID,)
    ).fetchone()[0]
    conn.close()

    assert n_results == n_alerts
