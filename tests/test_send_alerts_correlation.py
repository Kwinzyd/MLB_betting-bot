from unittest.mock import patch, AsyncMock
import pytest

from tests.fixtures.fixture_db import memory_conn


def _build_telegram_registry():
    from src.clients.execution.telegram_venue import TelegramVenue
    venue = TelegramVenue()
    venue._client.send_message = AsyncMock(return_value=True)
    return [venue], venue


def _add_candidate(conn, game_id, player, market, line, prob_over, snap_id, book='dk'):
    """Insert a playable candidate. devigged_* holds the sharp-devigged truth —
    set equal to prob_* so the model/sharp agreement gate passes."""
    conn.execute(
        "INSERT OR IGNORE INTO games (game_id, home_team, away_team, venue, status, date) "
        "VALUES (?, 'Yankees', 'Red Sox', 'Yankee Stadium', 'SCHEDULED', '2024-05-15')",
        (game_id,)
    )
    conn.execute('''
        INSERT INTO projections
        (game_id, player_name, market, projected_mean, prob_over, prob_under, context_json, timestamp)
        VALUES (?, ?, ?, ?, ?, ?, '{"sample_size": 20}', '2024-05-15T12:00:00')
    ''', (game_id, player, market, line, prob_over, 1.0 - prob_over))
    # odds=2.0 (implied 50%); sharp says prob_over → edge = (prob_over - 0.50)*100
    conn.execute('''
        INSERT INTO prop_snapshots
        (snapshot_id, game_id, player_name, market, line, over_odds, under_odds,
         bookmaker, timestamp, devigged_over, devigged_under)
        VALUES (?, ?, ?, ?, ?, 2.0, 2.0, ?, '2024-05-15T12:00:00', ?, ?)
    ''', (snap_id, game_id, player, market, line, book, prob_over, 1.0 - prob_over))


@pytest.fixture
def memory_db():
    return memory_conn()


@patch('src.pipelines.send_alerts.BETTING_ENABLED', True)
@patch('src.clients.execution.telegram_venue.BETTING_ENABLED', True)
@patch('src.pipelines.send_alerts.get_db_connection')
@patch('src.pipelines.send_alerts.build_venue_registry')
async def test_one_bet_per_player(mock_registry, mock_get_db, memory_db):
    """Same player with two playable markets → only the higher-edge one fires."""
    # Strikeouts: edge ~15% (prob_over=0.65)
    _add_candidate(memory_db, 'g1', 'Gerrit Cole', 'pitcher_strikeouts', 7.5, 0.65, 's1')
    # Earned runs: edge ~20% (prob_over=0.70) — should win
    _add_candidate(memory_db, 'g1', 'Gerrit Cole', 'pitcher_earned_runs', 2.5, 0.70, 's2')
    memory_db.commit()

    venues, _tg = _build_telegram_registry()
    mock_registry.return_value = venues
    mock_get_db.return_value.__enter__.return_value = memory_db
    mock_get_db.return_value.__exit__.return_value = None

    from src.pipelines.send_alerts import send_alerts
    await send_alerts()

    rows = memory_db.execute(
        "SELECT market FROM alerts_sent WHERE player_name = 'Gerrit Cole'"
    ).fetchall()
    assert len(rows) == 1
    assert rows[0]['market'] == 'pitcher_earned_runs'


@patch('src.pipelines.send_alerts.BETTING_ENABLED', True)
@patch('src.clients.execution.telegram_venue.BETTING_ENABLED', True)
@patch('src.pipelines.send_alerts.get_db_connection')
@patch('src.pipelines.send_alerts.build_venue_registry')
async def test_max_three_per_game(mock_registry, mock_get_db, memory_db):
    """Five playable candidates on one game → only the top 3 by edge fire."""
    # Edges: 20%, 18%, 15%, 12%, 10% — top 3 keep
    for i, (player, prob) in enumerate([
        ('Player A', 0.70),
        ('Player B', 0.68),
        ('Player C', 0.65),
        ('Player D', 0.62),
        ('Player E', 0.60),
    ]):
        _add_candidate(memory_db, 'g1', player, 'batter_hits', 1.5, prob, f's{i}')
    memory_db.commit()

    venues, _tg = _build_telegram_registry()
    mock_registry.return_value = venues
    mock_get_db.return_value.__enter__.return_value = memory_db
    mock_get_db.return_value.__exit__.return_value = None

    from src.pipelines.send_alerts import send_alerts
    await send_alerts()

    rows = memory_db.execute(
        "SELECT player_name FROM alerts_sent WHERE game_id = 'g1' ORDER BY edge DESC"
    ).fetchall()
    assert len(rows) == 3
    assert [r['player_name'] for r in rows] == ['Player A', 'Player B', 'Player C']
