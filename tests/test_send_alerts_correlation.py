from unittest.mock import patch, AsyncMock
import pytest

from tests.fixtures.fixture_db import memory_conn
from tests.test_send_alerts import seed_candidate


def _build_telegram_registry():
    from src.clients.execution.telegram_venue import TelegramVenue
    venue = TelegramVenue()
    venue._client.send_message = AsyncMock(return_value=True)
    return [venue], venue


@pytest.fixture
def memory_db():
    return memory_conn()


@patch('src.pipelines.send_alerts.BETTING_ENABLED', True)
@patch('src.clients.execution.telegram_venue.BETTING_ENABLED', True)
@patch('src.pipelines.send_alerts.get_db_connection')
@patch('src.pipelines.send_alerts.build_venue_registry')
async def test_one_bet_per_player(mock_registry, mock_get_db, memory_db):
    """Same player with two playable markets → only the higher-edge one fires."""
    seed_candidate(memory_db, game_id='g1', player='Gerrit Cole',
                   market='pitcher_strikeouts', line=7.5,
                   truth_prob=0.65, edge_pct=15.0)
    seed_candidate(memory_db, game_id='g1', player='Gerrit Cole',
                   market='pitcher_earned_runs', line=2.5,
                   truth_prob=0.70, edge_pct=20.0)
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
    for i, (player, prob, edge) in enumerate([
        ('Player A', 0.70, 20.0),
        ('Player B', 0.68, 18.0),
        ('Player C', 0.65, 15.0),
        ('Player D', 0.62, 12.0),
        ('Player E', 0.60, 10.0),
    ]):
        seed_candidate(memory_db, game_id='g1', player=player, player_id=300 + i,
                       market='batter_hits', line=1.5,
                       truth_prob=prob, edge_pct=edge)
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
