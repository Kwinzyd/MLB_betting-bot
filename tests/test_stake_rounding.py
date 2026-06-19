from unittest.mock import patch, AsyncMock

import pytest

from src.utils.stake_rounding import round_stake
from tests.fixtures.fixture_db import memory_conn


class TestRoundStake:
    def test_rounds_to_nearest_increment(self):
        with patch('src.utils.stake_rounding.STAKE_ROUNDING_INCREMENT', 5.0):
            assert round_stake(18.42) == 20.0
            assert round_stake(12.20) == 10.0

    def test_floors_to_one_increment_when_below(self):
        with patch('src.utils.stake_rounding.STAKE_ROUNDING_INCREMENT', 5.0):
            # 2.50 rounds to 0 with banker's rounding then floors up to inc.
            assert round_stake(2.50) == 5.0
            assert round_stake(0.10) == 5.0

    def test_passthrough_when_increment_zero(self):
        with patch('src.utils.stake_rounding.STAKE_ROUNDING_INCREMENT', 0.0):
            assert round_stake(18.42) == 18.42

    def test_passthrough_for_non_positive(self):
        with patch('src.utils.stake_rounding.STAKE_ROUNDING_INCREMENT', 5.0):
            assert round_stake(0.0) == 0.0
            assert round_stake(-1.0) == -1.0

    def test_respects_explicit_override(self):
        with patch('src.utils.stake_rounding.STAKE_ROUNDING_INCREMENT', 5.0):
            assert round_stake(18.42, increment=10.0) == 20.0
            assert round_stake(13.0, increment=10.0) == 10.0


# --- Integration: send_alerts persists rounded stake -------------------------


@pytest.fixture
def alerts_db():
    """Seed a fresh bet candidate with a known raw Kelly stake of $18.42."""
    from tests.test_send_alerts import seed_candidate
    conn = memory_conn()
    seed_candidate(conn, stake=18.42)
    conn.commit()
    return conn


# Neutralize portfolio Kelly so the candidate's persisted stake survives to
# the rounding step — this test isolates stake rounding, not portfolio sizing.
@patch('src.pipelines.send_alerts.apply_portfolio_kelly',
       new=lambda candidates, bankroll: candidates)
@patch('src.pipelines.send_alerts.BETTING_ENABLED', True)
@patch('src.clients.execution.telegram_venue.BETTING_ENABLED', True)
@patch('src.utils.stake_rounding.STAKE_ROUNDING_INCREMENT', 5.0)
@patch('src.pipelines.send_alerts.get_db_connection')
@patch('src.pipelines.send_alerts.build_venue_registry')
async def test_send_alerts_persists_rounded_stake(
    mock_registry, mock_get_db, alerts_db
):
    from src.clients.execution.telegram_venue import TelegramVenue
    tg = TelegramVenue()
    tg._client.send_message = AsyncMock(return_value=True)
    mock_registry.return_value = [tg]
    mock_bot = tg._client
    mock_get_db.return_value.__enter__.return_value = alerts_db
    mock_get_db.return_value.__exit__.return_value = None

    from src.pipelines.send_alerts import send_alerts
    await send_alerts()

    mock_bot.send_message.assert_called_once()
    msg = mock_bot.send_message.call_args[0][0]
    assert "$20.00" in msg

    row = alerts_db.execute("SELECT kelly_stake FROM alerts_sent").fetchone()
    assert row['kelly_stake'] == 20.0


@patch('src.pipelines.send_alerts.apply_portfolio_kelly',
       new=lambda candidates, bankroll: candidates)
@patch('src.pipelines.send_alerts.BETTING_ENABLED', True)
@patch('src.clients.execution.telegram_venue.BETTING_ENABLED', True)
@patch('src.utils.stake_rounding.STAKE_ROUNDING_INCREMENT', 0.0)
@patch('src.pipelines.send_alerts.get_db_connection')
@patch('src.pipelines.send_alerts.build_venue_registry')
async def test_send_alerts_rounding_disabled_passes_through(
    mock_registry, mock_get_db, alerts_db
):
    from src.clients.execution.telegram_venue import TelegramVenue
    tg = TelegramVenue()
    tg._client.send_message = AsyncMock(return_value=True)
    mock_registry.return_value = [tg]
    mock_get_db.return_value.__enter__.return_value = alerts_db
    mock_get_db.return_value.__exit__.return_value = None

    from src.pipelines.send_alerts import send_alerts
    await send_alerts()

    row = alerts_db.execute("SELECT kelly_stake FROM alerts_sent").fetchone()
    assert row['kelly_stake'] == 18.42


# --- Integration: SGP exposure cap returns rounded stake ---------------------


@pytest.fixture
def sgp_db():
    return memory_conn()


def _sgp_candidate(stake):
    return {
        'game_id': 'g1',
        'matchup': 'Red Sox @ Yankees',
        'legs': [],
        'joint_prob': 0.30,
        'naive_parlay_odds': 6.85,
        'fair_odds': 3.33,
        'edge_vs_naive': 1.05,
        'kelly_stake': stake,
        'kelly_pct': 0.018,
    }


@patch('src.utils.stake_rounding.STAKE_ROUNDING_INCREMENT', 5.0)
@patch('src.pipelines.find_sgp.MAX_BETS_PER_GAME', 3)
@patch('src.pipelines.find_sgp.get_current_bankroll', return_value=1000.0)
@patch('src.pipelines.find_sgp.get_db_connection')
def test_sgp_exposure_cap_returns_rounded_stake(mock_get_db, mock_bank, sgp_db):
    mock_get_db.return_value.__enter__.return_value = sgp_db
    mock_get_db.return_value.__exit__.return_value = None

    from src.pipelines.find_sgp import _apply_game_exposure_cap
    kept = _apply_game_exposure_cap([_sgp_candidate(stake=18.42)])
    assert len(kept) == 1
    assert kept[0]['kelly_stake'] == 20.0


@patch('src.utils.stake_rounding.STAKE_ROUNDING_INCREMENT', 0.0)
@patch('src.pipelines.find_sgp.MAX_BETS_PER_GAME', 3)
@patch('src.pipelines.find_sgp.get_current_bankroll', return_value=1000.0)
@patch('src.pipelines.find_sgp.get_db_connection')
def test_sgp_exposure_cap_rounding_disabled(mock_get_db, mock_bank, sgp_db):
    mock_get_db.return_value.__enter__.return_value = sgp_db
    mock_get_db.return_value.__exit__.return_value = None

    from src.pipelines.find_sgp import _apply_game_exposure_cap
    kept = _apply_game_exposure_cap([_sgp_candidate(stake=18.42)])
    assert len(kept) == 1
    assert kept[0]['kelly_stake'] == 18.42
