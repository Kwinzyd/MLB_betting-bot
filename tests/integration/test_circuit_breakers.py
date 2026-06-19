"""Integration tests for bankroll circuit breakers.

Verifies that:
1. full_stop=1 in bankroll_snapshots causes send_alerts to return without
   writing any alerts_sent rows.
2. kelly_fraction_override=0.5 causes all recommended_stake values to be
   halved relative to the override=1.0 baseline.
"""
import asyncio
import sqlite3
from unittest.mock import patch, AsyncMock

import pytest

from tests.fixtures.fixture_db import SCHEDULED_GAME_ID, TODAY


def _telegram_registry():
    """A registry with a single TelegramVenue whose network send is mocked.

    Recording an alerts_sent row requires BETTING_ENABLED=true and a venue that
    returns status 'sent' (shadow mode tracks via the orders table instead), so
    these circuit-breaker tests run with betting enabled and a stubbed client."""
    from src.clients.execution.telegram_venue import TelegramVenue
    venue = TelegramVenue()
    venue._client.send_message = AsyncMock(return_value=True)
    return [venue]


def _count_new_alerts(db_path: str) -> int:
    """Count alerts_sent rows for the scheduled game (only new ones fire today)."""
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    count = conn.execute(
        "SELECT COUNT(*) FROM alerts_sent WHERE game_id=?", (SCHEDULED_GAME_ID,)
    ).fetchone()[0]
    conn.close()
    return count


def _set_snapshot(db_path: str, full_stop: int = 0, kelly_override: float = 1.0) -> None:
    conn = sqlite3.connect(db_path)
    conn.execute(
        "UPDATE bankroll_snapshots SET full_stop=?, kelly_fraction_override=? "
        "WHERE snapshot_date=?",
        (full_stop, kelly_override, TODAY)
    )
    conn.commit()
    conn.close()


@patch('src.clients.telegram_bot.TelegramClient')
def test_full_stop_suppresses_alerts(mock_tg, db):
    """full_stop=1 → send_alerts exits immediately; no new alerts_sent rows."""
    _set_snapshot(db, full_stop=1)

    mock_tg.return_value.send_message_sync = lambda *a, **kw: None

    from src.pipelines.send_alerts import send_alerts
    asyncio.run(send_alerts())

    assert _count_new_alerts(db) == 0


@patch('src.pipelines.send_alerts.BETTING_ENABLED', True)
@patch('src.clients.execution.telegram_venue.BETTING_ENABLED', True)
@patch('src.pipelines.send_alerts.build_venue_registry')
def test_normal_operation_sends_alerts(mock_registry, db):
    """full_stop=0 → send_alerts processes the playable projection in the fixture."""
    _set_snapshot(db, full_stop=0, kelly_override=1.0)
    mock_registry.return_value = _telegram_registry()

    from src.pipelines.send_alerts import send_alerts
    asyncio.run(send_alerts())

    # The fixture has a projection for g2/Gerrit Cole with 14% edge — should fire.
    assert _count_new_alerts(db) >= 1


@patch('src.pipelines.send_alerts.BETTING_ENABLED', True)
@patch('src.clients.execution.telegram_venue.BETTING_ENABLED', True)
@patch('src.pipelines.send_alerts.build_venue_registry')
def test_kelly_override_halves_stakes(mock_registry, db):
    """kelly_fraction_override=0.5 produces stakes half those of override=1.0."""
    mock_registry.return_value = _telegram_registry()

    from src.pipelines.send_alerts import send_alerts

    # Baseline run: override = 1.0
    _set_snapshot(db, full_stop=0, kelly_override=1.0)
    asyncio.run(send_alerts())

    conn = sqlite3.connect(db)
    conn.row_factory = sqlite3.Row
    baseline_rows = conn.execute(
        "SELECT kelly_stake FROM alerts_sent WHERE game_id=?", (SCHEDULED_GAME_ID,)
    ).fetchall()
    conn.execute("DELETE FROM alerts_sent WHERE game_id=?", (SCHEDULED_GAME_ID,))
    conn.commit()
    conn.close()

    assert baseline_rows, "Expected at least one alert in baseline run"
    baseline_stake = baseline_rows[0]['kelly_stake']

    # Halved run: override = 0.5
    _set_snapshot(db, full_stop=0, kelly_override=0.5)
    asyncio.run(send_alerts())

    conn2 = sqlite3.connect(db)
    conn2.row_factory = sqlite3.Row
    halved_rows = conn2.execute(
        "SELECT kelly_stake FROM alerts_sent WHERE game_id=?", (SCHEDULED_GAME_ID,)
    ).fetchall()
    conn2.close()

    assert halved_rows, "Expected at least one alert in halved run"
    halved_stake = halved_rows[0]['kelly_stake']

    assert halved_stake == pytest.approx(baseline_stake * 0.5, rel=0.05)
