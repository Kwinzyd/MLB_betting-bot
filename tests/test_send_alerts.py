import pytest
from datetime import timedelta
from unittest.mock import patch, AsyncMock

from src.utils.time_utils import utcnow
from tests.fixtures.fixture_db import memory_conn


def seed_candidate(conn, game_id='g1', player='Gerrit Cole', player_id=101,
                   market='pitcher_strikeouts', line=6.5, side='over',
                   bookmaker='draftkings', odds=2.0, truth_prob=0.60,
                   edge_pct=10.0, created_at=None, stake=25.0):
    """Insert a game + projection-context + bet_candidates row the way
    scan_props persists winners."""
    conn.execute(
        "INSERT OR IGNORE INTO games (game_id, home_team, away_team, venue, status, date) "
        "VALUES (?, 'Yankees', 'Red Sox', 'Yankee Stadium', 'SCHEDULED', '2024-05-15')",
        (game_id,)
    )
    conn.execute('''
        INSERT OR IGNORE INTO projections
        (game_id, player_name, market, projected_mean, prob_over, prob_under, context_json, timestamp)
        VALUES (?, ?, ?, 7.5, ?, ?, '{"sample_size": 20}', '2024-05-15T12:00:00')
    ''', (game_id, player, market, truth_prob, 1.0 - truth_prob))
    conn.execute('''
        INSERT INTO bet_candidates
        (game_id, player_id, player_name, market, line, side, bookmaker, odds,
         sharp_book, anchor_line, truth_prob, model_prob, open_devig_prob,
         edge_pct, ev, kelly_fraction, recommended_stake, steam_detected, created_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, 'pinnacle', ?, ?, ?, ?, ?, ?, 0.02, ?, 0, ?)
    ''', (
        game_id, player_id, player, market, line, side, bookmaker, odds,
        line, truth_prob, truth_prob, truth_prob, edge_pct,
        truth_prob * odds - 1.0, stake,
        created_at or utcnow().isoformat(),
    ))


@pytest.fixture
def memory_db():
    """In-memory DB (full production schema) seeded with one fresh playable
    bet candidate, as persisted by scan_props."""
    conn = memory_conn()
    seed_candidate(conn)
    conn.commit()
    return conn


def _build_telegram_only_registry():
    from src.clients.execution.telegram_venue import TelegramVenue
    venue = TelegramVenue()
    venue._client.send_message = AsyncMock(return_value=True)
    return [venue], venue


def _build_telegram_and_paper_registry():
    from src.clients.execution.telegram_venue import TelegramVenue
    from src.clients.execution.paper_exchange import PaperExchange
    tg = TelegramVenue()
    tg._client.send_message = AsyncMock(return_value=True)
    return [tg, PaperExchange()], tg


@patch('src.pipelines.send_alerts.BETTING_ENABLED', True)
@patch('src.clients.execution.telegram_venue.BETTING_ENABLED', True)
@patch('src.pipelines.send_alerts.get_db_connection')
@patch('src.pipelines.send_alerts.build_venue_registry')
async def test_send_alerts_new_alert_inserted(mock_registry, mock_get_db, memory_db):
    """A fresh candidate dispatches to Telegram and is recorded delivered=1."""
    venues, tg = _build_telegram_only_registry()
    mock_registry.return_value = venues
    mock_get_db.return_value.__enter__.return_value = memory_db
    mock_get_db.return_value.__exit__.return_value = None

    from src.pipelines.send_alerts import send_alerts
    await send_alerts()

    tg._client.send_message.assert_called_once()

    alert = memory_db.execute("SELECT * FROM alerts_sent").fetchone()
    assert alert is not None
    assert alert['player_name'] == 'Gerrit Cole'
    assert alert['side'] == 'over'
    assert alert['bookmaker'] == 'draftkings'
    assert alert['delivered'] == 1
    assert alert['player_id'] == 101
    assert alert['open_devig_prob'] == pytest.approx(0.60)

    orders = memory_db.execute("SELECT * FROM orders").fetchall()
    assert len(orders) == 1
    assert orders[0]['venue'] == 'telegram'
    assert orders[0]['status'] == 'sent'
    assert orders[0]['alert_id'] == alert['alert_id']


@patch('src.pipelines.send_alerts.BETTING_ENABLED', False)
@patch('src.clients.execution.telegram_venue.BETTING_ENABLED', False)
@patch('src.pipelines.send_alerts.get_db_connection')
@patch('src.pipelines.send_alerts.build_venue_registry')
async def test_send_alerts_shadow_mode_records_undelivered(mock_registry, mock_get_db, memory_db):
    """BETTING_ENABLED=false still records the bet (delivered=0) and persists
    the skipped order — the paper-trading window must produce settle/CLV data."""
    venues, tg = _build_telegram_only_registry()
    mock_registry.return_value = venues
    mock_get_db.return_value.__enter__.return_value = memory_db
    mock_get_db.return_value.__exit__.return_value = None

    from src.pipelines.send_alerts import send_alerts
    await send_alerts()

    tg._client.send_message.assert_not_called()  # shadow: no delivery

    alert = memory_db.execute("SELECT * FROM alerts_sent").fetchone()
    assert alert is not None
    assert alert['delivered'] == 0
    assert alert['kelly_stake'] is not None

    orders = memory_db.execute("SELECT * FROM orders").fetchall()
    assert len(orders) == 1
    assert orders[0]['status'] == 'skipped'
    assert orders[0]['alert_id'] == alert['alert_id']


@patch('src.pipelines.send_alerts.get_db_connection')
@patch('src.pipelines.send_alerts.build_venue_registry')
async def test_send_alerts_deduplication(mock_registry, mock_get_db, memory_db):
    """An already-alerted prop is not sent again."""
    memory_db.execute('''
        INSERT INTO alerts_sent
        (player_name, market, line, side, edge, ev, kelly_stake,
         bookmaker, odds, opening_odds, game_id, timestamp)
        VALUES ('Gerrit Cole', 'pitcher_strikeouts', 6.5, 'over', 15.0, 0.30,
                10.0, 'draftkings', 2.0, 2.0, 'g1', '2024-05-15T11:00:00')
    ''')
    memory_db.commit()

    venues, tg = _build_telegram_only_registry()
    mock_registry.return_value = venues
    mock_get_db.return_value.__enter__.return_value = memory_db
    mock_get_db.return_value.__exit__.return_value = None

    from src.pipelines.send_alerts import send_alerts
    await send_alerts()

    tg._client.send_message.assert_not_called()
    rows = memory_db.execute("SELECT * FROM alerts_sent").fetchall()
    assert len(rows) == 1


@patch('src.pipelines.send_alerts.get_db_connection')
@patch('src.pipelines.send_alerts.build_venue_registry')
async def test_send_alerts_stale_candidate_ignored(mock_registry, mock_get_db):
    """Candidates older than the freshness window are never alerted — the
    odds they reference have likely moved."""
    conn = memory_conn()
    stale_ts = (utcnow() - timedelta(hours=2)).isoformat()
    seed_candidate(conn, created_at=stale_ts)
    conn.commit()

    venues, tg = _build_telegram_only_registry()
    mock_registry.return_value = venues
    mock_get_db.return_value.__enter__.return_value = conn
    mock_get_db.return_value.__exit__.return_value = None

    from src.pipelines.send_alerts import send_alerts
    await send_alerts()

    tg._client.send_message.assert_not_called()
    assert conn.execute("SELECT COUNT(*) FROM alerts_sent").fetchone()[0] == 0


@patch('src.pipelines.send_alerts.BETTING_ENABLED', True)
@patch('src.clients.execution.telegram_venue.BETTING_ENABLED', True)
@patch('src.pipelines.send_alerts.get_db_connection')
@patch('src.pipelines.send_alerts.build_venue_registry')
async def test_send_alerts_paper_exchange_records_filled_order(mock_registry, mock_get_db, memory_db):
    """When PaperExchange is registered, a filled order is persisted alongside the Telegram alert."""
    venues, _tg = _build_telegram_and_paper_registry()
    mock_registry.return_value = venues
    mock_get_db.return_value.__enter__.return_value = memory_db
    mock_get_db.return_value.__exit__.return_value = None

    from src.pipelines.send_alerts import send_alerts
    await send_alerts()

    orders = memory_db.execute("SELECT * FROM orders ORDER BY venue").fetchall()
    venues_seen = [o['venue'] for o in orders]
    assert 'telegram' in venues_seen
    assert 'paper_exchange' in venues_seen

    paper = next(o for o in orders if o['venue'] == 'paper_exchange')
    assert paper['status'] == 'filled'
    assert paper['fill_odds'] == 2.0
    assert paper['offered_odds'] == 2.0
    assert paper['venue_order_id'] is not None
    assert paper['venue_order_id'].startswith('paper-')


@patch('src.pipelines.send_alerts.get_db_connection')
@patch('src.pipelines.send_alerts.build_venue_registry')
async def test_send_alerts_no_venues_is_noop(mock_registry, mock_get_db, memory_db):
    """An empty venue registry skips work without touching the DB."""
    mock_registry.return_value = []
    mock_get_db.return_value.__enter__.return_value = memory_db
    mock_get_db.return_value.__exit__.return_value = None

    from src.pipelines.send_alerts import send_alerts
    await send_alerts()

    assert memory_db.execute("SELECT COUNT(*) FROM alerts_sent").fetchone()[0] == 0
    assert memory_db.execute("SELECT COUNT(*) FROM orders").fetchone()[0] == 0
