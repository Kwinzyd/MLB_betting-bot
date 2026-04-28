import pytest
import sqlite3
from unittest.mock import patch, AsyncMock


@pytest.fixture
def memory_db():
    """In-memory DB seeded with a playable projection and matching prop snapshot."""
    conn = sqlite3.connect(':memory:')
    conn.row_factory = sqlite3.Row
    conn.executescript('''
        CREATE TABLE games (
            game_id TEXT PRIMARY KEY, home_team TEXT, away_team TEXT,
            venue TEXT, status TEXT, date TEXT
        );
        CREATE TABLE projections (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            game_id TEXT, player_name TEXT, market TEXT,
            projected_mean REAL, prob_over REAL, prob_under REAL,
            context_json TEXT, timestamp TEXT,
            UNIQUE(game_id, player_name, market)
        );
        CREATE TABLE prop_snapshots (
            snapshot_id TEXT PRIMARY KEY, game_id TEXT, player_name TEXT,
            market TEXT, line REAL, over_odds REAL, under_odds REAL,
            bookmaker TEXT, timestamp TEXT, devigged_over REAL, devigged_under REAL
        );
        CREATE TABLE alerts_sent (
            alert_id INTEGER PRIMARY KEY AUTOINCREMENT,
            player_name TEXT, market TEXT, line REAL, side TEXT,
            edge REAL, ev REAL, kelly_stake REAL, bookmaker TEXT,
            odds REAL, opening_odds REAL,
            model_prob_over REAL, model_prob_under REAL,
            game_id TEXT, timestamp TEXT,
            UNIQUE(player_name, market, line, bookmaker)
        );
        CREATE TABLE orders (
            order_id INTEGER PRIMARY KEY AUTOINCREMENT,
            venue TEXT NOT NULL,
            alert_id INTEGER,
            player_name TEXT NOT NULL,
            market TEXT NOT NULL,
            line REAL NOT NULL,
            side TEXT NOT NULL,
            game_id TEXT,
            bookmaker TEXT,
            offered_odds REAL NOT NULL,
            fill_odds REAL,
            stake REAL NOT NULL,
            status TEXT NOT NULL DEFAULT 'pending',
            venue_order_id TEXT,
            placed_at TEXT NOT NULL,
            filled_at TEXT,
            notes TEXT
        );
    ''')
    conn.execute('''
        INSERT INTO games VALUES
        ('g1', 'Yankees', 'Red Sox', 'Yankee Stadium', 'SCHEDULED', '2024-05-15')
    ''')
    conn.execute('''
        INSERT INTO projections
        (game_id, player_name, market, projected_mean, prob_over, prob_under, context_json, timestamp)
        VALUES ('g1', 'Gerrit Cole', 'pitcher_strikeouts', 7.5, 0.62, 0.38, '{}', '2024-05-15T12:00:00')
    ''')
    conn.execute('''
        INSERT INTO prop_snapshots VALUES
        ('snap1', 'g1', 'Gerrit Cole', 'pitcher_strikeouts', 6.5, 2.0, 1.8,
         'draftkings', '2024-05-15T12:00:00', 0.60, 0.40)
    ''')
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
    """A playable edge dispatches to Telegram and is recorded in alerts_sent + orders."""
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

    orders = memory_db.execute("SELECT * FROM orders").fetchall()
    assert len(orders) == 1
    assert orders[0]['venue'] == 'telegram'
    assert orders[0]['status'] == 'sent'
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
async def test_send_alerts_no_edge_no_alert(mock_registry, mock_get_db, memory_db):
    """A projection below the edge threshold produces no alert."""
    memory_db.execute(
        "UPDATE projections SET prob_over = 0.51, prob_under = 0.49 WHERE player_name = 'Gerrit Cole'"
    )
    memory_db.commit()

    venues, tg = _build_telegram_only_registry()
    mock_registry.return_value = venues
    mock_get_db.return_value.__enter__.return_value = memory_db
    mock_get_db.return_value.__exit__.return_value = None

    from src.pipelines.send_alerts import send_alerts
    await send_alerts()

    tg._client.send_message.assert_not_called()
    alert = memory_db.execute("SELECT * FROM alerts_sent").fetchone()
    assert alert is None


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
