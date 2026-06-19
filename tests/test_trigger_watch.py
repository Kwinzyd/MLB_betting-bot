import sqlite3
from datetime import datetime, timezone, timedelta
from unittest.mock import patch, MagicMock, AsyncMock
import pytest


def _seed_schema(conn):
    conn.executescript('''
        CREATE TABLE games (
            game_id TEXT PRIMARY KEY, home_team TEXT, away_team TEXT,
            venue TEXT, status TEXT, date TEXT
        );
        CREATE TABLE umpire_stats (
            umpire_id INTEGER PRIMARY KEY,
            umpire_name TEXT NOT NULL,
            games_called INTEGER NOT NULL DEFAULT 0,
            total_strikeouts INTEGER NOT NULL DEFAULT 0,
            total_walks INTEGER NOT NULL DEFAULT 0,
            k_per_game REAL,
            bb_per_game REAL,
            k_factor REAL NOT NULL DEFAULT 1.0,
            updated_date TEXT
        );
        CREATE TABLE umpire_game_assignments (
            mlb_game_pk INTEGER PRIMARY KEY,
            game_id TEXT,
            umpire_id INTEGER NOT NULL,
            umpire_name TEXT NOT NULL,
            date TEXT NOT NULL
        );
        CREATE TABLE trigger_events (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            game_id TEXT NOT NULL,
            trigger_type TEXT NOT NULL,
            detail TEXT,
            triggered_at TEXT NOT NULL,
            UNIQUE(game_id, trigger_type, triggered_at)
        );
        CREATE TABLE probable_pitchers (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            game_id TEXT, team TEXT, player_name TEXT,
            player_id INTEGER, throws TEXT, date TEXT,
            UNIQUE(game_id, team)
        );
        CREATE TABLE game_totals_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            game_id TEXT NOT NULL,
            total REAL NOT NULL,
            source TEXT NOT NULL,
            timestamp TEXT NOT NULL
        );
    ''')


@pytest.fixture
def memory_db():
    conn = sqlite3.connect(':memory:')
    conn.row_factory = sqlite3.Row
    _seed_schema(conn)
    return conn


def _add_game(conn, game_id='g1', venue='Wrigley Field'):
    conn.execute(
        "INSERT INTO games VALUES (?, 'Cubs', 'Reds', ?, 'SCHEDULED', '2026-04-15')",
        (game_id, venue),
    )


def _add_umpire(conn, game_id, k_factor, games_called, ump_id=1, name='Angel Hernandez'):
    conn.execute(
        "INSERT INTO umpire_stats "
        "(umpire_id, umpire_name, games_called, k_factor, updated_date) "
        "VALUES (?, ?, ?, ?, '2026-04-14')",
        (ump_id, name, games_called, k_factor),
    )
    conn.execute(
        "INSERT INTO umpire_game_assignments "
        "(mlb_game_pk, game_id, umpire_id, umpire_name, date) "
        "VALUES (?, ?, ?, ?, '2026-04-15')",
        (1000 + ump_id, game_id, ump_id, name),
    )


class _DBCtx:
    """Mimics get_db_connection()'s context manager yielding a shared in-memory conn."""
    def __init__(self, conn):
        self._conn = conn

    def __call__(self):
        return self

    def __enter__(self):
        return self._conn

    def __exit__(self, *a):
        return False


def _patch_common(memory_db, weather_return=None):
    """Build the standard patch set. Returns a stack to enter via `with`."""
    db_ctx = _DBCtx(memory_db)
    weather_client = MagicMock()
    weather_client.get_game_weather.return_value = weather_return
    telegram_instance = MagicMock()
    telegram_instance.send_message = AsyncMock()
    odds_instance = MagicMock()
    odds_instance.get_event_odds = AsyncMock(return_value=None)
    return {
        'db': patch('src.pipelines.trigger_watch.get_db_connection', db_ctx),
        'weather_cls': patch('src.pipelines.trigger_watch.WeatherClient',
                             return_value=weather_client),
        'telegram_cls': patch('src.pipelines.trigger_watch.TelegramClient',
                              return_value=telegram_instance),
        'odds_cls': patch('src.pipelines.trigger_watch.OddsAPIClient',
                          return_value=odds_instance),
        'scan_props': patch('src.pipelines.trigger_watch.scan_props',
                            new_callable=AsyncMock),
        'send_alerts': patch('src.pipelines.trigger_watch.send_alerts',
                             new_callable=AsyncMock),
    }


async def test_umpire_extreme_fires_when_k_factor_above_threshold(memory_db):
    _add_game(memory_db, venue='Tropicana Field')  # dome — no weather trigger
    _add_umpire(memory_db, 'g1', k_factor=1.15, games_called=50)
    memory_db.commit()

    patches = _patch_common(memory_db, weather_return=None)
    with patches['db'], patches['weather_cls'], patches['telegram_cls'], patches['odds_cls'], \
         patches['scan_props'] as mock_scan, patches['send_alerts'] as mock_alerts:
        from src.pipelines.trigger_watch import run_trigger_watch
        await run_trigger_watch()

    rows = memory_db.execute(
        "SELECT game_id, trigger_type FROM trigger_events"
    ).fetchall()
    assert len(rows) == 1
    assert rows[0]['game_id'] == 'g1'
    assert rows[0]['trigger_type'] == 'umpire'
    mock_scan.assert_called_once_with(force=True, game_ids=['g1'])
    mock_alerts.assert_called_once()


async def test_umpire_no_fire_when_games_called_below_min(memory_db):
    _add_game(memory_db, venue='Tropicana Field')
    _add_umpire(memory_db, 'g1', k_factor=1.50, games_called=2)  # sample too small
    memory_db.commit()

    patches = _patch_common(memory_db, weather_return=None)
    with patches['db'], patches['weather_cls'], patches['telegram_cls'], patches['odds_cls'], \
         patches['scan_props'] as mock_scan, patches['send_alerts']:
        from src.pipelines.trigger_watch import run_trigger_watch
        await run_trigger_watch()

    assert memory_db.execute("SELECT COUNT(*) FROM trigger_events").fetchone()[0] == 0
    mock_scan.assert_not_called()


async def test_umpire_no_fire_when_deviation_below_threshold(memory_db):
    _add_game(memory_db, venue='Tropicana Field')
    _add_umpire(memory_db, 'g1', k_factor=1.05, games_called=50)  # within band
    memory_db.commit()

    patches = _patch_common(memory_db, weather_return=None)
    with patches['db'], patches['weather_cls'], patches['telegram_cls'], patches['odds_cls'], \
         patches['scan_props'] as mock_scan, patches['send_alerts']:
        from src.pipelines.trigger_watch import run_trigger_watch
        await run_trigger_watch()

    assert memory_db.execute("SELECT COUNT(*) FROM trigger_events").fetchone()[0] == 0
    mock_scan.assert_not_called()


async def test_weather_extreme_fires_on_hr_adjust(memory_db):
    _add_game(memory_db, venue='Wrigley Field')
    memory_db.commit()

    # Wrigley outfield_bearing=200, sensitivity=0.9. Wind from 20° blows straight
    # out toward CF → angle_diff=180 → wind_direction_factor=+1.0. 18mph speed
    # × 0.9 sensitivity × 0.12 HR coeff ≈ +9.7% HR — well above 5% threshold.
    weather = {'temp_f': 72, 'wind_mph': 18, 'wind_deg': 20, 'humidity': 50}

    patches = _patch_common(memory_db, weather_return=weather)
    with patches['db'], patches['weather_cls'], patches['telegram_cls'], patches['odds_cls'], \
         patches['scan_props'] as mock_scan, patches['send_alerts']:
        from src.pipelines.trigger_watch import run_trigger_watch
        await run_trigger_watch()

    rows = memory_db.execute(
        "SELECT trigger_type FROM trigger_events"
    ).fetchall()
    assert len(rows) == 1
    assert rows[0]['trigger_type'] == 'weather'
    mock_scan.assert_called_once_with(force=True, game_ids=['g1'])


async def test_weather_no_fire_when_dome(memory_db):
    _add_game(memory_db, venue='Tropicana Field')
    memory_db.commit()

    # Even a hurricane at Tropicana shouldn't trigger — it's a dome. The client
    # would return None in practice (we short-circuit on dome), but we belt-and-
    # suspenders it with a severe value to prove the dome guard holds.
    weather = {'temp_f': 72, 'wind_mph': 30, 'wind_deg': 180, 'humidity': 50}

    patches = _patch_common(memory_db, weather_return=weather)
    with patches['db'], patches['weather_cls'], patches['telegram_cls'], patches['odds_cls'], \
         patches['scan_props'] as mock_scan, patches['send_alerts']:
        from src.pipelines.trigger_watch import run_trigger_watch
        await run_trigger_watch()

    assert memory_db.execute("SELECT COUNT(*) FROM trigger_events").fetchone()[0] == 0
    mock_scan.assert_not_called()


async def test_dedup_prevents_second_fire_within_window(memory_db):
    _add_game(memory_db, venue='Tropicana Field')
    _add_umpire(memory_db, 'g1', k_factor=1.15, games_called=50)
    # Seed a prior trigger 1h ago — inside default 6h dedup window.
    prior_ts = (datetime.now(timezone.utc) - timedelta(hours=1)).isoformat()
    memory_db.execute(
        "INSERT INTO trigger_events (game_id, trigger_type, detail, triggered_at) "
        "VALUES ('g1', 'umpire', '{}', ?)",
        (prior_ts,),
    )
    memory_db.commit()

    patches = _patch_common(memory_db, weather_return=None)
    with patches['db'], patches['weather_cls'], patches['telegram_cls'], patches['odds_cls'], \
         patches['scan_props'] as mock_scan, patches['send_alerts']:
        from src.pipelines.trigger_watch import run_trigger_watch
        await run_trigger_watch()

    # Still just the one seed row; no new fire.
    assert memory_db.execute("SELECT COUNT(*) FROM trigger_events").fetchone()[0] == 1
    mock_scan.assert_not_called()


async def test_dedup_allows_second_fire_after_window(memory_db):
    _add_game(memory_db, venue='Tropicana Field')
    _add_umpire(memory_db, 'g1', k_factor=1.15, games_called=50)
    # Seed a prior trigger 12h ago — outside default 6h dedup window.
    prior_ts = (datetime.now(timezone.utc) - timedelta(hours=12)).isoformat()
    memory_db.execute(
        "INSERT INTO trigger_events (game_id, trigger_type, detail, triggered_at) "
        "VALUES ('g1', 'umpire', '{}', ?)",
        (prior_ts,),
    )
    memory_db.commit()

    patches = _patch_common(memory_db, weather_return=None)
    with patches['db'], patches['weather_cls'], patches['telegram_cls'], patches['odds_cls'], \
         patches['scan_props'] as mock_scan, patches['send_alerts']:
        from src.pipelines.trigger_watch import run_trigger_watch
        await run_trigger_watch()

    # Both the seed row and a new fire.
    assert memory_db.execute("SELECT COUNT(*) FROM trigger_events").fetchone()[0] == 2
    mock_scan.assert_called_once_with(force=True, game_ids=['g1'])


async def test_no_active_games_exits_cleanly(memory_db):
    memory_db.commit()

    patches = _patch_common(memory_db, weather_return=None)
    with patches['db'], patches['weather_cls'], patches['telegram_cls'], patches['odds_cls'], \
         patches['scan_props'] as mock_scan, patches['send_alerts']:
        from src.pipelines.trigger_watch import run_trigger_watch
        await run_trigger_watch()

    mock_scan.assert_not_called()
