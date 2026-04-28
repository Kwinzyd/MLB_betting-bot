from datetime import datetime, timezone, timedelta
from src.pipelines.scan_props import _should_scan_game


def _game(game_time=None, lineups_confirmed_at=None, last_scanned_at=None):
    """sqlite3.Row-like dict shim; _should_scan_game just indexes by key."""
    return {
        'game_time': game_time,
        'lineups_confirmed_at': lineups_confirmed_at,
        'last_scanned_at': last_scanned_at,
    }


def test_skips_when_no_lineups_and_far_from_game_time():
    now = datetime(2026, 4, 13, 18, 0, tzinfo=timezone.utc)
    g = _game(game_time='2026-04-13T23:00:00+00:00')  # 5h away
    assert _should_scan_game(g, now)[0] == "skip"


def test_scans_when_lineup_confirmed_and_never_scanned():
    now = datetime(2026, 4, 13, 18, 0, tzinfo=timezone.utc)
    g = _game(
        game_time='2026-04-13T23:00:00+00:00',
        lineups_confirmed_at='2026-04-13T17:55:00+00:00',
    )
    assert _should_scan_game(g, now)[0] == "scan"


def test_skips_when_already_scanned_since_lineup_drop():
    now = datetime(2026, 4, 13, 18, 0, tzinfo=timezone.utc)
    g = _game(
        game_time='2026-04-13T23:00:00+00:00',
        lineups_confirmed_at='2026-04-13T17:55:00+00:00',
        last_scanned_at='2026-04-13T17:56:00+00:00',
    )
    assert _should_scan_game(g, now)[0] == "skip"


def test_scans_inside_pregame_window():
    now = datetime(2026, 4, 13, 22, 30, tzinfo=timezone.utc)
    # First pitch in 30 minutes — inside 45-min window
    g = _game(
        game_time='2026-04-13T23:00:00+00:00',
        last_scanned_at='2026-04-13T20:00:00+00:00',
    )
    assert _should_scan_game(g, now)[0] == "scan"


def test_skips_after_first_pitch():
    now = datetime(2026, 4, 13, 23, 10, tzinfo=timezone.utc)
    g = _game(
        game_time='2026-04-13T23:00:00+00:00',
        last_scanned_at='2026-04-13T22:30:00+00:00',
    )
    assert _should_scan_game(g, now)[0] == "skip"


def test_scans_on_new_lineup_drop_after_prior_scan():
    """An updated lineup after a prior scan should trigger a re-scan."""
    now = datetime(2026, 4, 13, 20, 0, tzinfo=timezone.utc)
    g = _game(
        game_time='2026-04-13T23:00:00+00:00',
        lineups_confirmed_at='2026-04-13T19:55:00+00:00',
        last_scanned_at='2026-04-13T18:00:00+00:00',
    )
    assert _should_scan_game(g, now)[0] == "scan"
