from unittest.mock import patch

import pytest

from tests.fixtures.fixture_db import memory_conn


@pytest.fixture
def memory_db():
    """In-memory DB (full production schema) with three settled bets:
    WIN (+CLV), LOSS (-CLV), PUSH (0 CLV). Built via memory_conn() so the
    report's bankroll_snapshots query and every column it selects exist."""
    conn = memory_conn()
    # WIN bet: over 6.5, actual=8, prob_over=0.60, clv=+0.02
    conn.execute('''INSERT INTO alerts_sent
        (player_name, market, line, side, edge, ev, kelly_stake, bookmaker,
         odds, opening_odds, model_prob_over, model_prob_under, game_id, timestamp)
        VALUES ('Cole', 'pitcher_strikeouts', 6.5, 'over', 10.0, 0.2, 25.0,
                'draftkings', 2.0, 2.0, 0.60, 0.40, 'g1', '2026-04-20T12:00:00')''')
    # LOSS bet: over 1.5, actual=1, prob_over=0.62, clv=-0.01
    conn.execute('''INSERT INTO alerts_sent
        (player_name, market, line, side, edge, ev, kelly_stake, bookmaker,
         odds, opening_odds, model_prob_over, model_prob_under, game_id, timestamp)
        VALUES ('Judge', 'batter_hits', 1.5, 'over', 8.0, 0.15, 20.0,
                'draftkings', 2.1, 2.1, 0.62, 0.38, 'g2', '2026-04-20T12:00:00')''')
    # PUSH bet: over 2.5, actual=2.5 — no CLV (sentinel 0.0)
    conn.execute('''INSERT INTO alerts_sent
        (player_name, market, line, side, edge, ev, kelly_stake, bookmaker,
         odds, opening_odds, model_prob_over, model_prob_under, game_id, timestamp)
        VALUES ('Acuna', 'batter_total_bases', 2.5, 'over', 6.0, 0.10, 15.0,
                'draftkings', 2.0, 2.0, 0.58, 0.42, 'g3', '2026-04-20T12:00:00')''')

    conn.execute('INSERT INTO bet_results (alert_id, actual_value, result, profit, closing_odds, clv) '
                 'VALUES (1, 8.0, ?, ?, 1.95, 0.02)', ('WIN', 25.0))
    conn.execute('INSERT INTO bet_results (alert_id, actual_value, result, profit, closing_odds, clv) '
                 'VALUES (2, 1.0, ?, ?, 2.15, -0.01)', ('LOSS', -20.0))
    conn.execute('INSERT INTO bet_results (alert_id, actual_value, result, profit, closing_odds, clv) '
                 'VALUES (3, 2.5, ?, ?, 2.0, 0.0)', ('PUSH', 0.0))
    conn.commit()
    return conn


@patch('src.pipelines.report.get_db_connection')
def test_report_smoke(mock_get_db, memory_db, capsys):
    mock_get_db.return_value.__enter__.return_value = memory_db
    mock_get_db.return_value.__exit__.return_value = None

    from src.pipelines.report import generate_report
    generate_report()

    out = capsys.readouterr().out
    # Header
    assert "LIVE BET REPORT" in out
    assert "Settled bets scored : 3" in out
    # P&L — 1 win of 3 scorable (push counts as bet but not a win)
    assert "33.3%" in out
    # CLV section renders, averaging only the two non-sentinel values → +0.005
    assert "CLV" in out
    assert "+0.0050" in out
    assert "missing sharp close: 1" in out
    # Bankroll section renders
    assert "BANKROLL" in out


@patch('src.pipelines.report.get_db_connection')
def test_report_empty(mock_get_db, memory_db, capsys):
    # Wipe settled bets
    memory_db.execute("DELETE FROM bet_results")
    memory_db.commit()
    mock_get_db.return_value.__enter__.return_value = memory_db
    mock_get_db.return_value.__exit__.return_value = None

    from src.pipelines.report import generate_report
    generate_report()

    out = capsys.readouterr().out
    assert "No settled bets to report yet." in out
    assert "BANKROLL" in out


@patch('src.pipelines.report.get_db_connection')
def test_report_market_filter(mock_get_db, memory_db, capsys):
    mock_get_db.return_value.__enter__.return_value = memory_db
    mock_get_db.return_value.__exit__.return_value = None

    from src.pipelines.report import generate_report
    generate_report(markets=['pitcher_strikeouts'])

    out = capsys.readouterr().out
    assert "Settled bets scored : 1" in out
    assert "pitcher_strikeouts" in out
