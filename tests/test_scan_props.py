import pytest
import json
from unittest.mock import patch, AsyncMock

from tests.fixtures.fixture_db import memory_conn


@pytest.fixture
def memory_db():
    """In-memory DB on the full production schema (so every table scan_props
    writes — snapshots, projections, bet_candidates — exists), seeded with one
    active game."""
    conn = memory_conn()
    conn.execute('''
        INSERT INTO games (game_id, home_team, away_team, venue, status)
        VALUES ('game_123', 'Yankees', 'Red Sox', 'Yankee Stadium', 'SCHEDULED')
    ''')
    conn.commit()
    return conn


@patch('src.pipelines.scan_props.get_db_connection')
@patch('src.pipelines.scan_props.OddsAPIClient')
@patch('src.pipelines.scan_props.WeatherClient')
@patch('src.pipelines.scan_props._build_projection')
async def test_scan_props_database_inserts(
    mock_build_projection, mock_weather_client, mock_odds_client,
    mock_get_db, memory_db
):
    """Test that scan_props correctly saves snapshots and playable projections to the DB."""

    # 1. Setup database mock to return our in-memory DB context manager
    mock_get_db.return_value.__enter__.return_value = memory_db
    mock_get_db.return_value.__exit__.return_value = None

    # 2. Mock Odds API response — one sharp book (pinnacle) + one soft book (draftkings).
    # Pinnacle's no-vig devigs to ~60/40, matching the model's prob_over=0.60 so the
    # sharp/model agreement gate passes. Draftkings offers 2.0 on the Over, creating
    # a 10pp edge vs. the sharp 60% truth.
    sharp_outcomes = [
        {'name': 'Over',  'description': 'Gerrit Cole', 'point': 6.5, 'price': 1.67},
        {'name': 'Under', 'description': 'Gerrit Cole', 'point': 6.5, 'price': 2.50},
    ]
    soft_outcomes = [
        {'name': 'Over',  'description': 'Gerrit Cole', 'point': 6.5, 'price': 2.0},
        {'name': 'Under', 'description': 'Gerrit Cole', 'point': 6.5, 'price': 1.8},
    ]
    mock_odds_instance = mock_odds_client.return_value
    mock_odds_instance.get_event_odds = AsyncMock(return_value={
        'bookmakers': [
            {'key': 'pinnacle',   'markets': [{'key': 'pitcher_strikeouts', 'outcomes': sharp_outcomes}]},
            {'key': 'draftkings', 'markets': [{'key': 'pitcher_strikeouts', 'outcomes': soft_outcomes}]},
        ]
    })

    # 3. Mock the projection model to return a profitable edge on the OVER.
    mock_build_projection.return_value = {
        'projected_mean': 7.5,
        'prob_over': 0.60,
        'prob_under': 0.40,
        'injury_status': 'Healthy',
        'sample_size': 20,
        'context': {'mock_data': True}
    }

    # 4. Run the pipeline (force=True bypasses the quota gate)
    from src.pipelines.scan_props import scan_props
    await scan_props(force=True)

    # 5. Assert the data was saved successfully to the in-memory database.
    # We write one snapshot per bookmaker (sharp + soft); inspect the soft book.
    snapshots = memory_db.execute(
        "SELECT * FROM prop_snapshots WHERE bookmaker = 'draftkings'"
    ).fetchall()
    assert len(snapshots) > 0

    snapshot = dict(snapshots[0])
    assert snapshot['game_id'] == 'game_123'
    assert snapshot['player_name'] == 'Gerrit Cole'
    assert snapshot['market'] == 'pitcher_strikeouts'
    assert snapshot['bookmaker'] == 'draftkings'
    assert snapshot['line'] == 6.5
    assert snapshot['over_odds'] == 2.0

    projections = memory_db.execute("SELECT * FROM projections").fetchall()
    assert len(projections) == 1
    
    proj = dict(projections[0])
    assert proj['game_id'] == 'game_123'
    assert proj['player_name'] == 'Gerrit Cole'
    assert proj['market'] == 'pitcher_strikeouts'
    assert proj['prob_over'] == 0.60
    # scan_props merges the projection's sample_size into context_json so the
    # downstream rank_edge MIN_SAMPLE_SIZE gate can read it.
    assert json.loads(proj['context_json']) == {'mock_data': True, 'sample_size': 20}

    # The winning bet is persisted to bet_candidates — this is the handoff
    # send_alerts consumes (it never re-derives bets from snapshots).
    cands = memory_db.execute("SELECT * FROM bet_candidates").fetchall()
    assert len(cands) == 1
    cand = dict(cands[0])
    assert cand['game_id'] == 'game_123'
    assert cand['player_name'] == 'Gerrit Cole'
    assert cand['market'] == 'pitcher_strikeouts'
    assert cand['side'] == 'over'
    assert cand['bookmaker'] == 'draftkings'
    assert cand['odds'] == 2.0
    assert cand['sharp_book'] == 'pinnacle'
    assert cand['anchor_line'] == 6.5
    # Truth = sharp devig at the anchor (1.67/2.50 -> ~0.599 over)
    assert cand['truth_prob'] == pytest.approx(0.599, abs=0.005)
    # CLV open basis: draftkings' own two-sided devig (2.0/1.8 -> ~0.474 over)
    assert cand['open_devig_prob'] == pytest.approx(0.4737, abs=0.005)
    assert cand['recommended_stake'] > 0