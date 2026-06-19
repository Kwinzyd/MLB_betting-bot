from unittest.mock import patch

import pytest

from tests.fixtures.fixture_db import memory_conn
from src.utils.time_utils import get_eastern_local_date
from src.pipelines import enrich_injuries as ei


TODAY = str(get_eastern_local_date())


class _StubClient:
    def __init__(self, signal, available=True):
        self._signal = signal
        self.available = available

    async def complete_json(self, system, user, **kwargs):
        return self._signal


@pytest.fixture
def db():
    conn = memory_conn()
    conn.execute("INSERT INTO players (player_id, name) VALUES (101, 'Gerrit Cole')")
    conn.execute(
        "INSERT INTO injury_reports (date, player_name, status, injury_type) "
        "VALUES (?, 'Gerrit Cole', 'Day-To-Day', 'elbow')", (TODAY,))
    conn.commit()
    return conn


async def test_enrich_writes_signal(db):
    client = _StubClient({"play_status": "questionable", "play_probability": 0.4,
                          "impact_summary": "elbow, game-time decision"})
    with patch('src.pipelines.enrich_injuries.get_db_connection') as mock_db:
        mock_db.return_value.__enter__.return_value = db
        mock_db.return_value.__exit__.return_value = None
        out = await ei.enrich_injuries(client=client)
    assert out["enriched"] == 1
    row = db.execute("SELECT * FROM player_injury_signals WHERE player_id=101").fetchone()
    assert row["play_status"] == "questionable"
    assert row["play_probability"] == pytest.approx(0.4)
    assert "elbow" in row["impact_summary"]


async def test_enrich_failsafe_when_llm_off(db):
    client = _StubClient(None, available=False)
    out = await ei.enrich_injuries(client=client)
    assert out.get("skipped") == "llm_disabled"
    assert db.execute("SELECT COUNT(*) FROM player_injury_signals").fetchone()[0] == 0


async def test_enrich_failsafe_on_unparseable_signal(db):
    client = _StubClient(None)  # complete_json returned None (unparseable)
    with patch('src.pipelines.enrich_injuries.get_db_connection') as mock_db:
        mock_db.return_value.__enter__.return_value = db
        mock_db.return_value.__exit__.return_value = None
        out = await ei.enrich_injuries(client=client)
    assert out["enriched"] == 0
    assert db.execute("SELECT COUNT(*) FROM player_injury_signals").fetchone()[0] == 0


def test_get_injury_signal_roundtrip(db):
    db.execute(
        "INSERT INTO player_injury_signals "
        "(player_id, player_name, date, play_status, play_probability, impact_summary, created_at) "
        "VALUES (101, 'Gerrit Cole', ?, 'out', 0.05, 'shut down', 'now')", (TODAY,))
    db.commit()
    sig = ei.get_injury_signal(db, 101, TODAY)
    assert sig["play_status"] == "out"
    assert ei.get_injury_signal(db, 999, TODAY) is None


# --- The conservative gate inside scan_props._build_projection ---

def _gate_db():
    conn = memory_conn()
    conn.execute("INSERT INTO players (player_id, name, team_id, throws) VALUES (101,'Gerrit Cole',1,'R')")
    conn.execute("INSERT INTO teams (team_id, abbreviation, name) VALUES (1,'NYY','New York Yankees')")
    conn.execute(
        "INSERT INTO injury_reports (date, player_name, status) VALUES (?, 'Gerrit Cole', 'Day-To-Day')",
        (TODAY,))
    conn.commit()
    return conn


def test_low_play_probability_signal_skips_projection():
    """A non-clear status plus a low LLM play-probability gates the prop out."""
    conn = _gate_db()
    conn.execute(
        "INSERT INTO player_injury_signals "
        "(player_id, player_name, date, play_status, play_probability, impact_summary, created_at) "
        "VALUES (101,'Gerrit Cole',?, 'out', 0.05, 'shut down for the night', 'now')", (TODAY,))
    conn.commit()

    from src.pipelines.scan_props import _build_projection
    from src.models.projections import ProjectionModel
    with patch('src.pipelines.scan_props.get_db_connection') as mock_db:
        mock_db.return_value.__enter__.return_value = conn
        mock_db.return_value.__exit__.return_value = None
        proj = _build_projection(ProjectionModel(), 'Gerrit Cole', 'pitcher_strikeouts',
                                 6.5, 'g1', 'New York Yankees', 'Boston Red Sox', 'venue')
    assert proj is None


def test_healthy_player_not_gated_despite_pessimistic_signal():
    """The gate only tightens: a Healthy player still projects even if a stray
    pessimistic LLM signal exists (it's never consulted for Healthy status)."""
    conn = _gate_db()
    conn.execute("UPDATE injury_reports SET status='Healthy' WHERE player_name='Gerrit Cole'")
    conn.execute("INSERT INTO teams (team_id, abbreviation, name) VALUES (2,'BOS','Boston Red Sox')")
    # Confirmed-starter gate requires a probable_pitchers row to reach projection.
    conn.execute(
        "INSERT INTO probable_pitchers (game_id, team, player_name, player_id, throws, date) "
        "VALUES ('g1','New York Yankees','Gerrit Cole',101,'R',?)",
        (TODAY,),
    )
    # Seed enough pitcher logs for a real projection.
    conn.executemany(
        "INSERT INTO pitcher_game_logs (game_id, player_id, date, innings_pitched, "
        "strikeouts, earned_runs, pitches_thrown) VALUES (?,?,?,?,?,?,?)",
        [(i, 101, f'2026-05-{10+i:02d}', 6.0, 8, 2, 95) for i in range(1, 6)],
    )
    # A pessimistic signal that MUST be ignored because status is Healthy.
    conn.execute(
        "INSERT INTO player_injury_signals "
        "(player_id, player_name, date, play_status, play_probability, impact_summary, created_at) "
        "VALUES (101,'Gerrit Cole',?, 'out', 0.01, 'x', 'now')", (TODAY,))
    conn.commit()

    from src.pipelines.scan_props import _build_projection
    from src.models.projections import ProjectionModel
    with patch('src.pipelines.scan_props.get_db_connection') as mock_db:
        mock_db.return_value.__enter__.return_value = conn
        mock_db.return_value.__exit__.return_value = None
        proj = _build_projection(ProjectionModel(models_dir='__none__'), 'Gerrit Cole',
                                 'pitcher_strikeouts', 6.5, 'g1',
                                 'New York Yankees', 'Boston Red Sox', 'venue')
    assert proj is not None  # gate did not fire for a Healthy player
    assert proj['injury_status'] == 'Healthy'
