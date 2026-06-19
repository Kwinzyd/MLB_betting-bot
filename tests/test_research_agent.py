import json
from unittest.mock import patch

import pytest

from tests.fixtures.fixture_db import memory_conn
from src.pipelines import research_agent as ra


@pytest.fixture
def db():
    conn = memory_conn()
    conn.execute("INSERT INTO teams (team_id, abbreviation, name) VALUES (1,'NYY','New York Yankees')")
    conn.execute("INSERT INTO players (player_id, name, team_id, position, bats, throws) "
                 "VALUES (101,'Gerrit Cole',1,'P','R','R')")
    conn.executemany(
        "INSERT INTO pitcher_game_logs (game_id, player_id, date, innings_pitched, strikeouts, earned_runs) "
        "VALUES (?,?,?,?,?,?)",
        [(i, 101, f'2026-05-{10+i:02d}', 6.0, 7 + (i % 3), 2) for i in range(1, 6)],
    )
    conn.execute("INSERT INTO injury_reports (date, player_name, status) "
                 "VALUES ('2026-06-01','Gerrit Cole','Day-To-Day')")
    conn.commit()
    return conn


# --- Tool implementations (pure DB) ---

def test_find_player(db):
    out = ra._find_player(db, 'Cole')
    assert isinstance(out, list) and out[0]['player_id'] == 101
    assert out[0]['team'] == 'NYY'


def test_find_player_no_match(db):
    out = ra._find_player(db, 'Nobody')
    assert 'note' in out


def test_player_recent_stats(db):
    out = ra._player_recent_stats(db, 101, 'strikeouts', last_n=5)
    assert out['summary']['n'] == 5
    assert out['summary']['max'] >= out['summary']['min']
    assert len(out['games']) == 5


def test_player_recent_stats_unknown_stat(db):
    assert 'error' in ra._player_recent_stats(db, 101, 'bogus')


def test_player_injury_status(db):
    out = ra._player_injury_status(db, 'Cole')
    assert out['status'] == 'Day-To-Day'


def test_player_injury_status_healthy(db):
    out = ra._player_injury_status(db, 'Aaron Judge')
    assert 'healthy' in out['status'].lower()


# --- Agent loop with a stubbed LLM ---

class _StubClient:
    available = True

    def __init__(self, responses):
        self._responses = list(responses)

    async def chat(self, messages, **kwargs):
        return self._responses.pop(0)


def _tool_call_resp(name, args):
    return {"choices": [{"message": {
        "role": "assistant", "content": None,
        "tool_calls": [{"id": "c1", "type": "function",
                        "function": {"name": name, "arguments": json.dumps(args)}}],
    }}]}


def _answer_resp(text):
    return {"choices": [{"message": {"role": "assistant", "content": text}}]}


async def test_agent_runs_tool_then_answers(db):
    client = _StubClient([
        _tool_call_resp("player_recent_stats", {"player_id": 101, "stat": "strikeouts", "last_n": 5}),
        _answer_resp("Cole has averaged about 8 strikeouts over his last 5 starts."),
    ])
    with patch('src.pipelines.research_agent.get_db_connection') as mock_db:
        mock_db.return_value.__enter__.return_value = db
        mock_db.return_value.__exit__.return_value = None
        answer = await ra.ask("How many Ks has Cole had recently?", client=client)
    assert "8 strikeouts" in answer


async def test_agent_failsafe_when_unavailable():
    class _Off:
        available = False
    answer = await ra.ask("anything", client=_Off())
    assert "not configured" in answer.lower()


async def test_agent_failsafe_when_chat_returns_none():
    client = _StubClient([None])
    answer = await ra.ask("anything", client=client)
    assert "unavailable" in answer.lower()
