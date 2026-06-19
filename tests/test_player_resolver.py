from unittest.mock import patch

import pytest

from tests.fixtures.fixture_db import memory_conn
from src.data.player_resolver import normalize_name, resolve_player_id, cache_resolution
from src.pipelines import reconcile_names as rn


class TestNormalize:
    def test_strips_accents(self):
        assert normalize_name("José Ramírez") == "jose ramirez"

    def test_strips_punctuation_and_suffix(self):
        assert normalize_name("Ronald Acuña Jr.") == "ronald acuna"

    def test_collapses_space_and_case(self):
        assert normalize_name("  Juan   SOTO ") == "juan soto"


@pytest.fixture
def db():
    conn = memory_conn()
    conn.executemany(
        "INSERT INTO players (player_id, name) VALUES (?,?)",
        [(1, "José Ramírez"), (2, "Juan Soto"), (3, "Will Smith"),
         (4, "Will Smith")],  # genuine duplicate name -> ambiguous
    )
    conn.commit()
    return conn


def test_resolve_exact(db):
    pid, method = resolve_player_id(db, "Juan Soto")
    assert pid == 2 and method == "exact"


def test_resolve_normalized_accent(db):
    pid, method = resolve_player_id(db, "Jose Ramirez")  # no accents in query
    assert pid == 1 and method == "normalized"


def test_resolve_ambiguous(db):
    # Two "Will Smith" rows. A non-exact variant skips the exact path and the
    # normalized path finds >1 match -> ambiguous, never a wrong-player guess.
    pid, method = resolve_player_id(db, "will. smith")
    assert method == "ambiguous" and pid is None


def test_resolve_no_match(db):
    pid, method = resolve_player_id(db, "Nobody Here")
    assert pid is None and method == "no_match"


def test_cache_hit_short_circuits(db):
    cache_resolution(db, "Mr. Cached", 2, "llm")
    db.commit()
    pid, method = resolve_player_id(db, "Mr. Cached")
    assert pid == 2 and method == "cache"


def test_cached_miss_is_honored(db):
    cache_resolution(db, "Confirmed Unknown", None, "llm")
    db.commit()
    pid, method = resolve_player_id(db, "Confirmed Unknown")
    assert pid is None and method == "cached_miss"


# --- LLM reconciliation pass ---

class _StubClient:
    def __init__(self, pick, available=True):
        self._pick = pick
        self.available = available

    async def complete_json(self, system, user, **kwargs):
        return {"player_id": self._pick}


async def test_reconcile_caches_llm_pick():
    conn = memory_conn()
    conn.execute("INSERT INTO players (player_id, name) VALUES (2, 'Juan Soto')")
    conn.execute(
        "INSERT INTO prop_snapshots (snapshot_id, game_id, player_name, market, line, timestamp) "
        "VALUES ('s1','g1','J. Soto','batter_hits',1.5, ?)",
        (rn.utcnow().isoformat(),))
    conn.commit()

    with patch('src.pipelines.reconcile_names.get_db_connection') as mock_db:
        mock_db.return_value.__enter__.return_value = conn
        mock_db.return_value.__exit__.return_value = None
        out = await rn.reconcile_names(client=_StubClient(2))

    assert out["llm_resolved"] == 1
    # The cache now resolves 'J. Soto' deterministically.
    pid, method = resolve_player_id(conn, "J. Soto")
    assert pid == 2 and method == "cache"


async def test_reconcile_noop_when_llm_off():
    conn = memory_conn()
    conn.execute("INSERT INTO players (player_id, name) VALUES (2, 'Juan Soto')")
    conn.execute(
        "INSERT INTO prop_snapshots (snapshot_id, game_id, player_name, market, line, timestamp) "
        "VALUES ('s1','g1','J. Soto','batter_hits',1.5, ?)",
        (rn.utcnow().isoformat(),))
    conn.commit()
    with patch('src.pipelines.reconcile_names.get_db_connection') as mock_db:
        mock_db.return_value.__enter__.return_value = conn
        mock_db.return_value.__exit__.return_value = None
        out = await rn.reconcile_names(client=_StubClient(2, available=False))
    assert out["llm_resolved"] == 0
