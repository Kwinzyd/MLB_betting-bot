import sqlite3

import pytest

from src.pipelines.sync_statcast import (
    _rates_from_split_row,
    _parse_bdl_platoon_splits,
    sync_bdl_platoon_splits,
)


@pytest.fixture
def splits_db():
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    conn.executescript(
        """CREATE TABLE batter_platoon_splits (
               player_id  INTEGER NOT NULL, season INTEGER NOT NULL,
               vs_hand TEXT NOT NULL, market TEXT NOT NULL,
               rate_per_pa REAL, n_pa INTEGER, updated_at TEXT,
               PRIMARY KEY (player_id, season, vs_hand, market)
           );"""
    )
    yield conn
    conn.close()


def test_rates_from_split_row_basic():
    # player 130 vs. Right: AB=234 H=74 2B=17 3B=1 HR=11 BB=34 HBP=1
    row = {"at_bats": 234, "hits": 74, "doubles": 17, "triples": 1,
           "home_runs": 11, "walks": 34, "hit_by_pitch": 1, "strikeouts": 39}
    r = _rates_from_split_row(row)
    pa = 234 + 34 + 1  # 269
    assert r["n_pa"] == pa
    assert r["batter_hits"] == pytest.approx(74 / pa)
    assert r["batter_home_runs"] == pytest.approx(11 / pa)
    # TB = 74 + 17 + 2*1 + 3*11 = 126
    assert r["batter_total_bases"] == pytest.approx(126 / pa)


def test_rates_from_split_row_zero_pa_returns_none():
    assert _rates_from_split_row({"at_bats": 0, "walks": 0, "hit_by_pitch": 0}) is None
    assert _rates_from_split_row({}) is None


def test_parse_bdl_platoon_splits_extracts_both_hands():
    data = {
        "byBreakdown": [
            {"split_name": "Home", "at_bats": 100, "hits": 30},
            {"split_name": "vs. Left", "at_bats": 107, "hits": 26, "doubles": 5,
             "triples": 0, "home_runs": 4, "walks": 13, "hit_by_pitch": 2},
            {"split_name": "vs. Right", "at_bats": 234, "hits": 74, "doubles": 17,
             "triples": 1, "home_runs": 11, "walks": 34, "hit_by_pitch": 1},
        ]
    }
    out = _parse_bdl_platoon_splits(data)
    assert set(out) == {"L", "R"}
    assert out["L"]["n_pa"] == 107 + 13 + 2
    assert out["R"]["batter_hits"] == pytest.approx(74 / (234 + 34 + 1))


def test_parse_bdl_platoon_splits_empty_when_no_breakdown():
    assert _parse_bdl_platoon_splits({}) == {}
    assert _parse_bdl_platoon_splits({"split": [{"split_name": "All Splits"}]}) == {}
    # a pitcher with a handedness row but zero PAs contributes nothing
    assert _parse_bdl_platoon_splits(
        {"byBreakdown": [{"split_name": "vs. Left", "at_bats": 0, "walks": 0}]}
    ) == {}


class _StubClient:
    def __init__(self, per_player):
        self._per_player = per_player  # {player_id: splits_data}
        self.calls = []

    async def get_player_splits(self, player_id, season=None):
        self.calls.append(player_id)
        return self._per_player.get(player_id, {})


async def test_sync_upserts_and_overwrites_proxy(splits_db):
    memory_db = splits_db
    # Seed a log-derived proxy row that BDL should overwrite.
    memory_db.execute(
        """INSERT INTO batter_platoon_splits
           (player_id, season, vs_hand, market, rate_per_pa, n_pa, updated_at)
           VALUES (9, 2026, 'R', 'batter_hits', 0.111, 20, 'old')"""
    )
    memory_db.commit()

    splits = {9: {"byBreakdown": [
        {"split_name": "vs. Right", "at_bats": 234, "hits": 74, "doubles": 17,
         "triples": 1, "home_runs": 11, "walks": 34, "hit_by_pitch": 1},
    ]}}
    client = _StubClient(splits)

    import src.pipelines.sync_statcast as m
    from unittest.mock import patch

    class _Ctx:
        def __enter__(self_): return memory_db
        def __exit__(self_, *a): return False

    with patch.object(m, "get_db_connection", lambda: _Ctx()):
        n = await sync_bdl_platoon_splits(2026, player_ids=[9], client=client)

    assert n == 3  # hits, hr, tb
    row = memory_db.execute(
        "SELECT rate_per_pa, n_pa FROM batter_platoon_splits "
        "WHERE player_id=9 AND vs_hand='R' AND market='batter_hits'"
    ).fetchone()
    assert row["rate_per_pa"] == pytest.approx(74 / 269, abs=1e-5)  # overwrote 0.111
    assert row["n_pa"] == 269


async def test_sync_failsafe_on_player_error(splits_db):
    memory_db = splits_db
    class _Boom:
        async def get_player_splits(self, player_id, season=None):
            raise RuntimeError("BDL down")

    import src.pipelines.sync_statcast as m
    from unittest.mock import patch

    class _Ctx:
        def __enter__(self_): return memory_db
        def __exit__(self_, *a): return False

    with patch.object(m, "get_db_connection", lambda: _Ctx()):
        n = await sync_bdl_platoon_splits(2026, player_ids=[1, 2], client=_Boom())
    assert n == 0  # both errored, nothing upserted, no raise
