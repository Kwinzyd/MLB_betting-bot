import pytest

from src.models import bvp
from src.pipelines.sync_bvp import parse_versus_row


# --- parse_versus_row ---

def test_parse_versus_row_computes_tb_and_pa():
    row = {
        "opponent_player": {"id": 5407, "full_name": "Corbin Burnes"},
        "at_bats": 9, "hits": 2, "doubles": 1, "triples": 0,
        "home_runs": 1, "walks": 2, "strikeouts": 1,
    }
    p = parse_versus_row(row)
    assert p["pitcher_id"] == 5407
    assert p["ab"] == 9
    assert p["pa"] == 11             # ab + walks
    # TB = 2 + 1 + 0 + 3*1 = 6
    assert p["total_bases"] == 6
    assert p["home_runs"] == 1


def test_parse_versus_row_skips_missing_pitcher_or_ab():
    assert parse_versus_row({"at_bats": 5, "hits": 2}) is None            # no opponent_player
    assert parse_versus_row({"opponent_player": {"id": 1}, "at_bats": 0}) is None


# --- bvp_rate ---

def test_bvp_rate_per_market():
    stats = {"pa": 10, "ab": 9, "hits": 3, "total_bases": 6, "home_runs": 1}
    assert bvp.bvp_rate(stats, "batter_hits") == pytest.approx(0.3)
    assert bvp.bvp_rate(stats, "batter_total_bases") == pytest.approx(0.6)
    assert bvp.bvp_rate(stats, "batter_home_runs") == pytest.approx(0.1)
    assert bvp.bvp_rate(stats, "pitcher_strikeouts") is None   # not a BvP market
    assert bvp.bvp_rate({"pa": 0, "hits": 1}, "batter_hits") is None


# --- bvp_factor: shrinkage, floor, bounds ---

def test_factor_neutral_below_min_ab():
    stats = {"ab": 5, "pa": 6, "hits": 4}   # torrid but tiny sample
    assert bvp.bvp_factor(stats, 0.25, "batter_hits", min_ab=10) == 1.0


def test_factor_neutral_on_bad_baseline():
    stats = {"ab": 30, "pa": 33, "hits": 12}
    assert bvp.bvp_factor(stats, 0.0, "batter_hits") == 1.0


def test_factor_shrinks_toward_baseline():
    # 20 PA at .400 vs a .250 baseline, prior 40 PA.
    stats = {"ab": 18, "pa": 20, "hits": 8}  # bvp rate = 0.40
    f = bvp.bvp_factor(stats, 0.25, "batter_hits", prior_pa=40, min_ab=10,
                       lo=0.5, hi=2.0)
    # shrunk = (20*0.40 + 40*0.25)/60 = 0.30 ; factor = 0.30/0.25 = 1.20
    assert f == pytest.approx(1.20, abs=1e-6)


def test_factor_respects_bounds():
    stats = {"ab": 100, "pa": 110, "hits": 80}  # extreme .727 rate, big sample
    f = bvp.bvp_factor(stats, 0.20, "batter_hits", prior_pa=40, min_ab=10,
                       lo=0.95, hi=1.05)
    assert f == 1.05  # clamps to the tight upper bound


def test_factor_below_one_when_batter_struggles():
    stats = {"ab": 40, "pa": 44, "hits": 4}   # .091 vs .270 baseline
    f = bvp.bvp_factor(stats, 0.27, "batter_hits", prior_pa=40, min_ab=10,
                       lo=0.5, hi=2.0)
    assert f < 1.0


# --- bvp_factor_db ---

def test_factor_db_reads_row_and_is_failsafe():
    import sqlite3
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    conn.execute(
        """CREATE TABLE bvp_stats (
               batter_id INTEGER, pitcher_id INTEGER, ab INTEGER, pa INTEGER,
               hits INTEGER, total_bases INTEGER, home_runs INTEGER,
               strikeouts INTEGER, walks INTEGER, updated_at TEXT,
               PRIMARY KEY (batter_id, pitcher_id))"""
    )
    conn.execute(
        "INSERT INTO bvp_stats VALUES (7, 99, 18, 20, 8, 12, 1, 3, 2, 'now')"
    )
    conn.commit()
    f = bvp.bvp_factor_db(conn, 7, 99, "batter_hits", 0.25)
    assert f > 1.0                                   # .40 vs .25, shrunk
    assert bvp.bvp_factor_db(conn, 7, 12345, "batter_hits", 0.25) == 1.0  # no row
    assert bvp.bvp_factor_db(conn, None, 99, "batter_hits", 0.25) == 1.0  # no batter
    conn.close()
