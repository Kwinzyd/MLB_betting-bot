"""Tests for game-market settlement grading and BDL quote parsing."""
from unittest.mock import patch

import pytest

from tests.fixtures.fixture_db import memory_conn
from src.pipelines.settle_results import _grade_game_market
from src.pipelines.scan_game_markets import _price_markets
from src.clients.bdl_odds import get_game_quotes, american_to_decimal


class _DBCtx:
    def __init__(self, conn):
        self._conn = conn

    def __call__(self, *a, **k):
        return self

    def __enter__(self):
        return self._conn

    def __exit__(self, *a):
        return False


def _game(conn, home_score, away_score):
    conn.execute(
        "INSERT INTO games (game_id, home_team, away_team, venue, status, home_score, away_score) "
        "VALUES ('g1','Yankees','Red Sox','Yankee Stadium','COMPLETED',?,?)",
        (home_score, away_score),
    )
    conn.commit()


class TestGradeGameMarket:
    def _grade(self, conn, market, side, line):
        with patch('src.pipelines.settle_results.get_db_connection', _DBCtx(conn)):
            return _grade_game_market(market, side, line, 'g1')

    def test_moneyline(self):
        conn = memory_conn(); _game(conn, 5, 3)
        assert self._grade(conn, 'moneyline', 'home', 0.0)[0] == 'WIN'
        assert self._grade(conn, 'moneyline', 'away', 0.0)[0] == 'LOSS'

    def test_total_over_under_and_push(self):
        conn = memory_conn(); _game(conn, 5, 3)  # total 8
        assert self._grade(conn, 'game_total', 'over', 8.5)[0] == 'LOSS'
        assert self._grade(conn, 'game_total', 'under', 8.5)[0] == 'WIN'
        assert self._grade(conn, 'game_total', 'over', 8.0)[0] == 'PUSH'

    def test_run_line_home_favorite(self):
        conn = memory_conn(); _game(conn, 5, 3)  # margin +2, covers -1.5
        assert self._grade(conn, 'run_line', 'home', -1.5)[0] == 'WIN'
        conn2 = memory_conn(); _game(conn2, 4, 3)  # margin +1, fails -1.5
        with patch('src.pipelines.settle_results.get_db_connection', _DBCtx(conn2)):
            assert _grade_game_market('run_line', 'home', -1.5, 'g1')[0] == 'LOSS'

    def test_run_line_away_dog(self):
        conn = memory_conn(); _game(conn, 4, 3)  # away loses by 1, +1.5 covers
        assert self._grade(conn, 'run_line', 'away', 1.5)[0] == 'WIN'
        conn2 = memory_conn(); _game(conn2, 5, 3)  # away loses by 2, +1.5 fails
        with patch('src.pipelines.settle_results.get_db_connection', _DBCtx(conn2)):
            assert _grade_game_market('run_line', 'away', 1.5, 'g1')[0] == 'LOSS'

    def test_defers_when_scores_absent(self):
        conn = memory_conn()
        conn.execute(
            "INSERT INTO games (game_id, home_team, away_team, status) "
            "VALUES ('g1','Yankees','Red Sox','COMPLETED')")
        conn.commit()
        assert self._grade(conn, 'moneyline', 'home', 0.0) == (None, None)


def _even_quotes(total_line=9.0):
    side = lambda: {"devig": 0.5, "best_odds": 2.0, "best_vendor": "dk"}
    return {
        "run_line": 1.5, "total_line": total_line, "home_is_ml_favorite": True,
        "markets": {
            "moneyline": {"home": side(), "away": side()},
            "game_total": {"over": side(), "under": side()},
            "run_line": {"home": side(), "away": side()},
        },
    }


class TestPriceMarkets:
    def test_model_favorite_beats_even_book(self):
        # Model: home strong (lam 6) vs away weak (lam 3) -> home ML ~0.75 while
        # the book devigs to 0.50 -> a large home-moneyline edge.
        game = {"home_team": "Yankees", "away_team": "Red Sox"}
        edges = _price_markets(game, _even_quotes(9.0), lam_home=6.0, lam_away=3.0)
        ml_home = [e for e in edges if e["market"] == "moneyline" and e["side"] == "home"]
        assert ml_home, "expected a home moneyline edge"
        e = ml_home[0]
        assert e["edge_pct"] > 10.0
        assert e["ev"] > 0
        assert e["recommended_stake"] >= 0
        # The losing side (away ML) must NOT be flagged.
        assert not [x for x in edges if x["market"] == "moneyline" and x["side"] == "away"]

    def test_no_edge_when_model_matches_book(self):
        game = {"home_team": "Yankees", "away_team": "Red Sox"}
        # Symmetric lambdas -> ~50/50, matching the even book -> no edge clears.
        edges = _price_markets(game, _even_quotes(9.0), lam_home=4.5, lam_away=4.5)
        assert not [e for e in edges if e["market"] == "moneyline"]


class _StubClient:
    def __init__(self, records):
        self._records = records

    async def get_odds(self, game_ids=None, dates=None):
        return self._records


class TestGetGameQuotes:
    async def test_moneyline_only_uses_minus110_fallback_for_total(self):
        # Vendors quote ML + total line, but no over/under juice.
        records = [
            {"vendor": "fanduel", "total_value": "9", "moneyline_home_odds": -120, "moneyline_away_odds": 100},
            {"vendor": "dk", "total_value": "8.5", "moneyline_home_odds": -110, "moneyline_away_odds": -105},
        ]
        out = await get_game_quotes(123, client=_StubClient(records))
        assert out["total_line"] == pytest.approx(8.75)  # median of 8.5, 9
        assert out["run_line"] == 1.5
        # Moneyline devigs to real fair probs and shops the best price.
        ml = out["markets"]["moneyline"]
        assert 0.0 < ml["home"]["devig"] < 1.0
        assert ml["home"]["devig"] + ml["away"]["devig"] == pytest.approx(1.0, abs=1e-9)
        assert ml["away"]["best_odds"] == pytest.approx(american_to_decimal(100))  # best away price
        # Total has no juice -> 50/50 devig + assumed -110 price.
        tot = out["markets"]["game_total"]
        assert tot["over"]["devig"] == 0.5
        assert tot["over"]["best_odds"] == pytest.approx(1.909)

    async def test_none_when_empty(self):
        assert await get_game_quotes(123, client=_StubClient([])) is None
        assert await get_game_quotes(None) is None

    async def test_full_juice_devigs_total(self):
        records = [
            {"vendor": "dk", "total_value": "8.5",
             "moneyline_home_odds": -110, "moneyline_away_odds": -110,
             "total_over_odds": -105, "total_under_odds": -115},
        ]
        out = await get_game_quotes(123, client=_StubClient(records))
        tot = out["markets"]["game_total"]
        assert tot["over"]["devig"] != 0.5          # real juice -> real devig
        assert tot["over"]["devig"] + tot["under"]["devig"] == pytest.approx(1.0, abs=1e-9)
