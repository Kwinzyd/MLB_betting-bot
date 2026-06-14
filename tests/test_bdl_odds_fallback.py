import pytest

from src.models.pa_estimator import implied_team_total, _american_to_prob
from src.clients.bdl_odds import get_game_market, _median_or_none


# --- American odds -> probability ---

def test_american_to_prob():
    assert _american_to_prob(-110) == pytest.approx(110 / 210, abs=1e-4)
    assert _american_to_prob(150) == pytest.approx(100 / 250, abs=1e-4)
    assert _american_to_prob(None) is None
    assert _american_to_prob("") is None


# --- implied_team_total tilt ---

def test_symmetric_without_moneylines():
    assert implied_team_total(9.0) == pytest.approx(4.5)
    assert implied_team_total(8.0, None, None) == pytest.approx(4.0)


def test_tilt_favors_favorite_and_conserves_total():
    gt = 9.0
    home = implied_team_total(gt, -150, 130, side="home")
    away = implied_team_total(gt, -150, 130, side="away")
    assert home > away                       # home is the favorite
    assert home + away == pytest.approx(gt)   # split conserves the total


def test_tilt_is_bounded_for_extreme_favorite():
    home = implied_team_total(9.0, -100000, 5000, side="home")
    # share clamped to 0.60 -> 5.4, never the full total
    assert home == pytest.approx(9.0 * 0.60, abs=1e-6)


def test_tilt_falls_back_to_symmetric_on_one_sided_ml():
    assert implied_team_total(9.0, -150, None, side="home") == pytest.approx(4.5)


# --- BDL consensus helper ---

def test_median_or_none():
    assert _median_or_none([1, 2, 3]) == 2
    assert _median_or_none([None, "", "4", 6]) == 5
    assert _median_or_none([None, ""]) is None


class _StubClient:
    def __init__(self, records):
        self._records = records

    async def get_odds(self, game_ids=None, dates=None):
        return self._records


async def test_get_game_market_consensus():
    records = [
        {"vendor": "fanduel", "total_value": "9", "moneyline_home_odds": -120, "moneyline_away_odds": 100},
        {"vendor": "betmgm", "total_value": "9.5", "moneyline_home_odds": -130, "moneyline_away_odds": 110},
        {"vendor": "dk", "total_value": "8.5", "moneyline_home_odds": -110, "moneyline_away_odds": -105},
    ]
    out = await get_game_market(123, client=_StubClient(records))
    assert out["total"] == pytest.approx(9.0)       # median of 8.5, 9, 9.5
    assert out["ml_home"] == pytest.approx(-120)    # median of -110,-120,-130
    assert out["ml_away"] == pytest.approx(100)     # median of -105,100,110


async def test_get_game_market_none_when_empty():
    assert await get_game_market(123, client=_StubClient([])) is None
    assert await get_game_market(None) is None


async def test_get_game_market_failsafe_on_error():
    class _Boom:
        async def get_odds(self, **kw):
            raise RuntimeError("BDL down")
    assert await get_game_market(123, client=_Boom()) is None
