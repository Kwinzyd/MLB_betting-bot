import pytest

from src.clients.sportsdataio import SportsDataIOClient


def _client_with_props(props):
    """A client whose daily cache is pre-seeded, so no network is touched."""
    c = SportsDataIOClient()
    c._props_cache = props
    c._props_cache_date = "2026-07-08"  # non-None so _populate_props_cache no-ops

    async def _noop():
        return
    c._populate_props_cache = _noop  # type: ignore[assignment]
    return c


def test_american_to_decimal():
    c = SportsDataIOClient()
    assert c._american_to_decimal(100) == pytest.approx(2.0)
    assert c._american_to_decimal(-110) == pytest.approx(1.909, abs=1e-3)
    assert c._american_to_decimal(150) == pytest.approx(2.5)


async def test_get_event_odds_translates_matchup():
    props = [
        {"Name": "Aaron Judge", "Team": "NYY", "Opponent": "TB",
         "Description": "Hits", "OverUnder": 1.5, "OverPayout": 120, "UnderPayout": -150},
        {"Name": "Gerrit Cole", "Team": "NYY", "Opponent": "TB",
         "Description": "Pitching Strikeouts", "OverUnder": 6.5, "OverPayout": -110, "UnderPayout": -110},
        # different game — must be excluded
        {"Name": "Someone Else", "Team": "LAD", "Opponent": "SF",
         "Description": "Hits", "OverUnder": 0.5, "OverPayout": -200, "UnderPayout": 160},
    ]
    c = _client_with_props(props)
    ev = await c.get_event_odds(
        "g", ["batter_hits", "pitcher_strikeouts"],
        home_team_full="New York Yankees", away_team_full="Tampa Bay Rays",
    )
    assert list(ev.keys()) == ["bookmakers"]
    book = ev["bookmakers"][0]
    assert book["key"] == "sportsdataio"
    markets = {m["key"]: m for m in book["markets"]}
    assert set(markets) == {"batter_hits", "pitcher_strikeouts"}

    hits = markets["batter_hits"]["outcomes"]
    over = next(o for o in hits if o["name"] == "Over")
    assert over["description"] == "Aaron Judge"
    assert over["point"] == 1.5
    assert over["price"] == pytest.approx(2.2)  # +120 -> 2.2
    # the LAD/SF prop must not leak into this game
    assert all(o["description"] != "Someone Else" for o in hits)


async def test_get_event_odds_athletics_abbrev():
    # SportsDataIO uses "ATH"; the map must resolve the full name to it.
    props = [
        {"Name": "Brent Rooker", "Team": "ATH", "Opponent": "SEA",
         "Description": "Total Bases", "OverUnder": 1.5, "OverPayout": 110, "UnderPayout": -130},
    ]
    c = _client_with_props(props)
    ev = await c.get_event_odds(
        "g", ["batter_total_bases"],
        home_team_full="Seattle Mariners", away_team_full="Athletics",
    )
    assert ev["bookmakers"][0]["markets"][0]["key"] == "batter_total_bases"
    assert ev["bookmakers"][0]["markets"][0]["outcomes"][0]["description"] == "Brent Rooker"


async def test_get_event_odds_empty_without_team_names():
    c = _client_with_props([{"Name": "X", "Team": "NYY", "Opponent": "TB",
                             "Description": "Hits", "OverUnder": 1.5,
                             "OverPayout": -110, "UnderPayout": -110}])
    assert await c.get_event_odds("g", ["batter_hits"]) == {}
