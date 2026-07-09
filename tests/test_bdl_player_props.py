import pytest

from src.clients.bdl_odds import get_player_prop_odds, PROP_TYPE_TO_MARKET


def _record(player_id=1, vendor="draftkings", prop_type="hits", line="1.5",
            market_type="over_under", over=-115, under=-105, odds=None):
    return {
        "id": 99,
        "game_id": 777,
        "player_id": player_id,
        "vendor": vendor,
        "prop_type": prop_type,
        "line_value": line,
        "market": {
            "type": market_type,
            "over_odds": over,
            "under_odds": under,
            "odds": odds,
        },
        "updated_at": "2026-07-08T12:00:00Z",
    }


class _StubClient:
    def __init__(self, records):
        self._records = records
        self.bust_cache_seen = None

    async def get_player_props(self, game_id, bust_cache=False):
        self.bust_cache_seen = bust_cache
        return self._records


NAMES = {1: "Aaron Judge", 2: "Tarik Skubal"}


async def test_translates_to_odds_api_shape():
    records = [
        _record(player_id=1, vendor="draftkings", prop_type="hits", line="1.5"),
        _record(player_id=2, vendor="fanduel", prop_type="pitcher_strikeouts",
                line="6.5", over=100, under=-130),
    ]
    out = await get_player_prop_odds(777, client=_StubClient(records), name_map=NAMES)

    books = {b["key"]: b for b in out["bookmakers"]}
    assert set(books) == {"bdl_draftkings", "bdl_fanduel"}

    dk_market = books["bdl_draftkings"]["markets"][0]
    assert dk_market["key"] == "batter_hits"
    over = next(o for o in dk_market["outcomes"] if o["name"] == "Over")
    assert over["point"] == 1.5
    assert over["price"] == pytest.approx(1 + 100 / 115, abs=1e-3)
    assert over["description"] == "Aaron Judge"

    fd_market = books["bdl_fanduel"]["markets"][0]
    assert fd_market["key"] == "pitcher_strikeouts"
    over = next(o for o in fd_market["outcomes"] if o["name"] == "Over")
    assert over["price"] == pytest.approx(2.0)


async def test_skips_milestone_unmapped_and_unknown_players():
    records = [
        _record(market_type="milestone", over=None, under=None, odds=250),
        _record(prop_type="stolen_bases"),          # not a modeled market
        _record(player_id=42),                       # not in name_map
        _record(line="not-a-number"),
        _record(vendor=None),
    ]
    assert await get_player_prop_odds(777, client=_StubClient(records), name_map=NAMES) is None


async def test_one_sided_quote_keeps_available_side():
    records = [_record(over=-110, under=None)]
    out = await get_player_prop_odds(777, client=_StubClient(records), name_map=NAMES)
    outcomes = out["bookmakers"][0]["markets"][0]["outcomes"]
    assert [o["name"] for o in outcomes] == ["Over"]


async def test_failsafe_none_on_error_empty_and_no_game():
    class _Boom:
        async def get_player_props(self, game_id, bust_cache=False):
            raise RuntimeError("BDL down")

    assert await get_player_prop_odds(777, client=_Boom(), name_map=NAMES) is None
    assert await get_player_prop_odds(777, client=_StubClient([]), name_map=NAMES) is None
    assert await get_player_prop_odds(None) is None


async def test_bust_cache_passthrough():
    stub = _StubClient([_record()])
    await get_player_prop_odds(777, client=stub, bust_cache=True, name_map=NAMES)
    assert stub.bust_cache_seen is True


def test_prop_type_map_covers_all_modeled_markets():
    from src.config import MARKETS_MAPPING
    assert set(PROP_TYPE_TO_MARKET.values()) == set(MARKETS_MAPPING.keys())
