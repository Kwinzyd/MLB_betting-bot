from unittest.mock import patch


def _make_edge(playable=True, stake=10.0):
    return {
        "is_playable": playable,
        "edge_pct": 8.5,
        "ev": 0.04,
        "model_prob": 0.62,
        "book_implied": 0.50,
        "kelly": {"recommended_stake": stake},
    }


def _make_context(odds=2.0):
    return {
        "player_name": "Gerrit Cole",
        "market": "pitcher_strikeouts",
        "side": "over",
        "line": 6.5,
        "odds": odds,
        "bookmaker": "draftkings",
        "game_id": "g1",
        "home_team": "Yankees",
        "away_team": "Red Sox",
        "game_venue": "Yankee Stadium",
        "projection": {"projected_mean": 7.5},
        "model_context": {},
    }


@patch('src.clients.execution.PAPER_EXCHANGE_ENABLED', True)
@patch('src.clients.execution.TELEGRAM_VENUE_ENABLED', False)
def test_registry_paper_only():
    from src.clients.execution import build_venue_registry
    venues = build_venue_registry()
    assert len(venues) == 1
    assert venues[0].name == "paper_exchange"


@patch('src.clients.execution.PAPER_EXCHANGE_ENABLED', True)
@patch('src.clients.execution.TELEGRAM_VENUE_ENABLED', True)
def test_registry_both_enabled_returns_telegram_first():
    from src.clients.execution import build_venue_registry
    venues = build_venue_registry()
    assert [v.name for v in venues] == ["telegram", "paper_exchange"]


@patch('src.clients.execution.PAPER_EXCHANGE_ENABLED', False)
@patch('src.clients.execution.TELEGRAM_VENUE_ENABLED', False)
def test_registry_all_disabled_is_empty():
    from src.clients.execution import build_venue_registry
    assert build_venue_registry() == []


def test_polymarket_skips_cleanly_without_token_mapping():
    """A real edge (stake nested under kelly) reaches the token check — proving
    the stake key path is correct — and skips cleanly when no token is mapped,
    instead of erroring or faking a fill."""
    import asyncio
    from src.clients.execution.polymarket_exchange import PolymarketExchangeVenue
    venue = PolymarketExchangeVenue()
    record = asyncio.run(venue.place_order(_make_edge(stake=12.0), _make_context()))
    assert record["venue"] == "polymarket"
    assert record["status"] == "skipped"
    assert "token" in (record["notes"] or "").lower()


def test_polymarket_zero_stake_skips_before_token_check():
    import asyncio
    from src.clients.execution.polymarket_exchange import PolymarketExchangeVenue
    venue = PolymarketExchangeVenue()
    record = asyncio.run(venue.place_order(_make_edge(stake=0.0), _make_context()))
    assert record["status"] == "skipped"
    assert "stake" in (record["notes"] or "").lower()


def test_execution_venue_default_cancel_returns_false():
    from src.clients.execution.paper_exchange import PaperExchange
    import asyncio
    venue = PaperExchange()
    # asyncio.run() spins up a fresh loop each call; get_event_loop() would reuse
    # the loop pytest-asyncio already closed, breaking under suite ordering on 3.13.
    assert asyncio.run(venue.cancel_order("x")) is False


def test_execution_venue_default_open_orders_empty():
    from src.clients.execution.paper_exchange import PaperExchange
    import asyncio
    venue = PaperExchange()
    assert asyncio.run(venue.get_open_orders()) == []
