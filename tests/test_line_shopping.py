"""Tests for alt-line cross-book shopping in scan_props.

Covers:
  * _group_by_player_market regrouping
  * _pick_anchor_line selection
  * Anchor-level agreement gate blocks a (player, market) when model
    disagrees with sharp at the consensus line.
  * Alt-line picked when its EV beats the anchor's.
  * Alt-lines outside ALT_LINE_MAX_DISTANCE are excluded.
  * Same-line regression: behavior unchanged when only the anchor line
    is quoted.
  * MAX_BETS_PER_PLAYER caps the number of edges kept per player.
"""

import logging
import sqlite3
from unittest.mock import patch, MagicMock, AsyncMock

import pytest


# ---------------------------------------------------------------------------
# Helper utility tests (pure functions)
# ---------------------------------------------------------------------------


def test_group_by_player_market_regroups_by_two_keys():
    from src.pipelines.scan_props import _group_by_player_market
    player_lines = {
        ('Cole', 'pitcher_strikeouts', 7.5): {'pinnacle': {'over': 1.9, 'under': 1.9}},
        ('Cole', 'pitcher_strikeouts', 6.5): {'draftkings': {'over': 2.4, 'under': 1.6}},
        ('Cole', 'pitcher_earned_runs', 2.5): {'pinnacle': {'over': 1.95, 'under': 1.85}},
        ('Judge', 'batter_hits', 0.5):       {'pinnacle': {'over': 1.5, 'under': 2.5}},
    }
    grouped = _group_by_player_market(player_lines)
    assert set(grouped.keys()) == {
        ('Cole', 'pitcher_strikeouts'),
        ('Cole', 'pitcher_earned_runs'),
        ('Judge', 'batter_hits'),
    }
    assert set(grouped[('Cole', 'pitcher_strikeouts')].keys()) == {7.5, 6.5}


def test_pick_anchor_line_returns_first_sharp_quoted():
    from src.pipelines.scan_props import _pick_anchor_line
    lines_for_market = {
        # 6.5 not sharp-quoted both sides
        6.5: {'draftkings': {'over': 2.4, 'under': 1.6}},
        # 7.5 has full sharp pair
        7.5: {'pinnacle': {'over': 1.91, 'under': 1.91},
              'draftkings': {'over': 1.95, 'under': 1.85}},
    }
    anchor = _pick_anchor_line(lines_for_market, ['pinnacle', 'circasports'])
    assert anchor is not None
    line, sharp_o, sharp_u, book = anchor
    assert line == 7.5
    assert book == 'pinnacle'
    assert (sharp_o, sharp_u) == (1.91, 1.91)


def test_pick_anchor_line_returns_none_when_no_sharp_pair():
    from src.pipelines.scan_props import _pick_anchor_line
    lines_for_market = {
        7.5: {'draftkings': {'over': 1.95, 'under': 1.85}},
        6.5: {'fanduel': {'over': 2.4, 'under': 1.6}},
    }
    assert _pick_anchor_line(lines_for_market, ['pinnacle']) is None


# ---------------------------------------------------------------------------
# End-to-end loop tests with mocks
# ---------------------------------------------------------------------------


def _seed_schema(conn):
    conn.executescript('''
        CREATE TABLE games (
            game_id TEXT PRIMARY KEY, home_team TEXT, away_team TEXT,
            venue TEXT, status TEXT,
            lineups_confirmed_at TEXT, last_scanned_at TEXT, game_time TEXT,
            date TEXT
        );
        CREATE TABLE prop_snapshots (
            snapshot_id TEXT PRIMARY KEY,
            game_id TEXT, player_name TEXT, market TEXT, line REAL,
            over_odds REAL, under_odds REAL, bookmaker TEXT, timestamp TEXT,
            devigged_over REAL, devigged_under REAL
        );
        CREATE TABLE projections (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            game_id TEXT, player_name TEXT, market TEXT,
            projected_mean REAL, prob_over REAL, prob_under REAL,
            context_json TEXT, timestamp TEXT,
            UNIQUE(game_id, player_name, market)
        );
        CREATE TABLE game_totals_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            game_id TEXT NOT NULL, total REAL NOT NULL,
            source TEXT NOT NULL, timestamp TEXT NOT NULL
        );
        CREATE TABLE bet_candidates (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            game_id TEXT NOT NULL, player_id INTEGER, player_name TEXT NOT NULL,
            market TEXT NOT NULL, line REAL NOT NULL, side TEXT NOT NULL,
            bookmaker TEXT NOT NULL, odds REAL NOT NULL,
            sharp_book TEXT, anchor_line REAL, truth_prob REAL, model_prob REAL,
            open_devig_prob REAL, edge_pct REAL, ev REAL, kelly_fraction REAL,
            recommended_stake REAL, steam_detected INTEGER DEFAULT 0,
            created_at TEXT NOT NULL,
            UNIQUE(game_id, player_name, market, line, side, bookmaker)
        );
    ''')


@pytest.fixture
def memory_db():
    conn = sqlite3.connect(':memory:')
    conn.row_factory = sqlite3.Row
    _seed_schema(conn)
    conn.execute(
        "INSERT INTO games (game_id, home_team, away_team, venue, status) "
        "VALUES ('g1', 'Yankees', 'Red Sox', 'Yankee Stadium', 'SCHEDULED')"
    )
    conn.commit()
    return conn


class _DBCtx:
    def __init__(self, conn): self._conn = conn
    def __call__(self): return self
    def __enter__(self): return self._conn
    def __exit__(self, *a): return False


def _payload(player, market, lines_by_book):
    """Build an Odds-API-shaped event_odds dict.

    lines_by_book: {book: [(line, over_odds, under_odds), ...]}
    """
    bookmakers = []
    for book, entries in lines_by_book.items():
        outcomes = []
        for line, o, u in entries:
            outcomes.append({'name': 'Over', 'description': player,
                             'point': float(line), 'price': float(o)})
            outcomes.append({'name': 'Under', 'description': player,
                             'point': float(line), 'price': float(u)})
        bookmakers.append({'key': book, 'markets': [{'key': market,
                                                     'outcomes': outcomes}]})
    return {'bookmakers': bookmakers}


def _projection(prob_over, prob_under, projected_mean=6.5, market='pitcher_strikeouts'):
    return {
        'player_name': 'Cole', 'market': market, 'line': 7.5,
        'projected_mean': projected_mean,
        'prob_over': prob_over, 'prob_under': prob_under,
        'alpha': None, 'sigma': None,
        'injury_status': 'Healthy',
        'sample_size': 30,
        'context': {'model': 'test'},
    }


def _patch_stack(memory_db, payload, projection, prob_by_line):
    """Build the standard mock stack. prob_by_line maps line→(prob_over, prob_under)."""
    db_ctx = _DBCtx(memory_db)
    odds_instance = MagicMock()
    odds_instance.get_event_odds = AsyncMock(return_value=payload)
    weather_instance = MagicMock()
    weather_instance.get_game_weather.return_value = None

    def fake_probs(mean, line, market, alpha=None, sigma=None, **kw):
        return prob_by_line.get(line, (0.50, 0.50))

    return [
        patch('src.pipelines.scan_props.get_db_connection', db_ctx),
        patch('src.pipelines.scan_props.OddsAPIClient',
              return_value=odds_instance),
        patch('src.pipelines.scan_props.WeatherClient',
              return_value=weather_instance),
        patch('src.pipelines.scan_props._get_ump_k_factor', return_value=1.0),
        patch('src.pipelines.scan_props._record_total_snapshot'),
        patch('src.pipelines.scan_props._build_projection',
              return_value=projection),
        patch('src.pipelines.scan_props.get_probabilities',
              side_effect=fake_probs),
    ]


def _enter(stack):
    for p in stack:
        p.start()


def _exit(stack):
    for p in reversed(stack):
        p.stop()


async def test_anchor_validation_blocks_when_model_disagrees(memory_db, caplog):
    # Sharp at 7.5: devigged ≈ (0.559, 0.442). Model says 0.70 over → diff
    # 0.141 > SHARP_MODEL_AGREEMENT_TOL (0.05). Loop must skip the entire
    # (player, market) — no edges at any line, including a juicy alt.
    payload = _payload('Cole', 'pitcher_strikeouts', {
        'pinnacle': [(7.5, 1.74, 2.20)],
        'draftkings': [(7.5, 1.95, 1.95), (6.5, 2.50, 1.55)],
    })
    proj = _projection(prob_over=0.70, prob_under=0.30)
    prob_by_line = {7.5: (0.70, 0.30), 6.5: (0.85, 0.15)}
    stack = _patch_stack(memory_db, payload, proj, prob_by_line)
    _enter(stack)
    try:
        from src.pipelines.scan_props import scan_props
        with caplog.at_level(logging.INFO):
            await scan_props(force=True)
    finally:
        _exit(stack)

    assert "EDGE FOUND" not in caplog.text


async def test_alt_line_picked_when_ev_beats_anchor(memory_db, caplog):
    # Anchor 7.5 sharp devig ≈ (0.559, 0.442). Soft @7.5 over=1.85
    # (implied 0.541) → edge 1.8%, fails EDGE_MIN. Soft alt @6.5 over=1.85
    # with model_prob=0.75 → edge ~20% → playable. Winner must be the
    # alt-line, and no edge logged at the anchor.
    payload = _payload('Cole', 'pitcher_strikeouts', {
        'pinnacle': [(7.5, 1.74, 2.20)],
        'draftkings': [(7.5, 1.85, 1.95), (6.5, 1.85, 1.95)],
    })
    proj = _projection(prob_over=0.56, prob_under=0.44)  # agrees w/ sharp at anchor
    prob_by_line = {7.5: (0.56, 0.44), 6.5: (0.75, 0.25)}
    stack = _patch_stack(memory_db, payload, proj, prob_by_line)
    _enter(stack)
    try:
        from src.pipelines.scan_props import scan_props
        with caplog.at_level(logging.INFO):
            await scan_props(force=True)
    finally:
        _exit(stack)

    edge_lines = [r.message for r in caplog.records if "EDGE FOUND" in r.message]
    assert len(edge_lines) == 1
    msg = edge_lines[0]
    assert "OVER 6.5" in msg
    assert "anchor 7.5" in msg


async def test_alt_line_outside_distance_cap_excluded(memory_db, caplog):
    # Soft offers a juicy over at 5.0 (anchor 7.5, distance 2.5 > 1.0 cap).
    # That row must not be evaluated even though its raw EV would be huge.
    payload = _payload('Cole', 'pitcher_strikeouts', {
        'pinnacle': [(7.5, 1.91, 1.91)],
        'draftkings': [(7.5, 1.92, 1.92), (5.0, 3.00, 1.40)],
    })
    proj = _projection(prob_over=0.524, prob_under=0.476)  # agrees at anchor
    # If the cap were ignored, 5.0 over would be flagged with 0.95 truth.
    prob_by_line = {7.5: (0.524, 0.476), 5.0: (0.95, 0.05)}
    stack = _patch_stack(memory_db, payload, proj, prob_by_line)
    _enter(stack)
    try:
        from src.pipelines.scan_props import scan_props
        with caplog.at_level(logging.INFO):
            await scan_props(force=True)
    finally:
        _exit(stack)

    assert "5.0" not in "".join(
        r.message for r in caplog.records if "EDGE FOUND" in r.message
    )


async def test_same_line_regression(memory_db, caplog):
    # Only the anchor line is quoted by both sides. Behavior should match
    # the pre-refactor world: one playable EV at the anchor when EV beats
    # the threshold.
    payload = _payload('Cole', 'pitcher_strikeouts', {
        'pinnacle': [(7.5, 2.10, 1.78)],     # devigged over ≈ 0.460
        'draftkings': [(7.5, 2.40, 1.65)],   # over implied ≈ 0.417
    })
    # Bump sharp_prob_over above MIN_MODEL_PROB by tightening odds:
    # use over=1.74, under=2.20 → devig ≈ (0.559, 0.442).
    payload = _payload('Cole', 'pitcher_strikeouts', {
        'pinnacle': [(7.5, 1.74, 2.20)],
        'draftkings': [(7.5, 1.95, 2.20)],   # over implied 0.513
        # edge = 0.559 - 0.513 = 4.6% (below EDGE_MIN=5%) — bump to 1.99
    })
    payload = _payload('Cole', 'pitcher_strikeouts', {
        'pinnacle': [(7.5, 1.74, 2.20)],
        'draftkings': [(7.5, 1.99, 2.20)],   # over implied 0.5025 → edge ~5.7%
    })
    proj = _projection(prob_over=0.56, prob_under=0.44)
    prob_by_line = {7.5: (0.56, 0.44)}
    stack = _patch_stack(memory_db, payload, proj, prob_by_line)
    _enter(stack)
    try:
        from src.pipelines.scan_props import scan_props
        with caplog.at_level(logging.INFO):
            await scan_props(force=True)
    finally:
        _exit(stack)

    edge_lines = [r.message for r in caplog.records if "EDGE FOUND" in r.message]
    assert len(edge_lines) >= 1
    assert any("OVER 7.5" in m for m in edge_lines)


async def test_multiple_alts_share_per_player_cap(memory_db, caplog):
    # Three alt-lines all playable. With MAX_BETS_PER_PLAYER=2 only the
    # top 2 by EV survive into the EDGE FOUND log.
    payload = _payload('Cole', 'pitcher_strikeouts', {
        'pinnacle': [(7.5, 1.74, 2.20)],     # anchor sharp
        'draftkings': [
            (7.5, 1.85, 1.95),               # over implied 0.541
            (7.0, 1.85, 1.95),
            (6.5, 1.85, 1.95),
        ],
    })
    proj = _projection(prob_over=0.56, prob_under=0.44)
    prob_by_line = {
        7.5: (0.56, 0.44),
        7.0: (0.65, 0.35),
        6.5: (0.75, 0.25),
    }
    stack = _patch_stack(memory_db, payload, proj, prob_by_line)
    with patch('src.pipelines.scan_props.MAX_BETS_PER_PLAYER', 2):
        _enter(stack)
        try:
            from src.pipelines.scan_props import scan_props
            with caplog.at_level(logging.INFO):
                await scan_props(force=True)
        finally:
            _exit(stack)

    edge_lines = [r.message for r in caplog.records if "EDGE FOUND" in r.message]
    assert len(edge_lines) == 2
    # Top two by EV should be the lower lines (higher truth_over).
    assert any("6.5" in m for m in edge_lines)
    assert any("7.0" in m for m in edge_lines)
