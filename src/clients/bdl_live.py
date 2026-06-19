"""
BDL Live Client — real-time MLB game state via Ball Don't Lie API.

Provides two things the Live State Machine needs:
  1. Live game state per game (inning, outs, home/visitor scores, status).
  2. Per-player running box score stats so we know how many hits/TBs/etc.
     a batter has already accumulated, enabling "rest-of-game" re-projection.

Primary endpoint used:
    GET /mlb/v1/box_scores/live
        Returns all currently in-progress games with nested per-player stats.
        Single call, no game_id needed — very quota-friendly.

Secondary endpoint (batter slot derivation from play-by-play):
    GET /mlb/v1/plays?game_id=<id>
        Returns ordered plays with at_bat_number, batter_id, inning, outs.
        Used to determine which lineup slot is currently at the plate.

Rate-limit notes:
  - BDL GOAT tier: 600 req/min.  One live call every 20s for N in-progress
    games is well within that (1 call per poll cycle regardless of N).
  - The /plays endpoint is heavier; called at most once per game per minute
    via an in-memory TTL guard.
"""
from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from typing import Dict, List, Optional

import httpx

from src.config import BDL_API_KEY, MLB_SEASON
from src.data.cache import cache
from src.data.db import get_db_connection
from src.utils.circuit_breaker import AsyncCircuitBreaker
from src.utils.logging_utils import get_logger
from src.utils.retry import async_retry_api

logger = get_logger(__name__)

_BASE = "https://api.balldontlie.io/mlb/v1"
_HEADERS = {"Authorization": BDL_API_KEY} if BDL_API_KEY else {}

# Circuit breaker shared with other BDL clients.
_bdl_cb = AsyncCircuitBreaker(
    failure_threshold=5,
    recovery_timeout=120.0,
    exceptions=(httpx.RequestError,),
)

# Minimum seconds between /plays fetches per game (expensive, cursor-paginated).
_PLAYS_TTL_SECONDS = 60


# ---------------------------------------------------------------------------
# Low-level HTTP helpers
# ---------------------------------------------------------------------------

@_bdl_cb
@async_retry_api(max_retries=3, delay=1.0, backoff=2.0, exceptions=(httpx.HTTPError,))
async def _get(client: httpx.AsyncClient, endpoint: str, params: dict = None) -> dict:
    """Single BDL GET — no pagination, returns raw JSON body."""
    url = f"{_BASE}/{endpoint}"
    resp = await client.get(url, params=params, timeout=10.0)
    if resp.status_code == 429:
        logger.warning("BDL rate-limit hit on /%s", endpoint)
    resp.raise_for_status()
    return resp.json()


@_bdl_cb
@async_retry_api(max_retries=3, delay=1.0, backoff=2.0, exceptions=(httpx.HTTPError,))
async def _get_all_pages(
    client: httpx.AsyncClient, endpoint: str, params: dict = None
) -> List[dict]:
    """BDL GET with cursor-based pagination — collects all pages."""
    url = f"{_BASE}/{endpoint}"
    params = dict(params or {})
    results: List[dict] = []

    while url:
        resp = await client.get(url, params=params, timeout=15.0)
        if resp.status_code == 429:
            logger.warning("BDL rate-limit hit on /%s (paginated)", endpoint)
        resp.raise_for_status()
        body = resp.json()
        results.extend(body.get("data", []))
        cursor = body.get("meta", {}).get("next_cursor")
        if cursor:
            params = {"cursor": cursor}
        else:
            url = None  # type: ignore[assignment]

    return results


# ---------------------------------------------------------------------------
# Public client
# ---------------------------------------------------------------------------

class BDLLiveClient:
    """
    Async MLB live-data client.

    Methods used by the Live State Machine:
        fetch_live_box_scores()  — one call, all in-progress games.
        fetch_plays(game_id)     — play-by-play for one game (rate-limited).
        sync_game_states(...)    — update the games table with live state.
    """

    def __init__(self) -> None:
        self._client = httpx.AsyncClient(
            headers=_HEADERS,
            timeout=(3.0, 15.0),
        )

    # ------------------------------------------------------------------
    # Core live data
    # ------------------------------------------------------------------

    async def fetch_live_box_scores(self) -> List[dict]:
        """
        Fetch all currently in-progress MLB games with nested player stats.

        Returns a list of game dicts, each shaped like:
        {
          "id": <bdl_game_id>,
          "status": "In Progress",
          "inning": 5,
          "inning_half": "top",   # "top" | "bottom" | None
          "outs": 2,
          "home_team_score": 3,
          "visitor_team_score": 1,
          "home_team": { "id": ..., "abbreviation": ... },
          "visitor_team": { "id": ..., "abbreviation": ... },
          "home_team_stats": { <player_id>: {stat_dict} },
          "visitor_team_stats": { <player_id>: {stat_dict} },
        }

        Returns [] on any error (caller treats empty as "no live games").
        """
        cache_key = "bdl_mlb_live_box_scores"
        cached = cache.get(cache_key)
        if cached is not None:
            return cached

        try:
            body = await _get(self._client, "box_scores/live")
        except Exception as e:
            logger.warning("BDL live box_scores fetch failed: %s", e)
            return []

        games = []
        for raw in body.get("data", []):
            parsed = _parse_live_game(raw)
            if parsed:
                games.append(parsed)

        # Short TTL — 15s keeps us fresh without hammering the endpoint
        cache.set(cache_key, games, ttl_seconds=15)
        logger.debug("BDL live box scores: %d in-progress game(s)", len(games))
        return games

    async def fetch_plays(self, bdl_game_id: int) -> List[dict]:
        """
        Fetch play-by-play for a game. Rate-limited: at most once per
        _PLAYS_TTL_SECONDS per game to avoid quota bleed.

        Returns list of play dicts ordered by event sequence:
        {
          "inning": 5,
          "inning_half": "top",
          "outs": 1,
          "batter_id": <bdl_player_id>,
          "event": "Single",   # Home Run | Strikeout | Walk | etc.
          "description": "...",
        }
        """
        cache_key = f"bdl_mlb_plays_{bdl_game_id}"
        cached = cache.get(cache_key)
        if cached is not None:
            return cached

        try:
            plays = await _get_all_pages(
                self._client, "plays", params={"game_id": bdl_game_id}
            )
        except Exception as e:
            logger.warning("BDL plays fetch failed for game %s: %s", bdl_game_id, e)
            return []

        parsed = [_parse_play(p) for p in plays if p]
        cache.set(cache_key, parsed, ttl_seconds=_PLAYS_TTL_SECONDS)
        return parsed

    async def get_live_game(self, bdl_game_id: int) -> Optional[dict]:
        """
        Fetch a single game's live state. Lighter than the full box-score
        endpoint — useful when we only need inning/outs for one game.
        """
        try:
            body = await _get(self._client, f"games/{bdl_game_id}")
        except Exception as e:
            logger.warning("BDL game fetch failed for %s: %s", bdl_game_id, e)
            return None
        return _parse_live_game(body.get("data", {}))

    # ------------------------------------------------------------------
    # DB sync
    # ------------------------------------------------------------------

    async def sync_game_states(self) -> int:
        """
        Pull live box scores and write inning/outs/current_batter_slot
        back to the `games` table in the local DB.

        Also updates `status` to 'IN_PROGRESS' for games BDL reports as
        live, and accumulates running batter stat totals into
        `live_batter_stats` so the state machine can compute already-hit
        counts without re-querying BDL.

        Returns the number of games updated.
        """
        live_games = await self.fetch_live_box_scores()
        if not live_games:
            return 0

        updated = 0
        with get_db_connection() as conn:
            for g in live_games:
                bdl_id = g["id"]

                # Resolve the Odds-API game_id from bdl_game_id
                row = conn.execute(
                    "SELECT game_id FROM games WHERE bdl_game_id = ?", (bdl_id,)
                ).fetchone()
                if not row:
                    continue
                odds_game_id = row["game_id"]

                # Derive current batter slot from play-by-play
                current_slot = await self._derive_batter_slot(bdl_id, conn)

                conn.execute(
                    """UPDATE games
                       SET status = 'IN_PROGRESS',
                           inning = ?,
                           outs   = ?,
                           current_batter_slot = ?
                       WHERE game_id = ?""",
                    (
                        g.get("inning", 1),
                        g.get("outs", 0),
                        current_slot,
                        odds_game_id,
                    ),
                )

                # Write running batter stats so live_state_machine can read
                # "already accumulated" counts without extra API calls.
                _upsert_live_batter_stats(conn, odds_game_id, g)

                updated += 1

            conn.commit()

        logger.info("BDL live sync: updated %d in-progress game(s).", updated)
        return updated

    async def _derive_batter_slot(
        self, bdl_game_id: int, conn
    ) -> int:
        """
        Determine which lineup slot is currently at the plate using
        play-by-play data. Falls back to 1 if plays are unavailable.

        Strategy:
          1. Fetch ordered plays (cached 60s).
          2. Last play's batter_id → look up their batting_order in
             daily_lineups. That's the current slot.
        """
        plays = await self.fetch_plays(bdl_game_id)
        if not plays:
            return 1

        # Most recent play is last in the list
        last_batter_id = None
        for play in reversed(plays):
            if play.get("batter_id"):
                last_batter_id = play["batter_id"]
                break

        if last_batter_id is None:
            return 1

        # Map BDL player_id → lineup_position via players + daily_lineups
        player_row = conn.execute(
            "SELECT player_id FROM players WHERE bdl_player_id = ?",
            (last_batter_id,),
        ).fetchone()
        if not player_row:
            return 1

        lineup_row = conn.execute(
            "SELECT lineup_position FROM daily_lineups WHERE player_id = ?",
            (player_row["player_id"],),
        ).fetchone()
        return lineup_row["lineup_position"] if lineup_row else 1

    async def close(self) -> None:
        await self._client.aclose()


# ---------------------------------------------------------------------------
# Parse helpers
# ---------------------------------------------------------------------------

def _parse_live_game(raw: dict) -> Optional[dict]:
    """
    Normalise a BDL game/box-score dict into the shape the state machine
    and sync pipeline expect.

    BDL MLB game fields (confirmed from docs + live testing):
      id, status, inning, inning_half, outs,
      home_team_score, visitor_team_score,
      home_team { id, abbreviation, players: [...] },
      visitor_team { id, abbreviation, players: [...] }
    """
    if not raw:
        return None

    status = raw.get("status", "")
    # Only care about live games; skip Final / Scheduled / etc.
    if "progress" not in status.lower() and "live" not in status.lower():
        return None

    home_stats = _index_player_stats(raw.get("home_team", {}).get("players", []))
    away_stats = _index_player_stats(raw.get("visitor_team", {}).get("players", []))

    return {
        "id": raw.get("id"),
        "status": status,
        "inning": raw.get("inning") or 1,
        "inning_half": raw.get("inning_half"),   # "top" / "bottom"
        "outs": raw.get("outs") or 0,
        "home_team_score": raw.get("home_team_score") or 0,
        "visitor_team_score": raw.get("visitor_team_score") or 0,
        "home_team": {
            "id": raw.get("home_team", {}).get("id"),
            "abbreviation": raw.get("home_team", {}).get("abbreviation"),
        },
        "visitor_team": {
            "id": raw.get("visitor_team", {}).get("id"),
            "abbreviation": raw.get("visitor_team", {}).get("abbreviation"),
        },
        "home_team_stats": home_stats,
        "visitor_team_stats": away_stats,
    }


def _index_player_stats(players: List[dict]) -> Dict[int, dict]:
    """Index per-player box score stats by BDL player_id."""
    out: Dict[int, dict] = {}
    for p in players:
        player = p.get("player", {})
        pid = player.get("id")
        if not pid:
            continue
        out[pid] = {
            "at_bats":        p.get("ab") or p.get("at_bats") or 0,
            "hits":           p.get("h") or p.get("hits") or 0,
            "home_runs":      p.get("hr") or p.get("home_runs") or 0,
            "total_bases":    _calc_total_bases(p),
            "plate_appearances": (
                (p.get("ab") or p.get("at_bats") or 0)
                + (p.get("bb") or p.get("walks") or 0)
                + (p.get("hbp") or 0)
                + (p.get("sf") or 0)
            ),
            # Pitcher fields
            "strikeouts":     p.get("so") or p.get("strikeouts") or 0,
            "innings_pitched": p.get("ip") or p.get("innings_pitched") or 0.0,
            "earned_runs":    p.get("er") or p.get("earned_runs") or 0,
        }
    return out


def _calc_total_bases(p: dict) -> int:
    """Derive total_bases from singles/doubles/triples/HR when available."""
    hr = p.get("hr") or p.get("home_runs") or 0
    h3 = p.get("triples") or 0
    h2 = p.get("doubles") or 0
    h = p.get("h") or p.get("hits") or 0
    singles = max(0, h - h2 - h3 - hr)
    return singles + (h2 * 2) + (h3 * 3) + (hr * 4)


def _parse_play(raw: dict) -> dict:
    """Normalise a BDL play record."""
    return {
        "inning":      raw.get("inning"),
        "inning_half": raw.get("inning_half"),
        "outs":        raw.get("outs"),
        "batter_id":   (raw.get("batter") or {}).get("id"),
        "pitcher_id":  (raw.get("pitcher") or {}).get("id"),
        "event":       raw.get("event"),
        "description": raw.get("description"),
    }


def _upsert_live_batter_stats(conn, odds_game_id: str, game: dict) -> None:
    """
    Write running batter stat totals into live_batter_stats so the state
    machine can quickly read "already accumulated" counts.

    Schema expected (created by _migrate_live_batter_stats or init_db):
        live_batter_stats(game_id, bdl_player_id, hits, total_bases,
                          home_runs, plate_appearances, updated_at)
    """
    now = datetime.now(timezone.utc).isoformat()
    all_stats = {
        **game.get("home_team_stats", {}),
        **game.get("visitor_team_stats", {}),
    }
    for bdl_pid, stats in all_stats.items():
        try:
            conn.execute(
                """INSERT INTO live_batter_stats
                   (game_id, bdl_player_id, hits, total_bases, home_runs,
                    plate_appearances, updated_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?)
                   ON CONFLICT(game_id, bdl_player_id) DO UPDATE SET
                       hits             = excluded.hits,
                       total_bases      = excluded.total_bases,
                       home_runs        = excluded.home_runs,
                       plate_appearances = excluded.plate_appearances,
                       updated_at       = excluded.updated_at""",
                (
                    odds_game_id,
                    bdl_pid,
                    stats["hits"],
                    stats["total_bases"],
                    stats["home_runs"],
                    stats["plate_appearances"],
                    now,
                ),
            )
        except Exception as e:
            logger.debug("live_batter_stats upsert skipped (bdl_pid=%s): %s", bdl_pid, e)
