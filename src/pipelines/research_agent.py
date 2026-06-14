"""Natural-language research agent over the BDL-sourced database.

`python main.py ask "how has Gerrit Cole's strikeout rate trended lately?"`

The LLM is given read-only tools that query the bot's local store (game logs,
players, games, injuries, platoon splits — all sourced from BallDontLie). It
composes those tools to answer; it is instructed to report only what the tools
return and never to invent stats or give a probability/edge. This is the
"hand in hand with BDL" surface: BDL provides the data, the LLM navigates it.

Fail-safe: if the LLM layer is unavailable or errors, `ask` returns a clear
message instead of raising.
"""
from __future__ import annotations

import json
from typing import Any

from src.clients.llm import OpenRouterClient
from src.config import OPENROUTER_RESEARCH_MODEL
from src.data.db import get_db_connection
from src.utils.logging_utils import get_logger

logger = get_logger(__name__)

_MAX_ITERS = 6

_PITCHER_STATS = {"strikeouts", "earned_runs", "innings_pitched", "walks", "hits_allowed"}
_BATTER_STATS = {"hits", "total_bases", "home_runs", "doubles", "triples", "rbis", "runs", "walks", "strikeouts"}

_SYSTEM = (
    "You are an MLB research assistant for a prop-betting analyst. Answer using "
    "ONLY the data returned by the tools, which read the bot's BallDontLie-sourced "
    "database. Never invent statistics. If a tool returns no data, say so plainly. "
    "Do NOT give win probabilities, betting edges, or stake advice — only describe "
    "what the data shows. Be concise and quantitative; cite the numbers you used."
)

# OpenAI-compatible tool schemas.
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "find_player",
            "description": "Resolve a player name to id(s). Returns up to 5 matches "
                           "with player_id, name, team, position, bats, throws.",
            "parameters": {
                "type": "object",
                "properties": {"name": {"type": "string"}},
                "required": ["name"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "player_recent_stats",
            "description": "Recent game-by-game values of a stat for a player, plus "
                           "summary (mean/min/max/n). Pitcher stats: strikeouts, "
                           "earned_runs, innings_pitched, walks, hits_allowed. Batter "
                           "stats: hits, total_bases, home_runs, doubles, triples, rbis, runs.",
            "parameters": {
                "type": "object",
                "properties": {
                    "player_id": {"type": "integer"},
                    "stat": {"type": "string"},
                    "last_n": {"type": "integer", "description": "default 10"},
                },
                "required": ["player_id", "stat"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "player_vs_hand_split",
            "description": "A batter's per-PA rate for a market vs a pitcher hand "
                           "('L' or 'R'). Markets: batter_hits, batter_total_bases, batter_home_runs.",
            "parameters": {
                "type": "object",
                "properties": {
                    "player_id": {"type": "integer"},
                    "market": {"type": "string"},
                    "vs_hand": {"type": "string"},
                },
                "required": ["player_id", "market", "vs_hand"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "todays_games",
            "description": "Upcoming (not-yet-completed) games with matchup and first-pitch time.",
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "player_injury_status",
            "description": "Latest injury report status for a player name, if any.",
            "parameters": {
                "type": "object",
                "properties": {"player_name": {"type": "string"}},
                "required": ["player_name"],
            },
        },
    },
]


# ---------------------------------------------------------------------------
# Tool implementations (read-only, DB-backed)
# ---------------------------------------------------------------------------

def _find_player(conn, name: str) -> Any:
    rows = conn.execute(
        """SELECT p.player_id, p.name, COALESCE(t.abbreviation,'') AS team,
                  p.position, p.bats, p.throws
           FROM players p LEFT JOIN teams t ON p.team_id = t.team_id
           WHERE p.name = ? COLLATE NOCASE OR p.name LIKE ? COLLATE NOCASE
           LIMIT 5""",
        (name, f"%{name}%"),
    ).fetchall()
    return [dict(r) for r in rows] or {"note": f"no player matching '{name}'"}


def _player_recent_stats(conn, player_id: int, stat: str, last_n: int = 10) -> Any:
    last_n = max(1, min(int(last_n or 10), 40))
    if stat in _PITCHER_STATS:
        table = "pitcher_game_logs"
    elif stat in _BATTER_STATS:
        table = "batter_game_logs"
    else:
        return {"error": f"unknown stat '{stat}'"}
    # stat is validated against a fixed allowlist above, safe to interpolate.
    rows = conn.execute(
        f"SELECT date, {stat} AS v FROM {table} "
        f"WHERE player_id = ? AND date != '' ORDER BY date DESC LIMIT ?",
        (player_id, last_n),
    ).fetchall()
    if not rows:
        return {"note": f"no {stat} logs for player_id {player_id}"}
    vals = [float(r["v"] or 0) for r in rows]
    return {
        "stat": stat,
        "games": [{"date": r["date"], "value": r["v"]} for r in rows],
        "summary": {
            "n": len(vals), "mean": round(sum(vals) / len(vals), 3),
            "min": min(vals), "max": max(vals),
        },
    }


def _player_vs_hand_split(conn, player_id: int, market: str, vs_hand: str) -> Any:
    row = conn.execute(
        "SELECT rate_per_pa, n_pa, season FROM batter_platoon_splits "
        "WHERE player_id = ? AND market = ? AND vs_hand = ? "
        "ORDER BY season DESC LIMIT 1",
        (player_id, market, (vs_hand or "").upper()),
    ).fetchone()
    if not row:
        return {"note": "no platoon split data for that player/market/hand"}
    return {"market": market, "vs_hand": vs_hand.upper(),
            "rate_per_pa": row["rate_per_pa"], "n_pa": row["n_pa"], "season": row["season"]}


def _todays_games(conn) -> Any:
    rows = conn.execute(
        "SELECT away_team, home_team, game_time FROM games "
        "WHERE status NOT IN ('COMPLETED','IN_PROGRESS') ORDER BY game_time LIMIT 30"
    ).fetchall()
    return [{"matchup": f"{r['away_team']} @ {r['home_team']}", "first_pitch": r["game_time"]}
            for r in rows] or {"note": "no upcoming games"}


def _player_injury_status(conn, player_name: str) -> Any:
    row = conn.execute(
        "SELECT status, date FROM injury_reports WHERE player_name LIKE ? COLLATE NOCASE "
        "ORDER BY date DESC LIMIT 1",
        (f"%{player_name}%",),
    ).fetchone()
    if not row:
        return {"status": "no injury report found (assume healthy)"}
    return {"status": row["status"], "as_of": row["date"]}


_DISPATCH = {
    "find_player": _find_player,
    "player_recent_stats": _player_recent_stats,
    "player_vs_hand_split": _player_vs_hand_split,
    "todays_games": _todays_games,
    "player_injury_status": _player_injury_status,
}


def _execute_tool(name: str, args: dict) -> Any:
    fn = _DISPATCH.get(name)
    if fn is None:
        return {"error": f"unknown tool {name}"}
    try:
        with get_db_connection() as conn:
            return fn(conn, **args)
    except Exception as e:  # noqa: BLE001 — tool errors are reported to the model, not raised
        logger.warning("research tool %s failed: %s", name, e)
        return {"error": str(e)}


# ---------------------------------------------------------------------------
# Agent loop
# ---------------------------------------------------------------------------

async def ask(question: str, client: OpenRouterClient = None) -> str:
    """Answer a natural-language MLB research question. Returns a string always."""
    client = client or OpenRouterClient(model=OPENROUTER_RESEARCH_MODEL)
    if not client.available:
        return ("LLM layer is not configured. Set OPENROUTER_API_KEY (and "
                "LLM_ENABLED=true) to use the research agent.")

    messages = [
        {"role": "system", "content": _SYSTEM},
        {"role": "user", "content": question},
    ]
    for _ in range(_MAX_ITERS):
        data = await client.chat(messages, tools=TOOLS, tool_choice="auto", temperature=0.1)
        msg = OpenRouterClient._first_message(data)
        if msg is None:
            return "Sorry — the research agent is unavailable right now."
        tool_calls = msg.get("tool_calls")
        # Append the assistant turn verbatim so tool replies thread correctly.
        messages.append({
            "role": "assistant",
            "content": msg.get("content") or "",
            **({"tool_calls": tool_calls} if tool_calls else {}),
        })
        if not tool_calls:
            return msg.get("content") or "(no answer)"
        for tc in tool_calls:
            fn = tc.get("function", {})
            try:
                args = json.loads(fn.get("arguments") or "{}")
            except json.JSONDecodeError:
                args = {}
            result = _execute_tool(fn.get("name", ""), args)
            messages.append({
                "role": "tool",
                "tool_call_id": tc.get("id"),
                "content": json.dumps(result, default=str)[:4000],
            })
    return "I couldn't finish researching that within the step limit — try narrowing the question."
