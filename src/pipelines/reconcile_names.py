"""Offline LLM pass to resolve player names the deterministic matcher can't.

Walks recent prop-snapshot player names (the names that drive bets); any that
don't resolve deterministically are handed to the LLM with the candidate roster
rows, and its pick is written into player_name_resolutions. The money path never
calls the LLM — it just reads the cache this fills. Fully fail-safe: with the LLM
off, unresolved names are simply left for a human / future run.
"""
from __future__ import annotations

import json
from datetime import timedelta

from src.clients.llm import OpenRouterClient
from src.data.db import get_db_connection
from src.data.player_resolver import resolve_player_id, normalize_name, cache_resolution
from src.utils.time_utils import utcnow
from src.utils.logging_utils import get_logger

logger = get_logger(__name__)

_SYSTEM = (
    "You match a sportsbook player name to the correct MLB player from a list of "
    "candidate roster rows. Account for nicknames, accents, and abbreviated first "
    "names (e.g. 'J. Soto' -> 'Juan Soto'). Respond with JSON only: "
    '{"player_id": <id of the matching candidate, or null if none clearly match>}.'
)


def _prompt(name: str, candidates: list[dict]) -> str:
    rows = [{"player_id": c["player_id"], "name": c["name"], "team": c["team"]} for c in candidates]
    return f"Sportsbook name: {name!r}\nCandidates:\n{json.dumps(rows)}"


async def reconcile_names(client: OpenRouterClient = None, days: int = 3) -> dict:
    """Resolve recent unmatched prop names via the LLM and cache the answers."""
    logger.info("Executing pipeline: reconcile_names")
    client = client or OpenRouterClient()
    cutoff = (utcnow() - timedelta(days=days)).isoformat()

    with get_db_connection() as conn:
        names = [r["player_name"] for r in conn.execute(
            "SELECT DISTINCT player_name FROM prop_snapshots WHERE timestamp >= ?",
            (cutoff,),
        ).fetchall() if r["player_name"]]

    checked = resolved = llm_used = 0
    for name in names:
        checked += 1
        with get_db_connection() as conn:
            pid, _ = resolve_player_id(conn, name)
            if pid is not None:
                continue  # already resolves deterministically / from cache
            target = normalize_name(name)
            last = target.split(" ")[-1] if target else ""
            candidates = [dict(r) for r in conn.execute(
                "SELECT p.player_id, p.name, COALESCE(t.abbreviation,'') AS team "
                "FROM players p LEFT JOIN teams t ON p.team_id = t.team_id "
                "WHERE p.name LIKE ? COLLATE NOCASE LIMIT 25",
                (f"%{last}%",),
            ).fetchall()]
        if not candidates or not client.available:
            continue
        result = await client.complete_json(_SYSTEM, _prompt(name, candidates))
        chosen = result.get("player_id") if isinstance(result, dict) else None
        valid_ids = {c["player_id"] for c in candidates}
        if chosen in valid_ids:
            with get_db_connection() as conn:
                cache_resolution(conn, name, int(chosen), "llm")
                conn.commit()
            llm_used += 1
            resolved += 1
            logger.info("Reconciled '%s' -> player_id %s via LLM.", name, chosen)

    summary = {"checked": checked, "llm_resolved": resolved, "llm_calls": llm_used}
    logger.info("reconcile_names complete: %s", summary)
    return summary
