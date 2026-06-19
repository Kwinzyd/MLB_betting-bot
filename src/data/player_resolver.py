"""Robust player-name -> player_id resolution.

The pipeline matched names with `players WHERE name LIKE '%X%'` taking the first
row with no ordering — which can grade or project the WRONG player when names
collide or differ by diacritics/punctuation ("José Ramírez" vs "Jose Ramirez",
"J. Soto" vs "Juan Soto"). This resolver is deterministic and synchronous (safe
for the money path):

    cache -> exact (NOCASE) -> accent/suffix-normalized unique match

Genuinely ambiguous names (two players matching) are left for the offline LLM
reconciliation pass (src/pipelines/reconcile_names.py), which writes its answer
back into the cache so the next lookup is a deterministic hit.
"""
from __future__ import annotations

import re
import unicodedata
from typing import Optional, Tuple

_SUFFIX = re.compile(r"\b(jr|sr|ii|iii|iv)\b")


def normalize_name(name: str) -> str:
    """Lowercase, strip accents/punctuation/suffixes, collapse whitespace."""
    if not name:
        return ""
    s = unicodedata.normalize("NFKD", name)
    s = "".join(c for c in s if not unicodedata.combining(c))  # drop accents
    s = s.lower()
    s = re.sub(r"[^a-z0-9 ]", " ", s)         # punctuation -> space
    s = _SUFFIX.sub(" ", s)                    # drop jr/sr/iii...
    return re.sub(r"\s+", " ", s).strip()


def cache_resolution(conn, query: str, player_id: Optional[int], method: str) -> None:
    """Persist a resolution (including a confirmed None to avoid re-asking)."""
    from src.utils.time_utils import utcnow
    conn.execute(
        """INSERT INTO player_name_resolutions (query, player_id, method, created_at)
           VALUES (?, ?, ?, ?)
           ON CONFLICT(query) DO UPDATE SET
               player_id=excluded.player_id, method=excluded.method,
               created_at=excluded.created_at""",
        (query, player_id, method, utcnow().isoformat()),
    )


def resolve_player_id(conn, name: str) -> Tuple[Optional[int], Optional[str]]:
    """Resolve a name to (player_id, method). Returns (None, reason) on failure.

    Deterministic only — never calls the LLM. method is one of:
    cache | exact | normalized | None ('ambiguous' / 'no_match' reason).
    """
    if not name or not name.strip():
        return None, None

    # 1. Resolution cache (also caches LLM answers from the reconcile pass).
    row = conn.execute(
        "SELECT player_id FROM player_name_resolutions WHERE query = ? COLLATE NOCASE",
        (name,),
    ).fetchone()
    if row is not None:
        return (row["player_id"], "cache") if row["player_id"] is not None else (None, "cached_miss")

    # 2. Exact (case-insensitive).
    row = conn.execute(
        "SELECT player_id FROM players WHERE name = ? COLLATE NOCASE", (name,)
    ).fetchone()
    if row:
        return row["player_id"], "exact"

    # 3. Accent/suffix-normalized unique match. SQLite LIKE is accent-sensitive
    #    (a query 'ramirez' won't match a stored 'Ramírez'), so normalize every
    #    player name in Python. Only runs when exact failed, and the players
    #    table is small, so the full scan is cheap.
    target = normalize_name(name)
    if not target:
        return None, "no_match"
    candidates = conn.execute("SELECT player_id, name FROM players").fetchall()
    matches = [c for c in candidates if normalize_name(c["name"]) == target]
    if len(matches) == 1:
        return matches[0]["player_id"], "normalized"
    if len(matches) > 1:
        return None, "ambiguous"
    return None, "no_match"
