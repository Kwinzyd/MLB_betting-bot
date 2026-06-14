"""LLM injury enrichment: messy injury text -> structured availability signal.

BDL/CBS give a status string ("Day-To-Day", "Game-Time Decision", "10-Day IL",
"Questionable - wrist") that varies by source and is hard to gate on reliably.
This pipeline asks the LLM to normalize each into a structured signal:
  { play_status: active|questionable|out, play_probability: 0..1, impact_summary }
stored in player_injury_signals. The deterministic money path then reads
play_probability to conservatively skip likely-scratched players and surfaces
impact_summary in alerts. The LLM only classifies text — it never sets a
projection number, probability of a bet hitting, or stake.

Fail-safe: with the LLM layer off or erroring, no signal is written and the
existing injury-status gate is unaffected.
"""
from __future__ import annotations

import json

from src.clients.llm import OpenRouterClient
from src.data.db import get_db_connection
from src.utils.time_utils import get_eastern_local_date, utcnow
from src.utils.logging_utils import get_logger

logger = get_logger(__name__)

_SYSTEM = (
    "You normalize MLB injury report entries into a structured availability call "
    "for a given game day. Respond with JSON only: "
    '{"play_status": "active"|"questionable"|"out", '
    '"play_probability": number 0..1, "impact_summary": short string}. '
    "play_probability is your estimate the player APPEARS in the game. "
    "IL designations are 'out' (~0.02). 'Day-To-Day'/'Questionable'/'Game-Time "
    "Decision' are 'questionable' (~0.4-0.7 depending on wording). No designation "
    "or 'Active' is 'active' (~0.97). impact_summary is one short clause a bettor "
    "can read (e.g. 'wrist, may sit vs RHP'). Do not invent details not implied "
    "by the status text."
)


def _build_prompt(player_name: str, status: str, injury_type: str) -> str:
    return (
        f"Player: {player_name}\n"
        f"Status: {status or '(none)'}\n"
        f"Injury type: {injury_type or '(none)'}"
    )


async def enrich_injuries(client: OpenRouterClient = None) -> dict:
    """Normalize today's injury reports into structured signals. Returns a summary."""
    logger.info("Executing pipeline: enrich_injuries")
    client = client or OpenRouterClient()
    if not client.available:
        logger.info("LLM layer disabled; skipping injury enrichment.")
        return {"enriched": 0, "skipped": "llm_disabled"}

    today = str(get_eastern_local_date())
    with get_db_connection() as conn:
        rows = [dict(r) for r in conn.execute(
            """SELECT ir.player_name, ir.status, ir.injury_type,
                      (SELECT player_id FROM players p
                       WHERE p.name = ir.player_name COLLATE NOCASE LIMIT 1) AS player_id
               FROM injury_reports ir WHERE ir.date = ?""",
            (today,),
        ).fetchall()]

    enriched = 0
    for r in rows:
        if r["player_id"] is None:
            continue  # can't key a signal without a player_id
        signal = await client.complete_json(
            _SYSTEM, _build_prompt(r["player_name"], r["status"], r["injury_type"]),
        )
        if not isinstance(signal, dict):
            continue  # fail-safe: no signal -> existing gate handles it
        play_status = str(signal.get("play_status", "")).lower()
        if play_status not in ("active", "questionable", "out"):
            play_status = "questionable"
        try:
            prob = float(signal.get("play_probability"))
        except (TypeError, ValueError):
            prob = None
        if prob is None:
            prob = {"active": 0.97, "questionable": 0.5, "out": 0.02}[play_status]
        prob = max(0.0, min(1.0, prob))

        with get_db_connection() as conn:
            conn.execute(
                """INSERT INTO player_injury_signals
                   (player_id, player_name, date, play_status, play_probability,
                    impact_summary, source_status, created_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                   ON CONFLICT(player_id, date) DO UPDATE SET
                       play_status=excluded.play_status,
                       play_probability=excluded.play_probability,
                       impact_summary=excluded.impact_summary,
                       source_status=excluded.source_status,
                       created_at=excluded.created_at""",
                (
                    r["player_id"], r["player_name"], today, play_status, prob,
                    str(signal.get("impact_summary", ""))[:200], r["status"],
                    utcnow().isoformat(),
                ),
            )
            conn.commit()
        enriched += 1

    logger.info("Injury enrichment complete: %d signals written for %s.", enriched, today)
    return {"enriched": enriched, "date": today}


def get_injury_signal(conn, player_id: int, date: str) -> dict | None:
    """Fetch today's LLM injury signal for a player, or None. Used by scan_props."""
    if not player_id:
        return None
    row = conn.execute(
        "SELECT play_status, play_probability, impact_summary FROM player_injury_signals "
        "WHERE player_id = ? AND date = ?",
        (player_id, date),
    ).fetchone()
    return dict(row) if row else None
