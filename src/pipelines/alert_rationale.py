"""LLM-written one-line rationale for a prop alert.

Synthesizes the factors the quant pipeline ALREADY computed (edge, model vs
book probability, park, platoon, weather, umpire, injury context) into a single
human-readable sentence so you can sanity-check a pick before placing it. The
LLM explains the existing numbers — it never produces new ones, and it never
changes the edge or the bet decision.

Fail-safe: returns None when the LLM layer is off or errors, in which case the
alert is sent without a rationale line.
"""
from __future__ import annotations

import json
from typing import Optional

from src.clients.llm import OpenRouterClient
from src.utils.logging_utils import get_logger

logger = get_logger(__name__)

_SYSTEM = (
    "You write a single concise sentence explaining why a baseball prop bet may "
    "have value, for an experienced bettor. Use ONLY the factors provided. Do not "
    "invent statistics, do not state probabilities or odds the data didn't give "
    "you, and do not tell the user whether to bet. One sentence, <= 30 words."
)


def _facts(edge: dict, context: dict) -> dict:
    """Pull the human-relevant, already-computed factors for the prompt."""
    model_ctx = context.get("model_context") or {}
    facts = {
        "player": context.get("player_name"),
        "market": context.get("market"),
        "side": context.get("side"),
        "line": context.get("line"),
        "edge_pct": edge.get("edge_pct"),
        "model_prob": edge.get("model_prob"),
        "book_implied": edge.get("book_implied"),
        "matchup": f"{context.get('away_team')} @ {context.get('home_team')}",
    }
    # Optional context the projection attached — include only when present.
    for k in ("park_adj", "platoon_adj", "opp_k_rate", "ump_k_factor",
              "weather", "llm_injury_summary", "bp_adj", "projected_mean"):
        if model_ctx.get(k) is not None:
            facts[k] = model_ctx[k]
    return {k: v for k, v in facts.items() if v is not None}


async def generate_alert_rationale(edge: dict, context: dict,
                                   client: OpenRouterClient = None) -> Optional[str]:
    """Return a one-sentence rationale for the alert, or None (fail-safe)."""
    client = client or OpenRouterClient()
    if not client.available:
        return None
    facts = _facts(edge, context)
    text = await client.complete_text(
        _SYSTEM,
        "Explain why this prop may have value:\n" + json.dumps(facts, default=str),
        temperature=0.3,
    )
    if not text:
        return None
    return text.strip().strip('"')[:300]
