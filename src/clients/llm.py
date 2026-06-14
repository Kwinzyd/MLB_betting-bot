"""OpenRouter LLM client — the qualitative layer that complements BDL.

Design contract:
  - The LLM never produces projections, probabilities, edges, or stake sizes.
    It turns unstructured text into STRUCTURED context and answers research
    questions; the quant core stays deterministic and auditable.
  - Every entry point is FAIL-SAFE. With no API key, LLM_ENABLED=false, a
    timeout, a 4xx/5xx, or unparseable output, the public methods return None
    (or []), never raising into the pipeline. Callers fall back to non-LLM
    behavior. With real betting enabled, the LLM must never block the bot.
  - OpenRouter exposes an OpenAI-compatible Chat Completions API, so this is a
    thin httpx client (no extra SDK dependency), mirroring the other clients'
    circuit-breaker + cache + retry patterns.
"""
from __future__ import annotations

import hashlib
import json
import re
from typing import Any, Optional

import httpx

from src.config import (
    OPENROUTER_API_KEY, OPENROUTER_BASE_URL, OPENROUTER_MODEL,
    LLM_ENABLED, LLM_TIMEOUT_SECONDS, LLM_MAX_TOKENS, LLM_CACHE_TTL_SECONDS,
)
from src.data.cache import cache
from src.utils.circuit_breaker import AsyncCircuitBreaker, CircuitBreakerOpenException
from src.utils.logging_utils import get_logger

logger = get_logger(__name__)

# Fast-fail when OpenRouter is unhealthy so enrichment never stalls a slate.
_llm_circuit_breaker = AsyncCircuitBreaker(
    failure_threshold=4, recovery_timeout=120.0, exceptions=(httpx.HTTPError,),
)

_JSON_FENCE = re.compile(r"```(?:json)?\s*(.*?)\s*```", re.DOTALL)


def _extract_json(text: str) -> Optional[Any]:
    """Best-effort parse of a model's JSON reply.

    Cheap models sometimes wrap JSON in ``` fences or add prose. Try a direct
    parse, then a fenced block, then the first balanced {...} / [...] span.
    Returns None when nothing parses.
    """
    if not text:
        return None
    for candidate in (text, *(m.group(1) for m in _JSON_FENCE.finditer(text))):
        try:
            return json.loads(candidate)
        except (json.JSONDecodeError, TypeError):
            continue
    # Fall back to the first {...} or [...] span.
    for opener, closer in (("{", "}"), ("[", "]")):
        start = text.find(opener)
        end = text.rfind(closer)
        if 0 <= start < end:
            try:
                return json.loads(text[start:end + 1])
            except json.JSONDecodeError:
                pass
    return None


class OpenRouterClient:
    """Thin async OpenRouter chat client. All public methods are fail-safe."""

    def __init__(self, model: str = None):
        self.model = model or OPENROUTER_MODEL
        self.base_url = OPENROUTER_BASE_URL.rstrip("/")
        self.api_key = OPENROUTER_API_KEY

    @property
    def available(self) -> bool:
        return bool(LLM_ENABLED and self.api_key)

    # ------------------------------------------------------------------
    # Low-level chat
    # ------------------------------------------------------------------
    @_llm_circuit_breaker
    async def _post(self, payload: dict) -> Optional[dict]:
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            # OpenRouter attribution headers (optional but recommended).
            "HTTP-Referer": "https://github.com/mlb-betting-bot",
            "X-Title": "MLB Prop Betting Bot",
        }
        async with httpx.AsyncClient(timeout=LLM_TIMEOUT_SECONDS) as client:
            resp = await client.post(
                f"{self.base_url}/chat/completions", headers=headers, json=payload,
            )
            resp.raise_for_status()
            return resp.json()

    async def chat(self, messages: list[dict], *, model: str = None,
                   temperature: float = 0.2, max_tokens: int = None,
                   response_format: dict = None, tools: list = None,
                   tool_choice: Any = None) -> Optional[dict]:
        """Raw chat completion. Returns the OpenRouter response dict or None.

        Returns None (never raises) when the LLM layer is unavailable or the
        request fails — including circuit-breaker-open.
        """
        if not self.available:
            return None
        payload: dict = {
            "model": model or self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens or LLM_MAX_TOKENS,
        }
        if response_format:
            payload["response_format"] = response_format
        if tools:
            payload["tools"] = tools
            if tool_choice is not None:
                payload["tool_choice"] = tool_choice
        try:
            data = await self._post(payload)
        except CircuitBreakerOpenException:
            logger.debug("LLM circuit open; skipping call.")
            return None
        except Exception as e:  # noqa: BLE001 — fail-safe boundary
            logger.warning("LLM call failed: %s", e)
            return None
        usage = (data or {}).get("usage") or {}
        if usage:
            logger.debug(
                "LLM usage: model=%s prompt=%s completion=%s",
                payload["model"], usage.get("prompt_tokens"), usage.get("completion_tokens"),
            )
        return data

    @staticmethod
    def _first_message(data: Optional[dict]) -> Optional[dict]:
        if not data:
            return None
        choices = data.get("choices") or []
        if not choices:
            return None
        return choices[0].get("message")

    # ------------------------------------------------------------------
    # High-level helpers (cached, fail-safe)
    # ------------------------------------------------------------------
    def _cache_key(self, kind: str, system: str, user: str, model: str) -> str:
        h = hashlib.sha256(f"{kind}|{model}|{system}|{user}".encode()).hexdigest()[:32]
        return f"llm_{kind}_{h}"

    async def complete_text(self, system: str, user: str, *, model: str = None,
                            temperature: float = 0.2, use_cache: bool = True) -> Optional[str]:
        """Return the model's text reply, or None. Caches identical prompts."""
        if not self.available:
            return None
        mdl = model or self.model
        ck = self._cache_key("text", system, user, mdl)
        if use_cache:
            cached = cache.get(ck)
            if cached is not None:
                return cached
        data = await self.chat(
            [{"role": "system", "content": system}, {"role": "user", "content": user}],
            model=mdl, temperature=temperature,
        )
        msg = self._first_message(data)
        text = (msg or {}).get("content") if msg else None
        if text and use_cache:
            cache.set(ck, text, ttl_seconds=LLM_CACHE_TTL_SECONDS)
        return text

    async def complete_json(self, system: str, user: str, *, model: str = None,
                            temperature: float = 0.0, use_cache: bool = True) -> Optional[Any]:
        """Return a parsed JSON object/array from the model, or None.

        Requests JSON mode and tolerantly parses the reply. Deterministic by
        default (temperature 0) so cached enrichment is stable.
        """
        if not self.available:
            return None
        mdl = model or self.model
        ck = self._cache_key("json", system, user, mdl)
        if use_cache:
            cached = cache.get(ck)
            if cached is not None:
                return cached
        data = await self.chat(
            [{"role": "system", "content": system}, {"role": "user", "content": user}],
            model=mdl, temperature=temperature,
            response_format={"type": "json_object"},
        )
        msg = self._first_message(data)
        parsed = _extract_json((msg or {}).get("content", "")) if msg else None
        if parsed is not None and use_cache:
            cache.set(ck, parsed, ttl_seconds=LLM_CACHE_TTL_SECONDS)
        return parsed
