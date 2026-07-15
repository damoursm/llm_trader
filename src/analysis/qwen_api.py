"""Qwen route adapter — DashScope direct vs OpenRouter (2026-07-12).

The same qwen3.7-max model is reachable two ways, and the request dialect for
THINKING differs per route (verified live on both):

  • DashScope compatible-mode (direct Alibaba): ``extra_body={"enable_thinking":
    bool}``; ``thinking_budget`` omitted → the model's MAX chain-of-thought.
    Implicit prefix caching is automatic (hits at 20% of input).
  • OpenRouter (openrouter.ai/api/v1): the unified ``reasoning`` parameter —
    ``{"reasoning": {"enabled": bool}}`` — which OpenRouter translates to
    Alibaba's thinking fields; enabled-with-no-budget = the provider default,
    which IS the max chain-of-thought (verified 2026-07-12: reasoning_tokens
    flowed at the billed $1.25/$3.75 promo rates). DashScope's native
    ``enable_thinking`` is NOT guaranteed to pass through — always use
    ``reasoning`` on this route. NO implicit caching for Alibaba via OpenRouter
    (explicit cache_control only: 1.25x write / 0.1x read, 5-min TTL).

The route is derived from ``settings.qwen_base_url`` (no extra knob): pointing
it at openrouter.ai flips the dialect, the model id (``settings.qwen_model``,
OpenRouter needs the ``qwen/`` prefix), and the synthesis-prefix cache strategy
(explicit markers — see ``claude_analyst._qwen_user_content``). Flipping back to
direct DashScope is a 3-line .env change (base_url, key, model).
"""

from __future__ import annotations

from config.settings import settings


def is_openrouter() -> bool:
    """True when the Qwen route is OpenRouter (drives dialect + cache strategy)."""
    return "openrouter" in (settings.qwen_base_url or "").lower()


def qwen_model_id() -> str:
    """The configured Qwen model id for the active route (``qwen/qwen3.7-max``
    on OpenRouter, ``qwen3.7-max`` on DashScope direct)."""
    return settings.qwen_model


def thinking_body(thinking: bool) -> dict:
    """The ``extra_body`` that toggles Qwen reasoning on the ACTIVE route.

    Maximum thinking either way: no budget/effort is ever sent, so the model
    reasons at its default = maximum chain-of-thought budget."""
    if is_openrouter():
        return {"reasoning": {"enabled": bool(thinking)}}
    return {"enable_thinking": bool(thinking)}
