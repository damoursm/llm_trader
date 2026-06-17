"""Guard against the 'temperature is deprecated for this model' 400 (2026-06-17).

Opus 4.7+ and the Fable/Mythos 5 families removed temperature/top_p/top_k;
sending temperature 400s and silently kicks synthesis to the DeepSeek fallback.
"""

import pytest

from src.analysis.claude_analyst import (
    _anthropic_sampling_kwargs, _anthropic_thinking_kwargs,
    _engine_of, _synthesis_attempts_for,
)


@pytest.mark.parametrize("model", [
    "claude-opus-4-7",
    "claude-opus-4-8",
    "claude-opus-4-8-20260101",
    "claude-opus-5-0",        # future Opus — still no sampling params
    "claude-fable-5",
    "claude-mythos-5",
])
def test_newer_models_omit_temperature(model):
    assert _anthropic_sampling_kwargs(model) == {}


@pytest.mark.parametrize("model", [
    "claude-haiku-4-5-20251001",   # the default analyst model
    "claude-sonnet-4-6",
    "claude-opus-4-6",
    "claude-opus-4-5",
    "claude-opus-4-1-20250805",
    "claude-sonnet-4-5",
])
def test_older_models_keep_temperature(model):
    assert _anthropic_sampling_kwargs(model) == {"temperature": 0}


def test_empty_or_unknown_defaults_to_temperature():
    # Unknown/empty → keep temperature (older behavior); a genuine 400 still
    # degrades safely via the engine fallback.
    assert _anthropic_sampling_kwargs("") == {"temperature": 0}
    assert _anthropic_sampling_kwargs("some-other-model") == {"temperature": 0}


@pytest.mark.parametrize("model", [
    "claude-opus-4-6", "claude-opus-4-7", "claude-opus-4-8",
    "claude-sonnet-4-6", "claude-fable-5", "claude-mythos-5",
])
def test_adaptive_thinking_on_supported_models(model):
    assert _anthropic_thinking_kwargs(model) == {"thinking": {"type": "adaptive"}}


@pytest.mark.parametrize("model", [
    "claude-haiku-4-5-20251001",   # default analyst model — no adaptive thinking
    "claude-sonnet-4-5",
    "claude-opus-4-5",
    "",
])
def test_adaptive_thinking_omitted_on_unsupported_models(model):
    assert _anthropic_thinking_kwargs(model) == {}


# ── 3-way synthesis A/B bake-off routing ─────────────────────────────────────

def test_engine_of_maps_provider():
    assert _engine_of("deepseek-v4-flash") == "deepseek"
    assert _engine_of("claude-opus-4-8") == "anthropic"
    assert _engine_of("claude-haiku-4-5-20251001") == "anthropic"


@pytest.mark.parametrize("chosen", ["claude-opus-4-8", "claude-haiku-4-5-20251001"])
def test_attempts_anthropic_chosen_then_deepseek_fallback(chosen):
    # The chosen Claude model runs first (so it's the sampled+recorded engine);
    # DeepSeek is the resilience fallback.
    assert _synthesis_attempts_for(chosen, "claude-opus-4-8", "deepseek-v4-flash") == [
        ("anthropic", chosen), ("deepseek", "deepseek-v4-flash")]


def test_attempts_deepseek_chosen_then_anthropic_fallback():
    assert _synthesis_attempts_for("deepseek-v4-flash", "claude-opus-4-8", "deepseek-v4-flash") == [
        ("deepseek", "deepseek-v4-flash"), ("anthropic", "claude-opus-4-8")]
