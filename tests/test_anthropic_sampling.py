"""Guard against the 'temperature is deprecated for this model' 400 (2026-06-17).

Opus 4.7+ and the Fable/Mythos 5 families removed temperature/top_p/top_k;
sending temperature 400s and silently kicks synthesis to the DeepSeek fallback.
"""

import pytest

import src.analysis.claude_analyst as ca
from src.analysis.claude_analyst import (
    _anthropic_sampling_kwargs, _anthropic_thinking_kwargs,
    _engine_of, _synthesis_attempts_for, _deepseek_spec,
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


# ── 4-way bake-off: DeepSeek flash-thinking / pro-thinking arms ───────────────

@pytest.mark.parametrize("logical,api,thinking", [
    ("deepseek-v4-flash-thinking", "deepseek-v4-flash", True),
    ("deepseek-v4-pro-thinking",   "deepseek-v4-pro",   True),
    ("deepseek-v4-flash",          "deepseek-v4-flash", False),
    ("deepseek-v4-pro",            "deepseek-v4-pro",   False),
    (None,                         "deepseek-v4-flash", False),   # fallback default
])
def test_deepseek_spec_decodes_thinking_suffix(logical, api, thinking):
    assert _deepseek_spec(logical) == (api, thinking)


def test_thinking_ids_route_to_deepseek():
    # Provenance + hold-review pinning depend on these mapping to DeepSeek.
    assert _engine_of("deepseek-v4-flash-thinking") == "deepseek"
    assert _engine_of("deepseek-v4-pro-thinking") == "deepseek"


def test_attempts_pro_thinking_chosen_then_anthropic_fallback():
    # The logical id is preserved as the chosen attempt (so provenance records the
    # exact arm); the cross-engine fallback stays cheap flash non-thinking.
    assert _synthesis_attempts_for("deepseek-v4-pro-thinking", "claude-opus-4-8", "deepseek-v4-flash") == [
        ("deepseek", "deepseek-v4-pro-thinking"), ("anthropic", "claude-opus-4-8")]


def _capture_deepseek_create(monkeypatch):
    cap = {}

    class _Stream:
        def __enter__(self): return []          # no chunks → empty raw
        def __exit__(self, *a): return False

    class _Completions:
        def create(self, **kw):
            cap.update(kw)
            return _Stream()

    class _Client:
        chat = type("C", (), {"completions": _Completions()})()

    monkeypatch.setattr(ca, "_get_deepseek_analyst_client", lambda: _Client())
    return cap


def test_call_deepseek_threads_pro_and_thinking(monkeypatch):
    cap = _capture_deepseek_create(monkeypatch)
    ca._call_deepseek_analyst("hi", model="deepseek-v4-pro", thinking=True)
    assert cap["model"] == "deepseek-v4-pro"
    assert cap["extra_body"] == ca._DEEPSEEK_THINKING_ON


def test_call_deepseek_defaults_to_flash_nonthinking(monkeypatch):
    cap = _capture_deepseek_create(monkeypatch)
    ca._call_deepseek_analyst("hi")
    assert cap["model"] == ca._DEEPSEEK_ANALYST_MODEL
    assert cap["extra_body"] == ca._DEEPSEEK_THINKING_OFF
