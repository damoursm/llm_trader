"""Guard against the 'temperature is deprecated for this model' 400 (2026-06-17).

Opus 4.7+ and the Fable/Mythos 5 families removed temperature/top_p/top_k;
sending temperature 400s and silently kicks synthesis to the DeepSeek fallback.
"""

import pytest

import src.analysis.claude_analyst as ca
from src.analysis.claude_analyst import (
    _anthropic_sampling_kwargs, _anthropic_thinking_kwargs,
    _engine_of, _synthesis_attempts_for, _deepseek_spec, _qwen_spec,
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
    assert _engine_of("qwen3.7-max") == "qwen"
    assert _engine_of("qwen3.7-max-thinking") == "qwen"


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


# ── Qwen synthesis arm (2026-07-11 primary) ──────────────────────────────────

@pytest.mark.parametrize("logical,api,thinking", [
    ("qwen3.7-max-thinking", "qwen3.7-max", True),
    ("qwen3.7-max",          "qwen3.7-max", False),
    (None,                   "qwen3.7-max", False),   # default when blank
])
def test_qwen_spec_decodes_thinking_suffix(logical, api, thinking):
    assert _qwen_spec(logical) == (api, thinking)


def test_attempts_qwen_chosen_then_deepseek_fallback():
    # Qwen (the primary) runs first; DeepSeek is the resilient error fallback.
    assert _synthesis_attempts_for("qwen3.7-max-thinking", "claude-opus-4-8", "deepseek-v4-flash") == [
        ("qwen", "qwen3.7-max-thinking"), ("deepseek", "deepseek-v4-flash")]


def _capture_qwen_create(monkeypatch):
    cap = {}

    class _Stream:
        def __enter__(self): return []
        def __exit__(self, *a): return False

    class _Completions:
        def create(self, **kw):
            cap.update(kw)
            return _Stream()

    class _Client:
        chat = type("C", (), {"completions": _Completions()})()

    monkeypatch.setattr(ca, "_get_qwen_analyst_client", lambda: _Client())
    return cap


def test_call_qwen_threads_model_and_thinking(monkeypatch):
    cap = _capture_qwen_create(monkeypatch)
    ca._call_qwen_analyst("hi", model="qwen3.7-max", thinking=True)
    assert cap["model"] == "qwen3.7-max"
    assert cap["extra_body"] == {"enable_thinking": True}
    assert cap["stream"] is True


def test_call_qwen_defaults_to_thinking(monkeypatch):
    cap = _capture_qwen_create(monkeypatch)
    ca._call_qwen_analyst("hi")
    assert cap["model"] == ca.settings.qwen_model
    assert cap["extra_body"] == {"enable_thinking": True}


# ── Maximum-thinking policy (llm_max_thinking) ───────────────────────────────

def test_max_thinking_forces_qwen_thinking_and_omits_budget(monkeypatch):
    """Under the flag, even a thinking=False Qwen call reasons — and thinking_budget
    is never set (omitted → DashScope defaults it to the model's MAX chain-of-thought)."""
    monkeypatch.setattr(ca.settings, "llm_max_thinking", True)
    cap = _capture_qwen_create(monkeypatch)
    ca._call_qwen_analyst("hi", thinking=False)
    assert cap["extra_body"] == {"enable_thinking": True}
    assert "thinking_budget" not in cap["extra_body"]     # omitted = max budget


def test_max_thinking_forces_deepseek_fallback_thinking(monkeypatch):
    """The cross-engine DeepSeek fallback reasons too under the flag."""
    monkeypatch.setattr(ca.settings, "llm_max_thinking", True)
    cap = _capture_deepseek_create(monkeypatch)
    ca._call_deepseek_analyst("hi")                       # default thinking=False
    assert cap["extra_body"] == ca._DEEPSEEK_THINKING_ON


def test_max_thinking_anthropic_effort_max(monkeypatch):
    monkeypatch.setattr(ca.settings, "llm_max_thinking", True)
    kw = _anthropic_thinking_kwargs("claude-opus-4-8")
    assert kw == {"thinking": {"type": "adaptive"}, "output_config": {"effort": "max"}}
    # Haiku 4.5 supports neither thinking nor effort → still empty (would 400)
    assert _anthropic_thinking_kwargs("claude-haiku-4-5-20251001") == {}


def test_off_flag_leaves_effort_default(monkeypatch):
    monkeypatch.setattr(ca.settings, "llm_max_thinking", False)
    assert _anthropic_thinking_kwargs("claude-opus-4-8") == {"thinking": {"type": "adaptive"}}


# ── Prefix-cache observability (stream_options + usage capture) ──────────────

def test_streaming_calls_request_usage_for_cache_observability(monkeypatch):
    """Both OpenAI-compatible streaming calls must ask for the final usage chunk —
    without stream_options include_usage the provider never reports cached_tokens
    and cache hits are invisible."""
    cap_q = _capture_qwen_create(monkeypatch)
    ca._call_qwen_analyst("hi")
    assert cap_q["stream_options"] == {"include_usage": True}
    cap_d = _capture_deepseek_create(monkeypatch)
    ca._call_deepseek_analyst("hi")
    assert cap_d["stream_options"] == {"include_usage": True}


def test_usage_chunk_captured_and_logged(monkeypatch):
    """The usage-only final chunk (empty choices) must be captured, not skipped by
    the empty-choices guard, and routed to the cache log."""
    logged = {}
    monkeypatch.setattr(ca, "_log_openai_cache_usage",
                        lambda tag, usage: logged.update(tag=tag, usage=usage))

    class _Details:
        cached_tokens = 1200
    class _Usage:
        prompt_tokens = 1500
        prompt_tokens_details = _Details()
    class _UsageChunk:
        choices = []
        usage = _Usage()
    class _Stream:
        def __enter__(self): return [_UsageChunk()]
        def __exit__(self, *a): return False
    class _Completions:
        def create(self, **kw): return _Stream()
    class _Client:
        chat = type("C", (), {"completions": _Completions()})()

    monkeypatch.setattr(ca, "_get_qwen_analyst_client", lambda: _Client())
    ca._call_qwen_analyst("hi")
    assert logged["usage"].prompt_tokens_details.cached_tokens == 1200
    assert "Qwen" in logged["tag"]
