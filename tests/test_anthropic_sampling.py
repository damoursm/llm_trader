"""Guard against the 'temperature is deprecated for this model' 400 (2026-06-17).

Opus 4.7+ and the Fable/Mythos 5 families removed temperature/top_p/top_k;
sending temperature 400s and silently kicks synthesis to the DeepSeek fallback.
"""

import anthropic
import openai
import pytest

import src.analysis.claude_analyst as ca
from src.analysis.claude_analyst import (
    _anthropic_sampling_kwargs, _anthropic_thinking_kwargs,
    _engine_of, _synthesis_attempts_for, _deepseek_spec, _qwen_spec,
    _is_transient_llm_error, _call_with_retry,
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
    # DeepSeek then Qwen are the resilience fallbacks (all 3 providers, so a
    # single-provider outage can never fully block synthesis).
    assert _synthesis_attempts_for(chosen, "claude-opus-4-8", "deepseek-v4-flash", "qwen3.7-max") == [
        ("anthropic", chosen), ("deepseek", "deepseek-v4-flash"), ("qwen", "qwen3.7-max")]


def test_attempts_deepseek_chosen_then_anthropic_fallback():
    # DeepSeek chosen -> Qwen, THEN Anthropic (2026-07-22: widened from a single
    # fixed fallback so a transient DeepSeek failure isn't stranded when
    # Anthropic alone happens to be out of credits).
    assert _synthesis_attempts_for("deepseek-v4-flash", "claude-opus-4-8", "deepseek-v4-flash", "qwen3.7-max") == [
        ("deepseek", "deepseek-v4-flash"), ("qwen", "qwen3.7-max"), ("anthropic", "claude-opus-4-8")]


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
    # exact arm); the cross-engine fallbacks stay cheap/non-thinking defaults.
    assert _synthesis_attempts_for("deepseek-v4-pro-thinking", "claude-opus-4-8", "deepseek-v4-flash", "qwen3.7-max") == [
        ("deepseek", "deepseek-v4-pro-thinking"), ("qwen", "qwen3.7-max"), ("anthropic", "claude-opus-4-8")]


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
    # Qwen (the primary) runs first; DeepSeek then Anthropic are the fallbacks.
    assert _synthesis_attempts_for("qwen3.7-max-thinking", "claude-opus-4-8", "deepseek-v4-flash", "qwen3.7-max") == [
        ("qwen", "qwen3.7-max-thinking"), ("deepseek", "deepseek-v4-flash"), ("anthropic", "claude-opus-4-8")]


# ── Same-engine transient retry (2026-07-22) ─────────────────────────────────

class _FakeStatusError(Exception):
    def __init__(self, status_code, message="boom"):
        super().__init__(message)
        self.status_code = status_code


@pytest.mark.parametrize("status", [429, 500, 502, 503, 529])
def test_transient_status_codes_are_retried(status):
    assert _is_transient_llm_error(_FakeStatusError(status)) is True


@pytest.mark.parametrize("status", [400, 401, 402, 403, 404])
def test_hard_status_codes_are_not_retried(status):
    # 402 in particular is the exact "Insufficient credits/balance" shape seen
    # from both DeepSeek and Qwen/OpenRouter — retrying it just fails again.
    assert _is_transient_llm_error(_FakeStatusError(status)) is False


def test_connection_errors_are_transient():
    req = object()   # openai.APIConnectionError only touches .request for repr
    assert _is_transient_llm_error(openai.APIConnectionError(request=req)) is True
    assert _is_transient_llm_error(anthropic.APIConnectionError(request=req)) is True


def test_raw_readtimeout_leak_is_recognized_by_message():
    # 2026-07-22 case: a mid-stream read-timeout leaked through as a raw
    # exception with no status_code — must still be caught by message/type,
    # since this is precisely the failure that stranded a healthy DeepSeek.
    class ReadTimeout(Exception):
        pass
    assert _is_transient_llm_error(ReadTimeout("The read operation timed out")) is True


def test_unrelated_error_is_not_transient():
    assert _is_transient_llm_error(ValueError("unexpected ticker shape")) is False


def test_call_with_retry_recovers_after_one_transient_failure(monkeypatch):
    monkeypatch.setattr(ca.settings, "llm_transient_retries", 1)
    sleeps = []
    monkeypatch.setattr(ca.time, "sleep", lambda s: sleeps.append(s))
    calls = {"n": 0}

    def fake_engine(engine, model, prompt):
        calls["n"] += 1
        if calls["n"] == 1:
            raise _FakeStatusError(503, "server hiccup")
        return "ok"

    monkeypatch.setattr(ca, "_call_engine", fake_engine)
    assert _call_with_retry("deepseek", "deepseek-v4-flash", "hi") == "ok"
    assert calls["n"] == 2
    assert len(sleeps) == 1


def test_call_with_retry_raises_immediately_on_hard_failure(monkeypatch):
    monkeypatch.setattr(ca.settings, "llm_transient_retries", 1)
    sleeps = []
    monkeypatch.setattr(ca.time, "sleep", lambda s: sleeps.append(s))
    calls = {"n": 0}

    def fake_engine(engine, model, prompt):
        calls["n"] += 1
        raise _FakeStatusError(402, "Insufficient Balance")

    monkeypatch.setattr(ca, "_call_engine", fake_engine)
    with pytest.raises(_FakeStatusError):
        _call_with_retry("deepseek", "deepseek-v4-flash", "hi")
    assert calls["n"] == 1          # never retried — no point burning a hop on a hard fail
    assert sleeps == []


def test_call_with_retry_gives_up_after_exhausting_retries(monkeypatch):
    monkeypatch.setattr(ca.settings, "llm_transient_retries", 2)
    monkeypatch.setattr(ca.time, "sleep", lambda s: None)
    calls = {"n": 0}

    def fake_engine(engine, model, prompt):
        calls["n"] += 1
        raise _FakeStatusError(429, "rate limited")

    monkeypatch.setattr(ca, "_call_engine", fake_engine)
    with pytest.raises(_FakeStatusError):
        _call_with_retry("deepseek", "deepseek-v4-flash", "hi")
    assert calls["n"] == 3          # initial + 2 retries, then give up


def test_call_with_retry_zero_disables_retry(monkeypatch):
    monkeypatch.setattr(ca.settings, "llm_transient_retries", 0)
    monkeypatch.setattr(ca.time, "sleep", lambda s: None)
    calls = {"n": 0}

    def fake_engine(engine, model, prompt):
        calls["n"] += 1
        raise _FakeStatusError(503, "server hiccup")

    monkeypatch.setattr(ca, "_call_engine", fake_engine)
    with pytest.raises(_FakeStatusError):
        _call_with_retry("deepseek", "deepseek-v4-flash", "hi")
    assert calls["n"] == 1


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
    # effort (output_config) is routed through extra_body so it reaches the API as a
    # body field even on an anthropic SDK too old to accept a top-level output_config=
    # kwarg (0.49.0 raised TypeError → silent fallback to DeepSeek/rule-based).
    assert kw == {"thinking": {"type": "adaptive"},
                  "extra_body": {"output_config": {"effort": "max"}}}
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
