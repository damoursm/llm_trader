"""Qwen route adapter (2026-07-12): DashScope direct vs OpenRouter.

The route derives from settings.qwen_base_url; it flips (1) the thinking
dialect (DashScope enable_thinking vs OpenRouter's unified reasoning param —
both verified live), (2) the model id (qwen/ prefix on OpenRouter), and (3) the
synthesis-prefix cache strategy (DashScope = implicit/automatic → plain prompt;
OpenRouter = no implicit for Alibaba → explicit cache_control marker).
"""

import pytest

import src.analysis.claude_analyst as ca
from config.settings import settings
from src.analysis.qwen_api import is_openrouter, thinking_body

OR_URL = "https://openrouter.ai/api/v1"
DS_URL = "https://dashscope-intl.aliyuncs.com/compatible-mode/v1"


@pytest.fixture
def _openrouter(monkeypatch):
    monkeypatch.setattr(settings, "qwen_base_url", OR_URL)
    monkeypatch.setattr(settings, "qwen_model", "qwen/qwen3.7-max")


# ── dialect selection ───────────────────────────────────────────────────────

def test_route_detection(monkeypatch):
    monkeypatch.setattr(settings, "qwen_base_url", DS_URL)
    assert not is_openrouter()
    monkeypatch.setattr(settings, "qwen_base_url", OR_URL)
    assert is_openrouter()


def test_thinking_body_per_route(monkeypatch):
    monkeypatch.setattr(settings, "qwen_base_url", DS_URL)
    assert thinking_body(True) == {"enable_thinking": True}
    assert thinking_body(False) == {"enable_thinking": False}
    monkeypatch.setattr(settings, "qwen_base_url", OR_URL)
    assert thinking_body(True) == {"reasoning": {"enabled": True}}
    assert thinking_body(False) == {"reasoning": {"enabled": False}}


# ── synthesis call threads the route dialect + model id ─────────────────────

def _capture(monkeypatch):
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


def test_openrouter_call_uses_reasoning_and_prefixed_model(_openrouter, monkeypatch):
    cap = _capture(monkeypatch)
    ca._call_qwen_analyst("hi", thinking=True)
    assert cap["model"] == "qwen/qwen3.7-max"
    assert cap["extra_body"] == {"reasoning": {"enabled": True}}
    # no budget/effort sent → provider default = MAX chain-of-thought
    assert "thinking_budget" not in str(cap["extra_body"])


def test_dashscope_call_keeps_enable_thinking(monkeypatch):
    cap = _capture(monkeypatch)
    ca._call_qwen_analyst("hi", thinking=True)
    assert cap["model"] == "qwen3.7-max"
    assert cap["extra_body"] == {"enable_thinking": True}


def test_forced_hold_review_model_follows_route(_openrouter):
    # The pinned-review default model must carry the route's id + thinking suffix.
    assert ca._qwen_spec(None) == ("qwen/qwen3.7-max", False)
    assert ca._qwen_spec("qwen/qwen3.7-max-thinking") == ("qwen/qwen3.7-max", True)


# ── prefix cache strategy per route ─────────────────────────────────────────

def _prompt_with_sentinel(prefix_chars=6000):
    return ("P" * prefix_chars) + ca._CACHE_SENTINEL + "SUFFIX"


def test_openrouter_marks_prefix_with_cache_control(_openrouter, monkeypatch):
    monkeypatch.setattr(settings, "enable_prompt_caching", True)
    content = ca._qwen_user_content(_prompt_with_sentinel())
    assert isinstance(content, list) and len(content) == 2
    assert content[0]["cache_control"] == {"type": "ephemeral"}
    assert content[0]["text"] == "P" * 6000
    assert content[1]["text"] == "SUFFIX"


def test_openrouter_skips_marker_below_explicit_cache_minimum(_openrouter, monkeypatch):
    monkeypatch.setattr(settings, "enable_prompt_caching", True)
    content = ca._qwen_user_content(_prompt_with_sentinel(prefix_chars=1000))  # <1024 tok
    assert isinstance(content, str) and ca._CACHE_SENTINEL not in content


def test_dashscope_strips_sentinel_plain(monkeypatch):
    # Implicit caching is automatic on the direct route — plain string, no markers.
    monkeypatch.setattr(settings, "enable_prompt_caching", True)
    content = ca._qwen_user_content(_prompt_with_sentinel())
    assert isinstance(content, str)
    assert ca._CACHE_SENTINEL not in content
    assert content == "P" * 6000 + "SUFFIX"


def test_openrouter_caching_off_falls_back_to_plain(_openrouter, monkeypatch):
    monkeypatch.setattr(settings, "enable_prompt_caching", False)
    content = ca._qwen_user_content(_prompt_with_sentinel())
    assert isinstance(content, str) and ca._CACHE_SENTINEL not in content
