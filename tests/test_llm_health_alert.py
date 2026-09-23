"""Regression tests for the LLM-failure email alert (pipeline._assess_llm_health).

When credits run out or keys are invalid the failure is SILENT — sentiment
returns neutral 0.0 and synthesis falls through to the rule-based last resort.
_assess_llm_health collapses that into a 'down' verdict that drives the email's
🤖 banner + "🤖 LLM DOWN" subject tag (and, when healthy, the green ✅ LLM line).
"""
import src.pipeline as pl


def _patch(monkeypatch, synth_provider, sent_summary, errors=None):
    monkeypatch.setattr(
        pl, "get_last_synthesis_meta",
        lambda: ({"provider": synth_provider} if synth_provider is not None else None))
    monkeypatch.setattr(pl, "get_sentiment_provider_summary", lambda: sent_summary)
    monkeypatch.setattr(pl, "get_sentiment_engine_errors", lambda: dict(errors or {}))


def test_synthesis_rule_based_is_down(monkeypatch):
    # Both LLMs failed → synthesis fell to rule-based → recommendations not AI-made.
    _patch(monkeypatch, "rule-based", "deepseek×40")
    h = pl._assess_llm_health()
    assert h["down"] and h["synthesis_down"] and not h["sentiment_down"]
    assert "rule-based" in h["message"]


def test_sentiment_all_none_is_down(monkeypatch):
    # Sentiment was attempted but EVERY per-ticker call failed (all "none").
    _patch(monkeypatch, "deepseek", "none×42")
    h = pl._assess_llm_health()
    assert h["down"] and h["sentiment_down"] and not h["synthesis_down"]


def test_deepseek_served_run_is_healthy(monkeypatch):
    # A DeepSeek-served run is a working LLM layer — NOT down.
    _patch(monkeypatch, "deepseek", "deepseek×40, none×2")
    h = pl._assess_llm_health()
    assert not h["down"]
    assert h["synthesis_provider"] == "deepseek"          # feeds the green ✅ line
    assert h["sentiment_summary"] == "deepseek×40, none×2"


def test_anthropic_served_run_is_healthy(monkeypatch):
    _patch(monkeypatch, "anthropic", "anthropic×40")
    h = pl._assess_llm_health()
    assert not h["down"] and h["synthesis_provider"] == "anthropic"


def test_no_news_to_score_is_not_down(monkeypatch):
    # No tickers had news → no sentiment attempts (summary None) → NOT a degradation.
    _patch(monkeypatch, "deepseek", None)
    h = pl._assess_llm_health()
    assert not h["down"] and not h["sentiment_down"]


# ── the remedy names the engine that FAILED, never a fixed guess ────────────
# 2026-09-21: after a reboot the local server never came back and the sentiment
# tier was local-only, so every call fell to "none" for 20 ticks — while the
# alert told the operator to top up Anthropic + DeepSeek credits.

def test_a_dead_local_only_tier_points_at_the_local_server(monkeypatch):
    _patch(monkeypatch, "rank", "none×88, provider×49",
           errors={"local": (88, "Connection error.")})
    h = pl._assess_llm_health()
    assert h["down"] and h["sentiment_down"] and not h["synthesis_down"]
    assert "local: Connection error. ×88" in h["message"]
    assert "local LLM server" in h["remedy"] and "LlmTraderOllama" in h["remedy"]
    assert "credits" not in h["remedy"], "sent the operator to the hosted billing page"


def test_a_hosted_outage_points_at_that_engines_credits(monkeypatch):
    _patch(monkeypatch, "rank", "none×40",
           errors={"deepseek": (40, "Error code: 402 - Insufficient Balance")})
    h = pl._assess_llm_health()
    assert "deepseek" in h["remedy"] and "credits" in h["remedy"]
    assert "local LLM server" not in h["remedy"]


def test_local_then_hosted_failures_name_both_fixes(monkeypatch):
    _patch(monkeypatch, "rank", "none×42",
           errors={"local": (42, "Connection error."), "deepseek": (42, "402")})
    remedy = pl._assess_llm_health()["remedy"]
    assert "local LLM server" in remedy and "deepseek API credits" in remedy


def test_no_engine_available_to_try_says_so(monkeypatch):
    # Every client unconfigured: calls fall to "none" without any engine raising.
    _patch(monkeypatch, "rank", "none×42", errors={})
    h = pl._assess_llm_health()
    assert h["sentiment_down"] and "no sentiment engine was available" in h["remedy"]


def test_a_synthesis_outage_still_points_at_hosted_credits(monkeypatch):
    _patch(monkeypatch, "rule-based", "local×40")
    h = pl._assess_llm_health()
    assert h["synthesis_down"] and not h["sentiment_down"]
    assert "credits" in h["remedy"]


def test_a_healthy_run_carries_no_remedy(monkeypatch):
    _patch(monkeypatch, "rank", "local×68")
    h = pl._assess_llm_health()
    assert not h["down"] and h["remedy"] == ""
