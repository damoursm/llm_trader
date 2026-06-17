"""Regression tests for the LLM-failure email alert (pipeline._assess_llm_health).

When credits run out or keys are invalid the failure is SILENT — sentiment
returns neutral 0.0 and synthesis falls through to the rule-based last resort.
_assess_llm_health collapses that into a 'down' verdict that drives the email's
🤖 banner + "🤖 LLM DOWN" subject tag (and, when healthy, the green ✅ LLM line).
"""
import src.pipeline as pl


def _patch(monkeypatch, synth_provider, sent_summary):
    monkeypatch.setattr(
        pl, "get_last_synthesis_meta",
        lambda: ({"provider": synth_provider} if synth_provider is not None else None))
    monkeypatch.setattr(pl, "get_sentiment_provider_summary", lambda: sent_summary)


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
