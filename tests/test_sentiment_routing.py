"""Sentiment engine routing — DeepSeek-only unless Claude sentiment is enabled."""

import src.analysis.sentiment as sent


def test_order_deepseek_only_when_claude_disabled(monkeypatch):
    monkeypatch.setattr(sent.settings, "enable_claude_sentiment", False)
    monkeypatch.setattr(sent, "_PRIMARY_SENTIMENT_ENGINE", "deepseek")
    assert sent._sentiment_engine_order(None) == ["deepseek"]
    assert sent._sentiment_engine_order("deepseek") == ["deepseek"]
    # even an anthropic hold-review pin coerces to deepseek — Claude never scores sentiment
    assert sent._sentiment_engine_order("anthropic") == ["deepseek"]


def test_order_ab_when_claude_enabled(monkeypatch):
    monkeypatch.setattr(sent.settings, "enable_claude_sentiment", True)
    monkeypatch.setattr(sent, "_PRIMARY_SENTIMENT_ENGINE", "anthropic")
    assert sent._sentiment_engine_order(None) == ["anthropic", "deepseek"]
    monkeypatch.setattr(sent, "_PRIMARY_SENTIMENT_ENGINE", "deepseek")
    assert sent._sentiment_engine_order(None) == ["deepseek", "anthropic"]
    assert sent._sentiment_engine_order("anthropic") == ["anthropic"]   # pin honored when enabled


def test_reset_forces_deepseek_when_disabled(monkeypatch):
    monkeypatch.setattr(sent.settings, "enable_claude_sentiment", False)
    monkeypatch.setattr(sent.settings, "llm_ab_anthropic_share", 1.0)   # would pick anthropic if enabled
    sent.reset_sentiment_providers()
    assert sent._PRIMARY_SENTIMENT_ENGINE == "deepseek"


def test_reset_ab_when_enabled(monkeypatch):
    monkeypatch.setattr(sent.settings, "enable_claude_sentiment", True)
    monkeypatch.setattr(sent.settings, "llm_ab_anthropic_share", 1.0)   # always anthropic
    sent.reset_sentiment_providers()
    assert sent._PRIMARY_SENTIMENT_ENGINE == "anthropic"
    monkeypatch.setattr(sent.settings, "llm_ab_anthropic_share", 0.0)   # always deepseek
    sent.reset_sentiment_providers()
    assert sent._PRIMARY_SENTIMENT_ENGINE == "deepseek"
