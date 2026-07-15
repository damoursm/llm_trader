"""Sentiment engine routing (2026-07-13 cost tune): DeepSeek-flash scores ~90% of
runs and Qwen ~10% (sentiment_qwen_share), each the other's error fallback.
Hold-review pins are HONORED (Fix #2). Claude is reserved for synthesis unless
enable_claude_sentiment."""

import src.analysis.sentiment as sent


def test_order_deepseek_primary_qwen_fallback(monkeypatch):
    """DeepSeek primary, Claude disabled → Qwen is the error fallback (not stripped)."""
    monkeypatch.setattr(sent.settings, "enable_claude_sentiment", False)
    monkeypatch.setattr(sent, "_PRIMARY_SENTIMENT_ENGINE", "deepseek")
    assert sent._sentiment_engine_order(None) == ["deepseek", "qwen"]
    # pins honored (Fix #2 — the opener engine re-judges its own position)
    assert sent._sentiment_engine_order("deepseek") == ["deepseek"]
    assert sent._sentiment_engine_order("qwen") == ["qwen"]
    # an anthropic pin coerces to deepseek — Claude never scores sentiment when disabled
    assert sent._sentiment_engine_order("anthropic") == ["deepseek"]


def test_order_qwen_primary_run(monkeypatch):
    """A Qwen-primary run (the ~10%) tries Qwen first, DeepSeek as the fallback."""
    monkeypatch.setattr(sent.settings, "enable_claude_sentiment", False)
    monkeypatch.setattr(sent, "_PRIMARY_SENTIMENT_ENGINE", "qwen")
    assert sent._sentiment_engine_order(None) == ["qwen", "deepseek"]
    # pins still honored — a deepseek-opened position is re-judged by deepseek
    assert sent._sentiment_engine_order("deepseek") == ["deepseek"]
    assert sent._sentiment_engine_order("qwen") == ["qwen"]


def test_order_ab_when_claude_enabled(monkeypatch):
    monkeypatch.setattr(sent.settings, "enable_claude_sentiment", True)
    monkeypatch.setattr(sent, "_PRIMARY_SENTIMENT_ENGINE", "anthropic")
    assert sent._sentiment_engine_order(None) == ["anthropic", "deepseek"]
    monkeypatch.setattr(sent, "_PRIMARY_SENTIMENT_ENGINE", "deepseek")
    assert sent._sentiment_engine_order(None) == ["deepseek", "qwen"]
    assert sent._sentiment_engine_order("anthropic") == ["anthropic"]   # pin honored when enabled


def test_reset_deepseek_when_qwen_share_zero(monkeypatch):
    monkeypatch.setattr(sent.settings, "enable_claude_sentiment", False)
    monkeypatch.setattr(sent.settings, "sentiment_qwen_share", 0.0)
    monkeypatch.setattr(sent.settings, "qwen_api_key", "sk-test")
    sent.reset_sentiment_providers()
    assert sent._PRIMARY_SENTIMENT_ENGINE == "deepseek"


def test_reset_qwen_when_share_one_and_keyed(monkeypatch):
    monkeypatch.setattr(sent.settings, "enable_claude_sentiment", False)
    monkeypatch.setattr(sent.settings, "sentiment_qwen_share", 1.0)   # always Qwen this run
    monkeypatch.setattr(sent.settings, "qwen_api_key", "sk-test")
    sent.reset_sentiment_providers()
    assert sent._PRIMARY_SENTIMENT_ENGINE == "qwen"


def test_reset_deepseek_when_qwen_keyless(monkeypatch):
    """Even at share 1.0, a missing Qwen key falls to DeepSeek (no wasted primary attempt)."""
    monkeypatch.setattr(sent.settings, "enable_claude_sentiment", False)
    monkeypatch.setattr(sent.settings, "sentiment_qwen_share", 1.0)
    monkeypatch.setattr(sent.settings, "qwen_api_key", "")
    sent.reset_sentiment_providers()
    assert sent._PRIMARY_SENTIMENT_ENGINE == "deepseek"


def test_reset_anthropic_ab_precedes_qwen_share(monkeypatch):
    """Claude's A/B (when enabled) runs ahead of the Qwen split."""
    monkeypatch.setattr(sent.settings, "enable_claude_sentiment", True)
    monkeypatch.setattr(sent.settings, "llm_ab_anthropic_share", 1.0)   # always anthropic
    monkeypatch.setattr(sent.settings, "sentiment_qwen_share", 1.0)     # would be qwen otherwise
    monkeypatch.setattr(sent.settings, "qwen_api_key", "sk-test")
    sent.reset_sentiment_providers()
    assert sent._PRIMARY_SENTIMENT_ENGINE == "anthropic"
    # anthropic share 0 → falls through to the qwen split (share 1.0 → qwen)
    monkeypatch.setattr(sent.settings, "llm_ab_anthropic_share", 0.0)
    sent.reset_sentiment_providers()
    assert sent._PRIMARY_SENTIMENT_ENGINE == "qwen"
