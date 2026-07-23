"""Every LLM layer must reach DeepSeek when its preferred engine is down.

2026-07-22: Qwen ran out of OpenRouter credits and five HELD positions (WKC, MOH,
SEIC, GLD, AGEN) scored sentiment 0.0 + "Analysis error" on a DeepSeek-primary
run — they never tried DeepSeek at all, because an opener-pinned hold-review
returned the pinned engine ALONE. A neutral 0.0 then feeds the hold review that
decides whether to keep or close the position, so the failure was silent AND
consequential. Three chains had the same shape of gap.
"""

import pytest

from config.settings import settings
import src.analysis.sentiment as sent


# ── sentiment: a pin is a preference, not a dead end ─────────────────────────

@pytest.fixture(autouse=True)
def _deepseek_primary(monkeypatch):
    monkeypatch.setattr(sent, "_PRIMARY_SENTIMENT_ENGINE", "deepseek")
    monkeypatch.setattr(settings, "enable_claude_sentiment", False)


def test_qwen_pin_falls_back_to_deepseek():
    """THE regression: the pinned engine leads, but DeepSeek still follows."""
    order = sent._sentiment_engine_order("qwen")
    assert order[0] == "qwen", "opener pin must still be tried FIRST (Fix #2)"
    assert "deepseek" in order, "a down pin must not strand the ticker at 0.0"


def test_deepseek_pin_still_has_a_fallback():
    order = sent._sentiment_engine_order("deepseek")
    assert order[0] == "deepseek"
    assert len(order) > 1 and "qwen" in order


def test_unpinned_order_keeps_primary_first_and_reaches_every_engine():
    order = sent._sentiment_engine_order(None)
    assert order[0] == "deepseek"
    assert "qwen" in order


def test_claude_never_scores_sentiment_when_disabled():
    """Claude stays reserved for synthesis — including as a LAST-RESORT fallback,
    and including when explicitly pinned."""
    for pin in (None, "qwen", "deepseek", "anthropic"):
        order = sent._sentiment_engine_order(pin)
        assert "anthropic" not in order, f"anthropic leaked into order for pin={pin}"
        assert order, "order must never be empty"


def test_anthropic_pin_coerces_to_deepseek_when_disabled():
    assert sent._sentiment_engine_order("anthropic") == ["deepseek", "qwen"]


def test_claude_allowed_only_when_enabled(monkeypatch):
    monkeypatch.setattr(settings, "enable_claude_sentiment", True)
    assert "anthropic" in sent._sentiment_engine_order("anthropic")


def test_order_has_no_duplicates():
    for pin in (None, "qwen", "deepseek", "anthropic"):
        order = sent._sentiment_engine_order(pin)
        assert len(order) == len(set(order)), f"duplicate engine for pin={pin}"


@pytest.mark.parametrize("primary", ["qwen", "anthropic", "deepseek"])
def test_deepseek_is_always_reachable(monkeypatch, primary):
    monkeypatch.setattr(sent, "_PRIMARY_SENTIMENT_ENGINE", primary)
    for pin in (None, "qwen", "deepseek", "anthropic"):
        assert "deepseek" in sent._sentiment_engine_order(pin)


# ── macro-news classifier ────────────────────────────────────────────────────

def test_macro_news_deepseek_primary_still_carries_a_fallback(monkeypatch):
    from src.data import macro_news
    monkeypatch.setattr(settings, "llm_primary_provider", "deepseek")
    monkeypatch.setattr(settings, "deepseek_api_key", "k1")
    monkeypatch.setattr(settings, "qwen_api_key", "k2")
    providers = [a[0] for a in macro_news._macro_llm_attempts()]
    assert providers == ["deepseek", "qwen"]


def test_macro_news_qwen_primary_falls_back_to_deepseek(monkeypatch):
    from src.data import macro_news
    monkeypatch.setattr(settings, "llm_primary_provider", "qwen")
    monkeypatch.setattr(settings, "deepseek_api_key", "k1")
    monkeypatch.setattr(settings, "qwen_api_key", "k2")
    providers = [a[0] for a in macro_news._macro_llm_attempts()]
    assert providers == ["qwen", "deepseek"]


def test_macro_news_drops_keyless_providers(monkeypatch):
    from src.data import macro_news
    monkeypatch.setattr(settings, "llm_primary_provider", "deepseek")
    monkeypatch.setattr(settings, "deepseek_api_key", "k1")
    monkeypatch.setattr(settings, "qwen_api_key", "")
    assert [a[0] for a in macro_news._macro_llm_attempts()] == ["deepseek"]


# ── synthesis hold-review: try EVERY other engine, not one fixed alternate ───

@pytest.mark.parametrize("pin,expected", [
    ("deepseek", ["qwen", "anthropic"]),      # used to dead-end on anthropic alone
    ("qwen", ["deepseek", "anthropic"]),
    ("anthropic", ["deepseek", "qwen"]),
])
def test_hold_review_fallback_covers_all_other_engines(pin, expected):
    """DeepSeek leads (cheap + funded); every other engine is tried, not just one."""
    from src.pipeline import hold_review_fallbacks
    assert hold_review_fallbacks(pin) == expected


def test_hold_review_fallback_never_retries_the_pinned_engine():
    from src.pipeline import hold_review_fallbacks
    for pin in ("deepseek", "qwen", "anthropic"):
        assert pin not in hold_review_fallbacks(pin)
