"""Provider-sentiment hybrid: use pre-computed sentiment, skip the LLM scorer."""

from datetime import datetime, timezone

import src.analysis.sentiment as sent
import src.data.provider_news as pn
from src.models import NewsArticle


def _art(url, ticker, label=None, source="Polygon"):
    now = datetime.now(timezone.utc)
    return NewsArticle(
        title=f"news {url}", summary="x", url=url, source=source, published_at=now,
        tickers=[ticker],
        provider_insights=({ticker: label} if label else {}),
        provider_sentiment_source=("polygon" if label else None),
    )


def test_provider_score_helper(monkeypatch):
    monkeypatch.setattr(sent.settings, "enable_provider_sentiment", True)
    monkeypatch.setattr(sent.settings, "provider_sentiment_min_articles", 2)
    monkeypatch.setattr(sent.settings, "provider_sentiment_magnitude", 0.6)
    arts = [_art("u1", "AAPL", "positive"), _art("u2", "AAPL", "positive", source="Benzinga")]

    res = sent._provider_sentiment_score("AAPL", arts)
    assert res is not None
    score, rationale = res
    assert score > 0 and "provider" in rationale.lower()

    # flag off → defer to LLM (None)
    monkeypatch.setattr(sent.settings, "enable_provider_sentiment", False)
    assert sent._provider_sentiment_score("AAPL", arts) is None


def test_provider_score_needs_min_articles(monkeypatch):
    monkeypatch.setattr(sent.settings, "enable_provider_sentiment", True)
    monkeypatch.setattr(sent.settings, "provider_sentiment_min_articles", 2)
    # only one provider-scored article (the other has no insight) → defer to LLM
    arts = [_art("u1", "AAPL", "positive"), _art("u2", "AAPL", None)]
    assert sent._provider_sentiment_score("AAPL", arts) is None


def test_finnhub_style_articles_score_themselves_via_llm(monkeypatch):
    # Articles WITHOUT provider sentiment (Finnhub free, RSS, NewsAPI) must defer
    # to the LLM scorer — "we score it ourselves when the source doesn't have it"
    # — even when the provider-sentiment flag is ON.
    monkeypatch.setattr(sent.settings, "enable_provider_sentiment", True)
    monkeypatch.setattr(sent.settings, "provider_sentiment_min_articles", 2)
    arts = [_art("u1", "AAPL", None, source="Finnhub"),
            _art("u2", "AAPL", None, source="CNBC")]
    assert sent._provider_sentiment_score("AAPL", arts) is None   # → LLM path


def test_analyse_sentiment_uses_provider_and_skips_llm(monkeypatch):
    monkeypatch.setattr(sent.settings, "enable_provider_sentiment", True)
    monkeypatch.setattr(sent.settings, "provider_sentiment_min_articles", 2)
    monkeypatch.setattr(sent.settings, "provider_sentiment_magnitude", 0.6)

    def _boom(*a, **k):
        raise AssertionError("LLM must NOT be called when provider sentiment is used")

    monkeypatch.setattr(sent, "_get_deepseek", _boom)
    monkeypatch.setattr(sent, "_get_haiku", _boom)

    arts = [_art("u1", "AAPL", "positive"), _art("u2", "AAPL", "negative", source="Benzinga")]
    score, rationale = sent.analyse_sentiment("AAPL", arts)
    assert "provider" in rationale.lower()       # provider path taken (no LLM)


def test_force_engine_bypasses_provider(monkeypatch):
    # The opener-pinned hold-review (force_engine) must re-judge with its OWN LLM
    # engine, never the provider short-circuit.
    monkeypatch.setattr(sent.settings, "enable_provider_sentiment", True)
    monkeypatch.setattr(sent.settings, "provider_sentiment_min_articles", 2)
    called = {"n": 0}

    def _fake_score(*a, **k):
        called["n"] += 1
        return None

    monkeypatch.setattr(sent, "_provider_sentiment_score", _fake_score)
    # no deepseek/haiku key path: force a deepseek engine that returns None client
    monkeypatch.setattr(sent, "_get_deepseek", lambda: None)
    monkeypatch.setattr(sent, "_get_haiku", lambda: (_ for _ in ()).throw(RuntimeError("no key")))
    arts = [_art("u1", "AAPL", "positive"), _art("u2", "AAPL", "positive")]
    sent.analyse_sentiment("AAPL", arts, force_engine="deepseek")
    assert called["n"] == 0   # provider short-circuit never consulted under force_engine


def test_fetch_polygon_news_maps_insights(monkeypatch):
    monkeypatch.setattr(pn.settings, "enable_polygon_news", True)
    raw = [
        {"title": "T", "article_url": "http://x", "published_utc": "2026-06-17T12:00:00Z",
         "publisher": {"name": "Benzinga"}, "tickers": ["AAPL", "MSFT"], "description": "d",
         "insights": [{"ticker": "AAPL", "sentiment": "positive", "sentiment_reasoning": "r"},
                      {"ticker": "MSFT", "sentiment": "neutral"}]},
        {"title": "Other", "article_url": "http://y", "published_utc": "2026-06-17T11:00:00Z",
         "tickers": ["ZZZZ"], "insights": []},
    ]
    import src.data.polygon_client as pc
    monkeypatch.setattr(pc, "get_news", lambda limit=1000: raw)

    arts = pn.fetch_polygon_news(["AAPL"])
    assert len(arts) == 1                       # ZZZZ article filtered out (not in universe)
    a = arts[0]
    assert a.provider_insights == {"AAPL": "positive", "MSFT": "neutral"}
    assert a.provider_sentiment_source == "polygon"
    assert a.source == "Benzinga"


def test_provider_news_disabled_returns_empty(monkeypatch):
    monkeypatch.setattr(pn.settings, "enable_polygon_news", False)
    assert pn.fetch_polygon_news(["AAPL"]) == []
    monkeypatch.setattr(pn.settings, "enable_finnhub_news", False)
    assert pn.fetch_finnhub_news(["AAPL"]) == []
    # enabled but no key → still empty
    monkeypatch.setattr(pn.settings, "enable_finnhub_news", True)
    monkeypatch.setattr(pn.settings, "finnhub_api_key", "")
    assert pn.fetch_finnhub_news(["AAPL"]) == []
