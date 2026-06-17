"""Per-ticker news fetch + ticker-tagged relevance mapping (the #2 starvation fix).

Before this fix the per-ticker news / sentiment-velocity scores were ~0% populated:
general-market RSS/NewsAPI articles rarely name a specific mid-cap, and the
keyword aliases only covered ~30 mega-caps. yfinance Ticker.news supplies real
ticker-TAGGED articles for every symbol, mapped directly here.
"""
from datetime import datetime, timezone, timedelta

import yfinance

from src.models import NewsArticle
from src.analysis.sentiment import filter_relevant_articles
from src.data import news_fetcher
from src.data.news_fetcher import fetch_ticker_news, _parse_news_time


def _art(title="", summary="", tickers=None):
    return NewsArticle(title=title, summary=summary, url="u" + title, source="s",
                       published_at=datetime.now(timezone.utc), tickers=tickers or [])


def test_filter_relevant_matches_ticker_tags():
    # Two CRDO-tagged articles with NO "crdo" keyword in the text → matched by tag.
    arts = [_art("Chip momentum", "great quarter", ["CRDO"]),
            _art("Datacenter demand", "AI buildout", ["CRDO"])]
    assert len(filter_relevant_articles("CRDO", arts)) == 2
    assert filter_relevant_articles("ARM", arts) == []          # tagged to CRDO, not ARM


def test_filter_relevant_keyword_fallback_for_untagged():
    # Untagged general-market articles still map via the keyword aliases.
    arts = [_art("Intel cuts guidance", "intel chip", []),
            _art("Intel layoffs", "intel restructures", [])]
    assert len(filter_relevant_articles("INTC", arts)) == 2


def test_filter_relevant_below_threshold_returns_empty():
    assert filter_relevant_articles("CRDO", [_art("Credo soars", "great", ["CRDO"])]) == []


def test_parse_news_time_iso_epoch_none():
    assert _parse_news_time("2026-06-15T16:00:05Z").year == 2026
    assert _parse_news_time(1_700_000_000).tzinfo is not None   # legacy epoch
    assert _parse_news_time(None) is None
    assert _parse_news_time("") is None


def test_fetch_ticker_news_parses_content_schema(monkeypatch):
    now = datetime.now(timezone.utc)
    recent = now.strftime("%Y-%m-%dT%H:%M:%SZ")
    stale = (now - timedelta(days=30)).strftime("%Y-%m-%dT%H:%M:%SZ")

    class FakeTicker:
        def __init__(self, sym):
            self.sym = sym

        @property
        def news(self):
            return [
                {"content": {"title": f"{self.sym} beats", "summary": "good",
                             "pubDate": recent, "provider": {"displayName": "Zacks"},
                             "canonicalUrl": {"url": "http://x/1"}}},
                {"content": {"title": f"{self.sym} old news", "summary": "stale",
                             "pubDate": stale, "provider": {"displayName": "Zacks"},
                             "canonicalUrl": {"url": "http://x/2"}}},   # >7d → dropped
            ]

    monkeypatch.setattr(yfinance, "Ticker", FakeTicker)
    monkeypatch.setattr(news_fetcher.settings, "enable_ticker_news", True)

    arts = fetch_ticker_news(["CRDO"])
    assert len(arts) == 1                       # stale article dropped by the 7d cutoff
    a = arts[0]
    assert a.tickers == ["CRDO"] and a.title == "CRDO beats"
    assert a.source == "Zacks" and a.url == "http://x/1"


def test_fetch_ticker_news_legacy_schema(monkeypatch):
    class FakeTicker:
        def __init__(self, sym):
            self.sym = sym

        @property
        def news(self):                          # legacy flat schema (no 'content')
            return [{"title": f"{self.sym} up", "summary": "s", "link": "http://y/1",
                     "publisher": "Reuters",
                     "providerPublishTime": int(datetime.now(timezone.utc).timestamp())}]

    monkeypatch.setattr(yfinance, "Ticker", FakeTicker)
    monkeypatch.setattr(news_fetcher.settings, "enable_ticker_news", True)
    arts = fetch_ticker_news(["ARM"])
    assert len(arts) == 1 and arts[0].tickers == ["ARM"] and arts[0].source == "Reuters"


def test_fetch_ticker_news_disabled(monkeypatch):
    monkeypatch.setattr(news_fetcher.settings, "enable_ticker_news", False)
    assert fetch_ticker_news(["CRDO"]) == []
