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


# ── RSS fast-lane + press-release wires (reactivity changes) ─────────────────

def test_dedupe_by_url_first_wins():
    a, b, dup = _art("x"), _art("y"), _art("x")   # _art url = "u"+title → dup of "x"
    out = news_fetcher._dedupe_by_url([a, b, dup])
    assert [t.title for t in out] == ["x", "y"]


def test_fetch_cached_news_excludes_rss(monkeypatch):
    """The cache-worthy bundle must NOT pull RSS — RSS is the fresh fast-lane."""
    monkeypatch.setattr(news_fetcher, "fetch_ticker_news", lambda t: [_art("tk", tickers=["AAA"])])
    monkeypatch.setattr(news_fetcher, "fetch_newsapi", lambda q, max_age_hours=168: [])

    def boom():
        raise AssertionError("fetch_cached_news must not fetch RSS")
    monkeypatch.setattr(news_fetcher, "fetch_rss_news", boom)

    out = news_fetcher.fetch_cached_news(["AAA"], ["XLK"])
    assert [a.title for a in out] == ["tk"]


def test_fetch_all_news_merges_cached_and_rss_deduped(monkeypatch):
    monkeypatch.setattr(news_fetcher, "fetch_cached_news", lambda t, s: [_art("c")])
    monkeypatch.setattr(news_fetcher, "fetch_rss_news", lambda: [_art("r"), _art("c")])  # "c" dups
    titles = [a.title for a in news_fetcher.fetch_all_news(["AAA"], [])]
    assert titles.count("c") == 1 and "r" in titles


def test_fetch_rss_includes_pr_wires(monkeypatch):
    """Every RSS feed AND every press-release wire URL is fetched."""
    seen = []

    class _Feed:
        entries = []
        bozo = 0

    monkeypatch.setattr(news_fetcher.feedparser, "parse", lambda url: (seen.append(url) or _Feed()))
    news_fetcher.fetch_rss_news()
    assert news_fetcher.PR_WIRE_FEEDS                      # wires configured
    for u in {**news_fetcher.RSS_FEEDS, **news_fetcher.PR_WIRE_FEEDS}.values():
        assert u in seen


def test_pipeline_fetch_news_fast_lanes_rss(monkeypatch):
    """The core reactivity guarantee: even on a news-cache HIT, RSS/wires are
    refetched FRESH every tick and merged — so a breaking catalyst is never
    hidden behind the hourly cache."""
    import src.pipeline as pipeline

    cached = [_art("cached-bundle", tickers=["AAA"])]
    fresh = [_art("breaking wire")]
    rss_calls = []

    monkeypatch.setattr(pipeline, "load_news", lambda: cached)          # cache HIT
    monkeypatch.setattr(pipeline, "fetch_rss_news", lambda: (rss_calls.append(1) or fresh))
    monkeypatch.setattr(pipeline, "save_news", lambda a: None)

    def _no_cached_fetch(t, s):
        raise AssertionError("must not refetch the cached bundle on a cache hit")
    monkeypatch.setattr(pipeline, "fetch_cached_news", _no_cached_fetch)

    out = pipeline._fetch_news(["AAA"], [])
    assert rss_calls == [1]                                # RSS fetched fresh despite cache hit
    titles = {a.title for a in out}
    assert {"cached-bundle", "breaking wire"} <= titles    # merged
