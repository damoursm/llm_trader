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


def test_filter_relevant_below_threshold_returns_empty(monkeypatch):
    from src.analysis import sentiment
    monkeypatch.setattr(sentiment.settings, "news_relevance_min_articles", 2)
    assert filter_relevant_articles("CRDO", [_art("Credo soars", "great", ["CRDO"])]) == []


def test_filter_relevant_one_confirmed_article_is_a_digest():
    """Default minimum is ONE (2026-09-04): a single article a feed tagged to the
    company is evidence — the evidence-mass scaler discounts the thin digest;
    dropping it to nothing threw away a verdict for the min-2 of the old filter."""
    assert len(filter_relevant_articles("CRDO", [_art("Credo soars", "great", ["CRDO"])])) == 1


def test_filter_relevant_no_substring_sweep():
    """The 2026-09-04 root cause: ``"ar" in "market"`` used to be a hit, so a
    short symbol received the whole pool. Untagged prose that merely CONTAINS
    the letters is not about the company."""
    arts = [_art("Market rally broadens", "Traders start to price in cuts", []),
            _art("Barclays upgrades carmakers", "Sector view", [])]
    assert filter_relevant_articles("AR", arts) == []
    # ...but an explicit symbol or the registrant name IS.
    from src.data import company_names
    company_names._seed_for_tests({"AR": "Antero Resources Corp"})
    arts += [_art("Antero Resources raises output guidance", "", []),
             _art("Gas producers rally", "Antero (NYSE: AR) led the group", [])]
    got = filter_relevant_articles("AR", arts)
    assert [a.title for a in got] == ["Antero Resources raises output guidance", "Gas producers rally"]


def test_filter_relevant_legacy_behind_flag(monkeypatch):
    """``enable_name_relevance=False`` restores the pre-2026-09-04 filter
    (substring match, minimum two)."""
    from src.analysis import sentiment
    monkeypatch.setattr(sentiment.settings, "enable_name_relevance", False)
    arts = [_art("Market rally broadens", "", []), _art("Barclays upgrades carmakers", "", [])]
    assert len(filter_relevant_articles("AR", arts)) == 2          # the old sweep
    assert filter_relevant_articles("CRDO", [_art("x", "", ["CRDO"])]) == []   # old min 2


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
    monkeypatch.setattr(news_fetcher, "fetch_google_news", lambda t: [_art("g"), _art("c")])  # "c" dups
    titles = [a.title for a in news_fetcher.fetch_all_news(["AAA"], [])]
    assert titles.count("c") == 1 and "r" in titles and "g" in titles


# ── Google News RSS (per-ticker + Business Wire) ─────────────────────────────

def _gfeed(entries):
    class _F:
        pass
    f = _F()
    f.entries = entries
    return f


def _genable(monkeypatch):
    monkeypatch.setattr(news_fetcher.settings, "enable_google_news", True)
    monkeypatch.setattr(news_fetcher.settings, "google_news_max_tickers", 50)
    monkeypatch.setattr(news_fetcher.settings, "google_news_business_wire", True)


def test_fetch_google_news_maps_tickers_and_queries_business_wire(monkeypatch):
    import time as _t
    _genable(monkeypatch)
    recent = _t.gmtime()
    seen = []

    def fake_parse(url):
        seen.append(url)
        return _gfeed([{"title": "AAPL soars", "summary": "x", "link": f"http://g/{len(seen)}",
                        "published_parsed": recent, "source": {"title": "Reuters"}}])

    monkeypatch.setattr(news_fetcher.feedparser, "parse", fake_parse)
    arts = news_fetcher.fetch_google_news(["AAPL"])

    # symbol query + Business Wire query + the company-name query (AAPL carries a
    # curated alias, so a name phrase exists even with the SEC list offline).
    assert len(seen) == 3
    assert any("businesswire.com" in u for u in seen)
    assert any("%22apple%22+stock" in u for u in seen), seen
    assert arts and all(a.tickers == ["AAPL"] for a in arts)
    assert any(a.source == "google_news/Reuters" for a in arts)


def test_fetch_google_news_dedupes_identical_urls(monkeypatch):
    import time as _t
    _genable(monkeypatch)
    recent = _t.gmtime()
    # Both the general and BW query return the SAME article URL → one survives.
    monkeypatch.setattr(news_fetcher.feedparser, "parse", lambda url: _gfeed(
        [{"title": "dup", "summary": "", "link": "http://same", "published_parsed": recent}]))
    arts = news_fetcher.fetch_google_news(["AAPL"])
    assert len(arts) == 1


def test_fetch_google_news_drops_stale(monkeypatch):
    import time as _t
    _genable(monkeypatch)
    stale = _t.gmtime(_t.time() - 30 * 86400)             # 30 days old
    monkeypatch.setattr(news_fetcher.feedparser, "parse", lambda url: _gfeed(
        [{"title": "old", "summary": "", "link": "http://o", "published_parsed": stale}]))
    assert news_fetcher.fetch_google_news(["AAPL"], max_age_hours=24) == []


def test_fetch_google_news_skips_non_equity_and_caps(monkeypatch):
    _genable(monkeypatch)
    monkeypatch.setattr(news_fetcher.settings, "google_news_max_tickers", 2)
    queried = []
    monkeypatch.setattr(news_fetcher.feedparser, "parse",
                        lambda url: (queried.append(url) or _gfeed([])))
    # ^VIX (index) and GC=F (future) skipped; cap=2 keeps the first two equities.
    news_fetcher.fetch_google_news(["^VIX", "GC=F", "AAPL", "MSFT", "NVDA"])
    assert all("VIX" not in u and "GC" not in u for u in queried)
    tickers_hit = {t for t in ("AAPL", "MSFT", "NVDA") if any(t in u for u in queried)}
    assert tickers_hit == {"AAPL", "MSFT"}                # NVDA dropped by the cap


def test_fetch_google_news_disabled(monkeypatch):
    monkeypatch.setattr(news_fetcher.settings, "enable_google_news", False)
    assert news_fetcher.fetch_google_news(["AAPL"]) == []


# ── Search-feed tag confirmation (2026-09-04) ────────────────────────────────

def test_confirmed_tags_keeps_only_articles_that_mention_the_company():
    from src.data import company_names
    company_names._seed_for_tests({"AR": "Antero Resources Corp", "DIS": "Walt Disney Co"})
    ct = news_fetcher._confirmed_tags
    assert ct("AR", "Antero Resources beats", "") == ["AR"]             # name phrase
    assert ct("AR", "Gas names rally", "Antero (NYSE: AR) led") == ["AR"]  # explicit symbol
    assert ct("DIS", "Disney+ price hike", "") == ["DIS"]                # distinctive token, feed-vouched
    assert ct("AR", "AR-15 maker files for bankruptcy", "") == []        # not the company
    assert ct("AR", "Apple's AR headset delayed", "") == []              # AR = augmented reality
    assert ct("DIS", "Streaming wars heat up", "Netflix raises prices") == []  # a peer story


def test_confirmed_tags_symbol_word_for_unnamed_symbol():
    """No registrant name known (fresh symbol, SEC list lacking it): the bare
    symbol as a case-sensitive word still confirms, an ordinary acronym does not."""
    ct = news_fetcher._confirmed_tags
    assert ct("CRDO", "CRDO beats estimates", "") == ["CRDO"]
    assert ct("CRDO", "Semis rally", "chip names up") == []


def test_confirmed_tags_unconditional_behind_flag(monkeypatch):
    monkeypatch.setattr(news_fetcher.settings, "enable_name_relevance", False)
    assert news_fetcher._confirmed_tags("AR", "Apple's AR headset delayed", "") == ["AR"]


def test_fetch_ticker_news_related_item_stays_untagged(monkeypatch):
    """yfinance ``Ticker.news`` is Yahoo's RELATED feed: an item about a peer
    is returned for the queried symbol. It enters the pool (another name's
    filter may claim it) but is NOT tagged as the queried company's news."""
    from src.data import company_names
    company_names._seed_for_tests({"AMD": "Advanced Micro Devices Inc"})
    now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    class FakeTicker:
        def __init__(self, sym):
            self.sym = sym

        @property
        def news(self):
            return [
                {"content": {"title": "AMD unveils MI400", "summary": "", "pubDate": now,
                             "provider": {"displayName": "Reuters"},
                             "canonicalUrl": {"url": "http://x/1"}}},
                {"content": {"title": "Nvidia hits record high", "summary": "Blackwell demand",
                             "pubDate": now, "provider": {"displayName": "Reuters"},
                             "canonicalUrl": {"url": "http://x/2"}}},
            ]

    monkeypatch.setattr(yfinance, "Ticker", FakeTicker)
    monkeypatch.setattr(news_fetcher.settings, "enable_ticker_news", True)
    arts = {a.title: a for a in fetch_ticker_news(["AMD"])}
    assert arts["AMD unveils MI400"].tickers == ["AMD"]
    assert arts["Nvidia hits record high"].tickers == []            # in the pool, untagged


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


class _EmptyFeed:
    entries = []
    bozo = 0


def test_fetch_rss_includes_fda_when_enabled(monkeypatch):
    """FDA / MedWatch regulatory feeds ride the fresh fast-lane when enabled."""
    seen = []
    monkeypatch.setattr(news_fetcher.feedparser, "parse", lambda url: (seen.append(url) or _EmptyFeed()))
    monkeypatch.setattr(news_fetcher.settings, "enable_fda_news", True)
    news_fetcher.fetch_rss_news()
    for u in news_fetcher.FDA_FEEDS.values():
        assert u in seen


def test_fetch_rss_excludes_fda_when_disabled(monkeypatch):
    seen = []
    monkeypatch.setattr(news_fetcher.feedparser, "parse", lambda url: (seen.append(url) or _EmptyFeed()))
    monkeypatch.setattr(news_fetcher.settings, "enable_fda_news", False)
    news_fetcher.fetch_rss_news()
    for u in news_fetcher.FDA_FEEDS.values():
        assert u not in seen


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


# ── Google News: company-NAME query + feed-status tally (2026-09-04) ─────────

def test_fetch_google_news_adds_company_name_query_when_a_name_is_known(monkeypatch):
    import time as _t
    from src.data import company_names
    _genable(monkeypatch)
    company_names._seed_for_tests({"AR": "Antero Resources Corp"})
    recent = _t.gmtime()
    seen = []

    def fake_parse(url):
        seen.append(url)
        return _gfeed([{"title": "Antero Resources raises guidance", "summary": "",
                        "link": f"http://g/{len(seen)}", "published_parsed": recent}])

    monkeypatch.setattr(news_fetcher.feedparser, "parse", fake_parse)
    arts = news_fetcher.fetch_google_news(["AR"])

    # symbol query + name query + Business Wire query — the name query carries the
    # registrant title minus its suffix, quoted, so Google matches the phrase.
    assert len(seen) == 3
    assert any("%22antero+resources%22+stock" in u for u in seen), seen
    assert any("%22AR%22+stock" in u for u in seen), seen
    # The name-query hits are confirmed by the name-phrase tier, so they tag AR.
    assert arts and all(a.tickers == ["AR"] for a in arts)


def test_google_name_query_is_absent_without_a_name():
    # Offline fixture: no symbol has a name (and ZQZX has no curated alias) → no
    # name query, and no crash.
    assert news_fetcher._google_name_query("ZQZX") is None
    assert news_fetcher._google_name_query("") is None


def test_fetch_google_news_warns_on_non_200_feed_status(monkeypatch):
    from loguru import logger
    _genable(monkeypatch)
    monkeypatch.setattr(news_fetcher.settings, "google_news_business_wire", False)

    class _Throttled:
        status = 429
        entries = []

    monkeypatch.setattr(news_fetcher.feedparser, "parse", lambda url: _Throttled())
    msgs = []
    sink = logger.add(lambda m: msgs.append(str(m)), level="WARNING")
    try:
        # Alias-free symbols: one symbol query each, so exactly two feeds.
        assert news_fetcher.fetch_google_news(["ZQZX", "ZQZY"]) == []
    finally:
        logger.remove(sink)
    # One WARNING for the batch, carrying the count and the status — a throttle
    # must not read as a quiet news day.
    hits = [m for m in msgs if "Google News" in m and "non-200" in m]
    assert len(hits) == 1 and "429" in hits[0] and "2 feed" in hits[0], msgs


# ── freshest-cluster cut (A/B treatment arm, not wired into scoring) ─────────

def _aged(hours, title="h"):
    return NewsArticle(title=title, summary="s" * 40, url=f"u{hours}", source="Reuters",
                       published_at=datetime.now(timezone.utc) - timedelta(hours=hours))


def test_recent_cluster_keeps_the_fresh_group_and_drops_the_stale_one():
    from src.analysis.sentiment import recent_cluster
    arts = [_aged(0.5), _aged(2), _aged(6), _aged(96), _aged(100)]
    kept = recent_cluster(arts)
    assert [a.url for a in kept] == ["u0.5", "u2", "u6"]


def test_recent_cluster_is_relative_so_a_quiet_ticker_keeps_its_digest():
    """A FIXED window would empty the digest on a name whose only coverage is
    days old — and an empty digest is an abstention, a bigger change than the
    one under test. The cut is relative, so the freshest article always
    survives and nothing is dropped when everything is equally stale."""
    from src.analysis.sentiment import recent_cluster
    arts = [_aged(48), _aged(96), _aged(120)]
    assert len(recent_cluster(arts)) == 3
    assert recent_cluster([_aged(150)]) != []


def test_recent_cluster_floor_protects_the_same_days_coverage():
    """Without the 24h floor a 30-minute article would cut at 90 minutes and
    throw away this morning's coverage of the same story."""
    from src.analysis.sentiment import recent_cluster
    arts = [_aged(0.5), _aged(8), _aged(20)]
    assert len(recent_cluster(arts)) == 3
    assert len(recent_cluster(arts, floor_hours=2.0)) == 1


def test_recent_cluster_is_hour_quantised_and_empty_safe():
    """Same quantisation as `_recency_weight`: two runs inside one hour must cut
    identically, or a borderline article re-keys the verdict cache mid-hour."""
    from src.analysis.sentiment import recent_cluster
    assert recent_cluster([]) == []
    arts = [_aged(0.1), _aged(23.4), _aged(30)]
    assert [a.url for a in recent_cluster(arts)] == [a.url for a in recent_cluster(arts)]


def test_recent_cluster_is_not_wired_into_scoring():
    """It is the treatment arm of a pre-registered A/B
    (`scripts/compare_news_truncation.py`), so the live digest must still be
    built from every relevant article — a silently-shipped cut would make the
    measurement meaningless."""
    import inspect

    import src.analysis.sentiment as sent
    src = inspect.getsource(sent.analyse_sentiment)
    assert "recent_cluster" not in src
