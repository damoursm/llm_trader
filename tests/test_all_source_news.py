"""The ALL-SOURCE news ingestion (2026-09-25, `src/data/news_coverage.py`).

Every per-ticker news feed must be asked about EVERY name a tick scores. Before
it, Step 1 fetched on the ~130-name universe that exists before the smart-money
/ macro / peer additions — measured on three RTH runs: of ~253 smart-money
names only ~68 had any relevant digest, at 1.4 articles — and the date/hour-keyed
event caches served the first call's list to every later call, so a name that
joined later never got those feeds at all. These tests pin the three mechanisms
(previous-universe prefetch, top-up, per-file coverage sidecars) and the feed
attribution test set 2's per-source features will be rebuilt from.
"""
from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone

import pytest

from config.settings import settings
from src.data import news_coverage as nc
from src.models import NewsArticle

NOW = datetime(2026, 9, 25, 14, 0, tzinfo=timezone.utc)


def _art(title, url=None, tickers=("AAA",)):
    return NewsArticle(title=title, url=url if url is not None else f"http://x/{title}",
                       source="Reuters", summary="body", published_at=NOW, tickers=list(tickers))


# ── previous-universe prefetch ───────────────────────────────────────────────

def test_feed_universe_round_trips_and_goes_stale():
    nc.save_feed_universe(["aaa", "BBB", "aaa", " "], run_id="r1")
    assert nc.load_feed_universe() == ["AAA", "BBB"]
    assert nc.load_feed_universe(now=datetime.now(timezone.utc) + timedelta(hours=97)) == []


def test_a_missing_or_corrupt_universe_file_means_no_prefetch():
    assert nc.load_feed_universe() == []
    nc.FEED_UNIVERSE_PATH.parent.mkdir(parents=True, exist_ok=True)
    nc.FEED_UNIVERSE_PATH.write_text("{not json", encoding="utf-8")
    assert nc.load_feed_universe() == []


def test_news_tickers_keep_step0_order_then_add_the_previous_ticks_names():
    """Step 0's order is what the capped legs walk; the previous tick's names
    follow, never duplicating a Step-0 name."""
    assert nc.news_tickers(["SPY", "AAPL"], ["aapl", "LIVN", "HAL", "LIVN"]) == \
        ["SPY", "AAPL", "LIVN", "HAL"]


# ── per-file coverage sidecars ───────────────────────────────────────────────

def _cache_pair(tmp_path, name="events_2026-09-25.json"):
    path = tmp_path / name
    store = {"arts": None}

    def load():
        return None if store["arts"] is None else list(store["arts"])

    def save(arts):
        store["arts"] = list(arts)
        path.write_text("[]", encoding="utf-8")
    return path, store, load, save


def test_fetch_with_coverage_fetches_only_the_names_it_has_not_covered(tmp_path):
    path, store, load, save = _cache_pair(tmp_path)
    asked = []

    def fetch(missing):
        asked.append(list(missing))
        return [_art(f"{t} news", tickers=[t]) for t in missing]

    out1 = nc.fetch_with_coverage("t", path, ["A", "B"], load, save, fetch)
    out2 = nc.fetch_with_coverage("t", path, ["A", "B", "C"], load, save, fetch)
    out3 = nc.fetch_with_coverage("t", path, ["C", "A"], load, save, fetch)
    assert asked == [["A", "B"], ["C"]]                 # the third call asks nothing
    assert [a.title for a in out1] == ["A news", "B news"]
    assert [a.title for a in out2] == ["A news", "B news", "C news"]
    assert out3 == out2
    assert nc.load_covered(path) == {"A", "B", "C"}


def test_a_name_that_returned_nothing_is_still_covered(tmp_path):
    """Asking again the same day returns the same nothing — a name with no
    analyst action must not cost a call on every tick."""
    path, store, load, save = _cache_pair(tmp_path)
    asked = []
    nc.fetch_with_coverage("t", path, ["A"], load, save, lambda m: asked.append(m) or [])
    nc.fetch_with_coverage("t", path, ["A"], load, save, lambda m: asked.append(m) or [])
    assert asked == [["A"]]


def test_a_legacy_cache_without_a_sidecar_is_refetched_and_merged_without_duplicates(tmp_path):
    """A file written before the sidecar existed has unknown coverage: every
    name asked about is fetched once and merged on (url, title)."""
    path, store, load, save = _cache_pair(tmp_path)
    store["arts"] = [_art("A news", tickers=["A"])]
    out = nc.fetch_with_coverage("t", path, ["A", "B"], load, save,
                                 lambda m: [_art(f"{t} news", tickers=[t]) for t in m])
    assert [a.title for a in out] == ["A news", "B news"]
    assert nc.load_covered(path) == {"A", "B"}


def test_a_failed_fetch_leaves_cache_and_sidecar_untouched(tmp_path):
    path, store, load, save = _cache_pair(tmp_path)

    def boom(missing):
        raise RuntimeError("provider down")
    with pytest.raises(RuntimeError):
        nc.fetch_with_coverage("t", path, ["A"], load, save, boom)
    assert store["arts"] is None and nc.load_covered(path) is None


def test_merge_keys_on_url_and_title_not_the_url_alone():
    """Ticker events and Quiver contracts stamp ONE url on every article they
    build — a URL-only merge would collapse them to one."""
    a = _art("AAA: ticker change", url="https://massive.com/")
    b = _art("BBB: delisting", url="https://massive.com/")
    assert [x.title for x in nc.merge_articles([a], [b, a])] == [a.title, b.title]


def test_sidecars_live_outside_the_globbed_bundle_names():
    """`news_replay` globs cache/news_*.json as bundle files; a sidecar beside
    them would be read as one."""
    side = nc.coverage_path(nc.CACHE_DIR / "news_2026-09-25_14.json")
    assert side.parent == nc.COVERAGE_DIR and side.parent != nc.CACHE_DIR


@pytest.mark.parametrize("module_name,public,inner,args", [
    ("src.data.analyst_ratings", "fetch_analyst_ratings", "_fetch_ratings", (30,)),
    ("src.data.earnings", "fetch_earnings_surprises", "_fetch_surprises", (90,)),
    ("src.data.short_interest", "fetch_short_interest", "_fetch_short", ()),
    ("src.data.ticker_events", "fetch_ticker_events", "_fetch_events", ()),
])
def test_every_date_keyed_event_cache_is_incremental_per_ticker(tmp_path, monkeypatch, module_name,
                                                                public, inner, args):
    import importlib
    mod = importlib.import_module(module_name)
    monkeypatch.setattr(mod, "CACHE_DIR", tmp_path)
    monkeypatch.setattr(settings, "enable_fetch_data", True)
    if module_name.endswith("ticker_events"):
        monkeypatch.setattr(settings, "enable_ticker_events", True)
        monkeypatch.setattr(mod.polygon_client, "is_available", lambda: True)
    asked = []

    def fake(tickers, *a, **k):
        asked.append(list(tickers))
        return [_art(f"{t} {module_name}", tickers=[t]) for t in tickers]
    monkeypatch.setattr(mod, inner, fake)
    fn = getattr(mod, public)
    first = fn(["AAA", "BBB"], *args)
    second = fn(["AAA", "BBB", "CCC"], *args)
    assert asked == [["AAA", "BBB"], ["CCC"]]
    assert len(first) == 2 and len(second) == 3
    # the cache FILE keeps the list shape `news_history` reads back
    files = [p for p in tmp_path.glob("*.json")]
    assert len(files) == 1 and len(json.loads(files[0].read_text(encoding="utf-8"))) == 3


# ── the hourly yfinance bundle ───────────────────────────────────────────────

def test_the_hourly_bundle_fetches_only_names_the_hour_has_not_covered(tmp_path, monkeypatch):
    import src.pipeline as pipeline
    from src.data import cache
    monkeypatch.setattr(cache, "CACHE_DIR", tmp_path)
    bundle_calls, yf_calls = [], []
    monkeypatch.setattr(pipeline, "fetch_cached_news",
                        lambda t, s: bundle_calls.append(list(t)) or [_art("first", tickers=["AAA"])])
    monkeypatch.setattr(pipeline, "fetch_ticker_news",
                        lambda t: yf_calls.append(list(t)) or [_art(f"yf {x}", tickers=[x]) for x in t])
    monkeypatch.setattr(pipeline, "fetch_rss_news", lambda: [])
    out1 = pipeline._fetch_news(["AAA", "BBB"], [])
    out2 = pipeline._fetch_news(["AAA", "BBB", "CCC"], [])
    out3 = pipeline._fetch_news(["CCC", "AAA"], [])
    assert bundle_calls == [["AAA", "BBB"]]              # NewsAPI + yfinance once per hour
    assert yf_calls == [["CCC"]]                         # later: only the uncovered name
    assert {a.title for a in out2} == {"first", "yf CCC"} == {a.title for a in out3}
    assert {a.title for a in out1} == {"first"}


def test_fetch_news_hands_back_the_bundle_and_rss_legs_apart(tmp_path, monkeypatch):
    import src.pipeline as pipeline
    from src.data import cache
    monkeypatch.setattr(cache, "CACHE_DIR", tmp_path)
    monkeypatch.setattr(pipeline, "fetch_cached_news", lambda t, s: [_art("bundled")])
    monkeypatch.setattr(pipeline, "fetch_rss_news", lambda: [_art("wire"), _art("bundled")])
    parts: dict = {}
    out = pipeline._fetch_news(["AAA"], [], parts=parts)
    assert [a.title for a in parts["bundle"]] == ["bundled"]
    assert [a.title for a in parts["rss"]] == ["wire", "bundled"]
    assert [a.title for a in out] == ["bundled", "wire"]


# ── feed attribution ─────────────────────────────────────────────────────────

def test_feed_attribution_keeps_every_deliverer_of_a_story():
    """The pool keeps the first copy of a URL; the attribution keeps every feed
    that delivered it, with the tickers THAT feed tagged."""
    story = _art("Acme wins contract", url="http://s/1", tickers=["ACME"])
    story_g = _art("Acme wins contract", url="http://s/1", tickers=["ACME", "PEER"])
    other = _art("Other", url="http://s/2")
    att = nc.feed_attribution({"bundle": [story], "google": [story_g, story_g],
                               "finnhub": [other], "rss": None})
    h1, h2 = nc.url_hash(story), nc.url_hash(other)
    assert att[h1] == {"bundle": ["ACME"], "google": ["ACME", "PEER"]}
    assert att[h2] == {"finnhub": ["AAA"]}


def test_url_hash_is_the_archives_key():
    import hashlib
    a = _art("t", url="http://s/9")
    assert nc.url_hash(a) == hashlib.sha1(b"http://s/9").hexdigest()
    assert nc.url_hash(_art("only a title", url="")) == hashlib.sha1(b"only a title").hexdigest()


def test_news_article_feeds_bump_the_tail_never_the_head():
    from src.db import repo
    story = _art("Acme wins contract", url="http://s/1", tickers=["ACME"])
    h = nc.url_hash(story)
    out1 = repo.insert_news_article_feeds("r1", NOW.isoformat(), {h: {"google": ["ACME"]}})
    later = (NOW + timedelta(minutes=30)).isoformat()
    out2 = repo.insert_news_article_feeds("r2", later, {h: {"google": ["ACME"], "finnhub": ["ACME"]}})
    assert out1 == {"new": 1, "seen": 0} and out2 == {"new": 1, "seen": 1}
    d = repo.fetch_df("SELECT feed, first_seen_at, last_seen_at, n_sightings, first_run_id "
                      "FROM news_article_feeds ORDER BY feed")
    g = d[d.feed == "google"].iloc[0]
    f = d[d.feed == "finnhub"].iloc[0]
    assert (g.first_seen_at, g.last_seen_at, int(g.n_sightings), g.first_run_id) == \
        (NOW.isoformat(), later, 2, "r1")
    assert (f.first_seen_at, f.last_seen_at, int(f.n_sightings)) == (later, later, 1)


# ── the top-up ───────────────────────────────────────────────────────────────

def test_the_topup_runs_every_enabled_leg_and_survives_a_failing_one(monkeypatch):
    import src.pipeline as pipeline
    for flag in ("enable_google_news", "enable_polygon_news", "enable_8k_filings",
                 "enable_analyst_ratings", "enable_earnings", "enable_short_interest",
                 "enable_ticker_events", "enable_finnhub_news"):
        monkeypatch.setattr(settings, flag, True)
    monkeypatch.setattr(settings, "finnhub_api_key", "x")
    monkeypatch.setattr(settings, "quiver_api_key", "")
    names_seen = {}

    def leg(name):
        def f(names, *a, **k):
            names_seen[name] = list(names)
            return [_art(f"{name} {n}", tickers=[n]) for n in names]
        return f
    monkeypatch.setattr(pipeline, "_bundle_topup", lambda key, names: leg("bundle")(names))
    monkeypatch.setattr(pipeline, "fetch_google_news", leg("google"))
    monkeypatch.setattr(pipeline, "fetch_finnhub_news", leg("finnhub"))
    monkeypatch.setattr(pipeline, "fetch_polygon_news", leg("polygon"))
    monkeypatch.setattr(pipeline, "fetch_8k_articles", leg("8k"))
    monkeypatch.setattr(pipeline, "fetch_analyst_ratings", leg("analyst"))
    monkeypatch.setattr(pipeline, "fetch_earnings_surprises", leg("eps"))
    monkeypatch.setattr(pipeline, "fetch_ticker_events", leg("ticker_events"))

    def boom(names):
        raise RuntimeError("short interest provider down")
    monkeypatch.setattr(pipeline, "fetch_short_interest", boom)
    out = pipeline._news_topup(["NEW1", "NEW2"], [])
    assert set(out) == {"bundle", "google", "finnhub", "polygon", "8k", "analyst", "eps",
                        "ticker_events"}                  # short failed alone
    assert all(v == ["NEW1", "NEW2"] for v in names_seen.values())


# ── Google News: a throttled sweep stops itself ──────────────────────────────

def test_a_throttled_google_sweep_stops_issuing_queries(monkeypatch):
    from src.data import news_fetcher

    class _Feed:
        status = 429
        entries = []
    calls = []
    monkeypatch.setattr(settings, "enable_google_news", True)
    monkeypatch.setattr(settings, "google_news_max_tickers", 0)
    monkeypatch.setattr(settings, "google_news_business_wire", True)
    monkeypatch.setattr(news_fetcher, "_google_name_query", lambda tk: None)
    monkeypatch.setattr(news_fetcher.feedparser, "parse", lambda url: calls.append(url) or _Feed())
    tickers = [f"T{i:03d}" for i in range(200)]           # 400 queries
    assert news_fetcher.fetch_google_news(tickers) == []
    # in-flight workers may finish a few past the threshold, never the whole sweep
    assert news_fetcher._GOOGLE_STOP_AFTER_NON200 <= len(calls) < 60


def test_google_news_has_no_ticker_cap_by_default():
    from config.settings import Settings
    assert Settings.model_fields["google_news_max_tickers"].default == 0


# ── Quiver dark pool: past the API cap, the deep store ───────────────────────

def test_offexchange_reads_the_deep_store_past_the_api_cap(monkeypatch):
    from src.data import quiver
    monkeypatch.setattr(settings, "enable_quiver_offexchange", True)
    monkeypatch.setattr(settings, "quiver_offexchange_max_tickers", 1)
    monkeypatch.setattr(quiver, "is_available", lambda: True)
    monkeypatch.setattr(quiver.time, "sleep", lambda s: None)
    rows = [{"Date": f"2026-08-{d:02d}", "DPI": 0.40} for d in range(1, 26)] + \
           [{"Date": f"2026-09-{d:02d}", "DPI": 0.60} for d in (1, 2, 3)]
    api, deep = [], []
    monkeypatch.setattr(quiver, "_get", lambda path, **k: api.append(path) or rows)
    monkeypatch.setattr(quiver, "_deep_dpi_rows", lambda tk: deep.append(tk) or rows)
    out = quiver.fetch_offexchange(["AAA", "BBB", "CCC", "AAA"])
    assert api == ["/historical/offexchange/AAA"]
    assert deep == ["BBB", "CCC"]
    assert {a.tickers[0] for a in out} == {"AAA", "BBB", "CCC"}


# ── news_shock: one ingestion regime per baseline ────────────────────────────

@pytest.mark.parametrize("on", [True, False])
def test_the_attention_baseline_reads_only_the_all_source_regime(monkeypatch, on):
    """Coverage roughly tripled for the post-fetch names, so a baseline from
    before the switch would read every one of them as an attention SHOCK."""
    from src.db import repo
    from src.signals import news_shock
    seen = []
    monkeypatch.setattr(settings, "enable_all_source_news", on)
    monkeypatch.setattr(repo, "fetch_df", lambda sql, *a, **k: seen.append(sql) or None)
    news_shock.load_attention_baselines(force=True)
    assert (f"generated_at >= '{nc.ALL_SOURCE_SINCE}'" in seen[0]) is on
    news_shock.reset_cache()


def test_the_regime_instant_is_a_utc_iso_string_comparable_to_generated_at():
    ts = datetime.fromisoformat(nc.ALL_SOURCE_SINCE)
    assert ts.tzinfo is not None and ts.utcoffset() == timedelta(0)
    assert nc.ALL_SOURCE_SINCE.startswith("2026-09-2")


def test_the_topup_never_creates_an_hours_bundle(tmp_path, monkeypatch):
    """Without the hour's bundle (Step 1's fetch failed) the top-up must not
    write one: it would lack the NewsAPI part and every later tick of the hour
    would take it for the real thing."""
    import src.pipeline as pipeline
    from src.data import cache
    monkeypatch.setattr(cache, "CACHE_DIR", tmp_path)
    monkeypatch.setattr(pipeline, "fetch_ticker_news",
                        lambda t: [_art(f"yf {x}", tickers=[x]) for x in t])
    assert pipeline._bundle_topup(cache._hour_key(), ["AAA"]) == []
    assert not list(tmp_path.glob("news_*.json"))
