"""News re-scoring from the ARCHIVED pool, and the historical-provider pilot
(2026-09-23).

The archive source is what stops a news-scorer change from resetting the
models' news history: every run's merged pool is in `news_articles`, so a run
can be re-scored under new logic and restored into the panel. Each guarantee
that makes that legitimate is pinned here — the pool is exactly the run's, no
reconstruction can reach the panel, a retired scorer's re-scores retire with
it, and the historical fetchers cannot see past the tick or run over the live
pipeline's shared quotas.
"""

import time as _time
from datetime import date, datetime, timedelta, timezone

import pandas as pd
import pytest

from src.analysis import news_replay as nr
from src.models import NewsArticle

T1 = datetime(2026, 9, 15, 14, 0, tzinfo=timezone.utc)
T2 = T1 + timedelta(minutes=30)
T3 = T1 + timedelta(minutes=60)


@pytest.fixture
def tmp_db(tmp_path, monkeypatch):
    from config.settings import settings
    monkeypatch.setattr(settings, "db_path", str(tmp_path / "t.db"))


def _na(url, minutes_before_t1, tickers=(), title="headline"):
    return NewsArticle(title=title, summary="body", url=url, source="Reuters",
                       published_at=T1 - timedelta(minutes=minutes_before_t1),
                       tickers=list(tickers))


def _archive(*runs):
    from src.db import repo
    for rid, when, arts in runs:
        repo.insert_news_articles(rid, when.isoformat(), arts)


# ── the archived pool ───────────────────────────────────────────────────────

def test_archive_pool_is_exactly_what_the_run_held(tmp_db):
    """`first_seen_at <= run <= last_seen_at`: an article fetched before the run
    and still served at it is in; one that had left the feeds is out; one first
    fetched by a LATER run is the future. Tags are the archived (fetch-confirmed)
    ones, never re-derived — re-deriving strips Polygon's own tags."""
    _archive(("r1", T1, [_na("A", 60, ["AAA"], title="no symbol in this text"), _na("B", 90)]),
             ("r2", T2, [_na("A", 60, ["AAA"]), _na("C", 10)]),
             ("r3", T3, [_na("A", 60, ["AAA"]), _na("D", 5)]))
    at_t2 = {a.url: a for a in nr._archive_pool(T2)}
    assert set(at_t2) == {"A", "C"}                    # B left after r1, D is the future
    assert at_t2["A"].tickers == ["AAA"]               # kept although the text never names it
    assert {a.url for a in nr._archive_pool(T1)} == {"A", "B"}


def test_archive_pool_refuses_an_article_from_after_the_run(tmp_db):
    """first_seen_at <= run, so a fetch cannot have returned an article published
    hours later — the archive's stamps would be wrong, and that must be loud."""
    _archive(("r1", T1, [_na("late", -180)]))            # published 3 h after r1 fetched it
    with pytest.raises(RuntimeError, match="not point-in-time"):
        nr._archive_pool(T1)


def test_polygon_insights_are_reattached_by_url(monkeypatch):
    """The archive does not keep Polygon's sentiment labels; without them the
    provider shortcut cannot fire where live's did, and the re-score would pay
    for LLM calls live never made."""
    poly = NewsArticle(title="t", summary="s", url="P1", source="Benzinga",
                       published_at=T1, tickers=["AAA"], provider_insights={"AAA": "positive"},
                       provider_sentiment_source="polygon")
    monkeypatch.setattr(nr, "_polygon_as_of", lambda when, universe: [poly])
    pool, n = nr._attach_polygon_insights([_na("P1", 5, ["AAA"]), _na("X", 5)], T1, {"AAA"})
    by = {a.url: a for a in pool}
    assert n == 1 and by["P1"].provider_insights == {"AAA": "positive"}
    assert by["P1"].provider_sentiment_source == "polygon"
    assert by["X"].provider_insights == {}


def test_the_archive_source_skips_the_bundle_path(monkeypatch):
    monkeypatch.setattr(nr, "_archive_pool", lambda when: [_na("A", 5, ["AAA"])])
    monkeypatch.setattr(nr, "_polygon_as_of", lambda when, universe: [])
    monkeypatch.setattr(nr, "_bundle_index",
                        lambda: (_ for _ in ()).throw(AssertionError("bundle path used")))
    pool, prov = nr.build_tick_pool(T1, {"AAA"}, source="archive")
    assert [a.url for a in pool] == ["A"]
    assert prov["pool_spec"] == nr.ARCHIVE_POOL_SPEC and prov["n_archive"] == 1


def test_the_replay_row_carries_its_certificate_and_scoring_epoch():
    """The certificate (the digest the scorer READ, compared with the live row's
    `news_digest_id`) and the epoch the values were produced under ride every
    row, and the writer names both — a column the INSERT omits is silently lost
    (the 2026-09-12 `news_quiet` defect)."""
    import inspect

    from src.db.repo import _NEWS_REPLAY_COLS
    src = inspect.getsource(nr._replay_one)
    assert '"news_digest_id": (meta or {}).get("digest_id")' in src
    assert '"news_epoch": news_epoch_stamp()' in src
    assert {"news_digest_id", "news_epoch"} <= set(_NEWS_REPLAY_COLS)


# ── the EXACT scorer input: the stored live digest ──────────────────────────

def test_digest_text_override_applies_only_to_the_same_digest():
    """A later run's cache hit re-used a verdict scored at an earlier clock, so
    the ages the model read are those of the ORIGINAL call — which only the
    stored text carries. The override is taken only for the identical digest
    (the id is ticker-salted), never for a different selection."""
    from src.analysis import sentiment as sent
    a = NewsArticle(title="t", summary="s", url="u", source="Reuters",
                    published_at=T1 - timedelta(hours=3))
    did = sent.digest_id_for("AAA", [a])
    rebuilt = sent._digest_text("AAA", [a], T1)
    assert rebuilt == "[Reuters | 3h ago] t\ns"                    # the live construction
    assert sent._digest_text("AAA", [a], T1, (did, "ORIGINAL")) == "ORIGINAL"
    assert sent._digest_text("AAA", [a], T1, ("another-id", "ORIGINAL")) == rebuilt
    assert sent._digest_text("BBB", [a], T1, (did, "ORIGINAL")) != "ORIGINAL"


def test_stored_live_digests_round_trip_to_the_same_digest_id(tmp_db):
    """What the re-score scores for a live-scored ticker is the article set the
    live scorer read: parsing the stored digest back reproduces its id exactly
    (url, source, title and the published instant all survive the store)."""
    import json

    from src.analysis import sentiment as sent
    from src.db import repo
    from src.db.schema import SIGNAL_METHOD_COLUMNS
    arts = [NewsArticle(title=f"story {i}", summary="x" * 500, url=f"https://ex/{i}",
                        source="Reuters", published_at=T1 - timedelta(hours=i + 1))
            for i in range(3)]
    did = sent.digest_id_for("AAA", arts)
    repo.insert_signals("r1", T1.isoformat(), "2026-09-15", [{
        "ticker": "AAA", "type": "STOCK", "direction": "BULLISH", "combined_score": 0.1,
        "price": 10.0, "scores": {m: 0.0 for m in SIGNAL_METHOD_COLUMNS}, "news_digest_id": did}])
    repo.insert_sentiment_digests([{
        "digest_id": did, "ticker": "AAA", "run_id": "r1", "generated_at": T1.isoformat(),
        "n_articles": 3, "digest_text": "THE TEXT THE MODEL READ",
        "articles_json": json.dumps([{"url": a.url, "source": a.source, "title": a.title,
                                      "published_at": a.published_at.isoformat(),
                                      "summary": a.summary[:400]} for a in arts])}])
    got = nr._live_digests("r1")
    gid, text, parsed = got["AAA"]
    assert gid == did and text == "THE TEXT THE MODEL READ"
    assert sent.digest_id_for("AAA", parsed) == did


def test_the_archive_rescore_scores_the_stored_digest_first():
    import inspect
    src = inspect.getsource(nr._replay_one)
    assert "digest_override=(live_d[0], live_d[1]) if live_d else None" in src
    assert "allow_provider=live_d is None" in src
    assert 'info["live_digests"] = _live_digests(run_id)' in inspect.getsource(nr.replay_tick)


def test_compare_to_live_certifies_digests_and_reports_every_column(tmp_db):
    from src.db import repo
    from src.db.schema import SIGNAL_METHOD_COLUMNS
    gen = T1.isoformat()
    base = {"type": "STOCK", "direction": "BULLISH", "combined_score": 0.1, "price": 10.0}
    repo.insert_signals("r1", gen, "2026-09-15", [
        {**base, "ticker": "AAA", "news_digest_id": "d1", "news_raw_score": 0.4,
         "scores": {**{m: 0.0 for m in SIGNAL_METHOD_COLUMNS}, "news": 0.30}},
        {**base, "ticker": "BBB", "news_digest_id": "d2", "news_raw_score": -0.2,
         "scores": {**{m: 0.0 for m in SIGNAL_METHOD_COLUMNS}, "news": -0.10}}])
    repo.insert_news_replay([
        {**_replay_row("AAA", "archive", "e", 0.30, gen, "2026-09-15"), "run_id": "r1",
         "news_raw_score": 0.4, "news_digest_id": "d1"},
        {**_replay_row("BBB", "archive", "e", -0.12, gen, "2026-09-15"), "run_id": "r1",
         "news_raw_score": -0.2, "news_digest_id": "zzz"}])
    out = nr.compare_to_live("archive")
    assert out["rows"] == 2 and out["digest_match"] == pytest.approx(0.5)
    assert out["columns"]["news"]["exact"] == pytest.approx(0.5)
    assert out["columns"]["news"]["within_0.02"] == pytest.approx(1.0)
    assert out["raw_on_live_digests"]["exact"] == pytest.approx(1.0)


# ── the panel restore ───────────────────────────────────────────────────────

def _replay_row(ticker, pool_spec, epoch, news, gen, day="2026-09-10"):
    return {"run_id": "r0", "ticker": ticker, "signal_date": day, "generated_at": gen,
            "replayed_at": "2026-09-23T20:00:00+00:00", "replay_version": nr.REPLAY_VERSION,
            "engine": "local", "pool_spec": pool_spec, "news_epoch": epoch, "news": news,
            "news_raw_score": news, "news_catalyst": "earnings", "news_recency_mass": 1.5,
            "news_article_count": 3}


def test_restore_takes_only_certified_archive_rows_under_the_epoch_in_force(tmp_db, monkeypatch):
    """A scorer change retires every earlier re-score automatically (its stamp no
    longer matches), and no reconstruction can reach the panel: only
    `RESTORABLE_POOL_SPECS` rows restore, only onto pre-epoch rows, and the
    epoch mask then spares exactly those cells."""
    import src.analysis.signal_panel as sp
    from src.db import repo
    assert nr.RESTORABLE_POOL_SPECS == ("archive",)
    stamp = nr.news_epoch_stamp()
    gen = "2026-09-10T19:30:00+00:00"                    # before the news epoch in force
    repo.insert_news_replay([
        _replay_row("AAA", "archive", stamp, 0.31, gen),                        # restores
        _replay_row("BBB", "union168h", stamp, 0.32, gen),                      # a reconstruction
        _replay_row("CCC", "archive", "2026-09-01T00:00:00+00:00", 0.33, gen),  # retired epoch
        _replay_row("DDD", nr.HIST_POOL_SPEC, stamp, 0.34, gen),                # the pilot arm
    ])
    monkeypatch.setattr(sp, "_close_series", lambda tk: {})
    sig = pd.DataFrame([{"generated_at": gen, "signal_date": "2026-09-10", "ticker": t,
                         "news": 0.9, "news_raw_score": 0.9}
                        for t in ("AAA", "BBB", "CCC", "DDD")])
    panel = sp.build_panel(horizons=(1,), signals_df=sig).set_index("ticker")
    assert panel.loc["AAA", "news"] == pytest.approx(0.31)            # re-scored, kept
    assert bool(panel.loc["AAA", "news_rescored"])
    for t in ("BBB", "CCC", "DDD"):
        assert pd.isna(panel.loc[t, "news"]), t                        # masked as before


def test_restore_never_touches_a_post_epoch_row(tmp_db):
    """A post-epoch row was produced live by the current code; an archive row
    for it exists only to certify the pipeline and must never overwrite it."""
    from src.db import repo
    from src.signals.method_epochs import epoch_for
    late = (epoch_for("news") + timedelta(days=3)).isoformat()
    gen = f"{late}T15:00:00+00:00"
    repo.insert_news_replay([_replay_row("AAA", "archive", nr.news_epoch_stamp(), 0.2, gen, late)])
    df = pd.DataFrame([{"generated_at": gen, "signal_date": late, "ticker": "AAA", "news": 0.7}])
    out, restored = nr.restore_news_rescored(df)
    assert restored == {} and out.loc[0, "news"] == 0.7


def test_the_stacker_news_mask_spares_rescored_rows():
    from src.analysis.ml_stacker import add_news_features
    df = pd.DataFrame([{"signal_date": "2026-09-10", "ticker": "AAA", "news_raw_score": 0.3,
                        "news_catalyst": "earnings", "news_rescored": True},
                       {"signal_date": "2026-09-10", "ticker": "BBB", "news_raw_score": 0.4,
                        "news_catalyst": "earnings", "news_rescored": False}])
    out = add_news_features(df).set_index("ticker")
    assert out.loc["AAA", "news_raw_score"] == pytest.approx(0.3)
    assert pd.isna(out.loc["BBB", "news_raw_score"])


# ── courtesy to the live scheduler ──────────────────────────────────────────

def test_live_tick_running_reads_the_scheduler_log(tmp_path):
    d = date(2026, 9, 23)
    log = tmp_path / "llm_trader_2026-09-23.log"
    log.write_text("x | INFO | runner - [scheduler] tick for 15:00 ET (rth)\n"
                   "x | INFO | pipeline - [db] Persisted run 2026-09-23_190002: 10\n"
                   "x | INFO | runner - [scheduler] tick for 15:30 ET (rth)\n", encoding="utf-8")
    assert nr.live_tick_running(tmp_path, today=d) is True
    with open(log, "a", encoding="utf-8") as fh:
        fh.write("x | INFO | pipeline - [db] Persisted run 2026-09-23_193005: 10\n")
    assert nr.live_tick_running(tmp_path, today=d) is False
    assert nr.live_tick_running(tmp_path / "missing", today=d) is False   # unknown = not running


# ── the historical-provider pilot ───────────────────────────────────────────

class _Feed:
    def __init__(self, status, entries=()):
        self.status, self.entries = status, list(entries)


def _entry(title, when):
    e = {"title": title, "link": f"https://news.google.com/{title}", "summary": title}
    if when is not None:
        e["published_parsed"] = _time.gmtime(when.timestamp())
    return e


@pytest.fixture(autouse=True)
def _tmp_cache_dir(tmp_path, monkeypatch):
    """The historical fetchers cache per (source, ticker, tick) under
    CACHE_DIR; a test must never write there (nor read another test's file)."""
    import src.data.cache as cache
    monkeypatch.setattr(cache, "CACHE_DIR", tmp_path / "cache")
    (tmp_path / "cache").mkdir()


def _quiet_fetch(monkeypatch):
    import src.data.news_fetcher as nf
    from config.settings import settings
    monkeypatch.setattr(nr.time, "sleep", lambda s: None)
    monkeypatch.setattr(nr, "live_tick_running", lambda *a, **k: False)
    monkeypatch.setattr(nf, "_google_name_query", lambda tk: None)
    monkeypatch.setattr(nf, "_confirmed_tags", lambda tk, t, s: [tk])
    monkeypatch.setattr(settings, "google_news_business_wire", False)


def test_google_history_keeps_the_live_24h_window_at_the_tick(monkeypatch):
    import feedparser
    _quiet_fetch(monkeypatch)
    feed = _Feed(200, [_entry("in", T1 - timedelta(hours=3)),
                       _entry("stale", T1 - timedelta(hours=30)),      # past live's 24h cap
                       _entry("future", T1 + timedelta(hours=2)),      # after the tick
                       _entry("undated", None)])
    monkeypatch.setattr(feedparser, "parse", lambda url: feed)
    arts, stat = nr._google_history(["AAA"], T1)
    assert [a.title for a in arts] == ["in"] and arts[0].tickers == ["AAA"]
    assert stat["requests"] == 1 and not stat["aborted"]


def test_google_history_aborts_on_repeated_throttling(monkeypatch):
    """The live pipeline reads Google News from this machine every tick; a
    backfill that keeps querying through 429s risks its quota, so three non-200
    answers in a row stop the historical leg."""
    import feedparser
    _quiet_fetch(monkeypatch)
    calls = []
    monkeypatch.setattr(feedparser, "parse", lambda url: calls.append(url) or _Feed(429))
    arts, stat = nr._google_history(["AAA", "BBB", "CCC", "DDD"], T1)
    assert arts == [] and stat["aborted"] and len(calls) == 3


def test_finnhub_history_is_the_live_fetch_as_of_the_tick(monkeypatch):
    import httpx
    from config.settings import settings
    monkeypatch.setattr(settings, "finnhub_api_key", "k")
    monkeypatch.setattr(nr.time, "sleep", lambda s: None)
    monkeypatch.setattr(nr, "live_tick_running", lambda *a, **k: False)
    items = ([{"datetime": int((T1 + timedelta(hours=1)).timestamp()), "headline": "after the tick",
               "url": "u-future", "source": "Yahoo"}]
             + [{"datetime": int((T1 - timedelta(minutes=i + 1)).timestamp()),
                 "headline": f"story {i}", "url": f"u{i}", "source": "Yahoo"} for i in range(20)])

    class _R:
        status_code = 200

        def raise_for_status(self):
            return None

        def json(self):
            return items

    seen = {}
    monkeypatch.setattr(httpx, "get",
                        lambda url, params=None, timeout=None: seen.update(params) or _R())
    arts, stat = nr._finnhub_history(["aaa"], T1)
    assert len(arts) == 15                                  # live's per-ticker cap
    assert "u-future" not in {a.url for a in arts}          # not yet published at the tick
    assert all(a.tickers == ["AAA"] for a in arts)
    assert seen["from"] == (T1 - timedelta(days=3)).date().isoformat()
    assert seen["to"] == T1.date().isoformat()


def test_an_archive_rescore_refuses_a_run_before_the_archive_began(tmp_db):
    from src.db import repo
    from src.db.schema import SIGNAL_METHOD_COLUMNS
    _archive(("r2", T2, [_na("A", 5)]))                       # the archive starts at T2
    repo.insert_signals("early", T1.isoformat(), "2026-09-15", [{
        "ticker": "AAA", "type": "STOCK", "direction": "BULLISH", "combined_score": 0.1,
        "price": 10.0, "scores": {m: 0.0 for m in SIGNAL_METHOD_COLUMNS}}])
    out = nr.replay_tick("early", source="archive")
    assert out["status"] == "before the news archive began"
    assert nr.archive_start() == T2


def test_a_ticker_live_never_read_a_digest_for_keeps_live_s_verdict():
    """The pool rebuild cannot tell whether the run's true pool held a digest for
    a ticker live abstained on or provider-scored; scoring one anyway invented
    views live never had. The verdict is live's, the derived family is current."""
    import inspect
    src = inspect.getsource(nr._replay_one)
    assert 'info.get("live_verdicts") is not None and live_d is None' in src
    assert 'news, meta = float(kept[0] or 0.0), {"raw_score": kept[1], "catalyst": kept[2]}' in src
    assert 'info["live_verdicts"] = _live_verdicts(run_id)' in inspect.getsource(nr.replay_tick)


def test_live_verdicts_read_the_persisted_values(tmp_db):
    from src.db import repo
    from src.db.schema import SIGNAL_METHOD_COLUMNS
    base = {"type": "STOCK", "direction": "BULLISH", "combined_score": 0.1, "price": 10.0}
    repo.insert_signals("r1", T1.isoformat(), "2026-09-15", [
        {**base, "ticker": "PRV", "scores": {**{m: 0.0 for m in SIGNAL_METHOD_COLUMNS}, "news": 0.2}},
        {**base, "ticker": "ABS", "scores": {m: 0.0 for m in SIGNAL_METHOD_COLUMNS}}])
    got = nr._live_verdicts("r1")
    assert got["PRV"] == (pytest.approx(0.2), None, None)     # provider shortcut: no raw, no class
    assert got["ABS"][0] == 0.0
