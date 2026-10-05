"""Per-source news history (`src/analysis/news_history.py`): every leg is rebuilt
with live's own rule at the tick and nothing after it, the provider legs are
resumable and back off, and a group is stored whole or not at all."""
import json
from datetime import date, datetime, timedelta, timezone

import pytest

from src.analysis import news_history as nh
from src.analysis import news_replay as nr
from src.models import NewsArticle

# 2026-09-16 03:30 UTC is 2026-09-15 23:30 ET: the tick keyed its daily caches
# and its `date.today()` windows on the ET date.
T = datetime(2026, 9, 16, 3, 30, tzinfo=timezone.utc)
D = date(2026, 9, 15)
WK = date(2026, 9, 14)


@pytest.fixture(autouse=True)
def _tmp_cache_dir(tmp_path, monkeypatch):
    import src.data.cache as cache
    monkeypatch.setattr(cache, "CACHE_DIR", tmp_path / "cache")
    (tmp_path / "cache").mkdir()
    monkeypatch.setattr(nh, "_SEC_8K", None)


@pytest.fixture
def quiet(monkeypatch):
    import src.data.news_fetcher as nf
    from config.settings import settings
    monkeypatch.setattr(nh.time, "sleep", lambda s: None)
    monkeypatch.setattr(nr, "live_tick_running", lambda *a, **k: False)
    monkeypatch.setattr(nr, "live_tick_phase", lambda *a, **k: None)
    monkeypatch.setattr(nh, "live_google_trouble_since", lambda *a, **k: None)
    monkeypatch.setattr(nf, "_google_name_query", lambda tk: None)
    monkeypatch.setattr(nf, "_confirmed_tags", lambda tk, t, s: [tk] if tk in t else [])
    monkeypatch.setattr(settings, "google_news_business_wire", False)


def _run(tickers=("AAA", "BBB"), when=T):
    return {"run_id": "R1", "when": when, "day": nh.local_day(when), "tickers": list(tickers),
            "signal_date": "2026-09-15", "prices": {t: 10.0 for t in tickers}}


def _fetched(after=T + timedelta(hours=2)):
    return after.isoformat()


def test_the_tick_is_keyed_on_its_eastern_date():
    assert nh.local_day(T) == D
    assert nh.week_start(D) == WK
    assert nh.local_day(datetime(2026, 9, 15, 23, 50, tzinfo=timezone.utc)) == D


def test_finnhub_rule_reproduces_the_live_request_at_the_tick():
    """Live asks `from=D-3, to=D` (UTC dates on the provider side), keeps
    nothing published after the tick, and takes the 15 newest non-noise."""
    def item(ts, head="AAA wins contract", src="Reuters"):
        return {"datetime": int(ts.timestamp()), "headline": head, "url": f"u{ts.timestamp()}",
                "source": src, "summary": "s"}
    items = [item(T - timedelta(minutes=5)),                        # in (UTC date 09-16!)
             item(datetime(2026, 9, 15, 20, 0, tzinfo=timezone.utc)),
             item(datetime(2026, 9, 12, 1, 0, tzinfo=timezone.utc)),   # D-3 -> in
             item(datetime(2026, 9, 11, 23, 0, tzinfo=timezone.utc)),  # before D-3 -> out
             item(T + timedelta(minutes=5))]                        # after the tick -> out
    out = nh._finnhub_rule(items, "AAA", T, D)
    days = [a.published_at.date().isoformat() for a in out]
    # 09-16 03:25 UTC is past `to=D` on a UTC-date window: live never saw it
    assert days == ["2026-09-15", "2026-09-12"]
    assert all(a.tickers == ["AAA"] and a.provider_sentiment_source == "finnhub" for a in out)
    many = [item(datetime(2026, 9, 15, 10, 0, tzinfo=timezone.utc) + timedelta(minutes=i))
            for i in range(30)]
    assert len(nh._finnhub_rule(many, "AAA", T, D)) == 15


def _gweek(tk, entries, n=None, fetched=None, qkind="sym"):
    p = nh._gweek_path(tk, WK, qkind)
    nh._write_json(p, {"fetched_at": fetched or _fetched(), "n": len(entries) if n is None else n,
                       "entries": entries})


def _gentry(title, pub, link=None):
    return {"title": title, "summary": "", "link": link or f"g/{title}",
            "source": "google_news/Wire", "published": pub.isoformat() if pub else None}


def test_google_leg_keeps_live_window_tags_and_merges_duplicates(quiet):
    _gweek("AAA", [_gentry("AAA in", T - timedelta(hours=3)),
                   _gentry("AAA stale", T - timedelta(hours=30)),
                   _gentry("AAA after", T + timedelta(hours=1)),
                   _gentry("AAA undated", None),
                   _gentry("AAA and BBB", T - timedelta(hours=1), link="g/shared")])
    _gweek("BBB", [_gentry("AAA and BBB", T - timedelta(hours=1), link="g/shared")])
    arts, prov = nh.google_leg(_run())
    by = {a.title: a for a in arts}
    assert set(by) == {"AAA in", "AAA and BBB"}
    assert by["AAA and BBB"].tickers == ["AAA", "BBB"]      # both searches confirmed it
    assert prov["google_undated"] == 1 and prov["google_missing"] == 0


def test_google_leg_waits_for_every_ticker_and_for_a_fresh_answer(quiet):
    _gweek("AAA", [_gentry("AAA in", T - timedelta(hours=3))])
    assert nh.google_leg(_run())[0] is None                    # BBB not asked yet
    _gweek("BBB", [], fetched=(T - timedelta(hours=5)).isoformat())
    assert nh.google_leg(_run())[0] is None                    # asked BEFORE the tick
    kinds = [t[0] for t in nh.google_tasks([_run()])]
    assert kinds == ["week"]                                   # only the stale one again


def test_a_busy_google_week_is_reasked_per_run(quiet):
    _gweek("AAA", [_gentry("AAA week", T - timedelta(hours=2))], n=nh.GOOGLE_SPLIT_MIN)
    _gweek("BBB", [])
    run = _run()
    tasks = nh.google_tasks([run])
    assert [(t[0], t[1]) for t in tasks] == [("run", "AAA")]
    assert nh.google_leg(run)[0] is None
    nh._write_json(nh._grun_path("AAA", T, "sym"),
                   {"fetched_at": _fetched(), "n": 1,
                    "entries": [_gentry("AAA split", T - timedelta(hours=4))]})
    arts, _ = nh.google_leg(run)
    assert {a.title for a in arts} == {"AAA week", "AAA split"}


def test_acquire_google_backs_off_then_stops_and_caches_nothing(quiet, monkeypatch):
    calls = []
    monkeypatch.setattr(nh, "_google_fetch", lambda q, a, b: (calls.append(q), (429, []))[1])
    out = nh.acquire_google([_run()], pace_s=0)
    assert out["stopped"] and out["non200"] == {429: nh.GOOGLE_MAX_BAD + 1}
    assert len(calls) == nh.GOOGLE_MAX_BAD + 1
    assert not list(nh._raw_dir("google", "AAA").glob("*.json"))


def test_acquire_google_caches_answers_and_resumes(quiet, monkeypatch):
    monkeypatch.setattr(nh, "_google_fetch",
                        lambda q, a, b: (200, [_gentry("AAA x", T - timedelta(hours=1))]))
    out = nh.acquire_google([_run()], pace_s=0)
    assert out["ok"] == 2 and out["remaining"] == 0
    monkeypatch.setattr(nh, "_google_fetch", lambda *a: pytest.fail("refetched a cached answer"))
    assert nh.acquire_google([_run()], pace_s=0)["requests"] == 0


def test_the_live_google_alarm_is_read_from_the_scheduler_log(tmp_path):
    now = datetime.now()
    log = tmp_path / f"llm_trader_{now:%Y-%m-%d}.log"
    stamp = (now - timedelta(minutes=5)).strftime("%Y-%m-%d %H:%M:%S")
    log.write_text(
        f"{stamp}.123 | WARNING  | src.data.news_fetcher:fetch_google_news:352 - Google News: "
        f"12 feed(s) returned a non-200 status {{429: 12}} — throttled\n", encoding="utf-8")
    assert nh.live_google_trouble_since(now - timedelta(minutes=10), log_dir=tmp_path)
    assert nh.live_google_trouble_since(now - timedelta(minutes=1), log_dir=tmp_path) is None
    log.write_text(f"{stamp}.123 | INFO     | src.data.news_fetcher:fetch_google_news:355 - "
                   f"Google News: 0 articles across 118 tickers (BW=True)\n", encoding="utf-8")
    assert nh.live_google_trouble_since(now - timedelta(minutes=10), log_dir=tmp_path)


def _fweek(tk, items, n=None):
    nh._write_json(nh._fweek_path(tk, WK), {"fetched_at": _fetched(),
                                            "n": len(items) if n is None else n, "items": items})


def test_a_truncated_finnhub_week_uses_live_exact_day_request(quiet):
    item = {"datetime": int((T - timedelta(hours=8)).timestamp()), "headline": "AAA news",
            "url": "f/1", "source": "Reuters"}
    _fweek("AAA", [item], n=nh.FINNHUB_CAP)
    _fweek("BBB", [])
    run = _run()
    assert [(t[0], t[1], t[3]) for t in nh.finnhub_tasks([run])] == [("day", "AAA", D)]
    assert nh.finnhub_leg(run)[0] is None
    nh._write_json(nh._fday_path("AAA", D), {"fetched_at": _fetched(), "n": 1,
                                             "items": [dict(item, url="f/day")]})
    arts, prov = nh.finnhub_leg(run)
    assert [a.url for a in arts] == ["f/day"] and prov["finnhub_truncated_weeks"] == 1


def test_event_feeds_come_from_the_caches_the_tick_read(quiet):
    from src.data.cache import CACHE_DIR
    art = NewsArticle(title="AAA beats", summary="EPS", url="u/eps", source="Earnings/EPS",
                      published_at=T - timedelta(hours=20), tickers=["AAA"])
    (CACHE_DIR / f"earnings_surprises_{D.isoformat()}.json").write_text(
        json.dumps([art.model_dump(mode="json")]), encoding="utf-8")
    # the NEXT ET day's file must not leak into this tick
    (CACHE_DIR / f"analyst_ratings_{(D + timedelta(days=1)).isoformat()}.json").write_text(
        json.dumps([art.model_dump(mode="json")]), encoding="utf-8")
    legs, prov = nh.event_legs(_run())
    assert [a.title for a in legs["eps"]] == ["AAA beats"]
    assert legs["analyst"] == [] and "analyst" in prov["events_missing_caches"]


def test_quiver_legs_are_rebuilt_by_live_builders_from_that_days_payload(quiet, monkeypatch):
    import httpx

    from config.settings import settings
    from src.data.cache import CACHE_DIR
    monkeypatch.setattr(httpx, "get", lambda *a, **k: pytest.fail("fetched Quiver"))
    monkeypatch.setattr(settings, "enable_quiver_gov_contracts", True)
    rows = [{"Ticker": "AAA", "Date": "2026-09-10", "Amount": 5e6, "Agency": "DoD"},
            {"Ticker": "AAA", "Date": "2026-07-01", "Amount": 1e6, "Agency": "DoE"},   # > 30d before D
            {"Ticker": "ZZZ", "Date": "2026-09-10", "Amount": 1e6, "Agency": "DoD"}]   # not in universe
    (CACHE_DIR / f"quiver_live_govcontractsall_{D.isoformat()}.json").write_text(
        json.dumps(rows), encoding="utf-8")
    legs, _ = nh._quiver_legs(["AAA", "BBB"], D)
    assert [a.title[:14] for a in legs["quiver_contracts"]] == ["AAA awarded $5"]
    assert legs["quiver_lobbying"] == [] and legs["quiver_darkpool"] == []


def test_eight_k_rebuilt_from_the_deep_store_point_in_time(quiet, monkeypatch):
    import pandas as pd
    monkeypatch.setattr(nh, "_SEC_8K", pd.DataFrame([
        {"ticker": "AAA", "cik": "0000000001", "accession": "0001-26-000001", "filing_date": "2026-09-14",
         "acceptance": datetime(2026, 9, 14, 20, 0), "form": "8-K", "items": "2.02,9.01",
         "primary_doc": "a.htm"},
        {"ticker": "AAA", "cik": "0000000001", "accession": "0001-26-000002", "filing_date": "2026-09-16",
         "acceptance": datetime(2026, 9, 16, 12, 0), "form": "8-K", "items": "8.01",
         "primary_doc": "b.htm"},                                  # accepted AFTER the tick
        {"ticker": "AAA", "cik": "0000000001", "accession": "0001-26-000003", "filing_date": "2026-09-01",
         "acceptance": datetime(2026, 9, 1, 20, 0), "form": "8-K", "items": "8.01",
         "primary_doc": "c.htm"},                                  # beyond the 5-day look-back
        {"ticker": "BBB", "cik": "0000000002", "accession": "0002-26-000001", "filing_date": "2026-09-15",
         "acceptance": datetime(2026, 9, 15, 13, 0), "form": "8-K", "items": "9.01",
         "primary_doc": "d.htm"}]))                                # exhibit-only: no material item
    arts = nh._eight_k_leg(["AAA", "BBB"], T, D)
    assert len(arts) == 1 and arts[0].source == "SEC 8-K Filing"
    assert "a.htm" in arts[0].url and arts[0].tickers == []       # live's 8-Ks carry no tag


def test_the_all_group_merges_in_live_order_first_url_wins():
    a_bundle = NewsArticle(title="bundle copy", summary="", url="u/1", source="Zacks",
                           published_at=T, tickers=["AAA"])
    a_poly = NewsArticle(title="polygon copy", summary="", url="u/1", source="Zacks Investment",
                         published_at=T, tickers=["AAA", "BBB"])
    legs = {leg: [] for leg in nh.ALL_LEGS}
    legs["bundle"], legs["polygon"] = [a_bundle], [a_poly]
    assert [a.title for a in nh.group_pool("all", legs)] == ["bundle copy"]
    assert [a.title for a in nh.group_pool("polygon", legs)] == ["polygon copy"]
    legs["google"] = None
    assert nh.group_pool("all", legs) is None and nh.group_pool("bundle", legs) is not None


def test_mention_index_answers_exactly_like_the_real_function(monkeypatch):
    from src.data import company_names as cn
    real = lambda tk, text, allow_token=False: tk in text.split()       # noqa: E731
    monkeypatch.setattr(cn, "mentions", real)
    arts = [NewsArticle(title="AAA rises", summary="", url="1", source="x", published_at=T),
            NewsArticle(title="BBB and AAA", summary="deal", url="2", source="x", published_at=T)]
    with nh._MentionIndex(arts, ["AAA", "BBB", "CCC"]) as idx:
        assert cn.mentions is not real
        for a in arts:
            text = f"{a.title or ''} {a.summary or ''}"
            for tk in ("AAA", "BBB", "CCC"):
                assert cn.mentions(tk, text) == real(tk, text)
        assert cn.mentions("DDD", "DDD alone") is True                 # outside the index
    assert cn.mentions is real


def test_a_group_with_a_failed_call_is_not_stored(monkeypatch):
    stored = []
    monkeypatch.setattr(nh, "groups_done", lambda rid: {})
    monkeypatch.setattr(nh, "run_legs", lambda run, index=None, need=(): (
        {leg: [] for leg in nh.ALL_LEGS}, {"bundle_file": None}))
    monkeypatch.setattr(nh, "_write_run_prov", lambda *a: None)
    monkeypatch.setattr(nr, "baselines_as_of", lambda *a: {})
    monkeypatch.setattr(nr, "live_tick_running", lambda *a, **k: False)
    monkeypatch.setattr(nr, "live_tick_phase", lambda *a, **k: None)
    from src.db import repo
    monkeypatch.setattr(repo, "insert_news_replay", lambda rows: stored.append(rows))
    monkeypatch.setattr(nr, "_replay_one", lambda tk, pool, info, engine, prov, bl: {
        "ticker": tk, "news": 0.0, "scorer_failed": tk == "BBB"})
    out = nh.score_run(_run(), groups=("events", "polygon"), tick_aware=False)
    assert stored == []
    assert all("failure" in v for v in out["groups"].values())
    monkeypatch.setattr(nr, "_replay_one", lambda tk, pool, info, engine, prov, bl: {
        "ticker": tk, "news": 0.1, "scorer_failed": False, "pool_spec": prov["pool_spec"]})
    nh.score_run(_run(), groups=("events",), tick_aware=False)
    assert len(stored) == 1 and {r["pool_spec"] for r in stored[0]} == {"src:events"}
    assert {r["replay_version"] for r in stored[0]} == {nh.HISTORY_VERSION}


def test_live_tick_phase_follows_the_scheduler_log(tmp_path):
    """A live tick holds the feeds only while it fetches and the local LLM only
    while it scores sentiment; the phase is read off its own log lines."""
    day = date(2026, 9, 23)
    log = tmp_path / "llm_trader_2026-09-23.log"
    lines = ["2026-09-23 21:30:02.252 | INFO     | runner - [scheduler] tick for 21:30 ET (overnight)"]

    def phase(now=datetime(2026, 9, 23, 21, 50)):
        log.write_text("\n".join(lines) + "\n", encoding="utf-8")
        return nr.live_tick_phase(log_dir=tmp_path, today=day, now=now)

    assert phase() == "fetch"
    lines.append("2026-09-23 21:33:14.744 | INFO     | pipeline - Steps 1–3: 2796 total articles assembled")
    assert phase() == "prep"
    lines.append("2026-09-23 21:42:28.790 | INFO     | agg - Signal weights [TRENDING] — pattern=7%")
    assert phase() == "sentiment"
    lines.append("2026-09-23 21:49:02.993 | INFO     | agg - [aggregator] rank pool: 334/359 tradeable")
    assert phase() == "post"
    assert phase(now=datetime(2026, 9, 23, 22, 30)) is None       # never persisted: killed
    lines.append("2026-09-23 22:06:38.749 | INFO     | pipeline - [db] Persisted run 2026-09-24_013002")
    assert phase() is None


def test_paired_compares_two_sources_on_the_same_names(monkeypatch):
    """Each source's own IC is measured on a different population; the paired
    read keeps only the rows where BOTH sources hold a view."""
    import numpy as np
    import pandas as pd

    import src.analysis.signal_panel as sp
    rng = np.random.default_rng(0)
    rows, lab = [], []
    for d in range(12):
        for i in range(20):
            tk, y = f"T{i}", float(rng.normal())
            lab.append({"run_id": f"R{d}", "ticker": tk, "fwd_ret_pivot": y, "fwd_ret_5d": y})
            good = y + 0.1 * float(rng.normal())                 # source A reads the label
            noise = float(rng.normal())                          # source B does not
            rows.append({"run_id": f"R{d}", "ticker": tk, "signal_date": f"2026-07-{d + 1:02d}",
                         "generated_at": "", "pool_spec": "src:polygon", "news": good})
            rows.append({"run_id": f"R{d}", "ticker": tk, "signal_date": f"2026-07-{d + 1:02d}",
                         "generated_at": "", "pool_spec": "src:google",
                         "news": noise if i < 15 else 0.0})      # google abstains on 5 names
    monkeypatch.setattr(nh, "load_group_features", lambda groups=None: pd.DataFrame(rows))
    monkeypatch.setattr(sp, "build_panel", lambda **k: pd.DataFrame(lab))
    res = nh.paired(("polygon", "google"))
    assert res["set2_live_all_source"] == {} or "polygon - google" not in res["set2_live_all_source"]         or res["set2_live_all_source"]["polygon - google"]["rows"] == 0   # July rows belong to set 1 only
    out = res["set1_history"]["polygon - google"]
    assert out["days"] == 12 and out["rows"] == 12 * 15
    assert out["ic_diff"] > 0.5 and out["t"] > 2
    cov = nh.coverage_by_month(("polygon", "google"))
    assert cov["polygon"]["2026-07"] == 1.0 and cov["google"]["2026-07"] == 0.75


def test_the_pacer_holds_a_rate_not_a_sleep_after_each_request(monkeypatch):
    """The request's own latency counts toward the spacing: a request that took
    0.8 s is followed by a 1.2 s wait at a 2 s interval, not by a full 2 s."""
    clock = {"t": 100.0}
    slept = []
    monkeypatch.setattr(nh.time, "monotonic", lambda: clock["t"])
    monkeypatch.setattr(nh.time, "sleep", lambda s: (slept.append(round(s, 3)),
                                                      clock.__setitem__("t", clock["t"] + s)))
    p = nh._Pacer(2.0)
    p.wait()                       # first request: no wait
    clock["t"] += 0.8              # the request took 0.8 s
    p.wait()
    clock["t"] += 2.5              # a slow request: already past the interval
    p.wait()
    assert slept == [1.2]


def test_feed_universe_is_the_names_live_fetched_news_for(monkeypatch):
    """Live's per-ticker feeds ran on the universe as it stood at Step 1; the
    smart-money / macro / cointegration additions came after and were never asked."""
    import pandas as pd

    from src.db import repo
    rows = pd.DataFrame({"ticker": ["AAA", "BBB", "CCC", "DDD"],
                         "universe_source": ["watchlist", "smart_money", "trending", "coint_peer"]})
    monkeypatch.setattr(repo, "fetch_df", lambda sql, params=None: rows)
    names, known, sources = nh._feed_universe("R1", ["AAA", "BBB", "CCC", "DDD"])
    assert (names, known, sources["CCC"]) == (["AAA", "CCC"], True, "trending")
    rows.loc[0, "universe_source"] = None                   # before labels were recorded
    assert nh._feed_universe("R1", ["AAA", "BBB"]) == (["AAA", "BBB"], False, {})


def test_per_ticker_feeds_skip_names_added_after_the_fetch(quiet):
    run = dict(_run(), feed_tickers=["AAA"], feed_known=True)
    _gweek("AAA", [_gentry("AAA in", T - timedelta(hours=3))])
    assert {t[1] for t in nh.google_tasks([run])} == set()   # BBB is never asked
    arts, prov = nh.google_leg(run)
    assert [a.title for a in arts] == ["AAA in"] and prov["google_missing"] == 0
    assert {t[1] for t in nh.finnhub_tasks([run])} == {"AAA"}


def test_live_order_puts_the_pinned_lists_first_and_caps_finnhub_at_60(monkeypatch):
    """Live asks Finnhub for the first 60 of its Step-0 order: watchlist, sector
    ETFs and trending picks (from `get_trending_tickers`), then commodities and
    factor ETFs, then the discovered names."""
    from config.settings import Settings
    monkeypatch.setattr(Settings, "stocks_list", property(lambda self: ["WB", "WA"]))
    monkeypatch.setattr(Settings, "sectors_list", property(lambda self: ["XLK"]))
    monkeypatch.setattr(Settings, "commodities_list", property(lambda self: ["GLD"]))
    monkeypatch.setattr(Settings, "factor_list", property(lambda self: ["MTUM"]))
    screener = [f"S{i:03d}" for i in range(80)]
    sources = {"WA": "watchlist", "WB": "watchlist", "XLK": "sector_etf", "TRN": "trending",
               "GLD": "commodity", "MTUM": "factor_etf", **{t: "screener" for t in screener}}
    run = dict(_run(tickers=list(sources)), feed_tickers=sorted(sources), feed_known=True,
               feed_sources=sources)
    order = nh._live_order(run)
    assert order[:6] == ["WB", "WA", "XLK", "TRN", "GLD", "MTUM"]
    assert nh._finnhub_names(run) == order[:60] and len(nh._finnhub_names(run)) == 60


def test_coverage_groups_are_not_built_where_live_coverage_is_unknown(monkeypatch):
    """Before 2026-07-03 nothing records which names live fetched news for, so
    the Finnhub / Google / all groups are skipped there (never waited for)."""
    called = []
    monkeypatch.setattr(nh, "groups_done", lambda rid: {})
    monkeypatch.setattr(nh, "run_legs", lambda run, index=None, need=(): (
        called.append(tuple(need)) or {leg: [] for leg in nh.ALL_LEGS}, {}))
    monkeypatch.setattr(nh, "_write_run_prov", lambda *a: None)
    monkeypatch.setattr(nr, "baselines_as_of", lambda *a: {})
    monkeypatch.setattr(nr, "live_tick_phase", lambda *a, **k: None)
    from src.db import repo
    stored = []
    monkeypatch.setattr(repo, "insert_news_replay", lambda rows: stored.append(rows[0]["pool_spec"]))
    monkeypatch.setattr(nr, "_replay_one", lambda tk, pool, info, engine, prov, bl: {
        "ticker": tk, "news": 0.0, "scorer_failed": False, "pool_spec": prov["pool_spec"]})
    out = nh.score_run(dict(_run(), feed_known=False), tick_aware=False)
    assert sorted(stored) == ["src:bundle", "src:events", "src:polygon"]
    assert set(out["groups"]) == {"events", "polygon", "bundle"}
    assert nh.google_leg(dict(_run(), feed_known=False))[0] is None
    assert nh.finnhub_tasks([dict(_run(), feed_known=False)]) == []


def test_source_feature_frame_is_one_column_per_feature_and_source(monkeypatch):
    """The by-source matrix: `<feature>@<group>` columns keyed run-exact; a source
    read with no view is 0.0, a source not built for the run is NaN."""
    import math

    import pandas as pd

    from src.db import repo
    rows = pd.DataFrame([
        {"run_id": "R1", "ticker": "AAA", "signal_date": "2026-07-06", "generated_at": "g1",
         "pool_spec": "src:polygon", "news": 0.25, "news_catalyst": "earnings"},
        {"run_id": "R1", "ticker": "AAA", "signal_date": "2026-07-06", "generated_at": "g1",
         "pool_spec": "src:google", "news": 0.0, "news_catalyst": None},
        {"run_id": "R1", "ticker": "BBB", "signal_date": "2026-07-06", "generated_at": "g1",
         "pool_spec": "src:polygon", "news": -0.4, "news_catalyst": "analyst"},
    ])
    monkeypatch.setattr(repo, "fetch_df", lambda sql, params=None: rows.copy())
    wide = nh.source_feature_frame(groups=("polygon", "google"), features=("news", "news_catalyst"))
    assert list(wide.columns) == ["run_id", "ticker", "signal_date", "generated_at", "news@polygon",
                                  "news_catalyst@polygon", "news@google", "news_catalyst@google"]
    a = wide.set_index("ticker").loc["AAA"]
    b = wide.set_index("ticker").loc["BBB"]
    assert a["news@polygon"] == 0.25 and a["news@google"] == 0.0          # read, no view
    assert b["news@polygon"] == -0.4 and math.isnan(float(b["news@google"]))  # not built


def test_a_network_failure_is_retried_not_counted_as_throttling(quiet, monkeypatch):
    """No HTTP answer (a timeout, a DNS failure) is the network, not Google's
    verdict on this machine: it is retried, and never trips the throttle stop."""
    answers = [(-1, []), (-1, []), (-1, []), (200, [_gentry("AAA x", T - timedelta(hours=1))])]
    monkeypatch.setattr(nh, "_google_fetch", lambda q, a, b: answers.pop(0) if answers else (200, []))
    out = nh.acquire_google([_run(tickers=("AAA",))], pace_s=0)
    assert out["stopped"] is None and out["non200"] == {} and out["ok"] == 1
    assert nh._read_json(nh._gweek_path("AAA", WK, "sym"))["n"] == 1


def test_one_unreadable_answer_is_stored_like_live_but_a_server_outage_is_not(monkeypatch):
    """A single failure beside many readable verdicts is the model's (live
    records 0.0 for it); failures without readable verdicts beside them are the
    server's, and nothing is stored."""
    good = [{"ticker": f"G{i}", "news": 0.1, "news_raw_score": 0.1, "scorer_failed": False}
            for i in range(10)]
    bad = {"ticker": "SPY", "news": 0.0, "news_raw_score": None, "scorer_failed": True}
    assert nh.content_failures_only(good + [bad])
    assert not nh.content_failures_only(good[:3] + [bad])              # too few good answers
    assert not nh.content_failures_only([dict(bad, ticker=f"B{i}") for i in range(5)] + good)
    assert not nh.content_failures_only(good)                          # nothing failed

    stored = []
    monkeypatch.setattr(nh, "groups_done", lambda rid: {})
    monkeypatch.setattr(nh, "run_legs", lambda run, index=None, need=(): (
        {leg: [] for leg in nh.ALL_LEGS}, {"bundle_file": None}))
    monkeypatch.setattr(nh, "_write_run_prov", lambda *a: None)
    monkeypatch.setattr(nr, "baselines_as_of", lambda *a: {})
    monkeypatch.setattr(nr, "live_tick_phase", lambda *a, **k: None)
    from src.db import repo
    monkeypatch.setattr(repo, "insert_news_replay", lambda rows: stored.append(rows))
    tickers = [f"G{i}" for i in range(10)] + ["SPY"]
    monkeypatch.setattr(nr, "_replay_one", lambda tk, pool, info, engine, prov, bl: (
        dict(bad) if tk == "SPY" else {"ticker": tk, "news": 0.1, "news_raw_score": 0.1,
                                        "scorer_failed": False}))
    nh.score_run(_run(tickers=tickers), groups=("events",), tick_aware=False)
    assert len(stored) == 1 and len(stored[0]) == 11                    # stored, SPY at 0.0
    stored.clear()
    monkeypatch.setattr(nr, "_replay_one", lambda tk, pool, info, engine, prov, bl: dict(bad, ticker=tk))
    nh.score_run(_run(tickers=tickers), groups=("polygon",), tick_aware=False)
    assert stored == []                                                 # server down: refused


def test_after_the_all_source_ingestion_every_scored_name_is_covered_uncapped(monkeypatch):
    """From `news_coverage.ALL_SOURCE_SINCE` live asks every per-ticker feed
    about every scored name with no Finnhub / Google cap; a rebuild of such a
    run must not re-impose the pre-fetch restriction or the caps — and a rebuild
    of an EARLIER run must keep them, whatever the live setting says now."""
    from src.data.news_coverage import ALL_SOURCE_SINCE
    names = [f"N{i:03d}" for i in range(200)]
    sources = {t: "screener" for t in names}
    before = dict(_run(tickers=names), feed_tickers=names, feed_known=True, feed_sources=sources)
    after = dict(before, when=datetime.fromisoformat(ALL_SOURCE_SINCE) + timedelta(minutes=30))
    assert len(nh._finnhub_names(before)) == nh.LIVE_FINNHUB_MAX
    assert len(nh._google_names(before)) == nh.LIVE_GOOGLE_MAX
    assert len(nh._finnhub_names(after)) == len(nh._google_names(after)) == 200

    from src.db import repo
    import pandas as pd
    rows = pd.DataFrame({"ticker": ["PRE", "SM1"], "universe_source": ["watchlist", "smart_money"]})
    monkeypatch.setattr(repo, "fetch_df", lambda *a, **k: rows)
    t_after = datetime.fromisoformat(ALL_SOURCE_SINCE) + timedelta(hours=1)
    rid_after = t_after.strftime("%Y-%m-%d_%H%M%S")
    assert nh._feed_universe(rid_after, ["PRE", "SM1"])[0] == ["PRE", "SM1"]
    assert nh._feed_universe("2026-09-15_143000", ["PRE", "SM1"])[0] == ["PRE"]
