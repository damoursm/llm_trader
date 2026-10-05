"""The vol arm's added stocks, the short-sale restriction in the pick journal, the entry
step's journal and the trade log (user directives 2026-10-05: "Yes please do all of that" —
fix the vol arm's frozen universe; log, for every vol trade, the restriction state, the borrow
at entry and the broker's actual fill — and "Add them to live and backtest the results are
good": every common stock outside the store, whatever its listing date).

What must hold: the vol arm ranks the model's names plus its recorded stocks, the model arm
never one of them, the ETF arm never one of them (and the vol arm never an added product, with
or without the session snapshot); the screen reads only sessions BEFORE the day, keeps common
stocks / ADRs, whatever their listing date, that clear the dollar-volume floor over >= 10
sessions, and nothing the store, the model or the record already holds; adding a stock
ingests it, refreshes the deep universe, records it and seeds its vol history on existing day
files with the bars the live scorer would have scored; the restriction state is journaled with
every pick; the entry step journals every pick it settles with the borrow IBKR showed; the
trade log joins pick, entry step, ledger trade and broker events.
"""
from __future__ import annotations

import json
from datetime import date, datetime, timedelta
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import pytest

from config.settings import settings
from src.signals import sel_short as ss

ET = ZoneInfo("America/New_York")


@pytest.fixture
def on(monkeypatch):
    monkeypatch.setattr(settings, "enable_sel_short", True)
    monkeypatch.setattr(settings, "enable_sel_short_vol", True)
    monkeypatch.setattr(settings, "enable_sel_short_vol_added_stocks", True)
    return settings


def _write_listings(names):
    p = ss.listings_path()
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps({n: {"added": "2026-10-05", "type": "CS"} for n in names}), encoding="utf-8")


def _res(n=25):
    return pd.DataFrame({"ticker": [f"N{i:02d}" for i in range(n)], "score": np.linspace(1.0, 0.0, n),
                         "vol": np.linspace(0.0, 5.0, n), "px": 20.0, "dv20": 1e7, "pre5": 10.0, "status": "OK"})


def _stand(res):
    return pd.DataFrame({"n_prior": 0.0, "prior_max": np.nan, "prior_min": np.nan},
                        index=pd.Index(res.ticker, name="ticker"))


# ── routing ──────────────────────────────────────────────────────────────────

def test_the_vol_arm_ranks_its_added_stocks_and_no_other_arm_does(monkeypatch, on):
    monkeypatch.setattr(ss, "load_model", lambda: (None, {"tickers": ["AAA", "UVIX"]}))
    monkeypatch.setitem(ss._TYPES, "t", {"AAA": "CS", "UVIX": "ETF", "NEWETF": "ETF", "NEWCO": "CS"})
    _write_listings(["NEWCO"])
    res = pd.DataFrame({"ticker": ["AAA", "UVIX", "NEWETF", "NEWCO"], "status": ["OK", "OK", "VOL_ONLY", "VOL_ONLY"],
                        "score": [1.0, 0.5, np.nan, np.nan], "vol": [1.0, 2.0, 3.0, 4.0],
                        "px": 20.0, "dv20": 1e7, "pre5": 10.0})
    assert ss.arm_rows(res, "vol")["ticker"].tolist() == ["AAA", "UVIX", "NEWCO"]
    assert ss.arm_rows(res, "model")["ticker"].tolist() == ["AAA", "UVIX"]
    assert ss.arm_rows(res, "etf")["ticker"].tolist() == ["UVIX", "NEWETF"]
    # the most volatile name of the vol arm's rows is the added stock; it is never the model's pick
    monkeypatch.setattr(settings, "sel_short_min_run_rows", 1)
    sub = ss.arm_rows(res, "vol")
    assert ss.select(sub, date(2026, 10, 5), 2, _stand(sub), [], arm="vol")["ticker"] == "NEWCO"
    # a bar WITHOUT the session snapshot leaves every row at NO_SNAPSHOT: the vol arm still never
    # ranks an added product (it once did — membership was read off the VOL_ONLY status)
    res2 = res.assign(status="NO_SNAPSHOT")
    assert ss.arm_rows(res2, "vol")["ticker"].tolist() == ["AAA", "UVIX", "NEWCO"]
    from src.data import deep as _deep
    monkeypatch.setattr(_deep, "deep_universe", lambda refresh=False: ["AAA", "UVIX", "NEWETF", "NEWCO"])
    assert ss.vol_extra_names(["AAA", "UVIX"]) == ["NEWETF"]
    assert ss.etf_names(["AAA", "UVIX", "NEWETF", "NEWCO"], ["AAA", "UVIX"]) == ["NEWETF", "UVIX"]


def test_a_stock_in_the_store_outside_the_listings_is_never_an_etf_pick(monkeypatch, on):
    """The deep store's extras are the ETF arm's products; a COMMON STOCK that reached the store
    outside `add_listings` (a research backfill) must not become an ETF-arm name — an extra counts
    only when Polygon types it as a product or not at all."""
    monkeypatch.setitem(ss._TYPES, "t", {"AAA": "CS", "NEWETF": "ETF", "STRAY": "CS"})
    got = ss.etf_names(["AAA", "NEWETF", "STRAY", "UNTYPED"], ["AAA"])
    assert got == ["NEWETF", "UNTYPED"]


def test_no_model_name_list_means_every_name_is_the_models(monkeypatch, on):
    monkeypatch.setattr(ss, "load_model", lambda: (None, {}))
    res = pd.DataFrame({"ticker": ["A", "B"], "status": ["OK", "OK"]})
    assert ss.arm_rows(res, "vol")["ticker"].tolist() == ["A", "B"]
    assert ss.arm_rows(res, "model")["ticker"].tolist() == ["A", "B"]


# ── the screen ───────────────────────────────────────────────────────────────

def _screen_env(monkeypatch, d, details, rows_fn):
    asked = []

    def grouped(s):
        asked.append(s)
        return pd.DataFrame(rows_fn(s), columns=["ticker", "close", "volume"])
    monkeypatch.setattr(ss, "grouped_day", grouped)
    monkeypatch.setattr(ss, "ticker_details", lambda tks: {t: dict(details.get(t, {})) for t in tks})
    from src.data import deep as _deep
    monkeypatch.setattr(_deep, "deep_universe", lambda refresh=False: ["AAA", "BRK-B"])
    monkeypatch.setattr(ss, "load_model", lambda: (None, {"tickers": ["AAA"]}))
    return asked


def test_the_screen_takes_common_stocks_that_clear_the_floor_whatever_their_age(monkeypatch, on):
    d = date(2026, 10, 5)
    sess = ss.sessions_before(d, 20)
    details = {"NEWCO": {"type": "CS", "list_date": "2026-06-01", "active": True},
               "NEWADR": {"type": "ADRC", "list_date": "2025-12-01", "active": True},
               "NEWB-A": {"type": "CS", "list_date": "2026-08-01", "active": True},
               "OLDCO": {"type": "CS", "list_date": "2010-01-04", "active": True},     # established: in too
               "NODATE": {"type": "CS", "active": True},                              # no listing date on record
               "NEWETF": {"type": "ETF", "list_date": "2026-06-01", "active": True},  # a product
               "GONE": {"type": "CS", "list_date": "2026-06-01", "active": False},    # delisted already
               "NINE": {"type": "CS", "list_date": "2026-09-01", "active": True},     # 9 sessions only
               "THIN": {"type": "CS", "list_date": "2026-06-01", "active": True}}     # $2M a day

    def rows(s):
        r = [("NEWCO", 20.0, 1e6), ("NEWADR", 30.0, 1e6), ("NEWB.A", 20.0, 1e6), ("OLDCO", 20.0, 1e6),
             ("NODATE", 20.0, 1e6), ("NEWETF", 20.0, 1e6), ("GONE", 20.0, 1e6), ("THIN", 2.0, 1e6),
             ("BRK.B", 400.0, 1e6), ("AAA", 20.0, 1e6), ("ZVZZT", 20.0, 1e6)]
        if s >= sess[-9]:
            r.append(("NINE", 20.0, 1e6))
        return r
    asked = _screen_env(monkeypatch, d, details, rows)
    got = ss.screen_listings(d)
    # BRK.B -> BRK-B: already stored; AAA: the model's; ZVZZT: no record; NEWETF / GONE / NINE / THIN: out
    assert [r["ticker"] for r in got] == ["NEWADR", "NEWB-A", "NEWCO", "NODATE", "OLDCO"]
    assert got[2] == {"ticker": "NEWCO", "type": "CS", "list_date": "2026-06-01", "name": None,
                      "dollar_volume": 2e7}
    assert got[3]["list_date"] is None and got[4]["list_date"] == "2010-01-04"
    assert sorted(asked) == sess                                             # the 20 sessions before the day
    # a recorded stock is not screened again
    _write_listings(["NEWCO", "OLDCO"])
    assert [r["ticker"] for r in ss.screen_listings(d)] == ["NEWADR", "NEWB-A", "NODATE"]


def test_the_screen_needs_ten_sessions_of_whole_market_bars(monkeypatch, on):
    d = date(2026, 10, 5)
    monkeypatch.setattr(ss, "grouped_day", lambda s: None)
    assert ss.screen_listings(d) == []


def test_ticker_details_are_cached_and_unknown_symbols_asked_again_next_day(monkeypatch, on):
    from src.data import polygon_client as pc
    calls = []

    def details(t):
        calls.append(t)
        return {"type": "CS", "list_date": "2026-06-01", "active": True, "name": "New Co"} if t == "NEWCO" else None
    monkeypatch.setattr(pc, "get_ticker_details", details)
    got = ss.ticker_details(["NEWCO", "ZVZZT"])
    assert got["NEWCO"]["type"] == "CS" and got["ZVZZT"].get("type") is None
    assert ss.ticker_details(["NEWCO", "ZVZZT"])["NEWCO"]["list_date"] == "2026-06-01"
    assert sorted(calls) == ["NEWCO", "ZVZZT"]                               # the second call read the cache
    p = ss.root() / "ticker_details.json"
    cache = json.loads(p.read_text(encoding="utf-8"))
    cache["ZVZZT"]["at"] = (datetime.now(ET) - timedelta(hours=30)).isoformat()
    p.write_text(json.dumps(cache), encoding="utf-8")
    ss.ticker_details(["NEWCO", "ZVZZT"])
    assert sorted(calls) == ["NEWCO", "ZVZZT", "ZVZZT"]


# ── adding the stocks ────────────────────────────────────────────────────────

def test_add_listings_ingests_records_and_seeds_the_vol_history(monkeypatch, on):
    d = date(2026, 10, 5)
    from src.data import deep as _deep
    from src.data import intraday_store as ist
    calls = {}

    def extend(names, **k):
        calls["extend"] = (list(names), k)
        return {"new": len(names)}

    def universe(refresh=False):
        calls["refresh"] = refresh
        return []

    def backfill(days, **k):
        calls["backfill"] = (list(days), k)
        return {"vol:x": 1}
    monkeypatch.setattr(ist, "extend_deep_30m", extend)
    monkeypatch.setattr(ist, "load_deep_30m", lambda t: None if t == "NOBARS" else pd.DataFrame({"Close": [1.0]}))
    monkeypatch.setattr(_deep, "deep_universe", universe)
    monkeypatch.setattr(ss, "backfill_days", backfill)
    out = ss.add_listings(d, [{"ticker": "NEWCO", "type": "CS", "list_date": "2026-06-01", "name": "New Co"},
                              {"ticker": "NOBARS", "type": "CS", "list_date": "2026-06-01"}])
    assert out["added"] == ["NEWCO"] and ss.vol_listing_names() == ["NEWCO"]
    assert ss.read_listings()["NEWCO"] == {"added": "2026-10-05", "type": "CS", "list_date": "2026-06-01",
                                           "name": "New Co"}
    names, k = calls["extend"]
    assert names == ["NEWCO", "NOBARS"] and k["today"] == date(2026, 10, 2) and k["min_age_days"] == 0
    assert calls["refresh"] is True
    days, k = calls["backfill"]
    assert days == ss.sessions_before(d, ss.own_window("vol"))
    assert k == {"tickers": ["NEWCO"], "arms": ("vol",), "merge": True, "existing_only": True,
                 "min_visible_bars": 400}
    assert ss.add_listings(d, []) == {"added": []}


def test_the_backfill_keeps_the_bars_the_live_scorer_would_have_scored(monkeypatch):
    """The live scorer computes no feature before 400 bars are visible (NO_DATA): an added
    stock's seeded history holds only the bars it would have scored."""
    from src.analysis import ml30
    from src.analysis import ml_dataset as md
    dn = np.repeat([20720, 20721, 20722], 10)
    e = {"dn": dn, "bar": np.tile(np.arange(10), 3), "px": np.full(30, 20.0), "dv20": np.full(30, 1e7),
         "X": np.ones((30, 3), np.float32)}
    monkeypatch.setattr(ml30, "ticker_rows", lambda *a, **k: {"eval": e})
    monkeypatch.setattr(md, "hlc_30m", lambda tk: (pd.DatetimeIndex(np.arange(410)), None, None, None, None))
    days = [20720, 20721, 20722]
    r = ss._backfill_one(("NEWCO", 20720, days, 1.0, {}, False, False, 400))
    # the 30 eval rows are bars 381..410 of the series: bar >= 400 keeps the last 11
    assert len(r["dn"]) == 11 and r["D"].shape == (11, 0)
    assert len(ss._backfill_one(("NEWCO", 20720, days, 1.0, {}, False, False, 0))["dn"]) == 30


def test_a_listing_backfill_merges_only_into_day_files_that_exist(monkeypatch, on):
    d1, d2 = date(2026, 9, 28), date(2026, 9, 29)
    ss._write_pickle(pd.DataFrame({"bar": [0], "ticker": ["AAA"], "score": [3.0]}), ss.scores_path(d1, "vol"))
    _write_listings(["NEWCO"])
    from src.analysis import ml30
    nb = len(ml30.base_features())
    jv = ml30.base_features().index(str(settings.sel_short_vol_feature))

    def fake_one(job):
        X = np.zeros((2, nb), np.float32)
        X[:, jv] = 5.0
        return {"tk": job[0], "dn": np.array([ss.dnum(d1), ss.dnum(d2)]), "bar": np.array([0, 0]),
                "px": np.array([20.0, 20.0]), "dv20": np.array([1e7, 1e7]), "X": X, "D": np.zeros((2, 0))}
    monkeypatch.setattr(ss, "_backfill_one", fake_one)
    # the installed model is V2: its extra inputs (sx_*) are neither base nor deep features — a vol-only
    # backfill must never build the model's columns (2026-10-05 10:35: "'sx_runup5' is not in list")
    monkeypatch.setattr(ss, "load_model", lambda: (None, {"tickers": ["AAA"], "features": ["atr_pct_14", "sx_runup5"],
                                                          "extra_features": ["sx_runup5"]}))

    class _Pool:
        def __init__(self, *a, **k):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def map(self, fn, jobs, chunksize=1):
            return map(fn, jobs)
    monkeypatch.setattr(ss, "ProcessPoolExecutor", _Pool)
    ss.backfill_days([d1, d2], tickers=["NEWCO"], arms=("vol",), merge=True, existing_only=True,
                     min_visible_bars=400)
    got = pd.read_pickle(ss.scores_path(d1, "vol")).sort_values("ticker")
    assert got["ticker"].tolist() == ["AAA", "NEWCO"] and got["score"].tolist() == [3.0, 5.0]
    assert not ss.scores_path(d2, "vol").exists()                           # a missing day stays missing


def test_prepare_screens_first_and_ranks_the_listings(monkeypatch, on):
    d = date(2026, 10, 5)
    monkeypatch.setattr(settings, "enable_sel_short_etf", False)
    monkeypatch.setattr(ss, "load_model", lambda: (None, {"tickers": ["AAA"]}))
    seen = {}

    def screen(day):
        seen["screened"] = day
        return [{"ticker": "NEWCO"}]

    def add(day, found):
        _write_listings([r["ticker"] for r in found])
        return {"added": [r["ticker"] for r in found]}
    monkeypatch.setattr(ss, "screen_listings", screen)
    monkeypatch.setattr(ss, "add_listings", add)
    monkeypatch.setattr(ss, "_dv20", lambda t, day: 1e7)
    from src.data import intraday_store as ist
    monkeypatch.setattr(ist, "reset_split_tickers", lambda names, day: {})
    monkeypatch.setattr(ss, "write_trade_logs", lambda days: {})
    out = ss.prepare(d, extend=False)
    assert seen["screened"] == d and out["listings"]["added"] == ["NEWCO"] and out["vol_listings"] == 1
    assert json.loads(ss.universe_path(d).read_text(encoding="utf-8")) == {"AAA": 1e7, "NEWCO": 1e7}
    # the vol arm off: nothing screened, the listings stay out of the day's universe
    monkeypatch.setattr(settings, "enable_sel_short_vol", False)
    seen.clear()
    out = ss.prepare(d, extend=False)
    assert "screened" not in seen and out["vol_listings"] == 0
    assert json.loads(ss.universe_path(d).read_text(encoding="utf-8")) == {"AAA": 1e7}


# ── the short-sale restriction (Rule 201) at the pick ────────────────────────

def _bars(days_lows_closes):
    """A regular-hours 30-minute series: per session, 13 bars with the given lows/closes."""
    idx, low, close = [], [], []
    for day, lows, closes in days_lows_closes:
        for b in range(13):
            start = ss.bar_end_et(day, b) - timedelta(minutes=30)
            idx.append(pd.Timestamp(start).tz_convert("UTC").tz_localize(None))
            low.append(lows[b])
            close.append(closes[b])
    return pd.DatetimeIndex(idx), pd.Series(low), pd.Series(close)


def _bar_start(day, b):
    return pd.Timestamp(ss.bar_end_et(day, b) - timedelta(minutes=30)).tz_convert("UTC").tz_localize(None)


def test_ssr_state_reads_rule_201_off_the_regular_hours_bars():
    d2, d1, d0 = date(2026, 9, 24), date(2026, 9, 25), date(2026, 9, 28)
    flat = [10.0] * 13
    today_lows = [9.5, 9.4, 9.3, 8.9] + [9.6] * 9                         # the 10% line (9.0) breaks at bar 3
    idx, low, close = _bars([(d2, flat, flat), (d1, [9.8] * 13, flat), (d0, today_lows, [9.6] * 13)])
    before = ss.ssr_state(idx, low, close, d0, _bar_start(d0, 2))
    assert before["ssr"] is False and before["ssr_prev_close"] == 10.0 and before["ssr_day_low"] == 9.3
    after = ss.ssr_state(idx, low, close, d0, _bar_start(d0, 3))
    assert after["ssr"] is True and after["ssr_day_low"] == 8.9
    # triggered YESTERDAY (low 8.9 under 90% of the close before): in force all of today
    idx, low, close = _bars([(d2, flat, flat), (d1, [8.9] + [9.8] * 12, flat), (d0, [9.9] * 13, [9.9] * 13)])
    st = ss.ssr_state(idx, low, close, d0, _bar_start(d0, 0))
    assert st["ssr"] is True and st["ssr_prev_low"] == 8.9 and st["ssr_prev2_close"] == 10.0
    # no bar at the pick, or fewer than two earlier sessions: unknown
    assert ss.ssr_state(idx, low, close, d0, _bar_start(d0, 0) + timedelta(minutes=5))["ssr"] is None
    idx1, low1, close1 = _bars([(d0, [9.9] * 13, [9.9] * 13)])
    assert ss.ssr_state(idx1, low1, close1, d0, _bar_start(d0, 2))["ssr"] is None


def test_the_pick_journal_carries_the_restriction_state(on):
    d = date(2026, 9, 28)
    res = _res().assign(ssr=True, ssr_prev_close=11.0, ssr_day_low=9.5, ssr_prev_low=10.8, ssr_prev2_close=10.9)
    for arm in ("vol", "model"):
        rec = ss.select(res, d, 2, _stand(res), [], arm=arm)
        assert rec["ssr"] is True and rec["ssr_day_low"] == 9.5 and rec["ssr_prev2_close"] == 10.9
        json.dumps(rec)
    rec = ss.select(_res(), d, 2, _stand(_res()), [], arm="vol")              # no inputs on the rows: unknown
    assert rec["ssr"] is None and rec["ssr_prev_close"] is None


# ── the entry step's journal and the trade log ───────────────────────────────

def _pick(tk="ABC", day=date(2026, 9, 28), bar=1, **kw):
    rec = {"day": day.isoformat(), "bar_of_day": bar, "bar_end": ss.bar_end_et(day, bar).isoformat(),
           "ticker": tk, "decision": "short", "px": 10.0, "pre5": 8.0, "target": 9.0, "arm": "vol",
           "deadline": ss.bar_end_et(ss.session_after(day, 15), bar).isoformat(), "score": 4.2,
           "runup_pct": 25.0, "days_to_cover": 1.0, "rvol": 2.0, "ssr": True, "ssr_prev_close": 11.0,
           "ssr_day_low": 9.5}
    rec.update(kw)
    return rec


def _tracker_env(monkeypatch, picks):
    from src.data import ibkr_borrow
    from src.performance import tracker
    monkeypatch.setattr(ss, "pending_entries", lambda now=None: list(picks))
    monkeypatch.setattr(tracker, "_fetch_price", lambda t: 10.2)
    monkeypatch.setattr(tracker, "_reference_close", lambda t: None)
    monkeypatch.setattr(tracker, "_execution_iso", lambda: "2026-09-28T14:40:00+00:00")
    stamp = datetime(2026, 9, 28, 9, 0, tzinfo=ET)

    def block(ticker, price, base=None, now=None, max_fee_pct="default"):
        if ticker == "NOB":
            return "no_borrow", None                                         # not in IBKR's file at all
        if ticker == "LOW":
            return "no_borrow", ibkr_borrow.Borrow(ticker, 943.0, -900.0, 100, stamp)
        return None, ibkr_borrow.Borrow(ticker, 496.0, -80.0, 900_000, stamp)
    monkeypatch.setattr(ibkr_borrow, "short_block", block)
    return tracker


def test_the_entry_step_journals_every_pick_it_settles_with_the_borrow(monkeypatch, on):
    picks = [_pick("ABC"), _pick("NOB", ssr=False), _pick("LOW"), _pick("TGT")]
    tracker = _tracker_env(monkeypatch, picks)
    monkeypatch.setattr(tracker, "_fetch_price", lambda t: 8.9 if t == "TGT" else 10.2)
    assert tracker.record_sel_short_trades(run_id="r1") == 1
    lines = ss._read_jsonl(ss.entries_path(date(2026, 9, 28)))
    by = {x["ticker"]: x for x in lines}
    assert set(by) == {"ABC", "NOB", "LOW", "TGT"}
    a = by["ABC"]
    assert a["outcome"] == "opened" and a["price"] == 10.2 and a["borrow_available"] == 900_000
    assert a["borrow_fee_pct"] == 496.0 and a["borrow_checked"] is True and a["borrow_listed"] is True
    assert a["recommendation_id"] == ss.trade_id(ss.pick_key(picks[0])) and a["ssr"] is True
    assert by["NOB"]["outcome"] == "no_borrow" and by["NOB"]["borrow_listed"] is False and by["NOB"]["ssr"] is False
    assert by["LOW"]["outcome"] == "no_borrow" and by["LOW"]["borrow_available"] == 100
    assert by["TGT"]["outcome"] == "target_reached" and by["TGT"]["borrow_checked"] is None
    t = [x for x in tracker._load_trades() if x.get("entry_mechanism") == "sel_short"]
    assert len(t) == 1 and t[0]["sel_ssr"] is True and t[0]["sel_ssr_day_low"] == 9.5
    assert t[0]["recommendation_id"] == ss.trade_id(ss.pick_key(picks[0]))


def test_an_expired_pick_is_journaled(monkeypatch, on):
    d = date(2026, 9, 28)
    ss._journal(_pick("OLD", day=d, bar=0))
    now = ss.bar_end_et(d, 0) + timedelta(minutes=float(settings.sel_short_entry_max_age_minutes) + 5)
    assert ss.pending_entries(now) == []
    assert ss._read_jsonl(ss.entries_path(d))[0]["outcome"] == "expired"


def test_the_trade_log_joins_pick_entry_ledger_and_broker_fills(monkeypatch, on):
    d = date(2026, 9, 28)
    pick = _pick("ABC")
    ss._journal(pick)
    ss._journal(dict(_pick("NOB"), ssr=False))
    tracker = _tracker_env(monkeypatch, [pick, dict(_pick("NOB"), ssr=False)])
    assert tracker.record_sel_short_trades(run_id="r1") == 1
    rid = ss.trade_id(ss.pick_key(pick))
    trades = tracker._load_trades()
    t = next(x for x in trades if x.get("recommendation_id") == rid)
    t.update(broker_status="Filled", broker_requested_qty=137, broker_fill_qty=137, broker_fill_price=10.15,
             broker_commission=1.0, broker_order_id="9", broker_client_ref=rid + "-r1")
    tracker._save_trades(trades)
    from src.db import repo
    base = {"intent": "ENTRY", "ticker": "ABC", "side": "SELL", "order_type": "LMT", "requested_qty": 137,
            "model_price": 10.2}
    repo.insert_broker_report("r1", {"orders": [
        dict(base, event="SUBMIT_REFUSED", client_ref=rid, submitted_at="2026-09-28T14:40:05+00:00", filled_qty=0,
             ok=False, error="short sale price test"),
        dict(base, event="SUBMIT", client_ref=rid + "-r1", submitted_at="2026-09-28T14:40:10+00:00", filled_qty=0,
             limit_price=10.15, bid_at_submit=10.1, ask_at_submit=10.3, ok=True),
        dict(base, event="SETTLE_FILL", client_ref=rid + "-r1", submitted_at="2026-09-28T14:40:25+00:00",
             filled_qty=137, fill_price=10.15, ok=True, status="Filled")]})
    rows = ss.trade_log(d)
    assert [r["ticker"] for r in rows] == ["ABC", "NOB"]
    r = rows[0]
    assert r["arm"] == "vol" and r["ssr"] is True and r["entry_outcome"] == "opened"
    assert r["borrow_available"] == 900_000 and r["recommendation_id"] == rid and r["status"] == "OPEN"
    be = r["broker_entry"]
    assert be["status"] == "Filled" and be["filled_qty"] == 137 and be["fill_price"] == 10.15
    assert be["attempts"] == 2 and be["refused"] == 1 and be["errors"] == ["short sale price test"]
    assert be["slippage_bps"] == pytest.approx((10.2 - 10.15) / 10.2 * 1e4, abs=0.1)
    assert be["first_submit_at"] == "2026-09-28T14:40:05+00:00" and be["seconds_to_fill"] == 20.0
    assert r["broker_exit"]["attempts"] == 0 and r["broker_return_pct"] is None
    nob = rows[1]
    assert nob["entry_outcome"] == "no_borrow" and nob["borrow_listed"] is False and "broker_entry" not in nob
    assert ss.write_trade_logs([d]) == {"2026-09-28": 2}
    assert len(ss._read_jsonl(ss.tradelog_path(d))) == 2
