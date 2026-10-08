"""The vol arm's THIN STOCKS (user directive 2026-10-07: "Add the thin stocks to the live vol arm"; PREREG32).

What must hold: the "thin" arm runs only with the vol arm and its own switch; it ranks ONLY the rows the day's thin
universe names (the run flags them by name) and no other arm ranks or records one of them; its band is the thin
floor up to the vol floor; its target gives back the vol arm's share and it journals the same-bar backups; its
history is its own (`scores_thin`); the prepare writes the thin universe (common stocks between the floors — never a
product, never a liquid name) after screening the thin band outside the store, from the sessions BEFORE the day;
the scorer never builds a model vector for a row under the vol floor (it stays out of the V2 cross-section); the
account funds a thin short only while the reserve of live slices stays free; the entry step settles every other
arm's picks first.
"""
from __future__ import annotations

import json
from datetime import date, datetime, timezone
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
    monkeypatch.setattr(settings, "enable_sel_short_thin", True)
    monkeypatch.setattr(settings, "enable_sel_short_etf", False)
    monkeypatch.setattr(settings, "enable_sel_short_vol_added_stocks", True)
    return settings


def _write_listings(names):
    p = ss.listings_path()
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps({n: {"added": "2026-10-07", "type": "CS"} for n in names}), encoding="utf-8")


def _res():
    """Three liquid names and two thin ones, the thin ones the most volatile."""
    return pd.DataFrame({"ticker": ["AAA", "BBB", "CCC", "TH1", "TH2"],
                         "status": ["OK", "OK", "OK", "VOL_ONLY", "VOL_ONLY"],
                         "score": [0.5, 0.4, 0.3, np.nan, np.nan],
                         "vol": [3.0, 2.0, 1.0, 9.0, 8.0],
                         "px": [20.0, 20.0, 20.0, 12.0, 15.0],
                         "dv20": [1e7, 1e7, 1e7, 2e6, 3e6],
                         "pre5": [10.0, 10.0, 10.0, 8.0, 10.0],
                         "thin": [False, False, False, True, True]})


def _stand(res):
    return pd.DataFrame({"n_prior": 0.0, "prior_max": np.nan, "prior_min": np.nan},
                        index=pd.Index(res.ticker, name="ticker"))


def test_the_thin_arm_runs_with_the_vol_arm_and_its_own_switch(monkeypatch, on):
    assert ss.live_arms() == ["model", "vol", "thin"]
    monkeypatch.setattr(settings, "enable_sel_short_thin", False)
    assert ss.live_arms() == ["model", "vol"]
    monkeypatch.setattr(settings, "enable_sel_short_thin", True)
    monkeypatch.setattr(settings, "enable_sel_short_vol", False)
    assert ss.live_arms() == ["model"]
    assert "thin" in ss.ARMS and ss.scores_path(date(2026, 10, 8), "thin").parent.name == "scores_thin"


def test_the_thin_rows_belong_to_the_thin_arm_alone(monkeypatch, on):
    monkeypatch.setattr(ss, "load_model", lambda: (None, {"tickers": ["AAA", "BBB", "CCC", "TH1"]}))
    _write_listings(["TH2"])
    res = _res()
    assert ss.arm_rows(res, "thin")["ticker"].tolist() == ["TH1", "TH2"]
    assert ss.arm_rows(res, "vol")["ticker"].tolist() == ["AAA", "BBB", "CCC"]
    assert ss.arm_rows(res, "model")["ticker"].tolist() == ["AAA", "BBB", "CCC"]
    # a run without the flag (no thin universe): no row is the thin arm's
    assert ss.arm_rows(res.drop(columns="thin"), "thin").empty


def test_the_thin_arm_ranks_its_band_and_gives_back_the_vol_share(monkeypatch, on):
    monkeypatch.setattr(settings, "sel_short_min_run_rows", 1)
    monkeypatch.setattr(settings, "enable_sel_short_vol_fallback", True)
    d = date(2026, 10, 8)
    res = _res()
    thin = res[res.thin]
    rec = ss.select(thin, d, 2, _stand(thin), [], arm="thin")
    assert (rec["arm"], rec["ticker"], rec["decision"]) == ("thin", "TH1", "short")
    assert ss.give_back("thin") == ss.give_back("vol") == float(settings.sel_short_vol_give_back)
    assert rec["target"] == pytest.approx(12.0 - ss.give_back("vol") * (12.0 - 8.0))
    # the same-bar backups, as the vol arm's (the entry step's fallback when IBKR cannot lend)
    assert rec["fresh_pool"] == ["TH1", "TH2"] and [b["ticker"] for b in rec["backups"]] == ["TH2"]
    # the band: at the vol floor or under the thin floor, a name is not a thin stock
    out = thin.assign(dv20=[float(settings.sel_short_min_dollar_volume), 9e5])
    assert ss.select(out, d, 2, _stand(out), [], arm="thin")["decision"] == "thin_run"
    # the vol arm never ranks a thin name, even the most volatile
    vol = res[~res.thin]
    assert ss.select(vol, d, 2, _stand(vol), [], arm="vol")["ticker"] == "AAA"


def test_decide_keeps_the_thin_history_apart(monkeypatch, on):
    monkeypatch.setattr(ss, "load_model", lambda: (None, {"tickers": ["AAA", "BBB", "CCC", "TH1"]}))
    _write_listings(["TH2"])
    monkeypatch.setattr(settings, "sel_short_min_run_rows", 1)
    d = date(2026, 10, 8)
    run = ss.decide(_res(), d, 2)
    assert run["ticker"] == "AAA" and run["arms"]["vol"]["ticker"] == "AAA"
    assert run["arms"]["thin"]["ticker"] == "TH1" and run["arms"]["thin"]["arm"] == "thin"
    thin_hist = pd.read_pickle(ss.scores_path(d, "thin"))
    vol_hist = pd.read_pickle(ss.scores_path(d, "vol"))
    assert sorted(thin_hist["ticker"]) == ["TH1", "TH2"]
    assert not ({"TH1", "TH2"} & set(vol_hist["ticker"]))
    assert {r["arm"] for r in ss.read_picks(d)} == {"model", "vol", "thin"}


def test_the_run_scores_the_thin_universe_and_flags_its_rows_by_name(monkeypatch, on):
    d = date(2026, 10, 8)
    monkeypatch.setattr(ss, "load_model", lambda: (None, {"tickers": ["AAA", "BBB", "CCC", "TH1"]}))
    _write_listings(["TH2"])
    monkeypatch.setattr(ss, "load_universe", lambda day: {"AAA": 1e7, "BBB": 1e7, "CCC": 1e7})
    monkeypatch.setattr(ss, "load_thin_universe", lambda day: {"TH1": 2e6, "TH2": 3e6})
    monkeypatch.setattr(ss, "ensure_snapshot", lambda day, names: 1.0)
    from src.data import intraday_store as ist
    monkeypatch.setattr(ist, "reset_split_tickers", lambda names, day: {})
    seen = {}

    def score_bar(day, bar, uni, fetch=True):
        seen["uni"] = dict(uni)
        return _res().drop(columns="thin")                 # the scorer knows nothing of the band
    monkeypatch.setattr(ss, "score_bar", score_bar)
    monkeypatch.setattr(settings, "sel_short_min_run_rows", 1)
    run = ss.run(d, 2)
    assert seen["uni"] == {"AAA": 1e7, "BBB": 1e7, "CCC": 1e7, "TH1": 2e6, "TH2": 3e6}
    assert run["arms"]["thin"]["ticker"] == "TH1" and run["arms"]["vol"]["ticker"] == "AAA"
    # the thin stocks off: the run scores the universe alone
    monkeypatch.setattr(settings, "enable_sel_short_thin", False)
    ss.run(d, 3)
    assert seen["uni"] == {"AAA": 1e7, "BBB": 1e7, "CCC": 1e7}


def test_the_scorer_never_builds_a_model_vector_under_the_vol_floor(on, monkeypatch):
    from loguru import logger as _lg
    from src.analysis import deep_features as dfe
    from src.signals import ml_model
    monkeypatch.setattr(_lg, "remove", lambda *a, **k: None)
    d = date(2026, 10, 8)
    target_start = pd.Timestamp(ss.bar_end_et(d, 1)).tz_convert("UTC").tz_localize(None) - ss.BAR

    class Booster:
        def predict(self, M, num_iteration=None):
            return np.full(len(M), 0.7)

    monkeypatch.setattr(ss, "load_model", lambda: (Booster(), {"features": ["f1"], "num_iteration": 1,
                                                               "tickers": ["AAA", "TH1"]}))
    monkeypatch.setattr(ss, "_deep_and_recent", lambda t, day, fetch=True: (None, pd.DataFrame({"x": [1]})))
    monkeypatch.setattr(ss, "series", lambda tk, today, now, deep=None: (None,) * 5)
    monkeypatch.setattr(ss, "_close_at", lambda idx, c, day, bar: 8.0)
    frame = {"bar_ts": target_start, "close": 10.0, "sday": ss.dnum(d), "bar_idx": 1,
             "features": {"f1": 0.5, settings.sel_short_vol_feature: 3.0}}
    monkeypatch.setattr(ml_model, "features_30m_from_hlc", lambda tk, hlc, now: (frame, "OK"))
    monkeypatch.setattr(dfe, "load_session_snapshot", lambda day: {"AAA": {}, "TH1": {}})
    monkeypatch.setattr(dfe, "serving_vector", lambda tk, sday, close, bar, snap: {ss.DTC_FEATURE: 0.5})
    rows = {r["ticker"]: r for r in ss._score_chunk((["AAA", "TH1"], d.isoformat(), 1,
                                                     {"AAA": 1e7, "TH1": 2e6}))}
    assert rows["AAA"]["status"] == "OK" and rows["AAA"]["score"] == pytest.approx(0.7)
    # a model name under the vol floor (a thin stock): ATR% and days to cover, never a model score
    assert rows["TH1"]["status"] == "VOL_ONLY" and np.isnan(rows["TH1"]["score"])
    assert rows["TH1"]["vol"] == 3.0 and rows["TH1"]["dtc"] == 0.5


def _screen_env(monkeypatch, d, details, rows_fn):
    asked = []

    def grouped(s):
        asked.append(s)
        return pd.DataFrame(rows_fn(s), columns=["ticker", "close", "volume"])
    monkeypatch.setattr(ss, "grouped_day", grouped)
    monkeypatch.setattr(ss, "ticker_details", lambda tks: {t: dict(details.get(t, {})) for t in tks})
    from src.data import deep as _deep
    monkeypatch.setattr(_deep, "deep_universe", lambda refresh=False: ["AAA"])
    monkeypatch.setattr(ss, "load_model", lambda: (None, {"tickers": ["AAA"]}))
    return asked


def test_the_thin_screen_takes_the_band_from_the_sessions_before_the_day(monkeypatch, on):
    d = date(2026, 10, 8)
    details = {t: {"type": "CS", "active": True} for t in ("THINCO", "BIGCO", "TINYCO", "THINETF")}
    details["THINETF"]["type"] = "ETF"

    def rows(s):                    # mean daily $: THINCO 2M, BIGCO 20M, TINYCO 0.5M, THINETF 2M
        return [("THINCO", 20.0, 1e5), ("BIGCO", 20.0, 1e6), ("TINYCO", 5.0, 1e5), ("THINETF", 20.0, 1e5),
                ("AAA", 20.0, 1e5)]
    asked = _screen_env(monkeypatch, d, details, rows)
    assert [r["ticker"] for r in ss.screen_listings(d, band="thin")] == ["THINCO"]
    assert sorted(set(asked)) == ss.sessions_before(d, 20)                   # never a bar of the day or later
    assert [r["ticker"] for r in ss.screen_listings(d)] == ["BIGCO"]         # the vol band, unchanged


def test_prepare_writes_the_thin_universe(monkeypatch, on):
    d = date(2026, 10, 8)
    monkeypatch.setattr(ss, "load_model", lambda: (None, {"tickers": ["AAA", "TH1", "UVIX", "TINY"]}))
    monkeypatch.setitem(ss._TYPES, "t", {"AAA": "CS", "TH1": "CS", "UVIX": "ETF", "TINY": "CS", "NEWTH": "CS"})
    screened = []

    def screen(day, band="core"):
        screened.append(band)
        return [{"ticker": "NEWTH"}] if band == "thin" else []

    def add(day, found, backfill=True, arm="vol"):
        _write_listings([r["ticker"] for r in found])
        return {"added": [r["ticker"] for r in found], "arm": arm}
    monkeypatch.setattr(ss, "screen_listings", screen)
    monkeypatch.setattr(ss, "add_listings", add)
    dvs = {"AAA": 1e7, "TH1": 2e6, "UVIX": 3e6, "TINY": 5e5, "NEWTH": 4e6}
    monkeypatch.setattr(ss, "_dv20", lambda t, day: dvs[t])
    from src.data import intraday_store as ist
    monkeypatch.setattr(ist, "reset_split_tickers", lambda names, day: {})
    monkeypatch.setattr(ss, "write_trade_logs", lambda days: {})
    out = ss.prepare(d, extend=False)
    assert screened == ["core", "thin"] and out["thin_listings"]["arm"] == "thin"
    assert json.loads(ss.universe_path(d).read_text(encoding="utf-8")) == {"AAA": 1e7}
    # the thin universe: common stocks between the floors — never the ETF, never under the thin floor
    assert json.loads(ss.thin_universe_path(d).read_text(encoding="utf-8")) == {"TH1": 2e6, "NEWTH": 4e6}
    assert out["thin_universe"] == 2
    # the thin stocks off: no thin screen, no thin universe written
    monkeypatch.setattr(settings, "enable_sel_short_thin", False)
    ss.thin_universe_path(d).unlink()
    screened.clear()
    ss.prepare(d, extend=False)
    assert screened == ["core"] and not ss.thin_universe_path(d).exists()


def test_the_account_funds_a_thin_short_only_while_the_reserve_stays_free(monkeypatch):
    from src.performance import sim_account as sa
    monkeypatch.setattr(settings, "enable_sel_short_account_sizing", True)
    monkeypatch.setattr(settings, "sel_short_account_initial", 5000.0)
    now = datetime(2026, 10, 8, 15, 0, tzinfo=timezone.utc)               # a $5,000 account
    assert "thin" in sa.FUNDED_ARMS and sa.arm_funded("thin")
    n0, why0, _ = sa.size([], 20.0, 1e8, now)
    n1, why1, _ = sa.size([], 20.0, 1e8, now, reserve_slices=2.0)
    assert (why0, why1) == ("ok", "ok") and n1 == n0                       # an empty account: room for both
    # one open short of $1,400 holds $4,004 of initial margin (house 2.86): $996 of room left — a live
    # slice still fits (10 shares at $20), a thin one would eat the two live slices' $1,191 reserve
    open_ = [{"entry_mechanism": "sel_short", "sel_account_shares": 70, "entry_price": 20.0,
              "current_price": 20.0, "status": "OPEN", "return_pct": 0.0}]
    n2, why2, _ = sa.size(open_, 20.0, 1e8, now)
    n3, why3, _ = sa.size(open_, 20.0, 1e8, now, reserve_slices=2.0)
    assert (n2, why2) == (10, "ok") and (n3, why3) == (0, "thin_reserve")


def _pick(tk, arm, day=date(2026, 9, 28), bar=1):
    return {"day": day.isoformat(), "bar_of_day": bar, "bar_end": ss.bar_end_et(day, bar).isoformat(),
            "ticker": tk, "decision": "short", "px": 10.0, "pre5": 8.0, "target": 8.0, "arm": arm,
            "deadline": ss.bar_end_et(ss.session_after(day, 15), bar).isoformat(), "score": 4.2,
            "runup_pct": 25.0, "days_to_cover": 1.0, "rvol": 2.0, "atr_pct": 4.2, "dv20": 2e6}


def test_the_entry_step_settles_every_other_arm_before_a_thin_pick(monkeypatch, on):
    from src.data import ibkr_borrow
    from src.performance import tracker
    picks = [_pick("AAA", "thin"), _pick("ZZZ", "vol")]
    monkeypatch.setattr(ss, "pending_entries", lambda now=None: list(picks))
    monkeypatch.setattr(tracker, "_fetch_price", lambda t: 10.2)
    monkeypatch.setattr(tracker, "_reference_close", lambda t: None)
    monkeypatch.setattr(tracker, "_execution_iso", lambda: "2026-09-28T14:40:00+00:00")
    stamp = datetime(2026, 9, 28, 9, 0, tzinfo=ET)
    monkeypatch.setattr(ibkr_borrow, "short_block", lambda ticker, price, base=None, now=None, max_fee_pct="default":
                        (None, ibkr_borrow.Borrow(ticker, 12.0, -5.0, 900_000, stamp)))
    assert tracker.record_sel_short_trades(run_id="r1") == 2
    lines = ss._read_jsonl(ss.entries_path(date(2026, 9, 28)))
    assert [x["ticker"] for x in lines] == ["ZZZ", "AAA"]                   # the vol pick first, by arm
    t = {x["ticker"]: x for x in tracker._load_trades() if x.get("entry_mechanism") == "sel_short"}
    assert t["AAA"]["sel_arm"] == "thin" and t["AAA"]["sel_vol_score"] == 4.2 and t["AAA"]["sel_atr_pct"] == 4.2
    assert "THIN name" in t["AAA"]["rationale"] and t["ZZZ"]["sel_arm"] == "vol"
