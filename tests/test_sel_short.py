"""The selection-short strategy (2026-09-28, `src/signals/sel_short.py`): the
tail-regression LONG model's top-1 pick per regular-hours bar, fresh against its
own 30-session scores, shorted when it rose over the 5 sessions before, covered
at half the run-up given back or after 15 sessions — and the legacy book shadowed
and flattened once.

What must hold: the bar clock (which bar is complete when); the freshness
standing is `eval_metrics.own_history_standing`'s, and the whole selection rule
reproduces `eval_metrics.selection_entries` (the rule the model was evaluated
with); the target and deadline are the evaluated ones; a pick whose short
interest exceeds one day of volume is journaled `crowded`, never traded, and the
scorer carries the snapshot's days to cover to every row; the tracker opens only
journaled shorts IBKR can lend (ANY fee) and never twice; the strategy's shorts
take only their own exits (no legacy monitor, no reversal close); the legacy
book is flattened exactly once, in regular hours, after the instant; the
pipeline no longer opens legacy entries.
"""
from __future__ import annotations

import json
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import pytest

from config.settings import settings
from src.signals import sel_short as ss

ET = ZoneInfo("America/New_York")


def et(y, m, d, hh, mm):
    return datetime(y, m, d, hh, mm, tzinfo=ET)


@pytest.fixture
def on(monkeypatch):
    monkeypatch.setattr(settings, "enable_sel_short", True)
    return settings


# ── the bar clock ────────────────────────────────────────────────────────────

def test_latest_bar_and_bar_end():
    assert ss.latest_bar(et(2026, 9, 28, 9, 45)) is None                 # first bar not over yet
    assert ss.latest_bar(et(2026, 9, 28, 10, 0)) == (date(2026, 9, 28), 0)
    assert ss.latest_bar(et(2026, 9, 28, 10, 29)) == (date(2026, 9, 28), 0)
    assert ss.latest_bar(et(2026, 9, 28, 10, 30)) == (date(2026, 9, 28), 1)
    assert ss.latest_bar(et(2026, 9, 28, 16, 0)) == (date(2026, 9, 28), 12)
    assert ss.latest_bar(et(2026, 9, 28, 16, 45)) == (date(2026, 9, 28), 12)
    assert ss.latest_bar(et(2026, 9, 27, 12, 0)) is None                 # Sunday
    assert ss.bar_end_et(date(2026, 9, 28), 0) == et(2026, 9, 28, 10, 0)
    assert ss.bar_end_et(date(2026, 9, 28), 12) == et(2026, 9, 28, 16, 0)


def test_session_helpers_skip_weekends_and_holidays():
    assert ss.sessions_before(date(2026, 9, 28), 2) == [date(2026, 9, 24), date(2026, 9, 25)]
    assert ss.session_after(date(2026, 9, 25), 1) == date(2026, 9, 28)
    # 2026-11-26 is Thanksgiving: 15 sessions after Nov 20 skip it and the weekends
    got = ss.session_after(date(2026, 11, 20), 15)
    n, p = 0, date(2026, 11, 20)
    while p < got:
        p += timedelta(days=1)
        n += ss.is_session(p)
    assert n == 15 and not ss.is_session(date(2026, 11, 26))


# ── the freshness standing and the selection rule ─────────────────────────────

def _synthetic(days, n_tk=24, bars=3, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for d in days:
        for b in range(bars):
            for i in range(n_tk):
                rows.append({"d": d, "bar": b, "ticker": f"T{i:02d}", "score": float(rng.normal(i * 0.01, 1.0))})
    return pd.DataFrame(rows)


def _sessions(start, n):
    out, p = [], start
    while len(out) < n:
        if ss.is_session(p):
            out.append(p)
        p += timedelta(days=1)
    return out


def test_standing_is_eval_metrics_own_history(on):
    from src.analysis.eval_metrics import own_history_standing
    days = _sessions(date(2026, 6, 1), 40)
    df = _synthetic(days)
    for d, g in df.groupby("d"):
        ss._write_pickle(g[["bar", "ticker", "score"]].reset_index(drop=True), ss.scores_path(d))
    d = days[-1]
    tick = sorted(df.ticker.unique())
    got = ss.standing(d, tick)
    f = df.assign(signal_date=df.d.astype(str))
    ref = own_history_standing(f, "score", date="signal_date", ticker="ticker", window_days=30)
    ref = ref[f.d == d].assign(ticker=f.loc[f.d == d, "ticker"]).drop_duplicates("ticker").set_index("ticker")
    for c in ("n_prior", "prior_max", "prior_min"):
        np.testing.assert_allclose(got[c].to_numpy(), ref.loc[tick, c].to_numpy())


def test_the_rule_reproduces_the_evaluated_selection(on):
    """Run bar by bar through `select` (standing from the score files, earlier
    decisions of the day from the journal) and compare with
    `eval_metrics.selection_entries` on the same scores."""
    from src.analysis.eval_metrics import selection_entries
    days = _sessions(date(2026, 6, 1), 45)
    df = _synthetic(days, n_tk=24, bars=4, seed=7)
    # a steady climber, so the own-history rule has fresh new highs to find
    df.loc[df.ticker == "T05", "score"] += np.linspace(0, 6, int((df.ticker == "T05").sum()))
    frame = df.assign(signal_date=df.d.astype(str), run=[ss.dnum(d) * 100 + b for d, b in zip(df.d, df.bar)])
    ev = selection_entries(frame, "score", "long", run="run", label="none", bars="none")
    want = {(r.signal_date, int(r.bar), r.ticker) for r in ev.itertuples()}
    got = set()
    for d in days:
        g_day = df[df.d == d]
        journal = []
        for b in sorted(g_day.bar.unique()):
            g = g_day[g_day.bar == b]
            res = pd.DataFrame({"ticker": g.ticker, "score": g.score, "px": 10.0, "dv20": 1e7,
                                "pre5": 5.0, "status": "OK"})
            rec = ss.select(res, d, int(b), ss.standing(d, list(g.ticker)), journal)
            journal.append(rec)
            if rec["decision"] in ("short", "not_a_riser"):       # the rule's entry, before the run-up filter
                got.add((str(d), int(b), rec["ticker"]))
        ss._write_pickle(g_day[["bar", "ticker", "score"]].reset_index(drop=True), ss.scores_path(d))
    assert got == want and len(want) > 5


def test_select_decisions_target_and_deadline(on):
    d = date(2026, 9, 28)
    res = pd.DataFrame({"ticker": [f"N{i:02d}" for i in range(25)], "score": np.linspace(0, 1, 25),
                        "px": 20.0, "dv20": 1e7, "pre5": 10.0, "status": "OK"})
    stand = pd.DataFrame({"n_prior": 0.0, "prior_max": np.nan, "prior_min": np.nan},
                         index=pd.Index(res.ticker, name="ticker"))
    rec = ss.select(res, d, 2, stand, [])
    assert rec["decision"] == "short" and rec["ticker"] == "N24"
    assert rec["target"] == pytest.approx(20.0 - ss.give_back("model") * (20.0 - 10.0))
    assert datetime.fromisoformat(rec["deadline"]) == ss.bar_end_et(ss.session_after(d, 15), 2)
    # the top pick below its own 30-session high with enough history: not fresh
    st2 = stand.copy()
    st2.loc["N24", ["n_prior", "prior_max"]] = [50.0, 5.0]
    assert ss.select(res, d, 2, st2, [])["decision"] == "not_fresh"
    # never scored = no standing = fresh (a missing row must not read as history)
    assert ss.select(res, d, 2, stand.drop(index="N24"), [])["decision"] == "short"
    # an earlier fresh top pick of the same name today: not its first
    assert ss.select(res, d, 3, stand, [{"ticker": "N24", "fresh": True}])["decision"] == "not_first_today"
    # fell into the pick: no short
    assert ss.select(res.assign(pre5=25.0), d, 2, stand, [])["decision"] == "not_a_riser"
    # below the price floor the name is not in the cross-section; thin bars are skipped
    r2 = res.copy()
    r2.loc[r2.ticker == "N24", "px"] = 4.0
    assert ss.select(r2, d, 2, stand, [])["ticker"] == "N23"
    assert ss.select(res.head(10), d, 2, stand, [])["decision"] == "thin_run"


# ── the volatility arm (2026-09-26: "Short the most volatile name") ────────────

def _res(n=25):
    return pd.DataFrame({"ticker": [f"N{i:02d}" for i in range(n)],
                         "score": np.linspace(1.0, 0.0, n),            # the model prefers N00
                         "vol": np.linspace(0.0, 5.0, n),               # the vol arm prefers N24
                         "px": 20.0, "dv20": 1e7, "pre5": 10.0, "status": "OK"})


def test_vol_arm_ranks_atr_with_its_own_history_and_first_pick(on):
    d = date(2026, 9, 28)
    res = _res()
    stand = pd.DataFrame({"n_prior": 0.0, "prior_max": np.nan, "prior_min": np.nan},
                         index=pd.Index(res.ticker, name="ticker"))
    m = ss.select(res, d, 2, stand, [])
    v = ss.select(res, d, 2, stand, [], arm="vol")
    assert (m["ticker"], m["arm"]) == ("N00", "model")
    assert (v["ticker"], v["arm"], v["decision"]) == ("N24", "vol", "short")
    assert v["score"] == pytest.approx(5.0) and v["target"] == pytest.approx(20.0 - ss.give_back("vol") * 10.0)
    assert datetime.fromisoformat(v["deadline"]) == ss.bar_end_et(ss.session_after(d, 15), 2)
    # a row without the session snapshot still carries its ATR%: the vol arm ranks it, the model cannot
    r2 = res.assign(status="NO_SNAPSHOT")
    assert ss.select(r2, d, 2, stand, [])["decision"] == "thin_run"
    assert ss.select(r2, d, 2, stand, [], arm="vol")["ticker"] == "N24"
    # freshness against the vol history's own high
    st2 = stand.copy()
    st2.loc["N24", ["n_prior", "prior_max"]] = [50.0, 6.0]
    assert ss.select(res, d, 2, st2, [], arm="vol")["decision"] == "not_fresh"
    # "first pick of the day" counts only the same arm's earlier decisions; a record
    # journaled before the arms existed (no "arm") is the model's
    assert ss.select(res, d, 3, stand, [{"ticker": "N24", "fresh": True}], arm="vol")["decision"] == "short"
    assert ss.select(res, d, 3, stand, [{"ticker": "N24", "fresh": True, "arm": "vol"}],
                     arm="vol")["decision"] == "not_first_today"
    assert ss.select(res, d, 3, stand, [{"ticker": "N00", "fresh": True}])["decision"] == "not_first_today"
    assert ss.select(res.drop(columns="vol"), d, 2, stand, [], arm="vol")["decision"] == "thin_run"


def test_the_vol_rule_reproduces_the_evaluated_selection(on):
    """The vol arm through `select` + its own score files = `selection_entries`
    on the ATR% itself (the rule measured in scratchpad `short_models.py`)."""
    from src.analysis.eval_metrics import selection_entries
    days = _sessions(date(2026, 6, 1), 45)
    df = _synthetic(days, n_tk=24, bars=4, seed=11)
    df.loc[df.ticker == "T07", "score"] += np.linspace(0, 6, int((df.ticker == "T07").sum()))
    frame = df.assign(signal_date=df.d.astype(str), run=[ss.dnum(d) * 100 + b for d, b in zip(df.d, df.bar)])
    ev = selection_entries(frame, "score", "long", run="run", label="none", bars="none",
                           own_window_days=ss.own_window("vol"))
    want = {(r.signal_date, int(r.bar), r.ticker) for r in ev.itertuples()}
    got = set()
    for d in days:
        g_day = df[df.d == d]
        journal = []
        for b in sorted(g_day.bar.unique()):
            g = g_day[g_day.bar == b]
            res = pd.DataFrame({"ticker": g.ticker, "score": np.nan, "vol": g.score, "px": 10.0, "dv20": 1e7,
                                "pre5": 5.0, "status": "NO_SNAPSHOT"})
            rec = ss.select(res, d, int(b), ss.standing(d, list(g.ticker), "vol"), journal, arm="vol")
            journal.append(rec)
            if rec["decision"] in ("short", "not_a_riser"):
                got.add((str(d), int(b), rec["ticker"]))
        ss._write_pickle(g_day[["bar", "ticker", "score"]].reset_index(drop=True), ss.scores_path(d, "vol"))
    assert got == want and len(want) > 5


def test_each_arm_reads_its_own_freshness_window(on):
    """The vol arm judges freshness over 20 sessions (user directive 2026-09-27),
    the model over 30: a high 25 sessions back counts for the model only."""
    assert ss.own_window("model") == 30 and ss.own_window("vol") == 20
    d = date(2026, 9, 28)
    days = ss.sessions_before(d, 30)
    for i, s in enumerate(days):
        hi = 9.0 if i == 5 else 1.0                                  # 25 sessions before d
        for arm in ss.ARMS:
            ss._write_pickle(pd.DataFrame({"bar": [0], "ticker": ["AAA"], "score": [hi]}), ss.scores_path(s, arm))
    m, v = ss.standing(d, ["AAA"], "model").loc["AAA"], ss.standing(d, ["AAA"], "vol").loc["AAA"]
    assert (m["n_prior"], m["prior_max"]) == (30.0, 9.0)
    assert (v["n_prior"], v["prior_max"]) == (20.0, 1.0)


def test_decide_journals_both_arms_with_their_own_histories(on, monkeypatch):
    monkeypatch.setattr(settings, "enable_sel_short_etf", False)     # the ETF arm has its own test
    d = date(2026, 9, 28)
    res = _res()
    run = ss.decide(res, d, 2)
    picks = ss.read_picks(d)
    assert [(p["arm"], p["ticker"], p["decision"]) for p in picks] == [("model", "N00", "short"),
                                                                        ("vol", "N24", "short")]
    assert run["ticker"] == "N00" and run["arms"]["vol"]["ticker"] == "N24" and ss.run_done(d, 2)
    assert pd.read_pickle(ss.scores_path(d)).score.max() == pytest.approx(1.0)
    assert pd.read_pickle(ss.scores_path(d, "vol")).score.max() == pytest.approx(5.0)
    keys = [ss.pick_key(p) for p in picks]
    assert keys == ["2026-09-28|2|N00", "2026-09-28|2|N24|vol"]
    now = et(2026, 9, 28, 11, 20)
    assert sorted(ss.pick_key(p) for p in ss.pending_entries(now)) == sorted(keys)
    # the arm switched off: the model alone, as before
    monkeypatch.setattr(settings, "enable_sel_short_vol", False)
    run3 = ss.decide(res, d, 3)
    assert [p.get("arm") for p in ss.read_picks(d)][2:] == ["model"] and run3["arms"] == {}


# ── the short-interest filter (2026-09-27: "Under one day of volume") ──────────

def test_short_interest_filter_journals_crowded_picks_untraded(on, monkeypatch):
    """Both arms short a pick only when its short interest is under one day of
    volume (FINRA days to cover at its 1.00 floor). Above it the pick is
    journaled `crowded` — target and deadline kept so it can be followed, never
    pending — and it still counts as the name's pick of the day (the evaluation
    applied the filter at the trade step). An unknown value passes, as evaluated."""
    d = date(2026, 9, 28)
    res = _res()
    stand = pd.DataFrame({"n_prior": 0.0, "prior_max": np.nan, "prior_min": np.nan},
                         index=pd.Index(res.ticker, name="ticker"))
    assert settings.enable_sel_short_dtc_filter and settings.sel_short_max_days_to_cover == 1.0
    ok = ss.select(res.assign(dtc=1.0), d, 2, stand, [])
    assert (ok["decision"], ok["days_to_cover"]) == ("short", 1.0)
    for arm, tk in (("model", "N00"), ("vol", "N24")):
        r = ss.select(res.assign(dtc=np.where(res.ticker == tk, 1.47, 1.0)), d, 2, stand, [], arm=arm)
        assert (r["ticker"], r["decision"], r["days_to_cover"]) == (tk, "crowded", 1.47)
        assert r["target"] == pytest.approx(20.0 - ss.give_back(arm) * 10.0)
        assert datetime.fromisoformat(r["deadline"]) == ss.bar_end_et(ss.session_after(d, 15), 2)
    # unknown passes: a NaN in the scores, or scores without the column
    assert ss.select(res.assign(dtc=np.nan), d, 2, stand, [])["decision"] == "short"
    assert ss.select(res, d, 2, stand, [])["days_to_cover"] is None
    # the pick still counts as the name's fresh pick of the day
    crowd = ss.select(res.assign(dtc=3.0), d, 2, stand, [])
    assert ss.select(res.assign(dtc=1.0), d, 3, stand, [crowd])["decision"] == "not_first_today"
    # a not-a-riser is judged before the filter
    assert ss.select(res.assign(dtc=3.0, pre5=25.0), d, 2, stand, [])["decision"] == "not_a_riser"
    ss._journal(crowd)
    assert ss.pending_entries(et(2026, 9, 28, 11, 10)) == []
    monkeypatch.setattr(settings, "sel_short_max_days_to_cover", 3.0)
    assert ss.select(res.assign(dtc=3.0), d, 2, stand, [])["decision"] == "short"
    monkeypatch.setattr(settings, "sel_short_max_days_to_cover", 1.0)
    monkeypatch.setattr(settings, "enable_sel_short_dtc_filter", False)         # switched off: the old rule
    assert ss.select(res.assign(dtc=3.0), d, 2, stand, [])["decision"] == "short"


def test_decide_records_how_many_names_carried_days_to_cover(on):
    d = date(2026, 9, 28)
    run = ss.decide(_res().assign(dtc=[1.0] * 20 + [np.nan] * 5), d, 2)
    assert run["n_days_to_cover"] == 20
    assert run["days_to_cover"] == 1.0 and run["decision"] == "short"             # the model's N00
    assert run["arms"]["vol"]["days_to_cover"] is None and run["arms"]["vol"]["decision"] == "short"


def test_the_scorer_reads_days_to_cover_from_the_session_snapshot(on, monkeypatch):
    """`_score_chunk` hands every scored row the snapshot's `dp_si_dtc`. Without
    it every pick would pass the filter unjudged — which looks like normal
    operation — so the plumbing is pinned here, stores and model stubbed."""
    from loguru import logger as _lg
    from src.analysis import deep_features as dfe
    from src.signals import ml_model
    monkeypatch.setattr(_lg, "remove", lambda *a, **k: None)      # the worker resets loguru; not in-process
    assert ss.DTC_FEATURE in dfe.SNAPSHOT_FEATURES
    d = date(2026, 9, 28)
    target_start = pd.Timestamp(ss.bar_end_et(d, 1)).tz_convert("UTC").tz_localize(None) - ss.BAR

    class Booster:
        def predict(self, M, num_iteration=None):
            return np.zeros(len(M))

    monkeypatch.setattr(ss, "load_model", lambda: (Booster(), {"features": ["f1", ss.DTC_FEATURE],
                                                               "num_iteration": 1}))
    monkeypatch.setattr(ss, "_deep_and_recent", lambda t, day, fetch=True: (None, pd.DataFrame({"x": [1]})))
    monkeypatch.setattr(ss, "series", lambda tk, today, now, deep=None: (None,) * 5)
    monkeypatch.setattr(ss, "_close_at", lambda idx, c, day, bar: 8.0)
    frame = {"bar_ts": target_start, "close": 10.0, "sday": ss.dnum(d), "bar_idx": 1,
             "features": {"f1": 0.5, settings.sel_short_vol_feature: 3.0}}
    monkeypatch.setattr(ml_model, "features_30m_from_hlc", lambda tk, hlc, now: (frame, "OK"))
    monkeypatch.setattr(dfe, "load_session_snapshot", lambda day: {"AAA": {}, "BBB": {}})
    monkeypatch.setattr(dfe, "serving_vector",
                        lambda tk, sday, close, bar, snap: {ss.DTC_FEATURE: {"AAA": 1.0}.get(tk, 4.2)})
    rows = {r["ticker"]: r for r in ss._score_chunk((["AAA", "BBB"], d.isoformat(), 1,
                                                     {"AAA": 1e7, "BBB": 1e7}))}
    assert (rows["AAA"]["status"], rows["AAA"]["dtc"], rows["BBB"]["dtc"]) == ("OK", 1.0, 4.2)
    # no session snapshot: the vol arm still ranks the row, its days to cover unknown
    monkeypatch.setattr(dfe, "load_session_snapshot", lambda day: None)
    r = ss._score_chunk((["AAA"], d.isoformat(), 1, {"AAA": 1e7}))[0]
    assert r["status"] == "NO_SNAPSHOT" and np.isnan(r["dtc"]) and r["vol"] == 3.0


# ── the relative-volume filter (2026-10-01: "implement relative volume filter to prod") ──

def _rvol_reference(vol, j):
    """The evaluation's construction (scratchpad events._events_ticker ``vratio``): the
    bar's volume over the mean of the up-to-260 bars before it, >= 20 of them."""
    lo = max(0, j - 260)
    if j - lo < 20 or np.mean(vol[lo:j]) <= 0:
        return float("nan")
    return float(vol[j] / np.mean(vol[lo:j]))


def test_rvol_at_is_the_evaluations_relative_volume():
    idx = pd.date_range("2026-08-03 13:30", periods=400, freq="30min")        # naive-UTC bar starts
    vol = np.random.default_rng(4).integers(1_000, 90_000, 400).astype(float)
    for j in (20, 25, 259, 260, 261, 300, 399):
        assert ss.rvol_at(idx, pd.Series(vol), idx[j]) == pytest.approx(_rvol_reference(vol, j), rel=1e-12), j
    assert np.isnan(ss.rvol_at(idx, vol, idx[19]))                             # fewer than 20 bars before it
    assert np.isnan(ss.rvol_at(idx, vol, idx[5] + pd.Timedelta(minutes=10)))  # not a bar of the series
    assert np.isnan(ss.rvol_at(idx, np.r_[np.zeros(399), 5.0], idx[399]))     # a zero mean
    assert np.isnan(ss.rvol_at(None, vol, idx[30])) and np.isnan(ss.rvol_at(idx, None, idx[30]))


def test_vol_arm_relative_volume_filter_journals_low_rvol_untraded(on, monkeypatch):
    """The VOL arm shorts a pick only when its bar traded at least 1.58x the name's usual
    30-minute volume. Below it the pick is journaled `low_rvol` — target and deadline
    kept, never pending — and it still counts as the name's pick of the day. Unknown
    passes; the model arm is never filtered but journals the value."""
    d = date(2026, 9, 28)
    res = _res().assign(rvol=2.0)
    stand = pd.DataFrame({"n_prior": 0.0, "prior_max": np.nan, "prior_min": np.nan},
                         index=pd.Index(res.ticker, name="ticker"))
    assert settings.enable_sel_short_vol_rvol_filter and settings.sel_short_vol_min_rvol == pytest.approx(1.58)
    v = ss.select(res, d, 2, stand, [], arm="vol")
    assert (v["ticker"], v["decision"], v["rvol"]) == ("N24", "short", 2.0)
    low = res.assign(rvol=np.where(res.ticker == "N24", 1.2, 2.0))
    r = ss.select(low, d, 2, stand, [], arm="vol")
    assert (r["ticker"], r["decision"], r["rvol"]) == ("N24", "low_rvol", 1.2)
    assert r["target"] == pytest.approx(20.0 - ss.give_back("vol") * 10.0)
    assert datetime.fromisoformat(r["deadline"]) == ss.bar_end_et(ss.session_after(d, 15), 2)
    json.dumps(r)
    # at the cut passes; unknown passes (NaN, or scores without the column)
    assert ss.select(res.assign(rvol=1.58), d, 2, stand, [], arm="vol")["decision"] == "short"
    assert ss.select(res.assign(rvol=np.nan), d, 2, stand, [], arm="vol")["decision"] == "short"
    assert ss.select(_res(), d, 2, stand, [], arm="vol")["rvol"] is None
    # the MODEL arm is never filtered, but journals the value
    m = ss.select(low.assign(rvol=0.1), d, 2, stand, [])
    assert (m["ticker"], m["decision"], m["rvol"]) == ("N00", "short", 0.1)
    # the short-interest filter is judged first; a not-a-riser before both
    assert ss.select(low.assign(dtc=5.0), d, 2, stand, [], arm="vol")["decision"] == "crowded"
    assert ss.select(low.assign(pre5=25.0), d, 2, stand, [], arm="vol")["decision"] == "not_a_riser"
    # still the name's fresh pick of the day; never pending
    assert ss.select(res, d, 3, stand, [r], arm="vol")["decision"] == "not_first_today"
    ss._journal(r)
    assert ss.pending_entries(et(2026, 9, 28, 11, 10)) == []
    monkeypatch.setattr(settings, "sel_short_vol_min_rvol", 1.0)
    assert ss.select(low, d, 2, stand, [], arm="vol")["decision"] == "short"
    monkeypatch.setattr(settings, "sel_short_vol_min_rvol", 1.58)
    monkeypatch.setattr(settings, "enable_sel_short_vol_rvol_filter", False)  # switched off: the old rule
    assert ss.select(low, d, 2, stand, [], arm="vol")["decision"] == "short"


def test_decide_records_how_many_names_carried_a_relative_volume(on):
    d = date(2026, 9, 28)
    run = ss.decide(_res().assign(rvol=[2.0] * 18 + [np.nan] * 7), d, 2)
    assert run["n_rvol"] == 18 and run["arms"]["vol"]["rvol"] is None          # N24's is unknown: it passes
    assert run["arms"]["vol"]["decision"] == "short" and run["rvol"] == 2.0     # the model's N00 journals it


def test_the_scorer_hands_every_row_its_relative_volume(on, monkeypatch):
    """The plumbing: `_score_chunk` computes each row's relative volume from the
    series it scored (`rvol_at` at the target bar) — without it every vol pick would
    pass the filter unjudged, which looks like normal operation."""
    from loguru import logger as _lg
    from src.analysis import deep_features as dfe
    from src.signals import ml_model
    monkeypatch.setattr(_lg, "remove", lambda *a, **k: None)
    d = date(2026, 9, 28)
    target_start = pd.Timestamp(ss.bar_end_et(d, 1)).tz_convert("UTC").tz_localize(None) - ss.BAR

    class Booster:
        def predict(self, M, num_iteration=None):
            return np.zeros(len(M))
    monkeypatch.setattr(ss, "load_model", lambda: (Booster(), {"features": ["f1"], "num_iteration": 1}))
    monkeypatch.setattr(ss, "_deep_and_recent", lambda t, day, fetch=True: (None, pd.DataFrame({"x": [1]})))
    idx = pd.date_range(end=target_start, periods=40, freq="30min")
    vol = pd.Series(np.r_[np.full(39, 1_000.0), 3_000.0])
    monkeypatch.setattr(ss, "series", lambda tk, today, now, deep=None: (idx, None, None, pd.Series(np.full(40, 10.0)), vol))
    frame = {"bar_ts": target_start, "close": 10.0, "sday": ss.dnum(d), "bar_idx": 1,
             "features": {"f1": 0.5, settings.sel_short_vol_feature: 3.0}}
    monkeypatch.setattr(ml_model, "features_30m_from_hlc", lambda tk, hlc, now: (frame, "OK"))
    monkeypatch.setattr(dfe, "load_session_snapshot", lambda day: {"AAA": {}})
    monkeypatch.setattr(dfe, "serving_vector", lambda tk, sday, close, bar, snap: {})
    r = ss._score_chunk((["AAA"], d.isoformat(), 1, {"AAA": 1e7}))[0]
    assert r["status"] == "OK" and r["rvol"] == pytest.approx(3.0)
    # a row the scorer could not reach keeps an unknown value (it passes the filter)
    monkeypatch.setattr(ss, "_deep_and_recent", lambda t, day, fetch=True: (None, None))
    r = ss._score_chunk((["AAA"], d.isoformat(), 1, {"AAA": 1e7}))[0]
    assert r["status"] == "FETCH_FAILED" and np.isnan(r["rvol"])


def test_a_run_up_measured_across_a_history_gap_is_flagged_but_traded(on):
    """A reused ticker stitches two securities into one stored history — the bankrupt
    Akoustis at $0.04 read as a +63,878% run-up of the new AKTS at $23.80. A base close
    older than the session before its day is flagged `pre5_stale` for the journal, but
    the pick trades like any other: the stale-history guard was removed 2026-09-30
    (user: "Remove the Stale-history guard from live")."""
    d = date(2026, 9, 28)
    pre_day = ss.sessions_before(d, 5)[0]

    def last_bar(day, hh=9, mm=30):
        return pd.DatetimeIndex([pd.Timestamp(datetime(day.year, day.month, day.day, hh, mm, tzinfo=ET))
                                 .tz_convert("UTC").tz_localize(None)])
    assert not ss._stale_base(last_bar(pre_day), pre_day, 2)                              # its own session
    assert not ss._stale_base(last_bar(ss.sessions_before(pre_day, 1)[0], 15, 30), pre_day, 2)   # the one before
    assert ss._stale_base(last_bar(ss.sessions_before(pre_day, 2)[0], 15, 30), pre_day, 2)       # two before: a gap
    assert ss._stale_base(last_bar(date(2026, 5, 1)), pre_day, 2)                         # months before (AKTS)
    assert not ss._stale_base(None, pre_day, 2) and not ss._stale_base(pd.DatetimeIndex([]), pre_day, 2)
    res = _res()
    stand = pd.DataFrame({"n_prior": 0.0, "prior_max": np.nan, "prior_min": np.nan},
                         index=pd.Index(res.ticker, name="ticker"))
    v = ss.select(res.assign(pre5_stale=[False] * 24 + [True]), d, 2, stand, [], arm="vol")
    assert (v["ticker"], v["decision"], v["pre5_stale"]) == ("N24", "short", True)        # flagged, still traded
    w = ss.select(res, d, 2, stand, [], arm="vol")
    assert (w["decision"], w["pre5_stale"]) == ("short", False)
    assert v["target"] == pytest.approx(w["target"]) and v["deadline"] == w["deadline"]   # the flag changes nothing
    assert not any(ss.select(res.assign(pre5_stale=True), d, b, stand, [], arm=a)["decision"] == "stale_history"
                   for b in (1, 2, 3) for a in ("vol", "model"))


def test_the_scorer_flags_a_stale_run_up_base(on, monkeypatch):
    """The plumbing: `_score_chunk` flags the row when the history has a gap before the
    run-up's base, and keeps that base close (the last one at or before the bar, as the
    evaluation measured the run-up) — the flag is journal-only."""
    from loguru import logger as _lg
    from src.analysis import deep_features as dfe
    from src.signals import ml_model
    monkeypatch.setattr(_lg, "remove", lambda *a, **k: None)
    d = date(2026, 9, 28)
    target_start = pd.Timestamp(ss.bar_end_et(d, 1)).tz_convert("UTC").tz_localize(None) - ss.BAR

    class Booster:
        def predict(self, M, num_iteration=None):
            return np.zeros(len(M))
    monkeypatch.setattr(ss, "load_model", lambda: (Booster(), {"features": ["f1"], "num_iteration": 1}))
    monkeypatch.setattr(ss, "_deep_and_recent", lambda t, day, fetch=True: (None, pd.DataFrame({"x": [1]})))
    gap = pd.DatetimeIndex([pd.Timestamp("2026-05-01 13:30"), target_start])     # months-old base, then the bar
    monkeypatch.setattr(ss, "series", lambda tk, today, now, deep=None: (gap, None, None, pd.Series([0.04, 23.8]), None))
    frame = {"bar_ts": target_start, "close": 23.8, "sday": ss.dnum(d), "bar_idx": 1,
             "features": {"f1": 0.5, settings.sel_short_vol_feature: 30.0}}
    monkeypatch.setattr(ml_model, "features_30m_from_hlc", lambda tk, hlc, now: (frame, "OK"))
    monkeypatch.setattr(dfe, "load_session_snapshot", lambda day: {"AKTS": {}})
    monkeypatch.setattr(dfe, "serving_vector", lambda tk, sday, close, bar, snap: {})
    r = ss._score_chunk((["AKTS"], d.isoformat(), 1, {"AKTS": 1e7}))[0]
    assert r["status"] == "OK" and r["pre5_stale"] is True and r["pre5"] == 0.04
    fresh_idx = pd.DatetimeIndex([pd.Timestamp(ss.bar_end_et(ss.sessions_before(d, 5)[0], 1))
                                  .tz_convert("UTC").tz_localize(None) - ss.BAR, target_start])
    monkeypatch.setattr(ss, "series", lambda tk, today, now, deep=None: (fresh_idx, None, None,
                                                                         pd.Series([10.0, 23.8]), None))
    r = ss._score_chunk((["AKTS"], d.isoformat(), 1, {"AKTS": 1e7}))[0]
    assert r["pre5_stale"] is False and r["pre5"] == 10.0


def test_seed_vol_scores_from_the_arrays_feature_column(on):
    class Arr:                                         # a stand-in for sel_models.Arrays
        meta = {"base_features": ["x", settings.sel_short_vol_feature]}
        X = np.array([[0, 1.5], [0, 2.5], [0, np.nan], [0, 3.0]], np.float32)
        dn = np.array([ss.dnum(date(2026, 9, 24))] * 3 + [ss.dnum(date(2026, 9, 25))])
        bar = np.array([0, 0, 1, 0])
        tk = np.array([0, 1, 0, 1])
        tickers = ["AAA", "BBB"]
    assert ss.seed_vol_scores(Arr(), rows=np.arange(4)) == 2
    g = pd.read_pickle(ss.scores_path(date(2026, 9, 24), "vol"))
    assert sorted(g.ticker) == ["AAA", "BBB"] and set(g.score) == {1.5, 2.5}      # the NaN row dropped
    assert not ss.scores_path(date(2026, 9, 24)).exists()                         # the model's history untouched
    assert ss.seed_vol_scores(Arr(), rows=np.arange(4)) == 0                        # existing days kept


def test_pending_entries_expire_and_are_consumed_once(on):
    d = date(2026, 9, 28)
    rec = {"day": "2026-09-28", "bar_of_day": 1, "bar_end": ss.bar_end_et(d, 1).isoformat(),
           "ticker": "ABC", "decision": "short", "px": 10.0, "pre5": 8.0, "target": 9.0,
           "deadline": ss.bar_end_et(ss.session_after(d, 15), 1).isoformat(), "score": 1.0,
           "runup_pct": 25.0}
    ss._journal(rec)
    ss._journal(dict(rec, ticker="XYZ", decision="not_fresh"))
    now = et(2026, 9, 28, 10, 40)
    assert [r["ticker"] for r in ss.pending_entries(now)] == ["ABC"]
    ss.mark_consumed(d, ss.pick_key(rec), "opened")
    assert ss.pending_entries(now) == []
    ss._journal(dict(rec, bar_of_day=2, bar_end=ss.bar_end_et(d, 2).isoformat(), ticker="OLD"))
    assert ss.pending_entries(et(2026, 9, 28, 13, 0)) == []                 # 2.5 h later: stale
    assert ss.consumed(d)[ss.pick_key(dict(rec, bar_of_day=2, ticker="OLD"))] == "expired"


# ── the tracker ──────────────────────────────────────────────────────────────

def _pick(tk="ABC", day=date(2026, 9, 28), bar=1):
    return {"day": day.isoformat(), "bar_of_day": bar, "bar_end": ss.bar_end_et(day, bar).isoformat(),
            "ticker": tk, "decision": "short", "px": 10.0, "pre5": 8.0, "target": 9.0,
            "deadline": ss.bar_end_et(ss.session_after(day, 15), bar).isoformat(), "score": 1.23,
            "runup_pct": 25.0, "days_to_cover": 1.0}


def _tracker_env(monkeypatch, picks):
    from src.data import ibkr_borrow
    from src.performance import tracker
    monkeypatch.setattr(settings, "enable_sel_short", True)
    monkeypatch.setattr(ss, "pending_entries", lambda now=None: list(picks))
    consumed = {}
    monkeypatch.setattr(ss, "mark_consumed", lambda d, k, o: consumed.__setitem__(k, o))
    monkeypatch.setattr(tracker, "_fetch_price", lambda t: 10.2)
    monkeypatch.setattr(tracker, "_reference_close", lambda t: None)
    monkeypatch.setattr(tracker, "_execution_iso", lambda: "2026-09-28T14:40:00+00:00")
    calls = []

    def block(ticker, price, base=None, now=None, max_fee_pct="default"):
        calls.append((ticker, max_fee_pct))
        if ticker == "NOB":
            return "no_borrow", None
        return None, ibkr_borrow.Borrow(ticker, 496.0, -80.0, 900_000,
                                        datetime(2026, 9, 28, 9, 0, tzinfo=ET))
    monkeypatch.setattr(ibkr_borrow, "short_block", block)
    return tracker, consumed, calls


def test_record_opens_journaled_shorts_at_any_fee(monkeypatch):
    tracker, consumed, calls = _tracker_env(monkeypatch, [_pick("ABC"), _pick("NOB")])
    assert tracker.record_sel_short_trades(run_id="r1") == 1
    assert calls == [("ABC", None), ("NOB", None)]                     # no fee cap: "all borrowable"
    t = [x for x in tracker._load_trades() if x.get("entry_mechanism") == "sel_short"]
    assert len(t) == 1 and t[0]["ticker"] == "ABC" and t[0]["action"] == "SELL"
    assert t[0]["sel_target_price"] == 9.0 and t[0]["borrow_fee_pct"] == pytest.approx(496.0)
    assert t[0]["position_size_multiplier"] == pytest.approx(settings.sel_short_size_multiplier)
    assert t[0]["sel_days_to_cover"] == 1.0 and "days to cover 1.00" in t[0]["rationale"]
    assert consumed == {ss.pick_key(_pick("ABC")): "opened", ss.pick_key(_pick("NOB")): "no_borrow"}
    # the same name again while open, with the OLD rule (the switch at 1): not a second position
    monkeypatch.setattr(settings, "sel_short_max_open_per_ticker", 1)
    tracker2, consumed2, _ = _tracker_env(monkeypatch, [_pick("ABC", bar=3)])
    assert tracker2.record_sel_short_trades(run_id="r2") == 0
    assert list(consumed2.values()) == ["already_open"]


def test_two_arms_picking_one_name_open_one_trade_each_and_later_picks_are_noted(monkeypatch):
    """User directive 2026-10-04 (with the vol arm's quarter give-back): "Two trades, one per
    arm" — the same name picked by the model and the vol arm on one bar opens a trade for EACH
    arm, each with its own target and order ref, the model's first, netted at IBKR. With the
    old one-position rule (cap 1) a later pick of a held name by another arm is noted on the
    open trade, once per arm."""
    picks = [dict(_pick("ABC"), arm="vol", score=4.2, target=9.5), _pick("ABC"),
             dict(_pick("XYZ"), arm="vol", score=3.1, target=9.5)]
    tracker, consumed, calls = _tracker_env(monkeypatch, picks)
    assert tracker.record_sel_short_trades(run_id="r1") == 3
    t = [x for x in tracker._load_trades() if x.get("entry_mechanism") == "sel_short"]
    abc = [x for x in t if x["ticker"] == "ABC"]
    assert [x["sel_arm"] for x in abc] == ["model", "vol"]                       # the model's first
    assert [x["sel_target_price"] for x in abc] == [9.0, 9.5]                    # each arm's own target
    assert [x["sel_stack_n"] for x in abc] == [1, 2]
    assert abc[0]["sel_score"] == pytest.approx(1.23) and abc[0]["sel_vol_score"] is None
    assert abc[1]["sel_vol_score"] == pytest.approx(4.2) and abc[1]["sel_score"] is None
    assert "tail-regression" in abc[0]["rationale"] and "most volatile" in abc[1]["rationale"]
    assert f"{ss.give_back('model'):.0%} of the run-up given back" in abc[0]["rationale"]
    assert f"{ss.give_back('vol'):.0%} of the run-up given back" in abc[1]["rationale"]
    xyz = [x for x in t if x["ticker"] == "XYZ"][0]
    assert xyz["sel_arm"] == "vol" and f"a {ss.own_window('vol')}-session high" in xyz["rationale"]
    assert len({x["recommendation_id"] for x in t}) == 3                           # distinct IBKR order refs
    assert consumed == {"2026-09-28|1|ABC": "opened", "2026-09-28|1|ABC|vol": "opened",
                        "2026-09-28|1|XYZ|vol": "opened"}
    assert [c[0] for c in calls] == ["ABC", "XYZ"]                     # one borrow check per name per pass
    monkeypatch.setattr(settings, "sel_short_max_open_per_ticker", 1)   # the old rule: note, don't stack
    tracker2, consumed2, _ = _tracker_env(monkeypatch, [_pick("XYZ", bar=4)])
    assert tracker2.record_sel_short_trades(run_id="r2") == 0
    xyz = [x for x in tracker2._load_trades() if x["ticker"] == "XYZ"][0]
    assert [a["arm"] for a in xyz["sel_also"]] == ["model"] and list(consumed2.values()) == ["already_open"]
    tracker3, _, _ = _tracker_env(monkeypatch, [_pick("XYZ", bar=5)])   # the same arm again: noted once
    tracker3.record_sel_short_trades(run_id="r3")
    xyz = [x for x in tracker3._load_trades() if x["ticker"] == "XYZ"][0]
    assert [a["arm"] for a in xyz["sel_also"]] == ["model"]


def test_give_back_per_arm(on, monkeypatch):
    """User directive 2026-10-04 evening: the 100% give-back for the VOL arm only ("Vol arm only", once
    the per-arm numbers were in) — its target is the close 5 sessions before the pick, the whole run-up;
    the model and ETF arms keep half. Each arm reads its own knob."""
    assert settings.sel_short_vol_give_back == pytest.approx(1.0) and settings.sel_short_give_back == pytest.approx(0.5)
    assert [ss.give_back(a) for a in ("model", "vol", "etf")] == [0.5, 1.0, 0.5]
    d = date(2026, 9, 28)
    res = _res()
    stand = pd.DataFrame({"n_prior": 0.0, "prior_max": np.nan, "prior_min": np.nan},
                         index=pd.Index(res.ticker, name="ticker"))
    got = {a: ss.select(res, d, 2, stand, [], arm=a) for a in ("model", "vol", "etf")}
    assert all(r["decision"] == "short" for r in got.values())
    assert [got[a]["target"] for a in ("model", "vol", "etf")] == pytest.approx([15.0, 10.0, 15.0])   # px 20, 5 sessions before 10
    assert len({r["deadline"] for r in got.values()}) == 1                    # the deadline is the same 15 sessions
    monkeypatch.setattr(settings, "sel_short_vol_give_back", 0.25)            # the knobs stay per arm
    monkeypatch.setattr(settings, "sel_short_give_back", 1.0)
    assert [ss.give_back(a) for a in ("model", "vol", "etf")] == [1.0, 0.25, 1.0]
    got = {a: ss.select(res, d, 2, stand, [], arm=a) for a in ("model", "vol", "etf")}
    assert [got[a]["target"] for a in ("model", "vol", "etf")] == pytest.approx([10.0, 17.5, 10.0])


def test_a_repeat_pick_of_a_held_name_stacks_another_short(monkeypatch):
    """User directive 2026-09-28: "Remove the one position per ticker rule" — a
    repeat pick of a name the book already shorts opens ANOTHER trade;
    `sel_short_max_open_per_ticker` caps the stack (1 = the old rule), and a name
    another book holds is never stacked on."""
    tracker, _, _ = _tracker_env(monkeypatch, [_pick("ABC")])
    assert tracker.record_sel_short_trades(run_id="r1") == 1
    tracker2, consumed2, _ = _tracker_env(monkeypatch, [dict(_pick("ABC", bar=4), arm="vol", score=4.0)])
    assert tracker2.record_sel_short_trades(run_id="r2") == 1
    abc = [x for x in tracker2._load_trades() if x["ticker"] == "ABC" and x["status"] == "OPEN"]
    assert [t["sel_stack_n"] for t in abc] == [1, 2] and abc[0]["recommendation_id"] != abc[1]["recommendation_id"]
    assert "adds to 1 open short(s) on ABC" in abc[1]["rationale"] and not abc[0].get("sel_also")
    assert list(consumed2.values()) == ["opened"]
    monkeypatch.setattr(settings, "sel_short_max_open_per_ticker", 2)            # a stack of two at most
    tracker3, consumed3, _ = _tracker_env(monkeypatch, [_pick("ABC", bar=6)])
    assert tracker3.record_sel_short_trades(run_id="r3") == 0 and list(consumed3.values()) == ["already_open"]
    monkeypatch.setattr(settings, "sel_short_max_open_per_ticker", 0)
    _seed_trades(tracker3, [{"ticker": "LEG", "action": "BUY", "recommendation_id": "leg"}])   # another book
    tracker4, consumed4, _ = _tracker_env(monkeypatch, [_pick("LEG")])
    assert tracker4.record_sel_short_trades(run_id="r4") == 0 and list(consumed4.values()) == ["already_open"]


def test_a_pick_is_consumed_only_after_the_ledger_is_saved(monkeypatch):
    """Consumed first, a failed save (a DuckDB lock, 2026-09-25 16:39) lost the
    trade for good. Now the pick stays pending and the next tick opens it."""
    tracker, consumed, _ = _tracker_env(monkeypatch, [_pick("ABC")])

    def locked(trades):
        raise IOError("Cannot open file llm_trader.db: used by another process")
    monkeypatch.setattr(tracker, "_save_trades", locked)
    with pytest.raises(IOError):
        tracker.record_sel_short_trades(run_id="r1")
    assert consumed == {}


def test_record_skips_a_pick_already_at_its_target(monkeypatch):
    """A pick whose price has given back half its run-up before the entry step
    is not entered (the cover rule would close it at the next check)."""
    tracker, consumed, calls = _tracker_env(monkeypatch, [_pick("ABC")])
    monkeypatch.setattr(tracker, "_fetch_price", lambda t: 8.95)          # target 9.0
    assert tracker.record_sel_short_trades(run_id="r1") == 0
    assert consumed == {ss.pick_key(_pick("ABC")): "target_reached"}
    assert calls == []                                                   # never reached the borrow check
    assert not [x for x in tracker._load_trades() if x.get("entry_mechanism") == "sel_short"]


def test_launch_respects_a_live_scorer_and_ignores_a_dead_one(monkeypatch, on):
    import os
    import time as _time
    ss.root().mkdir(parents=True, exist_ok=True)
    (ss.root() / "model.json").write_text("{}", encoding="utf-8")
    launched = []

    class FakeProc:
        def __init__(self, args, **kw):
            launched.append(args)
            self.pid, self.rc = 4242, None

        def poll(self):
            return self.rc

    monkeypatch.setattr(ss.subprocess, "Popen", FakeProc)
    monkeypatch.setattr(ss, "_ACTIVE", {})
    now = et(2026, 9, 28, 10, 5)
    lock = ss.root() / "busy.lock"
    lock.write_text("1", encoding="utf-8")                               # a scorer beating right now
    assert ss.launch(now) is None and launched == []
    old = _time.time() - ss.LOCK_STALE_SECONDS - 5                       # its heartbeat stopped: dead
    os.utime(lock, (old, old))
    h = ss.launch(now)
    assert h is not None and launched[-1][-4:] == ["--day", "2026-09-28", "--bars", "0"]
    h["log"].close()
    lock.unlink()
    assert ss.launch(now) is None and len(launched) == 1                 # the one we launched still runs
    ss._ACTIVE["proc"].rc = 0
    h = ss.launch(now)
    assert h is not None and len(launched) == 2
    h["log"].close()


def test_every_missed_bar_is_scored_oldest_first(monkeypatch, on):
    """A tick that ran past the next bar used to leave the bar in between
    unscored (its pick lost, a hole in both freshness histories). Every
    completed bar without a run is scored, oldest first; a bar whose run failed
    twice is skipped so it cannot block the rest."""
    d = date(2026, 9, 28)
    (ss.root() / "runs").mkdir(parents=True, exist_ok=True)
    for b in (0, 2):
        (ss.root() / "runs" / f"2026-09-28_{b:02d}.json").write_text("{}", encoding="utf-8")
    now = et(2026, 9, 28, 12, 5)                                   # bars 0..4 complete
    assert ss.pending_bars(now) == [(d, 1), (d, 3), (d, 4)]
    ss.note_failure(d, 3, RuntimeError("boom"))
    assert ss.pending_bars(now) == [(d, 1), (d, 3), (d, 4)]         # one failure: retried
    ss.note_failure(d, 3, RuntimeError("boom"))
    assert ss.pending_bars(now) == [(d, 1), (d, 4)]                 # two: skipped
    assert ss.pending_bars(et(2026, 9, 28, 9, 50)) == []            # no bar complete yet


def test_one_crashing_bar_does_not_cost_the_bars_after_it(monkeypatch, on):
    ran = []

    def fake_run(d, b):
        if b == 1:
            raise RuntimeError("polygon down")
        ran.append(b)
        return {"bar": b}
    monkeypatch.setattr(ss, "run", fake_run)
    monkeypatch.setattr(ss.logger, "remove", lambda *a, **k: None)
    monkeypatch.setattr(ss.logger, "add", lambda *a, **k: 0)
    rc = ss.main(["--run", "--day", "2026-09-28", "--bars", "1,2"])
    assert rc == 1 and ran == [2]
    assert ss.run_failures(date(2026, 9, 28), 1) == 1 and not ss.run_done(date(2026, 9, 28), 1)


def test_a_defective_snapshot_takes_the_model_off_the_bar_not_the_vol_arm(monkeypatch, on):
    d = date(2026, 9, 28)
    monkeypatch.setattr(ss, "load_universe", lambda day: {f"N{i:02d}": 1e7 for i in range(25)})
    monkeypatch.setattr(ss, "ensure_snapshot", lambda day, names: 0.12)
    monkeypatch.setattr(ss, "score_bar", lambda day, bar, uni, fetch=True: _res())
    rec = ss.run(d, 2)
    assert rec["decision"] == "thin_run"                             # the model sat this bar out
    assert rec["arms"]["vol"]["decision"] == "short" and rec["arms"]["vol"]["ticker"] == "N24"
    assert rec["snapshot_coverage"] == pytest.approx(0.12)
    model_hist = pd.read_pickle(ss.scores_path(d)) if ss.scores_path(d).exists() else pd.DataFrame()
    assert len(model_hist) == 0                                      # no scores pushed into its history
    assert len(pd.read_pickle(ss.scores_path(d, "vol"))) == 25


def test_a_failed_snapshot_rebuild_is_recorded_and_does_not_raise(monkeypatch, on, tmp_path):
    from src.analysis import deep_features as dfe
    d = date(2026, 9, 28)
    monkeypatch.setattr(dfe, "snapshot_path", lambda day: tmp_path / f"{day}.parquet")   # never the real store
    monkeypatch.setattr(ss, "snapshot_coverage", lambda day, names: 0.0)

    def boom(day, workers=6):
        raise MemoryError("snapshot build ran out of memory")
    monkeypatch.setattr(dfe, "build_session_snapshot", boom)
    assert ss.ensure_snapshot(d, ["AAA"]) == 0.0
    st = ss.read_snapshot_status(d)
    assert st["found"] == 0.0 and st["exists"] is False and "MemoryError" in st["error"]


def test_health_names_unscored_bars_thin_runs_crashes_and_the_snapshot(monkeypatch, on):
    """User directive 2026-09-27: scorer failures and failed pre-open snapshots
    go in the email digest instead of reading as a quiet no-trade day."""
    d = date(2026, 9, 28)
    runs = ss.root() / "runs"
    runs.mkdir(parents=True, exist_ok=True)
    for b, decision in ((0, "short"), (1, "thin_run"), (3, "none")):
        (runs / f"2026-09-28_{b:02d}.json").write_text(json.dumps({"decision": decision}), encoding="utf-8")
    ss.note_failure(d, 2, RuntimeError("boom"))
    (ss.root() / "launches").mkdir(parents=True, exist_ok=True)
    (ss.root() / "launches" / "2026-09-28.jsonl").write_text(
        json.dumps({"args": ["--run", "--day", "2026-09-28", "--bars", "2"], "rc": 1}) + "\n", encoding="utf-8")
    ss._write_snapshot_status(d, found=0.1, exists=True, coverage=0.95, rebuilt=True, error=None)
    h = ss.health(et(2026, 9, 28, 12, 0))                           # bars 0..4 due; bar 4 just ended
    assert (h["runs"], h["bars_due"]) == (3, 5)
    text = " | ".join(h["problems"])
    assert "1 completed bar(s) never scored: 11:00 (1 crashed" in text
    assert "too thin to rank" in text and "10:30" in text
    assert "scorer exited with an error 1 time(s)" in text
    # resolved by the rebuild: in the digest as a note, not an alarm
    assert "pre-open snapshot" not in text
    assert h["notes"] == ["pre-open snapshot was defective (10% priced): rebuilt by the scorer (95%)"]
    ss._write_snapshot_status(d, coverage=0.5)
    assert "still 50% after the rebuild" in " | ".join(ss.health(et(2026, 9, 28, 12, 0))["problems"])
    monkeypatch.setattr(settings, "enable_sel_short", False)
    assert ss.health(et(2026, 9, 28, 12, 0))["problems"] == []


def test_lock_heartbeat_keeps_a_long_run_alive(monkeypatch, on):
    import os
    import time as _time
    monkeypatch.setattr(ss, "LOCK_HEARTBEAT_SECONDS", 0.05)
    lock = ss._hold_lock()
    old = _time.time() - 1000
    os.utime(lock, (old, old))
    _time.sleep(0.4)
    assert _time.time() - lock.stat().st_mtime < ss.LOCK_STALE_SECONDS
    lock.unlink()


def test_exit_reason_is_judged_at_any_time_on_a_fresh_mark():
    """User directive 2026-09-27: 'We should always have the possibility to enter
    or exit at any time' — no session window (it used to be 10:00-16:10 ET, which
    the 16:00 tick, reaching its monitor at ~16:16, could never meet). The target
    is judged only on a mark fetched within sel_short_mark_max_age_minutes."""
    from src.performance import tracker

    def t_at(when, px=8.9):
        return {"sel_target_price": 9.0, "current_price": px, "current_price_datetime": when.isoformat(),
                "sel_deadline": et(2026, 10, 19, 10, 30).isoformat()}
    for when in (et(2026, 9, 29, 9, 45), et(2026, 9, 29, 16, 16), et(2026, 9, 29, 21, 30),
                 et(2026, 10, 3, 11, 0)):                                   # pre-market, post-close, overnight, Saturday
        assert tracker._sel_short_exit_reason(t_at(when - timedelta(minutes=5)), when) == "sel_target"
    now = et(2026, 9, 29, 11, 0)
    assert tracker._sel_short_exit_reason(t_at(now, px=9.5), now) is None
    assert tracker._sel_short_exit_reason(t_at(now - timedelta(hours=3)), now) is None      # stale mark
    assert tracker._sel_short_exit_reason({"sel_target_price": 9.0, "current_price": 8.9}, now) is None
    late = et(2026, 10, 19, 21, 30)                                          # deadline passed, any session
    assert tracker._sel_short_exit_reason(t_at(late - timedelta(hours=6), px=9.5), late) == "sel_time"


def _seed_trades(tracker, rows):
    trades = tracker._load_trades()
    for r in rows:
        trades.append(dict({"type": "STOCK", "status": "OPEN", "entry_date": "2026-09-20",
                            "entry_datetime": "2026-09-20T14:00:00+00:00", "entry_session": "rth",
                            "entry_price": 10.0, "current_price": 10.0, "position_size_multiplier": 1.0},
                           **r))
    tracker._save_trades(trades)


def test_legacy_paths_never_touch_the_strategy_book(monkeypatch):
    from src.models import Recommendation
    from src.performance import tracker
    _seed_trades(tracker, [{"ticker": "SSS", "action": "SELL", "entry_mechanism": "sel_short",
                            "recommendation_id": "a"},
                           {"ticker": "LEG", "action": "SELL", "recommendation_id": "b"}])
    rec = lambda tk: Recommendation(ticker=tk, type="STOCK", direction="BULLISH", confidence=0.9,  # noqa: E731
                                    action="BUY", time_horizon="SWING", rationale="t",
                                    generated_at=datetime.now(timezone.utc))
    tracker.close_trades_on_signal_reversal([rec("SSS"), rec("LEG")])
    st = {t["ticker"]: t["status"] for t in tracker._load_trades()}
    assert st == {"SSS": "OPEN", "LEG": "CLOSED"}


def test_legacy_flatten_runs_once_in_regular_hours(monkeypatch):
    from src.performance import tracker
    monkeypatch.setattr(settings, "legacy_flatten_after", "2026-09-28T09:30:00-04:00")
    _seed_trades(tracker, [{"ticker": "SSS", "action": "SELL", "entry_mechanism": "sel_short",
                            "recommendation_id": "a"},
                           {"ticker": "LEG", "action": "BUY", "recommendation_id": "b"},
                           {"ticker": "FT", "action": "SELL", "entry_mechanism": "follow_through",
                            "recommendation_id": "c"}])
    assert tracker.flatten_legacy_positions(now=et(2026, 9, 28, 9, 0)) == 0      # before the instant
    assert tracker.flatten_legacy_positions(now=et(2026, 9, 28, 20, 30)) == 0    # after it, not RTH
    assert tracker.flatten_legacy_positions(now=et(2026, 9, 28, 9, 31)) == 2
    by = {t["ticker"]: t for t in tracker._load_trades()}
    assert by["SSS"]["status"] == "OPEN"
    assert by["LEG"]["exit_reason"] == by["FT"]["exit_reason"] == "legacy_flatten"
    _seed_trades(tracker, [{"ticker": "NEW", "action": "BUY", "recommendation_id": "d"}])
    assert tracker.flatten_legacy_positions(now=et(2026, 9, 29, 10, 0)) == 0     # once, ever
    assert json.loads(tracker._LEGACY_FLATTEN_MARKER.read_text())["after"] == "2026-09-28T09:30:00-04:00"


def test_monitor_covers_on_target_and_skips_legacy_monitor(monkeypatch):
    from src.performance import tracker
    marked = et(2026, 9, 29, 10, 55).isoformat()
    _seed_trades(tracker, [{"ticker": "HIT", "action": "SELL", "entry_mechanism": "sel_short",
                            "recommendation_id": "a", "current_price": 8.0, "sel_target_price": 9.0,
                            "current_price_datetime": marked,
                            "sel_deadline": et(2026, 10, 19, 10, 30).isoformat()},
                           {"ticker": "HOLD", "action": "SELL", "entry_mechanism": "sel_short",
                            "recommendation_id": "b", "current_price": 12.0, "sel_target_price": 9.0,
                            "current_price_datetime": marked,
                            "sel_deadline": et(2026, 10, 19, 10, 30).isoformat()}])
    assert tracker.monitor_sel_short_positions(now=et(2026, 9, 29, 11, 0)) == 1
    by = {t["ticker"]: t for t in tracker._load_trades()}
    assert by["HIT"]["exit_reason"] == "sel_target" and by["HOLD"]["status"] == "OPEN"
    # the legacy monitor passes over the strategy's trade even with a huge adverse move
    monkeypatch.setattr(settings, "enable_signal_decay_exits", True)
    trades = tracker._load_trades()
    for t in trades:
        if t["ticker"] == "HOLD":
            t["current_price"] = 100.0
    tracker._save_trades(trades)
    tracker.monitor_open_positions(signals_by_ticker={})
    assert {t["ticker"]: t["status"] for t in tracker._load_trades()}["HOLD"] == "OPEN"


# ── the volatility-normalised exit (user directive 2026-09-28: "Implement the
#    Cover when volatility halves (in profit) for the two live models") ─────────

def test_picks_record_their_atr_and_the_trade_stamps_it(on, monkeypatch):
    d = date(2026, 9, 28)
    res = _res().assign(vol=np.linspace(1.0, 5.0, 25))
    stand = pd.DataFrame({"n_prior": 0.0, "prior_max": np.nan, "prior_min": np.nan},
                         index=pd.Index(res.ticker, name="ticker"))
    m = ss.select(res, d, 2, stand, [])
    v = ss.select(res, d, 2, stand, [], arm="vol")
    assert (m["ticker"], m["atr_pct"]) == ("N00", pytest.approx(1.0))       # the model pick's own ATR%
    assert v["atr_pct"] == pytest.approx(v["score"]) == pytest.approx(5.0)  # the vol arm's score IS it
    monkeypatch.setattr(settings, "enable_sel_short_volnorm_exit", True)
    old_vol = {k: x for k, x in dict(_pick("XYZ"), arm="vol", score=3.1).items()}   # journaled before 09-28
    tracker, _, _ = _tracker_env(monkeypatch, [dict(_pick("ABC"), atr_pct=3.0), old_vol])
    assert tracker.record_sel_short_trades(run_id="r1") == 2
    t = {x["ticker"]: x for x in tracker._load_trades() if x.get("entry_mechanism") == "sel_short"}
    assert t["ABC"]["sel_atr_pct"] == pytest.approx(3.0)
    assert "ATR% falls to 50% of 3.00%" in t["ABC"]["rationale"]
    assert t["XYZ"]["sel_atr_pct"] == pytest.approx(3.1)                    # falls back to the vol score


def _vn_trade(**kw):
    now = et(2026, 9, 29, 11, 0)
    return dict({"entry_price": 10.0, "current_price": 9.5, "sel_target_price": 9.0, "sel_atr_pct": 4.0,
                 "current_price_datetime": (now - timedelta(minutes=5)).isoformat(),
                 "sel_deadline": et(2026, 10, 19, 10, 30).isoformat()}, **kw)


def test_volnorm_exit_reason_needs_profit_a_halved_atr_and_a_fresh_mark(monkeypatch):
    from src.performance import tracker
    now = et(2026, 9, 29, 11, 0)
    monkeypatch.setattr(settings, "enable_sel_short_volnorm_exit", True)
    r = tracker._sel_short_exit_reason
    assert r(_vn_trade(), now, atr_now=2.0) == "sel_volnorm"                 # 2.0 <= 0.5 x 4.0
    assert r(_vn_trade(), now, atr_now=2.1) is None
    assert r(_vn_trade(), now, atr_now=None) is None                         # no value, no exit
    assert r(_vn_trade(current_price=10.5), now, atr_now=1.0) is None        # losing: never
    assert r(_vn_trade(current_price=10.0), now, atr_now=1.0) is None        # flat: not in profit
    assert r(_vn_trade(current_price=8.9), now, atr_now=1.0) == "sel_target"  # the target first, as evaluated
    stale = _vn_trade(current_price_datetime=(now - timedelta(hours=2)).isoformat())
    assert r(stale, now, atr_now=1.0) is None
    assert r(_vn_trade(sel_atr_pct=None), now, atr_now=1.0) is None
    assert r(_vn_trade(sel_deadline=(now - timedelta(minutes=1)).isoformat()), now, atr_now=9.0) == "sel_time"
    monkeypatch.setattr(settings, "sel_short_volnorm_ratio", 0.6)
    assert r(_vn_trade(), now, atr_now=2.4) == "sel_volnorm"
    monkeypatch.setattr(settings, "enable_sel_short_volnorm_exit", False)
    assert r(_vn_trade(), now, atr_now=1.0) is None


def _cv_trade(**kw):
    now = et(2026, 9, 29, 11, 0)
    return dict({"ticker": "SQZ", "sel_arm": "vol", "entry_price": 10.0, "entry_date": "2026-09-22",
                 "current_price": 60.0, "sel_target_price": 9.0, "sel_atr_pct": 4.0,
                 "current_price_datetime": (now - timedelta(minutes=5)).isoformat(),
                 "sel_deadline": et(2026, 10, 19, 10, 30).isoformat()}, **kw)


def test_the_vol_arms_squeeze_cover_buys_back_at_a_multiple_of_the_split_adjusted_entry(monkeypatch):
    """User directive 2026-10-05: "have it tuned so that we don't have margin calls while still
    maximizing growth" — the vol arm covers a short whose fresh mark reaches
    `sel_short_vol_cover_multiple` x its entry price, the entry carried through every split
    executed since the entry day (these names reverse-split often: a 1-for-10 mid-hold is not a
    10-fold squeeze)."""
    from src.data import intraday_store as ist
    from src.performance import tracker
    now = et(2026, 9, 29, 11, 0)
    monkeypatch.setattr(settings, "enable_sel_short_vol_squeeze_cover", True)
    monkeypatch.setattr(settings, "sel_short_vol_cover_multiple", 6.0)
    rows = {}
    monkeypatch.setattr(ist, "split_rows", lambda: rows)
    r = tracker._sel_short_exit_reason
    assert r(_cv_trade(), now) == "sel_cover"                                # 60 >= 6 x 10
    assert r(_cv_trade(current_price=59.9), now) is None
    assert r(_cv_trade(sel_arm="model"), now) is None                        # tuned on the vol arm only
    assert r(_cv_trade(sel_arm="etf"), now) is None
    assert r(_cv_trade(sel_arm="model+vol"), now) == "sel_cover"             # a joint trade from before 10-04
    stale = _cv_trade(current_price_datetime=(now - timedelta(hours=2)).isoformat())
    assert r(stale, now) is None                                             # never on an old price
    assert r(_cv_trade(sel_deadline=(now - timedelta(minutes=1)).isoformat()), now) == "sel_time"
    rows["SQZ"] = [(date(2026, 9, 25), 10.0, 1.0)]                           # 1-for-10 since the entry
    assert r(_cv_trade(), now) is None                                       # 60 is 6 before the split
    assert r(_cv_trade(current_price=600.0), now) == "sel_cover"
    rows["SQZ"] = [(date(2026, 9, 25), 1.0, 2.0)]                            # 2-for-1 since the entry
    assert r(_cv_trade(current_price=30.0), now) == "sel_cover"              # 30 is 60 before it
    rows["SQZ"] = [(date(2026, 9, 22), 10.0, 1.0)]                           # on the entry day: in the entry
    assert r(_cv_trade(), now) == "sel_cover"
    monkeypatch.setattr(ist, "split_rows", lambda: None)                     # unreadable: the raw entry
    assert r(_cv_trade(), now) == "sel_cover"
    monkeypatch.setattr(settings, "enable_sel_short_vol_squeeze_cover", False)
    assert r(_cv_trade(), now) is None


def test_monitor_covers_when_volatility_halves_and_fetches_only_what_it_could_close(monkeypatch):
    from src.performance import tracker
    now = et(2026, 9, 29, 11, 0)
    marked = (now - timedelta(minutes=5)).isoformat()
    base = {"action": "SELL", "entry_mechanism": "sel_short", "current_price_datetime": marked,
            "sel_target_price": 9.0, "sel_deadline": et(2026, 10, 19, 10, 30).isoformat()}
    _seed_trades(tracker, [dict(base, ticker="WIN", recommendation_id="a", current_price=9.5, sel_atr_pct=4.0),
                           dict(base, ticker="LOSE", recommendation_id="b", current_price=10.5, sel_atr_pct=4.0),
                           dict(base, ticker="OLD", recommendation_id="c", current_price=9.6,
                                sel_bar_end=et(2026, 9, 25, 11, 0).isoformat()),
                           dict(base, ticker="CALM", recommendation_id="d", current_price=9.4, sel_atr_pct=4.0)])
    asked, backfilled = [], []
    monkeypatch.setattr(ss, "live_atr", lambda tks, now=None: asked.extend(tks) or {
        "WIN": {"atr_pct": 1.8, "bar_end": "2026-09-29T11:00:00-04:00"},
        "OLD": {"atr_pct": 1.0, "bar_end": "2026-09-29T11:00:00-04:00"},
        "CALM": {"atr_pct": 2.5, "bar_end": "2026-09-29T11:00:00-04:00"}})
    monkeypatch.setattr(ss, "atr_at", lambda tk, d, k: backfilled.append((tk, d, k)) or 3.0)
    monkeypatch.setattr(settings, "enable_sel_short_volnorm_exit", False)
    assert tracker.monitor_sel_short_positions(now=now) == 0 and asked == []   # off: nothing fetched
    monkeypatch.setattr(settings, "enable_sel_short_volnorm_exit", True)
    assert tracker.monitor_sel_short_positions(now=now) == 2
    assert sorted(asked) == ["CALM", "OLD", "WIN"]                           # never the losing trade
    assert backfilled == [("OLD", date(2026, 9, 25), 2)]                     # the pick bar's ATR%, once
    by = {t["ticker"]: t for t in tracker._load_trades()}
    assert by["WIN"]["exit_reason"] == "sel_volnorm" and by["WIN"]["sel_exit_atr_pct"] == pytest.approx(1.8)
    assert by["WIN"]["sel_exit_atr_bar"] == "2026-09-29T11:00:00-04:00"
    assert by["OLD"]["exit_reason"] == "sel_volnorm" and by["OLD"]["sel_atr_pct"] == pytest.approx(3.0)
    assert by["LOSE"]["status"] == by["CALM"]["status"] == "OPEN"            # 2.5 > 0.5 x 4.0


def _bars(days, seed=3):
    """Synthetic regular-hours 30-min OHLCV (naive-UTC bar starts) for ``days``."""
    rng = np.random.default_rng(seed)
    idx = []
    for d in days:
        for k in range(13):
            idx.append(pd.Timestamp(datetime(d.year, d.month, d.day, 9, 30, tzinfo=ET) + timedelta(minutes=30 * k))
                       .tz_convert("UTC").tz_localize(None))
    c = 20.0 * np.exp(np.cumsum(rng.normal(0, 0.02, len(idx))))
    h = c * (1 + np.abs(rng.normal(0, 0.01, len(idx))))
    lo = c * (1 - np.abs(rng.normal(0, 0.01, len(idx))))
    return pd.DataFrame({"Open": c, "High": h, "Low": lo, "Close": c, "Volume": 1e5}, index=pd.DatetimeIndex(idx))


def test_atr_at_is_the_scorers_feature_at_that_bar(monkeypatch):
    """The live value must be built like the pick's — the deep store + that
    session's bars through `features_30m_from_hlc`, float32 — or a halving could
    be an artefact of two constructions. A series whose newest bar is from an
    earlier session (a failed fetch) must give nothing, not an old value."""
    from src.signals.ml_model import features_30m_from_hlc
    d = date(2026, 9, 29)
    hist = ss.sessions_before(d, 40)
    deep, today = _bars(hist), _bars([d], seed=4)
    calls = []
    monkeypatch.setattr(ss, "_deep_and_recent", lambda tk, day, fetch=True: calls.append(tk) or (deep, today))
    ss._ATR_MEMO.clear()
    k = 3                                                   # the bar ending 11:30 ET
    end = pd.Timestamp(ss.bar_end_et(d, k)).tz_convert("UTC").tz_localize(None)
    frame, _ = features_30m_from_hlc("SYN", ss.series("SYN", today, end, deep=deep), end)
    want = float(np.float32(frame["features"]["atr_pct_14"]))
    assert ss.atr_at("SYN", d, k) == want
    assert ss.atr_at("SYN", d, k) == want and calls == ["SYN"]              # the exact bar is memoised
    got = ss.live_atr(["SYN"], ss.bar_end_et(d, k) + timedelta(minutes=5))
    assert got == {"SYN": {"atr_pct": want, "bar_end": ss.bar_end_et(d, k).isoformat()}}
    # the session's bars missing entirely: the newest bar is yesterday's -> nothing
    monkeypatch.setattr(ss, "_deep_and_recent", lambda tk, day, fetch=True: (deep, pd.DataFrame()))
    ss._ATR_MEMO.clear()
    assert ss.atr_at("SYN", d, k) is None
    # overnight / weekend: the latest completed bar is the last session's 16:00 bar
    assert ss.last_completed_bar(et(2026, 10, 3, 12, 0)) == (date(2026, 10, 2), 12)
    assert ss.last_completed_bar(et(2026, 9, 29, 9, 45)) == (date(2026, 9, 28), 12)
    assert ss.bar_from_end(ss.bar_end_et(d, 5).isoformat()) == (d, 5)


def test_short_block_fee_cap_override(monkeypatch, tmp_path):
    from src.data import ibkr_borrow as ib
    monkeypatch.setattr(settings, "enable_ibkr_borrow_snapshot", True)
    monkeypatch.setattr(settings, "enable_short_borrow_gate", True)
    monkeypatch.setattr(settings, "ibkr_borrow_max_age_minutes", 120.0)
    monkeypatch.setattr(settings, "short_borrow_max_fee_pct", 50.0)
    ib.reset()
    monkeypatch.setattr(settings, "short_borrow_min_available_usd", 10_000.0)
    text = ("#BOF|2026.09.28|09:00:00\n#SYM|CUR|NAME|CON|ISIN|REBATERATE|FEERATE|AVAILABLE|FIGI|\n"
            "HOT|USD|HOT|1|X|-80|496.15|500000|B|\nNONE|USD|NONE|1|X|0|3|0|B|\n#EOF|2\n")
    monkeypatch.setattr(ib, "_download", lambda timeout=30.0: text.encode())
    ib.snapshot()
    now = et(2026, 9, 28, 9, 30)
    assert ib.short_block("HOT", 10.0, now=now)[0] == "borrow_fee"                  # the setting's cap
    assert ib.short_block("HOT", 10.0, now=now, max_fee_pct=None)[0] is None        # "all borrowable"
    assert ib.short_block("NONE", 10.0, now=now, max_fee_pct=None)[0] == "no_borrow"


def test_pipeline_no_longer_opens_legacy_entries():
    """The rank rule's recommendations stay computed and persisted, but the only
    call that opens them sits behind `enable_legacy_entries` (False)."""
    src = Path("src/pipeline.py").read_text(encoding="utf-8")
    i = src.index("trade_diag = record_new_trades(")
    assert "if settings.enable_legacy_entries:" in src[i - 400:i]
    # gate_diag reads trade_diag on every tick: with legacy entries off it must
    # still be bound (a NameError at the end of every tick otherwise)
    g = src.index("if settings.enable_legacy_entries:")
    assert "trade_diag = {}" in src[g - 200:g]
    assert type(settings).model_fields["enable_legacy_entries"].default is False
    assert type(settings).model_fields["enable_follow_through_trading"].default is False
    assert "record_sel_short_trades(run_id=run_id)" in src
    assert src.index("flatten_legacy_positions(") < src.index("        monitor_open_positions(")


def test_features_from_supplied_bars_equal_the_served_ones(monkeypatch):
    """`features_30m_from_hlc` IS `features_30m` on the same bars."""
    from src.analysis import ml_dataset
    from src.signals import ml_model
    idx = pd.date_range("2026-01-02 14:30", periods=1000, freq="30min")
    et_idx = idx.tz_localize("UTC").tz_convert("America/New_York")
    idx = idx[(et_idx.hour * 60 + et_idx.minute >= 570) & (et_idx.hour * 60 + et_idx.minute <= 930)
              & (et_idx.dayofweek < 5)]
    rng = np.random.default_rng(3)
    c = pd.Series(50 * np.exp(np.cumsum(rng.normal(0, 0.01, len(idx)))))
    hlc = (idx, c * 1.004, c * 0.996, c, pd.Series(rng.integers(1e4, 1e5, len(idx)).astype(float)))
    monkeypatch.setattr(ml_dataset, "hlc_30m", lambda t: hlc)
    now = pd.Timestamp(idx[-1]) + pd.Timedelta(minutes=30)
    ml_model._FEAT_CACHE.clear()
    a, la = ml_model.features_30m("ZZZ", now)
    b, lb = ml_model.features_30m_from_hlc("ZZZ", hlc, now)
    assert la == lb
    if a is not None:
        assert a["bar_ts"] == b["bar_ts"] and a["close"] == b["close"]
        fa, fb = a["features"], b["features"]
        assert set(fa) == set(fb)
        for k in fa:
            assert (fa[k] == fb[k]) or (pd.isna(fa[k]) and pd.isna(fb[k]))


# ── the live trading path first (2026-09-28: "running the trading steps first",
#    after the two models' data fetching, engineering and inference) ─────────

def test_the_live_path_runs_before_the_shadow_pipeline_and_after_the_scorer():
    """Structural: the live path's marks + exits run right after the scorer is
    launched, its entries + broker sync run INSIDE the Steps 1-3 fetch pool (the
    main thread, overlapping the fetch) — and only after waiting for the scorer
    (the models' inference). The end-of-tick pass is kept."""
    src = Path("src/pipeline.py").read_text(encoding="utf-8")
    body = src[src.index("def run_pipeline("):]
    i_launch = body.index("_sel_short.launch(start)")
    i_marks = body.index("_live_marks_and_exits()")
    i_pool = body.index('with ThreadPoolExecutor(max_workers=14, thread_name_prefix="pipeline") as pool:')
    i_entries = body.index("live_broker_report = _live_entries_and_sync(run_id, _sel_handle, f_borrow)")
    i_collect = body.index("# ── Collect results")
    i_complete = body.index('f"Pipeline complete in')
    assert i_launch < i_marks < i_pool < i_entries < i_collect < i_complete
    helper = src[src.index("def _live_entries_and_sync("):src.index("def run_pipeline(")]
    assert helper.index("_ss.wait(sel_handle)") < helper.index("record_sel_short_trades(") \
        < helper.index("_broker_sync_watchdogged(")
    assert "_merge_broker_reports(live_broker_report, broker_report)" in body
    assert type(settings).model_fields["enable_live_path_first"].default is True


def test_live_entries_wait_for_the_scorer_then_open_then_sync(monkeypatch):
    import src.pipeline as pl
    from src.performance import tracker
    calls = []
    monkeypatch.setattr(ss, "wait", lambda h, timeout=None: calls.append(("wait", h)))
    monkeypatch.setattr(tracker, "record_sel_short_trades", lambda run_id=None: calls.append(("record", run_id)))
    monkeypatch.setattr(pl, "_broker_sync_watchdogged", lambda run_id, a: calls.append(("sync", a)) or {"ok": True})
    rep = pl._live_entries_and_sync("r1", {"proc": None})
    assert [c[0] for c in calls] == ["wait", "record", "sync"] and rep == {"ok": True}
    assert calls[2][1] is None                     # no legacy map on the live path


def test_the_two_syncs_merge_into_one_report():
    import src.pipeline as pl
    a = {"ok": True, "connected": True, "entries_submitted": 2, "orders": [1, 2], "errors": [],
         "drift": [{"ticker": "OLD"}], "account_equity": 100.0}
    b = {"ok": False, "connected": True, "entries_submitted": 1, "orders": [3], "errors": ["x"],
         "drift": [], "account_equity": 101.0}
    m = pl._merge_broker_reports(a, b)
    assert m["entries_submitted"] == 3 and m["orders"] == [1, 2, 3] and m["errors"] == ["x"]
    assert m["drift"] == [] and m["account_equity"] == 101.0 and m["ok"] is False
    assert pl._merge_broker_reports(None, b) is b and pl._merge_broker_reports(a, None) is a


def test_wait_skips_a_prepare_and_journals_a_run_once(monkeypatch, on):
    class Proc:
        def __init__(self, rc):
            self.rc, self.pid = rc, 1

        def wait(self, timeout=None):
            return self.rc

    class Log:
        def close(self):
            pass
    prep = {"proc": Proc(0), "args": ["--prepare", "--day", "2026-09-28"], "started": 0.0, "log": Log()}
    assert ss.wait(prep) is None                                     # never blocks on a prepare
    run = {"proc": Proc(1), "args": ["--run", "--day", "2026-09-28", "--bars", "0"], "started": 0.0, "log": Log()}
    import time as _t
    run["started"] = _t.time()
    assert ss.wait(run) == 1 and ss.wait(run) == 1                   # the second call: cached
    day = datetime.now(ET).date().isoformat()
    lines = (ss.root() / "launches" / f"{day}.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1 and json.loads(lines[0])["rc"] == 1



def test_every_pick_journals_its_confidence_components(on):
    """User 2026-09-29: "Add a few different scores to the pick journal" — the
    candidate confidence components ride every decision record, journal-only."""
    d = date(2026, 9, 28)
    res = _res()
    stand = pd.DataFrame({"n_prior": 0.0, "prior_max": np.nan, "prior_min": np.nan},
                         index=pd.Index(res.ticker, name="ticker"))
    m = ss.select(res, d, 2, stand, [])
    s = np.linspace(1.0, 0.0, 25)
    assert m["ticker"] == "N00" and m["runner_up"] == "N01"
    assert m["run_mean"] == pytest.approx(s.mean()) and m["run_std"] == pytest.approx(pd.Series(s).std())
    assert m["z_in_run"] == pytest.approx((1.0 - s.mean()) / pd.Series(s).std())
    assert m["gap2_z"] == pytest.approx((s[0] - s[1]) / pd.Series(s).std())
    assert m["own_margin_z"] is None                      # no standing: fresh by default
    # the other arm's view: N00 has the LOWEST ATR% (0.0) -> rank 25 of 25
    assert (m["other_arm_score"], m["other_arm_rank"], m["other_arm_n"]) == (0.0, 25, 25)
    assert m["dv20"] == pytest.approx(1e7)
    v = ss.select(res, d, 2, stand, [], arm="vol")
    assert v["ticker"] == "N24" and v["runner_up"] == "N23"
    assert (v["other_arm_score"], v["other_arm_rank"]) == (pytest.approx(0.0), 25)   # the model's last name
    # with a standing, the freshness margin is journaled on the run's scale
    st2 = stand.copy()
    st2.loc["N00", ["n_prior", "prior_max"]] = [50.0, 0.9]
    m2 = ss.select(res, d, 2, st2, [])
    assert m2["decision"] == "short" and m2["own_margin_z"] == pytest.approx((1.0 - 0.9) / pd.Series(s).std())
    # fewer than `sel_short_own_min_history` priors: fresh BY DEFAULT — no margin, even
    # when a prior max exists
    st3 = stand.copy()
    st3.loc["N00", ["n_prior", "prior_max"]] = [5.0, 0.9]
    assert ss.select(res, d, 2, st3, [])["own_margin_z"] is None
    # ties in the other arm rank as "1 + the names strictly above" (competition ranking)
    r4 = res.copy()
    r4.loc[r4.ticker == "N01", "vol"] = 0.0                # ties N00's ATR% of 0.0
    m4 = ss.select(r4, d, 2, stand, [])
    assert (m4["other_arm_score"], m4["other_arm_rank"]) == (0.0, 24)
    # a name the other arm cannot see (no snapshot for the model) journals None, never a crash
    r3 = res.copy()
    r3.loc[r3.ticker == "N24", "status"] = "NO_SNAPSHOT"
    v3 = ss.select(r3, d, 2, stand, [], arm="vol")
    assert v3["ticker"] == "N24" and v3["other_arm_score"] is None and v3["other_arm_n"] == 24
    json.dumps(v3)                                        # the journal line stays valid JSON



# ── the VOL arm's confidence: journal-only (its filter was removed 2026-09-30) ───────────
# Parity vectors: 2026 holdout vol trades from the evaluation (scratchpad
# confidence_test.py, corrected run) — the journal fields they map to and the
# confidence the evaluation computed; the journaled value must reproduce it.
_CONF_VECTORS = json.loads(r'''[{"bar_of_day": 0, "atr_pct": 34.3614501953125, "own_margin_z": 12.001483924356558, "gap2_z": 28.84054575703233, "conf": 0.11979166666666667, "kept": false}, {"bar_of_day": 0, "atr_pct": 25.583484649658203, "own_margin_z": null, "gap2_z": 14.096891527765111, "conf": 0.20138888888888887, "kept": false}, {"bar_of_day": 2, "atr_pct": 19.170337677001953, "own_margin_z": 6.954066683119714, "gap2_z": 2.858034614498714, "conf": 0.31158088235294124, "kept": false}, {"bar_of_day": 0, "atr_pct": 12.30855655670166, "own_margin_z": 7.81638741171217, "gap2_z": 0.9231890982833504, "conf": 0.43137254901960786, "kept": true}, {"bar_of_day": 1, "atr_pct": 13.654885292053223, "own_margin_z": null, "gap2_z": 4.25958471279632, "conf": 0.4444444444444445, "kept": true}, {"bar_of_day": 0, "atr_pct": 14.226902961730957, "own_margin_z": null, "gap2_z": 1.38232513443438, "conf": 0.5, "kept": true}, {"bar_of_day": 0, "atr_pct": 11.155540466308594, "own_margin_z": null, "gap2_z": 1.3754694102344553, "conf": 0.576388888888889, "kept": true}, {"bar_of_day": 10, "atr_pct": 15.341385841369629, "own_margin_z": null, "gap2_z": 2.0957780023579775, "conf": 0.6111111111111112, "kept": true}, {"bar_of_day": 0, "atr_pct": 8.067153930664062, "own_margin_z": null, "gap2_z": 1.1772758922000603, "conf": 0.6666666666666666, "kept": true}, {"bar_of_day": 8, "atr_pct": 10.789778709411621, "own_margin_z": null, "gap2_z": 1.3930382470837444, "conf": 0.7291666666666666, "kept": true}, {"bar_of_day": 4, "atr_pct": 9.250486373901367, "own_margin_z": null, "gap2_z": 0.6058694564532859, "conf": 0.7847222222222222, "kept": true}]''')


def test_vol_confidence_reproduces_the_evaluation_exactly():
    rule = ss.conf_rule_vol()
    assert rule and rule["arm"] == "vol" and abs(rule["cut"] - 0.405093) < 1e-6
    assert [i["field"] for i in rule["inputs"]] == ["bar_of_day", "atr_pct", "own_margin_z", "gap2_z"]
    for v in _CONF_VECTORS:
        got = ss.vol_confidence(v)
        assert got == pytest.approx(v["conf"], abs=1e-12), v
        assert (got >= rule["cut"]) is v["kept"]
    assert ss.vol_confidence({}) is None                  # no input -> no confidence


def test_the_vol_arm_never_filters_on_confidence_but_journals_it(on, monkeypatch):
    """The bottom-third confidence filter was removed 2026-09-30 (user: "Remove the
    confidence filter in live"): a vol pick trades whatever its confidence — the lowest,
    the highest or unknown — and the confidence is still journaled for the record."""
    d = date(2026, 9, 28)
    res = _res()
    stand = pd.DataFrame({"n_prior": 0.0, "prior_max": np.nan, "prior_min": np.nan},
                         index=pd.Index(res.ticker, name="ticker"))
    v = ss.select(res, d, 2, stand, [], arm="vol")
    assert v["decision"] == "short" and v["confidence"] is not None and "confidence_cut" not in v
    json.dumps(v)
    for c in (0.0, 0.10, 0.90, None):
        monkeypatch.setattr(ss, "vol_confidence", lambda rec, c=c: c)
        r = ss.select(res, d, 2, stand, [], arm="vol")
        assert r["decision"] == "short" and r["confidence"] == c and r["target"] == pytest.approx(20.0 - ss.give_back("vol") * 10.0), c
    assert not hasattr(settings, "enable_sel_short_vol_conf_filter")
    # the other gates still decide: a crowded pick stays crowded; the MODEL arm journals no confidence
    monkeypatch.setattr(ss, "vol_confidence", lambda rec: 0.10)
    assert ss.select(res.assign(dtc=5.0), d, 2, stand, [], arm="vol")["decision"] == "crowded"
    m = ss.select(res, d, 2, stand, [])
    assert m["decision"] == "short" and "confidence" not in m


# ── the vol arm's extra names: exchange-traded products (2026-10-02) ─────────

def test_a_vol_only_name_can_be_the_vol_pick_never_the_models(on):
    """User directive 2026-10-02 ("Expand to the ETFs and leveraged ETFs"): the
    scorer gives a name outside the model's training set its ATR% and days to
    cover but no model score (status VOL_ONLY) — the vol arm can pick it, the
    model arm cannot."""
    d = date(2026, 9, 28)
    res = _res()
    res.loc[res.ticker == "N24", ["status", "score"]] = ["VOL_ONLY", np.nan]     # the most volatile name
    stand = pd.DataFrame({"n_prior": 0.0, "prior_max": np.nan, "prior_min": np.nan},
                         index=pd.Index(res.ticker, name="ticker"))
    assert ss.select(res, d, 2, stand, [], arm="vol")["ticker"] == "N24"
    m = ss.select(res, d, 2, stand, [])
    assert m["ticker"] == "N00" and m["n_scored"] == 24


def test_snapshot_coverage_is_judged_on_the_models_names_only(monkeypatch, on):
    meta = {"tickers": ["AAA", "BBB"]}
    uni = {"AAA": 1e7, "BBB": 1e7, "ETF1": 1e7, "ETF2": 1e7}
    assert ss.model_universe(uni, meta) == ["AAA", "BBB"]
    assert ss.model_universe(uni, {}) == sorted(uni)                     # no list: every name is the model's
    from src.data import deep as _deep
    monkeypatch.setattr(_deep, "deep_universe", lambda refresh=False: ["AAA", "BBB", "ETF1", "ETF2"])
    assert ss.vol_extra_names(meta["tickers"]) == ["ETF1", "ETF2"]


def test_a_merge_backfill_adds_names_and_keeps_every_other_row(monkeypatch, on):
    """Seeding the ETF arm's history for newly added products adds their rows and
    keeps every other row of the day; the vol arm's history never takes a name
    outside the model's universe."""
    d = date(2026, 9, 28)
    ss._write_pickle(pd.DataFrame({"bar": [0, 0, 1], "ticker": ["UVIX", "SOXL", "UVIX"], "score": [1.0, 2.0, 3.0]}),
                     ss.scores_path(d, "etf"))
    dn = int((pd.Timestamp(d) - pd.Timestamp("1970-01-01")).days)
    from src.analysis import ml30
    nb = len(ml30.base_features())

    def fake_one(job):
        return {"tk": job[0], "dn": np.array([dn, dn]), "bar": np.array([0, 1]), "px": np.array([20.0, 20.0]),
                "dv20": np.array([1e7, 1e7]), "X": np.full((2, nb), 7.0), "D": np.zeros((2, 0))}
    monkeypatch.setattr(ss, "_backfill_one", fake_one)
    monkeypatch.setattr(ss, "load_model", lambda: (None, {"tickers": ["AAA", "UVIX", "SOXL"], "features": []}))

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
    from src.analysis import deep_features as dfe
    monkeypatch.setattr(dfe, "MarketTables", lambda tks: type("T", (), {"slices": lambda self, t: {}})())
    ss.backfill_days([d], tickers=["ETF1"], arms=("etf", "vol"), merge=True)
    got = pd.read_pickle(ss.scores_path(d, "etf")).sort_values(["bar", "ticker"]).reset_index(drop=True)
    assert got[["bar", "ticker"]].values.tolist() == [[0, "ETF1"], [0, "SOXL"], [0, "UVIX"], [1, "ETF1"], [1, "UVIX"]]
    assert got.loc[got.ticker == "SOXL", "score"].tolist() == [2.0]
    vol_p = ss.scores_path(d, "vol")
    assert not vol_p.exists() or "ETF1" not in set(pd.read_pickle(vol_p)["ticker"])


def test_each_arm_ranks_its_own_universe(monkeypatch, on):
    """The model and vol arms rank the model's names (never a VOL_ONLY product);
    the ETF arm ranks the exchange-traded products — the ones in the model's
    universe included."""
    monkeypatch.setattr(ss, "load_model", lambda: (None, {"tickers": ["AAA", "UVIX"]}))
    monkeypatch.setitem(ss._TYPES, "t", {"AAA": "CS", "UVIX": "ETF", "NEWETF": "ETF"})
    res = pd.DataFrame({"ticker": ["AAA", "UVIX", "NEWETF"], "status": ["OK", "OK", "VOL_ONLY"],
                        "score": [1.0, 0.5, np.nan], "vol": [1.0, 2.0, 3.0]})
    assert ss.arm_rows(res, "vol")["ticker"].tolist() == ["AAA", "UVIX"]
    assert ss.arm_rows(res, "model")["ticker"].tolist() == ["AAA", "UVIX"]
    assert ss.arm_rows(res, "etf")["ticker"].tolist() == ["UVIX", "NEWETF"]
    assert ss.low_rvol(1.0, "etf") and not ss.low_rvol(1.0, "model")      # the ETF arm keeps the volume filter


def test_the_etf_arm_journals_its_own_pick_beside_the_vol_arm(monkeypatch, on):
    """With the ETF arm on, one run journals three decisions: the model's, the vol
    arm's over the model's names, and the ETF arm's over the products — the most
    volatile product is the ETF arm's pick even when a stock is more volatile."""
    monkeypatch.setattr(settings, "enable_sel_short_vol", True)
    monkeypatch.setattr(settings, "enable_sel_short_etf", True)
    monkeypatch.setattr(settings, "enable_sel_short_vol_rvol_filter", False)
    d = date(2026, 9, 28)
    res = _res()
    names = list(res.ticker)
    monkeypatch.setattr(ss, "load_model", lambda: (None, {"tickers": names}))
    monkeypatch.setitem(ss._TYPES, "t", {"N20": "ETF", "N21": "ETF", "N22": "ETF", "N23": "ETF"})   # 4 products
    monkeypatch.setattr(settings, "sel_short_min_run_rows", 3)
    out = ss.decide(res.assign(dtc=0.5, rvol=2.0, pre5_stale=False), d, 2)
    picks = {r["arm"]: r for r in ss.read_picks(d)}
    assert set(picks) == {"model", "vol", "etf"}
    assert picks["vol"]["ticker"] == "N24" and picks["etf"]["ticker"] == "N23"
    assert picks["etf"]["decision"] == "short" and picks["etf"]["n_scored"] == 4
    assert out["arms"]["etf"]["ticker"] == "N23"
    assert len(pd.read_pickle(ss.scores_path(d, "etf"))) == 4                # the ETF arm's own history
