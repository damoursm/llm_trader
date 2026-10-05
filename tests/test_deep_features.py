"""Deep features (`src/analysis/deep_features.py`) — the point-in-time rules.

Every model row of session D must see the deep store exactly as it stood at
D 08:30 ET, because that is what the live store holds when the model is
served. These tests build synthetic parts in a temporary store and probe each
publication rule at its boundary, then perturb every row published AFTER the
cutoff and require the snapshot to be unchanged — the adversarial form of the
same guarantee.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.data import deep
from src.analysis import deep_features as F


@pytest.fixture(autouse=True)
def _tmp_store(tmp_path, monkeypatch):
    monkeypatch.setattr(deep, "DEEP_DIR", tmp_path / "deep")
    (tmp_path / "deep").mkdir()
    yield


def _day(s: str) -> int:
    return int(F.to_days([s])[0])


def _utc(et: str) -> pd.Timestamp:
    """An ET wall time -> naive UTC timestamp."""
    return pd.Timestamp(et).tz_localize("America/New_York").tz_convert("UTC").tz_localize(None)


def _part(family: str, tk: str, df: pd.DataFrame) -> None:
    deep.write_parquet(df, deep.DEEP_DIR / family / "parts" / f"{tk}.parquet")


def _rth(days=("2026-09-21", "2026-09-22", "2026-09-23"), px=100.0) -> F.RTH:
    """13 regular-hours bars per session, constant price, 1,000 shares a bar."""
    idx = []
    for d in days:
        for k in range(13):
            idx.append(_utc(f"{d} 09:30") + pd.Timedelta(minutes=30 * k))
    idx = pd.DatetimeIndex(idx)
    close = np.full(len(idx), px); close[13:] = px * 1.01          # sessions 2+ trade 1% higher
    return F.RTH(idx, close, np.full(len(idx), 1000.0))


# ── the clock ────────────────────────────────────────────────────────────────

def test_cutoff_is_0830_eastern_through_dst():
    summer, winter = _day("2026-09-23"), _day("2026-12-01")
    got = F.cutoff_ns(np.array([summer, winter]))
    assert pd.Timestamp(got[0]) == pd.Timestamp("2026-09-23 12:30")      # EDT = UTC-4
    assert pd.Timestamp(got[1]) == pd.Timestamp("2026-12-01 13:30")      # EST = UTC-5


# ── one boundary per publication rule ────────────────────────────────────────

def test_news_counts_only_articles_published_by_the_cutoff():
    tk = "AAA"
    _part("polygon_news", tk, pd.DataFrame({
        "published_utc": [_utc("2026-09-23 08:29"), _utc("2026-09-23 08:31")],
        "sentiment": ["positive", "negative"], "n_tickers": [1, 1]}))
    d = np.array([_day("2026-09-23"), _day("2026-09-24")])
    s = F.ticker_snapshots(tk, days=d, rth=_rth())
    assert s["dp_news_n_1d"].tolist() == [1.0, 1.0]            # 08:31 lands in the NEXT session, 08:29 ages out
    assert s["dp_news_sent_3d"].iloc[0] == 1.0                 # only the positive one is visible on the 23rd
    assert s["dp_news_sent_3d"].iloc[1] == 0.0                 # both visible on the 24th


def test_sec_filing_accepted_after_the_cutoff_waits_a_session():
    tk = "AAA"
    _part("sec_filings", tk, pd.DataFrame({
        "accession": ["a1", "a2"], "form": ["8-K", "8-K"], "items": ["2.02,9.01", "8.01"],
        "acceptance": [_utc("2026-09-22 16:05"), _utc("2026-09-23 09:00")],
        "filing_date": ["2026-09-22", "2026-09-23"]}))
    d = np.array([_day("2026-09-23"), _day("2026-09-24")])
    s = F.ticker_snapshots(tk, days=d, rth=_rth())
    assert s["dp_sec_n_8k_30d"].tolist() == [1.0, 2.0]
    assert s["dp_sec_d_earn"].iloc[0] == pytest.approx((16 * 60 + 25) / 1440)   # 16:05 -> 08:30 next day
    # the release's anchor is the last close known at 16:05 on the 22nd
    assert s["_earn_anchor"].iloc[0] == pytest.approx(101.0)


def test_insider_filing_is_visible_the_day_after_its_filing_date():
    tk = "AAA"
    ins = pd.DataFrame({"filing_date": ["2026-09-22"], "trans_code": ["P"], "notional": [5e5],
                        "is_officer": [True], "is_director": [False], "owner_cik": ["o1"]})
    d = np.array([_day("2026-09-22"), _day("2026-09-23")])
    s = F.ticker_snapshots(tk, days=d, rth=_rth(), slices={"insider": ins})
    assert s["dp_ins_buy_n_30d"].tolist() == [0.0, 1.0]
    assert s["dp_ins_nbuyers_90d"].tolist() == [0.0, 1.0]


def test_short_interest_is_visible_ten_business_days_after_settlement():
    tk = "AAA"
    _part("short_interest", tk, pd.DataFrame({"settlement_date": ["2026-08-31"], "short_interest": [1e6],
                                              "days_to_cover": [2.5], "avg_daily_volume": [4e5]}))
    known = np.busday_offset(np.datetime64("2026-08-31"), F.LAG_DAYS["short_interest_bdays"],
                             roll="forward").astype(np.int64)
    d = np.array([known - 1, known])
    s = F.ticker_snapshots(tk, days=d, rth=_rth())
    assert np.isnan(s["dp_si_dtc"].iloc[0]) and s["dp_si_dtc"].iloc[1] == 2.5


def test_premarket_uses_only_bars_ended_by_the_cutoff():
    tk = "AAA"
    rth = _rth()
    bars = pd.DataFrame({
        "ts": [_utc("2026-09-23 07:30"), _utc("2026-09-23 08:00"), _utc("2026-09-23 08:30")],
        "high": [102.0, 103.0, 150.0], "low": [101.0, 102.0, 90.0], "close": [102.0, 103.0, 150.0],
        "volume": [10.0, 10.0, 10.0], "session": ["pre", "pre", "pre"]})
    _part("bars30m_full", tk, bars)
    s = F.ticker_snapshots(tk, days=np.array([_day("2026-09-23")]), rth=rth)
    # prev close = the 22nd's last regular-hours close (101); the 08:30-09:00 bar is not known at 08:30
    assert s["dp_xh_pre_ret"].iloc[0] == pytest.approx(103.0 / 101.0 - 1.0)
    assert s["_xh_pre_last"].iloc[0] == 103.0
    assert s["dp_xh_pre_range"].iloc[0] == pytest.approx((103.0 - 101.0) / 101.0)


def test_preopen_snapshot_equals_the_one_built_with_the_sessions_bars():
    """At 08:30 on D the grid holds no bar of D. The snapshot built then must be
    the one a grid holding D's bars gives (the training rows' construction) —
    before 2026-09-26 it found no position for D and every price-dependent
    feature was missing. The fixtures above hand the builder D's own bars, which
    is how that shipped."""
    tk = "AAA"
    days = [d.strftime("%Y-%m-%d") for d in pd.bdate_range("2026-08-03", "2026-09-23")
            if d.strftime("%Y-%m-%d") != "2026-09-07"]                      # Labor Day
    full = _rth(tuple(days))
    pre_grid = _rth(tuple(days[:-1]))
    bars = pd.DataFrame({
        "ts": [_utc("2026-09-22 16:30"), _utc("2026-09-23 07:30"), _utc("2026-09-23 08:00")],
        "high": [101.5, 102.0, 103.0], "low": [101.0, 101.0, 102.0], "close": [101.2, 102.0, 103.0],
        "volume": [50.0, 10.0, 10.0], "session": ["post", "pre", "pre"]})
    _part("bars30m_full", tk, bars)
    ins = pd.DataFrame({"filing_date": ["2026-09-01"], "trans_code": ["P"], "is_officer": [True],
                        "is_director": [False], "notional": [5e5], "owner_cik": ["1"]})
    D = np.array([_day("2026-09-23")])
    a = F.ticker_snapshots(tk, days=D, rth=pre_grid, slices={"insider": ins}).iloc[0]
    b = F.ticker_snapshots(tk, days=D, rth=full, slices={"insider": ins}).iloc[0]
    for c in F.SNAPSHOT_FEATURES + F.ANCHOR_COLUMNS:
        assert (np.isnan(a[c]) and np.isnan(b[c])) or a[c] == pytest.approx(b[c], rel=1e-12), c
    assert a["_prev_close"] == pytest.approx(101.0)
    assert np.isfinite(a["dp_xh_pre_ret"]) and np.isfinite(a["dp_xh_post_ret"])
    assert np.isfinite(a["dp_ins_buy_rel_90d"])                              # needs the trailing $ volume
    # a grid that is BEHIND (ends two sessions before D) is not stretched over the gap
    behind = F.ticker_snapshots(tk, days=D, rth=_rth(tuple(days[:-2])), slices={"insider": ins}).iloc[0]
    assert np.isnan(behind["_prev_close"]) and np.isnan(behind["dp_xh_pre_ret"])


def test_analyst_action_is_visible_the_day_after_its_grade_date():
    tk = "AAA"
    _part("yf_analyst", tk, pd.DataFrame({"grade_date": [pd.Timestamp("2026-09-22 13:00")], "action": ["up"],
                                          "pt_current": [120.0], "pt_prior": [100.0]}))
    d = np.array([_day("2026-09-22"), _day("2026-09-23")])
    s = F.ticker_snapshots(tk, days=d, rth=_rth())
    assert s["dp_an_net_30d"].tolist() == [0.0, 1.0]
    assert s["dp_an_pt_chg"].iloc[1] == pytest.approx(0.2)


# ── the adversarial probe: nothing published after the cutoff may matter ────

def test_rows_published_after_the_cutoff_cannot_change_the_snapshot():
    tk = "AAA"
    D = _day("2026-09-23"); cut = pd.Timestamp(F.cutoff_ns(np.array([D]))[0])
    rng = np.random.default_rng(0)
    times = [cut - pd.Timedelta(hours=h) for h in rng.uniform(1, 400, 40)]
    later = [cut + pd.Timedelta(hours=h) for h in rng.uniform(0.01, 100, 40)]

    def build(future_sentiment: str, future_form: str):
        _part("polygon_news", tk, pd.DataFrame({
            "published_utc": times + later,
            "sentiment": ["positive"] * 40 + [future_sentiment] * 40, "n_tickers": [2] * 80}))
        _part("sec_filings", tk, pd.DataFrame({
            "accession": [f"a{i}" for i in range(80)], "items": ["2.02"] * 80,
            "form": ["8-K"] * 40 + [future_form] * 40, "acceptance": times + later,
            "filing_date": [str(t.date()) for t in times + later]}))
        return F.ticker_snapshots(tk, days=np.array([D]), rth=_rth())

    a = build("negative", "8-K")
    b = build("positive", "10-K")
    pd.testing.assert_frame_equal(a, b)


def test_earnings_surprise_waits_for_the_nightly_refetch():
    """A 07:00 ET release is before the 08:30 cutoff, but yfinance is only
    re-fetched nightly — the surprise must wait for the next day's snapshot,
    while the scheduled date itself is known in advance."""
    tk = "AAA"
    _part("yf_earnings", tk, pd.DataFrame({"event_ts": [_utc("2026-09-23 07:00"), _utc("2026-12-01 07:00")],
                                           "surprise_pct": [12.0, np.nan], "eps_reported": [1.1, np.nan]}))
    d = np.array([_day("2026-09-22"), _day("2026-09-23"), _day("2026-09-24")])
    s = F.ticker_snapshots(tk, days=d, rth=_rth())
    assert np.isnan(s["dp_earn_surprise"].iloc[0]) and np.isnan(s["dp_earn_surprise"].iloc[1])
    assert s["dp_earn_surprise"].iloc[2] == 12.0
    assert s["dp_earn_d_next"].iloc[0] == pytest.approx(22.5 / 24 + 0.0, abs=1.0)   # the 23rd, known ahead


def test_wiki_lags_by_the_refresh_cadence():
    tk = "AAA"
    days_ = pd.date_range("2026-07-01", "2026-09-22").strftime("%Y-%m-%d")
    views = np.full(len(days_), 100.0); views[-1] = 10_000.0             # a spike on the 22nd
    _part("wiki", tk, pd.DataFrame({"date": days_, "views": views}))
    d = np.array([_day("2026-09-23"), _day("2026-09-26")])
    s = F.ticker_snapshots(tk, days=d, rth=_rth())
    assert abs(s["dp_wiki_1d_rel"].iloc[0]) < 1e-9                        # the spike is not in the store yet
    assert s["dp_wiki_1d_rel"].iloc[1] > 4.0                              # 22nd + 4 days


def test_bar_features_move_with_the_bar_close():
    snap = pd.DataFrame({"_xh_pre_last": [100.0], "_earn_anchor": [50.0], "_an_anchor": [np.nan],
                         "_ins_anchor": [200.0], "_prev_close": [99.0]})
    bf = F.bar_features(snap, np.array([110.0]), np.array([3]))
    assert bf["dp_bar_index"][0] == 3
    assert bf["dp_xh_rth_vs_pre"][0] == pytest.approx(0.10)
    assert bf["dp_earn_ret_since"][0] == pytest.approx(1.20)
    assert np.isnan(bf["dp_an_ret_since"][0])
    assert bf["dp_ins_ret_since"][0] == pytest.approx(-0.45)


def test_session_snapshot_roundtrip_and_serving_vector(monkeypatch):
    """build -> load -> serve: the scorer's path. A ticker the snapshot does not
    carry gets NaN snapshot features and the session's market context."""
    from src.data.deep import sec
    monkeypatch.setattr(sec, "cik_map", lambda: {})
    tk = "AAA"
    _part("polygon_news", tk, pd.DataFrame({"published_utc": [_utc("2026-09-23 07:00")],
                                            "sentiment": ["positive"], "n_tickers": [1]}))
    monkeypatch.setattr(F.RTH, "for_ticker", classmethod(lambda cls, t: _rth()))
    D = _day("2026-09-23")
    n = F.build_session_snapshot(D, tickers=[tk, "BBB"], workers=1)
    assert n == 2 and F.snapshot_path(D).exists()
    snap = F.load_session_snapshot(D)
    assert set(snap) == {"AAA", "BBB"}
    v = F.serving_vector("AAA", D, close=101.0, bar_index=4, snapshot=snap)
    assert v["dp_news_n_1d"] == 1.0 and v["dp_bar_index"] == 4.0
    assert set(v) == set(F.DEEP_FEATURES)
    miss = F.serving_vector("ZZZ", D, close=10.0, bar_index=0, snapshot=snap)
    assert np.isnan(miss["dp_news_n_1d"]) and set(miss) == set(F.DEEP_FEATURES)


def test_feature_lists_are_disjoint_and_complete():
    assert len(set(F.DEEP_FEATURES)) == len(F.DEEP_FEATURES)
    grouped = [c for g in F.FEATURE_GROUPS.values() for c in g]
    assert sorted(grouped) == sorted(F.DEEP_FEATURES)
    assert all(c.startswith("dp_") for c in F.DEEP_FEATURES)
    assert not set(F.ANCHOR_COLUMNS) & set(F.DEEP_FEATURES)


def test_ftd_and_13f_file_names_parse_to_their_period_ends():
    assert F._ftd_period_end("cnsfails202608a") == _day("2026-08-15")
    assert F._ftd_period_end("cnsfails202608b") == _day("2026-08-31")
    assert F._13f_range_end("01jun2026-31aug2026_form13f") == _day("2026-08-31")
    assert F._13f_range_end("2013q2_form13f") == _day("2013-06-30")
    assert F._ftd_period_end("junk") is None and F._13f_range_end("junk") is None


# ── the scorer's missing-snapshot fallback ───────────────────────────────────

def test_recent_session_days_are_today_and_the_previous_session():
    assert F.recent_session_days(today=_day("2024-10-07")) == (_day("2024-10-07"), _day("2024-10-04"))  # Mon -> Fri
    assert F.recent_session_days(today=_day("2024-10-06")) == (_day("2024-10-06"), _day("2024-10-04"))  # Sun (overnight)
    assert F.previous_session_day(_day("2024-12-26")) == _day("2024-12-24")                             # Christmas


def test_snapshot_fallback_is_globally_single_flight_and_recent_only(monkeypatch):
    """Measured 2026-09-23: keyed per DAY, one scoring pass over names whose
    tick caches had stopped on older sessions launched eight concurrent
    whole-market builds. One build at a time, and only for a live session."""
    import subprocess
    import threading
    mon, fri = _day("2024-10-07"), _day("2024-10-04")
    monkeypatch.setattr(F, "recent_session_days", lambda today=None: (mon, fri))
    monkeypatch.setattr(F, "_BUILD_THREAD", None)
    gate, started = threading.Event(), []

    def fake_run(args, **kw):
        started.append(args[-1])
        gate.wait(5)
        return subprocess.CompletedProcess(args, 0)

    monkeypatch.setattr(subprocess, "run", fake_run)
    assert F.trigger_snapshot_build(_day("2024-09-30")) is False     # an old session: never built on the tick path
    assert F.trigger_snapshot_build(mon) is True
    assert F.trigger_snapshot_build(fri) is False                    # another day while a build runs: refused
    assert F.trigger_snapshot_build(mon) is False
    gate.set()
    F._BUILD_THREAD.join(5)
    assert started == ["2024-10-07"]
    assert F.trigger_snapshot_build(fri) is True                     # free again once it finished
    F._BUILD_THREAD.join(5)
    assert started == ["2024-10-07", "2024-10-04"]


def _stub_30m_scorer(monkeypatch, idx):
    from src.analysis import ml_dataset, pivot_target
    s = pd.Series(np.linspace(10.0, 11.0, len(idx)), index=idx)
    monkeypatch.setattr(ml_dataset, "hlc_30m", lambda tk: (idx, s, s, s, s))
    monkeypatch.setattr(ml_dataset, "ticker_feature_frame", lambda tk, hlc=None: pd.DataFrame([{"f": 1.0}]))
    monkeypatch.setattr(pivot_target, "leg_feature_rows", lambda *a, **k: [{}])
    return {"config": {"pivot_basis": pivot_target.pivot_basis()}, "features": ["f", "dp_x"], "model": None}


def test_scorer_never_builds_a_snapshot_for_a_stale_session(monkeypatch):
    """A name whose tick cache stopped on an older session asks for THAT
    session's snapshot: abstain, never launch a whole-market build."""
    from src.signals import ml_model
    idx = pd.date_range("2024-09-03 13:30", periods=400, freq="30min")          # naive UTC, long past
    art = _stub_30m_scorer(monkeypatch, idx)
    calls = []
    monkeypatch.setattr(F, "load_session_snapshot", lambda d: None)
    monkeypatch.setattr(F, "trigger_snapshot_build", lambda d: calls.append(d) or True)
    ml_model.reset_caches()
    assert ml_model._score_30m("ZZDEEP", art) == (0.0, "DEEP_STALE")
    assert calls == []
    # the same bar on a LIVE session: abstain and build it
    sday = int(F.session_days(idx[-1:])[0])
    monkeypatch.setattr(F, "recent_session_days", lambda today=None: (sday, sday - 1))
    ml_model.reset_caches()
    assert ml_model._score_30m("ZZDEEP", art) == (0.0, "DEEP_PENDING")
    assert calls == [sday]
    ml_model.reset_caches()


# ── Reg SHO threshold lists and IBKR borrow (2026-10-02) ────────────────────

def _rs_tables(on_dates, cal_dates):
    """Write the two Reg SHO tables: `on_dates` for AAA, every list complete on `cal_dates`."""
    from src.data.deep.regsho import MARKETS
    deep.write_parquet(pd.DataFrame({"date": list(on_dates), "market": "nasdaq", "symbol": "AAA",
                                     "symbol_raw": "AAA", "name": "A"}), deep.DEEP_DIR / "regsho.parquet")
    cal = pd.DataFrame([{"date": d, "market": m, "rows": 1} for d in cal_dates for m in MARKETS])
    deep.write_parquet(cal, deep.DEEP_DIR / "regsho_calendar.parquet")


def _rs_snap(days):
    t = F.MarketTables(["AAA"])
    return F.ticker_snapshots("AAA", days=np.array([_day(d) for d in days]), rth=_rth(), slices=t.slices("AAA"))


def test_regsho_list_of_a_day_is_visible_from_the_next_session():
    cal = ["2026-09-16", "2026-09-17", "2026-09-18", "2026-09-21", "2026-09-22"]
    _rs_tables(["2026-09-18", "2026-09-21", "2026-09-22"], cal)
    s = _rs_snap(["2026-09-22", "2026-09-23"])
    # at 09-22 08:30 the latest list is 09-21's (09-22's is published that night)
    assert s["dp_rs_on"].tolist() == [1.0, 1.0]
    assert s["dp_rs_streak"].tolist() == [2.0, 3.0]
    assert s["dp_rs_n_60"].tolist() == [2.0, 3.0]
    assert s["dp_rs_d_last"].tolist() == [1.0, 1.0]


def test_regsho_never_listed_name_reads_zero_and_a_stale_calendar_reads_missing():
    _rs_tables([], ["2026-09-21", "2026-09-22"])
    s = _rs_snap(["2026-09-23", "2026-10-05"])
    assert s["dp_rs_on"].iloc[0] == 0.0 and s["dp_rs_streak"].iloc[0] == 0.0
    assert np.isnan(s["dp_rs_d_last"].iloc[0])
    assert s[["dp_rs_on", "dp_rs_streak", "dp_rs_n_60"]].iloc[1].isna().all()      # newest list 13 days old


def test_regsho_day_with_a_market_missing_is_not_a_list_day():
    from src.data.deep.regsho import MARKETS
    _rs_tables(["2026-09-21"], ["2026-09-21"])
    cal = deep.read_parquet(deep.DEEP_DIR / "regsho_calendar.parquet")
    extra = pd.DataFrame([{"date": "2026-09-22", "market": m, "rows": 0} for m in MARKETS[:-1]])
    deep.write_parquet(pd.concat([cal, extra]), deep.DEEP_DIR / "regsho_calendar.parquet")
    s = _rs_snap(["2026-09-23"])
    assert s["dp_rs_on"].iloc[0] == 1.0          # 09-22 is incomplete: 09-21's list still stands


def _bw_table(rows):
    deep.write_parquet(pd.DataFrame(rows, columns=["ticker", "date", "source", "available", "fee"]),
                       deep.DEEP_DIR / "borrow_daily.parquet")


def _bw_snap(days):
    t = F.MarketTables(["AAA"])
    return F.ticker_snapshots("AAA", days=np.array([_day(d) for d in days]), rth=_rth(), slices=t.slices("AAA"))


def test_borrow_row_of_a_day_is_visible_from_the_next_session():
    dates = pd.bdate_range("2026-09-08", "2026-09-22").strftime("%Y-%m-%d")
    fee = np.linspace(1.0, 11.0, len(dates)); fee[-1] = 50.0
    avail = np.full(len(dates), 500_000.0); avail[-1] = 50.0
    _bw_table([("AAA", d, "ibd", a, f) for d, a, f in zip(dates, avail, fee)])
    s = _bw_snap(["2026-09-22", "2026-09-23"])
    assert s["dp_bw_fee"].iloc[0] == pytest.approx(fee[-2])               # 09-22's row waits a day
    assert s["dp_bw_fee"].iloc[1] == pytest.approx(50.0)
    assert s["dp_bw_htb"].tolist() == [0.0, 1.0]                          # 50 shares x ~$101 < $10k
    assert s["dp_bw_avail_usd"].iloc[1] == pytest.approx(np.log10(1 + 50 * 101.0), rel=1e-6)
    assert s["dp_bw_fee_chg5"].iloc[1] == pytest.approx(np.log((50.0 + 0.25) / (fee[-6] + 0.25)))
    assert s["dp_bw_fee_max20"].iloc[1] == pytest.approx(50.0)
    assert s["dp_bw_age"].iloc[1] == pytest.approx(8.5 / 24)              # midnight -> 08:30


def test_borrow_rows_older_than_ten_days_read_missing():
    _bw_table([("AAA", "2026-09-01", "ibd", 1e6, 0.3)])
    s = _bw_snap(["2026-09-23"])
    assert s[["dp_bw_fee", "dp_bw_htb", "dp_bw_age"]].iloc[0].isna().all()


def test_regsho_and_borrow_rows_dated_on_or_after_the_session_cannot_change_it():
    """The adversarial form for the two families: rewrite every row dated D or later
    (published after D's cutoff) — the snapshot of D must not move."""
    D = "2026-09-23"
    past = pd.bdate_range("2026-08-03", "2026-09-22").strftime("%Y-%m-%d").tolist()
    future = pd.bdate_range("2026-09-23", "2026-10-09").strftime("%Y-%m-%d").tolist()
    rng = np.random.default_rng(3)

    def build(seed):
        r = np.random.default_rng(seed)
        on = [d for d in past if rng.random() < 0.4] + [d for d in future if r.random() < 0.7]
        _rs_tables(on, past + future)
        rows = [("AAA", d, "ibd", 1e5 * (1 + (i % 7)), 1.0 + i % 5) for i, d in enumerate(past)]
        rows += [("AAA", d, "ibkr", r.uniform(0, 1e7), r.uniform(0, 900)) for d in future]
        _bw_table(rows)
        rng.bit_generator.state = np.random.default_rng(3).bit_generator.state     # same past every build
        return _rs_snap([D])
    a, b = build(11), build(12)
    pd.testing.assert_frame_equal(a, b)


def test_regsho_and_borrow_of_a_session_are_the_same_in_a_batch_and_alone():
    """Train / serve parity for the two families: the training rows compute every
    session of a name in one batch (`ml30`), the live pre-open snapshot one session
    alone — the session's features must be identical either way."""
    past = pd.bdate_range("2026-07-01", "2026-09-22").strftime("%Y-%m-%d").tolist()
    rng = np.random.default_rng(5)
    _rs_tables([d for d in past if rng.random() < 0.35], past)
    _bw_table([("AAA", d, "ibd", float(rng.integers(0, 2_000_000)), float(rng.uniform(0, 300))) for d in past])
    t = F.MarketTables(["AAA"])
    days = [_day(d) for d in ("2026-09-21", "2026-09-22", "2026-09-23")]
    batch = F.ticker_snapshots("AAA", days=np.array(days), rth=_rth(), slices=t.slices("AAA"))
    cols = F.FEATURE_GROUPS["rs"] + F.FEATURE_GROUPS["bw"]
    for d in days:
        alone = F.ticker_snapshots("AAA", days=np.array([d]), rth=_rth(), slices=t.slices("AAA"))
        pd.testing.assert_series_equal(batch.loc[d, cols], alone.iloc[0][cols], check_names=False)
