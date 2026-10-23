"""NO LOOK-AHEAD in the live selection short's decision path (user directive
2026-09-28: "We absolutely cannot have logic that looks into the future").

The backtest's evaluation arrays compute every bar's features on a ticker's WHOLE
30-minute history (`ml30.ticker_rows`: `ticker_feature_frame` + the pivot-leg state
over the full series); the live scorer computes the pick bar's features from the
bars up to that bar (`ml_model.features_30m_from_hlc`). Each guard below fails if
any input at a decision point could see something that happened after it:

* all 85 bar features (incl. the 30-min ATR% the vol arm ranks and the pivot-leg
  state) — backtest == live cut at the bar, and REWRITING every later bar changes
  nothing (a centred window, a back-fill, a whole-series normalisation would);
* the freshness rule — a name's own score history never includes the current
  session;
* the universe's dollar volume — the 20 sessions BEFORE the day, never the day;
* the volatility exit's ATR% — never a bar after the one it is judged on;
* the vol arm's relative-volume filter — the pick bar against the bars BEFORE it,
  never a later one (a past-session run reads a store that holds them);
* the journal's short-sale restriction (Rule 201) at a pick — the day's low through the
  pick bar, never a later bar (2026-10-05);
* the vol arm's added-stocks screen — the 20 sessions BEFORE the day, never the day
  (2026-10-05);
* the vol arm's squeeze cover — the entry carried through the splits executed by the tick's
  day only, never one the data lists for a later day (2026-10-05);
* the simulated account the vol arm is sized from — the money paid in BY the tick, never a
  deposit due later (2026-10-05);
* the corporate-action gap flag that keeps a spin-off out of the vol and ETF rankings — the bar
  and earlier ones only (2026-10-05);
* the borrow fee each open short carries into that account's equity — IBKR's rates, closes and
  charges of the days already over, the file in force for a day not over, never a later day's
  row or a rewrite of one (2026-10-06);
* the model's training rows — only labels CONFIRMED by the fit's cut;
* the dip long book (2026-10-08) — its signal and its exit at session D's open read the closes through
  D-1 only, never a bar of D or later (mutation-checked 2026-10-08);
* the dip book's meta-model filter (2026-10-14, PREREG96's RP33, off until deployed) — every one of its 17 inputs at
  session D (the name's daily bars, breadth, the name's previous signal, SPY's) and so its score read the closes
  through D-1 only; a store or SPY series already holding D and later (rewritten, or firing again) moves nothing
  (mutation-checked 2026-10-14: the name's inputs from the store's last bar, SPY's from its last bar, the previous
  signal searched past D-1 — each fails the guard);
* the ETF dual-momentum book (2026-10-08) — the month's pick reads the funds' closes through the previous
  month's last session and the T-bill rates PUBLISHED by its close (ALFRED vintages), never a bar of the month
  or a later vintage (mutation-checked 2026-10-08: closes uncut, the month's own close, the vintage cut dropped).
* the ledger's split adjustment (2026-10-09, PREREG64) — an open trade and a working dip order are rescaled
  through the splits executed by the tick only, from 04:00 ET of the split's date, never one the data lists for
  a later day or a later hour (mutation-checked 2026-10-09: the 04:00 shift dropped).
* the borrow pressure behind the vol arm's extra shorts (2026-10-10, PREREG71's V6) — session D's inputs read the
  daily bars and IBKR's fee of the sessions BEFORE D, FINRA's report settled 10+ days before, fails-to-deliver settled
  15-45 days before, the Reg SHO lists before D; rewriting every later row changes nothing (mutation-checked
  2026-10-10: the fee read on D itself).
* the turn of the fall and ETF picks (2026-10-10, PREREG76 / 77) — a waiting pick's turn (and the ETF target's
  cancel) at a tick reads only the bars COMPLETED by the tick, inside its window: the state from the whole series
  equals the series cut at the tick, a bar still in progress never counts, and rewriting every later bar changes
  nothing (mutation-checked 2026-10-10: the tick's cut dropped).
* the vol shorts' news-hold SHADOW (2026-10-17, PREREG136) — the decision at an exit counts the headlines published
  after the entry and at least `sel_short_news_hold_lag_minutes` before the exit only: adding or rewriting any
  headline after that cut changes nothing (mutation-checked 2026-10-17: the lag dropped from the window).
* the dip book's earnings-dip demotion (2026-10-17, PREREG137's DEMOTE_E) — session D's flags count the 8-K Item 2.02
  releases ACCEPTED by D's 09:30 only: releases accepted after it, in the store or in EDGAR's answer at the opening
  tick, change no flag, rank or pick (mutation-checked 2026-10-17: the window's 09:30 cut dropped).
* the dip book's own-security series (2026-10-18, `dip_security`: PREREG142's cut and join) — the break test at a new
  security's start and the series it leaves read the sessions through D-1 only: a store holding D and later
  (rewritten) moves no cut, bar or number (mutation-checked 2026-10-18: the series read past D-1 flips the break).
* the ledger's dividends (2026-10-19, AUDIT151) — a live book's trade carries an ex-date's cash only once its
  entitlement instant (20:00 ET of the session before the ex-date) has passed by the tick: a later ex-date's record,
  added or rewritten, moves no mark (mutation-checked 2026-10-19: the instant's cut dropped).
The deep features' point-in-time rules (the 08:30 cutoff, publication lags, rows
published after the cutoff) are pinned in `tests/test_deep_features.py` — the Reg SHO
(`rs`) and IBKR borrow (`bw`) groups there too (a list / borrow row of D is known from
D+1; rows dated on or after a session cannot move it; a session's values are the same
in a training batch and in the live snapshot; mutation-checked 2026-10-02); the
pivot-leg state's confirmation rule in `tests/test_pivot_target.py`; the run-up
base's history-gap guard and completed-bars-only scoring in `tests/test_sel_short.py`.
"""
from __future__ import annotations

from datetime import date, datetime, timedelta, timezone
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import pytest

from src.signals import sel_short as ss

ET = ZoneInfo("America/New_York")
DAY = date(2026, 9, 28)


def _frame(days, seed=5, scale=1.0):
    """Regular-hours 30-min OHLCV (naive-UTC bar starts) for ``days``: a random walk
    with real intraday ranges and volume."""
    rng = np.random.default_rng(seed)
    idx = [pd.Timestamp(datetime(d.year, d.month, d.day, 9, 30, tzinfo=ET) + timedelta(minutes=30 * k))
           .tz_convert("UTC").tz_localize(None) for d in days for k in range(13)]
    c = 20.0 * np.exp(np.cumsum(rng.normal(0, 0.012, len(idx))))
    o = np.r_[c[0], c[:-1]]
    h = np.maximum(o, c) * (1 + np.abs(rng.normal(0, 0.006, len(idx))))
    lo = np.minimum(o, c) * (1 - np.abs(rng.normal(0, 0.006, len(idx))))
    v = rng.integers(20_000, 200_000, len(idx)).astype(float)
    return pd.DataFrame({"Open": o * scale, "High": h * scale, "Low": lo * scale, "Close": c * scale,
                         "Volume": v}, index=pd.DatetimeIndex(idx))


def _hlc(df):
    from src.analysis.ml_dataset import hlc_from_frames
    return hlc_from_frames([df])


def _rewrite_after(df, t):
    """Every bar AFTER position ``t`` replaced by a wildly different path."""
    out = df.copy()
    rng = np.random.default_rng(99)
    k = len(out) - (t + 1)
    jump = np.exp(np.cumsum(rng.normal(0.02, 0.08, k)))
    for col in ("Open", "High", "Low", "Close"):
        out.iloc[t + 1:, out.columns.get_loc(col)] = out[col].iloc[t + 1:].to_numpy() * 3.0 * jump
    out.iloc[t + 1:, out.columns.get_loc("Volume")] = out["Volume"].iloc[t + 1:].to_numpy() * 25.0
    return out


def _same(a, b, where):
    a, b = np.asarray(a, float), np.asarray(b, float)
    both_nan = np.isnan(a) & np.isnan(b)
    close = np.isclose(a, b, rtol=1e-4, atol=1e-5) | both_nan
    assert close.all(), f"{where}: {int((~close).sum())} feature(s) differ"


def _arrays_rows(monkeypatch, df):
    """The backtest arrays' feature rows for EVERY bar (`ml30.ticker_rows`, eval block)."""
    from src.analysis import ml30, ml_dataset
    from src.analysis import deep_features as dfe
    hlc = _hlc(df)
    monkeypatch.setattr(ml_dataset, "hlc_30m", lambda tk: hlc)
    first_day = int(dfe.session_days(hlc[0])[0])
    r = ml30.ticker_rows("SYN", 1.0, deep=False, rows="fml", eval_since=first_day)
    assert r is not None and len(r["eval"]["X"]) == len(hlc[0])
    return r["eval"]["X"]


def _live_row(df, t):
    """The live scorer's features at bar ``t``: `features_30m_from_hlc`, cut at the bar's end."""
    from src.analysis import ml30
    from src.signals.ml_model import features_30m_from_hlc
    hlc = _hlc(df)
    end = pd.Timestamp(hlc[0][t]) + pd.Timedelta(minutes=30)
    frame, label = features_30m_from_hlc("SYN", hlc, end)
    assert frame is not None, label
    assert pd.Timestamp(frame["bar_ts"]) == pd.Timestamp(hlc[0][t])
    return np.array([frame["features"].get(f, np.nan) for f in ml30.base_features()], float)


CUTS = (430, 455, 519, 560, 600, 640)          # past the live 400-bar minimum; first, mid and last bars of sessions


def test_the_backtest_features_at_a_bar_equal_the_live_features_cut_at_it(monkeypatch):
    """The arrays (whole history) and the live scorer (history up to the bar) must
    produce the SAME 85 features at every pick bar — the backtest never used a
    feature the live system could not have computed."""
    df = _frame(ss.sessions_before(DAY, 52))
    X = _arrays_rows(monkeypatch, df)
    for t in CUTS:
        _same(X[t], _live_row(df, t), f"bar {t}: arrays vs live")


def test_rewriting_every_later_bar_changes_no_feature(monkeypatch):
    """A feature at bar t must not move when EVERY bar after t is replaced — in the
    live computation and in the backtest arrays."""
    df = _frame(ss.sessions_before(DAY, 52))
    X = _arrays_rows(monkeypatch, df)
    for t in CUTS:
        other = _rewrite_after(df, t)
        _same(_live_row(df, t), _live_row(other, t), f"bar {t}: live, later bars rewritten")
        Xo = _arrays_rows(monkeypatch, other)
        _same(X[t], Xo[t], f"bar {t}: arrays, later bars rewritten")


def test_the_freshness_history_never_includes_the_current_session(tmp_path, monkeypatch):
    """The own-history rule compares a score with the name's scores of the PREVIOUS
    sessions only; a score written for the current session (a later bar of the same
    day) must not raise the bar it is judged against."""
    from config.settings import settings
    from src.analysis.eval_metrics import own_history_standing
    monkeypatch.setattr(settings, "sel_short_dir", str(tmp_path / "sel_short"))
    prev = ss.sessions_before(DAY, 3)
    for i, d in enumerate(prev):
        ss.append_scores(d, pd.DataFrame({"bar": [2], "ticker": ["AAA"], "score": [0.10 + 0.01 * i]}), "model")
    ss.append_scores(DAY, pd.DataFrame({"bar": [5], "ticker": ["AAA"], "score": [9.99]}), "model")
    st = ss.standing(DAY, ["AAA"])
    assert st.loc["AAA", "n_prior"] == 3 and st.loc["AAA", "prior_max"] == pytest.approx(0.12)
    f = pd.DataFrame({"signal_date": [d.isoformat() for d in prev] + [DAY.isoformat()] * 2,
                      "ticker": ["AAA"] * 5, "score": [0.10, 0.11, 0.12, 9.99, 0.5]})
    ref = own_history_standing(f, "score", date="signal_date", ticker="ticker", window_days=30)
    assert ref["prior_max"].iloc[-1] == pytest.approx(0.12) and ref["n_prior"].iloc[-1] == 3


def test_the_universe_dollar_volume_never_reads_the_day_itself(monkeypatch):
    """The day's universe is fixed from the 20 sessions BEFORE it — the day's own
    (possibly huge) volume must not let a name in."""
    from src.data import intraday_store
    hist = _frame(ss.sessions_before(DAY, 25))
    today = _frame([DAY], seed=6)
    today["Volume"] *= 1_000
    monkeypatch.setattr(intraday_store, "load_deep_30m", lambda tk: hist)
    base = ss._dv20("SYN", DAY)
    monkeypatch.setattr(intraday_store, "load_deep_30m", lambda tk: pd.concat([hist, today]))
    assert ss._dv20("SYN", DAY) == pytest.approx(base)


def test_the_volatility_exit_never_reads_a_bar_after_its_own(monkeypatch):
    """The ATR% the volatility exit compares is the one AT the latest completed bar:
    later bars of the session (a spike after it) must not change it."""
    deep = _frame(ss.sessions_before(DAY, 40))
    today = _frame([DAY], seed=8)
    spiked = _rewrite_after(today, 3)
    for frame_today in (today, spiked):
        monkeypatch.setattr(ss, "_deep_and_recent", lambda tk, d, fetch=True, _t=frame_today: (deep, _t))
        ss._ATR_MEMO.clear()
        got = ss.atr_at("SYN", DAY, 3)
        if frame_today is today:
            want = got
    assert got == want and want is not None


def test_the_relative_volume_never_reads_a_bar_after_its_own():
    """The vol arm's relative-volume filter compares the pick bar's volume with the
    bars BEFORE it. Cutting the series at the bar, or rewriting every later bar (25x
    the volume, another price path), must not change it — the scorer of a past session
    reads a deep store that already holds the later bars."""
    df = _frame(ss.sessions_before(DAY, 52))
    full = _hlc(df)
    for t in CUTS:
        want = ss.rvol_at(full[0], full[4], full[0][t])
        assert np.isfinite(want)
        cut = _hlc(df.iloc[: t + 1])
        other = _hlc(_rewrite_after(df, t))
        assert ss.rvol_at(cut[0], cut[4], cut[0][t]) == pytest.approx(want, rel=1e-12), f"bar {t}: cut at the bar"
        assert ss.rvol_at(other[0], other[4], other[0][t]) == pytest.approx(want, rel=1e-12), \
            f"bar {t}: later bars rewritten"


def test_the_short_sale_restriction_never_reads_a_bar_after_its_own():
    """The journal's Rule 201 state at a pick bar (`ssr_state`) reads the day's low THROUGH
    that bar and the two sessions before it. Cutting the series at the bar, or rewriting every
    later bar (another path, or a crash that would trigger the restriction), must not change it."""
    df = _frame(ss.sessions_before(DAY, 4) + [DAY])
    n0 = len(df) - 13                                       # DAY's first bar
    for b in (0, 3, 7, 12):
        t = n0 + b
        full = _hlc(df)
        want = ss.ssr_state(full[0], full[2], full[3], DAY, full[0][t])
        assert want["ssr"] is not None, f"bar {b}"
        crash = df.copy()
        for col in ("Open", "High", "Low", "Close"):
            crash.iloc[t + 1:, crash.columns.get_loc(col)] = crash[col].iloc[t + 1:].to_numpy() * 0.2
        for other_df, why in ((df.iloc[: t + 1], "cut at the bar"), (_rewrite_after(df, t), "later bars rewritten"),
                              (crash, "a crash after the bar")):
            o = _hlc(other_df)
            assert ss.ssr_state(o[0], o[2], o[3], DAY, o[0][t]) == want, f"bar {b}: {why}"


def test_the_added_stocks_screen_never_reads_the_day_or_later(monkeypatch):
    """The vol arm's added-stocks screen (`screen_listings`) judges the dollar-volume floor on
    the 20 sessions BEFORE the day: a name dormant until the day (a 1,000,000x volume spike on
    it, or later) must stay out, and no session from the day on is ever read."""
    asked = []

    def grouped(s):
        asked.append(s)
        rows = [("STEADY", 20.0, 1e6 if s < DAY else 1e9), ("SPIKE", 20.0, 1.0 if s < DAY else 1e9)]
        return pd.DataFrame(rows, columns=["ticker", "close", "volume"])
    monkeypatch.setattr(ss, "grouped_day", grouped)
    monkeypatch.setattr(ss, "ticker_details",
                        lambda tks: {t: {"type": "CS", "list_date": "2026-06-01", "active": True} for t in tks})
    from src.data import deep as _deep
    monkeypatch.setattr(_deep, "deep_universe", lambda refresh=False: [])
    monkeypatch.setattr(ss, "load_model", lambda: (None, {"tickers": []}))
    got = ss.screen_listings(DAY)
    assert [r["ticker"] for r in got] == ["STEADY"] and got[0]["dollar_volume"] == pytest.approx(2e7)
    assert asked and max(asked) < DAY


def test_training_rows_only_use_labels_confirmed_by_the_cut():
    """A model fit on sessions <= cut may only learn labels whose pivot was
    CONFIRMED by then — a row near the cut whose outcome printed later is left out."""
    from src.analysis import sel_models as sm

    class Arr:
        dn = np.array([100, 100, 101, 102, 103])
        conf = np.array([101, 103, 102, 102, 104])          # the day each row's label confirmed
        y = np.array([1.0, 2.0, np.nan, 3.0, 4.0])
        ba = np.array([2, 5, 1, 1, 1])
        tk = np.array([0, 1, 0, 1, 0])
        run = dn * 100
        px = np.full(5, 20.0)
        dv20 = np.full(5, 1e7)

        def tradeable(self):
            return np.ones(5, bool)
    cut = str((pd.Timestamp("1970-01-01") + pd.Timedelta(days=102)).date())
    rows = sm.training_rows(Arr(), cut)
    assert sorted(rows.tolist()) == [0, 3]           # row 1 confirmed after the cut, 2 unlabelled, 4 after it


# ── the V2 model arm's extra inputs (src/signals/sel_v2.py, 2026-10-02) ──────

def _v2_history(df):
    """The history path's per-bar V2 inputs at EVERY bar (`sel_short._v2_extra_rows`: `bar_extras` over the
    whole series, as the training rows were built)."""
    from src.analysis.ml_dataset import ticker_feature_frame
    from src.signals import sel_v2
    hlc = _hlc(df)
    return sel_v2.bar_extras(hlc, ticker_feature_frame("SYN", hlc=hlc)["atr_pct_14"].to_numpy(float))


def _v2_live(df, t):
    """The live scorer's per-bar V2 inputs at bar ``t`` (`sel_short._score_chunk`: the series cut at the bar's
    end, its ATR% series from the same feature frame as the base features)."""
    from src.signals import sel_v2
    from src.signals.ml_model import features_30m_from_hlc
    hlc = _hlc(df)
    end = pd.Timestamp(hlc[0][t]) + pd.Timedelta(minutes=30)
    frame, label = features_30m_from_hlc("SYN", hlc, end, with_series=True)
    assert frame is not None, label
    n = int(frame["n_vis"])
    cut = (hlc[0][:n], hlc[1].iloc[:n], hlc[2].iloc[:n], hlc[3].iloc[:n], hlc[4].iloc[:n])
    return sel_v2.bar_extras(cut, frame["atr_series"])[-1]


def test_v2_bar_inputs_equal_live_and_ignore_every_later_bar():
    """sx_runup5 / sx_atr_own / sx_rvol260: the history's value at a bar == the live value cut at it, and
    neither moves when every later bar is rewritten (the same session's later bars included — the own-ATR
    reference is the PREVIOUS sessions' max, never the current session's)."""
    df = _frame(ss.sessions_before(DAY, 52))
    E = _v2_history(df)
    for t in CUTS:
        live = _v2_live(df, t)
        assert np.isfinite(live).all(), f"bar {t}: {live}"
        _same(E[t], live, f"bar {t}: V2 history vs live")
        other = _rewrite_after(df, t)
        _same(live, _v2_live(other, t), f"bar {t}: V2 live, later bars rewritten")
        _same(E[t], _v2_history(other)[t], f"bar {t}: V2 history, later bars rewritten")


def test_v2_session_inputs_ignore_rows_after_the_cutoff():
    """The per-session V2 inputs are known at the session's 08:30 ET cutoff: filings accepted after it, a
    reverse split executing later, never count; one accepted a minute before it does."""
    from src.signals import sel_v2
    cut = pd.Timestamp(datetime(DAY.year, DAY.month, DAY.day, 8, 30, tzinfo=ET)).tz_convert("UTC").tz_localize(None)
    sec = pd.DataFrame({"accession": ["a"], "form": ["8-K"], "acceptance": [cut - pd.Timedelta(days=20)],
                        "filing_date": ["2026-09-08"], "items": ["4.02"]})
    spl = pd.DataFrame({"ticker": ["SYN"], "execution_date": ["2026-06-01"], "split_from": [8.0], "split_to": [1.0]})
    dn = [ss.dnum(DAY)]
    base = sel_v2.session_extras("SYN", dn, splits=spl, sec=sec)[0]
    late = pd.DataFrame({"accession": ["b", "c", "d", "e"], "form": ["8-K", "NT 10-K", "424B5", "8-K"],
                         "acceptance": [cut + pd.Timedelta(minutes=1), cut + pd.Timedelta(hours=9),
                                        cut + pd.Timedelta(days=1), cut + pd.Timedelta(days=3)],
                         "filing_date": ["2026-09-28", "2026-09-28", "2026-09-29", "2026-10-01"],
                         "items": ["3.01,1.03,5.06", "", "", "4.02"]})
    late_spl = pd.DataFrame({"ticker": ["SYN"], "execution_date": ["2026-09-29"], "split_from": [20.0],
                             "split_to": [1.0]})
    _same(base, sel_v2.session_extras("SYN", dn, splits=pd.concat([spl, late_spl]),
                                      sec=pd.concat([sec, late], ignore_index=True))[0],
          "V2 session inputs, rows after the cutoff added")
    early = late.assign(acceptance=[cut - pd.Timedelta(minutes=1)] * 4)
    moved = sel_v2.session_extras("SYN", dn, splits=spl, sec=pd.concat([sec, early], ignore_index=True))[0]
    assert not np.allclose(np.nan_to_num(base, nan=-9), np.nan_to_num(moved, nan=-9))   # the guard is not vacuous


def test_v2_cross_sectional_inputs_read_only_their_own_bar():
    """The ranks and sector medians of a bar are over THAT bar's names: rewriting (or adding) the rows of a
    later bar changes nothing in it."""
    from src.signals import sel_v2
    rng = np.random.default_rng(4)

    def bar(run, n, scale):
        df = pd.DataFrame({"run": run, "tradeable": True}, index=range(n))
        for s in sel_v2.XS_SOURCES:
            df[s] = rng.normal(size=n) * scale
        df["sic2"] = rng.choice([28, 36, 73], size=n)
        return df
    now, later = bar(0, 40, 1.0), bar(1, 40, 1.0)
    first = sel_v2.cross_section(pd.concat([now, later], ignore_index=True)).iloc[:40]
    later2 = bar(1, 55, 50.0)
    second = sel_v2.cross_section(pd.concat([now, later2], ignore_index=True)).iloc[:40]
    _same(first.to_numpy(), second.to_numpy(), "V2 cross-section, a later bar rewritten")


def test_the_squeeze_cover_never_reads_a_split_after_the_tick(monkeypatch):
    """The vol arm's squeeze cover carries the entry price through the splits executed after the
    entry day and on/before the tick's day. Announced splits sit in the split data before their
    date (future-dated rows exist by design): one dated after the tick must not move the level,
    whatever it is."""
    from config.settings import settings
    from src.data import intraday_store as ist
    from src.performance import tracker
    monkeypatch.setattr(settings, "enable_sel_short_squeeze_cover", True)
    monkeypatch.setattr(settings, "sel_short_cover_multiple", 6.0)
    monkeypatch.setattr(settings, "sel_short_vol_cover_multiple", 6.0)     # the split's timing at one level (live: 4x)
    now = datetime(2026, 9, 29, 11, 0, tzinfo=ET)
    trade = {"ticker": "SQZ", "sel_arm": "vol", "entry_price": 10.0, "entry_date": "2026-09-22",
             "current_price": 60.0, "current_price_datetime": (now - timedelta(minutes=5)).isoformat(),
             "sel_target_price": 9.0, "sel_deadline": datetime(2026, 10, 19, 10, 30, tzinfo=ET).isoformat()}
    decisions = []
    for later in ([], [(date(2026, 9, 30), 10.0, 1.0)], [(date(2026, 9, 30), 1.0, 50.0), (date(2027, 1, 4), 9.0, 1.0)]):
        rows = {"SQZ": [(date(2026, 9, 25), 1.0, 1.0)] + later}
        monkeypatch.setattr(ist, "split_rows", lambda rows=rows: rows)
        decisions.append((tracker._sel_short_cover_level(trade, now), tracker._sel_short_exit_reason(trade, now)))
    assert decisions[0] == (60.0, "sel_cover")
    assert decisions[1] == decisions[0] and decisions[2] == decisions[0]


def test_the_simulated_account_never_counts_a_deposit_due_after_the_tick(monkeypatch):
    """The simulated account sizes a short from the money paid in BY the tick. Since 2026-10-07 it holds
    $10,000 and takes no deposits (user directive: "Change the $5000+$1000/14days to $10000"), so no later
    instant can move it: the eve and the next morning size the same short."""
    from config.settings import settings
    from src.performance import sim_account as sa
    monkeypatch.setattr(settings, "enable_sel_short_account_sizing", True)
    monkeypatch.setattr(settings, "sel_short_account_initial", 10000.0)
    monkeypatch.setattr(settings, "sel_short_account_slices", 10)
    eve = datetime(2026, 10, 18, 23, 59, tzinfo=ET)
    morning = datetime(2026, 10, 19, 0, 1, tzinfo=ET)
    assert sa.paid_in(eve) == sa.paid_in(morning) == 10000.0
    assert sa.size([], 10.0, 3e7, eve)[0] == sa.size([], 10.0, 3e7, morning)[0] == 100


def test_a_short_keeps_the_margin_ibkr_quoted_at_its_entry(monkeypatch):
    """IBKR's what-if rate for a name is read once, at the short's entry, and stamped on the trade
    (user directive 2026-10-05: "build the what-if"): a later quote for the same name or a later
    change of the default rates never moves the account's requirement for a short opened before it."""
    from config.settings import settings
    from src.performance import sim_account as sa
    monkeypatch.setattr(settings, "enable_sel_short_account_sizing", True)
    monkeypatch.setattr(settings, "sel_short_account_initial", 5000.0)
    monkeypatch.setattr(settings, "sel_short_account_house_maint", 2.0)
    monkeypatch.setattr(settings, "sel_short_account_house_init", 2.86)
    now = datetime(2026, 10, 6, 11, 0, tzinfo=ET)
    held = {"entry_mechanism": "sel_short", "sel_arm": "vol", "status": "OPEN", "sel_account_shares": 50,
            "entry_price": 10.0, "current_price": 12.0, "return_pct": -20.0,
            "sel_house_maint": 0.30, "sel_house_init": 0.356}
    base = sa.state([held], now)
    assert base["maint"] == pytest.approx(250.0)                       # Reg T's $5 a share beats 30%
    # later: the defaults change and IBKR quotes the name far higher
    monkeypatch.setattr(settings, "sel_short_account_house_maint", 5.0)
    monkeypatch.setattr(settings, "sel_short_account_house_init", 6.0)
    assert sa.state([held], now) == base
    new = dict(held, sel_house_maint=4.0, sel_house_init=5.72, entry_price=12.0, return_pct=0.0)
    assert sa.state([held, new], now)["maint"] == pytest.approx(base["maint"] + 4.0 * 50 * 12.0)


def test_the_corporate_gap_flag_never_reads_a_bar_after_its_own():
    """The vol and ETF arms leave out a bar whose ATR% is a corporate-action gap (`ca_gap_state`): the
    flag at a bar reads only that bar and earlier ones — a crash or a gap written after it never moves it,
    and the series cut at the bar gives the same answer."""
    rng = np.random.default_rng(5)
    sday = np.repeat(np.arange(20000, 20040), 13)
    c = np.where(sday < 20030, 100.0, 20.0) * (1 + rng.normal(0.0, 0.001, len(sday)))   # a spin-off at session 30
    h, lo = c * 1.0015, c * 0.9985
    idx = pd.DatetimeIndex(pd.Timestamp("1970-01-01") + pd.to_timedelta(sday, unit="D")
                           + pd.to_timedelta(14 * 60 + 30 * (np.arange(len(sday)) % 13), unit="min"))
    assert ss.ca_gap_state(idx, h, lo, c, idx[30 * 13 + 5])["ca_gap"] is True
    for j in (30 * 13 + 5, 32 * 13 + 2, 36 * 13):
        base = ss.ca_gap_state(idx, h, lo, c, idx[j])
        assert ss.ca_gap_state(idx[:j + 1], h[:j + 1], lo[:j + 1], c[:j + 1], idx[j]) == base
        h2, l2, c2 = h.copy(), lo.copy(), c.copy()
        for arr in (h2, l2, c2):
            arr[j + 1:] *= 0.1                                  # a 90% gap right after the bar
        assert ss.ca_gap_state(idx, h2, l2, c2, idx[j]) == base


def test_the_borrow_fee_at_a_tick_never_reads_a_later_day(monkeypatch):
    """Each open short carries IBKR's day-by-day borrow fee (`borrow_fees.cost_fraction`) into the simulated
    account's equity, which sizes the next short: at a tick it reads the rates, closes and IBKR charges of days
    already over and the file in force now. A row dated the tick's day or later — IBKR's rate, a close, a charge —
    added or rewritten never moves it; the same history cut at the tick gives the same schedule."""
    from config.settings import settings
    from src.performance import borrow_fees as bf
    monkeypatch.setattr(settings, "enable_ibkr_borrow_schedule", True)
    now = datetime(2026, 10, 6, 11, 0, tzinfo=ET)
    rates = [(date(2026, 10, 1), 36.0, "own_archive"), (date(2026, 10, 2), 40.0, "own_archive"),
             (date(2026, 10, 5), 44.0, "ibkr_api")]
    closes = [(date(2026, 10, 1), 9.8), (date(2026, 10, 2), 14.5), (date(2026, 10, 5), 10.0)]
    charged = {("XYZ", date(2026, 10, 2)): {"fee_ps": 0.02, "rate": 40.0, "price": 10.0}}
    src = {"rates": list(rates), "closes": list(closes), "charged": dict(charged)}
    monkeypatch.setattr(bf, "_rates", lambda t: src["rates"])
    monkeypatch.setattr(bf, "_file_rate", lambda t, when: None)
    monkeypatch.setattr(bf, "_closes", lambda t: src["closes"])
    monkeypatch.setattr(bf, "_charged", lambda: src["charged"])
    monkeypatch.setattr(bf, "charged_version", lambda: "v")
    trade = {"ticker": "XYZ", "action": "SELL", "type": "STOCK", "status": "OPEN", "entry_price": 10.0,
             "entry_datetime": "2026-10-01T14:00:00+00:00"}
    for end in ("2026-10-06T15:00:00+00:00", "2026-10-07T01:30:00+00:00"):   # a cover now; one in tonight's session
        base = bf.schedule(dict(trade), end, now)
        assert base["days"][-1][1].endswith("_carried")                       # today: the last day over, carried
        src["rates"] = rates + [(date(2026, 10, 6), 900.0, "ibkr_api"), (date(2026, 10, 7), 950.0, "own_archive")]
        src["closes"] = closes + [(date(2026, 10, 6), 77.0), (date(2026, 10, 7), 88.0)]
        src["charged"] = {**charged, ("XYZ", date(2026, 10, 6)): {"fee_ps": 9.9, "rate": 900.0, "price": 78.0}}
        assert bf.schedule(dict(trade), end, now) == base
        src.update(rates=list(rates), closes=list(closes), charged=dict(charged))


@pytest.mark.parametrize("arm", ["vol", "vol2"])
def test_the_fallback_candidates_never_read_a_later_bars_record(tmp_path, monkeypatch, arm):
    """PREREG16's same-bar fallback (2026-10-07): a bar's backups are judged on its own
    scores and the EARLIER bars' records only — a later bar's journaled fresh top 3 (a rerun
    out of order) must not make a candidate "not first today", and rewriting it changes nothing."""
    from config.settings import settings
    monkeypatch.setattr(settings, "sel_short_dir", str(tmp_path / "sel_short"))
    monkeypatch.setattr(settings, "enable_sel_short", True)
    monkeypatch.setattr(settings, "enable_sel_short_etf", False)
    monkeypatch.setattr(settings, "sel_short_min_run_rows", 3)
    monkeypatch.setattr(settings, "sel_short_vol_fallback_ranks", 3)
    monkeypatch.setattr(settings, "enable_sel_short_vol2", arm == "vol2")
    res = pd.DataFrame({"ticker": ["V0", "V1", "V2", "V3"], "score": [0.4, 0.3, 0.2, 0.1],
                        "vol": [9.0, 8.0, 7.0, 6.0], "px": 20.0, "dv20": 1e7, "pre5": 10.0, "status": "OK",
                        "dtc": 0.5, "rvol": 3.0})

    def backups_at_bar3():
        run = ss.decide(res, DAY, 3)
        return [b["ticker"] for b in run["arms"][arm].get("backups") or []]

    clean = backups_at_bar3()
    assert clean == ["V1", "V2"]
    later = {"day": DAY.isoformat(), "bar_of_day": 5, "arm": arm, "ticker": "V1", "fresh": True,
             "decision": "short", "fresh_pool": ["V1", "V2"]}
    ss._journal(later)
    assert backups_at_bar3() == clean
    ss._journal(dict(later, fresh_pool=["V2"], ticker="V2"))
    assert backups_at_bar3() == clean


# ── THE DIP LONG BOOK (2026-10-08) ───────────────────────────────────────────

def _dip_frame(days, drop_last=2, seed=21, rewrite_from=None):
    """A $1B+-a-day uptrend (flat 30-min bars at each session's close) whose last ``drop_last`` sessions
    fall 4% each — a 2-session-RSI dip above the 200-session average at the last close. Sessions at or
    after ``rewrite_from`` get a wildly different path (x3 price, x25 volume)."""
    rng = np.random.default_rng(seed)
    n = len(days)
    close = 100.0 * np.exp(np.cumsum(rng.normal(0.003, 0.004, n)))
    for k in range(drop_last):
        close[n - drop_last + k] = close[n - drop_last + k - 1] * 0.96
    rows, idx = [], []
    for d, c in zip(days, close):
        f = 3.0 if rewrite_from is not None and d >= rewrite_from else 1.0
        vol = 1_000_000.0 * (25.0 if f != 1.0 else 1.0)
        for k in range(13):
            idx.append(pd.Timestamp(datetime(d.year, d.month, d.day, 9, 30, tzinfo=ET) + timedelta(minutes=30 * k))
                       .tz_convert("UTC").tz_localize(None))
            rows.append((c * f, c * f * 1.001, c * f * 0.999, c * f, vol))
    return pd.DataFrame(rows, columns=["Open", "High", "Low", "Close", "Volume"], index=pd.DatetimeIndex(idx))


def test_the_dip_signal_never_reads_a_bar_of_its_own_session(monkeypatch):
    """The dip book buys at session D's open on the close of D-1: a store that already holds D's bars (and
    later ones) — rewritten wildly — must not move the numbers or the signal; read through D (the injected
    leak) they do move, so the guard can fail."""
    from src.data import intraday_store
    from src.signals import dip_long as dl
    hist = ss.sessions_before(DAY, 260)
    later = [DAY, ss.session_after(DAY, 1)]
    clean = _dip_frame(hist)
    future = pd.concat([clean, _dip_frame(hist + later, rewrite_from=DAY).iloc[len(clean):]])
    prev = dl.prev_session(DAY)
    assert prev == hist[-1]
    monkeypatch.setattr(intraday_store, "load_deep_30m", lambda tk: clean)
    base = dl.name_stats("SYN", prev)
    assert dl.qualifies(base) and base["session"] == prev          # the fixture is a real signal
    monkeypatch.setattr(intraday_store, "load_deep_30m", lambda tk: future)
    assert dl.name_stats("SYN", prev) == base
    assert dl.qualifies(dl.name_stats("SYN", prev))
    leak = dl.name_stats("SYN", DAY)                                 # the injected leak: reads D
    assert leak != base and leak["session"] == DAY


def test_the_day_signal_file_never_reads_the_trade_day(monkeypatch, tmp_path):
    """End to end: `compute_signals(D)` on a store holding D's (rewritten) bars equals the one on a store
    that ends at D-1."""
    from config.settings import settings
    from src.data import intraday_store
    from src.signals import dip_long as dl
    monkeypatch.setattr(settings, "dip_long_dir", str(tmp_path / "dip_long"))
    hist = ss.sessions_before(DAY, 260)
    clean = _dip_frame(hist)
    future = pd.concat([clean, _dip_frame(hist + [DAY], rewrite_from=DAY).iloc[len(clean):]])
    monkeypatch.setattr(dl, "candidates", lambda d: (["SYN"], "test"))
    monkeypatch.setattr(dl, "_extend", lambda tks, through: None)
    out = []
    for frame in (clean, future):
        monkeypatch.setattr(intraday_store, "load_deep_30m", lambda tk, _f=frame: _f)
        out.append([{k: v for k, v in r.items()} for r in dl.compute_signals(DAY)["signals"]])
    assert out[0] == out[1] and [r["ticker"] for r in out[0]] == ["SYN"]


def test_the_dip_exit_never_reads_a_close_of_its_own_session(monkeypatch):
    """The exit sold at D's open is decided on the closes through D-1: D's own close above its 5-session
    average (a rebound that has not happened at the open) must not sell, and rewriting D changes nothing."""
    from src.data import intraday_store
    from src.signals import dip_long as dl
    hist = ss.sessions_before(DAY, 260)
    clean = _dip_frame(hist, drop_last=4)                            # still falling through D-1
    rebound_d = pd.concat([clean, _dip_frame(hist + [DAY], drop_last=0, rewrite_from=DAY).iloc[len(clean):]])
    trade = {"ticker": "SYN", "dip_entry_day": hist[-3].isoformat()}
    monkeypatch.setattr(dl, "_extend", lambda tks, through: None)
    monkeypatch.setattr(intraday_store, "load_deep_30m", lambda tk: clean)
    base = dl.exit_check(trade, DAY)
    assert base[0] is None
    monkeypatch.setattr(intraday_store, "load_deep_30m", lambda tk: rebound_d)
    assert dl.exit_check(trade, DAY) == base
    nxt = ss.session_after(DAY, 1)                                   # the injected leak: judged a day later
    assert dl.exit_check(trade, nxt)[0] == "dip_rebound"


# ── THE DIP BOOK'S META-MODEL FILTER (PREREG96's RP33, 2026-10-14) ───────────

def _filter_frames():
    """SYN: a real dip signal at D-1 (`_dip_frame`); ``wild``: D and D+1 stored and rewritten (x3 price, x25 volume);
    ``deeper``: D and D+1 stored and the dip going on (the rule fires again on both); SPY: an uptrend, and the same
    with D and D+1 rewritten."""
    hist = ss.sessions_before(DAY, 260)
    later = [DAY, ss.session_after(DAY, 1)]
    clean = _dip_frame(hist)
    wild = pd.concat([clean, _dip_frame(hist + later, rewrite_from=DAY).iloc[len(clean):]])
    deeper = pd.concat([clean, _dip_frame(hist + later, drop_last=4).iloc[len(clean):]])
    spy = _dip_frame(hist, drop_last=0, seed=8)
    spy_wild = pd.concat([spy, _dip_frame(hist + later, drop_last=0, seed=8, rewrite_from=DAY).iloc[len(spy):]])
    return later, clean, wild, deeper, spy, spy_wild


def _same_inputs(a: dict, b: dict) -> bool:
    return a.keys() == b.keys() and all((isinstance(a[k], float) and isinstance(b[k], float) and np.isnan(a[k])
                                         and np.isnan(b[k])) or a[k] == b[k] for k in a)


def test_the_dip_filter_inputs_never_read_a_bar_of_the_trade_day(monkeypatch):
    """PREREG96's RP33 scores session D's signals on the closes through D-1: a store (and a SPY series) already holding
    D's bars and later ones — rewritten wildly, or with the dip going on so that the rule fires again on D and D+1 —
    must not move any of the 17 inputs (the name's 11, breadth, days_since_prev, SPY's 4); read through D (the
    injected leak) they move, and the fixture holds a later firing a look past D-1 would count."""
    from src.data import intraday_store
    from src.signals import dip_filter as DF
    from src.signals import dip_long as dl
    monkeypatch.setattr(dl, "_extend", lambda tks, through: None)
    later, clean, wild, deeper, spy, spy_wild = _filter_frames()
    prev = dl.prev_session(DAY)
    assert DF.rule_mask(dl.daily_bars(deeper))[-2:].all()            # the rule fires again on D and D+1

    def inputs(store, spy_):
        monkeypatch.setattr(intraday_store, "load_deep_30m", lambda tk: store if tk == "SYN" else spy_)
        monkeypatch.setattr(DF, "cache_daily", lambda tk: dl.daily_bars(spy_))
        got, src = DF.spy_inputs(prev)
        assert src == "cache"
        return DF.signal_inputs("SYN", prev, 3, got, {"SYN": "2019-05-01"})
    base = inputs(clean, spy)
    assert np.isfinite([base[k] for k in DF.FEATURES]).all()         # every input known at D-1
    for store, spy_ in ((wild, spy_wild), (deeper, spy_wild), (deeper, spy)):
        assert _same_inputs(inputs(store, spy_), base)
    # the injected leaks: the name's inputs, SPY's and the previous signal read past D-1
    leak = DF.name_features(dl.daily_bars(wild), later[-1])
    assert any(leak[k] != base[k] for k in DF.NAME_FEATURES)
    assert DF.spy_features_at(dl.daily_bars(spy_wild), later[-1]) != {k: base[k] for k in DF.SPY_FEATURES}
    assert DF.prev_signal(dl.daily_bars(deeper), ss.session_after(DAY, 2)) == later[-1]


def test_the_dip_filter_score_never_reads_the_trade_day(monkeypatch, tmp_path):
    """End to end: `compute_signals(D)` with the filter on (a stand-in model reading every input) scores and keeps the
    same on a store and a SPY series holding D's (rewritten) bars as on ones that end at D-1."""
    from config.settings import settings
    from src.data import intraday_store
    from src.signals import dip_filter as DF
    from src.signals import dip_long as dl
    monkeypatch.setattr(settings, "dip_long_dir", str(tmp_path / "dip_long"))
    monkeypatch.setattr(settings, "enable_dip_long_filter", True)
    monkeypatch.setattr(dl, "candidates", lambda d: (["SYN"], "test"))
    monkeypatch.setattr(dl, "_extend", lambda tks, through: None)
    w = np.random.default_rng(0).normal(size=len(DF.FEATURES))

    class Linear:                                                    # a score every input moves
        def predict(self, X):
            return np.nan_to_num(np.asarray(X, float)) @ w
    monkeypatch.setattr(DF, "load_model", lambda path=None: (Linear(), {"policy": {"cut": 0.0}, "prior_signals": {}}))
    later, clean, wild, deeper, spy, spy_wild = _filter_frames()
    out = []
    for store, spy_ in ((clean, spy), (wild, spy_wild), (deeper, spy_wild)):
        monkeypatch.setattr(intraday_store, "load_deep_30m", lambda tk, _s=store, _p=spy_: _s if tk == "SYN" else _p)
        monkeypatch.setattr(DF, "cache_daily", lambda tk, _p=spy_: dl.daily_bars(_p))
        day = dl.compute_signals(DAY)
        out.append([(r["ticker"], r["filter_score"], r["filter_keep"], r["filter_inputs"])
                    for r in day["signals"] + day["filtered"]])
    assert out[0] == out[1] == out[2] and [r[0] for r in out[0]] == ["SYN"]


def test_the_dip_daily_top_pick_never_reads_the_trade_day(monkeypatch, tmp_path):
    """PREREG116's Q1 (2026-10-16): the day's top pick is chosen on the meta-model's scores of the closes through D-1
    — two signals, stores (and SPY) already holding D's bars and later ones, rewritten wildly or with the dip going
    on, must not move a score, an input, the ranks or the pick; the scores read through D (the injected leak) move."""
    from config.settings import settings
    from src.data import intraday_store
    from src.signals import dip_filter as DF
    from src.signals import dip_long as dl
    monkeypatch.setattr(settings, "dip_long_dir", str(tmp_path / "dip_long"))
    for k, v in {"dip_long_daily_top": 1, "dip_long_rank": "score", "enable_dip_long_filter": False}.items():
        monkeypatch.setattr(settings, k, v)
    monkeypatch.setattr(dl, "candidates", lambda d: (["SYN", "TWO"], "test"))
    monkeypatch.setattr(dl, "_extend", lambda tks, through: None)
    w = np.random.default_rng(1).normal(size=len(DF.FEATURES))

    class Linear:                                                    # a score every input moves
        def predict(self, X):
            return np.nan_to_num(np.asarray(X, float)) @ w
    monkeypatch.setattr(DF, "load_model", lambda path=None: (Linear(), {"policy": {"cut": 0.0}, "prior_signals": {}}))
    later, clean, wild, deeper, spy, spy_wild = _filter_frames()
    hist = ss.sessions_before(DAY, 260)
    two = _dip_frame(hist, seed=33)
    two_wild = pd.concat([two, _dip_frame(hist + later, seed=33, rewrite_from=DAY).iloc[len(two):]])
    out = []
    for syn, tw, spy_ in ((clean, two, spy), (wild, two_wild, spy_wild), (deeper, two_wild, spy_wild)):
        frames = {"SYN": syn, "TWO": tw}
        monkeypatch.setattr(intraday_store, "load_deep_30m", lambda tk, _f=frames, _p=spy_: _f.get(tk, _p))
        monkeypatch.setattr(DF, "cache_daily", lambda tk, _p=spy_: dl.daily_bars(_p))
        day = dl.compute_signals(DAY)
        out.append([(r["ticker"], r["top_rank"], r["filter_score"], r["filter_inputs"])
                    for r in day["signals"] + day["not_top"]])
    assert out[0] == out[1] == out[2] and len(out[0]) == 2 and out[0][0][1] == 1
    # the injected leak: the names' inputs read through D+1 move the scores the pick is made on
    prev = dl.prev_session(DAY)
    leak = {tk: DF.name_features(dl.daily_bars(f), later[-1]) for tk, f in (("SYN", wild), ("TWO", two_wild))}
    base = {tk: DF.name_features(dl.daily_bars(f), prev) for tk, f in (("SYN", clean), ("TWO", two))}
    assert all(any(leak[tk][k] != base[tk][k] for k in DF.NAME_FEATURES) for tk in ("SYN", "TWO"))


def test_the_dip_earnings_demotion_never_reads_a_filing_after_its_cut(monkeypatch, tmp_path):
    """PREREG137's DEMOTE_E (2026-10-17): session D's earnings-dip flags read the filings ACCEPTED by D's 09:30 only —
    releases accepted after it (D's opening minutes, D's session, D+1) in the store's part and in EDGAR's answer at the
    opening tick, and every later filing rewritten, change no flag, rank or pick; a release accepted before the cut
    demotes the pick (so the guard can fail)."""
    from config.settings import settings
    from src.data import deep, intraday_store
    from src.signals import dip_earnings as DE
    from src.signals import dip_filter as DF
    from src.signals import dip_long as dl
    monkeypatch.setattr(settings, "dip_long_dir", str(tmp_path / "dip_long"))
    for k, v in {"dip_long_daily_top": 1, "dip_long_rank": "score", "enable_dip_long_filter": False,
                 "enable_dip_long_earnings_demote": True, "dip_long_earnings_demote_sessions": 3,
                 "dip_long_earnings_edgar": True}.items():
        monkeypatch.setattr(settings, k, v)
    monkeypatch.setattr(dl, "candidates", lambda d: (["SYN", "TWO"], "test"))
    monkeypatch.setattr(dl, "_extend", lambda tks, through: None)

    class Deepest:                                                    # the deeper dip scores higher
        def predict(self, X):
            return -np.asarray(X, float)[:, DF.FEATURES.index("rsi2")]
    monkeypatch.setattr(DF, "load_model", lambda path=None: (Deepest(), {"policy": {"cut": 0.0}, "prior_signals": {}}))
    later, clean, wild, deeper, spy, spy_wild = _filter_frames()
    hist = ss.sessions_before(DAY, 260)
    frames = {"SYN": clean, "TWO": _dip_frame(hist, seed=33)}
    monkeypatch.setattr(intraday_store, "load_deep_30m", lambda tk: frames.get(tk, spy))
    monkeypatch.setattr(DF, "cache_daily", lambda tk: dl.daily_bars(spy))
    parts = tmp_path / "sec_parts"
    monkeypatch.setattr(DE, "parts_dir", lambda: parts)
    monkeypatch.setattr(DE, "store_through", lambda: None)
    lo, hi = DE.window(DAY)

    cik = {"SYN": "0000000007", "TWO": "0000000008"}

    def filings(tk, when, items="2.02,9.01"):
        return pd.DataFrame([{"ticker": tk, "cik": cik[tk], "accession": f"{cik[tk]}-26-{k:06d}",
                              "acceptance": pd.Timestamp(a).tz_convert("UTC").tz_localize(None), "form": "8-K",
                              "items": items} for k, a in enumerate(when)])
    old = [lo - timedelta(days=20)]
    after = [hi + timedelta(microseconds=1), hi + timedelta(seconds=45), hi + timedelta(hours=7),
             hi + timedelta(days=1), hi + timedelta(days=6)]
    edgar_now, pages = {}, {}
    monkeypatch.setattr(DE, "edgar_filings", lambda tk, c, budget=None: edgar_now[tk])
    monkeypatch.setattr(DE, "accepted_at", lambda c, acc, budget=None: pages.get((c, acc)))

    def run(syn, two, edgar_only=None):
        """The store's parts and EDGAR's answer at the tick hold ``syn`` / ``two`` (the index pages agree); EDGAR alone
        also answers ``edgar_only`` = {ticker: [(the JSON's instant, the index page's)]}."""
        pages.clear()
        for tk, when in (("SYN", syn), ("TWO", two)):
            deep.write_parquet(filings(tk, when), parts / f"{tk}.parquet")
            extra = (edgar_only or {}).get(tk, [])
            f = filings(tk, list(when) + [j for j, _ in extra])
            edgar_now[tk] = f
            for (acc, a), t in zip(zip(f["accession"], f["acceptance"]), list(when) + [i for _, i in extra]):
                pages[(cik[tk], acc)] = int(pd.Timestamp(t).value)
        day = dl.compute_signals(DAY)
        return [(r["ticker"], r["top_rank"], r["earnings_dip"], r["earnings_status"]) for r in
                day["signals"] + day["not_top"]], day["top"]["earnings"]["flagged"]
    base = run(old, old)
    pick = base[0][0][0]
    other = "TWO" if pick == "SYN" else "SYN"
    assert base[1] == [] and len(base[0]) == 2
    for syn, two in ((old + after, old + after), (old + after[2:], old), (old, old + after[:2])):
        assert run(syn, two) == base
    # EDGAR's JSON putting a release accepted after the cut inside the window (an instant 4 h early) moves nothing:
    # the index page's instant decides
    assert run(old, old, {pick: [(hi - timedelta(hours=1), hi + timedelta(hours=3))]}) == base
    # a release accepted before the cut (D's pre-market) demotes the pick: the guard sees the filings it must
    leak = run(*([old + [hi - timedelta(minutes=1)], old] if pick == "SYN" else [old, old + [hi - timedelta(minutes=1)]]))
    assert leak[1] == [pick] and leak[0][0][0] == other


# ── THE ETF DUAL-MOMENTUM BOOK (2026-10-08) ──────────────────────────────────

def test_the_etf_decision_never_reads_a_close_after_the_month_end(monkeypatch, tmp_path):
    """The month's fund is decided at the previous month's last close: a store already holding the month's own
    sessions — rewritten wildly — must not move a score or the pick; scored at the trading session's own close
    (the injected leak) every score moves."""
    from config.settings import settings
    from src.data import intraday_store
    from src.signals import etf_dm as em
    monkeypatch.setattr(settings, "etf_dm_dir", str(tmp_path / "etf_dm"))
    monkeypatch.setattr(em, "_extend", lambda tks, through: None)
    monkeypatch.setattr(em, "tbill_growth", lambda a, b: 0.04)
    d = date(2026, 10, 1)
    hist = ss.sessions_before(d, 300)
    assert em.decision_session(d) == hist[-1] == date(2026, 9, 30)
    later = [d, ss.session_after(d, 1)]
    clean, future = {}, {}
    for i, tk in enumerate(em.universe()):
        clean[tk] = _dip_frame(hist, drop_last=0, seed=31 + i)
        future[tk] = pd.concat([clean[tk], _dip_frame(hist + later, drop_last=0, seed=31 + i,
                                                       rewrite_from=d).iloc[len(clean[tk]):]])
    out = []
    for store in (clean, future):
        monkeypatch.setattr(intraday_store, "load_deep_30m", lambda tk, _s=store: _s[tk])
        dec = em.compute_decision(d, write=False)
        out.append({k: dec[k] for k in ("decided_at_close", "funds", "stale", "top", "tbill_12m", "pick")})
    assert out[0] == out[1] and out[0]["pick"] in em.universe() and not out[0]["stale"]
    leak = {tk: em.fund_stats(em.closes(tk, d), d) for tk in em.universe()}       # the injected leak: D's close
    assert all(leak[tk]["score"] != out[1]["funds"][tk]["score"] for tk in em.universe())


def test_the_tbill_hurdle_never_reads_a_rate_published_after_the_month_end(monkeypatch, tmp_path):
    """The 12-month T-bill return the top fund must beat compounds the DTB3 values PUBLISHED by the month-end
    close (ALFRED vintages; a day's rate appears the next day): the month-end's own rate, a later revision of an
    earlier day and every later day must not move it."""
    import duckdb
    from src.data import deep
    from src.signals import etf_dm as em
    start, me = date(2025, 9, 30), date(2026, 9, 30)
    days = [x.date() for x in pd.bdate_range(start, me)]
    nxt = dict(zip(days, days[1:] + [date(2026, 10, 1)]))
    known = [("DTB3", x.isoformat(), nxt[x].isoformat(), "9999-12-31", 4.0 + 0.002 * i, True)
             for i, x in enumerate(days) if nxt[x] <= me]
    unknown = [("DTB3", me.isoformat(), "2026-10-01", "9999-12-31", 9.0, True),     # the month-end's own rate
               ("DTB3", "2026-06-01", "2026-10-01", "9999-12-31", 40.0, True),      # a revision published later
               ("DTB3", "2026-10-01", "2026-10-02", "9999-12-31", 40.0, True)]      # a later day

    def growth(rows):
        p = tmp_path / f"v{len(rows)}"
        p.mkdir()
        con = duckdb.connect()
        con.execute("create table t (series_id varchar, date varchar, realtime_start varchar, "
                    "realtime_end varchar, value double, vintage boolean)")
        con.executemany("insert into t values (?, ?, ?, ?, ?, ?)", rows)
        con.execute(f"copy t to '{(p / 'fred_vintages.parquet').as_posix()}' (format parquet)")
        con.close()
        monkeypatch.setattr(deep, "DEEP_DIR", p)
        return em.tbill_growth(start, me)

    base = growth(known)
    assert 0.03 < base < 0.06
    assert growth(known + unknown) == base


def test_a_dip_limit_touch_never_reads_a_bar_that_has_not_ended():
    """The limit entry (2026-10-09) fills at the END of the first 30-minute bar whose low reaches the limit, judged
    at a tick: a bar still forming at the tick, or any later one, rewritten to reach the limit, must not fill it (its
    limit is the close of D-1 minus half the ATR — the stats guard above); read past the tick (the injected leak) it
    does, so the guard can fail."""
    from src.signals import dip_long as dl
    d = DAY

    def frame(lows):
        idx = [(datetime(d.year, d.month, d.day, 9, 30, tzinfo=ET) + timedelta(minutes=30 * k))
               .astimezone(timezone.utc).replace(tzinfo=None) for k in range(len(lows))]
        return pd.DataFrame({"Open": 101.0, "High": 102.0, "Low": lows, "Close": 101.0, "Volume": 1e6},
                            index=pd.DatetimeIndex(idx))
    tick = datetime(d.year, d.month, d.day, 10, 30, 20, tzinfo=ET).astimezone(timezone.utc)
    after = datetime(d.year, d.month, d.day, 9, 30, tzinfo=ET).astimezone(timezone.utc)
    clean = frame([101.0, 100.5])                                  # the two bars ended by the tick
    future = frame([101.0, 100.5, 97.0, 96.0])                     # + the forming bar and a later one, through 98
    assert dl.first_touch(clean, 98.0, after, tick) is None
    assert dl.first_touch(future, 98.0, after, tick) is None
    leak = dl.first_touch(future, 98.0, after, tick + timedelta(minutes=30))   # the injected leak
    assert leak is not None and leak.astimezone(ET).strftime("%H:%M") == "11:00"


def test_a_split_after_the_tick_never_rescales_a_trade_or_a_dip_order(monkeypatch):
    """PREREG64: the ledger's split adjustment (`tracker.apply_splits`, `dip_long.apply_order_splits`) reads the split
    data, where announced splits sit before their date. A split executed after the tick — or on the tick's own date
    while the tick is before 04:00 ET, the pre-market open where it takes effect — must change nothing, whatever it
    is."""
    from config.settings import settings
    from src.data import intraday_store as ist
    from src.performance import tracker
    from src.signals import dip_long
    monkeypatch.setattr(settings, "enable_split_adjustment", True)
    now = datetime(2026, 10, 7, 3, 59, tzinfo=ET)

    def run(rows):
        monkeypatch.setattr(ist, "split_rows", lambda rows=rows: rows)
        t = {"ticker": "SYN", "status": "OPEN", "entry_price": 10.0, "sel_account_shares": 40,
             "entry_datetime": "2026-10-05T14:00:00+00:00", "entry_date": "2026-10-05", "current_price": 9.0,
             "current_price_datetime": "2026-10-06T19:00:00+00:00", "broker_fill_qty": 40}
        o = {"ticker": "SYN", "day": "2026-10-05", "status": "working", "limit": 9.5, "close": 10.0, "qty": 40,
             "events": []}
        tracker.apply_splits([t], now)
        dip_long.apply_order_splits([o], now)
        o.pop("events")
        return t, o

    early = [(date(2026, 9, 1), 10.0, 1.0)]                                 # before the entry: in its prices
    base = run({"SYN": early})
    for later in ([(date(2026, 10, 7), 10.0, 1.0)], [(date(2026, 10, 8), 1.0, 3.0), (date(2027, 1, 4), 20.0, 1.0)]):
        assert run({"SYN": early + later}) == base


def test_the_borrow_pressure_never_reads_the_session_or_later(tmp_path, monkeypatch):
    """PREREG71's V6: the pressure of session D (`sel_pressure.inputs`) — rewriting the daily bars of D and later,
    IBKR's fee of D and later, FINRA's reports settled after D-10, fails-to-deliver settled after D-15, the Reg SHO
    lists of D and later and a listing dated after D changes nothing; the panel refuses a session of D itself."""
    from config.settings import settings
    from src.signals import sel_pressure as SP
    monkeypatch.setattr(settings, "sel_short_dir", str(tmp_path / "sel_short"))
    days, k = [], date(2026, 6, 1)
    while len(days) < 66:
        if k.weekday() < 5:
            days.append(k)
        k += timedelta(days=1)
    before, d, later = days[:60], days[60], days[61:]
    rng = np.random.default_rng(11)

    def write_bars(scale_later):
        g = SP.root() / "grouped"
        g.mkdir(parents=True, exist_ok=True)
        for i, s in enumerate(days):
            px = np.array([20.0, 30.0]) * np.exp(0.01 * i) * (scale_later if s >= d else 1.0)
            pd.DataFrame({"sym": ["AAA", "BBB"], "o": px, "h": px * 1.05, "l": px * 0.95, "c": px,
                          "v": [2e6, 3e6]}).to_pickle(g / f"{s.isoformat()}.pkl")
    D = SP.dnum(d)
    cal = np.array([SP.dnum(s) for s in days], np.int64)

    def frames(rewrite):
        r = 1e3 if rewrite else 1.0
        fees = {"AAA": (cal.copy(), np.r_[np.full(60, 15.0), np.full(6, 15.0 * r)])}   # D and later rewritten
        return {"si": pd.DataFrame({"ticker": ["AAA", "AAA"], "settlement_date": [EPOCH_ + pd.Timedelta(days=D - 20),
                                                                                  EPOCH_ + pd.Timedelta(days=D - 5)],
                                    "short_interest": [1e5, 1e5 * r], "days_to_cover": [1.5, 1.5 * r]}),
                "ftd": pd.DataFrame({"symbol": ["AAA", "AAA"], "sd": [EPOCH_ + pd.Timedelta(days=D - 30),
                                                                      EPOCH_ + pd.Timedelta(days=D - 10)],
                                     "q": [4e5, 4e5 * r]}),
                "regsho": pd.DataFrame({"symbol": ["AAA"] * (2 if rewrite else 1),
                                        "date": ([days[50]] + ([d] if rewrite else []))}),
                "splits": pd.DataFrame({"ticker": ["BBB"] if rewrite else [], "d": [D + 2] if rewrite else [],
                                        "split_from": [10.0] if rewrite else [], "split_to": [1.0] if rewrite else []}),
                "details": pd.DataFrame({"ticker": ["AAA", "BBB"], "d": [EPOCH_ + pd.Timedelta(days=D - 900),
                                                                         EPOCH_ + pd.Timedelta(days=D + (30 if rewrite else 400))]}),
                "fees": fees, "cover": {"AAA": (int(cal[0]), int(cal[-1]))}, "types": {}}
    write_bars(1.0)
    a = SP.inputs(d, sessions=before, fetch=False, frames=frames(False))
    write_bars(7.0)                                                   # the bars of D and later rewritten
    b = SP.inputs(d, sessions=before, fetch=False, frames=frames(True))
    pd.testing.assert_frame_equal(a.reset_index(drop=True), b.reset_index(drop=True))
    assert np.isfinite(a.set_index("ticker").loc["AAA", "api_fee_prev"])
    with pytest.raises(ValueError):                                   # D's own bar is never asked for
        SP.panel(d, before + [d], fetch=False)


EPOCH_ = pd.Timestamp("1970-01-01")


def test_the_fall_turn_reads_only_bars_completed_by_the_tick(monkeypatch):
    """PREREG76's turn (`sel_short.turn_state`): at every bar end of the window, and in the middle of the next bar,
    the waiting pick's state from the whole series equals the series cut at the tick and the series with every
    later bar rewritten."""
    from config.settings import settings
    monkeypatch.setattr(settings, "sel_short_turn_atr", 0.5)
    monkeypatch.setattr(settings, "sel_short_turn_sessions", 5)
    monkeypatch.setattr(settings, "sel_short_entry_max_age_minutes", 75.0)
    days = [DAY] + [ss.session_after(DAY, i) for i in range(1, 7)]
    df = _frame(days, seed=17)
    pick = 2                                                           # bar 2 of the first day
    states = set()
    px0 = float(df["Close"].iloc[pick])
    for atr, target in ((1.0, None), (3.0, None), (6.0, None), (3.0, px0 * 0.97), (6.0, px0 * 0.99)):
        rec = {"day": DAY.isoformat(), "bar_of_day": pick, "ticker": "SYN", "atr_pct": atr, "px": px0,
               "target": target}                                   # the ETF arm's target cancels a pick reaching it
        for t in range(pick, len(df)):
            end = (df.index[t] + pd.Timedelta(minutes=30)).tz_localize("UTC").to_pydatetime()
            for now in (end, end + timedelta(minutes=10)):              # at the bar's end, and inside the next one
                full = ss.turn_state(rec, now, hlc=_hlc(df))
                assert full == ss.turn_state(rec, now, hlc=_hlc(df.iloc[:t + 1])), (atr, t, now)
                assert full == ss.turn_state(rec, now, hlc=_hlc(_rewrite_after(df, t))), (atr, t, now)
                states.add(full["state"])
    assert {"triggered", "waiting", "cancelled"} <= states             # the guard saw a turn, a wait and a cancel


# ── THE CRASH-REBOUND BOOK (2026-10-11, PREREG86) ─────────────────────────────

def _crash_frame(days, fall=25, sharp=2, rewrite_from=None):
    """A long rise, then ``fall`` falling sessions under the 200-session average whose last ``sharp`` drop 4% each — a
    2-session RSI under 5 below the trend at the last close. Sessions at or after ``rewrite_from`` get a wildly
    different path (x3 price)."""
    n = len(days)
    close = np.r_[np.linspace(100.0, 160.0, n - fall), np.linspace(158.0, 120.0, fall)]
    for k in range(sharp):
        close[n - sharp + k] = close[n - sharp + k - 1] * 0.96
    rows, idx = [], []
    for d, c in zip(days, close):
        f = 3.0 if rewrite_from is not None and d >= rewrite_from else 1.0
        for k in range(13):
            idx.append(pd.Timestamp(datetime(d.year, d.month, d.day, 9, 30, tzinfo=ET) + timedelta(minutes=30 * k))
                       .tz_convert("UTC").tz_localize(None))
            rows.append((c * f, c * f * 1.001, c * f * 0.999, c * f, 1_000_000.0))
    return pd.DataFrame(rows, columns=["Open", "High", "Low", "Close", "Volume"], index=pd.DatetimeIndex(idx))


def test_the_crash_signal_never_reads_a_bar_of_its_own_session(monkeypatch):
    """The crash book buys QQQ at session D's open on the close of D-1: a store already holding D's bars (and a
    later one) — rewritten wildly — must not move the numbers or the signal; read through D (the injected leak)
    they do move, so the guard can fail."""
    from src.data import intraday_store
    from src.signals import crash_long as cl
    from src.signals import dip_long as dl
    hist = ss.sessions_before(DAY, 260)
    later = [DAY, ss.session_after(DAY, 1)]
    clean = _crash_frame(hist)
    future = pd.concat([clean, _crash_frame(hist + later, rewrite_from=DAY).iloc[len(clean):]])
    prev = dl.prev_session(DAY)
    monkeypatch.setattr(intraday_store, "load_deep_30m", lambda tk: clean)
    base = cl.stats_at(cl._daily(), prev)
    assert cl.qualifies(base) and base["session"] == prev.isoformat()        # the fixture is a real signal
    monkeypatch.setattr(intraday_store, "load_deep_30m", lambda tk: future)
    assert cl.stats_at(cl._daily(), prev) == base
    leak = cl.stats_at(cl._daily(), DAY)                                     # the injected leak: reads D
    assert leak != base and leak["session"] == DAY.isoformat()


def test_the_crash_exit_never_reads_a_close_of_its_own_session(monkeypatch):
    """The crash long sold at D's open is decided on the closes through D-1: D's own close above its 5-session
    average (a rebound that has not happened at the open) must not sell, and rewriting D changes nothing."""
    from src.data import intraday_store
    from src.signals import crash_long as cl
    from src.signals import dip_long as dl
    hist = ss.sessions_before(DAY, 260)
    clean = _crash_frame(hist, sharp=4)                                     # still falling through D-1
    rebound_d = pd.concat([clean, _crash_frame(hist + [DAY], sharp=4, rewrite_from=DAY).iloc[len(clean):]])
    trade = {"ticker": "TQQQ", "crash_decision_symbol": "QQQ", "crash_entry_day": hist[-3].isoformat()}
    monkeypatch.setattr(dl, "_extend", lambda tks, through: None)
    # only QQQ's bars exist: the exit must read the DECISION symbol (QQQ), never the traded fund (TQQQ)
    monkeypatch.setattr(intraday_store, "load_deep_30m", lambda tk: clean if tk == "QQQ" else None)
    base = cl.exit_check(trade, DAY)
    assert base[0] is None
    monkeypatch.setattr(intraday_store, "load_deep_30m", lambda tk: rebound_d if tk == "QQQ" else None)
    assert cl.exit_check(trade, DAY) == base
    nxt = ss.session_after(DAY, 1)                                           # the injected leak: judged a day later
    assert cl.exit_check(trade, nxt)[0] == "crash_rebound"


def test_the_live_arms_atr_frame_is_the_full_frames_and_never_reads_a_later_bar():
    """The live arms' ATR-only frame (2026-10-12, `ml_model.atr_frame_30m_from_hlc`: the scorer
    decides the vol / ETF / fall arms on it before the model's full features) must equal the
    full frame's ATR% and bar at every pick bar, and not move when every later bar is
    rewritten."""
    from src.signals.ml_model import atr_frame_30m_from_hlc, features_30m_from_hlc
    df = _frame(ss.sessions_before(DAY, 52))
    for t in CUTS:
        hlc = _hlc(df)
        end = pd.Timestamp(hlc[0][t]) + pd.Timedelta(minutes=30)
        fast, _ = atr_frame_30m_from_hlc("SYN", hlc, end)
        full, _ = features_30m_from_hlc("SYN", hlc, end)
        assert pd.Timestamp(fast["bar_ts"]) == pd.Timestamp(hlc[0][t]) == pd.Timestamp(full["bar_ts"])
        assert fast["features"]["atr_pct_14"] == full["features"]["atr_pct_14"]
        assert (fast["close"], fast["sday"], fast["bar_idx"]) == (full["close"], full["sday"], full["bar_idx"])
        other = _hlc(_rewrite_after(df, t))
        moved, _ = atr_frame_30m_from_hlc("SYN", other, end)
        assert moved["features"]["atr_pct_14"] == fast["features"]["atr_pct_14"], f"bar {t}: a later bar leaked"
        assert moved["close"] == fast["close"]


# ── THE VOL SHORTS' NEWS HOLD, SHADOW (2026-10-17, PREREG136) ────────────────────────────────────────────────────

def test_the_news_hold_decision_never_reads_a_headline_after_its_cut(monkeypatch):
    """The decision at a vol short's exit (`news_hold.tone_decision`, the mean LLM tone of the hold's headlines) reads
    the headlines published after the entry and at least the lag before the exit only: the same decision with every
    headline after that cut dropped, added or rewritten to the opposite tone."""
    from config.settings import settings
    from src.signals import news_hold as NH
    monkeypatch.setattr(settings, "sel_short_news_hold_lag_minutes", 30)
    entry = datetime(2026, 9, 28, 14, 0, tzinfo=timezone.utc)
    exit_at = datetime(2026, 9, 30, 15, 0, tzinfo=timezone.utc)
    rng = np.random.default_rng(7)

    def art(i, t, s):
        return {"id": f"h{i}", "published_utc": t.isoformat(), "title": f"h{i}", "score": float(s)}
    before = [art(i, entry + timedelta(minutes=int(m)), s)
              for i, (m, s) in enumerate(zip(sorted(rng.integers(1, 2 * 1440, 12)), rng.choice([-1, 0, 1], 12)))]
    want = NH.tone_decision(entry, exit_at, before)[0]
    for cut_offset in (29, 15, 1, 0, -60, -1440):                # inside the lag, at the exit, after it
        t = exit_at - timedelta(minutes=cut_offset)
        for s in (-1, 1):
            late = [art(100 + j, t + timedelta(seconds=j), s) for j in range(5)]
            got = NH.tone_decision(entry, exit_at, before + late)[0]
            assert got == want, f"a headline {cut_offset} min before the exit leaked into the decision"
    # the earlier ones rewritten to the opposite tone DO move it (the guard is not vacuous)
    flipped = [dict(a, score=-a["score"]) for a in before]
    assert NH.tone_decision(entry, exit_at, flipped)[0] != want or want == 0


# ── THE OWN-SECURITY SERIES (2026-10-18, src/signals/dip_security.py) ────────────────────────────────────────────────

def test_the_own_security_series_never_reads_a_bar_after_d_minus_1(monkeypatch, tmp_path):
    """The dip book reads a name from its own security's start (PREREG142's cut): the BREAK test at the new security's
    first session compares the 20 sessions' median dollar volume after it — a security that started 5 sessions before
    D-1 has only 5 of them by D-1. A store already holding D and later (their volume rewritten to the old security's)
    must not move the cut, the series or the rule's numbers; read past D-1 (the injected leak) the later sessions
    flip the break and the stitched history comes back, so the guard can fail."""
    from config.settings import settings
    from src.data import intraday_store
    from src.signals import dip_long as dl
    from src.signals import dip_security as sec
    monkeypatch.setattr(settings, "enable_dip_long_own_security", True)
    monkeypatch.setattr(settings, "dip_long_dir", str(tmp_path / "dip_long"))
    monkeypatch.setattr(sec, "list_dates", lambda: {})
    monkeypatch.setattr(sec, "split_table", lambda asof=None: {})
    hist = ss.sessions_before(DAY, 300)
    later = [ss.session_after(DAY, k) for k in range(0, 20)]
    s0 = hist[-5]                                                   # the new security's first session

    def frame(days, wild_from=None):
        idx, rows = [], []
        for d in days:
            new = d >= s0
            c = 150.0 if new else 20.0
            dv = 5e9 if new else 2e6
            if wild_from is not None and d >= wild_from:
                dv = 2e6                                            # later sessions: the old security's volume
            for k in range(13):
                idx.append(pd.Timestamp(datetime(d.year, d.month, d.day, 9, 30, tzinfo=ET) + timedelta(minutes=30 * k))
                           .tz_convert("UTC").tz_localize(None))
                rows.append((c, c * 1.001, c * 0.999, c, dv / c / 13.0))
        return pd.DataFrame(rows, columns=["Open", "High", "Low", "Close", "Volume"], index=pd.DatetimeIndex(idx))
    clean = frame(hist)
    future = frame(hist + later, wild_from=DAY)
    sec.save_events({"SPCX": {"status": 200, "at": datetime.now(timezone.utc).isoformat(),
                              "events": [{"type": "ticker_change", "date": s0.isoformat(),
                                          "ticker_change": {"ticker": "SPCX"}}]}})
    prev = dl.prev_session(DAY)
    a, ra = sec.own_bars("SPCX", clean, through=prev)
    b, rb = sec.own_bars("SPCX", future, through=prev)
    assert ra["changed"] and ra["cut_from"] == s0.isoformat()
    assert a.equals(b) and {k: v for k, v in ra.items()} == {k: v for k, v in rb.items()}
    for store in (clean, future):
        monkeypatch.setattr(intraday_store, "load_deep_30m", lambda tk, _s=store: _s)
        d = dl._daily("SPCX", prev)
        assert d.index[0].date() == s0 and len(d) == 5
    leak, rl = sec.own_bars("SPCX", future)                        # the later sessions read: the break disappears
    assert not rl["changed"] and dl.daily_bars(leak).index[0].date() == hist[0]


def test_a_dividend_never_counts_before_its_entitlement_instant(monkeypatch):
    """The ledger's dividends (AUDIT151, 2026-10-19, `src/performance/dividends.py`): at a tick a live book's trade
    carries only the ex-dates whose entitlement instant (20:00 ET of the session before the ex-date) has passed by
    the tick; a record of a later ex-date — added, or its cash rewritten — never moves the mark, and the table cut
    at the tick gives the same events (mutation-checked 2026-10-19: the instant's cut dropped)."""
    from config.settings import settings
    from src.data import intraday_store
    from src.performance import dividends as dvd
    monkeypatch.setattr(settings, "enable_ledger_dividends", True)
    monkeypatch.setattr(intraday_store, "split_factor_between", lambda *a, **k: 1.0)
    tick = datetime(2026, 11, 20, 15, 0, tzinfo=timezone.utc)              # Friday 10:00 ET
    full = {"MSFT": (["2026-11-19", "2026-11-23", "2026-12-01"], [0.98, 0.50, 0.70])}   # 11-23's instant: Fri 20:00
    cut = {"MSFT": (["2026-11-19"], [0.98])}
    rewritten = {"MSFT": (["2026-11-19", "2026-11-23", "2026-11-24"], [0.98, 9.0, 3.0])}

    def events(table):
        monkeypatch.setattr(dvd, "table", lambda: table)
        t = {"ticker": "MSFT", "action": "BUY", "type": "STOCK", "status": "OPEN", "entry_price": 400.0,
             "entry_mechanism": "dip_long", "entry_datetime": "2026-11-17T15:00:00+00:00"}
        dvd.update(t, tick.isoformat(), now=tick)
        return t.get("dividends"), dvd.frac_through(t, tick.isoformat())
    base = events(full)
    assert [e["ex"] for e in base[0]] == ["2026-11-19"] and base[1] == pytest.approx(0.98 / 400)
    assert events(cut) == base
    assert events(rewritten) == base


# ── THE CAPITULATION BOOK (2026-10-21, PREREG165) and THE CRASH BOOK'S AFTER-HOURS EXIT (PREREG166's XC) ────────────

def _whole_market(days, rewrite_from=None):
    """Whole-market daily bars by session: four names (two falling hard at the end), the sessions at or after
    ``rewrite_from`` rewritten wildly (every name rallies x3)."""
    n = len(days)
    paths = {"FA1": np.r_[np.linspace(50, 100, n - 25), np.linspace(98, 70, 23), [65.0, 60.0]],
             "FA2": np.r_[np.linspace(40, 90, n - 25), np.linspace(88, 60, 23), [55.0, 50.0]],
             "UP1": np.linspace(50, 100, n), "UP2": np.linspace(30, 60, n)}
    vol = {"FA1": 9e8, "FA2": 8e8, "UP1": 7e7, "UP2": 6e7}
    out = {}
    for k, d in enumerate(days):
        f = 3.0 if rewrite_from is not None and d >= rewrite_from else 1.0
        out[d] = pd.DataFrame([{"sym": s, "o": p[k] * f, "h": p[k] * f, "l": p[k] * f, "c": p[k] * f, "v": vol[s]}
                               for s, p in paths.items()])
    return out


def test_the_capitulation_breadth_never_reads_a_session_after_its_own(monkeypatch):
    """The breadth deciding session D is read at the close of D-1: bars of D and later — rewritten wildly — must not
    move it, and no later session is even asked; read at D (the injected leak) it moves, so the guard can fail."""
    from config.settings import settings
    from src.data import company_names, intraday_store
    from src.signals import cap_long as cl
    from src.signals import sel_pressure
    from src.signals import dip_long as dl
    monkeypatch.setattr(settings, "cap_long_breadth_top", 4)
    prev = dl.prev_session(DAY)
    days = ss.sessions_before(DAY, cl.BREADTH_SESSIONS + 3) + [DAY, ss.session_after(DAY, 1)]
    clean = _whole_market(days)
    future = _whole_market(days, rewrite_from=DAY)
    asked = []

    def serve(src):
        def grouped(d, fetch=True):
            asked.append(d)
            return src.get(d)
        return grouped
    monkeypatch.setattr(company_names, "security_type", lambda s: "CS")
    monkeypatch.setattr(intraday_store, "split_rows", lambda: {})
    monkeypatch.setattr(sel_pressure, "grouped_ohlc", serve(clean))
    base = cl.breadth_at(prev)
    assert base["breadth"] == 0.5                                            # the fixture is a capitulation reading
    monkeypatch.setattr(sel_pressure, "grouped_ohlc", serve(future))
    asked.clear()
    assert cl.breadth_at(prev)["breadth"] == base["breadth"] and max(asked) == prev
    assert cl.breadth_at(DAY)["breadth"] != base["breadth"]                  # the injected leak: read at D


def test_the_capitulation_signal_never_reads_a_bar_of_its_own_session(monkeypatch):
    """The capitulation book's order of session D is decided on the close of D-1 (`cap_long.name_stats`): a store
    holding D's bars and a later session's — rewritten wildly — must not move the numbers; read through D (the
    injected leak) they move."""
    from config.settings import settings
    from src.data import intraday_store
    from src.signals import cap_long as cl
    from src.signals import dip_long as dl
    monkeypatch.setattr(settings, "enable_dip_long_own_security", False)
    hist = ss.sessions_before(DAY, 260)
    later = [DAY, ss.session_after(DAY, 1)]
    clean = _crash_frame(hist)
    future = pd.concat([clean, _crash_frame(hist + later, rewrite_from=DAY).iloc[len(clean):]])
    prev = dl.prev_session(DAY)
    monkeypatch.setattr(intraday_store, "load_deep_30m", lambda tk: clean)
    base = cl.name_stats("XYZ", prev)
    assert base["close"] < base["sma_trend"] and base["rsi2"] < 10 and base["session"] == prev
    monkeypatch.setattr(intraday_store, "load_deep_30m", lambda tk: future)
    assert cl.name_stats("XYZ", prev) == base
    leak = cl.name_stats("XYZ", DAY)                                         # the injected leak: reads D
    assert leak != base and leak["session"] == DAY


def test_the_crash_after_hours_exit_never_reads_a_close_after_its_session(monkeypatch):
    """The after-hours exit at session S reads QQQ's closes through S (the store through S-1, S's own last
    regular-hours bar): a store already holding S's and a later session's bars — rewritten — must not move the
    decision; judged a session later (the injected leak) it moves."""
    from config.settings import settings
    from src.data import intraday_store
    from src.signals import crash_long as cl
    from src.signals import dip_long as dl
    monkeypatch.setattr(settings, "crash_long_exit_sma_sessions", 5)
    monkeypatch.setattr(settings, "crash_long_max_hold_sessions", 10)
    monkeypatch.setattr(settings, "enable_dip_long_own_security", False)
    monkeypatch.setattr(dl, "_extend", lambda tks, through: None)
    hist = ss.sessions_before(DAY, 260)
    nxt = ss.session_after(DAY, 1)
    clean = _crash_frame(hist, sharp=4)                                     # still falling through D-1
    on_day = _crash_frame(hist + [DAY], sharp=5)                            # ... and on D (its last bar)
    rebound = pd.concat([clean, _crash_frame(hist + [DAY, nxt], sharp=5, rewrite_from=nxt).iloc[len(clean):]])
    day_bars = on_day.iloc[len(clean):]                                     # D's own 13 bars
    monkeypatch.setattr(dl, "session_bars", lambda tk, d: day_bars if (tk == "QQQ" and d == DAY) else
                        rebound.iloc[len(clean) + 13:] if (tk == "QQQ" and d == nxt) else None)
    trade = {"ticker": "TQQQ", "crash_decision_symbol": "QQQ", "crash_entry_day": hist[-3].isoformat()}
    monkeypatch.setattr(intraday_store, "load_deep_30m", lambda tk: clean if tk == "QQQ" else None)
    base = cl.pm_exit_check(trade, DAY)
    assert base[0] is None and base[1]["close_today"] == pytest.approx(float(day_bars["Close"].iloc[-1]))
    monkeypatch.setattr(intraday_store, "load_deep_30m", lambda tk: rebound if tk == "QQQ" else None)
    assert cl.pm_exit_check(trade, DAY) == base
    assert cl.pm_exit_check(trade, nxt)[0] == "crash_rebound"               # the injected leak: a session later
