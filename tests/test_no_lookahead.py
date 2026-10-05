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
* the model's training rows — only labels CONFIRMED by the fit's cut.
The deep features' point-in-time rules (the 08:30 cutoff, publication lags, rows
published after the cutoff) are pinned in `tests/test_deep_features.py` — the Reg SHO
(`rs`) and IBKR borrow (`bw`) groups there too (a list / borrow row of D is known from
D+1; rows dated on or after a session cannot move it; a session's values are the same
in a training batch and in the live snapshot; mutation-checked 2026-10-02); the
pivot-leg state's confirmation rule in `tests/test_pivot_target.py`; the run-up
base's history-gap guard and completed-bars-only scoring in `tests/test_sel_short.py`.
"""
from __future__ import annotations

from datetime import date, datetime, timedelta
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
    monkeypatch.setattr(settings, "enable_sel_short_vol_squeeze_cover", True)
    monkeypatch.setattr(settings, "sel_short_vol_cover_multiple", 6.0)
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
    """The vol arm's simulated account ($5,000 + $1,000 every 14 days) sizes a short from the money
    paid in BY the tick: the deposit due the next morning must not move it."""
    from config.settings import settings
    from src.performance import sim_account as sa
    monkeypatch.setattr(settings, "enable_sel_short_account_sizing", True)
    monkeypatch.setattr(settings, "sel_short_account_start", "2026-10-05")
    monkeypatch.setattr(settings, "sel_short_account_initial", 5000.0)
    monkeypatch.setattr(settings, "sel_short_account_deposit", 1000.0)
    monkeypatch.setattr(settings, "sel_short_account_deposit_days", 14)
    monkeypatch.setattr(settings, "sel_short_account_slices", 10)
    eve = datetime(2026, 10, 18, 23, 59, tzinfo=ET)
    assert sa.paid_in(eve) == 5000.0 and sa.size([], 10.0, 3e7, eve)[0] == 50
    morning = datetime(2026, 10, 19, 0, 1, tzinfo=ET)
    assert sa.paid_in(morning) == 6000.0 and sa.size([], 10.0, 3e7, morning)[0] == 60
