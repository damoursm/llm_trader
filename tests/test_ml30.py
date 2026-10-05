"""The in-tree 30-minute ml_ohlcv trainer (`src/analysis/ml30.py`).

Parity with the out-of-tree recipe that built the live artifact was verified on
real tickers (identical features, labels to float32 rounding, identical
confirmation days — AAPL/KFRC/SMCI/XOM, 2026-09-23); these tests pin the row
definition and the label's point-in-time cut on synthetic bars.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.analysis import ml30


def _bars(n_sessions=40, seed=0):
    rng = np.random.default_rng(seed)
    idx = []
    day = pd.Timestamp("2026-06-01")
    while len(idx) < n_sessions * 13:
        if day.weekday() < 5:
            for k in range(13):
                idx.append(pd.Timestamp(f"{day.date()} 09:30").tz_localize("America/New_York")
                           .tz_convert("UTC").tz_localize(None) + pd.Timedelta(minutes=30 * k))
        day += pd.Timedelta(days=1)
    idx = pd.DatetimeIndex(idx)
    c = 100 * np.exp(np.cumsum(rng.normal(0, 0.004, len(idx))))
    h = c * (1 + rng.uniform(0, 0.003, len(idx))); lo = c * (1 - rng.uniform(0, 0.003, len(idx)))
    return idx, pd.Series(h), pd.Series(lo), pd.Series(c), pd.Series(rng.uniform(1e4, 5e4, len(idx)))


def test_session_positions_match_the_live_recipe():
    """13-bar session: first/middle/last; a 1-bar session (a half-day's lone
    bar) is ONE row at position 2; a 2-bar session has no middle row."""
    sday = np.repeat([10, 11, 12], [13, 1, 2])
    bi, dd, pp = ml30.session_positions(sday)
    assert bi.tolist() == [0, 6, 12, 13, 14, 15]
    assert dd.tolist() == [10, 10, 10, 11, 12, 12]
    assert pp.tolist() == [0, 1, 2, 2, 0, 2]


def test_ticker_rows_labels_and_features_on_synthetic_bars(monkeypatch):
    from src.analysis import ml_dataset
    bars = _bars()
    monkeypatch.setattr(ml_dataset, "hlc_30m", lambda tk: bars)
    r = ml30.ticker_rows("SYN", 1.0, deep=False)
    assert r is not None
    n_sess = 40
    assert len(r["y"]) == 3 * n_sess
    assert r["X"].shape == (3 * n_sess, len(ml30.base_features()))
    lab = np.isfinite(r["y"])
    # a labelled row's pivot confirmed on or after the row's own session
    assert (r["conf"][lab] >= r["dn"][lab]).all()
    # the label is signed and bounded by the path
    assert np.nanmax(np.abs(r["y"])) < 50


def test_ticker_rows_refuses_short_history(monkeypatch):
    from src.analysis import ml_dataset
    bars = _bars(n_sessions=20)                                   # 260 bars < MIN_BARS
    monkeypatch.setattr(ml_dataset, "hlc_30m", lambda tk: bars)
    assert ml30.ticker_rows("SYN", 1.0, deep=False) is None


def test_bars_to_pivot_match_the_live_label(monkeypatch):
    """`ba` is what the selection objective divides the return by; on every row
    both resolve, it and the label equal the ONE live resolver's answer for a
    tick at the row bar's close."""
    from src.analysis import deep_features as dfe
    from src.analysis import ml_dataset
    from src.analysis.pivot_target import intraday_pivot_targets
    bars = _bars()
    monkeypatch.setattr(ml_dataset, "hlc_30m", lambda tk: bars)
    r = ml30.ticker_rows("SYN", 1.0, deep=False)
    idx, h, lo, c, _v = bars
    bi, _dd, _pp = ml30.session_positions(dfe.session_days(idx))
    anchors = [(idx[i] + pd.Timedelta(minutes=30), float(c.iloc[i])) for i in bi]
    live = intraday_pivot_targets(idx, c.to_numpy(), h.to_numpy(), lo.to_numpy(), anchors,
                                  basis="hl", thr_pct=1.0)
    lab = np.isfinite(r["y"])
    assert (r["ba"][lab] >= 1).all() and (r["ba"][~lab] == -1).all()
    checked = 0
    for k, lv in enumerate(live):
        if lv is not None and lv["resolved"] and lab[k]:
            assert r["ba"][k] == lv["bars_ahead"]
            assert r["y"][k] == pytest.approx(lv["target_pct"], rel=1e-4)
            checked += 1
    assert checked >= 60


def test_random_positions_share_bar_times_across_tickers_and_cover_the_day():
    """Three bars per session at day-seeded positions: every ticker with a full
    session gets the same bars (a full cross-section per run), and across days
    the positions cover the whole session, not just first/middle/last."""
    sday = np.repeat(np.arange(20000, 20060), 13)
    bi, dn, slot = ml30.random_positions(sday, 3)
    assert len(bi) == 60 * 3 and (np.bincount(slot) == 60).all()
    bi2, _, _ = ml30.random_positions(sday.copy(), 3)
    assert (bi == bi2).all()                                     # deterministic: same day -> same bars
    covered = set((bi - np.searchsorted(sday, dn)).tolist())
    assert covered == set(range(13))
    half = np.repeat([30000, 30001], [2, 13])                    # a 2-bar session keeps both bars
    b3, _, _ = ml30.random_positions(half, 3)
    assert b3[:2].tolist() == [0, 1] and len(b3) == 5


def test_rows_modes_eval_block_and_realizable_returns(monkeypatch):
    from src.analysis import deep_features as dfe
    from src.analysis import ml_dataset
    bars = _bars()
    monkeypatch.setattr(ml_dataset, "hlc_30m", lambda tk: bars)
    idx, h, lo, c, v = bars
    sday = dfe.session_days(idx)
    days = np.unique(sday)
    r = ml30.ticker_rows("SYN", 1.0, deep=False, rows="random3", eval_since=int(days[-10]))
    assert len(r["y"]) == 3 * len(days) and (r["pos"] <= 2).all()
    ev = r["eval"]
    assert len(ev["y"]) == 10 * 13 and ev["bar"].tolist()[:13] == list(range(13))
    cc = c.to_numpy()
    # realizable returns from a mid-session bar: this session's close, next, +5
    i = int(np.flatnonzero(sday == days[-10])[5])
    s_last = lambda d: cc[np.flatnonzero(sday == d)[-1]]           # noqa: E731
    assert ev["fsc"][5] == pytest.approx((s_last(days[-10]) / cc[i] - 1) * 100, rel=1e-5)
    assert ev["f1d"][5] == pytest.approx((s_last(days[-9]) / cc[i] - 1) * 100, rel=1e-5)
    assert ev["f5d"][5] == pytest.approx((s_last(days[-5]) / cc[i] - 1) * 100, rel=1e-5)
    # from a session's LAST bar, "the session close" is the next session's
    j = int(np.flatnonzero(sday == days[-10])[-1])
    assert ev["fsc"][12] == pytest.approx((s_last(days[-9]) / cc[j] - 1) * 100, rel=1e-5)
    assert np.isnan(ev["f5d"][-1]) and ev["px"][5] == pytest.approx(cc[i], rel=1e-6)
    # dollar volume is the 20 sessions BEFORE the row's session
    first = np.r_[0, np.flatnonzero(np.diff(sday) != 0) + 1]
    sdv = np.add.reduceat(cc * v.to_numpy(), first)
    s_i = int(np.searchsorted(days, days[-10]))
    assert ev["dv20"][0] == pytest.approx(sdv[s_i - 20:s_i].mean(), rel=1e-5)


def test_build_streams_train_and_eval_arrays(monkeypatch, tmp_path):
    from src.analysis import ml_dataset
    series = {"AAA": _bars(seed=1), "BBB": _bars(seed=2), "CCC": _bars(n_sessions=20, seed=3)}
    monkeypatch.setattr(ml_dataset, "hlc_30m", lambda tk: series[tk])
    meta = ml30.build(tickers=["AAA", "BBB", "CCC"], thr=1.0, deep=False, workers=1,
                      out_dir=tmp_path / "tr", rows="random3", eval_since="2026-07-20")
    assert meta["rows"] == "random3" and meta["n_rows"] == 2 * 3 * 40      # CCC is too short
    X = np.load(tmp_path / "tr" / "X.npy")
    tk = np.load(tmp_path / "tr" / "tk.npy")
    assert X.shape == (240, len(ml30.base_features())) and set(tk.tolist()) == {0, 1}
    ev_meta = __import__("json").loads((tmp_path / "tr_eval" / "meta.json").read_text())
    ev_bar = np.load(tmp_path / "tr_eval" / "bar.npy")
    assert ev_meta["rows"] == "all_bars" and len(ev_bar) == ev_meta["n_rows"] > 0
    assert not list((tmp_path / "tr").glob("*.bin"))                        # the raw parts are gone
    ref = ml30.ticker_rows("AAA", 1.0, deep=False, rows="random3")
    y = np.load(tmp_path / "tr" / "y.npy")
    ba = np.load(tmp_path / "tr" / "ba.npy")
    a = np.flatnonzero(tk == 0)
    assert np.allclose(y[a], ref["y"], equal_nan=True) and (ba[a] == ref["ba"]).all()


def test_daily_rows_use_the_previous_sessions_daily_features(monkeypatch):
    """A model served once a day, pre-market, knows the DAILY bars through the
    previous session: row D's X is the daily frame at the last daily bar BEFORE
    D; its label and returns are the first 30-minute bar's."""
    from src.analysis import deep_features as dfe
    from src.analysis import ml_dataset, predictability
    bars = _bars()
    idx = bars[0]
    sdays = sorted(set(pd.Timestamp(d).date() for d in
                       idx.tz_localize("UTC").tz_convert("America/New_York").date))
    rng = np.random.default_rng(5)
    ddates = [d.date() for d in pd.bdate_range(end=pd.Timestamp(sdays[-1]), periods=200)]
    dc = pd.Series(100 * np.exp(np.cumsum(rng.normal(0, 0.01, len(ddates)))), index=ddates)
    daily = (ddates, dc * 1.01, dc * 0.99, dc, pd.Series(rng.uniform(1e6, 2e6, len(ddates)), index=ddates))
    monkeypatch.setattr(ml_dataset, "hlc_30m", lambda tk: bars)
    monkeypatch.setattr(predictability, "_hlc_by_session", lambda tk: daily)
    r = ml30.ticker_rows("SYN", 1.0, deep=False, rows="daily")
    assert len(r["y"]) == len(sdays) and (r["bar"] == 0).all()
    ref = ml_dataset.ticker_feature_frame("SYN", hlc=daily).reindex(columns=ml_dataset.ALL_FEATURE_COLUMNS)
    k = 10
    d_k = np.datetime64(int(r["dn"][k]), "D").astype(object)
    prev = max(i for i, d in enumerate(ddates) if d < d_k)
    nf = len(ml_dataset.ALL_FEATURE_COLUMNS)
    assert np.allclose(r["X"][k, :nf], ref.iloc[prev].to_numpy(np.float32), equal_nan=True)
    first_bar = dfe.session_days(idx)
    i0 = int(np.flatnonzero(first_bar == r["dn"][k])[0])
    assert r["px"][k] == pytest.approx(float(bars[3].iloc[i0]), rel=1e-6)


def test_trailing_exits_long_short_gap_and_time_stop():
    c = np.array([100, 101, 103, 102.5, 101.9, 105, 106, 107], float)
    h = c + 0.2
    lo = c - 0.2
    h[4] = 102.5                    # bar 4 trades THROUGH the stop (not a gap): fill at the stop
    # long from bar 0: best rises to 103.2 (bar 2 high); bar 4 low 101.7 <= 103.2*0.99=102.168 -> exit at stop
    r, b = ml30.trailing_exits(c, h, lo, np.array([0]), 1.0, "long")
    assert b[0] == 4 and r[0] == pytest.approx((103.2 * 0.99 / 100 - 1) * 100)
    # short from bar 0: best is the entry (100), bar 1 high 101.2 >= 101 -> exit at 101 (a ~1% loss)
    r, b = ml30.trailing_exits(c, h, lo, np.array([0]), 1.0, "short")
    assert b[0] == 1 and r[0] == pytest.approx(1.0)
    # a bar that gaps through the stop fills at its own high (long)
    c2 = np.array([100, 100, 90], float); h2 = np.array([100, 100, 91.0]); l2 = np.array([100, 100, 89.0])
    r, b = ml30.trailing_exits(c2, h2, l2, np.array([0]), 1.0, "long")
    assert b[0] == 2 and r[0] == pytest.approx(-9.0)
    # time stop at the close when no stop is hit within max_bars
    c3 = np.linspace(100, 110, 30); h3 = c3 + 0.01; l3 = c3 - 0.01
    r, b = ml30.trailing_exits(c3, h3, l3, np.array([0]), 1.0, "long", max_bars=5)
    assert b[0] == 5 and r[0] == pytest.approx((c3[5] / 100 - 1) * 100)
    # the series ends first: no exit
    r, b = ml30.trailing_exits(c3, h3, l3, np.array([28]), 1.0, "long", max_bars=5)
    assert b[0] == -1 and np.isnan(r[0])


def test_add_exit_labels_aligns_to_the_rows(monkeypatch, tmp_path):
    from src.analysis import deep_features as dfe
    from src.analysis import ml_dataset
    series = {"AAA": _bars(seed=1), "BBB": _bars(seed=2)}
    monkeypatch.setattr(ml_dataset, "hlc_30m", lambda tk: series[tk])
    ml30.build(tickers=["AAA", "BBB"], thr=1.0, deep=False, workers=1, out_dir=tmp_path / "tr", rows="random3")
    ml30.add_exit_labels(tmp_path / "tr", workers=1)
    tk = np.load(tmp_path / "tr" / "tk.npy"); dn = np.load(tmp_path / "tr" / "dn.npy")
    bar = np.load(tmp_path / "tr" / "bar.npy"); xl = np.load(tmp_path / "tr" / "xl.npy")
    k = int(np.flatnonzero(tk == 1)[7])
    idx, h, lo, c, _v = series["BBB"]
    sday = dfe.session_days(idx)
    i = int(np.flatnonzero(sday == dn[k])[0]) + int(bar[k])
    r, _b = ml30.trailing_exits(c.to_numpy(), h.to_numpy(), lo.to_numpy(), np.array([i]), 1.0, "long")
    assert xl[k] == pytest.approx(r[0], rel=1e-5)
