"""Selection-objective models (`src/analysis/sel_models.py`, user spec 2026-09-25):
the per-side targets, the short side's orientation, the per-run IC, and one
end-to-end pass (build -> fit -> predict -> report) on synthetic tickers."""
import math

import numpy as np
import pandas as pd
import pytest

from src.analysis import sel_models as sm


def test_grades_are_top_heavy_and_the_tail_target_starts_at_the_80th_percentile():
    assert sm.grades(np.array([0.3, 0.6, 0.85, 0.92, 0.96, 0.985, 0.992, 0.999])).tolist() == \
        [0, 1, 2, 3, 4, 5, 6, 7]
    assert sm.tail_target(np.array([0.5, 0.8, 0.9, 1.0])).tolist() == pytest.approx([0, 0, 0.25, 1.0])


def test_side_rpd_orients_the_short_side_and_drops_unlabelled_rows():
    y = np.array([2.0, -3.0, 1.0]); ba = np.array([26, 13, -1])
    lng = sm.side_rpd(y, ba, "long")
    sht = sm.side_rpd(y, ba, "short")
    assert lng[:2].tolist() == [1.0, -3.0] and sht[:2].tolist() == [-1.0, 3.0]
    assert math.isnan(lng[2]) and math.isnan(sht[2])


def test_a_short_models_conviction_is_negated_but_the_symmetric_baseline_is_not():
    s = np.array([0.2, -0.1])
    assert sm.oriented(s, "short", "lambdarank").tolist() == [-0.2, 0.1]
    assert sm.oriented(s, "short", "rankreg").tolist() == [0.2, -0.1]
    assert sm.oriented(s, "long", "tailreg").tolist() == [0.2, -0.1]


def test_run_ic_is_the_day_mean_of_per_run_spearman():
    rows = []
    for run, d in ((1, "2026-07-01"), (2, "2026-07-01"), (3, "2026-07-02")):
        for i in range(25):
            rows.append({"run": run, "signal_date": d, "score": float(i),
                         "fwd_ret_pivot": float(i if run != 2 else -i)})
    out = sm.run_ic(pd.DataFrame(rows), "fwd_ret_pivot")
    assert out["days"] == 2 and out["mean"] == pytest.approx((0.0 + 1.0) / 2)


def _bars(n_sessions=32, seed=0):
    rng = np.random.default_rng(seed)
    idx, day = [], pd.Timestamp("2026-03-02")
    while len(idx) < n_sessions * 13:
        if day.weekday() < 5:
            for k in range(13):
                idx.append(pd.Timestamp(f"{day.date()} 09:30").tz_localize("America/New_York")
                           .tz_convert("UTC").tz_localize(None) + pd.Timedelta(minutes=30 * k))
        day += pd.Timedelta(days=1)
    idx = pd.DatetimeIndex(idx)
    c = 50 * np.exp(np.cumsum(rng.normal(0, 0.006, len(idx))))
    h = c * (1 + rng.uniform(0, 0.004, len(idx))); lo = c * (1 - rng.uniform(0, 0.004, len(idx)))
    return idx, pd.Series(h), pd.Series(lo), pd.Series(c), pd.Series(rng.uniform(2e4, 6e4, len(idx)))


def test_end_to_end_on_synthetic_tickers(monkeypatch, tmp_path):
    from src.analysis import ml30, ml_dataset
    series = {f"T{i:02d}": _bars(seed=i) for i in range(24)}
    monkeypatch.setattr(ml_dataset, "hlc_30m", lambda tk: series[tk])
    ml30.build(tickers=sorted(series), thr=1.0, deep=False, workers=1, out_dir=tmp_path / "tr",
               rows="random3", eval_since="2026-04-06")
    monkeypatch.setattr(sm, "MIN_QUERY", 10)
    arr = sm.Arrays(tmp_path / "tr")
    rows = sm.training_rows(arr, "2026-03-31")
    assert len(rows) and (np.diff(arr.run[rows]) >= 0).all()          # grouped by run
    assert (arr.conf[rows] <= sm.dnum("2026-03-31")).all()              # no label printed after the cut
    ev = sm.Arrays(tmp_path / "tr_eval")
    er = np.flatnonzero(ev.tradeable())
    for obj in sm.OBJECTIVES:
        side = "both" if obj == "rankreg" else "short"
        b = sm.fit(arr, rows, side, obj, fset="base", rounds=3, threads=1,
                   params=dict(min_child_samples=5, num_leaves=7))
        pred = sm.predict(b, ev, er, "base")
        assert pred.shape == (len(er),) and np.isfinite(pred).all()
        rep = sm.report(sm.frame(ev, er, sm.oriented(pred, "short", obj)), "short",
                        {"w": ("2026-04-06", None)})["w"]
        assert rep["selection"]["days"] > 0 and "f1d" in rep["picks_realized"]
        assert rep["selection"]["entries"] >= 1


def test_the_trail_target_is_the_realizable_exit_per_day():
    from types import SimpleNamespace
    arr = SimpleNamespace(dir="x", y=np.array([3.0, 3.0]), ba=np.array([4, 4]),
                          xl=np.array([2.6, -1.0], np.float32), xlb=np.array([26, 3], np.int32),
                          xs=np.array([-0.5, 1.0], np.float32), xsb=np.array([2, -1], np.int32))
    rows = np.array([0, 1])
    assert sm.side_target(arr, rows, "long", "trail").tolist() == pytest.approx([1.3, -1.0])
    s = sm.side_target(arr, rows, "short", "trail")
    assert s[0] == pytest.approx(0.5) and math.isnan(s[1])            # no exit -> no label
    assert sm.side_target(arr, rows, "long", "pivot").tolist() == pytest.approx([3.0, 3.0])


def test_validate_rescore_and_summary_on_synthetic_tickers(monkeypatch, tmp_path):
    from src.analysis import ml30, ml_dataset
    series = {f"T{i:02d}": _bars(seed=i) for i in range(24)}
    monkeypatch.setattr(ml_dataset, "hlc_30m", lambda tk: series[tk])
    ml30.build(tickers=sorted(series), thr=1.0, deep=False, workers=1, out_dir=tmp_path / "tr",
               rows="random3")
    ml30.add_exit_labels(tmp_path / "tr", workers=1)
    monkeypatch.setattr(sm, "MIN_QUERY", 10)
    monkeypatch.setattr(sm, "OUT_DIR", tmp_path / "out")
    monkeypatch.setitem(sm.KINDS, "intraday", (tmp_path / "tr", tmp_path / "tr", ""))
    val = ("2026-04-09", "2026-04-15")         # trail rows stop 16 days before the cut
    res = sm.validate(objectives=("lambdarank", "rankreg"), sides=("long",), fset="base",
                      cut="2026-04-08", val=val, rounds=4, checkpoints=(2, 4), threads=1,
                      params=dict(min_child_samples=5, num_leaves=7), target="trail")
    assert res["target"] == "trail" and {"lambdarank|long|2", "rankreg|long|4"} <= set(res["runs"])
    assert "trail_rpd" in res["runs"]["lambdarank|long|4"]["picks_realized"]
    rs = sm.rescore("val", phase="val", fset="base", checkpoints=(4,), val=val)
    assert "lambdarank|long|4" in rs and "rankreg|short|4" in rs
    tab = sm.summarize("val")
    assert {"rpd", "trail/d", "1d"} <= set(tab.columns) and len(tab) >= 4


def test_the_vol_matched_control_takes_same_run_names_of_similar_volatility():
    rows = []
    for i in range(40):                                   # one run; vol rank = i; f1d = i / 10
        rows.append({"run": 1, "signal_date": "2026-07-01", "ticker": i, "score": float(i),
                     "vol": float(i), "f1d": i / 10.0, "f5d": 0.0})
    df = pd.DataFrame(rows)
    e = df[df["ticker"] == 20].assign(_d="2026-07-01")
    ctl = sm.vol_control(df, e, "long", half=0.03)        # +-3% of 40 names: ranks 19 and 21
    assert ctl.loc[e.index[0], "f1d"] == pytest.approx((1.9 + 2.1) / 2)
    sht = sm.vol_control(df, e, "short", half=0.03)
    assert sht.loc[e.index[0], "f1d"] == pytest.approx(-(1.9 + 2.1) / 2)


def test_report_controls_every_entry_against_its_own_run():
    """Regression: entries once came back renumbered, so the control looked up
    the wrong rows and covered only a few entry-days."""
    rows = []
    for d in range(12):
        day = f"2026-07-{d + 1:02d}"
        for i in range(40):
            rows.append({"run": d, "signal_date": day, "ticker": i, "score": float(i == 7) + i / 1000.0,
                         "vol": float(i), "fwd_ret_pivot": 1.0, "bars_ahead": 13.0,
                         "fsc": 0.0, "f1d": i / 10.0, "f5d": 0.0})
    df = pd.DataFrame(rows).sample(frac=1.0, random_state=0)       # scrambled labels
    rep = sm.report(df, "long", {"w": ("2026-07-01", None)})["w"]
    exc = rep["vol_matched"]["f1d"]
    assert exc["control"]["days"] == rep["selection"]["days_with_entry"] > 0
    # the pick is ticker 7 (score 1.007) earning 0.7; its vol neighbours 6 and 8 average 0.7
    assert exc["excess"]["mean"] == pytest.approx(0.0, abs=1e-9)
