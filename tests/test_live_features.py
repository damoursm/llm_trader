"""The LIVE feature capture (2026-09-25, `src/analysis/live_features.py`).

Test set 2 (from 2026-09-28) reads the technical-indicator vectors as the live
tick computed them. What must hold: the captured 30-minute row IS the training
row `ml30` builds for the same bar (and the one `ml_ohlcv` was served), the
daily row IS `ml30.ticker_rows(rows="daily")`'s X for the session, each session
captures a name once, and the capture never blocks a tick. Parity on real data
was checked on AAPL / LIVN / XOM before the deploy (85/85 features, both kinds).
"""
from __future__ import annotations

import threading
import time
from datetime import date, datetime, timezone

import numpy as np
import pandas as pd
import pytest

from src.analysis import live_features as lf
from src.analysis import ml30


def _bars(n_sessions=40, seed=0, start="2026-06-01"):
    rng = np.random.default_rng(seed)
    idx = []
    day = pd.Timestamp(start)
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


def _daily(start="2025-01-02", end="2026-08-31", seed=1):
    rng = np.random.default_rng(seed)
    days = [d.date() for d in pd.bdate_range(start, end)]
    c = 50 * np.exp(np.cumsum(rng.normal(0, 0.01, len(days))))
    h = c * (1 + rng.uniform(0, 0.01, len(days))); lo = c * (1 - rng.uniform(0, 0.01, len(days)))
    v = rng.uniform(1e6, 3e6, len(days))
    return (days, pd.Series(h, index=days), pd.Series(lo, index=days), pd.Series(c, index=days),
            pd.Series(v, index=days))


@pytest.fixture
def synthetic(monkeypatch):
    from src.analysis import deep_features as dfe
    from src.analysis import ml_dataset, predictability
    from src.signals import ml_model
    bars, daily = _bars(), _daily()
    monkeypatch.setattr(ml_dataset, "hlc_30m", lambda tk: bars)
    monkeypatch.setattr(predictability, "_hlc_by_session", lambda tk: daily)
    monkeypatch.setattr(dfe, "load_session_snapshot", lambda day: None)
    ml_model.reset_caches()
    yield bars, daily
    ml_model.reset_caches()


# ── the session a tick serves ────────────────────────────────────────────────

@pytest.mark.parametrize("utc,expected", [
    ("2026-09-25 14:00", date(2026, 9, 25)),     # Fri 10:00 ET: that session
    ("2026-09-25 23:30", date(2026, 9, 25)),     # Fri 19:30 ET: still Friday's
    ("2026-09-26 00:30", date(2026, 9, 28)),     # Fri 20:30 ET: leads into Monday
    ("2026-09-28 00:30", date(2026, 9, 28)),     # Sun 20:30 ET overnight tick
    ("2026-09-28 06:00", date(2026, 9, 28)),     # Mon 02:00 ET
])
def test_capture_session(utc, expected):
    t = pd.Timestamp(utc, tz="UTC").to_pydatetime()
    assert lf.capture_session(t) == expected


# ── parity with the training rows ────────────────────────────────────────────

def test_the_30m_row_is_the_training_row_for_the_same_bar(synthetic):
    bars, _ = synthetic
    idx = bars[0]
    k = 13 * 35 + 6                                   # a mid-session bar of session 36
    now = idx[k] + pd.Timedelta(minutes=31)           # that bar has just completed
    got = lf.rows_30m(["SYN"], now.to_pydatetime().replace(tzinfo=timezone.utc))
    assert got["status"].tolist() == ["OK"] and got["deep_status"].tolist() == ["NO_SNAPSHOT"]
    assert pd.Timestamp(got["bar_ts"].iloc[0]) == idx[k]
    from src.analysis import deep_features as dfe
    sday = dfe.session_days(idx)
    ev = ml30.ticker_rows("SYN", 1.0, deep=False, rows="fml", eval_since=int(sday[k]))["eval"]
    j = int(np.flatnonzero((ev["dn"] == sday[k]) & (ev["bar"] == 6))[0])
    cols = ml30.base_features()
    cap = got[cols].iloc[0].to_numpy(np.float32)
    assert np.allclose(cap, ev["X"][j], rtol=1e-5, atol=1e-6, equal_nan=True)


def test_the_daily_row_is_the_daily_models_x(synthetic):
    rd = ml30.ticker_rows("SYN", 1.0, deep=False, rows="daily")
    cols = ml30.base_features()
    for j in (5, 20, len(rd["dn"]) - 1):
        sess = pd.Timestamp(np.datetime64(int(rd["dn"][j]), "D")).date()
        row = lf.daily_row("SYN", sess)
        assert row["status"] == "OK"
        assert row["feature_date"] < sess.isoformat()          # strictly the previous session
        got = np.array([row[c] for c in cols], np.float32)
        assert np.allclose(got, rd["X"][j], rtol=1e-5, atol=1e-6, equal_nan=True)


def test_too_little_daily_history_is_no_data(synthetic):
    assert lf.daily_row("SYN", date(2025, 2, 3))["status"] == "NO_DATA"


# ── the files ────────────────────────────────────────────────────────────────

def test_capture_writes_a_file_per_tick_and_each_name_once_per_session(synthetic, tmp_path):
    bars, _ = synthetic
    t1 = (bars[0][-1] + pd.Timedelta(hours=2)).to_pydatetime().replace(tzinfo=timezone.utc)
    s1 = lf.capture("r1", t1, ["SYN"], base_dir=tmp_path)
    s2 = lf.capture("r2", t1 + pd.Timedelta(minutes=30), ["SYN", "syn2"], base_dir=tmp_path)
    assert s1["rows_30m"] == 1 and s2["rows_30m"] == 2
    assert s1["daily_new"] == 1 and s2["daily_new"] == 1          # SYN captured once
    m = lf.load("30m", base_dir=tmp_path)
    assert sorted(m["run_id"].unique()) == ["r1", "r2"]
    assert set(ml30.base_features()) <= set(m.columns)
    from src.analysis import deep_features as dfe
    assert set(dfe.DEEP_FEATURES) <= set(m.columns)
    d = lf.load("daily", base_dir=tmp_path)
    assert sorted(d["ticker"]) == ["SYN", "SYN2"] and d["session"].nunique() == 1


def test_capture_async_is_single_flight(monkeypatch):
    gate = threading.Event()
    ran = []
    monkeypatch.setattr(lf, "capture", lambda *a, **k: (ran.append(a[0]), gate.wait(5)))
    now = datetime.now(timezone.utc)
    assert lf.capture_async("r1", now, ["A"]) is True
    assert lf.capture_async("r2", now, ["A"]) is False            # r1 still running
    gate.set()
    lf._THREAD.join(5)
    assert ran == ["r1"]


def test_the_pipeline_captures_after_persisting_and_the_suite_keeps_it_off():
    """The capture must follow `_persist_run` (it reads nothing the run writes,
    but it must never delay it) and stay off in tests (conftest)."""
    import inspect
    import src.pipeline as pipeline
    from config.settings import settings
    src = inspect.getsource(pipeline.run_pipeline)
    i_persist = src.index("_persist_run(")
    i_capture = src.index("live_features.capture_async(run_id, start, list(signals_by_ticker))")
    assert i_capture > i_persist
    assert settings.enable_live_feature_capture is False
