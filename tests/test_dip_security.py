"""The own-security series (user 2026-10-18: "Yes please apply the fix"; `src/signals/dip_security.py`, PREREG142's
construction): a reused ticker's older security cut at the new one's start, a renamed company's earlier ticker joined,
the same security left alone, the list date when Polygon has no events, a failure read as stored, and the dip book's
day computed on it (a stitched SpaceX-like name is no signal). The look-ahead guard is in `tests/test_no_lookahead.py`;
the parity with PREREG142's panel is `python -m src.signals.dip_security --parity`."""
from __future__ import annotations

import json
import pickle
from datetime import date, datetime, timedelta, timezone
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import pytest

from config.settings import settings
from src.signals import dip_long as dl
from src.signals import dip_security as sec
from src.signals import sel_short as ss

ET = ZoneInfo("America/New_York")
DAY = date(2026, 10, 19)                      # a Monday session
TODAY = pd.Timestamp("2026-10-19")


@pytest.fixture
def on(monkeypatch, tmp_path):
    monkeypatch.setattr(settings, "enable_dip_long_own_security", True)
    monkeypatch.setattr(settings, "dip_long_dir", str(tmp_path / "dip_long"))
    monkeypatch.setattr(settings, "dip_long_security_budget_seconds", 0.0)       # nothing is asked
    monkeypatch.setattr(sec, "list_dates", lambda: {})
    monkeypatch.setattr(sec, "split_table", lambda asof=None: {})
    return tmp_path


def _bars(days, closes, vols):
    """Flat 30-minute regular-hours bars at each session's close (naive-UTC bar starts)."""
    idx, rows = [], []
    for d, c, v in zip(days, closes, vols):
        for k in range(13):
            idx.append(pd.Timestamp(datetime(d.year, d.month, d.day, 9, 30, tzinfo=ET) + timedelta(minutes=30 * k))
                       .tz_convert("UTC").tz_localize(None))
            rows.append((c, c * 1.001, c * 0.999, c, v / 13.0))
    return pd.DataFrame(rows, columns=["Open", "High", "Low", "Close", "Volume"], index=pd.DatetimeIndex(idx))


def _events(name, runs):
    """A cached Polygon answer: ``runs`` = [(ticker, first day), ...]."""
    return {name: {"status": 200, "at": datetime.now(timezone.utc).isoformat(),
                   "events": [{"type": "ticker_change", "date": d, "ticker_change": {"ticker": t}} for t, d in runs]}}


def _stitched(n_old=300, n_new=100, seed=3):
    """A SPAC fund (~$20, $2M a day) for ``n_old`` sessions, then a mega listing under the same ticker (~$150, $5B a
    day) for ``n_new`` sessions, its last two closes falling — a 'dip above its 200-session average' only on the
    stitched series."""
    days = ss.sessions_before(DAY, n_old + n_new)
    rng = np.random.default_rng(seed)
    old = 20.0 * np.exp(np.cumsum(rng.normal(0.0, 0.002, n_old)))
    new = 150.0 * np.exp(np.cumsum(rng.normal(0.001, 0.01, n_new)))
    new[-2] = new[-3] * 0.95
    new[-1] = new[-2] * 0.95
    closes = np.r_[old, new]
    vols = np.r_[np.full(n_old, 2e6 / 20.0), np.full(n_new, 5e9 / 150.0)]
    return days, _bars(days, closes, vols), days[n_old]


def test_a_reused_tickers_older_security_is_cut(on):
    days, b30, s0 = _stitched()
    sec.save_events(_events("SPCX", [("SPCX", s0.isoformat())]))
    out, rec = sec.own_bars("SPCX", b30, today=TODAY)
    assert rec["changed"] and rec["cut_from"] == s0.isoformat()
    d = dl.daily_bars(out)
    assert d.index[0].date() == s0 and len(d) == 100
    assert dl.stats_at(d, days[-1]) is None                       # 100 sessions of its own: under the 220 the rule needs


def test_the_stitched_name_is_no_signal_of_the_day(on, monkeypatch):
    """The dip book's day: the stitched name qualifies on the stored series, not on its own security's."""
    from src.data import intraday_store
    days, b30, s0 = _stitched()
    monkeypatch.setattr(intraday_store, "load_deep_30m", lambda tk: b30)
    monkeypatch.setattr(dl, "candidates", lambda d: (["SPCX"], "test"))
    monkeypatch.setattr(dl, "_extend", lambda tks, through: None)
    monkeypatch.setattr(settings, "dip_long_daily_top", 0)
    monkeypatch.setattr(settings, "enable_dip_long_own_security", False)
    assert [r["ticker"] for r in dl.compute_signals(DAY)["signals"]] == ["SPCX"]          # as stored: a signal
    monkeypatch.setattr(settings, "enable_dip_long_own_security", True)
    sec.save_events(_events("SPCX", [("SPCX", s0.isoformat())]))
    out = dl.compute_signals(DAY)
    assert out["signals"] == []
    assert out["security"]["changed"]["SPCX"]["first_session"] == s0.isoformat()
    assert json.loads(dl.signals_path(DAY).read_text(encoding="utf-8"))["security"]["changed"]["SPCX"]


def test_a_renamed_companys_earlier_ticker_is_joined(on):
    """EchoStar-like: the store's NEW holds 120 sessions of its own; the earlier ticker OLD's bars (the cache's
    unadjusted run, 2021-01-04 onward) are prepended — a continuous price, so every junction holds."""
    days = ss.sessions_before(DAY, 500)
    rng = np.random.default_rng(5)
    c = 50.0 * np.exp(np.cumsum(rng.normal(0.0005, 0.01, len(days))))
    v = np.full(len(days), 2e9 / 50.0)
    k = len(days) - 120
    s1 = days[k]
    stored = _bars(days[k:], c[k:], v[k:])
    old = _bars(days[:k], c[:k], v[:k])
    a = max(pd.Timestamp("2015-01-02"), sec.WINDOW_FROM)       # the plan's join: the run inside the window
    end = pd.Timestamp(s1) - pd.Timedelta(days=1)
    p = sec.run_path("OLD", str(a.date()), str(end.date()))
    p.parent.mkdir(parents=True, exist_ok=True)
    pickle.dump(old, open(p, "wb"))
    sec.save_events(_events("NEW", [("OLD", "2015-01-02"), ("NEW", s1.isoformat())]))
    out, rec = sec.own_bars("NEW", stored, today=TODAY)
    assert rec["changed"] and list(rec["joined"]) == [f"OLD:{a.date()}"]
    d = dl.daily_bars(out)
    assert d.index[0] == pd.Timestamp(days[0]) and len(d) == len(days)
    np.testing.assert_allclose(d["c"].to_numpy(), c)
    assert dl.stats_at(d, days[-1]) is not None                   # its whole history: the rule can read it


def test_the_same_security_across_an_event_is_left_alone(on):
    """C / LIN / GOOG-like: Polygon dates a run start mid-series but the stored series does not break there."""
    days = ss.sessions_before(DAY, 400)
    rng = np.random.default_rng(9)
    c = 80.0 * np.exp(np.cumsum(rng.normal(0.0, 0.01, len(days))))
    b30 = _bars(days, c, np.full(len(days), 3e9 / 80.0))
    sec.save_events(_events("SAME", [("SAME", days[200].isoformat())]))
    out, rec = sec.own_bars("SAME", b30, today=TODAY)
    assert not rec["changed"] and out is b30


def test_the_list_date_decides_without_events(on, monkeypatch):
    days, b30, s0 = _stitched()
    monkeypatch.setattr(sec, "list_dates", lambda: {"SPCX": s0.isoformat()})
    out, rec = sec.own_bars("SPCX", b30, today=TODAY)
    assert rec["how"] == "list_date" and rec["changed"]
    assert dl.daily_bars(out).index[0].date() == s0


def test_a_name_without_a_run_of_its_own_ticker_is_read_as_stored(on):
    days, b30, s0 = _stitched()
    sec.save_events(_events("GOOGL", [("GOOG", "2015-10-06")]))
    out, rec = sec.own_bars("GOOGL", b30, today=TODAY)
    assert rec["how"] == "no_own_run" and out is b30


def test_a_failure_reads_the_stored_bars(on, monkeypatch):
    days, b30, s0 = _stitched()
    sec.save_events(_events("SPCX", [("SPCX", s0.isoformat())]))

    def boom(*a, **k):
        raise RuntimeError("broken")
    monkeypatch.setattr(sec, "assemble", boom)
    out, rec = sec.own_bars("SPCX", b30, today=TODAY)
    assert out is b30 and "broken" in rec["error"] and not rec["changed"]


def test_a_failed_answer_never_replaces_a_good_one(on):
    days, b30, s0 = _stitched()
    sec.save_events(_events("SPCX", [("SPCX", s0.isoformat())]))
    sec.save_events({"SPCX": {"status": None, "events": [], "at": datetime.now(timezone.utc).isoformat()}})
    e = sec.load_events()["SPCX"]
    assert e["status"] == 200 and e["last_failure"]["status"] is None
    assert sec.needs_events(["SPCX", "NEVER"]) == ["NEVER"]


def test_the_joined_runs_are_brought_into_todays_terms_by_the_securitys_splits(on, monkeypatch):
    """A 2-for-1 split under the new ticker after the rename: the earlier ticker's unadjusted bars are halved; a
    split dated after today is not in the store's bars yet and is not applied."""
    days = ss.sessions_before(DAY, 300)
    k = 200
    s1, sp = days[k], days[250]
    c = np.full(len(days), 100.0)
    c[250:] = 50.0                                     # the store is adjusted: before the split's day halved too
    adj = c.copy()
    adj[:250] = 50.0
    v = np.full(len(days), 2e9 / 100.0)
    stored = _bars(days[k:], adj[k:], v[k:])
    raw_old = _bars(days[:k], c[:k], v[:k])            # as traded then: $100
    a = max(pd.Timestamp("2015-01-02"), sec.WINDOW_FROM)       # the plan's join: the run inside the window
    end = pd.Timestamp(s1) - pd.Timedelta(days=1)
    p = sec.run_path("OLD", str(a.date()), str(end.date()))
    p.parent.mkdir(parents=True, exist_ok=True)
    pickle.dump(raw_old, open(p, "wb"))
    sec.save_events(_events("NEW", [("OLD", "2015-01-02"), ("NEW", s1.isoformat())]))
    later = date(2026, 12, 1)
    monkeypatch.setattr(sec, "split_table", lambda asof=None: {"NEW": [x for x in [(sp, 1.0, 2.0), (later, 1.0, 4.0)]
                                                                       if asof is None or x[0] <= asof]})
    out, rec = sec.own_bars("NEW", stored, today=TODAY)
    d = dl.daily_bars(out)
    assert rec["changed"] and len(d) == len(days)
    np.testing.assert_allclose(d["c"].to_numpy(), adj)
