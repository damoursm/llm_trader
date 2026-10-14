"""The scorer decides the LIVE arms first, the shadow arms after (2026-10-12).

User 2026-10-11: "run first only the necessary parts for live production, and after we do the
shadow arms ... MAKE SURE THE EXACT SAME LOGIC IS APPLIED FOLLOWING YOUR CHANGES."

The live arms (vol / ETF / fall) rank the 30-minute ATR%: their rows are built first — the
ATR from `ml_model.atr_frame_30m_from_hlc`, the same construction as the full frame's — and
decided and journaled at once; the model's inputs (the full feature frame, the V2 vector)
follow on the same fetched bars, then the shadow arms. These tests pin that the live rows and
the live decisions are the ones the one-phase scorer makes.
"""
from __future__ import annotations

import json
from datetime import date

import numpy as np
import pandas as pd
import pytest

import src.signals.sel_short as ss
from config.settings import settings
from src.signals import ml_model


# ── the ATR-only frame IS the full frame's ATR and bar position ──────────────

def _hlc(n_sessions: int, seed: int, flat_every: int = 0):
    """``(idx, high, low, close, volume)`` on the regular-hours 30-minute grid (naive UTC bar
    starts, October 2026 = EDT), a random walk; ``flat_every`` makes every k-th bar zero-range."""
    rng = np.random.default_rng(seed)
    days = pd.bdate_range("2026-08-03", periods=n_sessions)
    starts = [pd.Timestamp(day) + pd.Timedelta(hours=13, minutes=30) + pd.Timedelta(minutes=30 * k)
              for day in days for k in range(13)]
    idx = pd.DatetimeIndex(starts)
    c = 20.0 * np.exp(np.cumsum(rng.normal(0, 0.01, len(idx))))
    h = c * (1 + np.abs(rng.normal(0, 0.004, len(idx))))
    lo = c * (1 - np.abs(rng.normal(0, 0.004, len(idx))))
    if flat_every:
        h[::flat_every] = c[::flat_every]
        lo[::flat_every] = c[::flat_every]
    v = rng.integers(1_000, 100_000, len(idx)).astype(float)
    return idx, pd.Series(h), pd.Series(lo), pd.Series(c), pd.Series(v)


@pytest.mark.parametrize("seed,flat", [(0, 0), (1, 0), (2, 7), (3, 3)])
def test_the_atr_frame_equals_the_full_frame(seed, flat):
    hlc = _hlc(45, seed, flat)
    idx = hlc[0]
    for k in (399, 400, 401, 450, len(idx) - 1, len(idx)):           # the 400-bar floor and later cuts
        now = idx[k - 1] + pd.Timedelta(minutes=30) if k else idx[0]
        full, lf = ml_model.features_30m_from_hlc("T", hlc, now)
        fast, lq = ml_model.atr_frame_30m_from_hlc("T", hlc, now)
        assert lf == lq
        if full is None:
            assert fast is None
            continue
        for key in ("bar_ts", "close", "sday", "bar_idx", "n_bars"):
            assert fast[key] == full[key], key
        # bit for bit: the float the vol arm ranks (float32, as the scorer stores it) and its float64
        assert fast["features"]["atr_pct_14"] == full["features"]["atr_pct_14"]
        assert np.float32(fast["features"]["atr_pct_14"]) == np.float32(full["features"]["atr_pct_14"])


def test_the_atr_frame_keeps_the_full_frames_floor_and_labels():
    assert ml_model.atr_frame_30m_from_hlc("T", None, pd.Timestamp("2026-10-09 20:00")) == (None, "NO_DATA")
    hlc = _hlc(20, 5)                                                 # 260 bars < the 400-bar floor
    now = hlc[0][-1] + pd.Timedelta(minutes=30)
    assert ml_model.atr_frame_30m_from_hlc("T", hlc, now) == (None, "NO_DATA")
    assert ml_model.features_30m_from_hlc("T", hlc, now)[1] == "NO_DATA"


def test_ticker_feature_frame_builds_its_atr_with_the_shared_construction():
    from src.analysis import ml_dataset as md
    idx, h, lo, c, v = _hlc(40, 9)
    fs = md.ticker_feature_frame("T", hlc=(idx, h, lo, c, v))
    want = md.atr_pct_14(h.astype(float), lo.astype(float), c.astype(float)).values
    assert np.array_equal(fs["atr_pct_14"].to_numpy(), want, equal_nan=True)


# ── a name's live row is its full row's live part ────────────────────────────

class _Booster:
    def predict(self, M, num_iteration=None):
        return M[:, 0] * 0.0 + 0.5


def _ctx(monkeypatch, hlc_by_name, dv20, model_set=None):
    from src.analysis import deep_features as dfe
    monkeypatch.setattr(dfe, "serving_vector", lambda tk, sday, close, bar_idx, snap: {ss.DTC_FEATURE: 0.7})
    d = date(2026, 10, 2)
    bar = 4
    end = ss.bar_end_et(d, bar)
    now_naive = pd.Timestamp(end).tz_convert("UTC").tz_localize(None)
    loaded, c = {}, {}
    for tk, hlc in hlc_by_name.items():
        idx, h, lo, cl, v = hlc
        df = pd.DataFrame({"Open": cl.values, "High": h.values, "Low": lo.values, "Close": cl.values,
                           "Volume": v.values}, index=idx)
        loaded[tk] = (df, pd.DataFrame())
    return {"tickers": list(hlc_by_name), "d": d, "bar_of_day": bar, "dv20": dv20, "booster": _Booster(),
            "meta": {"features": ["atr_pct_14", "rsi_14"]}, "feats": ["atr_pct_14", "rsi_14"], "v2": False,
            "model_set": model_set, "day_ext": {}, "now_naive": now_naive, "target_start": now_naive - ss.BAR,
            "pre_day": ss.sessions_before(d, int(settings.sel_short_runup_sessions))[0], "loaded": loaded,
            "snap": object(), "vol_feature": "atr_pct_14"}


def test_live_rows_are_the_full_rows_live_fields(monkeypatch):
    names = {f"T{i}": _hlc(45, 10 + i, flat_every=5 if i == 2 else 0) for i in range(4)}
    names["SHORT"] = _hlc(20, 99)                                     # under the 400-bar floor: NO_DATA
    dv20 = {"T0": 1e7, "T1": 1e7, "T2": 2e6, "T3": 1e7, "SHORT": 1e7}
    c = _ctx(monkeypatch, names, dv20, model_set={"T0", "T1", "T2", "SHORT"})
    live = ss._score_rows(c, live_only=True)
    full = ss._score_rows(c)
    assert [r["ticker"] for r in live] == [r["ticker"] for r in full]
    statuses = {r["ticker"]: r["status"] for r in full}
    assert statuses["SHORT"] == "NO_DATA" and statuses["T2"] == "VOL_ONLY" and statuses["T3"] == "VOL_ONLY"
    assert statuses["T0"] == "OK"
    assert ss._live_rows_mismatch(pd.DataFrame(live), pd.DataFrame(full)) == []
    assert all(np.isnan(r["score"]) for r in live) and full[0]["score"] == 0.5    # only the full rows are scored


def test_a_mismatch_is_reported_field_by_field():
    a = pd.DataFrame({"ticker": ["A", "B"], "vol": [1.0, np.nan], "px": [5.0, 6.0], "status": ["OK", "VOL_ONLY"]})
    assert ss._live_rows_mismatch(a, a.copy()) == []
    b = a.copy()
    b.loc[1, "vol"] = 2.0
    out = ss._live_rows_mismatch(a, b)
    assert out == [{"ticker": "B", "field": "vol", "live": a.loc[1, "vol"], "full": 2.0}] or (
        len(out) == 1 and out[0]["ticker"] == "B" and out[0]["field"] == "vol")
    assert ss._live_rows_mismatch(a, b.iloc[:1])[0]["field"] == "names"


# ── run(): the live arms journaled first, the same decisions ─────────────────

def _res(n=25):
    return pd.DataFrame({"ticker": [f"N{i:02d}" for i in range(n)],
                         "score": np.linspace(1.0, 0.0, n),            # the model prefers N00
                         "vol": np.linspace(0.0, 5.0, n),               # the vol arm prefers N24
                         "px": 20.0, "dv20": 1e7, "pre5": 10.0, "status": "OK"})


@pytest.fixture
def arms_on(monkeypatch):
    monkeypatch.setattr(settings, "enable_sel_short", True)
    monkeypatch.setattr(settings, "enable_sel_short_vol", True)
    monkeypatch.setattr(settings, "enable_sel_short_etf", False)
    monkeypatch.setattr(settings, "sel_short_traded_arms", "vol")
    monkeypatch.setattr(ss, "load_universe", lambda day: {f"N{i:02d}": 1e7 for i in range(25)})
    monkeypatch.setattr(ss, "ensure_snapshot", lambda day, names: 1.0)
    return settings


_TIMING = ("seconds", "created_at")


def _same(u, v) -> bool:
    """Equal, NaN equal to NaN."""
    if isinstance(u, float) and isinstance(v, float) and np.isnan(u) and np.isnan(v):
        return True
    return u == v
_CROSS_ARM = ("other_arm_score", "other_arm_rank", "other_arm_n")   # journal-only: the model's view of a live pick


def _run_day(monkeypatch, tmp, live_first: bool, bars=(2, 3)):
    monkeypatch.setattr(settings, "sel_short_dir", str(tmp))
    monkeypatch.setattr(settings, "enable_sel_short_live_first_scoring", live_first)
    order = []

    def fake_phased(day, bar, uni, on_live):
        live = _res().assign(score=np.nan)
        on_live(live)
        order.append(("live_done", bar, [r["arm"] for r in ss.read_picks(day) if r["bar_of_day"] == bar]))
        return live, _res()
    monkeypatch.setattr(ss, "score_bar_phased", fake_phased)
    monkeypatch.setattr(ss, "score_bar", lambda day, bar, uni, fetch=True: _res())
    d = date(2026, 9, 28)
    runs = [ss.run(d, b) for b in bars]
    return d, runs, ss.read_picks(d), order


def test_live_first_journals_the_live_arms_first_with_the_same_decisions(monkeypatch, arms_on, tmp_path):
    d, runs_a, picks_a, _ = _run_day(monkeypatch, tmp_path / "one_phase", live_first=False)
    hist_a = {a: pd.read_pickle(ss.scores_path(d, a)) for a in ("model", "vol")}
    d, runs_b, picks_b, order = _run_day(monkeypatch, tmp_path / "live_first", live_first=True)
    hist_b = {a: pd.read_pickle(ss.scores_path(d, a)) for a in ("model", "vol")}
    # the vol arm's record was journaled BEFORE the model's (phase 1), every bar
    assert order == [("live_done", 2, ["vol"]), ("live_done", 3, ["vol"])]
    assert ss.live_done(d, 2) and ss.live_done(d, 3)

    def norm(r, cross=False):
        drop = _TIMING + (_CROSS_ARM if cross else ())
        return {k: ("NaN" if isinstance(v, float) and np.isnan(v) else v) for k, v in r.items() if k not in drop}
    key = lambda r: (r["bar_of_day"], r["arm"])                         # noqa: E731
    a = {key(r): r for r in picks_a}
    b = {key(r): r for r in picks_b}
    assert set(a) == set(b) and len(picks_a) == len(picks_b) == 4
    for k in a:
        live = k[1] == "vol"
        assert norm(a[k], cross=live) == norm(b[k], cross=live), k
    assert a[(2, "vol")]["decision"] == "short" and a[(2, "vol")]["ticker"] == "N24"
    # the score histories: identical
    for arm in hist_a:
        pd.testing.assert_frame_equal(hist_a[arm].reset_index(drop=True), hist_b[arm].reset_index(drop=True))
    # the run file: the same record and every arm, the live arm's from the marker
    for ra, rb in zip(runs_a, runs_b):
        diff = {k: (ra.get(k), rb.get(k)) for k in set(ra) | set(rb)
                if k not in _TIMING + ("arms",) and not _same(ra.get(k), rb.get(k))}
        assert diff == {}
        assert set(ra["arms"]) == set(rb["arms"]) == {"vol"}
        # the run file keeps the live pick's cross-arm view, read once the model's scores exist: the same values
        assert norm(ra["arms"]["vol"]) == norm(rb["arms"]["vol"])
        assert ra["arms"]["vol"]["other_arm_rank"] is not None


def test_a_failed_shadow_phase_is_rerun_without_journaling_the_live_arms_twice(monkeypatch, arms_on, tmp_path):
    monkeypatch.setattr(settings, "sel_short_dir", str(tmp_path))
    monkeypatch.setattr(settings, "enable_sel_short_live_first_scoring", True)

    def phased_then_crash(day, bar, uni, on_live):
        on_live(_res().assign(score=np.nan))
        raise RuntimeError("the shadow phase crashed")
    monkeypatch.setattr(ss, "score_bar_phased", phased_then_crash)
    monkeypatch.setattr(ss, "score_bar", lambda day, bar, uni, fetch=True: _res())
    d = date(2026, 9, 28)
    with pytest.raises(RuntimeError):
        ss.run(d, 2)
    assert ss.live_done(d, 2) and not ss.run_done(d, 2)
    assert [r["arm"] for r in ss.read_picks(d)] == ["vol"]
    rec = ss.run(d, 2)                                                  # the next tick's rerun
    assert ss.run_done(d, 2)
    assert sorted(r["arm"] for r in ss.read_picks(d)) == ["model", "vol"]  # the vol arm journaled once
    assert rec["arms"]["vol"]["ticker"] == "N24" and rec["ticker"] == "N00"
    assert len(pd.read_pickle(ss.scores_path(d, "vol"))) == 25            # its history appended once


def test_the_live_phase_takes_only_traded_arms_without_a_model_input(monkeypatch, arms_on):
    monkeypatch.setattr(settings, "enable_sel_short_live_first_scoring", True)
    monkeypatch.setattr(settings, "enable_sel_short_etf", True)
    monkeypatch.setattr(settings, "enable_sel_short_vol2", True)
    monkeypatch.setattr(settings, "enable_sel_short_fall", True)
    monkeypatch.setattr(settings, "sel_short_traded_arms", "vol,etf,fall,model")
    assert ss.live_phase_arms() == ["vol", "etf", "fall"]              # the model needs its scores; vol2 is shadow
    monkeypatch.setattr(settings, "sel_short_traded_arms", "")
    assert ss.live_phase_arms() == ["vol", "etf", "vol2", "fall"]      # every arm traded: every ATR% arm first
    monkeypatch.setattr(settings, "enable_sel_short_live_first_scoring", False)
    assert ss.live_phase_arms() == []


# ── the tick waits for the live arms' picks, not the shadow phase ────────────

class _Proc:
    def __init__(self):
        self.rc, self.pid = None, 1

    def poll(self):
        return self.rc


class _Log:
    def close(self):
        pass


def test_wait_returns_once_every_bar_has_its_live_picks(monkeypatch, arms_on, tmp_path):
    import time
    monkeypatch.setattr(settings, "sel_short_dir", str(tmp_path))
    d = date(2026, 9, 28)
    h = {"proc": _Proc(), "args": ["--run", "--day", d.isoformat(), "--bars", "2,3"], "started": time.time(),
         "log": _Log()}
    assert ss._launch_bars(h) == [(d, 2), (d, 3)]
    for b in (2, 3):
        p = ss.live_marker_path(d, b)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps({"arms": {}}), encoding="utf-8")
    t0 = time.time()
    assert ss.wait(h, timeout=30) is None and time.time() - t0 < 2      # the shadow phase still runs
    assert not h.get("done")
    h["proc"].rc = 0
    assert ss.wait(h, timeout=30) == 0 and h["done"]                    # its end, seen by a later wait


def test_wait_without_the_live_markers_waits_for_the_process(monkeypatch, arms_on, tmp_path):
    import threading
    import time
    monkeypatch.setattr(settings, "sel_short_dir", str(tmp_path))
    h = {"proc": _Proc(), "args": ["--run", "--day", "2026-09-28", "--bars", "2"], "started": time.time(),
         "log": _Log()}
    threading.Timer(0.5, lambda: setattr(h["proc"], "rc", 1)).start()
    assert ss.wait(h, timeout=30) == 1


# ── a worker's failure never hangs the scorer, and the shadow phase never cancels the live arms (2026-10-11 review) ──

def _fut(result=None, exc=None, done=True):
    from concurrent.futures import Future
    f = Future()
    if done:
        f.set_exception(exc) if exc is not None else f.set_result(result)
    return f


def test_collect_live_returns_every_chunks_live_rows():
    import queue
    q = queue.Queue()
    for i in range(3):
        q.put((i, [{"ticker": f"T{i}"}]))
    live = ss._collect_live(q, {i: _fut(done=False) for i in range(3)}, grace=0.0, poll=0.01)
    assert sorted(live) == [0, 1, 2] and live[2] == [{"ticker": "T2"}]


def test_a_worker_failing_before_its_live_rows_fails_the_bar():
    import queue
    q = queue.Queue()
    q.put((0, [{"ticker": "A"}]))
    with pytest.raises(ValueError, match="boom"):
        ss._collect_live(q, {0: _fut(done=False), 1: _fut(exc=ValueError("boom"))}, grace=0.0, poll=0.01)


def test_a_worker_failing_after_its_live_rows_never_cancels_the_live_arms():
    """The shadow phase's error (after the chunk's live rows were reported) surfaces at its result, once the live
    arms are decided — not here."""
    import queue
    q = queue.Queue()
    q.put((1, [{"ticker": "B"}]))
    q.put((0, [{"ticker": "A"}]))
    live = ss._collect_live(q, {0: _fut(exc=ValueError("shadow phase")), 1: _fut(done=False)}, grace=0.0, poll=0.01)
    assert sorted(live) == [0, 1]


def test_a_worker_ending_without_its_live_rows_fails_the_bar_after_the_grace():
    import queue
    with pytest.raises(RuntimeError, match="without reporting"):
        ss._collect_live(queue.Queue(), {0: _fut(result=[])}, grace=0.0, poll=0.01)


def test_live_rows_in_transit_arrive_within_the_grace():
    import queue
    import threading
    q = queue.Queue()
    threading.Timer(0.3, lambda: q.put((0, [{"ticker": "A"}]))).start()
    live = ss._collect_live(q, {0: _fut(exc=ValueError("shadow phase"))}, grace=5.0, poll=0.05)
    assert live == {0: [{"ticker": "A"}]}


def _put_big_and_return(i):
    """A worker that reports ~2 MB of live rows (far past a pipe's buffer) nobody reads, then ends."""
    from src.signals import sel_short as _ss
    _ss._LIVE_Q.put((i, ["x" * 1000] * 2000))
    return i


def test_a_worker_never_hangs_the_pools_shutdown_when_its_live_rows_are_not_read():
    """The parent stops reading the live queue when a bar fails; the other workers' rows still in the pipe must not
    hang their exit (the "joining processes that use queues" deadlock) — before the fix the pool's shutdown never
    returned, the scorer held busy.lock and no bar was scored again."""
    import multiprocessing as mp
    import threading
    from concurrent.futures import ProcessPoolExecutor
    ctx = mp.get_context("spawn")
    q = ctx.Queue()
    ex = ProcessPoolExecutor(1, mp_context=ctx, initializer=ss._init_live_queue, initargs=(q,))
    assert ex.submit(_put_big_and_return, 0).result(timeout=180) == 0
    t = threading.Thread(target=ex.shutdown, kwargs={"wait": True}, daemon=True)
    t.start()
    t.join(90)
    hung = t.is_alive()
    if hung:
        for proc in list((getattr(ex, "_processes", None) or {}).values()):
            proc.terminate()
    assert not hung


def test_a_live_mismatch_is_a_scorer_problem_in_the_email(monkeypatch, arms_on, tmp_path):
    from datetime import date as _date, datetime as _dt, timezone
    d = _date(2026, 10, 9)
    monkeypatch.setattr(ss, "is_session", lambda day: True)
    monkeypatch.setattr(ss, "latest_bar", lambda now: (d, 0))
    monkeypatch.setattr(ss, "read_snapshot_status", lambda day: None)
    p = ss.root() / "runs" / f"{d.isoformat()}_00.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps({"decision": "no_pick", "live_mismatch": [{"ticker": "AAA", "field": "vol"}]}),
                 encoding="utf-8")
    h = ss.health(now=_dt(2026, 10, 9, 14, 5, tzinfo=timezone.utc))
    assert any("live arms' rows differed" in x for x in h["problems"])
