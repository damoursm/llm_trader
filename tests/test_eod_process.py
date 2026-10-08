"""EOD maintenance as its own process (2026-10-07): the database fence between the
scheduler's ticks and the EOD process, the per-step record a relaunch resumes from, and
the scheduler's launch decision."""
import json
import os
import subprocess
import sys
import threading
import time
from datetime import date, datetime, time as dtime

import pytest

from config.settings import settings
from src.scheduler import db_fence, eod


# ── the database fence ───────────────────────────────────────────────────────

def test_pid_alive_tells_a_live_process_from_a_dead_one():
    assert db_fence.pid_alive(os.getpid())
    p = subprocess.Popen([sys.executable, "-c", "pass"])
    p.wait()
    assert not db_fence.pid_alive(p.pid)
    assert not db_fence.pid_alive(None) and not db_fence.pid_alive(0) and not db_fence.pid_alive("x")


def test_the_eod_side_waits_while_a_tick_holds_the_fence(monkeypatch):
    monkeypatch.setattr(db_fence, "POLL_SECONDS", 0.05)
    db_fence.raise_fence("tick")
    got = {}

    def eod_side():
        t0 = time.monotonic()
        with db_fence.db_turn():
            got["waited"] = time.monotonic() - t0
            got["busy"] = db_fence._read(db_fence.BUSY)

    th = threading.Thread(target=eod_side)
    th.start()
    time.sleep(0.6)
    assert "waited" not in got                       # still fenced out
    db_fence.drop_fence()
    th.join(timeout=5)
    assert got["waited"] >= 0.5
    assert got["busy"]["pid"] == os.getpid()
    assert db_fence._read(db_fence.BUSY) is None      # released with the connection


def test_a_tick_waits_out_the_eod_query_in_flight(monkeypatch):
    monkeypatch.setattr(db_fence, "POLL_SECONDS", 0.05)
    inside = threading.Event()

    def eod_side():
        with db_fence.db_turn():
            inside.set()
            time.sleep(0.8)

    th = threading.Thread(target=eod_side)
    th.start()
    assert inside.wait(5)
    db_fence.raise_fence("tick")
    waited = db_fence.wait_db_free(10)
    th.join(timeout=5)
    assert 0.4 <= waited < 5
    db_fence.drop_fence()


def test_flags_left_by_a_dead_process_or_too_old_are_ignored():
    p = subprocess.Popen([sys.executable, "-c", "pass"])
    p.wait()
    db_fence._write(db_fence.FENCE, {"pid": p.pid, "at": time.time()})
    db_fence._write(db_fence.BUSY, {"pid": p.pid, "at": time.time()})
    assert not db_fence.fence_up()
    assert db_fence.wait_db_free(5) < 0.5
    db_fence._write(db_fence.FENCE, {"pid": os.getpid(), "at": time.time() - db_fence.MAX_AGE_SECONDS - 60})
    assert not db_fence.fence_up()


def test_a_tick_stops_waiting_at_its_cap():
    db_fence._write(db_fence.BUSY, {"pid": os.getpid(), "at": time.time()})
    t0 = time.monotonic()
    waited = db_fence.wait_db_free(0.5)
    assert 0.5 <= waited < 3 and time.monotonic() - t0 < 3


def test_drop_fence_never_removes_another_process_fence():
    db_fence._write(db_fence.FENCE, {"pid": os.getpid() + 1, "at": time.time()})
    db_fence.drop_fence()
    assert db_fence._read(db_fence.FENCE) is not None


def test_connect_runs_through_the_gate_once_per_nesting(monkeypatch):
    from src.db import connection
    monkeypatch.setattr(db_fence, "POLL_SECONDS", 0.05)
    db_fence.install(lock_retries=9)
    assert connection._LOCK_RETRIES == 9
    with connection.connect() as c1:
        assert db_fence._read(db_fence.BUSY)["pid"] == os.getpid()
        db_fence.raise_fence("tick")                 # a tick arrives mid-query: the nested open must not wait
        with connection.connect() as c2:
            assert c2.execute("SELECT 1").fetchone()[0] == 1
        db_fence.drop_fence()
        assert c1.execute("SELECT 2").fetchone()[0] == 2
    assert db_fence._read(db_fence.BUSY) is None
    connection.set_gate(None)
    entered = []

    class _Gate:
        def __enter__(self):
            entered.append(1)

        def __exit__(self, *a):
            return False

    connection.set_gate(lambda: _Gate())
    with connection.connect():
        pass
    assert entered == [1]


# ── the chain and its record ─────────────────────────────────────────────────

def _fake_steps(monkeypatch, calls, fail=(), interrupt=()):
    def make(name):
        def fn(day):
            calls.append(name)
            if name in interrupt:
                raise KeyboardInterrupt(name)
            if name in fail:
                raise RuntimeError(f"{name} broke")
            return 7, f"{name} done"
        return fn
    monkeypatch.setattr(eod, "STEP_FNS", {s: make(s) for s in (eod.FLEX,) + eod.STEPS})
    monkeypatch.setattr(settings, "broker_mode", "ibkr_paper")
    for flag in ("enable_eod_replay_refresh", "enable_eod_walkforward", "enable_eod_shape_history",
                 "enable_eod_backtest_refresh", "enable_eod_spread_sweep"):
        monkeypatch.setattr(settings, flag, True)


def test_the_chain_records_every_step_and_finishes(monkeypatch):
    calls = []
    _fake_steps(monkeypatch, calls)
    st = eod.run(date(2026, 10, 6))
    assert sorted(calls) == sorted((eod.FLEX,) + eod.STEPS)
    assert [c for c in calls if c != eod.FLEX] == list(eod.STEPS)          # the chain's order
    assert all(st["steps"][s]["status"] == "ok" for s in (eod.FLEX,) + eod.STEPS)
    assert st["steps"]["replay"]["summary"] == "replay done" and st["steps"]["replay"]["result"] == 7
    assert st["finished"] and st["runs"][-1]["ended"]
    assert eod.current() is None                                          # cleared at the end


def test_a_relaunch_resumes_at_the_interrupted_step(monkeypatch):
    calls = []
    _fake_steps(monkeypatch, calls, interrupt=("replay",))
    with pytest.raises(KeyboardInterrupt):
        eod.run("2026-10-06")
    st = eod.load_state("2026-10-06")
    assert st["steps"]["replay"]["status"] == "running" and not st["finished"]
    assert "walkforward" not in st["steps"]
    calls2 = []
    _fake_steps(monkeypatch, calls2)
    st = eod.run("2026-10-06")
    assert [c for c in calls2 if c != eod.FLEX] == ["replay", "walkforward", "shape_history", "backtest",
                                                   "spread_sweep"]
    assert st["finished"] and len(st["runs"]) == 2


def test_a_failed_step_is_recorded_the_chain_goes_on_and_a_relaunch_skips_it(monkeypatch):
    calls = []
    _fake_steps(monkeypatch, calls, fail=("replay",))
    st = eod.run("2026-10-06")
    assert st["steps"]["replay"]["status"] == "failed" and "replay broke" in st["steps"]["replay"]["summary"]
    assert st["steps"]["backtest"]["status"] == "ok" and st["finished"]
    calls2 = []
    _fake_steps(monkeypatch, calls2)
    eod.run("2026-10-06")
    assert calls2 == []


def test_disabled_steps_are_recorded_off(monkeypatch):
    calls = []
    _fake_steps(monkeypatch, calls)
    monkeypatch.setattr(settings, "enable_eod_replay_refresh", False)
    monkeypatch.setattr(settings, "broker_mode", "off")
    st = eod.run("2026-10-06")
    assert st["steps"]["replay"]["status"] == "off" and "replay" not in calls
    assert st["steps"]["spread_sweep"]["status"] == "off" and "spread_sweep" not in calls
    assert st["finished"]


def test_named_steps_rerun_whatever_their_record(monkeypatch):
    calls = []
    _fake_steps(monkeypatch, calls)
    eod.run("2026-10-06")
    calls2 = []
    _fake_steps(monkeypatch, calls2)
    st = eod.run("2026-10-06", only=["replay"])
    assert calls2 == ["replay"] and st["steps"]["replay"]["status"] == "ok"


def test_walkforward_and_shape_history_are_pinned_to_the_market_day(monkeypatch):
    from src.analysis import walkforward
    from src.signals import rank_shaping
    got = {}
    monkeypatch.setattr(walkforward, "materialize", lambda **kw: got.setdefault("wf", kw) and 4)
    monkeypatch.setattr(rank_shaping, "materialize_shape_history", lambda **kw: got.setdefault("sh", kw) and 2)
    monkeypatch.setattr(settings, "walkforward_eod_days", 3)
    eod._walkforward(date(2026, 10, 6))
    eod._shape_history(date(2026, 10, 6))
    assert got["wf"] == {"start": "2026-10-03", "step_days": 1}
    assert got["sh"] == {"start": "2026-10-06", "end": "2026-10-06", "step_days": 1}


def test_one_eod_process_at_a_time(monkeypatch):
    fh = eod._try_lock(eod.DIR / eod.LOCK_NAME)
    try:
        assert eod.running()
        monkeypatch.setattr(eod, "_setup_logging", lambda day: None)
        assert eod.main(["--day", "2026-10-06"]) == 3
    finally:
        eod._unlock(fh)
    assert not eod.running()


def test_status_prints_the_record(capsys):
    eod.record("2026-10-06", "replay", "ok", "5 ticker-days rescored")
    assert eod.main(["--day", "2026-10-06", "--status"]) == 0
    out = json.loads(capsys.readouterr().out)
    assert out["steps"]["replay"]["summary"] == "5 ticker-days rescored"


# ── the scheduler's side ─────────────────────────────────────────────────────

@pytest.fixture
def fresh_runner(monkeypatch):
    import src.scheduler.runner as runner
    monkeypatch.setattr(runner, "_EOD_PROC_THREAD", None)
    monkeypatch.setattr(runner, "_EOD_LAUNCHES", {})
    monkeypatch.setattr(runner, "_EOD_GAVE_UP", set())
    monkeypatch.setattr(runner, "_EOD_ANNOUNCED", set())
    monkeypatch.setattr(runner, "_EOD_REFRESHED", set())
    monkeypatch.setattr(settings, "enable_eod_maintenance", True)
    return runner


def test_the_due_day(fresh_runner):
    due = fresh_runner._eod_due_day
    at = dtime(16, 20)
    assert due(datetime(2026, 10, 6, 16, 25), at) == date(2026, 10, 6)       # Tuesday after the trigger
    assert due(datetime(2026, 10, 6, 10, 0), at) == date(2026, 10, 5)        # before it: Monday's chain
    assert due(datetime(2026, 10, 5, 3, 0), at) == date(2026, 10, 2)         # Monday night: Friday's
    assert due(datetime(2026, 10, 10, 12, 0), at) == date(2026, 10, 9)       # Saturday: Friday's


def test_the_poll_launches_a_due_chain_once_and_respects_its_record(fresh_runner, monkeypatch):
    runner = fresh_runner
    launched = []
    monkeypatch.setattr(runner, "_eod_launch", lambda day, st: launched.append(day))
    now, at = datetime(2026, 10, 6, 16, 25), dtime(16, 20)
    runner._eod_poll(now, at)
    assert launched == [date(2026, 10, 6)]
    runner._eod_poll(now, at)                          # inside the relaunch gap: no second launch
    assert len(launched) == 1
    fh = eod._try_lock(eod.DIR / eod.LOCK_NAME)        # a running EOD process: never a second one
    try:
        runner._EOD_LAUNCHES.clear()
        runner._eod_poll(now, at)
        assert len(launched) == 1
    finally:
        eod._unlock(fh)
    for s in (eod.FLEX,) + eod.STEPS:
        eod.record("2026-10-06", s, "ok", "done")
    resets = []
    from src.analysis import market_relative
    from src.signals import aggregator
    monkeypatch.setattr(aggregator, "reset_winrate_filter_cache", lambda: resets.append("wr"))
    monkeypatch.setattr(market_relative, "reset_cache", lambda: resets.append("mr"))
    runner._EOD_LAUNCHES.clear()
    runner._eod_poll(now, at)
    runner._eod_poll(now, at)
    assert len(launched) == 1                          # finished: nothing to launch
    assert resets == ["wr", "mr"]                      # caches refreshed once per finished chain


def test_a_chain_that_keeps_dying_is_given_up(fresh_runner, monkeypatch):
    runner = fresh_runner
    launched = []
    monkeypatch.setattr(runner, "_eod_launch", lambda day, st: launched.append(day))
    monkeypatch.setattr(runner, "_EOD_RELAUNCH_GAP_SECONDS", 0.0)
    for _ in range(6):
        runner._eod_poll(datetime(2026, 10, 6, 16, 25), dtime(16, 20))
    assert len(launched) == runner._EOD_MAX_LAUNCHES


def test_a_tick_holds_the_fence_for_its_whole_run(fresh_runner, monkeypatch):
    runner = fresh_runner
    seen = []
    monkeypatch.setattr(settings, "tick_watchdog_seconds", 0)
    monkeypatch.setattr(runner, "run_pipeline", lambda **kw: seen.append(db_fence.fence_is_ours()))
    runner._run_tick_watchdogged(send_email=False)
    assert seen == [True]
    assert not db_fence.fence_is_ours()


def test_the_fence_rises_ahead_of_a_slot_and_drops_after(fresh_runner, monkeypatch):
    runner = fresh_runner
    monkeypatch.setattr(settings, "eod_tick_fence_lead_seconds", 120)
    slots = [(dtime(11, 0), "rth")]
    runner._eod_fence_ahead(datetime(2026, 10, 6, 10, 59), slots)
    assert db_fence.fence_is_ours()
    runner._eod_fence_ahead(datetime(2026, 10, 6, 11, 5), slots)
    assert not db_fence.fence_is_ours()


def test_step_outcomes_reach_the_scheduler_log_once(fresh_runner):
    from loguru import logger
    lines = []
    sink = logger.add(lambda m: lines.append(str(m)), level="INFO")
    try:
        eod.record("2026-10-06", "replay", "ok", "9 ticker-days rescored", finished="2999-01-01T00:00:00+00:00")
        eod.record("2026-10-06", "retention", "ok", "old news", finished="2000-01-01T00:00:00+00:00")
        fresh_runner._eod_announce("2026-10-06")
        fresh_runner._eod_announce("2026-10-06")
    finally:
        logger.remove(sink)
    hits = [l for l in lines if "EOD replay (2026-10-06): 9 ticker-days rescored" in l]
    assert len(hits) == 1
    assert not any("old news" in l for l in lines)       # finished before this process started
