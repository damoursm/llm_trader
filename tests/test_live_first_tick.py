"""The live path first, the shadow after, and the next slot's live path on time (2026-10-12).

User 2026-10-11: "improve the speed of the algorithm so that trades are ordered faster. That
means we need to run first only the necessary parts for live production, and after we do the
shadow arms, LLM scoring ..., any kind of backfill, etc. ... MAKE SURE THE EXACT SAME LOGIC IS
APPLIED FOLLOWING YOUR CHANGES."

* `run_pipeline` runs the WHOLE live path (marks, exits, entries after the scorer, broker sync)
  before any shadow step (structural test in tests/test_sel_short.py).
* While the shadow runs a long step, the scheduler's hook runs a slot that came due meanwhile —
  that slot's live path alone (`pipeline.run_live_path`), on time; its full tick still follows.
"""
from __future__ import annotations

import threading
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, time as dtime

import pytest

import src.pipeline as pl
from config.settings import settings


@pytest.fixture(autouse=True)
def _no_hook():
    pl.set_live_yield(None)
    yield
    pl.set_live_yield(None)


# ── the shadow step on a worker thread ───────────────────────────────────────

def test_shadow_call_without_a_hook_is_the_direct_call():
    seen = []
    out = pl._shadow_call(lambda a, b=0: seen.append(threading.current_thread()) or a + b, 2, b=3)
    assert out == 5 and seen == [threading.current_thread()]


def test_shadow_call_with_a_hook_runs_the_step_off_the_main_thread_and_yields(monkeypatch):
    monkeypatch.setattr(pl, "_YIELD_POLL_SECONDS", 0.02)
    yields, where = [], []
    pl.set_live_yield(lambda: yields.append(threading.current_thread()))

    def step(x):
        where.append(threading.current_thread())
        time.sleep(0.15)
        return x * 2
    assert pl._shadow_call(step, 21) == 42
    assert where and where[0] is not threading.current_thread()
    assert yields and all(t is threading.current_thread() for t in yields)


def test_shadow_call_raises_the_steps_own_exception_even_a_timeout(monkeypatch):
    """A step that raises TimeoutError must surface it — not be mistaken for the wait's own
    timeout and waited on forever."""
    monkeypatch.setattr(pl, "_YIELD_POLL_SECONDS", 0.02)
    pl.set_live_yield(lambda: None)

    def boom():
        raise TimeoutError("the step's own")
    with pytest.raises(TimeoutError, match="the step's own"):
        pl._shadow_call(boom)

    def bad():
        raise ValueError("x")
    with pytest.raises(ValueError):
        pl._shadow_call(bad)


def test_a_failing_hook_never_breaks_the_shadow(monkeypatch):
    monkeypatch.setattr(pl, "_YIELD_POLL_SECONDS", 0.02)

    def hook():
        raise RuntimeError("live path failed")
    pl.set_live_yield(hook)
    assert pl._shadow_call(lambda: time.sleep(0.1) or "done") == "done"


def test_the_pool_wait_yields_until_every_future_is_done(monkeypatch):
    monkeypatch.setattr(pl, "_YIELD_POLL_SECONDS", 0.02)
    yields = []
    pl.set_live_yield(lambda: yields.append(1))
    with pl._TrackedPool(max_workers=2) as pool:
        a = pool.submit(time.sleep, 0.1)
        b = pool.submit(lambda: 7)
        assert pool.futures == [a, b]
        pl._await_futures(pool.futures + [None])
        assert a.done() and b.done()
    assert yields


def test_the_pool_wait_without_a_hook_leaves_it_to_the_pool_exit():
    with pl._TrackedPool(max_workers=1) as pool:
        f = pool.submit(time.sleep, 0.05)
        pl._await_futures(pool.futures)                    # returns at once
    assert f.done()                                        # the exit waited, as before


# ── a due slot's live path ───────────────────────────────────────────────────

def test_run_live_path_runs_the_tick_starts_steps_in_order_and_no_per_run_reset(monkeypatch):
    from src.signals import sel_short as ss
    monkeypatch.setattr(settings, "enable_live_path_first", True)
    calls = []
    monkeypatch.setattr(ss, "launch", lambda start: calls.append("launch") or {"h": 1})
    monkeypatch.setattr(pl, "_start_borrow_snapshot", lambda record=True: calls.append(("borrow", record)) or "F")
    monkeypatch.setattr(pl, "_live_marks_and_exits", lambda: calls.append("marks") or 2)
    monkeypatch.setattr(pl, "_live_entries_and_sync",
                        lambda run_id, h, f, dip_closed=0: calls.append(("entries", h, f, dip_closed)) or {"mode": "x"})
    monkeypatch.setattr(pl, "_persist_live_report", lambda run_id, rep: calls.append(("persist", rep)))
    for name in ("_reset_source_log", "reset_sentiment_providers", "set_current_run", "reset_price_health"):
        monkeypatch.setattr(pl, name, lambda *a, **k: pytest.fail("a due slot's live path reset the running "
                                                                    "tick's per-run state"))
    assert pl.run_live_path() == {"mode": "x"}
    assert calls == ["launch", ("borrow", False), "marks", ("entries", {"h": 1}, "F", 2),
                     ("persist", {"mode": "x"})]


def test_run_live_path_is_off_with_the_live_path_first_off(monkeypatch):
    monkeypatch.setattr(settings, "enable_live_path_first", False)
    monkeypatch.setattr(pl, "_live_marks_and_exits", lambda: pytest.fail("ran"))
    assert pl.run_live_path() is None


def test_the_live_report_is_persisted_under_its_run_id(monkeypatch):
    seen = []
    monkeypatch.setattr(pl.repo, "insert_broker_report", lambda run_id, rep: seen.append((run_id, rep)))
    pl._persist_live_report("r1", {"mode": "ibkr_paper", "orders": []})
    pl._persist_live_report("r2", None)
    pl._persist_live_report("r3", {"mode": "off"})
    assert seen == [("r1", {"mode": "ibkr_paper", "orders": []})]


def test_the_borrow_file_is_fetched_on_its_own_thread(monkeypatch):
    from src.data import ibkr_borrow
    monkeypatch.setattr(settings, "enable_ibkr_borrow_snapshot", True)
    seen = []
    monkeypatch.setattr(ibkr_borrow, "snapshot", lambda: seen.append(threading.current_thread()) or "file")
    f = pl._start_borrow_snapshot(record=False)
    assert f.result(timeout=5) == "file" and seen[0] is not threading.current_thread()
    monkeypatch.setattr(settings, "enable_ibkr_borrow_snapshot", False)
    assert pl._start_borrow_snapshot() is None


# ── the scheduler's hook ─────────────────────────────────────────────────────

@pytest.fixture
def hooked(monkeypatch):
    from src import pipeline
    from src.scheduler import runner
    runs = []
    monkeypatch.setattr(settings, "enable_live_path_first", True)
    monkeypatch.setattr(pipeline, "run_live_path", lambda: runs.append(clock["now"]))
    clock = {"now": datetime(2026, 10, 12, 10, 2)}
    monkeypatch.setattr(runner, "now_et", lambda: clock["now"])
    monkeypatch.setattr(runner, "_slot_is_valid", lambda day, t, kind: True)
    slots = [(dtime(10, 0), "rth"), (dtime(10, 30), "rth"), (dtime(11, 0), "rth")]
    state = {"running": datetime(2026, 10, 12, 10, 0), "live_done": set()}
    hook = runner._make_live_yield(slots, 1500, dtime(16, 0), state)
    return hook, runs, clock, state, runner


def test_the_hook_runs_a_due_slots_live_path_once(hooked):
    hook, runs, clock, state, _ = hooked
    hook()                                            # 10:02: still the running tick's slot
    assert runs == []
    clock["now"] = datetime(2026, 10, 12, 10, 30, 6)
    hook()
    hook()                                            # once per slot
    assert runs == [datetime(2026, 10, 12, 10, 30, 6)]
    assert datetime(2026, 10, 12, 10, 30) in state["live_done"]
    clock["now"] = datetime(2026, 10, 12, 11, 0, 4)
    hook()
    assert len(runs) == 2


def test_the_hook_leaves_a_slot_past_the_grace_and_an_observation_slot(hooked, monkeypatch):
    hook, runs, clock, state, runner = hooked
    clock["now"] = datetime(2026, 10, 12, 10, 56)     # 10:30 slot 26 min late (> 25 min grace)
    hook()
    assert runs == []
    clock["now"] = datetime(2026, 10, 12, 11, 0, 5)
    monkeypatch.setattr(runner, "_tick_plan", lambda kind, t, end_t: (True, False))
    hook()
    assert runs == [] and datetime(2026, 10, 12, 11, 0) in state["live_tried"]
    assert datetime(2026, 10, 12, 11, 0) not in state["live_done"]        # nothing ran: its alert stays possible


def test_the_hook_pushes_the_running_ticks_watchdog_back(hooked, monkeypatch):
    hook, runs, clock, state, runner = hooked
    from src import pipeline
    monkeypatch.setattr(pipeline, "run_live_path", lambda: time.sleep(0.2))
    runner._TICK_DEADLINE["at"] = 100.0
    clock["now"] = datetime(2026, 10, 12, 10, 30, 3)
    hook()
    assert runner._TICK_DEADLINE["at"] >= 100.2
    runner._TICK_DEADLINE["at"] = None


def test_a_slot_tick_installs_the_hook_for_its_run_only(monkeypatch):
    from src.scheduler import runner
    seen = []
    monkeypatch.setattr(runner, "_run_tick_watchdogged", lambda **kw: seen.append((pl._LIVE_YIELD, kw)))
    state = {"running": None, "live_done": {datetime(2026, 10, 1, 10, 0), datetime(2026, 10, 12, 10, 0)}}
    hook = object()
    monkeypatch.setattr(settings, "enable_live_slot_yield", True)
    runner._run_slot_tick(datetime(2026, 10, 12, 10, 30), hook, state, send_email=False)
    assert seen == [(hook, {"send_email": False})]
    assert pl._LIVE_YIELD is None and state["running"] is None
    assert state["live_done"] == {datetime(2026, 10, 12, 10, 0)}          # old slots pruned
    # the switch off: no hook — every shadow step inline, the slot waits for the tick (as before)
    seen.clear()
    monkeypatch.setattr(settings, "enable_live_slot_yield", False)
    runner._run_slot_tick(datetime(2026, 10, 12, 11, 0), hook, state, send_email=False)
    assert seen == [(None, {"send_email": False})]


def test_the_watchdog_kills_past_its_deadline_and_a_push_spares_the_tick(monkeypatch):
    import os

    from src.scheduler import runner
    killed = []
    monkeypatch.setattr(os, "_exit", lambda code: killed.append(code))
    monkeypatch.setattr(settings, "tick_watchdog_seconds", 1)
    monkeypatch.setattr(runner, "run_pipeline", lambda **kw: time.sleep(2.6))
    runner._run_tick_under_watchdog()
    assert killed == [1]

    killed.clear()

    def pushed(**kw):
        runner._extend_tick_deadline(5.0)              # a due slot's live path took 5 s
        time.sleep(2.6)
    monkeypatch.setattr(runner, "run_pipeline", pushed)
    runner._run_tick_under_watchdog()
    assert killed == [] and runner._TICK_DEADLINE["at"] is None


def test_the_scheduler_polls_every_few_seconds():
    assert type(settings).model_fields["scheduler_poll_seconds"].default == 5


def test_the_hook_pushes_the_deadline_before_the_live_path_and_settles_it_after(hooked, monkeypatch):
    """The running tick's deadline is pushed by the live path's worst case BEFORE it runs (the watchdog never kills
    the scheduler in its middle), then set to the time it really took."""
    hook, runs, clock, state, runner = hooked
    from src import pipeline
    seen = []
    monkeypatch.setattr(pipeline, "run_live_path", lambda: seen.append(runner._TICK_DEADLINE["at"]) or time.sleep(0.2))
    runner._TICK_DEADLINE["at"] = 100.0
    clock["now"] = datetime(2026, 10, 12, 10, 30, 3)
    try:
        hook()
        assert seen and seen[0] >= 100.0 + runner._live_path_budget_seconds() - 1e-6
        assert 100.2 <= runner._TICK_DEADLINE["at"] < 100.0 + runner._live_path_budget_seconds()
    finally:
        runner._TICK_DEADLINE["at"] = None


def test_a_live_path_that_raised_is_not_retried_and_keeps_the_missed_alert(hooked, monkeypatch):
    hook, runs, clock, state, runner = hooked
    from src import pipeline
    calls = []

    def boom():
        calls.append(1)
        raise RuntimeError("gateway down")
    monkeypatch.setattr(pipeline, "run_live_path", boom)
    clock["now"] = datetime(2026, 10, 12, 10, 30, 3)
    with pytest.raises(RuntimeError):
        hook()                                         # (the pipeline's yield point swallows it)
    hook()
    assert calls == [1]                                # tried once
    assert datetime(2026, 10, 12, 10, 30) in state["live_tried"]
    assert datetime(2026, 10, 12, 10, 30) not in state["live_done"]


def test_step0_pins_the_trades_open_before_this_ticks_entries():
    """As when the entries ran beside Steps 1-3: the tick's new shorts never join its shadow universe (nor its
    price-provenance check against a snapshot taken minutes after their entry)."""
    import inspect
    src = inspect.getsource(pl.run_pipeline)
    i_marks = src.index("_dip_closed = _live_marks_and_exits()")
    i_open = src.index("_open_at_step0 = get_open_trade_tickers()")
    i_entries = src.index("live_broker_report = _live_entries_and_sync(")
    assert i_marks < i_open < i_entries
    assert "open_trade_tickers = _open_at_step0 if _open_at_step0 is not None else get_open_trade_tickers()" in src


def test_a_hooked_live_paths_broker_problem_is_critical(monkeypatch):
    from loguru import logger
    from src.signals import sel_short as ss
    monkeypatch.setattr(settings, "enable_live_path_first", True)
    monkeypatch.setattr(ss, "launch", lambda start: None)
    monkeypatch.setattr(pl, "_start_borrow_snapshot", lambda record=True: None)
    monkeypatch.setattr(pl, "_live_marks_and_exits", lambda: 0)
    monkeypatch.setattr(pl, "_live_entries_and_sync",
                        lambda run_id, h, f, dip_closed=0: {"mode": "ibkr_paper", "connected": False})
    monkeypatch.setattr(pl, "_persist_live_report", lambda run_id, rep: None)
    msgs = []
    sink = logger.add(lambda m: msgs.append(m.record), level="CRITICAL")
    try:
        pl.run_live_path()
    finally:
        logger.remove(sink)
    assert any("EXECUTION ISSUE" in r["message"] and "NOT connected" in r["message"] for r in msgs)
