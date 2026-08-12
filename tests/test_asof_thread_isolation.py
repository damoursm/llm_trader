"""The as-of cutoff must NOT leak between threads (2026-08-04).

`_ASOF` was a module GLOBAL. The nightly refactor runs its walk-forward in a
BACKGROUND THREAD (`runner._maybe_start_nightly_rescore`), so while it stepped
through history every CONCURRENT LIVE TICK saw the same cutoff and read a
TRUNCATED ledger:

    2026-08-02 22:41:39  nightly rescore STARTING in background
    2026-08-02 22:44:19  tick raised: save_trades refused ... shrink 356 -> ...

The save_trades shrink-guard caught the WRITE — the only reason history survived.
The silent half is worse: for the ~40 min a rescore runs, every overlapping tick
calibrated, sized and monitored positions against a ledger cut off weeks back.
"""

from __future__ import annotations

import ast
import threading
from pathlib import Path

from src.analysis.asof import analysis_asof, asof_sql_clause, before_cutoff, current_asof


def test_a_cutoff_does_not_leak_into_another_thread():
    """The live tick's thread must stay unrestricted while a walk-forward runs."""
    seen, ready, done = {}, threading.Event(), threading.Event()

    def live_tick():
        ready.wait(5)
        seen["asof"] = current_asof()
        seen["sql"] = asof_sql_clause("signal_date")
        seen["before"] = before_cutoff("2026-08-01")     # after the cutoff
        done.set()

    t = threading.Thread(target=live_tick)
    t.start()
    with analysis_asof("2026-06-01"):                    # walk-forward thread
        assert current_asof() == "2026-06-01"            # bound HERE
        ready.set()
        done.wait(5)
    t.join(5)

    assert seen["asof"] is None, "cutoff leaked into a concurrent thread"
    assert seen["sql"] == "", "cutoff leaked into another thread's SQL"
    assert seen["before"] is True, "another thread's rows were wrongly filtered"


def test_cutoff_still_binds_the_thread_that_set_it():
    with analysis_asof("2026-06-01"):
        assert current_asof() == "2026-06-01"
        assert asof_sql_clause("signal_date") == " AND signal_date < '2026-06-01'"
        assert before_cutoff("2026-05-31") is True
        assert before_cutoff("2026-06-02") is False
    assert current_asof() is None                        # restored on exit


def test_nesting_restores_the_outer_cutoff():
    with analysis_asof("2026-06-01"):
        with analysis_asof("2026-05-01"):
            assert current_asof() == "2026-05-01"
        assert current_asof() == "2026-06-01"
    assert current_asof() is None


def test_no_worker_pool_under_the_asof_read_path():
    """Thread-local is only safe because nothing beneath a cutoff fans out to a
    pool — a pool's workers would NOT inherit it and would silently read the
    future, which is exactly the look-ahead this module exists to prevent. That
    was the stated reason the cutoff was global, so it must stay mechanically
    true: if you add a pool here, propagate the cutoff into the workers."""
    root = Path(__file__).resolve().parent.parent
    watched = ["src/analysis/walkforward.py", "src/analysis/signal_panel.py",
               "src/analysis/simulated_trades.py", "src/performance/tracker.py",
               "src/analysis/market_relative.py"]
    banned = {"ThreadPoolExecutor", "ProcessPoolExecutor"}
    offenders = []
    for rel in watched:
        tree = ast.parse((root / rel).read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            name = None
            if isinstance(node, ast.Name):
                name = node.id
            elif isinstance(node, ast.Attribute):
                name = node.attr
            if name in banned:
                offenders.append(f"{rel}:{getattr(node, 'lineno', '?')} {name}")
    assert not offenders, (
        "a worker pool appeared under the as-of read path; its workers will NOT "
        "inherit the thread-local cutoff and will read the future: " + ", ".join(offenders))
