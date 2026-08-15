"""The dashboard warmer must never run heavy work in the web process (2026-08-14).

The outage this pins: the warm sweep (~400 s of pandas across 9 accessors, 1,587 s
against a busy pipeline tick) ran on a background THREAD inside waitress. Python's
GIL meant a `py-spy` dump showed all 8 workers idle and one thread `active+gil` in
`warm_caches`, while a **static 4 KB file timed out from localhost** — the server
could not even accept the connection. With the sweep grown past the ~30-minute
pipeline tick, the dashboard was unreachable nearly all the time.

Two things fix it and BOTH are required; either alone leaves the outage in place:
  1. the sweep runs in a child process (its own GIL);
  2. `_cached` serves the previous snapshot while that child runs, instead of
     recomputing on the request thread — otherwise the first visitor after every
     run simply re-does the whole sweep in the web process anyway.
"""

import pickle
import time

import pytest

from dashboard import data
from src.db import repo

# dashboard.data flips the repo read-only process-wide at import, which is right
# for the dashboard and wrong for the rest of the suite.
repo.set_read_only(False)


@pytest.fixture(autouse=True)
def _clean_cache():
    """Each test gets its own cache and warm state."""
    saved_cache = dict(data._perf_cache)
    saved_state = dict(data._warm_state)
    saved_ver = dict(data._data_ver)
    data._perf_cache.clear()
    yield
    data._perf_cache.clear()
    data._perf_cache.update(saved_cache)
    data._warm_state.clear()
    data._warm_state.update(saved_state)
    data._data_ver.clear()
    data._data_ver.update(saved_ver)


def _pin_version(monkeypatch, ver):
    monkeypatch.setattr(data, "_data_version", lambda: ver)


# ── the stale-serving policy ─────────────────────────────────────────────────

def test_fresh_entry_is_served_without_recomputing(monkeypatch):
    _pin_version(monkeypatch, "run-1")
    calls = []
    assert data._cached("k", lambda: calls.append(1) or "v1") == "v1"
    assert data._cached("k", lambda: calls.append(1) or "v2") == "v1"
    assert len(calls) == 1


def test_new_run_serves_the_stale_snapshot_while_a_warm_is_in_flight(monkeypatch):
    """THE regression. A new run landed and the warmer is rebuilding: the request
    thread must hand back the old value, not spend ~400 s recomputing under the
    GIL while every other request — including static files — waits."""
    _pin_version(monkeypatch, "run-1")
    data._cached("k", lambda: "old")

    _pin_version(monkeypatch, "run-2")               # a new pipeline run landed
    data._warm_state["in_flight"] = True
    called = []
    got = data._cached("k", lambda: called.append(1) or "new")
    assert got == "old", "recomputed on the request thread instead of serving stale"
    assert not called, "producer ran in the web process during a warm"


def test_new_run_recomputes_when_no_warm_is_running(monkeypatch):
    """The flip side: if nothing is rebuilding, stale data must not be served
    forever — the cache still refreshes itself on demand."""
    _pin_version(monkeypatch, "run-1")
    data._cached("k", lambda: "old")
    _pin_version(monkeypatch, "run-2")
    data._warm_state["in_flight"] = False
    assert data._cached("k", lambda: "new") == "new"


def test_stale_grace_is_capped(monkeypatch):
    """A permanently broken warmer must not pin the dashboard to ancient numbers:
    past the grace window, correctness wins and we pay the recompute."""
    _pin_version(monkeypatch, "run-1")
    data._cached("k", lambda: "old")
    data._perf_cache["k"]["ts"] = time.time() - data._WARM_STALE_GRACE - 1
    _pin_version(monkeypatch, "run-2")
    data._warm_state["in_flight"] = True
    assert data._cached("k", lambda: "new") == "new"


def test_missing_entry_still_computes_during_a_warm(monkeypatch):
    """Stale-serving must never invent a value that was never computed."""
    _pin_version(monkeypatch, "run-1")
    data._warm_state["in_flight"] = True
    assert data._cached("never-seen", lambda: "computed") == "computed"


# ── the sweep must leave the web process ─────────────────────────────────────

def test_warm_loop_does_not_call_warm_caches_in_process():
    """`_warm_loop` calling `warm_caches()` directly IS the outage. An in-process
    sweep is invisible from every other surface — the logs read normally and the
    workers look idle — so it is asserted at the source."""
    import inspect
    src = inspect.getsource(data._warm_loop)
    assert "warm_caches(" not in src, "the sweep is back on the web process's GIL"
    assert "_warm_in_subprocess" in src


def test_subprocess_failure_does_not_fall_back_to_an_in_process_sweep():
    """A fallback to in-process warming would quietly restore the outage on
    exactly the days the child is failing."""
    import inspect
    src = inspect.getsource(data._warm_in_subprocess)
    assert "warm_caches(" not in src


def test_in_flight_is_cleared_even_when_the_child_fails(monkeypatch):
    """A stuck `in_flight` would make `_cached` serve stale data forever."""
    def boom(*a, **k):
        raise OSError("no python")
    monkeypatch.setattr("subprocess.run", boom)
    data._warm_state["in_flight"] = False
    data._warm_in_subprocess("run-9")
    assert data._warm_state["in_flight"] is False


def test_merge_skips_one_bad_entry_and_keeps_the_rest():
    """One unpicklable accessor must cost that accessor only, not the sweep."""
    blob = {"cache": {
        "good1": pickle.dumps({"ts": 1.0, "data": "A", "ver": "v"}),
        "bad": b"not-a-pickle",
        "good2": pickle.dumps({"ts": 2.0, "data": "B", "ver": "v"}),
    }}
    assert data._merge_snapshot(blob, "test") == 2
    assert data._perf_cache["good1"]["data"] == "A"
    assert data._perf_cache["good2"]["data"] == "B"
    assert "bad" not in data._perf_cache


def test_snapshot_round_trips_so_a_restart_starts_warm(tmp_path, monkeypatch):
    """Without the persisted snapshot the first load after every restart re-does
    the whole sweep on the request thread — the ~400 s hang that reads as
    'Loading…' forever on a phone."""
    monkeypatch.setattr(data, "_repo_root", lambda: str(tmp_path))
    src = tmp_path / "src.pkl"
    with open(src, "wb") as fh:
        pickle.dump({"ver": "run-1", "cache": {
            "k": pickle.dumps({"ts": time.time(), "data": "warmed", "ver": "run-1"})}}, fh)

    data._save_snapshot(str(src))
    data._perf_cache.clear()
    data._load_snapshot()
    assert data._perf_cache["k"]["data"] == "warmed"


def test_load_snapshot_is_silent_when_absent_or_corrupt(tmp_path, monkeypatch):
    """A missing or half-written snapshot must never stop the dashboard booting."""
    monkeypatch.setattr(data, "_repo_root", lambda: str(tmp_path))
    data._load_snapshot()                            # absent
    (tmp_path / "cache").mkdir(exist_ok=True)
    (tmp_path / "cache" / "dashboard_warm.pkl").write_bytes(b"garbage")
    data._load_snapshot()                            # corrupt
    assert data._perf_cache == {}


def test_worker_writes_individually_pickled_entries(tmp_path, monkeypatch):
    """Train/serve parity for the cache: the child must emit exactly the keys and
    entry shape the parent merges into `_perf_cache`."""
    from dashboard import warm_worker

    monkeypatch.setattr(data, "_data_version", lambda: "run-7")
    monkeypatch.setattr(data, "warm_caches",
                        lambda reason="": data._perf_cache.update(
                            {"acc": {"ts": 1.0, "data": [1, 2, 3], "ver": "run-7"}}))
    out = tmp_path / "out.pkl"
    assert warm_worker.main(str(out)) == 0

    with open(out, "rb") as fh:
        blob = pickle.load(fh)
    assert blob["ver"] == "run-7"
    data._perf_cache.clear()
    assert data._merge_snapshot(blob, "test") == 1
    assert data._perf_cache["acc"]["data"] == [1, 2, 3]
