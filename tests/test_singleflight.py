"""Cache stampede on the heavy panel calibrations (2026-08-04).

`build_signals` runs CONCURRENTLY — the main pass, the `_HoldReviewBranch`
(overlapped with main steps 4-5 by design) and the shadow-arm branch. A plain
"check TTL → compute → store" cache does not stop duplicate work there: every
concurrent caller checks the same EMPTY cache, all miss, all compute. Measured on
a live 861 s tick: the win-rate filter ran 4x, two logging in the SAME SECOND.
"""

from __future__ import annotations

import threading
import time

from src.utils import ttl_single_flight


def _hammer(fn, n_threads=8):
    """Run fn() from n threads at once; return the results."""
    out, barrier = [], threading.Barrier(n_threads)
    def run():
        barrier.wait()                 # release all threads simultaneously
        out.append(fn())
    ts = [threading.Thread(target=run) for _ in range(n_threads)]
    for t in ts: t.start()
    for t in ts: t.join()
    return out


def test_concurrent_callers_share_one_computation():
    cache, lock, calls = {}, threading.Lock(), []
    def compute():
        calls.append(1)
        time.sleep(0.05)               # long enough that the others pile up
        return "value"
    res = _hammer(lambda: ttl_single_flight(lock, cache, "k", 999.0, compute))
    assert res == ["value"] * 8        # everyone got the same answer
    assert len(calls) == 1, f"stampede: computed {len(calls)}x instead of once"


def test_fresh_cache_never_takes_the_lock():
    # The steady state must stay lock-free, or serialising the miss path would
    # serialise every scorer call too.
    cache = {"k": {"ts": time.time(), "val": "fresh"}}
    class Boom:
        def __enter__(self): raise AssertionError("took the lock on a fresh hit")
        def __exit__(self, *a): pass
    assert ttl_single_flight(Boom(), cache, "k", 999.0, lambda: "recomputed") == "fresh"


def test_expired_entry_is_recomputed():
    cache = {"k": {"ts": time.time() - 500, "val": "stale"}}
    got = ttl_single_flight(threading.Lock(), cache, "k", 100.0, lambda: "fresh")
    assert got == "fresh" and cache["k"]["val"] == "fresh"


def test_a_failing_compute_is_not_cached():
    cache, lock, n = {}, threading.Lock(), []
    def boom():
        n.append(1)
        raise RuntimeError("panel unavailable")
    for _ in range(2):
        try:
            ttl_single_flight(lock, cache, "k", 999.0, boom)
        except RuntimeError:
            pass
    assert "k" not in cache            # nothing stored
    assert len(n) == 2                 # retried rather than serving a cached failure


def test_aggregator_winrate_filter_is_single_flight():
    """The real call site. One computation internally makes SEVERAL heavy passes
    (9 against the live ledger, fewer on an empty test DB), so the invariant is
    not "exactly one pass" — it is that N concurrent callers cost the SAME as one
    sequential caller. That is what the stampede broke."""
    from config.settings import settings
    import src.signals.aggregator as agg
    import src.performance.tracker as tracker

    settings.enable_winrate_method_filter = True
    orig = tracker.compute_solo_method_gross_winrate
    calls = []
    def counted(*a, **k):
        calls.append(1)
        time.sleep(0.02)                       # widen the stampede window
        return orig(*a, **k)
    tracker.compute_solo_method_gross_winrate = counted
    try:
        agg.reset_winrate_filter_cache()
        calls.clear()
        agg.winrate_filtered_methods()          # 1 sequential caller = the baseline
        baseline = len(calls)

        agg.reset_winrate_filter_cache()
        calls.clear()
        res = _hammer(agg.winrate_filtered_methods, n_threads=8)
        assert all(r == res[0] for r in res), "concurrent callers disagreed"
        # <= because the baseline run also warms DOWNSTREAM caches, so the
        # concurrent pass can legitimately need fewer. A stampede would be ~8x
        # baseline, so this still separates the two cleanly.
        assert len(calls) <= baseline, (
            f"stampede: 8 concurrent callers did {len(calls)} heavy passes, "
            f"a single caller does {baseline}")
    finally:
        tracker.compute_solo_method_gross_winrate = orig
        agg.reset_winrate_filter_cache()
