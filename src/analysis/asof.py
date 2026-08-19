"""Point-in-time ("as of") data cutoff — the mechanism behind walk-forward.

A backtest that applies TODAY's weights to ALL of history is measuring a
configuration that could not have existed: those weights were calibrated on data
that, relative to any historical row, is the FUTURE. Production never has that
luxury — it recalibrates every tick from whatever had happened so far.

This module makes "whatever had happened so far" enforceable. `analysis_asof(d)`
installs a cutoff that the three data choke points honour, so every calibration
downstream sees only rows strictly BEFORE ``d``:

    signal_panel._load_signals          the signals panel
    simulated_trades.load_sim_*         the solo-method simulation
    tracker._load_trades                the trade ledger

Those three are the same choke points the scorer-epoch mask uses, and for the
same reason: protecting the entry point protects every consumer, including ones
written later that never knew this existed.

**Why it also dissolves the circularity.** Tier 2's backtest was read-only —
never allowed to feed calibration — because weights fitted on the whole panel
and then applied back to it is self-confirmation. Under a cutoff that argument
no longer holds: weights at day D are fitted only on data before D, so they
cannot encode D's outcome. Walk-forward results are therefore legitimate
evidence in a way a fixed-weight backtest never was.

**The caches are the trap.** Every layer of the weighting stack memoises
(`_IC_WEIGHT_CACHE`, `_WINRATE_FILTER_CACHE`, `_SIDE_SKILL_CACHE`, the panel
memo, market-relative, method-horizons, and four in `tracker`). They are keyed
on time-to-live, NOT on the cutoff — so without an explicit flush, step two of a
walk-forward silently reuses step one's weights and the whole run collapses to a
single calibration wearing a walk-forward label. `analysis_asof` flushes on both
entry and exit; `reset_all_calibration_caches()` is exposed separately because
anything that changes the visible data has the same problem.
"""

from __future__ import annotations

import contextlib
import threading
from typing import Optional

from loguru import logger

# THREAD-LOCAL since 2026-08-04. It was a module global, with the reasoning that
# the walk-forward driver is single-threaded so a thread-local "would silently
# fail to reach a worker pool". That trade was backwards: there is no worker pool
# under this read path (asserted mechanically by
# tests/test_asof_thread_isolation.py), while the global DID leak — the nightly
# refactor runs its walk-forward in a BACKGROUND THREAD
# (`runner._maybe_start_nightly_rescore`), so a concurrent live tick read the
# same cutoff and saw a TRUNCATED ledger. Observed twice:
#
#   2026-08-02 22:41:39 nightly rescore STARTING in background
#   2026-08-02 22:44:19 tick raised: save_trades refused ... shrink 356 -> ...
#   2026-08-04 13:51:36 nightly rescore STARTING (triggered by that day's edits)
#
# The save_trades shrink-guard caught the WRITE, which is the only reason history
# survived. The silent half is worse: for the ~40 minutes a rescore runs, every
# overlapping tick calibrates, sizes and monitors positions against a ledger cut
# off weeks in the past, with nothing to notice it.
#
# Thread-local means the walk-forward's cutoff binds ONLY the thread that
# installed it; live ticks on other threads stay unrestricted, which is correct
# for both. If a worker pool is ever added beneath a calibration, the cutoff must
# be propagated into the workers EXPLICITLY — the test above fails to force that
# decision rather than letting look-ahead leak in quietly.
_ASOF_LOCAL = threading.local()


def current_asof() -> Optional[str]:
    """The active cutoff as ``YYYY-MM-DD``, or None when unrestricted."""
    return getattr(_ASOF_LOCAL, "value", None)


def asof_sql_clause(column: str = "signal_date") -> str:
    """A ``AND <column> < 'date'`` fragment, or ``""`` when no cutoff is set.

    Returned as literal SQL rather than a bound parameter so callers can splice
    it into queries whose parameter lists they do not control. The value is a
    validated ISO date (see `analysis_asof`), never user input.
    """
    _a = current_asof()
    return f" AND {column} < '{_a}'" if _a else ""


def before_cutoff(value) -> bool:
    """True when ``value`` (a date/ISO string) is visible under the cutoff."""
    _a = current_asof()
    if _a is None:
        return True
    if value is None:
        return False                    # unknown date cannot be proven visible
    return str(value)[:10] < _a


def reset_all_calibration_caches() -> None:
    """Flush every memo in the weighting stack.

    Each is keyed on a TTL, not on the as-of cutoff, so a stale entry survives a
    cutoff change and silently answers for the wrong point in time. Fail-soft per
    module: a missing hook must not abort a walk-forward, but it IS logged,
    because a cache that quietly refuses to clear produces a plausible-looking
    wrong answer — the failure mode this project keeps hitting.
    """
    import src.signals.aggregator as agg

    for fn in ("reset_ic_weight_cache", "reset_winrate_filter_cache"):
        try:
            getattr(agg, fn)()
        except Exception as e:
            logger.warning(f"[asof] {fn} failed: {e}")
    for mod, attr in ((agg, "_SIDE_SKILL_CACHE"),
                      (agg, "_IC_WEIGHT_CACHE"),
                      (agg, "_WINRATE_FILTER_CACHE")):
        try:
            getattr(mod, attr).clear()
        except Exception:
            pass
    for path in ("src.analysis.market_relative", "src.analysis.method_horizons",
                 "src.analysis.signal_panel", "src.analysis.simulated_trades",
                 "src.signals.rank_shaping", "src.signals.news_shock",
                 "src.signals.catalyst_tilt", "src.signals.ml_scale"):
        try:
            m = __import__(path, fromlist=["x"])
            for fn in ("reset_cache", "reset_panel_cache"):
                if hasattr(m, fn):
                    getattr(m, fn)()
        except Exception as e:
            logger.warning(f"[asof] cache reset failed for {path}: {e}")
    try:
        import src.performance.tracker as tr
        for attr in ("_SIDE_THRESHOLD_CACHE", "_AUTO_INVERSION_CACHE",
                     "_PANEL_INVERSION_CACHE", "_HORIZON_RAMP_CACHE"):
            if hasattr(tr, attr):
                getattr(tr, attr).clear()
    except Exception as e:
        logger.warning(f"[asof] tracker cache reset failed: {e}")


@contextlib.contextmanager
def analysis_asof(cutoff: Optional[str]):
    """Restrict every calibration read to rows strictly before ``cutoff``.

    ``cutoff`` is ``YYYY-MM-DD`` (or None for no restriction). Caches are flushed
    on entry AND exit — on exit because the calibrations computed inside are
    point-in-time and must never leak into live decisions afterwards.
    """
    if cutoff is not None:
        cutoff = str(cutoff)[:10]
        # A malformed cutoff would be spliced into SQL and silently match
        # nothing (or everything), so refuse rather than degrade.
        import datetime as _dt
        _dt.date.fromisoformat(cutoff)

    prev = current_asof()
    _ASOF_LOCAL.value = cutoff
    reset_all_calibration_caches()
    try:
        yield cutoff
    finally:
        _ASOF_LOCAL.value = prev
        reset_all_calibration_caches()
