"""Shared utilities."""

from datetime import datetime
from zoneinfo import ZoneInfo

ET = ZoneInfo("America/New_York")


def now_et() -> datetime:
    """Return the current datetime in US/Eastern (handles EST/EDT automatically)."""
    return datetime.now(ET)


def fmt_et(dt: datetime, include_date: bool = True) -> str:
    """
    Format a datetime in Eastern time.
    Converts from any timezone (including UTC) before formatting.
    """
    eastern = dt.astimezone(ET)
    tz_label = eastern.strftime("%Z")   # "EST" or "EDT"
    if include_date:
        return eastern.strftime(f"%Y-%m-%d %H:%M {tz_label}")
    return eastern.strftime(f"%H:%M {tz_label}")


def fmt_iso_et(iso_str: str, include_date: bool = True) -> str:
    """Parse an ISO 8601 string and format it as Eastern time.

    Trade dicts store ``entry_datetime`` / ``exit_datetime`` /
    ``decision_datetime`` as ISO 8601 strings (so they survive JSON
    round-trips cleanly). This helper turns one into the same human-friendly
    Eastern-time format ``fmt_et`` produces for live datetimes, so the email
    can show entry/exit time alongside the date. Returns an empty string for
    None / unparseable input so the template stays simple.
    """
    if not iso_str:
        return ""
    try:
        dt = datetime.fromisoformat(iso_str)
    except (TypeError, ValueError):
        return ""
    return fmt_et(dt, include_date=include_date)


# ── Single-flight TTL cache ──────────────────────────────────────────────────
#
# Single-flight TTL cache — one computation shared by concurrent callers.
#
# The pipeline runs ``build_signals`` from several places AT THE SAME TIME: the
# main pass, the concurrent ``_HoldReviewBranch`` (overlapped with main steps 4-5
# on purpose), and the shadow-arm branch. Each of those recomputes the same heavy
# panel calibrations, and a plain "check TTL → compute → store" cache does not stop
# that: every concurrent caller checks the *empty* cache, all of them miss, and all
# of them compute. A classic cache stampede.
#
# Measured on a live 861 s tick (2026-08-04): the win-rate filter ran **4×** — two
# of them logging in the SAME SECOND, which is the stampede caught red-handed —
# ``build_panel`` 3×, the market-relative filter 2×, IC weights 2×.
#
# ``ttl_single_flight`` fixes it with double-checked locking: the fast path stays
# lock-free (a fresh entry returns immediately, so the steady state costs nothing),
# and only a MISS takes the lock. The second caller then blocks, re-checks, and
# gets the first caller's result instead of duplicating the work.
#
# Deliberately NOT a general memoiser: callers keep their own cache dict and TTL so
# the existing cache-reset test hooks (``reset_winrate_filter_cache`` and friends)
# keep working untouched.


import threading
import time
from typing import Any, Callable, Dict


def ttl_single_flight(lock: threading.Lock, cache: Dict[str, dict], key: str,
                      ttl: float, compute: Callable[[], Any],
                      field: str = "val") -> Any:
    """Return the cached value for ``key``, computing it at most once.

    ``cache`` maps key → ``{"ts": float, field: value}`` (the shape the existing
    call sites already use). ``compute`` is only ever invoked by ONE thread at a
    time for a given lock; a caller that arrives during a computation waits and
    reuses the result.

    A ``compute`` that raises propagates to its caller and stores nothing, so the
    next call retries — failures are never cached.
    """
    hit = cache.get(key)
    if hit and (time.time() - hit["ts"]) < ttl:
        return hit[field]
    with lock:
        # Re-check: another thread may have filled it while we waited here.
        hit = cache.get(key)
        if hit and (time.time() - hit["ts"]) < ttl:
            return hit[field]
        value = compute()
        cache[key] = {"ts": time.time(), field: value}
        return value
