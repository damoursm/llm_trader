"""Per-ROW pivot labels — the one resolver every date/tick-keyed surface shares.

The label depends on the row's own tick (time + snapshot price), so the natural
unit is the row: ``pivot_fwd_row(tk, when, price, ...)`` returns the settled %
move to the next 30-minute pivot strictly after ``when`` (``None`` while
unsettled, or when the ticker has no 30-minute history — there is no other
basis to fall back to), with the resolving bar's ET session date for the
walk-forward embargo. Marks live on the bar's HIGH/LOW or CLOSE per
``pivot_target.pivot_label_basis``; the threshold is ``pivot_min_move_pct``.

Rows that carry a DATE but no tick — position-days, exit-day closes, a
follow-through candidate's signal day — anchor at that session's 16:00 ET
close with the close as the price, so the search starts at the next session's
first bar. State that on the surface that does it.

The series scan is memoised per (ticker, basis, threshold, window) for the life
of the process (`clear()` drops it), so a loop over thousands of rows scans
each ticker once. **The as-of cutoff is NOT part of the memo key** (2026-09-15):
the zigzag is causal — pivot ``k`` is emitted at its confirming bar ``CF[k]``
from bars ``<= CF[k]`` only — so scanning the full series once and admitting
only pivots whose confirming bar precedes the cutoff is EXACTLY the truncated
scan. Keying the memo on the cutoff made every walk-forward date (the nightly
rescore, `weights_for_date`) rescan every ticker's 30-minute series and starved
the ticks into watchdog kills. Point-in-time: under an as-of cutoff only bars of
EARLIER sessions are visible.
"""
from __future__ import annotations

from datetime import date as _date
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd

from src.analysis import pivot_target as _pt
from src.analysis.pivot_target import (
    MAX_PIVOT_BARS_30M, MIN_BARS_30M, _NY_TZ_NAME, _min_move_pct, _resolved_pivots,
    pivot_label_basis, session_close_utc, to_naive_utc,
)

_SCANS: Dict[Tuple[str, str, float, Optional[str]], Optional[tuple]] = {}


def clear() -> None:
    """Drop the per-ticker scan memo (tests; long-lived processes after a cache warm)."""
    _SCANS.clear()


def _asof_cut_utc(asof_day) -> Optional[pd.Timestamp]:
    """ET midnight of the as-of DATE as naive UTC: bars of earlier sessions only."""
    if asof_day is None:
        return None
    return (pd.Timestamp(asof_day).normalize().tz_localize(_NY_TZ_NAME)
            .tz_convert("UTC").tz_localize(None))


WARMUP_DAYS = 60          # calendar days of bars kept BEFORE `since`: the zigzag converges after its first confirmed swing


def _scan(tk: str, asof_day=None, since=None) -> Optional[tuple]:
    """Memoised per (ticker, basis, threshold, since) — ``asof_day`` is accepted
    for signature compatibility and deliberately NOT part of the key (see the
    module docstring; the cutoff is applied per row in `pivot_fwd_row`).
    ``since`` trims the series to bars from ``since − WARMUP_DAYS`` on: the
    tick cache holds a year per name, and a panel build only needs its own
    window plus warm-up."""
    basis = pivot_label_basis()
    thr = float(_min_move_pct())
    key = (tk, basis, thr, str(since) if since is not None else None)
    if key in _SCANS:
        return _SCANS[key]
    res = None
    s = _pt._series_30m(tk)               # resolved at call time: one seam for tests and callers
    if s is not None:
        idx, c, h, lo = s
        if since is not None:
            m0 = int(idx.searchsorted(pd.Timestamp(since) - pd.Timedelta(days=WARMUP_DAYS), side="left"))
            idx, c, h, lo = idx[m0:], c[m0:], h[m0:], lo[m0:]
        if basis == "close":
            h, lo = c, c                  # marks on the bar's CLOSE
        if len(c) >= MIN_BARS_30M:
            P, PP, FL, CF = _resolved_pivots(c, h, lo, thr)
            et_dates = idx.tz_localize("UTC").tz_convert(_NY_TZ_NAME).date
            res = (idx, c, np.asarray(P, dtype=int), np.asarray(PP, dtype=float),
                   np.asarray(FL, dtype=bool), et_dates, np.asarray(CF, dtype=int))
    _SCANS[key] = res
    return res


def _visible_bars(s: tuple, asof_day) -> int:
    """Number of bars visible under the cutoff (all of them without one)."""
    idx = s[0]
    cut = _asof_cut_utc(asof_day)
    return len(idx) if cut is None else int(idx.searchsorted(cut, side="left"))


def has_intraday_history(tk: str, asof_day=None, since=None) -> bool:
    """Same memo as `pivot_fwd_row` when called with the same ``since`` —
    a pre-check must not trigger an untrimmed full-series scan. Under a cutoff,
    "history" means at least ``MIN_BARS_30M`` bars visible before it."""
    s = _scan(tk, asof_day, since)
    return s is not None and _visible_bars(s, asof_day) >= MIN_BARS_30M


def pivot_fwd_row(tk: str, when, price: Optional[float], *, asof_day=None,
                  fallback_close: Optional[float] = None, since=None):
    """``(fwd_pct, end_session_date, end_ts)`` for one row, or ``None``.
    ``since`` (a date) trims the scanned series to the caller's window plus warm-up
    — pass the earliest anchor of the batch; see ``_scan``.

    ``when``  — the row's tick (any timestamp / ISO string, tz-aware or UTC-naive)
                OR a date; a date anchors at that session's 16:00 ET close.
    ``price`` — the snapshot price at the tick; falls back to ``fallback_close``
                (the session close) when missing, so a row is never anchored at
                a price it did not have.
    Settled rows only: the first RESOLVED pivot on a bar starting at/after the
    anchor, within ``MAX_PIVOT_BARS_30M``, whose CONFIRMING bar is visible under
    ``asof_day``. Unsettled → ``None`` (the evaluation surfaces apply the
    last-close rule themselves; a training label never does).
    """
    s = _scan(tk, asof_day, since)
    if s is None:
        return None
    idx, c, P, PP, _FL, et_dates, CF = s
    n_vis = _visible_bars(s, asof_day)
    if n_vis < MIN_BARS_30M:
        return None
    if isinstance(when, _date) and not isinstance(when, pd.Timestamp):
        t = session_close_utc(when)
    else:
        t = to_naive_utc(when)
    if t is None:
        return None
    px = price if (price is not None and price == price and price > 0) else fallback_close
    if not (px is not None and px == px and px > 0):
        return None
    i0 = int(idx.searchsorted(t, side="left"))
    if i0 >= n_vis:
        return None
    k = int(np.searchsorted(P, i0, side="left")) if len(P) else 0
    # Confirmations are monotone in k, so the first pivot after the anchor is the
    # only candidate: if its confirming bar is not yet visible, no later one is.
    if k < len(P) and int(CF[k]) < n_vis and (int(P[k]) - i0) <= MAX_PIVOT_BARS_30M:
        j = int(P[k])
        return (float(PP[k]) / float(px) - 1.0) * 100.0, et_dates[j], idx[j]
    return None
