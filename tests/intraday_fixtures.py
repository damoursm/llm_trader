"""Shared synthetic 30-MINUTE series for the pivot-label tests (2026-09-16).

The label runs on 30-minute regular-hours bars, so a test that wants a label
must supply such a series — never a daily one. Two builders:

* ``sessions_30m(start, n)`` — RTH bar starts (naive UTC) for ``n`` weekdays,
  13 per session, EDT offset (09:30 ET = 13:30 UTC); ``flat_series`` /
  ``plant_peak`` shape a story on them.
* ``replica_30m(dates, closes)`` — a FLAT replica of a daily close series: 13
  bars per session all at that session's close (O=H=L=C). Under the zigzag
  this reproduces the old daily-close semantics exactly (pivots at session
  closes, a date-anchored row's search starting at the next session's first
  bar), which is what the legacy panel tests were written against.

``stub_30m(monkeypatch, series_by_ticker)`` installs a ``_series_30m`` stub
AND clears the `pivot_rows` scan memo — the memo is keyed on the ticker, not
on the stub, so a test that re-stubs the same ticker without clearing reads
the previous test's series.
"""
from __future__ import annotations

from datetime import date
from typing import Dict, Sequence

import numpy as np
import pandas as pd

EDT_OPEN_UTC = 13.5          # 09:30 ET during daylight time = 13:30 UTC
BARS = 13


def sessions_30m(start: str, n_sessions: int) -> pd.DatetimeIndex:
    days = pd.bdate_range(start, periods=n_sessions)
    out = []
    for d in days:
        for k in range(BARS):
            out.append(d + pd.Timedelta(hours=EDT_OPEN_UTC) + pd.Timedelta(minutes=30 * k))
    return pd.DatetimeIndex(out)


def flat_series(idx, level: float = 100.0):
    n = len(idx)
    c = np.full(n, level); h = np.full(n, level + 0.05); lo = np.full(n, level - 0.05)
    return c, h, lo


def plant_peak(h, lo, c, at: int, peak: float = 103.0, trough: float = 99.0) -> None:
    """A swing HIGH on bar ``at`` confirmed by a >1% drop on the bars after it."""
    h[at] = peak; c[at] = peak - 0.5
    for k in range(1, 4):
        lo[at + k] = trough; c[at + k] = trough + 0.2; h[at + k] = trough + 0.4


def _session_bars_utc(d: date) -> pd.DatetimeIndex:
    """13 RTH bar starts of session ``d`` as naive UTC, DST-correct."""
    base = pd.Timestamp(d).tz_localize("America/New_York") + pd.Timedelta(hours=9, minutes=30)
    return pd.DatetimeIndex([(base + pd.Timedelta(minutes=30 * k)).tz_convert("UTC").tz_localize(None)
                             for k in range(BARS)])


def replica_30m(dates: Sequence[date], closes):
    """Flat replica: every bar of session ``d`` at ``closes[d]`` (dict or sequence)."""
    if isinstance(closes, dict):
        vals = [float(closes[d]) for d in dates]
    else:
        vals = [float(v) for v in closes]
    idx = pd.DatetimeIndex(np.concatenate([_session_bars_utc(d).values for d in dates]))
    c = np.repeat(np.asarray(vals, dtype=float), BARS)
    return idx, c, c.copy(), c.copy()


def stub_30m(monkeypatch, series_by_ticker: Dict[str, tuple]) -> None:
    """Install ``pivot_target._series_30m`` → the given ``(idx, c, h, lo)`` per
    ticker (None for unknown names) and clear the per-ticker scan memo."""
    from src.analysis import pivot_rows as pr
    from src.analysis import pivot_target as pt
    monkeypatch.setattr(pt, "_series_30m", lambda tk: series_by_ticker.get(tk))
    pr.clear()
