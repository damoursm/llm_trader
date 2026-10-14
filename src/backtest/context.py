"""THE MARKET CONTEXT a model is activated in (user directive 2026-10-14: "a model should understand the context and be
activated if historically it performs well in it. If it is activated, then it can be used to trade."; PREREG108).

POINT IN TIME: a decision on session D reads the context at the close of the last SPY session BEFORE D.

* trend — SPY's close against its `TREND_SESSIONS` (200) session average: "above" / "below";
* vol — SPY's `VOL_SESSIONS` (20) session realized volatility (daily log returns, annualised) against the terciles of
  that volatility's OWN HISTORY UP TO THAT CLOSE (an expanding window from SPY's first stored session, 2006-08-14;
  `MIN_HISTORY` (250) sessions at least): "low" / "mid" / "high".

The cell is "<trend>/<vol>" (`CELLS`, six). `reliability.regime_table`'s whole-history terciles stay a LABEL to read
results with; these cells DECIDE (the bank's activation maps, `Mimic(activation=...)`). Tests: tests/test_context.py
(the context of a day is the same on a history cut right after it, and rewriting every later close changes nothing).
"""
from __future__ import annotations

import bisect
import math
from typing import Iterable, Optional

import numpy as np
import pandas as pd

TREND_SESSIONS = 200
VOL_SESSIONS = 20
MIN_HISTORY = 250
CELLS = tuple(f"{t}/{v}" for t in ("above", "below") for v in ("low", "mid", "high"))
EPOCH = pd.Timestamp("1970-01-01")
_CACHE: dict = {}


def _quantile(sorted_vals: list, p: float) -> float:
    """numpy's default (linear) quantile of an already sorted list."""
    pos = (len(sorted_vals) - 1) * p
    i = int(math.floor(pos))
    j = min(i + 1, len(sorted_vals) - 1)
    return sorted_vals[i] + (pos - i) * (sorted_vals[j] - sorted_vals[i])


def table(spy: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    """Per SPY session (index: naive session dates): ``trend``, ``rvol`` (annualised), ``lo`` / ``hi`` (the terciles of
    every ``rvol`` up to and including that close), ``vol`` and ``cell`` (None until both are defined)."""
    if spy is None:
        from src.backtest import reliability as REL
        spy = REL.spy_daily()
    c = spy["Close"].astype(float)
    sma = c.rolling(TREND_SESSIONS).mean()
    rv = (np.log(c).diff().rolling(VOL_SESSIONS).std() * math.sqrt(252)).to_numpy(float)
    trend = np.where(sma.isna().to_numpy(), None, np.where(c.to_numpy() > sma.to_numpy(), "above", "below"))
    seen: list = []
    lo, hi = np.full(len(c), np.nan), np.full(len(c), np.nan)
    vol = np.empty(len(c), object)
    for i, x in enumerate(rv):
        vol[i] = None
        if not np.isfinite(x):
            continue
        bisect.insort(seen, float(x))
        if len(seen) < MIN_HISTORY:
            continue
        lo[i], hi[i] = _quantile(seen, 1.0 / 3.0), _quantile(seen, 2.0 / 3.0)
        vol[i] = "low" if x <= lo[i] else ("mid" if x <= hi[i] else "high")
    out = pd.DataFrame({"trend": trend, "rvol": rv, "lo": lo, "hi": hi, "vol": vol},
                       index=pd.DatetimeIndex(spy.index).tz_localize(None).normalize())
    out["cell"] = [f"{t}/{v}" if t is not None and v is not None else None for t, v in zip(out.trend, out.vol)]
    return out


def cached_table() -> pd.DataFrame:
    """`table()` of the SPY file, built once per process and file version."""
    from pathlib import Path
    p = Path("cache") / "ohlcv" / "SPY.json"
    key = (str(p.resolve()), p.stat().st_mtime if p.exists() else None)
    if _CACHE.get("key") != key:
        _CACHE["key"], _CACHE["table"] = key, table()
    return _CACHE["table"]


def at(days: Iterable, tab: Optional[pd.DataFrame] = None) -> np.ndarray:
    """The cell in force for a decision on each day (dates, or day numbers since 1970-01-01): the last SPY session
    STRICTLY BEFORE it; None where the table has none yet."""
    tab = cached_table() if tab is None else tab
    vals = list(days)
    if vals and all(isinstance(v, (int, np.integer)) for v in vals):
        d = pd.DatetimeIndex([EPOCH + pd.Timedelta(days=int(v)) for v in vals])
    else:
        d = pd.DatetimeIndex(pd.to_datetime(vals))
        if d.tz is not None:
            d = d.tz_localize(None)
    d = d.normalize()
    idx = tab.index.searchsorted(d, side="left") - 1
    cells = tab["cell"].to_numpy(object)
    return np.array([cells[i] if i >= 0 else None for i in idx], object)
