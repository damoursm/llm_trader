"""INTRADAY H/L pivot labels, anchored at the TICK — the evaluation skill's view
of the PRODUCTION label (2026-09-14; the ONLY label since 2026-09-16 — the daily
H/L label is retired, and this function also trains ml_ohlcv, via the deep store).

One implementation, not two: this module is a thin wrapper over
``src.analysis.pivot_target.intraday_pivot_targets`` — the same zigzag, the same
"eligible bars start at or after the tick" rule, the same last-visible-close
resolution for the unresolved tail — so an evaluation here and the panel's
training label can never drift apart. The production 30-minute cache
(``cache/ohlcv_30m``, Polygon-fed, the newest 260 sessions) is the default bar source;
pass ``bars`` (``{ticker: 30-min OHLCV frame}``) to evaluate on a scratch fetch.

Definitions, per (ticker, tick_time, tick_price):
  * eligible bars  = 30-min bars whose START is at or after tick_time — the bar
                     that contains the tick is excluded (it cannot be split).
  * resolved       = the first RESOLVED pivot on an eligible bar.
                     target = pivot extreme / tick_price − 1.
  * provisional    = no resolved pivot yet: marked at the LAST VISIBLE close,
                     never at the running leg's extreme.
  * same_day       = the resolving pivot's bar falls on the tick's ET date.
  * NaN            = the tick is after the last visible bar, or no history.

Point-in-time: ``asof`` truncates every series before scanning.

Guardrail: provisional labels are for EVALUATION and MONITORING ONLY — the panel
writes settled rows only, and training never sees an unresolved value.
"""
from __future__ import annotations

import sys
from typing import Dict, Optional

import numpy as np
import pandas as pd

for _p in (r"C:\Users\mathi\PycharmProjects\llm_trader",):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from src.analysis import pivot_target as _pt  # noqa: E402


def _frame_to_series(df: pd.DataFrame):
    idx = pd.DatetimeIndex(df.index)
    if idx.tz is not None:
        idx = idx.tz_convert("UTC").tz_localize(None)
    c = pd.to_numeric(df["Close"], errors="coerce").to_numpy(dtype=float)
    h = pd.to_numeric(df["High"], errors="coerce").to_numpy(dtype=float) if "High" in df.columns else c
    lo = pd.to_numeric(df["Low"], errors="coerce").to_numpy(dtype=float) if "Low" in df.columns else c
    ok = np.isfinite(c) & np.isfinite(h) & np.isfinite(lo) & (c > 0)
    if len(df) and ok.sum() < 0.9 * len(df):
        raise ValueError(f"{len(df) - int(ok.sum())} of {len(df)} 30-min bars have NaN prices")
    return idx[ok], c[ok], h[ok], lo[ok]


class IntradayLabeler:
    """Scan each ticker's 30-min series ONCE, then answer any number of anchors."""

    def __init__(self, bars: Optional[Dict[str, pd.DataFrame]] = None, asof=None):
        self._bars = bars
        self._asof = asof
        self._series: Dict[str, Optional[tuple]] = {}

    def _get(self, tk: str):
        if tk not in self._series:
            s = None
            if self._bars is not None:
                df = self._bars.get(tk)
                if df is not None and not df.empty:
                    s = _frame_to_series(df)
            else:
                s = _pt._series_30m(tk)                 # the production cache
            self._series[tk] = s
        return self._series[tk]

    def label(self, tk: str, tick_time_utc, tick_price: float):
        s = self._get(tk)
        if s is None:
            return None
        idx, c, h, lo = s
        r = _pt.intraday_pivot_targets(idx, c, h, lo, [(tick_time_utc, tick_price)], asof=self._asof)[0]
        if r is None:
            return None
        return dict(target_pct=r["target_pct"], resolved=r["resolved"], pivot_time=r["end_ts"],
                    bars_ahead=r["bars_ahead"], same_day=r["same_session"], confirm_ts=r.get("confirm_ts"))


def label_rows(rows: pd.DataFrame, bars: Optional[Dict[str, pd.DataFrame]] = None, *,
               time_col: str = "generated_at", price_col: str = "price", asof=None) -> pd.DataFrame:
    """Vectorised over a frame with ticker / tick time / tick price columns.
    ``bars=None`` reads the production 30-minute cache."""
    L = IntradayLabeler(bars, asof=asof)
    out = []
    for tk, tt, px in zip(rows["ticker"], rows[time_col], rows[price_col]):
        r = L.label(tk, tt, float(px) if px == px else np.nan)
        out.append(r or dict(target_pct=np.nan, resolved=False, pivot_time=pd.NaT,
                             bars_ahead=np.nan, same_day=False, confirm_ts=None))
    return pd.DataFrame(out, index=rows.index)
