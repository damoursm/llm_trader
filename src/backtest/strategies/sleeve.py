"""The QQQ sleeve's adapter — a RESEARCH book (PREREG154, rejected 2026-10-19: -6.9 log points a year, margin-call
draws 7 -> 19; no production book): one candidate LOT per ON session — the fund's previous close above its ``TREND``
(200) session average on the deep store's daily rows (`dip_long.daily_bars`, the crash book's construction; set A0:
A0's bars) — bought at that session's 09:30 open and held to the open of the first session that is not ON. The account
engine buys a lot only when the capital no book uses is at least ``MIN_FRAC`` (5%) of the equity (the strategy's
``min_frac``, `mimic.live_strategies` with ``Mimic(sleeve=True)``) and sells lots whole, the latest first, whenever a
book of higher quality needs room or the account de-levers. A lot's
30-minute marks and 09:30 opens are the fund's (`dm._hold`). Each set is its own account: a lot still held at the
set's last decision session is sold at the next session's open."""
from __future__ import annotations

from typing import List, Optional

import numpy as np
import pandas as pd

SYMBOL, TREND, MIN_FRAC = "QQQ", 200, 0.05


def symbol() -> str:
    return SYMBOL


def _daily(sym: str) -> Optional[pd.DataFrame]:
    """The fund's daily rows as production reads them (`dip_long._daily`: the deep 30-minute store, 2021 on), led by
    set A0's rows (the same construction on A0's bars) before the store's first session — production always has the
    200 sessions the trend needs; the backtest's set A would otherwise wait until October 2021 for its first lot."""
    from src.signals import dip_long
    deep = dip_long._daily(sym)
    try:
        from src.backtest import a0 as A0
        early = A0.daily(sym)
    except Exception:                                   # noqa: BLE001 — no A0 rows: the store's own history
        early = None
    if deep is None or deep.empty or early is None or early.empty:
        return deep
    early = early[pd.DatetimeIndex(early.index) < pd.DatetimeIndex(deep.index)[0]]
    cols = [c for c in deep.columns if c in early.columns]
    return pd.concat([early[cols], deep[cols]]) if len(early) else deep


def on_days(daily: pd.DataFrame, n: Optional[int] = None) -> pd.Series:
    """ON for each session D: D-1's close above its ``n``-session average (the first ``n`` sessions are off) — the
    decision at D's open reads the closes through D-1 only."""
    n = int(n or TREND)
    c = pd.Series(daily["c"].to_numpy(float), index=pd.DatetimeIndex(daily.index))
    above = (c > c.rolling(n).mean()).to_numpy(bool)
    return pd.Series(np.r_[False, above[:-1]], index=c.index)


def lots(lo, hi, daily: pd.DataFrame, n: Optional[int] = None) -> List[dict]:
    """The candidate lots for sessions [lo, hi]: {signal (D-1), entry (D), exit} — each ON session's open to the open
    of the first later session that is not ON, or of the first session after ``hi`` (the set's account ends); None
    past the data. ``n``: the trend's length (``TREND``)."""
    on = on_days(daily, n)
    days = pd.DatetimeIndex(on.index)
    onv = on.to_numpy(bool)
    n = len(days)
    nxt_off = np.full(n, n, np.int64)
    k = n
    for j in range(n - 1, -1, -1):
        if not onv[j]:
            k = j
        nxt_off[j] = k
    lo, hi = pd.Timestamp(lo), pd.Timestamp(hi)
    after_hi = int(np.searchsorted(days.values, np.datetime64(hi), side="right"))
    out = []
    for j in range(1, n):
        if not onv[j] or not (lo <= days[j] <= hi):
            continue
        x = min(int(nxt_off[j]), after_hi)
        out.append({"signal": days[j - 1], "entry": days[j], "exit": days[x] if x < n else None})
    return out


def pieces(lo, hi, end=None, strategy: str = "sleeve", sym: Optional[str] = None,
           period: Optional[str] = None) -> List[dict]:
    """The account engine's candidate lots for sessions [lo, hi] (``period="A0"``: set A0's bars, as the crash book)."""
    from src.backtest.strategies import dm as DM
    sym = sym or symbol()
    dd = DM.daily(sym)
    if period is None:
        bars, daily = DM.bars_30m(sym), _daily(sym)
    elif period == "A0":
        from src.backtest import a0 as A0
        bars, daily = A0.bars_rth_frame(sym), A0.daily(sym)
    else:
        raise ValueError(f"sleeve.pieces(period=...) knows set 'A0' only, not {period!r}")
    if dd is None or bars is None or daily is None or daily.empty:
        return []
    dv = (dd["Close"] * dd["Volume"]).rolling(20).mean()
    last = pd.Timestamp(end).normalize() if end is not None else pd.Timestamp(dd.index[-1])
    out = []
    for t in lots(lo, hi, daily):
        a, b = t["entry"], t["exit"]
        if a not in dd.index or (b is not None and b not in dd.index):
            continue
        x = dv[dv.index <= t["signal"]]
        p = DM._hold(sym, a, b, last, dd, bars, float(x.iloc[-1]) if len(x) else 0.0, strategy, t["signal"],
                     open_marks=True)
        if p is not None:
            p.pop("rebase", None)
            out.append(p)
    return out
