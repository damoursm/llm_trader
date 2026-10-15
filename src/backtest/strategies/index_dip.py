"""PREREG110's N3 — the BULL-MARKET INDEX DIP as account pieces (a research book, not production's): the crash book's
construction (`strategies/crash.py`) with its trend inverted and the dip book's RSI cut. At the close of D-1 the
DECISION symbol (QQQ) closed ABOVE its 200-session average with a 2-session RSI under 10 -> the TRADED fund (TQQQ) is
bought at D's 09:30 open and sold at the open after the decision symbol's first close above its 5-session average
since the entry session, or after 10 sessions held; one long at a time. The decision symbol's daily rows are
production's construction (`dip_long._daily`: the deep store's 30-minute bars as daily rows; set A0: `a0.daily`), the
held fund's marks its 30-minute bars (`dm._hold`). Joined to the live account by `mimic.register_research`
(`studies/prereg110/account110.py`)."""
from __future__ import annotations

from typing import List, Optional

import numpy as np
import pandas as pd

RULE = {"rsi_max": 10.0, "trend": 200, "exit_sma": 5, "max_hold": 10}


def _daily(sym: str) -> Optional[pd.DataFrame]:
    """The decision symbol's daily rows: the deep store's (`dip_long._daily`), preceded by set A0's (`a0.daily`) before
    the deep store's first session — the deep store starts in 2021, and a 200-session average needs the year before."""
    from src.signals import dip_long
    d = dip_long._daily(sym)
    try:
        from src.backtest import a0 as A0
        h = A0.daily(sym)
    except Exception:                                    # noqa: BLE001 — without A0's rows the deep store alone
        h = None
    if d is None or d.empty or h is None or h.empty:
        return d
    h = h[h.index < d.index.min()]
    return pd.concat([h[[c for c in d.columns if c in h.columns]], d]).sort_index()


def trades(lo, hi, daily: Optional[pd.DataFrame] = None, sym: str = "QQQ", rule: Optional[dict] = None) -> List[dict]:
    """The rule's longs for decisions on sessions [lo, hi]: {signal, entry, exit} session dates (an exit past the
    data's end is None). Each decision reads the closes through D-1 only."""
    from src.signals.dip_long import rsi2
    r = dict(RULE, **(rule or {}))
    d = daily if daily is not None else _daily(sym)
    if d is None or d.empty:
        return []
    days = pd.DatetimeIndex(d.index)
    c = d["c"].astype(float)
    cv = c.to_numpy(float)
    sma_t = c.rolling(int(r["trend"])).mean().to_numpy()
    sma_x = c.rolling(int(r["exit_sma"])).mean().to_numpy()
    r2 = rsi2(cv)
    lo, hi = pd.Timestamp(lo), pd.Timestamp(hi)
    out, busy_until = [], -1
    for j in range(1, len(days)):
        if not (lo <= days[j] <= hi) or j < busy_until:
            continue
        i = j - 1
        if not (np.isfinite(sma_t[i]) and cv[i] > sma_t[i] and np.isfinite(r2[i]) and r2[i] < float(r["rsi_max"])):
            continue
        mh = int(r["max_hold"])
        x = next((k for k in range(j, min(len(days), j + mh)) if cv[k] > sma_x[k]), j + mh - 1)
        exit_j = x + 1
        out.append({"signal": days[i], "entry": days[j], "exit": days[exit_j] if exit_j < len(days) else None})
        busy_until = exit_j if exit_j < len(days) else len(days)
    return out


def pieces(lo, hi, end=None, strategy: str = "idip", sym: str = "QQQ", trade_sym: str = "TQQQ",
           period: Optional[str] = None, rule: Optional[dict] = None) -> List[dict]:
    """The account engine's pieces of the rule's longs decided on sessions [lo, hi] (``period="A0"``: set A0's daily
    rows of the decision symbol and the fund's 30-minute bars, as `strategies.crash.pieces`)."""
    from src.backtest.strategies import dm as DM
    dd = DM.daily(trade_sym)
    if period is None:
        bars, dec = DM.bars_30m(trade_sym), None
    elif period == "A0":
        from src.backtest import a0 as A0
        bars, dec = A0.bars_rth_frame(trade_sym), A0.daily(sym)
        if dec is None:
            raise FileNotFoundError(f"set A0 holds no bars of the decision symbol {sym}")
    else:
        raise ValueError(f"index_dip.pieces(period=...) knows set 'A0' only, not {period!r}")
    if dd is None or bars is None:
        return []
    dv = (dd["Close"] * dd["Volume"]).rolling(20).mean()
    end = pd.Timestamp(end).normalize() if end is not None else None
    out = []
    for t in trades(lo, hi, daily=dec, sym=sym, rule=rule):
        a, b = t["entry"], t["exit"]
        if a not in dd.index or (b is not None and b not in dd.index):
            continue
        last = end if end is not None else pd.Timestamp(dd.index[-1])
        x = dv[dv.index <= t["signal"]]
        p = DM._hold(trade_sym, a, b, last, dd, bars, float(x.iloc[-1]) if len(x) else 0.0, strategy, t["signal"],
                     open_marks=True)
        if p is not None:
            p.pop("rebase", None)
            out.append(p)
    return out
