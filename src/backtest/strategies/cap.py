"""THE CAPITULATION BOOK as the account engine's pieces — production's `src/signals/cap_long.py` (from the 2026-10-21
session; PREREG165's CBg6_20, the research construction `llm_trader_research/studies/prereg165/cap165.py` +
`prereg163/wdip163.py`, which these pieces reproduce).

On a session whose BREADTH at the close before it (the share of the 100 most-traded common stocks / ADRs under their own
200-session average with a 2-session RSI under 10) is at least `cap_long_breadth_min`, every $2B+ name of the dip panel
that closed UNDER its own 200-session average with a 2-session RSI under 10 is the dip book's limit order
(`strategies.dip.order_pieces` with this book's signals: the limit at `cap_long_limit_atr` x the 14-session ATR under
the close, alive `cap_long_limit_sessions` sessions, the exit after the first close above the
`cap_long_exit_sma_sessions` average or `cap_long_max_hold_sessions` sessions), every signal an order, the deepest RSI2
first (the pieces' ``rank``). The breadth series is PREREG165's (`BREADTH`: Polygon's adjusted whole-market daily bars
2005-2026, `cap165.breadth`), which production's `cap_long.breadth_at` reproduces (`cap_long.parity`).

The account (`mimic.live_strategies`): 1/`cap_long_account_slices` of the equity, at most `cap_long_max_open` open, its
working orders counted, capital quality 0.1 (`mimic.QUALITY`: it gives way to every other book), preemptible whole, the
latest first, de-levered first (``delever_order`` 0); a dip order is sized as if its longs were freed and its fill sells
them (`Rules.fills_make_room`).
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

from config import settings
from src.backtest.strategies import dip as DIP

BREADTH = Path("studies") / "prereg165" / "data" / "breadth165.pkl"     # in `backtest_research_dir`


def params() -> DIP.DipParams:
    """The book's rule as `DipParams` (production's settings; PREREG165's `wdip163.params()`: the dip book's LIVE exit,
    RSI2 under 10, a 200-session trend, $2B)."""
    return DIP.DipParams(rsi_max=float(settings.cap_long_rsi_max), min_dv=float(settings.cap_long_min_dollar_volume),
                         trend=int(settings.cap_long_trend_sessions), exit_sma=int(settings.cap_long_exit_sma_sessions),
                         max_hold=int(settings.cap_long_max_hold_sessions))


@lru_cache(maxsize=1)
def breadth() -> pd.Series:
    """PREREG165's breadth by session (B at each session's close)."""
    return pd.read_pickle(Path(settings.backtest_research_dir) / BREADTH)


def gate_days(theta: Optional[float] = None) -> pd.DatetimeIndex:
    """The signal sessions D-1 whose breadth is at least ``theta`` (default `cap_long_breadth_min`): their next session
    is a capitulation session."""
    th = float(settings.cap_long_breadth_min if theta is None else theta)
    B = breadth()
    return pd.DatetimeIndex(sorted(B.index[(B >= th).to_numpy()]))


def signals_fn(theta: Optional[float] = None):
    """The book's signals in `strategies.dip.signals`' form: the dip rule with the trend inverted (`wdip163`'s, every
    context), kept only on the gate's signal sessions."""
    from src.signals.dip_long import rsi2
    keep = gate_days(theta)

    def signals(panel: Dict[str, pd.DataFrame], prm, lo, hi, names: Optional[Sequence[str]] = None) -> pd.DataFrame:
        lo, hi = pd.Timestamp(lo), pd.Timestamp(hi)
        data_last = max(v.index[-1] for v in panel.values() if len(v))
        rows = []
        for tk in names if names is not None else DIP.eligible_names(panel):
            d = panel.get(tk)
            if d is None or len(d) < DIP.MIN_BARS:
                continue
            c = d["c"].to_numpy(float)
            n = len(c)
            s = pd.Series(c)
            sma_t = s.rolling(int(prm.trend)).mean().to_numpy()
            sma_x = s.rolling(int(prm.exit_sma)).mean().to_numpy()
            dv20 = (d["c"] * d["v"]).rolling(20).mean().to_numpy()
            r2 = rsi2(c)
            dates = d.index
            with np.errstate(invalid="ignore"):
                sig = (c < sma_t) & (r2 < prm.rsi_max) & (dv20 >= prm.min_dv) & (dates >= lo) & (dates <= hi)
            sig[:DIP.MIN_BARS - 1] = False
            sig[n - 1:] = False
            ended = dates[-1] < data_last - pd.Timedelta(days=7)
            for i in np.flatnonzero(sig):
                j = i + 1
                x = next((k for k in range(j, min(n, i + 1 + int(prm.max_hold))) if DIP.exit_hit(prm, c, sma_x, r2, k)),
                         min(n - 1, i + int(prm.max_hold)))
                last_close = False
                if prm.exit == "next_open":
                    if x + 1 >= n:
                        if not ended:
                            continue
                        xd, last_close = n - 1, True
                    else:
                        xd = x + 1
                else:
                    xd = x
                if (dates[xd] - dates[j]).days > DIP.gap_days(prm):
                    continue
                rows.append((tk, dates[i], dates[j], dates[xd], float(r2[i]), float(dv20[i]), float(c[i]), last_close))
        S = pd.DataFrame(rows, columns=["tk", "sig", "d_in", "d_out", "rsi2", "dv20", "close", "last_close"])
        if not len(S):
            return S
        return S[pd.DatetimeIndex(S["sig"]).normalize().isin(keep)].reset_index(drop=True)
    return signals


def pieces(lo, hi, panel: Optional[Dict[str, pd.DataFrame]] = None, period: Optional[str] = None,
           strategy: str = "cap", theta: Optional[float] = None) -> List[dict]:
    """The book's orders decided on sessions [lo, hi] as the dip book's limit-order pieces (every signal's order) on
    ``panel`` (None: set A0's own-security panel with ``period="A0"``, else `strategies.dip.build_panel()`)."""
    if panel is None:
        if period == "A0":
            from src.backtest import a0 as A0
            panel = A0.dip_panel()
        else:
            panel = DIP.build_panel()
    return DIP.order_pieces(params(), lo, hi, panel=panel, strategy=strategy,
                            atr_mult=float(settings.cap_long_limit_atr), sessions=int(settings.cap_long_limit_sessions),
                            signals_fn=signals_fn(theta))
