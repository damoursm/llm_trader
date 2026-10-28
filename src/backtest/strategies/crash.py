"""The crash-rebound book's adapter (PREREG86 / PREREG88; production `src/signals/crash_long.py`): one piece per
long, built from production's own construction — the DECISION symbol's (QQQ's) deep-store 30-minute bars as daily
rows (`dip_long.daily_bars`), the decision at D-1's close (`crash_long.stats_at` / `qualifies`: below the 200-session
average with a 2-session RSI under 5), the TRADED fund (`crash_long_trade_symbol`, TQQQ) bought at D's 09:30 open and
sold at the open after the decision symbol's first close above its 5-session average since the entry session or
after 10 sessions held (`crash_long.rebound_session` / the held count) — one long at a time; a signal on the session
the previous long is sold at the open buys again (production's order: exits, then entries). The holding's 30-minute
marks and 09:30 opens are the traded fund's (`dm._hold`); its margin rate is the strategy's
(`mimic.live_strategies`: ``house_init`` / ``house_maint`` = `crash_long_margin_rate`)."""
from __future__ import annotations

from typing import List, Optional

import numpy as np
import pandas as pd

from config import settings

# PREREG98 S2: the exit's condition — "sma" (production: the decision symbol's close above its
# `crash_long_exit_sma_sessions` average) or "rsi2" (its RSI2 above ``level``: the capitulation has resolved)
EXIT = {"rule": "sma", "level": 50.0}


def _daily(sym: str) -> Optional[pd.DataFrame]:
    from src.signals import dip_long
    return dip_long._daily(sym)


def trades(lo, hi, daily: Optional[pd.DataFrame] = None, sym: Optional[str] = None, every: bool = False) -> List[dict]:
    """The rule's longs for decisions on sessions [lo, hi]: {signal, entry, exit} session dates (an exit past the
    data's end is the last session's close). ``every`` (PREREG227's S2, `Mimic.crash_reenter`): one long for EVERY
    qualifying session, each with its own exit — the account's one-at-a-time rule (`max_open` 1) rejects the ones that
    overlap a held long, as production settles them ``holding``, so a buy the account could not fund or a long a margin
    call sold is bought again at the next qualifying open; False: the first of each non-overlapping run only (the
    construction before it, which never re-entered)."""
    from src.signals import crash_long
    sym = sym or crash_long.symbol()
    d = daily if daily is not None else _daily(sym)
    if d is None or d.empty:
        return []
    days = pd.DatetimeIndex(d.index)
    c = d["c"]
    sma_x = c.rolling(int(settings.crash_long_exit_sma_sessions)).mean().to_numpy()
    cv = c.to_numpy(float)
    rule, level = str(EXIT.get("rule", "sma")), float(EXIT.get("level", 50.0))
    if rule == "rsi2":
        from src.signals.dip_long import rsi2
        r2 = rsi2(cv)
        hit = lambda k: r2[k] > level                                       # noqa: E731
    else:
        hit = lambda k: cv[k] > sma_x[k]                                    # noqa: E731
    lo, hi = pd.Timestamp(lo), pd.Timestamp(hi)
    out, busy_until = [], -1
    for j in range(1, len(days)):
        if not (lo <= days[j] <= hi) or (j < busy_until and not every):
            continue
        st = crash_long.stats_at(d.iloc[:j], days[j - 1].date())        # the close of D-1, nothing later
        if not crash_long.qualifies(st):
            continue
        mh = int(settings.crash_long_max_hold_sessions)
        x = next((k for k in range(j, min(len(days), j + mh)) if hit(k)), j + mh - 1)
        exit_j = x + 1
        out.append({"signal": days[j - 1], "entry": days[j],
                    "exit": days[exit_j] if exit_j < len(days) else None})
        busy_until = exit_j if exit_j < len(days) else len(days)
    return out


def pieces(lo, hi, end=None, strategy: str = "crash", sym: Optional[str] = None,
           trade_sym: Optional[str] = None, period: Optional[str] = None, every: bool = False) -> List[dict]:
    """The account engine's pieces of the crash book's longs decided on sessions [lo, hi]: decided on ``sym`` (the
    decision symbol, `crash_long_symbol`), held in ``trade_sym`` (`crash_long_trade_symbol`). ``period="A0"``
    (PREREG101): the decision symbol's daily rows and the fund's 30-minute bars from set A0's bars
    (`src.backtest.a0`), production's construction over 2016-2021 (the fund's daily bars stay the OHLCV cache's).
    ``every``: a piece for every qualifying session (`trades`; PREREG227's S2)."""
    from src.backtest.strategies import dm as DM
    from src.signals import crash_long
    sym = sym or crash_long.symbol()
    held = trade_sym or crash_long.trade_symbol()
    dd = DM.daily(held)
    if period is None:
        bars, dec = DM.bars_30m(held), None
    elif period == "A0":
        from src.backtest import a0 as A0
        bars, dec = A0.bars_rth_frame(held), A0.daily(sym)
        if dec is None:
            raise FileNotFoundError(f"set A0 holds no bars of the crash book's decision symbol {sym}")
    else:
        raise ValueError(f"crash.pieces(period=...) knows set 'A0' only, not {period!r}")
    if dd is None or bars is None:
        return []
    dv = (dd["Close"] * dd["Volume"]).rolling(20).mean()
    end = pd.Timestamp(end).normalize() if end is not None else None
    out = []
    for t in trades(lo, hi, daily=dec, sym=sym, **({"every": True} if every else {})):
        a, b = t["entry"], t["exit"]
        if a not in dd.index or (b is not None and b not in dd.index):
            continue
        last = end if end is not None else pd.Timestamp(dd.index[-1])
        x = dv[dv.index <= t["signal"]]
        p = DM._hold(held, a, b, last, dd, bars, float(x.iloc[-1]) if len(x) else 0.0, strategy, t["signal"],
                     open_marks=True)
        if p is not None:
            p.pop("rebase", None)
            out.append(p)
    return out
