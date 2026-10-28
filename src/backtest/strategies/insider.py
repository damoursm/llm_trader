"""THE INSIDER BOOK as the account engine's pieces — production's `src/signals/insider_long.py` (from the 2026-10-26
session; PREREG203's O1_20, PREREG205's construction of it as production trades it).

Every decision is production's own code on history: the market gate (`insider_long.gate_from_closes` on SPY's daily
closes — `reliability.spy_daily`, Polygon's daily aggregates: equal, close for close, to the whole-market bars'
closes production reads, checked on 300 sessions), the purchase events (`insider_long.load_rows` /
`events_from_rows`: the deep store's insider tables, point in time — only the rows filed before the session), the
stocks' numbers (`insider_long.bar_stats` on the whole-market daily bars as traded of the 252 sessions before it —
`sel_pressure.grouped_ohlc`'s cache, fetched where it lacks a session) and the ranking (`insider_long.decide`: the largest
purchase first). Each signal is a long bought at its session's 09:30 OPEN and sold at the 09:30 open of the session
after `insider_long_hold_sessions` (60) sessions held (`insider_long.exit_check`'s count: the entry session the first),
or at its last bar's close when it stops trading first; marked at every session's close and low and, from the second
session, at its open (``open_marks``) — every price in the entry session's terms (the whole-market bars are unadjusted:
a split executed during the hold brings the later bars back by the effective split set, `intraday_store.split_rows`).

The account (`mimic.live_strategies`): 1/`insider_long_account_slices` of the equity, at most `insider_long_max_open`
open, Reg T, at most 1% of the stock's dollar volume (the engine's ``adv_cap``); capital quality 0.05 (`mimic.QUALITY`:
it gives way to every other book), preemptible whole, the latest first, de-levered with the capitulation book
(``delever_order`` 0); a dip or capitulation order is sized as if its longs were freed and its fill sells them
(`Rules.fills_make_room`).
"""
from __future__ import annotations

from datetime import date
from functools import lru_cache
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from config import settings

ET = "America/New_York"


def _ns(day, hhmm: str) -> int:
    return int(pd.Timestamp(f"{pd.Timestamp(day).date()} {hhmm}", tz=ET).value)


@lru_cache(maxsize=1)
def _spy() -> pd.Series:
    from src.backtest import reliability as REL
    return REL.spy_daily()["Close"].astype(float)


@lru_cache(maxsize=1)
def _rows() -> pd.DataFrame:
    """Every insider row production reads (`insider_long.load_rows`, not deduplicated), in the files' order."""
    from src.signals import insider_long as IL
    return IL.load_rows()


def gate_sessions(lo, hi) -> List[date]:
    """The sessions in [``lo``, ``hi``] whose gate (at the close before them) is on."""
    from src.signals import dip_long as DL
    from src.signals import insider_long as IL
    out = []
    d = pd.Timestamp(lo).date()
    end = pd.Timestamp(hi).date()
    spy = _spy()
    while d <= end:
        if DL.is_session(d) and IL.gate_from_closes(spy, DL.prev_session(d)).get("on"):
            out.append(d)
        d = (pd.Timestamp(d) + pd.Timedelta(days=1)).date()
    return out


class _Bars:
    """The whole-market daily bars of the needed symbols, read once per session (production's reader)."""

    def __init__(self, fetch: bool = True):
        self.fetch = fetch
        self.by_sym: Dict[str, Dict[date, tuple]] = {}
        self.read: Dict[date, set] = {}               # each session -> the symbols already taken from it
        self.empty: set = set()                       # sessions without whole-market bars

    def ensure(self, sessions: Sequence[date], syms: Sequence[str]) -> None:
        from src.signals import cap_long
        want = set(syms)
        for s in sessions:
            if s in self.empty:
                continue
            todo = want - self.read.get(s, set())
            if not todo:
                continue
            g = cap_long._session_frame(s, fetch=self.fetch)
            if g is None:
                self.empty.add(s)
                continue
            self.read.setdefault(s, set()).update(todo)
            sub = g[g["sym"].isin(todo)]
            for r in sub.itertuples(index=False):
                self.by_sym.setdefault(r.sym, {})[s] = (float(r.o), float(r.h), float(r.l), float(r.c), r.v)

    def frame(self, sym: str, sessions: Sequence[date]) -> Optional[pd.DataFrame]:
        rows = self.by_sym.get(sym) or {}
        got = [(pd.Timestamp(s),) + rows[s] for s in sessions if s in rows]
        if not got:
            return None
        return pd.DataFrame(got, columns=["date", "o", "h", "l", "c", "v"]).set_index("date").sort_index()


def signals(lo, hi, bars: Optional[_Bars] = None) -> List[Tuple[date, dict]]:
    """Production's signals of the gate sessions in [``lo``, ``hi``]: [(session, signal)] in each session's rank order."""
    from src.signals import insider_long as IL
    from src.signals.sel_short import sessions_before
    bars = bars or _Bars()
    R = _rows()
    n = max(1, int(settings.insider_long_spy_high_sessions))
    days = gate_sessions(lo, hi)
    evs = {}
    for d in days:                                    # every gate session's events first (cheap) ...
        w_lo, w_hi = IL.filing_window(d)
        E = IL.events_from_rows(R, w_lo, w_hi, d)
        if not E.empty:
            evs[d] = E
    tks = sorted({IL.project_ticker(s) for E in evs.values() for s in E["sym"]})
    need = sorted({s for d in evs for s in sessions_before(d, n)})
    bars.ensure(need, tks)                            # ... then each session's bars read once for all their stocks
    out: List[Tuple[date, dict]] = []
    for d, E in evs.items():
        gate = IL.gate_from_closes(_spy(), IL.DL.prev_session(d))
        before = sessions_before(d, n)
        frames = {tk: bars.frame(tk, before) for tk in (IL.project_ticker(s) for s in E["sym"])}
        sig, _ = IL.decide(d, gate, E, {k: v for k, v in frames.items() if v is not None})
        out += [(d, s) for s in sig]
    return out


def _in_entry_terms(tk: str, entry: date, days: Sequence[date], vals: np.ndarray) -> np.ndarray:
    """Prices of ``days`` (unadjusted) in the entry session's terms: a split executed after the entry and on or before
    the day multiplies that day's price by split_to / split_from (the effective split set)."""
    from src.data.intraday_store import split_rows
    from src.data.polygon_client import to_polygon_symbol
    rows = split_rows() or {}
    out = np.asarray(vals, float).copy()
    dd = np.array([pd.Timestamp(x).value for x in days], np.int64)
    for e, sf, st in rows.get(to_polygon_symbol(tk).upper(), []):
        if e > entry:
            out[dd >= pd.Timestamp(e).value] *= float(st) / float(sf)
    return out


def piece(d: date, sig: dict, bars: _Bars, strategy: str = "ins", end=None) -> Optional[dict]:
    """The account engine's piece of one signal (module docstring)."""
    from src.backtest.strategies import dm as DM
    from src.signals.sel_short import session_after
    from src.signals import dip_long as DL
    tk = str(sig["ticker"])
    hold = max(1, int(settings.insider_long_hold_sessions))
    x_day = session_after(d, hold)
    last_day = pd.Timestamp(end).date() if end is not None else None
    sess = [d] + [session_after(d, k) for k in range(1, hold + 1)]
    if last_day is not None:
        sess = [s for s in sess if s <= last_day]
    bars.ensure(sess, [tk])
    f = bars.frame(tk, sess)
    if f is None or pd.Timestamp(d) not in f.index:
        return None                                   # no bar on the entry session: nothing bought
    days = [pd.Timestamp(x).date() for x in f.index]
    o = _in_entry_terms(tk, d, days, f["o"].to_numpy(float))
    c = _in_entry_terms(tk, d, days, f["c"].to_numpy(float))
    lo_ = _in_entry_terms(tk, d, days, f["l"].to_numpy(float))
    e = float(o[0])
    held = [k for k, x in enumerate(days) if x < x_day]
    after = [k for k, x in enumerate(days) if x >= x_day]
    if after:                                         # sold at the open of the first session at/after the exit's
        k_x = after[0]
        x, xns = float(o[k_x]), _ns(days[k_x], "09:30")
    else:                                             # stopped trading (or the data ends) first: its last close
        k_x = held[-1]
        x, xns = float(c[k_x]), _ns(days[k_x], "16:00")
    keep = [k for k in held if _ns(days[k], "16:00") <= xns]
    pt = np.array([_ns(days[k], "16:00") for k in keep], np.int64)
    pc, pl = c[keep], lo_[keep]
    later = [k for k in keep if days[k] != d]
    pt, pc, pl = DM._with_opens(pt, pc, pl, [_ns(days[k], "09:30") for k in later], o[later])
    if not (e > 0 and x > 0):
        return None
    dv20 = float(sig.get("dv20") or 0.0)
    hs = float(DM.half_spread_bps([dv20])[0])
    prev = DL.prev_session(d)
    return {"strategy": strategy, "tkn": tk, "ens": _ns(d, "09:30"), "xns": int(xns), "e": e, "x": x,
            "hs_in": hs, "hs_out": hs, "dv20": dv20, "pt": np.asarray(pt, np.int64), "pc": np.asarray(pc, float),
            "pl": np.asarray(pl, float), "rank": -float(sig.get("notional") or 0.0),
            "net": 100.0 * (x / e - 1) - 2 * hs / 100.0, "days": (int(xns) - _ns(d, "09:30")) / 86400e9,
            "asset": tk, "month": f"{d.year:04d}-{d.month:02d}", "open_marks": True,
            "pick_day": int((pd.Timestamp(prev) - pd.Timestamp("1970-01-01")).days),
            "ins_fd": sig.get("fd"), "ins_notional": float(sig.get("notional") or 0.0)}


def pieces(lo, hi, period: Optional[str] = None, strategy: str = "ins", end=None,
           fetch: bool = True) -> List[dict]:
    """The book's longs decided on sessions [``lo``, ``hi``] as the account engine's pieces. ``period`` is accepted for
    the other adapters' signature (the whole-market bars hold every listed and delisted stock, set A0's years too)."""
    from src.signals.sel_short import session_after
    bars = _Bars(fetch=fetch)
    sig = signals(lo, hi, bars)
    hold = max(1, int(settings.insider_long_hold_sessions))
    sess = sorted({session_after(d, k) for d, _ in sig for k in range(0, hold + 1)})
    bars.ensure(sess, sorted({str(s["ticker"]) for _, s in sig}))     # the holds' bars, each session read once
    out = []
    for d, s in sig:
        p = piece(d, s, bars, strategy=strategy, end=end)
        if p is not None:
            out.append(p)
    return out
