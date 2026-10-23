"""THE FORWARD CHECK — every live trade beside the simulator's own construction of it (PREREG185, research
2026-10-22; production from 2026-10-23 — the user, on wiring it into the end-of-day email and building the full
simulation of the live period: "Do both for number 4").

Both sides see the same market, so the market's noise cancels and only the execution model is measured: a gap shows
in a few dozen trades, where the live P&L needs a year or more (PREREG183). For every live-book trade (`books.is_live`)
ENTERED from the live account's fresh start (`live_account_fresh_start`; a margin lot folded into its trade) the
simulator's construction on Polygon's 30-minute bars AS TRADED (04:00-20:00 ET):

* a selection short enters at its decision bar's close (the pick bar, or the turning bar for a fall / ETF turn entry)
  and leaves at the first of the simulator's marks (every regular-hours bar close, production's off-grid ticks —
  `mimic.tick_marks`, the 09:30 tick the opening print) where its stamped target, its arm's cover multiple
  (`sel_short.cover_multiple`), its volatility exit in profit (`sel_short.volnorm_exit` / `volnorm_ratio`, the live
  ATR% by `sel_short.atr_at`) or its stamped deadline fires;
* a dip or capitulation order fills at AUDIT151's attainable open (an order arriving 09:30:30: the median trade of the
  next 5 seconds, else the minute's close; at or under the limit, else the first minute touching the limit at the
  limit, booked at that 30-minute bar's end) or at the first regular-hours bar whose low reaches the limit (at the
  limit, at the bar's end), inside its order life (`dip_long_limit_sessions` / `cap_long_limit_sessions`), and is sold
  at the opening print after the first close above its exit average since the fill or after its time limit (the
  book's `*_exit_sma_sessions` / `*_max_hold_sessions`) — PREREG187 found the opening print unbiased for these exits
  (-2 bp, t -0.95, on 665 exits 2018-2026);
* the crash book's long (TQQQ on QQQ's decisions) is bought at the entry session's opening print and sold after QQQ's
  first close above its `crash_long_exit_sma_sessions` average since the entry (or `crash_long_max_hold_sessions`
  sessions) — at that session's 16:00-16:30 bar close with `crash_long_pm_exit` (the simulator's `mimic.pm_exit`),
  else at the next opening print;
* each market leg pays Polygon's quoted half-spread at its instant (the last NBBO within 10 minutes, a 09:30:00 leg
  read at 09:30:30 — `quotes._read_at`; capped at `quotes.CAP_BP`) plus IBKR's commission and a sale's fees
  (`account.comm` / `sale_fees`) — the simulator's quote model; this module keeps the quotes it fetches in its own
  folder (never the research store).

Paired per trade (in % of the trade, + = live better than the simulator): the entry, the exit at the SAME instant (the
simulator's mark when live left), the exit's timing (the simulator's own exit against its mark at live's instant, own
exits only — a forced close is compared at the same instant only, `live_calibration.forced_exit`), the costs, the net
trade, and IBKR's fills against the ledger's. Per book: the mean over EPISODES (`live_calibration.episodes`: trades
entered or closed in one session count once) with its standard error.

THE TRACKING BAND: every n-session window (n = 1 .. `BAND_SESSIONS`) of every draw's daily NAV of the live
configuration's draws (`bank.LIVE_RUNS`, all sets): its percentiles, and a HELD-OUT band shifted from the draws' own
mean daily log rate to `HELD_OUT_GROWTH` a year (PREREG152's held-out reading, the cautious expectation); the live
account's growth since the fresh start ($50,000 + the P&L of the trades entered from it, open trades at each session's
close) and its percentile in both.

One record per market day (`cache/ml/forward_check/<day>.json`, the EOD chain's ``forward_check`` step after the live
calibration; `python -m src.backtest.forward_check [--day D]`). `health()` gives the email's scorer banner one line
and a NOTE — never a problem — when a book's live-minus-simulator net trade is beyond its noise (|t| >= 2 over at
least `MIN_EPISODES` episodes) or the account sits outside its band's 5-95% range. Nothing trades on it; a gap is the
next study's.
"""
from __future__ import annotations

import json
import math
import os
from collections import defaultdict
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from config import settings

ET = "America/New_York"
BAND_SESSIONS = 63
BAND_QUANTILES = (5, 25, 50, 75, 95)
HELD_OUT_GROWTH = 0.61          # PREREG152's held-out reading (each year at the settings tuned on the other eight)
MIN_EPISODES = 5                # a book's gap is judged once it rests on this many episodes
GAP_T = 2.0
SAME_TICK_NS = 10 * 60 * 10 ** 9   # one slot: a live tick runs seconds to minutes after the bar's end
GAP_KEYS = ("entry_gap_pct", "ibkr_entry_vs_ledger_pct", "exit_same_instant_gap_pct", "ibkr_exit_vs_ledger_pct",
            "exit_timing_gap_pct", "cost_gap_pct", "net_gap_pct", "ibkr_minus_sim_pct")
SHARE_KEYS = ("dip_account_shares", "sel_account_shares", "cap_account_shares", "crash_account_shares")
ARMS = ("vol", "etf", "fall", "dip", "cap", "crash")       # the live arms this module rebuilds


def out_dir() -> Path:
    return Path(getattr(settings, "forward_check_dir", "cache/ml/forward_check"))


def out_path(d: date) -> Path:
    return out_dir() / f"{d.isoformat()}.json"


def _ns(x) -> Optional[int]:
    if x in (None, ""):
        return None
    import pandas as pd
    try:
        t = pd.Timestamp(str(x))
    except (ValueError, TypeError):
        return None
    t = t.tz_localize("UTC") if t.tzinfo is None else t
    return int(t.value)


def _et(ns: int):
    import pandas as pd
    return pd.Timestamp(int(ns), unit="ns", tz="UTC").tz_convert(ET)


def _day(x) -> Optional[date]:
    try:
        return date.fromisoformat(str(x)[:10]) if x else None
    except ValueError:
        return None


def since_ns() -> int:
    """The live account's fresh start (`live_account_fresh_start`): the record's first instant."""
    return int(_ns(getattr(settings, "live_account_fresh_start", "") or "2026-10-20T09:30:00-04:00"))


def shares_of(t: dict) -> float:
    for k in SHARE_KEYS:
        if t.get(k) not in (None, ""):
            return float(t[k])
    return float(t.get("broker_fill_qty") or 0)


# ── the population ────────────────────────────────────────────────────────────────────────────────────────────────

def live_trades(trades: Sequence[dict], since: Optional[int] = None) -> List[dict]:
    """The live books' trades entered from ``since`` (default the fresh start), each margin lot folded into its trade
    (``_lots``: the lots split off by the liquidation, `tracker._split_margin_lot`)."""
    from src.performance import books
    s0 = since_ns() if since is None else int(since)
    live = [t for t in trades if books.is_live(t) and (_ns(t.get("entry_datetime")) or 0) >= s0]
    lots = defaultdict(list)
    for t in live:
        if t.get("margin_lot_of"):
            lots[str(t["margin_lot_of"])].append(t)
    out = [dict(t) for t in live if not t.get("margin_lot_of")]
    for t in out:
        t["_lots"] = lots.get(str(t.get("trade_id")), [])
    return out


def arm_of(t: dict) -> str:
    from src.backtest import live_calibration as LC
    return LC.arm_of(t) or str(t.get("entry_mechanism") or "")


# ── the market: Polygon's bars as traded, the attainable open, the quotes ─────────────────────────────────────────

class Market:
    """The prices the construction reads, fetched once per build (bars per name and span; the 09:30:30 trades per
    name and session and the quotes per leg cached on disk under `out_dir()`)."""

    def __init__(self, cache_dir: Optional[Path] = None):
        self.dir = Path(cache_dir) if cache_dir is not None else out_dir()
        self._bars: Dict[tuple, Optional[dict]] = {}
        self._open = self._load("open_trades.json")
        self._quotes = self._load("quotes.json")
        self.requests = 0

    def _load(self, name: str) -> dict:
        try:
            return json.loads((self.dir / name).read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return {}

    def save(self) -> None:
        self.dir.mkdir(parents=True, exist_ok=True)
        for name, obj in (("open_trades.json", self._open), ("quotes.json", self._quotes)):
            p = self.dir / name
            tmp = p.with_suffix(f".{os.getpid()}.tmp")
            tmp.write_text(json.dumps(obj), encoding="utf-8")
            os.replace(tmp, p)

    def bars(self, tk: str, lo: str, hi: str) -> Optional[dict]:
        """The name's 30-minute bars 04:00-20:00 ET over [lo, hi] as traded (`mimic.ExtBars`' form)."""
        import pandas as pd
        from src.backtest import mimic as M
        from src.data import polygon_client as PC
        key = (tk, lo, hi)
        if key in self._bars:
            return self._bars[key]
        df = PC.get_intraday_bars_range(tk, lo, hi, adjusted=False, rth_only=False)
        self.requests += 1
        if df is None or df.empty:
            self._bars[key] = None
            return None
        start = pd.DatetimeIndex(df.index)
        start = start.tz_localize("UTC") if start.tz is None else start.tz_convert("UTC")
        et = start.tz_convert(ET)
        mins = et.hour * 60 + et.minute
        b = {"start": start.asi8.astype(np.int64), "end": start.asi8.astype(np.int64) + M.BAR_NS,
             "o": df["Open"].to_numpy(float), "h": df["High"].to_numpy(float), "l": df["Low"].to_numpy(float),
             "c": df["Close"].to_numpy(float), "v": df["Volume"].to_numpy(float),
             "rth": np.asarray((mins >= 570) & (mins < 960), bool),
             "day": (et.normalize().tz_localize(None) - M.EPOCH).days.to_numpy().astype(np.int64),
             "date": [d.date() for d in et.normalize()]}
        self._bars[key] = b
        return b

    def open_arrival(self, tk: str, s: date) -> dict:
        """{"arrival": the median trade 09:30:30-35 (else the 09:30 minute's close), "how", "mins": [(minute start ns,
        low, close)]} of the session's regular-hours minutes — cached per (name, session)."""
        import pandas as pd
        from src.data import polygon_client as PC
        key = f"{tk}|{s.isoformat()}"
        if key not in self._open:
            sym = PC.to_polygon_symbol(tk)
            a = pd.Timestamp(datetime(s.year, s.month, s.day, 9, 30, 30), tz=ET).value
            js = PC._get(f"/v3/trades/{sym}", {"timestamp.gte": int(a), "timestamp.lt": int(a + 5 * 10 ** 9),
                                               "limit": 50000, "sort": "timestamp", "order": "asc"}) or {}
            m1 = PC._get(f"/v2/aggs/ticker/{sym}/range/1/minute/{s.isoformat()}/{s.isoformat()}",
                         {"adjusted": "false", "sort": "asc", "limit": 50000}) or {}
            self.requests += 2
            px = [float(r["price"]) for r in js.get("results") or [] if r.get("price")]
            mins = []
            for r in m1.get("results") or []:
                t = pd.Timestamp(r["t"], unit="ms", tz="UTC").tz_convert(ET)
                if (t.hour, t.minute) >= (9, 30) and t.hour < 16:
                    mins.append([int(t.value), float(r["l"]), float(r["c"])])
            arr = float(np.median(px)) if px else (mins[0][2] if mins else None)
            self._open[key] = {"arrival": arr, "how": "trades" if px else "minute_close", "mins": mins}
        return self._open[key]

    def half_spread_bp(self, tk: str, ns: int) -> Optional[float]:
        """Polygon's quoted half-spread at the leg (`quotes._read_at`: a 09:30:00 leg at 09:30:30), None without a
        quote in the 10 minutes before it."""
        from src.backtest import quotes as Q
        from src.data import polygon_client as PC
        key = f"{tk}|{int(ns)}"
        if key not in self._quotes:
            at = Q._read_at(int(ns))
            sym = PC.to_polygon_symbol(str(tk).split("@")[0])
            js = PC._get(f"/v3/quotes/{sym}", {"timestamp.lte": int(at.value),
                                               "timestamp.gte": int(at.value) - 600 * 10 ** 9,
                                               "order": "desc", "sort": "timestamp", "limit": 1}) or {}
            self.requests += 1
            res = js.get("results") or []
            hs = None
            if res:
                b, a = float(res[0].get("bid_price") or 0), float(res[0].get("ask_price") or 0)
                if a > 0 and b > 0 and a >= b:
                    hs = 1e4 * (a - b) / (a + b)
            self._quotes[key] = hs
        return self._quotes[key]


# ── the simulator's construction of one trade ─────────────────────────────────────────────────────────────────────

def open_print(b: dict, s: date) -> Optional[Tuple[int, float]]:
    """(the 09:30 instant, the session's opening print) — the first regular-hours bar's open."""
    import pandas as pd
    t = int(pd.Timestamp(datetime(s.year, s.month, s.day, 9, 30), tz=ET).value)
    j = int(np.searchsorted(b["start"], t))
    if j < len(b["start"]) and int(b["start"][j]) == t:
        return t, float(b["o"][j])
    return None


def marks(b: dict, t0: int, t1: int) -> Tuple[np.ndarray, np.ndarray]:
    """The simulator's marks strictly inside (t0, t1]: every regular-hours bar close at its end and production's
    off-grid ticks (`mimic.tick_marks`), sorted."""
    from src.backtest import mimic as M
    rth = b["rth"] & (b["end"] > t0) & (b["end"] <= t1)
    T = list(b["end"][rth])
    P = list(b["c"][rth])
    tt, pp, _h, _l = M.tick_marks(b, t0, t1 + 1)
    T += list(tt)
    P += list(pp)
    o = np.argsort(np.asarray(T, np.int64), kind="stable")
    return np.asarray(T, np.int64)[o], np.asarray(P, float)[o]


def mark_at(b: dict, t: int) -> Optional[float]:
    """The simulator's price at instant ``t``: the opening print at the 09:30 tick, else the close of the latest bar
    ending at or before it."""
    et = _et(t)
    if et.hour == 9 and et.minute == 30:
        op = open_print(b, et.date())
        if op is not None:
            return op[1]
    j = int(np.searchsorted(b["end"], t, side="right")) - 1
    return float(b["c"][j]) if j >= 0 else None


def daily(b: dict):
    """The book's daily construction (`dip_long.daily_bars`) from the regular-hours bars."""
    import pandas as pd
    r = b["rth"]
    d = pd.DataFrame({"date": pd.to_datetime([x for x, k in zip(b["date"], r) if k]),
                      "o": b["o"][r], "h": b["h"][r], "l": b["l"][r], "c": b["c"][r]})
    g = d.groupby("date", sort=True)
    return pd.DataFrame({"o": g.o.first(), "h": g.h.max(), "l": g.l.min(), "c": g.c.last()})


def attainable_open(mk: Market, tk: str, s: date, lim: Optional[float]) -> Optional[dict]:
    """AUDIT151's D_OPEN for an order resting at the open (`strategies.dip.reprice_open`): the 09:30:30 arrival at or
    under the limit -> filled at it; else the first minute whose low reaches the limit -> at the limit, booked at that
    30-minute bar's end; else None."""
    import pandas as pd
    from src.backtest import mimic as M
    r = mk.open_arrival(tk, s)
    if r.get("arrival") is None:
        return None
    if lim is None or r["arrival"] <= lim:
        return {"ns": int(pd.Timestamp(datetime(s.year, s.month, s.day, 9, 30), tz=ET).value),
                "price": float(r["arrival"]), "kind": "open", "how": r["how"]}
    for t, lo, _c in r["mins"]:
        if lo <= lim:
            start = _et(int(t)).floor("30min")
            return {"ns": int(start.value + M.BAR_NS), "price": float(lim), "kind": "touch", "how": "open_touch"}
    return None


def _sessions_held(e: date, last: date) -> int:
    from src.performance.market_calendar import market_days_between
    return int(market_days_between(e - timedelta(days=1), last))


def book_rules(book: str) -> dict:
    p = "dip_long" if book == "dip" else "cap_long"
    return {"exit_sma": int(getattr(settings, f"{p}_exit_sma_sessions")),
            "max_hold": int(getattr(settings, f"{p}_max_hold_sessions")),
            "order_life": int(getattr(settings, f"{p}_limit_sessions"))}


def sim_order_book(t: dict, b: dict, mk: Market, book: str) -> dict:
    """A dip or capitulation order filled and sold by the simulator's rule on the same bars."""
    import pandas as pd
    p = "dip" if book == "dip" else "cap"
    rules = book_rules(book)
    lim = float(t.get(f"{p}_limit") or 0) or None
    od = _day(t.get(f"{p}_order_day") or t.get(f"{p}_signal_day") or t.get("entry_date"))
    D = daily(b)
    out: dict = {"rule": rules}
    fill = None
    if lim is None:                                       # no limit stamped: an open entry at the attainable price
        e0 = _day(t.get(f"{p}_entry_day") or t.get("entry_date"))
        a = attainable_open(mk, str(t["ticker"]), e0, None) if e0 else None
        if a is not None:
            fill = dict(a, session=e0.isoformat())
        sess = []
    else:
        sess = [x.date() for x in D.index if od is not None and x.date() >= od][:rules["order_life"]]
    for s in sess:
        op = open_print(b, s)
        if op is not None and op[1] <= lim:
            a = attainable_open(mk, str(t["ticker"]), s, lim)
            if a is not None:
                fill = dict(a, session=s.isoformat())
                break
            continue
        for j in range(len(b["start"])):
            if b["rth"][j] and b["date"][j] == s and b["l"][j] <= lim:
                fill = {"session": s.isoformat(), "ns": int(b["end"][j]), "price": float(lim), "kind": "touch"}
                break
        if fill:
            break
    out["fill"] = fill
    if fill is None:
        return out
    e = date.fromisoformat(fill["session"])
    sma = D["c"].rolling(rules["exit_sma"]).mean()
    days = [x.date() for x in D.index]
    for i, dd in enumerate(days):
        if dd <= e or i == 0:
            continue
        prev = days[i - 1]
        win = D[(D.index >= pd.Timestamp(e)) & (D.index <= pd.Timestamp(prev))]
        hit = bool((win["c"] > sma.loc[win.index]).any())
        why = "rebound" if hit else ("time" if _sessions_held(e, prev) >= rules["max_hold"] else None)
        if why:
            op = open_print(b, dd)
            if op is not None:
                out["exit"] = {"session": dd.isoformat(), "ns": op[0], "price": op[1], "why": why}
            break
    return out


def sim_short(t: dict, b: dict) -> dict:
    """A selection short entered at its decision bar's close and closed on the simulator's marks by its stamped
    target, its arm's cover, its volatility exit in profit or its stamped deadline."""
    from src.signals import sel_short as SS
    arm = arm_of(t)
    if t.get("sel_turn_bar_end"):
        e_ns, e_px, kind = _ns(t["sel_turn_bar_end"]), float(t.get("sel_turn_close") or 0), "turn"
    else:
        e_ns, e_px, kind = _ns(t.get("sel_bar_end")), float(t.get("sel_bar_close") or 0), "immediate"
    out: dict = {"entry": {"ns": e_ns, "price": e_px, "kind": kind}}
    if not e_ns or not e_px:
        return out
    lvl = t.get("sel_target_price")
    lvl = float(lvl) if lvl not in (None, "") else None
    dl = _ns(t.get("sel_deadline"))
    apk = t.get("sel_atr_pct")
    apk = float(apk) if apk not in (None, "") else None
    mult = float(SS.cover_multiple(arm))
    vn, ratio = bool(SS.volnorm_exit(arm)), float(SS.volnorm_ratio(arm))
    T, P = marks(b, e_ns, int(b["end"][-1]))
    for tt, m in zip(T, P):
        tt = int(tt)
        if dl is not None and tt >= dl:
            out["exit"] = {"ns": tt, "price": float(m), "why": "time"}
            break
        if lvl is not None and m <= lvl:
            out["exit"] = {"ns": tt, "price": float(m), "why": "target"}
            break
        if m >= mult * e_px:
            out["exit"] = {"ns": tt, "price": float(m), "why": "cover"}
            break
        if vn and apk and m < e_px:
            d, k = SS.last_completed_bar(_et(tt).to_pydatetime())
            a = SS.atr_at(str(t["ticker"]), d, k)
            if a is not None and a <= ratio * apk:
                out["exit"] = {"ns": tt, "price": float(m), "why": "volnorm"}
                break
    return out


def sim_crash(t: dict, b: dict, q: dict) -> dict:
    """The crash book's long: the traded fund ``b`` (TQQQ) bought at the entry session's opening print, sold after the
    decision symbol's (``q``, QQQ's) first close above its exit average since the entry or its time limit — at that
    session's 16:00-16:30 bar close with `crash_long_pm_exit`, else at the next opening print."""
    import pandas as pd
    e0 = _day(t.get("entry_date"))
    out: dict = {"rule": {"exit_sma": int(settings.crash_long_exit_sma_sessions),
                          "max_hold": int(settings.crash_long_max_hold_sessions),
                          "pm_exit": bool(getattr(settings, "crash_long_pm_exit", False))}}
    op = open_print(b, e0) if e0 else None
    if op is None:
        return out
    out["fill"] = {"session": e0.isoformat(), "ns": op[0], "price": op[1], "kind": "open"}
    Q = daily(q)
    sma = Q["c"].rolling(out["rule"]["exit_sma"]).mean()
    days = [x.date() for x in Q.index]
    for i, s in enumerate(days):
        if s < e0:
            continue
        win = Q[(Q.index >= pd.Timestamp(e0)) & (Q.index <= pd.Timestamp(s))]
        hit = bool((win["c"] > sma.loc[win.index]).any())
        why = "rebound" if hit else ("time" if _sessions_held(e0, s) >= out["rule"]["max_hold"] else None)
        if not why:
            continue
        if out["rule"]["pm_exit"]:
            t16 = int(pd.Timestamp(datetime(s.year, s.month, s.day, 16, 0), tz=ET).value)
            j = int(np.searchsorted(b["start"], t16))
            if j < len(b["start"]) and int(b["start"][j]) == t16:
                out["exit"] = {"session": s.isoformat(), "ns": int(b["end"][j]), "price": float(b["c"][j]),
                               "why": f"{why}_pm"}
                break
        if i + 1 < len(days):
            nx = open_print(b, days[i + 1])
            if nx is not None:
                out["exit"] = {"session": days[i + 1].isoformat(), "ns": nx[0], "price": nx[1], "why": why}
        break
    return out


# ── the comparison ────────────────────────────────────────────────────────────────────────────────────────────────

def _lg(a, b) -> Optional[float]:
    return math.log(a / b) if a and b and a > 0 and b > 0 else None


def compare(t: dict, s: dict, hs_fn: Callable[[str, str, int, str], Tuple[float, bool]]) -> dict:
    """One trade's pairs, in % of the trade, + = live better than the simulator. ``hs_fn(arm, name, ns, leg)`` ->
    (the leg's half-spread bp, fell back)."""
    from src.backtest import account as A
    from src.backtest import live_calibration as LC
    from src.backtest import quotes as Q
    side = -1 if t.get("entry_mechanism") == "sel_short" else 1
    arm = arm_of(t)
    tk = str(t["ticker"])
    eL = float(t.get("entry_price") or 0)
    xL = float(t["exit_price"]) if t.get("exit_price") not in (None, "") else None
    closed = t.get("status") == "CLOSED"
    e_ns_L, x_ns_L = _ns(t.get("entry_datetime")), _ns(t.get("exit_datetime"))
    fe = (s.get("fill") if arm in ("dip", "cap", "crash") else s.get("entry")) or {}
    eS, e_ns_S = fe.get("price"), fe.get("ns")
    sx = s.get("exit") or {}
    xS, x_ns_S = sx.get("price"), sx.get("ns")
    b = s.get("_b")
    xS_at_L = mark_at(b, x_ns_L) if (b is not None and x_ns_L) else None
    n = shares_of(t)
    forced = LC.forced_exit(t) if closed else None
    r: dict = {"id": t.get("trade_id"), "ticker": tk, "arm": arm, "side": side, "status": t.get("status"),
               "entry_date": str(t.get("entry_date"))[:10], "exit_date": str(t.get("exit_date") or "")[:10] or None,
               "exit_reason": t.get("exit_reason"), "forced": forced, "shares": n, "fill_kind_sim": fe.get("kind"),
               "entry_live": {"at": str(_et(e_ns_L)) if e_ns_L else None, "price": eL,
                              "ibkr": t.get("broker_fill_price")},
               "entry_sim": {"at": str(_et(e_ns_S)) if e_ns_S else None, "price": eS},
               "exit_live": {"at": str(_et(x_ns_L)) if x_ns_L else None, "price": xL,
                             "ibkr": t.get("broker_exit_fill_price"), "sim_mark_same_instant": xS_at_L},
               "exit_sim": {"at": str(_et(x_ns_S)) if x_ns_S else None, "price": xS, "why": sx.get("why")},
               "sim_rule": s.get("rule")}
    g = _lg(eL, eS)
    r["entry_gap_pct"] = (-side * g * 100.0) if g is not None else None
    fi = t.get("broker_fill_price")
    g = _lg(float(fi), eL) if fi else None
    r["ibkr_entry_vs_ledger_pct"] = (-side * g * 100.0) if g is not None else None
    if not closed:
        return r
    g = _lg(xL, xS_at_L)
    r["exit_same_instant_gap_pct"] = (side * g * 100.0) if g is not None else None
    fo = t.get("broker_exit_fill_price")
    g = _lg(float(fo), xL) if fo else None
    r["ibkr_exit_vs_ledger_pct"] = (side * g * 100.0) if g is not None else None
    g = _lg(xS_at_L, xS)
    r["exit_timing_gap_pct"] = (side * g * 100.0) if (g is not None and not forced) else None
    r["same_tick"] = (abs(int(x_ns_S) - int(x_ns_L)) < SAME_TICK_NS) if (x_ns_S and x_ns_L) else None
    r["same_session"] = (_et(x_ns_S).date() == _et(x_ns_L).date()) if (x_ns_S and x_ns_L) else None
    hs_in, fb = 0.0, 0
    if arm in ("dip", "cap"):
        if fe.get("kind") == "open":
            v, m = hs_fn(arm, tk, int(e_ns_S), "in")
            hs_in, fb = min(Q.CAP_BP, v), fb + int(m)
    elif e_ns_S:
        v, m = hs_fn(arm, tk, int(e_ns_S), "in")
        hs_in, fb = min(Q.CAP_BP, v), fb + int(m)
    hs_out = None
    if x_ns_S:
        v, m = hs_fn(arm, tk, int(x_ns_S), "out")
        hs_out, fb = min(Q.CAP_BP, v), fb + int(m)
    cost_sim = None
    if eS and xS and n > 0 and hs_out is not None:
        fees = A.comm(n, eS) + A.comm(n, xS) + (A.sale_fees(n, eS) if side < 0 else A.sale_fees(n, xS))
        cost_sim = (hs_in + hs_out) / 1e4 + fees / (n * eS)
    borrow = float(t.get("borrow_fee_ps") or 0) / eL if (side < 0 and eL) else 0.0
    divs = 0.0
    for dv in t.get("dividends") or []:
        try:
            divs += float(dv.get("frac") or 0)
        except (AttributeError, TypeError, ValueError):
            pass
    gross_L = side * (xL / eL - 1.0) if (xL and eL) else None
    net_L = float(t["return_pct"]) / 100.0 if t.get("return_pct") is not None else None
    cost_L = (gross_L - net_L - borrow + side * divs) if (gross_L is not None and net_L is not None) else None
    r.update(cost_sim_pct=cost_sim * 100.0 if cost_sim is not None else None,
             cost_ledger_pct=cost_L * 100.0 if cost_L is not None else None,
             cost_gap_pct=((cost_sim - cost_L) * 100.0) if (cost_sim is not None and cost_L is not None) else None,
             quote_fallbacks=fb, borrow_pct=borrow * 100.0, dividends_pct=divs * 100.0)
    gross_S = side * (xS / eS - 1.0) if (xS and eS) else None
    net_S = (gross_S - cost_sim - borrow + side * divs) if (gross_S is not None and cost_sim is not None) else None
    r.update(net_ledger_pct=net_L * 100.0 if net_L is not None else None,
             net_sim_pct=net_S * 100.0 if net_S is not None else None,
             net_gap_pct=((net_L - net_S) * 100.0) if (net_L is not None and net_S is not None) else None)
    fi_, fo_ = t.get("broker_fill_price"), t.get("broker_exit_fill_price")
    if fi_ and fo_ and n > 0:
        fi_, fo_ = float(fi_), float(fo_)
        fees = A.comm(n, fi_) + A.comm(n, fo_) + (A.sale_fees(n, fi_) if side < 0 else A.sale_fees(n, fo_))
        net_I = side * (fo_ / fi_ - 1.0) - fees / (n * fi_) - borrow + side * divs
        r["net_ibkr_pct"] = net_I * 100.0
        r["ibkr_minus_sim_pct"] = (net_I - net_S) * 100.0 if net_S is not None else None
    return r


def by_book(rows: Sequence[dict], key: str) -> dict:
    """Per arm: the mean of ``key`` per trade and its standard error over EPISODES (`live_calibration.episodes`)."""
    from src.backtest import live_calibration as LC
    out = {}
    for arm in sorted({r["arm"] for r in rows}):
        R = [r for r in rows if r["arm"] == arm and r.get(key) is not None]
        if not R:
            continue
        ep = LC.episodes([{"entry_date": r["entry_date"], "exit_date": r["exit_date"]} for r in R])
        em = [float(np.mean([R[i][key] for i in g])) for g in ep]
        m = float(np.mean([r[key] for r in R]))
        se = float(np.std(em, ddof=1) / math.sqrt(len(em))) if len(em) > 1 else None
        out[arm] = {"n": len(R), "episodes": len(em), "mean": m, "se_episodes": se,
                    "t": (m / se) if se else None, "median": float(np.median([r[key] for r in R]))}
    return out


# ── the tracking band ─────────────────────────────────────────────────────────────────────────────────────────────

def _runs_dir() -> Tuple[Path, str]:
    from src.backtest import bank as BK
    rel, label = BK.LIVE_RUNS
    return Path(settings.backtest_research_dir) / rel, label


def band(runs: Optional[Dict[str, Dict[int, dict]]] = None) -> dict:
    """{n: {"in_sample": {q: growth %}, "held_out": {q: ...}, "windows"}} over every n-session window of every
    draw's daily NAV (`bank.LIVE_RUNS`), plus the per-session log rates — cached by the runs' label and files."""
    from src.backtest import bank as BK
    cache = out_dir() / "band.json"
    key = None
    if runs is None:
        rd, label = _runs_dir()
        files = sorted(rd.glob(f"*_{label}_s*.pkl"))
        key = f"{rd}|{label}|{len(files)}|{max((f.stat().st_mtime_ns for f in files), default=0)}"
        try:
            got = json.loads(cache.read_text(encoding="utf-8"))
            if got.get("key") == key:
                return got
        except (OSError, ValueError):
            pass
        runs = BK.load_runs(rd, label)
    W: Dict[int, list] = {n: [] for n in range(1, BAND_SESSIONS + 1)}
    rates = []
    for _st, by in runs.items():
        for _seed, rec in by.items():
            v = np.asarray(((rec.get("nav") or {}).get("values")) or [], float)
            if len(v) < 2:
                continue
            lv = np.log(v)
            rates.append((lv[-1] - lv[0]) / (len(lv) - 1))
            for n in W:
                if len(lv) > n:
                    W[n].append(lv[n:] - lv[:-n])
    r_in = float(np.mean(rates)) if rates else 0.0
    shift = r_in - math.log1p(HELD_OUT_GROWTH) / 252.0
    out = {"key": key, "source": "bank.LIVE_RUNS", "rate_in_sample_per_session": r_in,
           "held_out_growth": HELD_OUT_GROWTH, "shift_per_session": shift, "band": {}}
    for n, xs in W.items():
        if not xs:
            continue
        x = np.sort(np.concatenate(xs))
        out["band"][str(n)] = {"windows": int(len(x)),
                               "in_sample": {str(q): float(np.expm1(np.percentile(x, q)) * 100.0) for q in BAND_QUANTILES},
                               "held_out": {str(q): float(np.expm1(np.percentile(x - n * shift, q)) * 100.0)
                                            for q in BAND_QUANTILES},
                               "x": [float(v) for v in x[:: max(1, len(x) // 2000)]]}
    if key is not None:
        cache.parent.mkdir(parents=True, exist_ok=True)
        tmp = cache.with_suffix(f".{os.getpid()}.tmp")
        tmp.write_text(json.dumps(out), encoding="utf-8")
        os.replace(tmp, cache)
    return out


def percentile(bd: dict, n: int, g_pct: float, held_out: bool = False) -> Optional[float]:
    """The share (%) of the band's n-session windows below the live growth ``g_pct``."""
    b = (bd.get("band") or {}).get(str(n))
    if not b or not b.get("x"):
        return None
    x = np.asarray(b["x"], float) - (n * float(bd.get("shift_per_session", 0.0)) if held_out else 0.0)
    return float((np.expm1(x) * 100.0 < g_pct).mean() * 100.0)


def account_path(trades: Sequence[dict], mk: Market, today: date) -> List[dict]:
    """The live account since the fresh start at each session's close: `sel_short_account_initial` + the P&L of the
    trades entered from it — closed ones (a margin lot split off by a liquidation its own trade) at their ledger
    return, open ones at the session's close."""
    import pandas as pd
    from src.performance.market_calendar import market_days_between
    s0 = since_ns()
    T = [t for t in trades if (_ns(t.get("entry_datetime")) or 0) >= s0]
    start = _et(s0).date()
    sessions, d = [], start
    while d <= today:
        if market_days_between(d - timedelta(days=1), d) == 1:
            sessions.append(d)
        d += timedelta(days=1)
    if not sessions:
        return []
    init = float(getattr(settings, "sel_short_account_initial", 50_000.0))
    closes = {}
    for t in T:
        tk = str(t["ticker"])
        if tk not in closes:
            b = mk.bars(tk, sessions[0].isoformat(), sessions[-1].isoformat())
            closes[tk] = daily(b)["c"] if b is not None else pd.Series(dtype=float)
    path = []
    for i, s in enumerate(sessions):
        close_ns = int(pd.Timestamp(datetime(s.year, s.month, s.day, 16, 0), tz=ET).value)
        pnl = 0.0
        for t in T:
            e_ns = _ns(t.get("entry_datetime"))
            if e_ns is None or e_ns > close_ns:
                continue
            side = -1 if t.get("entry_mechanism") == "sel_short" else 1
            n, e = shares_of(t), float(t.get("entry_price") or 0)
            x_ns = _ns(t.get("exit_datetime"))
            if t.get("status") == "CLOSED" and x_ns is not None and x_ns <= close_ns:
                pnl += n * e * float(t.get("return_pct") or 0) / 100.0
            else:
                c = closes.get(str(t["ticker"]))
                cx = float(c.loc[pd.Timestamp(s)]) if c is not None and pd.Timestamp(s) in c.index else None
                if cx and e:
                    pnl += n * e * side * (cx / e - 1.0)
        path.append({"session": s.isoformat(), "n": i + 1, "growth_pct": pnl / init * 100.0, "equity": init + pnl})
    return path


# ── the record ────────────────────────────────────────────────────────────────────────────────────────────────────

def _hs_fn(mk: Market) -> Callable[[str, str, int, str], Tuple[float, bool]]:
    try:
        from src.backtest import quotes as Q
        med = Q.table().med
    except Exception:                                          # noqa: BLE001 — the research store may be absent
        med = {}

    def fn(arm: str, tk: str, ns: int, leg: str) -> Tuple[float, bool]:
        v = mk.half_spread_bp(tk, ns)
        if v is not None:
            return float(v), False
        return float(med.get((arm, leg), 50.0 if arm in ("vol", "etf", "fall") else 1.0)), True
    return fn


def build(day: date, trades: Optional[Sequence[dict]] = None, mk: Optional[Market] = None,
          runs: Optional[Dict[str, Dict[int, dict]]] = None) -> dict:
    """The day's record: every live trade from the fresh start beside the simulator's construction, the books'
    aggregates, the account in its band, and the notes."""
    import pandas as pd
    if trades is None:
        from src.db import repo
        trades = repo.load_trades()
    from src.performance import books
    mk = mk or Market()
    T = live_trades(trades)
    s0 = since_ns()
    funded = [t for t in trades if books.is_live(t) and (_ns(t.get("entry_datetime")) or 0) >= s0]  # lots included
    hs = _hs_fn(mk)
    rows, errors = [], []
    q_cache: Dict[str, Optional[dict]] = {}
    for t in T:
        if arm_of(t) not in ARMS:                              # a live book without a construction here (ETF DM)
            continue
        try:
            tk, arm = str(t["ticker"]), arm_of(t)
            od = _day(t.get("dip_order_day") or t.get("cap_order_day") or t.get("entry_date")) or day
            lo = (od - timedelta(days=21)).isoformat()
            x1 = _day(t.get("exit_date")) or day
            hi = min(x1 + timedelta(days=4), day).isoformat()
            b = mk.bars(tk, lo, hi)
            if b is None:
                rows.append({"id": t.get("trade_id"), "ticker": tk, "arm": arm, "error": "no bars"})
                continue
            if arm in ("dip", "cap"):
                s = sim_order_book(t, b, mk, arm)
            elif arm == "crash":
                sym = str(t.get("crash_decision_symbol") or getattr(settings, "crash_long_symbol", "QQQ"))
                if sym not in q_cache:
                    q_cache[sym] = mk.bars(sym, (od - timedelta(days=21)).isoformat(), hi)
                s = sim_crash(t, b, q_cache[sym]) if q_cache[sym] is not None else {}
            else:
                s = sim_short(t, b)
            s["_b"] = b
            rows.append(compare(t, s, hs))
        except Exception as e:                                 # noqa: BLE001 — one trade never stops the record
            errors.append(f"{t.get('trade_id')} {t.get('ticker')}: {type(e).__name__}: {str(e)[:160]}")
            rows.append({"id": t.get("trade_id"), "ticker": t.get("ticker"), "arm": arm_of(t),
                         "error": f"{type(e).__name__}"})
    ok = [r for r in rows if "error" not in r]
    agg = {k: by_book(ok, k) for k in GAP_KEYS}
    closed = [r for r in ok if r.get("status") == "CLOSED"]
    rec: dict = {"day": day.isoformat(), "since": str(_et(since_ns())),
                 "built_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                 "trades": len(rows), "closed": len(closed),
                 "closed_own_exit": sum(1 for r in closed if not r.get("forced")),
                 "same_tick": sum(1 for r in closed if r.get("same_tick")),
                 "rows": rows, "by_book": agg, "errors": errors}
    try:
        bd = band(runs)
        path = account_path(funded, mk, day)
        for p in path:
            p["percentile_in_sample"] = percentile(bd, p["n"], p["growth_pct"])
            p["percentile_held_out"] = percentile(bd, p["n"], p["growth_pct"], held_out=True)
        last = path[-1] if path else None
        bn = (bd.get("band") or {}).get(str(last["n"])) if last else None
        rec["account"] = {"path": path, "last": last,
                          "band_at_last": ({k: bn[k] for k in ("in_sample", "held_out", "windows")} if bn else None),
                          "band_source": bd.get("source"), "held_out_growth": HELD_OUT_GROWTH}
    except Exception as e:                                     # noqa: BLE001
        rec["account"] = {"error": f"{type(e).__name__}: {str(e)[:200]}"}
    rec["notes"] = notes(rec)
    try:
        mk.save()
    except OSError:
        pass
    rec["requests"] = mk.requests
    for r in rows:
        r.pop("_b", None)
    _ = pd                                                     # (pandas imported for the helpers above)
    return rec


def notes(rec: dict) -> List[str]:
    """A note for every book whose live-minus-simulator net trade is beyond its noise (|t| >= `GAP_T` over at least
    `MIN_EPISODES` episodes) and for the account outside its band's 5-95% range."""
    out = []
    for arm, a in ((rec.get("by_book") or {}).get("net_gap_pct") or {}).items():
        if a.get("episodes", 0) >= MIN_EPISODES and a.get("t") is not None and abs(a["t"]) >= GAP_T:
            out.append(f"{arm}: live minus the simulator {a['mean']:+.2f}% a trade (t {a['t']:+.1f}, "
                       f"{a['episodes']} episodes)")
    last = (rec.get("account") or {}).get("last") or {}
    p = last.get("percentile_in_sample")
    if p is not None and (p < 5.0 or p > 95.0):
        out.append(f"the account {last['growth_pct']:+.1f}% after {last['n']} sessions, at the {p:.0f}th "
                   f"percentile of the backtest's band (held out {last.get('percentile_held_out') or 0:.0f}th)")
    return out


def write(day: date, trades: Optional[Sequence[dict]] = None) -> Path:
    from src.scheduler import db_fence
    rec = build(day, trades)
    p = out_path(day)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(f".{os.getpid()}.tmp")
    tmp.write_text(json.dumps(rec, indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o)),
                   encoding="utf-8")
    db_fence.retry_os(lambda: os.replace(tmp, p))
    return p


def latest() -> Optional[dict]:
    try:
        files = sorted(out_dir().glob("????-??-??.json"))
        return json.loads(files[-1].read_text(encoding="utf-8")) if files else None
    except (OSError, ValueError):
        return None


def health() -> dict:
    """The email's scorer banner: one line of the forward check and its notes (never a problem)."""
    rec = latest()
    if not rec:
        return {"problems": [], "notes": []}
    parts = [f"forward check as of {rec.get('day')}: {rec.get('trades', 0)} live trades from the fresh start, "
             f"{rec.get('same_tick', 0)}/{rec.get('closed', 0)} closed on the simulator's own tick"]
    for arm, a in sorted(((rec.get("by_book") or {}).get("net_gap_pct") or {}).items()):
        parts.append(f"{arm} live - sim {a['mean']:+.2f}%/trade ({a['episodes']} ep.)")
    last = (rec.get("account") or {}).get("last") or {}
    if last.get("percentile_in_sample") is not None:
        parts.append(f"account {last['growth_pct']:+.1f}% after {last['n']} sessions = band pct "
                     f"{last['percentile_in_sample']:.0f} (held out {last.get('percentile_held_out') or 0:.0f})")
    return {"problems": [], "notes": [", ".join(parts)] + [f"forward check: {n}" for n in rec.get("notes") or []]}


def main(argv: Optional[Sequence[str]] = None) -> int:
    import argparse
    from src.scheduler import db_fence
    ap = argparse.ArgumentParser(description="Every live trade beside the simulator's construction of it.")
    ap.add_argument("--day", default=None)
    a = ap.parse_args(argv)
    if a.day:
        d = date.fromisoformat(a.day)
    else:
        from zoneinfo import ZoneInfo
        d = datetime.now(ZoneInfo(ET)).date()
    db_fence.install()
    p = write(d)
    rec = json.loads(p.read_text(encoding="utf-8"))
    print(f"{p}: {rec.get('trades')} trades, {rec.get('same_tick')}/{rec.get('closed')} on the simulator's tick; "
          f"notes: {rec.get('notes')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
