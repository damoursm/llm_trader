"""THE MULTI-STRATEGY ACCOUNT ENGINE (user directive 2026-10-08: "Rework our simulator that consider a real environment
with restrictions when trading so that it can consider multiple strategies ... It should be able to calculate the real
fees and be able to know the available capital from the 10000$ starting account").

One simulated IBKR margin account, every strategy's trades in it at once. The input is a list of PIECES — one trade
each, produced by a strategy adapter (`src/backtest/strategies/`) with its entry and exit already decided by the
strategy's own rules — and a `Strategy` per strategy name (its side, slices, limits, margin rates). The engine decides
which pieces the account can actually take and at what size, and books every dollar:

* EQUITY = the start balance ($10,000, no deposits) + the realized P&L of closed positions + the open ones marked at
  their latest bar (a short's accrued borrow included). Available capital = equity minus the initial requirement of
  the open positions (IBKR's "available funds").
* SIZE of a new position = floor(min(equity / slices, side cap x equity - the side's open value, the ADV cap) / price),
  cut to what the initial-margin room allows after the order's own costs; no new position under the strategy's
  `min_equity` (FINRA's $2,000 for shorts) or past its `max_open`.
* MARGIN: a short = the larger of Reg T (maintenance 30% or $5 a share at $5 and above, 100% or $2.50 below; initial
  50%) and IBKR's house rate (a multiple of the short's value); a long = Reg T 25% maintenance / 50% initial or the
  house rate if larger. At every bar (closes, or the bar's adverse extreme — high for a short, low for a long — with
  `trigger="high"`) equity below the maintenance of the open positions is a MARGIN CALL: every position is closed at
  that price paying twice its entry half-spread (the audited engine's rule).
* FEES, all real: the half-spread of each fill (the piece's ``hs_in`` / ``hs_out`` in bp, from the live NBBO or the
  fitted spread model), IBKR's fixed commission max($1, $0.005 a share) capped at 1% of the order, the SEC fee and
  FINRA's TAF on every SALE (a short's entry, a long's exit), and a short's IBKR borrow day by day (the piece's
  schedule: shares x roundup(1.02 x prior close) x the day's rate / 360 from settlement).

For one short strategy at 24 slices with IBKR's house margin this is, operation for operation, the audited engine the
vol-arm studies used (`cap5k7.simulate`; parity checked in tests/test_backtest_account.py and on the live book).

PIECE (dict) — required: ``strategy`` (a `Strategy` name), ``tkn``, ``ens`` / ``xns`` (entry / exit instants, ns UTC),
``e`` / ``x`` (fill prices), ``hs_in`` / ``hs_out`` (half-spreads, bp), ``dv20`` (20-session dollar volume), ``pt`` /
``pc`` (the 30-minute bars it is held through: bar times and closes). Optional: ``ph`` / ``pl`` (bar highs / lows,
default the closes), ``rank`` (order among entries at the same instant within a strategy, lower first), ``hm`` / ``hi``
(the name's own house rates), ``avail`` (shares lendable at entry), ``net`` / ``days`` (the strategy's own per-trade
net return and hold, for reference). A SHORT also carries its borrow schedule: ``d0`` (first charged day number),
``cum`` (cumulative collateral x rate per share), ``bfac`` (1 / 100 / 360), ``d_out`` (exit day number).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

SEC_FEE = 27.80e-6                     # SEC Section 31 fee per dollar sold
TAF_PER_SHARE, TAF_MAX = 0.000166, 8.30  # FINRA trading activity fee on shares sold, capped per trade
ET = "America/New_York"


def comm(n: float, p: float) -> float:
    """IBKR's fixed commission: max($1, $0.005 a share), at most 1% of the order's value."""
    return min(max(1.0, 0.005 * n), 0.01 * n * p)


def sale_fees(n: float, p: float) -> float:
    """The SEC fee and FINRA's TAF on a sale of ``n`` shares at ``p``."""
    return SEC_FEE * n * p + min(TAF_PER_SHARE * n, TAF_MAX)


def short_maint(n: float, p: float) -> float:
    """Reg T maintenance of ``n`` shares short at ``p``."""
    return max(5.0 * n, 0.30 * n * p) if p >= 5.0 else max(2.5 * n, n * p)


def short_init_rate(p: float) -> float:
    """Reg T initial requirement per share of a new short at ``p``."""
    return max(0.5 * p, max(5.0, 0.30 * p) if p >= 5.0 else max(2.5, p))


@dataclass
class Strategy:
    """One strategy in the account. ``slices``: each new position = equity / slices. ``priority``: entries at the same
    instant go in priority order (lower first), then by the piece's ``rank``. ``house_maint`` / ``house_init``: IBKR's
    house requirement as a multiple of the position's value (0 = Reg T only; a piece's own ``hm`` / ``hi`` wins)."""
    name: str
    side: int                                   # +1 long, -1 short
    slices: float = 24.0
    priority: int = 0
    max_open: Optional[int] = None
    adv_cap: Optional[float] = 0.01             # at most this share of the 20-session dollar volume
    min_equity: float = 0.0                     # no new position while equity is below it (shorts: FINRA's $2,000)
    house_maint: float = 0.0
    house_init: float = 0.0
    shrink_budget: Optional[float] = None       # a position whose house initial rate is above it is sized down
    avail_mult: Optional[float] = None          # skip unless the piece's ``avail`` >= this x the shares sized

    def __post_init__(self):
        if self.side not in (1, -1):
            raise ValueError(f"{self.name}: side must be +1 (long) or -1 (short)")


def vol_short(name: str = "vol", slices: float = 24.0, priority: int = 0) -> Strategy:
    """The live short arms' account rules: 24 slices, IBKR's house margin 2.00 / 2.86, FINRA's $2,000, 1% of ADV."""
    return Strategy(name, -1, slices=slices, priority=priority, min_equity=2000.0, house_maint=2.0, house_init=2.86)


def dip_long(name: str = "dip", slices: float = 10.0, priority: int = 1, max_open: Optional[int] = 10) -> Strategy:
    """The dip long book's account rules: 10 slices, at most 10 open, Reg T margin, 1% of ADV."""
    return Strategy(name, 1, slices=slices, priority=priority, max_open=max_open)


@dataclass
class Rules:
    """Account-wide rules. ``gross_cap``: each side's open value is at most this many times equity (1.0: the long
    book is never levered, the shorts never exceed the account — the live account's rule)."""
    start: float = 10_000.0
    trigger: str = "high"                       # "close": margin judged on bar closes; "high": intrabar extremes
    gross_cap: Dict[int, float] = field(default_factory=lambda: {1: 1.0, -1: 1.0})
    long_maint: float = 0.25
    long_init: float = 0.50


# ── calendar: settlement days for the borrow schedule ───────────────────────

_CAL: Dict[str, np.ndarray] = {}
T1_FROM = (pd.Timestamp("2024-05-28") - pd.Timestamp("1970-01-01")).days   # T+1 settlement from 2024-05-28


def day_numbers(ns) -> np.ndarray:
    """ET calendar day numbers (days since 1970-01-01) of UTC nanosecond instants."""
    ns = np.asarray(ns, np.int64)
    if not len(ns):
        return np.zeros(0, np.int64)
    u, inv = np.unique(ns, return_inverse=True)
    d = pd.to_datetime(u, unit="ns", utc=True).tz_convert(ET).tz_localize(None).normalize()
    return ((d - pd.Timestamp("1970-01-01")).days).to_numpy().astype(np.int64)[inv]


def default_settle() -> Callable[[int], int]:
    """Settlement session of a trade made on day number ``dn`` (T+2, T+1 from 2024-05-28) on the sessions AAPL traded
    in the deep 30-minute store, weekdays after its data — the research engine's own calendar."""
    if "s" not in _CAL:
        from src.data.intraday_store import load_deep_30m
        df = load_deep_30m("AAPL")
        days = pd.DatetimeIndex(df.index).tz_localize("UTC").tz_convert(ET).normalize().tz_localize(None)
        s = np.unique(((days - pd.Timestamp("1970-01-01")).days).to_numpy().astype(np.int64))
        extra = pd.bdate_range(pd.Timestamp("1970-01-01") + pd.Timedelta(days=int(s[-1]) + 1), "2027-12-31")
        _CAL["s"] = np.r_[s, ((extra - pd.Timestamp("1970-01-01")).days).to_numpy().astype(np.int64)]
    cal = _CAL["s"]
    memo: Dict[int, int] = {}

    def settle(dn: int) -> int:
        dn = int(dn)
        v = memo.get(dn)
        if v is None:
            i = int(np.searchsorted(cal, dn))
            v = int(cal[i + (1 if dn >= T1_FROM else 2)])
            memo[dn] = v
        return v
    return settle


# ── the book: pieces as arrays, in entry order ───────────────────────────────

class Book:
    """The pieces as arrays, ordered by entry instant, then strategy priority, then rank (a single strategy without
    ranks keeps its list order — the audited engine's stable sort)."""

    def __init__(self, pieces: Sequence[dict], strategies: Dict[str, Strategy], settle: Callable[[int], int]):
        n = len(pieces)
        missing = sorted({str(p.get("strategy")) for p in pieces} - set(strategies))
        if missing:
            raise KeyError(f"pieces of unknown strategies: {missing}")
        ens = np.array([int(p["ens"]) for p in pieces], np.int64)
        pri = np.array([strategies[p["strategy"]].priority for p in pieces], np.int64)
        rank = np.array([float(p.get("rank", k)) for k, p in enumerate(pieces)], float)
        o = np.lexsort((rank, pri, ens)) if n else np.zeros(0, np.int64)
        P = [pieces[i] for i in o]
        self.n = n
        self.pieces = P
        self.strat = [str(p["strategy"]) for p in P]
        self.side = np.array([strategies[s].side for s in self.strat], np.int64)
        self.tkn = np.array([p["tkn"] for p in P], dtype=object)
        self.ens = np.array([int(p["ens"]) for p in P], np.int64)
        self.xns = np.array([int(p["xns"]) for p in P], np.int64)
        for a in ("e", "x", "hs_in", "hs_out", "dv20"):
            setattr(self, a, np.array([float(p[a]) for p in P], float))
        self.pt = [np.asarray(p["pt"], np.int64) for p in P]
        self.pc = [np.asarray(p["pc"], float) for p in P]
        self.ph = [np.asarray(p.get("ph", p["pc"]), float) for p in P]
        self.pl = [np.asarray(p.get("pl", p["pc"]), float) for p in P]
        self.hm = np.array([float(p["hm"]) if p.get("hm") is not None else strategies[s].house_maint
                            for p, s in zip(P, self.strat)], float)
        self.hi = np.array([float(p["hi"]) if p.get("hi") is not None else strategies[s].house_init
                            for p, s in zip(P, self.strat)], float)
        self.avail = np.array([float(p["avail"]) if p.get("avail") is not None else np.inf for p in P], float)
        # a short's borrow schedule (a long carries none)
        big = np.iinfo(np.int64).max // 4
        self.d0 = np.array([int(p["d0"]) if self.side[k] < 0 and p.get("d0") is not None else big
                            for k, p in enumerate(P)], np.int64)
        self.cum = [np.asarray(p["cum"], float) if self.side[k] < 0 and p.get("cum") is not None else np.zeros(1)
                    for k, p in enumerate(P)]
        self.bfac = np.array([float(p.get("bfac", 0.0)) if self.side[k] < 0 else 0.0 for k, p in enumerate(P)], float)
        d_out = np.array([int(p["d_out"]) if p.get("d_out") is not None else -1 for p in P], np.int64)
        need = d_out < 0
        if need.any():
            d_out[need] = day_numbers(self.xns[need])
        self.d_out = d_out
        self.last_out = np.array([settle(d) - 1 for d in d_out], np.int64)
        self.full_ps = np.array([float(P[k]["full_ps"]) if self.side[k] < 0 and P[k].get("full_ps") is not None
                                 else self._borrow(k, self.last_out[k]) for k in range(n)], float)

    def _borrow(self, k: int, last: int) -> float:
        if last < self.d0[k]:
            return 0.0
        c = self.cum[k]
        return float(c[min(last - self.d0[k], len(c) - 1)] * self.bfac[k])

    def events(self):
        """(time, kind, row, bar, day): a piece's bar marks (kind 1), its exit (0) and its entry (2), sorted by time,
        then kind — at one instant exits first, then marks, then entries — then row and bar."""
        t, kind, ii, jj = [], [], [], []
        for i in range(self.n):
            m = len(self.pt[i])
            t.append(self.pt[i])
            kind.append(np.ones(m, np.int8))
            ii.append(np.full(m, i, np.int64))
            jj.append(np.arange(m, dtype=np.int64))
        t.append(self.xns)
        kind.append(np.zeros(self.n, np.int8))
        ii.append(np.arange(self.n))
        jj.append(np.full(self.n, -1))
        t.append(self.ens)
        kind.append(np.full(self.n, 2, np.int8))
        ii.append(np.arange(self.n))
        jj.append(np.full(self.n, -1))
        t, kind, ii, jj = (np.concatenate(z) for z in (t, kind, ii, jj))
        o = np.lexsort((jj, ii, kind, t))
        return t[o], kind[o], ii[o], jj[o], day_numbers(t[o])


# ── the simulation ───────────────────────────────────────────────────────────

def simulate(pieces: Sequence[dict], strategies: Sequence[Strategy] | Dict[str, Strategy], rules: Optional[Rules] = None,
             settle: Optional[Callable[[int], int]] = None, keep_curve: bool = False,
             keep_exposure: bool = False, keep_skips: bool = False) -> dict:
    """Run the account over the pieces. Returns ``final`` equity, ``calls`` (margin calls: instant, equity before,
    cash after), ``ruined`` (the instant equity hit zero, or None), ``maxdd`` / ``maxdd_nav`` (worst drawdown of
    equity), ``trades`` (positions opened), ``positions`` (every position: strategy, ticker, side, shares, entry / exit
    instant and price, P&L net of every cost, its costs, the equity at entry, how it ended), and with ``keep_curve`` the
    equity after every instant (``curve_t`` / ``curve_v``). ``keep_exposure`` (implies the curve) adds, at the same
    instants, the open longs' and shorts' value at their marks (``curve_long`` / ``curve_short``) and their initial
    margin requirement (``curve_req``: equity minus it is the capital still available). ``keep_skips`` adds ``skips``:
    every entry the account could not take in full — the shares its slice wanted, the shares it got (0 = skipped) and
    the binding limit (``margin``: the initial-margin room; ``side_cap``: the side's open value at 100% of equity;
    ``volume_cap``: 1% of the dollar volume; ``house_shrink``; ``max_open``; ``min_equity``: under FINRA's $2,000;
    ``below_one_share``; ``lendable``) — see `capital_summary`."""
    rules = rules or Rules()
    strat = {s.name: s for s in strategies} if not isinstance(strategies, dict) else dict(strategies)
    settle = settle or default_settle()
    start = float(rules.start)
    keep_curve = keep_curve or keep_exposure
    if not pieces:
        out = {"final": start, "calls": [], "ruined": None, "maxdd": 0.0, "maxdd_nav": 0.0, "trades": 0,
               "positions": [], "curve_t": np.zeros(0, np.int64), "curve_v": np.zeros(0)}
        if keep_exposure:
            out.update(curve_long=np.zeros(0), curve_short=np.zeros(0), curve_req=np.zeros(0))
        return out
    bk = Book(pieces, strat, settle)
    T, KIND, II, JJ, DN = bk.events()
    side, hm, hi, d0, cum, bfac, last_out = bk.side, bk.hm, bk.hi, bk.d0, bk.cum, bk.bfac, bk.last_out
    S = [strat[s] for s in bk.strat]
    lm, li = float(rules.long_maint), float(rules.long_init)
    gcap = {1: float(rules.gross_cap.get(1, 1.0)), -1: float(rules.gross_cap.get(-1, 1.0))}
    hi_trig = rules.trigger == "high"

    def acc(i, today):
        """A short's borrow accrued through ``today`` (at most its full hold's)."""
        last = min(last_out[i], today)
        if last < d0[i]:
            return 0.0
        c = cum[i]
        return float(c[min(last - d0[i], len(c) - 1)] * bfac[i])

    def acc_liq(i, today):
        """A short's borrow when it is bought back on ``today`` (through the day before its settlement)."""
        last = settle(today) - 1
        if last < d0[i]:
            return 0.0
        c = cum[i]
        return float(c[min(last - d0[i], len(c) - 1)] * bfac[i])

    def unreal(i, n, px, today):
        if side[i] < 0:
            return n * (o_e[i] - px) - o_cin[i] - (n * acc(i, today) if today >= d0[i] else 0.0)
        return n * (px - o_e[i]) - o_cin[i]

    def mreq(i, n, p):
        if side[i] < 0:
            r = short_maint(n, p)
            return r if hm[i] <= 0 else max(r, hm[i] * n * p)
        return max(lm * n * p, hm[i] * n * p)

    def ireq_open(i, n, p):
        if side[i] < 0:
            r = max(0.5 * n * p, short_maint(n, p))
            return r if hi[i] <= 0 else max(r, hi[i] * n * p)
        return max(li * n * p, hi[i] * n * p)

    def irate(i, e):
        if side[i] < 0:
            r = short_init_rate(e)
            return r if hi[i] <= 0 else max(r, hi[i] * e)
        return max(li * e, hi[i] * e)

    def entry_cost(i, n, e):
        """The order's half-spread + commission, + the SEC fee and TAF when it is a sale (a short) — summed in the
        audited engine's order (bit-identical results)."""
        if side[i] < 0:
            return n * e * bk.hs_in[i] / 1e4 + comm(n, e) + SEC_FEE * n * e + min(TAF_PER_SHARE * n, TAF_MAX)
        return n * e * bk.hs_in[i] / 1e4 + comm(n, e)

    cash = start
    o_n, o_e, o_cin, o_px, o_hi, o_eq = {}, {}, {}, {}, {}, {}
    open_by = {}                                       # strategy -> open positions
    calls, peak, maxdd = [], start, 0.0
    units, nav_peak, maxdd_nav = start, 1.0, 0.0      # the drawdown on NAV = equity / start (the audited engine's)
    curve_t, curve_v = [], []
    curve_l, curve_s, curve_r = [], [], []             # keep_exposure: longs' / shorts' value, initial requirement
    skips: List[dict] = []                             # keep_skips: entries not taken in full

    def skip_rec(i, t, why, want, got, eq, room):
        return {"strategy": bk.strat[i], "tkn": bk.tkn[i], "t": int(t), "reason": why, "want": int(want),
                "got": int(got), "price": float(bk.e[i]), "equity": eq, "room": room}
    positions: List[dict] = []
    pos_of: Dict[int, dict] = {}

    def close_position(i, xp, xt, pnl, cost, how):
        rec = pos_of.pop(i)
        rec.update(exit_ns=int(xt), exit_px=float(xp), pnl=float(pnl), cost_out=float(cost), how=how)
        positions.append(rec)
        open_by[bk.strat[i]] -= 1

    N = len(T)
    p = 0
    while p < N:
        t = T[p]
        q = p
        while q < N and T[q] == t:
            q += 1
        today = int(DN[p])
        bar_now = set()
        for z in range(p, q):
            i = II[z]
            if KIND[z] == 0:
                if i in o_n:
                    n = o_n.pop(i)
                    e = o_e.pop(i)
                    cin = o_cin.pop(i)
                    o_px.pop(i)
                    o_hi.pop(i)
                    o_eq.pop(i)
                    xp = bk.x[i]
                    if side[i] < 0:
                        cout = n * xp * bk.hs_out[i] / 1e4 + comm(n, xp)
                        pnl = n * (e - xp) - cin - cout - n * bk.full_ps[i]
                    else:
                        cout = n * xp * bk.hs_out[i] / 1e4 + comm(n, xp) + sale_fees(n, xp)
                        pnl = n * (xp - e) - cin - cout
                    cash += pnl
                    close_position(i, xp, t, pnl, cout, "exit")
            elif KIND[z] == 1:
                if i in o_n:
                    j = JJ[z]
                    o_px[i] = bk.pc[i][j]
                    o_hi[i] = bk.ph[i][j] if side[i] < 0 else bk.pl[i][j]
                    bar_now.add(i)
            else:
                break
        if o_n:
            px = {}
            for i in o_n:
                if hi_trig and i in bar_now:
                    px[i] = max(o_hi[i], o_px[i]) if side[i] < 0 else min(o_hi[i], o_px[i])
                else:
                    px[i] = o_px[i]
            need, eq = 0.0, cash
            for i, n in o_n.items():
                need += mreq(i, n, px[i])
                eq += unreal(i, n, px[i], today)
            if eq < need:
                eq0 = eq
                for i in sorted(o_n, key=lambda w: -mreq(w, o_n[w], px[w])):
                    n, e, cin, xp = o_n[i], o_e[i], o_cin[i], px[i]
                    if side[i] < 0:
                        cout = n * xp * 2 * bk.hs_in[i] / 1e4 + comm(n, xp)
                        pnl = n * (e - xp) - cin - cout - n * acc_liq(i, today)
                    else:
                        cout = n * xp * 2 * bk.hs_in[i] / 1e4 + comm(n, xp) + sale_fees(n, xp)
                        pnl = n * (xp - e) - cin - cout
                    cash += pnl
                    close_position(i, xp, t, pnl, cout, "margin_call")
                o_n.clear(), o_e.clear(), o_cin.clear(), o_px.clear(), o_hi.clear(), o_eq.clear()
                calls.append((int(t), eq0, cash))
                if cash <= 0:
                    curve_t.append(int(t))
                    curve_v.append(cash)
                    curve_l.append(0.0), curve_s.append(0.0), curve_r.append(0.0)
                    dd = (peak - cash) / peak if peak > 0 else 1.0
                    nav = cash / units
                    maxdd_nav = max(maxdd_nav, (nav_peak - nav) / nav_peak if nav_peak > 0 else 1.0)
                    out = {"final": cash, "calls": calls, "maxdd": max(maxdd, dd), "maxdd_nav": maxdd_nav,
                           "ruined": int(t), "trades": len(positions), "positions": positions,
                           "curve_t": np.array(curve_t), "curve_v": np.array(curve_v)}
                    if keep_exposure:
                        out.update(curve_long=np.array(curve_l), curve_short=np.array(curve_s),
                                   curve_req=np.array(curve_r))
                    if keep_skips:
                        out["skips"] = skips
                    return out
        for z in range(p, q):
            if KIND[z] != 2:
                continue
            i = II[z]
            st = S[i]
            if st.max_open is not None and open_by.get(st.name, 0) >= st.max_open:
                if keep_skips:
                    skips.append(skip_rec(i, t, "max_open", 0, 0, None, None))
                continue
            e = bk.e[i]
            eq = cash
            g = 0.0
            room_req = 0.0
            for w, n in o_n.items():
                pw = o_px[w]
                eq += unreal(w, n, pw, today)
                if side[w] == side[i]:
                    g += n * pw
                room_req += ireq_open(w, n, pw)
            if eq < st.min_equity:
                if keep_skips:
                    skips.append(skip_rec(i, t, "min_equity", 0, 0, eq, None))
                continue
            room = eq - room_req
            r_i = irate(i, e)
            nm = int(math.floor(room / r_i)) if room > 0 else 0
            if nm > 0:
                nm = int(math.floor(max(room - entry_cost(i, nm, e), 0.0) / r_i))
            alloc = min(eq / st.slices, gcap[int(side[i])] * eq - g)
            why_alloc = "side_cap" if gcap[int(side[i])] * eq - g < eq / st.slices else None
            if st.shrink_budget is not None and hi[i] > st.shrink_budget:
                alloc = alloc * st.shrink_budget / hi[i]
                why_alloc = "house_shrink"
            if st.adv_cap is not None and alloc > st.adv_cap * bk.dv20[i]:
                alloc = st.adv_cap * bk.dv20[i]
                why_alloc = "volume_cap"
            n_alloc = int(math.floor(alloc / e)) if e > 0 and alloc > 0 else 0
            n = min(n_alloc, nm)
            if keep_skips:
                want = int(math.floor(eq / st.slices / e)) if e > 0 and eq > 0 else 0
                if n < want or n <= 0:
                    why = "margin" if nm < n_alloc else (why_alloc or ("below_one_share" if want <= 0 else "rounding"))
                    skips.append(skip_rec(i, t, why, want, max(n, 0), eq, room))
            if n <= 0:
                continue
            if st.avail_mult is not None and n * st.avail_mult > bk.avail[i]:
                if keep_skips:
                    skips.append(skip_rec(i, t, "lendable", n, 0, eq, room))
                continue
            o_n[i], o_e[i] = n, e
            o_cin[i] = entry_cost(i, n, e)
            o_px[i], o_hi[i], o_eq[i] = e, e, eq
            open_by[st.name] = open_by.get(st.name, 0) + 1
            pos_of[i] = {"strategy": st.name, "tkn": bk.tkn[i], "side": int(side[i]), "shares": int(n),
                         "entry_ns": int(t), "entry_px": float(e), "cost_in": float(o_cin[i]), "equity_in": float(eq),
                         "net": bk.pieces[i].get("net"), "days": bk.pieces[i].get("days")}
        eq = cash
        for i, n in o_n.items():
            eq += unreal(i, n, o_px[i], today)
        if eq > peak:
            peak = eq
        dd = (peak - eq) / peak if peak > 0 else 1.0
        if dd > maxdd:
            maxdd = dd
        nav = eq / units
        nav_peak = max(nav_peak, nav)
        maxdd_nav = max(maxdd_nav, (nav_peak - nav) / nav_peak if nav_peak > 0 else 1.0)
        if keep_curve:
            curve_t.append(int(t))
            curve_v.append(eq)
            if keep_exposure:
                lv = sv = rq = 0.0
                for i, n in o_n.items():
                    if side[i] < 0:
                        sv += n * o_px[i]
                    else:
                        lv += n * o_px[i]
                    rq += ireq_open(i, n, o_px[i])
                curve_l.append(lv), curve_s.append(sv), curve_r.append(rq)
        p = q
    # positions still open at the end: marked at their last bar, no exit costs or further borrow (the audited
    # engine's final value)
    us = {i: (n * (o_e[i] - o_px[i]) - o_cin[i] if side[i] < 0 else n * (o_px[i] - o_e[i]) - o_cin[i])
          for i, n in o_n.items()}
    final = cash + sum(us[i] for i in o_n)
    for i in list(o_n):
        rec = pos_of.pop(i)
        rec.update(exit_ns=None, exit_px=float(o_px[i]), pnl=float(us[i]), cost_out=0.0, how="open_at_end")
        positions.append(rec)
    out = {"final": final, "calls": calls, "ruined": None, "maxdd": maxdd, "maxdd_nav": maxdd_nav,
           "trades": len(positions), "positions": positions,
           "curve_t": np.array(curve_t), "curve_v": np.array(curve_v)}
    if keep_exposure:
        out.update(curve_long=np.array(curve_l), curve_short=np.array(curve_s), curve_req=np.array(curve_r))
    if keep_skips:
        out["skips"] = skips
    return out


def capital_summary(res: dict, pieces: Sequence[dict]) -> Dict[str, dict]:
    """Per strategy, from ``simulate(..., keep_skips=True)``: the entries its rule produced, those taken in full, cut
    (fewer shares than the slice wanted) and skipped, each by binding limit, and the dollars its slices wanted but did
    not get."""
    rows = res.get("skips") or []
    out: Dict[str, dict] = {}
    names = sorted({str(p["strategy"]) for p in pieces})
    for n in names:
        R = [r for r in rows if r["strategy"] == n]
        cut = [r for r in R if r["got"] > 0]
        skip = [r for r in R if r["got"] <= 0]
        opened = sum(1 for p in res.get("positions") or [] if p["strategy"] == n)
        out[n] = {"entries": sum(1 for p in pieces if str(p["strategy"]) == n), "opened": opened,
                  "full": opened - len(cut), "cut": len(cut), "skipped": len(skip),
                  "cut_by": pd.Series([r["reason"] for r in cut], dtype=object).value_counts().to_dict(),
                  "skipped_by": pd.Series([r["reason"] for r in skip], dtype=object).value_counts().to_dict(),
                  "usd_not_deployed": float(sum((r["want"] - r["got"]) * r["price"] for r in R))}
    return out


# ── activity: trades, days in the market, capital in use ─────────────────────

def _et(ns) -> pd.Timestamp:
    return pd.Timestamp(int(ns), unit="ns", tz="UTC").tz_convert(ET).tz_localize(None)


def holding_days(positions: Sequence[dict], sessions, end=None) -> Dict[str, np.ndarray]:
    """Which ``sessions`` (session dates) each strategy held a position in during regular hours (09:30-16:00 ET):
    a position counts on a session it was open at some moment of — entered before that session's close and exited
    after its open (a long sold AT the open did not hold that day). Open at the end: held to ``end``. Returns
    {strategy: bool array, "any": bool array}."""
    days = pd.DatetimeIndex(pd.to_datetime(sessions)).normalize()
    end_ts = pd.Timestamp(end) if end is not None else days[-1] + pd.Timedelta(hours=16)
    out: Dict[str, np.ndarray] = {}
    anyh = np.zeros(len(days), bool)
    for p in positions:
        a = _et(p["entry_ns"])
        b = _et(p["exit_ns"]) if p.get("exit_ns") is not None else end_ts
        opn = days + pd.Timedelta(hours=9, minutes=30)
        cls = days + pd.Timedelta(hours=16)
        m = (a < cls) & (b > opn)
        h = out.setdefault(p["strategy"], np.zeros(len(days), bool))
        h |= np.asarray(m)
        anyh |= np.asarray(m)
    out["any"] = anyh
    return out


def activity(res: dict, sessions, lo, end) -> dict:
    """Per calendar year and over [lo, end]: positions opened per strategy, the sessions with a position open (any
    strategy, and each) and the sessions in the span."""
    days = pd.DatetimeIndex(pd.to_datetime(sessions)).normalize()
    days = days[(days >= pd.Timestamp(lo).normalize()) & (days <= pd.Timestamp(end).normalize())]
    rows = res.get("positions") or []
    held = holding_days(rows, days, end=pd.Timestamp(end) + pd.Timedelta(hours=16))
    years = sorted(set(days.year))
    out = {"years": {}, "span": {}}
    names = sorted({r["strategy"] for r in rows})
    for y in years + [None]:
        k = days.year == y if y is not None else np.ones(len(days), bool)
        ent = [r for r in rows if y is None or _et(r["entry_ns"]).year == y]
        rec = {"sessions": int(k.sum()), "days_held": int(held["any"][k].sum()),
               "trades": {n: sum(1 for r in ent if r["strategy"] == n) for n in names},
               "days_held_by": {n: int(held[n][k].sum()) for n in names if n in held}}
        rec["held_share"] = rec["days_held"] / rec["sessions"] if rec["sessions"] else float("nan")
        if y is None:
            out["span"] = rec
        else:
            out["years"][str(y)] = rec
    return out


def exposure_daily(res: dict, sessions, start: float = 10_000.0) -> pd.DataFrame:
    """End-of-session equity, open longs' and shorts' value at their marks and their initial margin requirement, per
    session (needs ``simulate(..., keep_exposure=True)``). A session without an event held nothing: its exposure is 0
    and its equity the last one (``start`` before the first event)."""
    days = pd.DatetimeIndex(pd.to_datetime(sessions)).normalize()
    t = np.asarray(res.get("curve_t", []), np.int64)
    df = pd.DataFrame(index=days, columns=["equity", "long", "short", "req"], dtype=float)
    if not len(t):
        df["equity"] = float(res.get("final", start))
        df[["long", "short", "req"]] = 0.0
        return df
    d = pd.to_datetime(t, unit="ns", utc=True).tz_convert(ET).tz_localize(None).normalize()
    cur = pd.DataFrame({"equity": res["curve_v"], "long": res["curve_long"], "short": res["curve_short"],
                        "req": res["curve_req"]}, index=d)
    last = cur.groupby(level=0).last()
    df.update(last)
    df["equity"] = df["equity"].ffill().fillna(start)
    df[["long", "short", "req"]] = df[["long", "short", "req"]].fillna(0.0)
    return df


# ── metrics ──────────────────────────────────────────────────────────────────

def per_day(ret_pct: float, days: float) -> float:
    """Return per day (user's definition): (1 + average trade)^(1 / average 24-hour hold) - 1, in %."""
    b = 1 + ret_pct / 100.0
    if days <= 0:
        return float("nan")
    return (b ** (1 / days) - 1) * 100.0 if b > 0 else -100.0


def metrics(res: dict, lo, end, start: float = 10_000.0) -> dict:
    """Both metrics of an account run over [lo, end]: GROWTH per year (and the log growth per year — the study
    objective), worst drawdown, margin calls; per strategy the positions taken, their P&L and costs, the average
    realized trade (net of every cost, on its notional), the average hold, RETURN PER DAY, and each position's log
    contribution log(1 + P&L / equity at entry) — whose sum is the account's log growth when positions do not
    overlap (the per-trade objective's account-level form)."""
    years = max((pd.Timestamp(end) - pd.Timestamp(lo)).days / 365.25, 1e-9)
    f = float(res["final"])
    out = {"final": f, "growth": (f / start) ** (1 / years) - 1 if f > 0 else -1.0,
           "log_growth": math.log(f / start) / years if f > 0 else float("-inf"),
           "maxdd": float(res.get("maxdd_nav", res.get("maxdd", 0.0))) * 100.0, "calls": len(res.get("calls") or []),
           "ruined": res.get("ruined"), "trades": int(res.get("trades", 0)), "years": years, "by_strategy": {}}
    rows = res.get("positions") or []
    total_lc = 0.0
    for name in sorted({r["strategy"] for r in rows}):
        R = [r for r in rows if r["strategy"] == name]
        ret = np.array([100.0 * r["pnl"] / (r["shares"] * r["entry_px"]) for r in R])
        hold = np.array([((r["exit_ns"] if r["exit_ns"] is not None else r["entry_ns"]) - r["entry_ns"]) / 86400e9
                         for r in R])
        lc = np.array([math.log(1 + r["pnl"] / r["equity_in"]) if r["equity_in"] > 0 and r["pnl"] > -r["equity_in"]
                       else float("-inf") for r in R])
        total_lc += float(lc.sum())
        out["by_strategy"][name] = {
            "positions": len(R), "pnl": float(sum(r["pnl"] for r in R)),
            "costs": float(sum(r["cost_in"] + r["cost_out"] for r in R)),
            "avg_ret": float(ret.mean()) if len(ret) else None, "median_ret": float(np.median(ret)) if len(ret) else None,
            "avg_days": float(hold.mean()) if len(hold) else None,
            "return_per_day": per_day(float(ret.mean()), float(hold.mean())) if len(ret) else None,
            "win_rate": float((ret > 0).mean()) if len(ret) else None,
            "margin_called": int(sum(1 for r in R if r["how"] == "margin_call")),
            "log_contrib_per_year": float(lc.sum()) / years}
    out["log_contrib_per_year"] = total_lc / years
    return out
