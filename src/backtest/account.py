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
* SIZE of a new position = floor(min(equity / slices, side cap x equity - the side's open value, the ADV cap) / price)
  (``slices``: the strategy's, or the piece's own ``slices`` when it carries one — a sizing re-tuned week by week),
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

LIMIT ORDERS (a strategy with ``orders``; PREREG61): one order = the pieces sharing an ``oid``, one per session it could
fill in (``ord_k``: the session's index in the order's window; ``ens`` / ``e``: the first touch in that session and its
fill price; the exit and marks of a fill there), each carrying the order's ``ord_place`` (the placement instants, one
per window session), ``ord_lim`` (the limit price) and ``ord_end`` (the instant it lapses). At each placement instant
an order that is neither working nor filled is sized at its limit like an entry (the strategy's slice, the room, its
side cap, the ADV cap; its working orders count toward ``max_open``) — orders placed at one instant go deepest
``rank`` first; it fills at the first of its pieces whose session is at/after its placement, with the quantity fixed at
placement CUT to what the account can fund at that instant (equity minus the open positions' initial requirement and
the other working orders' reservations, within its side cap): the rest of the order is not bought, and a touch that
finds no room buys nothing (the order keeps working). An order cancelled to make room may be placed again at a later
placement instant.

GROUP CAP (a strategy with ``group_cap``; PREREG100, off by default): each piece may carry a ``group`` (its sector,
say). A strategy takes no new position and places no order in a group that already holds ``group_cap`` open positions
plus working orders of the strategies in its ``group_scope`` (itself by default): a market entry is skipped
(``group_cap``), an order waits for its next placement instant (the path of an order without a slot). A piece without
a group is never capped and never counted; without any ``group_cap`` nothing changes.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence, Tuple

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
    # CAPITAL PRIORITY (all off by default): an entry of a strategy with a ``quality`` that finds too little room (its
    # side's cap or the margin room) sells just enough of the open positions of PREEMPTIBLE strategies of lower
    # quality (lowest first, the latest opened first) at their mark; a preemptible strategy with ``refill_min`` buys
    # back toward its slice at its own marks once the room it can fill is at least that share of equity
    quality: Optional[float] = None
    preemptible: bool = False
    refill_min: Optional[float] = None
    yield_whole: bool = False                   # preempted: the WHOLE position exits (not just the shares needed)
    reenter: bool = False                       # closed early (preempted, margin call): re-bought at the piece's
                                                # next 09:30 ET mark with room, never by preempting (PREREG53)
    room_frac: Optional[float] = None           # MARGIN BUFFER: an entry opens only while the account's initial
                                                # requirement after it is at most this share of equity (PREREG57)
    delever_order: Optional[int] = None         # DE-LEVERAGING (``Rules.delever``): its positions may be sold, the
                                                # lowest order first, the latest opened first (PREREG57)
    yield_first: str = "latest"                 # which of its positions give way first: "latest" / "earliest" opened
    delever_first: str = "latest"               # which of its positions de-leveraging sells first (PREREG58)
    # WORKING ORDERS (off: None; PREREG61): its pieces are LIMIT ORDERS (see `simulate`) placed at their placement
    # instants with the quantity sized then, filled at the first touch after placement. "reserve": a working order holds
    # its initial requirement in the account's room (and its value in its side); a higher-quality entry short of room
    # cancels the working orders of preemptible strategies (latest placed first) before selling their positions,
    # de-leveraging cancels them first, a margin call cancels them. "fill": it reserves nothing; only a margin call
    # cancels it. Either way a touch buys AT MOST what the room allows then (none: the order keeps working)
    orders: Optional[str] = None
    book: Optional[str] = None                  # its BOOK for `Rules.one_book_per_name` (None: its own name)
    fees: Optional[str] = None                  # its legs' cost model, "engine" / "ledger" (None: `Rules.fees`)
    # GROUP CAP (off: None; PREREG100): at most ``group_cap`` open positions PLUS working orders in one GROUP — a
    # piece's ``group`` (its sector, say; a piece without one is never capped and never counted) — among the
    # strategies named in ``group_scope`` (None: the strategy itself). A market entry whose group is at the cap is
    # skipped ("group_cap"); a limit order whose group is at the cap is not placed at that instant (logged
    # "group_cap") and is tried again at its next placement instant — the path of an order that finds no slot; a
    # re-entry waits the same way. A repeat of a held name (a stack) counts as one more position.
    group_cap: Optional[int] = None
    group_scope: Optional[Tuple[str, ...]] = None

    def __post_init__(self):
        if self.side not in (1, -1):
            raise ValueError(f"{self.name}: side must be +1 (long) or -1 (short)")
        for f in ("yield_first", "delever_first"):
            if getattr(self, f) not in ("latest", "earliest"):
                raise ValueError(f"{self.name}: {f} must be 'latest' or 'earliest'")
        if self.orders not in (None, "reserve", "fill"):
            raise ValueError(f"{self.name}: orders must be None, 'reserve' or 'fill'")
        if self.fees not in (None, "engine", "ledger"):
            raise ValueError(f"{self.name}: fees must be None, 'engine' or 'ledger'")
        if self.group_cap is not None and (isinstance(self.group_cap, bool) or int(self.group_cap) != self.group_cap
                                           or int(self.group_cap) < 0):
            raise ValueError(f"{self.name}: group_cap must be None or a whole number >= 0, not {self.group_cap!r}")
        if self.group_scope is not None:
            if isinstance(self.group_scope, str):
                raise ValueError(f"{self.name}: group_scope names strategies (a tuple), not {self.group_scope!r}")
            self.group_scope = tuple(str(s) for s in self.group_scope)
        if self.side < 0 and (self.reenter or self.refill_min is not None):
            # a re-entered or refilled lot would pay its borrow from the ORIGINAL piece's settlement
            raise ValueError(f"{self.name}: re-entry / refill are for long books (a short's borrow is per piece)")


def vol_short(name: str = "vol", slices: float = 24.0, priority: int = 0) -> Strategy:
    """The live short arms' account rules: 24 slices, IBKR's house margin 2.00 / 2.86, FINRA's $2,000, 1% of ADV."""
    return Strategy(name, -1, slices=slices, priority=priority, min_equity=2000.0, house_maint=2.0, house_init=2.86)


def dip_long(name: str = "dip", slices: float = 10.0, priority: int = 1, max_open: Optional[int] = 10) -> Strategy:
    """The dip long book's account rules: 10 slices, at most 10 open, Reg T margin, 1% of ADV."""
    return Strategy(name, 1, slices=slices, priority=priority, max_open=max_open)


def dm_long(name: str = "dm", slices: float = 1.0, priority: int = 2, max_open: Optional[int] = 1) -> Strategy:
    """The ETF dual-momentum book's account rules: the whole account in one fund (1 slice, 1 open), Reg T margin."""
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
    # DE-LEVERAGING (off: None): (trigger, target) — at a bar close where the maintenance requirement exceeds
    # trigger x equity, the positions of strategies with a ``delever_order`` are sold WHOLE at their marks until it is
    # at most target x equity (never bought back; PREREG57)
    delever: Optional[Tuple[float, float]] = None
    # ONE BOOK PER NAME (off: False): an entry — a limit order's placement or fill too — is skipped while a strategy
    # of ANOTHER book (`Strategy.book`) holds the name; strategies of one book may share it (live: the selection
    # short's arms vs the dip long book)
    one_book_per_name: bool = False
    # COSTS AS THE LEDGER'S (off: "engine", PREREG62): "ledger" = a leg's whole cost is its piece's ``hs_in`` /
    # ``hs_out`` (production's all-in one-way cost, commission included) — no separate commission, SEC fee or TAF; a
    # margin call closes at the exit cost; a new position's room keeps only its commission (production's sizing)
    fees: str = "engine"
    # open positions marked net of their exit cost (the ledger's marks): a piece's ``hs_mark`` (bp; default its
    # ``hs_out``) — the ledger marks an open trade net of its estimated exit cost whatever the exit later costs
    mark_net: bool = False
    # ORDERS MAKE ROOM (off: False; PREREG86): a limit order of a strategy with a ``quality`` is sized at placement as
    # if the lower-quality preemptible positions were freed (nothing sold then) and its fill sells just enough of them
    # for its quantity — so a sleeve that yields to every book never starves the dip book's orders
    fills_make_room: bool = False

    def __post_init__(self):
        if self.trigger not in ("close", "high"):
            raise ValueError(f"Rules.trigger must be 'close' or 'high', not {self.trigger!r}")
        if self.fees not in ("engine", "ledger"):
            raise ValueError(f"Rules.fees must be 'engine' or 'ledger', not {self.fees!r}")
        if self.delever is not None and not (len(self.delever) == 2 and 0 < self.delever[1] <= self.delever[0]):
            raise ValueError(f"Rules.delever must be (trigger, target) with 0 < target <= trigger, not {self.delever!r}")
        if set(self.gross_cap) - {1, -1}:
            raise ValueError(f"Rules.gross_cap keys must be 1 / -1, not {sorted(self.gross_cap)}")


def _plain(tkn) -> str:
    """The traded symbol of a backtest name (``XYZ@D`` / ``@X`` / ``@E`` -> ``XYZ``)."""
    return str(tkn).split("@")[0]


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


def settlement_days(sessions) -> np.ndarray:
    """The settlement days among ``sessions`` (day numbers): every session but the Federal Reserve holidays on which
    the NYSE trades and nothing settles (Columbus Day, Veterans Day — production's `borrow_fees.is_settlement_day`)."""
    from src.performance.borrow_fees import _settlement_only_holidays
    s = np.asarray(sessions, np.int64)
    years = pd.to_datetime(s, unit="D").year
    off = set()
    for y in sorted(set(years.tolist())):
        off |= {(pd.Timestamp(d) - pd.Timestamp("1970-01-01")).days for d in _settlement_only_holidays(int(y))}
    return s[~np.isin(s, np.array(sorted(off), np.int64))]


def default_settle() -> Callable[[int], int]:
    """Settlement day of a trade made on day number ``dn``: the 2nd SETTLEMENT day after it (the 1st from 2024-05-28;
    production's `borrow_fees.settlement_date`) — the sessions AAPL traded in the deep 30-minute store (weekdays after
    its data) less the bank holidays on which nothing settles (`settlement_days`)."""
    if "s" not in _CAL:
        from src.data.intraday_store import load_deep_30m
        df = load_deep_30m("AAPL")
        days = pd.DatetimeIndex(df.index).tz_localize("UTC").tz_convert(ET).normalize().tz_localize(None)
        s = np.unique(((days - pd.Timestamp("1970-01-01")).days).to_numpy().astype(np.int64))
        extra = pd.bdate_range(pd.Timestamp("1970-01-01") + pd.Timedelta(days=int(s[-1]) + 1), "2027-12-31")
        _CAL["s"] = settlement_days(np.r_[s, ((extra - pd.Timestamp("1970-01-01")).days).to_numpy().astype(np.int64)])
    return make_settle(_CAL["s"])


def make_settle(settlement: np.ndarray) -> Callable[[int], int]:
    """The settlement-day function over the sorted settlement days ``settlement`` (day numbers): a trade made on day
    ``dn`` settles on the 2nd of them after it (the 1st from 2024-05-28)."""
    cal = np.asarray(settlement, np.int64)
    memo: Dict[int, int] = {}

    def settle(dn: int) -> int:
        dn = int(dn)
        v = memo.get(dn)
        if v is None:
            i = int(np.searchsorted(cal, dn, side="right"))       # the first settlement day after the trade
            v = int(cal[i + (0 if dn >= T1_FROM else 1)])
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
        # a piece exits after it is entered: one that enters and exits at the same instant (a passive fill covered
        # within its bar, a window cut at its entry bar) exits 1 ns after — at one instant the engine takes exits
        # before entries, so it would otherwise open after its own exit and never close
        self.xns = np.maximum(np.array([int(p["xns"]) for p in P], np.int64), self.ens + 1)
        for a in ("e", "x", "hs_in", "hs_out", "dv20"):
            setattr(self, a, np.array([float(p[a]) for p in P], float))
        self.hs_mark = np.array([float(p["hs_mark"]) if p.get("hs_mark") is not None else float(p["hs_out"])
                                 for p in P], float)
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
    instants, the open longs' and shorts' value at their marks (``curve_long`` / ``curve_short``), the same per
    strategy (``curve_by``: {name: values}) and their initial margin requirement (``curve_req``: equity minus it is
    the capital still available). ``keep_skips`` adds ``skips``:
    every entry the account could not take in full — the shares its slice wanted, the shares it got (0 = skipped) and
    the binding limit (``margin``: the initial-margin room; ``side_cap``: the side's open value at 100% of equity;
    ``volume_cap``: 1% of the dollar volume; ``house_shrink``; ``max_open``; ``min_equity``: under FINRA's $2,000;
    ``below_one_share``; ``lendable``; ``group_cap``: its group at the strategy's `Strategy.group_cap`, with the
    ``group``) — see `capital_summary`. A limit order the group cap keeps from being placed is logged ``group_cap`` in
    ``orders``."""
    rules = rules or Rules()
    strat = {s.name: s for s in strategies} if not isinstance(strategies, dict) else dict(strategies)
    settle = settle or default_settle()
    start = float(rules.start)
    keep_curve = keep_curve or keep_exposure
    if not pieces:
        out = {"final": start, "calls": [], "ruined": None, "maxdd": 0.0, "maxdd_nav": 0.0, "trades": 0,
               "positions": [], "curve_t": np.zeros(0, np.int64), "curve_v": np.zeros(0)}
        if keep_exposure:
            out.update(curve_long=np.zeros(0), curve_short=np.zeros(0), curve_req=np.zeros(0),
                       curve_by={s: np.zeros(0) for s in strat})
        return out
    bk = Book(pieces, strat, settle)
    T, KIND, II, JJ, DN = bk.events()
    side, hm, hi, d0, cum, bfac, last_out = bk.side, bk.hm, bk.hi, bk.d0, bk.cum, bk.bfac, bk.last_out
    S = [strat[s] for s in bk.strat]
    # each entry's slice count: its strategy's, or the piece's own ``slices`` (a sizing re-tuned week by week, PREREG80)
    SL = np.array([float(pk.get("slices") or S[k].slices) for k, pk in enumerate(bk.pieces)], float)
    # LIMIT ORDERS: the orders of strategies with ``orders`` (pieces sharing an ``oid``); their placements (kind 3) and
    # lapses (kind 4) join the timeline, after the exits, marks and entries of the same instant
    oid_of = np.full(bk.n, -1, np.int64)
    ord_k = np.zeros(bk.n, np.int64)
    ORD: List[dict] = []
    if any(s.orders for s in strat.values()):
        idx_of: Dict[tuple, int] = {}
        for k in range(bk.n):
            pk = bk.pieces[k]
            if not S[k].orders or pk.get("oid") is None:
                continue
            key = (bk.strat[k], pk["oid"])
            o = idx_of.get(key)
            if o is None:
                o = idx_of[key] = len(ORD)
                ORD.append({"row": k, "place": np.asarray(pk["ord_place"], np.int64), "lim": float(pk["ord_lim"]),
                            "end": int(pk["ord_end"]), "rank": float(pk.get("rank", 0.0)), "oid": pk["oid"]})
            oid_of[k] = o
            ord_k[k] = int(pk["ord_k"])
        if ORD:
            et, ek, ei, ej = [T], [KIND], [II], [JJ]
            for o, od in enumerate(ORD):
                m = len(od["place"])
                et += [od["place"], np.array([od["end"]], np.int64)]
                ek += [np.full(m, 3, np.int8), np.array([4], np.int8)]
                ei += [np.full(m, o, np.int64), np.array([o], np.int64)]
                ej += [np.arange(m, dtype=np.int64), np.array([-1], np.int64)]
            T, KIND, II, JJ = (np.concatenate(z) for z in (et, ek, ei, ej))
            o_ = np.lexsort((JJ, II, KIND, T))
            T, KIND, II, JJ = T[o_], KIND[o_], II[o_], JJ[o_]
            DN = day_numbers(T)
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

    def acct(i, n, e, cin, raw):
        """Production's dollar P&L of the price-and-cost result ``raw``: shares x entry x the ledger's net return,
        which `spread._pct_return` prices on the cost-adjusted entry — ``raw`` / (1 - the entry cost's share) for a
        short, / (1 + it) for a long (ledger rules); ``raw`` under the audited engine's rules."""
        if not led_rules or n <= 0 or e <= 0:
            return raw
        c = cin / (n * e)
        return raw / (1.0 - c) if side[i] < 0 else raw / (1.0 + c)

    def unreal(i, n, px, today):
        xc = n * px * bk.hs_mark[i] / 1e4 if mark_net else 0.0
        if side[i] < 0:
            return acct(i, n, o_e[i], o_cin[i], n * (o_e[i] - px) - o_cin[i] - xc) \
                - (n * acc(i, today) if today >= d0[i] else 0.0)
        return acct(i, n, o_e[i], o_cin[i], n * (px - o_e[i]) - o_cin[i] - xc)

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

    rf = str(getattr(rules, "fees", "engine"))
    led_rules = rf == "ledger"                         # the account is the ledger's (production's sizing and P&L)
    led = np.array([str(S[k].fees or rf) == "ledger" for k in range(bk.n)], bool)
    mark_net = bool(getattr(rules, "mark_net", False))

    def entry_cost(i, n, e):
        """The order's half-spread + commission, + the SEC fee and TAF when it is a sale (a short) — summed in the
        audited engine's order (bit-identical results). Ledger costs: the piece's all-in entry cost."""
        if led[i]:
            return n * e * bk.hs_in[i] / 1e4
        if side[i] < 0:
            return n * e * bk.hs_in[i] / 1e4 + comm(n, e) + SEC_FEE * n * e + min(TAF_PER_SHARE * n, TAF_MAX)
        return n * e * bk.hs_in[i] / 1e4 + comm(n, e)

    def room_cost(i, n, e):
        """What a new position's sizing takes out of the margin room: under ledger rules production's — the
        commission, + the SEC fee and TAF for a short (`sim_account.size`; a long: `live_account.size_long` /
        `fill_cap`); otherwise its entry cost (the audited engine's)."""
        if led_rules:
            c = comm(n, e)
            return c + SEC_FEE * n * e + min(TAF_PER_SHARE * n, TAF_MAX) if side[i] < 0 else c
        return entry_cost(i, n, e)

    def exit_cost(i, n, xp, call=False):
        """A leg's exit cost: the half-spread + commission (+ the SEC fee and TAF on a long's sale); a margin call
        pays twice the ENTRY half-spread (the audited engine's rule). Ledger costs: the piece's all-in exit cost."""
        if led[i]:
            return n * xp * bk.hs_out[i] / 1e4
        h = 2 * bk.hs_in[i] if call else bk.hs_out[i]
        if side[i] < 0:
            return n * xp * h / 1e4 + comm(n, xp)
        return n * xp * h / 1e4 + comm(n, xp) + sale_fees(n, xp)

    cash = start
    o_n, o_e, o_cin, o_px, o_hi, o_eq = {}, {}, {}, {}, {}, {}
    open_by = {}                                       # strategy -> open positions
    calls, peak, maxdd = [], start, 0.0
    units, nav_peak, maxdd_nav = start, 1.0, 0.0      # the drawdown on NAV = equity / start (the audited engine's)
    curve_t, curve_v = [], []
    curve_l, curve_s, curve_r = [], [], []             # keep_exposure: longs' / shorts' value, initial requirement
    curve_b: Dict[str, list] = {s: [] for s in strat}  # keep_exposure: each strategy's open value
    skips: List[dict] = []                             # keep_skips: entries not taken in full

    def skip_rec(i, t, why, want, got, eq, room):
        return {"strategy": bk.strat[i], "tkn": bk.tkn[i], "t": int(t), "reason": why, "want": int(want),
                "got": int(got), "price": float(bk.e[i]), "equity": eq, "room": room}
    positions: List[dict] = []
    pos_of: Dict[int, dict] = {}

    def close_position(i, xp, xt, pnl, cost, how):
        rec = pos_of.pop(i)
        if "pnl_partial" in rec:                       # shares sold earlier to make room (capital priority)
            pnl += rec.pop("pnl_partial")
            cost += rec.pop("cost_partial")
        rec.update(exit_ns=int(xt), exit_px=float(xp), pnl=float(pnl), cost_out=float(cost), how=how)
        positions.append(rec)
        open_by[bk.strat[i]] -= 1
        if reenter_on and S[i].reenter and how in ("preempted", "margin_call") and int(xt) < int(bk.xns[i]):
            pending.add(i)

    reenter_on = any(s.reenter for s in strat.values())
    pending: set = set()                               # reenter: pieces closed early, waiting for an open with room
    reentries: List[dict] = []
    open_mark: Dict[int, np.ndarray] = {}              # reenter: which of a piece's marks fall at 09:30 ET
    if reenter_on:
        for k in range(bk.n):
            if S[k].reenter and len(bk.pt[k]):
                idx = pd.DatetimeIndex(pd.to_datetime(bk.pt[k], unit="ns", utc=True)).tz_convert(ET)
                open_mark[k] = np.asarray((idx.hour == 9) & (idx.minute == 30))
    pre_on = any(s.preemptible for s in strat.values())
    delev = tuple(rules.delever) if rules.delever is not None else None
    delevers: List[dict] = []                          # de-leveraging: positions sold to cut the requirement
    refill_on = any(s.preemptible and s.refill_min is not None for s in strat.values())
    preempts: List[dict] = []                          # capital priority: shares sold to make room
    refills: List[dict] = []                           # capital priority: shares bought back
    res: Dict[int, dict] = {}                          # limit orders working: order -> quantity, limit, session, placed
    ord_done: set = set()                              # limit orders filled
    working_by: Dict[str, int] = {}                    # strategy -> its working orders
    ord_log: List[dict] = []                           # every order's life: placed / filled / cancel_* / lapsed / ...

    def reserved(i):
        """The working orders' reserved initial requirement, and their value on entry ``i``'s side ("reserve")."""
        rq = gv = 0.0
        for r in res.values():
            if r["reserve"]:
                rq += ireq_open(r["row"], r["n"], r["lim"])
                if i is not None and side[r["row"]] == side[i]:
                    gv += r["n"] * r["lim"]
        return rq, gv

    def cancel(o, t, why):
        r = res.pop(o)
        working_by[bk.strat[r["row"]]] -= 1
        ord_log.append({"t": int(t), "what": why, "oid": ORD[o]["oid"], "tkn": bk.tkn[r["row"]], "n": int(r["n"])})

    book_of = [S[k].book or S[k].name for k in range(bk.n)]
    one_book = bool(getattr(rules, "one_book_per_name", False))

    def other_book_holds(i) -> bool:
        if not one_book:
            return False
        tk, b = _plain(bk.tkn[i]), book_of[i]
        return any(_plain(bk.tkn[w]) == tk and book_of[w] != b for w in o_n)

    # GROUP CAP (PREREG100; off unless a strategy has a ``group_cap``): each piece's group (None — or NaN — never
    # capped, never counted) and each strategy's scope (the strategies whose positions and working orders count)
    cap_on = any(s.group_cap is not None for s in strat.values())
    grp: List[Optional[str]] = []
    scope_of: Dict[str, frozenset] = {}
    if cap_on:
        for pk in bk.pieces:
            g_ = pk.get("group")
            grp.append(None if g_ is None or (isinstance(g_, float) and g_ != g_) else g_)
        scope_of = {s.name: frozenset(s.group_scope) if s.group_scope is not None else frozenset((s.name,))
                    for s in strat.values()}

    def group_full(i) -> bool:
        """Entry (or order) ``i``'s group already holds its strategy's ``group_cap`` open positions and working
        orders among the strategies of its scope."""
        st_ = S[i]
        if not cap_on or st_.group_cap is None or grp[i] is None:
            return False
        g_, sc = grp[i], scope_of[st_.name]
        k = sum(1 for w in o_n if grp[w] == g_ and bk.strat[w] in sc)
        k += sum(1 for r_ in res.values() if grp[r_["row"]] == g_ and bk.strat[r_["row"]] in sc)
        return k >= int(st_.group_cap)

    def gives_way(r, st):
        """A working order the entry of strategy ``st`` may cancel to make room."""
        so = S[r["row"]]
        return r["reserve"] and so.preemptible and so.quality is not None and so.quality < st.quality

    def sell_part(i, w, k, xp, t, today):
        """Sell ``k`` shares of open position ``w`` at ``xp`` to make room for entry ``i``: their P&L and exit costs
        are realized now; the position closes when no share is left."""
        nonlocal cash
        n = o_n[w]
        k = min(int(k), n)
        cin_k = o_cin[w] * k / n
        cout = exit_cost(w, k, xp)
        if side[w] < 0:
            pnl = acct(w, k, o_e[w], cin_k, k * (o_e[w] - xp) - cin_k - cout) - k * acc_liq(w, today)
        else:
            pnl = acct(w, k, o_e[w], cin_k, k * (xp - o_e[w]) - cin_k - cout)
        cash += pnl
        preempts.append({"t": int(t), "by": bk.strat[i], "by_tkn": bk.tkn[i], "strategy": bk.strat[w],
                         "tkn": bk.tkn[w], "shares": int(k), "px": float(xp), "pnl": float(pnl), "cost": float(cout),
                         "closed": bool(k >= n)})
        if k >= n:
            for d in (o_n, o_e, o_cin, o_px, o_hi, o_eq):
                d.pop(w)
            close_position(w, xp, t, pnl, cout, "preempted")
        else:
            o_n[w] = n - k
            o_cin[w] -= cin_k
            rec = pos_of[w]
            rec["pnl_partial"] = rec.get("pnl_partial", 0.0) + pnl
            rec["cost_partial"] = rec.get("cost_partial", 0.0) + cout
            rec["sold"] = rec.get("sold", 0) + int(k)

    def entry_size(i, st, e, today):
        """The entry step's sizing, recomputed after a sale (the same arithmetic as the inline step)."""
        eq = cash
        g = 0.0
        room_req = 0.0
        for w, n in o_n.items():
            pw = o_px[w]
            eq += unreal(w, n, pw, today)
            if side[w] == side[i]:
                g += n * pw
            room_req += ireq_open(w, n, pw)
        if res:
            rq, gv = reserved(i)
            room_req += rq
            g += gv
        room = (eq if st.room_frac is None else st.room_frac * eq) - room_req
        r_i = irate(i, e)
        nm = int(math.floor(room / r_i)) if room > 0 else 0
        if nm > 0:
            nm = int(math.floor(max(room - room_cost(i, nm, e), 0.0) / r_i))
        alloc = min(eq / SL[i], gcap[int(side[i])] * eq - g)
        want_v = eq / SL[i]
        if st.shrink_budget is not None and hi[i] > st.shrink_budget:
            alloc = alloc * st.shrink_budget / hi[i]
            want_v = want_v * st.shrink_budget / hi[i]
        if st.adv_cap is not None:
            alloc = min(alloc, st.adv_cap * bk.dv20[i])
            want_v = min(want_v, st.adv_cap * bk.dv20[i])
        n_alloc = int(math.floor(alloc / e)) if e > 0 and alloc > 0 else 0
        want = int(math.floor(want_v / e)) if e > 0 and want_v > 0 else 0
        return eq, room, r_i, nm, n_alloc, want

    def helps(i, st, e, today) -> bool:
        """Would freeing EVERY candidate (the working orders that give way, the preemptible positions of lower
        quality — as production's `live_account.give_way_plan` frees them: equity unchanged, their requirement and
        their side's value released) let entry ``i`` take more shares than it takes now? When not, nothing is sold:
        the shorts' own requirement past the equity, say, leaves the entry unfunded whatever the longs free (PREREG64)."""
        eq, room, r_i, nm, n_alloc, want = entry_size(i, st, e, today)
        return freed_size(i, st, e, today) > min(n_alloc, nm)

    def freed_size(i, st, e, today) -> int:
        """The shares entry ``i`` could take at ``e`` were every candidate of `helps` freed (nothing is sold)."""
        eq, room, r_i, nm, n_alloc, want = entry_size(i, st, e, today)
        d_room = d_g = 0.0
        g = sum(o_n[w] * o_px[w] for w in o_n if side[w] == side[i])
        for r in res.values():
            if r["reserve"] and side[r["row"]] == side[i]:
                g += r["n"] * r["lim"]
            if gives_way(r, st):
                d_room += ireq_open(r["row"], r["n"], r["lim"])
                d_g += r["n"] * r["lim"] if side[r["row"]] == side[i] else 0.0
        for w in o_n:
            if S[w].preemptible and S[w].quality is not None and S[w].quality < st.quality:
                d_room += ireq_open(w, o_n[w], o_px[w])
                d_g += o_n[w] * o_px[w] if side[w] == side[i] else 0.0
        room2 = room + d_room
        nm2 = int(math.floor(room2 / r_i)) if room2 > 0 else 0
        if nm2 > 0:
            nm2 = int(math.floor(max(room2 - room_cost(i, nm2, e), 0.0) / r_i))
        alloc2 = min(eq / SL[i], gcap[int(side[i])] * eq - (g - d_g))
        if st.shrink_budget is not None and hi[i] > st.shrink_budget:
            alloc2 = alloc2 * st.shrink_budget / hi[i]
        if st.adv_cap is not None:
            alloc2 = min(alloc2, st.adv_cap * bk.dv20[i])
        n2 = int(math.floor(alloc2 / e)) if e > 0 and alloc2 > 0 else 0
        return min(n2, nm2)

    def make_room(i, st, e, t, today, want_n=None):
        """Capital priority: sell lower-quality preemptible positions until entry ``i`` gets its whole slice (or no
        candidate is left) — production's give-way: while the MARGIN ROOM holds the entry below its side-capped size,
        any side's candidate (a sale frees room); once only its side's cap does, its own side's (only they free it);
        nothing at all when even selling every candidate would not let it take more (`helps`, PREREG64).
        ``want_n``: a limit order's fill wants its own quantity, not the slice at the fill price (PREREG86).
        Returns the shares it can now take."""
        if not helps(i, st, e, today):
            _, _, _, nm, n_alloc, _ = entry_size(i, st, e, today)
            return min(n_alloc, nm)
        for _ in range(500):
            eq, room, r_i, nm, n_alloc, want = entry_size(i, st, e, today)
            if want_n is not None:
                want = int(want_n)
            n = min(n_alloc, nm)
            if n >= want or want <= 0:
                return n
            room_bound = nm < n_alloc                  # else the side's cap (or the slice) holds it
            ocands = [o for o, r in res.items() if gives_way(r, st) and (room_bound or side[r["row"]] == side[i])]
            if ocands:                                 # working orders give way before positions
                o = min(ocands, key=lambda v: (S[res[v]["row"]].quality,
                                               (-1 if S[res[v]["row"]].yield_first == "latest" else 1) * res[v]["t"]))
                cancel(o, t, "cancel_giveway")
                continue
            cands = [w for w in o_n if S[w].preemptible and S[w].quality is not None and S[w].quality < st.quality
                     and (room_bound or side[w] == side[i])]
            if not cands:
                return n
            w = min(cands, key=lambda v: (S[v].quality, (-1 if S[v].yield_first == "latest" else 1)
                                          * pos_of[v]["entry_ns"]))
            pw = o_px[w]
            if not room_bound:
                k = int(math.ceil((want - n_alloc) * e / pw)) + 1
            else:
                tgt = want if side[w] == side[i] else min(want, n_alloc)
                per = ireq_open(w, 1, pw)
                k = int(math.ceil(((tgt - nm) * r_i + entry_cost(i, tgt, e)) / max(per * 0.95, 1e-9))) + 1
            sell_part(i, w, o_n[w] if S[w].yield_whole else max(1, min(k, o_n[w])), pw, t, today)
        _, _, _, nm, n_alloc, _ = entry_size(i, st, e, today)
        return min(n_alloc, nm)

    def refill(t, today, marked):
        """Capital priority: each preemptible position with ``refill_min`` and a mark at this instant buys back toward
        its slice (within its side's room and the margin room) once the gap is at least ``refill_min`` of equity."""
        for w in [v for v in o_n if v in marked and S[v].preemptible and S[v].refill_min is not None]:
            stw = S[w]
            pw = o_px[w]
            eq, g, rr = cash, 0.0, 0.0
            for v, nv in o_n.items():
                pv = o_px[v]
                eq += unreal(v, nv, pv, today)
                if side[v] == side[w]:
                    g += nv * pv
                rr += ireq_open(v, nv, pv)
            if res:
                rq_, gv_ = reserved(w)
                rr += rq_
                g += gv_
            own = o_n[w] * pw
            target = min(eq / SL[w], gcap[int(side[w])] * eq - (g - own))
            if stw.adv_cap is not None:
                target = min(target, stw.adv_cap * bk.dv20[w])
            gap = target - own
            if gap < stw.refill_min * eq or pw <= 0:
                continue
            room = eq - rr
            r_w = irate(w, pw)
            km = int(math.floor(room / r_w)) if room > 0 else 0
            if km > 0:
                km = int(math.floor(max(room - room_cost(w, km, pw), 0.0) / r_w))
            k = min(int(math.floor(gap / pw)), km)
            if k <= 0:
                continue
            cost = entry_cost(w, k, pw)
            n0 = o_n[w]
            o_e[w] = (n0 * o_e[w] + k * pw) / (n0 + k)
            o_n[w] = n0 + k
            o_cin[w] += cost
            rec = pos_of[w]
            rec["bought"] = rec.get("bought", 0) + int(k)
            rec["cost_in"] += cost
            refills.append({"t": int(t), "strategy": bk.strat[w], "tkn": bk.tkn[w], "shares": int(k), "px": float(pw),
                            "cost": float(cost)})

    N = len(T)
    p = 0
    while p < N:
        t = T[p]
        q = p
        while q < N and T[q] == t:
            q += 1
        today = int(DN[p])
        bar_now = set()
        re_now = []                                    # reenter: pending pieces at a 09:30 mark this instant
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
                    cout = exit_cost(i, n, xp)
                    if side[i] < 0:
                        pnl = acct(i, n, e, cin, n * (e - xp) - cin - cout) - n * bk.full_ps[i]
                    else:
                        pnl = acct(i, n, e, cin, n * (xp - e) - cin - cout)
                    cash += pnl
                    close_position(i, xp, t, pnl, cout, "exit")
                elif reenter_on and i in pending:
                    pending.discard(i)                 # its holding is over: nothing left to re-buy
            elif KIND[z] == 1:
                if i in o_n:
                    j = JJ[z]
                    o_px[i] = bk.pc[i][j]
                    o_hi[i] = bk.ph[i][j] if side[i] < 0 else bk.pl[i][j]
                    bar_now.add(i)
                elif reenter_on and i in pending and open_mark[i][JJ[z]]:
                    re_now.append((i, JJ[z]))
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
                    cout = exit_cost(i, n, xp, call=True)
                    if side[i] < 0:
                        pnl = acct(i, n, e, cin, n * (e - xp) - cin - cout) - n * acc_liq(i, today)
                    else:
                        pnl = acct(i, n, e, cin, n * (xp - e) - cin - cout)
                    cash += pnl
                    close_position(i, xp, t, pnl, cout, "margin_call")
                o_n.clear(), o_e.clear(), o_cin.clear(), o_px.clear(), o_hi.clear(), o_eq.clear()
                for o in list(res):                    # a margin call cancels every working order
                    cancel(o, t, "cancel_margin")
                calls.append((int(t), eq0, cash))
                if cash <= 0:
                    curve_t.append(int(t))
                    curve_v.append(cash)
                    curve_l.append(0.0), curve_s.append(0.0), curve_r.append(0.0)
                    for b_ in curve_b.values():
                        b_.append(0.0)
                    dd = (peak - cash) / peak if peak > 0 else 1.0
                    nav = cash / units
                    maxdd_nav = max(maxdd_nav, (nav_peak - nav) / nav_peak if nav_peak > 0 else 1.0)
                    out = {"final": cash, "calls": calls, "maxdd": max(maxdd, dd), "maxdd_nav": maxdd_nav,
                           "ruined": int(t), "trades": len(positions), "positions": positions,
                           "curve_t": np.array(curve_t), "curve_v": np.array(curve_v)}
                    if keep_exposure:
                        out.update(curve_long=np.array(curve_l), curve_short=np.array(curve_s),
                                   curve_req=np.array(curve_r), curve_by={s: np.array(v) for s, v in curve_b.items()})
                    if keep_skips:
                        out["skips"] = skips
                    if pre_on:
                        out.update(preemptions=preempts, refills=refills)
                    if reenter_on:
                        out["reentries"] = reentries
                    if delev is not None:
                        out["delevers"] = delevers
                    if ORD:
                        out["orders"] = ord_log
                    return out
        if delev is not None and o_n and bar_now:
            trig, tgt = delev
            need_c, eq_c = 0.0, cash
            for w, n in o_n.items():
                need_c += mreq(w, n, o_px[w])
                eq_c += unreal(w, n, o_px[w], today)
            if eq_c > 0 and need_c > trig * eq_c:
                for o in [o for o, r in res.items() if r["reserve"] and S[r["row"]].delever_order is not None]:
                    cancel(o, t, "cancel_delever")     # de-leveraging cancels the working orders first
                for w in sorted((w for w in o_n if S[w].delever_order is not None),
                                key=lambda w: (S[w].delever_order, (-1 if S[w].delever_first == "latest" else 1)
                                               * int(bk.ens[w]))):
                    if need_c <= tgt * eq_c:
                        break
                    n, e, cin, xp = o_n[w], o_e[w], o_cin[w], o_px[w]
                    cout = exit_cost(w, n, xp)
                    if side[w] < 0:
                        pnl = acct(w, n, e, cin, n * (e - xp) - cin - cout) - n * acc_liq(w, today)
                    else:
                        pnl = acct(w, n, e, cin, n * (xp - e) - cin - cout)
                    cash += pnl
                    for d in (o_n, o_e, o_cin, o_px, o_hi, o_eq):
                        d.pop(w)
                    close_position(w, xp, t, pnl, cout, "delever")
                    delevers.append({"t": int(t), "strategy": bk.strat[w], "tkn": bk.tkn[w], "shares": int(n),
                                     "px": float(xp), "pnl": float(pnl), "need": float(need_c), "equity": float(eq_c)})
                    need_c, eq_c = 0.0, cash
                    for v, m in o_n.items():
                        need_c += mreq(v, m, o_px[v])
                        eq_c += unreal(v, m, o_px[v], today)
        zs = [z for z in range(p, q) if KIND[z] == 2]
        if ORD:                                        # a limit order's fills first (the live tick's order)
            zs = [z for z in zs if oid_of[II[z]] >= 0] + [z for z in zs if oid_of[II[z]] < 0]
        for z in zs:
            i = II[z]
            st = S[i]
            if oid_of[i] >= 0:                         # a limit order's touch: it fills if working since before it
                o = int(oid_of[i])
                r = res.get(o)
                if r is None or ord_k[i] < r["k"]:
                    continue
                if other_book_holds(i):                # another book holds the name: no fill, the order keeps working
                    ord_log.append({"t": int(t), "what": "held", "oid": ORD[o]["oid"], "tkn": bk.tkn[i], "n": 0})
                    continue
                # the fill takes AT MOST what the account can fund now (PREREG61 amendment 1): equity minus the open
                # positions' initial requirement and the OTHER working orders' reservations, within its side cap; the
                # rest of the order is not bought. No room: nothing is bought and the order keeps working
                e = bk.e[i]
                if rules.fills_make_room and st.quality is not None and any(
                        S[w].preemptible and S[w].quality is not None and S[w].quality < st.quality for w in o_n):
                    make_room(i, st, e, t, today, want_n=int(r["n"]))   # PREREG86: lower-quality books give way
                eq, rq_now, g = cash, 0.0, 0.0
                for w, m in o_n.items():
                    pw = o_px[w]
                    eq += unreal(w, m, pw, today)
                    rq_now += ireq_open(w, m, pw)
                    if side[w] == side[i]:
                        g += m * pw
                for o2, r2 in res.items():
                    if o2 != o and r2["reserve"]:
                        rq_now += ireq_open(r2["row"], r2["n"], r2["lim"])
                        if side[r2["row"]] == side[i]:
                            g += r2["n"] * r2["lim"]
                room_before = eq - rq_now
                r_i = irate(i, e)
                nm = int(math.floor(room_before / r_i)) if room_before > 0 else 0
                if nm > 0:
                    nm = int(math.floor(max(room_before - room_cost(i, nm, e), 0.0) / r_i))
                n_side = int(math.floor((gcap[int(side[i])] * eq - g) / e)) if e > 0 else 0
                n = min(int(r["n"]), nm, n_side)
                need = ireq_open(i, int(r["n"]), e) + entry_cost(i, int(r["n"]), e)
                if n <= 0:
                    ord_log.append({"t": int(t), "what": "unfunded", "oid": ORD[o]["oid"], "tkn": bk.tkn[i], "n": 0,
                                    "n_order": int(r["n"]), "room": float(room_before), "need": float(need)})
                    continue
                res.pop(o)
                working_by[st.name] -= 1
                ord_done.add(o)
                o_n[i], o_e[i] = n, e
                o_cin[i] = entry_cost(i, n, e)
                o_px[i], o_hi[i], o_eq[i] = e, e, eq
                open_by[st.name] = open_by.get(st.name, 0) + 1
                pos_of[i] = {"strategy": st.name, "tkn": bk.tkn[i], "side": int(side[i]), "shares": n,
                             "entry_ns": int(t), "entry_px": float(e), "cost_in": float(o_cin[i]), "equity_in": float(eq),
                             "net": bk.pieces[i].get("net"), "days": bk.pieces[i].get("days"), "placed_ns": int(r["t"])}
                ord_log.append({"t": int(t), "what": "filled", "oid": ORD[o]["oid"], "tkn": bk.tkn[i], "n": n,
                                "n_order": int(r["n"]), "room": float(room_before), "need": float(need)})
                continue
            if st.max_open is not None and open_by.get(st.name, 0) + working_by.get(st.name, 0) >= st.max_open:
                if keep_skips:
                    skips.append(skip_rec(i, t, "max_open", 0, 0, None, None))
                continue
            if other_book_holds(i):
                if keep_skips:
                    skips.append(skip_rec(i, t, "other_book", 0, 0, None, None))
                continue
            if cap_on and group_full(i):               # PREREG100: its group is at the strategy's cap
                if keep_skips:
                    skips.append(dict(skip_rec(i, t, "group_cap", 0, 0, None, None), group=grp[i]))
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
            if res:
                rq, gv = reserved(i)
                room_req += rq
                g += gv
            if eq < st.min_equity:
                if keep_skips:
                    skips.append(skip_rec(i, t, "min_equity", 0, 0, eq, None))
                continue
            room = (eq if st.room_frac is None else st.room_frac * eq) - room_req
            r_i = irate(i, e)
            nm = int(math.floor(room / r_i)) if room > 0 else 0
            if nm > 0:
                nm = int(math.floor(max(room - room_cost(i, nm, e), 0.0) / r_i))
            alloc = min(eq / SL[i], gcap[int(side[i])] * eq - g)
            why_alloc = "side_cap" if gcap[int(side[i])] * eq - g < eq / SL[i] else None
            if st.shrink_budget is not None and hi[i] > st.shrink_budget:
                alloc = alloc * st.shrink_budget / hi[i]
                why_alloc = "house_shrink"
            if st.adv_cap is not None and alloc > st.adv_cap * bk.dv20[i]:
                alloc = st.adv_cap * bk.dv20[i]
                why_alloc = "volume_cap"
            n_alloc = int(math.floor(alloc / e)) if e > 0 and alloc > 0 else 0
            n = min(n_alloc, nm)
            # room is made only when the whole slice passes the lendability check (every smaller size then does):
            # never a sale for an entry the check then refuses
            lend_full = st.avail_mult is None or n_alloc * st.avail_mult <= bk.avail[i]
            if pre_on and lend_full and st.quality is not None and (any(S[w].preemptible and S[w].quality is not None
                                                                        and S[w].quality < st.quality for w in o_n)
                                                                    or any(gives_way(r, st) for r in res.values())):
                n = make_room(i, st, e, t, today)
                eq, room, r_i, nm, n_alloc, _ = entry_size(i, st, e, today)
            if keep_skips:
                want = int(math.floor(eq / SL[i] / e)) if e > 0 and eq > 0 else 0
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
        for i, j in re_now:
            st = S[i]
            if i not in pending or i in o_n or (st.max_open is not None and open_by.get(st.name, 0) >= st.max_open):
                continue
            if cap_on and group_full(i):               # PREREG100: waits for its group to fall below the cap
                continue
            e = float(bk.pc[i][j])
            eq, room, r_i, nm, n_alloc, _ = entry_size(i, st, e, today)
            n = min(n_alloc, nm)
            if eq < st.min_equity or n <= 0:
                continue
            o_n[i], o_e[i] = n, e
            o_cin[i] = entry_cost(i, n, e)
            o_px[i], o_hi[i], o_eq[i] = e, e, eq
            open_by[st.name] = open_by.get(st.name, 0) + 1
            pos_of[i] = {"strategy": st.name, "tkn": bk.tkn[i], "side": int(side[i]), "shares": int(n),
                         "entry_ns": int(t), "entry_px": float(e), "cost_in": float(o_cin[i]), "equity_in": float(eq),
                         "net": bk.pieces[i].get("net"), "days": bk.pieces[i].get("days"), "reentry": True}
            pending.discard(i)
            reentries.append({"t": int(t), "strategy": st.name, "tkn": bk.tkn[i], "shares": int(n), "px": e})
        if ORD:
            # limit orders placed now (deepest rank first), each sized at its limit like an entry; then the lapses
            plc = sorted((ORD[int(II[z])]["rank"], int(II[z]), int(JJ[z])) for z in range(p, q) if KIND[z] == 3)
            for _, o, m in plc:
                if o in res or o in ord_done:
                    continue
                row = ORD[o]["row"]
                st = S[row]
                if st.max_open is not None and open_by.get(st.name, 0) + working_by.get(st.name, 0) >= st.max_open:
                    ord_log.append({"t": int(t), "what": "no_slot", "oid": ORD[o]["oid"], "tkn": bk.tkn[row], "n": 0})
                    continue
                if other_book_holds(row):
                    ord_log.append({"t": int(t), "what": "held", "oid": ORD[o]["oid"], "tkn": bk.tkn[row], "n": 0})
                    continue
                if cap_on and group_full(row):         # PREREG100: not placed now, tried at its next placement
                    ord_log.append({"t": int(t), "what": "group_cap", "oid": ORD[o]["oid"], "tkn": bk.tkn[row], "n": 0,
                                    "group": grp[row]})
                    continue
                lim = ORD[o]["lim"]
                eq, room, r_i, nm, n_alloc, _ = entry_size(row, st, lim, today)
                n = min(n_alloc, nm)
                if rules.fills_make_room and st.quality is not None:   # PREREG86: sized through lower-quality books
                    n = max(n, freed_size(row, st, lim, today))
                if eq < st.min_equity or n <= 0:
                    ord_log.append({"t": int(t), "what": "no_room", "oid": ORD[o]["oid"], "tkn": bk.tkn[row], "n": 0})
                    continue
                res[o] = {"n": int(n), "lim": lim, "k": m, "t": int(t), "row": row, "reserve": st.orders == "reserve"}
                working_by[st.name] = working_by.get(st.name, 0) + 1
                ord_log.append({"t": int(t), "what": "placed", "oid": ORD[o]["oid"], "tkn": bk.tkn[row], "n": int(n)})
            for z in range(p, q):
                if KIND[z] == 4 and int(II[z]) in res:
                    cancel(int(II[z]), t, "lapsed")
        if refill_on and o_n and bar_now:
            refill(t, today, bar_now)
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
                bv = dict.fromkeys(curve_b, 0.0)
                for i, n in o_n.items():
                    if side[i] < 0:
                        sv += n * o_px[i]
                    else:
                        lv += n * o_px[i]
                    bv[bk.strat[i]] += n * o_px[i]
                    rq += ireq_open(i, n, o_px[i])
                if res:
                    rq += reserved(None)[0]
                curve_l.append(lv), curve_s.append(sv), curve_r.append(rq)
                for s, v in bv.items():
                    curve_b[s].append(v)
        p = q
    # positions still open at the end: valued as at every mark — the last mark, the borrow accrued to the last
    # instant's day, net of the exit-cost estimate with `Rules.mark_net` — so the final value is the curve's last
    # point (the audited engine left the accrued borrow out)
    end_day = int(DN[-1]) if len(DN) else 0
    us = {i: unreal(i, n, o_px[i], end_day) for i, n in o_n.items()}
    final = cash + sum(us[i] for i in o_n)
    for i in list(o_n):
        rec = pos_of.pop(i)
        pnl_i, cost_i = us[i], 0.0
        if "pnl_partial" in rec:
            pnl_i += rec.pop("pnl_partial")
            cost_i += rec.pop("cost_partial")
        rec.update(exit_ns=None, exit_px=float(o_px[i]), pnl=float(pnl_i), cost_out=float(cost_i), how="open_at_end")
        positions.append(rec)
    out = {"final": final, "calls": calls, "ruined": None, "maxdd": maxdd, "maxdd_nav": maxdd_nav,
           "trades": len(positions), "positions": positions,
           "curve_t": np.array(curve_t), "curve_v": np.array(curve_v)}
    if pre_on:
        out.update(preemptions=preempts, refills=refills)
    if reenter_on:
        out["reentries"] = reentries
    if delev is not None:
        out["delevers"] = delevers
    if ORD:
        out["orders"] = ord_log
    if keep_exposure:
        out.update(curve_long=np.array(curve_l), curve_short=np.array(curve_s), curve_req=np.array(curve_r),
                   curve_by={s: np.array(v) for s, v in curve_b.items()})
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
    log = res.get("orders") or []
    for n in names:
        mine = [p for p in pieces if str(p["strategy"]) == n]
        if any(p.get("oid") is not None for p in mine):
            # a limit-order book: its orders' lives (`simulate`'s ``orders``), not its touch pieces
            lim = {p["oid"]: float(p["ord_lim"]) for p in mine if p.get("oid") is not None}
            L = [x for x in log if x.get("oid") in lim]
            fills = [x for x in L if x["what"] == "filled"]
            cut = [x for x in fills if int(x["n"]) < int(x.get("n_order", x["n"]))]
            placed = {x["oid"] for x in L if x["what"] == "placed"}
            filled = {x["oid"] for x in fills}
            out[n] = {"orders": len(lim), "placed": len(placed), "filled": len(fills), "full": len(fills) - len(cut),
                      "cut": len(cut), "never_filled": len(placed - filled),
                      "not_placed_by": pd.Series([x["what"] for x in L if x["what"] in ("no_slot", "no_room", "held",
                                                                                         "group_cap")
                                                  and x["oid"] not in placed], dtype=object).value_counts().to_dict(),
                      "touches_unfunded": sum(1 for x in L if x["what"] == "unfunded"),
                      "ended_by": pd.Series([x["what"] for x in L if x["what"] not in
                                             ("placed", "filled", "unfunded", "held", "no_slot", "no_room",
                                              "group_cap")],
                                            dtype=object).value_counts().to_dict(),
                      "usd_not_deployed": float(sum((int(x.get("n_order", x["n"])) - int(x["n"])) * lim[x["oid"]]
                                                    for x in cut))}
            continue
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



# ── a run that starts in the middle of the pieces ────────────────────────────

def rebase(p: dict, lo_ns: int) -> Optional[dict]:
    """A STANDING holding (a piece with ``rebase``: the dual-momentum book's month pick) entered before ``lo_ns`` and
    still held after it, re-entered at its first mark at/after ``lo_ns`` — what an account starting then would buy;
    None when it is no longer held."""
    pt = np.asarray(p["pt"], np.int64)
    k = int(np.searchsorted(pt, int(lo_ns), side="left"))
    if k >= len(pt) or int(pt[k]) >= int(p["xns"]):
        return None
    q = dict(p)
    q["ens"], q["e"] = int(pt[k]), float(np.asarray(p["pc"], float)[k])
    q["pt"] = pt[k + 1:]
    for key in ("pc", "ph", "pl"):
        if key in p:
            q[key] = np.asarray(p[key], float)[k + 1:]
    q["rebased_from"] = int(p["ens"])
    return q


def from_start(pieces: Sequence[dict], lo_ns: int) -> List[dict]:
    """The pieces an account starting at ``lo_ns`` trades: those entered at/after it (a limit order: first placed at/after
    it), and the standing holdings (``rebase``) still held then, re-entered at their first mark (`rebase`)."""
    out: List[dict] = []
    for p in pieces:
        if p.get("oid") is not None:
            if int(np.min(np.asarray(p["ord_place"], np.int64))) >= lo_ns:
                out.append(p)
        elif int(p["ens"]) >= lo_ns:
            out.append(p)
        elif p.get("rebase") and int(p["xns"]) > lo_ns:
            q = rebase(p, lo_ns)
            if q is not None:
                out.append(q)
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
    """End-of-session equity, open longs' and shorts' value at their marks, each strategy's (``by:<name>``) and their
    initial margin requirement, per session (needs ``simulate(..., keep_exposure=True)``). A session without an event
    held nothing: its exposure is 0 and its equity the last one (``start`` before the first event)."""
    days = pd.DatetimeIndex(pd.to_datetime(sessions)).normalize()
    t = np.asarray(res.get("curve_t", []), np.int64)
    by = res.get("curve_by") or {}
    cols = ["equity", "long", "short", "req"] + [f"by:{s}" for s in by]
    df = pd.DataFrame(index=days, columns=cols, dtype=float)
    if not len(t):
        df["equity"] = float(res.get("final", start))
        df[cols[1:]] = 0.0
        return df
    d = pd.to_datetime(t, unit="ns", utc=True).tz_convert(ET).tz_localize(None).normalize()
    data = {"equity": res["curve_v"], "long": res["curve_long"], "short": res["curve_short"], "req": res["curve_req"]}
    data.update({f"by:{s}": v for s, v in by.items()})
    cur = pd.DataFrame(data, index=d)
    last = cur.groupby(level=0).last()
    df.update(last)
    df["equity"] = df["equity"].ffill().fillna(start)
    df[cols[1:]] = df[cols[1:]].fillna(0.0)
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
        Rc = [r for r in R if r.get("exit_ns") is not None]     # the realized trades (not the ones open at the end)
        ret = np.array([100.0 * r["pnl"] / (r["shares"] * r["entry_px"]) for r in Rc])
        hold = np.array([(r["exit_ns"] - r["entry_ns"]) / 86400e9 for r in Rc])
        gross = np.array([(r["exit_px"] - r["entry_px"]) * r["side"] > 0 for r in Rc], bool)
        lc = np.array([math.log(1 + r["pnl"] / r["equity_in"]) if r["equity_in"] > 0 and r["pnl"] > -r["equity_in"]
                       else float("-inf") for r in R])
        total_lc += float(lc.sum())
        out["by_strategy"][name] = {
            "positions": len(R), "closed": len(Rc), "open_at_end": len(R) - len(Rc),
            "pnl": float(sum(r["pnl"] for r in R)), "costs": float(sum(r["cost_in"] + r["cost_out"] for r in R)),
            "avg_ret": float(ret.mean()) if len(ret) else None, "median_ret": float(np.median(ret)) if len(ret) else None,
            "avg_days": float(hold.mean()) if len(hold) else None,
            "return_per_day": per_day(float(ret.mean()), float(hold.mean())) if len(ret) else None,
            # GROSS (the house standard: the right direction on raw prices); net of costs beside it
            "win_rate": float(gross.mean()) if len(gross) else None,
            "net_win_rate": float((ret > 0).mean()) if len(ret) else None,
            "margin_called": int(sum(1 for r in R if r["how"] == "margin_call")),
            "log_contrib_per_year": float(lc.sum()) / years}
    out["log_contrib_per_year"] = total_lc / years
    return out
