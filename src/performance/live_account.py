"""THE LIVE ACCOUNT — one $10,000 simulated IBKR margin account for the best combination (user directive 2026-10-09:
"Replace the current live arms by the best combination model: 'vol + ETF + fallers + dip, dip gives way,
de-leveraging; shorts 1/24, dip 1/10' ... from now on, only the best combination model will be used on the 10000$
account. If we want to test other strategies in live, we'll use another 10000$ account.").

The selection short's traded arms (`sel_short_traded_arms`: vol, ETF, fall) and the dip long book share ONE equity —
$10,000 plus the dollar P&L of every trade the account funded (shorts: ``sel_account_shares``, dip longs:
``dip_account_shares``) — and ONE margin room: the account's equity minus the initial requirement of everything it
holds (a short: the larger of Reg T's tiers and IBKR's house rate, `sim_account`; a long: Reg T's 50%). The rules are
the backtest engine's (`src/backtest/account.py`, PREREG57's best configuration, P4b-style give-way):

* SIZES — each short 1/its arm's slice count of the equity (`sim_account.arm_slices`: vol 20, ETF 16, fall 12 from the
  2026-10-12 session, PREREG80; `sim_account.size`, fed this account's state);
  each dip long 1/`dip_long_account_slices` (3 from the 2026-10-19 session, PREREG147's S3 — only each day's top pick
  among the $2B+ signals; 6 from 2026-10-16, PREREG116's Q1; 10 before) of the equity, at most that many open, the
  longs' value at most the equity, within the margin room (`size_long`).
* THE DIP BOOK GIVES WAY — a short the margin room cannot fund in full sells the dip book's longs, the latest first,
  WHOLE, until it can (`give_way_plan`) — none when selling them all would not let it open one more share (PREREG64:
  the shorts' own initial requirement past the equity used to cost every long for nothing); a long given up is never
  bought back. (The engine's "just enough shares"
  variant measured the same on this combination: +78.9 vs +77.0 %/yr mean over the starts, +56.1 vs +55.4 from 2021,
  no margin call either way — whole positions keep one ledger trade per broker position.)
* DE-LEVERAGING — at a tick where the maintenance requirement exceeds `live_account_delever_trigger` (0.8) x equity,
  the dip book's longs are sold, the latest first, whole, until it is at most `live_account_delever_target` (0.6) x
  equity; never bought back (`delever_plan`).
* MARGIN CALL — AS IBKR LIQUIDATES (from 2026-10-21, PREREG174 / 175; the user: "production and backtest needs to
  liquidate like IBKR does"): along what happened since the last check (`margin_watch`: the bars completed since, the
  live price), where equity falls under the maintenance requirement the most recently opened positions are sold in
  100-share lots until it is back at it, each lot 10% against the account from where the deficit appears, never past
  the extreme; once a session from 15:50 ET the Reg T floor at the marks (`plan_liquidation`, the account engine's
  model; `tracker.monitor_live_account` executes it). `live_account_liquidation` "all": every position closed
  (`margin_call`, the rule before).
* THE CRASH-REBOUND BOOK (2026-10-11, PREREG86 / PREREG88; `src/signals/crash_long.py`): one long at a time in
  the traded fund (TQQQ, on QQQ's decisions), 1/`crash_long_account_slices` (4) of the equity within the long side's
  cap and the room (`size_crash`); short of room it sells the dip book's latest longs whole, as a short does
  (`crash_give_way_plan`); its value and P&L are the account's like a dip long's, its requirement the fund's margin
  rate (`crash_long_margin_rate` 0.75 — FINRA's 3 x 25% for a 3x fund, initial = maintenance; stamped
  ``crash_margin_rate``; never under Reg T's); de-leveraging does not sell it.
* THE CAPITULATION BOOK (from the 2026-10-21 session, PREREG165 / 167; `src/signals/cap_long.py`): its limit orders
  1/`cap_long_account_slices` (6) of the equity at their limit within the long side's cap and the room at Reg T's
  50%, at most `cap_long_max_open` (3) open, working orders counted (`size_cap`), each fill cut to the room
  (`fill_cap`). Its capital GIVES WAY FIRST — the backtest's quality 0.1, under the dip book's 0.13: the shorts' and
  the crash book's give-way plans sell its latest longs whole before any dip long (`_give_way_longs`); a DIP order is
  sized as if they were sold (`size_long`, the engine's ``fills_make_room``) and its fill sells them, the latest first,
  whole, as far as that lets it buy its quantity (`fill_give_way_plan`); de-leveraging sells them first
  (`delever_plan`); a margin call closes them.
* THE INSIDER BOOK (from the 2026-10-26 session, PREREG203 / 205; `src/signals/insider_long.py`): its market buys at
  the open 1/`insider_long_account_slices` (10) of the equity within the long side's cap and the room at Reg T's 50%,
  at most 1% of the stock's dollar volume, at most `insider_long_max_open` (5) open (`size_insider`). Its capital
  GIVES WAY FIRST — the backtest's quality 0.05, under every other book's: the shorts', the crash book's, a dip or a
  capitulation order's fill short of room sells its latest longs whole before any other long (`_give_way_longs`); the
  dip and capitulation orders are sized as if they were sold (`size_long`, `size_cap`: the engine's
  ``fills_make_room``) and their fills sell them (`fill_give_way_plan`); de-leveraging sells them WITH the capitulation
  book's longs (the engine's ``delever_order`` 0 for both), the latest first, then the dip book's (`delever_plan`); a
  margin call liquidates them like any position.
* THE FEAR BOOK (from the 2026-10-28 session, PREREG219-226; `src/signals/fear_long.py`): its limit orders
  1/`fear_long_account_slices` (6) of the equity at their limit within the long side's cap and the room at Reg T's 50%,
  at most `fear_long_max_open` (4) open, working orders counted (`size_fear`), each fill cut to the room (`fill_cap`).
  Its capital GIVES WAY FIRST — the backtest's quality 0.04, under every other book's: the shorts', the crash book's, a
  dip, capitulation or insider entry's room sells its latest longs whole before any other long (`_give_way_longs`); the
  other books' orders are sized as if they were sold (`_freed`); de-leveraging sells them WITH the capitulation and
  insider books' longs (the engine's ``delever_order`` 0), the latest first; a margin call liquidates them like any
  position. It is ONE BOOK WITH THE DIP BOOK for one book per name (`books.SHARED_NAMES`).

THE GIVE-WAY ORDER is the engine's capital quality (`QUALITY`): the fear book 0.04, the insider book 0.05, the
capitulation book 0.1, the dip book 0.13 — the lowest first, the latest opened first within a book; the crash book (0.2)
and the shorts never give way.

Off (`enable_live_account` False): the two books keep their own accounts (`sim_account`, `dip_long.size`).
"""
from __future__ import annotations

import math
from datetime import datetime, timezone
from typing import Dict, List, Optional, Sequence, Tuple

from config import settings
from src.performance import sim_account

DIP_MECHANISM = "dip_long"
CRASH_MECHANISM = "crash_long"
CAP_MECHANISM = "cap_long"
INS_MECHANISM = "insider_long"
FEAR_MECHANISM = "fear_long"
LONG_INIT, LONG_MAINT = 0.50, 0.25        # Reg T on a long (the engine's Rules.long_init / long_maint)
# the long books' capital qualities (the backtest's `mimic.QUALITY`): a book gives way to every book above it, the
# lowest first; the crash book never gives way (not preemptible)
QUALITY = {"fear": 0.04, "ins": 0.05, "cap": 0.1, "dip": 0.13, "crash": 0.2}


def _crash_rate(trade: Optional[dict] = None) -> float:
    """The crash long's margin rate (initial = maintenance) as a share of its value: the one stamped on the trade,
    else `crash_long_margin_rate` (the engine's ``house_init`` / ``house_maint``: max(Reg T, it))."""
    r = trade.get("crash_margin_rate") if trade is not None else None
    return max(0.0, float(settings.crash_long_margin_rate if r is None else r))


def enabled() -> bool:
    """The shared account is on (it needs the selection short's simulated account)."""
    return sim_account.enabled() and bool(getattr(settings, "enable_live_account", False))


def dip_funded(trade: dict) -> bool:
    return trade.get("entry_mechanism") == DIP_MECHANISM and trade.get("dip_account_shares") is not None


def crash_funded(trade: dict) -> bool:
    return trade.get("entry_mechanism") == CRASH_MECHANISM and trade.get("crash_account_shares") is not None


def cap_funded(trade: dict) -> bool:
    return trade.get("entry_mechanism") == CAP_MECHANISM and trade.get("cap_account_shares") is not None


def ins_funded(trade: dict) -> bool:
    return trade.get("entry_mechanism") == INS_MECHANISM and trade.get("ins_account_shares") is not None


def fear_funded(trade: dict) -> bool:
    return trade.get("entry_mechanism") == FEAR_MECHANISM and trade.get("fear_account_shares") is not None


def long_book(trade: dict) -> Optional[str]:
    """The long book of a funded long ("fear", "ins", "cap", "dip", "crash"), else None."""
    if fear_funded(trade):
        return "fear"
    if ins_funded(trade):
        return "ins"
    if cap_funded(trade):
        return "cap"
    if dip_funded(trade):
        return "dip"
    if crash_funded(trade):
        return "crash"
    return None


def _long_shares(trade: dict) -> Optional[float]:
    """A funded long's share count (a dip long's, the crash book's, the capitulation book's, the insider book's or the
    fear book's), else None."""
    if dip_funded(trade):
        return float(trade["dip_account_shares"])
    if crash_funded(trade):
        return float(trade["crash_account_shares"])
    if cap_funded(trade):
        return float(trade["cap_account_shares"])
    if ins_funded(trade):
        return float(trade["ins_account_shares"])
    if fear_funded(trade):
        return float(trade["fear_account_shares"])
    return None


def _dip_dollars(trade: dict) -> float:
    try:
        return (float(_long_shares(trade) or 0.0) * float(trade.get("entry_price") or 0.0)
                * float(trade.get("return_pct") or 0.0) / 100.0)
    except (TypeError, ValueError):
        return 0.0


def state(trades: Sequence[dict], now: Optional[datetime] = None) -> Dict[str, float]:
    """The shared account: `sim_account.state` (its funded shorts: equity, ``gross`` = the open shorts' value,
    ``maint``, ``init_req``) with the funded longs added — their P&L in the equity and ``realized`` /
    ``unrealized``, their value in ``long_gross``, their requirements in ``maint`` / ``init_req`` (a dip long Reg
    T's, the crash long the larger of Reg T's and its fund's rate); ``long_open`` counts the dip longs,
    ``crash_open`` the crash long, ``cap_open`` the capitulation book's longs (``cap_gross`` their value), ``ins_open``
    the insider book's (``ins_gross``), ``fear_open`` the fear book's (``fear_gross``)."""
    st = dict(sim_account.state(trades, now))
    st.update(long_gross=0.0, long_open=0, long_funded=0, crash_open=0, cap_open=0, cap_gross=0.0, ins_open=0,
              ins_gross=0.0, fear_open=0, fear_gross=0.0)
    for t in trades:
        n = _long_shares(t)
        if n is None or not sim_account.counts(t):    # the fresh start: the account's trades before it are out
            continue
        st["long_funded"] += 1
        pnl = _dip_dollars(t)
        if t.get("status") == "OPEN":
            px = float(t.get("current_price") or t.get("entry_price") or 0.0)
            st["unrealized"] += pnl
            st["long_gross"] += n * px
            if dip_funded(t):
                st["maint"] += LONG_MAINT * n * px
                st["init_req"] += LONG_INIT * n * px
                st["long_open"] += 1                 # the dip book's slots
            elif cap_funded(t):
                st["maint"] += LONG_MAINT * n * px
                st["init_req"] += LONG_INIT * n * px
                st["cap_open"] += 1                  # the capitulation book's slots
                st["cap_gross"] += n * px
            elif ins_funded(t):
                st["maint"] += LONG_MAINT * n * px
                st["init_req"] += LONG_INIT * n * px
                st["ins_open"] += 1                  # the insider book's slots
                st["ins_gross"] += n * px
            elif fear_funded(t):
                st["maint"] += LONG_MAINT * n * px
                st["init_req"] += LONG_INIT * n * px
                st["fear_open"] += 1                 # the fear book's slots
                st["fear_gross"] += n * px
            else:
                r = _crash_rate(t)
                st["maint"] += max(LONG_MAINT, r) * n * px
                st["init_req"] += max(LONG_INIT, r) * n * px
                st["crash_open"] += 1
        else:
            st["realized"] += pnl
    st["equity"] = st["paid_in"] + st["realized"] + st["unrealized"]
    return st


def _long_size(st: Dict[str, float], price: float, slices: float) -> Tuple[int, str]:
    """``(shares, why)`` for a new long at ``price`` from the account state ``st`` — the engine's long sizing: the
    slice (equity / ``slices``) within the long side's cap (the open longs' value at most the equity) and the margin
    room at Reg T's 50% (the entry's commission out of it). ``why`` is "ok", "account_full" (no room / the side full)
    or "account_too_small" (a slice buys no share)."""
    eq, price = st["equity"], float(price)
    room = eq - st["init_req"]
    rate = LONG_INIT * price
    nm = int(math.floor(room / rate)) if room > 0 else 0
    if nm > 0:
        nm = int(math.floor(max(room - sim_account.comm(nm, price), 0.0) / rate))
    alloc = min(eq / slices, eq - st["long_gross"])
    n = min(int(math.floor(alloc / price)) if alloc > 0 else 0, nm)
    if n > 0:
        return n, "ok"
    if nm <= 0 or eq - st["long_gross"] < price:
        return 0, "account_full"
    return 0, "account_too_small"


def size_long(trades: Sequence[dict], price: float, now: Optional[datetime] = None,
              working: int = 0, free_cap: bool = True) -> Tuple[int, str, Dict[str, float]]:
    """``(shares, why, state)`` for a new dip long at ``price`` — the engine's long sizing (`_long_size`): the slice
    (equity / `dip_long_account_slices`) within the long side's cap and the margin room at Reg T's 50%, AS IF the
    lower-quality books' longs (the insider and capitulation books') were sold (they give way to a dip order's fill —
    the engine's ``fills_make_room``: `fill_give_way_plan` sells them when it fills); at most that many dip longs open,
    the book's ``working`` limit orders counted as slots taken (they reserve no money). ``free_cap`` False: from the room
    as it stands (a market buy at the open, the entry mode before the limit orders, sells nothing). ``why`` is "ok",
    "no_slot", "account_full" or "account_too_small"."""
    st = state(trades, now)
    slices = max(1, int(settings.dip_long_account_slices))
    if st["long_open"] + max(0, int(working)) >= slices:
        return 0, "no_slot", st
    if not price or price <= 0:
        return 0, "no_price", st
    n, why = _long_size(_freed(st, trades, QUALITY["dip"]) if free_cap else st, price, slices)
    return n, why, st


def size_cap(trades: Sequence[dict], price: float, now: Optional[datetime] = None,
             working: int = 0) -> Tuple[int, str, Dict[str, float]]:
    """``(shares, why, state)`` for a new capitulation-book order at ``price`` (its limit): the slice (equity /
    `cap_long_account_slices`) within the long side's cap and the margin room at Reg T's 50% (`_long_size`), AS IF the
    insider book's longs were sold (they give way to its fill: the engine's ``fills_make_room``); at most
    `cap_long_max_open` of its longs open, its ``working`` orders counted as slots taken."""
    st = state(trades, now)
    if st["cap_open"] + max(0, int(working)) >= max(1, int(settings.cap_long_max_open)):
        return 0, "no_slot", st
    if not price or price <= 0:
        return 0, "no_price", st
    n, why = _long_size(_freed(st, trades, QUALITY["cap"]), price, max(1.0, float(settings.cap_long_account_slices)))
    return n, why, st


def size_fear(trades: Sequence[dict], price: float, now: Optional[datetime] = None,
              working: int = 0) -> Tuple[int, str, Dict[str, float]]:
    """``(shares, why, state)`` for a new fear-book order at ``price`` (its limit): the slice (equity /
    `fear_long_account_slices`) within the long side's cap and the margin room at Reg T's 50% (`_long_size`) — no book
    gives way to it (its quality is the lowest); at most `fear_long_max_open` of its longs open, its ``working`` orders
    counted as slots taken. ``why`` is "ok", "no_slot", "account_full" or "account_too_small"."""
    st = state(trades, now)
    if st["fear_open"] + max(0, int(working)) >= max(1, int(settings.fear_long_max_open)):
        return 0, "no_slot", st
    if not price or price <= 0:
        return 0, "no_price", st
    n, why = _long_size(st, price, max(1.0, float(settings.fear_long_account_slices)))
    return n, why, st


def size_insider(trades: Sequence[dict], price: float, dv20: Optional[float] = None,
                 now: Optional[datetime] = None) -> Tuple[int, str, Dict[str, float]]:
    """``(shares, why, state)`` for a new insider-book long at ``price`` (a market buy at the open): the slice (equity /
    `insider_long_account_slices`) within the long side's cap and the margin room at Reg T's 50% (`_long_size`), at most
    1% of its 20-session dollar volume ``dv20`` (the engine's ``adv_cap``) — no book gives way to it (its quality is the
    lowest); at most `insider_long_max_open` of its longs open. ``why`` is "ok", "no_slot", "account_full",
    "account_too_small" or "volume_cap"."""
    st = state(trades, now)
    if st["ins_open"] >= max(1, int(settings.insider_long_max_open)):
        return 0, "no_slot", st
    if not price or price <= 0:
        return 0, "no_price", st
    n, why = _long_size(st, price, max(1.0, float(settings.insider_long_account_slices)))
    if n > 0 and dv20 is not None and math.isfinite(float(dv20)) and float(dv20) > 0:
        cap_n = int(math.floor(0.01 * float(dv20) / float(price)))
        if cap_n < n:
            n = cap_n
            why = "ok" if n > 0 else "volume_cap"
    return n, why, st


def _fill_room(st: Dict[str, float], price: float) -> int:
    """The shares a limit order's fill may buy at ``price`` from the state ``st``: what the margin room at Reg T's 50%
    funds (its commission out of it) within the long side's cap — no slice (the order's quantity was fixed when it
    was placed)."""
    eq, price = st["equity"], float(price)
    room = eq - st["init_req"]
    rate = LONG_INIT * price
    nm = int(math.floor(room / rate)) if room > 0 else 0
    if nm > 0:
        nm = int(math.floor(max(room - sim_account.comm(nm, price), 0.0) / rate))
    side = eq - st["long_gross"]
    ns = int(math.floor(side / price)) if side > 0 else 0
    return max(0, min(nm, ns))


def fill_cap(trades: Sequence[dict], price: float, now: Optional[datetime] = None,
             free_cap: bool = False, book: str = "dip") -> Tuple[int, Dict[str, float]]:
    """The shares a limit order may buy NOW at ``price`` (PREREG61 amendment 1, the engine's fill rule): what
    the margin room at Reg T's 50% funds (its commission out of it) within the long side's cap (the open longs'
    value at most the equity) — no slice: the order's own quantity was fixed when it was placed. ``free_cap``: as if
    the longs of the books under ``book``'s quality were sold (an order's resting size at IBKR — a dip order frees the
    insider and capitulation books', a capitulation order the insider book's: its fill sells them,
    `fill_give_way_plan`)."""
    st = state(trades, now)
    if not price or price <= 0:
        return 0, st
    return _fill_room(_freed(st, trades, QUALITY[book]) if free_cap else st, price), st


def margin_call(trades: Sequence[dict], now: Optional[datetime] = None) -> Tuple[bool, Dict[str, float]]:
    """True when the shared account's equity is below the maintenance requirement of what it holds."""
    st = state(trades, now)
    return bool((st["open"] or st["long_open"] or st["crash_open"] or st["cap_open"] or st["ins_open"]
                 or st["fear_open"]) and st["equity"] < st["maint"]), st


# ── IBKR's LIQUIDATION (PREREG174 / 175; the user 2026-10-21: "production and backtest needs to liquidate like IBKR
# does") — the account engine's `Rules(liquidation="ibkr", liq_fill="path")` (`src/backtest/account.py`), pure ──────

def funded_open(trades: Sequence[dict]) -> List[dict]:
    """Every OPEN position the account funded (its shorts and its longs)."""
    return [t for t in trades if t.get("status") == "OPEN"
            and (sim_account.funded(t) or _long_shares(t) is not None) and sim_account.counts(t)]


def shares_of(trade: dict) -> float:
    """The shares the account holds in a funded trade (a short's ``sel_account_shares`` or a long's)."""
    if sim_account.funded(trade):
        return float(trade["sel_account_shares"])
    return float(_long_shares(trade) or 0.0)


def side_of(trade: dict) -> int:
    return -1 if sim_account.funded(trade) else 1


def maint_of(trade: dict, n: float, px: float) -> float:
    """The maintenance requirement of ``n`` shares of the funded trade at ``px`` — `state`'s rules (a short: the larger
    of Reg T's tiers and its house rate; a dip, capitulation or insider long Reg T's 25%; the crash long the larger of it
    and the fund's rate)."""
    if sim_account.funded(trade):
        hm, _ = sim_account.trade_rates(trade)
        return max(sim_account.maint(n, px), hm * n * px)
    if crash_funded(trade):
        return max(LONG_MAINT, _crash_rate(trade)) * n * px
    return LONG_MAINT * n * px


def regt_of(trade: dict, n: float, px: float) -> float:
    """Reg T's initial requirement, no house rate: the floor of IBKR's end-of-day SMA check (the engine's `regt_req`)."""
    if sim_account.funded(trade):
        return max(0.5 * n * px, sim_account.maint(n, px))
    return LONG_INIT * n * px


def plan_liquidation(positions: Sequence[dict], equity: float, kind: str = "maint", lot: int = 100,
                     slip: float = 0.10) -> List[dict]:
    """IBKR's liquidation of the account (PREREG174's M_PATH, the engine's `liquidate_path` / `liquidate`, ported for
    the ledger's linear account): ``positions`` = [{"key", "side" (+1 long / -1 short), "n" (shares), "mark" (the
    current price), "prev" (the price at the previous check: the path's start), "ext" (the adverse extreme since then:
    the path's end), "order" (sortable: the most recently opened LAST), "whole" (True: never split), "maint" / "regt"
    (callables (n, px) -> the requirement)}]; ``equity`` = the account's equity with every position at its ``mark``.

    ``kind`` "maint": every position moves together from ``prev`` to ``ext`` (lambda 0 -> 1); at each first lambda where
    equity falls under the maintenance requirement the most recently opened positions are liquidated in lots of
    ``lot`` shares (a position of ``lot`` or fewer, or a ``whole`` one, whole) — the fewest that bring equity back to
    the requirement there —, each lot filled ``slip`` against the account from its price there, never past ``ext``;
    then the path goes on. "regt": the same lots against Reg T's initial requirement at the marks, filled there (the
    end-of-day check from 15:50 ET). Returns the lots in order: {"key", "shares", "held", "fill", "at" (the price where
    the deficit appeared), "kind", "closed", "equity", "need"}."""
    C = float(equity) - sum(p["side"] * p["n"] * p["mark"] for p in positions)
    P = {p["key"]: dict(p) for p in positions}
    order = sorted(P, key=lambda k: P[k]["order"], reverse=True)          # the most recently opened first
    lots: List[dict] = []
    req_key = "maint" if kind == "maint" else "regt"

    def eq_need(px):
        e, r = C, 0.0
        for k, p in P.items():
            if p["n"] > 0:
                e += p["side"] * p["n"] * px[k]
                r += p[req_key](p["n"], px[k])
        return e, r

    def liquidate(px, fill):
        nonlocal C
        eq, need = eq_need(px)
        for k in order:
            if eq >= need:
                break
            p = P[k]
            n = p["n"]
            if n <= 0:
                continue
            pw, pf, s = px[k], fill[k], p["side"]
            rr = p[req_key](n, pw) / n                     # the requirement each share releases (linear at one price)

            def effect(q):                                  # equity change of liquidating q shares filled at pf
                return s * q * (pf - pw)

            def cures(q):
                return eq + effect(q) >= need - q * rr
            if n <= lot or p.get("whole") or not cures(n):
                q = n
            else:
                per = (effect(n) + n * rr) / n
                q = n if per <= 0 else min(n, max(lot, int(math.ceil((need - eq) / per / lot)) * lot))
                while q > lot and cures(q - lot):
                    q -= lot
                while q < n and not cures(q):
                    q = min(n, q + lot)
            lots.append({"key": k, "shares": q, "held": n, "fill": float(pf), "at": float(pw), "kind": kind,
                         "closed": bool(q >= n), "equity": float(eq), "need": float(need)})
            C += s * q * pf
            p["n"] = n - q if q < n else 0.0
            eq, need = eq_need(px)

    if kind != "maint":
        px = {k: P[k]["mark"] for k in P}
        liquidate(px, px)
        return lots

    def at(lam):
        return {k: P[k]["prev"] + lam * (P[k]["ext"] - P[k]["prev"]) for k in P}

    def short_by(lam):
        e, r = eq_need(at(lam))
        return r - e

    lam = 0.0
    while any(p["n"] > 0 for p in P.values()) and short_by(1.0) > 0:
        if short_by(lam) > 0:
            lc = lam
        else:
            lo_, hi_ = lam, 1.0
            for _ in range(60):
                mid = 0.5 * (lo_ + hi_)
                if short_by(mid) > 0:
                    hi_ = mid
                else:
                    lo_ = mid
            lc = hi_
        x = at(lc)
        fill = {k: (min(P[k]["ext"], x[k] * (1.0 + slip)) if P[k]["side"] < 0 else max(P[k]["ext"], x[k] * (1.0 - slip)))
                for k in x}
        liquidate(x, fill)
        lam = lc
    return lots


def _latest_first(lst: List[dict]) -> List[dict]:
    return sorted(lst, key=lambda t: (str(t.get("entry_datetime") or ""), str(t.get("recommendation_id") or "")),
                  reverse=True)


def _open_dips_latest_first(trades: Sequence[dict]) -> List[dict]:
    return _latest_first([t for t in trades if t.get("status") == "OPEN" and dip_funded(t)])


def _open_caps_latest_first(trades: Sequence[dict]) -> List[dict]:
    return _latest_first([t for t in trades if t.get("status") == "OPEN" and cap_funded(t) and sim_account.counts(t)])


def _open_ins_latest_first(trades: Sequence[dict]) -> List[dict]:
    return _latest_first([t for t in trades if t.get("status") == "OPEN" and ins_funded(t) and sim_account.counts(t)])


def _open_fear_latest_first(trades: Sequence[dict]) -> List[dict]:
    return _latest_first([t for t in trades if t.get("status") == "OPEN" and fear_funded(t) and sim_account.counts(t)])


def _give_way_longs(trades: Sequence[dict], below: Optional[float] = None) -> List[dict]:
    """The longs that give way, in the order they are sold: the fear book's (the engine's quality 0.04), the insider
    book's (0.05), the capitulation book's (0.1), then the dip book's (0.13), the latest first within each — the
    engine's `make_room` order (the lowest quality first, the latest opened first within it); ``below``: only the books
    whose quality is under it (a fill makes room only from the books under its own)."""
    out: List[dict] = []
    for book, lst in (("fear", _open_fear_latest_first(trades)), ("ins", _open_ins_latest_first(trades)),
                      ("cap", _open_caps_latest_first(trades)), ("dip", _open_dips_latest_first(trades))):
        if below is None or QUALITY[book] < below:
            out += lst
    return out


def _delever_longs(trades: Sequence[dict]) -> List[dict]:
    """The longs de-leveraging sells, in order: the fear, insider and capitulation books' together (the engine's
    ``delever_order`` 0 for the three), the latest opened first, then the dip book's (1), the latest first."""
    first = _latest_first(_open_fear_latest_first(trades) + _open_ins_latest_first(trades)
                          + _open_caps_latest_first(trades))
    return first + _open_dips_latest_first(trades)


def _after_selling(st: Dict[str, float], t: dict) -> Dict[str, float]:
    """The account once long ``t`` (a dip, capitulation or insider long) is sold at its mark (its P&L realized; the
    exit's costs left to the ledger)."""
    n = float(_long_shares(t) or 0.0)
    px = float(t.get("current_price") or t.get("entry_price") or 0.0)
    out = dict(st)
    out["long_gross"] -= n * px
    out["maint"] -= LONG_MAINT * n * px
    out["init_req"] -= LONG_INIT * n * px
    if cap_funded(t):
        out["cap_open"] -= 1
        out["cap_gross"] -= n * px
    elif ins_funded(t):
        out["ins_open"] = out.get("ins_open", 0) - 1
        out["ins_gross"] = out.get("ins_gross", 0.0) - n * px
    elif fear_funded(t):
        out["fear_open"] = out.get("fear_open", 0) - 1
        out["fear_gross"] = out.get("fear_gross", 0.0) - n * px
    else:
        out["long_open"] -= 1
    return out


def _freed(st: Dict[str, float], trades: Sequence[dict], below: float = QUALITY["dip"]) -> Dict[str, float]:
    """The account were every long of the books under quality ``below`` sold (equity unchanged: the engine's
    `freed_size`) — for a dip order the insider and capitulation books', for a capitulation order the insider book's."""
    cur = st
    for t in _give_way_longs(trades, below=below):
        cur = _after_selling(cur, t)
    return cur


def delever_plan(trades: Sequence[dict], now: Optional[datetime] = None) -> Tuple[List[dict], Dict[str, float]]:
    """The longs to sell now, whole — the insider and capitulation books' first (the engine's ``delever_order`` 0 for
    both, the latest opened first across them), then the dip book's (1), the latest first: none unless the maintenance
    requirement exceeds `live_account_delever_trigger` x equity; then until it is at most
    `live_account_delever_target` x equity."""
    st = state(trades, now)
    eq = st["equity"]
    trig = float(getattr(settings, "live_account_delever_trigger", 0.8))
    tgt = float(getattr(settings, "live_account_delever_target", 0.6))
    if eq <= 0 or st["maint"] <= trig * eq:
        return [], st
    out, cur = [], st
    for t in _delever_longs(trades):
        if cur["maint"] <= tgt * eq:
            break
        out.append(t)
        cur = _after_selling(cur, t)
    return out, st


def give_way_plan(trades: Sequence[dict], price: float, dv20: Optional[float] = None,
                  now: Optional[datetime] = None,
                  rates: Optional[Tuple[float, float]] = None,
                  slices: Optional[float] = None) -> Tuple[List[dict], int, Dict[str, float]]:
    """The longs to sell, whole — the insider book's, the capitulation book's, then the dip book's, the latest first
    within each (`_give_way_longs`) — so that a new short at ``price`` gets the shares its slice and the side's cap allow
    (``want``) — none when the margin room already funds them, and none when selling them would not let the short
    open more shares than it gets now (its own book's initial requirement past the equity: the longs would be sold for
    nothing — PREREG64). Returns (the longs, want, the state before)."""
    st = state(trades, now)
    want = sim_account.size(trades, price, dv20, now, rates=rates, st=dict(st, init_req=0.0), slices=slices)[0]
    if want <= 0:
        return [], 0, st
    have = sim_account.size(trades, price, dv20, now, rates=rates, st=st, slices=slices)[0]
    if have >= want:
        return [], want, st
    out, cur = [], st
    for t in _give_way_longs(trades):
        out.append(t)
        cur = _after_selling(cur, t)
        if sim_account.size(trades, price, dv20, now, rates=rates, st=cur, slices=slices)[0] >= want:
            break
    if not out or sim_account.size(trades, price, dv20, now, rates=rates, st=cur, slices=slices)[0] <= have:
        return [], want, st                        # the sales would buy nothing: the long books keep their longs
    return out, want, st


def fill_give_way_plan(trades: Sequence[dict], price: float, want: int,
                       now: Optional[datetime] = None, book: str = "dip") -> Tuple[List[dict], Dict[str, float]]:
    """A limit order's fill (the engine's ``fills_make_room``; ``book`` the order's: "dip" or "cap"): the longs of the
    books under its quality to sell — the insider book's, then (for a dip order) the capitulation book's, the latest
    first within each, whole — until the fill can buy its order's ``want`` shares at ``price``; none when the room
    already lets it, and none when selling them all would not let it buy more than it buys now (the engine's `helps`).
    Returns (the longs, the state before)."""
    st = state(trades, now)
    q = QUALITY[book]
    cands = _give_way_longs(trades, below=q)
    if not cands or not price or price <= 0 or want <= 0:
        return [], st
    have = _fill_room(st, price)
    if have >= want or _fill_room(_freed(st, trades, q), price) <= have:
        return [], st
    out, cur = [], st
    for t in cands:
        out.append(t)
        cur = _after_selling(cur, t)
        if _fill_room(cur, price) >= want:
            break
    return out, st



def size_crash(trades: Sequence[dict], price: float, now: Optional[datetime] = None,
               st: Optional[Dict[str, float]] = None) -> Tuple[int, str, Dict[str, float]]:
    """``(shares, why, state)`` for the crash book's long at ``price`` (the traded fund's) — the engine's sizing of
    its book: the slice (equity / `crash_long_account_slices`) within the long side's cap (the open longs' value at
    most the equity) and the margin room at the larger of Reg T's 50% and the fund's rate (`crash_long_margin_rate`;
    the entry's commission out of it); one open at a time. ``why`` is "ok", "no_slot", "account_full" or
    "account_too_small"."""
    st = state(trades, now) if st is None else st
    if st.get("crash_open", 0) >= 1:
        return 0, "no_slot", st
    if not price or price <= 0:
        return 0, "no_price", st
    eq, price = st["equity"], float(price)
    slices = max(1.0, float(settings.crash_long_account_slices))
    room = eq - st["init_req"]
    rate = max(LONG_INIT, _crash_rate()) * price
    nm = int(math.floor(room / rate)) if room > 0 else 0
    if nm > 0:
        nm = int(math.floor(max(room - sim_account.comm(nm, price), 0.0) / rate))
    alloc = min(eq / slices, eq - st["long_gross"])
    n = min(int(math.floor(alloc / price)) if alloc > 0 else 0, nm)
    if n > 0:
        return n, "ok", st
    if nm <= 0 or eq - st["long_gross"] < price:
        return 0, "account_full", st
    return 0, "account_too_small", st


def crash_give_way_plan(trades: Sequence[dict], price: float,
                        now: Optional[datetime] = None) -> Tuple[List[dict], int, Dict[str, float]]:
    """The longs to sell, whole — the insider book's, the capitulation book's, then the dip book's, the latest first
    within each (`_give_way_longs`) — so the crash book's long at ``price`` gets its slice (``want``: its size were
    every one of them freed) — none when it already gets it, and none when selling them would not let it buy more than
    it buys now (PREREG64's rule; the backtest's crash book of quality 0.2 above the dip book's 0.13, the capitulation
    book's 0.1 and the insider book's 0.05). Returns (the longs, want, the state before)."""
    st = state(trades, now)
    dips = _give_way_longs(trades)
    free = st
    for t in dips:
        free = _after_selling(free, t)
    want = size_crash(trades, price, now, st=free)[0]
    have = size_crash(trades, price, now, st=st)[0]
    if want <= 0 or have >= want:
        return [], want, st
    out, cur = [], st
    for t in dips:
        out.append(t)
        cur = _after_selling(cur, t)
        if size_crash(trades, price, now, st=cur)[0] >= want:
            break
    if not out or size_crash(trades, price, now, st=cur)[0] <= have:
        return [], want, st
    return out, want, st

def summary(st: Dict[str, float]) -> str:
    return (f"live account equity ${st['equity']:,.0f} (paid in ${st['paid_in']:,.0f}, realized {st['realized']:+,.0f}, "
            f"open {st['unrealized']:+,.0f}): {int(st['open'])} short(s) ${st['gross']:,.0f}, "
            f"{int(st['long_open'])} dip long(s) + {int(st.get('crash_open', 0))} crash long + "
            f"{int(st.get('cap_open', 0))} capitulation long(s) + {int(st.get('ins_open', 0))} insider long(s) + "
            f"{int(st.get('fear_open', 0))} fear long(s) "
            f"${st['long_gross']:,.0f}, "
            f"maintenance ${st['maint']:,.0f}, "
            f"initial ${st['init_req']:,.0f}")
