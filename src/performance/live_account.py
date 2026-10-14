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
  each dip long 1/`dip_long_account_slices` (10) of the equity, at most that many open, the longs' value at most the
  equity, within the margin room (`size_long`).
* THE DIP BOOK GIVES WAY — a short the margin room cannot fund in full sells the dip book's longs, the latest first,
  WHOLE, until it can (`give_way_plan`) — none when selling them all would not let it open one more share (PREREG64:
  the shorts' own initial requirement past the equity used to cost every long for nothing); a long given up is never
  bought back. (The engine's "just enough shares"
  variant measured the same on this combination: +78.9 vs +77.0 %/yr mean over the starts, +56.1 vs +55.4 from 2021,
  no margin call either way — whole positions keep one ledger trade per broker position.)
* DE-LEVERAGING — at a tick where the maintenance requirement exceeds `live_account_delever_trigger` (0.8) x equity,
  the dip book's longs are sold, the latest first, whole, until it is at most `live_account_delever_target` (0.6) x
  equity; never bought back (`delever_plan`).
* MARGIN CALL — equity below the maintenance requirement: every short and every long the account funded is closed
  (`margin_call`).
* THE CRASH-REBOUND BOOK (2026-10-11, PREREG86 / PREREG88; `src/signals/crash_long.py`): one long at a time in
  the traded fund (TQQQ, on QQQ's decisions), 1/`crash_long_account_slices` (4) of the equity within the long side's
  cap and the room (`size_crash`); short of room it sells the dip book's latest longs whole, as a short does
  (`crash_give_way_plan`); its value and P&L are the account's like a dip long's, its requirement the fund's margin
  rate (`crash_long_margin_rate` 0.75 — FINRA's 3 x 25% for a 3x fund, initial = maintenance; stamped
  ``crash_margin_rate``; never under Reg T's); de-leveraging does not sell it.

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
LONG_INIT, LONG_MAINT = 0.50, 0.25        # Reg T on a long (the engine's Rules.long_init / long_maint)


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


def _long_shares(trade: dict) -> Optional[float]:
    """A funded long's share count (a dip long's or the crash book's), else None."""
    if dip_funded(trade):
        return float(trade["dip_account_shares"])
    if crash_funded(trade):
        return float(trade["crash_account_shares"])
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
    ``crash_open`` the crash long."""
    st = dict(sim_account.state(trades, now))
    st.update(long_gross=0.0, long_open=0, long_funded=0, crash_open=0)
    for t in trades:
        n = _long_shares(t)
        if n is None:
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
            else:
                r = _crash_rate(t)
                st["maint"] += max(LONG_MAINT, r) * n * px
                st["init_req"] += max(LONG_INIT, r) * n * px
                st["crash_open"] += 1
        else:
            st["realized"] += pnl
    st["equity"] = st["paid_in"] + st["realized"] + st["unrealized"]
    return st


def size_long(trades: Sequence[dict], price: float, now: Optional[datetime] = None,
              working: int = 0) -> Tuple[int, str, Dict[str, float]]:
    """``(shares, why, state)`` for a new dip long at ``price`` — the engine's long sizing: the slice
    (equity / `dip_long_account_slices`) within the long side's cap (the open longs' value at most the equity) and
    the margin room at Reg T's 50% (the entry's commission out of it); at most that many longs open, the book's
    ``working`` limit orders counted as slots taken (they reserve no money). ``why`` is "ok", "no_slot",
    "account_full" (no room / the side full) or "account_too_small" (a slice buys no share)."""
    st = state(trades, now)
    slices = max(1, int(settings.dip_long_account_slices))
    if st["long_open"] + max(0, int(working)) >= slices:
        return 0, "no_slot", st
    if not price or price <= 0:
        return 0, "no_price", st
    eq, price = st["equity"], float(price)
    room = eq - st["init_req"]
    rate = LONG_INIT * price
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


def fill_cap(trades: Sequence[dict], price: float, now: Optional[datetime] = None) -> Tuple[int, Dict[str, float]]:
    """The shares a dip limit order may buy NOW at ``price`` (PREREG61 amendment 1, the engine's fill rule): what
    the margin room at Reg T's 50% funds (its commission out of it) within the long side's cap (the open longs'
    value at most the equity) — no slice: the order's own quantity was fixed when it was placed."""
    st = state(trades, now)
    if not price or price <= 0:
        return 0, st
    eq, price = st["equity"], float(price)
    room = eq - st["init_req"]
    rate = LONG_INIT * price
    nm = int(math.floor(room / rate)) if room > 0 else 0
    if nm > 0:
        nm = int(math.floor(max(room - sim_account.comm(nm, price), 0.0) / rate))
    side = eq - st["long_gross"]
    ns = int(math.floor(side / price)) if side > 0 else 0
    return max(0, min(nm, ns)), st


def margin_call(trades: Sequence[dict], now: Optional[datetime] = None) -> Tuple[bool, Dict[str, float]]:
    """True when the shared account's equity is below the maintenance requirement of what it holds."""
    st = state(trades, now)
    return bool((st["open"] or st["long_open"] or st["crash_open"]) and st["equity"] < st["maint"]), st


def _open_dips_latest_first(trades: Sequence[dict]) -> List[dict]:
    dips = [t for t in trades if t.get("status") == "OPEN" and dip_funded(t)]
    return sorted(dips, key=lambda t: (str(t.get("entry_datetime") or ""), str(t.get("recommendation_id") or "")),
                  reverse=True)


def _after_selling(st: Dict[str, float], t: dict) -> Dict[str, float]:
    """The account once dip long ``t`` is sold at its mark (its P&L realized; the exit's costs left to the ledger)."""
    n = float(t["dip_account_shares"])
    px = float(t.get("current_price") or t.get("entry_price") or 0.0)
    out = dict(st)
    out["long_gross"] -= n * px
    out["maint"] -= LONG_MAINT * n * px
    out["init_req"] -= LONG_INIT * n * px
    out["long_open"] -= 1
    return out


def delever_plan(trades: Sequence[dict], now: Optional[datetime] = None) -> Tuple[List[dict], Dict[str, float]]:
    """The dip longs to sell now, the latest first, whole: none unless the maintenance requirement exceeds
    `live_account_delever_trigger` x equity; then until it is at most `live_account_delever_target` x equity."""
    st = state(trades, now)
    eq = st["equity"]
    trig = float(getattr(settings, "live_account_delever_trigger", 0.8))
    tgt = float(getattr(settings, "live_account_delever_target", 0.6))
    if eq <= 0 or st["maint"] <= trig * eq:
        return [], st
    out, cur = [], st
    for t in _open_dips_latest_first(trades):
        if cur["maint"] <= tgt * eq:
            break
        out.append(t)
        cur = _after_selling(cur, t)
    return out, st


def give_way_plan(trades: Sequence[dict], price: float, dv20: Optional[float] = None,
                  now: Optional[datetime] = None,
                  rates: Optional[Tuple[float, float]] = None,
                  slices: Optional[float] = None) -> Tuple[List[dict], int, Dict[str, float]]:
    """The dip longs to sell, the latest first, whole, so that a new short at ``price`` gets the shares its slice
    and the side's cap allow (``want``) — none when the margin room already funds them, and none when selling them
    would not let the short open more shares than it gets now (its own book's initial requirement past the equity:
    the longs would be sold for nothing — PREREG64). Returns (the longs, want, the state before)."""
    st = state(trades, now)
    want = sim_account.size(trades, price, dv20, now, rates=rates, st=dict(st, init_req=0.0), slices=slices)[0]
    if want <= 0:
        return [], 0, st
    have = sim_account.size(trades, price, dv20, now, rates=rates, st=st, slices=slices)[0]
    if have >= want:
        return [], want, st
    out, cur = [], st
    for t in _open_dips_latest_first(trades):
        out.append(t)
        cur = _after_selling(cur, t)
        if sim_account.size(trades, price, dv20, now, rates=rates, st=cur, slices=slices)[0] >= want:
            break
    if not out or sim_account.size(trades, price, dv20, now, rates=rates, st=cur, slices=slices)[0] <= have:
        return [], want, st                        # the sales would buy nothing: the dip book keeps its longs
    return out, want, st



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
    """The dip longs to sell, the latest first, whole, so the crash book's long at ``price`` gets its slice
    (``want``: its size were every dip long freed) — none when it already gets it, and none when selling them would
    not let it buy more than it buys now (PREREG64's rule; the backtest's crash book of quality 0.2 above the dip
    book's 0.13). Returns (the longs, want, the state before)."""
    st = state(trades, now)
    dips = _open_dips_latest_first(trades)
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
            f"{int(st['long_open'])} dip long(s) + {int(st.get('crash_open', 0))} crash long ${st['long_gross']:,.0f}, "
            f"maintenance ${st['maint']:,.0f}, "
            f"initial ${st['init_req']:,.0f}")
