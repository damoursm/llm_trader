"""The SIMULATED ACCOUNT the vol arm is sized from (user directive 2026-10-05: "Have the account based
sizing considering the simulated 5000$+1000$ every two weeks").

The live book trades a large paper account; the plan is a real account fed $5,000 at the start plus
$1,000 every 14 calendar days. Each new vol-arm short is sized from THAT account, replayed from the
ledger with the audited replay engine's rules (the compounding studies' `cap5k6`,
`memory/metrics-growth-and-return-per-day.md`):

* equity = the money paid in by now + every funded trade's dollar P&L: shares x entry price x the
  ledger's net return (spreads, commissions and borrow included; an open trade at its live mark);
* a new short = floor(min(equity / slices, equity - the open shorts' market value, 1% of the stock's
  20-session dollar volume) / price) whole shares, never more than the Reg T initial-margin room
  allows, none while equity is under FINRA's $2,000 minimum;
* a margin call = equity below the Reg T maintenance of the open shorts (30% or $5 a share at $5 and
  above; 100% or $2.50 a share below): every funded short is bought back, as the broker would.

The slices (10) and the vol arm's squeeze cover (6x) were tuned together for no margin call
(`memory/vol-arm-margin-protection-2026-10.md`). The account funds the VOL arm only — the arm the
account studies measured; the model and ETF arms keep the flat order size.
"""
from __future__ import annotations

import math
from datetime import date, datetime, timezone
from typing import Dict, Optional, Sequence, Tuple

from config import settings
from src.utils import ET

FUNDED_ARMS = ("vol",)
SEC_FEE, TAF_PER_SHARE, TAF_MAX = 27.80e-6, 0.000166, 8.30        # the engine's entry fees (cap5k4)


def enabled() -> bool:
    return bool(getattr(settings, "enable_sel_short_account_sizing", False))


def arm_funded(arm: Optional[str]) -> bool:
    """A pick of this arm is sized from the simulated account."""
    return enabled() and str(arm or "") in FUNDED_ARMS


def maint(n: float, p: float) -> float:
    """Reg T maintenance for a short of ``n`` shares at ``p`` (the engine's tiers)."""
    return max(5.0 * n, 0.30 * n * p) if p >= 5.0 else max(2.5 * n, n * p)


def init_rate(p: float) -> float:
    """Initial margin per share shorted at ``p``."""
    return max(0.5 * p, max(5.0, 0.30 * p) if p >= 5.0 else max(2.5, p))


def comm(n: float, p: float) -> float:
    """IBKR's fixed commission: max($1, $0.005 a share), at most 1% of the order's value."""
    return min(max(1.0, 0.005 * n), 0.01 * n * p)


def start_date() -> date:
    return date.fromisoformat(str(settings.sel_short_account_start)[:10])


def paid_in(now: datetime) -> float:
    """The start balance plus every deposit due on or before ``now``'s ET date (the first
    `sel_short_account_deposit_days` after the start)."""
    d0, today = start_date(), now.astimezone(ET).date()
    every = max(1, int(settings.sel_short_account_deposit_days))
    n_dep = max(0, (today - d0).days // every)
    return float(settings.sel_short_account_initial) + n_dep * float(settings.sel_short_account_deposit)


def funded(trade: dict) -> bool:
    """A selection-short trade the simulated account opened (it carries the account's share count)."""
    return trade.get("entry_mechanism") == "sel_short" and trade.get("sel_account_shares") is not None


def dollars(trade: dict) -> float:
    """The trade's dollar P&L in the account: shares x entry x the ledger's net return."""
    try:
        return (float(trade["sel_account_shares"]) * float(trade.get("entry_price") or 0.0)
                * float(trade.get("return_pct") or 0.0) / 100.0)
    except (TypeError, ValueError):
        return 0.0


def state(trades: Sequence[dict], now: Optional[datetime] = None) -> Dict[str, float]:
    """The account at ``now``: equity, paid in, realized / unrealized P&L, the open shorts' market
    value, maintenance and initial-margin requirements."""
    now = now or datetime.now(timezone.utc)
    out = {"paid_in": paid_in(now), "realized": 0.0, "unrealized": 0.0, "gross": 0.0, "maint": 0.0,
           "init_req": 0.0, "open": 0, "funded": 0}
    for t in trades:
        if not funded(t):
            continue
        out["funded"] += 1
        pnl = dollars(t)
        if t.get("status") == "OPEN":
            n = float(t["sel_account_shares"])
            px = float(t.get("current_price") or t.get("entry_price") or 0.0)
            out["unrealized"] += pnl
            out["gross"] += n * px
            m = maint(n, px)
            out["maint"] += m
            out["init_req"] += max(0.5 * n * px, m)
            out["open"] += 1
        else:
            out["realized"] += pnl
    out["equity"] = out["paid_in"] + out["realized"] + out["unrealized"]
    return out


def size(trades: Sequence[dict], price: float, dv20: Optional[float] = None,
         now: Optional[datetime] = None) -> Tuple[int, str, Dict[str, float]]:
    """``(shares, why, account state)`` for a new short at ``price``: the engine's sizing rule.
    ``why`` is "ok" or the reason no share fits."""
    st = state(trades, now)
    eq = st["equity"]
    if eq < float(settings.sel_short_account_min_equity):
        return 0, "account_below_minimum", st
    if not price or price <= 0:
        return 0, "no_price", st
    room = eq - st["init_req"]
    nm = int(math.floor(room / init_rate(price))) if room > 0 else 0
    if nm > 0:                                     # the entry's own fees out of the margin room
        c0 = comm(nm, price) + SEC_FEE * nm * price + min(TAF_PER_SHARE * nm, TAF_MAX)
        nm = int(math.floor(max(room - c0, 0.0) / init_rate(price)))
    alloc = min(eq / max(1, int(settings.sel_short_account_slices)), eq - st["gross"])
    liq = float(settings.sel_short_account_max_dollar_volume_share)
    vol_cap = None
    if dv20 is not None and math.isfinite(float(dv20)) and float(dv20) > 0 and liq > 0:
        vol_cap = liq * float(dv20)
        alloc = min(alloc, vol_cap)
    n = min(int(math.floor(alloc / price)) if alloc > 0 else 0, nm)
    if n > 0:
        return n, "ok", st
    if nm <= 0 or eq - st["gross"] < price:          # no room left: the open shorts or the margin
        return 0, "account_full", st
    if vol_cap is not None and vol_cap < price:      # 1% of the stock's dollar volume buys no share
        return 0, "volume_cap", st
    return 0, "account_too_small", st                # a slice buys no share


def margin_call(trades: Sequence[dict], now: Optional[datetime] = None) -> Tuple[bool, Dict[str, float]]:
    """True when the account's equity is below the maintenance of its open shorts."""
    st = state(trades, now)
    return bool(st["open"] and st["equity"] < st["maint"]), st


def summary(st: Dict[str, float]) -> str:
    return (f"equity ${st['equity']:,.0f} (paid in ${st['paid_in']:,.0f}, realized {st['realized']:+,.0f}, open "
            f"{st['unrealized']:+,.0f}), {int(st['open'])} open short(s) worth ${st['gross']:,.0f}, maintenance "
            f"${st['maint']:,.0f}")
