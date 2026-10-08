"""The SIMULATED ACCOUNT every arm's shorts are sized from (user directives 2026-10-05: "Have the account
based sizing considering the simulated 5000$+1000$ every two weeks", then 2026-10-07: "Change the
$5000+$1000/14days to $10000. We'll use this number from now on.").

The live book trades a large paper account; the plan is a real account of $10,000, no deposits. Each
new short is sized from THAT account, replayed from the ledger
with the audited replay engine's rules (the compounding studies' `cap5k6` / `cap5k7`,
`memory/metrics-growth-and-return-per-day.md`):

* equity = the $10,000 start balance + every funded trade's dollar P&L: shares x entry price x the
  ledger's net return (spreads, commissions and borrow included; an open trade at its live mark);
* a new short = floor(min(equity / slices, equity - the open shorts' market value, 1% of the stock's
  20-session dollar volume) / price) whole shares, never more than the initial-margin room allows,
  none while equity is under FINRA's $2,000 minimum;
* a margin call = equity below the maintenance of the open shorts: every funded short is bought
  back, as the broker would;
* the margin is IBKR's: the larger of Reg T's tiers (maintenance 30% or $5 a share at $5 and above,
  100% or $2.50 a share below) and IBKR's HOUSE rate, a multiple of the short's value. A short carries
  the rates IBKR's what-if quoted at its entry (`ibkr_margin`, stamped `sel_house_maint` /
  `sel_house_init`), else the defaults `sel_short_account_house_maint` / `_init` (2.00 / 2.86: the
  median IBKR quoted on 2026-10-05 for the volatile names the arms short,
  `memory/ibkr-house-margin-2026-10.md`). A short whose initial rate is above the default is shrunk
  by default / rate, so no short ties up more initial margin than one at the default rate; IBKR's
  refusal of any opening short skips the pick.

The 24 slices and the 6x squeeze cover were tuned together on the vol arm under IBKR's house margin
for no margin call (PREREG12); the per-name rates and the shrink were found after it (exploratory:
no margin call in 200 draws of the measured rates, a call in every draw without the shrink). The
account funds EVERY arm (user directive 2026-10-05): the model, vol and ETF arms share its equity
and its room; the model and ETF arms were not part of the tuning.
"""
from __future__ import annotations

import math
from datetime import datetime, timezone
from typing import Dict, Optional, Sequence, Tuple

from loguru import logger

from config import settings

FUNDED_ARMS = ("model", "vol", "etf", "thin", "vol2")
SEC_FEE, TAF_PER_SHARE, TAF_MAX = 27.80e-6, 0.000166, 8.30        # the engine's entry fees (cap5k4)
WHATIF_NOMINAL_USD = 500.0   # the what-if's size: IBKR's rate is a share of the value (measured at ~$500)


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


def default_rates() -> Tuple[float, float]:
    """The default house maintenance and initial requirement, multiples of a short's value (0 = Reg T
    only): what a short without IBKR's quote carries, and the initial rate is the shrink's budget."""
    return (max(0.0, float(getattr(settings, "sel_short_account_house_maint", 0.0) or 0.0)),
            max(0.0, float(getattr(settings, "sel_short_account_house_init", 0.0) or 0.0)))


def trade_rates(trade: dict) -> Tuple[float, float]:
    """The house rates a funded trade carries — the ones stamped at its entry — else the defaults."""
    d = default_rates()
    try:
        hm = float(trade["sel_house_maint"]) if trade.get("sel_house_maint") is not None else d[0]
        hi = float(trade["sel_house_init"]) if trade.get("sel_house_init") is not None else d[1]
    except (TypeError, ValueError):
        return d
    return (hm, hi) if math.isfinite(hm) and math.isfinite(hi) else d


def whatif_enabled() -> bool:
    return enabled() and bool(getattr(settings, "enable_sel_short_ibkr_whatif", False))


def ibkr_margin(ticker: str, price: float) -> Optional[dict]:
    """IBKR's what-if margin for a ~$500 short of ``ticker`` at ``price`` (`Broker.what_if_short`:
    priced by IBKR, never transmitted), or None without an IBKR broker. Fail-soft."""
    try:
        from src.broker import get_broker
        broker = get_broker()
        if broker is None or not price or price <= 0:
            return None
        qty = max(1, int(WHATIF_NOMINAL_USD // float(price)))
        return broker.what_if_short(ticker, qty, float(price))
    except Exception as e:                                     # noqa: BLE001
        logger.warning(f"[sel_short] IBKR what-if for {ticker} failed ({type(e).__name__}: {e}) — default rates")
        return None


def comm(n: float, p: float) -> float:
    """IBKR's fixed commission: max($1, $0.005 a share), at most 1% of the order's value."""
    return min(max(1.0, 0.005 * n), 0.01 * n * p)


def paid_in(now: datetime) -> float:
    """The money paid in: the start balance `sel_short_account_initial` ($10,000), no deposits (user
    directive 2026-10-07: "Change the $5000+$1000/14days to $10000. We'll use this number from now on.")."""
    return float(settings.sel_short_account_initial)


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
            hm, hi = trade_rates(t)
            m = max(maint(n, px), hm * n * px)
            out["maint"] += m
            out["init_req"] += max(0.5 * n * px, m, hi * n * px)
            out["open"] += 1
        else:
            out["realized"] += pnl
    out["equity"] = out["paid_in"] + out["realized"] + out["unrealized"]
    return out


def size(trades: Sequence[dict], price: float, dv20: Optional[float] = None,
         now: Optional[datetime] = None,
         rates: Optional[Tuple[float, float]] = None,
         reserve_slices: float = 0.0) -> Tuple[int, str, Dict[str, float]]:
    """``(shares, why, account state)`` for a new short at ``price``: the engine's sizing rule.
    ``rates`` = the name's house (maintenance, initial) from IBKR's what-if, else the defaults; an
    initial rate above the default shrinks the short by default / rate. ``reserve_slices`` (the vol
    arm's thin stocks, FREE CAPITAL ONLY — PREREG32's cap5k8): the initial-margin room of that many
    live slices at the default house rate stays free after this short ("thin_reserve" when it would
    not). ``why`` is "ok" or the reason no share fits."""
    st = state(trades, now)
    eq = st["equity"]
    if eq < float(settings.sel_short_account_min_equity):
        return 0, "account_below_minimum", st
    if not price or price <= 0:
        return 0, "no_price", st
    hi = (rates or default_rates())[1]
    room = eq - st["init_req"]
    if reserve_slices and reserve_slices > 0:
        keep = float(reserve_slices) * eq / max(1, int(settings.sel_short_account_slices)) * default_rates()[1]
        if room - keep <= 0:
            return 0, "thin_reserve", st
        room -= keep
    rate = max(init_rate(price), hi * price)       # the initial requirement per share shorted
    nm = int(math.floor(room / rate)) if room > 0 else 0
    if nm > 0:                                     # the entry's own fees out of the margin room
        c0 = comm(nm, price) + SEC_FEE * nm * price + min(TAF_PER_SHARE * nm, TAF_MAX)
        nm = int(math.floor(max(room - c0, 0.0) / rate))
    alloc = min(eq / max(1, int(settings.sel_short_account_slices)), eq - st["gross"])
    budget = default_rates()[1]
    if budget > 0 and hi > budget:                 # a dearer name: no more initial margin than a default slice
        alloc *= budget / hi
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
