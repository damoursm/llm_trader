"""Project the sim ledger onto ACTUAL IBKR executions — the "real" view.

The sim ledger records what the strategy DECIDED: decision-time prices run
through the modeled spread/commission stack, broker-independent by design.
This module answers the other question — what did the broker actually do?
It returns one trade-shaped dict per ledger trade whose entry genuinely
filled, anchored at real average fill prices with real IBKR commissions and
NO modeled costs anywhere (a fill price already contains the spread that was
actually paid; the commission is the one actually charged).

View semantics:
  • Only trades with a filled broker entry appear (``broker_fill_qty > 0``).
    Dry-run intents, rejected/expired/never-filled entries and pre-broker
    history simply don't exist in this view.
  • ``entry_price`` / ``exit_price`` are the actual average fill prices;
    entry/exit timestamps prefer the broker submission times.
  • A ledger-CLOSED trade whose exit hasn't filled is still OPEN here — the
    shares are genuinely held whatever the sim says — and marks at
    ``current_price`` like any open position.
  • ``return_pct`` = signed gross move on fill prices − actual commissions as
    a % of the filled entry notional. Open trades carry the entry commission
    only (the exit cost isn't known yet). Partial exits are treated as closed
    at the exit fill price (the residue is the drift pass's problem, not a
    performance event).
  • ``position_size_multiplier`` is replaced by the REAL filled notional in
    USD so capital-weighted stats weight by actual dollars at risk.

Used by the dashboard's Simulated ⇄ IBKR-fills toggle (``dashboard/data.py``).
"""
from __future__ import annotations

from statistics import median
from typing import List, Optional


def _f(v) -> Optional[float]:
    try:
        f = float(v)
        return f if f == f else None   # NaN-safe
    except (TypeError, ValueError):
        return None


def build_broker_trades(trades: List[dict]) -> List[dict]:
    """Trade-shaped dicts re-anchored at actual IBKR fills (see module doc)."""
    out: List[dict] = []
    for t in trades:
        fq = int(t.get("broker_fill_qty") or 0)
        fp = _f(t.get("broker_fill_price"))
        if fq <= 0 or not fp or fp <= 0:
            continue
        b = dict(t)
        b["broker_view"] = True
        b["entry_price"] = fp
        b["entry_datetime"] = t.get("broker_submitted_at") or t.get("entry_datetime")
        notional = fq * fp
        b["position_size_multiplier"] = notional
        b["filled_qty"] = fq
        b["filled_notional_usd"] = round(notional, 2)
        sign = 1.0 if t.get("action") == "BUY" else -1.0
        entry_comm = _f(t.get("broker_commission")) or 0.0

        xq = int(t.get("broker_exit_fill_qty") or 0)
        xp = _f(t.get("broker_exit_fill_price"))
        if xq > 0 and xp and xp > 0:
            exit_comm = _f(t.get("broker_exit_commission")) or 0.0
            b["status"] = "CLOSED"
            b["exit_price"] = xp
            b["exit_datetime"] = t.get("broker_exit_submitted_at") or t.get("exit_datetime")
            gross = sign * (xp - fp) / fp * 100.0
            b["return_pct"] = round(gross - (entry_comm + exit_comm) / notional * 100.0, 4)
        else:
            # Entry filled, exit not (or not yet) — genuinely still held.
            b["status"] = "OPEN"
            b["exit_price"] = None
            b["exit_date"] = None
            b["exit_datetime"] = None
            mark = _f(t.get("current_price")) or fp
            gross = sign * (mark - fp) / fp * 100.0
            b["return_pct"] = round(gross - entry_comm / notional * 100.0, 4)
        out.append(b)
    return out


def _dollar_pnl(b: dict) -> float:
    """Realized (CLOSED) or mark-to-market (OPEN) P&L in USD, net of the
    commissions actually charged so far."""
    sign = 1.0 if b.get("action") == "BUY" else -1.0
    fq = int(b.get("broker_fill_qty") or 0)
    entry = float(b["entry_price"])
    if b["status"] == "CLOSED":
        end = float(b["exit_price"])
        comm = (_f(b.get("broker_commission")) or 0.0) + (_f(b.get("broker_exit_commission")) or 0.0)
    else:
        end = _f(b.get("current_price")) or entry
        comm = _f(b.get("broker_commission")) or 0.0
    return sign * (end - entry) * fq - comm


def _exec_cost_pct(leg_is_buy: bool, model: Optional[float], fill: Optional[float]) -> float:
    """Cost-normalized execution cost of one leg vs its model/decision price,
    in % — positive = adverse (paid above model when buying, received below
    when selling). 0.0 when the model price is missing so the leg's cost
    degrades cleanly to commission-only rather than dropping out."""
    if not model or model <= 0 or not fill or fill <= 0:
        return 0.0
    return ((fill - model) if leg_is_buy else (model - fill)) / model * 100.0


def leg_one_way_cost_pct(leg: dict) -> Optional[float]:
    """All-in one-way cost (%) of ONE real filled LMT leg: real commission as %
    of the leg's notional PLUS the cost-normalized execution cost vs the
    decision price (the order's own side decides the adverse direction — a BUY
    order is adverse filling higher, a SELL adverse filling lower, covering
    entries and exits without needing to know which). None for unusable legs."""
    fq = int(leg.get("filled_qty") or 0)
    fp = _f(leg.get("fill_price"))
    if fq <= 0 or not fp or fp <= 0:
        return None
    comm = _f(leg.get("commission")) or 0.0
    comm_pct = comm / (fq * fp) * 100.0
    slip = _exec_cost_pct(str(leg.get("side") or "").upper() == "BUY",
                          _f(leg.get("model_price")), fp)
    return comm_pct + slip


def one_way_cost_pcts_from_legs(legs: List[dict]) -> List[float]:
    """Per-leg all-in one-way costs (%) over real **LMT** filled legs from
    ``broker_orders`` (``repo.fetch_filled_lmt_legs``). MKT legs never reach
    here — they're filtered out at the source, since LMT is what the system
    uses going forward. See :func:`leg_one_way_cost_pct` for the per-leg math."""
    out: List[float] = []
    for r in legs:
        pct = leg_one_way_cost_pct(r)
        if pct is not None:
            out.append(pct)
    return out


def real_one_way_cost_fraction(legs: List[dict], min_legs: int) -> Optional[float]:
    """Average all-in one-way cost as a FRACTION (e.g. 0.0056) over real LMT
    fills, for calibrating the simulated cost model. None until at least
    ``min_legs`` LMT legs exist — a flat average over a handful of fills is
    noise. Clamped ≥ 0 by the caller (``set_real_cost_override``)."""
    pcts = one_way_cost_pcts_from_legs(legs)
    if len(pcts) < max(1, int(min_legs)):
        return None
    return (sum(pcts) / len(pcts)) / 100.0


def avg_one_way_cost_pct_from_legs(legs: List[dict]) -> Optional[float]:
    """Average all-in one-way cost (%) over real LMT fills, no min-leg floor —
    the descriptive figure for the dashboard's IBKR cost tile. None when there
    are no LMT fills yet."""
    pcts = one_way_cost_pcts_from_legs(legs)
    return round(sum(pcts) / len(pcts), 4) if pcts else None


def summarize_broker_trades(btrades: List[dict], account_equity_usd: Optional[float] = None) -> dict:
    """Headline numbers for the dashboard's IBKR view.

    ``win_rate`` / ``avg_return`` / ``median_return`` are OPEN-INCLUSIVE — every
    trade counts, with an OPEN position's live-mark ``return_pct`` (already
    computed by ``build_broker_trades``) treated as a hypothetical exit. This
    mirrors the Simulated view's ``tracker._compute_segment_stats`` exactly, so
    the two sides of the dashboard's Simulated⇄IBKR toggle share the same
    trade-count denominator and differ ONLY in cost basis (modeled vs real
    fills) — the toggle's actual purpose. Before this they didn't: a 7-day
    window with 2 winners out of 7 CLOSED trades (28.6%) sat next to 8 winning
    OPEN positions excluded from the ratio entirely, while the trade tables on
    the same page told a 10-win/23-trade (43.5%) story a user counting by hand
    would reach (observed 2026-07-02).

    Dollar P&L is the primary lens, but ``weighted_return`` is the precise total
    % — each CLOSED round-trip weighted by its REAL filled notional, so sizing
    counts (the equal-weighted ``avg_return`` is kept alongside for comparison).
    ``weighted_return`` stays CLOSED-only deliberately (unlike the three above):
    an open position's filled notional is capital still AT RISK, not yet a
    realized dollar outcome to weight into a total-return figure.
    When ``account_equity_usd`` (the IBKR account NAV, in USD) is supplied,
    ``account_return_pct`` expresses cumulative P&L over the real account size —
    the account-relative impact, using more of the IBKR account data than fills
    alone."""
    closed = [b for b in btrades if b["status"] == "CLOSED"]
    open_ = [b for b in btrades if b["status"] == "OPEN"]
    all_rets = [float(b["return_pct"]) for b in btrades if b.get("return_pct") is not None]
    wins = [r for r in all_rets if r > 0]
    # Exit commissions only count once the exit actually filled — a stale
    # field on a cancelled/unfilled exit is not money spent.
    commissions = sum(
        (_f(b.get("broker_commission")) or 0.0)
        + ((_f(b.get("broker_exit_commission")) or 0.0) if b["status"] == "CLOSED" else 0.0)
        for b in btrades
    )
    realized = round(sum(_dollar_pnl(b) for b in closed), 2)
    unrealized = round(sum(_dollar_pnl(b) for b in open_), 2)
    total_pnl = round(realized + unrealized, 2)
    # Notional-weighted return — weight each closed round-trip by its REAL filled
    # dollars, so a $9k winner doesn't count the same as a $900 scratch (the
    # equal-weighted avg_return hides sizing; this is the precise total %).
    # Deliberately CLOSED-only — see the docstring.
    w_num = w_den = 0.0
    for b in closed:
        r = b.get("return_pct")
        nz = _f(b.get("filled_notional_usd")) or 0.0
        if r is not None and nz > 0:
            w_num += float(r) * nz
            w_den += nz
    weighted_return = round(w_num / w_den, 2) if w_den > 0 else None
    # Account-relative return — cumulative P&L over the IBKR account NAV (USD).
    account_return_pct = (round(total_pnl / account_equity_usd * 100.0, 2)
                          if account_equity_usd and account_equity_usd > 0 else None)
    # NOTE: the average one-way COST is no longer computed here — it is sourced
    # from real LMT fills in broker_orders (avg_one_way_cost_pct_from_legs), so
    # MKT fills are excluded and the figure matches the sim calibration.
    return {
        "trades": len(btrades),
        "closed": len(closed),
        "open": len(open_),
        "win_rate": round(100.0 * len(wins) / len(all_rets), 1) if all_rets else None,
        "avg_return": round(sum(all_rets) / len(all_rets), 2) if all_rets else None,
        "median_return": round(median(all_rets), 2) if all_rets else None,
        "weighted_return": weighted_return,
        "realized_pnl_usd": realized,
        "unrealized_pnl_usd": unrealized,
        "total_pnl_usd": total_pnl,
        "account_return_pct": account_return_pct,
        "account_equity_usd": round(float(account_equity_usd), 2) if account_equity_usd else None,
        "commissions_usd": round(commissions, 2),
    }
