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


def summarize_broker_trades(btrades: List[dict]) -> dict:
    """Headline numbers for the dashboard's IBKR view. Dollar P&L is the
    primary lens here — real executions have real notionals, so percentages
    alone hide sizing."""
    closed = [b for b in btrades if b["status"] == "CLOSED"]
    open_ = [b for b in btrades if b["status"] == "OPEN"]
    wins = [b for b in closed if (b.get("return_pct") or 0.0) > 0]
    rets = [float(b["return_pct"]) for b in closed if b.get("return_pct") is not None]
    # Exit commissions only count once the exit actually filled — a stale
    # field on a cancelled/unfilled exit is not money spent.
    commissions = sum(
        (_f(b.get("broker_commission")) or 0.0)
        + ((_f(b.get("broker_exit_commission")) or 0.0) if b["status"] == "CLOSED" else 0.0)
        for b in btrades
    )
    return {
        "trades": len(btrades),
        "closed": len(closed),
        "open": len(open_),
        "win_rate": round(100.0 * len(wins) / len(closed), 1) if closed else None,
        "avg_return": round(sum(rets) / len(rets), 2) if rets else None,
        "realized_pnl_usd": round(sum(_dollar_pnl(b) for b in closed), 2),
        "unrealized_pnl_usd": round(sum(_dollar_pnl(b) for b in open_), 2),
        "commissions_usd": round(commissions, 2),
    }
