"""Shadow & reconcile — converge broker state to the internal trade ledger.

Called once per pipeline tick, AFTER the internal open/close logic. This is the
single broker hook point: idempotent and self-healing.

  • Internal OPEN trade with no broker entry yet  → size + submit an entry, then
    record broker_* fields on the trade.
  • Internal CLOSED trade that still holds a broker position → submit the close,
    record the broker exit.
  • Broker position with no matching internal OPEN trade (drift) → report it
    (report-only; we never auto-flatten unexpected positions).

The internal NAV sim remains the source of truth for analytics; this adds the real
execution leg plus a reconciliation report (slippage, drift, rejects) for the email
and a broker-health alert. Every broker call is defensive — a broker failure logs,
sets ok=False, and degrades to report-only; it never raises into the pipeline.
"""
from __future__ import annotations

from typing import List, Optional

from loguru import logger

from config.settings import settings
from src.broker import Broker, OrderRequest, get_broker
from src.broker.fx import usd_per_unit
from src.broker.sizing import shares_for, shares_for_notional, within_caps
from src.db import repo


def _entry_side(action: str) -> str:
    return "BUY" if action == "BUY" else "SELL"


def _exit_side(action: str) -> str:
    # Flatten: a long (opened BUY) closes with SELL; a short (opened SELL) closes with BUY.
    return "SELL" if action == "BUY" else "BUY"


def _new_report() -> dict:
    return {
        "mode": settings.broker_mode, "connected": False, "ok": True,
        "entries_submitted": 0, "exits_submitted": 0, "rejects": 0,
        "drift": [], "slippage": [], "errors": [], "account_equity": None,
    }


def sync(broker: Optional[Broker] = None, trades: Optional[List[dict]] = None) -> dict:
    """Reconcile the broker against the internal ledger. Returns a report dict.

    No-op (empty report, ok=True) when broker_mode=off.
    """
    report = _new_report()
    broker = broker if broker is not None else get_broker()
    if broker is None:
        return report  # broker_mode=off → no broker calls at all

    try:
        report["connected"] = bool(broker.connect())
    except Exception as e:  # never let a broker hiccup break the pipeline
        report["ok"] = False
        report["errors"].append(f"connect: {e}")
        logger.warning(f"[broker] connect raised: {e}")
        return report
    if not report["connected"]:
        report["ok"] = False
        report["errors"].append("not connected")
        logger.warning("[broker] not connected — skipping sync (internal sim unaffected)")
        return report

    try:
        acct = broker.get_account()
        equity = acct.equity if acct else float(settings.broker_paper_equity)
        acct_ccy = acct.currency if acct else "USD"
        report["account_equity"] = equity
        report["account_currency"] = acct_ccy
        # Everything below sizes in USD (US securities are USD-priced). Convert the
        # account equity and the (CAD) base notional to USD via live FX.
        equity_usd = equity * usd_per_unit(acct_ccy)
        fx_notional = usd_per_unit(settings.broker_base_notional_ccy)
        positions = {p.ticker: p for p in broker.get_positions()}

        trades = trades if trades is not None else repo.load_trades()
        open_trades = [t for t in trades if t.get("status") == "OPEN"]
        closed_trades = [t for t in trades if t.get("status") == "CLOSED"]

        gross = sum(abs(p.quantity) * (p.market_price or p.avg_cost or 0.0)
                    for p in positions.values())
        n_open = sum(1 for p in positions.values() if p.quantity != 0)
        changed = False

        # ── ENTRIES: OPEN trades not yet sent to the broker ──────────────
        for t in open_trades:
            if t.get("broker_order_id"):
                continue  # idempotent — already submitted
            price = float(t.get("entry_price") or t.get("current_price") or 0.0)
            mult = t.get("position_size_multiplier", 1.0)
            if settings.broker_sizing_mode == "equity_pct":
                qty = shares_for(equity_usd, price, mult)
            else:
                qty = shares_for_notional(settings.broker_base_notional, fx_notional, price, mult)
            if qty <= 0:
                t["broker_status"] = "SKIPPED_ZERO_QTY"
                changed = True
                continue
            ok, reason = within_caps(n_open, gross + qty * price, equity_usd)
            if not ok:
                logger.info(f"[broker] entry {t['ticker']} skipped — {reason}")
                t["broker_status"] = f"SKIPPED_CAP: {reason}"
                changed = True
                continue
            res = broker.submit_order(OrderRequest(
                ticker=t["ticker"], side=_entry_side(t["action"]), quantity=qty,
                order_type=settings.broker_order_type,
                client_ref=t.get("recommendation_id") or f"{t.get('run_id', '')}-{t['ticker']}",
                intent="ENTRY",
            ))
            _apply_entry_result(t, res)
            changed = True
            if res.ok:
                report["entries_submitted"] += 1
                n_open += 1
                gross += qty * price
                if res.avg_fill_price and price:
                    report["slippage"].append({
                        "ticker": t["ticker"], "model": round(price, 4),
                        "fill": round(res.avg_fill_price, 4),
                        "bps": round((res.avg_fill_price - price) / price * 10000, 1),
                    })
            else:
                report["rejects"] += 1
                report["errors"].append(f"entry {t['ticker']}: {res.error}")

        # ── EXITS: CLOSED trades that still hold a broker position ────────
        for t in closed_trades:
            if not t.get("broker_order_id") or t.get("broker_exit_order_id"):
                continue  # never entered via broker, or already exited
            held = positions.get(t["ticker"])
            qty = int(t.get("broker_fill_qty") or (abs(held.quantity) if held else 0))
            if qty <= 0:
                t["broker_exit_status"] = "NOTHING_TO_CLOSE"
                changed = True
                continue
            res = broker.submit_order(OrderRequest(
                ticker=t["ticker"], side=_exit_side(t["action"]), quantity=qty,
                order_type=settings.broker_order_type,
                client_ref=(t.get("recommendation_id") or t["ticker"]) + "-exit",
                intent="EXIT",
            ))
            _apply_exit_result(t, res)
            changed = True
            if res.ok:
                report["exits_submitted"] += 1
            else:
                report["rejects"] += 1
                report["errors"].append(f"exit {t['ticker']}: {res.error}")

        # ── DRIFT: broker positions with no matching internal OPEN trade ──
        open_tickers = {t["ticker"] for t in open_trades}
        for tk, p in positions.items():
            if p.quantity != 0 and tk not in open_tickers:
                report["drift"].append({"ticker": tk, "broker_qty": p.quantity})

        if changed:
            repo.save_trades(trades)

    except Exception as e:  # belt-and-suspenders: never propagate into the pipeline
        report["ok"] = False
        report["errors"].append(str(e))
        logger.warning(f"[broker] sync failed (internal sim unaffected): {e}")
        return report

    logger.info(
        f"[broker:{settings.broker_mode}] sync — entries={report['entries_submitted']} "
        f"exits={report['exits_submitted']} rejects={report['rejects']} "
        f"drift={len(report['drift'])} equity={report['account_equity']:.0f}"
    )
    return report


def _apply_entry_result(t: dict, res) -> None:
    t["broker_order_id"]     = res.order_id
    t["broker_side"]         = res.side
    t["broker_fill_qty"]     = res.filled_qty
    t["broker_fill_price"]   = res.avg_fill_price
    t["broker_status"]       = res.status
    t["broker_submitted_at"] = res.submitted_at
    t["broker_client_ref"]   = res.client_ref


def _apply_exit_result(t: dict, res) -> None:
    t["broker_exit_order_id"]     = res.order_id
    t["broker_exit_fill_qty"]     = res.filled_qty
    t["broker_exit_fill_price"]   = res.avg_fill_price
    t["broker_exit_status"]       = res.status
    t["broker_exit_submitted_at"] = res.submitted_at
