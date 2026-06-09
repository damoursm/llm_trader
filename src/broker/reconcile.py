"""Shadow & reconcile — converge broker state to the internal trade ledger.

Called once per pipeline tick, AFTER the internal open/close logic. This is the
single broker hook point: idempotent and self-healing.

  • Fill refresh: orders submitted on an earlier tick that hadn't reached a
    terminal state inside submit_order's short poll (queued overnight, partial
    fill, late commission report) are repaired from today's executions.
  • Internal OPEN trade with no broker entry yet  → size + submit an entry, then
    record broker_* fields on the trade.
  • Internal CLOSED trade that still holds a broker position → submit the close,
    record the broker exit.
  • Broker position with no matching internal OPEN trade (drift) → report it
    (report-only; we never auto-flatten unexpected positions).

The internal NAV sim remains the source of truth for analytics; this adds the real
execution leg plus a reconciliation report (slippage, drift, rejects) for the email
and a broker-health alert. The full report — including one event row per order
submission / fill repair — is persisted to DuckDB by the pipeline
(``repo.insert_broker_report``) so the paper phase accumulates a durable
slippage / reject / drift record instead of losing it after each run.

Slippage convention: ``bps`` is **cost-normalized** — positive always means the
fill was worse than the model price (paid more on a BUY, received less on a
SELL), so entries and exits, longs and shorts, average meaningfully together.

Every broker call is defensive — a broker failure logs, sets ok=False, and
degrades to report-only; it never raises into the pipeline.
"""
from __future__ import annotations

import math
from datetime import datetime, timezone
from typing import List, Optional

from loguru import logger

from config.settings import settings
from src.broker import Broker, FillSummary, OrderRequest, get_broker
from src.broker.fx import usd_per_unit
from src.broker.sizing import shares_for, shares_for_notional, within_caps
from src.db import repo


def _entry_side(action: str) -> str:
    return "BUY" if action == "BUY" else "SELL"


def _exit_side(action: str) -> str:
    # Flatten: a long (opened BUY) closes with SELL; a short (opened SELL) closes with BUY.
    return "SELL" if action == "BUY" else "BUY"


def _utcnow_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _cost_bps(side: str, model: float, fill: float) -> Optional[float]:
    """Execution cost vs the model price in basis points, normalized so that
    positive = adverse for either side (BUY filled above model / SELL below)."""
    if not model or model <= 0 or not fill or fill <= 0:
        return None
    sign = 1.0 if side == "BUY" else -1.0
    return round(sign * (fill - model) / model * 10000, 1)


def _limit_price_for(side: str, model: float) -> Optional[float]:
    """Marketable-limit cap: model price ± broker_limit_cap_bps in the adverse
    direction, rounded *away* from the model to a valid US-equity tick so the
    cap is never tighter than configured. None for unusable model prices."""
    if not model or model <= 0:
        return None
    bps = float(settings.broker_limit_cap_bps)
    cap = model * (1 + bps / 10000.0) if side == "BUY" else model * (1 - bps / 10000.0)
    tick = 0.01 if model >= 1.0 else 0.0001
    steps = cap / tick
    cap = (math.ceil(steps) if side == "BUY" else math.floor(steps)) * tick
    return round(cap, 2 if model >= 1.0 else 4)


def _new_report() -> dict:
    return {
        "run_id": None, "mode": settings.broker_mode, "connected": False, "ok": True,
        "entries_submitted": 0, "exits_submitted": 0, "fills_repaired": 0, "rejects": 0,
        "drift": [], "slippage": [], "orders": [], "errors": [], "account_equity": None,
    }


def _record_slippage(report: dict, intent: str, ticker: str, side: str,
                     model: Optional[float], fill: Optional[float],
                     commission: Optional[float]) -> None:
    bps = _cost_bps(side, model or 0.0, fill or 0.0)
    if bps is None:
        return
    report["slippage"].append({
        "ticker": ticker, "intent": intent, "side": side,
        "model": round(model, 4), "fill": round(fill, 4),
        "bps": bps, "commission": commission,
    })


def _record_order(report: dict, *, event: str, intent: str, ticker: str, side: str,
                  order_type: str, requested_qty: Optional[int], filled_qty: Optional[int],
                  model_price: Optional[float], limit_price: Optional[float],
                  fill_price: Optional[float], commission: Optional[float],
                  status: Optional[str], ok: Optional[bool], error: Optional[str],
                  order_id: Optional[str], client_ref: Optional[str],
                  submitted_at: Optional[str]) -> None:
    """Append one order event row (SUBMIT or FILL_REFRESH) for DuckDB persistence."""
    report["orders"].append({
        "event": event, "intent": intent, "ticker": ticker, "side": side,
        "order_type": order_type, "requested_qty": requested_qty,
        "filled_qty": filled_qty, "model_price": model_price,
        "limit_price": limit_price, "fill_price": fill_price,
        "slippage_bps": _cost_bps(side, model_price or 0.0, fill_price or 0.0),
        "commission": commission, "status": status, "ok": ok, "error": error,
        "order_id": order_id, "client_ref": client_ref,
        "submitted_at": submitted_at or _utcnow_iso(),
    })


# ── fill refresh: repair stale broker_fill_* fields from today's executions ──

_TERMINAL_STATUSES = ("Filled", "Cancelled", "ApiCancelled", "Inactive", "DRYRUN")


def _leg_needs_refresh(t: dict, prefix: str) -> bool:
    """True when a submitted order leg ('broker_' or 'broker_exit_') is missing
    fill data or hasn't reached a terminal status yet."""
    if not t.get(f"{prefix}order_id"):
        return False
    if (t.get(f"{prefix}status") or "") not in _TERMINAL_STATUSES:
        return True
    return (not t.get(f"{prefix}fill_qty")
            or t.get(f"{prefix}fill_price") is None
            or t.get(f"{prefix}commission") is None)


def _apply_fill(t: dict, prefix: str, fs: FillSummary) -> bool:
    """Copy fresher fill data onto one order leg. Returns True if anything changed."""
    changed = False
    if fs.filled_qty and fs.filled_qty != t.get(f"{prefix}fill_qty"):
        t[f"{prefix}fill_qty"] = fs.filled_qty
        changed = True
    if fs.avg_fill_price and fs.avg_fill_price != t.get(f"{prefix}fill_price"):
        t[f"{prefix}fill_price"] = fs.avg_fill_price
        changed = True
    if fs.commission is not None and fs.commission != t.get(f"{prefix}commission"):
        t[f"{prefix}commission"] = fs.commission
        changed = True
    if changed:
        req = t.get(f"{prefix}requested_qty")
        t[f"{prefix}status"] = ("Filled" if not req or (fs.filled_qty or 0) >= int(req)
                                else "PartiallyFilled")
        t[f"{prefix}fill_refreshed_at"] = _utcnow_iso()
    return changed


def _refresh_fills(broker: Broker, trades: List[dict], report: dict) -> bool:
    """Repair fill qty / price / commission on previously submitted orders from
    the broker's execution feed (keyed by client_ref → IBKR orderRef). Covers
    orders that filled after submit_order's poll window: queued-overnight
    entries, partial fills, late commission reports. Returns True if any trade
    changed (caller persists)."""
    candidates = []
    for t in trades:
        if _leg_needs_refresh(t, "broker_"):
            candidates.append((t, "broker_", "ENTRY"))
        if _leg_needs_refresh(t, "broker_exit_"):
            candidates.append((t, "broker_exit_", "EXIT"))
    if not candidates:
        return False

    try:
        fills = {f.client_ref: f for f in broker.get_fills()}
    except Exception as e:
        logger.warning(f"[broker] get_fills failed (fill refresh skipped): {e}")
        return False
    if not fills:
        return False

    changed = False
    for t, prefix, intent in candidates:
        if intent == "ENTRY":
            ref = t.get("broker_client_ref") or t.get("recommendation_id")
            model = t.get("entry_price")
        else:
            ref = (t.get("broker_exit_client_ref")
                   or (t.get("recommendation_id") or t.get("ticker", "")) + "-exit")
            model = t.get("exit_price")
        fs = fills.get(ref) if ref else None
        if fs is None or not _apply_fill(t, prefix, fs):
            continue
        changed = True
        report["fills_repaired"] += 1
        side = _entry_side(t["action"]) if intent == "ENTRY" else _exit_side(t["action"])
        _record_slippage(report, intent, t["ticker"], side,
                         model, fs.avg_fill_price, fs.commission)
        _record_order(
            report, event="FILL_REFRESH", intent=intent, ticker=t["ticker"], side=side,
            order_type=settings.broker_order_type, requested_qty=t.get(f"{prefix}requested_qty"),
            filled_qty=fs.filled_qty, model_price=model, limit_price=None,
            fill_price=fs.avg_fill_price, commission=fs.commission,
            status=t.get(f"{prefix}status"), ok=True, error=None,
            order_id=t.get(f"{prefix}order_id"), client_ref=ref,
            submitted_at=fs.last_fill_at,
        )
        logger.info(
            f"[broker] fill refresh — {intent} {t['ticker']}: qty={fs.filled_qty} "
            f"fill={fs.avg_fill_price} commission={fs.commission}"
        )
    return changed


def sync(broker: Optional[Broker] = None, trades: Optional[List[dict]] = None,
         run_id: Optional[str] = None) -> dict:
    """Reconcile the broker against the internal ledger. Returns a report dict.

    No-op (empty report, ok=True) when broker_mode=off.
    """
    report = _new_report()
    report["run_id"] = run_id
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
        report["account_id"] = acct.account_id if acct else None

        # SAFETY: in paper mode, only ever trade a paper account. IBKR paper account
        # numbers are D-prefixed (e.g. DU…); live accounts are U-prefixed. A non-D
        # account here means TWS/Gateway is logged into the wrong login — refuse to
        # place ANY orders rather than risk trading a live account.
        if settings.broker_mode == "ibkr_paper" and acct and not str(acct.account_id).upper().startswith("D"):
            report["ok"] = False
            report["errors"].append(
                f"paper mode but connected account {acct.account_id} is not a paper (D…) account")
            logger.critical(
                f"[broker] SAFETY STOP — broker_mode=ibkr_paper but connected account is "
                f"{acct.account_id} (not a paper/D… account). NO orders placed — check your TWS/Gateway login."
            )
            return report

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

        # ── FILL REFRESH: repair orders that completed after a previous tick ──
        changed = _refresh_fills(broker, trades, report) or changed

        use_limit = settings.broker_order_type == "LMT"

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
            side = _entry_side(t["action"])
            limit = _limit_price_for(side, price) if use_limit else None
            res = broker.submit_order(OrderRequest(
                ticker=t["ticker"], side=side, quantity=qty,
                order_type=settings.broker_order_type, limit_price=limit,
                client_ref=t.get("recommendation_id") or f"{t.get('run_id', '')}-{t['ticker']}",
                intent="ENTRY",
            ))
            _apply_entry_result(t, res)
            changed = True
            _record_order(
                report, event="SUBMIT", intent="ENTRY", ticker=t["ticker"], side=side,
                order_type=settings.broker_order_type, requested_qty=qty,
                filled_qty=res.filled_qty, model_price=price, limit_price=limit,
                fill_price=res.avg_fill_price, commission=res.commission,
                status=res.status, ok=res.ok, error=res.error,
                order_id=res.order_id, client_ref=res.client_ref,
                submitted_at=res.submitted_at,
            )
            if res.ok:
                report["entries_submitted"] += 1
                n_open += 1
                gross += qty * price
                _record_slippage(report, "ENTRY", t["ticker"], side,
                                 price, res.avg_fill_price, res.commission)
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
            side = _exit_side(t["action"])
            model = float(t.get("exit_price") or t.get("current_price") or 0.0)
            limit = _limit_price_for(side, model) if use_limit else None
            res = broker.submit_order(OrderRequest(
                ticker=t["ticker"], side=side, quantity=qty,
                order_type=settings.broker_order_type, limit_price=limit,
                client_ref=(t.get("recommendation_id") or t["ticker"]) + "-exit",
                intent="EXIT",
            ))
            _apply_exit_result(t, res)
            changed = True
            _record_order(
                report, event="SUBMIT", intent="EXIT", ticker=t["ticker"], side=side,
                order_type=settings.broker_order_type, requested_qty=qty,
                filled_qty=res.filled_qty, model_price=model, limit_price=limit,
                fill_price=res.avg_fill_price, commission=res.commission,
                status=res.status, ok=res.ok, error=res.error,
                order_id=res.order_id, client_ref=res.client_ref,
                submitted_at=res.submitted_at,
            )
            if res.ok:
                report["exits_submitted"] += 1
                _record_slippage(report, "EXIT", t["ticker"], side,
                                 model, res.avg_fill_price, res.commission)
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
        f"exits={report['exits_submitted']} fills_repaired={report['fills_repaired']} "
        f"rejects={report['rejects']} drift={len(report['drift'])} "
        f"equity={report['account_equity']:.0f}"
    )
    return report


def _apply_entry_result(t: dict, res) -> None:
    t["broker_order_id"]     = res.order_id
    t["broker_side"]         = res.side
    t["broker_requested_qty"] = res.requested_qty
    t["broker_fill_qty"]     = res.filled_qty
    t["broker_fill_price"]   = res.avg_fill_price
    t["broker_commission"]   = res.commission
    t["broker_status"]       = res.status
    t["broker_submitted_at"] = res.submitted_at
    t["broker_client_ref"]   = res.client_ref


def _apply_exit_result(t: dict, res) -> None:
    t["broker_exit_order_id"]     = res.order_id
    t["broker_exit_requested_qty"] = res.requested_qty
    t["broker_exit_fill_qty"]     = res.filled_qty
    t["broker_exit_fill_price"]   = res.avg_fill_price
    t["broker_exit_commission"]   = res.commission
    t["broker_exit_status"]       = res.status
    t["broker_exit_submitted_at"] = res.submitted_at
    t["broker_exit_client_ref"]   = res.client_ref
