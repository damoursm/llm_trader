"""Shadow & reconcile — converge broker state to the internal trade ledger.

Called once per pipeline tick, AFTER the internal open/close logic. This is the
single broker hook point: idempotent and self-healing.

  • Fill refresh: orders submitted on an earlier tick that hadn't reached a
    terminal state inside submit_order's short poll (queued overnight, partial
    fill, late commission report) are repaired from today's executions.
  • Stale-unfilled cancel: an accepted order resting unfilled past
    ``broker_unfilled_cancel_minutes`` (its price cap was never reached) is
    cancelled and resubmitted re-anchored at the current mark.
  • Internal OPEN trade with no broker entry yet  → size + submit an entry, then
    record broker_* fields on the trade.
  • Internal CLOSED trade that still holds a broker position → submit the close,
    record the broker exit.
  • Broker position with no matching internal OPEN trade (drift) → flatten it
    with a price-capped LMT at a fresh live quote (``broker_drift_action``,
    default ``flatten``; ``report`` is surface-only and ibkr_live always
    downgrades to report). A same-side flatten that already filled within the
    guard window blocks a resubmit — a duplicate would flip the position, not
    flatten it.

Submission reliability: every submit is verified against the broker's answer
(``_submit_with_retry``) — TRANSIENT failures (connection drop, timeout,
pacing) retry a few times in-tick behind a duplicate guard (an attempt that
errored after transmission may exist at the broker; resubmitting blind would
double the position), while hard rejects return immediately. The retry window
is deliberately short and every retry goes out as a marketable LMT capped at
``broker_limit_cap_bps`` from the tick's model price, so a delayed fill can
never drift past the cap.

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
import time
from dataclasses import replace
from datetime import datetime, timezone
from typing import List, Optional

from loguru import logger

from config.settings import settings
from src.broker import Broker, FillSummary, OrderRequest, get_broker
from src.broker.base import OrderResult
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


def _limit_price_for(side: str, model: float, outside_rth: bool = False) -> Optional[float]:
    """Marketable-limit cap: model price ± the session's cap in the adverse
    direction, rounded *away* from the model to a valid US-equity tick so the
    cap is never tighter than configured. Off-RTH uses the wider
    ``broker_limit_cap_bps_extended`` — the extended-book spread runs ~4× RTH,
    and a cap inside the spread can never fill (the order would rest, get
    cancelled next tick, and chase the market on stale data). None for
    unusable model prices."""
    if not model or model <= 0:
        return None
    bps = float(settings.broker_limit_cap_bps_extended if outside_rth
                else settings.broker_limit_cap_bps)
    cap = model * (1 + bps / 10000.0) if side == "BUY" else model * (1 - bps / 10000.0)
    tick = 0.01 if model >= 1.0 else 0.0001
    steps = cap / tick
    cap = (math.ceil(steps) if side == "BUY" else math.floor(steps)) * tick
    return round(cap, 2 if model >= 1.0 else 4)


def _new_report() -> dict:
    return {
        "run_id": None, "mode": settings.broker_mode, "connected": False, "ok": True,
        "entries_submitted": 0, "exits_submitted": 0, "fills_repaired": 0, "rejects": 0,
        "retries": 0, "stale_cancels": 0, "entry_cancels_on_close": 0, "drift_flattened": 0,
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

# Statuses for ledger rows recovered from a backup or adopted from drift:
# their broker_order_id is a sentinel (blocks entry submission), and there is
# no live order to poll, so the fill-refresh pass must never treat them as
# candidates. On close, the exit path sizes from the live broker position, so
# a paper position that does exist for the ticker is still flattened.
#   RESTORED_NOT_SUBMITTED — position was never broker-entered
#   RESTORED_ADOPTED       — a real broker position exists, but the original
#                            order linkage was lost (e.g. the 2026-06-10 wipe)
_RESTORED_STATUSES = ("RESTORED_NOT_SUBMITTED", "RESTORED_ADOPTED")

_TERMINAL_STATUSES = ("Filled", "Cancelled", "ApiCancelled", "Inactive", "DRYRUN",
                      "RESTORED_NOT_SUBMITTED", "RESTORED_ADOPTED",
                      "DUPLICATE_REF_NOT_SUBMITTED")


def _leg_needs_refresh(t: dict, prefix: str) -> bool:
    """True when a submitted order leg ('broker_' or 'broker_exit_') is missing
    fill data or hasn't reached a terminal status yet."""
    if not t.get(f"{prefix}order_id"):
        return False
    status = t.get(f"{prefix}status") or ""
    if status in _RESTORED_STATUSES:
        return False              # sentinel linkage — nothing at the broker to poll
    if status not in _TERMINAL_STATUSES:
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


# ── submission reliability: verify, classify, retry (bounded) ───────────────

# Failure text that indicates a TRANSIENT condition worth one more try —
# everything else (insufficient funds, permissions, invalid contract, …) is a
# hard reject a retry cannot fix.
_TRANSIENT_MARKERS = (
    "disconnect", "connection", "timeout", "timed out", "pacing",
    "rate limit", "temporar", "busy", "not connected", "socket",
)


def _is_transient_failure(res: OrderResult) -> bool:
    """True when a failed OrderResult looks recoverable by a quick retry."""
    if res.ok:
        return False
    if (res.status or "").upper() == "DISCONNECTED":
        return True
    blob = f"{res.status or ''} {res.error or ''}".lower()
    return any(m in blob for m in _TRANSIENT_MARKERS)


def _known_order_result(broker: Broker, req: OrderRequest) -> Optional[OrderResult]:
    """Synthesize an OrderResult from broker state when ``req.client_ref``
    already exists there — the duplicate guard before any retry.

    An attempt that errored AFTER transmission (socket dropped mid-flight) may
    have reached the broker even though we recorded a failure; resubmitting
    blind would double the position (orderRef is a tag, IBKR does not dedupe
    on it). Filled → adopt the fill; still working → adopt the order and let
    the fill-refresh pass complete it on a later tick.
    """
    try:
        for f in broker.get_fills():
            if f.client_ref == req.client_ref and (f.filled_qty or 0) > 0:
                return OrderResult(
                    ok=True, ticker=req.ticker, side=req.side,
                    requested_qty=req.quantity, filled_qty=f.filled_qty,
                    avg_fill_price=f.avg_fill_price, order_id=f.order_id,
                    client_ref=req.client_ref, status="Filled",
                    commission=f.commission,
                )
        for o in broker.get_open_orders():
            if o.client_ref == req.client_ref:
                return OrderResult(
                    ok=True, ticker=req.ticker, side=req.side,
                    requested_qty=req.quantity, filled_qty=0,
                    order_id=o.order_id, client_ref=req.client_ref,
                    status=o.status or "Submitted",
                )
    except Exception as e:
        logger.debug(f"[broker] known-order check failed for {req.client_ref}: {e}")
    return None


def _submit_with_retry(broker: Broker, req: OrderRequest, model_price: float,
                       report: dict, intent: str) -> OrderResult:
    """Submit and verify; retry TRANSIENT failures a few times, briefly.

    The window is deliberately small (``broker_submit_retries`` ×
    ``broker_retry_wait_seconds``) so any eventual fill stays anchored near
    this tick's model price — and every retry is converted to a marketable
    LMT capped at ``broker_limit_cap_bps`` from that model price, so however
    late the retry lands, the worst acceptable fill never drifts past the
    cap. Hard rejects return immediately. Each failed attempt is persisted as
    a SUBMIT_FAILED event row so the paper phase accumulates a reliability
    record alongside slippage.
    """
    res = broker.submit_order(req)
    retries = max(0, int(settings.broker_submit_retries))
    wait = max(1, int(settings.broker_retry_wait_seconds))
    attempt = 1
    while (not res.ok) and attempt <= retries and _is_transient_failure(res):
        _record_order(
            report, event="SUBMIT_FAILED", intent=intent, ticker=req.ticker,
            side=req.side, order_type=req.order_type, requested_qty=req.quantity,
            filled_qty=0, model_price=model_price, limit_price=req.limit_price,
            fill_price=None, commission=None, status=res.status, ok=False,
            error=res.error, order_id=res.order_id, client_ref=req.client_ref,
            submitted_at=res.submitted_at,
        )
        report["retries"] = report.get("retries", 0) + 1
        logger.warning(
            f"[broker] {intent} {req.ticker} attempt {attempt} failed transiently "
            f"({res.status}: {res.error}) — retrying in {wait}s"
        )
        time.sleep(wait)
        known = _known_order_result(broker, req)
        if known is not None:
            logger.info(
                f"[broker] {intent} {req.ticker}: order {req.client_ref} already at "
                f"the broker ({known.status}) — adopting it, NOT resubmitting"
            )
            return known
        try:
            broker.connect()   # idempotent; revives a dropped session
        except Exception:
            pass
        cap = _limit_price_for(req.side, model_price, req.outside_rth)
        if cap:
            req = replace(req, order_type="LMT", limit_price=cap)
        res = broker.submit_order(req)
        attempt += 1
    return res


# ── stale-unfilled cancel: re-anchor resting orders at the current mark ─────

def _age_minutes(iso_ts) -> Optional[float]:
    try:
        dt = datetime.fromisoformat(str(iso_ts))
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return (datetime.now(timezone.utc) - dt).total_seconds() / 60.0
    except (TypeError, ValueError):
        return None


def _live_price(ticker: str) -> Optional[float]:
    """Fresh live quote for re-anchoring a stale-cancelled EXIT (None-safe).

    Lazy import: tracker has no dependency on src.broker, so this is acyclic.
    """
    try:
        from src.performance.tracker import _fetch_price
        px = _fetch_price(ticker)
        return float(px) if px and px > 0 else None
    except Exception as e:
        logger.debug(f"[broker] live re-anchor fetch failed for {ticker}: {e}")
        return None


def _predates(iso_ts, boundary: datetime) -> bool:
    """True when *iso_ts* parses and is strictly before *boundary* (tz-aware)."""
    try:
        dt = datetime.fromisoformat(str(iso_ts))
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt < boundary
    except (TypeError, ValueError):
        return False


def _order_is_gone(broker: Broker, ref: str) -> bool:
    """True when *ref* is neither working nor in today's fills — i.e. the
    order is dead at the broker (typically a DAY LMT expired at the session
    close). False when it's still working, has fills, or the broker can't be
    read (unknown → leave the leg alone this tick)."""
    if not ref:
        return False
    try:
        if any((o.client_ref or "") == ref for o in broker.get_open_orders()):
            return False
        if any((f.client_ref or "") == ref and (f.filled_qty or 0) > 0
               for f in broker.get_fills()):
            return False
        return True
    except Exception as e:
        logger.debug(f"[broker] dead-order check failed for {ref}: {e}")
        return False


def _cancel_stale_unfilled(broker: Broker, trades: List[dict], report: dict,
                           positions: Optional[dict] = None,
                           sync_started: Optional[datetime] = None) -> bool:
    """Cancel unfilled orders so nothing works the book on a previous tick's price.

    Tick-scoped mode (``broker_tick_scoped_orders``, default): an order lives
    exactly one tick. Anything submitted before THIS sync started and still
    unfilled was priced from a previous tick's data — cancel it and clear the
    leg so the entry/exit pass below resubmits THIS tick re-anchored at the
    current mark. Entries resubmit only when the trade survived this tick's
    signal pass (``monitor_open_positions`` runs before sync; a closed trade's
    entry is killed by cancel-on-close instead), so every working order always
    reflects the current tick's decision AND price. The age rule
    (``broker_unfilled_cancel_minutes``) is the fallback when tick-scoping is
    off, and an additional upper bound when it's on.

    Partial fills are left working (cancelling the remainder would strand a
    mismatched position). A cancel that loses the race to a fill is left for
    the fill-refresh pass.

    DEAD orders: every off-RTH submission is a DAY-limited LMT that IBKR
    expires at the 20:00 session close — ``cancel_order`` then finds nothing
    working, and without handling the leg would rest as a zombie 'Submitted'
    forever, never resubmitted. When the cancel fails AND the ref is neither
    working nor in today's fills, the order is gone: clear the leg as Expired
    so the pass below re-sends it. ENTRY legs are only cleared when no broker
    position backs them — the fills feed is day-scoped, so a fill from late
    yesterday is invisible here and resubmitting against an existing position
    would double it (the backed leg stays parked; the ledger already owns the
    position). A dead EXIT with no position left is stamped terminal instead
    of resubmitted — there is nothing to flatten, and an exit sized from the
    recorded entry fill would open a fresh short.
    """
    tick_scoped = bool(settings.broker_tick_scoped_orders)
    max_min = int(settings.broker_unfilled_cancel_minutes)
    if not tick_scoped and max_min <= 0:
        return False
    changed = False
    for t in trades:
        for prefix, intent, want_status in (("broker_", "ENTRY", "OPEN"),
                                            ("broker_exit_", "EXIT", "CLOSED")):
            if t.get("status") != want_status:
                continue
            if not t.get(f"{prefix}order_id"):
                continue
            status = t.get(f"{prefix}status") or ""
            if status in _TERMINAL_STATUSES or status in _RESTORED_STATUSES:
                continue
            if (t.get(f"{prefix}fill_qty") or 0) > 0:
                continue
            sub_at = t.get(f"{prefix}submitted_at")
            age = _age_minutes(sub_at)
            stale = bool(tick_scoped and sync_started and _predates(sub_at, sync_started))
            if not stale and max_min > 0 and age is not None and age >= max_min:
                stale = True
            if not stale:
                continue
            ref = t.get(f"{prefix}client_ref")
            try:
                cancelled = bool(ref) and broker.cancel_order(ref)
            except Exception as e:
                logger.warning(f"[broker] cancel_order {ref} raised: {e}")
                cancelled = False
            final_status = "STALE_CANCELLED"
            if not cancelled:
                if not _order_is_gone(broker, ref):
                    continue   # working or filled — the fill-refresh pass owns it
                expected_sign = 1 if t["action"] == "BUY" else -1
                held = (positions or {}).get(t["ticker"])
                backed = bool(held and held.quantity * expected_sign > 0)
                if intent == "ENTRY" and backed:
                    logger.info(
                        f"[broker] entry {t['ticker']}: order {ref} is dead but a "
                        "position backs it (possible prior-day fill outside the "
                        "day-scoped executions feed) — leaving the leg parked"
                    )
                    continue
                if intent == "EXIT" and not backed:
                    # Nothing held: the position is already gone (filled on an
                    # earlier day, or drift-flattened). Resubmitting an exit
                    # sized from the recorded entry fill would open a short.
                    t[f"{prefix}status"] = "Cancelled"
                    t[f"{prefix}cancel_reason"] = "expired_nothing_held"
                    changed = True
                    logger.info(
                        f"[broker] exit {t['ticker']}: order {ref} is dead and "
                        "nothing is held — stamped terminal, not resubmitted"
                    )
                    continue
                final_status = "EXPIRED"
            side = _entry_side(t["action"]) if intent == "ENTRY" else _exit_side(t["action"])
            _record_order(
                report, event="STALE_CANCEL", intent=intent, ticker=t["ticker"],
                side=side, order_type="LMT",
                requested_qty=t.get(f"{prefix}requested_qty"), filled_qty=0,
                model_price=None, limit_price=None, fill_price=None,
                commission=None, status=final_status, ok=True, error=None,
                order_id=t.get(f"{prefix}order_id"), client_ref=ref,
                submitted_at=None,
            )
            report["stale_cancels"] = report.get("stale_cancels", 0) + 1
            hist = t.get(f"{prefix}cancelled_order_ids") or []
            t[f"{prefix}cancelled_order_ids"] = hist + [t[f"{prefix}order_id"]]
            t[f"{prefix}order_id"] = None
            t[f"{prefix}status"] = final_status
            t[f"{prefix}resubmit_n"] = int(t.get(f"{prefix}resubmit_n") or 0) + 1
            changed = True
            age_s = f"{age:.0f} min" if age is not None else "unknown age"
            logger.info(
                f"[broker] {'expired' if final_status == 'EXPIRED' else 'stale unfilled'} "
                f"{intent} {t['ticker']} ({age_s}, from a previous tick) — "
                "resubmitting re-anchored at the current mark"
            )
    return changed


# ── drift auto-reconciliation: flatten orphan positions ─────────────────────

# A same-side drift flatten that FILLED within this window blocks a resubmit:
# the fill may postdate the positions snapshot (``ib.positions()`` is a
# subscription cache, so "position still shown" cannot be trusted against a
# fill this recent). The window must comfortably exceed the minutes between
# the snapshot read at sync start and the drift pass at its end, plus one tick
# interval for margin. A same-side fill OLDER than this, alongside a fresh
# snapshot that still shows the position, is genuine residue (e.g. a
# partial-fill leftover) — the flatten proceeds.
_FLATTEN_FILL_GUARD_MINUTES = 90


def _minutes_since(iso: Optional[str]) -> Optional[float]:
    """Minutes elapsed since an ISO timestamp; None when missing/unparseable."""
    if not iso:
        return None
    try:
        dt = datetime.fromisoformat(str(iso))
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return (datetime.now(timezone.utc) - dt).total_seconds() / 60.0
    except Exception:
        return None


def _recent_flatten_fill(broker: Broker, ref_prefix: str, side: str) -> Optional[str]:
    """client_ref of a same-side drift flatten filled within the guard window
    (a missing/unparseable fill time counts as recent — conservative), else None."""
    for f in broker.get_fills():
        if not (f.client_ref or "").startswith(ref_prefix):
            continue
        if f.side != side or f.filled_qty <= 0:
            continue
        age = _minutes_since(f.last_fill_at)
        if age is None or age <= _FLATTEN_FILL_GUARD_MINUTES:
            return f.client_ref
    return None


def _flatten_orphan(broker: Broker, ticker: str, broker_qty: float,
                    outside_rth: bool,
                    report: dict, run_id: Optional[str]) -> Optional[OrderResult]:
    """Submit a price-capped close for an orphan broker position.

    The ledger is the source of truth — a position it cannot explain is
    closed, never adopted (an adopted trade the strategy never signalled
    would contaminate every performance metric). Always a marketable LMT
    capped at ``broker_limit_cap_bps`` from a FRESH live quote: an unattended
    cleanup must carry price protection, whatever ``broker_order_type`` says.
    A flatten that rests unfilled is cancelled and re-anchored on the next
    tick (each cycle gets a fresh quote + ref), so convergence happens in
    bounded ≤cap steps. Returns None when skipped this tick (no quote,
    fractional position, a cancel race lost to a fill, or a same-side flatten
    already filled within the guard window).
    """
    qty = int(abs(broker_qty))
    if qty <= 0:
        logger.warning(
            f"[broker] drift {ticker}: fractional position ({broker_qty}) — "
            "cannot be closed via the API; flatten it manually in TWS"
        )
        return None

    # A flatten submitted on an earlier tick may still be working (its cap
    # was never reached). Cancel it and re-anchor at a fresh quote; losing
    # the cancel race to a fill means the position is gone — skip.
    ref_prefix = f"drift-{ticker}-"
    try:
        for o in broker.get_open_orders():
            if o.client_ref.startswith(ref_prefix):
                if not broker.cancel_order(o.client_ref):
                    logger.info(
                        f"[broker] drift {ticker}: working flatten {o.client_ref} "
                        "filled during cancel — nothing further this tick"
                    )
                    return None
                logger.info(
                    f"[broker] drift {ticker}: stale flatten {o.client_ref} "
                    "cancelled — re-anchoring at a fresh quote"
                )
                break
    except Exception as e:
        logger.debug(f"[broker] drift open-order check failed for {ticker}: {e}")

    side = "SELL" if broker_qty > 0 else "BUY"

    # A flatten that already FILLED is invisible to the open-orders pass above,
    # and the positions snapshot may predate the fill — submitting again on the
    # same side would not flatten, it would FLIP the position (TRUP 2026-06-11:
    # a short's BUY flatten filled after the previous tick's poll; the stale
    # snapshot still showed −32 and a second BUY made it +32 long). Any
    # same-side drift fill within the guard window → stand down this tick and
    # let the next tick's fresh snapshot decide. An opposite-side fill never
    # blocks — correcting an over-flatten needs the other side. Fail closed:
    # when fills can't be read, don't trade blind.
    try:
        prior = _recent_flatten_fill(broker, ref_prefix, side)
    except Exception as e:
        logger.warning(
            f"[broker] drift {ticker}: cannot verify prior flatten fills ({e}) — "
            "skipping this tick rather than risking a duplicate"
        )
        return None
    if prior:
        logger.info(
            f"[broker] drift {ticker}: {side} flatten {prior} already filled within "
            f"the last {_FLATTEN_FILL_GUARD_MINUTES} min — positions snapshot is "
            "likely stale; skipping this tick"
        )
        return None

    live = _live_price(ticker)
    if not live:
        logger.warning(
            f"[broker] drift {ticker}: no live quote for a price-capped flatten — "
            "skipping this tick (reported only)"
        )
        return None

    limit = _limit_price_for(side, live, outside_rth)
    req = OrderRequest(
        ticker=ticker, side=side, quantity=qty,
        order_type="LMT", limit_price=limit,
        client_ref=f"{ref_prefix}{run_id or _utcnow_iso()}",
        intent="EXIT", outside_rth=outside_rth,
    )
    res = _submit_with_retry(broker, req, model_price=live, report=report,
                             intent="DRIFT_FLATTEN")
    _record_order(
        report, event="DRIFT_FLATTEN", intent="EXIT", ticker=ticker, side=side,
        order_type="LMT", requested_qty=qty, filled_qty=res.filled_qty,
        model_price=live, limit_price=limit, fill_price=res.avg_fill_price,
        commission=res.commission, status=res.status, ok=res.ok, error=res.error,
        order_id=res.order_id, client_ref=res.client_ref,
        submitted_at=res.submitted_at,
    )
    if res.ok:
        logger.info(
            f"[broker] drift {ticker}: auto-flatten submitted — {side} {qty} "
            f"LMT @{limit} (live {live}); the broker converges to the ledger"
        )
    else:
        report["errors"].append(f"drift flatten {ticker}: {res.error}")
        logger.warning(f"[broker] drift {ticker}: flatten failed — {res.error}")
    return res


# ── drift prevention: never leave a working ENTRY behind a closed trade ─────

def _cancel_entries_for_closed(broker: Broker, trades: List[dict], report: dict) -> bool:
    """Cancel the still-working ENTRY order of any trade the ledger has CLOSED.

    This is the orphan factory: the ledger closes a trade (signal flip, decay
    exit) while its capped entry LMT is still resting at the broker. Nothing
    else would ever cancel that order — the stale-cancel pass only manages
    OPEN trades — so a later fill would create a position with no open trade
    behind it (permanent drift). Cancelling here closes the gap; any part
    that DID fill before the cancel is flattened by the exit pass right
    after (which sizes from the actual held position).
    """
    changed = False
    for t in trades:
        if t.get("status") != "CLOSED":
            continue
        if not t.get("broker_order_id"):
            continue
        status = t.get("broker_status") or ""
        if status in _TERMINAL_STATUSES or status in _RESTORED_STATUSES:
            continue
        req_qty = int(t.get("broker_requested_qty") or 0)
        fill_qty = int(t.get("broker_fill_qty") or 0)
        if req_qty and fill_qty >= req_qty:
            continue   # fully filled, just not stamped terminal yet — fill refresh's job
        ref = t.get("broker_client_ref")
        try:
            cancelled = bool(ref) and broker.cancel_order(ref)
        except Exception as e:
            logger.warning(f"[broker] cancel-on-close {ref} raised: {e}")
            cancelled = False
        if not cancelled:
            # Unknown at the broker (already terminal there) or lost the race
            # to a fill — the fill-refresh pass reconciles whichever it was.
            continue
        t["broker_status"] = "Cancelled"
        t["broker_cancel_reason"] = "closed_before_fill"
        changed = True
        report["entry_cancels_on_close"] = report.get("entry_cancels_on_close", 0) + 1
        _record_order(
            report, event="CANCEL_ON_CLOSE", intent="ENTRY", ticker=t["ticker"],
            side=_entry_side(t["action"]), order_type="LMT",
            requested_qty=req_qty or None, filled_qty=fill_qty,
            model_price=None, limit_price=None, fill_price=None, commission=None,
            status="Cancelled", ok=True, error=None,
            order_id=t.get("broker_order_id"), client_ref=ref, submitted_at=None,
        )
        logger.info(
            f"[broker] trade {t['ticker']} closed with its entry still working "
            f"({fill_qty}/{req_qty} filled) — entry order cancelled"
        )
    return changed


def sync(broker: Optional[Broker] = None, trades: Optional[List[dict]] = None,
         run_id: Optional[str] = None) -> dict:
    """Reconcile the broker against the internal ledger. Returns a report dict.

    No-op (empty report, ok=True) when broker_mode=off.
    """
    report = _new_report()
    report["run_id"] = run_id
    # Boundary for tick-scoped order lifetime: anything submitted before this
    # instant belongs to a previous tick and must not keep working the book.
    sync_started = datetime.now(timezone.utc)
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

        # ── STALE-UNFILLED CANCEL: tick-scoped order lifetime ─────────────
        # Unfilled orders from a previous tick are cancelled (or recognized as
        # dead/expired) and re-decided below from THIS tick's data and price.
        changed = _cancel_stale_unfilled(broker, trades, report,
                                         positions=positions,
                                         sync_started=sync_started) or changed

        # ── DRIFT PREVENTION: cancel working entries behind CLOSED trades ──
        changed = _cancel_entries_for_closed(broker, trades, report) or changed

        # Extended-session submissions: IBKR rejects MKT outside regular hours,
        # so off-RTH ticks force a marketable LMT (model price ± cap) flagged
        # outsideRth. An order submitted overnight rests until the 04:00
        # pre-market open — consistent with the tracker snapping the fill
        # timestamp to the next tradeable moment. RTH keeps the configured type.
        from src.performance.market_calendar import current_session
        outside_rth = current_session() != "rth"
        use_limit = settings.broker_order_type == "LMT" or outside_rth
        order_type = "LMT" if use_limit else "MKT"
        if outside_rth:
            logger.info(
                "[broker] off-RTH tick — orders submitted as marketable LMT "
                f"(extended cap {settings.broker_limit_cap_bps_extended:g} bp) "
                "with outsideRth=True"
            )

        # ── ENTRIES: OPEN trades not yet sent to the broker ──────────────
        # One broker order per client_ref, ever: the ref is the idempotency
        # key (→ IBKR orderRef) and IBKR does NOT dedupe orderRef, so two
        # ledger trades sharing a ref (duplicate LLM recommendations,
        # 2026-06-11 XLE) must produce exactly ONE order. The set spans the
        # entry AND exit passes of this tick; the DUPLICATE_REF_NOT_SUBMITTED
        # status makes the skip durable on later ticks.
        submitted_refs: set = set()
        for t in open_trades:
            if t.get("broker_order_id"):
                continue  # idempotent — already submitted
            if t.get("broker_status") == "DUPLICATE_REF_NOT_SUBMITTED":
                continue  # twin of an already-submitted trade — never re-sent
            price = float(t.get("entry_price") or t.get("current_price") or 0.0)
            resubmit_n = int(t.get("broker_resubmit_n") or 0)
            if resubmit_n and t.get("current_price"):
                # Stale-cancelled leg: re-anchor at the latest mark (refreshed
                # by update_open_trades this tick) so the capped LMT follows
                # the market in bounded steps instead of resting at the old
                # entry price forever.
                price = float(t["current_price"])
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
            limit = _limit_price_for(side, price, outside_rth) if use_limit else None
            ref = t.get("recommendation_id") or f"{t.get('run_id', '')}-{t['ticker']}"
            if resubmit_n:
                ref = f"{ref}-r{resubmit_n}"   # fresh ref per resubmission cycle
            if ref in submitted_refs:
                logger.warning(
                    f"[broker] entry {t['ticker']}: client_ref {ref} already "
                    "submitted this tick (duplicate ledger trade) — not sending "
                    "a second order"
                )
                t["broker_status"] = "DUPLICATE_REF_NOT_SUBMITTED"
                changed = True
                continue
            res = _submit_with_retry(broker, OrderRequest(
                ticker=t["ticker"], side=side, quantity=qty,
                order_type=order_type, limit_price=limit,
                client_ref=ref,
                intent="ENTRY", outside_rth=outside_rth,
            ), model_price=price, report=report, intent="ENTRY")
            submitted_refs.add(ref)
            _apply_entry_result(t, res)
            changed = True
            _record_order(
                report, event="SUBMIT", intent="ENTRY", ticker=t["ticker"], side=side,
                order_type=order_type, requested_qty=qty,
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
        exited_this_tick: set = set()
        for t in closed_trades:
            if not t.get("broker_order_id") or t.get("broker_exit_order_id"):
                continue  # never entered via broker, or already exited
            if t.get("broker_exit_status") == "DUPLICATE_REF_NOT_SUBMITTED":
                continue  # twin's exit already flattens the shared position
            # Size from what the broker ACTUALLY holds when the positions
            # feed has a row: a recorded fill count that lagged (partial
            # known at exit time, the rest discovered later by fill refresh)
            # would otherwise leave a residual position with no open trade
            # behind it. The goal after a ledger close is a FLAT broker book
            # in that ticker. Sign-checked: a holding OPPOSITE to the trade's
            # direction is never blind-traded (the drift pass handles it).
            # No position row at all → fall back to the recorded fill qty so
            # a confirmed entry is always covered even when the feed lags; a
            # stale record at worst leaves a residue the drift pass flattens.
            held = positions.get(t["ticker"])
            expected_sign = 1 if t["action"] == "BUY" else -1
            if held is None:
                qty = int(t.get("broker_fill_qty") or 0)
            elif held.quantity == 0:
                qty = 0
            elif held.quantity * expected_sign < 0:
                logger.warning(
                    f"[broker] exit {t['ticker']}: held qty {held.quantity} has the "
                    f"WRONG SIGN for a closed {t['action']} — not trading it blind "
                    "(drift pass will handle the position)"
                )
                qty = 0
            else:
                qty = int(abs(held.quantity))
            if qty <= 0:
                t["broker_exit_status"] = "NOTHING_TO_CLOSE"
                changed = True
                continue
            side = _exit_side(t["action"])
            model = float(t.get("exit_price") or t.get("current_price") or 0.0)
            exit_resubmit_n = int(t.get("broker_exit_resubmit_n") or 0)
            if exit_resubmit_n:
                # Stale-cancelled exit: the sim's exit_price is frozen, so
                # re-anchor at a fresh live quote (the position MUST flatten;
                # each cycle's cap then tracks the current market).
                live = _live_price(t["ticker"])
                if live:
                    model = live
            limit = _limit_price_for(side, model, outside_rth) if use_limit else None
            ref = (t.get("recommendation_id") or t["ticker"]) + "-exit"
            if exit_resubmit_n:
                ref = f"{ref}-r{exit_resubmit_n}"
            if ref in submitted_refs:
                # Each twin sizes its exit from the FULL held position — a
                # second same-ref exit would not flatten, it would flip the
                # book short (2026-06-11 XLE: 18 held, 36 sold).
                logger.warning(
                    f"[broker] exit {t['ticker']}: client_ref {ref} already "
                    "submitted this tick (duplicate ledger trade) — the first "
                    "exit flattens the whole position; not sending a second"
                )
                t["broker_exit_status"] = "DUPLICATE_REF_NOT_SUBMITTED"
                changed = True
                continue
            res = _submit_with_retry(broker, OrderRequest(
                ticker=t["ticker"], side=side, quantity=qty,
                order_type=order_type, limit_price=limit,
                client_ref=ref,
                intent="EXIT", outside_rth=outside_rth,
            ), model_price=model, report=report, intent="EXIT")
            submitted_refs.add(ref)
            _apply_exit_result(t, res)
            changed = True
            _record_order(
                report, event="SUBMIT", intent="EXIT", ticker=t["ticker"], side=side,
                order_type=order_type, requested_qty=qty,
                filled_qty=res.filled_qty, model_price=model, limit_price=limit,
                fill_price=res.avg_fill_price, commission=res.commission,
                status=res.status, ok=res.ok, error=res.error,
                order_id=res.order_id, client_ref=res.client_ref,
                submitted_at=res.submitted_at,
            )
            if res.ok:
                report["exits_submitted"] += 1
                exited_this_tick.add(t["ticker"])
                _record_slippage(report, "EXIT", t["ticker"], side,
                                 model, res.avg_fill_price, res.commission)
            else:
                report["rejects"] += 1
                report["errors"].append(f"exit {t['ticker']}: {res.error}")

        # ── DRIFT: broker positions the ledger cannot explain ─────────────
        # Exemptions (not drift, just close latency): an OPEN trade owns the
        # position; a CLOSED trade's exit order is still working; or the exit
        # was submitted within THIS tick (the positions snapshot above
        # predates it, so the position legitimately still appears there).
        open_tickers = {t["ticker"] for t in open_trades}
        pending_exit_tickers = {
            t["ticker"] for t in closed_trades
            if t.get("broker_exit_order_id")
            and (t.get("broker_exit_status") or "") not in _TERMINAL_STATUSES
        }
        drift_action = (settings.broker_drift_action or "report").lower()
        if drift_action == "flatten" and settings.broker_mode == "ibkr_live":
            logger.warning(
                "[broker] broker_drift_action=flatten is refused in ibkr_live — "
                "auto-selling unexpected REAL positions is a human decision; "
                "downgrading to report-only"
            )
            drift_action = "report"

        for tk, p in positions.items():
            if (p.quantity == 0 or tk in open_tickers
                    or tk in pending_exit_tickers or tk in exited_this_tick):
                continue
            entry = {"ticker": tk, "broker_qty": p.quantity, "action": drift_action}
            report["drift"].append(entry)
            if drift_action != "flatten":
                continue
            res = _flatten_orphan(broker, tk, p.quantity, outside_rth,
                                  report, run_id)
            if res is None:
                # deliberate stand-down this tick (no quote, cancel race lost
                # to a fill, or a same-side flatten already filled) — not an error
                entry["action"] = "flatten_skipped"
            elif res.ok:
                report["drift_flattened"] += 1
                entry["action"] = "flatten_submitted"
            else:
                entry["action"] = "flatten_failed"

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
