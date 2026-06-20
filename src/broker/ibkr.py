"""IBKRBroker — Interactive Brokers via ib_async + IB Gateway.

Paper (port 4002) and live (4001) use the IDENTICAL API; only the port/account
differ, so paper→live is a config swap. Trades US stocks/ETFs only (the entire
tradeable universe) — the CIRO restriction on API orders applies solely to
Canadian-listed securities, which this system never trades.

``ib_async`` is imported lazily so this module (and the whole src.broker package)
imports cleanly when the dependency isn't installed — only constructing/using an
IBKRBroker requires it.
"""
from __future__ import annotations

from typing import List, Optional

from loguru import logger

from config.settings import settings
from src.broker.base import (
    AccountSnapshot, Broker, FillSummary, OpenOrderInfo, OrderRequest, OrderResult, Position,
)

# Non-USD base currencies already flagged this process — the account config is
# static, so warn once instead of every get_account() call (29×/day in one log).
_WARNED_NON_USD_CCY: set = set()


def to_ib_symbol(ticker: str) -> str:
    """Map a yfinance-style ticker to IBKR symbology.

    US class shares use a separator in yfinance ("BRK-B", sometimes "BF.B")
    but a space in IBKR's ``symbol`` field ("BRK B"). Plain tickers pass
    through unchanged. (Preferred-share suffixes have messier mappings and are
    not handled — the universe doesn't trade them.)
    """
    return ticker.strip().upper().replace("-", " ").replace(".", " ")


def from_ib_symbol(symbol: str) -> str:
    """Inverse of ``to_ib_symbol``: IBKR "BRK B" → internal "BRK-B".

    Applied to symbols coming back from the broker (positions, fills) so they
    match the tickers stored on internal trades.
    """
    return symbol.strip().upper().replace(" ", "-")


class IBKRBroker(Broker):
    name = "ibkr"

    def __init__(self, host: Optional[str] = None, port: Optional[int] = None,
                 client_id: Optional[int] = None, account: Optional[str] = None):
        self.host = host or settings.ibkr_host
        self.port = int(port if port is not None else settings.ibkr_port)
        self.client_id = int(client_id if client_id is not None else settings.ibkr_client_id)
        self.account = account if account is not None else settings.ibkr_account
        self._ib = None

    # ── connection ────────────────────────────────────────────────────────
    def _get_ib(self):
        if self._ib is None:
            try:
                from ib_async import IB
            except ImportError as e:  # pragma: no cover - exercised only without the dep
                raise RuntimeError(
                    "ib_async is not installed — run `pip install ib_async` to use "
                    "broker_mode=ibkr_paper / ibkr_live."
                ) from e
            self._ib = IB()
        return self._ib

    def connect(self) -> bool:
        ib = self._get_ib()
        if ib.isConnected():
            return True
        try:
            ib.connect(self.host, self.port, clientId=self.client_id,
                       timeout=settings.ibkr_connect_timeout, readonly=False)
            logger.info(
                f"[broker:ibkr] connected {self.host}:{self.port} (clientId={self.client_id})"
            )
            return True
        except Exception as e:
            logger.warning(f"[broker:ibkr] connect failed ({self.host}:{self.port}): {e}")
            return False

    def is_connected(self) -> bool:
        return self._ib is not None and self._ib.isConnected()

    def disconnect(self) -> None:
        if self._ib is not None and self._ib.isConnected():
            self._ib.disconnect()

    def _account(self) -> str:
        if self.account:
            return self.account
        managed = self._ib.managedAccounts() or [""]
        return managed[0]

    def _qualify(self, ticker: str):
        from ib_async import Stock
        contract = Stock(to_ib_symbol(ticker), "SMART", "USD")
        self._ib.qualifyContracts(contract)
        return contract

    # ── reads ─────────────────────────────────────────────────────────────
    def get_account(self) -> Optional[AccountSnapshot]:
        if not self.is_connected():
            return None
        try:
            acct = self._account()
            rows = self._ib.accountValues(acct)
            vals = {r.tag: (r.value, r.currency) for r in rows}

            def num(tag: str) -> float:
                try:
                    return float(vals.get(tag, ("0", ""))[0])
                except (TypeError, ValueError):
                    return 0.0

            currency = vals.get("NetLiquidation", ("", "USD"))[1] or "USD"
            if currency != "USD" and currency not in _WARNED_NON_USD_CCY:
                _WARNED_NON_USD_CCY.add(currency)
                logger.warning(
                    f"[broker:ibkr] account base currency is {currency}, not USD — sizing "
                    "assumes USD. Set your IBKR paper account base currency to USD to avoid FX skew. "
                    "(warned once per run)"
                )
            return AccountSnapshot(
                equity=num("NetLiquidation"), cash=num("TotalCashValue"),
                buying_power=num("BuyingPower"), account_id=acct, currency=currency,
            )
        except Exception as e:
            logger.warning(f"[broker:ibkr] get_account failed: {e}")
            return None

    def get_positions(self) -> List[Position]:
        if not self.is_connected():
            return []
        out: List[Position] = []
        try:
            # ``ib.positions()`` reads a subscription cache that only advances
            # while the event loop runs — on the long-lived singleton
            # connection it can be a full tick stale by sync time (a flatten
            # that filled after the previous tick's last pump still shows as
            # held, and the drift pass would flatten it AGAIN, flipping the
            # position). A blocking re-request drains the queued events and
            # waits for positionEnd before we read.
            try:
                self._ib.reqPositions()
            except Exception as e:
                logger.warning(
                    f"[broker:ibkr] reqPositions refresh failed — reading "
                    f"cached positions: {e}"
                )
            for p in self._ib.positions(self.account or ""):
                out.append(Position(
                    ticker=from_ib_symbol(p.contract.symbol),
                    quantity=float(p.position),
                    avg_cost=float(p.avgCost) if p.avgCost else None,
                ))
        except Exception as e:
            logger.warning(f"[broker:ibkr] get_positions failed: {e}")
        return out

    # ── orders ────────────────────────────────────────────────────────────
    def submit_order(self, req: OrderRequest) -> OrderResult:
        if not self.is_connected():
            return OrderResult(ok=False, ticker=req.ticker, side=req.side,
                               requested_qty=req.quantity, client_ref=req.client_ref,
                               status="DISCONNECTED", error="broker not connected")
        try:
            from ib_async import LimitOrder, MarketOrder
            contract = self._qualify(req.ticker)
            if req.order_type == "LMT" and req.limit_price and req.limit_price > 0:
                order = LimitOrder(req.side, req.quantity, req.limit_price)
            else:
                order = MarketOrder(req.side, req.quantity)
            # Set the TIF explicitly. ib_async leaves Order.tif = '' (empty), which
            # makes IBKR's account order-preset fill it in as DAY and emit the noisy,
            # drift-prone warning 10349 ("Order TIF was set to DAY based on order
            # preset") — during which the order momentarily reports Cancelled in the
            # event feed before it actually fills (observed on MGNI 2026-06-17, filled
            # fine but logged a scary cancel). Sending DAY ourselves leaves nothing for
            # the preset to coerce. DAY is correct here — the settle pass cancels any
            # still-unfilled order each tick, so orders never need to live past the day.
            order.tif = "DAY"
            if req.outside_rth:
                # Extended-session eligibility. IBKR accepts this only on
                # limit orders; the reconciler forces LMT off-hours, so a MKT
                # falling through here would be an upstream bug surfaced by
                # IBKR's reject (captured in OrderResult.error).
                order.outsideRth = True
            if req.client_ref:
                order.orderRef = req.client_ref      # idempotency / reconciliation tag
            if self.account:
                order.account = self.account

            trade = self._ib.placeOrder(contract, order)
            # RTH market orders fill within ~ms; poll a few seconds for terminal status.
            # A still-working order (e.g. a marketable limit whose cap was exceeded)
            # exits this loop as Submitted — its fill/commission gets repaired by the
            # reconciler's fill-refresh pass on a later tick via get_fills().
            for _ in range(12):
                self._ib.sleep(1)
                if trade.orderStatus.status in ("Filled", "Cancelled", "ApiCancelled", "Inactive"):
                    break

            st = trade.orderStatus
            filled = int(float(st.filled or 0))
            ok = filled > 0 or st.status in ("Filled", "Submitted", "PreSubmitted")
            # Commission reports trail the fills by a moment; sum whatever has
            # arrived. 0.0 → None so the fill-refresh pass knows to repair it.
            commission = 0.0
            for f in (trade.fills or []):
                cr = getattr(f, "commissionReport", None)
                if cr is not None and cr.commission:
                    commission += float(cr.commission)
            return OrderResult(
                ok=ok, ticker=req.ticker, side=req.side, requested_qty=req.quantity,
                filled_qty=filled,
                avg_fill_price=float(st.avgFillPrice) if st.avgFillPrice else None,
                order_id=str(trade.order.orderId), client_ref=req.client_ref,
                status=st.status, commission=round(commission, 4) if commission else None,
                error=None if ok else (st.status or "not filled"),
            )
        except Exception as e:
            logger.warning(f"[broker:ibkr] submit_order {req.side} {req.quantity} {req.ticker} failed: {e}")
            return OrderResult(ok=False, ticker=req.ticker, side=req.side,
                               requested_qty=req.quantity, client_ref=req.client_ref,
                               status="ERROR", error=str(e))

    # ── fills (today's executions, for the reconciler's repair pass) ───────
    def get_fills(self) -> List[FillSummary]:
        """Today's executions aggregated per orderRef (= our client_ref).

        Uses ``reqExecutions``, which IBKR scopes to the current day — exactly
        the window that matters for repairing orders that filled after
        ``submit_order``'s poll (queued-overnight opens execute at today's
        open and appear here). Orders that filled on an earlier day while the
        app was down are not recoverable through this path; their positions
        still reconcile via ``get_positions``.

        Aggregation: IBKR maintains a running ``cumQty``/``avgPrice`` on each
        execution, so the latest execution per ref carries the order-level
        totals; commissions are summed across the individual fills.
        """
        if not self.is_connected():
            return []
        try:
            from ib_async import ExecutionFilter
            fills = self._ib.reqExecutions(ExecutionFilter())
        except Exception as e:
            logger.warning(f"[broker:ibkr] reqExecutions failed: {e}")
            return []

        agg: dict = {}
        max_cum: dict = {}
        for f in fills or []:
            ex = getattr(f, "execution", None)
            if ex is None:
                continue
            ref = (getattr(ex, "orderRef", "") or "").strip()
            if not ref:
                continue
            s = agg.get(ref)
            if s is None:
                s = FillSummary(client_ref=ref,
                                ticker=from_ib_symbol(f.contract.symbol),
                                side="BUY" if ex.side == "BOT" else "SELL",
                                order_id=str(ex.orderId), perm_id=str(ex.permId))
                agg[ref] = s
            cum = float(ex.cumQty or 0)
            if cum >= max_cum.get(ref, 0.0):   # latest execution carries the running totals
                max_cum[ref] = cum
                s.filled_qty = int(cum)
                s.avg_fill_price = float(ex.avgPrice) if ex.avgPrice else None
            cr = getattr(f, "commissionReport", None)
            if cr is not None and cr.commission:
                s.commission = round((s.commission or 0.0) + float(cr.commission), 4)
            t = getattr(f, "time", None)
            if t is not None:
                iso = t.isoformat() if hasattr(t, "isoformat") else str(t)
                if s.last_fill_at is None or iso > s.last_fill_at:
                    s.last_fill_at = iso
        return list(agg.values())

    # ── working orders (duplicate guard + stale-unfilled cancel pass) ──────
    def get_open_orders(self) -> List[OpenOrderInfo]:
        """Working orders for this API client, keyed by orderRef (= client_ref).

        ``openTrades()`` returns the orders this client id placed that have not
        reached a terminal state — exactly the set the reconciler needs to
        check "did my errored submission actually reach the broker?" and
        "which resting orders are stale?".
        """
        if not self.is_connected():
            return []
        out: List[OpenOrderInfo] = []
        try:
            for tr in self._ib.openTrades():
                ref = (getattr(tr.order, "orderRef", "") or "").strip()
                if not ref:
                    continue
                out.append(OpenOrderInfo(
                    client_ref=ref,
                    order_id=str(tr.order.orderId),
                    status=tr.orderStatus.status or "",
                    ticker=from_ib_symbol(tr.contract.symbol),
                    side=tr.order.action,
                ))
        except Exception as e:
            logger.warning(f"[broker:ibkr] get_open_orders failed: {e}")
        return out

    # ── real-time price (read-only; preferred live source for the tracker) ──
    def get_market_price(self, ticker: str) -> Optional[float]:
        """Real-time last/mark price via a blocking ``reqTickers`` snapshot.

        Uses IBKR's free Cboe One + IEX real-time feed (the same data that fills
        the orders), so the tracker's mark/decision price matches the execution
        venue. Returns None if not connected or no usable quote arrives — the
        caller then falls back to yfinance/Polygon. Prefers last trade, then the
        bid/ask midpoint (``marketPrice``), then the prior close; all NaN-filtered.
        """
        if not self.is_connected():
            return None
        try:
            contract = self._qualify(ticker)
            tickers = self._ib.reqTickers(contract)
            if not tickers:
                return None
            t = tickers[0]
            candidates = [getattr(t, "last", None), t.marketPrice(), getattr(t, "close", None)]
            for px in candidates:
                # px == px filters NaN (NaN != NaN); guard None and non-positive.
                if px is not None and px == px and float(px) > 0:
                    return float(px)
        except Exception as e:
            logger.debug(f"[broker:ibkr] get_market_price {ticker} failed: {e}")
        return None

    def cancel_order(self, client_ref: str) -> bool:
        """Cancel the working order tagged *client_ref*; True only on a
        confirmed cancel. An order that fills during the cancel race returns
        False — the caller leaves the trade leg intact and the fill-refresh
        pass records the execution instead."""
        if not self.is_connected() or not client_ref:
            return False
        try:
            for tr in self._ib.openTrades():
                if (getattr(tr.order, "orderRef", "") or "").strip() != client_ref:
                    continue
                self._ib.cancelOrder(tr.order)
                for _ in range(5):
                    self._ib.sleep(1)
                    if tr.orderStatus.status in ("Cancelled", "ApiCancelled", "Filled"):
                        break
                ok = tr.orderStatus.status in ("Cancelled", "ApiCancelled")
                logger.info(
                    f"[broker:ibkr] cancel {client_ref}: status={tr.orderStatus.status} "
                    f"({'confirmed' if ok else 'not cancelled'})"
                )
                return ok
        except Exception as e:
            logger.warning(f"[broker:ibkr] cancel_order {client_ref} failed: {e}")
        return False
