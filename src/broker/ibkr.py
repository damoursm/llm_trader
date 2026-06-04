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
from src.broker.base import AccountSnapshot, Broker, OrderRequest, OrderResult, Position


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
        contract = Stock(ticker, "SMART", "USD")
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
            if currency != "USD":
                logger.warning(
                    f"[broker:ibkr] account base currency is {currency}, not USD — sizing "
                    "assumes USD. Set your IBKR paper account base currency to USD to avoid FX skew."
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
            for p in self._ib.positions(self.account or ""):
                out.append(Position(
                    ticker=p.contract.symbol,
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
            from ib_async import MarketOrder
            contract = self._qualify(req.ticker)
            order = MarketOrder(req.side, req.quantity)
            if req.client_ref:
                order.orderRef = req.client_ref      # idempotency / reconciliation tag
            if self.account:
                order.account = self.account

            trade = self._ib.placeOrder(contract, order)
            # RTH market orders fill within ~ms; poll a few seconds for terminal status.
            for _ in range(12):
                self._ib.sleep(1)
                if trade.orderStatus.status in ("Filled", "Cancelled", "ApiCancelled", "Inactive"):
                    break

            st = trade.orderStatus
            filled = int(float(st.filled or 0))
            ok = filled > 0 or st.status in ("Filled", "Submitted", "PreSubmitted")
            return OrderResult(
                ok=ok, ticker=req.ticker, side=req.side, requested_qty=req.quantity,
                filled_qty=filled,
                avg_fill_price=float(st.avgFillPrice) if st.avgFillPrice else None,
                order_id=str(trade.order.orderId), client_ref=req.client_ref,
                status=st.status, error=None if ok else (st.status or "not filled"),
            )
        except Exception as e:
            logger.warning(f"[broker:ibkr] submit_order {req.side} {req.quantity} {req.ticker} failed: {e}")
            return OrderResult(ok=False, ticker=req.ticker, side=req.side,
                               requested_qty=req.quantity, client_ref=req.client_ref,
                               status="ERROR", error=str(e))
