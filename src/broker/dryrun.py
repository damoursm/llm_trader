"""DryRunBroker — logs the orders that WOULD be placed, submits nothing.

The Phase-1 safety mode (``broker_mode=dry_run``). Account equity comes from
``settings.broker_paper_equity`` so the full sizing path is exercised; positions are
always empty (nothing is really held), so the reconciler treats each internal OPEN
trade as a fresh intended entry — and, because a (synthetic) order id is returned,
records it on the trade so the same intended order is logged once, not every tick.
"""
from __future__ import annotations

from typing import List, Optional

from loguru import logger

from config.settings import settings
from src.broker.base import AccountSnapshot, Broker, OrderRequest, OrderResult, Position


class DryRunBroker(Broker):
    name = "dry_run"

    def connect(self) -> bool:
        return True

    def is_connected(self) -> bool:
        return True

    def get_account(self) -> Optional[AccountSnapshot]:
        eq = float(settings.broker_paper_equity)
        return AccountSnapshot(equity=eq, cash=eq, buying_power=eq, account_id="DRYRUN", currency="USD")

    def get_positions(self) -> List[Position]:
        return []

    def submit_order(self, req: OrderRequest) -> OrderResult:
        logger.info(
            f"[broker:dry_run] WOULD {req.intent} {req.side} {req.quantity} {req.ticker} "
            f"({req.order_type}, ref={req.client_ref})"
        )
        return OrderResult(
            ok=True, ticker=req.ticker, side=req.side, requested_qty=req.quantity,
            filled_qty=req.quantity, avg_fill_price=None,
            order_id=f"dryrun-{req.client_ref or req.ticker}", client_ref=req.client_ref,
            status="DRYRUN",
        )
