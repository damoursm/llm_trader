"""Broker abstraction — the interface the pipeline talks to for real execution.

Implementations: DryRunBroker (logs, submits nothing) and IBKRBroker (Interactive
Brokers via ib_async). The pipeline never imports a concrete broker directly — it
goes through ``src.broker.get_broker()`` and the reconciler in ``reconcile.py``.

Everything here is broker-agnostic and dependency-free so it imports cleanly even
when ib_async isn't installed (broker_mode=off / dry_run paths).
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import List, Optional


def _utcnow_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass
class OrderRequest:
    ticker: str
    side: str                 # "BUY" | "SELL"
    quantity: int             # whole shares, > 0
    order_type: str = "MKT"
    client_ref: str = ""      # idempotency key (the trade's recommendation_id → IBKR orderRef)
    intent: str = "ENTRY"     # "ENTRY" | "EXIT" — for logging/reconciliation only


@dataclass
class OrderResult:
    ok: bool
    ticker: str
    side: str
    requested_qty: int
    filled_qty: int = 0
    avg_fill_price: Optional[float] = None
    order_id: Optional[str] = None       # broker order id (string for portability)
    client_ref: str = ""
    status: str = ""                     # raw broker status (Filled / Submitted / Rejected / …)
    commission: Optional[float] = None
    error: Optional[str] = None
    submitted_at: str = field(default_factory=_utcnow_iso)


@dataclass
class Position:
    ticker: str
    quantity: float                      # signed: + long, − short
    avg_cost: Optional[float] = None
    market_price: Optional[float] = None


@dataclass
class AccountSnapshot:
    equity: float
    cash: float
    buying_power: float
    account_id: str = ""
    currency: str = "USD"                # base currency of the equity figure


class Broker(ABC):
    """Minimal synchronous broker surface the reconciler needs."""

    name: str = "broker"

    @abstractmethod
    def connect(self) -> bool:
        """Establish/confirm a session. Returns True on success (idempotent)."""

    @abstractmethod
    def is_connected(self) -> bool: ...

    @abstractmethod
    def get_account(self) -> Optional[AccountSnapshot]: ...

    @abstractmethod
    def get_positions(self) -> List[Position]: ...

    @abstractmethod
    def submit_order(self, req: OrderRequest) -> OrderResult: ...

    def disconnect(self) -> None:  # optional; default no-op
        return None
