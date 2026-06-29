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
from typing import Dict, List, Optional


def _utcnow_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass
class OrderRequest:
    ticker: str
    side: str                 # "BUY" | "SELL"
    quantity: int             # whole shares, > 0
    order_type: str = "MKT"   # "MKT" | "LMT" (LMT requires limit_price)
    limit_price: Optional[float] = None   # LMT only: marketable-limit cap price
    client_ref: str = ""      # idempotency key (the trade's recommendation_id → IBKR orderRef)
    intent: str = "ENTRY"     # "ENTRY" | "EXIT" — for logging/reconciliation only
    outside_rth: bool = False # allow the order to trade in extended sessions
                              # (IBKR: order.outsideRth; requires LMT — IBKR
                              # rejects MKT outside regular hours)


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


@dataclass
class BorrowInfo:
    """Short-borrow availability / cost for one ticker (IBKR-unique data).

    A low/zero ``shortable_shares`` or a high ``fee_pct`` means the name is hard or
    expensive to short — a squeeze tell the ``broker_advisor`` method scores
    bullishly (which fades a SELL). All fields are optional/best-effort: IBKR
    streams shortable shares (generic tick 236) reliably; the annualised fee rate
    needs the stock-loan feed and may be ``None``."""
    ticker: str
    shortable_shares: Optional[float] = None   # shares available to short; low/0 = hard to borrow
    fee_pct: Optional[float] = None            # annualised borrow fee % (if available); high = expensive
    is_shortable: Optional[bool] = None        # convenience flag (None = unknown)


@dataclass
class OpenOrderInfo:
    """One working (not yet terminal) order at the broker, keyed by client_ref.

    Produced by ``Broker.get_open_orders()`` so the reconciler can (a) verify
    whether a submission that errored mid-flight actually reached the broker
    before retrying — resubmitting blind would double the position — and
    (b) find resting stale orders to cancel.
    """
    client_ref: str
    order_id: Optional[str] = None
    status: str = ""
    ticker: str = ""
    side: str = ""


@dataclass
class FillSummary:
    """Aggregated fills for one order, keyed by its client_ref (IBKR orderRef).

    Produced by ``Broker.get_fills()`` from the broker's execution feed so the
    reconciler can repair ``broker_fill_*`` fields on trades whose order didn't
    reach a terminal state inside ``submit_order``'s short poll window (queued
    overnight, partial fill, late commission report).
    """
    client_ref: str
    ticker: str = ""
    side: str = ""                       # "BUY" | "SELL"
    filled_qty: int = 0
    avg_fill_price: Optional[float] = None
    commission: Optional[float] = None   # total, account currency
    order_id: Optional[str] = None
    perm_id: Optional[str] = None        # broker-stable id (survives session restarts)
    last_fill_at: Optional[str] = None   # ISO timestamp of the latest execution


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

    def get_fills(self) -> List[FillSummary]:
        """Recent (typically today's) executions aggregated per client_ref.

        Default: none. Brokers that can report executions override this so the
        reconciler's fill-refresh pass can repair stale fill/commission fields.
        """
        return []

    def get_open_orders(self) -> List["OpenOrderInfo"]:
        """Working (non-terminal) orders at the broker, keyed by client_ref.

        Default: none. Used by the reconciler's duplicate guard before a
        retry and by the stale-unfilled cancel pass.
        """
        return []

    def cancel_order(self, client_ref: str) -> bool:
        """Cancel the working order carrying *client_ref*. Returns True only
        when the broker confirms the cancel — False when the order is unknown,
        already terminal, or filled during the cancel race (the fill-refresh
        pass picks that up). Default: unsupported."""
        return False

    def get_market_price(self, ticker: str) -> Optional[float]:
        """Latest real-time last/mark price for *ticker*, or None if unavailable.

        Default: unsupported (None). IBKRBroker overrides this so the tracker's
        live-price path can prefer the broker's real-time feed — the same source
        that fills the orders — over yfinance. Read-only; never places an order.
        """
        return None

    def get_short_borrow(self, tickers: List[str]) -> Dict[str, BorrowInfo]:
        """Short-borrow availability / cost per ticker (the ``broker_advisor`` input).

        Default: none ({}). IBKRBroker overrides this to stream shortable-shares
        (and fee where available). Read-only; never places an order.
        """
        return {}

    def disconnect(self) -> None:  # optional; default no-op
        return None
