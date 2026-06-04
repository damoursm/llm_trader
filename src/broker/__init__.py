"""Broker package — real-execution layer (paper-first) for the pipeline.

``get_broker()`` returns a singleton Broker for the configured ``broker_mode``, or
``None`` when off. The reconciler (``reconcile.sync``) is the only caller in the
pipeline; everything is gated so ``broker_mode=off`` makes zero broker calls.
"""
from __future__ import annotations

from typing import Optional

from loguru import logger

from config.settings import settings
from src.broker.base import (
    AccountSnapshot,
    Broker,
    OrderRequest,
    OrderResult,
    Position,
)

__all__ = [
    "Broker", "OrderRequest", "OrderResult", "Position", "AccountSnapshot", "get_broker",
]

_LIVE_PORTS = {4001, 7496}   # IB Gateway live / TWS live — guard against paper→live misconfig

_BROKER: Optional[Broker] = None
_BROKER_MODE: Optional[str] = None


def get_broker(mode: Optional[str] = None) -> Optional[Broker]:
    """Singleton Broker for ``mode`` (defaults to ``settings.broker_mode``).

    off → None · dry_run → DryRunBroker · ibkr_paper/ibkr_live → IBKRBroker
    (paper vs live is the configured ``ibkr_port``: 4002 vs 4001).
    """
    global _BROKER, _BROKER_MODE
    mode = (mode or settings.broker_mode or "off").lower()

    if mode == "off":
        return None
    if _BROKER is not None and _BROKER_MODE == mode:
        return _BROKER

    if mode == "dry_run":
        from src.broker.dryrun import DryRunBroker
        _BROKER = DryRunBroker()
    elif mode in ("ibkr_paper", "ibkr_live"):
        from src.broker.ibkr import IBKRBroker
        if mode == "ibkr_paper" and settings.ibkr_port in _LIVE_PORTS:
            logger.warning(
                f"[broker] broker_mode=ibkr_paper but ibkr_port={settings.ibkr_port} is a LIVE "
                "port — refusing to start to avoid sending paper-intent orders to a live account."
            )
            return None
        _BROKER = IBKRBroker()
    else:
        logger.warning(f"[broker] unknown broker_mode '{mode}' — treating as off")
        return None

    _BROKER_MODE = mode
    return _BROKER
