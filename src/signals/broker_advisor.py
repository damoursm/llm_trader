"""Broker advisor — IBKR account / short-borrow-aware directional method (group v1).

The first method in the broker-aware group. It scores a ticker's SHORT-BORROW
state: a name that is hard or expensive to short is a squeeze tell — shorts are
impractical/costly and crowded shorts get squeezed UP — so the score is BULLISH
(positive), following the universal convention (+ = predicted up, |score| =
confidence; see ``aggregator`` docstring). The practical effect is exactly the
intended one: a positive score FADES a SELL (it pulls ``combined_score`` away from a
bearish call) and mildly reinforces a squeeze-long. Easy-to-borrow names score 0
(no view) — like every other sparse method.

It is the only method aware of IBKR-unique data (short availability/cost). Decision-
only: it never places an order. Default OFF — it needs a live IBKR connection
(``broker_mode`` != off). Future methods in this group (account-aware sizing, the
IBKR scanner for new-ticker discovery, an LLM advisor) live alongside it here.
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional

from loguru import logger

from config.settings import settings
from src.broker.base import BorrowInfo


def compute_broker_advisor_score(info: Optional[BorrowInfo]) -> float:
    """Short-borrow → [-1, +1] directional score (+ = bullish squeeze tilt).

    Prefers the borrow FEE when present (expensive borrow → bullish), else falls
    back to shortable-shares AVAILABILITY (scarce/zero → bullish). Ample borrow or
    no data → 0.0 (no view). Capped at ``broker_advisor_max_score`` so a single
    squeeze tell can never dominate the combine."""
    if info is None:
        return 0.0
    cap = float(settings.broker_advisor_max_score)

    # Fee path (preferred where available): expensive to borrow → strong squeeze tilt.
    if info.fee_pct is not None and info.fee_pct > 0:
        scale = max(0.5, float(settings.broker_advisor_expensive_fee_pct))
        return round(cap * math.tanh(info.fee_pct / scale), 3)

    # Availability path: scarce/zero shortable shares → squeeze tilt.
    if info.shortable_shares is not None:
        ss = max(0.0, float(info.shortable_shares))
        hard = max(1.0, float(settings.broker_advisor_hard_shares))
        if ss >= hard:
            return 0.0                       # ample availability → easy to borrow → no view
        frac = 1.0 - ss / hard               # hard → 0, 0 shares → 1
        return round(cap * frac, 3)

    # Only a boolean flag: explicitly not-shortable → full tilt; otherwise no view.
    if info.is_shortable is False:
        return round(cap, 3)
    return 0.0


def fetch_borrow_context(tickers: List[str], broker=None) -> Dict[str, BorrowInfo]:
    """Fetch short-borrow info for ``tickers`` via the broker (fail-soft → {}).

    Gated by ``enable_broker_advisor`` + a non-off ``broker_mode``. Uses the shared
    broker singleton (``get_broker``) — a disconnected/dropped session self-heals
    inside ``get_short_borrow`` (IBKRBroker._ensure_connected, throttled), and the
    reconciler reuses that same connection later in the tick. Bounded by the
    caller (pass a capped set — held names + a universe slice — since each ticker
    costs a market-data request). Never raises."""
    if not settings.enable_broker_advisor or not tickers:
        return {}
    if not settings.broker_mode or settings.broker_mode == "off":
        return {}
    try:
        if broker is None:
            from src.broker import get_broker
            broker = get_broker()
        if broker is None:
            return {}
        data = broker.get_short_borrow(list(tickers)) or {}
        if data:
            hard = settings.broker_advisor_hard_shares
            n_hard = sum(1 for b in data.values()
                         if b.shortable_shares is not None and b.shortable_shares < hard)
            logger.info(f"[broker_advisor] short-borrow fetched for {len(data)} ticker(s); "
                        f"{n_hard} hard-to-borrow")
        return data
    except Exception as e:  # pragma: no cover - defensive (no gateway / connection)
        logger.debug(f"[broker_advisor] borrow fetch unavailable: {e}")
        return {}
