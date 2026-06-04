"""Position sizing: convert (account equity, price, confidence multiplier) → shares.

shares = floor(equity × base_pct × size_multiplier / price)

``size_multiplier`` is the existing 1.0 / 1.5 / 2.0× confidence tier already stored
on every trade (``position_size_multiplier``). ``base_pct`` is the per-1.0× slice of
equity (default 5%). Risk caps (max concurrent positions, max gross exposure) are
enforced by ``within_caps`` in the reconciler before an entry is submitted.
"""
from __future__ import annotations

import math
from typing import Optional, Tuple

from config.settings import settings


def shares_for(equity: float, price: float, size_multiplier: float,
               base_pct: Optional[float] = None) -> int:
    """Whole-share quantity for one position. Returns 0 on bad inputs."""
    try:
        equity = float(equity)
        price = float(price)
        mult = max(0.0, float(size_multiplier if size_multiplier is not None else 1.0))
    except (TypeError, ValueError):
        return 0
    if equity <= 0 or price <= 0 or mult <= 0:
        return 0
    base = settings.broker_base_position_pct if base_pct is None else base_pct
    notional = equity * float(base) * mult
    return max(0, int(math.floor(notional / price)))


def within_caps(open_positions: int, gross_notional: float, equity: float) -> Tuple[bool, str]:
    """Risk gates checked before opening a new broker position.

    Returns (allowed, reason). ``reason`` is non-empty only when blocked.
    """
    if open_positions >= settings.broker_max_positions:
        return False, f"max_positions cap reached ({settings.broker_max_positions})"
    if equity > 0:
        gross_ratio = gross_notional / equity
        if gross_ratio > settings.broker_max_gross_exposure_pct:
            return False, (f"gross exposure {gross_ratio:.0%} exceeds cap "
                           f"{settings.broker_max_gross_exposure_pct:.0%}")
    return True, ""
