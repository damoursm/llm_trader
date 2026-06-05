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


def _round_shares(raw: float, mode: Optional[str] = None) -> int:
    """Fractional target → whole shares. IBKR's API rejects fractional equity (10243).

    "floor"   — strict budget: floor(raw); a name priced above one position skips (0).
    "nearest" — round half-up: a position still places if it's ≥ 0.5 share (priced up to
                ~2× the budget), so small bases don't needlessly skip reasonably-priced names.
    """
    if raw <= 0:
        return 0
    mode = (mode or settings.broker_share_rounding or "nearest").lower()
    if mode == "floor":
        return int(math.floor(raw))
    return int(math.floor(raw + 0.5))   # round half up


def shares_for(equity: float, price: float, size_multiplier: float,
               base_pct: Optional[float] = None, rounding: Optional[str] = None) -> int:
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
    return _round_shares(notional / price, rounding)


def shares_for_notional(base_notional: float, fx_to_usd: float, price: float,
                        size_multiplier: float, rounding: Optional[str] = None) -> int:
    """Whole shares for a fixed base notional on a USD-priced security.

    ``fx_to_usd`` converts 1 unit of the notional currency to USD (CAD→USD≈0.73;
    USD→USD=1.0). shares = round(base_notional × multiplier × fx_to_usd / price),
    rounded per ``broker_share_rounding`` (IBKR's API can't do fractional equity).
    """
    try:
        base = float(base_notional)
        fx = float(fx_to_usd)
        price = float(price)
        mult = max(0.0, float(size_multiplier if size_multiplier is not None else 1.0))
    except (TypeError, ValueError):
        return 0
    if base <= 0 or fx <= 0 or price <= 0 or mult <= 0:
        return 0
    budget_usd = base * fx * mult
    return _round_shares(budget_usd / price, rounding)


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
