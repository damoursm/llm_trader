"""FX helper — convert a notional currency to USD (the security currency).

US securities are priced in USD; a CAD-funded order for a USD stock needs the
CAD base notional converted to a USD share budget. The live rate comes from
yfinance (``<CCY>USD=X``, e.g. ``CADUSD=X`` ≈ 0.73), cached per calendar day, with
a configurable fallback so sizing stays robust if the quote is unavailable.
"""
from __future__ import annotations

import functools
from datetime import date
from typing import Optional

from loguru import logger

from config.settings import settings


@functools.lru_cache(maxsize=64)
def _live_rate(pair: str, _day: str) -> Optional[float]:
    """Latest close for an FX pair (e.g. 'CADUSD=X'). Cached by (pair, day)."""
    try:
        import yfinance as yf
        hist = yf.Ticker(pair).history(period="5d")
        if not hist.empty:
            close = hist["Close"].dropna()
            if not close.empty:
                return float(close.iloc[-1])
    except Exception as e:  # network/parse issue — caller falls back
        logger.debug(f"[broker:fx] {pair} fetch failed: {e}")
    return None


def usd_per_unit(currency: Optional[str]) -> float:
    """USD value of 1 unit of ``currency``. USD→1.0; others via ``<CCY>USD=X``.

    Falls back to ``broker_fx_fallback_cad_usd`` for CAD (and 1.0 for anything
    else) when the live quote is unavailable, so sizing never blocks on FX.
    """
    ccy = (currency or "USD").upper()
    if ccy == "USD":
        return 1.0
    rate = _live_rate(f"{ccy}USD=X", date.today().isoformat())
    if rate and rate > 0:
        return rate
    if ccy == "CAD":
        fb = float(settings.broker_fx_fallback_cad_usd)
        logger.warning(f"[broker:fx] live CAD→USD unavailable — using fallback {fb}")
        return fb
    logger.warning(f"[broker:fx] no {ccy}→USD rate available — assuming 1.0")
    return 1.0
