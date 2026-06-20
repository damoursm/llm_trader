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

# Per-run FX-conversion health. ``usd_per_unit`` records whether the LIVE quote
# was used or a fallback / assumed-1.0 was substituted, so a silent sizing-rate
# degradation (live FX feed down → stale config rate sizing every order) is
# surfaced through ``_collect_sources`` → run_sources (Data Quality tab) + the
# email/dashboard health banner instead of only scrolling past in the log.
_FX_HEALTH: dict = {"live": 0, "fallback": 0, "assumed_one": 0, "last_rate": None, "last_ccy": None}


def reset_fx_health() -> None:
    """Clear the per-run FX-conversion health counters (call at run start)."""
    _FX_HEALTH.update(live=0, fallback=0, assumed_one=0, last_rate=None, last_ccy=None)


def get_fx_health() -> dict:
    """Snapshot of this run's FX-conversion outcomes."""
    return dict(_FX_HEALTH)


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
        return 1.0          # no conversion needed — not an FX-health event
    rate = _live_rate(f"{ccy}USD=X", date.today().isoformat())
    if rate and rate > 0:
        _FX_HEALTH["live"] += 1
        _FX_HEALTH["last_rate"], _FX_HEALTH["last_ccy"] = rate, ccy
        return rate
    if ccy == "CAD":
        fb = float(settings.broker_fx_fallback_cad_usd)
        _FX_HEALTH["fallback"] += 1
        _FX_HEALTH["last_rate"], _FX_HEALTH["last_ccy"] = fb, ccy
        logger.warning(f"[broker:fx] live CAD→USD unavailable — using fallback {fb}")
        return fb
    _FX_HEALTH["assumed_one"] += 1
    _FX_HEALTH["last_rate"], _FX_HEALTH["last_ccy"] = 1.0, ccy
    logger.warning(f"[broker:fx] no {ccy}→USD rate available — assuming 1.0")
    return 1.0
