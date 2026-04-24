"""
VIX volatility regime and term structure.

Fetches six CBOE volatility indices from yfinance (no API key required):
  ^VIX9D  — 9-day expected volatility
  ^VIX    — 30-day S&P 500 implied volatility (the benchmark)
  ^VXN    — 30-day Nasdaq 100 implied volatility
  ^VVIX   — volatility of VIX itself (tail-risk gauge)
  ^VIX3M  — 3-month VIX
  ^VXMT   — 6-month mid-term VIX

Term structure (slope of the VIX curve):
  Contango     (VIX3M > VIX by >2pt):  normal; market expects future vol > current → calm
  Flat         (|VIX3M − VIX| ≤ 2pt):  transitional
  Backwardation(VIX > VIX3M by >2pt):  panic; near-term fear exceeds long-term → capitulation signal

VIX spot thresholds (contrarian — extreme fear signals potential lows):
  < 12              COMPLACENCY → contrarian BEARISH
  12–15             LOW → mild caution on aggressive longs
  15–20             NORMAL → no override
  20–25             ELEVATED → selective; prefer quality
  25–35             HIGH → start looking for reversal setups
  35–45             EXTREME_FEAR → fade SELLs, upgrade BUY conviction
  > 45              PANIC → very strong contrarian BUY (capitulation)

Cached daily (VIX moves intraday but daily context is sufficient for strategic overlay).
"""

import json
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Optional, Tuple

import yfinance as yf
from loguru import logger

from config import settings
from src.models import VIXContext

CACHE_DIR = Path("cache")

_INDICES = {
    "vix9d": "^VIX9D",
    "vix":   "^VIX",
    "vxn":   "^VXN",
    "vvix":  "^VVIX",
    "vix3m": "^VIX3M",
    "vix6m": "^VXMT",
}

# Contango / backwardation threshold (points)
_SLOPE_THRESHOLD = 2.0


# ─────────────────────────────────────────────────────────────────────────────
# Cache
# ─────────────────────────────────────────────────────────────────────────────

def _cache_path() -> Path:
    return CACHE_DIR / f"vix_{date.today().isoformat()}.json"


def _load_cache() -> Optional[VIXContext]:
    path = _cache_path()
    if not path.exists():
        return None
    try:
        ctx = VIXContext.model_validate(json.loads(path.read_text(encoding="utf-8")))
        logger.info(f"[vix] Loaded from cache — VIX={ctx.vix}, term={ctx.term_structure}")
        return ctx
    except Exception as e:
        logger.warning(f"[vix] Cache load failed: {e}")
        return None


def _save_cache(ctx: VIXContext) -> None:
    CACHE_DIR.mkdir(exist_ok=True)
    try:
        _cache_path().write_text(
            json.dumps(ctx.model_dump(mode="json"), indent=2, default=str),
            encoding="utf-8",
        )
    except Exception as e:
        logger.warning(f"[vix] Cache save failed: {e}")


# ─────────────────────────────────────────────────────────────────────────────
# Classification helpers
# ─────────────────────────────────────────────────────────────────────────────

def _classify_vix(vix: float) -> Tuple[str, str]:
    """Return (signal, direction). Direction is contrarian."""
    if vix > 45:
        return "PANIC", "BULLISH"
    if vix > 35:
        return "EXTREME_FEAR", "BULLISH"
    if vix > 25:
        return "HIGH", "BULLISH"
    if vix > 20:
        return "ELEVATED", "NEUTRAL"
    if vix > 15:
        return "NORMAL", "NEUTRAL"
    if vix > 12:
        return "LOW", "BEARISH"
    return "COMPLACENCY", "BEARISH"


def _classify_term_structure(slope: float) -> str:
    if slope > _SLOPE_THRESHOLD:
        return "CONTANGO"
    if slope < -_SLOPE_THRESHOLD:
        return "BACKWARDATION"
    return "FLAT"


def _fetch_price(symbol: str) -> Optional[float]:
    try:
        val = yf.Ticker(symbol).fast_info.last_price
        if val and float(val) > 0:
            return round(float(val), 2)
    except Exception:
        pass
    return None


# ─────────────────────────────────────────────────────────────────────────────
# Public entry point
# ─────────────────────────────────────────────────────────────────────────────

def fetch_vix_context() -> Optional[VIXContext]:
    """
    Fetch VIX, VXN, VVIX, and term structure indices from yfinance.
    Returns VIXContext or None if ^VIX itself is unavailable.
    Cached daily.
    """
    cached = _load_cache()
    if cached is not None:
        return cached

    if not settings.enable_fetch_data:
        logger.debug("[vix] ENABLE_FETCH_DATA=false — skipping yfinance fetch")
        return None

    # Fetch all indices
    values = {}
    for key, sym in _INDICES.items():
        values[key] = _fetch_price(sym)
        logger.debug(f"[vix] {sym}: {values[key]}")

    vix = values.get("vix")
    if vix is None:
        logger.warning("[vix] ^VIX unavailable — skipping VIX context")
        return None

    vix_signal, vix_direction = _classify_vix(vix)

    # Term structure: slope = VIX3M − VIX
    vix3m = values.get("vix3m")
    slope = round(vix3m - vix, 2) if vix3m else None
    term_structure = _classify_term_structure(slope) if slope is not None else "UNKNOWN"

    # Build summary
    vvix = values.get("vvix")
    vxn  = values.get("vxn")
    vix9d = values.get("vix9d")
    vix6m = values.get("vix6m")

    parts = [f"VIX at {vix:.1f} ({vix_signal})."]

    if slope is not None:
        ts_desc = {
            "CONTANGO":      f"Term structure in contango (VIX3M={vix3m:.1f}, slope={slope:+.1f}pt) — normal; market calm.",
            "FLAT":          f"Term structure flat (VIX3M={vix3m:.1f}, slope={slope:+.1f}pt) — transitional regime.",
            "BACKWARDATION": f"Term structure in BACKWARDATION (VIX3M={vix3m:.1f}, slope={slope:+.1f}pt) — near-term panic exceeds long-term fear; capitulation signal.",
        }.get(term_structure, "")
        parts.append(ts_desc)

    if vvix and vvix > 100:
        parts.append(f"VVIX={vvix:.1f} — elevated vol-of-vol signals high uncertainty about the VIX itself.")
    if vxn and vix and vxn > vix + 5:
        parts.append(f"VXN={vxn:.1f} significantly above VIX — tech sector experiencing higher relative fear.")

    ctx = VIXContext(
        vix=vix,
        vxn=vxn,
        vvix=vvix,
        vix9d=vix9d,
        vix3m=vix3m,
        vix6m=vix6m,
        term_structure=term_structure,
        slope_1m_3m=slope,
        vix_signal=vix_signal,
        vix_direction=vix_direction,
        report_date=date.today(),
        summary=" ".join(parts),
    )
    _save_cache(ctx)
    logger.info(
        f"[vix] VIX={vix:.1f} ({vix_signal}) | "
        f"Term={term_structure} (slope={slope:+.1f}pt) | "
        f"VVIX={vvix or 'N/A'}"
    )
    return ctx
