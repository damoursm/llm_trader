"""
NYSE TICK Index (^TICK) — market breadth / short-term exhaustion signal.

The NYSE TICK is the real-time count of NYSE-listed stocks trading on an uptick
minus those on a downtick.  It ranges from roughly −2000 to +2000 intraday.

Extreme readings are reliable short-term reversal signals (contrarian):
  > +1000  EXTREME_BULLS  → institutions buying en masse; short-term exhaustion;
                             contrarian BEARISH (fade the ramp)
  < -1000  EXTREME_BEARS  → panic selling cascade; short-term capitulation;
                             contrarian BULLISH (fade the flush)
  Both in same session    → WHIPSAW: institutions were active on both sides;
                             high-noise session; no directional edge
  Neither extreme         → NEUTRAL: breadth unremarkable

yfinance returns daily OHLC for ^TICK.  The High = session maximum TICK;
the Low = session minimum TICK.  We look at the most recent 5 trading days
to surface whether extremes are isolated or part of a pattern.

Cached daily (session closes don't change after the fact).
"""

import json
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Optional

import yfinance as yf
from loguru import logger

from config import settings
from src.models import TICKContext

CACHE_DIR = Path("cache")

_EXTREME_HIGH  = 1000.0   # above this → EXTREME_BULLS
_EXTREME_LOW   = -1000.0  # below this → EXTREME_BEARS
_LOOKBACK_DAYS = 5
_CACHE_STALE_DAYS = 5     # accept cache up to this many days old (covers weekends)


# ─────────────────────────────────────────────────────────────────────────────
# Cache
# ─────────────────────────────────────────────────────────────────────────────

def _cache_path() -> Path:
    return CACHE_DIR / f"tick_{date.today().isoformat()}.json"


def _load_cache() -> Optional[TICKContext]:
    """Load today's cache; if missing, fall back to any tick_*.json within 5 days."""
    today_path = _cache_path()
    if today_path.exists():
        try:
            ctx = TICKContext.model_validate(json.loads(today_path.read_text(encoding="utf-8")))
            logger.info(
                f"[tick] Loaded from cache — TICK high={ctx.tick_high}, low={ctx.tick_low}, "
                f"signal={ctx.signal}"
            )
            return ctx
        except Exception as e:
            logger.warning(f"[tick] Cache load failed: {e}")
            return None

    # Fallback: find the newest tick_*.json within 5 days
    candidates = sorted(CACHE_DIR.glob("tick_*.json"), reverse=True)
    for path in candidates:
        try:
            file_date = date.fromisoformat(path.stem.split("tick_")[1])
            age_days = (date.today() - file_date).days
            if age_days > _CACHE_STALE_DAYS:
                break  # files are sorted newest-first; nothing newer will follow
            ctx = TICKContext.model_validate(json.loads(path.read_text(encoding="utf-8")))
            logger.info(
                f"[tick] Loaded from cache ({age_days}d old) — "
                f"TICK high={ctx.tick_high}, low={ctx.tick_low}, signal={ctx.signal}"
            )
            return ctx
        except Exception:
            continue
    return None


def _save_cache(ctx: TICKContext) -> None:
    CACHE_DIR.mkdir(exist_ok=True)
    try:
        _cache_path().write_text(
            json.dumps(ctx.model_dump(mode="json"), indent=2, default=str),
            encoding="utf-8",
        )
    except Exception as e:
        logger.warning(f"[tick] Cache save failed: {e}")


# ─────────────────────────────────────────────────────────────────────────────
# Signal classification
# ─────────────────────────────────────────────────────────────────────────────

def _classify(tick_high: Optional[float], tick_low: Optional[float]):
    """Return (signal, direction) based on session extremes."""
    if tick_high is None or tick_low is None:
        return "UNKNOWN", "NEUTRAL"

    hit_high = tick_high >= _EXTREME_HIGH
    hit_low  = tick_low  <= _EXTREME_LOW

    if hit_high and hit_low:
        return "WHIPSAW", "NEUTRAL"
    if hit_high:
        return "EXTREME_BULLS", "BEARISH"   # contrarian — crowd over-extended
    if hit_low:
        return "EXTREME_BEARS", "BULLISH"   # contrarian — panic selling
    return "NEUTRAL", "NEUTRAL"


# ─────────────────────────────────────────────────────────────────────────────
# Public entry point
# ─────────────────────────────────────────────────────────────────────────────

def fetch_tick_context() -> Optional[TICKContext]:
    """
    Fetch the NYSE TICK daily OHLC from yfinance and derive a breadth signal.
    Returns TICKContext or None if data is unavailable.
    Cached daily.
    """
    cached = _load_cache()
    if cached is not None:
        return cached

    if not settings.enable_fetch_data:
        logger.debug("[tick] ENABLE_FETCH_DATA=false — skipping yfinance fetch")
        return None

    try:
        raw = yf.Ticker("^TICK").history(
            period=f"{_LOOKBACK_DAYS + 2}d",
            interval="1d",
            auto_adjust=False,
        )
    except Exception as e:
        logger.warning(f"[tick] yfinance fetch failed: {e}")
        return None

    if raw is None or raw.empty:
        logger.warning("[tick] ^TICK returned no data")
        return None

    # Drop rows where High/Low are NaN
    raw = raw.dropna(subset=["High", "Low"])
    if raw.empty:
        logger.warning("[tick] ^TICK has no usable rows after NaN drop")
        return None

    # Most recent session
    last_row = raw.iloc[-1]
    session_date = last_row.name.date() if hasattr(last_row.name, "date") else date.today()
    tick_high  = float(last_row["High"])
    tick_low   = float(last_row["Low"])
    tick_close = float(last_row["Close"]) if "Close" in last_row else None

    signal, direction = _classify(tick_high, tick_low)

    # Count extreme sessions in lookback window
    lookback = raw.tail(_LOOKBACK_DAYS)
    extreme_high_count = int((lookback["High"] >= _EXTREME_HIGH).sum())
    extreme_low_count  = int((lookback["Low"]  <= _EXTREME_LOW).sum())

    # Build summary
    parts = []
    if signal == "EXTREME_BULLS":
        parts.append(
            f"NYSE TICK reached +{tick_high:.0f} — aggressive institutional buying; "
            f"contrarian BEARISH (short-term exhaustion signal)."
        )
    elif signal == "EXTREME_BEARS":
        parts.append(
            f"NYSE TICK dropped to {tick_low:.0f} — panic selling cascade; "
            f"contrarian BULLISH (short-term capitulation signal)."
        )
    elif signal == "WHIPSAW":
        parts.append(
            f"NYSE TICK hit both extremes (high={tick_high:.0f}, low={tick_low:.0f}) — "
            f"institutions active on both sides; no directional edge."
        )
    else:
        parts.append(
            f"NYSE TICK unremarkable (high={tick_high:.0f}, low={tick_low:.0f}) — "
            f"no breadth extremes; market moving in orderly fashion."
        )

    if extreme_high_count >= 3:
        parts.append(
            f"{extreme_high_count}/{_LOOKBACK_DAYS} sessions showed TICK > +1000 — "
            f"sustained institutional buying pressure; late-cycle risk."
        )
    if extreme_low_count >= 3:
        parts.append(
            f"{extreme_low_count}/{_LOOKBACK_DAYS} sessions showed TICK < −1000 — "
            f"repeated panic flushes; distribution phase or forced selling."
        )

    ctx = TICKContext(
        tick_high=round(tick_high, 1),
        tick_low=round(tick_low, 1),
        tick_close=round(tick_close, 1) if tick_close is not None else None,
        session_date=session_date,
        signal=signal,
        direction=direction,
        extreme_high_count=extreme_high_count,
        extreme_low_count=extreme_low_count,
        report_date=date.today(),
        summary=" ".join(parts),
    )
    _save_cache(ctx)
    logger.info(
        f"[tick] TICK high={tick_high:.0f}  low={tick_low:.0f}  "
        f"→ {signal} ({direction})"
    )
    return ctx
