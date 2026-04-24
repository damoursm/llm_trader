"""ICE BofA MOVE Index — Treasury market implied volatility (bond VIX).

The MOVE Index measures Treasury market volatility the same way VIX measures
equity volatility.  It is constructed from 1-month OTM options on 2Y, 5Y,
10Y, and 30Y Treasuries.  Historically MOVE spikes precede equity dislocations
by 1–5 days: rising bond vol compresses risk appetite, triggers de-leveraging,
and widens credit spreads before equity markets fully reprice the risk.

MOVE signal thresholds:
  < 60              CALM       → bond market unusually quiet; no signal
  60–80             LOW        → below-normal vol; mild constructive backdrop
  80–100            NORMAL     → typical regime; no override
  100–120           ELEVATED   → above-average; watch for equity spillover
  120–150           HIGH       → significant stress; BEARISH for equities
  150–200           EXTREME    → major disruption; strong BEARISH warning
  > 200             PANIC      → extreme crisis (GFC-level)

Spike detection (5-day change > 20pt): early warning even from a low base.

Cross-asset divergence:
  MOVE / VIX ratio normally 4–7×.  Ratio > 8 = bond fear outpacing equity
  complacency → divergence warning; equities have not yet caught up.

Ticker priority:
  1. ^MOVE  — ICE BofA MOVE Index (Yahoo Finance)
  2. VXTLT  — CBOE/ErisX 30-day Treasury ETF Volatility Index (fallback)

Cached daily (yfinance, no API key required).
"""

import json
from datetime import date, timedelta
from pathlib import Path
from typing import Optional, Tuple

import yfinance as yf
from loguru import logger

from config import settings
from src.models import MOVEContext

_CACHE_DIR    = Path("cache")
_CACHE_PREFIX = "move_"
_TICKER       = "^MOVE"

_SPIKE_THRESHOLD  = 20.0   # points — absolute 5d change flagged as a spike
_RATIO_WARN       = 8.0    # MOVE/VIX above this = divergence warning


# ─────────────────────────────────────────────────────────────────────────────
# Cache helpers
# ─────────────────────────────────────────────────────────────────────────────

def _cache_path(today: date) -> Path:
    return _CACHE_DIR / f"{_CACHE_PREFIX}{today.isoformat()}.json"


def _load_cache(today: date) -> Optional[MOVEContext]:
    p = _cache_path(today)
    if not p.exists():
        return None
    try:
        ctx = MOVEContext.model_validate(json.loads(p.read_text(encoding="utf-8")))
        logger.info(f"[move] Loaded from cache — MOVE={ctx.move}, signal={ctx.signal}")
        return ctx
    except Exception as e:
        logger.warning(f"[move] Cache load failed: {e}")
        return None


def _save_cache(ctx: MOVEContext) -> None:
    _CACHE_DIR.mkdir(exist_ok=True)
    try:
        _cache_path(ctx.report_date).write_text(
            json.dumps(ctx.model_dump(mode="json"), indent=2, default=str),
            encoding="utf-8",
        )
    except Exception as e:
        logger.warning(f"[move] Cache save failed: {e}")


# ─────────────────────────────────────────────────────────────────────────────
# Classification helpers
# ─────────────────────────────────────────────────────────────────────────────

def _classify_move(level: float) -> Tuple[str, str]:
    """Return (signal, direction). Direction is the equity implication."""
    if level > 200:
        return "PANIC", "BEARISH"
    if level > 150:
        return "EXTREME", "BEARISH"
    if level > 120:
        return "HIGH", "BEARISH"
    if level > 100:
        return "ELEVATED", "BEARISH"
    if level > 80:
        return "NORMAL", "NEUTRAL"
    if level > 60:
        return "LOW", "NEUTRAL"
    return "CALM", "NEUTRAL"


def _pct_return(series, n: int) -> Optional[float]:
    if series is None or len(series) <= n:
        return None
    start = float(series.iloc[-(n + 1)])
    end   = float(series.iloc[-1])
    if start == 0:
        return None
    return round((end - start), 2)   # absolute point change (not %) for vol indices


# ─────────────────────────────────────────────────────────────────────────────
# VIX cache reader (to compute MOVE/VIX ratio without re-fetching)
# ─────────────────────────────────────────────────────────────────────────────

def _read_vix_from_cache(today: date) -> Optional[float]:
    """Read VIX spot from today's VIX cache file, if it exists."""
    vix_cache = Path("cache") / f"vix_{today.isoformat()}.json"
    if not vix_cache.exists():
        return None
    try:
        data = json.loads(vix_cache.read_text(encoding="utf-8"))
        return float(data["vix"]) if data.get("vix") else None
    except Exception:
        return None


# ─────────────────────────────────────────────────────────────────────────────
# Public entry point
# ─────────────────────────────────────────────────────────────────────────────

def _fetch_current_price() -> Optional[float]:
    """Get current MOVE level via fast_info (avoids heavy download when rate-limited)."""
    try:
        val = yf.Ticker(_TICKER).fast_info.last_price
        if val and float(val) > 0:
            return round(float(val), 2)
    except Exception:
        pass
    return None


def fetch_move_context(today: Optional[date] = None) -> Optional[MOVEContext]:
    """Fetch ^MOVE Index and compute bond-vol signals.

    Tries yf.download() for full history first (5d/20d avg).  If download
    fails (rate limit or weekend), falls back to fast_info for the current
    level only.  Returns None if data is completely unavailable.  Cached daily.
    """
    if today is None:
        today = date.today()

    cached = _load_cache(today)
    if cached is not None:
        return cached

    if not settings.enable_fetch_data:
        logger.debug("[move] ENABLE_FETCH_DATA=false — skipping yfinance fetch")
        return None

    start = (today - timedelta(days=40)).strftime("%Y-%m-%d")
    end   = today.strftime("%Y-%m-%d")

    raw = None
    try:
        data = yf.download(
            _TICKER,
            start=start,
            end=end,
            auto_adjust=True,
            progress=False,
        )
        if data is not None and not data.empty:
            # Single-ticker download returns flat columns
            closes = (
                data["Close"][_TICKER].dropna()
                if hasattr(data.columns, "levels")
                else data["Close"].dropna()
            )
            if len(closes) >= 3:
                raw = closes
                logger.debug(f"[move] Fetched {len(closes)} rows via download")
    except Exception as e:
        logger.debug(f"[move] download failed: {e}")

    # Fall back to fast_info for current price only
    move: Optional[float] = None
    if raw is not None and not raw.empty:
        move = round(float(raw.iloc[-1]), 2)
    else:
        move = _fetch_current_price()
        if move is None:
            logger.warning("[move] ^MOVE unavailable — skipping MOVE context")
            return None
        logger.debug(f"[move] Using fast_info fallback: MOVE={move}")

    # Historical comparisons
    move_5d_ago  = float(raw.iloc[-6]) if len(raw) >= 6  else None
    move_20d_avg = float(raw.tail(20).mean()) if len(raw) >= 10 else None

    spike_5d = round(move - move_5d_ago, 2) if move_5d_ago is not None else None
    is_spiking = abs(spike_5d) > _SPIKE_THRESHOLD if spike_5d is not None else False

    signal, direction = _classify_move(move)

    # A sharp downward spike from an extreme level can be constructive
    if is_spiking and spike_5d is not None and spike_5d < -_SPIKE_THRESHOLD and signal in ("HIGH", "EXTREME", "PANIC"):
        direction = "NEUTRAL"  # mean reversion / de-escalation underway

    # Cross-asset MOVE/VIX ratio
    vix         = _read_vix_from_cache(today)
    move_vix_ratio = round(move / vix, 2) if vix and vix > 0 else None

    # Build summary
    parts = [f"MOVE Index at {move:.1f} ({signal})."]
    if move_20d_avg:
        vs_avg = move - move_20d_avg
        parts.append(
            f"{'Above' if vs_avg > 0 else 'Below'} 20d avg ({move_20d_avg:.1f}) "
            f"by {abs(vs_avg):.1f}pt."
        )
    if is_spiking and spike_5d is not None:
        direction_word = "surged" if spike_5d > 0 else "dropped"
        parts.append(
            f"Bond vol {direction_word} {spike_5d:+.1f}pt in 5 days — "
            f"{'warning: equity dislocation risk rising' if spike_5d > 0 else 'de-escalation underway'}."
        )
    if move_vix_ratio and move_vix_ratio > _RATIO_WARN:
        parts.append(
            f"MOVE/VIX ratio = {move_vix_ratio:.1f}× (above {_RATIO_WARN:.0f}× warning) — "
            f"bond market pricing significantly more stress than equities; divergence risk."
        )
    if signal in ("HIGH", "EXTREME", "PANIC"):
        parts.append("Elevated MOVE typically precedes equity selling by 1–5 days.")

    summary = " ".join(parts)
    move = round(move, 2)

    ctx = MOVEContext(
        move=move,
        move_5d_ago=round(move_5d_ago, 2) if move_5d_ago else None,
        move_20d_avg=round(move_20d_avg, 2) if move_20d_avg else None,
        spike_5d=spike_5d,
        is_spiking=is_spiking,
        signal=signal,
        direction=direction,
        move_vix_ratio=move_vix_ratio,
        source=_TICKER,
        report_date=today,
        summary=summary,
    )
    _save_cache(ctx)
    logger.info(
        f"[move] MOVE={move:.1f} ({signal}) | "
        f"5d change={spike_5d:+.1f}pt | "
        f"source={source} | spiking={is_spiking}"
    )
    return ctx
