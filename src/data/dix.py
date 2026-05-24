"""Dark Pool Index (DIX) + market-wide GEX — SqueezeMetrics free feed.

DIX (Dark Pool Index) is the dollar- and volume-weighted short-volume across
off-exchange (dark pool) venues, expressed as a fraction (~0.38–0.48). Large
institutions route through dark pools to accumulate without moving the lit
market, so a HIGH DIX is a proxy for *hidden institutional buying* and is
historically bullish for forward S&P 500 returns — it leads price by ~1–4 weeks.

The same feed carries a market-wide GEX (gamma exposure): the whole-index dealer
gamma positioning. This is distinct from the per-ticker dealer gamma computed in
``gamma_exposure.py``. High/positive GEX = vol suppression (dealers long gamma →
pinning, mean-reversion); low/negative GEX = vol expansion (dealers short gamma →
moves amplified). The classic SqueezeMetrics bullish setup is **high DIX + low
GEX**: hidden accumulation with room to run.

Interpretation (relative to the ticker's own trailing-year range):
  DIX percentile ≥ 75   STRONG_ACCUMULATION  → BULLISH
  DIX percentile ≥ 58   ACCUMULATION         → BULLISH
  DIX percentile 42–58  NEUTRAL              → NEUTRAL
  DIX percentile ≤ 42   DISTRIBUTION         → BEARISH
  DIX percentile ≤ 25   STRONG_DISTRIBUTION  → BEARISH

DIX is a broad-market / regime overlay, NOT a per-ticker signal. It feeds the
Claude prompt and the Macro Regime Filter. Cached daily; no API key required.

Source: https://squeezemetrics.com/monitor/static/DIX.csv  (columns: date, price, dix, gex)
"""

import csv
import io
import json
from datetime import date
from pathlib import Path
from statistics import mean
from typing import List, Optional, Tuple

import httpx
from loguru import logger

from config import settings
from src.models import DIXContext

_CACHE_DIR    = Path("cache")
_CACHE_PREFIX = "dix_"
_CSV_URL      = "https://squeezemetrics.com/monitor/static/DIX.csv"
_WINDOW       = 252      # ~1 trading year for percentile computation
_TREND_EPS    = 0.005    # ≥0.5pp 5-day move in DIX counts as a trend


# ─────────────────────────────────────────────────────────────────────────────
# Cache helpers
# ─────────────────────────────────────────────────────────────────────────────

def _cache_path(today: date) -> Path:
    return _CACHE_DIR / f"{_CACHE_PREFIX}{today.isoformat()}.json"


def _load_cache(today: date) -> Optional[DIXContext]:
    p = _cache_path(today)
    if not p.exists():
        return None
    try:
        ctx = DIXContext.model_validate(json.loads(p.read_text(encoding="utf-8")))
        logger.info(f"[dix] Loaded from cache — DIX={ctx.dix_pct}%, signal={ctx.signal}")
        return ctx
    except Exception as e:
        logger.warning(f"[dix] Cache load failed: {e}")
        return None


def _save_cache(ctx: DIXContext) -> None:
    _CACHE_DIR.mkdir(exist_ok=True)
    try:
        _cache_path(ctx.report_date).write_text(
            json.dumps(ctx.model_dump(mode="json"), indent=2, default=str),
            encoding="utf-8",
        )
    except Exception as e:
        logger.warning(f"[dix] Cache save failed: {e}")


# ─────────────────────────────────────────────────────────────────────────────
# Classification helpers
# ─────────────────────────────────────────────────────────────────────────────

def _percentile_rank(values: List[float], current: float) -> Optional[float]:
    """Percentile (0–100) of ``current`` within ``values`` (fraction ≤ current)."""
    if not values:
        return None
    below = sum(1 for v in values if v <= current)
    return round(100.0 * below / len(values), 1)


def _classify(pct: Optional[float], dix: Optional[float]) -> Tuple[str, str]:
    """Return (signal, direction) from DIX percentile (primary) + absolute fallback."""
    if pct is not None:
        if pct >= 75:
            return "STRONG_ACCUMULATION", "BULLISH"
        if pct >= 58:
            return "ACCUMULATION", "BULLISH"
        if pct <= 25:
            return "STRONG_DISTRIBUTION", "BEARISH"
        if pct <= 42:
            return "DISTRIBUTION", "BEARISH"
        return "NEUTRAL", "NEUTRAL"
    # Absolute fallback when trailing history is too short for a percentile
    if dix is None:
        return "UNKNOWN", "NEUTRAL"
    if dix >= 0.45:
        return "STRONG_ACCUMULATION", "BULLISH"
    if dix >= 0.43:
        return "ACCUMULATION", "BULLISH"
    if dix <= 0.38:
        return "STRONG_DISTRIBUTION", "BEARISH"
    if dix <= 0.40:
        return "DISTRIBUTION", "BEARISH"
    return "NEUTRAL", "NEUTRAL"


def _classify_gex(gex: Optional[float], pct: Optional[float]) -> str:
    """Return whole-index gamma regime: VOL_SUPPRESSION | NEUTRAL | VOL_EXPANSION."""
    if gex is not None and gex < 0:
        return "VOL_EXPANSION"          # dealers net short gamma → moves amplified
    if pct is None:
        return "UNKNOWN"
    if pct >= 70:
        return "VOL_SUPPRESSION"        # high gamma → pinning / mean-reversion
    if pct <= 30:
        return "VOL_EXPANSION"
    return "NEUTRAL"


# ─────────────────────────────────────────────────────────────────────────────
# Feed parsing
# ─────────────────────────────────────────────────────────────────────────────

def _fetch_rows() -> Optional[List[dict]]:
    """Download and parse the SqueezeMetrics DIX CSV → list of {date, price, dix, gex}."""
    try:
        resp = httpx.get(_CSV_URL, timeout=30, headers={"User-Agent": "llm-trader/1.0"})
        resp.raise_for_status()
    except Exception as e:
        logger.warning(f"[dix] CSV fetch failed: {e}")
        return None

    rows: List[dict] = []
    try:
        reader = csv.DictReader(io.StringIO(resp.text))
        for raw in reader:
            r = {(k or "").strip().lower(): v for k, v in raw.items()}
            try:
                rows.append({
                    "date":  r.get("date"),
                    "price": float(r["price"]),
                    "dix":   float(r["dix"]),
                    "gex":   float(r["gex"]),
                })
            except (KeyError, ValueError, TypeError):
                continue
    except Exception as e:
        logger.warning(f"[dix] CSV parse failed: {e}")
        return None

    return rows or None


# ─────────────────────────────────────────────────────────────────────────────
# Public entry point
# ─────────────────────────────────────────────────────────────────────────────

def fetch_dix_context(today: Optional[date] = None) -> Optional[DIXContext]:
    """Fetch the Dark Pool Index + market-wide GEX and classify the regime.

    Cache-first (works offline once cached). When ENABLE_FETCH_DATA=false and no
    cache exists, returns None so the pipeline simply skips DIX. Any feed/parse
    failure also returns None — DIX never blocks the rest of the run.
    """
    if today is None:
        today = date.today()

    cached = _load_cache(today)
    if cached is not None:
        return cached

    if not settings.enable_fetch_data:
        logger.debug("[dix] ENABLE_FETCH_DATA=false — skipping DIX fetch")
        return None

    rows = _fetch_rows()
    if not rows:
        logger.warning("[dix] No DIX rows available — skipping DIX context")
        return None

    dix_series = [r["dix"] for r in rows]
    gex_series = [r["gex"] for r in rows]
    latest = rows[-1]
    dix = latest["dix"]
    gex = latest["gex"]
    spx = latest["price"]

    window_dix = dix_series[-_WINDOW:]
    window_gex = gex_series[-_WINDOW:]
    dix_pct = _percentile_rank(window_dix, dix)
    gex_pct = _percentile_rank(window_gex, gex)

    dix_5d_avg  = round(mean(dix_series[-5:]), 4)  if len(dix_series) >= 5  else None
    dix_20d_avg = round(mean(dix_series[-20:]), 4) if len(dix_series) >= 20 else None

    dix_5d_ago = dix_series[-6] if len(dix_series) >= 6 else None
    if dix_5d_ago is not None:
        delta = dix - dix_5d_ago
        dix_trend = "RISING" if delta > _TREND_EPS else ("FALLING" if delta < -_TREND_EPS else "FLAT")
    else:
        dix_trend = "FLAT"

    signal, direction = _classify(dix_pct, dix)
    gex_regime = _classify_gex(gex, gex_pct)

    # ── Summary ──────────────────────────────────────────────────────────────
    pct_str = f" ({dix_pct:.0f}th pct of trailing year)" if dix_pct is not None else ""
    parts = [
        f"Dark Pool Index at {dix * 100:.1f}%{pct_str} — "
        f"{signal.replace('_', ' ').lower()}; {dix_trend.lower()} over 5 days."
    ]
    if gex is not None:
        parts.append(f"Market GEX {gex / 1e9:+.2f}Bn ({gex_regime.replace('_', ' ').lower()}).")
    if direction == "BULLISH" and gex_regime == "VOL_EXPANSION":
        parts.append(
            "High dark-pool buying + low gamma = classic 'hidden accumulation with room to "
            "run' — among the most bullish DIX/GEX configurations."
        )
    elif direction == "BEARISH" and gex_regime == "VOL_EXPANSION":
        parts.append("Weak dark-pool support + low gamma = elevated downside-volatility risk.")
    elif direction == "BULLISH" and gex_regime == "VOL_SUPPRESSION":
        parts.append("Accumulation present but high gamma is pinning price — expect a grind, not a spike.")
    summary = " ".join(parts)

    ctx = DIXContext(
        dix=round(dix, 4),
        dix_pct=round(dix * 100, 2),
        dix_5d_avg=dix_5d_avg,
        dix_20d_avg=dix_20d_avg,
        dix_percentile_1y=dix_pct,
        dix_trend=dix_trend,
        gex=round(gex, 2),
        gex_percentile_1y=gex_pct,
        gex_regime=gex_regime,
        spx_price=round(spx, 2),
        obs_count=len(window_dix),
        signal=signal,
        direction=direction,
        source="SqueezeMetrics DIX.csv",
        report_date=today,
        summary=summary,
    )
    _save_cache(ctx)
    logger.info(
        f"[dix] DIX={dix * 100:.1f}% ({signal}, {direction}) | "
        f"pct_1y={dix_pct} | trend={dix_trend} | "
        f"GEX={gex / 1e9:+.2f}Bn ({gex_regime})"
    )
    return ctx
