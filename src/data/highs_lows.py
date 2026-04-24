"""
New 52-Week Highs vs. Lows — Market breadth divergence signal.

Tracks the proportion of tickers in the market basket (sector ETFs + broad indices +
watchlist stocks) that are trading near their 52-week highs vs. near their 52-week lows.

  High-Low Spread = %near_highs − %near_lows  ∈ [−100, +100]

Signal thresholds (spread):
  ≥ +50   STRONG_HIGHS    → BULLISH  (most names near cycle highs; broad strength)
  +20–50  HIGHS_DOMINATE  → BULLISH  (more highs than lows; trend intact)
  −20–20  BALANCED        → NEUTRAL  (mixed; no breadth edge)
  −50 – −20 LOWS_DOMINATE → BEARISH  (lows accumulating; trend weakening)
  ≤ −50   STRONG_LOWS     → BEARISH  (most names near cycle lows; broad weakness)

Divergence (highest-value signal — precedes reversals by 1–2 weeks):
  BEARISH DIVERGENCE: SPY near 52-week high, HL spread declining ≥10pp over 5 sessions
    → Rally led by fewer names; under-the-surface deterioration; distribution phase.
  BULLISH DIVERGENCE: SPY near 52-week low, HL spread rising ≥10pp over 5 sessions
    → Market still falling but new lows are contracting; selling exhaustion; coiling.

Data source: yfinance (sector ETFs + broad index ETFs + watchlist stocks). Cached daily.
"""

import json
import math
from datetime import date
from pathlib import Path
from typing import Optional

import pandas as pd
import yfinance as yf
from loguru import logger

from config import settings
from src.models import HighsLowsContext

CACHE_DIR = Path("cache")

_LOOKBACK      = "14mo"   # needs ≥ 252 trading days + 15-session lookback buffer
_ROLLING_DAYS  = 252      # standard 52-week window
_HIGH_ZONE_PCT = 5.0      # within 5% of 52-week high → "near high"
_LOW_ZONE_PCT  = 5.0      # within 5% of 52-week low  → "near low"
_SPY_HIGH_PCT  = 5.0      # SPY must be within 5% of its 52w high for bearish divergence
_SPY_LOW_PCT   = 5.0      # SPY must be within 5% of its 52w low  for bullish divergence
_DIV_SPREAD_DELTA = 10.0  # spread must change ≥ 10pp in 5 sessions to count as divergence


# ─────────────────────────────────────────────────────────────────────────────
# Cache helpers
# ─────────────────────────────────────────────────────────────────────────────

def _cache_path() -> Path:
    return CACHE_DIR / f"highs_lows_{date.today().isoformat()}.json"


def _load_cache() -> Optional[HighsLowsContext]:
    path = _cache_path()
    if not path.exists():
        return None
    try:
        ctx = HighsLowsContext.model_validate(
            json.loads(path.read_text(encoding="utf-8"))
        )
        logger.info(
            f"[highs_lows] Loaded from cache — spread={ctx.hl_spread:+.0f}pp "
            f"({ctx.signal}) | {ctx.highs_count}H / {ctx.lows_count}L / {ctx.total_count}"
        )
        return ctx
    except Exception as e:
        logger.warning(f"[highs_lows] Cache load failed: {e}")
        return None


def _save_cache(ctx: HighsLowsContext) -> None:
    CACHE_DIR.mkdir(exist_ok=True)
    try:
        _cache_path().write_text(
            json.dumps(ctx.model_dump(mode="json"), indent=2, default=str),
            encoding="utf-8",
        )
    except Exception as e:
        logger.warning(f"[highs_lows] Cache save failed: {e}")


# ─────────────────────────────────────────────────────────────────────────────
# Classification helpers
# ─────────────────────────────────────────────────────────────────────────────

def _classify(spread: float) -> tuple[str, str]:
    if spread >=  50: return "STRONG_HIGHS",   "BULLISH"
    if spread >=  20: return "HIGHS_DOMINATE",  "BULLISH"
    if spread >  -20: return "BALANCED",        "NEUTRAL"
    if spread >  -50: return "LOWS_DOMINATE",   "BEARISH"
    return               "STRONG_LOWS",    "BEARISH"


# ─────────────────────────────────────────────────────────────────────────────
# Public entry point
# ─────────────────────────────────────────────────────────────────────────────

def fetch_highs_lows_context() -> Optional[HighsLowsContext]:
    """
    Download 14 months of OHLCV data for sector ETFs + broad-market ETFs + watchlist stocks,
    then count how many are trading within 5% of their 52-week high (near-high zone) vs.
    within 5% of their 52-week low (near-low zone). Computes HL Spread today, 5d ago, and
    10d ago to detect divergence from the SPY index level. Cached daily.
    """
    cached = _load_cache()
    if cached is not None:
        return cached

    if not settings.enable_fetch_data:
        logger.debug("[highs_lows] ENABLE_FETCH_DATA=false — skipping yfinance fetch")
        return None

    # Build basket: sector ETFs + broad-market ETFs + watchlist stocks (deduplicated)
    seen: dict[str, None] = {}
    for t in settings.sectors_list + ["SPY", "QQQ", "IWM", "DIA"] + settings.stocks_list:
        seen[t] = None
    basket = list(seen.keys())

    try:
        data = yf.download(
            basket,
            period=_LOOKBACK,
            interval="1d",
            progress=False,
            auto_adjust=True,
        )
    except Exception as e:
        logger.warning(f"[highs_lows] yfinance download failed: {e}")
        return None

    if data is None or data.empty:
        logger.warning("[highs_lows] No data returned from yfinance")
        return None

    try:
        close = data["Close"]
    except Exception as e:
        logger.warning(f"[highs_lows] Could not extract Close prices: {e}")
        return None

    # Tickers to measure (exclude SPY — used only as index reference)
    measured = [t for t in basket if t != "SPY"]

    # ── Per-session zone counts ───────────────────────────────────────────────
    # We need counts for today (offset=1), 5d ago (offset=6), 10d ago (offset=11).
    offsets = {"now": 1, "5d": 6, "10d": 11}
    counts: dict[str, dict[str, int]] = {
        k: {"highs": 0, "lows": 0, "total": 0} for k in offsets
    }

    for ticker in measured:
        try:
            if ticker not in close.columns:
                continue
            s = close[ticker].dropna()
            min_needed = _ROLLING_DAYS + max(offsets.values()) + 1
            if len(s) < min_needed:
                logger.debug(f"[highs_lows] {ticker}: only {len(s)} bars, need {min_needed}")
                continue

            rh = s.rolling(_ROLLING_DAYS, min_periods=int(_ROLLING_DAYS * 0.8)).max()
            rl = s.rolling(_ROLLING_DAYS, min_periods=int(_ROLLING_DAYS * 0.8)).min()

            for key, offset in offsets.items():
                price = float(s.iloc[-offset])
                hi    = float(rh.iloc[-offset])
                lo    = float(rl.iloc[-offset])

                if math.isnan(hi) or math.isnan(lo) or hi <= 0 or lo <= 0:
                    continue

                pct_from_high = (price - hi) / hi * 100  # ≤ 0
                pct_from_low  = (price - lo) / lo * 100  # ≥ 0

                counts[key]["total"] += 1
                if pct_from_high >= -_HIGH_ZONE_PCT:
                    counts[key]["highs"] += 1
                elif pct_from_low <= _LOW_ZONE_PCT:
                    counts[key]["lows"] += 1
        except Exception as exc:
            logger.debug(f"[highs_lows] Skipping {ticker}: {exc}")

    c = counts["now"]
    if c["total"] == 0:
        logger.warning("[highs_lows] No valid tickers to compute HL spread")
        return None

    total    = c["total"]
    h_count  = c["highs"]
    l_count  = c["lows"]
    n_count  = total - h_count - l_count

    pct_highs = round(h_count / total * 100, 1)
    pct_lows  = round(l_count / total * 100, 1)
    hl_spread = round(pct_highs - pct_lows, 1)

    def _spread_for(key: str) -> Optional[float]:
        d = counts[key]
        if d["total"] == 0:
            return None
        h = d["highs"] / d["total"] * 100
        lv = d["lows"]  / d["total"] * 100
        return round(h - lv, 1)

    hl_spread_5d  = _spread_for("5d")
    hl_spread_10d = _spread_for("10d")

    signal, direction = _classify(hl_spread)

    # ── SPY reference ─────────────────────────────────────────────────────────
    spy_pct_from_high: Optional[float] = None
    spy_pct_from_low:  Optional[float] = None
    try:
        if "SPY" in close.columns:
            spy_s  = close["SPY"].dropna()
            if len(spy_s) >= _ROLLING_DAYS:
                spy_rh = spy_s.rolling(_ROLLING_DAYS, min_periods=200).max()
                spy_rl = spy_s.rolling(_ROLLING_DAYS, min_periods=200).min()
                spy_price    = float(spy_s.iloc[-1])
                spy_52w_high = float(spy_rh.iloc[-1])
                spy_52w_low  = float(spy_rl.iloc[-1])
                if spy_52w_high > 0:
                    spy_pct_from_high = round((spy_price - spy_52w_high) / spy_52w_high * 100, 2)
                if spy_52w_low > 0:
                    spy_pct_from_low  = round((spy_price - spy_52w_low)  / spy_52w_low  * 100, 2)
    except Exception as e:
        logger.debug(f"[highs_lows] SPY reference failed: {e}")

    # ── Divergence detection ─────────────────────────────────────────────────
    is_bearish_divergence = False
    is_bullish_divergence = False

    if hl_spread_5d is not None:
        delta_5d = hl_spread - hl_spread_5d
        # Bearish: SPY near 52-week high but HL spread contracting
        if (spy_pct_from_high is not None
                and spy_pct_from_high >= -_SPY_HIGH_PCT
                and delta_5d <= -_DIV_SPREAD_DELTA):
            is_bearish_divergence = True
            direction = "BEARISH"
        # Bullish: SPY near 52-week low but HL spread expanding
        if (spy_pct_from_low is not None
                and spy_pct_from_low <= _SPY_LOW_PCT
                and delta_5d >= _DIV_SPREAD_DELTA):
            is_bullish_divergence = True
            direction = "BULLISH"

    # ── Summary ───────────────────────────────────────────────────────────────
    signal_desc = {
        "STRONG_HIGHS":   "Most names near cycle highs — broad strength and leadership.",
        "HIGHS_DOMINATE": "More names near highs than lows — trend has broad participation.",
        "BALANCED":       "Mixed breadth — roughly equal highs and lows.",
        "LOWS_DOMINATE":  "More names near lows than highs — underlying weakness accumulating.",
        "STRONG_LOWS":    "Most names near cycle lows — broad deterioration.",
    }.get(signal, "")

    trend_str = ""
    if hl_spread_5d is not None:
        delta = hl_spread - hl_spread_5d
        trend_str = f" (5d ago: {hl_spread_5d:+.0f}pp, Δ{delta:+.0f}pp)."

    spy_str = ""
    if spy_pct_from_high is not None:
        spy_str = f" SPY: {spy_pct_from_high:+.1f}% from 52w high"
        if spy_pct_from_low is not None:
            spy_str += f", {spy_pct_from_low:+.1f}% from 52w low."
        else:
            spy_str += "."

    div_str = ""
    if is_bearish_divergence:
        div_str = " ⚠ BEARISH DIVERGENCE: SPY near highs but breadth eroding."
    elif is_bullish_divergence:
        div_str = " ⚡ BULLISH DIVERGENCE: SPY near lows but new lows contracting."

    summary = (
        f"HL Spread: {hl_spread:+.0f}pp ({h_count} near-highs, {l_count} near-lows, "
        f"{n_count} neutral / {total} total). {signal_desc}{trend_str}{spy_str}{div_str}"
    )

    ctx = HighsLowsContext(
        highs_count=h_count,
        lows_count=l_count,
        neutral_count=n_count,
        total_count=total,
        pct_near_highs=pct_highs,
        pct_near_lows=pct_lows,
        hl_spread=hl_spread,
        hl_spread_5d_ago=hl_spread_5d,
        hl_spread_10d_ago=hl_spread_10d,
        spy_pct_from_52w_high=spy_pct_from_high,
        spy_pct_from_52w_low=spy_pct_from_low,
        signal=signal,
        direction=direction,
        is_bearish_divergence=is_bearish_divergence,
        is_bullish_divergence=is_bullish_divergence,
        report_date=date.today(),
        summary=summary,
    )
    _save_cache(ctx)

    div_tag = " | BEARISH DIV" if is_bearish_divergence else (" | BULLISH DIV" if is_bullish_divergence else "")
    logger.info(
        f"[highs_lows] Spread={hl_spread:+.0f}pp ({h_count}H/{l_count}L/{total}) "
        f"→ {signal} ({direction}){div_tag}"
    )
    return ctx
