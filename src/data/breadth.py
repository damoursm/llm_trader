"""
S&P 500 Market Breadth — % of sector ETFs above their 200-day SMA.

When the majority of market sectors trade above their 200-day moving average, the rally
has broad participation — a healthy sign. When fewer than 30% are above their 200d SMA,
the market is broadly oversold across sectors, not just at the index level. A rapid
recovery from sub-30% breadth is a "confirmed breadth thrust" — one of the most reliable
bullish setups in market history, associated with 3–6 month sustained rallies.

Signal thresholds (sector ETF breadth):
  ≥ 85%   BREADTH_EXTENDED  → contrarian BEARISH (extended; limited upside, complacency)
  70–84%  BREADTH_HEALTHY   → BULLISH (broad participation; trend confirmed)
  50–69%  BREADTH_MIXED     → NEUTRAL (mixed; stock-picking environment)
  30–49%  BREADTH_WEAK      → BEARISH (more sectors below 200d than above)
  < 30%   BREADTH_COLLAPSE  → BEARISH (broad market stress; watch for thrust)

Breadth thrust (strongest bullish signal):
  Rising ≥ 8 percentage points in 5 trading days from a sub-35% base → confirmed thrust.
  Override direction → BULLISH.

Approximated via sector ETF internals: 11 SPDR sector ETFs (XLK, XLF, XLE, XLV, XLY, XLP,
XLI, XLB, XLU, XLRE, XLC) + SPY for index-level context. Cached daily.
"""

import json
from datetime import date
from pathlib import Path
from typing import List, Optional, Tuple

import yfinance as yf
from loguru import logger

from config import settings
from src.models import BreadthContext

CACHE_DIR = Path("cache")
_LOOKBACK = "14mo"   # needs 205+ trading days; 14 months ≈ 295 trading days

_THRUST_SESSIONS    = 5     # compare current vs N sessions ago for thrust
_THRUST_MIN_RISE_PP = 8.0   # need ≥ 8 percentage-point rise in _THRUST_SESSIONS days
_THRUST_MAX_BASE    = 35.0  # previous reading must be below this to qualify as thrust


# ─────────────────────────────────────────────────────────────────────────────
# Cache helpers
# ─────────────────────────────────────────────────────────────────────────────

def _cache_path() -> Path:
    return CACHE_DIR / f"breadth_{date.today().isoformat()}.json"


def _load_cache() -> Optional[BreadthContext]:
    path = _cache_path()
    if not path.exists():
        return None
    try:
        ctx = BreadthContext.model_validate(json.loads(path.read_text(encoding="utf-8")))
        logger.info(
            f"[breadth] Loaded from cache — {ctx.pct_above_200d:.0f}% above 200d SMA "
            f"({ctx.signal})"
        )
        return ctx
    except Exception as e:
        logger.warning(f"[breadth] Cache load failed: {e}")
        return None


def _save_cache(ctx: BreadthContext) -> None:
    CACHE_DIR.mkdir(exist_ok=True)
    try:
        _cache_path().write_text(
            json.dumps(ctx.model_dump(mode="json"), indent=2, default=str),
            encoding="utf-8",
        )
    except Exception as e:
        logger.warning(f"[breadth] Cache save failed: {e}")


# ─────────────────────────────────────────────────────────────────────────────
# Classification
# ─────────────────────────────────────────────────────────────────────────────

def _classify(pct: float, prev_pct: Optional[float]) -> Tuple[str, str, bool]:
    """Return (signal, direction, is_breadth_thrust).

    A breadth thrust fires when breadth was in the BREADTH_COLLAPSE zone (<30%)
    and has now risen OUT of it (≥30%) with a significant move (≥8pp in 5 sessions).
    This mirrors the classic definition: rapid escape from broadly oversold territory.
    """
    is_thrust = (
        prev_pct is not None
        and prev_pct < 30.0          # was in collapse zone
        and pct >= 30.0              # has crossed the 30% threshold
        and (pct - prev_pct) >= _THRUST_MIN_RISE_PP  # significant rise
    )

    if pct >= 85.0:
        signal, direction = "BREADTH_EXTENDED", "BEARISH"
    elif pct >= 70.0:
        signal, direction = "BREADTH_HEALTHY", "BULLISH"
    elif pct >= 50.0:
        signal, direction = "BREADTH_MIXED", "NEUTRAL"
    elif pct >= 30.0:
        signal, direction = "BREADTH_WEAK", "BEARISH"
    else:
        signal, direction = "BREADTH_COLLAPSE", "BEARISH"

    # Thrust overrides direction to BULLISH regardless of the raw level
    if is_thrust:
        direction = "BULLISH"

    return signal, direction, is_thrust


# ─────────────────────────────────────────────────────────────────────────────
# Public entry point
# ─────────────────────────────────────────────────────────────────────────────

def fetch_breadth_context() -> Optional[BreadthContext]:
    """
    Download 14 months of sector ETF closes from yfinance and compute the % of
    ETFs trading above their 200-day SMA (with 5-day lookback for thrust detection).
    Returns BreadthContext or None if data is unavailable. Cached daily.
    """
    cached = _load_cache()
    if cached is not None:
        return cached

    if not settings.enable_fetch_data:
        logger.debug("[breadth] ENABLE_FETCH_DATA=false — skipping yfinance fetch")
        return None

    sector_etfs: List[str] = settings.sectors_list   # 11 SPDR sector ETFs
    all_tickers = sector_etfs + ["SPY"]

    try:
        data = yf.download(
            all_tickers,
            period=_LOOKBACK,
            interval="1d",
            progress=False,
            auto_adjust=True,
        )
    except Exception as e:
        logger.warning(f"[breadth] yfinance download failed: {e}")
        return None

    if data is None or data.empty:
        logger.warning("[breadth] No data returned from yfinance")
        return None

    try:
        close = data["Close"]
    except Exception as e:
        logger.warning(f"[breadth] Could not extract Close prices: {e}")
        return None

    # ── Per-ticker SMA200 calculation ─────────────────────────────────────────
    n_total     = 0
    n_above_now = 0
    n_above_5d  = 0
    n_valid_5d  = 0

    spy_above_200d: Optional[bool]  = None
    spy_200d_dist:  Optional[float] = None

    offset_5d = _THRUST_SESSIONS + 1   # iloc[-7] for 5 sessions back relative to iloc[-1]

    for ticker in all_tickers:
        try:
            series = close[ticker].dropna()
        except (KeyError, TypeError):
            logger.debug(f"[breadth] No data for {ticker}")
            continue

        if len(series) < 202:
            logger.debug(f"[breadth] Insufficient history for {ticker}: {len(series)} bars")
            continue

        sma200 = series.rolling(200).mean()

        spot      = float(series.iloc[-1])
        sma_now   = float(sma200.iloc[-1])
        above_now = spot > sma_now

        # 5-session-ago reading
        above_5d_ago: Optional[bool] = None
        if len(series) >= 202 + offset_5d:
            spot_5d  = float(series.iloc[-offset_5d])
            sma_5d   = float(sma200.iloc[-offset_5d])
            above_5d_ago = spot_5d > sma_5d

        if ticker == "SPY":
            spy_above_200d = above_now
            if sma_now > 0:
                spy_200d_dist = round((spot - sma_now) / sma_now * 100, 2)
            continue

        # Sector ETF counts
        n_total += 1
        if above_now:
            n_above_now += 1
        if above_5d_ago is not None:
            n_valid_5d += 1
            if above_5d_ago:
                n_above_5d += 1

    if n_total == 0:
        logger.warning("[breadth] No valid sector ETFs found")
        return None

    pct_now = round(n_above_now / n_total * 100, 1)
    pct_5d  = round(n_above_5d  / n_valid_5d * 100, 1) if n_valid_5d > 0 else None

    signal, direction, is_thrust = _classify(pct_now, pct_5d)

    # ── Summary ───────────────────────────────────────────────────────────────
    signal_desc = {
        "BREADTH_COLLAPSE":  "Broad market stress — most sectors below 200d SMA.",
        "BREADTH_WEAK":      "More sectors below than above 200d SMA — cautious environment.",
        "BREADTH_MIXED":     "Mixed breadth — stock-picking environment, no broad trend confirmation.",
        "BREADTH_HEALTHY":   "Healthy breadth — broad participation confirms the prevailing trend.",
        "BREADTH_EXTENDED":  "Nearly all sectors above 200d SMA — limited further upside; complacency risk.",
    }.get(signal, "")

    spy_str = ""
    if spy_above_200d is not None:
        pos = "above" if spy_above_200d else "below"
        dist_str = f" ({spy_200d_dist:+.1f}% from 200d SMA)" if spy_200d_dist is not None else ""
        spy_str = f" SPY is {pos} its 200d SMA{dist_str}."

    delta_str = ""
    if pct_5d is not None:
        delta = pct_now - pct_5d
        delta_str = f" 5-session ago: {pct_5d:.0f}% (Δ{delta:+.0f}pp)."

    thrust_str = ""
    if is_thrust:
        rise = pct_now - (pct_5d or pct_now)
        thrust_str = (
            f" BREADTH THRUST confirmed: +{rise:.0f}pp rise from oversold levels — "
            f"historically one of the strongest multi-month bullish setups."
        )

    summary = (
        f"{n_above_now}/{n_total} sector ETFs above 200d SMA ({pct_now:.0f}%). "
        f"{signal_desc}{spy_str}{delta_str}{thrust_str}"
    )

    ctx = BreadthContext(
        pct_above_200d=pct_now,
        pct_above_200d_5d_ago=pct_5d,
        etf_count=n_total,
        etfs_above=n_above_now,
        signal=signal,
        direction=direction,
        is_breadth_thrust=is_thrust,
        spy_above_200d=spy_above_200d,
        spy_200d_distance_pct=spy_200d_dist,
        report_date=date.today(),
        summary=summary,
    )
    _save_cache(ctx)
    thrust_tag = f" | BREADTH THRUST (from {pct_5d:.0f}%)" if is_thrust else ""
    logger.info(
        f"[breadth] {pct_now:.0f}% above 200d SMA ({n_above_now}/{n_total} ETFs) "
        f"→ {signal} ({direction}){thrust_tag}"
    )
    return ctx
