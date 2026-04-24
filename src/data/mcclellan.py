"""
McClellan Oscillator & Summation Index — NYSE advance/decline breadth momentum.

The McClellan Oscillator measures the momentum of market breadth by applying two
exponential moving averages to the daily net advances (advancing NYSE issues minus
declining issues):

  McClellan Oscillator = EMA(19) of Net Advances − EMA(39) of Net Advances

  When positive: the 19-day EMA is above the 39-day EMA → breadth accelerating
  When negative: the 19-day EMA is below the 39-day EMA → breadth decelerating

The McClellan Summation Index is the running cumulative total of oscillator readings
and gives a longer-term perspective on market breadth:

  Summation Index(t) = Summation Index(t-1) + Oscillator(t)

Oscillator signal thresholds:
  > +100   OVERBOUGHT        → contrarian BEARISH (momentum stretched; fade rallies)
  > +50    BULLISH_MOMENTUM  → BULLISH (strong net advances; support buying)
  -50–50   NEUTRAL           → no directional edge
  < -50    BEARISH_MOMENTUM  → BEARISH (net declines dominating; avoid longs)
  < -100   OVERSOLD          → contrarian BULLISH (capitulation; coiling for reversal)

Summation Index thresholds (relative to our data window):
  > +500   EXTENDED_BULL → trend overstretched; reduce new long exposure
  > 0      BULL_TREND    → positive breadth trend; buy dips
  < 0      BEAR_TREND    → negative breadth trend; sell rallies
  < -500   EXTENDED_BEAR → deeply oversold trend; approaching major reversal zone

Zero-line crossings of the oscillator are the best swing-trade timing signals:
  Crossing above 0 → bullish momentum shift (EMA19 now above EMA39)
  Crossing below 0 → bearish momentum shift (EMA19 now below EMA39)

Data source: ^NYAD (NYSE Advance-Decline issues) via yfinance. Cached daily.
"""

import json
from datetime import date
from pathlib import Path
from typing import Optional, Tuple

import pandas as pd
import yfinance as yf
from loguru import logger

from config import settings
from src.models import McClellanContext

CACHE_DIR  = Path("cache")
_LOOKBACK  = "9mo"    # ~190 trading days; needs ≥ 44 days for EMA39 to converge
_AD_SYMBOL = "^NYAD"  # NYSE Advance-Decline issues (daily or cumulative A/D)
_EMA_SHORT = 19       # "10% trend" — fast EMA
_EMA_LONG  = 39       # "5% trend"  — slow EMA

# Oscillator thresholds
_OB_HIGH = 100.0   # overbought
_OB_MID  =  50.0   # bullish momentum
_OS_MID  = -50.0   # bearish momentum
_OS_HIGH = -100.0  # oversold

# Summation thresholds (relative to our data window; not tied to the absolute scale)
_SI_BULL_EXT =  500.0
_SI_BEAR_EXT = -500.0


# ─────────────────────────────────────────────────────────────────────────────
# Cache helpers
# ─────────────────────────────────────────────────────────────────────────────

def _cache_path() -> Path:
    return CACHE_DIR / f"mcclellan_{date.today().isoformat()}.json"


def _load_cache() -> Optional[McClellanContext]:
    path = _cache_path()
    if not path.exists():
        return None
    try:
        ctx = McClellanContext.model_validate(json.loads(path.read_text(encoding="utf-8")))
        logger.info(
            f"[mcclellan] Loaded from cache — osc={ctx.oscillator:+.1f} ({ctx.osc_signal}), "
            f"SI={ctx.summation:+.0f} ({ctx.sum_signal})"
        )
        return ctx
    except Exception as e:
        logger.warning(f"[mcclellan] Cache load failed: {e}")
        return None


def _save_cache(ctx: McClellanContext) -> None:
    CACHE_DIR.mkdir(exist_ok=True)
    try:
        _cache_path().write_text(
            json.dumps(ctx.model_dump(mode="json"), indent=2, default=str),
            encoding="utf-8",
        )
    except Exception as e:
        logger.warning(f"[mcclellan] Cache save failed: {e}")


# ─────────────────────────────────────────────────────────────────────────────
# Classification helpers
# ─────────────────────────────────────────────────────────────────────────────

def _classify_oscillator(osc: float) -> str:
    if osc >  _OB_HIGH: return "OVERBOUGHT"
    if osc >  _OB_MID:  return "BULLISH_MOMENTUM"
    if osc >= _OS_MID:  return "NEUTRAL"
    if osc >= _OS_HIGH: return "BEARISH_MOMENTUM"
    return "OVERSOLD"


def _classify_summation(si: float) -> str:
    if si >  _SI_BULL_EXT: return "EXTENDED_BULL"
    if si >  0:            return "BULL_TREND"
    if si >= _SI_BEAR_EXT: return "BEAR_TREND"
    return "EXTENDED_BEAR"


def _compute_direction(
    osc: float,
    osc_prev: Optional[float],
    si: float,
    is_bull_cross: bool,
    is_bear_cross: bool,
) -> str:
    """Determine directional bias. Crossings take priority."""
    if is_bull_cross:
        return "BULLISH"
    if is_bear_cross:
        return "BEARISH"
    # Contrarian at extremes: oversold + turning up / overbought + turning down
    if osc <= _OS_HIGH and osc_prev is not None and osc > osc_prev:
        return "BULLISH"   # deeply oversold and starting to recover
    if osc >= _OB_HIGH and osc_prev is not None and osc < osc_prev:
        return "BEARISH"   # overbought and rolling over
    # Momentum + trend alignment
    if osc >= _OB_MID and si > 0:
        return "BULLISH"
    if osc <= _OS_MID and si < 0:
        return "BEARISH"
    # Summation-only direction
    if si >  200:
        return "BULLISH"
    if si < -200:
        return "BEARISH"
    return "NEUTRAL"


# ─────────────────────────────────────────────────────────────────────────────
# Public entry point
# ─────────────────────────────────────────────────────────────────────────────

def fetch_mcclellan_context() -> Optional[McClellanContext]:
    """
    Compute the McClellan Oscillator and Summation Index from NYSE A/D data.
    Returns McClellanContext or None if data is unavailable. Cached daily.
    """
    cached = _load_cache()
    if cached is not None:
        return cached

    if not settings.enable_fetch_data:
        logger.debug("[mcclellan] ENABLE_FETCH_DATA=false — skipping yfinance fetch")
        return None

    try:
        data = yf.download(
            _AD_SYMBOL,
            period=_LOOKBACK,
            interval="1d",
            progress=False,
            auto_adjust=True,
        )
    except Exception as e:
        logger.warning(f"[mcclellan] yfinance download failed: {e}")
        return None

    if data is None or data.empty:
        logger.warning(f"[mcclellan] No data returned for {_AD_SYMBOL}")
        return None

    try:
        # Single-ticker download may be flat or MultiIndex depending on yfinance version
        if isinstance(data.columns, pd.MultiIndex):
            close = data["Close"][_AD_SYMBOL].dropna()
        else:
            close = data["Close"].dropna()
    except Exception as e:
        logger.warning(f"[mcclellan] Could not extract Close series: {e}")
        return None

    if len(close) < _EMA_LONG + 10:
        logger.warning(
            f"[mcclellan] Insufficient history: {len(close)} bars (need {_EMA_LONG + 10})"
        )
        return None

    # Determine if ^NYAD gives the cumulative A/D line or raw daily net-advances.
    # If values span > 10_000, it's likely the cumulative line → take daily differences.
    val_range = float(close.max() - close.min())
    if val_range > 10_000:
        logger.debug(f"[mcclellan] ^NYAD appears cumulative (range={val_range:.0f}); differencing")
        net_adv = close.diff().dropna()
    else:
        logger.debug(f"[mcclellan] ^NYAD appears to be daily net advances (range={val_range:.0f})")
        net_adv = close

    if len(net_adv) < _EMA_LONG + 5:
        logger.warning(f"[mcclellan] Insufficient net-advance data: {len(net_adv)} bars")
        return None

    # McClellan Oscillator = EMA(19) − EMA(39) of net advances
    ema19_s = net_adv.ewm(span=_EMA_SHORT, adjust=False).mean()
    ema39_s = net_adv.ewm(span=_EMA_LONG,  adjust=False).mean()
    osc_s   = ema19_s - ema39_s

    # McClellan Summation Index = cumulative sum of oscillator (relative to our window)
    si_s = osc_s.cumsum()

    # Extract current and historical snapshots
    osc_now  = float(osc_s.iloc[-1])
    osc_5d   = float(osc_s.iloc[-6]) if len(osc_s) >= 6 else None
    si_now   = float(si_s.iloc[-1])
    si_5d    = float(si_s.iloc[-6])  if len(si_s) >= 6  else None
    ema19_now = float(ema19_s.iloc[-1])
    ema39_now = float(ema39_s.iloc[-1])

    # Detect zero crossings in the last 3 sessions
    if len(osc_s) >= 4:
        prev3 = osc_s.iloc[-4:-1].tolist()
        is_bull_cross = (osc_now >= 0) and any(v < 0 for v in prev3)
        is_bear_cross = (osc_now < 0)  and any(v >= 0 for v in prev3)
    else:
        is_bull_cross = is_bear_cross = False

    osc_signal = _classify_oscillator(osc_now)
    sum_signal = _classify_summation(si_now)
    direction  = _compute_direction(osc_now, osc_5d, si_now, is_bull_cross, is_bear_cross)

    # ── Summary ───────────────────────────────────────────────────────────────
    osc_desc = {
        "OVERBOUGHT":       "Net-advance momentum is stretched — breadth exhaustion; fade further rallies.",
        "BULLISH_MOMENTUM": "Strong positive breadth momentum — advancing issues dominating on a sustained basis.",
        "NEUTRAL":          "Breadth momentum neutral — no clear A/D directional edge.",
        "BEARISH_MOMENTUM": "Negative breadth momentum — declining issues dominating; avoid new longs.",
        "OVERSOLD":         "Breadth deeply oversold — potential mean-reversion and swing-low setup forming.",
    }.get(osc_signal, "")

    cross_str = ""
    if is_bull_cross:
        cross_str = " Oscillator just crossed ABOVE zero → bullish momentum shift confirmed."
    elif is_bear_cross:
        cross_str = " Oscillator just crossed BELOW zero → bearish momentum shift confirmed."

    si_trend_str = ""
    if si_5d is not None:
        si_delta = si_now - si_5d
        si_trend_str = f" (Δ{si_delta:+.0f} over 5 sessions)"

    osc_delta_str = ""
    if osc_5d is not None:
        osc_delta_str = f" (5d ago: {osc_5d:+.1f})"

    summary = (
        f"McClellan Oscillator: {osc_now:+.1f}{osc_delta_str} ({osc_signal}). "
        f"Summation Index: {si_now:+.0f}{si_trend_str} ({sum_signal}). "
        f"{osc_desc}{cross_str}"
    )

    ctx = McClellanContext(
        oscillator=round(osc_now, 2),
        oscillator_5d_ago=round(osc_5d, 2) if osc_5d is not None else None,
        summation=round(si_now, 1),
        summation_5d_ago=round(si_5d, 1) if si_5d is not None else None,
        ema19=round(ema19_now, 1),
        ema39=round(ema39_now, 1),
        osc_signal=osc_signal,
        sum_signal=sum_signal,
        direction=direction,
        is_bullish_cross=is_bull_cross,
        is_bearish_cross=is_bear_cross,
        report_date=date.today(),
        summary=summary,
    )
    _save_cache(ctx)
    cross_tag = " | BULL CROSS" if is_bull_cross else (" | BEAR CROSS" if is_bear_cross else "")
    logger.info(
        f"[mcclellan] Osc={osc_now:+.1f} ({osc_signal}) | SI={si_now:+.0f} ({sum_signal}) "
        f"| {direction}{cross_tag}"
    )
    return ctx
