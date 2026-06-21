"""
Trend Strength — ADX/DMI directional movement + Donchian channel breakout.

Two of the most empirically durable, decades-validated technical systems, combined
into one trend-quality signal. They measure a dimension the rest of the stack does
NOT: not "how big was the return" (price momentum) or "overbought/oversold"
(RSI/Bollinger/MFI), but **is price in a strong, confirmed directional trend, and
which way?**

1. Directional Movement Index (Welles Wilder, 1978)
   - +DI / -DI measure upward vs downward directional movement (Wilder-smoothed).
   - ADX measures TREND STRENGTH regardless of direction:
       ADX < 20  → no trend (chop) — directional signals are unreliable here
       ADX 25-40 → established trend
       ADX > 40  → very strong trend
   - Direction = (+DI − -DI) / (+DI + -DI) ∈ [-1, +1]; strength scales it.

2. Donchian Channel breakout (the "Turtle" system)
   - 20-day channel of the prior highs/lows. A close above the prior 20-day high is
     a classic long breakout; below the prior 20-day low, a short breakout.
   - Between the bands, a mild positional lean toward the nearer band.

Composite score ∈ [-1, +1]:
   adx_dir   = ((+DI − -DI) / (+DI + -DI)) × clip((ADX − 15)/30, 0, 1)
   donchian  = +1 on a 20-day-high breakout, −1 on a 20-day-low breakout,
               else 0.5 × (2 × channel_position − 1)
   score     = clip(0.60 × adx_dir + 0.40 × donchian, -1, +1)

Positive = strong/confirmed UPtrend (trend-following BUY bias).
Negative = strong/confirmed DOWNtrend (trend-following SELL bias).
Near zero = no trend / chop — by design this dampens conviction rather than guessing.

Cache strategy mirrors the other technical signals: prefers the incremental OHLCV
chart cache (cache/ohlcv/<TICKER>.json), falls back to a live fetch on cold cache,
and works with ENABLE_FETCH_DATA=false when chart caches are populated.
Minimum 50 bars required (ADX needs ~2× the period to stabilise); returns
(0.0, 0.0, "NO_DATA") when data is insufficient.
"""

from typing import Optional, Tuple

import numpy as np
import pandas as pd
from loguru import logger

from config import settings
from src.data.market_data import get_history
from src.data.cache import load_ohlcv

_MIN_ROWS = 50   # ADX(14) needs ~2× period + Donchian(20) warmup to be meaningful


def _get_ohlcv(ticker: str) -> pd.DataFrame:
    cached = load_ohlcv(ticker)
    if cached is not None and len(cached) >= _MIN_ROWS:
        return cached
    return get_history(ticker, period="18mo")


def _wilder(series: pd.Series, period: int) -> pd.Series:
    """Wilder's smoothing (RMA) — equivalent to an EMA with alpha = 1/period."""
    return series.ewm(alpha=1.0 / period, adjust=False).mean()


def _compute_dmi(df: pd.DataFrame, period: int) -> Tuple[float, float, float]:
    """Return (adx, plus_di, minus_di) using Wilder's standard formulation."""
    high  = df["High"]
    low   = df["Low"]
    close = df["Close"]
    prev_close = close.shift(1)

    up_move   = high.diff()
    down_move = -low.diff()
    plus_dm  = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    plus_dm  = pd.Series(plus_dm, index=df.index)
    minus_dm = pd.Series(minus_dm, index=df.index)

    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low - prev_close).abs(),
    ], axis=1).max(axis=1)

    atr = _wilder(tr, period)
    atr_safe = atr.replace(0, np.nan)
    plus_di  = 100.0 * _wilder(plus_dm, period) / atr_safe
    minus_di = 100.0 * _wilder(minus_dm, period) / atr_safe

    di_sum = (plus_di + minus_di).replace(0, np.nan)
    dx = 100.0 * (plus_di - minus_di).abs() / di_sum
    adx = _wilder(dx.fillna(0.0), period)

    return (
        float(adx.iloc[-1]),
        float(plus_di.iloc[-1]) if np.isfinite(plus_di.iloc[-1]) else 0.0,
        float(minus_di.iloc[-1]) if np.isfinite(minus_di.iloc[-1]) else 0.0,
    )


def _compute_donchian(df: pd.DataFrame, period: int) -> Tuple[int, float]:
    """Return (breakout, position).

    breakout: +1 close above prior `period`-day high, −1 below prior low, else 0.
    position: where the latest close sits in the current `period`-day range, 0–1.
    """
    high  = df["High"]
    low   = df["Low"]
    close = float(df["Close"].iloc[-1])

    prior_high = float(high.rolling(period).max().shift(1).iloc[-1])
    prior_low  = float(low.rolling(period).min().shift(1).iloc[-1])

    breakout = 0
    if np.isfinite(prior_high) and close > prior_high:
        breakout = 1
    elif np.isfinite(prior_low) and close < prior_low:
        breakout = -1

    chan_hi = float(high.rolling(period).max().iloc[-1])
    chan_lo = float(low.rolling(period).min().iloc[-1])
    rng = chan_hi - chan_lo
    position = (close - chan_lo) / rng if rng > 0 else 0.5
    position = max(0.0, min(1.0, position))

    return breakout, position


def _classify(adx: float, direction: float, breakout: int) -> str:
    if adx < 20:
        return "NO_TREND"
    if breakout == 1:
        return "BREAKOUT_UP"
    if breakout == -1:
        return "BREAKOUT_DOWN"
    if direction > 0.10:
        return "STRONG_UPTREND" if adx >= 25 else "UPTREND"
    if direction < -0.10:
        return "STRONG_DOWNTREND" if adx >= 25 else "DOWNTREND"
    return "NEUTRAL"


def compute_trend_strength_score(ticker: str, df: Optional[pd.DataFrame] = None) -> Tuple[float, float, str]:
    """Return (score, adx_value, label).

    score ∈ [−1.0, +1.0] — positive = confirmed uptrend, negative = confirmed downtrend,
    near zero = no trend / chop. Returns (0.0, 0.0, "NO_DATA") when data is insufficient.

    ``df``: optional pre-fetched OHLCV frame (any timeframe). When ``None`` the
    daily cache-first fetch is used — identical to the legacy behaviour.
    """
    adx_period      = max(2, settings.trend_adx_period)
    donchian_period = max(2, settings.trend_donchian_period)

    if df is None:
        df = _get_ohlcv(ticker)
    required = {"High", "Low", "Close"}
    if df.empty or len(df) < _MIN_ROWS or not required.issubset(df.columns):
        logger.debug(f"[trend] {ticker}: insufficient data ({len(df)} rows)")
        return 0.0, 0.0, "NO_DATA"

    df = df.copy()
    for col in required:
        df[col] = df[col].astype(float)

    try:
        adx, plus_di, minus_di = _compute_dmi(df, adx_period)
        breakout, position     = _compute_donchian(df, donchian_period)
    except Exception as exc:
        logger.debug(f"[trend] {ticker}: computation error — {exc}")
        return 0.0, 0.0, "NO_DATA"

    if not np.isfinite(adx):
        return 0.0, 0.0, "NO_DATA"

    di_sum = plus_di + minus_di
    direction = (plus_di - minus_di) / di_sum if di_sum > 0 else 0.0
    adx_strength = max(0.0, min(1.0, (adx - 15.0) / 30.0))   # ADX 15→0, 45→1
    adx_dir = direction * adx_strength

    if breakout == 1:
        donchian = 1.0
    elif breakout == -1:
        donchian = -1.0
    else:
        donchian = 0.5 * (2.0 * position - 1.0)   # mild lean toward the nearer band

    raw = 0.60 * adx_dir + 0.40 * donchian
    score = round(max(-1.0, min(1.0, raw)), 3)
    label = _classify(adx, direction, breakout)

    logger.debug(
        f"[trend] {ticker}: ADX={adx:.1f} +DI={plus_di:.1f} -DI={minus_di:.1f} "
        f"dir={direction:+.2f}×str={adx_strength:.2f}={adx_dir:+.2f}  "
        f"donch={donchian:+.2f}(bo={breakout},pos={position:.2f})  "
        f"score={score:+.3f} [{label}]"
    )
    return score, round(adx, 2), label
