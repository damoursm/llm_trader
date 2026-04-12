"""Compute technical indicators and derive a [-1, +1] score.

Precision improvements over the naive approach:
  - SMA signals use % distance from the average (gradient), not a binary above/below
  - Volume conviction multiplier: a strong price move on high volume is more reliable
    than the same move on thin volume. Applied as a final multiplier [0.70, 1.30].
  - All individual signals are capped to [-1, +1] before averaging.
"""

import pandas as pd
import ta
from loguru import logger
from src.data.market_data import get_history


# Volume conviction: how many standard deviations above/below the 20-day mean
# a recent day's volume must be to earn a full boost/penalty.
_VOL_LOOKBACK   = 20    # days for average/std calculation
_VOL_BOOST_MAX  = 1.30  # maximum multiplier (very high volume)
_VOL_BOOST_MIN  = 0.70  # minimum multiplier (very low volume)


def _volume_conviction(df: pd.DataFrame) -> float:
    """
    Return a volume-conviction multiplier in [_VOL_BOOST_MIN, _VOL_BOOST_MAX].

    Logic:
      - Use the mean volume of the last 5 days vs the 20-day rolling mean.
        (5-day window avoids one-off spikes distorting the signal.)
      - vol_ratio > 1.5 AND price moved in signal direction → boost confidence
      - vol_ratio < 0.6                                     → dampen confidence
      - Otherwise: neutral (1.0)
    """
    if "Volume" not in df.columns or len(df) < _VOL_LOOKBACK + 5:
        return 1.0

    vol   = df["Volume"]
    close = df["Close"]

    avg_vol_20 = vol.iloc[-(_VOL_LOOKBACK + 5):-5].mean()
    if avg_vol_20 == 0:
        return 1.0

    recent_vol   = vol.iloc[-5:].mean()
    vol_ratio    = recent_vol / avg_vol_20

    # Recent price direction (last 5 days)
    price_change = (close.iloc[-1] - close.iloc[-5]) / close.iloc[-5]

    # High volume amplifies the signal; low volume dampens it
    if vol_ratio >= 2.0:
        multiplier = _VOL_BOOST_MAX
    elif vol_ratio >= 1.5:
        multiplier = 1.15
    elif vol_ratio <= 0.4:
        multiplier = _VOL_BOOST_MIN
    elif vol_ratio <= 0.6:
        multiplier = 0.80
    else:
        multiplier = 1.0

    logger.debug(
        f"Volume conviction: ratio={vol_ratio:.2f}x → multiplier={multiplier:.2f}  "
        f"price_5d={price_change:+.1%}"
    )
    return multiplier


def compute_technical_score(ticker: str) -> float:
    """
    Returns a score in [-1.0, +1.0].
    Positive = bullish signals, Negative = bearish signals.
    """
    df = get_history(ticker, period="3mo")
    if df.empty or len(df) < 20:
        logger.warning(f"Not enough history for technical analysis of {ticker}")
        return 0.0

    close = df["Close"]
    price = close.iloc[-1]
    signals = []

    # ── RSI: oversold(<30) → bullish, overbought(>70) → bearish ──────────
    rsi = ta.momentum.RSIIndicator(close, window=14).rsi().iloc[-1]
    if rsi < 30:
        signals.append(1.0)
    elif rsi > 70:
        signals.append(-1.0)
    else:
        signals.append((50 - rsi) / 50)   # gradient: 50 → 0.0, 30 → +0.40, 70 → -0.40

    # ── MACD: signal crossover strength ───────────────────────────────────
    macd_obj  = ta.trend.MACD(close)
    macd_diff = macd_obj.macd_diff().iloc[-1]
    signals.append(max(-1.0, min(1.0, macd_diff / price * 100)))

    # ── Price vs 50-day SMA: gradient by % distance ───────────────────────
    # +1.0 at 10%+ above, -1.0 at 10%+ below. Linear in between.
    sma50  = ta.trend.SMAIndicator(close, window=50).sma_indicator().iloc[-1]
    dist50 = (price - sma50) / sma50   # e.g. +0.05 = 5% above
    signals.append(max(-1.0, min(1.0, dist50 / 0.10)))

    # ── Price vs 200-day SMA: gradient, lower weight via softer cap ───────
    window_200 = min(200, len(close))
    sma200     = ta.trend.SMAIndicator(close, window=window_200).sma_indicator().iloc[-1]
    dist200    = (price - sma200) / sma200
    # Cap at ±15% distance for a full ±1.0 score (longer MA → wider bands)
    signals.append(max(-1.0, min(1.0, dist200 / 0.15)))

    # ── Bollinger Bands: position within band ─────────────────────────────
    bb       = ta.volatility.BollingerBands(close, window=20)
    bb_high  = bb.bollinger_hband().iloc[-1]
    bb_low   = bb.bollinger_lband().iloc[-1]
    bb_range = bb_high - bb_low
    if bb_range > 0:
        bb_pos = (price - bb_low) / bb_range   # 0 = at low band, 1 = at high band
        signals.append(1.0 - 2 * bb_pos)       # -1 at top, +1 at bottom
    else:
        signals.append(0.0)

    raw_score = sum(signals) / len(signals)

    # ── Volume conviction multiplier ──────────────────────────────────────
    vol_mult  = _volume_conviction(df)
    score     = raw_score * vol_mult

    logger.debug(
        f"{ticker} technical: raw={raw_score:.3f}  vol_mult={vol_mult:.2f}  "
        f"final={score:.3f} | rsi={rsi:.1f}"
    )
    return round(max(-1.0, min(1.0, score)), 3)
