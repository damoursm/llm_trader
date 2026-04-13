"""Compute technical indicators and derive a [-1, +1] score.

Also returns market-microstructure metadata (vol_ratio, atr_pct, bb_width_pct)
that the aggregator uses for cross-method confidence adjustments:

  vol_ratio    — recent 5-day volume vs 20-day average; used by aggregator to
                 amplify confidence when multiple methods AND volume agree.
  atr_pct      — ATR(14) / close; answers "how much does this stock typically
                 move per day?" — high → price is likely to move on a signal.
  bb_width_pct — (upper - lower) / middle BB; measures current volatility
                 regime (squeeze = narrow, expansion = wide).

Score improvements vs a naive approach:
  - SMA signals use % distance from the average (gradient), not a binary above/below.
  - Volume conviction multiplier shapes the *technical* sub-score [0.70, 1.30].
    A separate vol_ratio is also exposed for cross-method use in the aggregator.
  - All individual indicator signals are capped to [-1, +1] before averaging.
"""

from dataclasses import dataclass

import pandas as pd
import ta
from loguru import logger
from src.data.market_data import get_history


_VOL_LOOKBACK  = 20    # days for volume average
_VOL_BOOST_MAX = 1.30
_VOL_BOOST_MIN = 0.70


@dataclass
class TechnicalResult:
    """All outputs from the technical analysis pass for a single ticker."""
    score: float        # [-1, +1] directional signal
    vol_ratio: float    # recent_5d_vol / 20d_avg_vol  (1.0 = average)
    atr_pct: float      # ATR(14) / close price        (typical daily range as % of price)
    bb_width_pct: float # (BB_upper − BB_lower) / BB_middle (squeeze → narrow, expansion → wide)


# Sentinel returned when there is insufficient data.
EMPTY_RESULT = TechnicalResult(score=0.0, vol_ratio=1.0, atr_pct=0.01, bb_width_pct=0.05)


def _volume_conviction(df: pd.DataFrame) -> tuple[float, float]:
    """
    Return (vol_multiplier, vol_ratio) where:
      vol_multiplier ∈ [_VOL_BOOST_MIN, _VOL_BOOST_MAX] shapes the technical sub-score.
      vol_ratio      is the raw recent/average ratio exposed to the aggregator.
    Uses the 5-day mean vs 20-day rolling mean to avoid one-off spike distortion.
    """
    if "Volume" not in df.columns or len(df) < _VOL_LOOKBACK + 5:
        return 1.0, 1.0

    vol = df["Volume"]
    avg_vol_20 = vol.iloc[-(_VOL_LOOKBACK + 5):-5].mean()
    if avg_vol_20 == 0:
        return 1.0, 1.0

    recent_vol = vol.iloc[-5:].mean()
    vol_ratio  = recent_vol / avg_vol_20

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

    return multiplier, vol_ratio


def compute_technical_score(ticker: str) -> TechnicalResult:
    """
    Compute technical indicators and return a TechnicalResult.

    score ∈ [-1.0, +1.0] — positive = bullish, negative = bearish.
    Also returns vol_ratio, atr_pct, bb_width_pct for use in the aggregator.
    """
    df = get_history(ticker, period="3mo")
    if df.empty or len(df) < 20:
        logger.warning(f"Not enough history for technical analysis of {ticker}")
        return EMPTY_RESULT

    close = df["Close"]
    price = float(close.iloc[-1])
    signals = []

    # ── RSI: oversold(<30) → bullish, overbought(>70) → bearish ──────────
    rsi = ta.momentum.RSIIndicator(close, window=14).rsi().iloc[-1]
    if rsi < 30:
        signals.append(1.0)
    elif rsi > 70:
        signals.append(-1.0)
    else:
        signals.append((50 - rsi) / 50)   # gradient: 50→0.0, 30→+0.40, 70→-0.40

    # ── MACD: signal crossover strength ───────────────────────────────────
    macd_diff = ta.trend.MACD(close).macd_diff().iloc[-1]
    signals.append(max(-1.0, min(1.0, macd_diff / price * 100)))

    # ── Price vs 50-day SMA: gradient by % distance ───────────────────────
    sma50  = ta.trend.SMAIndicator(close, window=50).sma_indicator().iloc[-1]
    dist50 = (price - sma50) / sma50
    signals.append(max(-1.0, min(1.0, dist50 / 0.10)))

    # ── Price vs 200-day SMA: gradient, softer cap ────────────────────────
    window_200 = min(200, len(close))
    sma200     = ta.trend.SMAIndicator(close, window=window_200).sma_indicator().iloc[-1]
    dist200    = (price - sma200) / sma200
    signals.append(max(-1.0, min(1.0, dist200 / 0.15)))

    # ── Bollinger Bands: position within band ─────────────────────────────
    bb      = ta.volatility.BollingerBands(close, window=20)
    bb_high = bb.bollinger_hband().iloc[-1]
    bb_mid  = bb.bollinger_mavg().iloc[-1]
    bb_low  = bb.bollinger_lband().iloc[-1]
    bb_range = bb_high - bb_low
    if bb_range > 0:
        bb_pos = (price - bb_low) / bb_range   # 0 = at low band, 1 = at high band
        signals.append(1.0 - 2 * bb_pos)       # -1 at top, +1 at bottom
    else:
        signals.append(0.0)

    raw_score = sum(signals) / len(signals)

    # ── Volume conviction (shapes technical sub-score) ────────────────────
    vol_mult, vol_ratio = _volume_conviction(df)
    score = round(max(-1.0, min(1.0, raw_score * vol_mult)), 3)

    # ── ATR%: typical daily range as % of price ───────────────────────────
    atr_pct = 0.01  # default ~1%
    if "High" in df.columns and "Low" in df.columns and len(df) >= 15:
        atr_val = ta.volatility.AverageTrueRange(
            df["High"], df["Low"], close, window=14
        ).average_true_range().iloc[-1]
        if price > 0:
            atr_pct = float(atr_val / price)

    # ── BB width%: (upper − lower) / middle ──────────────────────────────
    bb_width_pct = float(bb_range / bb_mid) if bb_mid > 0 else 0.05

    logger.debug(
        f"{ticker} technical: raw={raw_score:.3f}  vol_mult={vol_mult:.2f}  "
        f"score={score:.3f} | rsi={rsi:.1f}  atr%={atr_pct:.3f}  "
        f"bb_width%={bb_width_pct:.3f}  vol_ratio={vol_ratio:.2f}x"
    )
    return TechnicalResult(
        score=score,
        vol_ratio=round(vol_ratio, 2),
        atr_pct=round(atr_pct, 4),
        bb_width_pct=round(bb_width_pct, 4),
    )
