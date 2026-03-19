"""Compute technical indicators and derive a score."""

import pandas as pd
import ta
from loguru import logger
from src.data.market_data import get_history


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
    signals = []

    # RSI: oversold(<30) → bullish, overbought(>70) → bearish
    rsi = ta.momentum.RSIIndicator(close, window=14).rsi().iloc[-1]
    if rsi < 30:
        signals.append(1.0)
    elif rsi > 70:
        signals.append(-1.0)
    else:
        signals.append((50 - rsi) / 50)  # normalised: 0 is neutral

    # MACD: signal crossover
    macd_obj = ta.trend.MACD(close)
    macd_diff = macd_obj.macd_diff().iloc[-1]
    signals.append(max(-1.0, min(1.0, macd_diff / close.iloc[-1] * 100)))

    # Price vs 50-day SMA
    sma50 = ta.trend.SMAIndicator(close, window=50).sma_indicator().iloc[-1]
    price = close.iloc[-1]
    signals.append(1.0 if price > sma50 else -1.0)

    # Price vs 200-day SMA (use all available if < 200 days)
    window_200 = min(200, len(close))
    sma200 = ta.trend.SMAIndicator(close, window=window_200).sma_indicator().iloc[-1]
    signals.append(1.0 if price > sma200 else -1.0)

    # Bollinger Bands: near lower band → bullish, near upper band → bearish
    bb = ta.volatility.BollingerBands(close, window=20)
    bb_high = bb.bollinger_hband().iloc[-1]
    bb_low = bb.bollinger_lband().iloc[-1]
    bb_range = bb_high - bb_low
    if bb_range > 0:
        bb_pos = (price - bb_low) / bb_range  # 0 = at low, 1 = at high
        signals.append(1.0 - 2 * bb_pos)      # -1 at high, +1 at low
    else:
        signals.append(0.0)

    score = sum(signals) / len(signals)
    logger.debug(f"{ticker} technical score: {score:.3f} | rsi={rsi:.1f}")
    return round(max(-1.0, min(1.0, score)), 3)
