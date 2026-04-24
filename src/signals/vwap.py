"""
VWAP Distance signal — mean-reversion indicator.

Rolling multi-day VWAP is the volume-weighted average price over the last N
trading sessions.  Institutions use VWAP as their execution benchmark, so
large deviations attract institutional order flow back toward it.

  price >> VWAP  →  bearish mean-reversion signal  (score < 0)
  price << VWAP  →  bullish mean-reversion signal   (score > 0)
  price ≈ VWAP   →  neutral

Score normalisation:
  We use a z-score approach against the rolling distribution of (price − VWAP)
  distances over the last 60 sessions.  This makes the score adaptive to each
  security's typical volatility:

      distance_pct  = (close − VWAP) / VWAP × 100
      z             = distance_pct / rolling_std(distance_pct, 60)
      score         = −clip(z / 2.0, −1.0, +1.0)

  Dividing by 2 means a 2-sigma deviation yields a full ±1.0 score.
  The sign is inverted because the signal is mean-reversion (above → bearish).

Important caveats:
  - VWAP pull is weaker in strong trending markets (momentum overrides gravity).
  - VWAP distance is best used as a confirming / dampening layer, not a
    standalone directional call.
  - Requires at least 25 rows of history to compute; returns 0.0 otherwise.
"""

from typing import Tuple

import numpy as np
import pandas as pd
from loguru import logger

from src.data.market_data import get_history
from src.data.cache import load_ohlcv

_VWAP_WINDOW = 20   # rolling sessions for VWAP
_STD_WINDOW  = 60   # sessions for normalising the distance
_MIN_ROWS    = 25   # minimum rows needed to produce a score


def _get_ohlcv(ticker: str) -> pd.DataFrame:
    """
    Return OHLCV history, preferring the incremental chart cache over a live fetch.

    The OHLCV chart cache (cache/ohlcv/<TICKER>.json) is written on every run
    that generates charts and typically holds 3+ months of data.  Using it here
    means VWAP works even when ENABLE_FETCH_DATA=false.
    """
    cached = load_ohlcv(ticker)
    if cached is not None and len(cached) >= _MIN_ROWS:
        return cached
    return get_history(ticker, period="6mo")


def compute_vwap_score(ticker: str) -> Tuple[float, float]:
    """
    Return (score, vwap_distance_pct) for the ticker.

    score ∈ [-1.0, +1.0] — positive = price below VWAP (bullish reversion),
                            negative = price above VWAP (bearish reversion).
    vwap_distance_pct     — raw (price − VWAP) / VWAP × 100 (positive = above VWAP).
    Returns (0.0, 0.0) if there is insufficient data.
    """
    df = _get_ohlcv(ticker)

    required = {"High", "Low", "Close", "Volume"}
    if df.empty or len(df) < _MIN_ROWS or not required.issubset(df.columns):
        logger.debug(f"[vwap] {ticker}: insufficient data ({len(df)} rows)")
        return 0.0, 0.0

    close  = df["Close"].astype(float)
    high   = df["High"].astype(float)
    low    = df["Low"].astype(float)
    volume = df["Volume"].astype(float).replace(0, np.nan)

    # Typical price = (H + L + C) / 3  (classic VWAP input)
    typical = (high + low + close) / 3.0

    # Rolling VWAP over _VWAP_WINDOW sessions
    tp_x_vol     = (typical * volume).rolling(_VWAP_WINDOW, min_periods=_VWAP_WINDOW)
    vol_rolling  = volume.rolling(_VWAP_WINDOW, min_periods=_VWAP_WINDOW)
    vwap_series  = tp_x_vol.sum() / vol_rolling.sum()

    # Drop rows where VWAP couldn't be computed
    valid = vwap_series.dropna()
    if valid.empty:
        return 0.0, 0.0

    # Current values (most recent row with valid VWAP)
    current_vwap  = float(valid.iloc[-1])
    current_price = float(close.loc[valid.index[-1]])

    if current_vwap <= 0:
        return 0.0, 0.0

    # Distance as % of VWAP
    dist_pct = (current_price - current_vwap) / current_vwap * 100.0

    # Normalise using rolling std of the distance series
    dist_series = (close - vwap_series) / vwap_series * 100.0
    dist_std    = float(dist_series.rolling(_STD_WINDOW, min_periods=10).std().iloc[-1])

    if dist_std > 0:
        z     = dist_pct / dist_std
        score = -float(np.clip(z / 2.0, -1.0, 1.0))   # invert: above VWAP → bearish
    else:
        score = 0.0

    score = round(score, 3)
    dist_pct = round(dist_pct, 2)

    logger.debug(
        f"[vwap] {ticker}: price={current_price:.2f}  vwap={current_vwap:.2f}  "
        f"dist={dist_pct:+.2f}%  z={dist_pct / dist_std if dist_std else 0:.2f}  "
        f"score={score:+.3f}"
    )
    return score, dist_pct
