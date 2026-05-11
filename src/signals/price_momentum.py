"""
Price Momentum signal — Perceived Value tracker.

When investors observe rising prices, perceived value increases and attracts
more capital, reinforcing the trend. This self-fulfilling dynamic means
that multi-period price momentum is one of the most robust factors in
academic finance (Jegadeesh & Titman 1993; Asness et al. 2013).

Score derivation:
  1. Compute 1-month (21-bar) and 2-month (42-bar) returns from OHLCV history.
  2. Normalise each against the ticker's own trailing 252-bar return distribution
     (rolling std of 21-bar returns). Adapts to each security's volatility regime.
  3. Composite z-score = 0.6 × z_1m + 0.4 × z_3m  (recent momentum weighted more)
  4. Map to [-1, +1] via tanh(z / 1.5):
       z = 0   → score =  0.00  (in-line with own history)
       z = 1   → score ≈ +0.62  (1-sigma above baseline — clear uptrend)
       z = 2   → score ≈ +0.90  (2-sigma above — market is chasing this name)
       z = -1  → score ≈ -0.62  (momentum selling territory)

Volume adjustment (±0.10 max):
  Strong uptrend + rising volume → institutional participation → +0.10
  Strong downtrend + rising volume → confirmed distribution → -0.10
  Any trend + falling volume → "thin air" → slight dampening

Cache strategy:
  Prefers the incremental OHLCV chart cache (cache/ohlcv/<TICKER>.json).
  Falls back to a live yfinance fetch on cold cache.
  Works with ENABLE_FETCH_DATA=false when chart caches are populated.
  Minimum 70 bars required; returns 0.0 when data is insufficient.
"""

from typing import Tuple

import numpy as np
import pandas as pd
from loguru import logger

from src.data.market_data import get_history
from src.data.cache import load_ohlcv


_MIN_ROWS      = 50    # minimum bars to compute meaningful momentum
_DIST_WINDOW   = 252   # trailing bars for normalisation distribution
_MOM_SHORT     = 21    # 1-month lookback (trading days)
_MOM_MED       = 42    # 2-month lookback (trading days); 42 bars ≈ 8-9 weeks
_VOL_WINDOW    = 21    # rolling window for volume ratio
_SHORT_WEIGHT  = 0.6   # weight on 1m in composite z-score
_MED_WEIGHT    = 0.4   # weight on 3m in composite z-score
_TANH_SCALE    = 1.5   # divisor before tanh; 1.5-sigma maps to ≈0.74


def _get_ohlcv(ticker: str) -> pd.DataFrame:
    cached = load_ohlcv(ticker)
    if cached is not None and len(cached) >= _MIN_ROWS:
        return cached
    return get_history(ticker, period="18mo")


def compute_price_momentum_score(ticker: str) -> Tuple[float, float, float]:
    """Return (score, mom_1m_pct, mom_3m_pct).

    score ∈ [-1.0, +1.0].
    Positive = price trending above its own historical baseline (upward momentum).
    Negative = price trending below historical baseline (downward momentum).
    Returns (0.0, 0.0, 0.0) when data is insufficient.
    """
    df = _get_ohlcv(ticker)

    if df.empty or len(df) < _MIN_ROWS or "Close" not in df.columns:
        logger.debug(f"[momentum] {ticker}: insufficient data ({len(df)} rows)")
        return 0.0, 0.0, 0.0

    close  = df["Close"].astype(float)
    volume = df["Volume"].astype(float).replace(0, np.nan) if "Volume" in df.columns else None

    if len(close) < _MOM_MED + 2:
        return 0.0, 0.0, 0.0

    # ── Raw multi-period returns ─────────────────────────────────────────────
    mom_1m = float((close.iloc[-1] - close.iloc[-_MOM_SHORT]) / close.iloc[-_MOM_SHORT])
    mom_3m = float((close.iloc[-1] - close.iloc[-_MOM_MED])   / close.iloc[-_MOM_MED])

    # ── Normalise against own trailing return distribution ───────────────────
    rolling_returns = close.pct_change(_MOM_SHORT)
    dist_slice = rolling_returns.dropna().tail(_DIST_WINDOW)

    if len(dist_slice) < 30:
        return 0.0, 0.0, 0.0

    std_1m = float(dist_slice.std())
    if std_1m < 1e-8:
        return 0.0, 0.0, 0.0

    z_1m = mom_1m / std_1m
    z_3m = mom_3m / (std_1m * (2 ** 0.5))  # 2m std ≈ sqrt(2) × 1m std

    z_composite = _SHORT_WEIGHT * z_1m + _MED_WEIGHT * z_3m

    # ── Volume confirmation adjustment ───────────────────────────────────────
    vol_adj = 0.0
    if volume is not None and len(volume) >= _DIST_WINDOW:
        vol_recent = float(volume.tail(_VOL_WINDOW).mean())
        vol_longer = float(volume.tail(_DIST_WINDOW).mean())
        if vol_longer > 0:
            vol_ratio = vol_recent / vol_longer
            if z_composite > 0.5 and vol_ratio > 1.3:
                vol_adj = +0.10   # uptrend confirmed by rising institutional volume
            elif z_composite < -0.5 and vol_ratio > 1.3:
                vol_adj = -0.10   # downtrend confirmed by rising distribution volume
            elif vol_ratio < 0.6:
                vol_adj = -0.05 * float(np.sign(z_composite))  # thin volume → less conviction

    # ── Map to [-1, +1] ──────────────────────────────────────────────────────
    raw_score = float(np.tanh(z_composite / _TANH_SCALE)) + vol_adj
    score     = round(max(-1.0, min(1.0, raw_score)), 3)

    mom_1m_pct = round(mom_1m * 100, 2)
    mom_3m_pct = round(mom_3m * 100, 2)

    logger.debug(
        f"[momentum] {ticker}: 1m={mom_1m_pct:+.1f}%  3m={mom_3m_pct:+.1f}%  "
        f"z_1m={z_1m:+.2f}  z_3m={z_3m:+.2f}  z_comp={z_composite:+.2f}  "
        f"vol_adj={vol_adj:+.2f}  score={score:+.3f}"
    )
    return score, mom_1m_pct, mom_3m_pct
