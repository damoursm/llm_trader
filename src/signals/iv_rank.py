"""
IV Rank + Directional signal — Volatility-regime-aware directional bias.

True historical implied volatility per ticker would require options-chain
history we don't keep. Realized volatility (RV) is a tight proxy: RV and IV
are highly correlated (corr typically 0.7–0.9 on 20-day windows), and the
*rank* of current vol within a ticker's own trailing distribution carries
the same regime information as IV Rank — what matters is "is this stock
unusually volatile vs its own past", not the absolute level.

  IV Rank ≈ percentile_rank(current_RV_21d, trailing_252_RV_21d)  ∈ [0, 100]

The "directional" component combines IV Rank with normalised 5-day price
action (return ÷ ATR%) — i.e. how many average-daily-ranges the stock has
moved this week. This is robust to regime shifts because both inputs are
self-normalised against each ticker's own volatility footprint.

Score logic (regime-aware):

  High IV Rank (≥ 70)
    Indicates a fear / event-pricing regime — options are expensive,
    market is bracing for a move. Apply contrarian logic:
      • strong negative move (z_ret ≤ -1)  →  +0.55  (capitulation buy)
      • strong positive move (z_ret ≥ +1)  →  -0.55  (euphoric chase, fade)
      • mild move                          →  -0.20 × sign(z_ret)  (event risk caution)

  Low IV Rank (≤ 30)
    Calm regime — options are cheap, no event priced in. Trend confirmation:
      • positive trend  →  +0.40 × tanh(z_ret / 1.5)   (room for vol expansion)
      • negative trend  →  −0.40 × tanh(|z_ret| / 1.5) (downtrend not yet feared)

  Mid IV Rank (30 < ir < 70)
    Mild trend-following bias only:
      • score = 0.30 × tanh(z_ret / 1.5)

The returned ``direction_label`` is a one-token regime summary used in the
email leaderboard ("CAPITULATION_BUY" | "FADE_EXTREME" | "CALM_UPTREND" |
"CALM_DOWNTREND" | "TREND_FOLLOWING" | "EVENT_CAUTION" | "NEUTRAL").

Cache strategy:
  Prefers the incremental OHLCV chart cache (cache/ohlcv/<TICKER>.json).
  Falls back to a live yfinance fetch on cold cache.
  Works with ENABLE_FETCH_DATA=false when chart caches are populated.
  Minimum 65 bars required; returns (0.0, 50.0, 0.0, "NEUTRAL") otherwise.
"""

from typing import Tuple

import numpy as np
import pandas as pd
from loguru import logger

from src.data.market_data import get_history
from src.data.cache import load_ohlcv


_MIN_ROWS    = 65        # minimum bars to compute RV rank + 5d return
_RV_WINDOW   = 21        # bars per realized-vol calculation (~1 month)
_DIST_WINDOW = 252       # trailing bars for the RV percentile distribution
_RET_WINDOW  = 5         # bars for the short-term directional return
_ATR_WINDOW  = 14        # bars for ATR%
_TANH_SCALE  = 1.5       # divisor inside tanh for the trend-follow leg
_HIGH_IR     = 70.0      # IV rank threshold for "high"
_LOW_IR      = 30.0      # IV rank threshold for "low"


def _get_ohlcv(ticker: str) -> pd.DataFrame:
    cached = load_ohlcv(ticker)
    if cached is not None and len(cached) >= _MIN_ROWS:
        return cached
    return get_history(ticker, period="18mo")


def _atr_pct(df: pd.DataFrame) -> float:
    """14-period ATR as % of last close. Returns 0.0 when insufficient."""
    high  = df["High"].astype(float)
    low   = df["Low"].astype(float)
    close = df["Close"].astype(float)
    prev_close = close.shift(1)
    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low - prev_close).abs(),
    ], axis=1).max(axis=1)
    atr = tr.rolling(_ATR_WINDOW).mean().iloc[-1]
    last = float(close.iloc[-1])
    if not np.isfinite(atr) or last <= 0:
        return 0.0
    return float(atr) / last


def compute_iv_rank_score(ticker: str) -> Tuple[float, float, float, str]:
    """Return ``(score, iv_rank, ret_5d_pct, direction_label)``.

    ``score`` ∈ [-1, +1]: regime-aware directional bias.
    ``iv_rank`` ∈ [0, 100]: percentile rank of current 21-day realized vol
      within the trailing 252-day RV distribution.
    ``ret_5d_pct``: raw 5-day percent return.
    ``direction_label``: short token describing the regime + direction.
    """
    df = _get_ohlcv(ticker)

    if df.empty or len(df) < _MIN_ROWS or "Close" not in df.columns:
        logger.debug(f"[iv_rank] {ticker}: insufficient data ({len(df)} rows)")
        return 0.0, 50.0, 0.0, "NEUTRAL"

    close = df["Close"].astype(float)

    # ── Realized volatility series: 21-day stdev of log returns × √252 ──
    log_ret = np.log(close / close.shift(1))
    rv = log_ret.rolling(_RV_WINDOW).std() * np.sqrt(252)
    rv_clean = rv.dropna()
    if len(rv_clean) < 30:
        return 0.0, 50.0, 0.0, "NEUTRAL"

    dist = rv_clean.tail(_DIST_WINDOW)
    current_rv = float(dist.iloc[-1])
    if not np.isfinite(current_rv) or current_rv <= 0:
        return 0.0, 50.0, 0.0, "NEUTRAL"

    # IV-Rank-style percentile of current RV inside the trailing distribution
    iv_rank = float((dist < current_rv).sum()) / float(len(dist)) * 100.0
    iv_rank = round(max(0.0, min(100.0, iv_rank)), 1)

    # ── Directional component: 5-day return normalised by ATR% ──
    if len(close) < _RET_WINDOW + 1:
        return 0.0, iv_rank, 0.0, "NEUTRAL"
    ret_5d = float((close.iloc[-1] - close.iloc[-_RET_WINDOW - 1]) / close.iloc[-_RET_WINDOW - 1])
    atr_pct = _atr_pct(df)
    if atr_pct < 1e-6:
        return 0.0, iv_rank, ret_5d * 100, "NEUTRAL"
    # z_ret ≈ how many average daily ranges this move covered
    z_ret = ret_5d / atr_pct

    # ── Regime-aware scoring ─────────────────────────────────────────
    score = 0.0
    label = "NEUTRAL"
    if iv_rank >= _HIGH_IR:
        # Expensive options → contrarian
        if z_ret <= -1.0:
            score = +0.55
            label = "CAPITULATION_BUY"
        elif z_ret >= 1.0:
            score = -0.55
            label = "FADE_EXTREME"
        else:
            sign = float(np.sign(z_ret)) if abs(z_ret) > 0.1 else 0.0
            score = -0.20 * sign
            label = "EVENT_CAUTION" if sign != 0 else "NEUTRAL"
    elif iv_rank <= _LOW_IR:
        # Cheap options → trend confirmation
        if z_ret > 0:
            score = +0.40 * float(np.tanh(z_ret / _TANH_SCALE))
            label = "CALM_UPTREND" if score > 0.05 else "NEUTRAL"
        elif z_ret < 0:
            score = -0.40 * float(np.tanh(abs(z_ret) / _TANH_SCALE))
            label = "CALM_DOWNTREND" if score < -0.05 else "NEUTRAL"
    else:
        # Mid-IV-rank: mild trend-following
        score = 0.30 * float(np.tanh(z_ret / _TANH_SCALE))
        if abs(score) >= 0.05:
            label = "TREND_FOLLOWING"

    score = round(max(-1.0, min(1.0, score)), 3)
    ret_5d_pct = round(ret_5d * 100, 2)

    logger.debug(
        f"[iv_rank] {ticker}: rv={current_rv:.2%}  ir={iv_rank:.0f}  "
        f"ret_5d={ret_5d_pct:+.1f}%  atr={atr_pct:.2%}  z_ret={z_ret:+.2f}  "
        f"label={label}  score={score:+.3f}"
    )
    return score, iv_rank, ret_5d_pct, label