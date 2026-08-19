"""
Money Flow — Chaikin Money Flow accumulation / distribution signal (v3).

The score is CMF ALONE since 2026-08-16:

    cmf   = Σ₂₀( MFM × V ) / Σ₂₀ V,   MFM = ((C−L) − (H−C)) / (H−L)
    score = tanh( tanh(cmf / 0.15) / 0.6 )

(The outer tanh/0.6 is the old composite's output squash, kept so v3 is exactly
the measured candidate and the score scale is continuous with the composite's
CMF-only branch.)

Why the other two terms were REMOVED (v3 basis — the 2026-08-15 review's 3-year
gated battery: 754 daily cross-sections, 752k rows, 1,206 tickers, signed pivot
target, memory/news-interaction-methods... see money-flow-review memory):

  - The v2 composite measured IC +0.0349 (t +10.2) — the 2026-07-24 directional
    fix genuinely works — but CMF alone measured +0.0479 (t +11.8), beating the
    composite PAIRED at t +9.0, best in every year 2023–2026, and balanced
    per side (buy +0.028 / sell +0.031, where the composite's buy side was
    2.5× weaker).
  - OBV 21-bar slope: IC +0.0038 (t +0.97) — no information at any tested
    normalisation (full-history and trailing-252 scale both), pure dilution,
    and the single most expensive computation in the scorer (a Python loop of
    per-bar polyfits over the full 20-year frame).
  - MFI contrarian-at-extremes: IC −0.0140 (t −2.72), negative every year;
    the SIGN-FLIPPED variant merely tied the no-MFI composite (paired t −0.3),
    i.e. the term carries nothing either way. MFI survives only as an AUX
    display value (`mfi_value` feeds the synthesis prompt and the email
    template) — it no longer touches the score.

History: v1 could not express direction at all (contrarian MFI cancelled the
trend-following CMF; OBV z-score measured acceleration — the 2026-07-24 audit,
tests/test_method_directionality.py). v2 fixed both terms. v3 removes them.
Scorer epochs registered at each step; money_flow is REPLAYABLE, so the panel's
history is REGENERATED under the current formula rather than masked (run
`python -m src.analysis.replay --write` after any change here).

Latency: all rolling windows only need the trailing slice, so the scorer
computes on the last ``_TAIL_BARS`` bars — identical outputs, no full-history
loops (the frame is ~5,000 bars deep since the 20-year backfill).

Cache strategy: prefers the incremental OHLCV cache, falls back to a live
yfinance fetch on cold cache. Minimum 30 bars; returns (0.0, 50.0, 0.0) when
data is insufficient. ``df=`` accepts any timeframe (the multi-timeframe layer
passes 30-min and weekly frames); ``df=None`` reproduces the daily behaviour.
"""

from typing import Optional, Tuple

import numpy as np
import pandas as pd
from loguru import logger

from src.data.market_data import get_history
from src.data.cache import load_ohlcv


_MIN_ROWS    = 30    # minimum bars to compute meaningful indicators
_MFI_PERIOD  = 14    # standard MFI lookback (aux display value only since v3)
_CMF_PERIOD  = 20    # standard CMF lookback
_TAIL_BARS   = 60    # rolling windows at the last bar need only this slice


def _get_ohlcv(ticker: str) -> pd.DataFrame:
    cached = load_ohlcv(ticker)
    if cached is not None and len(cached) >= _MIN_ROWS:
        return cached
    return get_history(ticker, period="18mo")


def _compute_mfi(df: pd.DataFrame) -> float:
    """14-period Money Flow Index, 0–100. AUX DISPLAY VALUE only (v3): feeds
    `mfi_value` on the signal (synthesis prompt + email), not the score."""
    typical = (df["High"] + df["Low"] + df["Close"]) / 3
    mf = typical * df["Volume"]
    d = typical.diff()
    pos_sum = float(mf.where(d > 0, 0.0).rolling(_MFI_PERIOD).sum().iloc[-1])
    neg_sum = float(mf.where(d < 0, 0.0).rolling(_MFI_PERIOD).sum().iloc[-1])
    if not np.isfinite(pos_sum) or not np.isfinite(neg_sum):
        return 50.0
    if neg_sum == 0:
        return 100.0
    return round(100.0 - 100.0 / (1.0 + pos_sum / neg_sum), 2)


def _compute_cmf(df: pd.DataFrame) -> float:
    """20-period Chaikin Money Flow, returns [−1, +1]."""
    hl_range = (df["High"] - df["Low"]).replace(0, np.nan)
    mfm = ((df["Close"] - df["Low"]) - (df["High"] - df["Close"])) / hl_range
    mfv = mfm * df["Volume"]
    vol_sum = df["Volume"].rolling(_CMF_PERIOD).sum()
    cmf_ser = mfv.rolling(_CMF_PERIOD).sum() / vol_sum
    val = float(cmf_ser.iloc[-1])
    return round(val, 4) if np.isfinite(val) else 0.0


def compute_money_flow_score(ticker: str, df: Optional[pd.DataFrame] = None) -> Tuple[float, float, float]:
    """Return (score, mfi_value, cmf_value).

    score ∈ [−1.0, +1.0] — CMF only (v3).
    Positive = institutional accumulation / buying pressure.
    Negative = distribution / selling pressure.
    Returns (0.0, 50.0, 0.0) when data is insufficient.

    ``df``: optional pre-fetched OHLCV frame (any timeframe). When ``None`` the
    daily cache-first fetch is used — identical to the legacy behaviour.
    """
    if df is None:
        df = _get_ohlcv(ticker)

    required_cols = {"High", "Low", "Close", "Volume"}
    if df.empty or len(df) < _MIN_ROWS or not required_cols.issubset(df.columns):
        logger.debug(f"[money_flow] {ticker}: insufficient data ({len(df)} rows)")
        return 0.0, 50.0, 0.0

    # Rolling windows at the LAST bar only need the trailing slice — identical
    # outputs, none of the full-history cost (frames are ~5,000 bars deep).
    df = df.tail(_TAIL_BARS).copy()
    for col in required_cols:
        df[col] = df[col].astype(float)
    df["Volume"] = df["Volume"].replace(0, np.nan).fillna(1.0)

    try:
        mfi = _compute_mfi(df)
        cmf = _compute_cmf(df)
    except Exception as exc:
        logger.debug(f"[money_flow] {ticker}: computation error — {exc}")
        return 0.0, 50.0, 0.0

    # CMF: directional — positive = institutional accumulation = bullish.
    cmf_score = float(np.tanh(cmf / 0.15))
    score = round(max(-1.0, min(1.0, float(np.tanh(cmf_score / 0.6)))), 3)

    logger.debug(
        f"[money_flow] {ticker}: cmf={cmf:+.3f}→score={score:+.3f}  "
        f"(mfi={mfi:.1f} aux-only)"
    )
    return score, round(mfi, 2), round(cmf, 4)
