"""
Money Flow Indicators — Accumulation / Distribution signal.

Traders use "money flow" indicators to track volume-adjusted price movement
as a proxy for institutional accumulation (buying) and distribution (selling).
Rising price on shrinking volume often signals a weak move; rising price with
expanding volume signals genuine institutional participation.

Three complementary indicators are combined:

1. Money Flow Index (MFI, 14-period)
   - Oscillator 0–100 using typical price × volume (volume-weighted RSI)
   - MFI < 20 → oversold / accumulation zone  →  bullish reading
   - MFI > 80 → overbought / distribution zone → bearish reading
   - Contrarian interpretation applies ONLY at those extremes: inside the
     35–65 neutral band MFI ABSTAINS and the other components renormalise
     (2026-07-24 — the old linear map read every ordinary uptrend as
     "overbought" and cancelled the trend-following cmf term)
   - Score: 0 inside 35–65; ramps linearly to +1 at MFI 20, −1 at MFI 80

2. Chaikin Money Flow (CMF, 20-period)
   - Volume-weighted sum of Money Flow Multiplier over 20 days
   - Positive CMF → accumulation (buyers in control)
   - Negative CMF → distribution (sellers in control)
   - Score: tanh(CMF / 0.15)        maps 0 → 0, ±0.15 → ±0.96

3. On-Balance Volume (OBV) slope, scale-normalised
   - Cumulative volume: +volume on up days, −volume on down days
   - Rising OBV slope → sustained buying pressure
   - Score: 21-bar regression slope ÷ mean |slope| — SIGN-PRESERVING
     (2026-07-24: was a z-score vs its own history, which measures
     acceleration and read a steadily-falling OBV as neutral-to-bullish)
   - tanh(obv / 1.0)                maps ±1× typical flow → ±0.76

Composite score (weighted mean over the ACTIVE components — an abstaining MFI
drops out of the denominator rather than shrinking the composite toward zero):
  raw = Σ(wᵢ × scoreᵢ) / Σ(wᵢ)   over cmf (0.40), obv (0.20), mfi (0.40 if active)
  score = tanh(raw / 0.6) clamped to [−1, +1]

Cache strategy:
  Prefers the incremental OHLCV chart cache (cache/ohlcv/<TICKER>.json).
  Falls back to a live yfinance fetch on cold cache.
  Works with ENABLE_FETCH_DATA=false when chart caches are populated.
  Minimum 30 bars required; returns (0.0, 50.0, 0.0) when data is insufficient.
"""

from typing import Optional, Tuple

import numpy as np
import pandas as pd
from loguru import logger

from src.data.market_data import get_history
from src.data.cache import load_ohlcv


_MIN_ROWS    = 30    # minimum bars to compute meaningful indicators
_MFI_PERIOD  = 14    # standard MFI lookback
_CMF_PERIOD  = 20    # standard CMF lookback
_OBV_WINDOW  = 21    # OBV slope window (bars per regression)

# MFI neutral band (2026-07-24 directional fix). MFI's contrarian reading is
# only standard at the EXTREMES ("<20 oversold / >80 overbought" — the module
# docstring's own interpretation). The old score, tanh((50−MFI)/20), applied it
# LINEARLY across the whole range, so an ordinary uptrend (MFI≈91) scored −0.97
# "bearish" purely for trending up — exactly cancelling the equally-weighted,
# trend-FOLLOWING cmf term. Measured on synthetic mirror series the composite
# came out +0.19 in an uptrend and +0.17 in a downtrend: the method could not
# express direction at all, which matches its ~0 IC on the live panel
# (buy −0.018 / sell −0.006 over 120 days). Inside the band MFI now abstains
# and the remaining components renormalise, so the composite reads clean
# accumulation/distribution; outside it MFI applies its contrarian override.
_MFI_NEUTRAL_LO = 35.0
_MFI_NEUTRAL_HI = 65.0
_MFI_EXTREME_LO = 20.0   # ≤ this → full-strength bullish (oversold)
_MFI_EXTREME_HI = 80.0   # ≥ this → full-strength bearish (overbought)


def _get_ohlcv(ticker: str) -> pd.DataFrame:
    cached = load_ohlcv(ticker)
    if cached is not None and len(cached) >= _MIN_ROWS:
        return cached
    return get_history(ticker, period="18mo")


def _compute_mfi(df: pd.DataFrame) -> float:
    """14-period Money Flow Index, returns 0–100."""
    typical = (df["High"] + df["Low"] + df["Close"]) / 3
    mf = typical * df["Volume"]

    pos_mf = np.zeros(len(df))
    neg_mf = np.zeros(len(df))
    typ_arr = typical.values
    mf_arr  = mf.values

    for i in range(1, len(df)):
        if typ_arr[i] > typ_arr[i - 1]:
            pos_mf[i] = mf_arr[i]
        elif typ_arr[i] < typ_arr[i - 1]:
            neg_mf[i] = mf_arr[i]

    pos_ser = pd.Series(pos_mf, index=df.index)
    neg_ser = pd.Series(neg_mf, index=df.index)
    pos_sum = float(pos_ser.rolling(_MFI_PERIOD).sum().iloc[-1])
    neg_sum = float(neg_ser.rolling(_MFI_PERIOD).sum().iloc[-1])

    if neg_sum == 0:
        return 100.0
    mfr = pos_sum / neg_sum
    return round(100.0 - 100.0 / (1.0 + mfr), 2)


def _compute_cmf(df: pd.DataFrame) -> float:
    """20-period Chaikin Money Flow, returns [−1, +1]."""
    hl_range = (df["High"] - df["Low"]).replace(0, np.nan)
    mfm = ((df["Close"] - df["Low"]) - (df["High"] - df["Close"])) / hl_range
    mfv = mfm * df["Volume"]
    vol_sum = df["Volume"].rolling(_CMF_PERIOD).sum()
    cmf_ser = mfv.rolling(_CMF_PERIOD).sum() / vol_sum
    val = float(cmf_ser.iloc[-1])
    return round(val, 4) if np.isfinite(val) else 0.0


def _compute_obv_z(df: pd.DataFrame) -> float:
    """21-bar OBV linear-regression slope, scale-normalised and SIGN-PRESERVING.

    Returns ``last_slope / mean(|slope|)`` — a dimensionless measure of how
    hard volume is flowing in (+) or out (−) relative to this name's typical
    flow intensity.

    It used to return a z-score of the slope against its own history
    (``(last − mean) / std``), which measures ACCELERATION, not direction, and
    so destroyed the sign the composite depends on: in a steady decline every
    slope is negative, the latest is near the mean, and the z-score lands at
    ~0 — or POSITIVE when the drop merely slows. Measured on a mirror-image
    downtrend the raw slope was −2.56M (correctly bearish) while the z-score
    read +1.27 (bullish). Dividing by the mean ABSOLUTE slope keeps the scale
    normalisation but preserves the sign (2026-07-24)."""
    close  = df["Close"].values
    volume = df["Volume"].values
    obv    = np.zeros(len(df))

    for i in range(1, len(df)):
        if close[i] > close[i - 1]:
            obv[i] = obv[i - 1] + volume[i]
        elif close[i] < close[i - 1]:
            obv[i] = obv[i - 1] - volume[i]
        else:
            obv[i] = obv[i - 1]

    if len(obv) < _OBV_WINDOW:
        return 0.0

    x = np.arange(_OBV_WINDOW, dtype=float)
    slopes = [
        np.polyfit(x, obv[i - _OBV_WINDOW + 1:i + 1], 1)[0]
        for i in range(_OBV_WINDOW - 1, len(obv))
    ]
    slopes_arr = np.array(slopes)
    scale = np.abs(slopes_arr).mean()
    if not np.isfinite(scale) or scale < 1e-8:
        return 0.0
    return float(slopes_arr[-1] / scale)


def _mfi_contrarian_score(mfi: float) -> float:
    """MFI → [−1, +1] contrarian score, ABSTAINING inside the neutral band.

    Oversold (low MFI) is bullish, overbought bearish — but only outside
    ``_MFI_NEUTRAL_LO/HI``. Between them the reading carries no contrarian
    information and the old linear map merely inverted the prevailing trend,
    cancelling the trend-following cmf/obv terms (see the constants above).
    Ramps linearly from 0 at the band edge to ±1 at the extreme."""
    if not np.isfinite(mfi):
        return 0.0
    if mfi <= _MFI_NEUTRAL_LO:                       # oversold → bullish
        span = _MFI_NEUTRAL_LO - _MFI_EXTREME_LO
        return float(min(1.0, (_MFI_NEUTRAL_LO - mfi) / span)) if span > 0 else 1.0
    if mfi >= _MFI_NEUTRAL_HI:                       # overbought → bearish
        span = _MFI_EXTREME_HI - _MFI_NEUTRAL_HI
        return -float(min(1.0, (mfi - _MFI_NEUTRAL_HI) / span)) if span > 0 else -1.0
    return 0.0


def compute_money_flow_score(ticker: str, df: Optional[pd.DataFrame] = None) -> Tuple[float, float, float]:
    """Return (score, mfi_value, cmf_value).

    score ∈ [−1.0, +1.0].
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

    df = df.copy()
    for col in required_cols:
        df[col] = df[col].astype(float)
    df["Volume"] = df["Volume"].replace(0, np.nan).fillna(1.0)

    try:
        mfi   = _compute_mfi(df)
        cmf   = _compute_cmf(df)
        obv_z = _compute_obv_z(df)
    except Exception as exc:
        logger.debug(f"[money_flow] {ticker}: computation error — {exc}")
        return 0.0, 50.0, 0.0

    # MFI: contrarian, but ONLY at the extremes where that reading is standard.
    # Inside the neutral band it abstains (weight renormalised away) instead of
    # fighting the trend-following cmf/obv terms — see _MFI_NEUTRAL_LO/HI.
    mfi_score = _mfi_contrarian_score(mfi)
    mfi_active = mfi_score != 0.0

    # CMF: directional — positive = institutional accumulation = bullish
    cmf_score = float(np.tanh(cmf / 0.15))

    # OBV slope normalised — rising trend = buying pressure = bullish
    obv_score = float(np.tanh(obv_z / 1.0))

    # Weighted mean over the ACTIVE components, so an abstaining MFI does not
    # simply shrink the composite toward zero (the aggregator's own idiom).
    parts = [(0.40, cmf_score), (0.20, obv_score)]
    if mfi_active:
        parts.append((0.40, mfi_score))
    wsum = sum(w for w, _ in parts)
    composite = sum(w * v for w, v in parts) / wsum if wsum else 0.0

    raw_score = float(np.tanh(composite / 0.6))
    score = round(max(-1.0, min(1.0, raw_score)), 3)

    logger.debug(
        f"[money_flow] {ticker}: mfi={mfi:.1f}→{mfi_score:+.2f}"
        f"{'' if mfi_active else ' (neutral band — abstains)'}  "
        f"cmf={cmf:+.3f}→{cmf_score:+.2f}  obv_z={obv_z:+.2f}→{obv_score:+.2f}  "
        f"composite={composite:+.3f}  score={score:+.3f}"
    )
    return score, round(mfi, 2), round(cmf, 4)
