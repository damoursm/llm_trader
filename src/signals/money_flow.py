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
   - Contrarian interpretation: extreme low = smart money accumulating
   - Score: tanh((50 − MFI) / 20)   maps 0 → 0, 20 → +0.96, 80 → −0.96

2. Chaikin Money Flow (CMF, 20-period)
   - Volume-weighted sum of Money Flow Multiplier over 20 days
   - Positive CMF → accumulation (buyers in control)
   - Negative CMF → distribution (sellers in control)
   - Score: tanh(CMF / 0.15)        maps 0 → 0, ±0.15 → ±0.96

3. On-Balance Volume (OBV) slope z-score
   - Cumulative volume: +volume on up days, −volume on down days
   - Rising OBV slope → sustained buying pressure
   - Score: normalised 21-bar linear regression slope of OBV
   - tanh(obv_z / 1.0)              maps ±1σ → ±0.76

Composite score:
  raw = 0.40 × mfi_score + 0.40 × cmf_score + 0.20 × obv_score
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
    """21-bar OBV linear-regression slope, returned as a z-score over the full series."""
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
    std = slopes_arr.std()
    if std < 1e-8:
        return 0.0
    return float((slopes_arr[-1] - slopes_arr.mean()) / std)


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

    # MFI: contrarian — low MFI = accumulation zone = bullish
    mfi_score = float(np.tanh((50.0 - mfi) / 20.0))

    # CMF: directional — positive = institutional accumulation = bullish
    cmf_score = float(np.tanh(cmf / 0.15))

    # OBV slope normalised — rising trend = buying pressure = bullish
    obv_score = float(np.tanh(obv_z / 1.0))

    composite = 0.40 * mfi_score + 0.40 * cmf_score + 0.20 * obv_score
    raw_score = float(np.tanh(composite / 0.6))
    score = round(max(-1.0, min(1.0, raw_score)), 3)

    logger.debug(
        f"[money_flow] {ticker}: mfi={mfi:.1f}→{mfi_score:+.2f}  "
        f"cmf={cmf:+.3f}→{cmf_score:+.2f}  obv_z={obv_z:+.2f}→{obv_score:+.2f}  "
        f"composite={composite:+.3f}  score={score:+.3f}"
    )
    return score, round(mfi, 2), round(cmf, 4)
