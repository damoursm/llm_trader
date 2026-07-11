"""TTM Squeeze — Bollinger-inside-Keltner volatility coil (2026-07-08, panel-first).

John Carter's squeeze: when the Bollinger Bands (20, 2σ) contract INSIDE the
Keltner Channels (20, 1.5×ATR), volatility is coiling — and the release
(bands re-expanding outside the channel) reliably precedes a directional
expansion. The squeeze itself is a STATE, not a direction; per the house
score-sign convention the direction comes from Carter's momentum oscillator
(linear-regression value of close minus the Donchian/SMA midline), so the
emitted score is signed by momentum:

  • squeeze FIRED within the last few bars (after a real coil) → full-strength
    score in the momentum direction — the classic entry.
  • squeeze currently ON → small anticipatory score (×0.35) in the momentum
    direction — the coil is loaded but not yet released.
  • no squeeze → 0.0 (no view) — this method only speaks around coils, so the
    panel's IC measures exactly the release edge, not general momentum.

PANEL-FIRST at weight 0 (same contract as classic_anomalies): scored on every
ticker + IC-tracked + trade-attributed, but excluded from combined_score,
coherence, sources_agreeing, and the exit consensus until the IC earns it.
Daily-only by design. Score ∈ [-1, +1]; 0.0 = no view.
"""

from typing import Optional, Tuple

import numpy as np
import pandas as pd
from loguru import logger

from src.data.cache import load_ohlcv
from src.data.market_data import get_history

_MIN_ROWS         = 60    # enough bars for the 20-bar bands + a momentum window
_WINDOW           = 20    # BB / Keltner / Donchian / momentum window (Carter's default)
_BB_STD           = 2.0
_KC_ATR_MULT      = 1.5
_MIN_SQUEEZE_BARS = 5     # a coil shorter than this is noise, not compression
_RELEASE_WINDOW   = 3     # a release counts as "fired" for this many bars
_ANTICIPATORY     = 0.35  # confidence haircut while the squeeze is still ON
_DEADBAND         = 0.05  # |score| below this → no view


def _get_ohlcv(ticker: str) -> pd.DataFrame:
    cached = load_ohlcv(ticker)
    if cached is not None and len(cached) >= _MIN_ROWS:
        return cached
    return get_history(ticker, period="6mo")


def compute_ttm_squeeze_score(ticker: str, df: Optional[pd.DataFrame] = None) -> Tuple[float, str, int]:
    """Return (score, label, bars) — label ∈ NONE|SQUEEZE_ON|FIRED_UP|FIRED_DOWN;
    ``bars`` = bars spent in the coil (SQUEEZE_ON) / bars since the release
    (FIRED_*) / 0 (NONE)."""
    if df is None:
        df = _get_ohlcv(ticker)
    if df is None or df.empty or len(df) < _MIN_ROWS:
        logger.debug(f"[squeeze] {ticker}: insufficient data ({0 if df is None else len(df)} rows)")
        return 0.0, "NONE", 0
    for col in ("Close", "High", "Low"):
        if col not in df.columns:
            return 0.0, "NONE", 0

    close = pd.to_numeric(df["Close"], errors="coerce")
    high  = pd.to_numeric(df["High"], errors="coerce")
    low   = pd.to_numeric(df["Low"], errors="coerce")
    frame = pd.DataFrame({"c": close, "h": high, "l": low}).dropna()
    if len(frame) < _MIN_ROWS or (frame["c"] <= 0).any():
        return 0.0, "NONE", 0
    c, h, l = frame["c"], frame["h"], frame["l"]

    # Bollinger (SMA mid) vs Keltner (same SMA mid, ATR-based half-width).
    mid    = c.rolling(_WINDOW).mean()
    bb_dev = _BB_STD * c.rolling(_WINDOW).std()
    tr     = pd.concat([h - l, (h - c.shift(1)).abs(), (l - c.shift(1)).abs()], axis=1).max(axis=1)
    atr    = tr.rolling(_WINDOW).mean()
    kc_dev = _KC_ATR_MULT * atr
    squeeze_on = (bb_dev < kc_dev)

    if not bool(squeeze_on.notna().iloc[-1]) or not np.isfinite(float(atr.iloc[-1])) or float(atr.iloc[-1]) <= 0:
        return 0.0, "NONE", 0

    # Carter momentum: linreg-fitted value of close − (Donchian mid + SMA)/2.
    donch_mid = (h.rolling(_WINDOW).max() + l.rolling(_WINDOW).min()) / 2.0
    delta = (c - (donch_mid + mid) / 2.0).dropna().tail(_WINDOW)
    if len(delta) < _WINDOW:
        return 0.0, "NONE", 0
    x = np.arange(len(delta), dtype=float)
    slope, intercept = np.polyfit(x, delta.to_numpy(dtype=float), 1)
    mom = slope * x[-1] + intercept                      # fitted value at the last bar
    mom_norm = mom / float(atr.iloc[-1])
    if not np.isfinite(mom_norm):
        return 0.0, "NONE", 0

    # State machine over the tail of the squeeze series.
    flags = squeeze_on.dropna().astype(bool)
    on_now = bool(flags.iloc[-1])

    def _run_length(series: pd.Series) -> int:
        n = 0
        for v in series.iloc[::-1]:
            if not v:
                break
            n += 1
        return n

    if on_now:
        run = _run_length(flags)
        if run < _MIN_SQUEEZE_BARS:
            return 0.0, "NONE", 0
        score = float(_ANTICIPATORY * np.tanh(mom_norm))
        if abs(score) < _DEADBAND:
            return 0.0, "NONE", 0
        logger.debug(f"[squeeze] {ticker}: ON {run} bars  mom={mom_norm:+.2f}  score={score:+.3f}")
        return round(score, 3), "SQUEEZE_ON", run

    # Not on now — did a real coil release within the last few bars?
    tail = flags.iloc[-(_RELEASE_WINDOW + 1):]
    off_run = _run_length(~tail)                          # bars since the release (≥1)
    if 1 <= off_run <= _RELEASE_WINDOW and len(flags) > off_run:
        prior = flags.iloc[:-off_run]
        coil = _run_length(prior)
        if coil >= _MIN_SQUEEZE_BARS:
            score = float(np.tanh(mom_norm))
            if abs(score) < _DEADBAND:
                return 0.0, "NONE", 0
            label = "FIRED_UP" if score > 0 else "FIRED_DOWN"
            logger.debug(f"[squeeze] {ticker}: {label} (coil {coil} bars, {off_run} bars ago)  "
                         f"mom={mom_norm:+.2f}  score={score:+.3f}")
            return round(score, 3), label, off_run
    return 0.0, "NONE", 0
