"""Volume Profile — Point of Control / Value Area (2026-07-08, panel-first).

Auction-market read of WHERE the volume traded, not when: a volume-at-price
histogram over the trailing quarter. The Point of Control (POC — the
most-traded price) and the 70% Value Area around it are levels institutional
desks actually reference. One coherent hypothesis, per auction market theory:

  • price ABOVE the value-area high → initiative buying / acceptance above
    value → bullish continuation (scaled by ATR-distance beyond the band);
  • price BELOW the value-area low → acceptance below value → bearish;
  • price INSIDE the value area → balance: a small POC-gravity score
    (×0.35, like the squeeze's anticipatory haircut) toward the magnet —
    below POC pulls up, above POC pulls down.

Daily-bar approximation: each session's volume is spread UNIFORMLY across its
[Low, High] range over fixed price bins — deterministic, no intraday data
needed. PANEL-FIRST at weight 0 (same contract as classic_anomalies):
IC-tracked + trade-attributed, excluded from combined_score / coherence /
sources_agreeing / the exit consensus. Daily-only. 0.0 = no view (missing/zero
volume fails closed).
"""

from typing import Optional, Tuple

import numpy as np
import pandas as pd
from loguru import logger

from src.data.cache import load_ohlcv
from src.data.market_data import get_history

_LOOKBACK   = 60     # ~one quarter — the classic composite profile
_MIN_ROWS   = 60
_BINS       = 40
_VALUE_AREA = 0.70   # market-profile convention
_ATR_LEN    = 14
_TANH_ATR   = 1.5    # 1.5 ATRs beyond the band ≈ 0.76
_GRAVITY    = 0.35   # inside-value POC-gravity haircut
_DEADBAND   = 0.05


def _get_ohlcv(ticker: str) -> pd.DataFrame:
    cached = load_ohlcv(ticker)
    if cached is not None and len(cached) >= _MIN_ROWS:
        return cached
    return get_history(ticker, period="6mo")


def _profile(h: np.ndarray, l: np.ndarray, v: np.ndarray) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """(bin_centers, volume_per_bin) — each day's volume spread uniformly over
    its [low, high] overlap with fixed bins across the window's price range."""
    lo, hi = float(np.min(l)), float(np.max(h))
    if not (np.isfinite(lo) and np.isfinite(hi)) or hi <= lo:
        return None
    edges = np.linspace(lo, hi, _BINS + 1)
    vol_at = np.zeros(_BINS)
    for dl, dh, dv in zip(l, h, v):
        if not (np.isfinite(dl) and np.isfinite(dh) and np.isfinite(dv)) or dv <= 0:
            continue
        span = max(dh - dl, 1e-12)
        # Overlap of [dl, dh] with each bin, as a fraction of the day's range.
        overlap = np.clip(np.minimum(edges[1:], dh) - np.maximum(edges[:-1], dl), 0.0, None)
        vol_at += dv * overlap / span
    if vol_at.sum() <= 0:
        return None
    centers = (edges[:-1] + edges[1:]) / 2.0
    return centers, vol_at


def _value_area(centers: np.ndarray, vol_at: np.ndarray) -> Tuple[float, float, float]:
    """(poc, val, vah) — expand from the POC bin toward the higher-volume
    neighbour until the accumulated share reaches the value-area fraction
    (the standard market-profile construction)."""
    poc_i = int(np.argmax(vol_at))
    total = float(vol_at.sum())
    lo_i = hi_i = poc_i
    acc = float(vol_at[poc_i])
    while acc / total < _VALUE_AREA and (lo_i > 0 or hi_i < len(vol_at) - 1):
        below = vol_at[lo_i - 1] if lo_i > 0 else -1.0
        above = vol_at[hi_i + 1] if hi_i < len(vol_at) - 1 else -1.0
        if above >= below:
            hi_i += 1
            acc += float(vol_at[hi_i])
        else:
            lo_i -= 1
            acc += float(vol_at[lo_i])
    return float(centers[poc_i]), float(centers[lo_i]), float(centers[hi_i])


def compute_volume_profile_score(ticker: str, df: Optional[pd.DataFrame] = None) -> Tuple[float, str, float]:
    """Return (score, label, poc_dist_pct) — label ∈ ABOVE_VALUE | BELOW_VALUE |
    IN_VALUE | NO_DATA; ``poc_dist_pct`` = close vs POC in % (+ = above the POC)."""
    if df is None:
        df = _get_ohlcv(ticker)
    if df is None or df.empty or len(df) < _MIN_ROWS:
        logger.debug(f"[vol_profile] {ticker}: insufficient data ({0 if df is None else len(df)} rows)")
        return 0.0, "NO_DATA", 0.0
    for col in ("Close", "High", "Low", "Volume"):
        if col not in df.columns:
            return 0.0, "NO_DATA", 0.0

    w = df.tail(_LOOKBACK)
    frame = pd.DataFrame({
        "c": pd.to_numeric(w["Close"], errors="coerce"),
        "h": pd.to_numeric(w["High"], errors="coerce"),
        "l": pd.to_numeric(w["Low"], errors="coerce"),
        "v": pd.to_numeric(w["Volume"], errors="coerce"),
    }).dropna()
    if len(frame) < _MIN_ROWS or (frame["c"] <= 0).any():
        return 0.0, "NO_DATA", 0.0

    prof = _profile(frame["h"].to_numpy(), frame["l"].to_numpy(), frame["v"].to_numpy())
    if prof is None:
        return 0.0, "NO_DATA", 0.0                       # zero-volume window fails closed
    centers, vol_at = prof
    poc, val, vah = _value_area(centers, vol_at)

    c, h, l = frame["c"], frame["h"], frame["l"]
    tr = pd.concat([h - l, (h - c.shift(1)).abs(), (l - c.shift(1)).abs()], axis=1).max(axis=1)
    atr = float(tr.rolling(_ATR_LEN).mean().iloc[-1])
    last = float(c.iloc[-1])
    poc_dist_pct = round((last / poc - 1.0) * 100, 2) if poc > 0 else 0.0
    if not np.isfinite(atr) or atr <= 0:
        return 0.0, "NO_DATA", poc_dist_pct

    if last > vah:
        score = float(np.tanh((last - vah) / (_TANH_ATR * atr)))
        label = "ABOVE_VALUE"
    elif last < val:
        score = float(-np.tanh((val - last) / (_TANH_ATR * atr)))
        label = "BELOW_VALUE"
    else:
        # Balance: gravitate toward the POC magnet (below POC → pull up).
        score = float(_GRAVITY * np.tanh((poc - last) / (_TANH_ATR * atr)))
        label = "IN_VALUE"
    if not np.isfinite(score) or abs(score) < _DEADBAND:
        return 0.0, label, poc_dist_pct
    logger.debug(f"[vol_profile] {ticker}: {label}  poc={poc:.2f}({poc_dist_pct:+.1f}%)  "
                 f"va=[{val:.2f},{vah:.2f}]  score={score:+.3f}")
    return round(score, 3), label, poc_dist_pct
