"""Anchored VWAP — 52-week high/low anchors, positioning semantics (2026-07-08).

A VWAP anchored at a reference EVENT is the average price every participant has
paid since that event — a level institutional desks actually defend. Two
structural anchors that exist for every name with a year of history:

  • the 52-week HIGH-close day — everyone who bought since the top; price above
    this AVWAP means the average post-top buyer is in profit (overhead supply
    absorbed), below it means trapped supply overhead.
  • the 52-week LOW-close day — everyone who bought since the bottom; price
    above it means the average post-bottom buyer defends their gain (support).

score = mean of tanh(distance/ATR) for the two legs: above BOTH anchors →
strongly bullish, below both → strongly bearish, between them → small/mixed.

SIGN CONVENTION NOTE: this is POSITIONING (above the anchor = support = bullish
continuation), the OPPOSITE reading of the rolling ``vwap`` method, which
scores distance from a 20-day VWAP as MEAN-REVERSION (above → bearish). They
are different hypotheses about different reference frames; the signals panel
adjudicates both. PANEL-FIRST at weight 0 (same contract as classic_anomalies):
IC-tracked + trade-attributed, excluded from combined_score / coherence /
sources_agreeing / the exit consensus. Daily-only. 0.0 = no view.
"""

from typing import Optional, Tuple

import numpy as np
import pandas as pd
from loguru import logger

from src.data.cache import load_ohlcv
from src.data.market_data import get_history

_WINDOW    = 252   # the anchor-search year
_MIN_ROWS  = 200   # an honest year of anchors (same floor as hi52)
_ATR_LEN   = 14
_TANH_Z    = 2.0   # z of 2 ATRs from an anchor ≈ 0.76 on that leg
_DEADBAND  = 0.05


def _get_ohlcv(ticker: str) -> pd.DataFrame:
    cached = load_ohlcv(ticker)
    if cached is not None and len(cached) >= _MIN_ROWS:
        return cached
    return get_history(ticker, period="18mo")


def _avwap_from(idx_pos: int, tp: pd.Series, vol: pd.Series) -> float:
    v = vol.iloc[idx_pos:]
    p = tp.iloc[idx_pos:]
    denom = float(v.sum())
    if denom <= 0 or not np.isfinite(denom):
        return float("nan")
    return float((p * v).sum() / denom)


def compute_anchored_vwap_score(ticker: str, df: Optional[pd.DataFrame] = None) -> Tuple[float, float, float]:
    """Return (score, dist_hi_pct, dist_lo_pct) — the signed distances (%) of the
    last close from the high-anchored and low-anchored VWAPs. (0.0, 0, 0) = no
    view (short history, missing/zero volume — fail-closed — or inside the
    deadband)."""
    if df is None:
        df = _get_ohlcv(ticker)
    if df is None or df.empty or len(df) < _MIN_ROWS:
        logger.debug(f"[avwap] {ticker}: insufficient data ({0 if df is None else len(df)} rows)")
        return 0.0, 0.0, 0.0
    for col in ("Close", "High", "Low", "Volume"):
        if col not in df.columns:
            return 0.0, 0.0, 0.0

    w = df.tail(_WINDOW).copy()
    close = pd.to_numeric(w["Close"], errors="coerce")
    high  = pd.to_numeric(w["High"], errors="coerce")
    low   = pd.to_numeric(w["Low"], errors="coerce")
    vol   = pd.to_numeric(w["Volume"], errors="coerce")
    frame = pd.DataFrame({"c": close, "h": high, "l": low, "v": vol}).dropna()
    if len(frame) < _MIN_ROWS or (frame["c"] <= 0).any() or (frame["v"] < 0).any():
        return 0.0, 0.0, 0.0
    c, h, l, v = frame["c"], frame["h"], frame["l"], frame["v"]
    last = float(c.iloc[-1])

    tp = (h + l + c) / 3.0
    hi_pos = int(np.argmax(c.to_numpy()))
    lo_pos = int(np.argmin(c.to_numpy()))
    avwap_hi = _avwap_from(hi_pos, tp, v)
    avwap_lo = _avwap_from(lo_pos, tp, v)

    tr  = pd.concat([h - l, (h - c.shift(1)).abs(), (l - c.shift(1)).abs()], axis=1).max(axis=1)
    atr = float(tr.rolling(_ATR_LEN).mean().iloc[-1])
    if not all(np.isfinite(x) for x in (avwap_hi, avwap_lo, atr)) or atr <= 0:
        return 0.0, 0.0, 0.0

    z_hi = (last - avwap_hi) / atr
    z_lo = (last - avwap_lo) / atr
    score = float((np.tanh(z_hi / _TANH_Z) + np.tanh(z_lo / _TANH_Z)) / 2.0)
    dist_hi_pct = round((last / avwap_hi - 1.0) * 100, 2)
    dist_lo_pct = round((last / avwap_lo - 1.0) * 100, 2)
    if not np.isfinite(score) or abs(score) < _DEADBAND:
        return 0.0, dist_hi_pct, dist_lo_pct
    logger.debug(f"[avwap] {ticker}: hi_anchor {dist_hi_pct:+.1f}%  lo_anchor {dist_lo_pct:+.1f}%  "
                 f"z=({z_hi:+.2f},{z_lo:+.2f})  score={score:+.3f}")
    return round(score, 3), dist_hi_pct, dist_lo_pct
