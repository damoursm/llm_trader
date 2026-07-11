"""Residual Momentum — beta-adjusted 12-1 momentum (2026-07-08, panel-first).

Blitz, Huij & Martens (2011): momentum computed on the RESIDUALS of a factor
regression carries a similar premium to raw momentum with roughly half the
volatility and far smaller crashes — because raw momentum is contaminated by
a time-varying beta tilt (in an up-market the "winners" are just the high-beta
names, and they crater together on reversals). Daily adaptation with the
market factor:

  1. beta = cov(r_stock, r_SPY) / var(r_SPY) over the trailing 252 sessions.
  2. resid_t = r_stock,t − beta × r_SPY,t (daily residuals), SUMMED over the
     window excluding the most-recent skip-month (the same 12-1 convention as
     `mom_12_1`). The arithmetic sum is the BHM construction — NOT compound
     stock return − beta × compound market return, which leaks compounding
     convexity for beta ≠ 1 (a pure 2-beta market ride must net ~zero here).
  3. z = resid_12_1 / (σ_daily_residual × √window)  — the BHM vol-scaling, so
     a name is ranked by residual return PER UNIT of residual risk.
  4. score = tanh(z / 1.5).

Distinct from `sector_momentum` / `market_momentum`, which subtract the
benchmark return one-for-one (an implicit beta of 1): here the ESTIMATED beta
is removed, so a 2-beta name in a +10% market needs > +20% to score positive.
PANEL-FIRST at weight 0 (same contract as classic_anomalies): IC-tracked +
trade-attributed, excluded from combined_score / coherence / sources_agreeing
/ the exit consensus. Daily-only. 0.0 = no view.
"""

import threading
import time
from typing import Optional, Tuple

import numpy as np
import pandas as pd
from loguru import logger

from src.data.cache import load_ohlcv
from src.data.market_data import get_history

_BENCHMARK   = "SPY"
_LOOKBACK    = 252   # regression + momentum window (needs the full year, like mom_12_1)
_SKIP        = 21    # skip-month (the short-term-reversal regime)
_MIN_ALIGNED = 200   # minimum overlapping sessions after the date join
_TANH_SCALE  = 1.5
_MIN_MKT_VAR = 1e-10

# One parsed benchmark frame per ~10-minute bucket — 400 tickers/tick would
# otherwise re-parse SPY.json 400×; the bucket keeps the long-lived scheduler
# from serving a stale frame forever. Lock: build_signals scores tickers on a
# 32-thread pool, so an unlocked cold miss would stampede SPY with concurrent
# fetches. A failed load is memoized as None for the bucket (one attempt per
# ~10 min, not one per ticker).
_BENCH_MEMO: dict = {}
_BENCH_LOCK = threading.Lock()
_MISS = object()


def _benchmark_df() -> Optional[pd.DataFrame]:
    bucket = int(time.time() // 600)
    hit = _BENCH_MEMO.get(bucket, _MISS)
    if hit is not _MISS:
        return hit
    with _BENCH_LOCK:
        hit = _BENCH_MEMO.get(bucket, _MISS)
        if hit is not _MISS:
            return hit
        df = load_ohlcv(_BENCHMARK)
        if df is None or df.empty or len(df) < _LOOKBACK:
            try:
                df = get_history(_BENCHMARK, period="18mo")
            except Exception as e:
                logger.debug(f"[resid_mom] benchmark fetch failed: {e}")
                df = None
        if df is not None and df.empty:
            df = None
        _BENCH_MEMO.clear()
        _BENCH_MEMO[bucket] = df
        return df


def _get_ohlcv(ticker: str) -> pd.DataFrame:
    cached = load_ohlcv(ticker)
    if cached is not None and len(cached) >= _LOOKBACK:
        return cached
    return get_history(ticker, period="18mo")


def compute_residual_momentum_score(ticker: str, df: Optional[pd.DataFrame] = None,
                                    mkt_df: Optional[pd.DataFrame] = None) -> Tuple[float, float, float]:
    """Return (score, resid_12_1_pct, beta). (0.0, 0.0, 0.0) = no view
    (short/unaligned history, degenerate market variance, or non-finite math)."""
    if ticker.upper() == _BENCHMARK:
        return 0.0, 0.0, 1.0                     # the market has no residual vs itself
    if df is None:
        df = _get_ohlcv(ticker)
    if mkt_df is None:
        mkt_df = _benchmark_df()
    if df is None or df.empty or mkt_df is None or mkt_df.empty:
        return 0.0, 0.0, 0.0
    if "Close" not in df.columns or "Close" not in mkt_df.columns:
        return 0.0, 0.0, 0.0

    s = pd.to_numeric(df["Close"], errors="coerce")
    m = pd.to_numeric(mkt_df["Close"], errors="coerce")
    joined = pd.DataFrame({"s": s, "m": m}).dropna()
    joined = joined[(joined["s"] > 0) & (joined["m"] > 0)].tail(_LOOKBACK)
    if len(joined) < _MIN_ALIGNED:
        logger.debug(f"[resid_mom] {ticker}: only {len(joined)} aligned sessions")
        return 0.0, 0.0, 0.0

    rs = joined["s"].pct_change().dropna()
    rm = joined["m"].pct_change().dropna()
    mkt_var = float(rm.var())
    if not np.isfinite(mkt_var) or mkt_var < _MIN_MKT_VAR:
        return 0.0, 0.0, 0.0
    beta = float(rs.cov(rm) / mkt_var)

    skip = min(_SKIP, len(joined) - 2)
    resid_daily = rs - beta * rm
    resid_daily = resid_daily.iloc[: max(1, len(resid_daily) - skip)]   # exclude the skipped month
    sigma = float(resid_daily.std())
    if not np.isfinite(sigma) or sigma < 1e-8:
        return 0.0, 0.0, 0.0

    resid_12_1 = float(resid_daily.sum())
    z = resid_12_1 / (sigma * np.sqrt(len(resid_daily)))
    if not np.isfinite(z):
        return 0.0, 0.0, 0.0
    score = float(np.tanh(z / _TANH_SCALE))
    resid_pct = round(resid_12_1 * 100, 2)
    logger.debug(f"[resid_mom] {ticker}: beta={beta:+.2f}  resid_12_1={resid_pct:+.1f}%  "
                 f"z={z:+.2f}  score={score:+.3f}")
    return round(score, 3), resid_pct, round(beta, 3)
