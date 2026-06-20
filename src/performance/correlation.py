"""Realized pairwise correlations for position-sizing.

Used by ``tracker.record_new_trades`` to detect cross-sector factor
concentration the static per-sector cap misses (e.g. NVDA + AVGO + SMH
all loading semis even though SMH lives in the ETF bucket; three high-beta
growth names rated independently moving together when growth rolls).

Pearson correlation on **log returns** (returns are symmetric, immune to
scale, and additive — the standard quant choice) over the last N trading
days, read from the on-disk OHLCV cache. No live fetches happen here —
when a ticker has no cache, the correlation is reported as ``None`` and
the caller treats it as "no view" (no haircut).

Public entry points:
    * ``pairwise_correlation(a, b, days)``         → single ρ or ``None``
    * ``mean_pairwise_correlation(cand, others)``  → (mean_ρ, max_ρ, n_used)
    * ``correlation_haircut(cand, others, dir)``   → final multiplier + diagnostics
"""

from __future__ import annotations

from datetime import date
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
from loguru import logger

from config import settings
from src.data.cache import load_ohlcv

# Per-run correlation-sizing health. The haircut silently drops pairs it can't
# compute (missing OHLCV cache, too little overlap) — a *failure* is then
# indistinguishable from a genuine ρ≈0, weakening the concentration haircut
# unseen. These counters let ``_collect_sources`` surface a high failure rate
# (run_sources → Data Quality tab + health banner).
_CORR_HEALTH: dict = {"attempted": 0, "failed": 0}


def reset_correlation_health() -> None:
    """Clear the per-run correlation-sizing health counters (call at run start)."""
    _CORR_HEALTH.update(attempted=0, failed=0)


def get_correlation_health() -> dict:
    """Snapshot of this run's correlation-pair compute outcomes."""
    return dict(_CORR_HEALTH)


def _load_log_returns(ticker: str, days: int) -> Optional[pd.Series]:
    """Return a date-indexed log-return Series for *ticker* over the last
    *days* trading bars. ``None`` when the cache is missing / too short.

    Dedupe is critical: when a ticker's OHLCV cache holds rows from both
    Polygon (tz-naive midnight UTC) and yfinance (tz-aware ET midnight)
    for the same session — which can happen across a force-refresh — the
    ``.normalize()`` step below collapses both onto the same date. Without
    a dedupe, downstream ``pd.concat(join="inner")`` raises
    ``cannot reindex on an axis with duplicate labels``.
    """
    df = load_ohlcv(ticker)
    if df is None or df.empty or "Close" not in df.columns:
        return None
    close = df["Close"].astype(float)
    # Normalise the index to date-only so cache rows coming from yfinance
    # (tz-aware) and Polygon (tz-naive) align without timezone clashes.
    idx = pd.DatetimeIndex(close.index)
    try:
        if idx.tz is not None:
            idx = idx.tz_convert("America/New_York").tz_localize(None)
    except (TypeError, AttributeError):
        pass
    close.index = pd.DatetimeIndex(idx).normalize()
    close = close[close > 0]
    # Drop duplicate dates — keep the last value per session (which is the
    # most-recently-written cache row). Without this, the post-normalize
    # index can hold same-date duplicates from mixed-provider caches and
    # subsequent pd.concat() calls explode.
    close = close[~close.index.duplicated(keep="last")]
    if len(close) < 5:
        return None
    log_ret = np.log(close).diff().dropna()
    # Keep only the most recent ``days`` bars
    return log_ret.tail(days)


def pairwise_correlation(ticker_a: str, ticker_b: str, days: Optional[int] = None) -> Optional[float]:
    """Pearson correlation of log returns over the last *days* bars.

    Returns ``None`` when:
      * either ticker has no usable OHLCV cache,
      * the date overlap is < ``correlation_min_overlap_days``,
      * either series has zero variance (constant price, e.g. delisted).

    Same-ticker pairs return 1.0 by convention.
    """
    if not ticker_a or not ticker_b:
        return None
    if ticker_a.upper() == ticker_b.upper():
        return 1.0
    n = int(days or settings.correlation_lookback_days)
    a = _load_log_returns(ticker_a, n)
    b = _load_log_returns(ticker_b, n)
    if a is None or b is None:
        return None
    df = pd.concat([a, b], axis=1, join="inner").dropna()
    if len(df) < max(2, int(settings.correlation_min_overlap_days)):
        return None
    x = df.iloc[:, 0].values
    y = df.iloc[:, 1].values
    if float(np.std(x)) < 1e-10 or float(np.std(y)) < 1e-10:
        return None
    try:
        r = float(np.corrcoef(x, y)[0, 1])
    except Exception:
        return None
    if np.isnan(r):
        return None
    # Clip floating-point creep outside [-1, 1].
    return max(-1.0, min(1.0, r))


def mean_pairwise_correlation(
    candidate: str,
    others: Iterable[str],
    days: Optional[int] = None,
) -> Tuple[Optional[float], Optional[float], int, Dict[str, float]]:
    """Return ``(mean_ρ, max_ρ, n_pairs_used, per_ticker_rho)``.

    Same-ticker entries in *others* are skipped (you can't be correlated
    with yourself for sizing purposes). Pairs we can't compute (missing
    cache, etc.) are dropped from the mean — the function returns
    ``(None, None, 0, {})`` only when no usable pair exists.
    """
    rhos: Dict[str, float] = {}
    cand = candidate.upper()
    for other in others:
        other_up = other.upper()
        if other_up == cand:
            continue
        _CORR_HEALTH["attempted"] += 1
        r = pairwise_correlation(cand, other_up, days=days)
        if r is None:
            _CORR_HEALTH["failed"] += 1   # couldn't compute — NOT a true ρ≈0
            continue
        rhos[other_up] = r
    if not rhos:
        return None, None, 0, {}
    values = list(rhos.values())
    return float(np.mean(values)), float(max(values)), len(values), rhos


def _haircut_multiplier(mean_corr: float) -> float:
    """Linear-interpolate between low/high thresholds.

    ρ̄ ≤ low_threshold  → 1.0× (full size)
    ρ̄ ≥ high_threshold → min_multiplier
    in between          → linear interpolation
    """
    lo  = float(settings.correlation_low_threshold)
    hi  = float(settings.correlation_high_threshold)
    mn  = float(settings.correlation_min_multiplier)
    if hi <= lo:
        return 1.0
    if mean_corr <= lo:
        return 1.0
    if mean_corr >= hi:
        return mn
    # Linear: t = 0 at lo → 1.0; t = 1 at hi → mn
    t = (mean_corr - lo) / (hi - lo)
    return round(1.0 + t * (mn - 1.0), 3)


def correlation_haircut(
    candidate: str,
    same_direction_positions: List[dict],
    days: Optional[int] = None,
) -> Tuple[float, dict]:
    """Compute the size haircut for *candidate* given currently-OPEN
    same-direction positions (each a tracker trade dict with ``ticker`` and
    ``position_size_multiplier``).

    Returns ``(multiplier, diagnostics)`` where *diagnostics* is:

        {
          "mean_corr":          float or None,
          "max_corr":           float or None,
          "n_pairs":            int,
          "weighted_exposure":  float,       # Σ(size · |ρ|)
          "portfolio_cap_hit":  bool,        # True if adding candidate would exceed cap
          "per_ticker":         {ticker: ρ},
        }

    Behaviour:
      * Empty positions list, or no usable correlations → ``(1.0, …)``.
      * The portfolio cap check uses |ρ| (a strongly negative correlation
        is also concentrated factor exposure — just in the inverse).
      * The multiplier curve is linear between configured thresholds.
    """
    diagnostics = {
        "mean_corr": None,
        "max_corr":  None,
        "n_pairs":   0,
        "weighted_exposure": 0.0,
        "portfolio_cap_hit": False,
        "per_ticker": {},
    }
    if not settings.enable_correlation_sizing or not same_direction_positions:
        return 1.0, diagnostics

    tickers = [p.get("ticker", "").upper() for p in same_direction_positions if p.get("ticker")]
    if not tickers:
        return 1.0, diagnostics

    mean_r, max_r, n, per_ticker = mean_pairwise_correlation(candidate, tickers, days=days)
    diagnostics["per_ticker"] = {k: round(v, 3) for k, v in per_ticker.items()}
    diagnostics["mean_corr"] = round(mean_r, 3) if mean_r is not None else None
    diagnostics["max_corr"]  = round(max_r, 3)  if max_r  is not None else None
    diagnostics["n_pairs"]   = n

    if mean_r is None:
        return 1.0, diagnostics

    # Portfolio cap: Σ(other_size · |ρ|) — even a strong negative correlation
    # represents factor exposure (just inverted), so we use |ρ|.
    weighted = 0.0
    for p in same_direction_positions:
        tk = p.get("ticker", "").upper()
        if tk in per_ticker:
            weighted += float(p.get("position_size_multiplier", 1.0)) * abs(per_ticker[tk])
    diagnostics["weighted_exposure"] = round(weighted, 3)
    diagnostics["portfolio_cap_hit"] = weighted > float(settings.correlation_portfolio_cap)

    multiplier = _haircut_multiplier(mean_r)
    return multiplier, diagnostics


def trim_old_correlation_cache() -> None:
    """No-op stub for symmetry with the rest of the cache layer.

    Correlations aren't persisted today — they're computed on demand from
    the OHLCV cache, which has its own retention. The helper exists so the
    pipeline can grow a persistent correlation matrix later without
    touching call sites.
    """
    return None
