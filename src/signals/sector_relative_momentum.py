"""Relative momentum — beta-stripped alpha factors.

Same structural pattern as ``price_momentum`` but on residual returns:

    r_rel[t] = r_ticker[t] − r_benchmark[t]

Two public scorers share the same core machinery:

  * ``compute_sector_relative_momentum_score(ticker)`` — benchmark is the
    ticker's sector ETF (e.g. XLK for NVDA) resolved by
    ``sector_benchmark.get_sector_benchmark``. Feeds the weighted aggregator
    as a real method.
  * ``compute_market_relative_momentum_score(ticker)`` — benchmark is SPY
    (the broad market). PROMOTED into the weighted combine but kept LIGHT
    (``market_momentum`` = 0.08): market_relative = sector_relative
    + (sector − market), so a heavy weight would double-count the beta already
    in ``sector_momentum`` — the small weight bounds that overlap. Also feeds
    the email/prompt to show whether a stock lags because its *sector* is weak
    vs the market, or on its own merits.

Why these are cleaner factors than absolute momentum
----------------------------------------------------
Absolute momentum mixes idiosyncratic alpha with sector beta — if XLK is
up 20% over the last month, every constituent looks like it has momentum.
Subtracting the benchmark strips out that shared beta and leaves the
ticker-specific excess. In factor research (Moskowitz & Grinblatt 1999;
Asness 1997) residual/sector-neutral momentum dominates the raw version
on forward returns and is a meaningfully less-crowded factor.

For ETFs/commodities (no sector applies) the sector function returns 0;
the market function still produces a meaningful number against SPY (e.g.
GLD leading SPY in a risk-off tape).
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
import pandas as pd
from loguru import logger

from src.data.cache import load_ohlcv
from src.data.market_data import get_history, get_weekly_history
from src.signals.sector_benchmark import get_sector_benchmark


_MIN_ROWS      = 50    # minimum bars to compute meaningful momentum
_DIST_WINDOW   = 252   # trailing bars for normalisation distribution
_MOM_SHORT     = 21    # 1-month lookback (trading days)
_MOM_MED       = 42    # 2-month lookback (trading days)
_SHORT_WEIGHT  = 0.6
_MED_WEIGHT    = 0.4
_TANH_SCALE    = 1.5


def _get_close(ticker: str, interval: str = "1d") -> Optional[pd.Series]:
    """Return the Close series for *ticker* at ``interval`` from cache, with a
    one-shot fetch fallback when missing. Returns ``None`` on any failure.

    ``1d`` = daily cache (legacy); ``30m`` = intraday fetch; ``1w`` = weekly
    resampled from the daily cache."""
    try:
        if interval == "1d":
            df = load_ohlcv(ticker)
            if df is None or df.empty or "Close" not in df.columns:
                df = get_history(ticker, period="18mo")
        elif interval == "30m":
            df = get_history(ticker, interval="30m")
        elif interval == "1w":
            df = get_weekly_history(ticker)
        else:
            return None
    except Exception as e:
        logger.debug(f"[sector_mom] history fetch failed for {ticker}[{interval}]: {e}")
        return None
    if df is None or df.empty or "Close" not in df.columns:
        return None
    s = df["Close"].astype(float)
    # Drop non-positive closes — they can't produce valid pct_change.
    s = s[s > 0]
    return s if len(s) >= _MIN_ROWS else None


def _align(ticker_close: pd.Series, bench_close: pd.Series) -> Tuple[pd.Series, pd.Series]:
    """Align on common dates and return the intersection.

    Dedupe is critical: mixed Polygon/yfinance cache rows can collide on the
    same session-date after timezone normalization. Without ``~duplicated``
    the ``.loc[common]`` step expands each duplicate hit, breaking downstream
    pct_change / std computations.
    """
    # Ensure tz-naive date-level index alignment to avoid timezone mismatches
    # between Polygon (naive UTC) and yfinance (tz-aware) OHLCV rows.
    def _normalize(idx: pd.DatetimeIndex) -> pd.DatetimeIndex:
        try:
            if idx.tz is not None:
                idx = idx.tz_convert("America/New_York").tz_localize(None)
        except (TypeError, AttributeError):
            pass
        return pd.DatetimeIndex(idx).normalize()

    a = ticker_close.copy()
    a.index = _normalize(a.index)
    a = a[~a.index.duplicated(keep="last")]
    b = bench_close.copy()
    b.index = _normalize(b.index)
    b = b[~b.index.duplicated(keep="last")]
    common = a.index.intersection(b.index)
    return a.loc[common], b.loc[common]


def _compute_relative_momentum(
    ticker: str,
    benchmark: str,
    tag: str = "rel_mom",
    interval: str = "1d",
) -> Tuple[float, float, float, str]:
    """Generic ticker-minus-benchmark residual momentum scorer.

    Shared implementation backing both the sector and market variants. The
    ``tag`` argument is only used in debug logging so the two callers
    show up distinctly in logs. ``interval`` pulls BOTH legs at the same
    timeframe so the residual is apples-to-apples.

    Returns ``(score, rel_1m_pct, rel_3m_pct, benchmark)``. Returns zeros
    (and the benchmark string, so callers can still report it) whenever
    either leg lacks usable history or self-comparison would occur.
    """
    if not benchmark or benchmark.upper() == ticker.upper():
        return 0.0, 0.0, 0.0, ""

    tk_close = _get_close(ticker, interval)
    bm_close = _get_close(benchmark, interval)
    if tk_close is None or bm_close is None:
        logger.debug(
            f"[{tag}] {ticker} vs {benchmark}: missing OHLCV "
            f"(ticker={tk_close is not None}, bench={bm_close is not None})"
        )
        return 0.0, 0.0, 0.0, benchmark

    tk_close, bm_close = _align(tk_close, bm_close)
    # Upfront gate: enough overlap to compute both 42-bar returns. The
    # normalisation step below also enforces a 30-point distribution check.
    if len(tk_close) < _MOM_MED + 2:
        return 0.0, 0.0, 0.0, benchmark

    # Raw multi-period returns (ticker_return - benchmark_return).
    rel_1m = float(
        (tk_close.iloc[-1] - tk_close.iloc[-_MOM_SHORT]) / tk_close.iloc[-_MOM_SHORT]
        - (bm_close.iloc[-1] - bm_close.iloc[-_MOM_SHORT]) / bm_close.iloc[-_MOM_SHORT]
    )
    rel_3m = float(
        (tk_close.iloc[-1] - tk_close.iloc[-_MOM_MED]) / tk_close.iloc[-_MOM_MED]
        - (bm_close.iloc[-1] - bm_close.iloc[-_MOM_MED]) / bm_close.iloc[-_MOM_MED]
    )

    # Normalise against the ticker's own historical 21-bar excess-return
    # distribution, so an unusually volatile name doesn't blow out the score.
    excess_21 = tk_close.pct_change(_MOM_SHORT) - bm_close.pct_change(_MOM_SHORT)
    dist_slice = excess_21.dropna().tail(_DIST_WINDOW)
    if len(dist_slice) < 30:
        return 0.0, 0.0, 0.0, benchmark

    std_1m = float(dist_slice.std())
    if std_1m < 1e-8:
        return 0.0, 0.0, 0.0, benchmark

    z_1m = rel_1m / std_1m
    z_3m = rel_3m / (std_1m * (2 ** 0.5))  # 2m std ≈ sqrt(2) × 1m std
    z_composite = _SHORT_WEIGHT * z_1m + _MED_WEIGHT * z_3m

    raw_score = float(np.tanh(z_composite / _TANH_SCALE))
    score     = round(max(-1.0, min(1.0, raw_score)), 3)

    rel_1m_pct = round(rel_1m * 100, 2)
    rel_3m_pct = round(rel_3m * 100, 2)

    logger.debug(
        f"[{tag}] {ticker} vs {benchmark}: "
        f"1m={rel_1m_pct:+.2f}pp  3m={rel_3m_pct:+.2f}pp  "
        f"z_comp={z_composite:+.2f}  score={score:+.3f}"
    )
    return score, rel_1m_pct, rel_3m_pct, benchmark


def compute_sector_relative_momentum_score(
    ticker: str,
    asset_type: Optional[str] = None,
    interval: str = "1d",
) -> Tuple[float, float, float, str]:
    """Return ``(score, rel_1m_pct, rel_3m_pct, benchmark)`` — sector benchmark.

    * ``score`` ∈ [-1.0, +1.0]. Positive = ticker outperforming its sector
      ETF over recent periods relative to its own historical excess-return
      distribution. Zero = no data / ETFs / commodities.
    * ``rel_1m_pct``/``rel_3m_pct`` = raw ticker-minus-benchmark return %
      over 21 / 42 trading days. Useful diagnostic for the email.
    * ``benchmark`` = ETF symbol actually used (``""`` if none).

    ``interval`` pulls both the ticker and its sector ETF at the same timeframe
    ("1d" daily/legacy, "30m" intraday, "1w" weekly).
    """
    benchmark = get_sector_benchmark(ticker, asset_type=asset_type)
    if not benchmark:
        return 0.0, 0.0, 0.0, ""
    return _compute_relative_momentum(ticker, benchmark, tag="sector_mom", interval=interval)


def compute_market_relative_momentum_score(
    ticker: str,
) -> Tuple[float, float, float, str]:
    """Return ``(score, rel_1m_pct, rel_3m_pct, "SPY")`` — broad-market benchmark.

    Always benchmarked against SPY (broad market). Answers "is this name lagging
    the market?" independently of whether it's lagging its sector. Weighted into
    the aggregator combine but kept LIGHT (``market_momentum`` = 0.08) because

        market_rel = sector_rel + (sector − market)

    so a heavy weight would double-count the beta already in ``sector_momentum``.

    Applies to commodities too — e.g. GLD leading SPY in a risk-off tape
    is a useful read even though it lacks a sector ETF.
    """
    if ticker.upper() == "SPY":
        return 0.0, 0.0, 0.0, ""
    return _compute_relative_momentum(ticker, "SPY", tag="market_mom")
