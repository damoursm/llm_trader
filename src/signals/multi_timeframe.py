"""Multi-timeframe technical orchestration (30-min / daily / weekly).

Every OHLCV-based method already produces a *daily* score inside
``aggregator.build_signals``. This module computes the same methods on the two
*non-daily* candles — a faster 30-minute bar and a slower weekly bar — and
exposes a blend so the live ``combined_score`` reflects all three timeframes.

Daily is intentionally NOT recomputed here: the aggregator already has it (and
its scorers fetch their full daily history), so re-deriving it from a thinner
frame would degrade the daily signal. This module returns only the ``*_30m`` and
``*_1w`` components; ``blend_timeframes`` then mixes them with the daily value
the aggregator passes in.

Design notes
------------
* **30-min** bars come from yfinance (``get_history(interval="30m")``) and are
  rate-limited / capped (``intraday_30m_max_tickers``); a frame that fails to
  fetch simply yields no 30m keys (the method abstains at that timeframe).
* **Weekly** bars are resampled from the daily cache (free, no fetch).
* A scorer that returns exactly ``0.0`` is treated as **"no view"** at that
  timeframe and EXCLUDED from the blend's renormalisation — same convention the
  IC panel and the aggregator's coherence use (non-zero = a view). So when the
  non-daily candles say nothing, the blend collapses to the daily value and the
  strategy is unchanged.
"""

from __future__ import annotations

from typing import Dict, Optional

from loguru import logger

from config import settings
from src.data.market_data import get_history, get_weekly_history

# The 8 OHLCV-based methods recomputed per timeframe. Order is the canonical one
# used for the signals-panel columns + the dashboard's technical IC categories.
TECHNICAL_METHODS = (
    "tech", "vwap", "momentum", "money_flow",
    "trend_strength", "iv_rank", "pattern", "sector_momentum",
)

# Non-daily timeframes this module produces (daily comes from build_signals).
NON_DAILY_TIMEFRAMES = ("30m", "1w")

_EPS = 1e-9


def _score_one(method: str, ticker: str, df, interval: str) -> Optional[float]:
    """Run a single method's scorer on a pre-fetched frame. Each scorer is
    isolated so one failure never poisons the rest of the frame."""
    try:
        if method == "tech":
            from src.analysis.technical import compute_technical_score
            return float(compute_technical_score(ticker, df=df).score)
        if method == "vwap":
            from src.signals.vwap import compute_vwap_score
            return float(compute_vwap_score(ticker, df=df)[0])
        if method == "momentum":
            from src.signals.price_momentum import compute_price_momentum_score
            return float(compute_price_momentum_score(ticker, df=df)[0])
        if method == "money_flow":
            from src.signals.money_flow import compute_money_flow_score
            return float(compute_money_flow_score(ticker, df=df)[0])
        if method == "trend_strength":
            from src.signals.trend_strength import compute_trend_strength_score
            return float(compute_trend_strength_score(ticker, df=df)[0])
        if method == "iv_rank":
            from src.signals.iv_rank import compute_iv_rank_score
            return float(compute_iv_rank_score(ticker, df=df)[0])
        if method == "pattern":
            from src.signals.pattern_recognition import compute_pattern_score
            return float(compute_pattern_score(ticker, df=df, interval=interval)[0])
        if method == "sector_momentum":
            # sector_momentum fetches BOTH legs at the interval (no df param).
            from src.signals.sector_relative_momentum import compute_sector_relative_momentum_score
            return float(compute_sector_relative_momentum_score(ticker, interval=interval)[0])
    except Exception as e:  # pragma: no cover - defensive
        logger.debug(f"[mtf] {ticker} {method}[{interval}] scorer failed: {e}")
    return None


def _score_frame(ticker: str, df, interval: str) -> Dict[str, float]:
    """Score all 8 technical methods on one timeframe's frame.

    ``sector_momentum`` ignores ``df`` and re-fetches both legs at ``interval``
    (it needs the benchmark ETF at the same candle); the rest score on ``df``.
    Only finite scores are returned (a missing/failed scorer is simply absent)."""
    out: Dict[str, float] = {}
    for method in TECHNICAL_METHODS:
        s = _score_one(method, ticker, df, interval)
        if s is not None:
            out[f"{method}_{interval}"] = round(s, 3)
    return out


def compute_timeframe_scores(ticker: str, allow_30m: bool = True) -> Dict[str, float]:
    """Return the NON-DAILY per-(method, timeframe) scores for ``ticker``.

    Keys are ``f"{method}_30m"`` / ``f"{method}_1w"``. The daily components are
    NOT included (the aggregator already holds them). ``allow_30m`` lets the
    caller honour ``intraday_30m_max_tickers`` by skipping the 30m fetch for
    tickers beyond the cap. Returns ``{}`` when the master flag is off."""
    if not settings.enable_multi_timeframe_signals:
        return {}

    out: Dict[str, float] = {}

    if settings.enable_intraday_30m and allow_30m:
        try:
            df30 = get_history(ticker, interval="30m")
        except Exception as e:  # pragma: no cover - defensive
            logger.debug(f"[mtf] {ticker} 30m fetch failed: {e}")
            df30 = None
        if df30 is not None and not df30.empty:
            out.update(_score_frame(ticker, df30, "30m"))

    if settings.enable_weekly_signals:
        try:
            dfw = get_weekly_history(ticker)
        except Exception as e:  # pragma: no cover - defensive
            logger.debug(f"[mtf] {ticker} weekly resample failed: {e}")
            dfw = None
        if dfw is not None and not dfw.empty:
            out.update(_score_frame(ticker, dfw, "1w"))

    return out


def blend_timeframes(scores_by_tf: Dict[str, Optional[float]]) -> float:
    """Renormalised blend of a method's per-timeframe scores.

    ``scores_by_tf`` maps ``"30m" | "1d" | "1w"`` → score (or ``None``). The
    configured ``tf_blend_*`` weights are applied over only the timeframes with
    a *view* (non-None, |score| > eps); a timeframe that abstains drops out and
    the remaining weights are renormalised. With no views the blend is 0.0; with
    only the daily view it returns the daily value — i.e. legacy behaviour."""
    weights = {
        "30m": max(0.0, float(settings.tf_blend_30m)),
        "1d":  max(0.0, float(settings.tf_blend_1d)),
        "1w":  max(0.0, float(settings.tf_blend_1w)),
    }
    num = 0.0
    den = 0.0
    for tf, score in scores_by_tf.items():
        if score is None or abs(score) < _EPS:
            continue
        w = weights.get(tf, 0.0)
        if w <= 0.0:
            continue
        num += w * float(score)
        den += w
    return round(num / den, 4) if den > 0 else 0.0
