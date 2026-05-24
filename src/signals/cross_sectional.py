"""Cross-sectional ranking signal.

The 11 base methods all score in absolute terms: a +0.5 sentiment is "fairly
bullish" regardless of what every other ticker in the universe is scoring.
That convention is informative when the universe is mixed, but breaks down
in regime-dominated tape:

  * Bull market: news sentiment, technical, momentum, money_flow all skew
    positive across most stocks. A +0.5 stock looks bullish but is actually
    average — the *real* bullish names are at +0.9.
  * Bear market: most absolute scores skew negative. A −0.3 ticker is
    "moderately bearish" in isolation but might be the best-positioned name
    in a sea of −0.8 readings.

Cross-sectional ranking corrects for the regime by measuring how each
ticker's score deviates from the universe mean on each method, then
aggregating those z-scores into a single "stand-out" score per ticker. It
adds a relative-value dimension to the otherwise-absolute aggregator.

Algorithm
─────────
For each method m ∈ {news, tech, insider, put_call, max_pain, oi_skew, vwap,
pattern, momentum, money_flow, pead}:

  1. Collect all per-ticker scores across the universe.
  2. Drop near-zero readings (a method that did not fire on a ticker
     shouldn't pull that ticker's z-score toward the mean — the absence of
     a view is not the same as a "0" view).
  3. Compute μ_m and σ_m over the remaining non-zero scores.
  4. For each ticker t with a non-zero score: z_m[t] = (score - μ_m) / σ_m,
     clipped to ±zcap (default 2.5) to stop a single outlier from dominating
     the average.

Per ticker:
  cs_score = mean(z_m[t] for all m with non-zero score for t) / zcap
  clipped to [-1, +1].

A ticker that's average on every active method gets cs_score ≈ 0. A ticker
that's two standard deviations above the universe on most active methods
gets cs_score ≈ +0.8. The aggregator then *adds* ``cross_sectional_weight ×
cs_score`` to ``combined_score`` so the cross-sectional view composes with
the absolute aggregation rather than replacing it.

Tickers with zero non-zero method scores get cs_score = 0 (no relative
view — no penalty). A universe of one ticker also returns 0 (no relative
comparison possible).
"""

from __future__ import annotations

import statistics
from typing import Dict, List, Tuple

from loguru import logger


# The methods we'll z-score across the universe. Mirrors aggregator._BASE_WEIGHTS
# but excludes ``cross_sectional`` itself (no recursion). vwap_distance_pct,
# pead_surprise_pct, mfi_value, cmf_value, etc. are *raw indicator readings*,
# not scores — we stay with the normalized [-1, +1] outputs.
_METHODS_FOR_RANKING: Tuple[str, ...] = (
    "sentiment_score",   # news
    "technical_score",   # tech
    "insider_score",     # insider
    "put_call_score",    # put_call
    "max_pain_score",    # max_pain
    "oi_skew_score",     # oi_skew
    "vwap_score",        # vwap
    "pattern_score",     # pattern
    "momentum_score",    # momentum
    "money_flow_score",  # money_flow
    "pead_score",        # pead
)

_ZERO_THRESHOLD = 1e-3   # treat |score| < this as "method did not fire"


def _clip(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


def compute_cross_sectional_scores(
    signals: List,
    zcap: float = 2.5,
) -> Dict[str, float]:
    """Return ``{ticker: cs_score}`` mapping. Empty dict on insufficient input.

    Each TickerSignal in ``signals`` is expected to expose the attributes in
    ``_METHODS_FOR_RANKING``. Missing attributes are treated as 0 (no view).

    Cross-sectional z-scores need at least 3 tickers with a non-zero reading
    on a method to compute a usable σ. Methods with fewer than that are
    silently skipped — they contribute zero to every ticker's average.
    """
    if not signals or len(signals) < 3:
        return {}

    tickers = [s.ticker for s in signals]
    # method → {ticker: signed_score}
    by_method: Dict[str, Dict[str, float]] = {}
    for attr in _METHODS_FOR_RANKING:
        by_method[attr] = {}
        for s in signals:
            v = float(getattr(s, attr, 0.0) or 0.0)
            by_method[attr][s.ticker] = v

    # Compute per-method universe stats over non-zero scores only
    method_stats: Dict[str, Tuple[float, float]] = {}
    for attr, ticker_scores in by_method.items():
        non_zero = [v for v in ticker_scores.values() if abs(v) >= _ZERO_THRESHOLD]
        if len(non_zero) < 3:
            continue
        mu = statistics.mean(non_zero)
        try:
            sigma = statistics.stdev(non_zero)
        except statistics.StatisticsError:
            continue
        if sigma <= 0:
            continue
        method_stats[attr] = (mu, sigma)

    if not method_stats:
        return {t: 0.0 for t in tickers}

    # Per-ticker aggregation
    cs_scores: Dict[str, float] = {}
    for t in tickers:
        zs: List[float] = []
        for attr, (mu, sigma) in method_stats.items():
            v = by_method[attr][t]
            if abs(v) < _ZERO_THRESHOLD:
                continue   # no view on this method
            z = (v - mu) / sigma
            zs.append(_clip(z, -zcap, zcap))
        if not zs:
            cs_scores[t] = 0.0
            continue
        avg_z = sum(zs) / len(zs)
        cs_scores[t] = round(_clip(avg_z / zcap, -1.0, 1.0), 4)

    # Log a brief diagnostic so you can see which tickers stand out
    extremes = sorted(cs_scores.items(), key=lambda kv: -abs(kv[1]))[:6]
    if extremes and abs(extremes[0][1]) >= 0.3:
        formatted = "  ".join(f"{tk}={v:+.2f}" for tk, v in extremes if abs(v) >= 0.2)
        if formatted:
            logger.info(f"[cross_sectional] Standouts (|z| ≥ 0.20): {formatted}")

    return cs_scores
