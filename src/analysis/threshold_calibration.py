"""Engine-relative actionable threshold — quantile-anchored confidence gates.

The actionable gate compares LLM confidence to absolute thresholds (0.85
baseline, the macro-regime ladder 0.79–0.95, session bumps). But confidence
DISTRIBUTIONS are engine-specific: DeepSeek hands out 1.00s where Claude tops
out lower, so swapping ``ANALYST_MODEL`` (or the A/B flip landing on the other
engine) silently changes what every threshold means — the same scale-invariance
bug that absolute breadth thresholds had for method-set growth.

Fix: translate each static threshold through the two distributions —

    effective = Q_engine( F_global(static) )

where ``F_global`` is the ECDF of recent BUY/SELL recommendation confidences
across ALL engines and ``Q_engine`` the quantile function of THIS run's engine.
The static threshold's meaning becomes "how selective the gate is" (the share
of calls it rejects) instead of a raw number, so the regime ladder's semantics
survive any engine's confidence inflation/deflation. The translated value is
Bayesian-shrunk toward the static one while the engine's history is thin
(``shrink``, prior_n = ``threshold_engine_prior_n``) and safety-clamped to
±``threshold_max_shift`` around the static — the gate can drift with evidence,
never jump.

Both distributions come from the ``recommendations`` table (top-10 per run —
a selection bias shared by numerator and denominator, so the translation is
unaffected to first order). Deterministic given the DB; cached briefly.
"""

from __future__ import annotations

import time
from bisect import bisect_left, bisect_right
from datetime import date, timedelta
from typing import List, Optional, Tuple

from loguru import logger

from config.settings import settings
from src.performance.calibration import report_calibration, shrink

_CACHE_TTL_S = 600.0
_cache: dict = {}     # (engine, days) -> {"ts", "global": [...], "engine": [...]}


def _confidence_samples(engine_model: Optional[str], days: int) -> Tuple[List[float], List[float]]:
    """(global, engine) sorted BUY/SELL recommendation-confidence samples over
    the window. Empty lists on any read problem (fail-soft to static)."""
    key = (engine_model or "", int(days))
    now = time.time()
    hit = _cache.get(key)
    if hit and (now - hit["ts"]) < _CACHE_TTL_S:
        return hit["global"], hit["engine"]
    g: List[float] = []
    e: List[float] = []
    try:
        from src.db import repo
        from src.performance.tracker import _is_rule_based_fill
        cutoff = (date.today() - timedelta(days=days)).isoformat()
        df = repo.fetch_df(
            "SELECT llm_provider, confidence, rationale FROM recommendations "
            "WHERE action IN ('BUY','SELL') AND confidence IS NOT NULL "
            "AND generated_at >= ?", [cutoff])
        if df is not None and not df.empty and "rationale" in df.columns:
            # Drop rule-based BACK-FILLS: their confidence is the AGGREGATOR's,
            # copied verbatim, not the engine's own. They were ~22% of rows and
            # carried the run's model name before 2026-07-22, so leaving them in
            # skews the engine's confidence ECDF — and this calibration converts
            # the regime threshold into that scale, so it moves the live gate.
            df = df[~df.apply(lambda r: _is_rule_based_fill(r["llm_provider"],
                                                            r["rationale"]), axis=1)]
        if df is not None and not df.empty:
            g = sorted(float(c) for c in df["confidence"] if c == c)
            if engine_model:
                sub = df[df["llm_provider"].astype(str) == str(engine_model)]
                e = sorted(float(c) for c in sub["confidence"] if c == c)
    except Exception as exc:
        logger.debug(f"[threshold] confidence-sample read failed ({exc}) — static gate")
    _cache[key] = {"ts": now, "global": g, "engine": e}
    return g, e


def _ecdf(sorted_vals: List[float], x: float) -> float:
    """P(value < x) over the sample (left-continuous ECDF)."""
    return bisect_left(sorted_vals, x) / len(sorted_vals)


def _quantile(sorted_vals: List[float], q: float) -> float:
    """The q-quantile of the sample (nearest-rank, clamped)."""
    idx = min(len(sorted_vals) - 1, max(0, int(round(q * (len(sorted_vals) - 1)))))
    return sorted_vals[idx]


def engine_relative_threshold(static_threshold: float,
                              engine_model: Optional[str]) -> Tuple[float, dict]:
    """Translate ``static_threshold`` into this engine's confidence scale.

    Returns ``(effective_threshold, meta)``. Falls back to the static value
    when disabled, the engine is unknown, or history is thin (the shrinkage
    handles the ramp between). Never moves more than
    ``threshold_max_shift`` from the static anchor."""
    static = float(static_threshold)
    meta = {"static": round(static, 4), "engine": engine_model or "",
            "n_engine": 0, "n_global": 0, "applied": False}
    if not settings.threshold_engine_relative_enabled or not engine_model:
        return static, meta
    g, e = _confidence_samples(engine_model,
                               int(settings.threshold_calibration_days))
    meta["n_global"], meta["n_engine"] = len(g), len(e)
    if len(g) < int(settings.threshold_min_global_recs) or not e:
        return static, meta

    selectivity = _ecdf(g, static)                 # share of ALL calls the gate rejects
    translated = _quantile(e, selectivity)         # same selectivity on THIS engine
    post = shrink(static, int(settings.threshold_engine_prior_n),
                  translated, len(e))
    max_shift = float(settings.threshold_max_shift)
    eff = min(static + max_shift, max(static - max_shift, post))
    eff = min(0.95, max(0.50, eff))
    meta.update(applied=True, selectivity=round(selectivity, 4),
                translated=round(translated, 4), effective=round(eff, 4))
    report_calibration(
        "actionable_threshold", value=eff, prior=static, n_evidence=len(e),
        unit="confidence",
        note=f"engine-relative ({engine_model}); gate selectivity "
             f"{selectivity:.0%} of recent BUY/SELL calls")
    return round(eff, 4), meta


def reset_cache() -> None:
    """Tests."""
    _cache.clear()
