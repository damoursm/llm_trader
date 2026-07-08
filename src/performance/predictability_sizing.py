"""Predictability sizing (Tier 1) — allocate more capital to names whose
direction is more forecastable at a swing horizon.

The Tier-0 predictability panel (``src/analysis/predictability.py``) measured
that clean-trend names — high Kaufman efficiency ratio / high ADX — carry a
markedly higher directional hit rate for ``combined_score`` at the 5-day swing
horizon (~60% vs a ~50% coin flip for chop). This module turns that measurement
into a **self-calibrating sizing tilt**, built on the exact evidence-throttled
idiom as breadth sizing (``tracker._breadth_calibration``):

    score    = w_er·ER + w_adx·min(ADX, cap)/cap          (per stock, as of entry)
    mult     = 1 + span_eff × clamp((score − center)/half_width, −1, +1)
    span_eff = predictability_size_span × clamp(d_post / edge_ref, 0, 1)
    d_post   = (prior_n·d_prior + n_days·d_obs) / (prior_n + n_days)

``d_obs`` is the measured directional-hit gap between high- and low-score names
over the UNBIASED signals panel (every scored ticker, not just gated trades);
``center``/``half_width`` are the median / IQR of the recent score distribution
so a name is ranked against the current cross-section. Crucially the evidence
count is **signal-days** (not rows) — same-day names are cross-sectionally
correlated, so counting days keeps the shrinkage honest and makes the tilt
strengthen over WEEKS AND MONTHS as independent days accrue, exactly as intended.
It starts ~inert (heavily shrunk toward the small documented prior on the ~2
weeks of one regime we have), grows as the edge confirms, and decays to neutral
if it fades. Never a gate (every name still trades, so outcomes keep
accumulating), never inverts a position's sign.

Deterministic given the DB + OHLCV cache; the heavy calibration is cached
(``predictability_cal_ttl_seconds``) and fully fail-soft — any error yields an
inert calibration, so sizing is never broken by this layer.
"""

from __future__ import annotations

import time
from bisect import bisect_left
from statistics import median
from typing import Optional

import pandas as pd
from loguru import logger

from config.settings import settings

_EPS = 1e-12
_cal_cache: dict = {"ts": 0.0, "cal": None}

# An inert calibration: span_eff 0 ⇒ predictability_multiplier is exactly 1.0.
_INERT = {"center": 0.5, "half_width": 1.0, "span_eff": 0.0, "edge": 0.0,
          "d_obs": None, "n_days": 0, "n_rows": 0, "horizon": 0}


def _clip01(x: float) -> float:
    return 0.0 if x < 0.0 else (1.0 if x > 1.0 else x)


def blend_score(eff_ratio: Optional[float], adx: Optional[float]) -> Optional[float]:
    """The per-stock predictability score in [0, 1] — a weighted blend of the
    Kaufman efficiency ratio (already [0, 1]) and normalized ADX. None when
    neither component is available. Weights renormalise over whatever is present,
    so a close-only series (no ADX) still yields an ER-only score."""
    w_er = float(settings.predictability_er_weight)
    w_adx = float(settings.predictability_adx_weight)
    cap = max(1e-9, float(settings.predictability_adx_cap))
    total = 0.0
    wsum = 0.0
    if eff_ratio is not None and eff_ratio == eff_ratio:      # == drops NaN
        total += w_er * _clip01(float(eff_ratio))
        wsum += w_er
    if adx is not None and adx == adx:
        total += w_adx * _clip01(float(adx) / cap)
        wsum += w_adx
    return (total / wsum) if wsum > 0 else None


def predictability_score(ticker: str, asof=None) -> Optional[float]:
    """Live predictability score for a ticker — the latest (or as-of ``asof``
    date) blend of trend efficiency + ADX from the OHLCV cache. Fail-soft
    (None on any error / no history), so a sizing caller degrades to neutral."""
    try:
        from src.analysis.predictability import _feature_frame
        fr = _feature_frame(ticker, int(settings.predictability_er_window),
                            int(settings.predictability_er_window),
                            int(settings.predictability_adx_period))
        if fr is None:
            return None
        idx, fs = fr
        if not len(idx):
            return None
        if asof is None:
            row = fs.iloc[-1]
        else:
            i = bisect_left(idx, asof)
            if i >= len(idx):
                return None
            row = fs.iloc[i]
        return blend_score(row.get("eff_ratio"), row.get("adx"))
    except Exception as e:
        logger.debug(f"[predictability] score failed for {ticker}: {e}")
        return None


def _score_panel(feature_panel: pd.DataFrame) -> pd.Series:
    """The predictability score for each row of a built feature panel (uses the
    eff_ratio / adx columns already attached by ``build_feature_panel``)."""
    er = pd.to_numeric(feature_panel.get("eff_ratio"), errors="coerce")
    adx = pd.to_numeric(feature_panel.get("adx"), errors="coerce")
    return pd.Series([blend_score(e, a) for e, a in zip(er, adx)], index=feature_panel.index)


def calibrate_predictability(feature_panel: Optional[pd.DataFrame] = None) -> dict:
    """Self-calibrating ``center`` / ``half_width`` / ``span_eff`` for the
    predictability tilt, measured over the signals panel. Cached
    ``predictability_cal_ttl_seconds`` for callers that don't pass a panel;
    fully fail-soft (returns the inert calibration on any error or thin data).

    ``span_eff`` grows with the number of SIGNAL-DAYS of confirming evidence — the
    tilt is nearly off today and strengthens over weeks/months as independent
    cross-sections accrue (or fades to 0 if the edge doesn't hold)."""
    if not settings.enable_predictability_sizing:
        return dict(_INERT)
    now = time.time()
    if feature_panel is None and _cal_cache["cal"] is not None \
            and (now - _cal_cache["ts"]) < float(settings.predictability_cal_ttl_seconds):
        return _cal_cache["cal"]

    cal = dict(_INERT)
    try:
        horizon = int(settings.predictability_horizon)
        fp = feature_panel
        if fp is None:
            from src.analysis.signal_panel import build_panel
            from src.analysis.predictability import build_feature_panel
            panel = build_panel(horizons=(horizon,), days=int(settings.predictability_cal_days))
            fp = build_feature_panel(panel) if panel is not None and not panel.empty else None
        if fp is None or fp.empty:
            _store(now, cal, feature_panel)
            return cal

        scores = _score_panel(fp)
        valid_scores = scores.dropna()
        if len(valid_scores) < int(settings.predictability_cal_min_rows):
            _store(now, cal, feature_panel)
            return cal

        # Ramp geometry — rank a new name against the CURRENT score cross-section.
        center = float(median(valid_scores))
        try:
            q1, _, q3 = pd.Series(valid_scores).quantile([0.25, 0.5, 0.75]).tolist()
            half = max(float(settings.predictability_halfwidth_floor), float(q3 - q1))
        except Exception:
            half = max(float(settings.predictability_halfwidth_floor), 0.1)

        # Measured edge — the directional-hit gap between high- and low-score
        # names for combined_score at the swing horizon, shrunk toward the prior
        # by SIGNAL-DAYS (correlated same-day names count once).
        col = f"fwd_ret_{horizon}d"
        cs = pd.to_numeric(fp.get("combined_score"), errors="coerce")
        fwd = pd.to_numeric(fp.get(col), errors="coerce")
        used = fp.assign(_score=scores, _cs=cs, _fwd=fwd)
        used = used[scores.notna() & cs.notna() & (cs.abs() > _EPS)
                    & fwd.notna() & (fwd != 0)]
        n_rows = int(len(used))
        n_days = int(used["signal_date"].nunique()) if "signal_date" in used.columns else 0
        d_obs = None
        if n_rows >= int(settings.predictability_cal_min_rows):
            correct = (used["_cs"] > 0) == (used["_fwd"] > 0)
            hi = correct[used["_score"] >= center]
            lo = correct[used["_score"] < center]
            if len(hi) and len(lo):
                p_hi = (int(hi.sum()) + 1) / (len(hi) + 2)        # Laplace-smoothed
                p_lo = (int(lo.sum()) + 1) / (len(lo) + 2)
                d_obs = float(p_hi - p_lo)

        d_prior = float(settings.predictability_edge_prior)
        prior_n = max(0, int(settings.predictability_edge_prior_n))
        if d_obs is not None and n_days > 0:
            d_post = (prior_n * d_prior + n_days * d_obs) / (prior_n + n_days)
        else:
            d_post = d_prior
        edge_ref = max(1e-9, float(settings.predictability_edge_ref))
        edge = _clip01(d_post / edge_ref)
        span_eff = round(float(settings.predictability_size_span) * edge, 4)

        cal = {"center": round(center, 4), "half_width": round(half, 4),
               "span_eff": span_eff, "edge": round(edge, 4),
               "d_obs": (round(d_obs, 4) if d_obs is not None else None),
               "n_days": n_days, "n_rows": n_rows, "horizon": horizon}
        _report(cal, d_prior, edge_ref)
    except Exception as e:
        logger.debug(f"[predictability] calibration failed: {e}")
        cal = dict(_INERT)
    _store(now, cal, feature_panel)
    return cal


def _store(now: float, cal: dict, feature_panel: Optional[pd.DataFrame]) -> None:
    if feature_panel is None:                 # only cache the live (DB-built) calibration
        _cal_cache.update(ts=now, cal=cal)


def _report(cal: dict, d_prior: float, edge_ref: float) -> None:
    try:
        from src.performance.calibration import report_calibration
        report_calibration(
            "predictability_span_eff", value=cal["span_eff"],
            prior=round(float(settings.predictability_size_span)
                        * _clip01(d_prior / edge_ref), 4),
            n_evidence=cal["n_days"], unit="± size tilt",
            note=f"evidence-throttled predictability sizing "
                 f"(hit-gap {cal['d_obs']} over {cal['n_days']} signal-day(s), "
                 f"{cal['horizon']}d horizon)")
    except Exception:
        pass


def predictability_multiplier(score: Optional[float], cal: Optional[dict] = None) -> float:
    """CONTINUOUS position-size tilt from the predictability score:

        mult = 1 + span_eff × clamp((score − center) / half_width, −1, +1)

    Bounded to [1 − span_eff, 1 + span_eff], neutral 1.0 when disabled, when the
    score is unknown (no OHLCV history → absence of evidence, not a bearish
    signal), or while ``span_eff`` is 0 (thin/regime-limited evidence)."""
    if not settings.enable_predictability_sizing or score is None:
        return 1.0
    c = cal if cal is not None else calibrate_predictability()
    span = float(c.get("span_eff", 0.0))
    if span <= 0:
        return 1.0
    ramp = (float(score) - c["center"]) / max(1e-9, c["half_width"])
    ramp = min(1.0, max(-1.0, ramp))
    return round(1.0 + span * ramp, 4)


def reset_cache() -> None:
    """Tests."""
    _cal_cache.update(ts=0.0, cal=None)
