"""Kaufman efficiency + ADX·DMI as directional trend METHODS (× trend context),
predicting continuation OR reversal via a LEARNED orientation.

Four methods, one per (feature × trend context): ``kaufman_long`` / ``adx_long``
are active only when the stock is in an **uptrend**; ``kaufman_short`` /
``adx_short`` only in a **downtrend** (0 = context not present = no view). Within
its context, a method does NOT blindly bet the trend continues — its raw signed
trend strength is multiplied by a **learned orientation** ∈ [−1, +1]:

    orientation = +1 → predict CONTINUATION (score keeps the trend's sign:
                       bullish in an uptrend, bearish in a downtrend)
    orientation = −1 → predict REVERSAL (score flips: a downtrend-context method
                       outputs a BULLISH score to predict a bounce)
    |orientation|     → confidence; magnitude also scales with trend strength

The orientation is measured per method from the signals panel on the PIVOT
basis (2026-08-16 rebase): the per-day rank IC of the raw signed trend FEATURE
against the signed pivot target within the method's active context, shrunk
toward 0 = ABSTAIN by signal-days — an unproven context contributes NOTHING
(the catalyst_tilt idiom). The old formula shrank toward a +1 CONTINUATION
prior on a drift-biased continuation-rate statistic at the fixed horizon,
which held all four contexts at +0.28..+0.39 while the panel measured them
DESCENDING (adx_long daily IC −0.049, t −3.2 — the 2026-08-16 decile-direction
audit). See ``calibrate_trend_orientation``. The final score follows the house
sign convention (positive = predicted up, negative = predicted down,
|score| = confidence).

Because a name is up- OR down-trending (never both), each method is sparse and
one-sided in its CONTEXT, which is why they fold into ``combined_score`` as an
additive overlay OUTSIDE the normalised weight pool (aggregator.build_signals),
not pooled methods that would dampen the non-trending names.

Reuses the daily cache-first OHLCV + Wilder DMI from ``trend_strength`` (called
just before this in the aggregator loop, so the cache is warm — no extra fetch).
"""

import time
from typing import Optional

import numpy as np
import pandas as pd
from loguru import logger

from config.settings import settings
from src.signals.trend_strength import _MIN_ROWS, _compute_dmi, _get_ohlcv

TREND_PREDICT_METHODS = ("kaufman_long", "kaufman_short", "adx_long", "adx_short")
_ZERO = {m: 0.0 for m in TREND_PREDICT_METHODS}
# Default orientation: +1 = predict CONTINUATION (with the trend) for every
# method. calibrate_trend_orientation() learns per-method deviations toward
# reversal (−1) as the forward returns confirm it.
_CONTINUATION = {m: 1.0 for m in TREND_PREDICT_METHODS}
_EPS = 1e-9
# Each method's raw signed feature column + its trend side (context): +1 = the
# method is active in UPtrends, −1 = active in DOWNtrends.
_METHOD_SPEC = {
    "kaufman_long":  ("er_signed", +1),
    "kaufman_short": ("er_signed", -1),
    "adx_long":      ("adx_signed", +1),
    "adx_short":     ("adx_signed", -1),
}
_orient_cache: dict = {"ts": 0.0, "orient": None}


def _signed_efficiency_ratio(close: pd.Series, window: int) -> Optional[float]:
    """Kaufman efficiency ratio with the trend's SIGN kept: net move over the
    window ÷ the total path length, in [−1, +1] (positive = an efficient up-move,
    negative = an efficient down-move). None when the path is degenerate."""
    if len(close) < window + 1:
        return None
    seg = close.iloc[-(window + 1):].astype(float)
    net = float(seg.iloc[-1] - seg.iloc[0])
    path = float(seg.diff().abs().sum())
    if not (path > 0):
        return None
    return max(-1.0, min(1.0, net / path))


def compute_trend_predictability_scores(ticker: str, df: Optional[pd.DataFrame] = None,
                                        orientation: Optional[dict] = None) -> dict:
    """``{kaufman_long, kaufman_short, adx_long, adx_short}`` each ∈ [−1, +1] for a
    ticker. Each method is active only in its trend CONTEXT (``*_long`` on an
    uptrend, ``*_short`` on a downtrend) and its raw signed trend strength is
    multiplied by the LEARNED ``orientation`` for that method: a value of +1
    predicts continuation (score keeps the trend's sign), −1 predicts reversal
    (score flips — e.g. a downtrend-context method outputs a BULLISH score to
    predict a bounce), and the magnitude scales the conviction. ``orientation``
    defaults to all +1 (pure continuation); pass ``calibrate_trend_orientation()``
    to use the learned values. All zeros when data is insufficient. ``df`` lets a
    caller pass a pre-fetched OHLCV frame; otherwise the daily cache-first fetch."""
    er_window = max(2, int(settings.predictability_er_window))
    adx_period = max(2, int(settings.predictability_adx_period))
    cap = max(1e-9, float(settings.predictability_adx_cap))
    o = orientation or _CONTINUATION

    if df is None:
        df = _get_ohlcv(ticker)
    if df is None or df.empty or len(df) < _MIN_ROWS or "Close" not in df.columns:
        return dict(_ZERO)

    def _clip(v: float) -> float:
        return round(max(-1.0, min(1.0, v)), 3)

    out = dict(_ZERO)
    try:
        close = pd.to_numeric(df["Close"], errors="coerce").dropna()
        er = _signed_efficiency_ratio(close, er_window)
        if er is not None:
            out["kaufman_long"] = _clip(o.get("kaufman_long", 1.0) * er) if er > 0 else 0.0
            out["kaufman_short"] = _clip(o.get("kaufman_short", 1.0) * er) if er < 0 else 0.0

        if {"High", "Low"}.issubset(df.columns):
            d = df.copy()
            for c in ("High", "Low", "Close"):
                d[c] = pd.to_numeric(d[c], errors="coerce")
            adx, plus_di, minus_di = _compute_dmi(d, adx_period)
            if np.isfinite(adx):
                sign = 1.0 if plus_di > minus_di else (-1.0 if minus_di > plus_di else 0.0)
                adx_signed = max(-1.0, min(1.0, sign * min(float(adx), cap) / cap))
                out["adx_long"] = _clip(o.get("adx_long", 1.0) * adx_signed) if adx_signed > 0 else 0.0
                out["adx_short"] = _clip(o.get("adx_short", 1.0) * adx_signed) if adx_signed < 0 else 0.0
    except Exception as exc:
        logger.debug(f"[trend_predict] {ticker}: {exc}")
        return dict(_ZERO)
    return out


# ── learned continuation/reversal orientation ────────────────────────────────

def _store_orient(now: float, orient: dict, feature_panel: Optional[pd.DataFrame]) -> None:
    if feature_panel is None:                 # only cache the live (DB-built) calibration
        _orient_cache.update(ts=now, orient=orient)


# A mean daily IC at/above this magnitude maps to a FULL ±1 orientation
# (before the days-based shrinkage). 0.04 is a strong IC for these features.
_ORIENT_IC_SCALE = 0.04


def calibrate_trend_orientation(feature_panel: Optional[pd.DataFrame] = None) -> dict:
    """Per-method orientation ∈ [−1, +1] — CONTINUATION (+) or REVERSAL (−),
    learned from the signals panel on the PIVOT basis (2026-08-16 rebase).

    For each method's active context (uptrend for ``*_long``, downtrend for
    ``*_short``): the per-day Spearman IC of the raw signed trend FEATURE
    (``er_signed``/``adx_signed`` — deliberately the orientation-FREE inputs,
    because the persisted METHOD scores carry the served orientation, and
    calibrating on those would let a served flip feed back into its own next
    calibration) against the signed PIVOT target, on the tradeable-price
    subset. ``orient_obs = clip(mean_daily_IC / _ORIENT_IC_SCALE, −1, +1)``,
    shrunk toward **0 = ABSTAIN** by signal-days (``trend_orientation_prior_n``)
    — an unproven context contributes NOTHING (the catalyst_tilt idiom). The
    old formula shrank toward a +1 continuation prior on a drift-biased
    continuation-rate at the fixed horizon, which held every context at
    +0.28..+0.39 while the panel measured all four DESCENDING (the 2026-08-16
    decile-direction audit; epoch-registered on the four methods because the
    persisted scores are orientation-inclusive). Cached
    (``trend_orientation_cal_ttl_seconds``) and fully fail-soft (→ all 0.0,
    abstain), so this layer never breaks scoring. Reports each orientation to
    the calibration registry."""
    if not settings.enable_trend_predictability_methods:
        return dict(_ZERO)
    now = time.time()
    if feature_panel is None and _orient_cache["orient"] is not None \
            and (now - _orient_cache["ts"]) < float(settings.trend_orientation_cal_ttl_seconds):
        return _orient_cache["orient"]

    orient = dict(_ZERO)
    diag: dict = {}
    try:
        horizon = int(settings.predictability_horizon)
        fp = feature_panel
        if fp is None:
            from src.analysis.signal_panel import build_panel
            from src.analysis.predictability import attach_feature_signals
            panel = build_panel(horizons=(horizon,), days=int(settings.trend_orientation_cal_days))
            fp = attach_feature_signals(panel) if (panel is not None and not panel.empty) else None
        if fp is None or getattr(fp, "empty", True):
            _store_orient(now, orient, feature_panel)
            return orient

        # Pivot label first (the decision basis); fixed horizon only as the
        # fallback for frames that predate the pivot column.
        _pv = fp.get("fwd_ret_pivot")
        fwd = pd.to_numeric(_pv, errors="coerce") if _pv is not None else None
        if fwd is None or not fwd.notna().any():
            _alt = fp.get(f"fwd_ret_{horizon}d")
            fwd = pd.to_numeric(_alt, errors="coerce") if _alt is not None else None
        if fwd is None:
            _store_orient(now, orient, feature_panel)
            return orient
        # Tradeable-price subset (pivot numbers are only meaningful gated);
        # fail-soft to the full frame when the column is absent (test fixtures).
        if "price" in fp.columns:
            pr = pd.to_numeric(fp["price"], errors="coerce")
            gated = pr >= float(getattr(settings, "trade_min_price", 5.0))
            if int(gated.sum()) >= int(settings.trend_orientation_cal_min_rows):
                fp = fp[gated]
                fwd = fwd[gated]
        prior_n = max(0, int(settings.trend_orientation_prior_n))
        min_rows = int(settings.trend_orientation_cal_min_rows)
        for m, (fcol, side) in _METHOD_SPEC.items():
            raw = pd.to_numeric(fp.get(fcol), errors="coerce")
            if raw is None or fwd is None:
                continue
            active = (raw > _EPS) if side > 0 else (raw < -_EPS)
            valid = active & fwd.notna() & raw.notna()
            if int(valid.sum()) < min_rows or "signal_date" not in fp.columns:
                continue                                   # keep the 0 abstain prior
            sub = pd.DataFrame({"day": fp.loc[valid, "signal_date"].astype(str).str[:10],
                                "x": raw[valid], "y": fwd[valid]})
            ics = []
            for _, g in sub.groupby("day"):
                if len(g) < 5 or g["x"].nunique() < 3:
                    continue
                ics.append(g["x"].rank().corr(g["y"].rank()))
            ics = pd.Series(ics, dtype=float).dropna()
            n_days = int(len(ics))
            if n_days < 5:
                continue
            ic_mean = float(ics.mean())
            orient_obs = max(-1.0, min(1.0, ic_mean / _ORIENT_IC_SCALE))
            o = n_days * orient_obs / (n_days + prior_n) if (n_days + prior_n) > 0 else 0.0
            orient[m] = round(max(-1.0, min(1.0, o)), 3)
            diag[m] = (round(ic_mean, 4), n_days)
        _report_orient(orient, diag)
    except Exception as e:
        logger.debug(f"[trend_predict] orientation calibration failed: {e}")
        orient = dict(_ZERO)
    _store_orient(now, orient, feature_panel)
    return orient


def _report_orient(orient: dict, diag: dict) -> None:
    try:
        from src.performance.calibration import report_calibration
        for m in TREND_PREDICT_METHODS:
            ic_mean, n_days = diag.get(m, (None, 0))
            report_calibration(
                f"trend_orient_{m}", value=orient.get(m, 0.0), prior=0.0,
                n_evidence=n_days, unit="orient (+cont/−rev)",
                note=(f"pivot daily IC {ic_mean} over {n_days} signal-day(s)"
                      if ic_mean is not None else "abstain prior (thin data)"))
    except Exception:
        pass


def reset_cache() -> None:
    """Tests."""
    _orient_cache.update(ts=0.0, orient=None)
