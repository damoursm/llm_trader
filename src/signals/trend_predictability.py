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

The orientation is measured per method from the signals panel — how often that
trend context has actually continued vs reversed at the swing horizon — and shrunk
toward a CONTINUATION prior (+1) by signal-days, so each method starts as
continuation and flips toward reversal only as the forward returns confirm it
(e.g. it learns "clean downtrends bounce → predict up"). See
``calibrate_trend_orientation``. The final score follows the house sign convention
(positive = predicted up, negative = predicted down, |score| = confidence).

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


def calibrate_trend_orientation(feature_panel: Optional[pd.DataFrame] = None) -> dict:
    """Per-method orientation ∈ [−1, +1] — whether each trend context should
    predict CONTINUATION (+1) or REVERSAL (−1), learned from the signals panel.

    For each method's active context (uptrend for ``*_long``, downtrend for
    ``*_short``), measures how often the trend CONTINUED at the swing horizon
    (``predictability_horizon``): ``agree`` = share whose forward move matched the
    trend, ``orient_obs = 2·agree − 1`` (+1 pure continuation, −1 pure reversal).
    That is shrunk toward the CONTINUATION prior (+1) by SIGNAL-DAYS
    (``trend_orientation_prior_n``), so a method starts as continuation and flips
    toward reversal only as independent days of forward returns confirm it — the
    same evidence-throttled idiom as the sizing calibrations. Cached
    (``trend_orientation_cal_ttl_seconds``) and fully fail-soft (→ all +1
    continuation), so this layer never breaks scoring. Reports each orientation to
    the calibration registry."""
    if not settings.enable_trend_predictability_methods:
        return dict(_CONTINUATION)
    now = time.time()
    if feature_panel is None and _orient_cache["orient"] is not None \
            and (now - _orient_cache["ts"]) < float(settings.trend_orientation_cal_ttl_seconds):
        return _orient_cache["orient"]

    orient = dict(_CONTINUATION)
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

        fwd = pd.to_numeric(fp.get(f"fwd_ret_{horizon}d"), errors="coerce")
        prior_n = max(0, int(settings.trend_orientation_prior_n))
        min_rows = int(settings.trend_orientation_cal_min_rows)
        for m, (fcol, side) in _METHOD_SPEC.items():
            raw = pd.to_numeric(fp.get(fcol), errors="coerce")
            if raw is None or fwd is None:
                continue
            active = (raw > _EPS) if side > 0 else (raw < -_EPS)
            valid = active & fwd.notna()
            if int(valid.sum()) < min_rows:
                continue                                   # keep the +1 continuation prior
            f = fwd[valid]
            moved = f != 0
            if not bool(moved.any()):
                continue
            cont = (f > 0) if side > 0 else (f < 0)        # did the trend continue?
            agree = float(cont[moved].mean())
            orient_obs = 2.0 * agree - 1.0                 # +1 continuation … −1 reversal
            n_days = int(fp.loc[valid, "signal_date"].nunique()) if "signal_date" in fp.columns else 0
            denom = prior_n + n_days
            o = ((prior_n * 1.0 + n_days * orient_obs) / denom) if denom > 0 else 1.0
            orient[m] = round(max(-1.0, min(1.0, o)), 3)
            diag[m] = (round(agree, 3), n_days)
        _report_orient(orient, diag)
    except Exception as e:
        logger.debug(f"[trend_predict] orientation calibration failed: {e}")
        orient = dict(_CONTINUATION)
    _store_orient(now, orient, feature_panel)
    return orient


def _report_orient(orient: dict, diag: dict) -> None:
    try:
        from src.performance.calibration import report_calibration
        for m in TREND_PREDICT_METHODS:
            agree, n_days = diag.get(m, (None, 0))
            report_calibration(
                f"trend_orient_{m}", value=orient.get(m, 1.0), prior=1.0,
                n_evidence=n_days, unit="orient (+cont/−rev)",
                note=(f"continuation rate {agree} over {n_days} signal-day(s)"
                      if agree is not None else "continuation prior (thin data)"))
    except Exception:
        pass


def reset_cache() -> None:
    """Tests."""
    _orient_cache.update(ts=0.0, orient=None)
