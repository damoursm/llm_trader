"""Unified expected-edge sizing — the learned successor to the sizing tiers.

Position size is currently a PRODUCT of hand-shaped conviction tiers
(compressed confidence × calibrated breadth), each individually evidence-based
but combined by construction, not by data. This module learns the combination:
a ridge-regularized linear model of REALIZED returns on standardized
entry-time features, whose influence on the final size grows with closed-trade
evidence and is exactly zero when data is thin.

    prior     = the existing conviction product (confidence × breadth tiers)
    edge_mult = 1 + edge_size_span × clamp(pred / target_std, −1, +1)
    blended   = (1 − w) × prior + w × edge_mult,   w = n_closed/(n_closed+prior_n)
    final     = existing size chain × (blended / prior)          (ratio form —
                w = 0 reproduces today's sizing bit-for-bit)

Features (fixed order, all derivable both at entry time and from stored
trades, oriented in the trade's direction where signed):

    breadth_frac · confidence · combined_score · news · momentum · off-RTH

The ridge penalty (λ·n on standardized features) shrinks coefficients toward
zero — a feature that doesn't predict realized returns contributes nothing —
and the evidence blend keeps the whole layer honest: below ``edge_min_closed``
realized trades the model has NO say, and it takes ``edge_prior_n`` closes for
an even split with the prior tiers. Never a gate, never inverts the sign of a
position — it only scales size within the span. Deterministic given the
ledger; cached briefly for callers that don't pass trades.
"""

from __future__ import annotations

import time
from typing import List, Optional

import numpy as np
from loguru import logger

from config.settings import settings
from src.performance.calibration import report_calibration

FEATURES = ("breadth_frac", "confidence", "combined_score", "news", "momentum", "off_rth")

_CACHE_TTL_S = 600.0
_cache: dict = {"ts": 0.0, "model": None}


def trade_features(trade: dict) -> Optional[List[float]]:
    """The model's feature vector for a stored trade (None when the trade
    carries no attribution). Signed scores are oriented in the trade's
    direction, so 'the signal supported this position' is always positive.

    The combined-score feature prefers the ABSOLUTE-basis twin
    (``signal_at_entry.combined_score_abs``, snapshotted at entry since
    2026-08-14) and falls back to ``combined_score`` — which on pre-shadow rows
    IS absolute — so the series stays on ONE basis across the rank cutover.
    The exact ``ex_combine`` convention from ml_exit, for the exact same
    reason: a basis change must never shift a fitted model's feature
    distribution mid-life."""
    ms = trade.get("method_scores") or {}
    if not ms:
        return None
    from src.performance.tracker import _trade_breadth_frac
    frac = _trade_breadth_frac(trade)
    if frac is None:
        return None
    sign = 1.0 if trade.get("action") == "BUY" else -1.0
    sig = trade.get("signal_at_entry") or {}
    combined = sig.get("combined_score_abs")
    if combined is None:
        combined = sig.get("combined_score") or 0.0
    return [
        float(frac),
        float(trade.get("confidence") or 0.0),
        sign * float(combined),
        sign * float(ms.get("news") or 0.0),
        sign * float(ms.get("momentum") or 0.0),
        1.0 if (trade.get("entry_session") or "rth") != "rth" else 0.0,
    ]


def _row_is_comparable(trade: dict) -> bool:
    """STANDING RULE (calibration comparability): a calibration may only fit
    data the current code produced — satisfied by RESTRICTING, never by
    converting. This model regresses returns on entry-time CONFIDENCE and the
    NEWS / MOMENTUM method scores, all of which have scale epochs:
    ``confidence`` re-scaled at CONFIDENCE_EPOCH (2026-08-14; measured 75% of
    eligible closes carried the pre-split scale when the sibling calibrations
    were gated) and ``news`` re-scaled at its 2026-08-17 scorer epoch (mean
    |news| halved at the v3/v4 prompt). Fitting across a scale break makes a
    coefficient describe the MIXTURE, not the relationship. Found ungated
    2026-08-22 (the only decision-feeding confidence consumer without the
    gate); the evidence blend ``w = n/(n+prior_n)`` is exactly what makes the
    smaller clean sample safe. Fail-OPEN on missing dates, like the helpers
    themselves — a malformed timestamp must not erase history."""
    from src.signals.method_epochs import (confidence_is_comparable,
                                           score_is_comparable)
    when = trade.get("entry_datetime") or trade.get("entry_date")
    return (confidence_is_comparable(when)
            and score_is_comparable("news", when)
            and score_is_comparable("momentum", when))


def fit_edge_model(trades: List[dict]) -> Optional[dict]:
    """Ridge-fit E[realized return | entry features] over CLOSED attributed
    trades. Returns ``{beta, mean, std, y_mean, target_std, n}`` or None when
    fewer than ``edge_min_closed`` usable closes exist (→ the layer is inert)."""
    rows, ys = [], []
    n_incomparable = 0
    for t in trades:
        if t.get("status") != "CLOSED" or t.get("return_pct") is None:
            continue
        if not _row_is_comparable(t):
            n_incomparable += 1              # superseded scale — never fit it
            continue
        f = trade_features(t)
        if f is None:
            continue
        rows.append(f)
        ys.append(float(t["return_pct"]))
    if n_incomparable:
        logger.debug(f"[edge_sizing] {n_incomparable} closed trade(s) excluded "
                     f"from the fit (pre-epoch confidence/news scale); "
                     f"{len(rows)} comparable rows remain")
    n = len(rows)
    if n < max(int(settings.edge_min_closed), len(FEATURES) + 2):
        return None
    X = np.asarray(rows, dtype=float)
    y = np.asarray(ys, dtype=float)
    mean = X.mean(axis=0)
    std = X.std(axis=0)
    std[std < 1e-9] = 1.0                      # constant feature → coef stays ~0
    Xs = (X - mean) / std
    y_mean = float(y.mean())
    yc = y - y_mean
    lam = float(settings.edge_ridge_lambda) * n
    beta = np.linalg.solve(Xs.T @ Xs + lam * np.eye(Xs.shape[1]), Xs.T @ yc)
    target_std = float(y.std()) or 1.0
    return {"beta": beta.tolist(), "mean": mean.tolist(), "std": std.tolist(),
            "y_mean": y_mean, "target_std": target_std, "n": n}


def get_edge_model(trades: Optional[List[dict]] = None) -> Optional[dict]:
    """The current model — recomputed from the passed ledger, or cached."""
    now = time.time()
    if trades is None and (now - _cache["ts"]) < _CACHE_TTL_S:
        return _cache["model"]
    if trades is None:
        from src.performance.tracker import _load_trades
        trades = _load_trades()
    model = fit_edge_model(trades) if settings.edge_sizing_enabled else None
    if trades is None or True:                 # always refresh the shared cache
        _cache.update(ts=now, model=model)
    return model


def edge_blend_ratio(features: Optional[List[float]],
                     prior_conviction: float,
                     model: Optional[dict]) -> tuple:
    """The multiplicative adjustment that hands sizing over from the prior
    tiers to the learned model as evidence accrues.

    Returns ``(ratio, meta)`` where ``final size = existing chain × ratio``.
    ratio = blended/prior with blended = (1−w)·prior + w·edge_mult; w=0 (thin
    data, missing features, disabled) → ratio exactly 1.0. Clamped to a sane
    band so a degenerate fit can never blow up a position."""
    meta = {"w": 0.0, "pred": None, "edge_mult": None}
    if (not settings.edge_sizing_enabled or model is None or features is None
            or prior_conviction <= 0):
        return 1.0, meta
    x = np.asarray(features, dtype=float)
    xs = (x - np.asarray(model["mean"])) / np.asarray(model["std"])
    pred = float(model["y_mean"] + xs @ np.asarray(model["beta"]))
    span = float(settings.edge_size_span)
    edge_mult = 1.0 + span * min(1.0, max(-1.0, pred / model["target_std"]))
    n = int(model["n"])
    w = n / (n + float(settings.edge_prior_n))
    blended = (1.0 - w) * prior_conviction + w * edge_mult
    ratio = min(1.6, max(0.6, blended / prior_conviction))
    meta.update(w=round(w, 4), pred=round(pred, 4), edge_mult=round(edge_mult, 4))
    report_calibration(
        "edge_model_weight", value=w, prior=0.0, n_evidence=n, unit="blend share",
        note=f"learned-sizing handover ({n} realized closes; "
             f"full parity at {int(settings.edge_prior_n)})")
    return round(ratio, 4), meta


def reset_cache() -> None:
    """Tests."""
    _cache.update(ts=0.0, model=None)
