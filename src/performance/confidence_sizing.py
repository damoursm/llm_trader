"""Confidence recalibration for sizing — feed sizing the EMPIRICAL win rate of a
stated-confidence band, not the stated number (2026-07-12).

LLM-stated confidence is weakly calibrated (ledger ρ ≈ +0.07; the 2026-06
diagnostics even measured it anti-predictive for a stretch). The existing
mitigation — the compressed ``confidence_size_span`` ramp inside
``_position_multiplier`` — merely LIMITS how much the unreliable number can
move size. This module adds the missing half: a **self-calibrating tilt whose
input is the ledger's own evidence** about each confidence band, through the
house calibration idiom (documented prior → evidence → Bayesian shrinkage →
clamps → fail-soft → registry-reported):

    p_pool     = pooled win rate over eligible closed trades      (the prior/center)
    p_shrunk_b = (wins_b + k·p_pool) / (n_b + k)                  (per band, k = prior_n)
    mult       = 1 + span × clamp((p_shrunk_b − p_pool) / half_width, −1, +1)

The shrinkage IS the evidence throttle: a thin band pulls to ``p_pool`` and the
tilt vanishes; if stated confidence carries no information at all, every band
shrinks to the pool and the layer is exactly neutral. Crucially the tilt follows
the EVIDENCE wherever it points — a high-stated-confidence band that empirically
LOSES is sized DOWN (the inversion the raw ramp can never express).

Deliberately a STANDALONE multiplicative layer (like breadth / expected-edge /
predictability) rather than a replacement of ``_position_multiplier``: the
edge-sizing model uses the tier product as its Bayesian prior, so swapping the
ramp out would silently shift that model's prior; layering keeps every existing
calibration honest and makes this one independently auditable
(``confidence_recal_multiplier`` stamped per trade, registry row per run).

Bands are shared with the dashboard's confidence-calibration analysis
(``src/analysis/confidence_calibration._BUCKETS``) so the Method-Performance
buckets and the sizing tilt always read the same structure. Fail-soft
throughout — any error yields an inert calibration and neutral 1.0 multipliers.
"""

from __future__ import annotations

from typing import List, Optional

from loguru import logger

from config.settings import settings

# Single source of truth for the band edges — the dashboard analysis module.
from src.analysis.confidence_calibration import _BUCKETS

# An inert calibration: no bands ⇒ confidence_sizing_multiplier is exactly 1.0.
_INERT = {"pool_win": None, "n": 0, "bands": []}


def calibrate_confidence_sizing(trades: Optional[List[dict]]) -> dict:
    """Per-band empirical win rates from the closed ledger, shrunk toward the
    pooled win rate by ``confidence_recal_prior_n`` observations.

    Eligible = CLOSED trades carrying a numeric entry ``confidence`` and
    ``return_pct`` (win = return_pct > 0, the spread-adjusted convention).
    Below ``confidence_recal_min_trades`` eligible closes the calibration is
    inert — the layer starts neutral and only speaks once the ledger can.
    Registry-reported so the Data Quality tab shows the live band spread."""
    if not settings.enable_confidence_recal_sizing:
        return dict(_INERT)
    cal = dict(_INERT)
    try:
        rows = []
        for t in trades or []:
            if t.get("status") != "CLOSED":
                continue
            conf, ret = t.get("confidence"), t.get("return_pct")
            if conf is None or ret is None:
                continue
            try:
                rows.append((float(conf), float(ret) > 0.0))
            except (TypeError, ValueError):
                continue
        n = len(rows)
        if n < int(settings.confidence_recal_min_trades):
            return cal

        pool_win = sum(1 for _, w in rows if w) / n
        k = max(0, int(settings.confidence_recal_prior_n))
        bands = []
        for lo, hi, label in _BUCKETS:
            seg = [w for c, w in rows if lo <= c < hi]
            n_b = len(seg)
            wins_b = sum(1 for w in seg if w)
            # Bayesian shrinkage toward the pool — thin bands say nothing.
            p_shrunk = (wins_b + k * pool_win) / (n_b + k) if (n_b + k) else pool_win
            bands.append({"lo": lo, "hi": hi, "label": label, "n": n_b,
                          "win_rate": round(wins_b / n_b, 4) if n_b else None,
                          "p_shrunk": round(p_shrunk, 4)})
        cal = {"pool_win": round(pool_win, 4), "n": n, "bands": bands}

        # Registry: the live shrunk win-rate spread across POPULATED bands — the
        # strength the tilt actually has right now (0 = neutral layer).
        populated = [b["p_shrunk"] for b in bands if b["n"] > 0]
        spread = (max(populated) - min(populated)) if len(populated) >= 2 else 0.0
        try:
            from src.performance.calibration import report_calibration
            report_calibration(
                "confidence_recal_spread", value=round(spread, 4), prior=0.0,
                n_evidence=n, unit="win-rate pts",
                note="shrunk per-band win-rate spread feeding the confidence-recal "
                     "sizing tilt (bands: "
                     + ", ".join(f"{b['label']}→{b['p_shrunk']:.2f}(n={b['n']})"
                                 for b in bands if b["n"]) + ")")
        except Exception:
            pass
    except Exception as e:
        logger.debug(f"[conf_recal] calibration failed: {e}")
        cal = dict(_INERT)
    return cal


def _band_for(confidence: float, cal: dict) -> Optional[dict]:
    for b in cal.get("bands") or []:
        if b["lo"] <= confidence < b["hi"]:
            return b
    return None


def confidence_sizing_multiplier(confidence, cal: Optional[dict]) -> float:
    """CONTINUOUS size tilt from the trade's confidence band's EMPIRICAL win rate:

        mult = 1 + span × clamp((p_shrunk − p_pool) / half_width, −1, +1)

    Bounded to [1 − span, 1 + span]; neutral 1.0 when the layer is disabled, the
    confidence is missing, the calibration is inert (thin ledger), or the band
    holds no closed trades (absence of evidence, not a verdict)."""
    if not settings.enable_confidence_recal_sizing or confidence is None or not cal:
        return 1.0
    pool = cal.get("pool_win")
    if pool is None:
        return 1.0
    try:
        band = _band_for(float(confidence), cal)
    except (TypeError, ValueError):
        return 1.0
    if band is None or not band.get("n"):
        return 1.0
    span = max(0.0, float(settings.confidence_recal_span))
    half = max(1e-9, float(settings.confidence_recal_half_width))
    ramp = (float(band["p_shrunk"]) - float(pool)) / half
    ramp = min(1.0, max(-1.0, ramp))
    return round(1.0 + span * ramp, 4)
