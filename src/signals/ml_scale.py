"""Self-calibrating ML confidence divisor (2026-08-18).

``ml_raw_confidence_scale`` maps ``|combined_score|`` onto ``raw_confidence``
for rows the ML stackers drove (``combine_source == "ml"``). It is not a Gate-1
formula — the LLM states its own confidence — it is a **SATURATION control**:
every row with ``|combined| >= divisor`` clips to ``raw_confidence = 1.0`` and
is rendered into the synthesis prompt as ``combined_confidence=100%``. Measured
2026-08-18 on ml-source BUY/SELL candidates, that anchor bin passed Gate 1 at
**67.2%** against 6-17% for every lower bin, and the LLM tracks the ML anchor
closely (spearman +0.68) precisely BECAUSE it varies — on weighted runs nearly
every top-40 prompt slot already reads 100%, the anchor carries no information,
and the model falls back to its own ~0.86 prior (spearman +0.015). A 100%
anchor only persuades when its neighbours are not also at 100%.

So the divisor sets how much of the ML cross-section shouts, and THAT is what
drives the arm's exposure. A hardcoded divisor cannot hold it: the stackers
retrain WEEKLY and their conviction scale moves with the artifact (all-row
saturation went 16.3% -> ~26% across the 2026-08-15 retrain), so a constant
tuned this week is stale next week — the A/B silently drifts back to unequal
exposure, which is the failure mode that makes the arm's P&L unreadable.

**What is pinned is the SATURATION SHARE, not the divisor.** Each refresh:

1.  measure the WEIGHTED arm's own saturation share over the trailing window
    (its rows clipped at ``raw_confidence >= 0.999``) — the parity target,
    shrunk toward the documented prior ``ml_saturation_target`` and clamped;
2.  read the ml-source ``|combined_score|`` distribution over the same window
    and take the ``1 - target`` quantile — the divisor that reproduces that
    share on THIS artifact's conviction scale;
3.  shrink that toward the static ``ml_raw_confidence_scale`` setting by
    evidence (``ml_scale_prior_n``), clamp to a fixed multiplicative band
    around it, and report to the calibration registry;
4.  fail soft to the static setting on ANY failure — a DB hiccup must never
    silently re-scale the live book.

**No circularity.** The evidence is the weighted arm's saturation (governed by
``rank_raw_confidence_scale``) and the ml-source ``combined_score`` — and
``combined_score`` is persisted UNCLIPPED, so the quantile is not censored by
the divisor currently in force. The ML divisor never appears in its own inputs,
which is what makes re-deriving it from the panel sound rather than a backtest.

Point-in-time: the query honours ``analysis_asof`` and the TTL cache is
registered in ``asof.reset_all_calibration_caches``, so a walk-forward step
cannot reuse a divisor solved on later rows.

**Why this does NOT register a CONFIDENCE_EPOCH** (the question a reader will
reach for, since the divisor does move ``raw_confidence`` on ml-source rows).
Three reasons, in order of weight: (1) ``CONFIDENCE_EPOCH`` is a single GLOBAL
instant covering every row of every combine, so registering one for an ML-only
divisor would discard the WEIGHTED majority's confidence history too — a
disproportionate cost for a change that touches a minority of rows; (2) the
standing rule registers CATEGORICAL changes, not refinements, and this is a
recalibrated divisor on the same quantity — the same class as the real-fill
cost recalibration, which likewise does not epoch; (3) most importantly, this
mechanism HOLDS THE MEANING CONSTANT while the number moves. Pinning the
saturation share is what makes ML confidence mean the same thing across
artifacts, so the drifting value is the thing that keeps the series
comparable, not the thing that breaks it. If the SHARE target itself is ever
redefined, that IS categorical — epoch it then.

History: memory/ml-arm-exposure-calibration-2026-08.md.
"""

from __future__ import annotations

import threading
import time
from datetime import date, timedelta
from typing import Optional

from loguru import logger

from config.settings import settings

# Saturation is judged at the clip point; raw_confidence is rounded on the way
# to the panel, so compare with a tolerance rather than == 1.0.
_SATURATED = 0.999

# SAFETY GUARDRAILS — fixed module constants, never settings. A knob that
# weakens a guardrail is a knob that eventually gets turned. The band is
# multiplicative around the static setting: the calibration may retune the
# divisor, it may not run away with it.
_BAND_LO, _BAND_HI = 1.0 / 3.0, 3.0
# The measured target share is itself clamped: a weighted arm that saturated
# almost everything (or nothing) must not drag the ML arm to an extreme.
_TARGET_LO, _TARGET_HI = 0.03, 0.25

_TTL_SECONDS = 3 * 3600
_CACHE: dict = {"ts": 0.0, "scale": None}
_LOCK = threading.Lock()      # single-flight: the combine runs on many threads


def _static_scale() -> float:
    """The documented prior — the hand-tuned constant this shrinks toward."""
    return float(getattr(settings, "ml_raw_confidence_scale", 0.12))


def _window_bounds() -> tuple:
    """``(start, end)`` ISO dates for the evidence window. ``end`` is the
    ``analysis_asof`` cutoff when one is installed (exclusive), else None =
    up to now."""
    end: Optional[str] = None
    try:
        from src.analysis.asof import current_asof
        end = current_asof()
    except Exception:
        end = None
    days = max(2, int(getattr(settings, "ml_scale_window_days", 5)))
    anchor = date.fromisoformat(str(end)[:10]) if end else date.today() + timedelta(days=1)
    # Calendar span covering `days` TRADING days: two weekend days per week
    # plus one for a holiday. Deliberately TIGHT — over-padding widens the
    # window across a retrain boundary and mixes two artifacts' conviction
    # scales, which is the drift this calibration exists to track.
    start = (anchor - timedelta(days=days + 2 * ((days + 4) // 5) + 1)).isoformat()
    return start, (str(end)[:10] if end else None)


def measure(days: Optional[int] = None) -> dict:
    """The raw measurement behind the calibration, as a dict (also the CLI /
    test surface): weighted saturation share, ml row count, and the solved
    quantile. Returns ``{}`` when the evidence is unusable."""
    import numpy as np

    from src.db import repo

    start, end = _window_bounds()
    if days:
        anchor = date.fromisoformat(end) if end else date.today() + timedelta(days=1)
        d = max(2, int(days))
        start = (anchor - timedelta(days=d + 2 * ((d + 4) // 5) + 1)).isoformat()
    where = ["signal_date >= ?", "combined_score IS NOT NULL",
             "raw_confidence IS NOT NULL"]
    params: list = [start]
    if end:
        where.append("signal_date < ?")
        params.append(end)
    df = repo.fetch_df(
        "SELECT combine_source, combined_score, raw_confidence FROM signals "
        f"WHERE {' AND '.join(where)}", params=params)
    if df is None or df.empty:
        return {}
    src = df["combine_source"].fillna("weighted").replace({"": "weighted"})
    # The ML divisor governs the FULL swap only; a partial ml_buy/ml_sell row
    # keeps the weighted divisor, so it belongs to neither population here.
    ml = df[src == "ml"]
    wt = df[src == "weighted"]
    if len(ml) == 0 or len(wt) == 0:
        return {"n_ml": int(len(ml)), "n_weighted": int(len(wt))}
    wt_sat = float((wt["raw_confidence"].astype(float) >= _SATURATED).mean())
    ml_sat = float((ml["raw_confidence"].astype(float) >= _SATURATED).mean())
    target = float(np.clip(
        _shrink(float(getattr(settings, "ml_saturation_target", 0.09)),
                float(getattr(settings, "ml_scale_prior_n", 4000)), wt_sat, len(wt)),
        _TARGET_LO, _TARGET_HI))
    absx = ml["combined_score"].astype(float).abs()
    absx = absx[np.isfinite(absx) & (absx > 0)]
    if len(absx) == 0:
        return {"n_ml": 0, "n_weighted": int(len(wt))}
    solved = float(np.quantile(absx, max(0.0, min(1.0, 1.0 - target))))
    return {"n_ml": int(len(ml)), "n_weighted": int(len(wt)),
            "weighted_saturation": round(wt_sat, 4),
            "ml_saturation": round(ml_sat, 4),
            "target_share": round(target, 4),
            "solved_scale": round(solved, 6),
            "window_start": start, "window_end": end}


def _shrink(prior: float, prior_n: float, observed: Optional[float], n_obs: int) -> float:
    from src.performance.calibration import shrink
    return shrink(prior, prior_n, observed, n_obs)


def calibrate_ml_confidence_scale(force: bool = False) -> float:
    """The live ML confidence divisor. Cached ``_TTL_SECONDS``; fail-soft to
    the static ``ml_raw_confidence_scale`` on every failure path."""
    static = _static_scale()
    if not getattr(settings, "enable_ml_scale_calibration", False):
        return static
    now = time.time()
    hit = _CACHE["scale"]
    if not force and hit is not None and (now - _CACHE["ts"]) < _TTL_SECONDS:
        return hit
    with _LOCK:
        hit = _CACHE["scale"]                      # re-check under the lock
        if not force and hit is not None and (time.time() - _CACHE["ts"]) < _TTL_SECONDS:
            return hit
        value, note, n_ev = static, "static (no evidence)", 0
        try:
            m = measure()
            n_ml = int(m.get("n_ml", 0))
            min_rows = max(200, int(getattr(settings, "ml_scale_min_rows", 2000)))
            if m.get("solved_scale") and n_ml >= min_rows:
                prior_n = float(getattr(settings, "ml_scale_prior_n", 4000))
                blended = _shrink(static, prior_n, float(m["solved_scale"]), n_ml)
                value = float(min(max(blended, static * _BAND_LO), static * _BAND_HI))
                n_ev = n_ml
                note = (f"target {m['target_share']:.3f} (weighted "
                        f"{m['weighted_saturation']:.3f}) | ml sat "
                        f"{m['ml_saturation']:.3f} | solved {m['solved_scale']:.4f}")
                logger.info(
                    f"[ml_scale] divisor {value:.4f} (static {static:.4f}, solved "
                    f"{m['solved_scale']:.4f}, {n_ml} ml rows) — pinning saturation "
                    f"at {m['target_share']*100:.1f}% (weighted arm "
                    f"{m['weighted_saturation']*100:.1f}%)")
            else:
                note = f"static — only {n_ml} ml rows (< {min_rows})"
                logger.info(f"[ml_scale] {note}; keeping {static:.4f}")
        except Exception as e:
            logger.warning(f"[ml_scale] calibration failed (keeping static "
                           f"{static:.4f}): {e}")
            value, note, n_ev = static, f"static — calibration failed: {e}", 0
        try:
            from src.performance.calibration import report_calibration
            report_calibration("ml_confidence_scale", value=value, prior=static,
                               n_evidence=n_ev, unit="divisor", note=note)
        except Exception:
            pass
        _CACHE.update(ts=time.time(), scale=value)
        return value


def reset_cache() -> None:
    """Test / asof hook."""
    _CACHE.update(ts=0.0, scale=None)


if __name__ == "__main__":       # pragma: no cover - operator CLI
    import json
    print(json.dumps(measure(), indent=2, default=str))
    print(f"\nlive divisor: {calibrate_ml_confidence_scale(force=True):.6f}"
          f"   (static {_static_scale():.6f})")
