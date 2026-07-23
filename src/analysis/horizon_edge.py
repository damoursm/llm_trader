"""True horizon-edge decay of combined_score — the realized edge by holding horizon.

Measured tick-by-tick, ticker-by-ticker over the signals panel: every scored
ticker at every tick is a hypothetical entry in its combined_score direction, and
we record the realized *oriented* forward return at each holding horizon. This is
the ground truth the horizon time-stop rests on — does the signal's edge peak,
then decay / reverse? — at THOUSANDS of observations, where the held-panel
``horizon`` IC can't validate it (only ~5 real positions have ever outlived their
window, so that IC is stuck at n≈1).

First result (actionable subset, confidence ≥ 0.85 — the traded population): the
edge peaks at ~1–2 days (+1% at 2d), is gone by 3d, and turns NEGATIVE by 5–7d
(win ~37%). So holding a gated position past ~2–3 days gives the edge back — a
SHORTER window than the entry ``target_horizon``.

Two products:

* ``compute_horizon_edge_curve`` — the curve (n / IC / win / edge per horizon),
  for the dashboard + offline inspection.
* ``calibrate_edge_horizon`` — the measured edge-positive WINDOW (trading days) +
  an evidence STRENGTH, consumed by the exit logic (``exit_methods._edge_decay_pressure``
  emits a per-position exit signal; ``edge_decay_floor_adjustment`` turns it into
  an evidence-throttled raise of the confidence-loss floor). It starts inert on
  the current thin data and tightens the effective time-stop toward the realized
  decay point only as the finding accrues — the same measure-then-act, never-
  assumed idiom as the other calibrations. Registry-reported.
"""

from __future__ import annotations

import time
from typing import Optional, Sequence

import pandas as pd
from loguru import logger

from config.settings import settings

_EPS = 1e-12
_DEFAULT_HORIZONS = (1, 2, 3, 5, 7, 10)
_cal_cache: dict = {"ts": 0.0, "cal": None}
# An inert calibration: no window ⇒ _edge_decay_pressure / the floor nudge are 0.
_INERT = {"edge_days": None, "strength": 0.0, "n_obs": 0, "peak_day": None}


def compute_horizon_edge_curve(days: Optional[int] = None,
                               horizons: Sequence[int] = _DEFAULT_HORIZONS,
                               conf_min: Optional[float] = 0.85, min_n: int = 20,
                               panel: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    """Realized edge of ``combined_score`` by holding horizon, over the signals
    panel (optionally restricted to ``confidence ≥ conf_min`` — the traded
    population). One row per horizon: ``n`` (joint obs with a forward return),
    ``ic`` (Spearman of combined_score vs forward return), ``win`` % (directional
    hit), ``edge`` % (mean ``sign(combined_score) × forward_return`` — the P&L of
    following the signal that long). Horizons below ``min_n`` report None."""
    from src.analysis.signal_panel import build_panel, _spearman
    if panel is None:
        panel = build_panel(horizons=horizons, days=days)
    if panel is None or panel.empty:
        return pd.DataFrame()
    cs = pd.to_numeric(panel.get("combined_score"), errors="coerce")
    base = cs.notna() & (cs.abs() > _EPS)
    if conf_min is not None and "confidence" in panel.columns:
        conf = pd.to_numeric(panel["confidence"], errors="coerce")
        base = base & conf.notna() & (conf >= float(conf_min))

    rows = []
    for h in horizons:
        col = f"fwd_ret_{h}d"
        if col not in panel.columns:
            rows.append({"horizon": int(h), "n": 0, "ic": None, "win": None, "edge": None})
            continue
        fwd = pd.to_numeric(panel[col], errors="coerce")
        valid = base & fwd.notna()
        n = int(valid.sum())
        row = {"horizon": int(h), "n": n}
        if n < min_n:
            row.update(ic=None, win=None, edge=None)
        else:
            s, f = cs[valid], fwd[valid]
            ic = _spearman(s, f)
            oriented = f.where(s > 0, -f)
            row["ic"] = round(ic, 4) if ic is not None else None
            row["win"] = round(float((oriented > 0).mean() * 100.0), 2)
            row["edge"] = round(float(oriented.mean()), 4)
        rows.append(row)
    return pd.DataFrame(rows)


def edge_exhaustion_days(curve: pd.DataFrame, min_n: int = 20) -> Optional[int]:
    """The largest holding horizon at which the realized edge is still POSITIVE
    (mean edge > 0 and win ≥ 50%, ≥ ``min_n`` obs) — the point past which holding
    stops helping. None when no horizon clears it, OR when the edge is still
    positive at the LONGEST measured horizon (no decay observed → don't impose a
    stop while the edge may still extend)."""
    if curve is None or curve.empty:
        return None
    ok = curve[(curve["n"] >= min_n) & curve["edge"].notna() & (curve["edge"] > 0)
               & (curve["win"].fillna(0.0) >= 50.0)]
    if ok.empty:
        return None
    last_pos = int(ok["horizon"].max())
    # Require an OBSERVED decay: some longer horizon whose edge went non-positive.
    beyond = curve[(curve["horizon"] > last_pos) & curve["edge"].notna()
                   & (curve["edge"] <= 0)]
    return last_pos if not beyond.empty else None


def _peak_day(curve: pd.DataFrame) -> Optional[int]:
    if curve is None or curve.empty or curve["edge"].dropna().empty:
        return None
    sub = curve[curve["edge"].notna()]
    return int(sub.loc[sub["edge"].idxmax(), "horizon"])


def _store(now: float, cal: dict, panel: Optional[pd.DataFrame]) -> None:
    if panel is None:                          # only cache the live (DB-built) calibration
        _cal_cache.update(ts=now, cal=cal)


def calibrate_edge_horizon(panel: Optional[pd.DataFrame] = None) -> dict:
    """``{edge_days, strength, n_obs, peak_day}`` — the measured edge-positive
    window + its evidence weight, driving the edge-decay time-stop.

    ``edge_days`` = the last horizon whose realized edge is still positive (the
    exit window). ``strength`` ∈ [0, 1] ramps the DECISION nudge with the sample
    at the first NEGATIVE-edge horizon (the decision point), shrunk by
    ``edge_decay_prior_obs`` — so it is ~inert on today's thin long-horizon data
    and firms up as the decay confirms. Cached ``edge_decay_cal_ttl_seconds`` and
    fully fail-soft (→ inert, so the exit logic is never broken by this layer)."""
    if not settings.enable_edge_decay_exit:
        return dict(_INERT)
    now = time.time()
    if panel is None and _cal_cache["cal"] is not None \
            and (now - _cal_cache["ts"]) < float(settings.edge_decay_cal_ttl_seconds):
        return _cal_cache["cal"]

    cal = dict(_INERT)
    try:
        min_n = int(settings.edge_decay_min_n)
        curve = compute_horizon_edge_curve(
            days=int(settings.edge_decay_cal_days), horizons=_DEFAULT_HORIZONS,
            conf_min=float(settings.edge_decay_conf_min), min_n=min_n, panel=panel)
        if curve is None or curve.empty:
            _store(now, cal, panel)
            return cal
        edge_days = edge_exhaustion_days(curve, min_n)
        # Evidence = obs at the FIRST non-positive-edge horizon (the decision point);
        # fall back to the largest sample if the edge never turns.
        neg = curve[curve["edge"].notna() & (curve["edge"] <= 0)]
        n_dec = int(neg["n"].iloc[0]) if not neg.empty else int(curve["n"].max() or 0)
        prior = max(1, int(settings.edge_decay_prior_obs))
        strength = round(n_dec / (n_dec + prior), 4) if edge_days else 0.0
        cal = {"edge_days": edge_days, "strength": strength, "n_obs": n_dec,
               "peak_day": _peak_day(curve)}
        _report(cal)
    except Exception as e:
        logger.debug(f"[edge_decay] calibration failed: {e}")
        cal = dict(_INERT)
    _store(now, cal, panel)
    return cal


def _report(cal: dict) -> None:
    try:
        from src.performance.calibration import report_calibration
        report_calibration(
            "edge_decay_window", value=(cal.get("edge_days") or 0), prior=0.0,
            n_evidence=cal.get("n_obs", 0), unit="trading days",
            note=f"realized edge-positive window (combined_score, conf≥"
                 f"{settings.edge_decay_conf_min}); time-stop tightens toward this "
                 f"(peak ~{cal.get('peak_day')}d)")
        report_calibration(
            "edge_decay_strength", value=cal.get("strength", 0.0), prior=0.0,
            n_evidence=cal.get("n_obs", 0), unit="floor-nudge weight",
            note="evidence-throttled edge-decay exit strength (inert until the decay confirms)")
    except Exception:
        pass


def edge_decay_floor_adjustment(escores: Optional[dict], cal: Optional[dict] = None) -> float:
    """Evidence-throttled raise of the confidence-loss close floor from the
    edge-decay signal. Reads the RAW ``edge_decay`` pressure (negative once a
    position is held past the measured edge window) from ``escores`` and scales it
    by the calibration ``strength`` and ``edge_decay_floor_cap``. ``+`` raises the
    floor (close a lukewarm reaffirm sooner); 0 when disabled / within the window /
    thin evidence. Bounded by the cap so it can never run away."""
    if not settings.enable_edge_decay_exit:
        return 0.0
    c = cal if cal is not None else calibrate_edge_horizon()
    strength = float(c.get("strength") or 0.0)
    if strength <= 0:
        return 0.0
    pressure = float((escores or {}).get("edge_decay") or 0.0)   # ≤ 0 past the window
    if pressure >= 0:
        return 0.0
    cap = float(settings.edge_decay_floor_cap)
    return round(min(cap, -pressure * strength * cap), 4)


def reset_cache() -> None:
    """Tests."""
    _cal_cache.update(ts=0.0, cal=None)
