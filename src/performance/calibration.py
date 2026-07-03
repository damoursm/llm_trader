"""Calibration registry — the one place every self-calibrating parameter reports to.

The system keeps replacing hardcoded constants with values learned from its own
data (real-fill costs, breadth sizing, session spread multipliers, the horizon
cost hurdle, …). Each of those mechanisms follows the same shape — a documented
PRIOR, an EVIDENCE source, Bayesian shrinkage between them, and clamps — but
they used to be invisible snowflakes: nothing recorded what value was actually
in force on a given run. This module fixes the observability half:

* every calibrated computation calls :func:`report_calibration` with its
  current effective value, its prior, and how much evidence backs it;
* the pipeline snapshots :func:`get_calibrations` into ``gate_diag`` at
  persist time (the established no-schema-change channel, like the
  price-provenance verdict), so every run records the exact calibrations it
  traded with;
* the dashboard's Data Quality tab renders the latest snapshot — current value
  vs prior vs evidence-n — so a drifting or mis-learning parameter is visible
  the way a dark data feed is.

Deliberately dependency-free (imported by ``spread``/``tracker``/…) and
process-local: the registry is a live view of THIS process; durable history
lives in the per-run ``gate_diag`` snapshots.
"""

from __future__ import annotations

from typing import Optional

_REGISTRY: dict = {}


def report_calibration(name: str, *, value, prior=None, n_evidence: int = 0,
                       unit: str = "", note: str = "") -> None:
    """Record the CURRENT effective value of one calibrated parameter.

    ``prior`` is the documented fallback the value shrinks toward (None when
    the parameter has no meaningful prior, e.g. a purely measured quantity);
    ``n_evidence`` is the number of observations behind the current value —
    0 means the prior is fully in force. Overwrites any previous report for
    the same name (the registry holds current state, not history)."""
    try:
        _REGISTRY[name] = {
            "name": name,
            "value": round(float(value), 6) if value is not None else None,
            "prior": round(float(prior), 6) if prior is not None else None,
            "n_evidence": int(n_evidence),
            "unit": unit,
            "note": note,
        }
    except (TypeError, ValueError):
        _REGISTRY[name] = {"name": name, "value": None, "prior": None,
                           "n_evidence": int(n_evidence), "unit": unit, "note": note}


def get_calibrations() -> list:
    """Sorted snapshot of every reported calibration (for gate_diag / dashboard)."""
    return [dict(_REGISTRY[k]) for k in sorted(_REGISTRY)]


def reset_calibrations() -> None:
    """Clear the registry (tests)."""
    _REGISTRY.clear()


def shrink(prior: float, prior_n: float, observed: Optional[float], n_obs: int) -> float:
    """The shared Bayesian-shrinkage helper every calibration uses:
    ``(prior_n·prior + n_obs·observed) / (prior_n + n_obs)``. With no
    observations (or ``observed`` None) the prior holds exactly; evidence
    pulls the posterior toward the measurement in proportion to sample size."""
    if observed is None or n_obs <= 0:
        return float(prior)
    return (float(prior_n) * float(prior) + float(n_obs) * float(observed)) / (float(prior_n) + float(n_obs))
