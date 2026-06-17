"""Tests for src.analysis.broker_forensics."""

import pandas as pd

from src.analysis.broker_forensics import (
    compute_forensics, fill_outcomes, slippage_by_session, reject_reasons, drift_frequency,
)


def _orders():
    # 14:00 EDT = 18:00 UTC (rth); 07:00 EDT = 11:00 UTC (extended).
    return pd.DataFrame([
        {"status": "Filled", "filled_qty": 10, "slippage_bps": 12.0, "ok": True,
         "submitted_at": "2026-06-12T18:00:00+00:00", "error": None},
        {"status": "Filled", "filled_qty": 5, "slippage_bps": 30.0, "ok": True,
         "submitted_at": "2026-06-12T18:30:00+00:00", "error": None},
        {"status": "Filled", "filled_qty": 5, "slippage_bps": 70.0, "ok": True,
         "submitted_at": "2026-06-12T11:00:00+00:00", "error": None},
        {"status": "SETTLE_KILL", "filled_qty": 0, "slippage_bps": None, "ok": True,
         "submitted_at": "2026-06-12T18:00:00+00:00", "error": None},
        {"status": "SUBMIT_FAILED", "filled_qty": 0, "slippage_bps": None, "ok": False,
         "submitted_at": "2026-06-12T18:00:00+00:00", "error": "insufficient funds"},
        {"status": "Submitted", "filled_qty": 0, "slippage_bps": None, "ok": True,
         "submitted_at": "2026-06-12T18:00:00+00:00", "error": None},
    ])


def test_fill_outcomes_counts_and_rate():
    fo = fill_outcomes(_orders())
    assert fo["counts"]["filled"] == 3
    assert fo["counts"]["killed"] == 1
    assert fo["counts"]["failed"] == 1
    assert fo["counts"]["working"] == 1
    # terminal = filled(3)+killed(1)+failed(1) = 5; working excluded.
    assert fo["n_terminal"] == 5
    assert fo["fill_rate"] == 60.0


def test_slippage_grouped_by_session():
    sl = slippage_by_session(_orders())
    by = {r["session"]: r for r in sl.to_dict("records")}
    assert by["rth"]["n"] == 2
    assert by["extended"]["n"] == 1
    assert by["extended"]["mean_bps"] == 70.0


def test_reject_reasons():
    rr = reject_reasons(_orders())
    reasons = dict(zip(rr["reason"], rr["n"]))
    assert reasons.get("insufficient funds") == 1


def test_drift_frequency():
    rec = pd.DataFrame([{"n_drift": 0}, {"n_drift": 2}, {"n_drift": 0}])
    d = drift_frequency(rec)
    assert d["n_runs"] == 3
    assert d["runs_with_drift"] == 1
    assert d["total_drift_events"] == 2
    assert d["pct_runs_with_drift"] == 33.3


def test_empty_inputs_are_safe():
    rep = compute_forensics(pd.DataFrame(), pd.DataFrame())
    assert rep["n_orders"] == 0
    assert rep["fill_outcomes"]["fill_rate"] is None
