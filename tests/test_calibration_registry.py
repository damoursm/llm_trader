"""Calibration registry + the first registry-native calibrations (2026-07-03):
the derived horizon cost hurdle and the per-session real-fill cost split
(session spread multipliers Bayesian-shrunk toward the x4/x10 priors).
"""

import pytest

from config.settings import settings


# ── registry primitives ───────────────────────────────────────────────────────

def test_registry_roundtrip_and_reset():
    from src.performance.calibration import (get_calibrations, report_calibration,
                                             reset_calibrations)
    report_calibration("alpha", value=1.23, prior=1.0, n_evidence=7, unit="x", note="n")
    report_calibration("alpha", value=1.30, prior=1.0, n_evidence=9)   # overwrite, not append
    report_calibration("beta", value=0.5)
    cals = get_calibrations()
    assert [c["name"] for c in cals] == ["alpha", "beta"]
    a = cals[0]
    assert a["value"] == pytest.approx(1.30) and a["n_evidence"] == 9
    assert cals[1]["prior"] is None
    reset_calibrations()
    assert get_calibrations() == []


def test_shrink_math():
    from src.performance.calibration import shrink
    assert shrink(4.0, 20, None, 0) == pytest.approx(4.0)          # no evidence → prior
    assert shrink(4.0, 20, 2.0, 0) == pytest.approx(4.0)
    assert shrink(4.0, 20, 2.0, 20) == pytest.approx(3.0)          # equal weight → midpoint
    assert shrink(4.0, 20, 2.0, 2000) == pytest.approx(2.0, abs=0.02)  # evidence dominates


# ── derived cost hurdle ───────────────────────────────────────────────────────

def test_effective_cost_hurdle(monkeypatch):
    from src.performance import spread
    monkeypatch.setattr(settings, "horizon_cost_hurdle_pct", 0.40)
    monkeypatch.setattr(settings, "cost_hurdle_use_calibrated", True)
    monkeypatch.setattr(settings, "cost_hurdle_safety", 1.5)
    # No calibration → static fallback.
    spread.set_real_cost_override(None)
    assert spread.effective_cost_hurdle_pct() == pytest.approx(0.40)
    # Calibrated 8 bp one-way → 2 × 0.08 × 1.5 = 0.24% round trip.
    spread.set_real_cost_override(0.0008)
    assert spread.effective_cost_hurdle_pct() == pytest.approx(0.24)
    # Sanity clamps: degenerate calibrations can't zero the gate or shut it.
    spread.set_real_cost_override(0.00001)
    assert spread.effective_cost_hurdle_pct() == pytest.approx(0.05)
    spread.set_real_cost_override(0.02)
    assert spread.effective_cost_hurdle_pct() == pytest.approx(2.0)
    # Feature off → static, even with a calibration installed.
    monkeypatch.setattr(settings, "cost_hurdle_use_calibrated", False)
    spread.set_real_cost_override(0.0008)
    assert spread.effective_cost_hurdle_pct() == pytest.approx(0.40)


# ── session-aware real-cost override ─────────────────────────────────────────

def test_one_side_cost_uses_session_split(monkeypatch):
    from src.performance import spread
    monkeypatch.setattr(settings, "sim_real_fill_min_price", 1.0)
    spread.set_real_cost_override(
        0.001, by_session={"rth": 0.0008, "extended": 0.0032, "overnight": 0.008})
    assert spread._one_side_cost(100.0, "STOCK", "rth") == pytest.approx(0.0008)
    assert spread._one_side_cost(100.0, "STOCK", "extended") == pytest.approx(0.0032)
    assert spread._one_side_cost(100.0, "STOCK", "overnight") == pytest.approx(0.008)
    assert spread._one_side_cost(100.0, "STOCK", None) == pytest.approx(0.0008)  # None → rth
    # Sub-$1 legs still take the modeled tiered cost, never the calibration.
    assert spread._one_side_cost(0.05, "STOCK", "rth") == pytest.approx(0.0250)
    # Flat-only install behaves as before (session ignored).
    spread.set_real_cost_override(0.001)
    assert spread.get_real_cost_session_overrides() is None
    assert spread._one_side_cost(100.0, "STOCK", "overnight") == pytest.approx(0.001)
    # Clearing wipes both.
    spread.set_real_cost_override(None)
    assert spread.get_real_cost_override() is None
    assert spread.get_real_cost_session_overrides() is None


# ── calibrate_sim_costs: per-session split + registry reporting ──────────────

def _leg(session_iso, cost_pct, qty=10, model=100.0):
    """A filled LMT leg whose all-in one-way cost is exactly ``cost_pct``
    (built as commission-only: fill at model → zero slippage)."""
    fill = model
    comm = cost_pct / 100.0 * qty * fill
    return {"client_ref": f"r{session_iso}{cost_pct}{qty}", "side": "BUY",
            "filled_qty": qty, "model_price": model, "fill_price": fill,
            "commission": comm, "submitted_at": session_iso}


RTH_TS = "2026-07-02T15:00:00+00:00"        # 11:00 ET
EXT_TS = "2026-07-02T21:00:00+00:00"        # 17:00 ET after-hours


def test_calibrate_sim_costs_session_split(monkeypatch):
    from src.performance import spread, tracker
    monkeypatch.setattr(settings, "sim_use_real_fill_costs", True)
    monkeypatch.setattr(settings, "sim_real_fill_costs_min_legs", 10)
    monkeypatch.setattr(settings, "session_spread_calibration_enabled", True)
    monkeypatch.setattr(settings, "session_cost_min_legs", 5)
    monkeypatch.setattr(settings, "session_spread_prior_n", 20)
    monkeypatch.setattr(settings, "spread_extended_multiplier", 4.0)
    monkeypatch.setattr(settings, "spread_overnight_multiplier", 10.0)
    # 10 RTH legs at 0.10% + 6 extended legs at 0.20% (obs ratio = 2.0).
    legs = ([_leg(RTH_TS, 0.10, qty=10 + i) for i in range(10)]
            + [_leg(EXT_TS, 0.20, qty=30 + i) for i in range(6)])
    monkeypatch.setattr(tracker.repo, "fetch_filled_lmt_legs", lambda: legs)

    frac = tracker.calibrate_sim_costs()
    assert frac == pytest.approx((0.10 * 10 + 0.20 * 6) / 16 / 100.0, rel=1e-6)

    by = spread.get_real_cost_session_overrides()
    assert by["rth"] == pytest.approx(0.001, rel=1e-6)              # measured RTH mean
    # Extended: obs 2.0 over 6 legs, shrunk toward prior 4.0 with prior_n 20:
    # (20×4 + 6×2)/26 = 3.5385 → cost = rth × mult.
    assert by["extended"] == pytest.approx(0.001 * (20 * 4 + 6 * 2) / 26, rel=1e-4)
    # Overnight: zero legs → the ×10 prior holds exactly.
    assert by["overnight"] == pytest.approx(0.001 * 10.0, rel=1e-6)

    from src.performance.calibration import get_calibrations
    cals = {c["name"]: c for c in get_calibrations()}
    assert cals["spread_mult_extended"]["value"] == pytest.approx(3.5385, abs=1e-3)
    assert cals["spread_mult_extended"]["n_evidence"] == 6
    assert cals["spread_mult_overnight"]["value"] == pytest.approx(10.0)
    assert cals["spread_mult_overnight"]["n_evidence"] == 0
    assert cals["sim_one_way_cost"]["n_evidence"] == 16
    assert "cost_hurdle" in cals


def test_calibrate_sim_costs_split_disabled(monkeypatch):
    from src.performance import spread, tracker
    monkeypatch.setattr(settings, "sim_use_real_fill_costs", True)
    monkeypatch.setattr(settings, "sim_real_fill_costs_min_legs", 10)
    monkeypatch.setattr(settings, "session_spread_calibration_enabled", False)
    legs = [_leg(RTH_TS, 0.10, qty=10 + i) for i in range(12)]
    monkeypatch.setattr(tracker.repo, "fetch_filled_lmt_legs", lambda: legs)
    frac = tracker.calibrate_sim_costs()
    assert frac is not None
    assert spread.get_real_cost_session_overrides() is None          # flat-only, legacy
