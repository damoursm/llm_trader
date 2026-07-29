"""Calibrate the simulated cost model to REAL IBKR fills (2026-06-13).

When enough real fills exist, the sim charges the measured average all-in
one-way cost (commission + execution vs decision price) on every leg instead
of the modeled half-spread + commission — so simulated returns reflect what
execution actually costs. Gated by a minimum-leg floor and clamped ≥ 0.
"""

import pytest

from config.settings import settings
from src.performance import spread
from src.performance.broker_view import real_one_way_cost_fraction
from src.performance.spread import _one_side_cost, _pct_return


def _leg(side="BUY", filled_qty=10, model_price=100.0, fill_price=100.0, commission=1.0):
    """A broker_orders-shaped LMT fill row (what fetch_filled_lmt_legs returns)."""
    return {"side": side, "filled_qty": filled_qty, "model_price": model_price,
            "fill_price": fill_price, "commission": commission}


# ── the override short-circuits the model ──────────────────────────────────

def test_one_side_cost_returns_override_flat_ignoring_price_and_session():
    spread.set_real_cost_override(0.004)   # 0.4% flat
    try:
        assert _one_side_cost(5.0, "STOCK") == pytest.approx(0.004)
        assert _one_side_cost(500.0, "ETF", "extended") == pytest.approx(0.004)
    finally:
        spread.set_real_cost_override(None)


def test_override_clamped_non_negative():
    spread.set_real_cost_override(-0.01)
    try:
        assert _one_side_cost(100.0) == 0.0
    finally:
        spread.set_real_cost_override(None)


def test_override_flows_into_pct_return():
    # No override → spread-only model (conftest pins commission 'none').
    base = _pct_return("BUY", 100.0, 110.0, "STOCK")
    spread.set_real_cost_override(0.01)    # 1% per leg
    try:
        calibrated = _pct_return("BUY", 100.0, 110.0, "STOCK")
    finally:
        spread.set_real_cost_override(None)
    # eff_entry = 100×1.01 = 101; eff_exit = 110×0.99 = 108.9 → (108.9-101)/101
    assert calibrated == pytest.approx((108.9 - 101.0) / 101.0 * 100, abs=1e-6)
    assert calibrated < base               # 1% legs cost more than the spread-only model


def test_none_override_uses_the_model():
    spread.set_real_cost_override(None)
    assert _one_side_cost(100.0, "STOCK") == pytest.approx(spread._dynamic_half_spread(100.0, "STOCK"))


# ── deriving the fraction from real LMT legs, with the min-leg floor ────────

def test_real_fraction_none_below_min_legs():
    assert real_one_way_cost_fraction([_leg(), _leg()], min_legs=10) is None


def test_real_fraction_averages_legs_above_floor():
    # 10 legs, each commission only (fill == model) = 1/(10×100)×100 = 0.1%
    legs = [_leg() for _ in range(10)]
    assert real_one_way_cost_fraction(legs, min_legs=10) == pytest.approx(0.001, abs=1e-5)


def test_real_fraction_includes_execution_drift():
    # BUY legs filled 0.5% adverse on top of ~0.1% commission → ~0.5995%.
    legs = [_leg(side="BUY", model_price=100.0, fill_price=100.5) for _ in range(10)]
    frac = real_one_way_cost_fraction(legs, min_legs=10)
    expected = (0.5 + 1.0 / (10 * 100.5) * 100) / 100.0
    assert frac == pytest.approx(expected, abs=1e-5)


# ── end-to-end through the tracker (reads repo.fetch_filled_lmt_legs) ───────

def test_calibrate_sim_costs_sets_and_clears(monkeypatch):
    from src.performance import tracker
    monkeypatch.setattr(settings, "sim_use_real_fill_costs", True)
    monkeypatch.setattr(settings, "sim_real_fill_costs_min_legs", 10)
    legs = [_leg() for _ in range(10)]                     # 10 LMT legs @ 0.1%
    monkeypatch.setattr(tracker.repo, "fetch_filled_lmt_legs", lambda: legs)
    frac = tracker.calibrate_sim_costs()
    assert frac == pytest.approx(0.001, abs=1e-5)
    assert spread.get_real_cost_override() == pytest.approx(0.001, abs=1e-5)
    # disabled → cleared
    monkeypatch.setattr(settings, "sim_use_real_fill_costs", False)
    assert tracker.calibrate_sim_costs() is None
    assert spread.get_real_cost_override() is None


def test_calibrate_below_floor_falls_back_to_model(monkeypatch):
    from src.performance import tracker
    monkeypatch.setattr(settings, "sim_use_real_fill_costs", True)
    monkeypatch.setattr(settings, "sim_real_fill_costs_min_legs", 10)
    monkeypatch.setattr(tracker.repo, "fetch_filled_lmt_legs", lambda: [_leg(), _leg()])
    assert tracker.calibrate_sim_costs() is None           # too few LMT legs
    assert spread.get_real_cost_override() is None          # model in effect


# ── session-split outlier band (2026-07-23 regression) ─────────────────────
#
# The per-SESSION split (calibrate_sim_costs) averaged raw per-leg costs with
# no outlier band, so a handful of corrupt all-negative legs (stale decision
# prices) dragged each session mean below zero → clamped to 0 → _one_side_cost
# returned ZERO cost for every leg that hit the session path (checked before
# the flat override). That silently zeroed the sim's cost basis.

def test_session_split_excludes_outlier_legs(monkeypatch, tmp_path):
    import src.db.repo as repo
    from src.performance import tracker

    good = [_leg(fill_price=100.20, commission=0.0) for _ in range(12)]  # +0.20% one-way, RTH
    bad = _leg(fill_price=93.0, commission=0.0)                          # -7% one-way, corrupt
    for leg in good + [bad]:
        leg["submitted_at"] = "2026-07-10T14:00:00+00:00"  # all RTH
    monkeypatch.setattr(repo, "fetch_filled_lmt_legs", lambda: good + [bad])
    monkeypatch.setattr(settings, "sim_use_real_fill_costs", True)
    monkeypatch.setattr(settings, "sim_real_fill_costs_min_legs", 5)
    monkeypatch.setattr(settings, "session_spread_calibration_enabled", True)
    monkeypatch.setattr(settings, "session_cost_min_legs", 5)
    monkeypatch.setattr(settings, "sim_real_fill_cost_sanity_pct", 2.0)

    tracker.calibrate_sim_costs([])
    try:
        by_sess = spread.get_real_cost_session_overrides()
        assert by_sess is not None
        # RTH must be the ~0.20% of the good legs, NOT dragged to 0 by the -7% leg.
        assert by_sess["rth"] == pytest.approx(0.0020, abs=3e-4)
        assert _one_side_cost(100.0, "STOCK", "rth") > 0.001    # not zero
    finally:
        spread.set_real_cost_override(None)
        spread.set_cost_attribution(None, None, None)
