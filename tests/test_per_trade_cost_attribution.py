"""Per-trade cost attribution (2026-07-23).

Each sim leg is charged the cost that best fits it: its OWN realized cost if it
filled at the broker; else the average of the trades that filled in the same
tick (run); else the average for its time-of-day period; else the modeled cost.
Ledger purity is preserved — every sim trade still assumes it fills; only the
COST charged is sourced more precisely. All pure/fakes, no network.
"""

import pytest

from config.settings import settings
from src.performance import spread
from src.performance.spread import (resolve_leg_cost, resolve_trade_leg_cost,
                                    real_leg_cost_frac, set_cost_attribution,
                                    session_bucket_fine, _pct_return)


@pytest.fixture(autouse=True)
def _clean(monkeypatch):
    monkeypatch.setattr(settings, "sim_per_trade_cost_attribution", True)
    monkeypatch.setattr(settings, "sim_real_fill_min_price", 1.0)
    monkeypatch.setattr(settings, "sim_real_fill_cost_sanity_pct", 2.0)
    set_cost_attribution(None, None, None)
    yield
    set_cost_attribution(None, None, None)


# ── real per-leg cost ───────────────────────────────────────────────────────

def test_real_leg_cost_is_commission_plus_slippage():
    # BUY filled 100 @ $50.10 vs $50.00 decision, $1 commission on $5010 notional.
    frac = real_leg_cost_frac("BUY", 100, 50.0, 50.10, 1.0)
    slip = (50.10 - 50.0) / 50.0          # +0.20%
    comm = 1.0 / (100 * 50.10)            # ~0.02%
    assert frac == pytest.approx(slip + comm, abs=1e-6)


def test_real_leg_cost_rejects_outliers():
    # -7% one-way is impossible for a capped LMT → discarded (None), so the
    # caller falls through to an estimate instead of paying you to trade.
    assert real_leg_cost_frac("BUY", 100, 100.0, 93.0, 0.0) is None


def test_real_leg_cost_clamps_favorable_to_zero():
    # A mildly favorable fill (inside the band) floors at 0 — never negative cost.
    assert real_leg_cost_frac("BUY", 100, 100.0, 99.5, 0.0) == 0.0


# ── the four-tier hierarchy ─────────────────────────────────────────────────

def _leg(**kw):
    base = dict(side="BUY", price=100.0, asset_type="STOCK", session_fine="rth",
                ref="ref1", filled_qty=0, model_price=100.0, fill_price=None,
                commission=None, run=None)
    base.update(kw)
    return base


def test_tier1_real_fill_wins():
    set_cost_attribution({"R": 0.05}, {"rth": 0.09}, {"ref1": "R"})
    c = resolve_leg_cost(**_leg(filled_qty=100, fill_price=100.10, commission=1.0, run="R"))
    assert c == pytest.approx(real_leg_cost_frac("BUY", 100, 100.0, 100.10, 1.0), abs=1e-9)


def test_tier2_tick_average_when_unfilled():
    set_cost_attribution({"R": 0.05}, {"rth": 0.09}, {})
    c = resolve_leg_cost(**_leg(run="R"))          # unfilled, run R has an average
    assert c == pytest.approx(0.05)


def test_tier3_session_average_when_tick_too_thin():
    set_cost_attribution({}, {"rth": 0.09}, {})    # no run average available
    c = resolve_leg_cost(**_leg(run="R"))
    assert c == pytest.approx(0.09)


def test_tier4_model_when_nothing_installed():
    from src.performance.spread import _one_side_cost
    c = resolve_leg_cost(**_leg(run="R"))
    assert c == pytest.approx(_one_side_cost(100.0, "STOCK", "rth"))


def test_subdollar_leg_always_uses_model():
    from src.performance.spread import _one_side_cost
    set_cost_attribution({"R": 0.05}, {"rth": 0.09}, {})
    c = resolve_leg_cost(**_leg(price=0.50, run="R"))
    assert c == pytest.approx(_one_side_cost(0.50, "STOCK", "rth"))


def test_master_switch_off_is_pure_model(monkeypatch):
    from src.performance.spread import _one_side_cost
    monkeypatch.setattr(settings, "sim_per_trade_cost_attribution", False)
    set_cost_attribution({"R": 0.05}, {"rth": 0.09}, {"ref1": "R"})
    c = resolve_leg_cost(**_leg(filled_qty=100, fill_price=100.10, commission=1.0, run="R"))
    assert c == pytest.approx(_one_side_cost(100.0, "STOCK", "rth"))


# ── trade-level extraction (both engines share this) ────────────────────────

def test_trade_leg_cost_entry_uses_real_fill():
    t = {"action": "BUY", "type": "STOCK", "entry_price": 100.0, "run_id": "R",
         "broker_fill_qty": 100, "broker_fill_price": 100.10, "broker_commission": 1.0,
         "broker_client_ref": "e1"}
    c = resolve_trade_leg_cost(t, "entry", 100.0, "rth")
    assert c == pytest.approx(real_leg_cost_frac("BUY", 100, 100.0, 100.10, 1.0), abs=1e-9)


def test_trade_leg_cost_unfilled_entry_uses_its_run_average():
    set_cost_attribution({"R": 0.042}, {"rth": 0.09}, {})
    t = {"action": "BUY", "type": "STOCK", "entry_price": 100.0, "run_id": "R"}
    assert resolve_trade_leg_cost(t, "entry", 100.0, "rth") == pytest.approx(0.042)


def test_stored_session_drives_model_fallback_without_a_timestamp():
    # entry_session='extended' but no datetime: the modeled fallback must still
    # charge the EXTENDED spread, not RTH (regression for the phase-1 test).
    from src.performance.spread import _one_side_cost
    t = {"action": "BUY", "type": "STOCK", "entry_price": 100.0, "entry_session": "extended"}
    c = resolve_trade_leg_cost(t, "entry", 100.0, session_bucket_fine(None))
    assert c == pytest.approx(_one_side_cost(100.0, "STOCK", "extended"))


# ── the two return engines must agree on cost ───────────────────────────────

def test_both_engines_charge_identical_per_trade_costs():
    """_pct_return (tracker) and the daily-NAV effective prices must resolve the
    SAME per-leg cost for the same trade — the invariant the shared
    resolve_trade_leg_cost enforces."""
    from src.performance.daily_nav import _effective_entry, _effective_exit
    set_cost_attribution({"R": 0.03}, {"rth": 0.09}, {})
    t = {"action": "BUY", "type": "STOCK", "entry_price": 100.0, "exit_price": 110.0,
         "run_id": "R"}                       # unfilled both legs → tick avg 0.03
    e_cost = resolve_trade_leg_cost(t, "entry", 100.0, "rth")
    x_cost = resolve_trade_leg_cost(t, "exit", 110.0, "rth")
    # _pct_return with these costs
    r = _pct_return("BUY", 100.0, 110.0, "STOCK", entry_cost=e_cost, exit_cost=x_cost)
    # daily-NAV single-step compound from the same effective anchors
    ee = _effective_entry(100.0, "BUY", "STOCK", cost=e_cost)
    xx = _effective_exit(110.0, "BUY", "STOCK", cost=x_cost)
    r_nav = (xx - ee) / ee * 100
    assert r == pytest.approx(r_nav, abs=1e-9)
