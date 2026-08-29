"""Liquidity-class cost buckets (2026-08-25): unfilled legs priced from the
realized costs of fills that TRADE like them, the parametric model demoted to
the cold-start prior. Each tier of the resolver hierarchy is pinned — a wrong
tier order silently reprices the whole ledger."""

from __future__ import annotations

import pytest

from config.settings import settings
from src.performance import spread


@pytest.fixture(autouse=True)
def _clean():
    spread.set_cost_attribution(None, None, None)
    spread.set_real_cost_override(None)
    yield
    spread.set_cost_attribution(None, None, None)
    spread.set_real_cost_override(None)


def test_band_edges():
    assert spread.price_band(5.0) == 0
    assert spread.price_band(20.0) == 1          # edge belongs to the upper band
    assert spread.price_band(99.9) == 1
    assert spread.price_band(100.0) == 2
    assert spread.price_band(None) is None and spread.price_band(-1) is None
    assert spread.adv_band(5e6) == 0
    assert spread.adv_band(20e6) == 1
    assert spread.adv_band(None) is None


def _resolve(price=50.0, adv=50e6, run="r1", fine="rth"):
    return spread.resolve_leg_cost(
        side="BUY", price=price, asset_type="STOCK", session_fine=fine,
        ref=None, filled_qty=0, model_price=None, fill_price=None,
        commission=None, run=run, adv=adv)


def test_hierarchy_tick_bucket_beats_tick_and_session(monkeypatch):
    monkeypatch.setattr(settings, "sim_per_trade_cost_attribution", True)
    pb, ab = spread.price_band(50.0), spread.adv_band(50e6)
    spread.set_cost_attribution(
        tick_costs={"r1": 0.0030}, session_costs={"rth": 0.0040}, ref_to_run={},
        tick_bucket={("r1", pb, ab): 0.0010},
        session_bucket={("rth", pb, ab): 0.0020})
    assert _resolve() == pytest.approx(0.0010)               # tick-bucket first
    spread.set_cost_attribution(
        tick_costs={"r1": 0.0030}, session_costs={"rth": 0.0040}, ref_to_run={},
        tick_bucket={}, session_bucket={("rth", pb, ab): 0.0020})
    assert _resolve() == pytest.approx(0.0030)               # then plain tick
    assert _resolve(run="other") == pytest.approx(0.0020)    # then session-bucket
    spread.set_cost_attribution(
        tick_costs={}, session_costs={"rth": 0.0040}, ref_to_run={},
        tick_bucket={}, session_bucket={})
    assert _resolve(run="other") == pytest.approx(0.0040)    # then plain session


def test_unknown_adv_skips_bucket_tiers(monkeypatch):
    monkeypatch.setattr(settings, "sim_per_trade_cost_attribution", True)
    pb, ab = spread.price_band(50.0), spread.adv_band(50e6)
    spread.set_cost_attribution(
        tick_costs={"r1": 0.0030}, session_costs={}, ref_to_run={},
        tick_bucket={("r1", pb, ab): 0.0010}, session_bucket={})
    assert _resolve(adv=None) == pytest.approx(0.0030)       # no class -> no guess


def test_cold_start_reaches_the_model(monkeypatch):
    # No fills anywhere: no overrides, no tables -> the hand-built formula is
    # the last resort (fresh-install prior), not an error.
    monkeypatch.setattr(settings, "sim_per_trade_cost_attribution", True)
    got = _resolve(run=None)
    modeled = spread._one_side_cost(50.0, "STOCK", "rth")
    assert got == pytest.approx(modeled) and got > 0


def test_calibrate_builds_shrunk_clamped_buckets(monkeypatch):
    """End-to-end through tracker.calibrate_sim_costs with synthetic fills:
    the session-class bucket is shrunk toward the session mean and clamped."""
    from src.performance import tracker
    monkeypatch.setattr(settings, "sim_use_real_fill_costs", True)
    monkeypatch.setattr(settings, "sim_per_trade_cost_attribution", True)
    monkeypatch.setattr(settings, "sim_cost_bucket_min_legs", 2)
    monkeypatch.setattr(settings, "sim_cost_bucket_prior_n", 0)   # no shrink -> raw mean
    monkeypatch.setattr(spread, "adv_of", lambda tk: 50e6)
    legs = []
    # 10 cheap-liquid legs at ~0.10% and 4 expensive-liquid legs at ~0.40%,
    # all RTH (submitted_at date-only -> 'rth').
    for i in range(10):
        legs.append({"client_ref": f"a{i}", "run_id": "rA", "ticker": "CHEAP",
                     "side": "BUY", "filled_qty": 10, "model_price": 50.0,
                     "fill_price": 50.05, "commission": 0.0,
                     "submitted_at": "2026-08-20"})
    for i in range(4):
        legs.append({"client_ref": f"b{i}", "run_id": "rB", "ticker": "EXP",
                     "side": "BUY", "filled_qty": 10, "model_price": 200.0,
                     "fill_price": 200.8, "commission": 0.0,
                     "submitted_at": "2026-08-20"})
    monkeypatch.setattr(tracker.repo, "fetch_filled_lmt_legs", lambda: legs)
    tracker.calibrate_sim_costs()
    pb_cheap, pb_exp = spread.price_band(50.05), spread.price_band(200.8)
    ab = spread.adv_band(50e6)
    sb = spread._SESSION_BUCKET_COST
    assert ("rth", pb_cheap, ab) in sb and ("rth", pb_exp, ab) in sb
    assert sb[("rth", pb_cheap, ab)] == pytest.approx(0.001, rel=0.05)
    assert sb[("rth", pb_exp, ab)] == pytest.approx(0.004, rel=0.05)
    # clamp: bucket can never leave [0.25x, 4x] of the session mean
    base = spread._SESSION_COST_FRAC["rth"]
    for v in sb.values():
        assert base * 0.25 - 1e-12 <= v <= base * 4.0 + 1e-12
    # and an unfilled expensive leg now prices like its class, not the blend
    got = _resolve(price=200.0, adv=50e6, run="nope")
    assert got == pytest.approx(sb[("rth", pb_exp, ab)])
