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


# ── AUDIT151 (2026-10-19): a leg with no usable fill of its own is costed at its OWN quote at submit ──

def test_a_legs_own_quote_prices_it_with_ibkrs_fees_on_its_shares(monkeypatch):
    monkeypatch.setattr(settings, "commission_model", "ibkr_fixed")
    # a short entry (a SELL) of 33 shares at 12.50, the book 12.40 x 12.60: 80 bp half-spread + $1 + SEC + TAF
    got = spread.quote_leg_cost_frac("SELL", 12.50, 12.40, 12.60, 33)
    value = 33 * 12.50
    want = 0.2 / 25.0 + (1.0 + spread.SEC_FEE_RATE * value + spread.TAF_PER_SHARE * 33) / value
    assert got == pytest.approx(want)
    assert spread.quote_leg_cost_frac("BUY", 12.50, 12.40, 12.60, 33) == pytest.approx(0.2 / 25.0 + 1.0 / value)
    assert spread.quote_leg_cost_frac("BUY", 10.0, 9.0, 11.5, 100) == pytest.approx(0.04 + 1.0 / 1000.0)  # capped
    for bad in ((12.5, 0.0, 12.6, 33), (12.5, 12.6, 12.4, 33), (12.5, 12.4, 12.6, 0), (12.5, None, 12.6, 33)):
        assert spread.quote_leg_cost_frac("BUY", *bad) is None
    monkeypatch.setattr(settings, "commission_model", "none")
    assert spread.quote_leg_cost_frac("SELL", 12.50, 12.40, 12.60, 33) == pytest.approx(0.008)   # spread only


def test_the_quote_tier_sits_between_the_own_fill_and_the_bucket_averages(monkeypatch):
    monkeypatch.setattr(settings, "sim_per_trade_cost_attribution", True)
    monkeypatch.setattr(settings, "commission_model", "none")
    pb, ab = spread.price_band(50.0), spread.adv_band(50e6)
    spread.set_cost_attribution(tick_costs={"r1": 0.0030}, session_costs={"rth": 0.0040}, ref_to_run={},
                                tick_bucket={("r1", pb, ab): 0.0010}, session_bucket={})
    kw = dict(side="BUY", price=50.0, asset_type="STOCK", session_fine="rth", ref=None, run="r1", adv=50e6,
              commission=None, quote_bid=49.75, quote_ask=50.25, shares=20)
    assert spread.resolve_leg_cost(filled_qty=0, model_price=None, fill_price=None, **kw) == pytest.approx(0.005)
    own = spread.resolve_leg_cost(filled_qty=20, model_price=50.0, fill_price=50.10, **kw)
    assert own == pytest.approx(0.002)                         # its own fill first: 10 bp of slippage
    kw.update(quote_bid=None, quote_ask=None)
    assert spread.resolve_leg_cost(filled_qty=0, model_price=None, fill_price=None, **kw) == pytest.approx(0.0010)


def test_a_trades_unfilled_exit_takes_its_exit_quote_and_its_entry_its_own(monkeypatch):
    monkeypatch.setattr(settings, "sim_per_trade_cost_attribution", True)
    monkeypatch.setattr(settings, "commission_model", "none")
    t = {"ticker": "AEHL", "action": "SELL", "type": "STOCK", "entry_price": 11.975, "exit_price": 11.51,
         "sel_account_shares": 29, "broker_requested_qty": 29, "broker_fill_qty": 0,
         "broker_bid_at_submit": 11.90, "broker_ask_at_submit": 12.05,
         "broker_exit_requested_qty": 29, "broker_exit_bid_at_submit": 11.10, "broker_exit_ask_at_submit": 11.88}
    assert spread.resolve_trade_leg_cost(t, "entry", 11.975, "rth") == pytest.approx(0.15 / 23.95)
    assert spread.resolve_trade_leg_cost(t, "exit", 11.51, "rth") == pytest.approx(0.78 / 22.98)


def test_the_broker_stamps_the_book_it_priced_the_order_on(monkeypatch):
    import time
    from types import SimpleNamespace
    from src.broker import reconcile as rec
    res = SimpleNamespace(order_id="1", side="SELL", requested_qty=29, filled_qty=0, avg_fill_price=None,
                          commission=None, status="Submitted", submitted_at="x", client_ref="ref-1")
    rec._LAST_QUOTE.clear()
    rec._LAST_QUOTE["AEHL"] = (11.90, 12.05, time.time())
    t = {"ticker": "AEHL", "action": "SELL"}
    rec._apply_entry_result(t, res)
    assert (t["broker_bid_at_submit"], t["broker_ask_at_submit"]) == (11.90, 12.05)
    rec._LAST_QUOTE["AEHL"] = (11.10, 11.88, time.time())
    rec._apply_exit_result(t, res)
    assert (t["broker_exit_bid_at_submit"], t["broker_exit_ask_at_submit"]) == (11.10, 11.88)
    rec._LAST_QUOTE["XYZ"] = (5.0, 5.1, time.time() - 3600)      # a stale book never stamps
    u = {"ticker": "XYZ", "action": "BUY"}
    rec._apply_entry_result(u, res)
    assert "broker_bid_at_submit" not in u
    rec._LAST_QUOTE.clear()
