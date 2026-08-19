"""Limits must be able to CROSS the book, not just sit near the mid (2026-08-17).

The measured problem: `get_market_price` returns a LAST/MID price — it never
returns the ask — so a limit priced at `mid x (1 + cap)` is marketable only when
the cap happens to exceed the HALF-SPREAD. That holds in RTH and fails badly
off-hours, where a thin book quotes hundreds of bp:

    rth        56.6% fill      afterhours 13.9%
    premarket  10.2%           overnight   4.8%

The trap in the evidence: the fills that DID happen consumed only a small slice
of their cap (p50 ~8-20 bp of an 80-150 bp allowance), which reads as "the cap is
generous". That was survivorship — the wide-spread names never filled, so they
never entered the statistic at all. The non-fills are invisible in a distribution
conditioned on filling.

Fix: when a real two-sided quote exists, widen the cap to cover
`half_spread x broker_spread_cap_mult` and push the limit to the far side of the
book when the cap covers it. Reaching the far side costs nothing — a marketable
limit executes at the touch — so this changes WHETHER an order can trade, not
what it pays.
"""

import pytest

from config import settings
from src.broker.base import Quote
from src.broker.reconcile import (_effective_cap_bps, _limit_price_for,
                                  _session_cap_bps)


@pytest.fixture(autouse=True)
def _spread_aware(monkeypatch):
    monkeypatch.setattr(settings, "broker_spread_aware_limits", True)
    monkeypatch.setattr(settings, "broker_spread_cap_mult", 1.5)
    monkeypatch.setattr(settings, "broker_limit_cap_bps", 45.0)
    monkeypatch.setattr(settings, "broker_limit_cap_bps_max", 400.0)


def _q(bid, ask):
    return Quote(ticker="AAA", bid=bid, ask=ask)


# ── the cap widens only when the spread demands it ──────────────────────────

def test_tight_spread_leaves_the_configured_cap_alone():
    """A 10 bp book needs 5 bp to cross; the 45 bp cap already covers it, so
    nothing widens — the cap must never be RAISED without cause."""
    assert _effective_cap_bps(False, _q(99.95, 100.05)) == 45.0


def test_wide_spread_widens_the_cap_enough_to_cross():
    """A 200 bp book needs 100 bp to reach the far side; 45 could never fill."""
    eff = _effective_cap_bps(False, _q(99.0, 101.0))
    assert eff == pytest.approx(150.0, abs=1.0)   # 100 bp half-spread x 1.5


def test_the_cap_is_never_lowered_below_the_configured_value():
    """Widening is one-directional: a tight book must not shrink our headroom."""
    assert _effective_cap_bps(False, _q(99.999, 100.001)) == 45.0


def test_absolute_ceiling_bounds_a_garbage_quote():
    """A stale one-tick 4am book must not authorise an unbounded price."""
    eff = _effective_cap_bps(False, _q(50.0, 150.0))   # 10,000 bp spread
    assert eff == 400.0


def test_off_rth_does_not_widen_by_default(monkeypatch):
    """Measured, not cautious. Widening only creates fills whose half-spread
    exceeded the old cap, so the newly-fillable off-hours cohort pays >=160 bp
    round trip (extended) / >=300 bp (overnight). Off-hours gross returns are an
    order of magnitude below that — real trades +1.09%/-0.46% mean, the
    simulated panel -0.09%/-0.14% over 15k rows — so those fills book a measured
    loss whatever the signal says."""
    monkeypatch.setattr(settings, "broker_spread_aware_off_rth", False)
    # Compared against _session_cap_bps rather than a literal: which off-RTH cap
    # applies (extended vs overnight) is resolved from the CLOCK, so a hardcoded
    # number makes the test pass or fail by time of day.
    assert _effective_cap_bps(True, _q(97.0, 103.0)) == _session_cap_bps(True)


def test_off_rth_widening_can_be_re_enabled(monkeypatch):
    """Kept as one flag, not deleted: the verdict is about the CURRENT measured
    returns, and it is re-testable if those change."""
    monkeypatch.setattr(settings, "broker_spread_aware_off_rth", True)
    assert _effective_cap_bps(True, _q(97.0, 103.0)) > _session_cap_bps(True)


def test_rth_still_widens_while_off_rth_does_not(monkeypatch):
    """The asymmetry is the point: RTH's cap measured BINDING (fills piling at
    p90 19.1 of a 20 bp cap) and its spreads are tight, so widening there is
    cheap and useful."""
    monkeypatch.setattr(settings, "broker_spread_aware_off_rth", False)
    assert _effective_cap_bps(False, _q(99.0, 101.0)) > 45.0     # RTH widens
    assert _effective_cap_bps(True, _q(99.0, 101.0)) == _session_cap_bps(True)


def test_no_quote_keeps_the_old_behaviour_exactly():
    assert _effective_cap_bps(False, None) == 45.0


def test_disabled_flag_keeps_the_old_behaviour(monkeypatch):
    monkeypatch.setattr(settings, "broker_spread_aware_limits", False)
    assert _effective_cap_bps(False, _q(99.0, 101.0)) == 45.0


# ── the limit reaches the far side ──────────────────────────────────────────

def test_buy_limit_reaches_the_ask_in_a_wide_book():
    """THE fix. Priced off the mid at 45 bp the limit lands ~100.45 and can
    never trade against a 101.00 ask; it must reach the ask."""
    old = _limit_price_for("BUY", 100.0, False, None)
    new = _limit_price_for("BUY", 100.0, False, _q(99.0, 101.0))
    assert old < 101.0, "fixture no longer reproduces the unfillable case"
    assert new >= 101.0, f"limit {new} still cannot cross a 101.00 ask"


def test_sell_limit_reaches_the_bid_in_a_wide_book():
    old = _limit_price_for("SELL", 100.0, False, None)
    new = _limit_price_for("SELL", 100.0, False, _q(99.0, 101.0))
    assert old > 99.0
    assert new <= 99.0, f"limit {new} still cannot cross a 99.00 bid"


def test_limit_never_exceeds_the_effective_cap():
    """Cost discipline survives: an order the cap cannot reach is one we have
    decided not to pay for, so the ceiling still binds."""
    monkeypatch_ceiling = 50.0
    settings.broker_limit_cap_bps_max = monkeypatch_ceiling
    try:
        # 1,000 bp book; ceiling allows only 50 bp => limit stops at the cap.
        lim = _limit_price_for("BUY", 100.0, False, _q(95.0, 105.0))
        assert lim <= 100.0 * (1 + monkeypatch_ceiling / 10_000.0) + 0.01
    finally:
        settings.broker_limit_cap_bps_max = 400.0


def test_a_one_sided_quote_is_ignored():
    """A book quoting only one side cannot tell us what crossing costs."""
    assert _effective_cap_bps(False, Quote("AAA", bid=99.0, ask=None)) == 45.0
    assert _limit_price_for("BUY", 100.0, False, Quote("AAA", bid=99.0, ask=None)) \
        == _limit_price_for("BUY", 100.0, False, None)


def test_unusable_model_price_still_returns_none():
    assert _limit_price_for("BUY", 0.0, False, _q(99.0, 101.0)) is None
    assert _limit_price_for("BUY", None, False, _q(99.0, 101.0)) is None


# ── Quote itself ────────────────────────────────────────────────────────────

def test_quote_helpers():
    q = _q(99.0, 101.0)
    assert q.mid == 100.0
    assert q.spread_bps == pytest.approx(200.0)
    assert q.opposite("BUY") == 101.0
    assert q.opposite("SELL") == 99.0
    # incomplete books report nothing rather than guessing
    assert Quote("AAA", bid=None, ask=101.0).mid is None
    assert Quote("AAA", bid=None, ask=101.0).spread_bps is None
    assert Quote("AAA", bid=0.0, ask=0.0).opposite("BUY") is None
