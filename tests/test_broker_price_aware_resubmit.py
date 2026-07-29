"""Price-aware next-tick resubmit for a previously-unfilled ENTRY (ask 2, 2026-07-07).

An entry that didn't fill is re-decided on the next tick: still-wanted AND the
price hasn't run away → chase; decayed but price as-good-or-better than the
decision → resubmit at the better price; price adverse of the decision by more
than broker_resubmit_max_adverse_bps → HOLD, even if the signal still fires
(2026-07-23: that ceiling used to apply only after the signal decayed, so a
still-actionable name was chased at any price). Exits are never gated.

Also covers entry SIZING against the held position: an order asks for the GAP
to target, so a repeat submission can never multiply the position.

Pure-logic table plus integration checks through reconcile.sync(); all fakes,
no network.
"""

import pytest

from config.settings import settings
from src.broker.base import AccountSnapshot, OrderResult
from src.broker.reconcile import _resubmit_decision


# ── decision table (pure) ───────────────────────────────────────────────────

def _t(action, entry, price):
    return {"ticker": "X", "action": action, "entry_price": entry, "current_price": price}


def test_still_actionable_same_direction_submits_while_price_is_sane():
    # Inside broker_resubmit_max_adverse_bps (default 100 bp) the chase is on.
    assert _resubmit_decision(_t("BUY", 100, 100.5), {"X": "BUY"}) == "submit"
    assert _resubmit_decision(_t("SELL", 100, 99.5), {"X": "SELL"}) == "submit"
    # A favorable move is always fine, however far.
    assert _resubmit_decision(_t("BUY", 100, 90), {"X": "BUY"}) == "submit"
    assert _resubmit_decision(_t("SELL", 100, 110), {"X": "SELL"}) == "submit"


def test_still_actionable_is_NOT_chased_once_price_ran_away():
    """The decision (and its sizing) was made at entry_price — past the adverse
    ceiling the modeled edge is gone, so a still-firing signal is not enough."""
    assert _resubmit_decision(_t("BUY", 100, 110), {"X": "BUY"}) == "skip"
    assert _resubmit_decision(_t("SELL", 100, 90), {"X": "SELL"}) == "skip"


def test_adverse_ceiling_is_configurable(monkeypatch):
    t = _t("BUY", 100, 102)                            # 200 bp adverse
    monkeypatch.setattr(settings, "broker_resubmit_max_adverse_bps", 300.0)
    assert _resubmit_decision(t, {"X": "BUY"}) == "submit"
    monkeypatch.setattr(settings, "broker_resubmit_max_adverse_bps", 100.0)
    assert _resubmit_decision(t, {"X": "BUY"}) == "skip"
    monkeypatch.setattr(settings, "broker_resubmit_max_adverse_bps", 0.0)   # disabled
    assert _resubmit_decision(t, {"X": "BUY"}) == "submit"


def test_flipped_to_opposite_side_skips_even_on_a_better_price():
    # BUY entry, price dropped (looks "favorable" for a long) BUT the signal is
    # now SELL — never buy into a bearish flip.
    assert _resubmit_decision(_t("BUY", 100, 95), {"X": "SELL"}) == "skip"
    assert _resubmit_decision(_t("SELL", 100, 105), {"X": "BUY"}) == "skip"


def test_decayed_long_resubmits_only_at_equal_or_better_price():
    assert _resubmit_decision(_t("BUY", 100, 100), {}) == "submit"   # equal is ok
    assert _resubmit_decision(_t("BUY", 100, 98), {}) == "submit"    # cheaper = better
    assert _resubmit_decision(_t("BUY", 100, 101), {}) == "skip"     # more expensive = hold


def test_decayed_short_resubmits_only_at_equal_or_better_price():
    assert _resubmit_decision(_t("SELL", 100, 100), {}) == "submit"  # equal is ok
    assert _resubmit_decision(_t("SELL", 100, 102), {}) == "submit"  # higher = better for a short
    assert _resubmit_decision(_t("SELL", 100, 99), {}) == "skip"     # lower = hold


def test_missing_prices_default_to_submit_not_strand():
    assert _resubmit_decision(_t("BUY", 0, 0), {}) == "submit"
    assert _resubmit_decision({"ticker": "X", "action": "BUY"}, {}) == "submit"


# ── integration: the entry pass actually holds vs. resubmits ─────────────────

class _FakeBroker:
    name = "fake"

    def __init__(self):
        self.requests = []
        self.cancelled = []

    def connect(self):
        return True

    def is_connected(self):
        return True

    def get_account(self):
        return AccountSnapshot(equity=100000.0, cash=100000.0, buying_power=100000.0,
                               account_id="DU", currency="USD")

    def get_positions(self):
        return []

    def get_fills(self):
        return []

    def get_open_orders(self):
        return []

    def cancel_order(self, ref):
        self.cancelled.append(ref)
        return True

    def submit_order(self, req):
        self.requests.append(req)
        return OrderResult(ok=True, ticker=req.ticker, side=req.side,
                           requested_qty=req.quantity, filled_qty=0,
                           order_id=str(len(self.requests)), client_ref=req.client_ref,
                           status="Submitted")


@pytest.fixture(autouse=True)
def _env(monkeypatch):
    from src.performance import market_calendar
    from src.broker import reconcile
    monkeypatch.setattr(reconcile.time, "sleep", lambda s: None)
    monkeypatch.setattr(settings, "broker_mode", "ibkr_paper")
    monkeypatch.setattr(settings, "broker_order_type", "LMT")
    monkeypatch.setattr(settings, "broker_limit_cap_bps", 20.0)
    monkeypatch.setattr(settings, "broker_base_notional_ccy", "USD")
    monkeypatch.setattr(settings, "broker_base_notional", 1000.0)
    monkeypatch.setattr(settings, "broker_sizing_mode", "notional")
    monkeypatch.setattr(settings, "broker_submit_retries", 0)
    monkeypatch.setattr(settings, "broker_connect_retries", 0)
    monkeypatch.setattr(settings, "broker_settle_seconds", 0)   # isolate the entry decision
    monkeypatch.setattr(settings, "broker_price_aware_resubmit", True)
    monkeypatch.setattr(settings, "broker_drift_action", "report")
    monkeypatch.setattr(market_calendar, "current_session", lambda now=None: "rth")


def _sync(monkeypatch, broker, trades, actionable):
    from src.broker import reconcile
    monkeypatch.setattr(reconcile, "get_broker", lambda: broker)
    monkeypatch.setattr(reconcile, "_live_price", lambda t: 100.0)
    return reconcile.sync(run_id="r1", trades=trades, actionable_by_ticker=actionable)


def _killed_entry(action="BUY", entry=100.0, price=100.0):
    # a previously-unfilled entry the settle pass killed last tick
    return {"ticker": "TEST", "type": "STOCK", "action": action, "status": "OPEN",
            "entry_price": entry, "current_price": price,
            "position_size_multiplier": 1.0, "recommendation_id": "abc", "run_id": "r1",
            "broker_status": "UNFILLED_KILLED", "broker_order_id": None,
            "broker_resubmit_n": 1}


def test_decayed_adverse_entry_is_held_not_chased(monkeypatch):
    broker = _FakeBroker()
    trade = _killed_entry(action="BUY", entry=100.0, price=105.0)   # adverse for a long
    _sync(monkeypatch, broker, [trade], actionable={})             # no longer actionable
    assert broker.requests == []                                    # NOT chased
    assert trade["broker_status"] == "RESUBMIT_HELD_ADVERSE"


def test_decayed_favorable_entry_is_resubmitted(monkeypatch):
    broker = _FakeBroker()
    trade = _killed_entry(action="BUY", entry=100.0, price=97.0)    # cheaper than decision
    _sync(monkeypatch, broker, [trade], actionable={})
    assert len(broker.requests) == 1                                # resubmitted at the better price
    assert broker.requests[0].side == "BUY"


def test_still_actionable_entry_is_resubmitted_when_price_is_close(monkeypatch):
    broker = _FakeBroker()
    trade = _killed_entry(action="BUY", entry=100.0, price=100.5)   # 50 bp — inside the ceiling
    _sync(monkeypatch, broker, [trade], actionable={"TEST": "BUY"})
    assert len(broker.requests) == 1


def test_still_actionable_entry_is_held_once_price_ran_away(monkeypatch):
    broker = _FakeBroker()
    trade = _killed_entry(action="BUY", entry=100.0, price=105.0)   # 500 bp adverse
    _sync(monkeypatch, broker, [trade], actionable={"TEST": "BUY"})
    assert broker.requests == []                                    # NOT chased
    assert trade["broker_status"] == "RESUBMIT_HELD_ADVERSE"


def test_no_actionable_map_keeps_legacy_chase(monkeypatch):
    broker = _FakeBroker()
    trade = _killed_entry(action="BUY", entry=100.0, price=105.0)   # adverse
    _sync(monkeypatch, broker, [trade], actionable=None)           # None → legacy behavior
    assert len(broker.requests) == 1


# ── sizing against what is actually held ────────────────────────────────────
#
# An entry orders the GAP to target, never the full target again. Re-requesting
# the whole size when the broker already holds part of it multiplies the
# position instead of reaching it — the sizing half of the 2026-07-23 cascade,
# where repeated full-size orders sold 112 shares against a 14-share long.

class _PositionBroker(_FakeBroker):
    def __init__(self, qty):
        super().__init__()
        self._qty = qty

    def get_positions(self):
        from src.broker.base import Position
        return [Position(ticker="TEST", quantity=self._qty, avg_cost=100.0)]


def _fresh_entry(action="BUY"):
    return {"ticker": "TEST", "type": "STOCK", "action": action, "status": "OPEN",
            "entry_price": 100.0, "current_price": 100.0,
            "position_size_multiplier": 1.0, "recommendation_id": "abc", "run_id": "r1",
            "broker_order_id": None}


def test_entry_orders_only_the_remainder_when_partly_held(monkeypatch):
    # $1000 notional @ $100 → target 10 shares; 4 already held → order 6.
    broker = _PositionBroker(4)
    trade = _fresh_entry()
    _sync(monkeypatch, broker, [trade], actionable={"TEST": "BUY"})
    assert len(broker.requests) == 1
    assert broker.requests[0].quantity == 6


def test_entry_is_skipped_when_target_is_already_held(monkeypatch):
    broker = _PositionBroker(10)          # already at target
    trade = _fresh_entry()
    _sync(monkeypatch, broker, [trade], actionable={"TEST": "BUY"})
    assert broker.requests == []
    assert trade["broker_status"] == "POSITION_ALREADY_AT_TARGET"


def test_oversized_holding_never_adds_more(monkeypatch):
    """The 2026-07-23 aftermath: broker holds 10x the intended size. The entry
    pass must add NOTHING (correcting the excess is the drift pass's call)."""
    broker = _PositionBroker(100)
    trade = _fresh_entry()
    _sync(monkeypatch, broker, [trade], actionable={"TEST": "BUY"})
    assert broker.requests == []
    assert trade["broker_status"] == "POSITION_ALREADY_AT_TARGET"


def test_opposite_sign_holding_is_not_treated_as_progress(monkeypatch):
    """A short holding is not partial progress toward a long — it belongs to
    the drift pass, so the long entry is sized normally."""
    broker = _PositionBroker(-4)
    trade = _fresh_entry(action="BUY")
    _sync(monkeypatch, broker, [trade], actionable={"TEST": "BUY"})
    assert len(broker.requests) == 1
    assert broker.requests[0].quantity == 10        # full target, no offset
