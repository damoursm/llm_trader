"""Overnight drift-flatten noise fix (2026-07-08).

A drift position whose flatten the OVERNIGHT venue won't accept (thin / ineligible
book) is PENDING until the pre-market open — self-resolving, not a failure. It must
NOT force a broker-health CRITICAL / email every overnight tick (it did for V/META
on 2026-07-07). Daytime / extended flatten failures stay real problems.
"""

import pytest

from config.settings import settings
from src.broker.base import AccountSnapshot, OrderResult, Position
from src.pipeline import _assess_broker_health


# ── the health verdict gating (pure) ────────────────────────────────────────

def _report(drift, **over):
    r = {"mode": "ibkr_paper", "connected": True, "ok": True, "errors": [],
         "rejects": 0, "drift": drift, "drift_flattened": 0}
    r.update(over)
    return r


def test_overnight_pending_flatten_is_not_a_problem():
    v = _assess_broker_health(_report([
        {"ticker": "V", "action": "flatten_pending_open"},
        {"ticker": "META", "action": "flatten_pending_open"}]))
    assert v["down"] is False                       # no CRITICAL / no forced email
    assert v["drift_pending"] == ["V", "META"]      # still surfaced
    assert v["message"] == ""


def test_daytime_flatten_failed_still_alerts():
    v = _assess_broker_health(_report([{"ticker": "V", "action": "flatten_failed"}]))
    assert v["down"] is True
    assert "auto-flatten FAILED" in v["message"]


def test_mixed_pending_and_hard_alerts_only_on_the_hard_one():
    v = _assess_broker_health(_report([
        {"ticker": "V", "action": "flatten_pending_open"},
        {"ticker": "META", "action": "flatten_failed"}]))
    assert v["down"] is True
    assert "META" in v["message"] and "1 position" in v["message"]
    assert v["drift_pending"] == ["V"]


def test_flatten_submitted_still_surfaces_the_drift():
    v = _assess_broker_health(_report([{"ticker": "V", "action": "flatten_submitted"}],
                                      drift_flattened=1))
    assert v["down"] is True
    assert "auto-flatten submitted for 1" in v["message"]


# ── the reconcile labels an overnight unfillable flatten 'pending_open' ──────

class _DriftBroker:
    """Holds one drift position; the flatten submit comes back Cancelled (the
    overnight venue won't accept it)."""
    name = "fake"

    def __init__(self):
        self.requests = []

    def connect(self):
        return True

    def is_connected(self):
        return True

    def get_account(self):
        return AccountSnapshot(equity=100000.0, cash=100000.0, buying_power=100000.0,
                               account_id="DU", currency="USD")

    def get_positions(self):
        return [Position(ticker="V", quantity=3, avg_cost=356.0)]

    def get_fills(self):
        return []

    def get_open_orders(self):
        return []

    def cancel_order(self, ref):
        return True

    def submit_order(self, req):
        self.requests.append(req)
        return OrderResult(ok=False, ticker=req.ticker, side=req.side,
                           requested_qty=req.quantity, filled_qty=0,
                           order_id=None, client_ref=req.client_ref,
                           status="Cancelled", error="Cancelled")


@pytest.fixture(autouse=True)
def _env(monkeypatch):
    from src.broker import reconcile
    monkeypatch.setattr(reconcile.time, "sleep", lambda s: None)
    monkeypatch.setattr(settings, "broker_mode", "ibkr_paper")
    monkeypatch.setattr(settings, "broker_order_type", "LMT")
    monkeypatch.setattr(settings, "broker_limit_cap_bps", 20.0)
    monkeypatch.setattr(settings, "broker_limit_cap_bps_overnight", 150.0)
    monkeypatch.setattr(settings, "broker_submit_retries", 0)
    monkeypatch.setattr(settings, "broker_connect_retries", 0)
    monkeypatch.setattr(settings, "broker_settle_seconds", 0)
    monkeypatch.setattr(settings, "broker_drift_action", "flatten")


def test_reconcile_marks_overnight_unfillable_flatten_pending(monkeypatch):
    from src.broker import reconcile
    broker = _DriftBroker()
    monkeypatch.setattr(reconcile, "get_broker", lambda: broker)
    monkeypatch.setattr(reconcile, "_live_price", lambda t: 356.0)
    monkeypatch.setattr(reconcile, "_overnight_routing_active", lambda: True)   # overnight
    report = reconcile.sync(run_id="r1", trades=[])           # empty ledger → V is drift

    drift = report["drift"]
    assert len(drift) == 1 and drift[0]["ticker"] == "V"
    assert drift[0]["action"] == "flatten_pending_open"        # benign, not 'flatten_failed'
    assert "drift flatten V" not in " ".join(report["errors"])  # not a reconcile error
    assert _assess_broker_health(report)["down"] is False       # → no alert


def test_reconcile_daytime_unfillable_flatten_is_a_failure(monkeypatch):
    from src.broker import reconcile
    broker = _DriftBroker()
    monkeypatch.setattr(reconcile, "get_broker", lambda: broker)
    monkeypatch.setattr(reconcile, "_live_price", lambda t: 356.0)
    monkeypatch.setattr(reconcile, "_overnight_routing_active", lambda: False)  # RTH/extended
    report = reconcile.sync(run_id="r1", trades=[])

    assert report["drift"][0]["action"] == "flatten_failed"     # real problem daytime
    assert _assess_broker_health(report)["down"] is True
