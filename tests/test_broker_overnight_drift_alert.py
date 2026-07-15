"""Overnight drift-flatten noise fix (2026-07-08) + off-RTH accepted-flatten
follow-up (2026-07-11).

A drift position whose flatten the OVERNIGHT venue won't accept (thin / ineligible
book) is PENDING until the pre-market open — self-resolving, not a failure. It must
NOT force a broker-health CRITICAL / email every overnight tick (it did for V/META
on 2026-07-07). Daytime / extended flatten failures stay real problems.

Follow-up (PRCH 2026-07-10): a flatten the broker ACCEPTS but hasn't filled yet
('flatten_submitted') is the SAME self-resolving convergence off-RTH — it isn't
a rejection, just a capped order still working a thin pre-market/overnight book.
It fired 6 straight CRITICAL alerts pre-market before PRCH filled cleanly. Off-RTH
this now labels 'flatten_pending_fill' and is excluded from the health verdict the
same way; RTH keeps alerting (a capped marketable LMT not filling in a deep book
is genuinely unusual there).
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


def test_off_rth_pending_fill_is_not_a_problem():
    """PRCH 2026-07-10: accepted-but-unfilled off-RTH is convergence, not a
    failure — must not force a CRITICAL every pre-market/overnight tick."""
    v = _assess_broker_health(_report([{"ticker": "PRCH", "action": "flatten_pending_fill"}],
                                      drift_flattened=1))
    assert v["down"] is False
    assert v["drift_pending"] == ["PRCH"]
    assert v["message"] == ""


def test_mixed_pending_fill_and_hard_alerts_only_on_the_hard_one():
    v = _assess_broker_health(_report([
        {"ticker": "PRCH", "action": "flatten_pending_fill"},
        {"ticker": "META", "action": "flatten_failed"}]))
    assert v["down"] is True
    assert "META" in v["message"] and "1 position" in v["message"]
    assert v["drift_pending"] == ["PRCH"]


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


# ── the reconcile labels an off-RTH ACCEPTED-but-unfilled flatten 'pending_fill' ──

class _AcceptedDriftBroker(_DriftBroker):
    """Holds one drift position; the flatten submit is ACCEPTED (resting) but
    hasn't filled yet — a thin off-RTH book working toward a fill, not a
    rejection (PRCH 2026-07-10)."""

    def submit_order(self, req):
        self.requests.append(req)
        return OrderResult(ok=True, ticker=req.ticker, side=req.side,
                           requested_qty=req.quantity, filled_qty=0,
                           order_id="o1", client_ref=req.client_ref,
                           status="Submitted", error=None)


def test_reconcile_marks_off_rth_accepted_flatten_pending_fill(monkeypatch):
    """Extended (pre-market) session, venue ACCEPTS the flatten but it hasn't
    filled yet — pending convergence, not a CRITICAL alert."""
    from src.broker import reconcile
    from src.performance import market_calendar
    broker = _AcceptedDriftBroker()
    monkeypatch.setattr(reconcile, "get_broker", lambda: broker)
    monkeypatch.setattr(reconcile, "_live_price", lambda t: 356.0)
    monkeypatch.setattr(reconcile, "_overnight_routing_active", lambda: False)
    monkeypatch.setattr(market_calendar, "current_session", lambda now=None: "extended")
    report = reconcile.sync(run_id="r1", trades=[])

    assert report["drift"][0]["action"] == "flatten_pending_fill"
    assert report["drift_flattened"] == 1          # still counted as working activity
    assert _assess_broker_health(report)["down"] is False


def test_reconcile_rth_accepted_flatten_still_alerts(monkeypatch):
    """The same accepted-but-unfilled outcome during RTH stays a hard alert —
    a capped marketable LMT not filling in a deep book there is genuinely
    unusual (no regression from the pre-fix RTH behaviour)."""
    from src.broker import reconcile
    from src.performance import market_calendar
    broker = _AcceptedDriftBroker()
    monkeypatch.setattr(reconcile, "get_broker", lambda: broker)
    monkeypatch.setattr(reconcile, "_live_price", lambda t: 356.0)
    monkeypatch.setattr(reconcile, "_overnight_routing_active", lambda: False)
    monkeypatch.setattr(market_calendar, "current_session", lambda now=None: "rth")
    report = reconcile.sync(run_id="r1", trades=[])

    assert report["drift"][0]["action"] == "flatten_submitted"
    assert _assess_broker_health(report)["down"] is True


def test_reconcile_overnight_accepted_flatten_is_also_pending(monkeypatch):
    """An overnight venue that for once ACCEPTS the flatten (rather than the
    usual outright reject) must not be treated as WORSE than a reject."""
    from src.broker import reconcile
    from src.performance import market_calendar
    broker = _AcceptedDriftBroker()
    monkeypatch.setattr(reconcile, "get_broker", lambda: broker)
    monkeypatch.setattr(reconcile, "_live_price", lambda t: 356.0)
    monkeypatch.setattr(reconcile, "_overnight_routing_active", lambda: True)
    monkeypatch.setattr(market_calendar, "current_session", lambda now=None: "overnight")
    report = reconcile.sync(run_id="r1", trades=[])

    assert report["drift"][0]["action"] == "flatten_pending_fill"
    assert _assess_broker_health(report)["down"] is False
