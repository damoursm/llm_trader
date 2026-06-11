"""Order-submission reliability: acceptance verification, bounded transient
retry (price-capped), the duplicate guard, and the stale-unfilled cancel +
re-anchor pass. All fakes, no sleeps (time.sleep is patched out), no network.
"""

from datetime import datetime, timedelta, timezone

import pytest

from config.settings import settings
from src.broker.base import AccountSnapshot, OpenOrderInfo, OrderRequest, OrderResult


@pytest.fixture(autouse=True)
def _no_sleep(monkeypatch):
    from src.broker import reconcile
    monkeypatch.setattr(reconcile.time, "sleep", lambda s: None)


def _ok(req, status="Filled"):
    return OrderResult(ok=True, ticker=req.ticker, side=req.side,
                       requested_qty=req.quantity, filled_qty=req.quantity,
                       avg_fill_price=req.limit_price or 100.0, order_id="1",
                       client_ref=req.client_ref, status=status)


def _fail(req, status="DISCONNECTED", error="socket dropped"):
    return OrderResult(ok=False, ticker=req.ticker, side=req.side,
                       requested_qty=req.quantity, client_ref=req.client_ref,
                       status=status, error=error)


class _ScriptedBroker:
    """submit_order pops the next scripted outcome ('ok' | 'transient' | 'hard')."""
    name = "fake"

    def __init__(self, script, open_orders=(), fills=(), positions=()):
        self.script = list(script)
        self.requests = []
        self.cancelled = []
        self._open_orders = list(open_orders)
        self._fills = list(fills)
        self._positions = list(positions)

    def connect(self):
        return True

    def is_connected(self):
        return True

    def get_account(self):
        return AccountSnapshot(equity=100000.0, cash=100000.0, buying_power=100000.0,
                               account_id="DU000", currency="USD")

    def get_positions(self):
        return list(self._positions)

    def get_fills(self):
        return list(self._fills)

    def get_open_orders(self):
        return list(self._open_orders)

    def cancel_order(self, client_ref):
        self.cancelled.append(client_ref)
        return True

    def submit_order(self, req):
        self.requests.append(req)
        kind = self.script.pop(0) if self.script else "ok"
        if kind == "ok":
            return _ok(req)
        if kind == "transient":
            return _fail(req)
        return _fail(req, status="Inactive", error="insufficient buying power")


# ── transient vs hard classification ──────────────────────────────────────

@pytest.mark.parametrize("status,error,want", [
    ("DISCONNECTED", "broker not connected", True),
    ("ERROR", "connection reset by peer", True),
    ("ERROR", "request timed out", True),
    ("Cancelled", "pacing violation", True),
    ("Inactive", "insufficient buying power", False),
    ("Inactive", "no trading permissions", False),
    ("ERROR", "invalid contract", False),
])
def test_is_transient_failure(status, error, want):
    from src.broker.reconcile import _is_transient_failure
    res = OrderResult(ok=False, ticker="T", side="BUY", requested_qty=1,
                      status=status, error=error)
    assert _is_transient_failure(res) is want


# ── _submit_with_retry ─────────────────────────────────────────────────────

def _req(**kw):
    base = dict(ticker="TEST", side="BUY", quantity=10, order_type="MKT",
                limit_price=None, client_ref="ref1", intent="ENTRY")
    base.update(kw)
    return OrderRequest(**base)


def test_retry_transient_then_success_uses_capped_lmt(monkeypatch):
    from src.broker import reconcile
    monkeypatch.setattr(settings, "broker_submit_retries", 2)
    monkeypatch.setattr(settings, "broker_retry_wait_seconds", 1)
    monkeypatch.setattr(settings, "broker_limit_cap_bps", 20.0)
    broker = _ScriptedBroker(["transient", "ok"])
    report = reconcile._new_report()

    res = reconcile._submit_with_retry(broker, _req(), model_price=100.0,
                                       report=report, intent="ENTRY")
    assert res.ok and len(broker.requests) == 2
    # The retry must be price-protected: LMT at model +20 bp, even though the
    # first attempt was MKT.
    second = broker.requests[1]
    assert second.order_type == "LMT"
    assert second.limit_price == pytest.approx(100.20, abs=1e-6)
    # Reliability record: one SUBMIT_FAILED event row + the retries counter.
    assert report["retries"] == 1
    assert [o for o in report["orders"] if o["event"] == "SUBMIT_FAILED"]


def test_hard_reject_never_retries(monkeypatch):
    from src.broker import reconcile
    monkeypatch.setattr(settings, "broker_submit_retries", 3)
    broker = _ScriptedBroker(["hard", "ok"])
    report = reconcile._new_report()
    res = reconcile._submit_with_retry(broker, _req(), 100.0, report, "ENTRY")
    assert not res.ok and len(broker.requests) == 1
    assert report["retries"] == 0


def test_retry_budget_is_bounded(monkeypatch):
    from src.broker import reconcile
    monkeypatch.setattr(settings, "broker_submit_retries", 2)
    broker = _ScriptedBroker(["transient", "transient", "transient", "transient"])
    report = reconcile._new_report()
    res = reconcile._submit_with_retry(broker, _req(), 100.0, report, "ENTRY")
    # 1 initial + 2 retries, then give up (the next tick takes over).
    assert not res.ok and len(broker.requests) == 3


def test_duplicate_guard_adopts_order_already_at_broker(monkeypatch):
    """A submission that errored after transmission may exist at the broker —
    the retry must adopt it, not resubmit (double-position protection)."""
    from src.broker import reconcile
    monkeypatch.setattr(settings, "broker_submit_retries", 2)
    broker = _ScriptedBroker(
        ["transient", "ok"],
        open_orders=[OpenOrderInfo(client_ref="ref1", order_id="42",
                                   status="Submitted", ticker="TEST", side="BUY")],
    )
    report = reconcile._new_report()
    res = reconcile._submit_with_retry(broker, _req(), 100.0, report, "ENTRY")
    assert res.ok and res.order_id == "42" and res.status == "Submitted"
    assert len(broker.requests) == 1          # never resubmitted
    # The adopted working order completes via the fill-refresh pass later.


# ── stale-unfilled cancel + same-tick re-anchored resubmit ────────────────

def _stale_iso(minutes=180):
    return (datetime.now(timezone.utc) - timedelta(minutes=minutes)).isoformat(timespec="seconds")


def _broker_settings(monkeypatch):
    monkeypatch.setattr(settings, "broker_mode", "ibkr_paper")
    monkeypatch.setattr(settings, "broker_order_type", "LMT")
    monkeypatch.setattr(settings, "broker_limit_cap_bps", 20.0)
    monkeypatch.setattr(settings, "broker_base_notional_ccy", "USD")
    monkeypatch.setattr(settings, "broker_base_notional", 1000.0)
    monkeypatch.setattr(settings, "broker_sizing_mode", "notional")
    monkeypatch.setattr(settings, "broker_unfilled_cancel_minutes", 90)
    monkeypatch.setattr(settings, "broker_submit_retries", 0)


def test_stale_entry_cancelled_and_resubmitted_at_current_mark(monkeypatch):
    from src.broker import reconcile
    from src.performance import market_calendar
    _broker_settings(monkeypatch)
    monkeypatch.setattr(market_calendar, "current_session", lambda now=None: "rth")
    broker = _ScriptedBroker(["ok"])
    monkeypatch.setattr(reconcile, "get_broker", lambda: broker)

    trade = {
        "ticker": "TEST", "type": "STOCK", "action": "BUY", "status": "OPEN",
        "entry_price": 100.0, "current_price": 90.0,        # market moved away
        "position_size_multiplier": 1.0,
        "recommendation_id": "abc", "run_id": "r1",
        "broker_order_id": "11", "broker_client_ref": "abc",
        "broker_status": "Submitted", "broker_fill_qty": 0,
        "broker_requested_qty": 10, "broker_submitted_at": _stale_iso(),
    }
    report = reconcile.sync(run_id="r1", trades=[trade])

    assert broker.cancelled == ["abc"]
    assert report["stale_cancels"] == 1
    assert report["entries_submitted"] == 1
    req = broker.requests[0]
    assert req.client_ref == "abc-r1"                       # fresh ref per cycle
    # Re-anchored at the CURRENT mark (90), not the stale entry price (100).
    assert req.limit_price == pytest.approx(90.0 * 1.002, abs=0.01)
    assert trade["broker_order_id"] == "1"                  # new order recorded
    assert trade["broker_cancelled_order_ids"] == ["11"]


def test_stale_exit_resubmits_at_live_quote(monkeypatch):
    from src.broker import reconcile
    from src.broker.base import Position
    from src.performance import market_calendar
    _broker_settings(monkeypatch)
    monkeypatch.setattr(market_calendar, "current_session", lambda now=None: "rth")
    monkeypatch.setattr(reconcile, "_live_price", lambda t: 95.0)
    broker = _ScriptedBroker(
        ["ok"], positions=[Position(ticker="TEST", quantity=10, avg_cost=100.0)],
    )
    monkeypatch.setattr(reconcile, "get_broker", lambda: broker)

    trade = {
        "ticker": "TEST", "type": "STOCK", "action": "BUY", "status": "CLOSED",
        "entry_price": 100.0, "exit_price": 101.0, "current_price": 101.0,
        "position_size_multiplier": 1.0,
        "recommendation_id": "abc", "run_id": "r1",
        "broker_order_id": "11", "broker_status": "Filled",
        "broker_fill_qty": 10, "broker_requested_qty": 10,
        "broker_exit_order_id": "12", "broker_exit_client_ref": "abc-exit",
        "broker_exit_status": "Submitted", "broker_exit_fill_qty": 0,
        "broker_exit_requested_qty": 10, "broker_exit_submitted_at": _stale_iso(),
    }
    report = reconcile.sync(run_id="r1", trades=[trade])

    assert broker.cancelled == ["abc-exit"]
    assert report["stale_cancels"] == 1 and report["exits_submitted"] == 1
    req = broker.requests[0]
    assert req.side == "SELL" and req.client_ref == "abc-exit-r1"
    # SELL cap anchored at the fresh live quote: 95 × (1 − 20 bp).
    assert req.limit_price == pytest.approx(95.0 * 0.998, abs=0.01)


def test_partial_fills_and_fresh_orders_left_alone(monkeypatch):
    from src.broker import reconcile
    _broker_settings(monkeypatch)
    broker = _ScriptedBroker([])
    report = reconcile._new_report()
    partial = {
        "ticker": "P", "action": "BUY", "status": "OPEN",
        "broker_order_id": "1", "broker_client_ref": "p",
        "broker_status": "Submitted", "broker_fill_qty": 5,
        "broker_submitted_at": _stale_iso(),
    }
    fresh = {
        "ticker": "F", "action": "BUY", "status": "OPEN",
        "broker_order_id": "2", "broker_client_ref": "f",
        "broker_status": "Submitted", "broker_fill_qty": 0,
        "broker_submitted_at": _stale_iso(minutes=10),
    }
    assert reconcile._cancel_stale_unfilled(broker, [partial, fresh], report) is False
    assert broker.cancelled == []

    # And the whole pass is off when the age cap is 0.
    monkeypatch.setattr(settings, "broker_unfilled_cancel_minutes", 0)
    stale = dict(fresh, broker_submitted_at=_stale_iso())
    assert reconcile._cancel_stale_unfilled(broker, [stale], report) is False
