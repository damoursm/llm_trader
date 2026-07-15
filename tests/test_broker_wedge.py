"""Alive-but-wedged IB Gateway hardening (2026-07-13).

A wedged gateway keeps the socket open — isConnected() stays True — while EVERY
API request times out with a BLANK-message TimeoutError. Before this fix that
produced: blank errors everywhere (undebuggable), a whole tick of 45s-timeouts,
and the failures mislabeled "N order(s) rejected". Now:

  • _exc_text turns a blank exception into its class name (honest logs + a
    non-blank broker_orders.error);
  • the broker counts CONSECUTIVE request timeouts and, at the threshold, treats
    the connected-but-wedged session as dropped → force-recycles the client
    (disconnect + redial), which on a still-wedged gateway arms the cooldown so
    the rest of the tick fast-fails instead of each call hanging;
  • the reconcile classifies a timeout/disconnect failure as a broker TIMEOUT,
    and the health verdict says "broker NOT RESPONDING", not "rejected".
"""

import pytest

from config.settings import settings
from src.broker.base import OrderRequest, OrderResult
from src.broker.ibkr import IBKRBroker, _exc_text, _is_timeout_exc


# ── the diagnosis-blocker: blank exception text ─────────────────────────────

def test_exc_text_recovers_blank_timeout():
    assert _exc_text(TimeoutError()) == "TimeoutError"     # str('') → class name
    assert _exc_text(ValueError("insufficient funds")) == "insufficient funds"
    assert _is_timeout_exc(TimeoutError()) is True
    assert _is_timeout_exc(RuntimeError("request timeout")) is True
    assert _is_timeout_exc(ValueError("no security definition")) is False


# ── wedge detection at the broker level ─────────────────────────────────────

class WedgeIB:
    """isConnected() True (socket alive) but every request raises a blank
    TimeoutError — the alive-but-wedged gateway. `recovered` flips it healthy."""

    def __init__(self):
        self.connected = True
        self.recovered = False
        self.dials = 0
        self.disconnect_calls = 0
        self.RequestTimeout = 45.0

    def isConnected(self):
        return self.connected

    def connect(self, host, port, clientId=None, timeout=None, readonly=False):
        self.dials += 1
        if not self.recovered:
            raise TimeoutError()                 # dial to a wedged gateway times out
        self.connected = True

    def disconnect(self):
        self.disconnect_calls += 1
        self.connected = False

    def reqPositions(self, *a, **k):
        if not self.recovered:
            raise TimeoutError()                 # blank-message request timeout
        return None                              # healthy gateway responds
    def reqExecutions(self, *a, **k):
        if not self.recovered:
            raise TimeoutError()
        return []
    def positions(self, account=""):
        return []


@pytest.fixture
def _wedge_env(monkeypatch):
    monkeypatch.setattr(settings, "broker_wedge_timeout_threshold", 3)
    monkeypatch.setattr(settings, "broker_reconnect_cooldown_seconds", 300.0)


def _broker(fake):
    b = IBKRBroker(host="127.0.0.1", port=4002, client_id=99, account="DU0")
    b._ib = fake
    return b


def test_counter_accumulates_through_get_positions(_wedge_env):
    fake = WedgeIB()
    b = _broker(fake)
    # get_positions: reqPositions() times out (counted), positions() cache read
    # returns [] — the method still returns a (cached, empty) list.
    for _ in range(2):
        assert b.get_positions() == []
    assert b._consecutive_timeouts == 2
    assert not b.is_wedged()                     # below threshold


def test_wedge_triggers_forced_recycle_then_fast_fails(_wedge_env):
    fake = WedgeIB()
    b = _broker(fake)
    # 3 timeouts reach the threshold. The 3rd get_positions call: reqPositions
    # times out → counter hits 3, but _ensure_connected already returned True at
    # entry (checked before the request). The NEXT touchpoint sees the wedge.
    for _ in range(3):
        b.get_positions()
    assert b.is_wedged()
    # Next touchpoint: _ensure_connected sees wedged → connect(force=True) →
    # disconnect() + redial; redial times out → cooldown armed, session dropped.
    b.get_positions()
    assert fake.disconnect_calls >= 1            # client was RECYCLED (not trusted)
    assert fake.dials == 1
    dials_after = fake.dials
    # Inside the cooldown every further touchpoint fast-fails — NO more dials,
    # NO 45s hangs. This is the fix for the whole-tick-of-timeouts symptom.
    b.get_positions(); b.get_fills(); b.get_positions()
    assert fake.dials == dials_after


def test_wedge_auto_recovers_when_gateway_returns(_wedge_env, monkeypatch):
    fake = WedgeIB()
    b = _broker(fake)
    for _ in range(3):
        b.get_positions()
    assert b.is_wedged()
    monkeypatch.setattr(settings, "broker_reconnect_cooldown_seconds", 0.0)  # no throttle
    fake.recovered = True                        # IBC restarted the gateway
    b.get_positions()                            # forced redial now succeeds
    assert b.is_connected()
    assert b._consecutive_timeouts == 0          # counter cleared on the fresh session
    assert not b.is_wedged()


def test_submit_order_fast_fails_once_wedged(_wedge_env):
    fake = WedgeIB()
    b = _broker(fake)
    for _ in range(3):
        b.get_positions()                        # drive the wedge
    assert b.is_wedged()
    # submit_order gates on _ensure_connected → forced recycle fails → cooldown →
    # returns DISCONNECTED without ever attempting a 45s placeOrder timeout.
    res = b.submit_order(OrderRequest(ticker="AAPL", side="BUY", quantity=1,
                                      client_ref="r", intent="ENTRY"))
    assert res.ok is False
    assert res.status == "DISCONNECTED"
    assert "not connected" in (res.error or "").lower()


# ── honest reporting: timeouts ≠ rejects ────────────────────────────────────

def test_reconcile_classifies_timeout_as_broker_timeout():
    from src.broker.reconcile import _new_report, _tally_submit_failure
    rep = _new_report()
    # blank-message timeout surfaced as its class name by submit_order
    _tally_submit_failure(rep, "entry", "HNGE",
                          OrderResult(ok=False, ticker="HNGE", side="BUY",
                                      requested_qty=3, status="ERROR", error="TimeoutError"))
    _tally_submit_failure(rep, "exit", "FTNT",
                          OrderResult(ok=False, ticker="FTNT", side="SELL",
                                      requested_qty=3, status="DISCONNECTED",
                                      error="broker not connected"))
    # a genuine reject still counts as a reject
    _tally_submit_failure(rep, "entry", "XYZ",
                          OrderResult(ok=False, ticker="XYZ", side="BUY",
                                      requested_qty=1, status="ERROR",
                                      error="Order rejected: insufficient funds"))
    assert rep["broker_timeouts"] == 2
    assert rep["rejects"] == 1
    assert any("TimeoutError" in e for e in rep["errors"])     # non-blank error persisted


def test_health_verdict_says_not_responding_not_rejected():
    from src.pipeline import _assess_broker_health
    v = _assess_broker_health({"mode": "ibkr_paper", "connected": True, "ok": True,
                               "broker_timeouts": 6, "rejects": 0, "drift": []})
    assert v["down"] is True
    assert "NOT RESPONDING" in v["message"]
    assert "wedged" in v["message"].lower()
    assert "rejected" not in v["message"]
    assert v["broker_timeouts"] == 6


def test_health_verdict_still_flags_real_rejects_separately():
    from src.pipeline import _assess_broker_health
    v = _assess_broker_health({"mode": "ibkr_paper", "connected": True, "ok": True,
                               "broker_timeouts": 0, "rejects": 2, "drift": []})
    assert v["down"] is True and "2 order(s) rejected" in v["message"]
    assert "NOT RESPONDING" not in v["message"]


def test_health_verdict_carries_exact_errors_deduped():
    """The verdict threads the EXACT per-order reasons (order-preserving dedupe)
    so the email banner + CRITICAL log can show them, not just the count."""
    from src.pipeline import _assess_broker_health
    errs = [
        "exit NET: Cancelled: Error 10329 … directly routed to OVERNIGHT",
        "exit BLLN: Cancelled: Error 10329 … directly routed to OVERNIGHT",
        "exit NET: Cancelled: Error 10329 … directly routed to OVERNIGHT",  # dup
    ]
    v = _assess_broker_health({"mode": "ibkr_paper", "connected": True, "ok": True,
                               "broker_timeouts": 0, "rejects": 2, "drift": [],
                               "errors": errs})
    assert v["errors"] == errs[:2]           # deduped, order preserved
    assert "10329" in " ".join(v["errors"])
