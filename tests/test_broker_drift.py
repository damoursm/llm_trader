"""Drift prevention + auto-reconciliation:
(P1) working ENTRY orders are cancelled the moment their trade closes,
(P2) exits flatten what the broker ACTUALLY holds (sign-checked),
(P3) in-flight closes are not reported as drift,
(P4) true orphans are auto-flattened via a price-capped LMT at a live quote
     (report-only is configurable; ibkr_live always refuses auto-flatten).
All fakes, no sleeps, no network.
"""

import pytest

from config.settings import settings
from src.broker.base import AccountSnapshot, OpenOrderInfo, OrderResult, Position


@pytest.fixture(autouse=True)
def _no_sleep(monkeypatch):
    from src.broker import reconcile
    monkeypatch.setattr(reconcile.time, "sleep", lambda s: None)


@pytest.fixture(autouse=True)
def _env(monkeypatch):
    from src.performance import market_calendar
    monkeypatch.setattr(settings, "broker_mode", "ibkr_paper")
    monkeypatch.setattr(settings, "broker_order_type", "LMT")
    monkeypatch.setattr(settings, "broker_limit_cap_bps", 20.0)
    monkeypatch.setattr(settings, "broker_base_notional_ccy", "USD")
    monkeypatch.setattr(settings, "broker_base_notional", 1000.0)
    monkeypatch.setattr(settings, "broker_sizing_mode", "notional")
    monkeypatch.setattr(settings, "broker_submit_retries", 0)
    monkeypatch.setattr(settings, "broker_unfilled_cancel_minutes", 0)
    monkeypatch.setattr(settings, "broker_drift_action", "flatten")
    monkeypatch.setattr(market_calendar, "current_session", lambda now=None: "rth")


class _FakeBroker:
    name = "fake"

    def __init__(self, positions=(), open_orders=(), cancel_ok=True):
        self.requests = []
        self.cancelled = []
        self._positions = list(positions)
        self._open_orders = list(open_orders)
        self._cancel_ok = cancel_ok

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
        return []

    def get_open_orders(self):
        return list(self._open_orders)

    def cancel_order(self, client_ref):
        self.cancelled.append(client_ref)
        return self._cancel_ok

    def submit_order(self, req):
        self.requests.append(req)
        return OrderResult(ok=True, ticker=req.ticker, side=req.side,
                           requested_qty=req.quantity, filled_qty=req.quantity,
                           avg_fill_price=req.limit_price or 100.0,
                           order_id=str(len(self.requests)), client_ref=req.client_ref,
                           status="Filled")


def _sync(monkeypatch, broker, trades, live_price=100.0):
    from src.broker import reconcile
    monkeypatch.setattr(reconcile, "get_broker", lambda: broker)
    monkeypatch.setattr(reconcile, "_live_price", lambda t: live_price)
    return reconcile.sync(run_id="r1", trades=trades)


# ── P1: working entry cancelled when its trade closes ─────────────────────

def test_entry_cancelled_when_trade_closes_before_fill(monkeypatch):
    """The orphan factory: ledger closed, entry LMT still resting → cancel it.
    Nothing held → no exit, no drift, no order left to fill later."""
    broker = _FakeBroker()   # no position: the entry never filled
    trade = {
        "ticker": "TEST", "type": "STOCK", "action": "BUY", "status": "CLOSED",
        "entry_price": 100.0, "exit_price": 101.0, "position_size_multiplier": 1.0,
        "recommendation_id": "abc", "run_id": "r1",
        "broker_order_id": "11", "broker_client_ref": "abc",
        "broker_status": "Submitted", "broker_fill_qty": 0, "broker_requested_qty": 10,
    }
    report = _sync(monkeypatch, broker, [trade])
    assert broker.cancelled == ["abc"]
    assert report["entry_cancels_on_close"] == 1
    assert trade["broker_status"] == "Cancelled"
    assert trade["broker_cancel_reason"] == "closed_before_fill"
    assert broker.requests == []          # nothing held → nothing submitted
    assert report["drift"] == []
    assert trade["broker_exit_status"] == "NOTHING_TO_CLOSE"


def test_partially_filled_entry_cancelled_then_residual_flattened(monkeypatch):
    """Closed with 4/10 filled: cancel the working remainder, exit the 4 held."""
    broker = _FakeBroker(positions=[Position(ticker="TEST", quantity=4, avg_cost=100.0)])
    trade = {
        "ticker": "TEST", "type": "STOCK", "action": "BUY", "status": "CLOSED",
        "entry_price": 100.0, "exit_price": 101.0, "position_size_multiplier": 1.0,
        "recommendation_id": "abc", "run_id": "r1",
        "broker_order_id": "11", "broker_client_ref": "abc",
        "broker_status": "Submitted", "broker_fill_qty": 4, "broker_requested_qty": 10,
    }
    report = _sync(monkeypatch, broker, [trade])
    assert broker.cancelled == ["abc"]
    assert report["entry_cancels_on_close"] == 1
    assert report["exits_submitted"] == 1
    req = broker.requests[0]
    assert req.side == "SELL" and req.quantity == 4
    assert report["drift"] == []


# ── P2: exits flatten the ACTUAL holding, sign-checked ────────────────────

def test_exit_sizes_from_actual_held_not_stale_fill_qty(monkeypatch):
    """Recorded fill said 5, broker actually holds 10 → exit 10 (no residue)."""
    broker = _FakeBroker(positions=[Position(ticker="TEST", quantity=10, avg_cost=100.0)])
    trade = {
        "ticker": "TEST", "type": "STOCK", "action": "BUY", "status": "CLOSED",
        "entry_price": 100.0, "exit_price": 101.0, "position_size_multiplier": 1.0,
        "recommendation_id": "abc", "run_id": "r1",
        "broker_order_id": "11", "broker_client_ref": "abc",
        "broker_status": "Filled", "broker_fill_qty": 5, "broker_requested_qty": 10,
    }
    report = _sync(monkeypatch, broker, [trade])
    assert report["exits_submitted"] == 1
    assert broker.requests[0].quantity == 10


def test_exit_never_blind_trades_wrong_sign_position(monkeypatch):
    """Closed BUY but the broker holds a SHORT → don't touch it; it surfaces
    as drift for the auto-reconcile pass (which BUYs it back, capped)."""
    broker = _FakeBroker(positions=[Position(ticker="TEST", quantity=-7, avg_cost=100.0)])
    trade = {
        "ticker": "TEST", "type": "STOCK", "action": "BUY", "status": "CLOSED",
        "entry_price": 100.0, "exit_price": 101.0, "position_size_multiplier": 1.0,
        "recommendation_id": "abc", "run_id": "r1",
        "broker_order_id": "11", "broker_client_ref": "abc",
        "broker_status": "Filled", "broker_fill_qty": 7, "broker_requested_qty": 7,
    }
    report = _sync(monkeypatch, broker, [trade], live_price=100.0)
    assert trade["broker_exit_status"] == "NOTHING_TO_CLOSE"
    # The drift pass handles it: BUY back the short, price-capped.
    assert [d["ticker"] for d in report["drift"]] == ["TEST"]
    flat = broker.requests[-1]
    assert flat.side == "BUY" and flat.quantity == 7
    assert flat.client_ref.startswith("drift-TEST-")


# ── P3: in-flight closes are not drift ────────────────────────────────────

def test_exit_submitted_this_tick_is_not_drift(monkeypatch):
    broker = _FakeBroker(positions=[Position(ticker="TEST", quantity=10, avg_cost=100.0)])
    trade = {
        "ticker": "TEST", "type": "STOCK", "action": "BUY", "status": "CLOSED",
        "entry_price": 100.0, "exit_price": 101.0, "position_size_multiplier": 1.0,
        "recommendation_id": "abc", "run_id": "r1",
        "broker_order_id": "11", "broker_client_ref": "abc",
        "broker_status": "Filled", "broker_fill_qty": 10, "broker_requested_qty": 10,
    }
    report = _sync(monkeypatch, broker, [trade])
    assert report["exits_submitted"] == 1
    assert report["drift"] == []          # position explained by the in-flight close


def test_working_exit_from_earlier_tick_is_not_drift(monkeypatch):
    broker = _FakeBroker(positions=[Position(ticker="TEST", quantity=10, avg_cost=100.0)])
    trade = {
        "ticker": "TEST", "type": "STOCK", "action": "BUY", "status": "CLOSED",
        "entry_price": 100.0, "exit_price": 101.0, "position_size_multiplier": 1.0,
        "recommendation_id": "abc", "run_id": "r1",
        "broker_order_id": "11", "broker_client_ref": "abc",
        "broker_status": "Filled", "broker_fill_qty": 10, "broker_requested_qty": 10,
        "broker_exit_order_id": "12", "broker_exit_client_ref": "abc-exit",
        "broker_exit_status": "Submitted", "broker_exit_fill_qty": 0,
    }
    report = _sync(monkeypatch, broker, [trade])
    assert report["drift"] == []


# ── P4: true orphans auto-flattened (capped LMT at a live quote) ──────────

def test_orphans_auto_flattened_long_and_short(monkeypatch):
    broker = _FakeBroker(positions=[
        Position(ticker="VGT", quantity=7, avg_cost=110.0),
        Position(ticker="TRUP", quantity=-32, avg_cost=22.0),
    ])
    report = _sync(monkeypatch, broker, [], live_price=100.0)
    assert report["drift_flattened"] == 2
    assert {d["action"] for d in report["drift"]} == {"flatten_submitted"}
    by_ticker = {r.ticker: r for r in broker.requests}
    assert by_ticker["VGT"].side == "SELL" and by_ticker["VGT"].quantity == 7
    # SELL cap: live 100 × (1 − 20 bp)
    assert by_ticker["VGT"].limit_price == pytest.approx(99.80, abs=1e-6)
    assert by_ticker["TRUP"].side == "BUY" and by_ticker["TRUP"].quantity == 32
    assert by_ticker["TRUP"].limit_price == pytest.approx(100.20, abs=1e-6)
    assert all(r.order_type == "LMT" for r in broker.requests)


def test_resting_flatten_recancelled_and_reanchored(monkeypatch):
    """A flatten from an earlier tick still working → cancel, resubmit fresh."""
    broker = _FakeBroker(
        positions=[Position(ticker="VGT", quantity=7, avg_cost=110.0)],
        open_orders=[OpenOrderInfo(client_ref="drift-VGT-r0", order_id="9",
                                   status="Submitted", ticker="VGT", side="SELL")],
    )
    report = _sync(monkeypatch, broker, [], live_price=95.0)
    assert broker.cancelled == ["drift-VGT-r0"]
    assert report["drift_flattened"] == 1
    req = broker.requests[0]
    assert req.client_ref == "drift-VGT-r1"
    assert req.limit_price == pytest.approx(95.0 * 0.998, abs=0.01)


def test_flatten_lost_cancel_race_skips_tick(monkeypatch):
    broker = _FakeBroker(
        positions=[Position(ticker="VGT", quantity=7, avg_cost=110.0)],
        open_orders=[OpenOrderInfo(client_ref="drift-VGT-r0", order_id="9",
                                   status="Submitted", ticker="VGT", side="SELL")],
        cancel_ok=False,   # the working flatten filled during the cancel
    )
    report = _sync(monkeypatch, broker, [])
    assert broker.requests == []
    assert report["drift_flattened"] == 0
    assert report["drift"][0]["action"] == "flatten_failed"


def test_no_live_quote_reports_without_trading(monkeypatch):
    broker = _FakeBroker(positions=[Position(ticker="VGT", quantity=7, avg_cost=110.0)])
    report = _sync(monkeypatch, broker, [], live_price=None)
    assert broker.requests == []
    assert report["drift"][0]["action"] == "flatten_failed"


def test_report_mode_and_live_mode_never_flatten(monkeypatch):
    broker = _FakeBroker(positions=[Position(ticker="VGT", quantity=7, avg_cost=110.0)])
    monkeypatch.setattr(settings, "broker_drift_action", "report")
    report = _sync(monkeypatch, broker, [])
    assert broker.requests == [] and report["drift"][0]["action"] == "report"

    # ibkr_live downgrades flatten → report (real-money safety).
    monkeypatch.setattr(settings, "broker_drift_action", "flatten")
    monkeypatch.setattr(settings, "broker_mode", "ibkr_live")
    broker2 = _FakeBroker(positions=[Position(ticker="VGT", quantity=7, avg_cost=110.0)])
    report2 = _sync(monkeypatch, broker2, [])
    assert broker2.requests == [] and report2["drift"][0]["action"] == "report"
