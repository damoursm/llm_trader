"""Tests for src.broker.reconcile.sync — shadow execution + idempotency + drift.

Uses a FakeBroker and a monkeypatched repo so nothing touches IB Gateway or the
real trades database.
"""

import pytest

import src.broker.reconcile as rec
from config.settings import settings
from src.broker.base import AccountSnapshot, Broker, OrderResult, Position


class FakeBroker(Broker):
    name = "fake"

    def __init__(self, positions=None, equity=100_000.0):
        self._positions = positions or []
        self._equity = equity
        self.orders = []

    def connect(self):
        return True

    def is_connected(self):
        return True

    def get_account(self):
        return AccountSnapshot(self._equity, self._equity, self._equity, "FAKE")

    def get_positions(self):
        return list(self._positions)

    def submit_order(self, req):
        self.orders.append(req)
        return OrderResult(
            ok=True, ticker=req.ticker, side=req.side, requested_qty=req.quantity,
            filled_qty=req.quantity, avg_fill_price=100.05, order_id=f"o{len(self.orders)}",
            client_ref=req.client_ref, status="Filled",
        )


@pytest.fixture
def repo_store(monkeypatch):
    store = {"trades": []}
    monkeypatch.setattr(rec.repo, "load_trades", lambda: store["trades"])
    monkeypatch.setattr(rec.repo, "save_trades", lambda t: store.update(trades=t))
    monkeypatch.setattr(settings, "broker_mode", "ibkr_paper")
    return store


def _open_trade(ticker="AAPL", action="BUY"):
    return {"ticker": ticker, "action": action, "status": "OPEN",
            "entry_price": 100.0, "position_size_multiplier": 1.0,
            "recommendation_id": f"rec-{ticker}"}


def test_entry_submitted_and_idempotent(repo_store):
    repo_store["trades"] = [_open_trade("AAPL")]
    b = FakeBroker()
    r1 = rec.sync(broker=b)
    assert r1["entries_submitted"] == 1
    assert repo_store["trades"][0]["broker_order_id"] == "o1"
    assert repo_store["trades"][0]["broker_client_ref"] == "rec-AAPL"  # idempotency tag
    # Second run: trade already carries broker_order_id → no duplicate order.
    r2 = rec.sync(broker=b)
    assert r2["entries_submitted"] == 0
    assert len(b.orders) == 1


def test_entry_records_slippage(repo_store):
    repo_store["trades"] = [_open_trade("AAPL")]            # model price 100.0
    b = FakeBroker()                                        # fills at 100.05
    r = rec.sync(broker=b)
    assert r["slippage"] and r["slippage"][0]["ticker"] == "AAPL"
    assert r["slippage"][0]["bps"] == pytest.approx(5.0, abs=0.1)


def test_long_exit_uses_sell(repo_store):
    t = _open_trade("MSFT")
    t.update(status="CLOSED", broker_order_id="o1", broker_fill_qty=10)
    repo_store["trades"] = [t]
    b = FakeBroker(positions=[Position("MSFT", 10, 100.0)])
    r = rec.sync(broker=b)
    assert r["exits_submitted"] == 1
    assert b.orders[0].side == "SELL"
    assert repo_store["trades"][0]["broker_exit_order_id"] == "o1"


def test_short_entry_then_cover(repo_store):
    repo_store["trades"] = [_open_trade("TSLA", action="SELL")]
    b = FakeBroker()
    rec.sync(broker=b)
    assert b.orders[0].side == "SELL"             # opening a short
    repo_store["trades"][0]["status"] = "CLOSED"
    rec.sync(broker=b)
    assert b.orders[-1].side == "BUY"             # covering the short


def test_drift_reported(repo_store):
    repo_store["trades"] = []                                   # no internal opens
    b = FakeBroker(positions=[Position("NVDA", 5, 100.0)])
    r = rec.sync(broker=b)
    assert any(d["ticker"] == "NVDA" for d in r["drift"])


def test_off_mode_is_noop(monkeypatch):
    monkeypatch.setattr(settings, "broker_mode", "off")
    r = rec.sync()                                  # get_broker() → None
    assert r["entries_submitted"] == 0
    assert r["exits_submitted"] == 0
    assert r["connected"] is False
