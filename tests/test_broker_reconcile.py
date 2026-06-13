"""Tests for src.broker.reconcile.sync — shadow execution + idempotency + drift.

Uses a FakeBroker and a monkeypatched repo so nothing touches IB Gateway or the
real trades database.
"""

import pytest

import src.broker.reconcile as rec
from config.settings import settings
from src.broker.base import AccountSnapshot, Broker, FillSummary, OrderResult, Position


class FakeBroker(Broker):
    name = "fake"

    def __init__(self, positions=None, equity=100_000.0, account_id="DU0000001"):
        self._positions = positions or []
        self._equity = equity
        self._account_id = account_id
        self.orders = []
        self.fills = []          # FillSummary list served by get_fills()

    def connect(self):
        return True

    def is_connected(self):
        return True

    def get_account(self):
        return AccountSnapshot(self._equity, self._equity, self._equity, self._account_id)

    def get_positions(self):
        return list(self._positions)

    def get_fills(self):
        return list(self.fills)

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
    monkeypatch.setattr(settings, "broker_sizing_mode", "notional")
    monkeypatch.setattr(settings, "broker_base_notional", 500.0)
    # Patch FX so tests never hit the network (1.0 = treat base notional as USD).
    monkeypatch.setattr(rec, "usd_per_unit", lambda ccy: 1.0)
    # Deterministic session: caps and order typing are session-aware, so pin
    # RTH regardless of the wall-clock the suite runs at.
    from src.performance import market_calendar
    monkeypatch.setattr(market_calendar, "current_session", lambda now=None: "rth")
    # Settle pass (fill-fast-or-kill) has its own suite — disabled here so
    # these tests exercise the submit/refresh passes in isolation (and never
    # real-sleep through the watch loop).
    monkeypatch.setattr(settings, "broker_settle_seconds", 0)
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


def test_notional_sizing_quantity(repo_store, monkeypatch):
    monkeypatch.setattr(rec, "usd_per_unit", lambda ccy: 0.73)   # CAD→USD
    t = _open_trade("AAPL")
    t["entry_price"] = 50.0
    repo_store["trades"] = [t]
    b = FakeBroker()
    rec.sync(broker=b)
    assert b.orders[0].quantity == 7      # floor(500 × 0.73 / 50) = 7


def test_drift_reported(repo_store):
    repo_store["trades"] = []                                   # no internal opens
    b = FakeBroker(positions=[Position("NVDA", 5, 100.0)])
    r = rec.sync(broker=b)
    assert any(d["ticker"] == "NVDA" for d in r["drift"])


def test_safety_stop_on_non_paper_account(repo_store):
    # broker_mode=ibkr_paper but connected to a live (U-prefixed) account → refuse all orders.
    repo_store["trades"] = [_open_trade("AAPL")]
    b = FakeBroker(account_id="U26348620")
    r = rec.sync(broker=b)
    assert r["ok"] is False
    assert r["entries_submitted"] == 0
    assert b.orders == []


def test_off_mode_is_noop(monkeypatch):
    monkeypatch.setattr(settings, "broker_mode", "off")
    r = rec.sync()                                  # get_broker() → None
    assert r["entries_submitted"] == 0
    assert r["exits_submitted"] == 0
    assert r["connected"] is False


# ── execution feedback: exit slippage, order rows, fill refresh, LMT ────────

def test_exit_records_slippage_and_order_row(repo_store):
    t = _open_trade("MSFT")
    t.update(status="CLOSED", exit_price=101.0,
             broker_order_id="o1", broker_status="Filled", broker_fill_qty=10,
             broker_fill_price=100.0, broker_commission=0.35, broker_requested_qty=10)
    repo_store["trades"] = [t]
    b = FakeBroker(positions=[Position("MSFT", 10, 100.0)])   # fills at 100.05
    r = rec.sync(broker=b)
    assert r["exits_submitted"] == 1
    exits = [s for s in r["slippage"] if s["intent"] == "EXIT"]
    assert exits and exits[0]["ticker"] == "MSFT"
    # SELL exit filled at 100.05 vs model 101 → received less → positive (adverse) bps
    assert exits[0]["bps"] == pytest.approx((101.0 - 100.05) / 101.0 * 10000, abs=0.5)
    rows = [o for o in r["orders"] if o["event"] == "SUBMIT" and o["intent"] == "EXIT"]
    assert rows and rows[0]["model_price"] == pytest.approx(101.0)
    assert rows[0]["fill_price"] == pytest.approx(100.05)


def test_entry_slippage_is_cost_normalized_for_shorts(repo_store):
    # Short entry (SELL) filled at 100.05 vs model 100 → sold HIGHER than model
    # → favourable → negative cost bps.
    repo_store["trades"] = [_open_trade("TSLA", action="SELL")]
    r = rec.sync(broker=FakeBroker())
    assert r["slippage"][0]["bps"] == pytest.approx(-5.0, abs=0.1)


def test_run_id_lands_on_report(repo_store):
    repo_store["trades"] = []
    r = rec.sync(broker=FakeBroker(), run_id="run-42")
    assert r["run_id"] == "run-42"


def test_limit_order_buy_caps_above_model(repo_store, monkeypatch):
    monkeypatch.setattr(settings, "broker_order_type", "LMT")
    monkeypatch.setattr(settings, "broker_limit_cap_bps", 20.0)
    repo_store["trades"] = [_open_trade("AAPL")]              # model 100.0
    b = FakeBroker()
    rec.sync(broker=b)
    req = b.orders[0]
    assert req.order_type == "LMT"
    assert req.limit_price == pytest.approx(100.20)           # +20 bps, away-rounded to a cent


def test_limit_order_sell_caps_below_model(repo_store, monkeypatch):
    monkeypatch.setattr(settings, "broker_order_type", "LMT")
    monkeypatch.setattr(settings, "broker_limit_cap_bps", 20.0)
    repo_store["trades"] = [_open_trade("TSLA", action="SELL")]
    b = FakeBroker()
    rec.sync(broker=b)
    assert b.orders[0].limit_price == pytest.approx(99.80)


def test_limit_cap_widens_off_rth(repo_store, monkeypatch):
    """Off-RTH the 20 bp RTH cap sits INSIDE the ~4× wider extended spread —
    the order would rest unfilled every time. The session-aware cap keeps
    off-RTH orders genuinely marketable in their decision tick."""
    from src.performance import market_calendar
    monkeypatch.setattr(market_calendar, "current_session", lambda now=None: "extended")
    monkeypatch.setattr(settings, "broker_order_type", "LMT")
    monkeypatch.setattr(settings, "broker_limit_cap_bps", 20.0)
    monkeypatch.setattr(settings, "broker_limit_cap_bps_extended", 80.0)
    repo_store["trades"] = [_open_trade("AAPL")]              # model 100.0
    b = FakeBroker()
    rec.sync(broker=b)
    req = b.orders[0]
    assert req.order_type == "LMT" and req.outside_rth
    assert req.limit_price == pytest.approx(100.80)           # +80 bps extended cap


def test_fill_refresh_repairs_stale_entry(repo_store):
    # Entry submitted on an earlier tick: order id recorded but no fill data yet
    # (e.g. queued overnight). The refresh pass repairs it from today's executions.
    t = _open_trade("AAPL")
    t.update(broker_order_id="o9", broker_client_ref="rec-AAPL",
             broker_status="Submitted", broker_fill_qty=0, broker_fill_price=None,
             broker_requested_qty=7)
    repo_store["trades"] = [t]
    b = FakeBroker()
    b.fills = [FillSummary(client_ref="rec-AAPL", ticker="AAPL", side="BUY",
                           filled_qty=7, avg_fill_price=100.10, commission=0.35)]
    r = rec.sync(broker=b)
    assert r["fills_repaired"] == 1
    tr = repo_store["trades"][0]
    assert tr["broker_fill_qty"] == 7
    assert tr["broker_fill_price"] == pytest.approx(100.10)
    assert tr["broker_commission"] == pytest.approx(0.35)
    assert tr["broker_status"] == "Filled"
    assert b.orders == []          # repaired, not resubmitted
    # The repaired fill contributes a slippage record and a FILL_REFRESH event row.
    assert any(s["intent"] == "ENTRY" and s["ticker"] == "AAPL" for s in r["slippage"])
    assert any(o["event"] == "FILL_REFRESH" for o in r["orders"])
    # Second sync: leg is terminal and complete → nothing repaired twice.
    r2 = rec.sync(broker=b)
    assert r2["fills_repaired"] == 0


def test_fill_refresh_repairs_partial_fill_then_completion(repo_store):
    t = _open_trade("AAPL")
    t.update(broker_order_id="o9", broker_client_ref="rec-AAPL",
             broker_status="Submitted", broker_fill_qty=0, broker_requested_qty=10)
    repo_store["trades"] = [t]
    b = FakeBroker()
    b.fills = [FillSummary(client_ref="rec-AAPL", ticker="AAPL", side="BUY",
                           filled_qty=4, avg_fill_price=100.0, commission=0.35)]
    rec.sync(broker=b)
    assert repo_store["trades"][0]["broker_status"] == "PartiallyFilled"
    # The rest fills later → still a candidate (non-terminal status) → repaired again.
    b.fills = [FillSummary(client_ref="rec-AAPL", ticker="AAPL", side="BUY",
                           filled_qty=10, avg_fill_price=100.05, commission=0.70)]
    r = rec.sync(broker=b)
    assert r["fills_repaired"] == 1
    assert repo_store["trades"][0]["broker_fill_qty"] == 10
    assert repo_store["trades"][0]["broker_status"] == "Filled"


def test_adopted_drift_row_is_quiet_until_close(repo_store):
    """A row adopted from drift (sentinel order id, RESTORED_ADOPTED status) must
    submit nothing and never be a fill-refresh candidate while open — and on
    close, the exit must size from the LIVE broker position (no stored fill qty)."""
    t = _open_trade("VGT")
    t.update(broker_order_id="ADOPTED_DRIFT_2026-06-10",
             broker_status="RESTORED_ADOPTED")
    repo_store["trades"] = [t]
    b = FakeBroker(positions=[Position("VGT", 6, 117.0)])
    b.fills = [FillSummary(client_ref="rec-VGT", ticker="VGT", side="BUY",
                           filled_qty=6, avg_fill_price=117.0, commission=0.35)]
    r = rec.sync(broker=b)
    assert b.orders == []                       # no entry resubmitted
    assert r["fills_repaired"] == 0             # not a refresh candidate
    assert r["drift"] == []                     # position matches the OPEN row

    t.update(status="CLOSED", exit_price=120.0)
    r2 = rec.sync(broker=b)
    assert r2["exits_submitted"] == 1
    assert b.orders[-1].side == "SELL"
    assert b.orders[-1].quantity == 6           # sized from the live position


def test_submit_order_rows_recorded_for_entries(repo_store):
    repo_store["trades"] = [_open_trade("AAPL")]
    r = rec.sync(broker=FakeBroker())
    rows = [o for o in r["orders"] if o["event"] == "SUBMIT" and o["intent"] == "ENTRY"]
    assert len(rows) == 1
    assert rows[0]["ticker"] == "AAPL"
    assert rows[0]["model_price"] == pytest.approx(100.0)
    assert rows[0]["slippage_bps"] == pytest.approx(5.0, abs=0.1)
    assert rows[0]["client_ref"] == "rec-AAPL"
