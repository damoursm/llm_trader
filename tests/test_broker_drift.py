"""Drift prevention + auto-reconciliation:
(P1) working ENTRY orders are cancelled the moment their trade closes,
(P2) exits flatten what the broker ACTUALLY holds (sign-checked),
(P3) in-flight closes are not reported as drift,
(P4) true orphans are auto-flattened via a price-capped LMT at a live quote
     (report-only is configurable; ibkr_live always refuses auto-flatten),
(P5) a flatten that already FILLED is never submitted twice — a second
     same-side order would flip the position, not flatten it,
(P6) duplicate ledger trades sharing one client_ref produce exactly ONE broker
     order per leg (IBKR does not dedupe orderRef; a twin exit sized from the
     full held position would flip the book short),
(P7) tick-scoped order lifetime — an unfilled order from a previous tick is
     cancelled and re-decided from THIS tick's data; dead/expired orders are
     recovered instead of resting as zombies.
All fakes, no sleeps, no network.
"""

from datetime import datetime, timedelta, timezone

import pytest

from config.settings import settings
from src.broker.base import (AccountSnapshot, FillSummary, OpenOrderInfo,
                             OrderResult, Position)


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
    monkeypatch.setattr(settings, "broker_tick_scoped_orders", True)
    # These tests target the tick-scope/dedupe/drift passes — the settle pass
    # (fill-fast-or-kill) has its own suite (test_broker_settle.py).
    monkeypatch.setattr(settings, "broker_settle_seconds", 0)
    monkeypatch.setattr(settings, "broker_drift_action", "flatten")
    monkeypatch.setattr(market_calendar, "current_session", lambda now=None: "rth")


class _FakeBroker:
    name = "fake"

    def __init__(self, positions=(), open_orders=(), cancel_ok=True, fills=()):
        self.requests = []
        self.cancelled = []
        self._positions = list(positions)
        self._open_orders = list(open_orders)
        self._cancel_ok = cancel_ok
        self._fills = list(fills)

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
    assert report["drift"][0]["action"] == "flatten_skipped"


def test_no_live_quote_reports_without_trading(monkeypatch):
    broker = _FakeBroker(positions=[Position(ticker="VGT", quantity=7, avg_cost=110.0)])
    report = _sync(monkeypatch, broker, [], live_price=None)
    assert broker.requests == []
    assert report["drift"][0]["action"] == "flatten_skipped"


# ── P5: an already-filled flatten is never submitted twice ────────────────

def _iso_minutes_ago(minutes):
    return (datetime.now(timezone.utc) - timedelta(minutes=minutes)).isoformat()


def _drift_fill(ticker, side, minutes_ago, qty=32, ref="r0"):
    return FillSummary(client_ref=f"drift-{ticker}-{ref}", ticker=ticker,
                       side=side, filled_qty=qty, avg_fill_price=100.0,
                       last_fill_at=_iso_minutes_ago(minutes_ago))


def test_recent_same_side_flatten_fill_blocks_resubmit(monkeypatch):
    """The TRUP incident (2026-06-11): the short's BUY flatten filled after the
    previous tick's poll window; the next tick's stale positions snapshot still
    showed −32 and a SECOND BUY flipped the account +32 long. A same-side drift
    fill within the guard window must stand the flatten down for the tick."""
    broker = _FakeBroker(
        positions=[Position(ticker="TRUP", quantity=-32, avg_cost=22.0)],
        fills=[_drift_fill("TRUP", "BUY", minutes_ago=16)],
    )
    report = _sync(monkeypatch, broker, [])
    assert broker.requests == []
    assert report["drift_flattened"] == 0
    assert report["drift"][0]["action"] == "flatten_skipped"


def test_opposite_side_fill_never_blocks_the_correction(monkeypatch):
    """If a double-flatten DID flip the position (+32 long after two BUYs), the
    corrective SELL must not be blocked by the earlier BUY fill — and fills on
    non-drift refs (normal exits) never count either."""
    broker = _FakeBroker(
        positions=[Position(ticker="TRUP", quantity=32, avg_cost=23.28)],
        fills=[
            _drift_fill("TRUP", "BUY", minutes_ago=16),
            FillSummary(client_ref="abc-exit", ticker="TRUP", side="SELL",
                        filled_qty=32, avg_fill_price=100.0,
                        last_fill_at=_iso_minutes_ago(5)),
        ],
    )
    report = _sync(monkeypatch, broker, [])
    assert len(broker.requests) == 1
    req = broker.requests[0]
    assert req.side == "SELL" and req.quantity == 32
    assert report["drift_flattened"] == 1


def test_old_same_side_fill_does_not_block_new_drift(monkeypatch):
    """A same-side drift fill from hours ago alongside a position still in a
    fresh snapshot is genuine residue / new drift — the flatten proceeds."""
    broker = _FakeBroker(
        positions=[Position(ticker="TRUP", quantity=-32, avg_cost=22.0)],
        fills=[_drift_fill("TRUP", "BUY", minutes_ago=8 * 60)],
    )
    report = _sync(monkeypatch, broker, [])
    assert len(broker.requests) == 1 and broker.requests[0].side == "BUY"
    assert report["drift_flattened"] == 1


def test_fill_with_missing_timestamp_blocks_conservatively(monkeypatch):
    broker = _FakeBroker(
        positions=[Position(ticker="TRUP", quantity=-32, avg_cost=22.0)],
        fills=[FillSummary(client_ref="drift-TRUP-r0", ticker="TRUP",
                           side="BUY", filled_qty=32, last_fill_at=None)],
    )
    report = _sync(monkeypatch, broker, [])
    assert broker.requests == []
    assert report["drift"][0]["action"] == "flatten_skipped"


def test_unreadable_fills_fail_closed(monkeypatch):
    """When the execution feed can't be read, don't trade blind — skip the tick."""
    class _Boom(_FakeBroker):
        def get_fills(self):
            raise RuntimeError("reqExecutions timeout")

    broker = _Boom(positions=[Position(ticker="TRUP", quantity=-32, avg_cost=22.0)])
    report = _sync(monkeypatch, broker, [])
    assert broker.requests == []
    assert report["drift"][0]["action"] == "flatten_skipped"


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


# ── P7: tick-scoped order lifetime + dead-order recovery ──────────────────

def _resting_entry(**over):
    t = {
        "ticker": "RDW", "type": "STOCK", "action": "BUY", "status": "OPEN",
        "entry_price": 100.0, "current_price": 105.0,
        "position_size_multiplier": 1.0,
        "recommendation_id": "abc", "run_id": "r0",
        "broker_order_id": "9", "broker_client_ref": "abc",
        "broker_status": "Submitted", "broker_fill_qty": 0,
        "broker_requested_qty": 7,
        "broker_submitted_at": _iso_minutes_ago(40),   # previous tick
    }
    t.update(over)
    return t


def test_previous_tick_entry_recancelled_and_reanchored_this_tick(monkeypatch):
    """Tick-scoped lifetime: an entry resting from the last tick is cancelled
    and resubmitted THIS tick at the current mark — never left working the
    book at a previous tick's price (age rule is 0 = disabled here)."""
    broker = _FakeBroker(
        open_orders=[OpenOrderInfo(client_ref="abc", order_id="9",
                                   status="Submitted", ticker="RDW", side="BUY")],
    )
    trade = _resting_entry()
    report = _sync(monkeypatch, broker, [trade], live_price=105.0)
    assert broker.cancelled == ["abc"]
    assert report["stale_cancels"] == 1
    assert report["entries_submitted"] == 1
    req = broker.requests[0]
    assert req.client_ref == "abc-r1"
    # Re-anchored at current_price (105), not the stale entry price (100).
    assert req.limit_price == pytest.approx(105.0 * 1.002, abs=0.01)


def test_order_submitted_this_tick_is_not_recancelled(monkeypatch):
    """The tick-scope boundary is sync start — an order from THIS tick rests."""
    broker = _FakeBroker(
        open_orders=[OpenOrderInfo(client_ref="abc", order_id="9",
                                   status="Submitted", ticker="RDW", side="BUY")],
    )
    from datetime import datetime, timezone, timedelta
    future = (datetime.now(timezone.utc) + timedelta(seconds=30)).isoformat()
    trade = _resting_entry(broker_submitted_at=future)
    report = _sync(monkeypatch, broker, [trade])
    assert broker.cancelled == []
    assert report["stale_cancels"] == 0
    assert broker.requests == []          # leg intact — nothing resubmitted


def test_expired_entry_cleared_and_resubmitted_when_positionless(monkeypatch):
    """Every off-RTH order is a DAY LMT that dies at the session close: the
    cancel finds nothing working, no fills, no position → the leg must be
    cleared and re-sent, not rest as a zombie 'Submitted' forever."""
    broker = _FakeBroker(cancel_ok=False)   # nothing working, no fills, no position
    trade = _resting_entry()
    report = _sync(monkeypatch, broker, [trade], live_price=105.0)
    assert report["stale_cancels"] == 1
    assert report["entries_submitted"] == 1
    assert broker.requests[0].client_ref == "abc-r1"


def test_expired_entry_with_backing_position_is_parked_not_resubmitted(monkeypatch):
    """Dead order but a position exists (e.g. filled late yesterday, outside
    the day-scoped executions feed) — resubmitting would DOUBLE it."""
    broker = _FakeBroker(
        cancel_ok=False,
        positions=[Position(ticker="RDW", quantity=7, avg_cost=100.0)],
    )
    trade = _resting_entry()
    report = _sync(monkeypatch, broker, [trade])
    assert broker.requests == []
    assert trade["broker_order_id"] == "9"   # leg parked, not cleared
    assert report["drift"] == []             # open trade owns the position


def test_dead_exit_with_position_held_resubmitted(monkeypatch):
    """An expired exit with the position still held MUST re-send — the
    position has to flatten."""
    broker = _FakeBroker(
        cancel_ok=False,
        positions=[Position(ticker="RDW", quantity=7, avg_cost=100.0)],
    )
    trade = _resting_entry(
        status="CLOSED", exit_price=104.0,
        broker_status="Filled", broker_fill_qty=7, broker_fill_price=100.0,
        broker_commission=0.35,
        broker_exit_order_id="11", broker_exit_client_ref="abc-exit",
        broker_exit_status="Submitted", broker_exit_fill_qty=0,
        broker_exit_requested_qty=7,
        broker_exit_submitted_at=_iso_minutes_ago(40),
    )
    report = _sync(monkeypatch, broker, [trade], live_price=104.0)
    assert report["stale_cancels"] == 1
    assert report["exits_submitted"] == 1
    req = broker.requests[0]
    assert req.side == "SELL" and req.quantity == 7
    assert req.client_ref == "abc-exit-r1"


def test_dead_exit_with_nothing_held_stamped_terminal(monkeypatch):
    """Dead exit and the position is already gone — resubmitting an exit sized
    from the recorded entry fill would open a fresh short. Stamp it terminal."""
    broker = _FakeBroker(cancel_ok=False)
    trade = _resting_entry(
        status="CLOSED", exit_price=104.0,
        broker_status="Filled", broker_fill_qty=7, broker_fill_price=100.0,
        broker_commission=0.35,
        broker_exit_order_id="11", broker_exit_client_ref="abc-exit",
        broker_exit_status="Submitted", broker_exit_fill_qty=0,
        broker_exit_requested_qty=7,
        broker_exit_submitted_at=_iso_minutes_ago(40),
    )
    report = _sync(monkeypatch, broker, [trade])
    assert broker.requests == []
    assert report["exits_submitted"] == 0
    assert trade["broker_exit_status"] == "Cancelled"
    assert trade["broker_exit_cancel_reason"] == "expired_nothing_held"


# ── P6: duplicate ledger trades sharing one client_ref ────────────────────

def _xle_twin(**over):
    base = {
        "ticker": "XLE", "type": "ETF", "action": "BUY", "status": "OPEN",
        "entry_price": 57.12, "position_size_multiplier": 0.75,
        "recommendation_id": "e6ac0487a035e3fb", "run_id": "r1",
    }
    base.update(over)
    return base


def test_duplicate_open_trades_submit_one_entry(monkeypatch):
    """Two OPEN twins sharing a recommendation_id → exactly one broker order;
    the second is marked DUPLICATE_REF_NOT_SUBMITTED and never sent."""
    broker = _FakeBroker()
    t1, t2 = _xle_twin(), _xle_twin()
    report = _sync(monkeypatch, broker, [t1, t2], live_price=57.12)
    assert report["entries_submitted"] == 1
    assert len(broker.requests) == 1
    assert t1.get("broker_order_id")
    assert not t2.get("broker_order_id")
    assert t2["broker_status"] == "DUPLICATE_REF_NOT_SUBMITTED"


def test_duplicate_ref_skip_is_durable_across_ticks(monkeypatch):
    """On a later tick the marked twin must STILL not submit (a same-ref order
    would reach the broker then, since the per-tick set has reset)."""
    broker = _FakeBroker()
    twin = _xle_twin(broker_status="DUPLICATE_REF_NOT_SUBMITTED")
    report = _sync(monkeypatch, broker, [twin], live_price=57.12)
    assert broker.requests == []
    assert report["entries_submitted"] == 0


def test_duplicate_closed_twins_exit_once_not_double(monkeypatch):
    """The 2026-06-11 XLE incident: both CLOSED twins sized their exit from the
    FULL held position (18) → sold 36, account flipped −18 short. One exit only."""
    broker = _FakeBroker(positions=[Position(ticker="XLE", quantity=18, avg_cost=57.15)])
    base = dict(
        status="CLOSED", exit_price=57.10,
        broker_client_ref="e6ac0487a035e3fb", broker_status="Filled",
        broker_fill_qty=9, broker_requested_qty=9,
    )
    t1 = _xle_twin(**base, broker_order_id="72")
    t2 = _xle_twin(**base, broker_order_id="74")
    report = _sync(monkeypatch, broker, [t1, t2], live_price=57.10)
    assert report["exits_submitted"] == 1
    sells = [r for r in broker.requests if r.side == "SELL"]
    assert len(sells) == 1 and sells[0].quantity == 18
    assert t1.get("broker_exit_order_id")
    assert not t2.get("broker_exit_order_id")
    assert t2["broker_exit_status"] == "DUPLICATE_REF_NOT_SUBMITTED"
    assert report["drift"] == []   # position fully explained — nothing flipped
