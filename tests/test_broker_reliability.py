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
    broker = _ScriptedBroker(["transient", "ok"])
    # The order only becomes visible at the broker once it has been sent — the
    # errored first attempt DID reach it. (Pre-populating this would instead
    # exercise the pre-submit guard below.)
    _orig_submit = broker.submit_order

    def _submit(req):
        res = _orig_submit(req)
        broker._open_orders = [OpenOrderInfo(client_ref="ref1", order_id="42",
                                             status="Submitted", ticker="TEST",
                                             side="BUY")]
        return res

    broker.submit_order = _submit
    report = reconcile._new_report()
    res = reconcile._submit_with_retry(broker, _req(), 100.0, report, "ENTRY")
    assert res.ok and res.order_id == "42" and res.status == "Submitted"
    assert len(broker.requests) == 1          # never resubmitted
    # The adopted working order completes via the fill-refresh pass later.


def test_presubmit_guard_never_stacks_a_second_order_on_one_ref(monkeypatch):
    """A ref that ALREADY has a working order is adopted without submitting.

    IBKR does not dedupe orderRef: every submission is an independent order
    that fills independently. On 2026-07-23 a watchdog-killed reconcile lost
    the record of orders it had placed, so later ticks resubmitted the same
    ref — 8 live orders stacked on one HQY exit ref and filled together,
    selling 112 shares against a 14-share long.
    """
    from src.broker import reconcile
    broker = _ScriptedBroker(
        ["ok"],
        open_orders=[OpenOrderInfo(client_ref="ref1", order_id="42",
                                   status="Submitted", ticker="TEST", side="BUY")],
    )
    report = reconcile._new_report()
    res = reconcile._submit_with_retry(broker, _req(), 100.0, report, "ENTRY")
    assert res.ok and res.order_id == "42"
    assert broker.requests == []                          # nothing was sent
    assert report["duplicate_submits_blocked"] == 1


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


# ── orphan-order sweep: no working order may outlive its ledger leg ────────
#
# The 2026-07-23 drift incident: the reconcile watchdog force-exits with
# os._exit(1), so a kill between placing an order and persisting the ledger
# left the order working with nothing pointing at it. Later ticks resubmitted
# the same ref, the duplicates filled together, and the resulting position
# read as "drifted from the ledger" every tick thereafter.

def _owned_leg(**kw):
    base = {"ticker": "OWNED", "action": "BUY", "status": "OPEN",
            "broker_order_id": "1", "broker_client_ref": "owned-ref",
            "broker_status": "Submitted"}
    base.update(kw)
    return base


def test_orphan_sweep_cancels_only_unowned_refs(monkeypatch):
    from src.broker import reconcile
    monkeypatch.setattr(settings, "broker_orphan_order_sweep", True)
    broker = _ScriptedBroker(
        [],
        open_orders=[
            OpenOrderInfo(client_ref="owned-ref", order_id="1",
                          status="Submitted", ticker="OWNED", side="BUY"),
            OpenOrderInfo(client_ref="ghost-ref", order_id="2",
                          status="Submitted", ticker="GHOST", side="SELL"),
            OpenOrderInfo(client_ref="drift-XYZ-2026-07-23_120000", order_id="3",
                          status="Submitted", ticker="XYZ", side="SELL"),
        ],
    )
    report = reconcile._new_report()
    reconcile._sweep_orphan_orders(broker, [_owned_leg()], report)
    # The live leg is spared; the drift flatten is _flatten_orphan's business.
    assert broker.cancelled == ["ghost-ref"]
    assert report["orphan_orders_cancelled"] == 1


def test_orphan_sweep_cancels_leg_whose_status_went_terminal(monkeypatch):
    """A leg cleared/killed by an earlier pass no longer owns its order."""
    from src.broker import reconcile
    monkeypatch.setattr(settings, "broker_orphan_order_sweep", True)
    broker = _ScriptedBroker(
        [],
        open_orders=[OpenOrderInfo(client_ref="owned-ref", order_id="1",
                                   status="Submitted", ticker="OWNED", side="BUY")],
    )
    report = reconcile._new_report()
    reconcile._sweep_orphan_orders(broker, [_owned_leg(broker_status="Cancelled")], report)
    assert broker.cancelled == ["owned-ref"]


def test_orphan_sweep_can_be_disabled(monkeypatch):
    from src.broker import reconcile
    monkeypatch.setattr(settings, "broker_orphan_order_sweep", False)
    broker = _ScriptedBroker(
        [],
        open_orders=[OpenOrderInfo(client_ref="ghost-ref", order_id="2",
                                   status="Submitted", ticker="GHOST", side="SELL")],
    )
    report = reconcile._new_report()
    reconcile._sweep_orphan_orders(broker, [], report)
    assert broker.cancelled == []


def test_submission_persists_the_leg_immediately(monkeypatch):
    """An order that is live at the broker must be in the ledger before the
    next line runs — the watchdog can os._exit at any moment."""
    from src.broker import reconcile
    saved = []
    monkeypatch.setattr(reconcile.repo, "save_trades",
                        lambda trades: saved.append([dict(t) for t in trades]))
    trades = [{"ticker": "T", "broker_order_id": None}]
    reconcile._persist_legs(trades)
    assert len(saved) == 1 and saved[0][0]["ticker"] == "T"


def test_persist_legs_never_raises(monkeypatch):
    """A DuckDB hiccup must not abort a reconcile mid-order-cycle."""
    from src.broker import reconcile

    def _boom(_trades):
        raise RuntimeError("db locked")

    monkeypatch.setattr(reconcile.repo, "save_trades", _boom)
    reconcile._persist_legs([{"ticker": "T"}])   # must not propagate


# ── settle pass: the budget is WALL-CLOCK, not a poll count ────────────────
#
# broker_settle_seconds was spent as `n_polls = budget // poll_seconds`, which
# silently assumed the work inside a poll was free. Each poll actually cancels
# and re-submits every unfilled leg (~13 s each against a real gateway), so on
# 2026-07-23 a 30 s budget ran 10–20 min with ~9 legs. Every tick then blew the
# 600 s reconcile watchdog, which force-exited mid-order-cycle and orphaned the
# orders it had just placed — the root of the drift cascade.

def test_settle_stops_at_the_wall_clock_budget(monkeypatch):
    from src.broker import reconcile

    monkeypatch.setattr(settings, "broker_settle_seconds", 30)
    monkeypatch.setattr(settings, "broker_settle_poll_seconds", 3)
    monkeypatch.setattr(settings, "broker_settle_reanchor_every", 2)
    monkeypatch.setattr(reconcile, "_live_price", lambda _t: 100.0)

    clock = {"t": 0.0}
    monkeypatch.setattr(reconcile.time, "monotonic", lambda: clock["t"])
    monkeypatch.setattr(reconcile.time, "sleep",
                        lambda s: clock.__setitem__("t", clock["t"] + s))

    broker = _ScriptedBroker([])
    _orig_submit = broker.submit_order

    def _slow_submit(req):                    # a re-anchor costs real seconds
        clock["t"] += 15.0
        return _orig_submit(req)

    broker.submit_order = _slow_submit

    sync_started = datetime.now(timezone.utc) - timedelta(minutes=5)
    leg = {
        "ticker": "T", "action": "BUY", "status": "OPEN",
        "broker_order_id": "1", "broker_client_ref": "r0",
        "broker_status": "Submitted", "broker_fill_qty": 0,
        "broker_requested_qty": 10,
        "broker_submitted_at": datetime.now(timezone.utc).isoformat(),
    }
    report = reconcile._new_report()
    reconcile._settle_unfilled_this_tick(broker, [leg], report, False, sync_started)

    # 10 polls x 15 s of submit work would be minutes. The deadline caps it:
    # only the re-anchors that start inside the 30 s budget are allowed.
    assert clock["t"] <= 30 + 15          # at most one re-anchor may overrun
    assert len(broker.requests) <= 2
