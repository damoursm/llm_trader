"""Settle pass — fill fast or kill (2026-06-12).

An order either executes within ~its decision window or it does not exist:
the settle pass watches this tick's unfilled submissions, repairs fills the
moment they land, re-anchors zero-fill orders at a fresh quote EARLY and
REPEATEDLY (every ``broker_settle_reanchor_every`` polls, 2026-07-07), and
CANCELS survivors at the deadline so nothing ever rests across ticks to fill
late at a stale price. Plus: connect retries at sync start (gateway re-login
window). All fakes, no sleeps, no network.
"""

import pytest

from config.settings import settings
from src.broker.base import AccountSnapshot, FillSummary, OrderResult, Position


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
    monkeypatch.setattr(settings, "broker_limit_cap_bps_extended", 80.0)
    monkeypatch.setattr(settings, "broker_base_notional_ccy", "USD")
    monkeypatch.setattr(settings, "broker_base_notional", 1000.0)
    monkeypatch.setattr(settings, "broker_sizing_mode", "notional")
    monkeypatch.setattr(settings, "broker_submit_retries", 0)
    monkeypatch.setattr(settings, "broker_connect_retries", 0)
    monkeypatch.setattr(settings, "broker_unfilled_cancel_minutes", 0)
    monkeypatch.setattr(settings, "broker_tick_scoped_orders", True)
    monkeypatch.setattr(settings, "broker_settle_seconds", 60)   # 12 polls, re-anchor at 6
    monkeypatch.setattr(settings, "broker_drift_action", "report")
    monkeypatch.setattr(market_calendar, "current_session", lambda now=None: "rth")


class _SettleBroker:
    """Submissions rest as 'Submitted'; fills appear in the executions feed
    after a configurable number of get_fills polls, keyed by client_ref."""
    name = "fake"

    def __init__(self, positions=(), fill_ref_after=None, cancel_ok=True):
        self.requests = []
        self.cancelled = []
        self.connect_calls = 0
        self._positions = list(positions)
        self._fills_calls = 0
        self._fill_ref_after = dict(fill_ref_after or {})   # ref -> appears after N polls
        self._cancel_ok = cancel_ok
        self._req_by_ref = {}

    def connect(self):
        self.connect_calls += 1
        return True

    def is_connected(self):
        return True

    def get_account(self):
        return AccountSnapshot(equity=100000.0, cash=100000.0, buying_power=100000.0,
                               account_id="DU000", currency="USD")

    def get_positions(self):
        return list(self._positions)

    def get_fills(self):
        self._fills_calls += 1
        out = []
        for ref, after in self._fill_ref_after.items():
            req = self._req_by_ref.get(ref)
            if req is not None and self._fills_calls >= after:
                out.append(FillSummary(client_ref=ref, ticker=req.ticker, side=req.side,
                                       filled_qty=req.quantity, avg_fill_price=100.05,
                                       commission=1.0))
        return out

    def get_open_orders(self):
        return []

    def cancel_order(self, client_ref):
        self.cancelled.append(client_ref)
        return self._cancel_ok

    def submit_order(self, req):
        self.requests.append(req)
        self._req_by_ref[req.client_ref] = req
        return OrderResult(ok=True, ticker=req.ticker, side=req.side,
                           requested_qty=req.quantity, filled_qty=0,
                           order_id=str(len(self.requests)), client_ref=req.client_ref,
                           status="Submitted")


def _sync(monkeypatch, broker, trades, live_price=100.0):
    from src.broker import reconcile
    monkeypatch.setattr(reconcile, "get_broker", lambda: broker)
    monkeypatch.setattr(reconcile, "_live_price", lambda t: live_price)
    return reconcile.sync(run_id="r1", trades=trades)


def _open_trade(**over):
    t = {"ticker": "TEST", "type": "STOCK", "action": "BUY", "status": "OPEN",
         "entry_price": 100.0, "current_price": 100.0,
         "position_size_multiplier": 1.0, "recommendation_id": "abc", "run_id": "r1"}
    t.update(over)
    return t


def test_settle_repairs_fill_that_lands_during_the_watch(monkeypatch):
    """Happy path: the entry fills 2 polls after submission — the leg is
    repaired in-tick (fill price + commission), nothing cancelled."""
    broker = _SettleBroker(fill_ref_after={"abc": 2})
    trade = _open_trade()
    report = _sync(monkeypatch, broker, [trade])
    assert report["settled_fills"] == 1
    assert report["unfilled_killed"] == 0
    assert broker.cancelled == []
    assert trade["broker_status"] == "Filled"
    assert trade["broker_fill_price"] == pytest.approx(100.05)
    assert trade["broker_commission"] == pytest.approx(1.0)


def test_settle_reanchors_repeatedly_then_kills_at_deadline(monkeypatch):
    """Never fills: the order is re-anchored at a FRESH quote REPEATEDLY (every
    `broker_settle_reanchor_every` polls, ask 1) within the budget — chasing the
    spread in bounded steps — then the survivor is killed at the deadline so
    nothing rests across ticks."""
    monkeypatch.setattr(settings, "broker_settle_seconds", 30)         # 10 polls @3s
    monkeypatch.setattr(settings, "broker_settle_poll_seconds", 3)
    monkeypatch.setattr(settings, "broker_settle_reanchor_every", 2)   # polls 2,4,6,8 → r1..r4
    broker = _SettleBroker()
    trade = _open_trade()
    report = _sync(monkeypatch, broker, [trade], live_price=101.0)
    assert report["settle_reanchors"] == 4     # re-anchored 4×, not once
    assert report["unfilled_killed"] == 1
    # original + each re-anchor cancelled in turn (last one killed at deadline)
    assert broker.cancelled == ["abc", "abc-r1", "abc-r2", "abc-r3", "abc-r4"]
    r1 = broker.requests[1]
    assert r1.client_ref == "abc-r1"
    assert r1.limit_price == pytest.approx(101.0 * 1.002, abs=0.01)   # fresh capped LMT
    assert trade["broker_order_id"] is None
    assert trade["broker_status"] == "UNFILLED_KILLED"
    assert trade["broker_resubmit_n"] == 5     # 4 re-anchors + kill → next tick is -r5


def test_settle_fill_on_a_reanchored_ref_is_repaired(monkeypatch):
    """A fill that lands on a re-anchored ref while it is the live order is
    repaired (not killed), even though the pass re-anchors repeatedly."""
    monkeypatch.setattr(settings, "broker_settle_seconds", 30)
    monkeypatch.setattr(settings, "broker_settle_poll_seconds", 3)
    monkeypatch.setattr(settings, "broker_settle_reanchor_every", 2)
    # abc-r1 is submitted at poll_i=1 (get_fills call 2) and live until poll_i=3;
    # make it fill at get_fills call 3 (poll_i=2), while it is the live ref.
    broker = _SettleBroker(fill_ref_after={"abc-r1": 3})
    trade = _open_trade()
    report = _sync(monkeypatch, broker, [trade], live_price=101.0)
    assert report["settled_fills"] == 1
    assert report["unfilled_killed"] == 0
    assert trade["broker_status"] == "Filled"
    assert trade["broker_client_ref"] == "abc-r1"


def test_killed_exit_is_resent_next_sync_sized_from_held(monkeypatch):
    """An exit killed at the deadline MUST come back next tick (the position
    has to flatten) — re-anchored under a fresh ref and sized from the held qty."""
    monkeypatch.setattr(settings, "broker_settle_seconds", 6)          # 2 polls @3s
    monkeypatch.setattr(settings, "broker_settle_poll_seconds", 3)
    monkeypatch.setattr(settings, "broker_settle_reanchor_every", 1)   # one re-anchor (r1) then kill
    broker = _SettleBroker(positions=[Position(ticker="TEST", quantity=10, avg_cost=100.0)])
    trade = _open_trade(
        status="CLOSED", exit_price=102.0,
        broker_order_id="9", broker_client_ref="abc", broker_status="Filled",
        broker_fill_qty=10, broker_fill_price=100.0, broker_commission=1.0,
        broker_requested_qty=10,
    )
    report1 = _sync(monkeypatch, broker, [trade], live_price=102.0)
    assert report1["exits_submitted"] == 1
    assert report1["unfilled_killed"] == 1
    assert trade["broker_exit_status"] == "UNFILLED_KILLED"
    assert trade["broker_exit_order_id"] is None

    report2 = _sync(monkeypatch, broker, [trade], live_price=102.5)
    assert report2["exits_submitted"] == 1
    # The kill bumped resubmit_n, so sync #2's resend carries the NEXT ref
    # (-r2) — fresh ref per cycle, sized from the actual held position.
    resent = next(r for r in broker.requests if r.client_ref == "abc-exit-r2")
    assert resent.side == "SELL" and resent.quantity == 10
    assert resent.limit_price == pytest.approx(102.5 * 0.998, abs=0.01)  # re-anchored at the LIVE quote


def test_settle_disabled_keeps_legacy_resting_behavior(monkeypatch):
    monkeypatch.setattr(settings, "broker_settle_seconds", 0)
    broker = _SettleBroker()
    trade = _open_trade()
    report = _sync(monkeypatch, broker, [trade])
    assert report["unfilled_killed"] == 0
    assert broker.cancelled == []
    assert trade["broker_status"] == "Submitted"   # rests until the next tick


# ── connect retries ────────────────────────────────────────────────────────

class _FlakyConnect(_SettleBroker):
    def __init__(self, fail_first_n):
        super().__init__()
        self._fail_n = fail_first_n

    def connect(self):
        self.connect_calls += 1
        return self.connect_calls > self._fail_n


def test_connect_retry_survives_gateway_relogin_window(monkeypatch):
    monkeypatch.setattr(settings, "broker_connect_retries", 2)
    monkeypatch.setattr(settings, "broker_connect_retry_wait_seconds", 1)
    broker = _FlakyConnect(fail_first_n=1)
    report = _sync(monkeypatch, broker, [])
    assert report["connected"] is True
    assert broker.connect_calls == 2


def test_no_retries_keeps_old_single_attempt(monkeypatch):
    monkeypatch.setattr(settings, "broker_connect_retries", 0)
    broker = _FlakyConnect(fail_first_n=1)
    report = _sync(monkeypatch, broker, [])
    assert report["connected"] is False
    assert "not connected" in report["errors"]
    assert broker.connect_calls == 1
