"""Extended-hours Phase 1 (trade mode): scheduler tick plan + email gating,
session-stamped trade lifecycle, extended sizing haircut, session-aware
return normalisation, per-session performance rows, and off-RTH broker
submissions (marketable LMT + outsideRth).

The suite-wide conftest fixtures isolate the DB and pin commission_model='none'.
"""

from datetime import time

import pytest

from config.settings import settings
from src.performance.spread import _pct_return


# ── scheduler tick plan: observe vs trade, email gating ───────────────────

@pytest.mark.parametrize("mode,kind,slot,want_observe,want_email", [
    # trade mode (default): extended slots are FULL ticks, never email
    ("trade",   "extended", time(8, 0),   False, False),
    ("trade",   "extended", time(17, 0),  False, False),  # past 16:00 — the bug case
    ("trade",   "extended", time(20, 0),  False, False),
    ("trade",   "rth",      time(12, 0),  False, False),
    ("trade",   "rth",      time(16, 0),  False, True),   # closing slot emails
    # observe mode: extended slots are observation, never email
    ("observe", "extended", time(8, 0),   True,  False),
    ("observe", "extended", time(17, 0),  True,  False),
    ("observe", "rth",      time(16, 0),  False, True),
])
def test_tick_plan(monkeypatch, mode, kind, slot, want_observe, want_email):
    monkeypatch.setattr(settings, "extended_hours_mode", mode)
    monkeypatch.setattr(settings, "scheduler_email_every_tick", False)
    from src.scheduler.runner import _tick_plan
    observe, send_email = _tick_plan(kind, slot, time(16, 0))
    assert (observe, send_email) == (want_observe, want_email)


def test_tick_plan_email_every_tick_covers_all_slots(monkeypatch):
    """With email_every_tick on, EVERY slot emails — extended included
    (extended ticks are full trading ticks, their report is as real as an
    RTH one). With it off (parametrized test above), only the 16:00 closing
    RTH slot sends the daily report."""
    monkeypatch.setattr(settings, "extended_hours_mode", "trade")
    monkeypatch.setattr(settings, "scheduler_email_every_tick", True)
    from src.scheduler.runner import _tick_plan
    assert _tick_plan("rth", time(10, 0), time(16, 0)) == (False, True)
    assert _tick_plan("extended", time(17, 0), time(16, 0)) == (False, True)
    assert _tick_plan("extended", time(4, 0), time(16, 0)) == (False, True)


# ── _execution_iso: extended fills are real fills in trade mode ───────────

def test_execution_iso_extended_trade_mode(monkeypatch):
    from src.performance import market_calendar, tracker
    monkeypatch.setattr(tracker, "_now_iso", lambda: "2026-06-10T12:35:00+00:00")  # 08:35 ET
    monkeypatch.setattr(market_calendar, "current_session", lambda now=None: "extended")
    monkeypatch.setattr(settings, "extended_hours_mode", "trade")
    # trade mode + extended session: the fill is NOW, not snapped to 09:30.
    assert tracker._execution_iso() == "2026-06-10T12:35:00+00:00"
    # observe mode: legacy snap to the next regular open.
    monkeypatch.setattr(settings, "extended_hours_mode", "observe")
    assert tracker._execution_iso() != "2026-06-10T12:35:00+00:00"


# ── record_new_trades: entry_session stamp + extended sizing haircut ──────

def _mk_rec(ticker="TEST", action="BUY", confidence=0.78):
    from datetime import datetime as _dt, timezone as _tz
    from src.models import Recommendation
    return Recommendation(
        ticker=ticker, type="STOCK",
        direction="BULLISH" if action == "BUY" else "BEARISH",
        confidence=confidence, action=action, time_horizon="SWING",
        rationale="test", generated_at=_dt.now(_tz.utc),
    )


def test_record_new_trades_stamps_session_and_haircuts(monkeypatch):
    from src.performance import tracker
    monkeypatch.setattr(settings, "enable_intraday_timing", False)
    monkeypatch.setattr(settings, "enable_correlation_sizing", False)
    monkeypatch.setattr(settings, "extended_size_multiplier", 0.5)
    # Freeze an extended-session fill instant (08:35 ET) and the price feed.
    monkeypatch.setattr(tracker, "_execution_iso", lambda: "2026-06-10T12:35:00+00:00")
    monkeypatch.setattr(tracker, "_fetch_price", lambda t: 100.0)
    monkeypatch.setattr(tracker, "_reference_close", lambda t: None)

    diag = tracker.record_new_trades([_mk_rec()], run_id="testrun")
    assert diag["opened"] == 1
    assert diag["extended_haircut_applied"] == 1

    t = next(t for t in tracker._load_trades() if t["ticker"] == "TEST")
    assert t["entry_session"] == "extended"
    assert t["entry_datetime"] == "2026-06-10T12:35:00+00:00"      # not snapped
    # conf 0.78 = 1.0x tier, x0.5 extended haircut.
    assert t["position_size_multiplier"] == pytest.approx(0.5)
    assert t["extended_size_multiplier"] == pytest.approx(0.5)


def test_record_new_trades_rth_entry_no_haircut(monkeypatch):
    from src.performance import tracker
    monkeypatch.setattr(settings, "enable_intraday_timing", False)
    monkeypatch.setattr(settings, "enable_correlation_sizing", False)
    monkeypatch.setattr(settings, "extended_size_multiplier", 0.5)
    monkeypatch.setattr(tracker, "_execution_iso", lambda: "2026-06-10T15:00:00+00:00")  # 11:00 ET
    monkeypatch.setattr(tracker, "_fetch_price", lambda t: 50.0)
    monkeypatch.setattr(tracker, "_reference_close", lambda t: None)

    diag = tracker.record_new_trades([_mk_rec(ticker="RTHT")], run_id="testrun2")
    assert diag["opened"] == 1 and diag["extended_haircut_applied"] == 0
    t = next(t for t in tracker._load_trades() if t["ticker"] == "RTHT")
    assert t["entry_session"] == "rth"
    assert t["position_size_multiplier"] == pytest.approx(1.0)


# ── session-aware return normalisation ────────────────────────────────────

def test_normalize_closed_returns_honors_stored_sessions():
    from src.performance.tracker import _normalize_closed_returns
    trade = {
        "status": "CLOSED", "action": "BUY", "type": "STOCK",
        "entry_price": 100.0, "exit_price": 110.0,
        "entry_session": "extended", "exit_session": None,
        "return_pct": 0.0, "position_size_multiplier": 1.0,
    }
    assert _normalize_closed_returns([trade]) == 1
    # Same hand-math as the Phase-0 leg test: 12 bp extended entry, 3 bp RTH exit.
    assert trade["return_pct"] == pytest.approx(
        _pct_return("BUY", 100.0, 110.0, "STOCK", entry_session="extended"), abs=1e-3
    )


# ── performance table: per-session breakdown rows ─────────────────────────

def _mk_trade(ticker, entry_dt, ret, action="BUY"):
    return {
        "ticker": ticker, "type": "STOCK", "action": action, "direction": action,
        "status": "CLOSED", "entry_date": entry_dt[:10], "entry_datetime": entry_dt,
        "entry_price": 100.0, "exit_price": 100.0 * (1 + ret / 100.0),
        "exit_date": entry_dt[:10], "return_pct": ret,
        "position_size_multiplier": 1.0, "methods_agreeing": [],
    }


def test_performance_table_session_rows(monkeypatch):
    from src.performance import daily_nav, tracker
    # Daily-NAV compound needs OHLCV — stub it; we only check row presence.
    monkeypatch.setattr(daily_nav, "compute_compound_return", lambda trades, **kw: 0.0)
    trades = [
        _mk_trade("AAA", "2026-06-09T15:00:00+00:00", 2.0),    # 11:00 ET = rth
        _mk_trade("BBB", "2026-06-09T21:00:00+00:00", -1.0),   # 17:00 ET = extended
    ]
    rows = tracker._compute_performance_table(trades)
    session_rows = {r["label"]: r for r in rows if r["group"] == "session"}
    assert session_rows["Regular hours (RTH) only"]["trades"] == 1
    assert session_rows["Extended hours only"]["trades"] == 1

    # RTH-only ledger: no session group (it would duplicate All Trades).
    rows_rth = tracker._compute_performance_table(
        [_mk_trade("AAA", "2026-06-09T15:00:00+00:00", 2.0)]
    )
    assert not [r for r in rows_rth if r["group"] == "session"]


# ── broker: off-RTH submissions are marketable LMT + outsideRth ───────────

class _RecordingBroker:
    name = "fake"

    def __init__(self):
        self.requests = []

    def connect(self):
        return True

    def is_connected(self):
        return True

    def get_account(self):
        from src.broker.base import AccountSnapshot
        return AccountSnapshot(equity=100000.0, cash=100000.0, buying_power=100000.0,
                               account_id="DU000", currency="USD")

    def get_positions(self):
        return []

    def get_fills(self):
        return []

    def submit_order(self, req):
        from src.broker.base import OrderResult
        self.requests.append(req)
        return OrderResult(ok=True, ticker=req.ticker, side=req.side,
                           requested_qty=req.quantity, filled_qty=req.quantity,
                           avg_fill_price=req.limit_price or 100.0,
                           order_id="1", client_ref=req.client_ref, status="Filled")


def _broker_env(monkeypatch, session: str):
    from src.broker import reconcile
    from src.performance import market_calendar
    monkeypatch.setattr(settings, "broker_mode", "ibkr_paper")
    monkeypatch.setattr(settings, "broker_order_type", "MKT")   # RTH default stays MKT
    monkeypatch.setattr(settings, "broker_limit_cap_bps", 20.0)
    monkeypatch.setattr(settings, "broker_limit_cap_bps_extended", 80.0)
    monkeypatch.setattr(settings, "broker_base_notional_ccy", "USD")
    monkeypatch.setattr(settings, "broker_base_notional", 1000.0)
    monkeypatch.setattr(settings, "broker_sizing_mode", "notional")
    monkeypatch.setattr(settings, "broker_settle_seconds", 0)   # settle has its own suite
    monkeypatch.setattr(market_calendar, "current_session", lambda now=None: session)
    fake = _RecordingBroker()
    monkeypatch.setattr(reconcile, "get_broker", lambda: fake)
    return reconcile, fake


def test_reconcile_extended_session_forces_lmt_outside_rth(monkeypatch):
    reconcile, fake = _broker_env(monkeypatch, "extended")
    trade = {
        "ticker": "TEST", "type": "STOCK", "action": "BUY", "status": "OPEN",
        "entry_price": 100.0, "position_size_multiplier": 1.0,
        "recommendation_id": "abc123", "run_id": "r1",
    }
    report = reconcile.sync(run_id="r1", trades=[trade])
    assert report["entries_submitted"] == 1
    req = fake.requests[0]
    assert req.order_type == "LMT"
    assert req.outside_rth is True
    # BUY cap off-RTH uses the EXTENDED cap: 100 × (1 + 80 bp) = 100.80 —
    # the 20 bp RTH cap sits inside the ~4× wider extended spread and would
    # rest unfilled every time (the order must fill in its decision tick).
    assert req.limit_price == pytest.approx(100.80, abs=1e-6)


def test_reconcile_rth_keeps_configured_mkt(monkeypatch):
    reconcile, fake = _broker_env(monkeypatch, "rth")
    trade = {
        "ticker": "TEST", "type": "STOCK", "action": "BUY", "status": "OPEN",
        "entry_price": 100.0, "position_size_multiplier": 1.0,
        "recommendation_id": "abc124", "run_id": "r2",
    }
    report = reconcile.sync(run_id="r2", trades=[trade])
    assert report["entries_submitted"] == 1
    req = fake.requests[0]
    assert req.order_type == "MKT" and req.outside_rth is False and req.limit_price is None
