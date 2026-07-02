"""Overnight trading session (20:00 ET → 04:00 ET; IBKR overnight venue).

The overnight session becomes a first-class trading period: scheduler slots on
the venue's own calendar (Sunday night → Thursday night, 20:00–03:50 ET),
real execution timestamps in trade mode, a harder sizing haircut and
confidence bump, session-aware broker LMT caps, and OVERNIGHT-venue routing.
Fixed dates used: 2026-07-05 = Sunday, 2026-07-06 = Monday, 2026-07-10 =
Friday, 2026-07-03 = observed Independence Day holiday (Friday).
"""

from datetime import datetime, time

import pytest

from config.settings import settings


# ── market_calendar.is_overnight_session_open ────────────────────────────────

def test_overnight_session_calendar():
    from src.performance.market_calendar import is_overnight_session_open
    # Sunday evening leads into Monday → open.
    assert is_overnight_session_open(datetime(2026, 7, 5, 21, 0)) is True
    # Friday evening leads into Saturday → closed.
    assert is_overnight_session_open(datetime(2026, 7, 10, 21, 0)) is False
    # Weekday morning half on a market day → open.
    assert is_overnight_session_open(datetime(2026, 7, 7, 1, 0)) is True
    # Saturday morning → closed.
    assert is_overnight_session_open(datetime(2026, 7, 11, 1, 0)) is False
    # 03:50–04:00 venue gap → closed even on a market day.
    assert is_overnight_session_open(datetime(2026, 7, 7, 3, 55)) is False
    # Holiday eve: Thursday 2026-07-02 evening leads into the observed
    # Independence Day holiday (Friday 2026-07-03) → closed.
    assert is_overnight_session_open(datetime(2026, 7, 2, 21, 0)) is False
    # Daytime is never the overnight session.
    assert is_overnight_session_open(datetime(2026, 7, 7, 14, 0)) is False


# ── scheduler slots ───────────────────────────────────────────────────────────

def test_session_slots_include_overnight_when_on(monkeypatch):
    from src.scheduler import runner
    monkeypatch.setattr(settings, "overnight_hours_mode", "trade")
    slots, _end = runner._session_slots()
    on = [t for t, k in slots if k == "overnight"]
    assert time(20, 30) in on and time(1, 0) in on and time(3, 30) in on
    # Slots stay time-sorted, so the morning half leads the day.
    assert slots[0][0] < time(4, 0)


def test_session_slots_exclude_overnight_when_off(monkeypatch):
    from src.scheduler import runner
    monkeypatch.setattr(settings, "overnight_hours_mode", "off")
    slots, _end = runner._session_slots()
    assert not [t for t, k in slots if k == "overnight"]


def test_slot_validity_follows_venue_calendar():
    from src.scheduler.runner import _slot_is_valid
    from datetime import date
    # Sunday evening overnight slot: valid (leads into Monday).
    assert _slot_is_valid(date(2026, 7, 5), time(20, 30), "overnight") is True
    # Friday evening overnight slot: invalid (leads into Saturday).
    assert _slot_is_valid(date(2026, 7, 10), time(20, 30), "overnight") is False
    # Monday-morning overnight slot: valid.
    assert _slot_is_valid(date(2026, 7, 6), time(1, 0), "overnight") is True
    # Saturday-morning overnight slot: invalid.
    assert _slot_is_valid(date(2026, 7, 11), time(1, 0), "overnight") is False
    # RTH/extended slots still require the day itself to be a market day.
    assert _slot_is_valid(date(2026, 7, 5), time(9, 30), "rth") is False
    assert _slot_is_valid(date(2026, 7, 6), time(9, 30), "rth") is True


def test_current_slot_sunday_evening_and_friday_evening():
    from src.scheduler.runner import _current_slot
    slots = [(time(20, 30), "overnight"), (time(21, 30), "overnight")]
    # Sunday 21:00 → the 20:30 overnight slot fires.
    got = _current_slot(datetime(2026, 7, 5, 21, 0), slots)
    assert got == (datetime(2026, 7, 5, 20, 30), "overnight")
    # Friday 21:00 → no valid slot (no Friday-night session).
    assert _current_slot(datetime(2026, 7, 10, 21, 0), slots) is None


def test_tick_plan_overnight_modes(monkeypatch):
    from src.scheduler.runner import _tick_plan
    monkeypatch.setattr(settings, "scheduler_email_every_tick", False)
    monkeypatch.setattr(settings, "overnight_hours_mode", "trade")
    observe, email = _tick_plan("overnight", time(1, 0), time(16, 0))
    assert observe is False and email is False          # full tick, no email slot
    monkeypatch.setattr(settings, "overnight_hours_mode", "observe")
    observe, _ = _tick_plan("overnight", time(1, 0), time(16, 0))
    assert observe is True
    # The overnight mode knob must not affect extended slots.
    monkeypatch.setattr(settings, "extended_hours_mode", "trade")
    observe, _ = _tick_plan("extended", time(17, 0), time(16, 0))
    assert observe is False


# ── tracker: execution instant + sizing ───────────────────────────────────────

def test_execution_iso_overnight_trade_mode(monkeypatch):
    from src.performance import market_calendar, tracker
    monkeypatch.setattr(tracker, "_now_iso", lambda: "2026-07-07T01:30:00+00:00")
    monkeypatch.setattr(market_calendar, "current_session", lambda now=None: "overnight")
    monkeypatch.setattr(market_calendar, "is_overnight_session_open", lambda now=None: True)
    monkeypatch.setattr(settings, "overnight_hours_mode", "trade")
    assert tracker._execution_iso() == "2026-07-07T01:30:00+00:00"   # a real fill, now
    # Venue closed (Fri/Sat night) → snapped to the next regular open.
    monkeypatch.setattr(market_calendar, "is_overnight_session_open", lambda now=None: False)
    assert tracker._execution_iso() != "2026-07-07T01:30:00+00:00"
    # Observe mode → snapped as before.
    monkeypatch.setattr(market_calendar, "is_overnight_session_open", lambda now=None: True)
    monkeypatch.setattr(settings, "overnight_hours_mode", "observe")
    assert tracker._execution_iso() != "2026-07-07T01:30:00+00:00"


def _mk_rec(ticker="ONT", action="BUY", confidence=0.78):
    from datetime import timezone
    from src.models import Recommendation
    return Recommendation(
        ticker=ticker, type="STOCK",
        direction="BULLISH" if action == "BUY" else "BEARISH",
        confidence=confidence, action=action, time_horizon="SWING",
        rationale="test", generated_at=datetime.now(timezone.utc),
    )


def test_record_new_trades_overnight_sizing(monkeypatch):
    from src.performance import tracker
    monkeypatch.setattr(settings, "enable_intraday_timing", False)
    monkeypatch.setattr(settings, "enable_correlation_sizing", False)
    monkeypatch.setattr(settings, "overnight_size_multiplier", 0.25)
    # 01:30 UTC = 21:30 ET the previous evening → overnight session stamp.
    monkeypatch.setattr(tracker, "_execution_iso", lambda: "2026-07-07T01:30:00+00:00")
    monkeypatch.setattr(tracker, "_fetch_price", lambda t: 100.0)
    monkeypatch.setattr(tracker, "_reference_close", lambda t: None)

    diag = tracker.record_new_trades([_mk_rec()], run_id="on1")
    assert diag["opened"] == 1
    t = next(t for t in tracker._load_trades() if t["ticker"] == "ONT")
    assert t["entry_session"] == "overnight"
    # conf 0.78 = 1.0× tier, ×0.25 overnight haircut (not the ×0.5 extended one).
    assert t["position_size_multiplier"] == pytest.approx(0.25)


# ── broker: session-aware LMT cap + overnight venue routing ───────────────────

def test_limit_cap_uses_overnight_bps(monkeypatch):
    from src.performance import market_calendar
    from src.broker import reconcile
    monkeypatch.setattr(settings, "broker_limit_cap_bps_extended", 80.0)
    monkeypatch.setattr(settings, "broker_limit_cap_bps_overnight", 150.0)
    monkeypatch.setattr(market_calendar, "current_session", lambda now=None: "overnight")
    assert reconcile._limit_price_for("BUY", 100.0, outside_rth=True) == pytest.approx(101.5)
    monkeypatch.setattr(market_calendar, "current_session", lambda now=None: "extended")
    assert reconcile._limit_price_for("BUY", 100.0, outside_rth=True) == pytest.approx(100.8)
    # RTH ignores the session entirely.
    assert reconcile._limit_price_for("BUY", 100.0, outside_rth=False) == pytest.approx(
        100.0 * (1 + settings.broker_limit_cap_bps / 10000.0), abs=0.011)


def test_overnight_routing_active_gates(monkeypatch):
    from src.performance import market_calendar
    from src.broker import reconcile
    monkeypatch.setattr(market_calendar, "current_session", lambda now=None: "overnight")
    monkeypatch.setattr(market_calendar, "is_overnight_session_open", lambda now=None: True)
    monkeypatch.setattr(settings, "broker_overnight_routing", True)
    assert reconcile._overnight_routing_active() is True
    monkeypatch.setattr(settings, "broker_overnight_routing", False)
    assert reconcile._overnight_routing_active() is False
    monkeypatch.setattr(settings, "broker_overnight_routing", True)
    monkeypatch.setattr(market_calendar, "is_overnight_session_open", lambda now=None: False)
    assert reconcile._overnight_routing_active() is False        # Fri/Sat night
    monkeypatch.setattr(market_calendar, "current_session", lambda now=None: "extended")
    assert reconcile._overnight_routing_active() is False        # not overnight at all


def test_order_request_overnight_field_defaults_false():
    from src.broker.base import OrderRequest
    req = OrderRequest(ticker="SPY", side="BUY", quantity=1)
    assert req.overnight is False
    req = OrderRequest(ticker="SPY", side="BUY", quantity=1, overnight=True)
    assert req.overnight is True
