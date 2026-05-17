"""Tests for src.performance.market_calendar — NYSE holidays + session snapping."""

from datetime import date, datetime, timezone

import pytest

from src.performance.market_calendar import (
    is_market_day,
    market_days_between,
    next_market_session_open_utc,
)


# ── is_market_day ─────────────────────────────────────────────────────────

@pytest.mark.parametrize("d,expected,reason", [
    (date(2026, 5, 25), False, "Memorial Day"),
    (date(2026, 12, 25), False, "Christmas"),
    (date(2026, 1, 19), False, "MLK Day"),
    (date(2026, 4, 3),  False, "Good Friday"),
    (date(2026, 7, 3),  False, "Independence Day observed (July 4 = Sat)"),
    (date(2026, 11, 26), False, "Thanksgiving"),
    (date(2026, 5, 16), False, "Saturday"),
    (date(2026, 5, 17), False, "Sunday"),
    (date(2026, 5, 18), True,  "ordinary Monday"),
    (date(2026, 5, 26), True,  "Tuesday after Memorial Day"),
    (date(2025, 6, 19), False, "Juneteenth 2025"),
    (date(2027, 7, 5),  False, "Independence Day observed (July 4 = Sun)"),
])
def test_is_market_day(d, expected, reason):
    assert is_market_day(d) is expected, reason


def test_is_market_day_outside_table_falls_back_to_weekday():
    """Outside the hard-coded table the helper assumes any weekday is a session."""
    far_future_weekday = date(2099, 7, 22)   # a Wednesday
    far_future_weekend = date(2099, 7, 19)   # a Sunday
    assert is_market_day(far_future_weekday) is True
    assert is_market_day(far_future_weekend) is False


# ── market_days_between ───────────────────────────────────────────────────

def test_market_days_between_excludes_memorial_day():
    """Fri 2026-05-22 → Tue 2026-06-02 contains Memorial Day (Mon 2026-05-25)."""
    # Days in (Fri, Tue]: Sat, Sun, Mon (holiday), Tue, Wed, Thu, Fri, Sat, Sun, Mon, Tue
    # Weekdays: Mon, Tue, Wed, Thu, Fri, Mon, Tue = 7
    # Market days: weekdays minus Memorial Day = 6
    assert market_days_between(date(2026, 5, 22), date(2026, 6, 2)) == 6


def test_market_days_between_thanksgiving_week():
    """Thanksgiving 2026 (Thu Nov 26) — 4-day week."""
    # Mon Nov 23 → Fri Nov 27 = (Tue Wed Thu(holiday) Fri) = 3 market days
    assert market_days_between(date(2026, 11, 23), date(2026, 11, 27)) == 3


def test_market_days_between_same_day_zero():
    assert market_days_between(date(2026, 5, 15), date(2026, 5, 15)) == 0


def test_market_days_between_reverse_zero():
    """End before start returns 0 (defensive — should not auto-negate)."""
    assert market_days_between(date(2026, 5, 20), date(2026, 5, 15)) == 0


# ── next_market_session_open_utc ─────────────────────────────────────────

def _utc(*args):
    return datetime(*args, tzinfo=timezone.utc)


def test_snap_premarket_to_today_open():
    """8 AM ET on a market day → today's 9:30 AM ET (= 13:30 UTC in EDT)."""
    fri_8am_et = _utc(2026, 5, 15, 12, 0)
    out = next_market_session_open_utc(fri_8am_et)
    assert out == _utc(2026, 5, 15, 13, 30)


def test_in_session_returns_unchanged():
    """Mid-session timestamps return as-is (we'd execute immediately)."""
    fri_130pm_et = _utc(2026, 5, 15, 17, 30)
    assert next_market_session_open_utc(fri_130pm_et) == fri_130pm_et


def test_snap_after_close_to_next_trading_day():
    """Fri 6 PM ET → Mon 9:30 AM ET."""
    fri_6pm_et = _utc(2026, 5, 15, 22, 0)
    assert next_market_session_open_utc(fri_6pm_et) == _utc(2026, 5, 18, 13, 30)


def test_snap_weekend_to_monday():
    sat_morning_et = _utc(2026, 5, 16, 14, 0)
    assert next_market_session_open_utc(sat_morning_et) == _utc(2026, 5, 18, 13, 30)


def test_snap_holiday_to_next_trading_day():
    """Memorial Day Mon 8 AM ET → Tue 9:30 AM ET."""
    mem_day_8am = _utc(2026, 5, 25, 12, 0)
    assert next_market_session_open_utc(mem_day_8am) == _utc(2026, 5, 26, 13, 30)


def test_snap_handles_naive_datetime():
    """Naive datetime is assumed UTC (matches datetime.now() conventions)."""
    naive_premarket = datetime(2026, 5, 15, 12, 0)
    out = next_market_session_open_utc(naive_premarket)
    assert out == _utc(2026, 5, 15, 13, 30)
