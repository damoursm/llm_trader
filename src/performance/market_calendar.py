"""NYSE market calendar: trading-day counting and execution-time snapping.

Self-contained — no `pandas_market_calendars` dependency.  Holiday table is
maintained by hand for 2024–2030 (the NYSE publishes the next 3 years; this
table extends a few years out to give the rest of the trading system room).

What this fixes:
  * The old ``_trading_days_held`` counted Mon–Fri without removing market
    holidays, so a position held through Memorial Day / Christmas / etc.
    auto-closed one trading day too early.
  * The old ``entry_datetime`` was the wall-clock fetch instant, which lands
    at ~12:00 UTC (8 AM ET) — *before* the equity market opens.  ``record_
    new_trades`` now snaps that to the next real session open via
    ``effective_execution_datetime`` so the audit trail reflects when a fill
    could actually have happened, while ``decision_datetime`` preserves the
    precise fetch instant for full traceability.

If today's date moves past the end of the hard-coded table the helpers fall
back to "any weekday is a market day" (worst-case behaviour: matches the
pre-fix logic).  Refresh the table annually from the NYSE calendar.
"""

from __future__ import annotations

from datetime import date, datetime, time, timedelta, timezone
from typing import Optional

try:
    from zoneinfo import ZoneInfo
except ImportError:  # pragma: no cover — Python < 3.9
    from backports.zoneinfo import ZoneInfo  # type: ignore


NY_TZ = ZoneInfo("America/New_York")

# Regular session: 9:30 AM – 4:00 PM Eastern.  Half-days (3 close days the
# day after Thanksgiving + Christmas Eve in some years) are not tracked
# separately — for execution-time snapping that's a non-issue (the market
# is open), and for trading-day counting they still count as 1 session.
MARKET_OPEN_LOCAL  = time(9, 30)
MARKET_CLOSE_LOCAL = time(16, 0)

# Extended-session boundaries (ET): pre-market 04:00–09:30, after-hours
# 16:00–20:00, overnight 20:00–04:00. Same convention as tracker._trade_session.
EXTENDED_OPEN_LOCAL  = time(4, 0)
EXTENDED_CLOSE_LOCAL = time(20, 0)


def current_session(now: Optional[datetime] = None) -> str:
    """US-market session for a clock instant: ``rth | extended | overnight``.

    Mirrors the boundaries of ``tracker._trade_session`` (which classifies a
    stored trade dict) but takes a plain datetime, so callers like the
    snapshot fetcher can branch on "what session is it right now". Naive
    datetimes are assumed to already be Eastern.
    """
    dt = now or datetime.now(NY_TZ)
    dt = dt.astimezone(NY_TZ) if dt.tzinfo is not None else dt
    t = dt.time()
    if MARKET_OPEN_LOCAL <= t < MARKET_CLOSE_LOCAL:
        return "rth"
    if EXTENDED_OPEN_LOCAL <= t < MARKET_OPEN_LOCAL or MARKET_CLOSE_LOCAL <= t < EXTENDED_CLOSE_LOCAL:
        return "extended"
    return "overnight"


# IBKR's overnight venue (IBEOS) stops matching at 03:50 ET, ten minutes before
# the 04:00 pre-market open.
OVERNIGHT_VENUE_CLOSE_LOCAL = time(3, 50)


def is_overnight_session_open(now: Optional[datetime] = None) -> bool:
    """True when the US OVERNIGHT session (IBKR: 20:00 ET → 03:50 ET next day,
    Sunday night through Thursday night) is actually trading at *now*.

    The ``overnight`` label from :func:`current_session` is purely clock-based —
    it is also "overnight" all Saturday, Friday night, and holiday eves, when no
    venue is open. Trading decisions need this stricter test: the EVENING half
    (≥ 20:00) is open only when the NEXT day is a market day (Sunday evening
    leads into Monday ✓; Friday evening → Saturday ✗; the eve of a holiday ✗),
    and the MORNING half (< 03:50) only when TODAY is a market day. Naive
    datetimes are assumed Eastern.
    """
    dt = now or datetime.now(NY_TZ)
    dt = dt.astimezone(NY_TZ) if dt.tzinfo is not None else dt
    t = dt.time()
    if t >= EXTENDED_CLOSE_LOCAL:                       # 20:00–23:59 evening half
        return is_market_day(dt.date() + timedelta(days=1))
    if t < OVERNIGHT_VENUE_CLOSE_LOCAL:                 # 00:00–03:50 morning half
        return is_market_day(dt.date())
    return False                                        # 03:50–04:00 gap / daytime


# NYSE full-day holidays, 2024–2030.  Includes observance shifts (e.g.
# Independence Day 2026 falls on Saturday → observed Friday 2026-07-03).
# When the table runs out, _is_market_day() falls back to "weekday".
NYSE_HOLIDAYS: frozenset[date] = frozenset({
    # 2024
    date(2024, 1, 1), date(2024, 1, 15), date(2024, 2, 19), date(2024, 3, 29),
    date(2024, 5, 27), date(2024, 6, 19), date(2024, 7, 4), date(2024, 9, 2),
    date(2024, 11, 28), date(2024, 12, 25),
    # 2025
    date(2025, 1, 1), date(2025, 1, 20), date(2025, 2, 17), date(2025, 4, 18),
    date(2025, 5, 26), date(2025, 6, 19), date(2025, 7, 4), date(2025, 9, 1),
    date(2025, 11, 27), date(2025, 12, 25),
    # 2026
    date(2026, 1, 1), date(2026, 1, 19), date(2026, 2, 16), date(2026, 4, 3),
    date(2026, 5, 25), date(2026, 6, 19), date(2026, 7, 3), date(2026, 9, 7),
    date(2026, 11, 26), date(2026, 12, 25),
    # 2027
    date(2027, 1, 1), date(2027, 1, 18), date(2027, 2, 15), date(2027, 3, 26),
    date(2027, 5, 31), date(2027, 6, 18), date(2027, 7, 5), date(2027, 9, 6),
    date(2027, 11, 25), date(2027, 12, 24),
    # 2028
    date(2028, 1, 17), date(2028, 2, 21), date(2028, 4, 14), date(2028, 5, 29),
    date(2028, 6, 19), date(2028, 7, 4), date(2028, 9, 4), date(2028, 11, 23),
    date(2028, 12, 25),
    # 2029
    date(2029, 1, 1), date(2029, 1, 15), date(2029, 2, 19), date(2029, 3, 30),
    date(2029, 5, 28), date(2029, 6, 19), date(2029, 7, 4), date(2029, 9, 3),
    date(2029, 11, 22), date(2029, 12, 25),
    # 2030
    date(2030, 1, 1), date(2030, 1, 21), date(2030, 2, 18), date(2030, 4, 19),
    date(2030, 5, 27), date(2030, 6, 19), date(2030, 7, 4), date(2030, 9, 2),
    date(2030, 11, 28), date(2030, 12, 25),
})

# Range covered by the table.  Outside this range _is_market_day falls back
# to a plain weekday check so the helpers still work — but the user should
# extend the table well before reaching that edge.
_TABLE_START = date(2024, 1, 1)
_TABLE_END   = date(2030, 12, 31)


def is_market_day(d: date) -> bool:
    """True for NYSE regular sessions: weekday and not a known holiday."""
    if d.weekday() >= 5:
        return False
    if _TABLE_START <= d <= _TABLE_END:
        return d not in NYSE_HOLIDAYS
    return True   # outside table — assume weekday is a session (fail-open)


def previous_market_day(d: date) -> date:
    """Most recent NYSE session strictly before *d*."""
    p = d - timedelta(days=1)
    while not is_market_day(p):
        p -= timedelta(days=1)
    return p


def market_days_between(start: date, end: date) -> int:
    """Count NYSE sessions in (start, end] — exclusive of start, inclusive of end.

    Used to populate each trade's ``days_held`` field (display / observability).
    Walking calendar days here is fine: the loop is cheap and the implementation
    needs no external dependency.
    """
    if end <= start:
        return 0
    count = 0
    d = start + timedelta(days=1)
    while d <= end:
        if is_market_day(d):
            count += 1
        d += timedelta(days=1)
    return count


def next_market_session_open_utc(now_utc: Optional[datetime] = None) -> datetime:
    """Return the next NYSE regular-session open instant, as a UTC datetime.

    Logic:
      * If ``now_utc`` is already within today's session window AND today is
        a market day → returns ``now_utc`` (we'd execute immediately).
      * Otherwise → returns the next market-day's 9:30 AM ET, in UTC.

    Used by ``record_new_trades`` to stamp ``entry_datetime`` as the time a
    real fill could have occurred.  ``decision_datetime`` (the raw fetch
    instant) is stored alongside so the audit trail still records the exact
    moment the pipeline made the call.
    """
    if now_utc is None:
        now_utc = datetime.now(timezone.utc)
    elif now_utc.tzinfo is None:
        now_utc = now_utc.replace(tzinfo=timezone.utc)

    now_et = now_utc.astimezone(NY_TZ)
    today_et = now_et.date()

    if is_market_day(today_et):
        open_et  = datetime.combine(today_et, MARKET_OPEN_LOCAL,  tzinfo=NY_TZ)
        close_et = datetime.combine(today_et, MARKET_CLOSE_LOCAL, tzinfo=NY_TZ)
        if now_et < open_et:
            return open_et.astimezone(timezone.utc)
        if now_et < close_et:
            return now_utc  # in session — execute now

    # After close or non-market day → roll forward to the next session.
    d = today_et + timedelta(days=1)
    while not is_market_day(d):
        d += timedelta(days=1)
    next_open_et = datetime.combine(d, MARKET_OPEN_LOCAL, tzinfo=NY_TZ)
    return next_open_et.astimezone(timezone.utc)


def effective_execution_iso(now_utc: Optional[datetime] = None) -> str:
    """ISO 8601 UTC string for the next realistic fill time. See above."""
    return next_market_session_open_utc(now_utc).isoformat(timespec="seconds")
