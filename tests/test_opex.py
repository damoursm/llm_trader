"""Options-expiration calendar (`src/data/opex.py`) — pure date math.

Feeds the OpEx max-pain weight boost in the weighting stack and the §OpEx email
block. Zero network calls, so every branch is reachable from a fixed date, and
the whole module is a set of boundary conditions: month rollovers, year
rollovers, the third-Friday rule when the 1st IS a Friday, and the difference
between the week containing expiry and the week after it.

The one that actually bites is the "next vs previous expiry" switch. On expiry
day itself the next expiry is TODAY; one day later it jumps to next month, and
so does the whole opex-week window it is derived from.
"""

from __future__ import annotations

from datetime import date, timedelta

import pytest

from src.data.opex import _third_friday, compute_opex_context


# ── the third-Friday rule ───────────────────────────────────────────────────

@pytest.mark.parametrize("year,month,expected", [
    (2026, 1, date(2026, 1, 16)),
    (2026, 3, date(2026, 3, 20)),
    (2026, 5, date(2026, 5, 15)),    # May 1 2026 IS a Friday -> 3rd is the 15th
    (2026, 8, date(2026, 8, 21)),
    (2026, 12, date(2026, 12, 18)),
    (2027, 1, date(2027, 1, 15)),
])
def test_third_friday_is_correct(year, month, expected):
    got = _third_friday(year, month)
    assert got == expected
    assert got.weekday() == 4, "not a Friday"


def test_third_friday_is_always_between_the_15th_and_21st():
    """The arithmetic invariant — an off-by-one week is otherwise only visible
    in months where the 1st happens to be a Friday."""
    for year in (2025, 2026, 2027):
        for month in range(1, 13):
            d = _third_friday(year, month)
            assert d.weekday() == 4
            assert 15 <= d.day <= 21


# ── the next/previous switch ────────────────────────────────────────────────

def test_on_expiry_day_the_next_expiry_is_today():
    ctx = compute_opex_context(date(2026, 8, 21))
    assert ctx.next_opex == date(2026, 8, 21)
    assert ctx.days_to_opex == 0
    assert ctx.prev_opex == date(2026, 7, 17)
    assert ctx.signal == "OPEX_DAY"


def test_the_day_after_expiry_rolls_to_next_month():
    ctx = compute_opex_context(date(2026, 8, 22))
    assert ctx.next_opex == date(2026, 9, 18)
    assert ctx.prev_opex == date(2026, 8, 21)
    assert ctx.days_since_prev_opex == 1
    assert ctx.in_opex_week is False, "the expiry week ended with the expiry"


def test_december_rolls_the_year_forward():
    ctx = compute_opex_context(date(2026, 12, 19))
    assert ctx.next_opex == date(2027, 1, 15)
    assert ctx.prev_opex == date(2026, 12, 18)


def test_january_rolls_the_year_back():
    ctx = compute_opex_context(date(2026, 1, 5))
    assert ctx.next_opex == date(2026, 1, 16)
    assert ctx.prev_opex == date(2025, 12, 19)
    assert ctx.days_since_prev_opex > 0


# ── the signal ladder ───────────────────────────────────────────────────────

@pytest.mark.parametrize("day,expected", [
    (date(2026, 8, 21), "OPEX_DAY"),           # the Friday
    (date(2026, 8, 20), "OPEX_IMMINENT"),      # Thursday
    (date(2026, 8, 17), "OPEX_WEEK"),          # Monday of expiry week
    (date(2026, 8, 24), "POST_OPEX"),          # 3d after
    (date(2026, 8, 5), "NEUTRAL"),             # mid-cycle
])
def test_signal_ladder_for_a_standard_month(day, expected):
    assert compute_opex_context(day).signal == expected


def test_quarterly_expiry_week_is_flagged_triple_witching():
    ctx = compute_opex_context(date(2026, 9, 14))       # Monday of Sep expiry week
    assert ctx.is_triple_witching is True
    assert ctx.signal == "TRIPLE_WITCHING_WEEK"
    assert "TRIPLE WITCHING" in ctx.summary


@pytest.mark.parametrize("month", [3, 6, 9, 12])
def test_all_four_quarterly_months_are_triple_witching(month):
    monday = _third_friday(2026, month) - timedelta(days=4)
    assert compute_opex_context(monday).is_triple_witching is True


@pytest.mark.parametrize("month", [1, 2, 4, 5, 7, 8, 10, 11])
def test_non_quarterly_months_are_not(month):
    monday = _third_friday(2026, month) - timedelta(days=4)
    ctx = compute_opex_context(monday)
    assert ctx.is_triple_witching is False
    assert ctx.signal == "OPEX_WEEK"


def test_expiry_day_outranks_triple_witching_in_the_label():
    """Both are true on a quarterly Friday; the ladder reports the more
    time-critical one, and the witching status stays on its own field."""
    ctx = compute_opex_context(date(2026, 9, 18))
    assert ctx.signal == "OPEX_DAY" and ctx.is_triple_witching is True


# ── the windows ─────────────────────────────────────────────────────────────

def test_opex_week_runs_monday_through_the_expiry_friday():
    expiry = date(2026, 8, 21)
    monday = expiry - timedelta(days=4)
    for offset in range(0, 5):
        assert compute_opex_context(monday + timedelta(days=offset)).in_opex_week
    assert compute_opex_context(monday - timedelta(days=1)).in_opex_week is False
    assert compute_opex_context(expiry + timedelta(days=1)).in_opex_week is False


def test_opex_week_monday_is_always_a_monday():
    for month in range(1, 13):
        ctx = compute_opex_context(_third_friday(2026, month))
        assert ctx.opex_week_monday.weekday() == 0
        assert (ctx.next_opex - ctx.opex_week_monday).days == 4


def test_post_opex_window_is_one_to_five_days_after_expiry():
    expiry = date(2026, 8, 21)
    assert compute_opex_context(expiry).in_post_opex_window is False   # day 0
    for d in range(1, 6):
        assert compute_opex_context(expiry + timedelta(days=d)).in_post_opex_window
    assert compute_opex_context(expiry + timedelta(days=6)).in_post_opex_window is False


def test_opex_week_takes_precedence_over_the_post_opex_window():
    """They can overlap in a short month; the imminent pin is the live effect."""
    ctx = compute_opex_context(date(2026, 8, 17))
    assert ctx.in_opex_week and ctx.signal.startswith("OPEX")


# ── the payload ─────────────────────────────────────────────────────────────

def test_context_is_internally_consistent_every_day_for_two_years():
    """Cheap exhaustive sweep — the module is pure, so there is no reason to
    test only the interesting dates."""
    d = date(2025, 1, 1)
    while d < date(2027, 1, 1):
        ctx = compute_opex_context(d)
        assert ctx.today == d
        assert ctx.prev_opex < ctx.next_opex
        assert ctx.prev_opex <= d <= ctx.next_opex
        assert ctx.days_to_opex == (ctx.next_opex - d).days >= 0
        assert ctx.days_since_prev_opex == (d - ctx.prev_opex).days >= 0
        assert ctx.next_opex.weekday() == 4 and ctx.prev_opex.weekday() == 4
        assert ctx.signal in {"OPEX_DAY", "OPEX_IMMINENT", "TRIPLE_WITCHING_WEEK",
                              "OPEX_WEEK", "POST_OPEX", "NEUTRAL"}
        assert ctx.summary
        d += timedelta(days=1)


def test_defaults_to_today():
    assert compute_opex_context().today == date.today()


def test_summary_names_the_expiry_it_is_talking_about():
    ctx = compute_opex_context(date(2026, 8, 5))
    assert ctx.next_opex.strftime("%b %d") in ctx.summary
    post = compute_opex_context(date(2026, 8, 24))
    assert post.prev_opex.strftime("%b %d") in post.summary
