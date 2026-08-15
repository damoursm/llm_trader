"""Seasonal calendar effects (`src/data/seasonality.py`) — pure date math.

Feeds the §Seasonality email block and the synthesis prompt. Zero network, so
every branch is reachable from a fixed date and the whole module is boundary
conditions: month lengths (including February in a leap year), quarter edges,
and the composite score that combines a monthly bias with the active windows.

The composite is the part worth pinning. It sums a monthly bias with the net of
the active window effects and then buckets the total, so a sign or an off-by-one
in either term silently relabels a headwind as a tailwind while every individual
flag still reads correctly.
"""

from __future__ import annotations

import calendar
from datetime import date, timedelta

import pytest

from src.data.seasonality import (_MONTHLY_BIAS, _QUARTER_END_MONTHS,
                                  _QUARTER_START_MONTHS,
                                  compute_seasonality_context)


def _ctx(d: date):
    return compute_seasonality_context(d)


# ── month-boundary windows ──────────────────────────────────────────────────

@pytest.mark.parametrize("year,month", [(2026, 2), (2024, 2), (2026, 4), (2026, 12)])
def test_month_end_window_is_the_last_three_calendar_days(year, month):
    """Driven off `calendar.monthrange`, so a 28/29/30/31-day month all have to
    land on the same three days — the leap-February case is the one a hard-coded
    30 would get wrong."""
    last = calendar.monthrange(year, month)[1]
    for d in range(last - 2, last + 1):
        assert _ctx(date(year, month, d)).in_month_end_window is True
    assert _ctx(date(year, month, last - 3)).in_month_end_window is False


def test_month_start_window_is_the_first_three_calendar_days():
    for d in (1, 2, 3):
        assert _ctx(date(2026, 8, d)).in_month_start_window is True
    assert _ctx(date(2026, 8, 4)).in_month_start_window is False


def test_the_two_month_windows_do_not_overlap_in_a_normal_month():
    for d in range(1, 32):
        c = _ctx(date(2026, 8, d))
        assert not (c.in_month_end_window and c.in_month_start_window)


# ── quarter windows ─────────────────────────────────────────────────────────

@pytest.mark.parametrize("month", sorted(_QUARTER_END_MONTHS))
def test_quarter_end_window_is_the_last_five_days_of_a_quarter_end_month(month):
    last = calendar.monthrange(2026, month)[1]
    assert _ctx(date(2026, month, last)).in_quarter_end_window is True
    assert _ctx(date(2026, month, last - 4)).in_quarter_end_window is True
    assert _ctx(date(2026, month, last - 5)).in_quarter_end_window is False


@pytest.mark.parametrize("month", [1, 2, 4, 5, 7, 8, 10, 11])
def test_non_quarter_end_months_never_open_the_window(month):
    last = calendar.monthrange(2026, month)[1]
    assert _ctx(date(2026, month, last)).in_quarter_end_window is False


@pytest.mark.parametrize("month", sorted(_QUARTER_START_MONTHS))
def test_quarter_start_window_is_the_first_five_days(month):
    assert _ctx(date(2026, month, 1)).in_quarter_start_window is True
    assert _ctx(date(2026, month, 5)).in_quarter_start_window is True
    assert _ctx(date(2026, month, 6)).in_quarter_start_window is False


def test_quarter_number_matches_the_month():
    for month in range(1, 13):
        assert _ctx(date(2026, month, 15)).quarter == (month - 1) // 3 + 1


def test_the_quarter_month_sets_are_consistent():
    """Every quarter-start month must follow a quarter-end month; a set that
    drifted would open the two windows on the wrong side of a boundary."""
    assert len(_QUARTER_END_MONTHS) == len(_QUARTER_START_MONTHS) == 4
    for m in _QUARTER_START_MONTHS:
        prev = 12 if m == 1 else m - 1
        assert prev in _QUARTER_END_MONTHS


# ── fiscal-year-end intensity ───────────────────────────────────────────────

@pytest.mark.parametrize("month", [6, 12])
def test_fiscal_year_end_needs_both_the_month_and_the_window(month):
    last = calendar.monthrange(2026, month)[1]
    assert _ctx(date(2026, month, last)).is_fiscal_year_end is True
    assert _ctx(date(2026, month, 10)).is_fiscal_year_end is False


@pytest.mark.parametrize("month", [3, 9])
def test_other_quarter_ends_are_not_fiscal_year_ends(month):
    last = calendar.monthrange(2026, month)[1]
    c = _ctx(date(2026, month, last))
    assert c.in_quarter_end_window is True and c.is_fiscal_year_end is False


# ── the January effect ──────────────────────────────────────────────────────

def test_january_effect_covers_the_first_fifteen_days_only():
    assert _ctx(date(2026, 1, 1)).in_january_effect is True
    assert _ctx(date(2026, 1, 15)).in_january_effect is True
    assert _ctx(date(2026, 1, 16)).in_january_effect is False
    assert _ctx(date(2026, 2, 5)).in_january_effect is False


# ── the monthly bias table ──────────────────────────────────────────────────

def test_every_month_has_a_bias_entry():
    """A missing month is a KeyError on that date — a whole-month outage of the
    seasonality block, one month a year."""
    assert set(_MONTHLY_BIAS) == set(range(1, 13))


def test_the_documented_extremes_are_encoded():
    """April strongest, September worst — the two the module's docstring
    commits to."""
    assert _MONTHLY_BIAS[4][0] == "BULLISH"
    assert _MONTHLY_BIAS[9][0] == "BEARISH"


def test_bias_directions_are_valid_labels():
    for month, (direction, signal, desc) in _MONTHLY_BIAS.items():
        assert direction in {"BULLISH", "BEARISH", "NEUTRAL"}, month
        assert signal and desc


def test_the_context_reports_the_months_own_bias():
    for month in range(1, 13):
        c = _ctx(date(2026, month, 15))
        assert c.monthly_bias == _MONTHLY_BIAS[month][0]
        assert c.month == month
        assert c.month_name == date(2026, month, 15).strftime("%B")


# ── the composite ───────────────────────────────────────────────────────────

def test_composite_direction_agrees_with_its_signal():
    """The two are derived from one score; letting them disagree would put a
    BEARISH direction under a TAILWIND label in the email."""
    d = date(2026, 1, 1)
    while d < date(2027, 1, 1):
        c = _ctx(d)
        if c.composite_signal in ("STRONG_TAILWIND", "TAILWIND"):
            assert c.composite_direction == "BULLISH", d
        elif c.composite_signal in ("STRONG_HEADWIND", "HEADWIND"):
            assert c.composite_direction == "BEARISH", d
        else:
            assert c.composite_direction == "NEUTRAL", d
        d += timedelta(days=1)


def test_a_stacked_bullish_window_beats_a_quiet_bullish_month():
    """April is bullish all month; April 1 also carries quarter-start AND
    month-start effects, so the composite must be strictly stronger."""
    quiet = _ctx(date(2026, 4, 15))
    stacked = _ctx(date(2026, 4, 1))
    order = {"STRONG_HEADWIND": -2, "HEADWIND": -1, "NEUTRAL": 0,
             "TAILWIND": 1, "STRONG_TAILWIND": 2}
    assert order[stacked.composite_signal] > order[quiet.composite_signal]
    assert stacked.in_quarter_start_window and stacked.in_month_start_window


def test_active_effects_are_named_in_the_summary():
    c = _ctx(date(2026, 3, 31))          # quarter-end + month-end
    assert c.active_effects
    for e in c.active_effects:
        assert e.name in c.summary
        assert e.direction in {"BULLISH", "BEARISH", "NEUTRAL"}


def test_a_quiet_mid_month_day_has_no_window_effects():
    c = _ctx(date(2026, 8, 14))
    assert not c.in_month_end_window and not c.in_month_start_window
    assert not c.in_quarter_end_window and not c.in_quarter_start_window
    assert c.active_effects == []


# ── whole-year sweep ────────────────────────────────────────────────────────

def test_every_day_of_two_years_produces_a_consistent_context():
    """The module is pure and cheap, so there is no reason to test only the
    interesting dates — this is what catches a month-length or leap-year slip."""
    d = date(2024, 1, 1)                 # includes a leap February
    while d < date(2026, 1, 1):
        c = _ctx(d)
        assert c.today == d and 1 <= c.month <= 12 and 1 <= c.quarter <= 4
        assert c.composite_signal in {"STRONG_TAILWIND", "TAILWIND", "NEUTRAL",
                                      "HEADWIND", "STRONG_HEADWIND"}
        assert c.summary and c.monthly_description in c.summary
        d += timedelta(days=1)


def test_defaults_to_today():
    assert compute_seasonality_context().today == date.today()
