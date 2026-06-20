"""Tests for src.performance.daily_nav — the path-faithful NAV engine.

These tests are the highest-priority regression net: they pin the math of
every code path the engine takes, so any future edit that perturbs the
spread treatment, split adjustment, sign convention, zero-mark guard, or
open-trade anchor will fail loudly here before reaching production.
"""

from datetime import date

import pandas as pd
import pytest

import src.performance.daily_nav as dn
from src.performance.daily_nav import (
    compute_compound_return,
    compute_trade_compound,
    daily_pnl_breakdown,
    _session_date,
)


# ── Helper to inject a synthetic close series ────────────────────────────

@pytest.fixture
def fake_closes(monkeypatch):
    """Returns a function that installs a synthetic OHLCV close series."""
    store: dict[str, dict[date, float]] = {}

    def install(ticker: str, dates: list[date], closes: list[float]):
        store[ticker] = dict(zip(dates, closes))

    def fake_loader(tk: str) -> dict[date, float]:
        return store.get(tk, {})

    monkeypatch.setattr(dn, "_load_close_series", fake_loader)
    return install


# ── _session_date timezone normalisation (Fix #17) ───────────────────────

def test_session_date_tz_aware_yfinance_summer():
    """EDT-midnight bar → 04:00 UTC → session date 2026-05-14."""
    ts = pd.Timestamp("2026-05-14T04:00:00.000Z")
    assert _session_date(ts) == date(2026, 5, 14)


def test_session_date_tz_aware_yfinance_winter():
    """EST-midnight bar → 05:00 UTC → session date 2026-01-14."""
    ts = pd.Timestamp("2026-01-14T05:00:00.000Z")
    assert _session_date(ts) == date(2026, 1, 14)


def test_session_date_tz_naive_polygon():
    """Polygon midnight UTC → session date matches the UTC date directly."""
    ts = pd.Timestamp("2026-05-14T00:00:00")
    assert _session_date(ts) == date(2026, 5, 14)


# ── Single-trade compounds ───────────────────────────────────────────────

def test_long_compound_telescopes_to_buy_and_hold(fake_closes):
    """For longs, the path-faithful daily walk compounds back to (eff_exit/eff_entry − 1).
    Spread for $100 large-cap = 4 bp half. Round-trip cost ≈ 8 bp."""
    fake_closes("STK", [date(2026, 5, 11), date(2026, 5, 12), date(2026, 5, 13)], [105.0, 108.0, 110.0])
    trade = {
        "ticker": "STK", "type": "STOCK", "action": "BUY",
        "entry_date": "2026-05-11", "exit_date": "2026-05-13",
        "entry_price": 100.0, "exit_price": 110.0,
        "position_size_multiplier": 1.0, "status": "CLOSED",
    }
    # Hand math: eff_entry = 100.04, eff_exit = 109.956 → +9.911 %
    result = compute_trade_compound(trade)
    assert result == pytest.approx(9.91, abs=0.02)


def test_short_path_faithful_diverges_from_buy_and_hold(fake_closes):
    """Volatile short: round-trip back to entry price → still a small *loss* from
    daily marking (volatility decay), not zero like simple buy-and-hold would say."""
    fake_closes("STK",
                [date(2026, 5, 12), date(2026, 5, 13), date(2026, 5, 14)],
                [110.0, 90.0, 100.0])
    trade = {
        "ticker": "STK", "type": "STOCK", "action": "SELL",
        "entry_date": "2026-05-11", "exit_date": "2026-05-15",
        "entry_price": 100.0, "exit_price": 100.0,
        "position_size_multiplier": 1.0, "status": "CLOSED",
    }
    # Path-faithful short with 100 → 110 → 90 → 100:
    #   r1 = -(110-100)/100 = -10%
    #   r2 = -(90-110)/110  = +18.18%
    #   r3 = -(100-90)/90   = -11.11%
    # Compound ≈ 0.9 × 1.1818 × 0.8889 ≈ 0.9454 → -5.5% before spread cost.
    # Buy-and-hold equivalent would be 0% (entry and exit prices equal).
    result = compute_trade_compound(trade)
    assert result is not None
    assert result < -2.0, "Path-faithful short on round-trip prices should show volatility decay"


# ── Split adjustment (Fix #1) ────────────────────────────────────────────

def test_split_adjusted_walk(fake_closes):
    """A 2-for-1 split mid-walk is silently absorbed by the entry_ref_close anchor.

    Scenario: entered at $100 on 2026-05-01.  Cache today shows close[2026-05-01]
    = $50 because a 2-for-1 retroactively adjusted everything.  Without the
    ref_close fix, the walk reads a phantom -50 % drop on day 2.
    """
    fake_closes("STK",
                [date(2026, 5, 1), date(2026, 5, 2), date(2026, 5, 3), date(2026, 5, 4)],
                [50.0, 55.0, 55.0, 60.0])    # all in current (post-split) scale
    trade = {
        "ticker": "STK", "type": "STOCK", "action": "BUY",
        "entry_date": "2026-05-01", "exit_date": "2026-05-04",
        "entry_price": 100.0,        # recorded BEFORE split
        "exit_price":  60.0,         # recorded AFTER split (in current scale)
        "entry_ref_close": 100.0,    # frozen at trade time on old scale
        "entry_ref_close_date": "2026-05-01",
        "exit_ref_close": 60.0,      # frozen at exit time on current scale
        "exit_ref_close_date": "2026-05-04",
        "position_size_multiplier": 1.0, "status": "CLOSED",
    }
    # Real economic story: $50 → $60 in current scale = +20 % minus spread.
    result = compute_trade_compound(trade)
    assert result == pytest.approx(19.94, abs=0.05)


def test_no_ref_close_falls_back_to_no_adjustment(fake_closes):
    """Without ref data we conservatively skip adjustment.

    A trade that has experienced a split during its holding period and lacks
    the new ref_close fields WILL show the phantom jump — this is the
    documented fallback for legacy entries.  Test pins the behaviour so a
    future change that silently auto-adjusts (and is therefore unsafe for
    intraday-vs-close entries) shows up here.
    """
    fake_closes("STK",
                [date(2026, 5, 1), date(2026, 5, 4)],
                [50.0, 60.0])
    trade = {
        "ticker": "STK", "type": "STOCK", "action": "BUY",
        "entry_date": "2026-05-01", "exit_date": "2026-05-04",
        "entry_price": 100.0, "exit_price": 60.0,
        "position_size_multiplier": 1.0, "status": "CLOSED",
    }
    # No ref data → no rescale → walks 100 → 50 → 60 → phantom heavy loss.
    result = compute_trade_compound(trade)
    assert result < -30.0   # ≈ -40 %


def test_split_adjustment_ignores_subsplit_noise():
    """The split-adjustment must fire ONLY for a genuine corporate action
    (≥~20% discrete move). A near-1.0 ratio means the reference close wasn't a
    finalized daily close — an intraday/forming-bar snapshot or a mixed-source
    cache disagreement — not a split, and applying it injects a phantom return.

    Regression for the 2026-06-16 daily-NAV reconciliation bug: INTC's same-day
    intraday exit ref (112.46) vs the finalized 06-11 close (116.96) was read as
    a 1.04× 'split', inflating its compound to +5.1% vs a true +1.5%; RDW's
    mixed-source ref (16.06 vs 15.75) as a 0.98×.
    """
    # ~4% intraday-vs-finalized gap (INTC) → NOT a split → ignored.
    assert dn._split_adjustment_factor(112.46, "2026-06-11", {date(2026, 6, 11): 116.96}) == 1.0
    # ~2% mixed-source gap (RDW) → ignored.
    assert dn._split_adjustment_factor(16.06, "2026-06-09", {date(2026, 6, 9): 15.75}) == 1.0
    # Genuine 2:1 split → applied.
    assert dn._split_adjustment_factor(100.0, "2026-05-01", {date(2026, 5, 1): 50.0}) == pytest.approx(0.5)
    # Genuine 5:4 split (0.8 — the smallest common forward split) → applied.
    assert dn._split_adjustment_factor(100.0, "2026-05-01", {date(2026, 5, 1): 80.0}) == pytest.approx(0.8)


def test_intraday_exit_ref_no_phantom_compound(fake_closes):
    """A same-day intraday trade whose exit_ref_close was captured mid-session
    must reconcile to the real entry→exit move — the intraday-snapshot reference
    must not inject a phantom split adjustment. End-to-end regression for the
    INTC bug above (+5.1% phantom vs ~+1.5% real)."""
    fake_closes("STK", [date(2026, 6, 10), date(2026, 6, 11)], [107.04, 116.96])
    base = {
        "ticker": "STK", "type": "STOCK", "action": "BUY",
        "entry_date": "2026-06-11", "exit_date": "2026-06-11",
        "entry_price": 112.33, "exit_price": 114.07,
        "entry_ref_close": 107.04, "entry_ref_close_date": "2026-06-10",  # finalized prior close → 1.0×
        "position_size_multiplier": 1.0, "status": "CLOSED",
    }
    polluted = dict(base, exit_ref_close=112.46, exit_ref_close_date="2026-06-11")  # intraday snap → would-be 1.04×
    clean    = dict(base, exit_ref_close=116.96, exit_ref_close_date="2026-06-11")  # finalized close → 1.0×
    r_pol = compute_trade_compound(polluted)
    r_cln = compute_trade_compound(clean)
    assert r_pol == r_cln           # guard neutralises the intraday-snapshot ref
    assert r_pol is not None and 0.0 < r_pol < 2.0   # ~+1.5% real move, NOT the +5% phantom


# ── Zero-mark guard (Fix #15) ────────────────────────────────────────────

def test_zero_mark_skipped_walk_continues(monkeypatch):
    """A bad row (close = 0) must not silence the rest of the walk.

    Old behaviour: prev_mark = 0 → every subsequent r_d guard returned 0 →
    every event after the bad row was lost.
    """
    def force_series(_):
        return {
            date(2026, 5, 12): 100.0,
            date(2026, 5, 13): 0.0,     # CORRUPT row (must be skipped)
            date(2026, 5, 14): 110.0,
            date(2026, 5, 15): 120.0,
        }
    monkeypatch.setattr(dn, "_load_close_series", force_series)
    trade = {
        "ticker": "STK", "type": "STOCK", "action": "BUY",
        "entry_date": "2026-05-12", "exit_date": "2026-05-15",
        "entry_price": 100.0, "exit_price": 120.0,
        "entry_ref_close": 100.0, "entry_ref_close_date": "2026-05-12",
        "exit_ref_close":  120.0, "exit_ref_close_date":  "2026-05-15",
        "position_size_multiplier": 1.0, "status": "CLOSED",
    }
    # Real walk: 100 → 110 → 120 → +20 % minus spread.
    # Old code would have collapsed to ~0 % on the corrupt row.
    result = compute_trade_compound(trade)
    assert result == pytest.approx(19.94, abs=0.05)

    # Breakdown must not include the bad day and must reach the exit anchor.
    bd = daily_pnl_breakdown(trade)
    dates_in_walk = [d for d, _, _ in bd]
    assert date(2026, 5, 13) not in dates_in_walk
    assert date(2026, 5, 15) in dates_in_walk


# ── Open-trade end anchor: live mark primary, cached-close fallback ───────────
# Design note: an OPEN trade is marked to the LIVE current_price each tick so the
# equity curve reflects the latest intraday price (CLAUDE.md: "Open-position NAV is
# marked to the live price each tick"). The deterministic cached-close anchor
# (_open_trade_end_anchor, the original Fix #9) remains the FALLBACK for legacy
# trades with no live mark — and that fallback is still invariant to wall-clock.

_OPEN_BASE = {
    "ticker": "STK", "type": "STOCK", "action": "BUY",
    "entry_date": "2026-05-12", "entry_price": 100.0,
    "entry_ref_close": 100.0, "entry_ref_close_date": "2026-05-12",
    "position_size_multiplier": 1.0, "status": "OPEN",
}


def test_open_trade_tracks_live_mark_when_present(fake_closes):
    """The open-trade compound moves WITH the live current_price mark — a higher
    intraday mark yields a higher compound (the live-mark design)."""
    fake_closes("STK",
                [date(2026, 5, 12), date(2026, 5, 13), date(2026, 5, 14)],
                [100.0, 105.0, 110.0])
    today = date(2026, 5, 15)
    a = compute_trade_compound(dict(_OPEN_BASE, current_price=108.0), today=today)
    b = compute_trade_compound(dict(_OPEN_BASE, current_price=115.0), today=today)
    assert a is not None and b is not None
    assert b > a, f"a higher live mark must raise the compound (got {a} vs {b})"


def test_open_trade_fallback_anchor_is_deterministic(fake_closes):
    """With NO live mark (legacy trade), the engine anchors to the most recent
    cached close — invariant to the wall-clock time the run happened (Fix #9),
    and sitting strictly between the live-mark results above/below that close
    (confirming it anchors at the 110 close, not the live price)."""
    fake_closes("STK",
                [date(2026, 5, 12), date(2026, 5, 13), date(2026, 5, 14)],
                [100.0, 105.0, 110.0])
    legacy = dict(_OPEN_BASE)                       # no current_price
    r_early = compute_trade_compound(legacy, today=date(2026, 5, 15))
    r_late  = compute_trade_compound(legacy, today=date(2026, 5, 16))  # same cache, later run
    assert r_early is not None
    assert r_early == r_late, "fallback compound must not depend on wall-clock given the cache"

    # Anchored at the 110 cached close → between the 108 and 115 live-mark walks.
    lo = compute_trade_compound(dict(_OPEN_BASE, current_price=108.0), today=date(2026, 5, 15))
    hi = compute_trade_compound(dict(_OPEN_BASE, current_price=115.0), today=date(2026, 5, 15))
    assert lo < r_early < hi


# ── Portfolio aggregation ────────────────────────────────────────────────

def test_compound_empty_returns_none():
    assert compute_compound_return([]) is None


def test_capital_weighted_aggregation(fake_closes):
    """Two overlapping positions with different weights → portfolio daily
    return = capital-weighted average of position returns each day."""
    fake_closes("A", [date(2026, 5, 12), date(2026, 5, 13)], [110.0, 121.0])
    fake_closes("B", [date(2026, 5, 12), date(2026, 5, 13)], [90.0, 81.0])

    a = {
        "ticker": "A", "type": "STOCK", "action": "BUY",
        "entry_date": "2026-05-11", "exit_date": "2026-05-13",
        "entry_price": 100.0, "exit_price": 121.0,
        "entry_ref_close": 100.0, "entry_ref_close_date": "2026-05-12",
        "exit_ref_close":  121.0, "exit_ref_close_date":  "2026-05-13",
        "position_size_multiplier": 2.0, "status": "CLOSED",   # higher weight
    }
    b = {
        "ticker": "B", "type": "STOCK", "action": "BUY",
        "entry_date": "2026-05-11", "exit_date": "2026-05-13",
        "entry_price": 100.0, "exit_price": 81.0,
        "entry_ref_close": 100.0, "entry_ref_close_date": "2026-05-12",
        "exit_ref_close":  81.0,  "exit_ref_close_date":  "2026-05-13",
        "position_size_multiplier": 1.0, "status": "CLOSED",
    }
    # Compound should be biased toward A (the heavier weight, the winner).
    portfolio = compute_compound_return([a, b])
    single_a  = compute_trade_compound(a)
    single_b  = compute_trade_compound(b)
    assert portfolio is not None
    # Portfolio sits between the two trades.
    assert single_b < portfolio < single_a
    # And lies ABOVE the equal-weight midpoint — the 2× weight on A pulls the
    # portfolio compound up.  (The portfolio doesn't reach A's level because
    # the weight isn't infinite; it just leans that way.)
    midpoint = (single_a + single_b) / 2
    assert portfolio > midpoint, (
        f"portfolio ({portfolio}) should sit above equal-weight midpoint "
        f"({midpoint}) given A's 2× weight"
    )
