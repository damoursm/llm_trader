"""Probes for the LIVE (provisional) pivot target — ``live_next_pivot`` and the
tracker's per-tick ``_update_pivot_targets``.

The contract under test: for an anchor bar *i0*, the surface reports the FIRST
pivot after *i0* — settled once its ``pivot_min_move_pct`` reversal has
printed, PROVISIONAL until then. The provisional value is the running leg
extreme, so it must (a) keep EXTENDING through sub-threshold wiggles — never
freeze early, the user's explicit requirement — and (b) CONVERGE to the settled
label at the confirming bar without a jump. The scan refactor must leave the
resolved sequence byte-identical (``_resolved_pivots`` is the label machine).
"""

from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd
import pytest

from src.analysis.pivot_target import (_pivot_scan, _resolved_pivots,
                                       live_next_pivot, next_pivot_targets)


def _dates(n):
    return list(pd.date_range("2024-01-02", periods=n, freq="B").date)


def _hl(c, band=0.001):
    c = np.asarray(c, dtype=float)
    return c, c * (1 + band), c * (1 - band)


def _story_series():
    """50 flat bars → 9-bar fall (trough @58, conf 59) → 20-bar rise to a
    running high @79 → 3-bar sub-1% wiggle (must NOT confirm) → 3 new highs
    (extreme extends to @85) → crash @86 (peak @85 CONFIRMS) → 13-bar drift
    down (new down leg, pending). Bar ranges are ±0.1% so the ±1% zigzag
    threshold is exercised by the closes, not the synthetic ranges."""
    c = np.concatenate([
        np.full(50, 100.0),
        [98.9, 97.8, 96.7, 95.6, 94.5, 93.4, 92.3, 91.2, 90.0],   # 50..58
        [91.0],                                                    # 59 confirms trough@58
        91.5 + 0.5 * np.arange(20),                                # 60..79 → 101.0
        [100.6, 100.7, 100.8],                                     # 80..82 wiggle < 1%
        [101.5, 102.0, 102.5],                                     # 83..85 new highs
        [99.0],                                                    # 86 confirms peak@85
        98.8 - 0.2 * np.arange(13),                                # 87..99 drift down
    ])
    return _hl(c)


PEAK_PX = 102.5 * 1.001          # the settled peak extreme (bar 85's high)
I0 = 65                          # anchor inside the up leg


# ── the scan refactor must not move the resolved sequence ────────────────────

def test_scan_and_resolved_pivots_agree():
    c, h, lo = _story_series()
    (P, PP, FL, CF), pending = _pivot_scan(c, h, lo)
    P2, PP2, FL2, CF2 = _resolved_pivots(c, h, lo)
    assert np.array_equal(P, P2) and np.array_equal(PP, PP2)
    assert np.array_equal(FL, FL2) and np.array_equal(CF, CF2)
    # full story: peak@0 (flat-prefix high), trough@58, peak@85 — down leg pending
    assert list(P) == [0, 58, 85]
    assert pending is not None and pending[2] is False        # pending trough
    assert pending[0] == 99                                    # running low @ last bar


# ── provisional semantics ────────────────────────────────────────────────────

def test_provisional_survives_subthreshold_wiggle():
    """Post-wiggle truncation: the candidate peak must still be the pre-wiggle
    high — extended, not frozen or confirmed, by a sub-1% pullback."""
    c, h, lo = _story_series()
    n = 83                                        # bars 0..82 (wiggle included)
    piv = live_next_pivot(c[:n], h[:n], lo[:n], I0)
    assert piv is not None and piv["resolved"] is False
    assert piv["is_peak"] is True
    assert piv["idx"] == 79
    assert piv["price"] == pytest.approx(101.0 * 1.001)


def test_provisional_is_monotone_and_converges_at_confirmation():
    """The up-leg provisional target never decreases across truncations, and
    the settled value at the confirming bar equals the last provisional —
    convergence without a jump."""
    c, h, lo = _story_series()
    last = -np.inf
    for n in range(67, 87):                       # provisional throughout
        piv = live_next_pivot(c[:n], h[:n], lo[:n], I0)
        assert piv is not None and piv["resolved"] is False, f"n={n}"
        assert piv["price"] >= last - 1e-12, f"n={n}"
        last = piv["price"]
    assert last == pytest.approx(PEAK_PX)
    piv = live_next_pivot(c[:87], h[:87], lo[:87], I0)    # bar 86 = confirming bar
    assert piv["resolved"] is True
    assert piv["idx"] == 85 and piv["confirm_idx"] == 86
    assert piv["price"] == pytest.approx(last)             # no jump at resolution


def test_resolved_matches_next_pivot_targets():
    c, h, lo = _story_series()
    piv = live_next_pivot(c, h, lo, I0)
    assert piv["resolved"] is True
    sp, end = next_pivot_targets(c, h, lo)
    assert end[I0] == piv["idx"] == 85
    assert sp[I0] == pytest.approx((piv["price"] / c[I0] - 1.0) * 100.0)


def test_anchor_past_running_extreme_walks_to_next_leg():
    """Anchor AT the leg extreme: the first pivot after it belongs to the NEXT
    leg — the running trough candidate, under the as-if-the-leg-stands view."""
    c, h, lo = _story_series()
    piv = live_next_pivot(c, h, lo, 85)
    assert piv is not None and piv["resolved"] is False and piv["seed"] is False
    assert piv["is_peak"] is False                         # next leg: trough candidate
    assert piv["idx"] == 99
    assert piv["price"] == pytest.approx(c[99] * 0.999)


def test_anchor_on_newest_session_returns_leg_continuation_seed():
    """No completed bar after the anchor yet (an entry on the newest session,
    e.g. Sunday overnight): the pending candidate is returned as a SEED rather
    than nothing — superseded once the next completed bar lands."""
    c, h, lo = _story_series()
    piv = live_next_pivot(c[:86], h[:86], lo[:86], 85)    # pending peak IS bar 85
    assert piv is not None and piv["seed"] is True
    assert piv["resolved"] is False
    assert piv["is_peak"] is True
    assert piv["idx"] == 85 and piv["price"] == pytest.approx(PEAK_PX)
    # anchor past the extreme with nothing after it: the walk alternates
    # (peak@79 stands → wiggle trough@80 stands → running up-candidate @82)
    # and seeds at the freshest candidate it reached
    piv2 = live_next_pivot(c[:83], h[:83], lo[:83], 82)   # wiggle bars after peak@79
    assert piv2 is not None and piv2["seed"] is True
    assert piv2["is_peak"] is True
    assert piv2["idx"] == 82
    assert piv2["price"] == pytest.approx(100.8 * 1.001)


def test_no_training_cap_on_resolved_targets():
    """``next_pivot_targets`` excludes pivots > MAX_PIVOT_DAYS out (a training
    hygiene rule); the monitoring surface deliberately reports them."""
    c = np.r_[np.linspace(10, 30, 100), np.linspace(29.9, 20, 20)]
    c, h, lo = _hl(c, band=0.01)
    sp, _end = next_pivot_targets(c, h, lo)
    assert np.isnan(sp[5])                                 # capped out of training
    piv = live_next_pivot(c, h, lo, 5)
    assert piv is not None and piv["resolved"] is True
    assert piv["idx"] == 99


def test_short_window_returns_none():
    c, h, lo = _story_series()
    assert live_next_pivot(c[:49], h[:49], lo[:49], 10) is None


# ── tracker integration ──────────────────────────────────────────────────────

def _fake_series(c, h, lo, n=None):
    n = len(c) if n is None else n
    d = _dates(len(c))[:n]
    return d, c[:n], h[:n], lo[:n]


def _open_trade(dates, entry_bar, entry_px, cur_px, action="BUY"):
    return {
        "ticker": "TEST", "status": "OPEN", "action": action,
        "entry_date": dates[entry_bar].isoformat(),
        "entry_price": entry_px, "current_price": cur_px,
    }


def test_update_pivot_targets_resolved_long(monkeypatch):
    from src.performance import tracker
    c, h, lo = _story_series()
    d = _dates(len(c))
    monkeypatch.setattr("src.analysis.pivot_target._series",
                        lambda tk: _fake_series(c, h, lo))
    t = _open_trade(d, 66, 95.0, 100.0)          # anchor bar = 65
    tracker._update_pivot_targets([t])
    assert t["pivot_resolved"] is True
    assert t["pivot_target_price"] == pytest.approx(PEAK_PX, abs=1e-3)
    assert t["pivot_target_date"] == d[85].isoformat()
    assert t["pivot_confirmed_date"] == d[86].isoformat()
    assert t["pivot_target_pct"] == pytest.approx((PEAK_PX / 95.0 - 1) * 100, abs=1e-2)
    assert t["pivot_capture_pct"] == pytest.approx(
        (100.0 - 95.0) / (PEAK_PX - 95.0) * 100, abs=0.1)


def test_update_pivot_targets_live_extension(monkeypatch):
    """Provisional peak + a live mark above the completed-bars extreme: the
    target extends to the mark (dated today) — but only extends, never resolves."""
    from src.performance import tracker
    c, h, lo = _story_series()
    d = _dates(len(c))
    monkeypatch.setattr("src.analysis.pivot_target._series",
                        lambda tk: _fake_series(c, h, lo, n=86))   # pre-confirmation
    t = _open_trade(d, 66, 95.0, 103.5)          # mark above 102.6 extreme
    tracker._update_pivot_targets([t])
    assert t["pivot_resolved"] is False
    assert t["pivot_target_price"] == pytest.approx(103.5)
    assert t["pivot_target_date"] == date.today().isoformat()
    t2 = _open_trade(d, 66, 95.0, 101.0)         # mark below the extreme: no extension
    tracker._update_pivot_targets([t2])
    assert t2["pivot_target_price"] == pytest.approx(PEAK_PX, abs=1e-3)
    assert t2["pivot_target_date"] == d[85].isoformat()


def test_update_pivot_targets_provisional_short(monkeypatch):
    """A SELL riding the pending down leg: target below entry (negative pct),
    capture positive as price falls toward it; a mark ABOVE the running low
    must not shrink the trough candidate."""
    from src.performance import tracker
    c, h, lo = _story_series()
    d = _dates(len(c))
    monkeypatch.setattr("src.analysis.pivot_target._series",
                        lambda tk: _fake_series(c, h, lo))
    t = _open_trade(d, 90, 98.0, 97.0, action="SELL")   # anchor 89, pending trough @99
    tracker._update_pivot_targets([t])
    trough = c[99] * 0.999
    assert t["pivot_resolved"] is False
    assert t["pivot_is_peak"] is False
    assert t["pivot_target_price"] == pytest.approx(trough, abs=1e-3)
    assert t["pivot_target_pct"] == pytest.approx((trough / 98.0 - 1) * 100, abs=1e-2)
    assert t["pivot_capture_pct"] == pytest.approx(
        (97.0 - 98.0) / (trough - 98.0) * 100, abs=0.1)
    assert t["pivot_capture_pct"] > 0
