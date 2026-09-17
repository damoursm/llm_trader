"""Probes for the LIVE pivot target — ``live_next_pivot`` and the tracker's
per-tick ``_update_pivot_targets`` — on 30-minute bars.

The contract: for an open trade anchored at its entry fill, the surface
reports the first RESOLVED 30-minute pivot strictly after the fill (settled at
its confirming bar), and until then a PROVISIONAL value at the FRESHEST close
— the live mark when it is newer than the last cached bar — never the running
leg's extreme. The scan refactor must leave the resolved sequence identical
(``_resolved_pivots`` is the label machine).
"""

from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd
import pytest

from src.analysis.pivot_target import _pivot_scan, _resolved_pivots
from tests.intraday_fixtures import sessions_30m, stub_30m


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
IDX = sessions_30m("2026-08-03", 8)      # 104 bar starts; the story uses the first 100


# ── the scan refactor must not move the resolved sequence ────────────────────

def test_scan_and_resolved_pivots_agree():
    c, h, lo = _story_series()
    (P, PP, FL, CF), pending = _pivot_scan(c, h, lo)
    P2, PP2, FL2, CF2 = _resolved_pivots(c, h, lo)
    assert list(P) == list(P2) and list(PP) == list(PP2)
    assert list(FL) == list(FL2) and list(CF) == list(CF2)
    # the flat opening seeds a peak at bar 0 (confirmed by the fall at 50), then the
    # trough @58 (confirmed 59) and the peak @85 (confirmed by the crash @86)
    assert [int(x) for x in P] == [0, 58, 85] and [bool(x) for x in FL] == [True, False, True]
    assert [int(x) for x in CF] == [50, 59, 86]
    assert pending is not None and pending[2] is False     # a down leg is pending


# ── tracker integration ──────────────────────────────────────────────────────

def _series(n=None):
    c, h, lo = _story_series()
    n = len(c) if n is None else n
    return IDX[:n], c[:n], h[:n], lo[:n]


def _open_trade(entry_bar, entry_px, cur_px, action="BUY"):
    """An open trade whose fill lands one minute INTO bar ``entry_bar`` — so the
    first eligible bar is ``entry_bar + 1``."""
    return {
        "ticker": "TEST", "status": "OPEN", "action": action,
        "entry_datetime": (IDX[entry_bar] + pd.Timedelta(minutes=1)).isoformat(),
        "entry_date": IDX[entry_bar].date().isoformat(),
        "entry_price": entry_px, "current_price": cur_px,
    }


def test_update_pivot_targets_resolved_long(monkeypatch):
    from src.performance import tracker
    stub_30m(monkeypatch, {"TEST": _series()})
    t = _open_trade(65, 95.0, 100.0)
    tracker._update_pivot_targets([t])
    assert t["pivot_resolved"] is True and t["pivot_is_peak"] is True
    assert t["pivot_target_price"] == pytest.approx(PEAK_PX, abs=1e-3)
    assert t["pivot_target_ts"] == IDX[85].isoformat()
    assert t["pivot_target_date"] == IDX[85].date().isoformat()          # EDT: bar date == session date
    assert t["pivot_confirmed_date"] == IDX[86].date().isoformat()
    assert t["pivot_target_pct"] == pytest.approx((PEAK_PX / 95.0 - 1) * 100, abs=1e-2)
    assert t["pivot_capture_pct"] == pytest.approx((100.0 - 95.0) / (PEAK_PX - 95.0) * 100, abs=0.1)


def test_update_pivot_targets_provisional_at_last_close_and_live_mark(monkeypatch):
    """Pre-confirmation (series cut before bar 86): the target is the LAST
    CLOSE (bar 85's close), never the running extreme; a live mark replaces
    it — whichever way it points — but never resolves it."""
    from src.performance import tracker
    stub_30m(monkeypatch, {"TEST": _series(n=86)})
    t = _open_trade(65, 95.0, 103.5)
    tracker._update_pivot_targets([t])
    assert t["pivot_resolved"] is False and t["pivot_target_ts"] is None
    assert t["pivot_target_price"] == pytest.approx(103.5)
    assert t["pivot_target_date"] == date.today().isoformat()
    t2 = _open_trade(65, 95.0, None)
    tracker._update_pivot_targets([t2])
    assert t2["pivot_resolved"] is False
    assert t2["pivot_target_price"] == pytest.approx(102.5)              # the last close, not 102.6 (the high)


def test_update_pivot_targets_provisional_short(monkeypatch):
    """A SELL riding the pending down leg: no resolved pivot after the fill, so
    the target is the last close (below entry → negative pct), capture positive
    as price falls toward it."""
    from src.performance import tracker
    stub_30m(monkeypatch, {"TEST": _series()})
    t = _open_trade(89, 98.0, 97.0, action="SELL")
    tracker._update_pivot_targets([t])
    last_close = 98.8 - 0.2 * 12
    assert t["pivot_resolved"] is False and t["pivot_is_peak"] is False
    assert t["pivot_target_price"] == pytest.approx(97.0)                # the live mark stands in
    t2 = _open_trade(89, 98.0, None, action="SELL")
    tracker._update_pivot_targets([t2])
    assert t2["pivot_target_price"] == pytest.approx(last_close, abs=1e-6)
    assert t2["pivot_target_pct"] == pytest.approx((last_close / 98.0 - 1) * 100, abs=1e-2)


def test_update_pivot_targets_date_only_entry_anchors_at_the_open(monkeypatch):
    """A legacy trade with no entry timestamp anchors at its session's 09:30 ET
    open, so the entry session's own swing can be the target."""
    from src.performance import tracker
    stub_30m(monkeypatch, {"TEST": _series()})
    t = {"ticker": "TEST", "status": "OPEN", "action": "BUY",
         "entry_date": IDX[6 * 13].date().isoformat(), "entry_price": 95.0, "current_price": 100.0}
    tracker._update_pivot_targets([t])
    assert t["pivot_resolved"] is True and t["pivot_target_ts"] == IDX[85].isoformat()


def test_no_30m_history_leaves_the_trade_untouched(monkeypatch):
    from src.performance import tracker
    stub_30m(monkeypatch, {})
    t = _open_trade(65, 95.0, 100.0)
    tracker._update_pivot_targets([t])
    assert "pivot_target_price" not in t
