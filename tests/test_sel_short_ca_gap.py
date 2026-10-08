"""Corporate-action gaps in the vol and ETF arms (user 2026-10-05: "Spin-offs aren't in the split data, so a
CTVA-style gap can block the slot again" — fix it).

What must hold: a session that opens at least 40% under the previous close while the name trades calmly around
it is flagged for the 5 sessions from the gap (while the ATR% without session-opening gaps is under 40% of the
full one); a genuine crash that keeps trading wildly is not flagged past its first bars; a smaller gap or a gap
up is never flagged; nothing before the gap is; the vol and ETF arms leave flagged bars out of their ranking
(and so out of their freshness history), the model arm does not.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.signals import sel_short as ss


def _series(gap: float, after_range: float, sessions_before: int = 30, sessions_after: int = 10, seed: int = 3):
    """Regular-hours 30-minute bars: calm at 100 (0.3% bar ranges), then a session opening at
    100 x (1 + gap) whose bars range ``after_range`` of the price."""
    rng = np.random.default_rng(seed)
    sday, hi, lo, cl = [], [], [], []
    for s in range(sessions_before + sessions_after):
        level = 100.0 if s < sessions_before else 100.0 * (1.0 + gap)
        width = 0.003 if s < sessions_before else after_range
        for _ in range(13):
            c = level * (1.0 + rng.normal(0.0, width / 3.0))
            sday.append(20000 + s)
            hi.append(c * (1.0 + width / 2.0))
            lo.append(c * (1.0 - width / 2.0))
            cl.append(c)
    return np.array(sday), np.array(hi), np.array(lo), np.array(cl)


def _flagged_sessions(sday, flags):
    return sorted(set(int(d) - 20000 for d in sday[flags]))


def test_a_calm_level_shift_is_flagged_while_the_gap_dominates_and_nothing_before_it():
    sday, hi, lo, cl = _series(gap=-0.80, after_range=0.004)       # a spin-off: 100 -> 20, then calm
    fl = ss.corporate_gap_flags(sday, hi, lo, cl)
    got = _flagged_sessions(sday, fl)
    assert got[:3] == [30, 31, 32] and set(got) <= {30, 31, 32, 33, 34}   # from the gap, at most five sessions
    assert fl[30 * 13: 31 * 13].all()                               # every bar of the gap session
    assert not fl[: 30 * 13].any() and not fl[35 * 13:].any()


def test_a_crash_that_keeps_trading_wildly_and_smaller_or_upward_gaps_are_not_flagged():
    sday, hi, lo, cl = _series(gap=-0.60, after_range=0.15)        # a genuine crash, 15% bar ranges after
    fl = ss.corporate_gap_flags(sday, hi, lo, cl)
    assert not fl[31 * 13:].any()                                   # from the next session on: volatility, ranked
    for gap in (-0.30, +1.00):                                      # a 30% gap down; a gap up
        sday, hi, lo, cl = _series(gap=gap, after_range=0.004)
        assert not ss.corporate_gap_flags(sday, hi, lo, cl).any()


def test_the_vol_and_etf_arms_leave_flagged_bars_out_and_the_model_arm_does_not(monkeypatch):
    monkeypatch.setattr(ss, "load_model", lambda: (None, {"tickers": ["AAA", "CTVA", "UVIX"]}))
    monkeypatch.setitem(ss._TYPES, "t", {"AAA": "CS", "CTVA": "CS", "UVIX": "ETF", "GAPE": "ETF"})
    monkeypatch.setattr(ss, "vol_listing_names", lambda: [])
    res = pd.DataFrame({"ticker": ["AAA", "CTVA", "UVIX", "GAPE"], "status": "OK", "score": [1.0, 2.0, 0.5, 0.1],
                        "vol": [3.0, 9.0, 4.0, 8.0], "px": 20.0, "dv20": 1e7, "pre5": 10.0,
                        "ca_gap": [False, True, None, True], "ca_gap_pct": [None, -81.0, None, -45.0]})
    assert ss.arm_rows(res, "vol")["ticker"].tolist() == ["AAA", "UVIX"]
    assert ss.arm_rows(res, "vol", ca_gaps=True)["ticker"].tolist() == ["CTVA"]
    assert ss.arm_rows(res, "etf")["ticker"].tolist() == ["UVIX"]
    assert ss.arm_rows(res, "etf", ca_gaps=True)["ticker"].tolist() == ["GAPE"]
    assert ss.arm_rows(res, "model")["ticker"].tolist() == ["AAA", "CTVA", "UVIX"]
    assert ss.arm_rows(res, "model", ca_gaps=True).empty


def test_the_backfill_reads_the_flags_of_its_rows_from_the_whole_series():
    sday, hi, lo, cl = _series(gap=-0.80, after_range=0.004)
    idx = pd.DatetimeIndex(pd.Timestamp("1970-01-01") + pd.to_timedelta(sday, unit="D")
                           + pd.to_timedelta(14 * 60 + 30 * (np.arange(len(sday)) % 13), unit="min"))
    tail = ss._ca_gap_tail((idx, hi, lo, cl, np.ones(len(cl))), 13 * 12)     # the last 12 sessions
    full = ss.corporate_gap_flags(sday, hi, lo, cl)
    assert (tail == full[-13 * 12:]).all() and tail.any()
    assert len(ss._ca_gap_tail((idx, hi, lo, cl, np.ones(len(cl))), len(cl) + 5)) == len(cl) + 5
