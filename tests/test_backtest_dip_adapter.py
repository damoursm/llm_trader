"""The dip book's backtest adapter (src/backtest/strategies/dip.py) against production's rule (src/signals/dip_long.py),
on synthetic daily bars (PREREG63): a signal needs MIN_BARS daily bars through its own close, as `stats_at` does, and
a name that stops trading before its exit (delisted, acquired) is sold at its last close instead of vanishing."""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.backtest.strategies import dip as DIP


def series(n, dips=(), start="2020-01-02", end_drop=None):
    """A rising daily series (close above its 200-session average) with two-day 3% drops at ``dips`` (RSI2 < 10);
    ``end_drop``: a drop over the last sessions that never recovers."""
    idx = pd.bdate_range(start, periods=n)
    c = 100.0 * 1.002 ** np.arange(n)
    for k in dips:
        c[k] *= 0.97
        c[k + 1] *= 0.97 * 0.97
    if end_drop is not None:
        c[end_drop:] = c[end_drop - 1] * 0.95 ** np.arange(1, n - end_drop + 1)
    v = np.full(n, 2e7)                                           # $2B a day: above the $1B floor
    return pd.DataFrame({"o": c, "h": c * 1.01, "l": c * 0.99, "c": c, "v": v}, index=idx)


def test_a_signal_needs_min_bars_through_its_own_close():
    """Production's `stats_at` returns None below 220 bars through D-1: a dip at bar 210 is no signal."""
    d = series(300, dips=(210, 260))
    S = DIP.signals({"AAA": d}, DIP.LIVE, d.index[0], d.index[-1], names=["AAA"])
    sig_bars = [int(d.index.get_loc(t)) for t in S["sig"]]
    assert all(i >= DIP.MIN_BARS - 1 for i in sig_bars)
    assert any(i >= 261 for i in sig_bars)                        # the later dip is a signal


def test_a_name_that_stops_trading_before_its_exit_is_sold_at_its_last_close():
    full = series(320)
    gone = series(300, dips=(296,), end_drop=298)                 # its data ends 20 sessions before the panel's
    panel = {"AAA": full, "GONE@D": gone}
    S = DIP.signals(panel, DIP.LIVE, gone.index[250], full.index[-1], names=["GONE@D"])
    row = S[S["sig"] >= gone.index[296]]
    assert len(row) and bool(row.iloc[-1]["last_close"]) and row.iloc[-1]["d_out"] == gone.index[-1]
    P = DIP.pieces(DIP.LIVE, gone.index[250], full.index[-1], panel=panel, granularity="daily", names=["GONE@D"])
    last = [p for p in P if p["xns"] == pd.Timestamp(f"{gone.index[-1].date()} 16:00", tz=DIP.ET).value]
    assert last and last[-1]["x"] == float(gone["c"].iloc[-1])


def test_a_trade_whose_exit_lies_past_the_data_is_still_left_out():
    """A name still trading at the data's end: its exit is unknown, not a delisting."""
    d = series(300, dips=(296,), end_drop=298)
    S = DIP.signals({"AAA": d}, DIP.LIVE, d.index[250], d.index[-1], names=["AAA"])
    assert not (S["sig"] >= d.index[296]).any()


def test_a_class_share_stored_under_both_spellings_is_one_name(monkeypatch):
    """PREREG64: the deep store holds Berkshire as BRK-B.pkl AND BRK.B.pkl (identical bars); production's candidates
    hold it once in the project's form (`dip_long.candidates`), so the adapter does too."""
    from src.data import company_names
    monkeypatch.setattr(company_names, "security_type", lambda t: "CS")
    d = series(260)
    panel = {"BRK-B": d, "BRK.B": d, "AAPL": d, "OLD.A@D": d, "OLD-A@D": d, "ONLY.C": d}
    assert DIP.eligible_names(panel) == ["AAPL", "BRK-B", "OLD-A@D", "ONLY.C"]


def test_the_gap_guard_scales_with_the_time_limit():
    """PREREG97: a hold spanning more calendar days than the guard is read as a data gap and skipped; at the live
    10-session limit the guard stays 20 days (parity), a 15- or 20-session limit gets twice its sessions — a fixed 20
    dropped the longest real holds, the worst trades."""
    import dataclasses
    from src.backtest.strategies import dip as DIP
    assert DIP.gap_days(DIP.LIVE) == 20
    assert DIP.gap_days(dataclasses.replace(DIP.LIVE, max_hold=5)) == 20
    assert DIP.gap_days(dataclasses.replace(DIP.LIVE, max_hold=15)) == 30
    assert DIP.gap_days(dataclasses.replace(DIP.LIVE, max_hold=20)) == 40


def test_the_rsi2_recovery_exit_sells_once_the_oscillator_recovers():
    """PREREG98 S2: exit_rule "rsi2" exits at the first session (from the entry's) whose RSI2 is above exit_level;
    the live "sma" exit is unchanged."""
    import dataclasses
    import numpy as np
    from src.backtest.strategies import dip as DIP
    c = np.array([100, 99, 98, 97, 96, 97, 99, 101, 102, 103], float)
    sma = np.full(len(c), 98.5)
    r2 = DIP.rsi2(c)
    live = DIP.LIVE
    assert DIP.exit_hit(live, c, sma, None, 6) and not DIP.exit_hit(live, c, sma, None, 3)
    p50 = dataclasses.replace(live, exit_rule="rsi2", exit_level=50.0)
    first = next(k for k in range(4, len(c)) if DIP.exit_hit(p50, c, sma, r2, k))
    assert first == next(k for k in range(4, len(c)) if r2[k] > 50.0) == 5      # the first up close lifts RSI2
    p90 = dataclasses.replace(live, exit_rule="rsi2", exit_level=90.0)
    assert next(k for k in range(4, len(c)) if DIP.exit_hit(p90, c, sma, r2, k)) > first
