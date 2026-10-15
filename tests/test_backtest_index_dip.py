"""PREREG110's N3 adapter (`src/backtest/strategies/index_dip.py`): the bull-market index dip's decisions read the
decision symbol's closes through D-1 only, enter at D, exit at the open after the first close above the 5-session
average (or after 10 sessions), one long at a time."""
import numpy as np
import pandas as pd

from src.backtest.strategies import index_dip as ID


def _frame(n=320, seed=0):
    rng = np.random.default_rng(seed)
    r = 0.0006 + 0.01 * rng.standard_normal(n)
    r[250:253] = -0.03                                  # a dip in an uptrend
    c = 100 * np.exp(np.cumsum(r))
    days = pd.bdate_range("2020-01-01", periods=n)
    return pd.DataFrame({"o": c, "h": c * 1.01, "l": c * 0.99, "c": c}, index=days)


def test_a_dip_in_an_uptrend_is_bought_the_next_session_and_one_at_a_time():
    d = _frame()
    T = ID.trades(d.index[0], d.index[-1], daily=d)
    assert T, "the injected dip must trigger"
    for t in T:
        assert t["entry"] > t["signal"] and (t["exit"] is None or t["exit"] > t["entry"])
    ends = [t["exit"] for t in T if t["exit"] is not None]
    for a, b in zip(ends, [t["entry"] for t in T][1:]):
        assert b >= a                                   # never two longs at once


def _decisions(d, lo, hi):
    return [(t["signal"], t["entry"]) for t in ID.trades(lo, hi, daily=d)]


def _violations(d):
    """Entries whose decision changes when every close from the entry day on is rewritten (a crash, then a rally): a
    decision must read the closes through D-1 only."""
    full = _decisions(d, d.index[0], d.index[-1])
    bad = 0
    for sig, ent in full:
        for k in (0.5, 2.0):
            later = d.copy()
            later.loc[later.index >= ent, ["o", "h", "l", "c"]] *= k
            if (sig, ent) not in _decisions(later, d.index[0], d.index[-1]):
                bad += 1
    return bad, len(full)


def test_no_lookahead_rewriting_the_entry_day_and_later_changes_no_decision():
    res = [_violations(_frame(seed=seed)) for seed in range(6)]
    assert sum(n for _, n in res) > 0 and all(bad == 0 for bad, _ in res)


def test_the_guard_catches_an_injected_leak(monkeypatch):
    """A rule whose D-1 RSI reads D's close must be caught (the guard has teeth)."""
    from src.signals import dip_long

    real = dip_long.rsi2

    def leaky(c):
        r = real(np.asarray(c, float))
        return np.r_[r[1:], np.nan]
    monkeypatch.setattr(dip_long, "rsi2", leaky)
    assert sum(_violations(_frame(seed=seed))[0] for seed in range(6)) > 0
