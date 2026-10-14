"""The market context a model is activated in (src/backtest/context.py; PREREG108): point in time, no look-ahead."""
import numpy as np
import pandas as pd

from src.backtest import context as CX


def _spy(n=900, seed=0):
    rng = np.random.default_rng(seed)
    days = pd.bdate_range("2015-01-02", periods=n)
    r = rng.normal(0.0004, 0.01, size=n)
    r[500:540] *= 4                                         # a volatile stretch
    return pd.DataFrame({"Close": 100 * np.exp(np.cumsum(r))}, index=days)


def test_a_decision_reads_the_close_before_its_day():
    spy = _spy()
    tab = CX.table(spy)
    d = spy.index[700]
    assert CX.at([d], tab)[0] == tab.cell.iloc[699]
    day_number = int((d - CX.EPOCH).days)
    assert CX.at([day_number], tab)[0] == tab.cell.iloc[699]
    assert CX.at([spy.index[0]], tab)[0] is None


def test_no_look_ahead_a_cut_history_and_rewritten_later_closes_change_nothing():
    spy = _spy()
    full = CX.table(spy)
    for k in (300, 520, 610, 899):
        cut = CX.table(spy.iloc[:k + 1])
        assert list(cut.cell) == list(full.cell.iloc[:k + 1])
        later = spy.copy()
        later.iloc[k + 1:, 0] = later.iloc[k + 1:, 0].to_numpy() * np.linspace(0.5, 3.0, len(later) - k - 1)
        assert list(CX.table(later).cell.iloc[:k + 1]) == list(full.cell.iloc[:k + 1])


def test_the_terciles_are_the_expanding_history_s_and_the_volatile_stretch_reads_high():
    spy = _spy()
    tab = CX.table(spy)
    i = 400
    rv = tab.rvol.iloc[:i + 1].dropna().to_numpy()
    assert abs(tab.lo.iloc[i] - np.quantile(rv, 1 / 3)) < 1e-12 and abs(tab.hi.iloc[i] - np.quantile(rv, 2 / 3)) < 1e-12
    assert (tab.vol.iloc[525:540] == "high").all()
    assert tab.cell.dropna().isin(CX.CELLS).all()
    first = tab.vol.first_valid_index()
    assert tab.rvol.loc[:first].notna().sum() == CX.MIN_HISTORY      # 250 values of volatility before the first label
