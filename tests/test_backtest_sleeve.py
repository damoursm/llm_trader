"""PREREG154's research book, the QQQ sleeve (rejected 2026-10-19; no production book): the engine's minimum position
(`Strategy.min_frac`), the sleeve adapter's trend and lots (`src/backtest/strategies/sleeve.py`, no look-ahead) and
its switch (`Mimic(sleeve=...)`, off by default)."""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.backtest import account as A
from src.backtest import mimic as M
from src.backtest.strategies import sleeve as SLEEVE

DAY = 86400 * 10**9
T0 = pd.Timestamp("2024-01-08 15:00", tz="UTC").value


def settle(dn):
    return int(dn) + 1


def piece(strategy, tkn, e, x, days=2, t0=T0):
    bars = list(np.linspace(e, x, days + 1))[1:]
    return {"strategy": strategy, "tkn": tkn, "ens": t0, "xns": t0 + days * DAY, "e": e, "x": x, "hs_in": 1.0,
            "hs_out": 1.0, "dv20": 1e12, "pt": [t0 + (k + 1) * DAY for k in range(len(bars))], "pc": bars}


def _run(min_frac):
    big = A.Strategy("big", 1, slices=1.25, priority=0)               # takes 80% of the long side's cap
    sl = A.Strategy("sleeve", 1, slices=1.0, priority=99, min_frac=min_frac)
    P = [piece("big", "AAA", 100.0, 100.0), piece("sleeve", "QQQ", 10.0, 10.0)]
    return A.simulate(P, [big, sl], A.Rules(), settle=settle, keep_skips=True)


def test_min_frac_skips_a_lot_under_its_share_of_equity():
    res = _run(0.30)                                                  # ~20% left on the long side: under 30%
    assert [p["strategy"] for p in res["positions"]] == ["big"]
    assert any(s.get("reason") == "min_frac" for s in res["skips"])


def test_min_frac_lets_a_lot_at_or_above_it_open():
    res = _run(0.10)
    assert sorted(p["strategy"] for p in res["positions"]) == ["big", "sleeve"]


def test_min_frac_off_by_default():
    assert A.Strategy("x", 1).min_frac is None


def _daily(closes):
    idx = pd.bdate_range("2024-01-01", periods=len(closes))
    return pd.DataFrame({"c": closes, "o": closes}, index=idx)


def test_on_days_read_only_the_previous_close():
    c = [10.0] * 5 + [12.0, 8.0, 12.0, 12.0]
    d = _daily(c)
    on = SLEEVE.on_days(d, n=3)
    later = d.copy()
    later.iloc[6:, later.columns.get_loc("c")] = 1000.0               # rewrite every bar from D = 6 on
    on2 = SLEEVE.on_days(later, n=3)
    assert list(on.iloc[:7]) == list(on2.iloc[:7])                    # D's decision never reads D or later
    assert bool(on.iloc[6]) == bool(c[5] > np.mean(c[3:6]))


def test_lots_exit_at_the_first_off_session_and_at_the_set_end():
    c = [10.0] * 3 + [11.0, 12.0, 13.0, 9.0, 9.0, 14.0, 15.0]
    d = _daily(c)
    on = SLEEVE.on_days(d, n=3)
    days = d.index
    L = SLEEVE.lots(days[0], days[-1], d, n=3)
    assert L, "an ON spell gives lots"
    for lot in L:
        assert bool(on.loc[lot["entry"]])
        if lot["exit"] is not None:
            assert not bool(on.loc[lot["exit"]]) or lot["exit"] > days[-1]
    capped = SLEEVE.lots(days[0], days[4], d, n=3)                          # a set ending at session 4
    assert all(x["exit"] is None or x["exit"] <= days[5] for x in capped)


def test_mimic_sleeve_is_off_by_default_and_switches_its_strategy_and_rule():
    m = M.Mimic()
    assert m.sleeve is False
    assert "sleeve" not in [s.name for s in M.live_strategies(m)]
    assert M.live_rules(m).fills_make_room is False
    on = M.Mimic(sleeve=True)
    s = [x for x in M.live_strategies(on) if x.name == "sleeve"]
    assert len(s) == 1 and s[0].min_frac == SLEEVE.MIN_FRAC and s[0].quality == 0.0 and s[0].preemptible
    assert M.live_rules(on).fills_make_room is True
