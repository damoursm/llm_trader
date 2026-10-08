"""The weekly walk-forward driver (src/backtest/study.py): pieces sliced by decision day and truncated at the Sunday
cut (nothing after the cut is seen), the Reality Check keeping the reference without a real edge, the chained
account assembling each week's chosen rules. Reproduction of PREREG35's per-week live scores on the real vol book is
checked by the research harness (optvol/bt_study_check.py, 2026-10-08: 42 windows, same trades, |diff| <= 2e-4)."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.backtest import account as A
from src.backtest import study as S

DAY = 86400 * 10**9


def settle(dn):
    return int(dn) + 1


def ns(ts):
    return pd.Timestamp(ts, tz="America/New_York").value


def piece(tkn, pick, enter, exit_, e=100.0, x=110.0, strategy="L"):
    pt = [ns(f"{d} 16:00") for d in pd.bdate_range(enter, exit_)]
    return {"strategy": strategy, "tkn": tkn, "pick_day": S.dnum(pick), "ens": ns(f"{enter} 09:30"),
            "xns": ns(f"{exit_} 16:00"), "e": e, "x": x, "hs_in": 1.0, "hs_out": 1.0, "dv20": 1e9,
            "pt": pt, "pc": list(np.linspace(e, x, len(pt))), "pl": list(np.linspace(e, x, len(pt)))}


def test_a_trade_open_at_the_cut_is_marked_at_its_last_bar_before_it():
    p = piece("A", "2024-01-04", "2024-01-05", "2024-01-12")          # Fri entry, exit the next Friday
    cut = S.cut_of("2024-01-07")                                      # Sunday 19:00
    (q,) = S.window_pieces([p], S.dnum("2024-01-01"), S.dnum("2024-01-06"), cut)
    assert q["xns"] == ns("2024-01-05 16:00") and q["x"] == p["pc"][0] and q["marked_at_cut"]
    assert all(t <= cut for t in q["pt"])
    assert S.window_pieces([p], S.dnum("2024-01-01"), S.dnum("2024-01-06"), None)[0]["xns"] == p["xns"]


def test_a_decision_outside_the_window_or_an_entry_after_the_cut_is_left_out():
    late = piece("B", "2024-01-05", "2024-01-08", "2024-01-09")        # decided Friday, entered Monday
    cut = S.cut_of("2024-01-07")
    assert S.window_pieces([late], S.dnum("2024-01-01"), S.dnum("2024-01-06"), cut) == []
    old = piece("C", "2023-12-01", "2023-12-04", "2023-12-05")
    assert S.window_pieces([old], S.dnum("2024-01-01"), S.dnum("2024-01-06"), None) == []


def test_the_reality_check_keeps_the_reference_without_a_real_edge():
    rng = np.random.default_rng(1)
    X = rng.normal(0, 0.01, (20, 250))                                # 20 combos of pure noise
    g = X.sum(axis=1)
    pick, diag = S.select(X, g, ref=3, seed=0)
    assert pick["M4"] == 3 and diag["rc_p"] >= 0.10
    assert pick["M0"] == int(np.argmax(g)) or g[3] >= g.max() - 1e-12
    X2 = X.copy()
    X2[7] += 0.004                                                    # a real, large edge
    pick2, diag2 = S.select(X2, X2.sum(axis=1), ref=3, seed=0)
    assert pick2["M4"] == 7 and diag2["rc_p"] < 0.10


def test_the_chained_account_takes_each_weeks_chosen_rule():
    a = [piece("A1", "2024-01-08", "2024-01-09", "2024-01-10"), piece("A2", "2024-01-15", "2024-01-16", "2024-01-17")]
    b = [piece("B1", "2024-01-08", "2024-01-09", "2024-01-10", x=90.0),
         piece("B2", "2024-01-15", "2024-01-16", "2024-01-17", x=90.0)]
    arms = {"L": S.Arm("L", A.Strategy("L", 1, slices=10), {"a": a, "b": b})}
    combos = [S.Combo("a", (("L", "a"),)), S.Combo("b", (("L", "b"),))]
    st = S.Study(arms, combos, reference=0, first="2024-01-07", last="2024-01-14", test_cap="2024-01-31",
                 data_lo="2023-01-01", windows=(12,))
    c = S.chained(st, [0, 1], settle)
    got = sorted(p["tkn"] for p in c["res"]["positions"])
    assert got == ["A1", "B2"]
