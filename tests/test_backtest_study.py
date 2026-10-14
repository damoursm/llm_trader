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


# ── PREREG63: the bugs the 2026-10-09 review confirmed ───────────────────────

def test_a_tie_with_the_reference_keeps_the_reference_itself():
    X = np.array([[0.01, 0.02], [0.01, 0.02], [0.0, 0.0]])          # combos 0 and 1 identical in the window
    g = np.array([0.5, 0.5, 0.0])
    choice, _ = S.select(X, g, ref=1, seed=0)
    assert choice["M0"] == 1 and choice["M4"] == 1


def test_a_working_order_before_the_cut_holds_its_place_in_the_window():
    """An order placed before the cut whose touch comes after it was dropped from the window: the window's account had
    a free slot it never had."""
    p = piece("A", "2024-01-03", "2024-01-09", "2024-01-12")         # touch Tuesday after the Sunday cut
    p.update(oid="A|1", ord_place=[ns("2024-01-04 09:30") + 1, ns("2024-01-05 09:30") + 1,
                                   ns("2024-01-08 09:30") + 1], ord_lim=99.0, ord_end=ns("2024-01-10 16:00"), ord_k=2)
    cut = S.cut_of("2024-01-07")
    (q,) = S.window_pieces([p], S.dnum("2024-01-01"), S.dnum("2024-01-06"), cut)
    assert q["ord_k"] == -1 and q["placeholder"] and q["ens"] >= cut
    assert q["ord_place"] == [ns("2024-01-04 09:30") + 1, ns("2024-01-05 09:30") + 1]


def test_a_standing_holding_is_bought_at_a_windows_and_a_runs_start():
    """A dual-momentum holding decided before a window (or a run's start) and still held then was invisible to it."""
    p = piece("QQQ", "2024-01-31", "2024-02-01", "2024-03-28", e=100.0, x=120.0, strategy="dm")
    p["rebase"] = True
    lo = "2024-03-04"
    (q,) = S.window_pieces([p], S.dnum(lo), S.dnum("2024-03-08"), None)
    assert q["ens"] >= ns(f"{lo} 00:00") and q["e"] > 100.0 and q["rebased_from"] == p["ens"]
    (r,) = A.from_start([p], ns(f"{lo} 00:00"))
    assert r["ens"] == q["ens"] and r["xns"] == p["xns"]
    plain = dict(p, rebase=False)
    assert A.from_start([plain], ns(f"{lo} 00:00")) == []
    assert S.window_pieces([plain], S.dnum(lo), S.dnum("2024-03-08"), None) == []


def test_the_drawdown_counts_a_loss_from_the_start():
    assert S.max_drawdown([0.95, 0.90, 1.2]) == pytest.approx(10.0)
