"""IBKR's liquidation in the account engine (PREREG174, `Rules(liquidation="ibkr")`; the user 2026-10-21: "I want the
backtest to do exactly the same [as IBKR during margin calls], and, if we can't be exact, approximate closely and be
only slightly on the safe side"). IBKR's "Margin Calculations for Securities" page: equity under the maintenance
requirement is liquidated — the most recent transactions first, in 100-share increments, until excess liquidity is
back at zero —, its Soft Edge Margin tolerates a deficit up to 10% of the account's value until 15 minutes before the
close, and its end-of-day Reg T check (from 15:50 ET) wants SMA >= 0, whose floor is equity minus Reg T's initial
requirement. These tests pin each rule on synthetic accounts."""
from __future__ import annotations

import pandas as pd
import pytest

from src.backtest import account as A

ET = "America/New_York"


def settle(dn):
    return int(dn) + 1                                        # T+1 on every day (synthetic calendar)


def et(s: str) -> int:
    return pd.Timestamp(s, tz=ET).value


def leg(strategy, tkn, e, ens, xns, x, bars, hs=1.0, **kw):
    """A trade entered at ``ens`` at ``e``, exited at ``xns`` at ``x``, marked at ``bars`` [(instant, close)]."""
    p = {"strategy": strategy, "tkn": tkn, "ens": et(ens), "xns": et(xns), "e": e, "x": x, "hs_in": hs, "hs_out": hs,
         "dv20": 1e9, "pt": [et(t) for t, _ in bars], "pc": [float(c) for _, c in bars]}
    p.update(kw)
    return p


IB = A.Rules(trigger="close", liquidation="ibkr")


def squeeze(rules):
    """Three Reg T shorts of 125 shares at $20 (1/4 slices): OLD (Monday 10:00), NEW1 (11:00), NEW2 (12:00); OLD's
    Tuesday 10:00 bar closes at $80 — equity about $2,500 against a $4,500 requirement."""
    s = A.Strategy("S", -1, slices=4, min_equity=2000.0)
    P = [leg("S", "OLD", 20.0, "2024-01-08 10:00", "2024-01-11 10:00", 20.0,
             [("2024-01-09 10:00", 80.0), ("2024-01-10 10:00", 20.0)]),
         leg("S", "NEW1", 20.0, "2024-01-08 11:00", "2024-01-11 11:00", 20.0, [("2024-01-10 11:00", 20.0)]),
         leg("S", "NEW2", 20.0, "2024-01-08 12:00", "2024-01-11 12:00", 20.0, [("2024-01-10 12:00", 20.0)])]
    return A.simulate(P, [s], rules, settle=settle)


def test_the_default_rules_still_close_everything():
    assert A.Rules().liquidation == "all"
    r = squeeze(A.Rules(trigger="close"))
    assert len(r["calls"]) == 1 and {p["how"] for p in r["positions"]} == {"margin_call"}
    assert "liquidations" not in r


def test_ibkr_liquidates_the_most_recent_positions_first_in_lots_until_back_at_maintenance():
    r = squeeze(IB)
    assert [p["shares"] for p in r["positions"] if p["tkn"] == "OLD"] == [125]
    assert len(r["calls"]) == 1 and r["call_kinds"] == ["maint"]
    L = r["liquidations"]
    # the most recent first: NEW2 and NEW1 whole (each cures too little), then ONE lot of OLD cures the rest
    assert [(x["tkn"], x["closed"]) for x in L] == [("NEW2", True), ("NEW1", True), ("OLD", False)]
    assert [x["shares"] == x["held"] for x in L[:2]] == [True, True] and L[2]["shares"] == 100
    assert all(x["t"] == et("2024-01-09 10:00") and x["px"] == (80.0 if x["tkn"] == "OLD" else 20.0) for x in L)
    how = {p["tkn"]: p for p in r["positions"]}
    assert how["NEW1"]["how"] == how["NEW2"]["how"] == "margin_call"
    assert how["OLD"]["how"] == "exit" and how["OLD"]["liquidated"] == 100
    # after it equity is back above the requirement: OLD's 25 shares at $80, Reg T's 30%
    t, eq_before, eq_after = r["calls"][0]
    assert eq_before < 4500.0 and eq_after >= 0.30 * 80.0 * 25
    # the 25 shares left ride to the exit at $20 (about flat): the position's P&L is the lot's loss, about
    lot_pnl = next(x["pnl"] for x in L if x["tkn"] == "OLD")
    assert lot_pnl < -5900.0 and how["OLD"]["pnl"] == pytest.approx(lot_pnl, abs=5.0)


def test_the_requirement_order_covers_the_squeezed_short_alone():
    r = squeeze(A.Rules(trigger="close", liquidation="ibkr", liq_order="requirement"))
    assert [(x["tkn"], x["shares"]) for x in r["liquidations"]] == [("OLD", 100)]
    assert {p["tkn"]: p["how"] for p in r["positions"]} == {"OLD": "exit", "NEW1": "exit", "NEW2": "exit"}


def test_a_position_under_one_lot_goes_whole_and_lots_are_hundreds():
    s = A.Strategy("S", -1, slices=1, min_equity=2000.0)
    # 500 shares at $20 (Reg T: 30% maintenance); Tuesday's bar at $32: equity $4,000 against $4,800
    P = [leg("S", "SQ", 20.0, "2024-01-08 10:00", "2024-01-11 10:00", 20.0, [("2024-01-09 11:00", 32.0)])]
    r = A.simulate(P, [s], IB, settle=settle)
    L = r["liquidations"]
    assert [x["shares"] for x in L] == [100] and r["positions"][0]["liquidated"] == 100
    # a lot of 100 at $32 releases $960 of requirement: the $800 deficit needs one; none would leave it short
    s2 = A.Strategy("S", -1, slices=1 / 0.08, min_equity=0.0)    # 40 shares: under one lot
    # $210: equity about $2,400 against $2,520
    P2 = [leg("S", "SQ", 20.0, "2024-01-08 10:00", "2024-01-11 10:00", 20.0, [("2024-01-09 11:00", 210.0)])]
    r2 = A.simulate(P2, [s2], IB, settle=settle)
    assert [(x["shares"], x["closed"]) for x in r2["liquidations"]] == [(40, True)]


def _one_short(bars, sem, xns="2024-01-12 10:00"):
    s = A.Strategy("S", -1, slices=1, min_equity=2000.0)        # 500 shares at $20
    P = [leg("S", "SQ", 20.0, "2024-01-08 10:00", xns, 20.0, bars)]
    return A.simulate(P, [s], A.Rules(trigger="close", liquidation="ibkr", sem=sem), settle=settle)


def test_soft_edge_margin_tolerates_a_small_deficit_in_regular_hours_until_15_45():
    # $31.20: equity $4,400 against $4,680 — a deficit of 6% of equity
    assert _one_short([("2024-01-09 11:00", 31.2)], sem=0.10)["calls"] == []
    assert len(_one_short([("2024-01-09 11:00", 31.2)], sem=0.0)["calls"]) == 1      # strict: liquidated
    assert len(_one_short([("2024-01-09 16:00", 31.2)], sem=0.10)["calls"]) == 1     # after 15:45: the full requirement
    assert len(_one_short([("2024-01-09 17:00", 31.2)], sem=0.10)["calls"]) == 1     # after hours: the full requirement
    assert len(_one_short([("2024-01-09 09:30", 31.2)], sem=0.10)["calls"]) == 0     # from the open
    # $32: a 20% deficit breaks it at 11:00; the session's later small deficit is then liquidated at once
    r = _one_short([("2024-01-09 11:00", 32.0), ("2024-01-09 12:00", 32.5)], sem=0.10)
    assert [pd.Timestamp(c[0], tz="UTC").tz_convert(ET).strftime("%H:%M") for c in r["calls"]] == ["11:00", "12:00"]
    # the same small deficit on a fresh session is tolerated again
    r = _one_short([("2024-01-09 11:00", 32.0), ("2024-01-10 12:00", 32.5)], sem=0.10)
    assert len(r["calls"]) == 1


def _two_levered_longs(bar_at, regt_close=True):
    s = A.Strategy("L", 1, slices=1)                           # Reg T: 25% maintenance, 50% initial
    bars = [(bar_at, 80.0)]
    P = [leg("L", "L1", 100.0, "2024-01-08 10:00", "2024-01-10 10:00", 80.0, bars),
         leg("L", "L2", 100.0, "2024-01-08 11:00", "2024-01-10 10:00", 80.0, bars)]
    return A.simulate(P, [s], A.Rules(trigger="close", gross_cap={1: 2.0, -1: 1.0}, liquidation="ibkr",
                                      regt_close=regt_close), settle=settle)


def test_the_reg_t_floor_is_checked_from_15_50_and_sells_the_latest_long():
    # about $16,000 of longs on $6,000 of equity: maintenance (25%) holds, Reg T's 50% does not
    r = _two_levered_longs("2024-01-09 16:00")
    assert r["call_kinds"] == ["regt"]
    assert [(x["tkn"], x["why"], x["closed"]) for x in r["liquidations"]] == [("L2", "regt", True)]
    assert {p["tkn"]: p["how"] for p in r["positions"]} == {"L1": "exit", "L2": "margin_call"}
    assert _two_levered_longs("2024-01-09 11:00")["calls"] == []                 # before 15:50: not checked
    assert _two_levered_longs("2024-01-09 16:00", regt_close=False)["calls"] == []


def test_ruin_when_liquidating_everything_leaves_no_equity():
    s = A.Strategy("S", -1, slices=1, min_equity=2000.0)
    P = [leg("S", "SQ", 20.0, "2024-01-08 10:00", "2024-01-11 10:00", 20.0, [("2024-01-09 11:00", 50.0)])]
    r = A.simulate(P, [s], IB, settle=settle)
    assert r["ruined"] == et("2024-01-09 11:00") and r["final"] < 0
    assert [(x["shares"], x["closed"]) for x in r["liquidations"]] == [(500, True)]


def test_the_path_sells_at_each_crossing_one_band_above_it_never_past_the_bar():
    """PREREG174 Amendments 2-3: IBKR's real-time check along the bar — a lot when the deficit appears, filled one LULD
    band (10%) above that price, more lots as the price keeps rising, never above the bar's high."""
    s = A.Strategy("S", -1, slices=1, min_equity=2000.0)        # 500 shares at $20, Reg T's 30%
    P = [leg("S", "SQ", 20.0, "2024-01-08 10:00", "2024-01-11 10:00", 20.0, [("2024-01-09 11:00", 25.0)], ph=[40.0])]
    path = A.simulate(P, [s], A.Rules(trigger="high", liquidation="ibkr", liq_fill="path"), settle=settle)
    L = path["liquidations"]
    # equity $20,000 - $650 x the price: the deficit appears at $30.76; then at $31.95, $33.59 and $36.17
    assert [x["shares"] for x in L] == [100, 100, 100, 100]
    assert [x["mark"] for x in L] == pytest.approx([30.76, 31.95, 33.59, 36.17], abs=0.06)
    assert all(x["px"] == pytest.approx(min(40.0, 1.1 * x["mark"])) for x in L)
    assert path["ruined"] is None and next(p for p in path["positions"])["liquidated"] == 400
    # the same bar judged at its extreme all at once: the whole short covered at $40 — the account is gone
    ext = A.simulate(P, [s], A.Rules(trigger="high", liquidation="ibkr"), settle=settle)
    assert [(x["shares"], x["px"]) for x in ext["liquidations"]] == [(500, 40.0)] and ext["ruined"] is not None
    # a bar that never breaks the requirement liquidates nothing on either rule
    P2 = [dict(P[0], ph=[30.0])]
    assert A.simulate(P2, [s], A.Rules(trigger="high", liquidation="ibkr", liq_fill="path"), settle=settle)["calls"] == []


def test_the_rules_refuse_ibkr_options_without_ibkr_and_bad_values():
    with pytest.raises(ValueError):
        A.Rules(sem=0.1)
    with pytest.raises(ValueError):
        A.Rules(regt_close=True)
    with pytest.raises(ValueError):
        A.Rules(liquidation="partial")
    with pytest.raises(ValueError):
        A.Rules(liquidation="ibkr", liq_order="largest")
    with pytest.raises(ValueError):
        A.Rules(liquidation="ibkr", liq_lot=0)
    with pytest.raises(ValueError):
        A.Rules(liquidation="ibkr", sem=1.0)
    with pytest.raises(ValueError):
        A.Rules(liquidation="ibkr", liq_fill="worst")
    with pytest.raises(ValueError):
        A.Rules(liquidation="ibkr", liq_slip=1.5)
