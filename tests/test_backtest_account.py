"""The multi-strategy account engine (src/backtest/account.py; user directive 2026-10-08): real fees, available
capital, margin calls, several strategies sharing one $10,000 account. Bit-parity with the audited short engine
(cap5k7) is checked on the real live vol book by the research harness (scratchpad optvol/bt_parity.py, 2026-10-08:
4 of 4 runs identical); these tests pin the rules on synthetic trades."""
from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from src.backtest import account as A

DAY = 86400 * 10**9
T0 = pd.Timestamp("2024-01-08 15:00", tz="UTC").value       # a Monday, 10:00 ET


def settle(dn):
    return int(dn) + 1                                        # T+1 on every day (synthetic calendar)


def piece(strategy, tkn, e, x, days=2, bars=None, hs=1.0, dv20=1e9, t0=T0, **kw):
    """A trade entered at ``t0`` at ``e`` and exited ``days`` days later at ``x``, marked once a day at ``bars``."""
    bars = list(bars) if bars is not None else list(np.linspace(e, x, days + 1))[1:]
    pt = [t0 + (k + 1) * DAY for k in range(len(bars))]
    p = {"strategy": strategy, "tkn": tkn, "ens": t0, "xns": t0 + days * DAY, "e": e, "x": x, "hs_in": hs,
         "hs_out": hs, "dv20": dv20, "pt": pt, "pc": bars}
    p.update(kw)
    return p


def short_borrow(rate_pct_yr, price, days):
    """A short's schedule: the day's collateral roundup(1.02 x price) x rate, cumulated from its first charged day."""
    d_in = A.day_numbers([T0])[0]
    d0 = settle(d_in)
    v = np.ceil(1.02 * price) * rate_pct_yr
    return {"d0": d0, "cum": np.cumsum(np.full(days + 30, v)), "bfac": 1 / 100 / 360}


def test_a_long_books_its_fees_exactly():
    s = A.Strategy("L", 1, slices=10)
    r = A.simulate([piece("L", "AAA", 100.0, 110.0)], [s], A.Rules(), settle=settle)
    pos = r["positions"][0]
    n = pos["shares"]
    assert n == 10                                            # 1/10 of $10,000 at $100
    cin = n * 100.0 * 1.0 / 1e4 + A.comm(n, 100.0)            # a buy: half-spread + commission
    cout = n * 110.0 * 1.0 / 1e4 + A.comm(n, 110.0) + A.sale_fees(n, 110.0)   # a sale: + SEC fee + TAF
    assert pos["pnl"] == pytest.approx(n * 10.0 - cin - cout)
    assert r["final"] == pytest.approx(10_000.0 + pos["pnl"])


def test_a_short_pays_its_sale_fees_and_its_borrow():
    s = A.vol_short("S")
    b = short_borrow(50.0, 50.0, 5)
    r = A.simulate([piece("S", "BBB", 50.0, 45.0, days=3, **b)], [s], A.Rules(), settle=settle)
    pos = r["positions"][0]
    n = pos["shares"]
    assert n == int((10_000 / 24) // 50.0)                    # 1/24 of the account
    cin = n * 50.0 * 1.0 / 1e4 + A.comm(n, 50.0) + A.sale_fees(n, 50.0)
    cout = n * 45.0 * 1.0 / 1e4 + A.comm(n, 45.0)
    d_out = A.day_numbers([T0 + 3 * DAY])[0]
    charged_days = (settle(d_out) - 1) - b["d0"] + 1
    borrow = b["cum"][charged_days - 1] * b["bfac"]
    assert pos["pnl"] == pytest.approx(n * 5.0 - cin - cout - n * borrow)
    assert borrow > 0


def test_a_squeeze_through_the_bar_high_is_a_margin_call_and_closes_everything():
    s = A.vol_short("S", slices=2)                            # two big shorts: a squeeze breaks the margin
    p1 = piece("S", "SQZ", 10.0, 10.0, days=3, bars=[10.0, 10.0, 10.0], ph=[10.0, 60.0, 10.0])
    p2 = piece("S", "CALM", 10.0, 10.0, days=3, bars=[10.0, 10.0, 10.0], rank=1)
    hi = A.simulate([p1, p2], [s], A.Rules(trigger="high"), settle=settle)
    assert len(hi["calls"]) == 1
    assert {r["how"] for r in hi["positions"]} == {"margin_call"}
    cl = A.simulate([p1, p2], [s], A.Rules(trigger="close"), settle=settle)
    assert cl["calls"] == [] and {r["how"] for r in cl["positions"]} == {"exit"}


def test_a_levered_long_is_margin_called_on_its_bar_low_only_with_the_intrabar_trigger():
    """An unlevered long can never be called (its equity always covers 25% of its value); a 2x long whose bar low
    is 60% under the entry is — with the intrabar trigger, not on the unchanged close."""
    s = A.Strategy("L", 1, slices=0.5)
    p = piece("L", "AAA", 100.0, 100.0, days=2, bars=[100.0, 100.0], pl=[40.0, 100.0])   # the low before the exit bar
    lev = A.Rules(trigger="high", gross_cap={1: 2.0, -1: 1.0})
    r = A.simulate([p], [s], lev, settle=settle)
    assert r["positions"][0]["shares"] > 150 and len(r["calls"]) == 1
    assert A.simulate([p], [s], A.Rules(trigger="close", gross_cap={1: 2.0, -1: 1.0}), settle=settle)["calls"] == []
    unlev = A.simulate([dict(p, pl=[1.0, 100.0])], [A.Strategy("L", 1, slices=1)], A.Rules(), settle=settle)
    assert unlev["calls"] == []


def test_slices_max_open_and_the_side_cap_limit_a_strategy():
    s = A.Strategy("L", 1, slices=4, max_open=3)
    ps = [piece("L", f"T{k}", 100.0, 100.0, rank=k) for k in range(6)]
    r = A.simulate(ps, [s], A.Rules(), settle=settle)
    assert r["trades"] == 3                                   # max_open
    s2 = A.Strategy("L", 1, slices=2)                         # half the account each: the third finds no room
    r2 = A.simulate(ps[:3], [s2], A.Rules(), settle=settle)
    assert [p["shares"] for p in r2["positions"]] == [50, 49] or r2["trades"] == 2
    assert sum(p["shares"] * 100.0 for p in r2["positions"]) <= 10_000.0


def test_entries_at_one_instant_follow_strategy_priority_then_rank():
    a = A.Strategy("first", 1, slices=1, priority=0)          # takes the whole account
    b = A.Strategy("second", 1, slices=1, priority=1)
    ps = [piece("second", "B", 100.0, 100.0), piece("first", "A", 100.0, 100.0)]
    r = A.simulate(ps, [a, b], A.Rules(), settle=settle)
    assert [p["strategy"] for p in r["positions"]] == ["first"]


def test_the_short_book_needs_room_the_long_book_has_used():
    """Longs at 50% initial margin + shorts at IBKR's 2.86x house rate share ONE room: a full long book leaves the
    shorts a third of the size they would have alone."""
    longs = [piece("L", f"L{k}", 100.0, 100.0, rank=k) for k in range(10)]
    shorts = [piece("S", f"S{k}", 10.0, 10.0, t0=T0 + 3600 * 10**9, rank=k) for k in range(30)]
    st = [A.Strategy("L", 1, slices=10, max_open=10, priority=0), A.vol_short("S", priority=1)]
    both = A.simulate(longs + shorts, st, A.Rules(), settle=settle)
    alone = A.simulate(shorts, [A.vol_short("S")], A.Rules(), settle=settle)
    gross = lambda r, side: sum(p["shares"] * p["entry_px"] for p in r["positions"] if p["side"] == side)  # noqa: E731
    assert gross(both, 1) >= 8_500.0                          # ten longs of ~1/10 (net of their entry costs)
    assert 0 < gross(both, -1) < 0.7 * gross(alone, -1)
    room = 10_000.0 - 0.5 * gross(both, 1)                    # what the longs leave of the initial-margin room
    assert gross(both, -1) * 2.86 <= room + 1e-6


def test_a_short_needs_finras_minimum_equity():
    s = A.vol_short("S")
    r = A.simulate([piece("S", "X", 10.0, 9.0)], [s], A.Rules(start=1_500.0), settle=settle)
    assert r["trades"] == 0


def test_the_adv_cap_limits_the_order():
    s = A.vol_short("S", slices=1)
    r = A.simulate([piece("S", "THIN", 10.0, 9.0, dv20=50_000.0)], [s], A.Rules(), settle=settle)
    assert r["positions"][0]["shares"] * 10.0 <= 0.01 * 50_000.0


def test_metrics_split_both_metrics_by_strategy_and_add_up():
    st = [A.Strategy("L", 1, slices=10), A.vol_short("S")]
    ps = [piece("L", "AAA", 100.0, 105.0, days=4), piece("S", "BBB", 20.0, 18.0, days=2, rank=1)]
    r = A.simulate(ps, st, A.Rules(), settle=settle)
    m = A.metrics(r, pd.Timestamp(T0, unit="ns"), pd.Timestamp(T0 + 365 * DAY, unit="ns"))
    assert set(m["by_strategy"]) == {"L", "S"}
    assert sum(v["pnl"] for v in m["by_strategy"].values()) == pytest.approx(r["final"] - 10_000.0)
    assert m["log_growth"] == pytest.approx(math.log(r["final"] / 10_000.0), rel=1e-3)
    L = m["by_strategy"]["L"]
    assert L["avg_days"] == pytest.approx(4.0) and L["return_per_day"] == pytest.approx(A.per_day(L["avg_ret"], 4.0))
    assert m["log_contrib_per_year"] == pytest.approx(m["log_growth"], abs=1e-3)   # no overlap -> they add up


def test_an_unknown_strategy_is_refused():
    with pytest.raises(KeyError):
        A.simulate([piece("ghost", "X", 10.0, 10.0)], [A.vol_short("S")], A.Rules(), settle=settle)


def test_the_exposure_record_holds_each_sides_value_and_its_initial_margin():
    lo, sh = A.Strategy("L", 1, slices=10), A.vol_short("S", slices=24)
    pl = piece("L", "AAA", 100.0, 100.0, days=3, bars=[100.0, 100.0, 100.0])
    ps = piece("S", "BBB", 50.0, 50.0, days=3, bars=[50.0, 50.0, 50.0])
    plain = A.simulate([pl, ps], [lo, sh], A.Rules(), settle=settle, keep_curve=True)
    r = A.simulate([pl, ps], [lo, sh], A.Rules(), settle=settle, keep_exposure=True)
    assert r["final"] == plain["final"] and np.array_equal(r["curve_v"], plain["curve_v"])   # recording changes nothing
    nl = next(p["shares"] for p in r["positions"] if p["strategy"] == "L")
    ns = next(p["shares"] for p in r["positions"] if p["strategy"] == "S")
    k = 1                                                     # the first mark after both entries
    assert r["curve_long"][k] == pytest.approx(nl * 100.0) and r["curve_short"][k] == pytest.approx(ns * 50.0)
    # a long's Reg T 50% + a short's IBKR house rate 2.86 x its value
    assert r["curve_req"][k] == pytest.approx(0.5 * nl * 100.0 + 2.86 * ns * 50.0)
    assert r["curve_long"][-1] == 0 and r["curve_short"][-1] == 0 and r["curve_req"][-1] == 0


def test_activity_counts_trades_per_year_and_the_sessions_holding_a_position():
    s = A.Strategy("L", 1, slices=10)
    mon = pd.Timestamp("2024-01-08 09:30", tz="America/New_York").value         # bought at Monday's open
    wed = pd.Timestamp("2024-01-10 09:30", tz="America/New_York").value         # sold at Wednesday's open
    p = piece("L", "AAA", 100.0, 101.0, days=2, t0=mon)
    p["xns"] = wed
    r = A.simulate([p], [s], A.Rules(), settle=settle)
    days = pd.bdate_range("2024-01-08", "2024-01-12")
    act = A.activity(r, days, "2024-01-08", "2024-01-12")
    assert act["span"]["trades"] == {"L": 1}
    # held Monday and Tuesday; sold AT Wednesday's open, so Wednesday is not a holding day
    assert act["span"]["days_held"] == 2 and act["span"]["sessions"] == 5
    assert act["years"]["2024"]["held_share"] == pytest.approx(0.4)


def test_entries_the_account_cannot_take_in_full_are_recorded_with_their_binding_limit():
    sh = A.vol_short("S", slices=2)                           # half the account a short: IBKR's 2.86x cannot fit
    P = [piece("S", "AAA", 50.0, 50.0, days=3, bars=[50.0] * 3), piece("S", "BBB", 50.0, 50.0, days=3, bars=[50.0] * 3)]
    plain = A.simulate(P, [sh], A.Rules(), settle=settle)
    r = A.simulate(P, [sh], A.Rules(), settle=settle, keep_skips=True)
    assert r["final"] == plain["final"]                       # the record changes nothing
    first, second = r["skips"]
    assert first["reason"] == "margin" and 0 < first["got"] < first["want"] == 100   # cut to the room
    assert second["reason"] == "margin" and second["got"] == 0                       # no room left: skipped
    s = A.capital_summary(r, P)["S"]
    assert (s["entries"], s["opened"], s["cut"], s["skipped"]) == (2, 1, 1, 1)
    lo = A.Strategy("L", 1, slices=10, max_open=1)
    r2 = A.simulate([piece("L", "C", 10.0, 11.0), piece("L", "D", 10.0, 11.0)], [lo], A.Rules(), settle=settle,
                    keep_skips=True)
    assert [x["reason"] for x in r2["skips"]] == ["max_open"]
