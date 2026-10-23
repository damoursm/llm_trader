"""The multi-strategy account engine (src/backtest/account.py; user directive 2026-10-08): real fees, available
capital, margin calls, several strategies sharing one $10,000 account. Bit-parity with the audited short engine
(cap5k7) is checked on the real live vol book by the research harness (scratchpad optvol/bt_parity.py, 2026-10-08:
4 of 4 runs identical); these tests pin the rules on synthetic trades."""
from __future__ import annotations

import dataclasses
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


def test_the_exposure_record_splits_the_open_value_by_strategy():
    lo, sh = A.Strategy("L", 1, slices=10), A.vol_short("S", slices=24)
    pl = piece("L", "AAA", 100.0, 100.0, days=3, bars=[100.0, 100.0, 100.0])
    ps = piece("S", "BBB", 50.0, 50.0, days=3, bars=[50.0, 50.0, 50.0])
    plain = A.simulate([pl, ps], [lo, sh], A.Rules(), settle=settle, keep_curve=True)
    r = A.simulate([pl, ps], [lo, sh], A.Rules(), settle=settle, keep_exposure=True)
    assert r["final"] == plain["final"] and np.array_equal(r["curve_v"], plain["curve_v"])   # recording changes nothing
    assert set(r["curve_by"]) == {"L", "S"}
    assert np.allclose(r["curve_by"]["L"], r["curve_long"]) and np.allclose(r["curve_by"]["S"], r["curve_short"])
    days = pd.bdate_range("2024-01-01", "2024-01-10")
    ex = A.exposure_daily(r, days)
    assert {"by:L", "by:S"} <= set(ex.columns)
    assert np.allclose(ex["by:L"], ex["long"]) and np.allclose(ex["by:S"], ex["short"])


# ── capital priority (PREREG52) ─────────────────────────────────────────────

def test_a_better_entry_sells_just_enough_of_the_filler_to_take_its_whole_slice():
    """The filler (the whole account in one fund, quality 1) gives way to a better long (quality 2) that finds the
    long side full: it sells about a tenth of its shares at its mark, the better long gets its whole 1/10 slice."""
    fill = A.Strategy("F", 1, slices=1, max_open=1, priority=0, quality=1.0, preemptible=True)
    best = A.Strategy("D", 1, slices=10, priority=1, quality=2.0)
    pf = piece("F", "FUND", 100.0, 100.0, days=5, bars=[100.0] * 5)
    pd_ = piece("D", "DIP", 50.0, 50.0, days=2, bars=[50.0, 50.0], t0=T0 + DAY // 2)
    r = A.simulate([pf, pd_], [fill, best], A.Rules(), settle=settle, keep_skips=True)
    f = next(p for p in r["positions"] if p["strategy"] == "F")
    d = next(p for p in r["positions"] if p["strategy"] == "D")
    assert d["shares"] == int(d["equity_in"] / 10 // 50.0)          # its whole slice
    (s,) = r["preemptions"]
    assert s["strategy"] == "F" and s["by"] == "D" and not s["closed"]
    assert 9 <= s["shares"] <= 12 and f["sold"] == s["shares"]      # ~10% of the fund's ~99 shares
    assert not [k for k in r["skips"] if k["strategy"] == "D"]      # nothing cut or skipped
    # without priorities the same entry finds no room
    plain = A.simulate([pf, pd_], [A.Strategy("F", 1, slices=1, max_open=1, priority=0),
                                   A.Strategy("D", 1, slices=10, priority=1)], A.Rules(), settle=settle)
    assert [p["strategy"] for p in plain["positions"]] == ["F"] and "preemptions" not in plain


def test_a_short_short_of_margin_room_sells_the_filler():
    fill = A.Strategy("F", 1, slices=1, max_open=1, priority=0, quality=1.0, preemptible=True)
    sh = A.Strategy("S", -1, slices=4, priority=1, min_equity=2000.0, house_maint=2.0, house_init=2.86, quality=3.0)
    pf = piece("F", "FUND", 100.0, 100.0, days=5, bars=[100.0] * 5)
    ps = piece("S", "HOT", 10.0, 10.0, days=2, bars=[10.0, 10.0], t0=T0 + DAY // 2)
    r = A.simulate([pf, ps], [fill, sh], A.Rules(), settle=settle)
    s = next(p for p in r["positions"] if p["strategy"] == "S")
    assert s["shares"] == int(s["equity_in"] / 4 // 10.0)           # the whole quarter: the fund made the margin room
    assert sum(x["shares"] for x in r["preemptions"]) > 0


def test_the_filler_buys_back_once_the_room_returns():
    fill = A.Strategy("F", 1, slices=1, max_open=1, priority=0, quality=1.0, preemptible=True, refill_min=0.05)
    best = A.Strategy("D", 1, slices=10, priority=1, quality=2.0)
    pf = piece("F", "FUND", 100.0, 100.0, days=6, bars=[100.0] * 6)
    pd_ = piece("D", "DIP", 50.0, 50.0, days=2, bars=[50.0, 50.0], t0=T0 + DAY // 2)
    r = A.simulate([pf, pd_], [fill, best], A.Rules(), settle=settle)
    (s,) = r["preemptions"]
    assert r["refills"] and r["refills"][0]["t"] > pd_["xns"]       # after the better position left
    assert sum(x["shares"] for x in r["refills"]) >= s["shares"] - 2
    f = next(p for p in r["positions"] if p["strategy"] == "F")
    assert f["bought"] == sum(x["shares"] for x in r["refills"])


def test_no_priority_between_equals_and_never_for_a_non_preemptible_position():
    a = A.Strategy("A", 1, slices=1, max_open=1, priority=0, quality=2.0)            # not preemptible
    b = A.Strategy("B", 1, slices=10, priority=1, quality=5.0)
    pa = piece("A", "AAA", 100.0, 100.0, days=5, bars=[100.0] * 5)
    pb = piece("B", "BBB", 50.0, 50.0, days=2, t0=T0 + DAY // 2)
    r = A.simulate([pa, pb], [a, b], A.Rules(), settle=settle)
    assert r["preemptions"] == [] if "preemptions" in r else True
    assert [p["strategy"] for p in r["positions"]] == ["A"]
    eq = A.Strategy("E", 1, slices=1, max_open=1, priority=0, quality=2.0, preemptible=True)
    r2 = A.simulate([dict(pa, strategy="E"), pb], [eq, A.Strategy("B", 1, slices=10, priority=1, quality=2.0)],
                    A.Rules(), settle=settle)
    assert r2["preemptions"] == [] and [p["strategy"] for p in r2["positions"]] == ["E"]


def _session_piece(strategy, tkn, px, d0, d1, opens=True):
    """A holding bought at the 09:30 open of ``d0`` and sold at the 09:30 open of ``d1`` (ET dates), flat at ``px``,
    marked at each later session's 09:30 open and every session's 16:00 close."""
    ET_ = "America/New_York"
    days = pd.bdate_range(d0, d1)
    pt, pc = [], []
    for k, d in enumerate(days[:-1]):
        if opens and k > 0:
            pt.append(pd.Timestamp(f"{d.date()} 09:30", tz=ET_).value)
            pc.append(px)
        pt.append(pd.Timestamp(f"{d.date()} 16:00", tz=ET_).value)
        pc.append(px)
    return {"strategy": strategy, "tkn": tkn, "ens": pd.Timestamp(f"{days[0].date()} 09:30", tz=ET_).value,
            "xns": pd.Timestamp(f"{days[-1].date()} 09:30", tz=ET_).value, "e": px, "x": px, "hs_in": 1.0, "hs_out": 1.0,
            "dv20": 1e10, "pt": pt, "pc": pc}


def test_a_whole_exit_closes_the_lower_ranked_position_entirely():
    fund = A.Strategy("F", 1, slices=1, max_open=1, priority=3, quality=0.06, preemptible=True, yield_whole=True)
    dip = A.Strategy("D", 1, slices=10, priority=2, quality=0.13)
    pf = _session_piece("F", "FUND", 100.0, "2024-03-04", "2024-03-11")
    pd_ = dict(_session_piece("D", "DIP", 50.0, "2024-03-05", "2024-03-07"))
    r = A.simulate([pf, pd_], [fund, dip], A.Rules(), settle=settle)
    (s,) = r["preemptions"]
    assert s["closed"] and s["strategy"] == "F"
    f = [p for p in r["positions"] if p["strategy"] == "F"]
    assert len(f) == 1 and f[0]["how"] == "preempted"            # no re-entry without ``reenter``
    d = next(p for p in r["positions"] if p["strategy"] == "D")
    assert d["shares"] == int(d["equity_in"] / 10 // 50.0)


def test_an_exited_holding_rebuys_at_the_next_open_with_room():
    fund = A.Strategy("F", 1, slices=1, max_open=1, priority=3, quality=0.06, preemptible=True, yield_whole=True,
                      reenter=True)
    dip = A.Strategy("D", 1, slices=10, priority=2, quality=0.13)
    pf = _session_piece("F", "FUND", 100.0, "2024-03-04", "2024-03-11")
    pd_ = _session_piece("D", "DIP", 50.0, "2024-03-05", "2024-03-07")
    r = A.simulate([pf, pd_], [fund, dip], A.Rules(), settle=settle)
    (re,) = r["reentries"]
    # the dip book still holds its tenth on 03-06: the fund re-buys at that open with the room left (about 90%)
    assert re["t"] == pd.Timestamp("2024-03-06 09:30", tz="America/New_York").value
    f = [p for p in r["positions"] if p["strategy"] == "F"]
    assert [p["how"] for p in f] == ["preempted", "exit"] and f[1].get("reentry")
    assert 0.80 * 100 <= f[1]["shares"] <= 0.92 * 100
    # never re-bought by exiting another book: the dip position kept its whole slice
    assert len(r["preemptions"]) == 1


def test_after_a_margin_call_the_holding_rebuys_at_the_next_open():
    fund = A.Strategy("F", 1, slices=1, max_open=1, priority=0, reenter=True)
    sh = A.vol_short("S", slices=24, priority=1)
    pf = _session_piece("F", "FUND", 100.0, "2024-03-04", "2024-03-11")
    ET_ = "America/New_York"
    t_in = pd.Timestamp("2024-03-04 09:30", tz=ET_).value
    ps = {"strategy": "S", "tkn": "HOT", "ens": t_in, "xns": pd.Timestamp("2024-03-08 09:30", tz=ET_).value, "e": 10.0,
          "x": 10.0, "hs_in": 1.0, "hs_out": 1.0, "dv20": 1e9,
          "pt": [pd.Timestamp("2024-03-05 12:00", tz=ET_).value], "pc": [100.0]}
    r = A.simulate([pf, ps], [fund, sh], A.Rules(), settle=settle)
    assert r["calls"] and r["reentries"]
    assert r["reentries"][0]["t"] == pd.Timestamp("2024-03-06 09:30", tz=ET_).value


def test_a_margin_buffer_opens_a_short_only_within_its_share_of_equity():
    """PREREG57's margin buffer: with ``room_frac`` 0.5 a short may use only half of equity as initial requirement
    (IBKR's 2.86 house rate on a $10 short: 28.6 a share)."""
    p = piece("S", "BUF", 10.0, 10.0, days=2, bars=[10.0, 10.0])
    free = A.simulate([p], [A.vol_short("S", slices=1)], A.Rules(), settle=settle)
    half = A.simulate([p], [A.Strategy("S", -1, slices=1, min_equity=2000.0, house_maint=2.0, house_init=2.86,
                                       room_frac=0.5)], A.Rules(), settle=settle)
    assert free["positions"][0]["shares"] == 349                 # floor(10,000 / 28.6)
    assert half["positions"][0]["shares"] == 174                 # floor(5,000 / 28.6)


def test_deleveraging_sells_the_long_book_before_a_slow_squeeze_calls_the_account():
    """PREREG57's de-leveraging: a long filling the account and a short squeezed bar by bar (10 -> 13 -> 15 -> 18).
    Without it the account is margin-called at 18; with (0.8, 0.6) the long is sold at the close of 15 (requirement
    85% of equity), which keeps the account out of the call."""
    lng = A.Strategy("DM", 1, slices=1, priority=0, max_open=1, delever_order=0)
    sht = A.vol_short("S", slices=4, priority=1)
    pl = piece("DM", "SPY", 100.0, 100.0, days=6, bars=[100.0] * 6)
    ps = piece("S", "SQZ", 10.0, 18.0, days=4, bars=[13.0, 15.0, 18.0, 18.0])
    off = A.simulate([pl, ps], [lng, sht], A.Rules(trigger="close"), settle=settle)
    assert len(off["calls"]) == 1 and "delevers" not in off
    on = A.simulate([pl, ps], [lng, sht], A.Rules(trigger="close", delever=(0.8, 0.6)), settle=settle)
    assert on["calls"] == []
    assert [(d["strategy"], d["tkn"]) for d in on["delevers"]] == [("DM", "SPY")]
    assert on["delevers"][0]["need"] > 0.8 * on["delevers"][0]["equity"]
    how = {r["tkn"]: r["how"] for r in on["positions"]}
    assert how == {"SPY": "delever", "SQZ": "exit"}
    # a strategy without a delever order is never sold by it
    keep = A.simulate([pl, ps], [dataclasses_replace(lng, delever_order=None), sht],
                      A.Rules(trigger="close", delever=(0.8, 0.6)), settle=settle)
    assert keep["delevers"] == [] and len(keep["calls"]) == 1


def dataclasses_replace(obj, **kw):
    import dataclasses
    return dataclasses.replace(obj, **kw)


def _three_dips_and_a_short():
    """Three dip longs bought on days 0, 1, 2 (marks flat) and, on day 3, a short the margin room cannot fund."""
    lng = A.Strategy("D", 1, slices=3.2, priority=1, max_open=3, quality=0.1, preemptible=True, yield_whole=True)
    sht = A.Strategy("S", -1, slices=1.2, priority=0, quality=1.0, min_equity=2000.0, house_maint=2.0, house_init=2.86)
    P = [piece("D", f"L{k}", 100.0, 100.0, days=8 - k, bars=[100.0] * (8 - k), t0=T0 + k * DAY) for k in range(3)]
    P.append(piece("S", "SQ", 10.0, 10.0, days=2, bars=[10.0, 10.0], t0=T0 + 3 * DAY + 3600 * 10**9))
    return P, lng, sht


def test_the_dip_book_gives_way_latest_or_earliest_first():
    """PREREG58: ``yield_first`` picks which position gives way — the latest opened (default) or the earliest."""
    P, lng, sht = _three_dips_and_a_short()
    late = A.simulate(P, [lng, sht], A.Rules(trigger="close"), settle=settle)
    early = A.simulate(P, [dataclasses_replace(lng, yield_first="earliest"), sht], A.Rules(trigger="close"),
                       settle=settle)
    assert [p["tkn"] for p in late["preemptions"]][:1] == ["L2"]
    assert [p["tkn"] for p in early["preemptions"]][:1] == ["L0"]
    with pytest.raises(ValueError):
        dataclasses_replace(lng, yield_first="oldest")


def test_the_dip_book_never_gives_way_for_nothing():
    """PREREG64: a short whose book's initial requirement is already past the equity cannot be funded whatever the
    longs free — production's give-way (`live_account.give_way_plan`) sells nothing then, nor does the engine (it
    used to sell the long, for nothing)."""
    lng = A.Strategy("D", 1, slices=4, priority=1, max_open=3, quality=0.1, preemptible=True, yield_whole=True)
    sht = A.Strategy("S", -1, slices=4, priority=0, quality=1.0, min_equity=2000.0, house_maint=2.0, house_init=2.86)
    h = 3600 * 10**9
    P = [piece("D", "L0", 100.0, 100.0, days=8, bars=[100.0] * 8),
         piece("S", "A", 10.0, 14.0, days=6, bars=[12.0, 14.0, 14.0, 14.0, 14.0, 14.0], t0=T0 + h),
         piece("S", "B", 10.0, 10.0, days=2, bars=[10.0, 10.0], t0=T0 + 2 * DAY + 2 * h)]
    r = A.simulate(P, [lng, sht], A.Rules(trigger="close"), settle=settle)
    assert not r.get("preemptions")
    assert [p["how"] for p in r["positions"] if p["tkn"] == "L0"] == ["exit"]
    assert not [p for p in r["positions"] if p["tkn"] == "B"]       # the short is still not funded
    assert {p["how"] for p in r["positions"]} == {"exit"}           # and no margin call


def test_the_other_side_gives_way_while_the_margin_room_holds_an_entry_below_its_sides_cap():
    """PREREG64, production's rule: an entry the margin room holds below its side-capped size sells the other
    side's preemptible positions (they free room) even when its side's cap also binds below its slice."""
    lng = A.Strategy("D", 1, slices=2.5, priority=1, max_open=3, quality=0.1, preemptible=True, yield_whole=True)
    sht = A.Strategy("S", -1, slices=2, priority=0, quality=1.0, min_equity=2000.0)
    h = 3600 * 10**9
    P = [piece("D", "L0", 100.0, 100.0, days=8, bars=[100.0] * 8),
         piece("S", "A", 10.0, 10.0, days=6, bars=[10.0] * 6, t0=T0 + h, dv20=3.98e5, hm=1.5, hi=2.0),
         piece("S", "B", 10.0, 10.0, days=2, bars=[10.0, 10.0], t0=T0 + DAY + 2 * h, hm=1.5, hi=2.0)]
    rules = A.Rules(trigger="close", gross_cap={1: 1.0, -1: 0.5})
    r = A.simulate(P, [lng, sht], rules, settle=settle)
    (a,) = [p for p in r["positions"] if p["tkn"] == "A"]
    (b,) = [p for p in r["positions"] if p["tkn"] == "B"]
    assert a["shares"] == 398                                       # 1% of its dollar volume: $40 of room left
    assert [(x["tkn"], x["by_tkn"]) for x in r["preemptions"]] == [("L0", "B")]
    assert 95 <= b["shares"] <= 102                                 # the side's cap: ~$1,020 of the $5,000 slice


def test_deleveraging_sells_latest_or_earliest_first():
    """PREREG58: ``delever_first`` picks which position de-leveraging sells first."""
    lng = A.Strategy("D", 1, slices=4, priority=0, max_open=4, delever_order=0)
    sht = A.vol_short("S", slices=4, priority=1)
    P = [piece("D", f"L{k}", 100.0, 100.0, days=6, bars=[100.0] * 6, t0=T0 + k * 3600 * 10**9) for k in range(4)]
    P.append(piece("S", "SQZ", 10.0, 18.0, days=4, bars=[13.0, 15.0, 18.0, 18.0], t0=T0 + 5 * 3600 * 10**9))
    late = A.simulate(P, [lng, sht], A.Rules(trigger="close", delever=(0.8, 0.6)), settle=settle)
    early = A.simulate(P, [dataclasses_replace(lng, delever_first="earliest"), sht],
                       A.Rules(trigger="close", delever=(0.8, 0.6)), settle=settle)
    assert late["delevers"] and early["delevers"]
    assert late["delevers"][0]["tkn"] == "L3" and early["delevers"][0]["tkn"] == "L0"


# ── limit orders (PREREG61) ──────────────────────────────────────────────────

H = 3600 * 10**9


def order(strategy, tkn, oid, lim, place, end, fills, rank=0.0):
    """One limit order: placed at the instants ``place``, lapsing at ``end``; ``fills`` = {its window session k:
    (touch instant, fill price, days held, exit price)}, one piece each — or one placeholder that never fills."""
    out = []
    for k, (te, e, days, x) in fills.items():
        p = piece(strategy, tkn, e, x, days=days, bars=[e] * days, t0=te, rank=rank)
        p.update(oid=oid, ord_k=k, ord_place=list(place), ord_lim=lim, ord_end=end)
        out.append(p)
    if not out:
        out.append({"strategy": strategy, "tkn": tkn, "ens": end, "xns": end + DAY, "e": lim, "x": lim, "hs_in": 1.0,
                    "hs_out": 1.0, "dv20": 1e9, "pt": [], "pc": [], "rank": rank, "oid": oid, "ord_k": -1,
                    "ord_place": list(place), "ord_lim": lim, "ord_end": end})
    return out


def test_a_limit_order_fills_once_at_its_first_touch_after_placement_with_the_quantity_sized_then():
    d = A.Strategy("D", 1, slices=10, max_open=10, orders="reserve")
    P = order("D", "DIP", "o1", 95.0, place=[T0, T0 + DAY, T0 + 2 * DAY], end=T0 + 3 * DAY,
              fills={1: (T0 + DAY + H, 95.0, 3, 99.0), 2: (T0 + 2 * DAY + H, 95.0, 3, 99.0)})
    r = A.simulate(P, [d], A.Rules(), settle=settle)
    (pos,) = r["positions"]
    assert pos["shares"] == int(10_000 / 10 // 95.0)              # sized at placement, at the limit
    assert pos["entry_ns"] == T0 + DAY + H and pos["placed_ns"] == T0
    assert [o["what"] for o in r["orders"]] == ["placed", "filled"]
    with pytest.raises(ValueError):
        A.Strategy("D", 1, orders="maybe")


def _order_and_a_short(mode):
    """A big dip order (80 shares at $100, $4,000 of Reg T room) placed at T0 and touched 5 h later; between, a short
    whose house margin (2.86 x $2,500) needs more room than the order leaves."""
    d = A.Strategy("D", 1, slices=1.25, max_open=3, priority=1, quality=0.1, preemptible=True, yield_whole=True,
                   orders=mode)
    s = A.Strategy("S", -1, slices=4, priority=0, quality=1.0, min_equity=2000.0, house_maint=2.0, house_init=2.86)
    P = order("D", "DIP", "o1", 100.0, place=[T0, T0 + DAY], end=T0 + 2 * DAY,
              fills={0: (T0 + 5 * H, 100.0, 3, 101.0), 1: (T0 + DAY + H, 100.0, 3, 101.0)})
    P.append(piece("S", "HOT", 10.0, 10.0, days=4, bars=[10.0] * 4, t0=T0 + 2 * H))
    return A.simulate(P, [d, s], A.Rules(trigger="close"), settle=settle)


def test_a_reserved_order_holds_its_room_and_gives_way_to_a_short_before_any_position():
    r = _order_and_a_short("reserve")
    what = [o["what"] for o in r["orders"]]
    assert what[:2] == ["placed", "cancel_giveway"]                # the short took the reserved room
    sh = next(p for p in r["positions"] if p["strategy"] == "S")
    assert sh["shares"] == int(10_000 / 4 // 10.0)                 # its whole slice
    assert r["preemptions"] == []                                  # no position had to be sold
    assert what[2:] == ["placed", "filled"]                        # placed again next session, in the room left
    dp = next(p for p in r["positions"] if p["strategy"] == "D")
    assert dp["entry_ns"] == T0 + DAY + H and 0 < dp["shares"] < 80


def test_an_unreserved_order_fills_only_into_the_room_the_shorts_left():
    """PREREG61 amendment 1: an order that reserved nothing buys at its touch only what the room still allows."""
    r = _order_and_a_short("fill")
    log = r["orders"]
    assert [o["what"] for o in log] == ["placed", "filled"]
    sh = next(p for p in r["positions"] if p["strategy"] == "S")
    dp = next(p for p in r["positions"] if p["strategy"] == "D")
    assert sh["shares"] == 250                                     # nothing was reserved: the short took its slice
    assert dp["entry_ns"] == T0 + 5 * H and log[1]["n_order"] == 80
    room = log[1]["room"]                                          # what the short left: ~$2,840 of Reg T room
    assert 2000 < room < 3000
    assert dp["shares"] == int((room - A.comm(dp["shares"], 100.0) - dp["shares"] * 100.0 * 1e-4) // 50.0)
    assert dp["shares"] < 80                                       # not the 80 shares the order promised


def test_an_order_that_finds_no_room_buys_nothing_and_keeps_working():
    d = A.Strategy("D", 1, slices=1.25, max_open=3, priority=1, orders="fill")
    s = A.Strategy("S", -1, slices=1.4, priority=0, min_equity=2000.0, house_maint=2.0, house_init=2.86)
    P = order("D", "DIP", "o1", 100.0, place=[T0, T0 + DAY], end=T0 + 2 * DAY,
              fills={0: (T0 + 5 * H, 100.0, 3, 101.0), 1: (T0 + DAY + 5 * H, 100.0, 3, 101.0)})
    P.append(piece("S", "HOT", 10.0, 10.0, days=1, bars=[10.0], t0=T0 + 2 * H))    # takes the whole room for a day
    r = A.simulate(P, [d, s], A.Rules(trigger="close"), settle=settle)
    assert [o["what"] for o in r["orders"]] == ["placed", "unfunded", "filled"]
    dp = next(p for p in r["positions"] if p["strategy"] == "D")
    assert dp["entry_ns"] == T0 + DAY + 5 * H and dp["shares"] == 80   # the short gone, the whole order fills


def test_working_orders_hold_their_slots_and_a_lapsed_order_frees_its_slot():
    d = A.Strategy("D", 1, slices=10, max_open=1, orders="reserve")
    a = order("D", "AAA", "a", 50.0, place=[T0], end=T0 + 4 * H, fills={}, rank=1.0)     # placed, never touched
    b = order("D", "BBB", "b", 40.0, place=[T0, T0 + DAY], end=T0 + 2 * DAY, rank=2.0,
              fills={1: (T0 + DAY + H, 40.0, 2, 41.0)})
    r = A.simulate(a + b, [d], A.Rules(), settle=settle)
    assert [(o["oid"], o["what"]) for o in r["orders"]] == [("a", "placed"), ("b", "no_slot"), ("a", "lapsed"),
                                                            ("b", "placed"), ("b", "filled")]


def test_deleveraging_cancels_the_working_orders_before_selling_and_a_margin_call_cancels_them():
    d = A.Strategy("D", 1, slices=2, priority=0, max_open=2, delever_order=0, orders="reserve")
    sht = A.vol_short("S", slices=4, priority=1)
    P = order("D", "SPY", "o1", 100.0, place=[T0], end=T0 + 6 * DAY, fills={0: (T0 + 1, 100.0, 6, 100.0)}, rank=1.0)
    P += order("D", "QQQ", "o2", 100.0, place=[T0], end=T0 + 6 * DAY, fills={}, rank=2.0)   # works, never touched
    P.append(piece("S", "SQZ", 10.0, 18.0, days=4, bars=[13.0, 15.0, 18.0, 18.0], t0=T0 + H))
    on = A.simulate(P, [d, sht], A.Rules(trigger="close", delever=(0.8, 0.6)), settle=settle)
    assert ("o2", "cancel_delever") in [(o["oid"], o["what"]) for o in on["orders"]]
    assert [x["tkn"] for x in on["delevers"]] == ["SPY"]
    calls = A.simulate(P, [dataclasses_replace(d, delever_order=None, orders="fill"), A.vol_short("S", slices=1.5)],
                       A.Rules(trigger="close"), settle=settle)
    assert calls["calls"] and ("o2", "cancel_margin") in [(o["oid"], o["what"]) for o in calls["orders"]]


def test_a_limit_fill_comes_before_the_other_entries_of_its_instant():
    """Production judges the dip book's touches at the START of the tick, before the shorts' entries: at one instant
    the order's fill takes the room first (whatever the strategies' priorities)."""
    d = A.Strategy("D", 1, slices=1.25, max_open=3, priority=1, orders="fill")
    s = A.Strategy("S", -1, slices=1.4, priority=0, min_equity=2000.0, house_maint=2.0, house_init=2.86)
    P = order("D", "DIP", "o1", 100.0, place=[T0], end=T0 + DAY, fills={0: (T0 + 5 * H, 100.0, 3, 101.0)})
    P.append(piece("S", "HOT", 10.0, 10.0, days=1, bars=[10.0], t0=T0 + 5 * H))
    r = A.simulate(P, [d, s], A.Rules(trigger="close"), settle=settle)
    dp = next(p for p in r["positions"] if p["strategy"] == "D")
    sh = next(p for p in r["positions"] if p["strategy"] == "S")
    assert dp["shares"] == 80                                      # the fill first: the whole order
    assert sh["shares"] < int(10_000 / 1.4 // 10.0)                # the short got what was left


def test_one_book_per_name_keeps_the_books_off_each_others_names():
    """`Rules(one_book_per_name)`: a dip order is neither placed nor filled on a name the short book holds, and a
    short never enters a name the dip book holds; two strategies of one book may share a name."""
    dip_o = A.Strategy("D", 1, slices=10, max_open=10, priority=1, orders="fill", book="dip")
    dip_p = A.Strategy("DP", 1, slices=10, max_open=10, priority=1, book="dip")
    s1 = A.Strategy("S1", -1, slices=24, priority=0, min_equity=2000.0, house_maint=2.0, house_init=2.86, book="short")
    s2 = A.Strategy("S2", -1, slices=24, priority=0, min_equity=2000.0, house_maint=2.0, house_init=2.86, book="short")
    P = [piece("S1", "AAA", 50.0, 50.0, days=3, bars=[50.0] * 3),
         piece("S2", "AAA", 50.0, 50.0, days=2, bars=[50.0] * 2, t0=T0 + H)]          # same book: allowed
    P += order("D", "AAA", "o1", 49.0, place=[T0 + 2 * H], end=T0 + DAY, fills={0: (T0 + 3 * H, 49.0, 2, 50.0)})
    P += [piece("DP", "BBB", 20.0, 20.0, days=3, bars=[20.0] * 3),
          piece("S1", "BBB", 20.0, 20.0, days=2, bars=[20.0] * 2, t0=T0 + H)]         # the dip book holds BBB
    on = A.simulate(P, [dip_o, dip_p, s1, s2], A.Rules(one_book_per_name=True), settle=settle)
    held = {(p["strategy"], p["tkn"]) for p in on["positions"]}
    assert ("S2", "AAA") in held and ("D", "AAA") not in held and ("S1", "BBB") not in held
    assert ("o1", "held") in [(o["oid"], o["what"]) for o in on["orders"]]
    off = A.simulate(P, [dip_o, dip_p, s1, s2], A.Rules(), settle=settle)
    held0 = {(p["strategy"], p["tkn"]) for p in off["positions"]}
    assert ("D", "AAA") in held0 and ("S1", "BBB") in held0      # the default lets them share the name


def test_ledger_costs_charge_the_pieces_all_in_cost_and_marks_net_of_the_exit_cost():
    """PREREG62: `Rules(fees="ledger")` charges each leg its piece's all-in one-way cost (hs_in / hs_out, commission
    included: no separate commission, SEC fee or TAF), and books the P&L as the live account does — shares x entry x
    the ledger's net return (`spread._pct_return`, priced on the cost-adjusted entry; PREREG63); `mark_net` marks an
    open position net of its exit cost, as the ledger marks its open trades."""
    from src.performance.spread import _pct_return
    s = A.Strategy("L", 1, slices=10)
    p = piece("L", "AAA", 100.0, 110.0, hs=30.0)                  # 30 bp each way, all in
    r = A.simulate([p], [s], A.Rules(fees="ledger"), settle=settle)
    pos = r["positions"][0]
    n = pos["shares"]
    ret = _pct_return("BUY", 100.0, 110.0, entry_cost=30e-4, exit_cost=30e-4) / 100.0
    assert pos["pnl"] == pytest.approx(n * 100.0 * ret)
    assert pos["pnl"] == pytest.approx((n * 10.0 - n * 100.0 * 30e-4 - n * 110.0 * 30e-4) / (1 + 30e-4))
    eng = A.simulate([p], [s], A.Rules(), settle=settle)["positions"][0]
    assert eng["pnl"] < pos["pnl"] + n * 100 * 30e-4                # the engine adds commission + sale fees
    # marks: an open position's equity net of the exit cost
    q = piece("L", "BBB", 100.0, 100.0, days=3, bars=[100.0] * 3, hs=30.0)
    on = A.simulate([q], [s], A.Rules(fees="ledger", mark_net=True), settle=settle, keep_curve=True)
    off = A.simulate([q], [s], A.Rules(fees="ledger"), settle=settle, keep_curve=True)
    k = on["positions"][0]["shares"]
    assert on["curve_v"][1] == pytest.approx(off["curve_v"][1] - k * 100.0 * 30e-4)


# ── PREREG63: the bugs the 2026-10-09 review confirmed ───────────────────────

def test_a_piece_entered_and_exited_at_the_same_instant_closes():
    """A passive fill covered within its bar (or a window cut at the entry bar) has xns == ens: at one instant the
    engine takes exits before entries, so it opened after its own exit and stayed open to the end."""
    p = piece("S", "AAA", 10.0, 9.0, days=0, bars=[])
    p["xns"] = p["ens"]
    r = A.simulate([p], [A.vol_short("S")], A.Rules(), settle=settle)
    pos = r["positions"][0]
    assert pos["how"] == "exit" and pos["exit_px"] == 9.0 and pos["pnl"] > 0


def test_trade_statistics_count_wins_on_gross_prices():
    """Per strategy the win rate is GROSS — the right direction on raw prices (the house standard) — not P&L > 0
    after costs (``net_win_rate``)."""
    s = A.Strategy("L", 1, slices=10)
    up_but_lost = piece("L", "AAA", 100.0, 100.05, days=2, hs=40.0)     # up 0.05%, costs 0.8%: a net loss
    r = A.simulate([up_but_lost], [s], A.Rules(), settle=settle)
    m = A.metrics(r, "2024-01-08", "2024-03-31")["by_strategy"]["L"]
    assert (m["positions"], m["closed"], m["open_at_end"]) == (1, 1, 0)
    assert m["win_rate"] == 1.0 and m["net_win_rate"] == 0.0


def test_a_ruin_keeps_the_exposure_curves_aligned():
    s = A.Strategy("S", -1, slices=1, gross_cap=None) if False else A.Strategy("S", -1, slices=1)
    p = piece("S", "AAA", 10.0, 10.0, days=3, bars=[10.0, 60.0, 60.0])
    r = A.simulate([p], [s], A.Rules(trigger="close", gross_cap={1: 1.0, -1: 1.0}), settle=settle, keep_exposure=True)
    assert r["ruined"] is not None
    assert all(len(v) == len(r["curve_t"]) for v in r["curve_by"].values())


def test_room_is_not_made_for_an_entry_the_lendability_check_refuses():
    lo = A.Strategy("L", 1, slices=1, preemptible=True, quality=0.1)
    hi_ = A.Strategy("S", -1, slices=1, quality=1.0, avail_mult=1.0, house_maint=2.0, house_init=2.0)
    long_ = piece("L", "AAA", 100.0, 100.0, days=5)                      # half the room: the short needs a sale
    short = piece("S", "BBB", 50.0, 50.0, days=2, t0=T0 + DAY, avail=1.0)      # 1 share lendable: refused
    r = A.simulate([long_, short], [lo, hi_], A.Rules(), settle=settle)
    assert [p["tkn"] for p in r["positions"]] == ["AAA"] and r["positions"][0]["how"] == "exit"
    assert not r.get("preemptions") and "sold" not in r["positions"][0]


def test_reentry_and_refill_are_refused_on_a_short_strategy():
    with pytest.raises(ValueError):
        A.Strategy("S", -1, reenter=True)
    with pytest.raises(ValueError):
        A.Strategy("S", -1, refill_min=0.1)


def test_rules_are_validated():
    for bad in ({"fees": "Ledger"}, {"trigger": "intrabar"}, {"delever": (0.6, 0.8)}, {"gross_cap": {0: 1.0}}):
        with pytest.raises(ValueError):
            A.Rules(**bad)


def test_a_short_sized_under_ledger_rules_keeps_its_sale_fees_out_of_the_room():
    """Production's `sim_account.size` takes the commission, the SEC fee and the TAF out of the margin room."""
    c_comm = A.comm(100, 50.0)
    c_all = c_comm + A.SEC_FEE * 100 * 50.0 + min(A.TAF_PER_SHARE * 100, A.TAF_MAX)
    start = 100 * 100.0 + (c_comm + c_all) / 2          # 100 shares fit after the commission, not after every fee
    s = A.Strategy("S", -1, slices=1, house_maint=2.0, house_init=2.0)
    p = piece("S", "AAA", 50.0, 50.0, days=2, hs=0.0)
    r = A.simulate([p], [s], A.Rules(fees="ledger", start=start), settle=settle)
    assert r["positions"][0]["shares"] == 99


def test_settlement_skips_the_bank_holidays_on_which_stocks_trade():
    """Columbus Day 2025 (Oct 13) is a session but no settlement day: a T+1 trade on Friday Oct 10 settles Tuesday."""
    sessions = [A.day_numbers([pd.Timestamp(d, tz="America/New_York").value])[0]
                for d in pd.bdate_range("2025-10-06", "2025-10-17")]
    st = A.make_settle(A.settlement_days(sessions))
    fri = A.day_numbers([pd.Timestamp("2025-10-10 12:00", tz="America/New_York").value])[0]
    tue = A.day_numbers([pd.Timestamp("2025-10-14 12:00", tz="America/New_York").value])[0]
    assert st(fri) == tue


def test_one_book_per_name_compares_the_traded_symbol():
    dip = A.Strategy("D", 1, slices=10, book="dip_long")
    vol = dataclasses.replace(A.vol_short("V"), book="sel_short")
    long_ = piece("D", "XYZ", 100.0, 100.0, days=5)
    short = piece("V", "XYZ@X", 100.0, 90.0, days=2, t0=T0 + DAY)
    r = A.simulate([long_, short], [dip, vol], A.Rules(one_book_per_name=True), settle=settle)
    assert [p["tkn"] for p in r["positions"]] == ["XYZ"]


def test_an_order_books_capital_summary_reads_the_order_log():
    """An order sized at placement and cut at its fill (another book took the side's room in between) is reported as
    cut from the order log — the touch pieces used to be counted as entries and the cut never seen."""
    s = A.Strategy("L", 1, slices=1, orders="fill", max_open=10)
    y = A.Strategy("Y", 1, slices=1.25)
    o = piece("L", "AAA", 50.0, 55.0, days=2, t0=T0 + DAY)
    o.update(oid="AAA|1", ord_place=[T0 - 10], ord_lim=50.0, ord_end=T0 + 3 * DAY, ord_k=0)
    other = piece("Y", "YYY", 100.0, 100.0, days=5)                  # 8,000 of the 10,000 side cap, after the placement
    r = A.simulate([other, o], [y, s], A.Rules(), settle=settle, keep_skips=True)
    cs = A.capital_summary(r, [other, o])["L"]
    assert (cs["orders"], cs["placed"], cs["filled"], cs["cut"]) == (1, 1, 1, 1) and cs["usd_not_deployed"] > 0


def test_a_piece_may_carry_its_own_slices_and_without_them_nothing_changes():
    """PREREG80 (a sizing re-tuned week by week): a piece's own ``slices`` sizes its entry instead of its strategy's;
    a piece carrying its strategy's count, or none, sizes exactly as before."""
    st = [A.Strategy("L", 1, slices=10, max_open=10)]
    ps = [piece("L", f"T{k}", 100.0, 101.0, rank=k, t0=T0 + k * DAY) for k in range(4)]
    base = A.simulate(ps, st, A.Rules(), settle=settle)
    same = A.simulate([dict(p, slices=10) for p in ps], st, A.Rules(), settle=settle)
    assert [p["shares"] for p in same["positions"]] == [p["shares"] for p in base["positions"]]
    assert same["final"] == base["final"]
    half = A.simulate([dict(p, slices=5) if p["tkn"] == "T0" else p for p in ps], st, A.Rules(), settle=settle)
    sh = {p["tkn"]: p["shares"] for p in half["positions"]}
    bs = {p["tkn"]: p["shares"] for p in base["positions"]}
    assert sh["T0"] == pytest.approx(2 * bs["T0"], abs=1)              # 1/5 of the account instead of 1/10
    assert sh["T1"] == pytest.approx(bs["T1"], rel=0.02)               # the next entries keep their strategy's 1/10


# ── order fills make room (PREREG86's trend sleeve) ─────────────────────────

def _sleeve_and_a_dip_order(**rules):
    """A sleeve (quality 0, preemptible, slices 1) buys the whole long side at T0; a dip order (quality 0.13, "fill")
    is placed an hour later and touched four hours after that."""
    sl = A.Strategy("SL", 1, slices=1.0, max_open=1, priority=9, quality=0.0, preemptible=True)
    d = A.Strategy("D", 1, slices=10, max_open=10, priority=1, quality=0.13, preemptible=True, orders="fill")
    P = [piece("SL", "IDX", 100.0, 100.0, days=5, bars=[100.0] * 5, t0=T0)]
    P += order("D", "DIP", "o1", 50.0, place=[T0 + H], end=T0 + DAY, fills={0: (T0 + 5 * H, 50.0, 3, 51.0)})
    return A.simulate(P, [sl, d], A.Rules(trigger="close", **rules), settle=settle)


def test_without_the_flag_a_sleeve_holding_the_long_side_leaves_the_dip_order_no_room():
    r = _sleeve_and_a_dip_order()
    assert [o["what"] for o in r["orders"]] == ["no_room"]
    assert r["preemptions"] == []


def test_with_the_flag_an_order_is_placed_through_the_sleeve_and_its_fill_sells_just_enough_of_it():
    r = _sleeve_and_a_dip_order(fills_make_room=True)
    log = r["orders"]
    assert [o["what"] for o in log] == ["placed", "filled"]
    dp = next(p for p in r["positions"] if p["strategy"] == "D")
    assert dp["shares"] == log[0]["n"] == log[1]["n_order"]           # the whole order: a tenth of the equity
    assert log[0]["n"] == int(9_995 / 10 // 50.0)                        # (the sleeve's entry costs: 19 shares)
    assert r["preemptions"] and {x["strategy"] for x in r["preemptions"]} == {"SL"}
    sold = sum(x["shares"] for x in r["preemptions"])
    assert 0 < sold < 100                                              # just enough of the sleeve, not all of it
    assert all(x["t"] == T0 + 5 * H for x in r["preemptions"])         # sold at the fill, not at the placement


def _caps_and_a_dip_order(**rules):
    """Three capitulation-like lots (quality 0.1, preemptible whole, 25 shares at $100 each) and a plain long Y fill the
    long side at T0; a dip order (quality 0.13, 1/2, "fill") placed an hour later is sized as if the lots were freed
    (100 at $50); Y's mark falls to $80 before the order is touched, so the order's quantity is above its slice at
    the fill (the equity $9,500: 94 shares)."""
    c = A.Strategy("C", 1, slices=4.0, max_open=3, priority=7, quality=0.1, preemptible=True, yield_whole=True)
    y = A.Strategy("Y", 1, slices=4.0, max_open=1, priority=0)
    d = A.Strategy("D", 1, slices=2.0, max_open=2, priority=6, quality=0.13, preemptible=True, orders="fill")
    P = [piece("Y", "YYY", 100.0, 80.0, days=3, bars=[80.0] * 3, t0=T0)]
    P += [piece("C", f"C{k}", 100.0, 100.0, days=3, bars=[100.0] * 3, t0=T0 + k * 60 * 10**9) for k in (1, 2, 3)]
    P += order("D", "DIP", "o1", 50.0, place=[T0 + H], end=T0 + 2 * DAY, fills={0: (T0 + DAY + H, 50.0, 3, 51.0)})
    return A.simulate(P, [y, c, d], A.Rules(trigger="close", fills_make_room=True, **rules), settle=settle)


def test_the_fills_own_size_decides_the_give_way_with_fill_room_exact():
    """`Rules.fill_room_exact` (production's `live_account.fill_give_way_plan`, 2026-10-20): the fill sells the lots,
    the latest first, only until its own size (the room and its side's cap, no slice) reaches the order's quantity;
    without it the slice-capped size keeps selling the earliest lot too, for no share."""
    old = _caps_and_a_dip_order()
    new = _caps_and_a_dip_order(fill_room_exact=True)
    assert sorted(x["tkn"] for x in old["preemptions"]) == ["C1", "C2", "C3"]
    assert sorted(x["tkn"] for x in new["preemptions"]) == ["C2", "C3"]                # the earliest lot kept
    for r in (old, new):
        dp = next(p for p in r["positions"] if p["strategy"] == "D")
        placed = next(o for o in r["orders"] if o["what"] == "placed")
        assert dp["shares"] == placed["n"] == 99                       # the whole order either way (99: the costs)


def test_the_flag_changes_nothing_without_a_lower_quality_preemptible_position():
    d = A.Strategy("D", 1, slices=1.25, max_open=3, priority=1, orders="fill", quality=0.13, preemptible=True)
    s = A.Strategy("S", -1, slices=1.4, priority=0, min_equity=2000.0, house_maint=2.0, house_init=2.86)
    P = order("D", "DIP", "o1", 100.0, place=[T0, T0 + DAY], end=T0 + 2 * DAY,
              fills={0: (T0 + 5 * H, 100.0, 3, 101.0), 1: (T0 + DAY + 5 * H, 100.0, 3, 101.0)})
    P.append(piece("S", "HOT", 10.0, 10.0, days=1, bars=[10.0], t0=T0 + 2 * H))
    a = A.simulate(P, [d, s], A.Rules(trigger="close"), settle=settle)
    b = A.simulate(P, [d, s], A.Rules(trigger="close", fills_make_room=True), settle=settle)
    assert a["final"] == b["final"] and a["orders"] == b["orders"]
