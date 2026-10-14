"""The account engine's GROUP CAP (src/backtest/account.py, `Strategy.group_cap` / `group_scope`; PREREG100's
per-sector cap): at most K open positions plus working orders in one group among the strategies of the scope — a
market entry at the cap is skipped, a limit order waits for its next placement instant (the path of an order without a
slot), a stack counts as one more position, a piece without a group is never capped. Without a cap nothing changes."""
from __future__ import annotations

import dataclasses

import numpy as np
import pandas as pd
import pytest

from src.backtest import account as A

DAY = 86400 * 10**9
H = 3600 * 10**9
T0 = pd.Timestamp("2024-01-08 15:00", tz="UTC").value       # a Monday, 10:00 ET
ET = "America/New_York"


def settle(dn):
    return int(dn) + 1


def piece(strategy, tkn, e, x, days=2, bars=None, hs=1.0, dv20=1e9, t0=T0, **kw):
    bars = list(bars) if bars is not None else list(np.linspace(e, x, days + 1))[1:]
    p = {"strategy": strategy, "tkn": tkn, "ens": t0, "xns": t0 + days * DAY, "e": e, "x": x, "hs_in": hs,
         "hs_out": hs, "dv20": dv20, "pt": [t0 + (k + 1) * DAY for k in range(len(bars))], "pc": bars}
    p.update(kw)
    return p


def order(strategy, tkn, oid, lim, place, end, fills, rank=0.0, **kw):
    """One limit order (the engine's convention): placed at ``place``, lapsing at ``end``; ``fills`` = {window session
    k: (touch instant, fill price, days held, exit price)}."""
    out = []
    for k, (te, e, days, x) in fills.items():
        p = piece(strategy, tkn, e, x, days=days, bars=[e] * days, t0=te, rank=rank, **kw)
        p.update(oid=oid, ord_k=k, ord_place=list(place), ord_lim=lim, ord_end=end)
        out.append(p)
    return out


def short(name="S", **kw):
    return A.Strategy(name, -1, slices=24, min_equity=2000.0, house_maint=2.0, house_init=2.86, **kw)


def flat_short(strategy, tkn, t0, days=5, **kw):
    return piece(strategy, tkn, 20.0, 20.0, days=days, bars=[20.0] * days, t0=t0, **kw)


# ── market entries ───────────────────────────────────────────────────────────

def test_market_entries_beyond_the_cap_are_skipped_and_recorded():
    P = [flat_short("S", f"T{k}", T0 + k * H, group="BusEq") for k in range(4)]
    P.append(flat_short("S", "OTH", T0 + 4 * H, group="Money"))
    P.append(flat_short("S", "UNK", T0 + 5 * H))                                  # no group
    P.append(flat_short("S", "LATE", T0 + 6 * DAY, group="BusEq"))                # after the first two closed
    r = A.simulate(P, [short(group_cap=2)], A.Rules(), settle=settle, keep_skips=True)
    assert sorted(p["tkn"] for p in r["positions"]) == ["LATE", "OTH", "T0", "T1", "UNK"]
    sk = [x for x in r["skips"] if x["reason"] == "group_cap"]
    assert [(x["tkn"], x["group"], x["got"]) for x in sk] == [("T2", "BusEq", 0), ("T3", "BusEq", 0)]
    off = A.simulate(P, [short()], A.Rules(), settle=settle, keep_skips=True)      # no cap: all seven
    assert len(off["positions"]) == 7 and not any(x["reason"] == "group_cap" for x in off["skips"])


def test_a_piece_without_a_group_is_never_capped_nor_counted():
    P = [flat_short("S", "A1", T0, group="BusEq"), flat_short("S", "N1", T0 + H),
         flat_short("S", "N2", T0 + 2 * H, group=None), flat_short("S", "N3", T0 + 3 * H, group=float("nan")),
         flat_short("S", "A2", T0 + 4 * H, group="BusEq")]
    r = A.simulate(P, [short(group_cap=0)], A.Rules(), settle=settle)             # a zero cap: no grouped entry
    assert sorted(p["tkn"] for p in r["positions"]) == ["N1", "N2", "N3"]
    r1 = A.simulate(P, [short(group_cap=1)], A.Rules(), settle=settle)            # the ungrouped never count
    assert sorted(p["tkn"] for p in r1["positions"]) == ["A1", "N1", "N2", "N3"]


def test_a_stack_of_a_held_name_counts_as_one_more_position():
    P = [flat_short("S", "AAA", T0, group="BusEq"), flat_short("S", "AAA", T0 + H, group="BusEq"),
         flat_short("S", "BBB", T0 + 2 * H, group="BusEq")]
    r = A.simulate(P, [short(group_cap=2)], A.Rules(), settle=settle, keep_skips=True)
    assert [p["tkn"] for p in r["positions"]] == ["AAA", "AAA"]
    assert [(x["tkn"], x["reason"]) for x in r["skips"] if x["reason"] == "group_cap"] == [("BBB", "group_cap")]


def test_the_scope_counts_the_named_strategies_and_an_uncapped_strategy_is_never_capped():
    """The vol and fall arms share one cap (scope V + F); the ETF-like arm E is neither capped nor counted."""
    P = [flat_short("V", "AAA", T0, group="BusEq"), flat_short("E", "FND", T0 + H, group="BusEq"),
         flat_short("E", "FND2", T0 + 2 * H, group="BusEq"), flat_short("F", "BBB", T0 + 3 * H, group="BusEq"),
         flat_short("F", "CCC", T0 + 4 * H, group="Money"), flat_short("V", "DDD", T0 + 5 * H, group="BusEq")]
    shared = [short("V", group_cap=1, group_scope=("V", "F")), short("F", group_cap=1, group_scope=["V", "F"]),
              short("E")]
    r = A.simulate(P, shared, A.Rules(), settle=settle, keep_skips=True)
    held = sorted((p["strategy"], p["tkn"]) for p in r["positions"])
    assert held == [("E", "FND"), ("E", "FND2"), ("F", "CCC"), ("V", "AAA")]
    assert sorted(x["tkn"] for x in r["skips"] if x["reason"] == "group_cap") == ["BBB", "DDD"]
    own = [short("V", group_cap=1), short("F", group_cap=1), short("E")]            # each counts only itself
    r2 = A.simulate(P, own, A.Rules(), settle=settle)
    held2 = sorted((p["strategy"], p["tkn"]) for p in r2["positions"])
    assert ("F", "BBB") in held2 and ("V", "DDD") not in held2
    assert shared[1].group_scope == ("V", "F")                                     # a list is kept as a tuple


# ── limit orders ─────────────────────────────────────────────────────────────

def _two_orders(cap):
    """Order ``a`` (placed at T0, filled an hour later, held a day) and order ``b`` in the same group, placeable at
    each of three sessions' openings and touched two hours after each."""
    d = A.Strategy("D", 1, slices=10, max_open=10, orders="fill", group_cap=cap)
    a = order("D", "AAA", "a", 100.0, place=[T0], end=T0 + 8 * H, fills={0: (T0 + H, 100.0, 1, 101.0)}, rank=1.0,
              group="BusEq")
    b = order("D", "BBB", "b", 90.0, place=[T0, T0 + DAY, T0 + 2 * DAY], end=T0 + 2 * DAY + 8 * H, rank=2.0,
              group="BusEq", fills={0: (T0 + 2 * H, 90.0, 2, 95.0), 1: (T0 + DAY + 2 * H, 90.0, 2, 95.0),
                                    2: (T0 + 2 * DAY + 2 * H, 90.0, 2, 95.0)})
    return A.simulate(a + b, [d], A.Rules(), settle=settle, keep_skips=True), a + b


def test_an_order_at_the_cap_is_not_placed_and_is_placed_at_a_later_session_once_a_position_closes():
    r, P = _two_orders(1)
    log = [(o["oid"], o["what"]) for o in r["orders"]]
    # T0: a is placed (working: it counts) and b finds its group full; the next opening a is held; the one after,
    # a has been sold and b is placed — and fills at that session's touch
    assert log == [("a", "placed"), ("b", "group_cap"), ("a", "filled"), ("b", "group_cap"), ("b", "placed"),
                   ("b", "filled")]
    caps = [o for o in r["orders"] if o["what"] == "group_cap"]
    assert [o["t"] for o in caps] == [T0, T0 + DAY] and all(o["group"] == "BusEq" for o in caps)
    b = next(p for p in r["positions"] if p["tkn"] == "BBB")
    assert b["entry_ns"] == T0 + 2 * DAY + 2 * H and b["placed_ns"] == T0 + 2 * DAY
    cs = A.capital_summary(r, P)["D"]
    assert cs["filled"] == 2 and "group_cap" not in cs["ended_by"]
    off, _ = _two_orders(None)
    assert [(o["oid"], o["what"]) for o in off["orders"]] == [("a", "placed"), ("b", "placed"), ("a", "filled"),
                                                             ("b", "filled")]
    assert next(p for p in off["positions"] if p["tkn"] == "BBB")["entry_ns"] == T0 + 2 * H


def test_an_order_kept_out_by_the_cap_for_its_whole_life_is_reported_as_not_placed():
    d = A.Strategy("D", 1, slices=10, max_open=10, orders="fill", group_cap=1)
    a = order("D", "AAA", "a", 100.0, place=[T0], end=T0 + 8 * H, fills={0: (T0 + H, 100.0, 5, 101.0)}, rank=1.0,
              group="BusEq")
    b = order("D", "BBB", "b", 90.0, place=[T0, T0 + DAY], end=T0 + DAY + 8 * H, rank=2.0, group="BusEq",
              fills={1: (T0 + DAY + 2 * H, 90.0, 2, 95.0)})
    r = A.simulate(a + b, [d], A.Rules(), settle=settle)
    assert [(o["oid"], o["what"]) for o in r["orders"]] == [("a", "placed"), ("b", "group_cap"), ("a", "filled"),
                                                           ("b", "group_cap")]
    cs = A.capital_summary(r, a + b)["D"]
    assert cs["not_placed_by"] == {"group_cap": 2} and cs["ended_by"] == {}


def test_a_working_order_counts_toward_a_market_entry_of_its_scope():
    d = A.Strategy("D", 1, slices=10, max_open=10, orders="fill", group_cap=1, group_scope=("D", "M"))
    m = A.Strategy("M", 1, slices=10, max_open=10, group_cap=1, group_scope=("D", "M"))
    P = [{"strategy": "D", "tkn": "AAA", "ens": T0 + 8 * H, "xns": T0 + 8 * H + DAY, "e": 100.0, "x": 100.0,
          "hs_in": 1.0, "hs_out": 1.0, "dv20": 1e9, "pt": [], "pc": [], "oid": "a", "ord_k": -1,
          "ord_place": [T0], "ord_lim": 100.0, "ord_end": T0 + 8 * H, "group": "BusEq"}]    # placed, never touched
    P.append(piece("M", "MMM", 50.0, 50.0, days=1, bars=[50.0], t0=T0 + H, group="BusEq"))  # while a works
    P.append(piece("M", "NNN", 50.0, 50.0, days=1, bars=[50.0], t0=T0 + 9 * H, group="BusEq"))  # after it lapsed
    r = A.simulate(P, [d, m], A.Rules(), settle=settle, keep_skips=True)
    assert [p["tkn"] for p in r["positions"]] == ["NNN"]
    assert [x["tkn"] for x in r["skips"] if x["reason"] == "group_cap"] == ["MMM"]


# ── re-entry ─────────────────────────────────────────────────────────────────

def _session_piece(strategy, tkn, px, d0, d1, **kw):
    """Bought at the 09:30 open of ``d0``, sold at the 09:30 open of ``d1``, flat at ``px``, marked at each later
    session's 09:30 open and every session's 16:00 close."""
    days = pd.bdate_range(d0, d1)
    pt, pc = [], []
    for k, d in enumerate(days[:-1]):
        if k > 0:
            pt.append(pd.Timestamp(f"{d.date()} 09:30", tz=ET).value)
            pc.append(px)
        pt.append(pd.Timestamp(f"{d.date()} 16:00", tz=ET).value)
        pc.append(px)
    p = {"strategy": strategy, "tkn": tkn, "ens": pd.Timestamp(f"{days[0].date()} 09:30", tz=ET).value,
         "xns": pd.Timestamp(f"{days[-1].date()} 09:30", tz=ET).value, "e": px, "x": px, "hs_in": 1.0, "hs_out": 1.0,
         "dv20": 1e10, "pt": pt, "pc": pc}
    p.update(kw)
    return p


def test_a_reentry_waits_while_its_group_is_at_the_cap():
    """The fund is sold to make room for the dip (03-05) and re-buys at the next open with room (03-06) — unless a
    position of its scope holds its group then: it waits until that one is sold (03-07's open)."""
    def run(cap):
        fund = A.Strategy("F", 1, slices=1, max_open=1, priority=3, quality=0.06, preemptible=True, yield_whole=True,
                          reenter=True, group_cap=cap, group_scope=("F", "G"))
        dip = A.Strategy("D", 1, slices=10, priority=2, quality=0.13)
        g = short("G", priority=4)
        P = [_session_piece("F", "FUND", 100.0, "2024-03-04", "2024-03-11", group="X"),
             _session_piece("D", "DIP", 50.0, "2024-03-05", "2024-03-07"),
             dict(_session_piece("G", "GGG", 20.0, "2024-03-05", "2024-03-07", group="X"),
                  ens=pd.Timestamp("2024-03-05 10:00", tz=ET).value)]
        return A.simulate(P, [fund, dip, g], A.Rules(), settle=settle)
    free, capped = run(None), run(1)
    assert [x["t"] for x in free["reentries"]] == [pd.Timestamp("2024-03-06 09:30", tz=ET).value]
    assert [x["t"] for x in capped["reentries"]] == [pd.Timestamp("2024-03-07 09:30", tz=ET).value]


# ── defaults ─────────────────────────────────────────────────────────────────

def _same(a, b, path="out"):
    """Exact equality of two `simulate` results (arrays element by element, NaN equal to NaN)."""
    if isinstance(a, dict):
        assert set(a) == set(b), f"{path}: {sorted(set(a) ^ set(b))}"
        for k in a:
            _same(a[k], b[k], f"{path}.{k}")
    elif isinstance(a, (list, tuple)):
        assert len(a) == len(b), f"{path}: {len(a)} vs {len(b)}"
        for k, (x, y) in enumerate(zip(a, b)):
            _same(x, y, f"{path}[{k}]")
    elif isinstance(a, np.ndarray):
        assert np.array_equal(a, np.asarray(b), equal_nan=a.dtype.kind == "f"), path
    elif isinstance(a, float) and a != a:
        assert b != b, path
    else:
        assert a == b, f"{path}: {a!r} vs {b!r}"


def _scenarios():
    """Account scenarios reaching the engine's paths: shorts with borrow and house rates, a margin call, limit orders
    (fill and reserve) with give-way, de-leveraging, one book per name and the ledger's costs."""
    out = []
    d = A.Strategy("D", 1, slices=1.25, max_open=3, priority=1, quality=0.1, preemptible=True, yield_whole=True,
                   orders="reserve", delever_order=0)
    s = A.Strategy("S", -1, slices=4, priority=0, quality=1.0, min_equity=2000.0, house_maint=2.0, house_init=2.86)
    P = order("D", "DIP", "o1", 100.0, place=[T0, T0 + DAY], end=T0 + 2 * DAY,
              fills={0: (T0 + 5 * H, 100.0, 3, 101.0), 1: (T0 + DAY + H, 100.0, 3, 101.0)}, group="BusEq")
    P += order("D", "DIP2", "o2", 50.0, place=[T0, T0 + DAY], end=T0 + 2 * DAY,
               fills={1: (T0 + DAY + 3 * H, 50.0, 2, 52.0)}, rank=1.0, group="BusEq")
    P.append(piece("S", "HOT", 10.0, 10.0, days=4, bars=[10.0] * 4, t0=T0 + 2 * H, group="Money"))
    P.append(piece("S", "SQZ", 10.0, 18.0, days=4, bars=[13.0, 15.0, 18.0, 18.0], t0=T0 + 3 * H, group="BusEq"))
    for rules in (A.Rules(trigger="close"), A.Rules(trigger="high", delever=(0.8, 0.6)),
                  A.Rules(one_book_per_name=True, fees="ledger", mark_net=True)):
        out.append((P, [d, s], rules))
        out.append((P, [dataclasses.replace(d, orders="fill"), A.vol_short("S", slices=1.5)], rules))
    lng = A.Strategy("L", 1, slices=10, max_open=3)
    Q = [piece("L", f"T{k}", 100.0, 100.0 + k, rank=k, t0=T0 + k * H, days=3, group=("A" if k % 2 else "B"))
         for k in range(6)]
    out.append((Q, [lng], A.Rules()))
    return out


def test_without_a_cap_the_groups_change_nothing_and_a_cap_that_never_binds_changes_nothing():
    for P, strategies, rules in _scenarios():
        bare = [{k: v for k, v in p.items() if k != "group"} for p in P]
        kw = dict(settle=settle, keep_exposure=True, keep_skips=True)
        base = A.simulate([dict(p) for p in bare], strategies, rules, **kw)
        grouped = A.simulate([dict(p) for p in P], strategies, rules, **kw)
        _same(base, grouped)
        loose = [dataclasses.replace(s, group_cap=100, group_scope=tuple(x.name for x in strategies))
                 for s in strategies]
        _same(base, A.simulate([dict(p) for p in P], loose, rules, **kw))
        explicit = [dataclasses.replace(s, group_cap=None, group_scope=None) for s in strategies]
        _same(base, A.simulate([dict(p) for p in P], explicit, rules, **kw))


def test_a_binding_cap_changes_the_scenario():
    P, strategies, rules = _scenarios()[-1]
    base = A.simulate(P, strategies, rules, settle=settle)
    capped = A.simulate(P, [dataclasses.replace(strategies[0], group_cap=1)], rules, settle=settle, keep_skips=True)
    assert len(capped["positions"]) < len(base["positions"])
    assert {x["reason"] for x in capped["skips"]} >= {"group_cap"}


def test_the_cap_is_validated():
    for bad in (-1, 1.5, True):
        with pytest.raises(ValueError):
            A.Strategy("X", 1, group_cap=bad)
    with pytest.raises(ValueError):
        A.Strategy("X", 1, group_cap=2, group_scope="vol")
    assert A.Strategy("X", 1, group_cap=2.0).group_cap == 2.0 and A.Strategy("X", 1).group_cap is None
