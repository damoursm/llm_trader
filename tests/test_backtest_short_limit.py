"""PREREG66 — the dip book's limit entry mirrored for the short arms, in the simulator (src/backtest/strategies/vol.py
`short_limit_candidates`, `window_instants`; the engine's `Strategy(orders="fill")` for a short; `mimic.Mimic.short_limit`).

What must hold: a SELL limit above the pick rests from the pick through the W-th session counting the pick's own and
no later than the deadline; a bar whose HIGH reaches it is a touch (one tick past it with ``through``), one candidate
per window session; the first close at or under the target cancels it — a touch inside that bar still fills; the
engine sizes a short order when placed and fills it at a touch within the room, waiting for a later session's touch
when there is none, and never fills after the order ended."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.backtest import account as A
from src.backtest.mimic import Mimic
from src.backtest.strategies.vol import BIG, short_limit_candidates, window_instants

DAY = 86400 * 10**9
H = 3600 * 10**9
T0 = pd.Timestamp("2024-01-08 15:00", tz="UTC").value       # a Monday, 10:00 ET
CAL = np.arange(19730, 19760)                                # day numbers (synthetic sessions)


def path():
    """Six bars after a pick on session 19730: two more that session, then one per session."""
    pc = np.array([[10.0, 10.4, 10.2, 9.0, 10.0, 10.0]])
    ph = np.array([[10.2, 10.6, 10.3, 9.5, 10.7, 10.8]])
    psd = np.array([[19730, 19730, 19731, 19731, 19732, 19733]])
    return pc, ph, psd


def test_each_window_session_has_its_first_touch():
    pc, ph, psd = path()
    rows, ms, js, jc = short_limit_candidates(pc, ph, psd, np.array([19730]), CAL, np.array([10.5]),
                                              np.array([-np.inf]), np.array([5]), 5)
    assert list(ms) == [0, 2, 3] and list(js) == [1, 4, 5] and jc[0] == BIG
    rows, ms, js, _ = short_limit_candidates(pc, ph, psd, np.array([19730]), CAL, np.array([10.5]),
                                             np.array([-np.inf]), np.array([5]), 3)
    assert list(ms) == [0, 2]                                 # the 4th session is past a 3-session window
    rows, ms, js, _ = short_limit_candidates(pc, ph, psd, np.array([19730]), CAL, np.array([10.5]),
                                             np.array([-np.inf]), np.array([4]), 5)
    assert list(js) == [1, 4]                                 # never past the deadline bar


def test_the_target_cancels_the_order_but_a_touch_in_that_bar_fills():
    pc, ph, psd = path()
    rows, ms, js, jc = short_limit_candidates(pc, ph, psd, np.array([19730]), CAL, np.array([10.5]),
                                              np.array([9.2]), np.array([5]), 5)
    assert jc[0] == 3 and list(js) == [1]                    # bar 3 closed at 9.0 <= 9.2: nothing after it
    rows, ms, js, jc = short_limit_candidates(pc, ph, psd, np.array([19730]), CAL, np.array([9.4]),
                                              np.array([9.2]), np.array([5]), 5)
    assert list(js) == [0, 2] and jc[0] == 3                 # bar 2's high 10.3 reached 9.4: filled, then cancelled


def test_through_needs_a_bar_one_tick_past_the_limit():
    pc, ph, psd = path()
    _, _, js, _ = short_limit_candidates(pc, ph, psd, np.array([19730]), CAL, np.array([10.6]),
                                         np.array([-np.inf]), np.array([5]), 5, through=np.array([0.01]))
    assert list(js) == [4, 5]                                 # 10.6 itself (bar 1) is not enough


def test_the_window_counts_the_picks_own_session():
    place, end = window_instants(19730, T0, CAL, 3)
    assert place[0] == T0 + 1 and len(place) == 3
    et = [pd.Timestamp(t - 1, unit="ns", tz="UTC").tz_convert("America/New_York") for t in place[1:]]
    assert [(x.hour, x.minute) for x in et] == [(9, 30), (9, 30)]
    e = pd.Timestamp(end - 1, unit="ns", tz="UTC").tz_convert("America/New_York")
    assert (e.hour, e.minute) == (16, 0) and (e.normalize().tz_localize(None) - pd.Timestamp("1970-01-01")).days == 19732


def _short_order(fills, end, place):
    out = []
    for k, (te, e, days, x) in fills.items():
        p = {"strategy": "S", "tkn": "HOT", "ens": te, "xns": te + days * DAY, "e": e, "x": x, "hs_in": 0.0,
             "hs_out": 1.0, "dv20": 1e9, "pt": [te + (j + 1) * DAY for j in range(days)], "pc": [e] * days,
             "oid": "o1", "ord_k": k, "ord_place": list(place), "ord_lim": e, "ord_end": end}
        out.append(p)
    return out


def test_a_short_order_is_sized_when_placed_and_fills_at_its_touch_within_the_room():
    s = A.Strategy("S", -1, slices=4, house_maint=2.0, house_init=2.86, min_equity=2000.0, orders="fill")
    P = _short_order({0: (T0 + 3 * H, 10.5, 3, 10.0)}, end=T0 + 4 * DAY, place=[T0])
    r = A.simulate(P, [s], A.Rules(trigger="close"), settle=lambda d: int(d) + 1)
    (pos,) = r["positions"]
    assert pos["side"] == -1 and pos["entry_ns"] == T0 + 3 * H and pos["entry_px"] == 10.5
    assert pos["shares"] == int(10_000 / 4 / 10.5)            # the slice at the limit, sized when placed
    assert [o["what"] for o in r["orders"]] == ["placed", "filled"]


def test_a_short_order_without_room_waits_for_the_next_sessions_touch_and_never_fills_after_its_end():
    s = A.Strategy("S", -1, slices=4, priority=1, house_maint=2.0, house_init=2.86, min_equity=2000.0, orders="fill")
    big = A.Strategy("B", -1, slices=1, priority=0, house_maint=2.0, house_init=2.86, min_equity=2000.0)
    blocker = {"strategy": "B", "tkn": "BIG", "ens": T0 + H, "xns": T0 + DAY + 2 * H, "e": 10.0, "x": 10.0,
               "hs_in": 0.0, "hs_out": 0.0, "dv20": 1e12, "pt": [T0 + DAY + 2 * H], "pc": [10.0]}
    P = _short_order({0: (T0 + 3 * H, 10.5, 2, 10.0), 1: (T0 + DAY + 3 * H, 10.5, 2, 10.0)},
                     end=T0 + 2 * DAY, place=[T0, T0 + DAY])
    r = A.simulate(P + [blocker], [s, big], A.Rules(trigger="close"), settle=lambda d: int(d) + 1)
    what = [o["what"] for o in r["orders"]]
    assert what == ["placed", "unfunded", "filled"]          # the blocker took the room; the next touch had it
    pos = next(p for p in r["positions"] if p["strategy"] == "S")
    assert pos["entry_ns"] == T0 + DAY + 3 * H
    late = _short_order({1: (T0 + 3 * DAY, 10.5, 2, 10.0)}, end=T0 + 2 * DAY, place=[T0, T0 + DAY])
    r2 = A.simulate(late, [s], A.Rules(trigger="close"), settle=lambda d: int(d) + 1)
    assert not r2["positions"]                                # a touch after the order ended: nothing


def test_the_mimic_switches_only_the_named_arms_to_limit_orders():
    m = Mimic(short_limit=(0.5, 5), short_limit_arms=("vol",))
    assert m.limited("vol") and not m.limited("etf") and not Mimic().limited("vol")
    from src.backtest.mimic import live_strategies
    st = {s.name: s for s in live_strategies(m)}
    assert st["vol"].orders == "fill" and st["etf"].orders is None and st["dip"].orders == "fill"
    with pytest.raises(ValueError):
        Mimic(short_limit=(0.5, 5), short_limit_arms=("dip",))
