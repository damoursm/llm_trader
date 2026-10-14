"""THE TURN in the simulator (PREREG76; production's `sel_short.turn_state`, the faller shorts from 2026-10-12):
`strategies.vol.turn_candidates` finds the first bar inside the window whose close is at least k x the pick's ATR$
under the highest close since the pick; the simulator that mimics production uses it for the fall arm by default.

What must hold: an immediate faller triggers on its first bar; a riser triggers when it turns, the running maximum
including the later highs; a turn after the window, or on the deadline bar, is no trade; a close at/under the target
first (the same bar included) cancels the pick; the simulator and production find the SAME trigger bar on the same
path; `Mimic()` carries the turn, `Mimic.off()` does not, and nonsense parameters are refused.
"""
from __future__ import annotations

from datetime import date, timedelta

import numpy as np
import pandas as pd
import pytest

from src.backtest.mimic import Mimic
from src.backtest.strategies.vol import BIG, turn_candidates

CAL = np.array([20000 + i for i in range(12)], np.int64)        # 12 consecutive sessions (day numbers)


def _path(closes, day0=20000, per=13, first_bar=1):
    """One pick's forward path (the bars after the pick bar ``first_bar - 1`` of session ``day0``)."""
    psd, d, k = [], day0, first_bar
    for _ in closes:
        psd.append(d)
        k += 1
        if k >= per:
            d, k = d + 1, 0
    return np.asarray(closes, float)[None, :], np.asarray(psd, np.int64)[None, :]


def _turn(closes, px=10.0, drop=0.2, lvl=-np.inf, eff=BIG, W=5):
    pc, psd = _path(closes)
    return int(turn_candidates(pc, psd, np.array([20000]), CAL, np.array([px]), np.array([drop]),
                               np.array([lvl]), np.array([eff]), W)[0])


def test_an_immediate_faller_triggers_on_its_first_bar():
    assert _turn([9.75, 9.5]) == 0


def test_a_riser_triggers_when_it_turns_from_its_highest_close():
    assert _turn([10.5, 10.8, 10.7, 10.55, 10.2]) == 3                 # 10.55 <= 10.8 - 0.2


def test_no_turn_no_trade():
    assert _turn([10.1, 10.2, 10.3]) == -1


def test_a_turn_after_the_window_is_no_trade():
    n = 12 + 13 * 4                                                    # the window's bars after the pick (bar 1)
    closes = [10.1] * n + [9.0]
    assert _turn(closes) == -1
    assert _turn(closes, W=6) == n


def test_the_target_first_cancels_the_pick_and_the_same_bar_too():
    assert _turn([10.5, 10.25, 9.5], lvl=9.0) == 1                     # 10.25 <= 10.5 - 0.2: the turn
    assert _turn([10.5, 10.25, 9.5], lvl=10.25) == -1                  # the turning bar itself reaches the target
    assert _turn([9.95, 10.5, 10.2], lvl=9.95) == -1                   # the target first (bar 0), the turn at bar 2


def test_a_turn_on_the_deadline_bar_is_no_trade():
    assert _turn([10.5, 10.25], eff=1) == -1
    assert _turn([10.5, 10.25], eff=2) == 1


def test_the_simulator_and_production_find_the_same_turn(monkeypatch):
    """The same random paths through production's `sel_short.turn_state` and the simulator's `turn_candidates`."""
    from config.settings import settings
    from src.analysis.ml_dataset import hlc_from_frames
    from src.signals import sel_short as ss
    monkeypatch.setattr(settings, "sel_short_turn_atr", 0.5)
    monkeypatch.setattr(settings, "sel_short_turn_sessions", 5)
    monkeypatch.setattr(settings, "sel_short_entry_max_age_minutes", 75.0)
    day0 = date(2026, 9, 28)
    days = [day0] + [ss.session_after(day0, i) for i in range(1, 8)]
    cal = np.array([ss.dnum(d) for d in days], np.int64)
    idx = pd.DatetimeIndex([pd.Timestamp(ss.bar_end_et(d, b) - timedelta(minutes=30)).tz_convert("UTC").tz_localize(None)
                            for d in days for b in range(13)])
    seen = {"hit": 0, "none": 0}
    for seed in range(44):
        rng = np.random.default_rng(seed)
        steps = rng.normal(0, 0.015, len(idx)) if seed < 40 else np.full(len(idx), 0.002)   # the last four never turn
        c = 20.0 * np.exp(np.cumsum(steps))
        df = pd.DataFrame({"Open": c, "High": c * 1.004, "Low": c * 0.996, "Close": c, "Volume": 1e5}, index=idx)
        pick, atr = int(rng.integers(0, 13)), float(rng.choice([1.0, 2.5, 6.0]))
        rec = {"day": day0.isoformat(), "bar_of_day": pick, "ticker": "SYN", "atr_pct": atr, "px": float(c[pick])}
        st = ss.turn_state(rec, ss.turn_window_end(rec) + timedelta(days=3), hlc=hlc_from_frames([df]))
        path = c[pick + 1:][None, :]
        psd = np.array([ss.dnum(t.tz_localize("UTC").tz_convert(ss.ET).date()) for t in idx[pick + 1:]], np.int64)[None, :]
        j = int(turn_candidates(path, psd, np.array([cal[0]]), cal, np.array([c[pick]]),
                                np.array([0.5 * c[pick] * atr / 100.0]), np.array([-np.inf]), np.array([BIG]), 5)[0])
        if j < 0:
            assert st["state"] == "lapsed", seed
            seen["none"] += 1
        else:
            end = pd.Timestamp(idx[pick + 1 + j] + pd.Timedelta(minutes=30)).tz_localize("UTC").tz_convert(ss.ET)
            assert st["state"] == "triggered" and pd.Timestamp(st["bar_end"]) == end, seed
            assert st["close"] == pytest.approx(c[pick + 1 + j])
            seen["hit"] += 1
    assert seen["hit"] > 5 and seen["none"] > 0                        # both outcomes exercised


def test_the_mimic_carries_the_turn_and_the_old_simulator_does_not():
    assert Mimic().fall_turn == (0.5, 5)
    assert Mimic.off().fall_turn is None
    with pytest.raises(ValueError):
        Mimic(fall_turn=(0.0, 5))
    with pytest.raises(ValueError):
        Mimic(fall_turn=(0.5, 0))


# ── PREREG77: the live rules the simulator now mimics ──

def test_the_vol_rule_covers_at_4x_without_its_volatility_exit_and_the_fall_arm_keeps_both():
    from src.backtest.strategies.vol import FALL, LIVE, VOL2
    assert LIVE["cap"] == 4.0 and LIVE["volnorm_on"] is False
    fall = dict(LIVE, **FALL)                                          # how `VolEngine.cfg` merges an arm's params
    assert fall["cap"] == 6.0 and fall["volnorm_on"] is True and fall["fallback"] == 0 and fall["give"] is None
    assert dict(LIVE, **VOL2)["cap"] == 6.0


def test_the_mimic_carries_the_etf_turn_and_the_old_simulator_does_not():
    assert Mimic().etf_turn == (0.5, 5)
    assert Mimic.off().etf_turn is None
    with pytest.raises(ValueError):
        Mimic(etf_turn=(0.5, 0))


def _etf_stub():
    from src.backtest import account as A
    from src.backtest.strategies.etf import EtfArm

    class _CM:
        @staticmethod
        def predict(price, dv, atr, bod, year):
            return np.zeros(len(price))

    class _RF:
        @staticmethod
        def daily(tkn, days, fb):
            return np.zeros(len(days))
    arm = object.__new__(EtfArm)
    arm.cal = np.array([20000 + i for i in range(30)], np.int64)
    arm.cm, arm.rf, arm.settle = _CM(), _RF(), A.default_settle()
    return arm


def _etf_bars(closes, per=13):
    n = len(closes)
    sday = np.array([20000 + i // per for i in range(n)], np.int64)
    bod = np.array([i % per for i in range(n)], np.int64)
    ns = np.array([(int(d) * 86400 + 14 * 3600 + 30 * 60 * (b + 1)) * 10**9 for d, b in zip(sday, bod)], np.int64)
    c = np.asarray(closes, float)
    return {"c": c, "h": c * 1.001, "l": c * 0.999, "atr": np.full(n, 4.0), "ns": ns, "sday": sday, "bod": bod,
            "pos": bod}


def test_the_etf_turn_enters_at_the_turning_close_and_its_target_first_cancels_it():
    import types
    arm = _etf_stub()
    r = types.SimpleNamespace(tkn="EEE@E", dv20=5e7, year=2022, fee_file=float("nan"), dn=20000)
    a = _etf_bars([10.0, 10.5, 10.8, 10.55, 10.3, 10.2, 10.1, 10.0])
    q = arm._turn_piece(r, a, 0, 0, 7, False, 10.0, 9.0, 1.0, (0.5, 5), 6.0, "etf", True)
    assert q is not None and q["e"] == pytest.approx(10.55) and q["ens"] == int(a["ns"][3])   # 10.55 <= 10.8 - 0.2
    assert q["pt"][0] == int(a["ns"][4])                                                     # exits from the next bar
    s = _etf_bars([10.0, 10.6, 10.8, 10.55, 10.3])                                             # the turning bar
    assert arm._turn_piece(r, s, 0, 0, 4, False, 10.0, 10.5, 1.0, (0.5, 5), 6.0, "etf", True)["e"] == pytest.approx(10.55)
    assert arm._turn_piece(r, s, 0, 0, 4, False, 10.0, 10.56, 1.0, (0.5, 5), 6.0, "etf", True) is None  # reaches it too
    b = _etf_bars([10.0, 9.85, 10.5, 10.2, 10.0])                                              # the target first
    assert arm._turn_piece(r, b, 0, 0, 4, False, 10.0, 9.9, 1.0, (0.5, 5), 6.0, "etf", True) is None
