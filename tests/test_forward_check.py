"""THE FORWARD CHECK (`src/backtest/forward_check.py`; PREREG185 -> production 2026-10-23): every live trade beside
the simulator's construction of it on the same prices, the books' gaps over episodes, the live account in the
backtest's tracking band, the EOD step and the email's notes. No network: the market is a fake."""
from __future__ import annotations

import json
import math
from datetime import date, datetime

import numpy as np
import pandas as pd
import pytest

from config import settings
from src.backtest import forward_check as FC
from src.backtest import mimic as M

ET = "America/New_York"
RTH_SLOTS = ["09:30", "10:00", "10:30", "11:00", "11:30", "12:00", "12:30", "13:00", "13:30", "14:00", "14:30",
             "15:00", "15:30"]


def _ns(day: str, hhmm: str) -> int:
    return int(pd.Timestamp(f"{day} {hhmm}", tz=ET).value)


def mkbars(days: dict) -> dict:
    """{ISO day: [(o, h, l, c) for each regular-hours bar, 13 of them]} -> the `Market.bars` form."""
    st, o, h, l, c = [], [], [], [], []
    for d in sorted(days):
        for slot, (bo, bh, bl, bc) in zip(RTH_SLOTS, days[d]):
            st.append(_ns(d, slot))
            o.append(bo), h.append(bh), l.append(bl), c.append(bc)
    start = np.asarray(st, np.int64)
    et = pd.DatetimeIndex(pd.to_datetime(start, utc=True)).tz_convert(ET)
    mins = et.hour * 60 + et.minute
    return {"start": start, "end": start + M.BAR_NS, "o": np.asarray(o, float), "h": np.asarray(h, float),
            "l": np.asarray(l, float), "c": np.asarray(c, float), "v": np.ones(len(start)),
            "rth": np.asarray((mins >= 570) & (mins < 960), bool),
            "day": (et.normalize().tz_localize(None) - M.EPOCH).days.to_numpy().astype(np.int64),
            "date": [x.date() for x in et.normalize()]}


def flat(px: float) -> list:
    return [(px, px, px, px)] * 13


class FakeMarket(FC.Market):
    def __init__(self, bars: dict, arrivals: dict = None, hs: float = 10.0, tmp=None):
        self.dir = tmp
        self._bars, self._open, self._quotes, self.requests = {}, {}, {}, 0
        self.B, self.A, self.H = bars, arrivals or {}, hs

    def bars(self, tk, lo, hi):
        return self.B.get(tk)

    def open_arrival(self, tk, s):
        return self.A.get((tk, s.isoformat()), {"arrival": None, "how": None, "mins": []})

    def half_spread_bp(self, tk, ns):
        return self.H

    def save(self):
        return None


@pytest.fixture
def prod(monkeypatch):
    for k, v in {"sel_short_vol_cover_multiple": 4.0, "sel_short_cover_multiple": 6.0,
                 "enable_sel_short_vol_volnorm_exit": False, "dip_long_exit_sma_sessions": 3,
                 "dip_long_max_hold_sessions": 5, "dip_long_limit_sessions": 3, "cap_long_exit_sma_sessions": 3,
                 "cap_long_max_hold_sessions": 5, "cap_long_limit_sessions": 3, "crash_long_exit_sma_sessions": 3,
                 "crash_long_max_hold_sessions": 5, "crash_long_pm_exit": True,
                 "live_account_fresh_start": "2026-10-20T09:30:00-04:00", "sel_short_account_initial": 50000.0}.items():
        monkeypatch.setattr(settings, k, v)


def short_trade(**kw) -> dict:
    t = {"trade_id": "s1", "ticker": "ZZZ", "entry_mechanism": "sel_short", "sel_arm": "vol", "action": "SELL",
         "status": "CLOSED", "entry_date": "2026-10-21", "exit_date": "2026-10-21",
         "entry_datetime": "2026-10-21T10:01:00-04:00", "sel_bar_end": "2026-10-21T10:00:00-04:00",
         "sel_bar_close": 10.0, "sel_target_price": 9.0, "sel_deadline": "2026-11-11T10:00:00-04:00",
         "sel_account_shares": 100, "entry_price": 10.0, "exit_price": 9.0,
         "exit_datetime": "2026-10-21T11:00:20-04:00", "exit_reason": "sel_target", "return_pct": 9.5}
    t.update(kw)
    return t


def test_a_short_leaves_at_its_target_on_the_simulators_marks(prod):
    bars = [(10, 10, 10, 10), (10, 10, 9.6, 9.6), (9.6, 9.6, 8.9, 8.95)] + [(8.95,) * 4] * 10
    b = mkbars({"2026-10-21": bars})
    s = FC.sim_short(short_trade(), b)
    assert s["entry"]["price"] == 10.0 and s["entry"]["kind"] == "immediate"
    assert s["exit"]["why"] == "target" and s["exit"]["price"] == 8.95
    assert s["exit"]["ns"] == _ns("2026-10-21", "11:00")             # the 10:30 bar's end


def test_a_short_covers_at_its_arms_multiple_and_leaves_at_its_deadline(prod):
    up = [(10, 10, 10, 10), (10, 41, 10, 40.5)] + [(40.5,) * 4] * 11
    s = FC.sim_short(short_trade(sel_target_price=1.0), mkbars({"2026-10-21": up}))
    assert s["exit"]["why"] == "cover"                                 # the vol arm's 4x
    s = FC.sim_short(short_trade(sel_arm="etf", sel_target_price=1.0), mkbars({"2026-10-21": up}))
    assert "exit" not in s                                             # the ETF arm's 6x not reached
    t = short_trade(sel_target_price=1.0, sel_deadline="2026-10-22T10:00:00-04:00")
    s = FC.sim_short(t, mkbars({"2026-10-21": flat(10.0), "2026-10-22": flat(10.5)}))
    assert s["exit"]["why"] == "time" and s["exit"]["ns"] >= _ns("2026-10-22", "10:00")


def dip_trade(**kw) -> dict:
    t = {"trade_id": "d1", "ticker": "AAA", "entry_mechanism": "dip_long", "action": "BUY", "status": "CLOSED",
         "entry_date": "2026-10-21", "exit_date": "2026-10-23", "entry_datetime": "2026-10-21T11:00:30-04:00",
         "dip_limit": 99.0, "dip_order_day": "2026-10-21", "dip_account_shares": 10, "entry_price": 99.0,
         "exit_price": 101.0, "exit_datetime": "2026-10-23T09:30:15-04:00", "exit_reason": "dip_rebound",
         "return_pct": 1.9}
    t.update(kw)
    return t


def test_a_dip_order_fills_at_the_touch_and_sells_at_the_open_after_its_rebound(prod):
    d1 = [(100, 100, 100, 100), (100, 100, 98.5, 99.2)] + [(99.2,) * 4] * 11      # the 10:00 bar touches 99
    days = {"2026-10-16": flat(103), "2026-10-19": flat(102), "2026-10-20": flat(101), "2026-10-21": d1,
            "2026-10-22": flat(104), "2026-10-23": [(105, 105, 105, 105)] + [(105,) * 4] * 12}
    s = FC.sim_order_book(dip_trade(), mkbars(days), FakeMarket({}), "dip")
    assert s["fill"]["kind"] == "touch" and s["fill"]["price"] == 99.0
    assert s["fill"]["ns"] == _ns("2026-10-21", "10:30")              # the touching bar's end
    assert s["exit"]["why"] == "rebound" and s["exit"]["session"] == "2026-10-23" and s["exit"]["price"] == 105


def test_an_open_fill_takes_the_attainable_price(prod):
    d1 = [(98.0, 99, 97.5, 98.5)] + [(98.5,) * 4] * 12                 # opened under the limit
    days = {"2026-10-16": flat(103), "2026-10-19": flat(102), "2026-10-20": flat(101), "2026-10-21": d1,
            "2026-10-22": flat(97), "2026-10-23": flat(96)}
    mk = FakeMarket({}, arrivals={("AAA", "2026-10-21"): {"arrival": 97.8, "how": "trades", "mins": []}})
    s = FC.sim_order_book(dip_trade(), mkbars(days), mk, "dip")
    assert s["fill"]["kind"] == "open" and s["fill"]["price"] == 97.8


def test_the_pairs_read_plus_when_live_does_better(prod):
    b = mkbars({"2026-10-21": [(10, 10, 10, 10), (10, 10, 9.6, 9.6), (9.6, 9.6, 8.9, 8.95)] + [(8.95,) * 4] * 10})
    t = short_trade(entry_price=10.1, exit_price=8.9)                  # sold higher and covered lower than the sim
    s = FC.sim_short(t, b)
    s["_b"] = b
    r = FC.compare(t, s, lambda arm, tk, ns, leg: (10.0, False))
    assert r["entry_gap_pct"] > 0 and r["exit_same_instant_gap_pct"] > 0
    assert r["same_tick"] is True and r["cost_sim_pct"] > 0.2


def test_the_books_gap_is_read_over_episodes():
    rows = [{"arm": "dip", "entry_date": "2026-10-21", "exit_date": "2026-10-23", "net_gap_pct": 1.0},
            {"arm": "dip", "entry_date": "2026-10-21", "exit_date": "2026-10-24", "net_gap_pct": 3.0},
            {"arm": "dip", "entry_date": "2026-10-27", "exit_date": "2026-10-29", "net_gap_pct": -1.0}]
    a = FC.by_book(rows, "net_gap_pct")["dip"]
    assert a["n"] == 3 and a["episodes"] == 2 and a["mean"] == pytest.approx(1.0)
    assert a["se_episodes"] == pytest.approx(np.std([2.0, -1.0], ddof=1) / math.sqrt(2))


def _runs(rate: float) -> dict:
    v = list(np.exp(np.arange(300) * rate))
    days = [str(d.date()) for d in pd.bdate_range("2024-01-02", periods=300)]
    return {"A": {0: {"nav": {"days": days, "values": v}}, 1: {"nav": {"days": days, "values": v}}}}


def test_the_band_and_its_held_out_shift():
    bd = FC.band(_runs(math.log(1.75) / 252))
    b1 = bd["band"]["1"]
    assert b1["in_sample"]["50"] == pytest.approx(100 * (1.75 ** (1 / 252) - 1), rel=1e-6)
    assert bd["shift_per_session"] == pytest.approx((math.log(1.75) - math.log(1.61)) / 252, rel=1e-6)
    assert b1["held_out"]["50"] == pytest.approx(100 * (1.61 ** (1 / 252) - 1), rel=1e-5)
    assert FC.percentile(bd, 1, -1.0) == 0.0 and FC.percentile(bd, 1, 5.0) == 100.0


def test_notes_flag_a_gap_beyond_noise_and_the_account_outside_its_band():
    rec = {"by_book": {"net_gap_pct": {"dip": {"mean": -0.9, "t": -2.5, "episodes": 6},
                                       "vol": {"mean": -5.0, "t": -3.0, "episodes": 2}}},
           "account": {"last": {"n": 3, "growth_pct": -4.3, "percentile_in_sample": 4.0,
                                "percentile_held_out": 9.0}}}
    n = FC.notes(rec)
    assert len(n) == 2 and n[0].startswith("dip:") and "4th percentile" in n[1]   # vol: too few episodes


def test_live_trades_fold_the_margin_lots_and_skip_the_record_before_the_fresh_start(prod):
    T = [dip_trade(trade_id="d1"), dip_trade(trade_id="d1-mc1", margin_lot_of="d1"),
         dip_trade(trade_id="old", entry_datetime="2026-10-19T10:00:00-04:00")]
    got = FC.live_trades(T)
    assert [t["trade_id"] for t in got] == ["d1"] and [x["trade_id"] for x in got[0]["_lots"]] == ["d1-mc1"]


def test_build_and_health_end_to_end(prod, monkeypatch, tmp_path):
    d1 = [(100, 100, 100, 100), (100, 100, 98.5, 99.2)] + [(99.2,) * 4] * 11
    days = {"2026-10-16": flat(103), "2026-10-19": flat(102), "2026-10-20": flat(101), "2026-10-21": d1,
            "2026-10-22": flat(104), "2026-10-23": flat(105)}
    mk = FakeMarket({"AAA": mkbars(days)}, tmp=tmp_path)
    rec = FC.build(date(2026, 10, 23), trades=[dip_trade()], mk=mk, runs=_runs(math.log(1.75) / 252))
    assert rec["trades"] == 1 and rec["closed"] == 1 and not rec["errors"]
    r = rec["rows"][0]
    assert r["arm"] == "dip" and r["exit_sim"]["why"] == "rebound" and r["net_gap_pct"] is not None
    assert rec["account"]["last"]["session"] == "2026-10-23"
    FC.out_dir().mkdir(parents=True, exist_ok=True)
    FC.out_path(date(2026, 10, 23)).write_text(json.dumps(rec, default=str), encoding="utf-8")
    h = FC.health()
    assert h["problems"] == [] and h["notes"][0].startswith("forward check as of 2026-10-23: 1 live trades")


def test_the_eod_chain_runs_it_after_the_live_calibration():
    from src.scheduler import eod
    assert eod.STEPS[-1] == "forward_check" and "forward_check" in eod.STEP_FNS
    assert type(settings).model_fields["enable_eod_forward_check"].default is True
    assert str(FC.out_dir()).endswith("forward_check")                # the test's own folder (conftest)
