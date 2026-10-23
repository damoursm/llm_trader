"""THE LIVE ACCOUNT LIQUIDATES AS IBKR DOES (PREREG174 / 175; the user 2026-10-21: "production and backtest needs to
liquidate like IBKR does"): `live_account.plan_liquidation` is the account engine's `Rules(liquidation="ibkr",
liq_fill="path")` for the ledger's account — the most recently opened positions first, in 100-share lots, only until
equity is back at the maintenance requirement, each lot filled 10% against the account from where the deficit appears
along the bar, never past the bar's extreme; the end-of-day Reg T floor at the marks. PARITY (PREREG175's G4): on the
engine's own scenarios the plan's lots and fills equal the engine's. Then `margin_watch` (what a check reads), the
tracker's execution (a part of a trade split off as its own trade) and the broker's legs of a lot."""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo

import pandas as pd
import pytest

from config.settings import settings
from src.backtest import account as A
from src.performance import live_account as la
from src.performance import margin_watch as mw
from src.performance import sim_account as sa
from tests.test_broker_reconcile import FakeBroker, repo_store  # noqa: F401 — the fixture
from tests.test_live_account import NOW, dip, on, short  # noqa: F401 — the fixture

ET = ZoneInfo("America/New_York")


def et(s: str) -> int:
    return pd.Timestamp(s, tz="America/New_York").value


def settle(dn):
    return int(dn) + 1


def leg(strategy, tkn, e, ens, xns, x, bars, **kw):
    """A cost-free piece (the plan prices no cost: the ledger's own costs price the legs)."""
    p = {"strategy": strategy, "tkn": tkn, "ens": et(ens), "xns": et(xns), "e": e, "x": x, "hs_in": 0.0, "hs_out": 0.0,
         "dv20": 1e9, "pt": [et(t) for t, _ in bars], "pc": [float(c) for _, c in bars], "fees": "ledger"}
    p.update(kw)
    return p


def short_pos(key, n, mark, prev=None, ext=None, order=0, whole=False, hm=0.0):
    return {"key": key, "side": -1, "n": float(n), "mark": float(mark), "prev": float(mark if prev is None else prev),
            "ext": float(mark if ext is None else ext), "order": order, "whole": whole,
            "maint": lambda q, p: max(sa.maint(q, p), hm * q * p),
            "regt": lambda q, p: max(0.5 * q * p, sa.maint(q, p))}


def long_pos(key, n, mark, order=0):
    return {"key": key, "side": 1, "n": float(n), "mark": float(mark), "prev": float(mark), "ext": float(mark),
            "order": order, "whole": False, "maint": lambda q, p: 0.25 * q * p, "regt": lambda q, p: 0.5 * q * p}


# ── PARITY with the engine (PREREG175's G4) ───────────────────────────────────

def test_parity_the_most_recent_first_in_lots_until_back_at_maintenance():
    """Three Reg T shorts of 125 shares at $20: OLD (10:00), NEW1 (11:00), NEW2 (12:00); OLD marked at $80."""
    s = A.Strategy("S", -1, slices=4, min_equity=2000.0)
    P = [leg("S", "OLD", 20.0, "2024-01-08 10:00", "2024-01-11 10:00", 20.0,
             [("2024-01-09 10:00", 80.0), ("2024-01-10 10:00", 20.0)]),
         leg("S", "NEW1", 20.0, "2024-01-08 11:00", "2024-01-11 11:00", 20.0, [("2024-01-10 11:00", 20.0)]),
         leg("S", "NEW2", 20.0, "2024-01-08 12:00", "2024-01-11 12:00", 20.0, [("2024-01-10 12:00", 20.0)])]
    r = A.simulate(P, [s], A.Rules(trigger="close", liquidation="ibkr", fees="ledger"), settle=settle)
    eng = [(x["tkn"], x["shares"], x["closed"], x["px"]) for x in r["liquidations"]]
    held = {p["tkn"]: p["shares"] for p in r["positions"]}
    eq = 10_000.0 + held["OLD"] * (20.0 - 80.0)                    # every position at its mark, no cost
    pos = [short_pos("OLD", held["OLD"], 80.0, order=1), short_pos("NEW1", held["NEW1"], 20.0, order=2),
           short_pos("NEW2", held["NEW2"], 20.0, order=3)]
    lots = la.plan_liquidation(pos, eq, "maint")
    assert [(x["key"], x["shares"], x["closed"], x["fill"]) for x in lots] == eng
    assert eng == [("NEW2", 125, True, 20.0), ("NEW1", 125, True, 20.0), ("OLD", 100, False, 80.0)]


def test_parity_the_path_sells_at_each_crossing_one_band_above_it():
    """One short of 500 at $20 (Reg T's 30%); the next bar closes at $25 with a high of $40."""
    s = A.Strategy("S", -1, slices=1, min_equity=2000.0)
    P = [leg("S", "SQ", 20.0, "2024-01-08 10:00", "2024-01-11 10:00", 20.0, [("2024-01-09 11:00", 25.0)], ph=[40.0])]
    r = A.simulate(P, [s], A.Rules(trigger="high", liquidation="ibkr", liq_fill="path", fees="ledger"), settle=settle)
    eng = r["liquidations"]
    assert [x["shares"] for x in eng] == [100, 100, 100, 100]
    pos = [short_pos("SQ", 500, 25.0, prev=20.0, ext=40.0)]
    lots = la.plan_liquidation(pos, 10_000.0 + 500 * (20.0 - 25.0), "maint")
    assert [x["shares"] for x in lots] == [x["shares"] for x in eng]
    assert [x["at"] for x in lots] == pytest.approx([x["mark"] for x in eng], abs=1e-6)
    assert [x["fill"] for x in lots] == pytest.approx([x["px"] for x in eng], abs=1e-6)
    assert [x["at"] for x in lots] == pytest.approx([30.76, 31.95, 33.59, 36.17], abs=0.06)
    # a bar that never breaks the requirement sells nothing
    assert la.plan_liquidation([short_pos("SQ", 500, 25.0, prev=20.0, ext=30.0)], 7_500.0, "maint") == []


def test_parity_the_reg_t_floor_sells_the_latest_long_at_the_marks():
    s = A.Strategy("L", 1, slices=1)
    bars = [("2024-01-09 16:00", 80.0)]
    P = [leg("L", "L1", 100.0, "2024-01-08 10:00", "2024-01-10 10:00", 80.0, bars),
         leg("L", "L2", 100.0, "2024-01-08 11:00", "2024-01-10 10:00", 80.0, bars)]
    r = A.simulate(P, [s], A.Rules(trigger="close", gross_cap={1: 2.0, -1: 1.0}, liquidation="ibkr", regt_close=True,
                                   fees="ledger"), settle=settle)
    assert r["call_kinds"] == ["regt"]
    held = {p["tkn"]: p["shares"] for p in r["positions"]}
    eq = 10_000.0 + (held["L1"] + held["L2"]) * (80.0 - 100.0)
    pos = [long_pos("L1", held["L1"], 80.0, order=1), long_pos("L2", held["L2"], 80.0, order=2)]
    assert la.plan_liquidation(pos, eq, "maint") == []              # maintenance (25%) holds
    lots = la.plan_liquidation(pos, eq, "regt")
    assert [(x["key"], x["shares"], x["closed"]) for x in lots] == \
        [(x["tkn"], x["shares"], x["closed"]) for x in r["liquidations"]] == [("L2", held["L2"], True)]


def test_the_engine_s_reg_t_check_reads_the_marks_not_the_bar_s_extreme():
    """PREREG175: the end-of-day check at the prices of its instant — a long whose bar dipped through Reg T's floor but
    closed above it is not sold at the end of the day (the maintenance path already met the bar's low)."""
    s = A.Strategy("L", 1, slices=1)
    P = [leg("L", "L1", 100.0, "2024-01-08 10:00", "2024-01-10 10:00", 99.0, [("2024-01-09 16:00", 99.0)], pl=[60.0]),
         leg("L", "L2", 100.0, "2024-01-08 11:00", "2024-01-10 10:00", 99.0, [("2024-01-09 16:00", 99.0)], pl=[60.0])]
    r = A.simulate(P, [s], A.Rules(trigger="high", gross_cap={1: 2.0, -1: 1.0}, liquidation="ibkr", regt_close=True,
                                   liq_fill="path", fees="ledger"), settle=settle)
    assert "regt" not in "+".join(r["call_kinds"])


# ── the plan ─────────────────────────────────────────────────────────────────

def test_a_position_that_cannot_be_split_goes_whole():
    pos = [short_pos("SQ", 500, 32.0, order=1, whole=True)]
    lots = la.plan_liquidation(pos, 10_000.0 + 500 * (20.0 - 32.0), "maint")
    assert [(x["shares"], x["closed"]) for x in lots] == [(500, True)]
    lots = la.plan_liquidation([short_pos("SQ", 500, 32.0, order=1)], 10_000.0 + 500 * (20.0 - 32.0), "maint")
    assert [(x["shares"], x["closed"]) for x in lots] == [(100, False)]          # one lot cures it


def test_a_fractional_holding_sells_its_hundreds_then_the_rest():
    pos = [short_pos("RS", 342.5, 40.0, order=1)]
    eq = 10_000.0 + 342.5 * (20.0 - 40.0)                           # $3,150 against 30% of $13,700 = $4,110
    lots = la.plan_liquidation(pos, eq, "maint")
    assert [x["shares"] for x in lots] == [100]                     # 100 x $12 of requirement released
    assert [x["shares"] for x in la.plan_liquidation(pos, 1_000.0, "maint")] == [300]
    assert [(x["shares"], x["closed"]) for x in la.plan_liquidation(pos, 100.0, "maint")] == [(342.5, True)]


# ── what a check reads (margin_watch) ────────────────────────────────────────

def test_the_latest_bar_end_skips_the_overnight_hours():
    def at(s):
        return datetime.fromisoformat(s).replace(tzinfo=ET).astimezone(timezone.utc)
    assert mw.latest_bar_end(at("2026-10-21 18:00:30")) == at("2026-10-21 18:00:00")
    assert mw.latest_bar_end(at("2026-10-21 19:50:10")) == at("2026-10-21 19:30:00")
    assert mw.latest_bar_end(at("2026-10-21 21:30:00")) == at("2026-10-21 20:00:00")
    assert mw.latest_bar_end(at("2026-10-22 03:30:00")) == at("2026-10-21 20:00:00")
    assert mw.latest_bar_end(at("2026-10-22 05:00:20")) == at("2026-10-22 05:00:00")


def test_a_check_reads_the_bars_since_the_last_and_paths_from_its_price(on, monkeypatch):
    t0 = datetime(2026, 9, 28, 17, 30, tzinfo=ET).astimezone(timezone.utc)
    now = datetime(2026, 9, 28, 18, 0, 20, tzinfo=ET).astimezone(timezone.utc)
    tr = short("SQ", n=300, px=10.0, mark=12.0, ret=-20.0, at="2026-09-28T14:00:00+00:00")
    st = {"checked_at": t0.isoformat(), "trades": {"sel-SQ": {"px": 11.0, "bar_end": t0.isoformat(), "f": 1.0}}}
    seen = {}

    def bars(ticker, after, now_):
        seen[ticker] = (after, now_)
        return [(now_.replace(second=0), 19.0, 10.5)]
    monkeypatch.setattr(mw, "bars_since", bars)
    pos, new, notes = mw.inputs([tr], now, st)
    assert seen["SQ"][0] == t0 and notes == []
    assert pos[0]["prev"] == 11.0 and pos[0]["ext"] == 19.0 and pos[0]["mark"] == 12.0 and pos[0]["n"] == 300
    assert new["sel-SQ"]["bar_end"] == now.replace(second=0).isoformat()
    # a stock split since the last check: the last price in today's terms
    tr2 = dict(tr, split_factor=0.5)
    pos2, _, _ = mw.inputs([tr2], now, st)
    assert pos2[0]["prev"] == pytest.approx(5.5)
    # the first check (no state): the marks, no bar read before it
    pos3, new3, _ = mw.inputs([tr], now, {})
    assert pos3[0]["prev"] == pos3[0]["ext"] == 12.0
    assert new3["sel-SQ"]["bar_end"] == mw.latest_bar_end(now).isoformat()
    # Polygon unreadable: judged at the mark, the bars read again next time
    monkeypatch.setattr(mw, "bars_since", lambda ticker, after, now_: None)
    pos4, new4, notes4 = mw.inputs([tr], now, st)
    assert pos4[0]["ext"] == 12.0 and new4["sel-SQ"]["bar_end"] == t0.isoformat() and notes4


def test_the_reg_t_check_runs_once_a_session_from_15_50(monkeypatch):
    def at(s):
        return datetime.fromisoformat(s).replace(tzinfo=ET).astimezone(timezone.utc)
    assert not mw.regt_due(at("2026-10-21 15:30:00"), {})
    assert mw.regt_due(at("2026-10-21 16:00:10"), {})
    assert not mw.regt_due(at("2026-10-21 16:00:10"), {"regt_done": "2026-10-21"})
    assert not mw.regt_due(at("2026-10-21 20:30:00"), {})                   # the overnight session: not that day's
    assert not mw.regt_due(at("2026-10-24 16:00:00"), {})                   # a Saturday
    monkeypatch.setattr(settings, "live_account_regt_close", False)
    assert not mw.regt_due(at("2026-10-21 16:00:10"), {})


# ── the ledger (tracker.monitor_live_account) ─────────────────────────────────

def _ledger(monkeypatch, trades):
    from src.db import repo
    from src.performance import tracker
    repo.save_trades(list(trades), allow_shrink=True)
    monkeypatch.setattr(tracker, "_fetch_price", lambda t: None)
    monkeypatch.setattr(tracker, "_reference_close", lambda t: None)
    monkeypatch.setattr(tracker, "_execution_iso", lambda: "2026-09-28T14:40:00+00:00")
    return tracker


def test_a_lot_is_split_off_as_its_own_trade_and_the_rest_stays_open(monkeypatch, on):
    sq = short("SQ", n=300, px=10.0, mark=35.0, ret=-250.0, at="2026-09-28T14:10:00+00:00")
    sq.update(sel_house_maint=0.0, sel_house_init=0.0, sel_account_notional=3000.0, broker_order_id="o1",
              broker_client_ref="sel-SQ", broker_status="Filled", broker_fill_qty=300, broker_requested_qty=300,
              broker_commission=1.5, broker_fill_price=10.0)
    old = short("OLD", n=50, px=10.0, mark=10.0, at="2026-09-28T13:40:00+00:00")
    old.update(sel_house_maint=0.0, sel_house_init=0.0)
    # equity $10,000 - $7,500 = $2,500 against 30% of $10,500 + OLD's $250 = $3,400: one lot of SQ (the latest)
    # releases $1,050 — back at the requirement ($2,350)
    tracker = _ledger(monkeypatch, [old, sq])
    st = la.state(tracker._load_trades(), NOW)
    assert st["equity"] == pytest.approx(2_500.0) and st["maint"] == pytest.approx(3_400.0)
    assert tracker.monitor_live_account(now=NOW) == 1
    T = {x["recommendation_id"]: x for x in tracker._load_trades()}
    lot = T["sel-SQ-mc1"]
    assert lot["margin_lot_of"] == "sel-SQ" and lot["status"] == "CLOSED"
    assert lot["exit_reason"] == "sel_margin_call" and lot["sel_account_shares"] == 100
    assert lot["exit_price"] == pytest.approx(35.0)                 # the first check: judged at the marks
    rest = T["sel-SQ"]
    assert rest["status"] == "OPEN" and rest["sel_account_shares"] == 200 and rest["margin_lots"] == 1
    assert rest["broker_fill_qty"] + lot["broker_fill_qty"] == 300 and rest["broker_split_out_qty"] == lot["broker_fill_qty"]
    assert lot["broker_status"] == "MARGIN_LOT" and rest["broker_status"] == "Filled"
    assert rest["broker_commission"] + lot["broker_commission"] == pytest.approx(1.5)
    assert rest["sel_account_notional"] + lot["sel_account_notional"] == pytest.approx(3000.0)
    assert rest["position_size_multiplier"] + lot["position_size_multiplier"] == pytest.approx(1.0)   # the NAV's weight
    assert lot["margin_liquidation"]["kind"] == "maint" and T["sel-OLD"]["status"] == "OPEN"   # SQ is the latest
    from src.db.repo import _trade_id
    assert _trade_id(lot) != _trade_id(rest)
    assert mw.STATE.exists() and "sel-SQ" in mw.load_state()["trades"] and "sel-SQ-mc1" not in mw.load_state()["trades"]
    after = la.state(tracker._load_trades(), NOW)
    assert after["equity"] >= after["maint"] - 1.0                   # back at the requirement (the ledger's costs aside)


def test_the_account_check_liquidates_the_latest_whole_shorts_under_one_lot(monkeypatch, on):
    worse = [dip("D0")] + [short(f"S{k}", n=40, px=10.0, mark=40.0, ret=-300.0) for k in range(3)]
    tracker = _ledger(monkeypatch, worse)
    assert tracker.monitor_live_account(now=NOW) == 2
    t = {x["ticker"]: x for x in tracker._load_trades()}
    assert t["S2"]["exit_reason"] == t["S1"]["exit_reason"] == "sel_margin_call"
    assert t["S0"]["status"] == "OPEN" and t["D0"]["status"] == "OPEN"
    # the rule before: every position the account funded
    monkeypatch.setattr(settings, "live_account_liquidation", "all")
    tracker = _ledger(monkeypatch, worse)
    assert tracker.monitor_live_account(now=NOW) == 4


def test_a_failing_check_falls_back_to_closing_everything(monkeypatch, on):
    worse = [dip("D0")] + [short(f"S{k}", n=40, px=10.0, mark=40.0, ret=-300.0) for k in range(3)]
    tracker = _ledger(monkeypatch, worse)

    def boom(*a, **k):
        raise RuntimeError("bars")
    monkeypatch.setattr(mw, "inputs", boom)
    assert tracker.monitor_live_account(now=NOW) == 4


# ── the broker's legs of a lot (reconcile) ───────────────────────────────────

def test_the_lot_covers_its_shares_and_the_rest_keeps_its_own(repo_store):  # noqa: F811
    import src.broker.reconcile as rec
    from src.broker.base import Position
    rest = {"ticker": "SQ", "action": "SELL", "status": "OPEN", "entry_price": 10.0, "position_size_multiplier": 1.0,
            "recommendation_id": "sel-SQ", "entry_mechanism": "sel_short", "sel_account_shares": 200,
            "broker_order_id": "o1", "broker_client_ref": "sel-SQ", "broker_status": "Filled", "broker_fill_qty": 200,
            "broker_requested_qty": 200, "broker_fill_price": 10.0, "broker_commission": 1.0,
            "broker_split_out_qty": 100}
    lot = dict(rest, recommendation_id="sel-SQ-mc1", margin_lot_of="sel-SQ", status="CLOSED", sel_account_shares=100,
               broker_fill_qty=100, broker_requested_qty=100, broker_status="MARGIN_LOT", exit_price=33.0,
               broker_commission=0.5)
    lot.pop("broker_split_out_qty")
    repo_store["trades"] = [rest, lot]
    b = FakeBroker(positions=[Position("SQ", -300, 30.0)])
    r = rec.sync(broker=b)
    covers = [o for o in b.orders if o.side == "BUY"]
    assert r["exits_submitted"] == 1 and len(covers) == 1 and covers[0].quantity == 100
    assert covers[0].client_ref == "sel-SQ-mc1-exit"
    assert not [o for o in b.orders if o.side == "SELL"]           # the remainder sends no new entry


def test_the_fill_refresh_leaves_the_lots_shares_to_the_lots():
    import src.broker.reconcile as rec
    from src.broker.base import FillSummary
    t = {"ticker": "SQ", "action": "SELL", "status": "OPEN", "broker_order_id": "o1", "broker_status": "PartiallyFilled",
         "broker_fill_qty": 200, "broker_requested_qty": 200, "broker_split_out_qty": 100, "entry_price": 10.0}
    fs = FillSummary(client_ref="sel-SQ", filled_qty=300, avg_fill_price=10.0, commission=1.5)
    assert rec._apply_fill(t, "broker_", fs) is True                # the price and the commission are new
    assert t["broker_fill_qty"] == 200 and t["broker_status"] == "Filled"
    assert t["broker_commission"] == pytest.approx(1.0)
