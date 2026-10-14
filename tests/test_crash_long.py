"""THE CRASH-REBOUND BOOK (user 2026-10-11, PREREG86: "Crash book only"; PREREG88: "deploy the QQQ decisions, trade
TQQQ"; `src/signals/crash_long.py`).

What must hold: TQQQ is bought at the open when QQQ's previous close was under its 200-session average with a
2-session RSI under 5 — decided from QQQ's closes through D-1 only, kept once; sold at the open after QQQ's first close
above its 5-session average since the entry (never TQQQ's), or after 10 sessions; one long at a time; 1/4 of the LIVE
ACCOUNT's equity within the long side's cap and the margin room at the fund's 75% rate, selling the dip book's latest
longs whole when that makes room; a margin call closes it; the broker orders its share count; the simulator builds
the same longs as production, held in the traded fund.
"""
from __future__ import annotations

from datetime import date, datetime, timedelta, timezone
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import pytest

from config.settings import settings
from src.performance import books
from src.performance import live_account as la
from src.performance import tracker
from src.signals import crash_long as cl
from src.signals import dip_long as dl

ET = ZoneInfo("America/New_York")
DAY = date(2026, 10, 8)                        # a Thursday session; D-1 = 2026-10-07
OPEN_TICK = datetime(2026, 10, 8, 9, 30, 20, tzinfo=ET).astimezone(timezone.utc)
NOW = OPEN_TICK


def _sessions(end: date, n: int):
    """The last ``n`` market sessions on or before ``end`` (the real calendar)."""
    out, d = [], end
    while len(out) < n:
        if dl.is_session(d):
            out.append(d)
        d -= timedelta(days=1)
    return out[::-1]


def _frame(closes, end=date(2026, 10, 7)):
    days = _sessions(end, len(closes))
    c = np.asarray(closes, float)
    return pd.DataFrame({"o": c, "h": c * 1.01, "l": c * 0.99, "c": c, "v": np.full(len(c), 1e7)},
                        index=pd.DatetimeIndex([pd.Timestamp(x) for x in days]))


def _crash_closes(n=240):
    """A long rise, then a fall under the 200-session average ending with two sharp down days (RSI2 near 0)."""
    up = list(np.linspace(300, 500, n - 30))
    down = list(np.linspace(495, 420, 28))
    return up + down + [400.0, 380.0]


@pytest.fixture
def on(monkeypatch, tmp_path):
    monkeypatch.setattr(settings, "enable_crash_long", True)
    monkeypatch.setattr(settings, "crash_long_dir", str(tmp_path / "crash_long"))
    monkeypatch.setattr(settings, "crash_long_symbol", "QQQ")
    monkeypatch.setattr(settings, "crash_long_trade_symbol", "TQQQ")
    monkeypatch.setattr(settings, "crash_long_margin_rate", 0.75)
    monkeypatch.setattr(settings, "crash_long_trend_sessions", 200)
    monkeypatch.setattr(settings, "crash_long_rsi_max", 5.0)
    monkeypatch.setattr(settings, "crash_long_exit_sma_sessions", 5)
    monkeypatch.setattr(settings, "crash_long_max_hold_sessions", 10)
    monkeypatch.setattr(settings, "crash_long_account_slices", 4.0)
    monkeypatch.setattr(settings, "crash_long_entry_window_minutes", 60.0)
    monkeypatch.setattr(settings, "enable_sel_short", True)
    monkeypatch.setattr(settings, "enable_sel_short_account_sizing", True)
    monkeypatch.setattr(settings, "enable_sel_short_ibkr_whatif", False)
    monkeypatch.setattr(settings, "enable_live_account", True)
    monkeypatch.setattr(settings, "sel_short_account_initial", 10_000.0)
    monkeypatch.setattr(settings, "dip_long_account_slices", 10)
    monkeypatch.setattr(dl, "_extend", lambda tks, through: None)
    return tmp_path


# ── the rule ─────────────────────────────────────────────────────────────────

def test_the_signal_is_a_close_under_the_trend_average_with_a_2_session_rsi_under_5(on):
    f = _frame(_crash_closes())
    st = cl.stats_at(f, date(2026, 10, 7))
    assert st["close"] == 380.0 and st["close"] < st["sma_trend"] and st["rsi2"] < 5
    assert cl.qualifies(st)
    assert not cl.qualifies(dict(st, close=st["sma_trend"] + 1))             # above the trend average: no
    assert not cl.qualifies(dict(st, rsi2=6.0))                              # not oversold enough: no
    assert cl.stats_at(f.iloc[:150], date(2026, 10, 7)) is None              # under 200 sessions: no decision


def test_the_decision_reads_the_previous_close_and_is_kept_once(on, monkeypatch):
    f = _frame(_crash_closes())
    monkeypatch.setattr(dl, "_daily", lambda tk: f)
    sig = cl.signal_for(DAY)
    assert sig["signal"] and sig["signal_session"] == "2026-10-07" and not sig["stale"]
    assert cl.signals_path(DAY).exists()
    monkeypatch.setattr(dl, "_daily", lambda tk: _frame(list(np.linspace(300, 500, 240))))
    assert cl.signal_for(DAY)["signal"]                                      # the kept decision, not recomputed


def test_a_store_missing_the_previous_close_is_retried_not_kept(on, monkeypatch):
    f = _frame(_crash_closes(), end=date(2026, 10, 6))                       # D-1 (10-07) missing
    monkeypatch.setattr(dl, "_daily", lambda tk: f)
    sig = cl.signal_for(DAY)
    assert sig["stale"] and not sig["signal"] and not cl.signals_path(DAY).exists()


# ── the exit ─────────────────────────────────────────────────────────────────

def test_the_long_leaves_after_the_first_close_above_its_5_session_average(on, monkeypatch):
    closes = list(np.linspace(300, 500, 230)) + [400.0, 380.0, 370.0, 360.0, 395.0]
    f = _frame(closes)
    monkeypatch.setattr(dl, "_daily", lambda tk: f)
    t = {"ticker": "TQQQ", "crash_decision_symbol": "QQQ", "crash_entry_day": "2026-10-02"}
    reason, info = cl.exit_check(t, DAY)
    assert reason == "crash_rebound" and info["rebound_session"] == "2026-10-07" and info["decision_symbol"] == "QQQ"


def test_the_exit_reads_the_decision_symbols_closes_never_the_traded_funds(on, monkeypatch):
    rebound = _frame(list(np.linspace(300, 500, 230)) + [400.0, 380.0, 370.0, 360.0, 395.0])
    falling = _frame(list(np.linspace(300, 500, 230)) + [400.0, 380.0, 370.0, 360.0, 350.0])
    t = {"ticker": "TQQQ", "crash_decision_symbol": "QQQ", "crash_entry_day": "2026-10-02"}
    monkeypatch.setattr(dl, "_daily", lambda tk: {"QQQ": rebound, "TQQQ": falling}[tk])
    assert cl.exit_check(t, DAY)[0] == "crash_rebound"                       # QQQ rebounded: sell
    monkeypatch.setattr(dl, "_daily", lambda tk: {"QQQ": falling, "TQQQ": rebound}[tk])
    assert cl.exit_check(t, DAY)[0] is None                                 # only TQQQ rebounded: hold


def test_the_long_leaves_after_10_sessions_without_a_rebound(on, monkeypatch):
    closes = list(np.linspace(300, 500, 220)) + list(np.linspace(400, 300, 20))      # falling every day
    f = _frame(closes)
    monkeypatch.setattr(dl, "_daily", lambda tk: f)
    entry = _sessions(date(2026, 10, 7), 10)[0]                                       # 10 sessions through D-1
    reason, info = cl.exit_check({"ticker": "TQQQ", "crash_entry_day": entry.isoformat()}, DAY)
    assert reason == "crash_time" and info["held"] == 10
    later = _sessions(date(2026, 10, 7), 9)[0]
    assert cl.exit_check({"ticker": "TQQQ", "crash_entry_day": later.isoformat()}, DAY)[0] is None


# ── the live account ─────────────────────────────────────────────────────────

def dip(tk, n=10, px=100.0, at="2026-10-07T13:31:00+00:00"):
    return {"ticker": tk, "entry_mechanism": "dip_long", "action": "BUY", "type": "STOCK", "status": "OPEN",
            "dip_account_shares": n, "entry_price": px, "current_price": px, "return_pct": 0.0,
            "entry_datetime": at, "entry_date": at[:10], "entry_session": "rth", "position_size_multiplier": 1.0,
            "recommendation_id": f"dip-{tk}", "current_price_datetime": NOW.isoformat()}


def crash(n=6, px=400.0, ret=0.0, status="OPEN", rate=0.75):
    return {"ticker": "TQQQ", "entry_mechanism": "crash_long", "action": "BUY", "type": "STOCK", "status": status,
            "crash_decision_symbol": "QQQ", "crash_margin_rate": rate,
            "crash_account_shares": n, "entry_price": px, "current_price": px, "return_pct": ret,
            "entry_datetime": "2026-10-07T13:30:30+00:00", "entry_date": "2026-10-07", "entry_session": "rth",
            "position_size_multiplier": 1.0, "recommendation_id": "crash-1", "crash_entry_day": "2026-10-07",
            "current_price_datetime": NOW.isoformat()}


def test_the_crash_long_is_in_the_accounts_equity_side_and_margin_with_its_own_slot(on):
    st = la.state([crash(n=6, px=400.0), dip("D1")], NOW)
    assert st["long_gross"] == pytest.approx(2_400.0 + 1_000.0)
    assert st["init_req"] == pytest.approx(0.75 * 2_400.0 + 0.5 * 1_000.0)  # the fund's 75%; the dip's Reg T
    assert st["maint"] == pytest.approx(0.75 * 2_400.0 + 0.25 * 1_000.0)
    assert st["crash_open"] == 1 and st["long_open"] == 1                     # the dip book's slots: dips only
    assert la.state([crash(ret=10.0, status="CLOSED")], NOW)["realized"] == pytest.approx(240.0)


def test_a_quarter_of_the_equity_one_long_at_a_time(on):
    n, why, _ = la.size_crash([], 400.0, NOW)
    assert (n, why) == (6, "ok")                                              # $2,500 / $400
    assert la.size_crash([crash()], 400.0, NOW)[:2] == (0, "no_slot")


def test_the_traded_funds_margin_rate_sizes_and_margins_it(on, monkeypatch):
    st = {"equity": 10_000.0, "init_req": 9_100.0, "long_gross": 0.0, "crash_open": 0}
    assert la.size_crash([], 400.0, NOW, st=dict(st))[:2] == (2, "ok")       # $900 of room at 75%: 2 x $400
    lost = crash(n=30, px=400.0, ret=-30.0)                                  # equity $6,400, the long's value $12,000
    assert la.margin_call([lost], NOW)[0]                                    # 75% maintenance ($9,000): called
    assert not la.margin_call([dict(lost, crash_margin_rate=0.0)], NOW)[0]   # Reg T's 25% ($3,000): not
    monkeypatch.setattr(settings, "crash_long_margin_rate", 0.0)
    assert la.size_crash([], 400.0, NOW, st=dict(st))[:2] == (4, "ok")       # at Reg T's 50%: 4


def test_the_dip_book_gives_way_latest_first_whole_only_as_many_as_needed(on):
    dips = [dip(f"D{k}", at=f"2026-10-07T13:3{k}:00+00:00") for k in range(10)]   # $10,000: the long side full
    assert la.size_crash(dips, 400.0, NOW)[:2] == (0, "account_full")
    plan, want, _ = la.crash_give_way_plan(dips, 400.0, NOW)
    assert want == 6 and [t["ticker"] for t in plan] == ["D9", "D8", "D7"]   # $3,000 freed buys 6 x $400
    assert la.crash_give_way_plan(dips[:5], 400.0, NOW)[0] == []              # room already there: nothing sold


def test_a_margin_call_counts_the_crash_long(on):
    called, st = la.margin_call([crash(n=100, px=400.0, ret=-80.0)], NOW)     # equity $10,000 - $32,000 < maint
    assert called and st["crash_open"] == 1


# ── the ledger steps ─────────────────────────────────────────────────────────

@pytest.fixture
def entry(monkeypatch, on):
    sig = {"day": DAY.isoformat(), "symbol": "QQQ", "signal_session": "2026-10-07", "stale": False, "signal": True,
           "close": 380.0, "sma_trend": 450.0, "rsi2": 1.5}
    monkeypatch.setattr(cl, "signal_for", lambda d: sig)
    monkeypatch.setattr(tracker, "_fetch_price", lambda t: {"TQQQ": 50.0}.get(t))    # only the traded fund's
    monkeypatch.setattr(tracker, "_reference_close", lambda t: None)
    monkeypatch.setattr(tracker, "_execution_iso", lambda: OPEN_TICK.isoformat())
    return sig


def test_the_open_buys_tqqq_on_qqqs_decision_at_the_accounts_size_once(entry):
    assert tracker.record_crash_long_trades(run_id="r1", now=OPEN_TICK) == 1
    (t,) = [x for x in tracker._load_trades() if x.get("entry_mechanism") == "crash_long"]
    assert t["ticker"] == "TQQQ" and t["action"] == "BUY" and t["crash_account_shares"] == 50   # $2,500 / $50
    assert t["entry_price"] == 50.0 and t["crash_decision_symbol"] == "QQQ" and t["crash_margin_rate"] == 0.75
    assert t["crash_close"] == 380.0                                          # QQQ's close, the decision's
    assert t["crash_signal_day"] == "2026-10-07" and t["crash_entry_day"] == DAY.isoformat()
    assert cl.settled(DAY) == {"QQQ": "opened"}
    assert cl._read_jsonl(cl.entries_path(DAY))[0]["trade_symbol"] == "TQQQ"
    assert tracker.record_crash_long_trades(now=OPEN_TICK + timedelta(minutes=30)) == 0
    late = datetime(2026, 10, 8, 10, 31, tzinfo=ET).astimezone(timezone.utc)
    assert not cl.in_entry_window(late)


def test_a_signal_while_holding_is_not_stacked(entry):
    tracker._save_trades([crash()])
    assert tracker.record_crash_long_trades(now=OPEN_TICK) == 0
    assert cl.settled(DAY) == {"QQQ": "holding"}


def test_the_entry_sells_the_dip_books_latest_longs_for_room(entry):
    tracker._save_trades([dip(f"D{k}", at=f"2026-10-07T13:3{k}:00+00:00") for k in range(10)])
    assert tracker.record_crash_long_trades(now=OPEN_TICK) == 1
    tr = tracker._load_trades()
    gone = sorted(x["ticker"] for x in tr if x.get("exit_reason") == "dip_gave_way")
    assert gone == ["D7", "D8", "D9"]
    (c,) = [x for x in tr if x.get("entry_mechanism") == "crash_long"]
    assert c["crash_account_shares"] == 49                                   # $9,998 after the sales' costs / 4 / $50
    assert sorted(c["crash_gave_way"]) == ["dip-D7", "dip-D8", "dip-D9"]


def test_another_book_holding_the_traded_fund_settles_it_held(entry):
    held = dict(dip("TQQQ"), entry_mechanism="sel_short", action="SELL", recommendation_id="short-TQQQ")
    held.pop("dip_account_shares")
    tracker._save_trades([held])
    assert tracker.record_crash_long_trades(now=OPEN_TICK) == 0
    assert cl.settled(DAY) == {"QQQ": "held"}


def test_the_open_sells_the_long_whose_exit_is_due(on, monkeypatch):
    tracker._save_trades([crash()])
    monkeypatch.setattr(cl, "exit_check", lambda t, d: ("crash_rebound", {"held": 1, "close": 410.0,
                                                                           "sma_exit": 400.0,
                                                                           "rebound_session": "2026-10-07"}))
    monkeypatch.setattr(tracker, "_fetch_price", lambda t: 412.0)
    monkeypatch.setattr(tracker, "_reference_close", lambda t: None)
    monkeypatch.setattr(tracker, "_execution_iso", lambda: OPEN_TICK.isoformat())
    assert tracker.monitor_crash_long_positions(now=OPEN_TICK) == 1
    t = tracker._load_trades()[0]
    assert t["status"] == "CLOSED" and t["exit_reason"] == "crash_rebound" and t["exit_price"] == 412.0
    assert cl._read_jsonl(cl.exits_path(DAY))[0]["reason"] == "crash_rebound"


# ── the broker and the simulator ─────────────────────────────────────────────

def test_the_book_is_live_and_the_broker_orders_its_share_count():
    from src.broker import reconcile
    assert books.is_live(crash()) and "crash_long" in books.LIVE_MECHANISMS
    assert reconcile._entry_qty(crash(n=7), 400.0, 1e6, 1.0) == 7


def test_a_split_of_the_traded_fund_rescales_its_shares_never_the_decision_symbols_closes():
    assert "crash_account_shares" in tracker.SPLIT_SHARE_FIELDS and "entry_price" in tracker.SPLIT_PRICE_FIELDS
    assert "crash_close" not in tracker.SPLIT_PRICE_FIELDS and "crash_sma_trend" not in tracker.SPLIT_PRICE_FIELDS


def test_the_simulator_runs_the_crash_book_as_production_does(on):
    from src.backtest import mimic as M
    names = {s.name: s for s in M.live_strategies(M.Mimic())}
    c = names["crash"]
    assert c.side == 1 and c.slices == 4.0 and c.max_open == 1 and c.quality == 0.2
    assert c.house_init == 0.75 and c.house_maint == 0.75                     # the fund's margin rate
    assert c.quality > names["dip"].quality and names["dip"].preemptible
    assert "crash" not in {s.name for s in M.live_strategies(M.Mimic.off())}


def test_the_adapters_longs_are_productions_decisions(on, monkeypatch):
    from src.backtest.strategies import crash as CR
    closes = (list(np.linspace(300, 500, 230)) + [400.0, 380.0, 385.0, 390.0, 395.0, 420.0, 430.0, 380.0, 360.0,
                                                    350.0, 355.0, 362.0, 370.0, 380.0, 390.0, 400.0])
    f = _frame(closes)
    got = CR.trades(f.index[200], f.index[-1], daily=f, sym="QQQ")
    want = [f.index[j] for j in range(201, len(f))
            if cl.qualifies(cl.stats_at(f.iloc[:j], f.index[j - 1].date()))]
    assert [t["entry"] for t in got] == [d for d in want if all(not (t["entry"] < d < (t["exit"] or d))
                                                                 for t in got)]
    assert got and all(t["exit"] is None or t["exit"] > t["entry"] for t in got)


def test_the_adapter_holds_the_traded_fund_on_the_decision_symbols_longs(on, monkeypatch):
    from src.backtest.strategies import crash as CR
    from src.backtest.strategies import dm as DM
    days = [pd.Timestamp(x) for x in _sessions(date(2026, 10, 7), 30)]
    a, b = days[-10], days[-6]
    seen = []

    def trades(lo, hi, daily=None, sym=None):
        seen.append(sym)
        return [{"signal": days[-11], "entry": a, "exit": b}]
    px = np.arange(30, dtype=float) + 50.0
    tq = pd.DataFrame({"Open": px, "High": px + 1, "Low": px - 1, "Close": px + 0.5, "Volume": np.full(30, 1e7)},
                      index=pd.DatetimeIndex(days))
    rows, idx = [], []
    for d, o in zip(days, px):                                              # 13 regular-hours bars a session
        for k in range(13):
            idx.append(pd.Timestamp(datetime(d.year, d.month, d.day, 9, 30, tzinfo=ET) + timedelta(minutes=30 * k))
                       .tz_convert("UTC").tz_localize(None))
            rows.append((o, o + 1, o - 1, o + 0.5, 1e6))
    bars = pd.DataFrame(rows, columns=["Open", "High", "Low", "Close", "Volume"], index=pd.DatetimeIndex(idx))
    monkeypatch.setattr(CR, "trades", trades)
    monkeypatch.setattr(DM, "daily", lambda t: tq if t == "TQQQ" else None)
    monkeypatch.setattr(DM, "bars_30m", lambda t: bars if t == "TQQQ" else None)
    (p,) = CR.pieces(days[0], days[-1])
    assert seen == ["QQQ"]                                                   # decided on QQQ
    assert p["tkn"] == "TQQQ" and p["e"] == tq.loc[a, "Open"] and p["x"] == tq.loc[b, "Open"]
