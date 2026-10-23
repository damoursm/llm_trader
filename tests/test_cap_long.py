"""THE CAPITULATION BOOK (PREREG165's CBg6_20; user 2026-10-20: "Deploy all three."; `src/signals/cap_long.py`).

What must hold: the breadth at a session's close is the oversold share (under the 200-session average with a 2-session
RSI under 10) of the 100 most-traded common stocks / ADRs with 220 bars in their run, on closes in that session's
split terms, reading no later session and never guessing over a missing one; on a session whose breadth at the close
before is at least 20% every $2B+ dip under its own trend is the dip book's limit order — 1/6 of the live account at
its limit, at most 3 open with the working orders, the deepest dip first, filled at the open or at the end of the first
bar reaching the limit within the room, lapsing after its window, sold at the open after the first close above its
3-session average or after 5 sessions; its capital gives way FIRST (the shorts', the crash book's and a dip fill's
room, de-leveraging), a margin call closes its longs and cancels its orders; the broker rests its orders and orders its
share count; the simulator builds the same book with PREREG165's account strategy.
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
from src.signals import cap_long as cl
from src.signals import dip_long as dl
from src.signals import sel_short

ET = ZoneInfo("America/New_York")
DAY = date(2026, 10, 8)                        # a Thursday session; D-1 = 2026-10-07
PREV = date(2026, 10, 7)


def at(d, h, m, s=20):
    return datetime(d.year, d.month, d.day, h, m, s, tzinfo=ET).astimezone(timezone.utc)


OPEN_TICK = at(DAY, 9, 30)
NOW = OPEN_TICK


def _sessions(end: date, n: int):
    out, d = [], end
    while len(out) < n:
        if dl.is_session(d):
            out.append(d)
        d -= timedelta(days=1)
    return out[::-1]


# ── the breadth ──────────────────────────────────────────────────────────────

def _uptrend(n):
    return list(np.linspace(50.0, 100.0, n))


def _crash(n):
    """Up for most of the window, then a fall under the 200-session average ending in two sharp down days."""
    return list(np.linspace(50.0, 100.0, n - 25)) + list(np.linspace(98.0, 70.0, 23)) + [65.0, 60.0]


@pytest.fixture
def market(monkeypatch, tmp_path):
    """A whole market of synthetic names over the breadth's sessions through PREV: ``series`` = {sym: (closes, volume)}
    (closes aligned on the last sessions; None = no bar that session)."""
    monkeypatch.setattr(settings, "enable_cap_long", True)
    monkeypatch.setattr(settings, "cap_long_dir", str(tmp_path / "cap_long"))
    monkeypatch.setattr(settings, "cap_long_breadth_top", 4)
    monkeypatch.setattr(settings, "cap_long_breadth_min", 0.20)
    days = _sessions(PREV, cl.BREADTH_SESSIONS + 5)
    state = {"series": {}, "types": {}, "splits": {}, "asked": [], "missing": set()}

    def grouped(d, fetch=True):
        state["asked"].append(d)
        if d in state["missing"]:
            return None
        rows = []
        for sym, (closes, vol) in state["series"].items():
            k = len(days) - 1 - days.index(d) if d in days else None
            if k is None or k >= len(closes):
                continue
            c = closes[len(closes) - 1 - k]
            if c is None:
                continue
            rows.append({"sym": sym, "o": c, "h": c * 1.01, "l": c * 0.99, "c": c, "v": vol})
        return pd.DataFrame(rows, columns=["sym", "o", "h", "l", "c", "v"])
    from src.signals import sel_pressure
    monkeypatch.setattr(sel_pressure, "grouped_ohlc", grouped)
    from src.data import company_names
    monkeypatch.setattr(company_names, "security_type", lambda s: state["types"].get(s, "CS"))
    from src.data import intraday_store
    monkeypatch.setattr(intraday_store, "split_rows", lambda: state["splits"])
    state["days"] = days
    return state


def test_the_breadth_is_the_oversold_share_of_the_most_traded_stocks(market):
    n = cl.BREADTH_SESSIONS
    market["series"].update({
        "AAA": (_crash(n), 9e8), "BBB": (_crash(n), 8e8), "CCC": (_uptrend(n), 7e7), "DDD": (_uptrend(n), 6e7),
        "EEE": (_crash(n), 1e6),                 # oversold, but not among the 4 most traded: never counted
        "FND": (_crash(n), 9e9)})                # the most traded of all, but a fund: never a stock
    market["types"]["FND"] = "ETF"
    out = cl.breadth_at(PREV)
    assert out["error"] is None and [x["ticker"] for x in out["top"]] == ["AAA", "BBB", "CCC", "DDD"]   # by close x volume
    assert sorted(out["oversold"]) == ["AAA", "BBB"] and out["breadth"] == pytest.approx(0.5)
    assert max(market["asked"]) == PREV                                      # no session after the close read


def test_a_name_needs_220_bars_in_its_run_and_a_long_gap_starts_a_new_run(market):
    n = cl.BREADTH_SESSIONS
    gapped = _crash(n)
    for k in range(n - 150, n - 120):                                        # 30 sessions without a bar (> 30 days)
        gapped[k] = None
    market["series"].update({"AAA": (_crash(n), 9e7), "GAP": (gapped, 8.5e7), "YNG": (_crash(200), 8e7),
                             "CCC": (_uptrend(n), 7e7), "DDD": (_uptrend(n), 6e7), "EEE": (_uptrend(n), 5e7)})
    out = cl.breadth_at(PREV)
    assert sorted(x["ticker"] for x in out["top"]) == ["AAA", "CCC", "DDD", "EEE"]   # GAP's run, YNG: under 220 bars
    assert out["breadth"] == pytest.approx(0.25)


def test_a_split_inside_the_window_never_reads_as_a_fall(market):
    n = cl.BREADTH_SESSIONS
    raw = _uptrend(n)
    raw[-1] = raw[-1] / 2.0                                                  # a 2-for-1 split executed at the close's session
    market["series"].update({"SPL": (raw, 9e7), "CCC": (_uptrend(n), 7e7), "DDD": (_uptrend(n), 6e7),
                             "EEE": (_uptrend(n), 5e7)})
    split_day = market["days"][-1]
    assert split_day == PREV
    market["splits"]["SPL"] = [(split_day, 1.0, 2.0), (PREV + timedelta(days=30), 1.0, 3.0)]   # a later one: unread
    out = cl.breadth_at(PREV)
    spl = next(x for x in out["top"] if x["ticker"] == "SPL")
    assert not spl["oversold"] and out["breadth"] == 0.0
    market["splits"].clear()                                                 # the record missing: the fall is read
    assert cl.breadth_at(PREV)["breadth"] == pytest.approx(0.25)


def test_a_session_without_whole_market_bars_is_an_error_not_a_guess(market):
    n = cl.BREADTH_SESSIONS
    market["series"].update({k: (_uptrend(n), v) for k, v in (("A1", 9e7), ("A2", 8e7), ("A3", 7e7), ("A4", 6e7))})
    market["missing"].add(market["days"][-40])
    out = cl.breadth_at(PREV)
    assert out["breadth"] is None and "no whole-market bars" in out["error"]
    assert cl.compute_signals(DAY) is None and not cl.signals_path(DAY).exists()    # retried, nothing kept
    assert not cl.breadth_path(PREV).exists()


def test_the_breadth_is_kept_once_computed(market):
    n = cl.BREADTH_SESSIONS
    market["series"].update({k: (_crash(n), v) for k, v in (("A1", 9e7), ("A2", 8e7), ("A3", 7e7), ("A4", 6e7))})
    assert cl.breadth_for(DAY)["breadth"] == 1.0 and cl.breadth_path(PREV).exists()
    market["series"].clear()
    assert cl.breadth_for(DAY)["breadth"] == 1.0                             # the kept value, not recomputed


# ── the rule ─────────────────────────────────────────────────────────────────

def _st(close, sma, rsi, dv=3e9, atr=2.0):
    return {"session": PREV, "close": close, "sma_trend": sma, "sma_exit": close, "rsi2": rsi, "dv20": dv,
            "bars": 900, "atr14": atr}


@pytest.fixture
def rule(monkeypatch, tmp_path):
    monkeypatch.setattr(settings, "enable_cap_long", True)
    monkeypatch.setattr(settings, "cap_long_dir", str(tmp_path / "cap_long"))
    stats = {"DEEP": _st(80.0, 100.0, 2.0), "SOFT": _st(90.0, 100.0, 8.0), "ABOV": _st(110.0, 100.0, 3.0),
             "SMAL": _st(80.0, 100.0, 1.0, dv=1.5e9), "WARM": _st(80.0, 100.0, 12.0)}
    monkeypatch.setattr(dl, "candidates", lambda d, min_dv=None: (sorted(stats), "test"))
    monkeypatch.setattr(cl, "name_stats", lambda tk, through: stats.get(tk))
    breadth = {"breadth": 0.25, "oversold": ["DEEP"], "error": None}
    monkeypatch.setattr(cl, "breadth_for", lambda d, fetch=True: breadth)
    return breadth


def test_a_capitulation_session_orders_every_big_dip_under_its_trend_deepest_first(rule):
    day = cl.compute_signals(DAY)
    assert day["gate"] and day["breadth"] == 0.25 and day["signal_session"] == "2026-10-07"
    assert [s["ticker"] for s in day["signals"]] == ["DEEP", "SOFT"]         # above the trend / under $2B / RSI2 >= 10: no
    assert day["signals"][0]["limit"] == cl.limit_price(80.0, 2.0) == 79.0
    assert cl.signals_path(DAY).exists()


def test_under_the_gate_nothing_is_ordered(rule):
    rule["breadth"] = 0.19
    day = cl.compute_signals(DAY)
    assert not day["gate"] and day["signals"] == [] and cl.signals_path(DAY).exists()


def _daily(closes, end=PREV):
    days = _sessions(end, len(closes))
    c = np.asarray(closes, float)
    return pd.DataFrame({"o": c, "h": c * 1.01, "l": c * 0.99, "c": c, "v": np.full(len(c), 1e7)},
                        index=pd.DatetimeIndex([pd.Timestamp(x) for x in days]))


def test_the_long_leaves_after_the_first_close_above_its_3_session_average_or_5_sessions(rule, monkeypatch):
    monkeypatch.setattr(dl, "_extend", lambda tks, through: None)
    up = _daily(list(np.linspace(100, 80, 230)) + [78.0, 76.0, 80.0])
    monkeypatch.setattr(dl, "_daily", lambda tk, through=None: up)
    entry = _sessions(PREV, 3)[0]
    reason, info = cl.exit_check({"ticker": "DEEP", "cap_entry_day": entry.isoformat()}, DAY)
    assert reason == "cap_rebound" and info["rebound_session"] == "2026-10-07"
    down = _daily(list(np.linspace(100, 60, 240)))
    monkeypatch.setattr(dl, "_daily", lambda tk, through=None: down)
    e5 = _sessions(PREV, 5)[0]
    assert cl.exit_check({"ticker": "DEEP", "cap_entry_day": e5.isoformat()}, DAY)[0] == "cap_time"
    e4 = _sessions(PREV, 4)[0]
    assert cl.exit_check({"ticker": "DEEP", "cap_entry_day": e4.isoformat()}, DAY)[0] is None


# ── the live account ─────────────────────────────────────────────────────────

@pytest.fixture
def acct(monkeypatch):
    for k, v in {"enable_sel_short": True, "enable_sel_short_account_sizing": True,
                 "enable_sel_short_ibkr_whatif": False, "enable_live_account": True,
                 "sel_short_account_initial": 10_000.0, "dip_long_account_slices": 2,
                 "cap_long_account_slices": 6.0, "cap_long_max_open": 3, "crash_long_account_slices": 3.0,
                 "crash_long_margin_rate": 0.75, "live_account_delever_trigger": 0.7,
                 "live_account_delever_target": 0.5, "sel_short_account_house_maint": 2.0,
                 "sel_short_account_house_init": 2.86}.items():
        monkeypatch.setattr(settings, k, v)


def long_(mech, tk, n=10, px=100.0, at_iso="2026-10-07T13:31:00+00:00", ret=0.0):
    pre = {"dip_long": "dip", "cap_long": "cap"}[mech]
    return {"ticker": tk, "entry_mechanism": mech, "action": "BUY", "type": "STOCK", "status": "OPEN",
            f"{pre}_account_shares": n, "entry_price": px, "current_price": px, "return_pct": ret,
            "entry_datetime": at_iso, "entry_date": at_iso[:10], "entry_session": "rth", "position_size_multiplier": 1.0,
            "recommendation_id": f"{pre}-{tk}", "current_price_datetime": NOW.isoformat()}


def cap(tk, **kw):
    return long_("cap_long", tk, **kw)


def dip(tk, **kw):
    return long_("dip_long", tk, **kw)


def short(tk, n, px=10.0, at_iso="2026-10-08T14:00:00+00:00"):
    return {"ticker": tk, "entry_mechanism": "sel_short", "sel_arm": "vol", "action": "SELL", "type": "STOCK",
            "status": "OPEN", "sel_account_shares": n, "entry_price": px, "current_price": px, "return_pct": 0.0,
            "sel_house_maint": 2.0, "sel_house_init": 2.86, "entry_datetime": at_iso, "entry_date": at_iso[:10],
            "entry_session": "rth", "position_size_multiplier": 1.0, "recommendation_id": f"sel-{tk}",
            "current_price_datetime": at_iso}


def test_the_capitulation_longs_count_in_the_account_with_their_own_slots(acct):
    st = la.state([cap("C1"), dip("D1")], NOW)
    assert st["long_gross"] == pytest.approx(2_000.0) and st["cap_gross"] == pytest.approx(1_000.0)
    assert st["init_req"] == pytest.approx(1_000.0) and st["maint"] == pytest.approx(500.0)   # Reg T on both
    assert st["cap_open"] == 1 and st["long_open"] == 1                                     # separate slots
    assert la.state([dict(cap("C1", ret=10.0), status="CLOSED")], NOW)["realized"] == pytest.approx(100.0)


def test_each_order_is_a_sixth_at_most_three_open_the_working_orders_counted(acct):
    n, why, _ = la.size_cap([], 100.0, NOW)
    assert (n, why) == (16, "ok")                                            # $10,000 / 6 / $100
    assert la.size_cap([cap("C1"), cap("C2")], 100.0, NOW, working=1)[:2] == (0, "no_slot")
    assert la.size_cap([cap("C1")], 100.0, NOW, working=1)[0] > 0


def test_the_long_books_give_way_to_a_short_capitulation_first_then_dip_latest_first(acct):
    held = [dip("D1", n=40, at_iso="2026-10-07T13:31:00+00:00"), dip("D2", n=40, at_iso="2026-10-07T13:35:00+00:00"),
            cap("C1", n=10, at_iso="2026-10-07T13:32:00+00:00"), cap("C2", n=10, at_iso="2026-10-07T13:33:00+00:00")]
    # $10,000: $8,000 dip + $2,000 cap at 50% = $5,000 initial; a short of 1/20 at $10 needs 50 x $28.6 = $1,430
    plan, want, _ = la.give_way_plan(held + [short("HOT", 150)], 10.0, None, NOW, slices=20)
    assert want == 50 and [t["ticker"] for t in plan][:2] == ["C2", "C1"]   # the capitulation longs first, latest first
    plan2, _, _ = la.give_way_plan(held + [short("HOT", 150)], 10.0, None, NOW, slices=1)
    order = [t["ticker"] for t in plan2]
    assert order[:2] == ["C2", "C1"] and set(order[2:]) <= {"D2", "D1"} and order[2:3] in (["D2"], [])


def test_the_crash_buy_sells_the_capitulation_longs_before_the_dip_longs(acct):
    held = [dip("D1", n=40), dip("D2", n=40, at_iso="2026-10-07T13:40:00+00:00"), cap("C1", n=20)]
    plan, want, _ = la.crash_give_way_plan(held, 50.0, NOW)
    assert want > 0 and [t["ticker"] for t in plan][0] == "C1"


def test_deleveraging_sells_the_capitulation_longs_first(acct):
    held = [dip("D1", n=50, at_iso="2026-10-07T13:40:00+00:00"), cap("C1", n=30), cap("C2", n=30,
            at_iso="2026-10-07T13:35:00+00:00"), short("HOT", 250, px=10.0)]
    st = la.state(held, NOW)
    assert st["maint"] > 0.7 * st["equity"]
    plan, _ = la.delever_plan(held, NOW)
    names = [t["ticker"] for t in plan]
    assert names[:2] == ["C2", "C1"]                                         # the capitulation book first, latest first


def test_a_dip_order_is_sized_as_if_the_capitulation_longs_were_sold_and_its_fill_sells_them(acct):
    held = [cap("C1", n=50, at_iso="2026-10-07T13:31:00+00:00"), cap("C2", n=40, at_iso="2026-10-07T13:32:00+00:00")]
    n, why, _ = la.size_long(held, 100.0, NOW)                               # $9,000 of capitulation longs held
    assert (n, why) == (50, "ok")                                            # the 1/2 slice as if they were sold
    assert la.fill_cap(held, 100.0, NOW)[0] == 10                            # the room as it stands: $1,000 of side
    assert la.fill_cap(held, 100.0, NOW, free_cap=True)[0] == 100            # as if freed (the resting size)
    plan, _ = la.fill_give_way_plan(held, 100.0, 50, NOW)
    assert [t["ticker"] for t in plan] == ["C2"]                             # the latest, whole: room for 50
    plan, _ = la.fill_give_way_plan(held, 100.0, 90, NOW)
    assert [t["ticker"] for t in plan] == ["C2", "C1"]
    assert la.fill_give_way_plan(held, 100.0, 5, NOW)[0] == []               # the room already lets it
    assert la.fill_give_way_plan([dip("D1")], 100.0, 50, NOW)[0] == []       # never a dip long


def test_a_margin_call_counts_the_capitulation_longs(acct):
    called, st = la.margin_call([cap("C1", n=300, px=100.0, ret=-50.0)], NOW)   # equity $10,000 - $15,000
    assert called and st["cap_open"] == 1


# ── the ledger steps ─────────────────────────────────────────────────────────

def bars(d, rows):
    idx = [(datetime(d.year, d.month, d.day, 9, 30, tzinfo=ET) + timedelta(minutes=30 * k)).astimezone(timezone.utc)
           .replace(tzinfo=None) for k in range(len(rows))]
    return pd.DataFrame(rows, columns=["Open", "High", "Low", "Close"], index=pd.DatetimeIndex(idx)).assign(Volume=1e6)


def _sig(tk, rsi, close, atr):
    return {"ticker": tk, "session": "2026-10-07", "close": close, "sma_trend": close * 1.2, "sma_exit": close * 0.96,
            "rsi2": rsi, "dv20": 3e9, "bars": 1400, "atr14": atr, "limit": cl.limit_price(close, atr)}


@pytest.fixture
def book(monkeypatch, tmp_path, acct):
    for k, v in {"enable_cap_long": True, "cap_long_dir": str(tmp_path / "cap_long"), "cap_long_limit_atr": 0.5,
                 "cap_long_limit_sessions": 3, "enable_dip_long": True, "dip_long_dir": str(tmp_path / "dip_long"),
                 "dip_long_entry_mode": "limit", "dip_long_limit_atr": 0.5, "dip_long_limit_sessions": 3,
                 "enable_sel_short_vol": True, "sel_short_traded_arms": "vol,etf,fall"}.items():
        monkeypatch.setattr(settings, k, v)
    monkeypatch.setattr(dl, "_extend", lambda tks, through: None)
    monkeypatch.setattr(tracker, "_reference_close", lambda t: None)
    day = {"day": DAY.isoformat(), "signal_session": "2026-10-07", "breadth": 0.31, "gate": True,
           "signals": [_sig("AAA", 2.0, 100.0, 4.0), _sig("BBB", 6.0, 50.0, 2.0)]}     # limits 98.00 / 49.00
    monkeypatch.setattr(cl, "signals_for", lambda d: day if d == DAY else {"signals": [], "gate": False})
    monkeypatch.setattr(dl, "signals_for", lambda d: {"signals": []})
    feed = {}
    monkeypatch.setattr(dl, "session_bars", lambda tk, d: feed.get((tk, d), bars(d, [])))
    return feed


def caps():
    return {t["ticker"]: t for t in tracker._load_trades() if t.get("entry_mechanism") == "cap_long"}


def orders_of(d=DAY):
    return {o["ticker"]: o for o in cl.load_orders(d)}


def test_the_open_places_every_order_deepest_first_at_a_sixth_and_fills_one_opening_under_its_limit(book):
    book[("AAA", DAY)] = bars(DAY, [(97.5, 99.0, 97.0, 98.5)])              # opened under 98.00: filled at the open
    book[("BBB", DAY)] = bars(DAY, [(51.0, 51.2, 50.5, 51.0)])
    assert tracker.record_cap_long_trades(now=OPEN_TICK) == 3               # 2 placed + 1 bought at the open
    o = orders_of()
    assert o["AAA"]["status"] == "filled" and o["BBB"]["status"] == "working"
    assert o["AAA"]["qty"] == int(10_000 / 6 // 98.0) and o["BBB"]["qty"] == int(10_000 / 6 // 49.0)
    assert o["AAA"]["window"] == ["2026-10-08", "2026-10-09", "2026-10-12"]
    t = caps()["AAA"]
    assert t["entry_price"] == 97.5 and t["cap_fill_kind"] == "open" and t["cap_account_shares"] == o["AAA"]["qty"]
    assert t["cap_breadth"] == 0.31 and t["cap_entry_day"] == "2026-10-08" and books.is_live(t)
    assert tracker.record_cap_long_trades(now=OPEN_TICK + timedelta(minutes=30)) == 0   # one placement a session


def test_a_touch_fills_at_the_bar_end_and_an_order_lapses_after_its_window(book):
    book[("AAA", DAY)] = bars(DAY, [(101.0, 101.5, 100.5, 101.0)])
    book[("BBB", DAY)] = bars(DAY, [(51.0, 51.2, 50.5, 51.0)])
    tracker.record_cap_long_trades(now=OPEN_TICK)
    book[("AAA", DAY)] = bars(DAY, [(101.0, 101.5, 97.8, 98.6), (98.6, 99.0, 98.2, 98.9)])
    assert tracker.monitor_cap_long_orders(now=at(DAY, 10, 0)) == 1
    t = caps()["AAA"]
    assert t["entry_price"] == 98.0 and t["cap_fill_kind"] == "touch" and t["entry_datetime"].startswith("2026-10-08T14:00")
    last = date(2026, 10, 12)
    tracker.monitor_cap_long_orders(now=at(last, 16, 1))
    assert orders_of()["BBB"]["status"] == "lapsed"


def test_without_todays_breadth_the_carried_orders_are_still_armed(book, monkeypatch):
    book[("AAA", DAY)] = bars(DAY, [(101.0, 101.5, 100.5, 101.0)])
    book[("BBB", DAY)] = bars(DAY, [(51.0, 51.2, 50.5, 51.0)])
    tracker.record_cap_long_trades(now=OPEN_TICK)
    nxt = date(2026, 10, 9)
    day = cl.signals_for(DAY)
    monkeypatch.setattr(cl, "signals_for", lambda d: day if d == DAY else None)      # the next day's breadth unknown
    assert tracker.record_cap_long_trades(now=at(nxt, 9, 30)) == 2                  # both carried orders armed
    assert all(nxt.isoformat() in (o.get("session_armed_at") or {}) for o in orders_of().values())


def test_another_books_name_is_neither_placed_nor_filled(book):
    tracker._save_trades([dip("AAA")])
    book[("AAA", DAY)] = bars(DAY, [(97.0, 99.0, 96.0, 98.5)])
    tracker.record_cap_long_trades(now=OPEN_TICK)
    assert orders_of()["AAA"]["status"] == "pending" and "AAA" not in caps()


def test_a_dip_fill_sells_the_capitulation_long_for_its_room(book, monkeypatch):
    tracker._save_trades([cap("C1", n=90, px=100.0)])                         # $9,000 of the $10,000 long side
    monkeypatch.setattr(tracker, "_execution_iso", lambda: at(DAY, 10, 0).isoformat())
    o = {"oid": "DIP1|2026-10-08", "ticker": "DIP1", "day": "2026-10-08", "signal_session": "2026-10-07", "rank": 1,
         "rsi2": 3.0, "close": 100.0, "atr14": 4.0, "limit": 98.0, "dv20": 3e9, "sma_trend": 90.0,
         "window": ["2026-10-08", "2026-10-09", "2026-10-12"], "status": "working", "qty": 50, "events": [],
         "session_armed_at": {"2026-10-08": OPEN_TICK.isoformat()}, "ibkr_qty": 50, "ibkr": None}
    dl.save_orders(DAY, [o])
    book[("DIP1", DAY)] = bars(DAY, [(101.0, 101.5, 97.8, 98.6)])
    assert tracker.monitor_dip_long_orders(now=at(DAY, 10, 0)) == 1
    tr = {t["ticker"]: t for t in tracker._load_trades()}
    assert tr["C1"]["status"] == "CLOSED" and tr["C1"]["exit_reason"] == "cap_gave_way"
    assert tr["DIP1"]["dip_account_shares"] == 50
    assert dl._read_jsonl(cl.exits_path(DAY))[0]["reason"] == "cap_gave_way"        # its own book's journal


def test_a_margin_call_closes_the_capitulation_longs_and_cancels_its_orders(book, monkeypatch):
    book[("AAA", DAY)] = bars(DAY, [(101.0, 101.5, 100.5, 101.0)])
    book[("BBB", DAY)] = bars(DAY, [(51.0, 51.2, 50.5, 51.0)])
    tracker.record_cap_long_trades(now=OPEN_TICK)
    tracker._save_trades(tracker._load_trades() + [cap("LOS", n=300, px=100.0, ret=-50.0)])
    monkeypatch.setattr(tracker, "_execution_iso", lambda: at(DAY, 10, 0).isoformat())
    assert tracker.monitor_live_account(now=at(DAY, 10, 0)) == 1
    t = caps()["LOS"]
    assert t["status"] == "CLOSED" and t["exit_reason"] == "cap_margin_call"
    assert {x["status"] for x in orders_of().values()} == {"pending"}


def test_deleveraging_closes_the_capitulation_longs_as_cap_delever(book, monkeypatch):
    tracker._save_trades([cap("C1", n=30), dip("D1", n=50, at_iso="2026-10-07T13:40:00+00:00"),
                          short("HOT", 300, px=10.0)])                         # maintenance $8,000 > 70% of $10,000
    monkeypatch.setattr(tracker, "_execution_iso", lambda: at(DAY, 10, 0).isoformat())
    assert tracker.monitor_live_account(now=at(DAY, 10, 0)) >= 1
    t = {x["ticker"]: x for x in tracker._load_trades()}
    assert t["C1"]["exit_reason"] == "cap_delever"


def test_the_open_sells_the_long_whose_exit_is_due(book, monkeypatch):
    tracker._save_trades([cap("AAA")])
    monkeypatch.setattr(cl, "exit_check", lambda t, d: ("cap_rebound", {"held": 2, "close": 105.0, "sma_exit": 101.0,
                                                                        "rebound_session": "2026-10-07"}))
    monkeypatch.setattr(tracker, "_fetch_price", lambda t: 106.0)
    monkeypatch.setattr(tracker, "_execution_iso", lambda: OPEN_TICK.isoformat())
    assert tracker.monitor_cap_long_positions(now=OPEN_TICK) == 1
    t = caps()["AAA"]
    assert t["status"] == "CLOSED" and t["exit_reason"] == "cap_rebound" and t["exit_price"] == 106.0


def test_the_give_way_reasons_name_the_longs_own_book():
    assert tracker._long_give_way_reason(cap("C1"), "gave_way") == ("cap_gave_way", "cap_long")
    assert tracker._long_give_way_reason(dip("D1"), "delever") == ("dip_delever", "dip_long")


# ── the broker ───────────────────────────────────────────────────────────────

def test_the_broker_orders_its_share_count_rests_its_orders_and_books_an_open_fill_at_ibkrs_price(book, monkeypatch):
    from src.broker import reconcile as rec
    from src.broker.reconcile import _book_dip_open_fill
    assert rec._entry_qty(cap("C1", n=7), 100.0, 1e6, 1.0) == 7
    t = dict(cap("C1"), cap_fill_kind="open", broker_fill_price=97.62, broker_fill_qty=10, entry_price=97.5)
    _book_dip_open_fill(t)
    assert t["entry_price"] == 97.62 and t["cap_open_print"] == 97.5
    book[("AAA", DAY)] = bars(DAY, [(101.0, 101.5, 100.5, 101.0)])
    book[("BBB", DAY)] = bars(DAY, [(51.0, 51.2, 50.5, 51.0)])
    tracker.record_cap_long_trades(now=OPEN_TICK)
    from tests.test_dip_limit import _broker
    store = {"trades": tracker._load_trades()}
    r, b = _broker(monkeypatch, store)
    r.sync(broker=b)
    rest = [(o.ticker, o.side, o.quantity, o.limit_price, o.rest, o.client_ref[:5]) for o in b.orders]
    assert ("AAA", "BUY", orders_of()["AAA"]["qty"], 98.0, True, "capw-") in rest
    assert orders_of()["AAA"]["ibkr"]["state"] == "working"
    assert orders_of()["AAA"]["ibkr"]["ref"] in r._dip_resting_refs()      # owned: the orphan sweep keeps it


# ── the simulator ────────────────────────────────────────────────────────────

def test_the_simulator_runs_the_book_as_production_does(monkeypatch):
    from src.backtest import mimic as M
    monkeypatch.setattr(settings, "enable_cap_long", True)
    m = M.Mimic()
    names = {s.name: s for s in M.live_strategies(m)}
    c = names["cap"]
    assert c.side == 1 and c.slices == 6.0 and c.max_open == 3 and c.priority == 7 and c.quality == 0.1
    assert c.preemptible and c.yield_whole and c.yield_first == "latest" and c.delever_order == 0
    assert c.orders == "fill" and c.quality < names["dip"].quality < names["crash"].quality
    assert names["dip"].delever_order == 1 and M.live_rules(m).fills_make_room and M.live_rules(m).fill_room_exact
    assert M.BOOK["cap"] == books.CAP_LONG
    off = M.Mimic(**M.PRE167)
    assert "cap" not in {s.name for s in M.live_strategies(off)} and not M.live_rules(off).fills_make_room
    assert not M.live_rules(off).fill_room_exact
    monkeypatch.setattr(settings, "enable_cap_long", False)
    assert "cap" not in {s.name for s in M.live_strategies(M.Mimic())}


def test_the_mimics_defaults_follow_production(monkeypatch):
    from src.backtest import mimic as M
    for flag, field in (("enable_dip_long_filter", "dip_filter"), ("crash_long_pm_exit", "crash_pm_exit")):
        monkeypatch.setattr(settings, flag, True)
        assert getattr(M.Mimic(), field) is True
        monkeypatch.setattr(settings, flag, False)
        assert getattr(M.Mimic(), field) is False
    assert M.Mimic.off().cap is False and M.Mimic.off().dip_filter is False and M.Mimic.off().crash_pm_exit is False


def test_the_adapters_signals_are_the_rule_under_the_trend_on_the_gates_sessions(monkeypatch):
    from src.backtest.strategies import cap as CAP
    days = pd.DatetimeIndex([pd.Timestamp(x) for x in _sessions(PREV, 270)])
    c = np.r_[np.linspace(50, 100, 230), np.linspace(98, 70, 23), [65.0, 60.0], np.linspace(62, 80, 15)]
    d = pd.DataFrame({"o": c, "h": c * 1.01, "l": c * 0.99, "c": c, "v": np.full(len(c), 1e8)}, index=days)
    panel = {"XXX": d}
    monkeypatch.setattr(CAP.DIP, "eligible_names", lambda p: ["XXX"])
    gate, under = days[254], days[252]                                        # the 60.0 close / an earlier fall day
    monkeypatch.setattr(CAP, "breadth", lambda: pd.Series([0.3, 0.1], index=[gate, under]))
    S = CAP.signals_fn()(panel, CAP.params(), days[0], days[-1])
    assert list(pd.DatetimeIndex(S["sig"])) == [gate]                         # the oversold close on the gate day only
    monkeypatch.setattr(CAP, "breadth", lambda: pd.Series([0.1], index=[gate]))
    assert len(CAP.signals_fn()(panel, CAP.params(), days[0], days[-1])) == 0
    up = d.assign(c=np.linspace(50, 120, 270))                                # above its trend: never a signal
    monkeypatch.setattr(CAP, "breadth", lambda: pd.Series([0.9] * 270, index=days))
    assert len(CAP.signals_fn()({"XXX": up}, CAP.params(), days[0], days[-1])) == 0
