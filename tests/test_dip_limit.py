"""THE DIP BOOK'S LIMIT ENTRY (user directive 2026-10-09: "deploy the Order holds nothing, fills into the room left
for the 1/2 ATR limit in live for the dip buying model. Remember that the simulator needs to mimic production as much
as possible."; `src/signals/dip_long.py` "THE LIMIT ENTRY", `tracker._place_dip_orders` / `monitor_dip_long_orders`,
`reconcile._sync_dip_orders`).

What must hold — the engine's rule (`src/backtest/account.py` orders="fill", `strategies/dip.py` `order_pieces`):
the limit is the signal close minus 1/2 ATR14, floored to the cent; at the session's opening tick the orders are
placed deepest dip first, the quantity fixed then (the dip slice at the limit), working orders counting as dip slots,
nothing reserved; an order fills at the open when the session opens at/under it, else at the END of the first
30-minute bar whose low reaches it — at most what the live account's room allows then; a touch with no room buys
nothing and the order waits for the next session; carried orders are armed again at each window session's open; an
order lapses after its last session; a margin call cancels the working orders; the broker rests each working order at
IBKR at the room's size and links the fill.
"""
from __future__ import annotations

from datetime import date, datetime, timedelta, timezone
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import pytest

from config.settings import settings
from src.performance import tracker
from src.signals import dip_long as dl

ET = ZoneInfo("America/New_York")
DAY = date(2026, 10, 8)                                    # a Thursday session; its window: 10-08 .. 10-14
NEXT = date(2026, 10, 9)


def at(d, h, m, s=20):
    return datetime(d.year, d.month, d.day, h, m, s, tzinfo=ET).astimezone(timezone.utc)


OPEN_TICK = at(DAY, 9, 30)


def bars(d, rows):
    """30-minute bars of session ``d`` from 09:30: rows of (open, high, low, close)."""
    idx = [(datetime(d.year, d.month, d.day, 9, 30, tzinfo=ET) + timedelta(minutes=30 * k)).astimezone(timezone.utc)
           .replace(tzinfo=None) for k in range(len(rows))]
    return pd.DataFrame(rows, columns=["Open", "High", "Low", "Close"], index=pd.DatetimeIndex(idx)).assign(Volume=1e6)


def _sig(tk, rsi, close, atr):
    return {"ticker": tk, "session": "2026-10-07", "close": close, "sma_trend": close * 0.9, "sma_exit": close * 1.04,
            "rsi2": rsi, "dv20": 2e9, "bars": 1400, "atr14": atr, "limit": dl.limit_price(close, atr)}


@pytest.fixture
def lim(monkeypatch, tmp_path):
    for k, v in {"enable_dip_long": True, "dip_long_dir": str(tmp_path / "dip_long"), "dip_long_entry_mode": "limit",
                 "dip_long_limit_atr": 0.5, "dip_long_limit_sessions": 5, "dip_long_account_slices": 10,
                 "enable_sel_short": True, "enable_sel_short_vol": True, "enable_sel_short_account_sizing": True,
                 "enable_sel_short_ibkr_whatif": False, "enable_live_account": True,
                 "sel_short_traded_arms": "vol,etf,fall", "sel_short_account_initial": 10_000.0,
                 "sel_short_account_slices": 24, "sel_short_account_house_maint": 2.0,
                 "sel_short_account_house_init": 2.86}.items():
        monkeypatch.setattr(settings, k, v)
    monkeypatch.setattr(dl, "_extend", lambda tks, through: None)
    monkeypatch.setattr(tracker, "_reference_close", lambda t: None)
    day = {"day": DAY.isoformat(), "signal_session": "2026-10-07",
           "signals": [_sig("AAA", 4.0, 100.0, 4.0), _sig("BBB", 8.0, 50.0, 2.0)]}       # limits 98.00 / 49.00
    monkeypatch.setattr(dl, "signals_for", lambda d: day if d == DAY else {"signals": []})
    feed = {}
    monkeypatch.setattr(dl, "session_bars", lambda tk, d: feed.get((tk, d), bars(d, [])))
    return feed


def short(tk, n, px=10.0, at_iso="2026-10-08T14:00:00+00:00"):
    return {"ticker": tk, "entry_mechanism": "sel_short", "sel_arm": "vol", "action": "SELL", "type": "STOCK",
            "status": "OPEN", "sel_account_shares": n, "entry_price": px, "current_price": px, "return_pct": 0.0,
            "sel_house_maint": 2.0, "sel_house_init": 2.86, "entry_datetime": at_iso, "entry_date": at_iso[:10],
            "entry_session": "rth", "position_size_multiplier": 1.0, "recommendation_id": f"sel-{tk}",
            "current_price_datetime": at_iso}


def orders_of(d=DAY):
    return {o["ticker"]: o for o in dl.load_orders(d)}


def dips():
    return {t["ticker"]: t for t in tracker._load_trades() if t.get("entry_mechanism") == "dip_long"}


def test_the_limit_is_the_close_minus_half_the_atr_floored_to_the_cent(lim):
    assert dl.limit_price(100.0, 4.0) == 98.0
    assert dl.limit_price(100.007, 1.0) == 99.5
    assert dl.limit_price(100.0, float("nan")) is None and dl.limit_price(1.0, 4.0) is None
    d = pd.DataFrame({"h": np.r_[np.full(20, 11.0)], "l": np.r_[np.full(20, 9.0)], "c": np.full(20, 10.0)})
    assert dl.atr14(d) == pytest.approx(2.0)                       # the true range's 14-session mean


def test_the_open_places_the_orders_deepest_first_and_buys_nothing_above_the_limit(lim):
    lim[("AAA", DAY)] = bars(DAY, [(101.0, 101.5, 100.5, 101.0)])
    lim[("BBB", DAY)] = bars(DAY, [(51.0, 51.2, 50.5, 51.0)])
    assert tracker.record_dip_long_trades(now=OPEN_TICK) == 2                 # 2 placed, none bought
    o = orders_of()
    assert [o[t]["status"] for t in ("AAA", "BBB")] == ["working", "working"]
    assert o["AAA"]["limit"] == 98.0 and o["AAA"]["qty"] == int(1_000 // 98.0)      # the slice at the limit
    assert o["BBB"]["qty"] == int(1_000 // 49.0)
    assert o["AAA"]["window"] == ["2026-10-08", "2026-10-09", "2026-10-12", "2026-10-13", "2026-10-14"]
    assert dips() == {}
    # the same session's next tick never places twice
    assert tracker.record_dip_long_trades(now=OPEN_TICK + timedelta(minutes=30)) == 0   # nothing placed again
    assert [e["what"] for e in orders_of()["AAA"]["events"]].count("placed") == 1


def test_an_order_fills_at_the_open_when_the_session_opens_under_its_limit(lim):
    lim[("AAA", DAY)] = bars(DAY, [(97.5, 99.0, 97.0, 98.5)])
    lim[("BBB", DAY)] = bars(DAY, [(51.0, 51.2, 50.5, 51.0)])
    assert tracker.record_dip_long_trades(now=OPEN_TICK) == 3                 # 2 placed + 1 bought at the open
    t = dips()["AAA"]
    assert t["entry_price"] == 97.5 and t["dip_fill_kind"] == "open" and t["dip_account_shares"] == 10
    assert t["entry_datetime"].startswith("2026-10-08T13:30") and t["dip_entry_day"] == "2026-10-08"
    assert orders_of()["AAA"]["status"] == "filled" and orders_of()["BBB"]["status"] == "working"


def test_a_bar_that_reaches_the_limit_fills_at_its_end_within_the_room_left(lim):
    lim[("AAA", DAY)] = bars(DAY, [(101.0, 101.5, 100.5, 101.0)])
    lim[("BBB", DAY)] = bars(DAY, [(51.0, 51.2, 50.5, 51.0)])
    tracker.record_dip_long_trades(now=OPEN_TICK)                  # placed: AAA 10 sh at 98.00
    tracker._save_trades(tracker._load_trades() + [short("HOT", 340)])   # a short takes most of the room after
    lim[("AAA", DAY)] = bars(DAY, [(101.0, 101.5, 97.8, 98.6), (98.6, 99.0, 98.2, 98.9)])
    assert tracker.monitor_dip_long_orders(now=at(DAY, 10, 0)) == 1      # judged at the 10:00 tick
    t = dips()["AAA"]
    assert t["entry_price"] == 98.0 and t["dip_fill_kind"] == "touch"
    assert t["entry_datetime"].startswith("2026-10-08T14:00")             # the end of the 09:30 bar
    assert 0 < t["dip_account_shares"] < 10 and t["dip_order_qty"] == 10   # cut to the room the short left
    assert t["dip_account_shares"] == t["dip_fill_cap"]


def test_a_touch_with_no_room_buys_nothing_and_waits_for_the_next_session(lim):
    lim[("AAA", DAY)] = bars(DAY, [(101.0, 101.5, 100.5, 101.0)])
    lim[("BBB", DAY)] = bars(DAY, [(51.0, 51.2, 50.5, 51.0)])
    tracker.record_dip_long_trades(now=OPEN_TICK)
    tracker._save_trades(tracker._load_trades() + [short("HOT", 349)])   # the whole room
    lim[("AAA", DAY)] = bars(DAY, [(101.0, 101.5, 97.8, 98.6), (98.6, 98.7, 97.0, 97.5)])
    assert tracker.monitor_dip_long_orders(now=at(DAY, 10, 0)) == 0
    o = orders_of()["AAA"]
    assert o["status"] == "working" and o["unfunded_session"] == "2026-10-08"
    assert tracker.monitor_dip_long_orders(now=at(DAY, 10, 30)) == 0     # not again this session
    # the next session: the short is gone, the order is armed at the open and fills when a bar reaches it
    tracker._save_trades([t for t in tracker._load_trades() if t["ticker"] != "HOT"])
    lim[("AAA", NEXT)] = bars(NEXT, [(99.0, 99.5, 98.4, 99.2)])
    assert tracker.record_dip_long_trades(now=at(NEXT, 9, 30)) >= 1             # AAA armed again
    assert "2026-10-09" in orders_of()["AAA"]["session_armed_at"]
    lim[("AAA", NEXT)] = bars(NEXT, [(99.0, 99.5, 98.4, 99.2), (99.2, 99.3, 97.9, 98.1)])
    assert tracker.monitor_dip_long_orders(now=at(NEXT, 10, 30)) == 1
    t = dips()["AAA"]
    assert t["dip_account_shares"] == 10 and t["entry_datetime"].startswith("2026-10-09T14:30")


def test_an_order_lapses_after_its_last_session_and_a_margin_call_cancels_the_working_ones(lim):
    lim[("AAA", DAY)] = bars(DAY, [(101.0, 101.5, 100.5, 101.0)])
    lim[("BBB", DAY)] = bars(DAY, [(51.0, 51.2, 50.5, 51.0)])
    tracker.record_dip_long_trades(now=OPEN_TICK)
    assert tracker.cancel_dip_orders("cancel_margin", now=at(DAY, 11, 0)) == 2
    assert {o["status"] for o in orders_of().values()} == {"pending"}
    last = date(2026, 10, 14)
    tracker.monitor_dip_long_orders(now=at(last, 15, 45))
    assert {o["status"] for o in orders_of().values()} == {"pending"}
    tracker.monitor_dip_long_orders(now=at(last, 16, 1))
    assert {o["status"] for o in orders_of().values()} == {"lapsed"}


def test_another_books_name_is_neither_placed_nor_filled(lim):
    tracker._save_trades([short("AAA", 10, px=100.0)])
    lim[("AAA", DAY)] = bars(DAY, [(97.0, 99.0, 96.0, 98.5)])
    lim[("BBB", DAY)] = bars(DAY, [(51.0, 51.2, 50.5, 51.0)])
    tracker.record_dip_long_trades(now=OPEN_TICK)
    o = orders_of()["AAA"]
    assert o["status"] == "pending" and o["events"][-1]["what"] == "held"
    assert "AAA" not in dips()


# ── the broker ───────────────────────────────────────────────────────────────

def _broker(monkeypatch, store):
    import src.broker.reconcile as rec
    from tests.test_broker_reconcile import FakeBroker
    monkeypatch.setattr(rec.repo, "load_trades", lambda: store["trades"])
    monkeypatch.setattr(rec.repo, "save_trades", lambda t: store.update(trades=t))
    monkeypatch.setattr(settings, "broker_mode", "ibkr_paper")
    monkeypatch.setattr(settings, "broker_settle_seconds", 0)
    monkeypatch.setattr(settings, "enable_legacy_entries", False)
    monkeypatch.setattr(rec, "usd_per_unit", lambda ccy: 1.0)
    from src.performance import market_calendar
    monkeypatch.setattr(market_calendar, "current_session", lambda now=None: "rth")
    monkeypatch.setattr(dl, "today_et", lambda now=None: DAY)

    class Resting(FakeBroker):
        def __init__(self):
            super().__init__()
            self.cancelled = []

        def submit_order(self, req):
            res = super().submit_order(req)
            if getattr(req, "rest", False):
                res.filled_qty, res.status, res.avg_fill_price = 0, "Submitted", None
            return res

        def cancel_order(self, client_ref):
            self.cancelled.append(client_ref)
            return True
    return rec, Resting()


def test_the_broker_rests_each_working_order_at_the_rooms_size(lim, monkeypatch):
    lim[("AAA", DAY)] = bars(DAY, [(101.0, 101.5, 100.5, 101.0)])
    lim[("BBB", DAY)] = bars(DAY, [(51.0, 51.2, 50.5, 51.0)])
    tracker.record_dip_long_trades(now=OPEN_TICK)
    store = {"trades": tracker._load_trades()}
    rec, b = _broker(monkeypatch, store)
    rec.sync(broker=b)
    rest = [(o.ticker, o.side, o.quantity, o.limit_price, o.rest) for o in b.orders]
    assert rest == [("AAA", "BUY", 10, 98.0, True), ("BBB", "BUY", 20, 49.0, True)]
    assert orders_of()["AAA"]["ibkr"]["state"] == "working" and orders_of()["AAA"]["ibkr"]["ref"].startswith("dipw-AAA-")
    rec.sync(broker=b)
    assert len(b.orders) == 2                                       # unchanged: nothing re-sent
    store["trades"] = store["trades"] + [short("HOT", 340)]           # the room shrinks
    rec.sync(broker=b)
    aaa = [o for o in b.orders if o.ticker == "AAA"]
    assert "dipw-AAA-2026-10-08-1" in b.cancelled                     # resized: cancelled ...
    assert len(aaa) == 2 and 0 < aaa[-1].quantity < 10 and aaa[-1].client_ref.endswith("-2")   # ... re-placed smaller


def test_a_fill_links_the_long_to_its_resting_order_and_sends_no_second_order(lim, monkeypatch):
    from src.broker.base import FillSummary
    lim[("AAA", DAY)] = bars(DAY, [(101.0, 101.5, 100.5, 101.0)])
    lim[("BBB", DAY)] = bars(DAY, [(51.0, 51.2, 50.5, 51.0)])
    tracker.record_dip_long_trades(now=OPEN_TICK)
    store = {"trades": tracker._load_trades()}
    rec, b = _broker(monkeypatch, store)
    rec.sync(broker=b)
    ref = orders_of()["AAA"]["ibkr"]["ref"]
    lim[("AAA", DAY)] = bars(DAY, [(101.0, 101.5, 97.8, 98.6)])
    tracker.monitor_dip_long_orders(now=at(DAY, 10, 0))
    t = dips()["AAA"]
    assert t["broker_client_ref"] == ref and t["broker_order_id"]
    b.fills = [FillSummary(client_ref=ref, ticker="AAA", side="BUY", filled_qty=10, avg_fill_price=98.0,
                           order_id=t["broker_order_id"])]
    store["trades"] = tracker._load_trades()
    n_before = len(b.orders)
    rec.sync(broker=b)
    assert len([o for o in b.orders[n_before:] if o.ticker == "AAA"]) == 0
    aaa = next(x for x in store["trades"] if x["ticker"] == "AAA" and x.get("entry_mechanism") == "dip_long")
    assert aaa["broker_fill_qty"] == 10


def test_a_resting_order_filled_before_the_ledger_sees_it_is_not_drift(lim, monkeypatch):
    """IBKR fills a resting order mid-tick; the ledger links it only at the next tick (once the bar has ended): until
    then the end-of-tick sync must not flatten that position as drift."""
    from src.broker.base import Position
    lim[("AAA", DAY)] = bars(DAY, [(101.0, 101.5, 100.5, 101.0)])
    lim[("BBB", DAY)] = bars(DAY, [(51.0, 51.2, 50.5, 51.0)])
    monkeypatch.setattr(settings, "broker_drift_action", "flatten")
    tracker.record_dip_long_trades(now=OPEN_TICK)
    store = {"trades": tracker._load_trades()}
    rec, b = _broker(monkeypatch, store)
    rec.sync(broker=b)                                   # AAA's limit rests at IBKR
    n = len(b.orders)
    b._positions = [Position(ticker="AAA", quantity=10, avg_cost=98.0)]   # ... and fills before the next tick
    rec.sync(broker=b)
    assert not [o for o in b.orders[n:] if o.ticker == "AAA" and o.side == "SELL"]   # not flattened
