"""SPLITS DURING A HOLD (PREREG64; user 2026-10-09: "Fix the bugs in production and in the simulator").

A split rescales a position's quoted price and its share count at once — IBKR adjusts both. A ledger trade that
kept its pre-split entry and share count read a 1-for-10 reverse split as a 900% loss on a short: a false margin call
for the whole live account, and share counts that no longer matched IBKR's (a false buy-in alarm on a short, the
"excess" shares of a forward-split long flattened as drift). What must hold: from 04:00 ET of the split's execution
date every open trade is in the terms quoted now — prices x the factor, share counts / it; the dollars, the return
and the account are unchanged; it happens once; the squeeze cover is not adjusted twice; IBKR's borrow charges and a
fill IBKR reports late come into the same terms; a working dip limit order is rescaled like a position.
"""
from __future__ import annotations

from datetime import date, datetime, timezone
from zoneinfo import ZoneInfo

import pytest

from config.settings import settings
from src.broker import FillSummary
from src.broker import reconcile
from src.data import intraday_store as ist
from src.performance import borrow_fees as bf
from src.performance import live_account as la
from src.performance import tracker

ET = ZoneInfo("America/New_York")
SPLIT_DAY = date(2026, 10, 7)                                       # a Wednesday


def et(*a):
    return datetime(*a, tzinfo=ET)


@pytest.fixture
def on(monkeypatch):
    monkeypatch.setattr(settings, "enable_split_adjustment", True)
    monkeypatch.setattr(settings, "enable_sel_short", True)
    monkeypatch.setattr(settings, "enable_sel_short_account_sizing", True)
    monkeypatch.setattr(settings, "enable_live_account", True)
    monkeypatch.setattr(settings, "sel_short_traded_arms", "vol,etf,fall")
    monkeypatch.setattr(settings, "sel_short_account_initial", 10_000.0)
    monkeypatch.setattr(settings, "sel_short_account_slices", 24)
    monkeypatch.setattr(settings, "sel_short_account_house_maint", 2.0)
    monkeypatch.setattr(settings, "sel_short_account_house_init", 2.86)
    rows = {"RVS": [(SPLIT_DAY, 10.0, 1.0)], "FWD": [(SPLIT_DAY, 1.0, 2.0)]}
    monkeypatch.setattr(ist, "split_rows", lambda: rows)
    return rows


def short(tk="RVS", n=40, px=10.0, mark=9.6, at="2026-10-05T14:00:00+00:00", mark_at="2026-10-06T19:00:00+00:00"):
    return {"ticker": tk, "entry_mechanism": "sel_short", "sel_arm": "vol", "action": "SELL", "type": "STOCK",
            "status": "OPEN", "sel_account_shares": n, "entry_price": px, "current_price": mark,
            "current_price_datetime": mark_at, "return_pct": round((px - mark) / px * 100, 3),
            "sel_house_maint": 2.0, "sel_house_init": 2.86, "entry_datetime": at, "entry_date": at[:10],
            "entry_session": "rth", "position_size_multiplier": 1.0, "recommendation_id": f"sel-{tk}",
            "entry_ref_close": 10.1, "entry_ref_close_date": "2026-10-02", "sel_target_price": 8.0,
            "sel_bar_close": 10.05, "sel_pre5_close": 8.0, "broker_fill_qty": n, "broker_requested_qty": n,
            "broker_fill_price": 10.02, "broker_client_ref": f"sel-{tk}", "broker_status": "Filled",
            "broker_submitted_at": "2026-10-05T14:00:05+00:00", "sel_deadline": "2026-10-26T14:00:00+00:00"}


def dip(tk="FWD", n=10, px=200.0, mark=204.0, at="2026-10-05T13:30:00+00:00"):
    return {"ticker": tk, "entry_mechanism": "dip_long", "action": "BUY", "type": "STOCK", "status": "OPEN",
            "dip_account_shares": n, "entry_price": px, "current_price": mark,
            "current_price_datetime": "2026-10-06T19:00:00+00:00", "return_pct": 2.0, "entry_datetime": at,
            "entry_date": at[:10], "entry_session": "rth", "position_size_multiplier": 1.0,
            "recommendation_id": f"dip-{tk}", "dip_close": 205.0, "dip_sma_trend": 180.0, "dip_limit": 200.0,
            "dip_atr14": 10.0, "broker_fill_qty": n, "broker_requested_qty": n, "broker_fill_price": 200.0}


def test_a_reverse_split_rescales_an_open_short_once_from_the_pre_market_open(on):
    t = short()
    assert tracker.apply_splits([t], et(2026, 10, 7, 3, 59)) == 0         # the overnight hours: the old terms
    assert t["entry_price"] == 10.0 and "split_factor" not in t
    assert tracker.apply_splits([t], et(2026, 10, 7, 4, 0)) == 1
    assert t["entry_price"] == 100.0 and t["sel_account_shares"] == 4.0 and t["broker_fill_qty"] == 4.0
    assert t["sel_target_price"] == 80.0 and t["entry_ref_close"] == 101.0 and t["broker_fill_price"] == 100.2
    assert t["current_price"] == 96.0                                       # Tuesday's mark, in today's terms
    assert t["split_factor"] == 10.0 and t["split_orig"]["entry_price"] == 10.0
    assert t["sel_account_shares"] * t["entry_price"] == 40 * 10.0          # the same dollars
    assert tracker.apply_splits([t], et(2026, 10, 8, 10, 0)) == 0           # once
    assert t["entry_price"] == 100.0 and len(t["split_log"]) == 1
    on["RVS"].append((date(2026, 10, 9), 1.0, 2.0))                         # another split later: from the original
    assert tracker.apply_splits([t], et(2026, 10, 9, 9, 30)) == 1
    assert t["entry_price"] == 50.0 and t["sel_account_shares"] == 8.0 and t["split_factor"] == 5.0
    closed = dict(short(), status="CLOSED")
    assert tracker.apply_splits([closed], et(2026, 10, 8, 10, 0)) == 0      # a closed trade keeps its terms


def test_a_split_on_the_entry_session_is_already_in_the_entry(on):
    t = short(at="2026-10-07T13:45:00+00:00")                               # entered 09:45 ET on the split day
    assert tracker.apply_splits([t], et(2026, 10, 8, 10, 0)) == 0
    early = short(at="2026-10-07T07:30:00+00:00")                           # 03:30 ET: still the old terms
    assert tracker.apply_splits([early], et(2026, 10, 7, 10, 0)) == 1 and early["entry_price"] == 100.0


def test_a_reverse_split_never_reads_as_a_loss_or_a_margin_call(monkeypatch, on):
    """The live mark after a 1-for-10 reverse split: $95 is a 5% gain on a $10 short, not an 850% loss."""
    from src.db import repo
    book = [short()] + [short(tk=f"S{k}", mark=10.0) for k in range(5)]
    repo.save_trades(book, allow_shrink=True)
    prices = {"RVS": 95.0, **{f"S{k}": 10.0 for k in range(5)}}
    monkeypatch.setattr(tracker, "_fetch_price", lambda tk: prices[tk])
    monkeypatch.setattr(tracker, "_now_iso", lambda: "2026-10-07T14:00:00+00:00")
    monkeypatch.setattr(tracker, "_refresh_open_trade_ohlcv", lambda trades: None)
    monkeypatch.setattr(tracker, "_update_pivot_targets", lambda trades: None)
    tracker.update_open_trades()                                            # now: any time after the split
    t = {x["ticker"]: x for x in tracker._load_trades()}
    assert t["RVS"]["entry_price"] == 100.0 and t["RVS"]["sel_account_shares"] == 4.0
    assert 4.0 < t["RVS"]["return_pct"] < 5.1
    hit, st = la.margin_call(list(t.values()), datetime(2026, 10, 7, 14, 0, tzinfo=timezone.utc))
    assert not hit and st["equity"] > 9_990.0
    # the adjustment off: the bug — the account reads an 850% loss and calls the margin
    monkeypatch.setattr(settings, "enable_split_adjustment", False)
    repo.save_trades(book, allow_shrink=True)
    tracker.update_open_trades()
    raw = tracker._load_trades()
    assert next(x for x in raw if x["ticker"] == "RVS")["return_pct"] < -800.0
    assert la.margin_call(raw, datetime(2026, 10, 7, 14, 0, tzinfo=timezone.utc))[0]


def test_a_forward_split_doubles_a_dip_longs_shares_and_the_broker_owns_them(on):
    t = dip()
    assert tracker.apply_splits([t], et(2026, 10, 7, 9, 30)) == 1
    assert t["entry_price"] == 100.0 and t["dip_account_shares"] == 20.0 and t["broker_fill_qty"] == 20.0
    assert t["dip_limit"] == 100.0 and t["dip_close"] == 102.5 and t["dip_sma_trend"] == 90.0
    assert t["current_price"] == 102.0
    assert reconcile._owned_by_open([t]) == {"FWD": 20}                     # IBKR's 20 shares are the trade's


def test_the_squeeze_cover_is_not_adjusted_twice(on, monkeypatch):
    monkeypatch.setattr(settings, "enable_sel_short_squeeze_cover", True)
    monkeypatch.setattr(settings, "sel_short_cover_multiple", 6.0)
    monkeypatch.setattr(settings, "sel_short_vol_cover_multiple", 6.0)     # the split's count at one level (live: 4x)
    now = et(2026, 10, 7, 10, 0)
    raw = short()
    assert tracker._sel_short_cover_level(raw, now) == 600.0                # 6 x $10, carried through the split
    done = short()
    tracker.apply_splits([done], now)
    assert tracker._sel_short_cover_level(done, now) == 600.0               # 6 x the rescaled $100, once
    assert tracker._sel_short_cover_level(short(), et(2026, 10, 7, 3, 30)) == 60.0   # before 04:00: old terms


def test_ibkrs_charges_before_a_split_count_in_todays_shares(on, monkeypatch):
    monkeypatch.setattr(settings, "enable_ibkr_borrow_schedule", True)
    monkeypatch.setattr(settings, "enable_short_borrow_cost", True)
    monkeypatch.setattr(bf, "_rates", lambda t: [(date(2026, 10, 1), 36.0, "own_archive")])
    monkeypatch.setattr(bf, "_file_rate", lambda t, when: None)
    monkeypatch.setattr(bf, "_closes", lambda t: [(date(2026, 10, 5), 98.0), (date(2026, 10, 6), 97.0)])
    monkeypatch.setattr(bf, "_charged", lambda: {("RVS", date(2026, 10, 6)): {"fee_ps": 0.01, "rate": 36.0,
                                                                                "price": 10.0}})
    monkeypatch.setattr(bf, "charged_version", lambda: "v1")
    t = short()
    now = et(2026, 10, 8, 11, 0)
    tracker.apply_splits([t], now)
    s = bf.schedule(t, now=now)
    day = {r[0]: r for r in s["days"]}
    assert day["2026-10-06"] == ["2026-10-06", "ibkr_charged", 36.0, 100.0, 0.1]   # 1 share now = 10 then
    assert day["2026-10-07"][1] == "own_archive_carried" and day["2026-10-07"][3] == 99.0   # the formula: today's close
    assert s["frac"] == pytest.approx(s["fee_ps"] / 100.0)


def test_a_fill_ibkr_reports_after_a_split_lands_in_the_trades_terms(on):
    t = short()
    t.update(broker_fill_qty=0, broker_fill_price=None, broker_status="Submitted")
    tracker.apply_splits([t], et(2026, 10, 7, 10, 0))                      # the order went out before the split
    assert t["broker_requested_qty"] == 4.0
    fs = FillSummary(client_ref="sel-RVS", ticker="RVS", side="SELL", filled_qty=40, avg_fill_price=10.0,
                     commission=1.0)
    assert reconcile._apply_fill(t, "broker_", fs)
    assert t["broker_fill_qty"] == 4.0 and t["broker_fill_price"] == 100.0 and t["broker_status"] == "Filled"
    assert not reconcile._apply_fill(t, "broker_", fs)                     # the same report: nothing new
    later = short()
    later.update(broker_submitted_at="2026-10-07T14:00:00+00:00")
    tracker.apply_splits([later], et(2026, 10, 7, 10, 0))
    fs2 = FillSummary(client_ref="sel-RVS", ticker="RVS", side="SELL", filled_qty=4, avg_fill_price=100.0,
                      commission=1.0)
    reconcile._apply_fill(later, "broker_", fs2)                            # resent after the split: as reported
    assert later["broker_fill_qty"] == 4 and later["broker_fill_price"] == 100.0


def test_a_working_dip_order_is_rescaled_like_a_position(on):
    from src.signals import dip_long
    o = {"oid": "FWD|2026-10-05", "ticker": "FWD", "day": "2026-10-05", "status": "working", "close": 152.0,
         "atr14": 3.3, "sma_trend": 140.0, "limit": 150.37, "qty": 7, "ibkr_qty": 7, "events": []}
    gone = dict(o, oid="FWD|x", status="filled", events=[])
    assert dip_long.apply_order_splits([o, gone], et(2026, 10, 7, 3, 0)) == 0
    assert dip_long.apply_order_splits([o, gone], et(2026, 10, 7, 9, 30)) == 1
    assert o["limit"] == 75.18 and o["qty"] == 14 and o["ibkr_qty"] == 14 and o["close"] == 76.0
    assert gone["limit"] == 150.37 and gone["qty"] == 7                     # a filled order is history
    assert dip_long.apply_order_splits([o], et(2026, 10, 8, 9, 30)) == 0    # once
    r = {"oid": "RVS|2026-10-05", "ticker": "RVS", "day": "2026-10-05", "status": "pending", "close": 5.0,
         "atr14": 0.4, "sma_trend": 4.0, "limit": 4.8, "qty": None, "ibkr_qty": 0, "events": []}
    assert dip_long.apply_order_splits([r], et(2026, 10, 7, 9, 30)) == 1
    assert r["limit"] == 48.0 and r["qty"] is None
    on["FWD"] = [(date(2026, 10, 5), 1.0, 2.0)]                             # a split on the order's own day: in it
    o2 = {"oid": "FWD|y", "ticker": "FWD", "day": "2026-10-05", "status": "working", "limit": 75.18, "qty": 14,
          "events": []}
    assert dip_long.apply_order_splits([o2], et(2026, 10, 8, 9, 30)) == 0 and o2["limit"] == 75.18


def test_a_withdrawn_split_restores_a_dip_orders_quantity_with_its_limit(on):
    """A pending order rescaled by a 2-for-1, then placed (its quantity in the new terms), then the record withdrawn
    (the split table called it false, 2026-10-16): the order comes back to its day's terms — its quantity WITH its
    limit (the quantity was kept in the new terms: x2 the notional)."""
    from src.signals import dip_long
    r = {"oid": "FWD|2026-10-05", "ticker": "FWD", "day": "2026-10-05", "status": "pending", "close": 152.0,
         "atr14": 3.3, "sma_trend": 140.0, "limit": 150.37, "qty": None, "ibkr_qty": 0, "events": []}
    assert dip_long.apply_order_splits([r], et(2026, 10, 7, 9, 30)) == 1 and r["limit"] == 75.18
    r.update(status="working", qty=12)                                      # placed after the split: 12 x $75.18
    on["FWD"] = []                                                          # the record withdrawn
    assert dip_long.apply_order_splits([r], et(2026, 10, 8, 9, 30)) == 1
    assert r["limit"] == 150.37 and r["qty"] == 6                           # ~$902 either way


def test_off_nothing_moves(on, monkeypatch):
    monkeypatch.setattr(settings, "enable_split_adjustment", False)
    t, o = short(), {"ticker": "FWD", "day": "2026-10-05", "status": "working", "limit": 150.0, "qty": 7}
    from src.signals import dip_long
    assert tracker.apply_splits([t], et(2026, 10, 8, 10, 0)) == 0 and t["entry_price"] == 10.0
    assert dip_long.apply_order_splits([o], et(2026, 10, 8, 10, 0)) == 0 and o["limit"] == 150.0


def test_the_broker_sync_reads_a_split_position_in_ibkrs_shares(on, monkeypatch):
    """IBKR holds 4 short shares of RVS after the 1-for-10 and 20 of FWD after the 2-for-1: the broker's own load of
    the ledger is rescaled too — no false buy-in alarm on the short, no "excess" long shares flattened as drift."""
    import src.broker.reconcile as rec
    from src.broker.base import AccountSnapshot, Broker, OrderResult, Position

    class Fake(Broker):
        name = "fake"

        def __init__(self, positions):
            self._positions, self.orders = positions, []

        def connect(self):
            return True

        def is_connected(self):
            return True

        def get_account(self):
            return AccountSnapshot(100_000.0, 100_000.0, 100_000.0, "DU0000001")

        def get_positions(self):
            return list(self._positions)

        def get_fills(self):
            return []

        def submit_order(self, req):
            self.orders.append(req)
            return OrderResult(ok=True, ticker=req.ticker, side=req.side, requested_qty=req.quantity,
                               filled_qty=req.quantity, avg_fill_price=100.0, order_id=f"o{len(self.orders)}",
                               client_ref=req.client_ref, status="Filled")

    s, d = short(), dip()
    for t in (s, d):
        t.update(broker_order_id=f"id-{t['ticker']}", broker_client_ref=t["recommendation_id"], broker_status="Filled",
                 broker_commission=1.0)
    store = {"trades": [s, d]}
    monkeypatch.setattr(rec.repo, "load_trades", lambda: store["trades"])
    monkeypatch.setattr(rec.repo, "save_trades", lambda t: store.update(trades=t))
    monkeypatch.setattr(settings, "broker_mode", "ibkr_paper")
    monkeypatch.setattr(settings, "broker_settle_seconds", 0)
    monkeypatch.setattr(settings, "broker_buy_in_confirm_syncs", 2)
    monkeypatch.setattr(rec, "usd_per_unit", lambda ccy: 1.0)
    from src.performance import market_calendar
    monkeypatch.setattr(market_calendar, "current_session", lambda now=None: "rth")
    b = Fake([Position("RVS", -4, 95.0), Position("FWD", 20, 102.0)])
    for _ in range(3):
        rep = rec.sync(broker=b)
        assert "buy_in_suspects" not in rep
    assert b.orders == []                                                   # nothing flattened, nothing resent
    got = {t["ticker"]: t for t in store["trades"]}
    assert got["RVS"]["broker_fill_qty"] == 4.0 and got["FWD"]["broker_fill_qty"] == 20.0
