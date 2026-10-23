"""THE FRESH START (2026-10-20; the user: "can you exit all positions in IBKR so that we start fresh?", the live
account restarting at $50,000) and PREREG159's per-arm rules: `tracker.fresh_start_live_books`,
`sim_account.counts`, `sel_short.before_fresh_start`, `sel_short.max_hold`, `sel_short.volnorm_ratio`."""
import json
from datetime import date, datetime
from zoneinfo import ZoneInfo

from config.settings import settings
from src.performance import live_account, tracker
from src.signals import dip_long, sel_short

ET = ZoneInfo("America/New_York")
FS = "2026-10-20T09:30:00-04:00"


def et(y, m, d, hh, mm):
    return datetime(y, m, d, hh, mm, tzinfo=ET)


def _seed(rows):
    trades = tracker._load_trades()
    for r in rows:
        trades.append(dict({"type": "STOCK", "status": "OPEN", "entry_date": "2026-10-14",
                            "entry_datetime": "2026-10-14T13:30:00+00:00", "entry_session": "rth",
                            "entry_price": 100.0, "current_price": 90.0, "position_size_multiplier": 1.0}, **r))
    tracker._save_trades(trades)


def test_fresh_start_closes_every_live_book_once_in_regular_hours(monkeypatch):
    monkeypatch.setattr(settings, "live_account_fresh_start", FS)
    monkeypatch.setattr(tracker, "_fetch_price", lambda tk: 91.0)
    _seed([{"ticker": "DIP", "action": "BUY", "entry_mechanism": "dip_long", "dip_account_shares": 3,
            "recommendation_id": "a"},
           {"ticker": "SHO", "action": "SELL", "entry_mechanism": "sel_short", "sel_account_shares": 5,
            "sel_arm": "vol", "recommendation_id": "b"},
           {"ticker": "TQQQ", "action": "BUY", "entry_mechanism": "crash_long", "crash_account_shares": 2,
            "recommendation_id": "c"},
           {"ticker": "LEG", "action": "BUY", "recommendation_id": "d"}])       # not a live book: left alone
    assert tracker.fresh_start_live_books(now=et(2026, 10, 20, 9, 0)) == 0      # before the instant
    assert tracker.fresh_start_live_books(now=et(2026, 10, 20, 20, 30)) == 0    # after it, not regular hours
    assert tracker.fresh_start_live_books(now=et(2026, 10, 20, 9, 31)) == 3
    by = {t["ticker"]: t for t in tracker._load_trades()}
    for tk in ("DIP", "SHO", "TQQQ"):
        assert by[tk]["status"] == "CLOSED" and by[tk]["exit_reason"] == "fresh_start"
        assert by[tk]["exit_price"] == 91.0                                    # a price fetched now
    assert by["LEG"]["status"] == "OPEN"
    _seed([{"ticker": "NEW", "action": "BUY", "entry_mechanism": "dip_long", "dip_account_shares": 1,
            "entry_datetime": "2026-10-20T14:00:00+00:00", "recommendation_id": "e"}])
    assert tracker.fresh_start_live_books(now=et(2026, 10, 21, 10, 0)) == 0     # once, ever
    assert json.loads(tracker._FRESH_START_MARKER.read_text())["after"] == datetime.fromisoformat(FS).isoformat()


def test_fresh_start_waits_for_a_price(monkeypatch):
    monkeypatch.setattr(settings, "live_account_fresh_start", FS)
    monkeypatch.setattr(tracker, "_fetch_price", lambda tk: None)
    _seed([{"ticker": "DIP", "action": "BUY", "entry_mechanism": "dip_long", "dip_account_shares": 3,
            "recommendation_id": "a"}])
    assert tracker.fresh_start_live_books(now=et(2026, 10, 20, 9, 31)) == 0
    assert not tracker._FRESH_START_MARKER.exists()                              # retried next tick
    monkeypatch.setattr(tracker, "_fetch_price", lambda tk: 95.0)
    assert tracker.fresh_start_live_books(now=et(2026, 10, 20, 10, 1)) == 1
    assert tracker._FRESH_START_MARKER.exists()


def test_fresh_start_settles_the_dip_books_earlier_orders(monkeypatch):
    monkeypatch.setattr(settings, "live_account_fresh_start", FS)
    monkeypatch.setattr(settings, "dip_long_entry_mode", "limit")
    monkeypatch.setattr(settings, "dip_long_limit_sessions", 3)
    d0, today = date(2026, 10, 19), date(2026, 10, 20)
    dip_long.save_orders(d0, [{"oid": "A|2026-10-19", "ticker": "A", "day": "2026-10-19", "status": "working",
                               "window": ["2026-10-19", "2026-10-20", "2026-10-21"]},
                              {"oid": "B|2026-10-19", "ticker": "B", "day": "2026-10-19", "status": "not_top"}])
    dip_long.save_orders(today, [{"oid": "C|2026-10-20", "ticker": "C", "day": "2026-10-20", "status": "pending",
                                  "window": ["2026-10-20", "2026-10-21", "2026-10-22"]}])
    tracker.fresh_start_live_books(now=et(2026, 10, 20, 9, 31))
    assert [o["status"] for o in dip_long.load_orders(d0)] == ["fresh_start", "not_top"]
    assert [o["status"] for o in dip_long.load_orders(today)] == ["pending"]       # today's: the new account's
    assert json.loads(tracker._FRESH_START_MARKER.read_text())["dip_orders_settled"] == 1


def test_the_account_restarts_at_its_initial_balance(monkeypatch):
    monkeypatch.setattr(settings, "sel_short_account_initial", 50000.0)
    old = [{"ticker": "OLD", "status": "CLOSED", "entry_mechanism": "dip_long", "dip_account_shares": 10,
            "entry_price": 100.0, "return_pct": -20.0, "entry_datetime": "2026-10-09T13:30:00+00:00"},
           {"ticker": "OLDS", "status": "CLOSED", "entry_mechanism": "sel_short", "sel_account_shares": 10,
            "entry_price": 50.0, "return_pct": 10.0, "entry_datetime": "2026-10-15T16:00:00+00:00"}]
    new = [{"ticker": "NEW", "status": "CLOSED", "entry_mechanism": "dip_long", "dip_account_shares": 10,
            "entry_price": 100.0, "return_pct": 5.0, "entry_datetime": "2026-10-20T13:30:00+00:00"}]
    monkeypatch.setattr(settings, "live_account_fresh_start", "")
    assert live_account.state(old + new)["equity"] == 50000.0 - 200.0 + 50.0 + 50.0
    monkeypatch.setattr(settings, "live_account_fresh_start", FS)
    assert live_account.state(old + new)["equity"] == 50000.0 + 50.0         # only the trades entered from it
    held = dict(old[0], status="OPEN", current_price=80.0)                    # an open trade always counts
    assert live_account.state([held])["equity"] == 50000.0 - 200.0


def test_prereg159_per_arm_rules(monkeypatch):
    monkeypatch.setattr(settings, "sel_short_arm_max_hold_sessions", 10)
    monkeypatch.setattr(settings, "sel_short_vol_max_hold_sessions", 10)
    monkeypatch.setattr(settings, "sel_short_max_hold_sessions", 15)
    assert [sel_short.max_hold(a) for a in ("vol", "etf", "fall", "vol2", "model")] == [10, 10, 10, 10, 15]
    monkeypatch.setattr(settings, "sel_short_vol_max_hold_sessions", 15)     # PREREG173's V15: the vol arm's own
    assert [sel_short.max_hold(a) for a in ("vol", "etf", "fall", "vol2", "model")] == [15, 10, 10, 15, 15]
    monkeypatch.setattr(settings, "sel_short_vol_max_hold_sessions", 20)
    assert (sel_short.max_hold("vol"), sel_short.max_hold("fall"), sel_short.max_hold("etf")) == (20, 10, 10)
    monkeypatch.setattr(settings, "sel_short_fall_volnorm_ratio", 0.7)
    monkeypatch.setattr(settings, "sel_short_volnorm_ratio", 0.5)
    assert [sel_short.volnorm_ratio(a) for a in ("fall", "etf", "model")] == [0.7, 0.5, 0.5]
    assert tracker._sel_volnorm_ratio({"sel_arm": "fall"}) == 0.7
    assert tracker._sel_volnorm_ratio({"sel_arm": "etf"}) == 0.5


def test_picks_before_the_fresh_start_never_enter(monkeypatch):
    monkeypatch.setattr(settings, "live_account_fresh_start", "2026-10-20T11:00:00-04:00")
    monkeypatch.setattr(settings, "sel_short_traded_arms", "vol,etf,fall")
    old = {"bar_end": et(2026, 10, 20, 10, 0).isoformat()}
    assert not sel_short.before_fresh_start(old, et(2026, 10, 20, 10, 30))     # the instant not passed: as before
    assert sel_short.before_fresh_start(old, et(2026, 10, 20, 11, 5))
    assert not sel_short.before_fresh_start({"bar_end": et(2026, 10, 20, 11, 30).isoformat()},
                                            et(2026, 10, 20, 11, 35))
    d = date(2026, 10, 20)
    p = sel_short.picks_path(d)
    p.parent.mkdir(parents=True, exist_ok=True)
    rec = {"day": "2026-10-20", "bar_of_day": 0, "ticker": "XYZ", "arm": "vol", "decision": "short",
           "bar_end": et(2026, 10, 20, 10, 0).isoformat()}
    p.write_text(json.dumps(rec) + "\n", encoding="utf-8")
    assert sel_short.pending_entries(et(2026, 10, 20, 11, 5)) == []
    assert sel_short.consumed(d)[sel_short.pick_key(rec)] == "fresh_start"
    monkeypatch.setattr(settings, "live_account_fresh_start", "")
    assert not sel_short.before_fresh_start(old, et(2026, 10, 20, 11, 5))
