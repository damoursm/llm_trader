"""Share recalls (2026-10-07, user: "protecting ourselves against share recalls ... IBKR then buys back our short at
market"): the journal-only watch of each held short's place in IBKR's short-stock file, and the broker sync's
alert when IBKR holds fewer short shares than the open trades own (a forced buy-in)."""
from datetime import datetime, timezone
from types import SimpleNamespace
from zoneinfo import ZoneInfo

from config.settings import settings

ET = ZoneInfo("America/New_York")


def _short(tk, qty, rid):
    return {"ticker": tk, "status": "OPEN", "entry_mechanism": "sel_short", "action": "SELL",
            "broker_fill_qty": qty, "recommendation_id": rid}


def test_a_short_ibkr_no_longer_holds_is_alerted_after_the_confirmation_syncs(monkeypatch):
    from src.broker import reconcile
    monkeypatch.setattr(settings, "broker_buy_in_confirm_syncs", 3)
    trades = [_short("AAA", 100, "a1"), _short("AAA", 50, "a2"), _short("BBB", 10, "b1"),
              dict(_short("CCC", 10, "c1"), status="CLOSED")]
    pos = {"AAA": SimpleNamespace(quantity=-150), "BBB": SimpleNamespace(quantity=-10)}
    rep = {}
    assert reconcile._buy_in_check(trades, pos, rep) is False and "buy_in_suspects" not in rep
    pos["AAA"] = SimpleNamespace(quantity=-50)                        # 100 short shares gone
    for n in (1, 2):
        rep = {}
        assert reconcile._buy_in_check(trades, pos, rep) is True
        assert "buy_in_suspects" not in rep and trades[0]["broker_short_missing_n"] == n
    rep = {}
    reconcile._buy_in_check(trades, pos, rep)
    sus = rep["buy_in_suspects"]
    assert {s["trade"] for s in sus} == {"a1", "a2"} and sus[0]["missing"] == 100 and sus[0]["ibkr_short"] == 50
    assert "broker_short_missing_n" not in trades[2]                  # BBB is whole
    del pos["AAA"]                                                    # no position row at all: every share gone
    rep = {}
    reconcile._buy_in_check(trades, pos, rep)
    assert rep["buy_in_suspects"][0]["ibkr_short"] == 0 and rep["buy_in_suspects"][0]["missing"] == 150
    pos["AAA"] = SimpleNamespace(quantity=-150)                       # back: the stamps clear
    assert reconcile._buy_in_check(trades, pos, {}) is True
    assert "broker_short_missing_n" not in trades[0] and "broker_short_missing_since" not in trades[1]
    pos["AAA"] = SimpleNamespace(quantity=-200)                       # MORE short than owned (a pending exit): fine
    rep = {}
    assert reconcile._buy_in_check(trades, pos, rep) is False and "buy_in_suspects" not in rep


def test_the_broker_banner_names_a_suspected_buy_in():
    from src.pipeline import _assess_broker_health
    h = _assess_broker_health({"mode": "ibkr_paper", "connected": True, "ok": True,
                               "buy_in_suspects": [{"ticker": "AAA"}, {"ticker": "AAA"}]})
    assert h["down"] and "BUY-IN" in h["message"] and "AAA" in h["message"] and h["message"].count("AAA") == 1
    assert not _assess_broker_health({"mode": "ibkr_paper", "connected": True, "ok": True})["down"]


def test_the_borrow_watch_stamps_held_shorts_and_never_closes_them(monkeypatch):
    from src.data import ibkr_borrow
    from src.performance import tracker
    monkeypatch.setattr(settings, "enable_sel_short_borrow_watch", True)
    ts = datetime(2026, 10, 7, 9, 30, tzinfo=ET)
    table = {"AAA": ibkr_borrow.Borrow("AAA", 5.0, 1.0, 50_000, ts)}
    monkeypatch.setattr(ibkr_borrow, "latest", lambda *a, **k: table)
    book = [{"ticker": "AAA", "status": "OPEN"}, {"ticker": "BBB", "status": "OPEN"}]
    now = datetime(2026, 10, 7, 14, 0, tzinfo=timezone.utc)
    assert tracker._sel_short_borrow_watch(book, now) is True
    assert (book[0]["sel_borrow_state"], book[0]["sel_borrow_available"]) == ("listed", 50_000)
    assert (book[1]["sel_borrow_state"], book[1]["sel_borrow_absent_count"]) == ("absent", 1)
    since = book[1]["sel_borrow_absent_since"]
    assert tracker._sel_short_borrow_watch(book, now) is False        # nothing moved
    table["BBB"] = ibkr_borrow.Borrow("BBB", 30.0, -10.0, 2_000, ts)
    assert tracker._sel_short_borrow_watch(book, now) is True
    assert book[1]["sel_borrow_state"] == "listed" and book[1]["sel_borrow_absent_since"] == since
    del table["BBB"]
    tracker._sel_short_borrow_watch(book, now)
    assert book[1]["sel_borrow_absent_count"] == 2 and book[1]["sel_borrow_absent_since"] == since
    assert all(t["status"] == "OPEN" for t in book)                   # journal-only
    monkeypatch.setattr(ibkr_borrow, "latest", lambda *a, **k: None)  # no current file: nothing stamped
    assert tracker._sel_short_borrow_watch([{"ticker": "ZZZ"}], now) is False
    monkeypatch.setattr(settings, "enable_sel_short_borrow_watch", False)
    assert tracker._sel_short_borrow_watch([{"ticker": "ZZZ"}], now) is False
