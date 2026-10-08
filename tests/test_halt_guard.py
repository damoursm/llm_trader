"""Trading halts (2026-10-07, user: "We can't buy back during a halt, and the stock can reopen much higher"):
execution safety — a pick whose name is halted at the entry tick is not entered that tick (re-checked next tick
inside its window), a held short that is halted is stamped; NYSE's current-halt list is parsed fail-soft."""
from datetime import datetime, timezone

from config.settings import settings
from src.data import trade_halts
from src.signals import sel_short as ss

CSV = ("﻿Halt Date,Halt Time,Symbol,Name,Exchange,Reason,Resume Date,NYSE Resume Time\n"
       "2026-10-07,10:01:00,HLT,Halted Co,Nasdaq,LULD pause,,\n"
       "2026-10-07,09:45:00,OLD,Old Halt Co,Nasdaq,LULD pause,2026-10-07,09:50:00\n"
       "2026-10-06,19:50:00,BRK B,Berkshire B,NYSE,News Pending,,\n")


def test_the_current_list_keeps_only_halts_without_a_resume(monkeypatch):
    import httpx

    class R:
        status_code, text = 200, CSV

        def raise_for_status(self):
            return None
    monkeypatch.setattr(httpx, "get", lambda *a, **k: R())
    trade_halts.reset()
    rows = trade_halts.current()
    assert set(rows) == {"HLT", "BRK B"} and rows["HLT"]["reason"] == "LULD pause"
    assert trade_halts.halted("BRK-B")["since"] == "2026-10-06 19:50:00"
    assert trade_halts.halted("OLD") is None and trade_halts.halted("XYZ") is None

    def boom(*a, **k):
        raise httpx.ConnectError("down")
    monkeypatch.setattr(httpx, "get", boom)
    trade_halts.reset()
    assert trade_halts.current() is None and trade_halts.halted("HLT") is None    # unknown blocks nothing


def _env(monkeypatch, halted):
    import test_sel_short as T
    tracker, consumed, calls = T._tracker_env(monkeypatch, [T._pick("HLT"), T._pick("ABC")])
    monkeypatch.setattr(settings, "enable_sel_short_halt_guard", True)
    monkeypatch.setattr(trade_halts, "current", lambda max_age_seconds=60.0: halted)
    journal = []
    monkeypatch.setattr(ss, "journal_entry", lambda rec, outcome, **kw: journal.append((rec["ticker"], outcome)))
    return tracker, consumed, calls, journal


def test_a_halted_pick_waits_for_the_next_tick(monkeypatch):
    tracker, consumed, calls, journal = _env(monkeypatch, {"HLT": {"since": "2026-09-28 10:01:00",
                                                                     "reason": "LULD pause", "exchange": "Nasdaq"}})
    assert tracker.record_sel_short_trades(run_id="r1") == 1
    t = [x for x in tracker._load_trades() if x.get("entry_mechanism") == "sel_short"]
    assert [x["ticker"] for x in t] == ["ABC"]
    assert ss.pick_key({"day": "2026-09-28", "bar_of_day": 1, "ticker": "HLT"}) not in consumed
    assert ("HLT", "halted_retry") in journal and [c[0] for c in calls] == ["ABC"]   # never priced nor borrowed


def test_a_held_short_that_is_halted_is_stamped(monkeypatch):
    from src.performance import tracker
    monkeypatch.setattr(settings, "enable_sel_short_halt_guard", True)
    state = {"HLT": {"since": "2026-10-07 10:01:00", "reason": "LULD pause", "exchange": "Nasdaq"}}
    monkeypatch.setattr(trade_halts, "current", lambda max_age_seconds=60.0: state)
    book = [{"ticker": "HLT", "status": "OPEN"}, {"ticker": "ABC", "status": "OPEN"}]
    now = datetime(2026, 10, 7, 14, 5, tzinfo=timezone.utc)
    assert tracker._sel_short_halt_watch(book, now) is True
    assert book[0]["sel_halted"] and book[0]["sel_halt_count"] == 1 and book[0]["sel_halt_reason"] == "LULD pause"
    assert "sel_halted" not in book[1]
    assert tracker._sel_short_halt_watch(book, now) is False             # the same halt: nothing new
    state.clear()
    assert tracker._sel_short_halt_watch(book, now) is True and book[0]["sel_halted"] is False
    state["HLT"] = {"since": "2026-10-07 11:30:00", "reason": "News pending", "exchange": "Nasdaq"}
    tracker._sel_short_halt_watch(book, now)
    assert book[0]["sel_halt_count"] == 2 and book[0]["status"] == "OPEN"
