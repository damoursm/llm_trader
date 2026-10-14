"""THE LIVE ACCOUNT (user directive 2026-10-09: only the best combination — vol + ETF + fallers + dip, the dip book
giving way, de-leveraging; shorts 1/24, dip 1/10 — on the $10,000 account; `src/performance/live_account.py`).

What must hold: one equity and one margin room for both books (a dip long's Reg T 50% / 25%, a short's house rate);
a short sized from that room; a dip long sized 1/10 of the shared equity within the room and the long side's cap, at
most 10 open; a short the room cannot fund in full sells the dip book's longs WHOLE, the latest first, only as many as
needed, before it is sized; at a tick where the maintenance requirement exceeds 80% of equity the dip longs are sold,
the latest first, until it is at most 60%; a margin call closes every short and long the account funded; off, the two
books keep their own accounts.
"""
from __future__ import annotations

from datetime import date, datetime, timedelta, timezone
from zoneinfo import ZoneInfo

import pytest

from config.settings import settings
from src.performance import live_account as la
from src.performance import sim_account as sa
from src.signals import sel_short as ss

ET = ZoneInfo("America/New_York")
NOW = datetime(2026, 9, 28, 14, 40, tzinfo=timezone.utc)


@pytest.fixture
def on(monkeypatch):
    monkeypatch.setattr(settings, "enable_sel_short", True)
    monkeypatch.setattr(settings, "enable_sel_short_vol", True)
    monkeypatch.setattr(settings, "enable_sel_short_fall", True)
    monkeypatch.setattr(settings, "enable_sel_short_account_sizing", True)
    monkeypatch.setattr(settings, "enable_sel_short_ibkr_whatif", False)
    monkeypatch.setattr(settings, "enable_live_account", True)
    monkeypatch.setattr(settings, "sel_short_traded_arms", "vol,etf,fall")
    monkeypatch.setattr(settings, "sel_short_account_initial", 10_000.0)
    monkeypatch.setattr(settings, "sel_short_account_slices", 24)
    for arm in ("vol", "etf", "fall"):                     # the give-way arithmetic at 1/24 (live: 20 / 16 / 12)
        monkeypatch.setattr(settings, f"sel_short_{arm}_account_slices", 24.0)
    monkeypatch.setattr(settings, "sel_short_account_house_maint", 2.0)
    monkeypatch.setattr(settings, "sel_short_account_house_init", 2.86)
    monkeypatch.setattr(settings, "dip_long_account_slices", 10)
    monkeypatch.setattr(settings, "live_account_delever_trigger", 0.8)
    monkeypatch.setattr(settings, "live_account_delever_target", 0.6)
    return settings


def dip(tk, n=10, px=100.0, mark=None, ret=0.0, at="2026-09-28T13:31:00+00:00"):
    return {"ticker": tk, "entry_mechanism": "dip_long", "action": "BUY", "type": "STOCK", "status": "OPEN",
            "dip_account_shares": n, "entry_price": px, "current_price": mark or px, "return_pct": ret,
            "entry_datetime": at, "entry_date": at[:10], "entry_session": "rth", "position_size_multiplier": 1.0,
            "recommendation_id": f"dip-{tk}", "current_price_datetime": NOW.isoformat()}


def short(tk, n=40, px=10.0, mark=None, ret=0.0, at="2026-09-28T14:00:00+00:00"):
    return {"ticker": tk, "entry_mechanism": "sel_short", "sel_arm": "vol", "action": "SELL", "type": "STOCK",
            "status": "OPEN", "sel_account_shares": n, "entry_price": px, "current_price": mark or px,
            "return_pct": ret, "sel_house_maint": 2.0, "sel_house_init": 2.86, "entry_datetime": at,
            "entry_date": at[:10], "entry_session": "rth", "position_size_multiplier": 1.0,
            "recommendation_id": f"sel-{tk}", "current_price_datetime": NOW.isoformat()}


def test_one_equity_and_one_margin_room_for_both_books(on):
    st = la.state([short("S1", n=10, px=20.0), dip("D1", n=10, px=100.0)], NOW)
    assert st["equity"] == pytest.approx(10_000.0)
    assert st["gross"] == pytest.approx(200.0) and st["long_gross"] == pytest.approx(1_000.0)
    assert st["maint"] == pytest.approx(400.0 + 250.0)            # the short at house 2.00x, the long at Reg T 25%
    assert st["init_req"] == pytest.approx(572.0 + 500.0)         # the short at house 2.86x, the long at Reg T 50%
    pnl = la.state([short("S1", n=10, px=20.0, ret=-50.0), dip("D1", n=10, px=100.0, ret=10.0, mark=110.0)], NOW)
    assert pnl["equity"] == pytest.approx(10_000.0 - 100.0 + 100.0)
    closed = la.state([dict(dip("D2", ret=5.0), status="CLOSED")], NOW)
    assert closed["realized"] == pytest.approx(50.0) and closed["long_open"] == 0


def _crowded_book():
    """9 dip longs of $1,000 (initial $4,500) and 4 shorts of $400 (initial $4,576): $924 of room left."""
    dips = [dip(f"D{k}", at=f"2026-09-28T13:3{k}:00+00:00") for k in range(9)]
    shorts = [short(f"S{k}") for k in range(4)]
    return dips + shorts


def test_a_short_is_sized_from_the_shared_room(on):
    book = _crowded_book()
    st = la.state(book, NOW)
    assert st["equity"] - st["init_req"] == pytest.approx(10_000.0 - 4_500.0 - 4 * 400.0 * 2.86)
    alone, _, _ = sa.size([t for t in book if t["entry_mechanism"] == "sel_short"], 10.0, 5e7, NOW)
    shared, why, _ = sa.size(book, 10.0, 5e7, NOW, st=st)
    assert alone == 41 and why == "ok" and 0 < shared < alone      # the dip longs' requirement shrinks the room


def test_the_dip_book_gives_way_whole_latest_first_only_as_needed(on):
    book = _crowded_book()
    plan, want, _ = la.give_way_plan(book, 10.0, 5e7, NOW)
    assert want == 41 and [t["ticker"] for t in plan] == ["D8"]     # one long, the latest, frees enough
    after = [t for t in book if t is not plan[0]]
    assert sa.size(after, 10.0, 5e7, NOW, st=la.state(after, NOW))[0] == want
    roomy = [dip("D0")] + [short("S0")]
    assert la.give_way_plan(roomy, 10.0, 5e7, NOW)[0] == []         # room enough: nothing sold
    # 3 more shorts: the initial requirement ($12,508) is past the equity — 8 longs ($4,000 of room) are needed
    tight = _crowded_book() + [short(f"T{k}") for k in range(3)]
    plan2, _, _ = la.give_way_plan(tight, 10.0, 5e7, NOW)
    assert [t["ticker"] for t in plan2] == [f"D{k}" for k in (8, 7, 6, 5, 4, 3, 2, 1)]
    rest = [t for t in tight if t not in plan2]
    assert sa.size(rest, 10.0, 5e7, NOW, st=la.state(rest, NOW))[0] == want
    seven = [t for t in tight if t not in plan2[:7]]
    assert sa.size(seven, 10.0, 5e7, NOW, st=la.state(seven, NOW))[0] < want    # one long fewer is not enough


def test_the_dip_book_never_gives_way_for_nothing(on):
    """PREREG64: when the shorts' own initial requirement is past the equity, selling every dip long still leaves a
    new short unfunded — the dip book keeps its longs (the plan used to sell them all, for nothing). Sales that fund
    even part of the short still go ahead, the latest long first."""
    dips = [dip(f"D{k}", at=f"2026-09-28T13:3{k}:00+00:00") for k in range(3)]
    shorts = [short(f"S{k}") for k in range(9)]                     # $3,600 of shorts: $10,296 initial
    book = dips + shorts
    st = la.state(book, NOW)
    assert st["maint"] < 0.8 * st["equity"]                         # no de-leveraging, no margin call
    assert sa.size(book, 10.0, 5e7, NOW, st=st)[0] == 0
    plan, want, _ = la.give_way_plan(book, 10.0, 5e7, NOW)
    assert want == 41 and plan == []
    without = shorts
    assert sa.size(without, 10.0, 5e7, NOW, st=la.state(without, NOW))[0] == 0   # selling all three buys nothing
    part = dips + shorts[:8]                                        # one short fewer: the longs free 29 shares
    plan2, _, _ = la.give_way_plan(part, 10.0, 5e7, NOW)
    assert [t["ticker"] for t in plan2] == ["D2", "D1", "D0"]
    rest = shorts[:8]
    assert 0 < sa.size(rest, 10.0, 5e7, NOW, st=la.state(rest, NOW))[0] < 41


def test_a_dip_long_is_sized_from_the_shared_account(on):
    n, why, st = la.size_long([short(f"S{k}") for k in range(4)], 100.0)
    assert why == "ok" and n == 10                                  # 1/10 of $10,000 at $100
    tight = [short(f"S{k}", n=400) for k in range(2)]               # $8,000 of shorts: $22,880 initial — no room
    assert la.size_long(tight, 100.0)[:2] == (0, "account_full")
    full = [dip(f"D{k}") for k in range(10)]
    assert la.size_long(full, 100.0)[1] == "no_slot"
    side = [dip(f"D{k}", n=99) for k in range(1)] + [dip("E", n=1, px=50.0)]   # $9,950 of longs: the side's cap
    assert la.size_long(side, 100.0)[1] == "account_full"
    assert la.size_long([], 20_000.0)[1] == "account_full"          # a slice buys no share... nor the room


def test_deleveraging_sells_the_latest_dip_longs_until_60_percent(on):
    book = [dip(f"D{k}", at=f"2026-09-28T13:3{k}:00+00:00") for k in range(9)]
    book += [short(f"S{k}", n=40, px=10.0, mark=20.0, ret=-100.0) for k in range(3)]   # squeezed: -$1,200
    plan, st = la.delever_plan(book, NOW)
    assert st["equity"] == pytest.approx(8_800.0)
    assert st["maint"] == pytest.approx(3 * 1_600.0 + 9 * 250.0)      # 7,050 = 80.1% of the equity
    assert [t["ticker"] for t in plan] == [f"D{k}" for k in (8, 7, 6, 5, 4, 3, 2, 1)]   # down to <= 60%
    calm = book[:9] + [short(f"S{k}") for k in range(3)]
    assert la.delever_plan(calm, NOW)[0] == []


def _ledger(monkeypatch, trades, price=9.8):
    from src.data import ibkr_borrow
    from src.db import repo
    from src.performance import tracker
    repo.save_trades(list(trades), allow_shrink=True)               # each scenario replaces the test ledger
    monkeypatch.setattr(tracker, "_fetch_price", lambda t: price)
    monkeypatch.setattr(tracker, "_reference_close", lambda t: None)
    monkeypatch.setattr(tracker, "_execution_iso", lambda: "2026-09-28T14:40:00+00:00")
    stamp = datetime(2026, 9, 28, 9, 0, tzinfo=ET)
    monkeypatch.setattr(ibkr_borrow, "short_block",
                        lambda ticker, price, base=None, now=None, max_fee_pct="default":
                        (None, ibkr_borrow.Borrow(ticker, 12.0, -5.0, 900_000, stamp)))
    return tracker


def _pick(tk, arm="fall", day=date(2026, 9, 28), bar=1):
    return {"day": day.isoformat(), "bar_of_day": bar, "bar_end": ss.bar_end_et(day, bar).isoformat(),
            "ticker": tk, "decision": "short", "px": 10.0, "pre5": 12.0, "target": None, "arm": arm,
            "deadline": ss.bar_end_et(ss.session_after(day, 15), bar).isoformat(), "score": 6.5,
            "runup_pct": -16.7, "days_to_cover": 0.5, "rvol": 2.0, "atr_pct": 6.5, "dv20": 5e7}


def test_the_entry_step_sells_the_latest_dip_long_and_opens_the_whole_short(monkeypatch, on):
    tracker = _ledger(monkeypatch, _crowded_book())
    monkeypatch.setattr(ss, "pending_entries", lambda now=None: [_pick("NEW")])
    assert tracker.record_sel_short_trades(run_id="r1", now=NOW) == 1
    t = {x["ticker"]: x for x in tracker._load_trades()}
    assert t["D8"]["status"] == "CLOSED" and t["D8"]["exit_reason"] == "dip_gave_way"
    assert all(t[f"D{k}"]["status"] == "OPEN" for k in range(8))
    want = sa.size([], 9.8, 5e7, NOW)[0]                            # 1/24 of $10,000 at $9.80
    assert t["NEW"]["sel_account_shares"] == want and t["NEW"]["sel_arm"] == "fall"


def test_the_account_check_deleverages_and_a_margin_call_closes_both_books(monkeypatch, on):
    book = [dip(f"D{k}", at=f"2026-09-28T13:3{k}:00+00:00") for k in range(9)]
    book += [short(f"S{k}", n=40, px=10.0, mark=20.0, ret=-100.0) for k in range(3)]
    tracker = _ledger(monkeypatch, book)
    assert tracker.monitor_live_account(now=NOW) == 8
    t = {x["ticker"]: x for x in tracker._load_trades()}
    assert [k for k in range(9) if t[f"D{k}"]["status"] == "CLOSED"] == [1, 2, 3, 4, 5, 6, 7, 8]
    assert {t[f"D{k}"]["exit_reason"] for k in range(1, 9)} == {"dip_delever"}
    assert all(t[f"S{k}"]["status"] == "OPEN" for k in range(3))
    # squeezed past the equity: every short and long the account funded closed
    worse = [dip("D0")] + [short(f"S{k}", n=40, px=10.0, mark=40.0, ret=-300.0) for k in range(3)]
    tracker = _ledger(monkeypatch, worse)
    assert tracker.monitor_live_account(now=NOW) == 4
    t = {x["ticker"]: x for x in tracker._load_trades()}
    assert t["D0"]["exit_reason"] == "dip_margin_call"
    assert {t[f"S{k}"]["exit_reason"] for k in range(3)} == {"sel_margin_call"}


def test_the_dip_entry_step_sizes_from_the_shared_account(monkeypatch, on):
    from src.signals import dip_long
    tracker = _ledger(monkeypatch, [short(f"S{k}") for k in range(4)], price=100.0)
    monkeypatch.setattr(settings, "enable_dip_long", True)
    monkeypatch.setattr(dip_long, "in_entry_window", lambda now=None: True)
    monkeypatch.setattr(dip_long, "today_et", lambda now=None: date(2026, 9, 28))
    sig = {"ticker": "MEGA", "close": 99.0, "sma_trend": 90.0, "rsi2": 4.0, "dv20": 2e9}
    monkeypatch.setattr(dip_long, "signals_for", lambda d: {"signals": [sig], "signal_session": "2026-09-25"})
    assert tracker.record_dip_long_trades(run_id="r1", now=NOW) == 1
    t = next(x for x in tracker._load_trades() if x["ticker"] == "MEGA")
    assert t["dip_account_shares"] == la.size_long([short(f"S{k}") for k in range(4)], 100.0)[0] == 10
    assert t["dip_live_account"] is True and t["dip_account_cash"] is None


def test_off_the_two_books_keep_their_own_accounts(monkeypatch, on):
    monkeypatch.setattr(settings, "enable_live_account", False)
    assert not la.enabled()
    tracker = _ledger(monkeypatch, _crowded_book())
    assert tracker.monitor_live_account(now=NOW) == 0
    monkeypatch.setattr(ss, "pending_entries", lambda now=None: [_pick("NEW")])
    tracker.record_sel_short_trades(run_id="r1", now=NOW)
    t = {x["ticker"]: x for x in tracker._load_trades()}
    assert all(t[f"D{k}"]["status"] == "OPEN" for k in range(9))     # no dip long sold for a short
