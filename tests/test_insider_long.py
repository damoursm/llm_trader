"""THE INSIDER BOOK (PREREG203's O1_20; user 2026-10-24: "Ok. Deploy the Opportunistic, SPY 20%+ down to the live
combine."; `src/signals/insider_long.py`).

What must hold: the gate is SPY's close at D-1 at least 20% under its highest close of the 252 sessions through it,
never a guess over a missing session; an EVENT is one company's officer / director open-market purchases (Form 4 /
4/A, code P, acquired, a price, a non-derivative title) filed on one date, a transaction reported twice counted at its
first report, only the filings made before the session read; Cohen, Malloy & Pomorski's classes from the insider's
trades filed before the purchase (routine = the purchase's month in each of the three calendar years before,
opportunistic = traded in each of them otherwise); on a gate session every opportunistic event of the filing window
(from the session before through the eve) whose stock closed at $5+, traded $5M a day over its last 20 bars and holds 61
bars in its run is bought at the open, the largest purchase first, 1/10 of the live account within the room, at most 1%
of its dollar volume, at most 5 open, sold at the open after 60 sessions; its capital gives way FIRST — to the shorts,
the crash book, the dip and capitulation fills — and de-leverages with the capitulation book's, the latest first; the
broker orders its share count; the simulator builds the same book (production's code) and prices each long in its entry
session's terms.
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
from src.signals import dip_long as dl
from src.signals import insider_long as il

ET = ZoneInfo("America/New_York")
DAY = date(2026, 10, 8)                        # a Thursday session; D-1 = 2026-10-07
PREV = date(2026, 10, 7)


def at(d, h, m, s=20):
    return datetime(d.year, d.month, d.day, h, m, s, tzinfo=ET).astimezone(timezone.utc)


OPEN_TICK = at(DAY, 9, 30)


def _sessions_before(end: date, n: int):
    out, d = [], end - timedelta(days=1)
    while len(out) < n:
        if dl.is_session(d):
            out.append(d)
        d -= timedelta(days=1)
    return out[::-1]


# ── the gate ─────────────────────────────────────────────────────────────────

def _closes(values, end=PREV):
    days = _sessions_before(end + timedelta(days=1), len(values))
    return pd.Series(values, index=pd.DatetimeIndex(days), dtype=float)


def test_the_gate_is_spy_twenty_percent_under_its_252_session_high():
    up = list(np.linspace(300.0, 400.0, 200))
    g = il.gate_from_closes(_closes(up + list(np.linspace(399.0, 319.0, 60))), PREV)   # 319 / 400 = 20.25% under
    assert g["on"] and g["error"] is None and g["high"] == pytest.approx(400.0) and g["drawdown"] > 0.20
    g = il.gate_from_closes(_closes(up + list(np.linspace(399.0, 330.0, 60))), PREV)   # 17.5% under
    assert not g["on"] and g["drawdown"] == pytest.approx(1 - 330.0 / 400.0)
    old_high = [500.0] + up + list(np.linspace(399.0, 330.0, 60))                        # 261 closes: 500 is older
    assert il.gate_from_closes(_closes(old_high), PREV)["high"] == pytest.approx(400.0)  # than 252 sessions


def test_the_gate_never_guesses_without_its_close_or_its_history():
    s = _closes(list(np.linspace(300.0, 400.0, 260)))
    assert il.gate_from_closes(s.iloc[:-1], PREV)["error"]                   # no close at D-1
    assert il.gate_from_closes(s.iloc[-150:], PREV)["error"]                 # fewer than 200 sessions


def test_a_missing_whole_market_session_is_an_error_not_a_guess(monkeypatch):
    frames = [(d, pd.DataFrame({"sym": ["SPY"], "o": [1.0], "h": [1.0], "l": [1.0], "c": [300.0], "v": [1.0]}))
              for d in _sessions_before(DAY, 252)]
    monkeypatch.setattr(il, "_frames", lambda d, n, fetch=True: (frames[1:], [frames[0][0].isoformat()]))
    assert il.gate_at(DAY)["error"]


# ── the filings ──────────────────────────────────────────────────────────────

def row(owner, tk, fd, td, code="P", shares=1000.0, price=10.0, form="4", title="Common Stock", officer=True,
        director=False, acq=None, issuer="0001"):
    return {"ticker": tk, "issuer_cik": issuer, "form": form, "filing_date": fd, "trans_date": td,
            "trans_code": code, "acq_disp": acq or ("A" if code == "P" else "D"), "shares": shares, "price": price,
            "shares_owned_after": 5000.0, "security_title": title, "is_director": director, "is_officer": officer,
            "owner_cik": owner, "owner_title": "CEO", "aff_10b5_1": None}


def rows(*rr):
    return il.clean_rows(pd.DataFrame(list(rr)))


def _history(owner, years, month, tk="OLD", fd_pad=5):
    """An insider's sales in ``month`` of each of ``years`` (filed ``fd_pad`` days after)."""
    out = []
    for y in years:
        td = date(y, month, 10)
        out.append(row(owner, tk, (td + timedelta(days=fd_pad)).isoformat(), td.isoformat(), code="S"))
    return out


def test_the_classes_are_cohen_malloy_pomorskis_point_in_time():
    R = rows(*_history("O", (2023, 2024, 2025), 3), *_history("R", (2023, 2024, 2025), 10),
             *_history("N", (2024, 2025), 10),
             row("O", "AAA", "2026-10-07", "2026-10-06"), row("R", "BBB", "2026-10-07", "2026-10-06"),
             row("N", "CCC", "2026-10-07", "2026-10-06"))
    E = il.events_from_rows(R, PREV, PREV, DAY).set_index("sym")
    assert E.loc["AAA", "n_opp"] == 1 and E.loc["AAA", "opp"]                # traded in 3 years, never in October
    assert E.loc["BBB", "n_rout"] == 1 and not E.loc["BBB", "opp"]           # every October: routine
    assert E.loc["CCC", "n_opp"] == 0 and not E.loc["CCC", "opp"]            # two years only: neither class
    fd = np.array(["2026-10-07"], dtype="datetime64[ns]")
    assert il.klass(fd, np.array(["2023-03-01"], dtype="datetime64[ns]"), "2026-10-06", "2026-10-07") == "na"


def test_only_trades_filed_before_the_purchase_classify_it():
    late = [row("O", "OLD", "2026-10-07", "2025-03-10", code="S")]           # its 2025 trade filed WITH the purchase
    R = rows(*_history("O", (2023, 2024), 3), *late, row("O", "AAA", "2026-10-07", "2026-10-06"))
    assert il.events_from_rows(R, PREV, PREV, DAY).iloc[0]["n_opp"] == 0


def test_an_event_is_a_companys_purchases_filed_on_one_date_its_notional_summed():
    hist = _history("O1", (2023, 2024, 2025), 3) + _history("O2", (2023, 2024, 2025), 4)
    R = rows(*hist, row("O1", "AAA", "2026-10-07", "2026-10-05", shares=1000, price=20.0),
             row("O2", "AAA", "2026-10-07", "2026-10-06", shares=500, price=21.0),
             row("O1", "AAA", "2026-10-07", "2026-10-06", code="S"),                  # a sale: no purchase
             row("O1", "AAA", "2026-10-07", "2026-10-06", shares=10, title="Stock Option (right to buy)"),
             row("O1", "AAA", "2026-10-07", "2026-10-06", shares=10, price=0.0),      # no price: no purchase
             row("O1", "AAA", "2026-10-07", "2026-10-06", shares=10, form="5"),       # a Form 5: not read
             row("O1", "BRK/B", "2026-10-07", "2026-10-06", shares=1, price=500.0))
    E = il.events_from_rows(R, PREV, PREV, DAY).set_index("sym")
    assert E.loc["AAA", "notional"] == pytest.approx(1000 * 20.0 + 500 * 21.0) and E.loc["AAA", "buyers"] == 2
    assert E.loc["AAA", "n_opp"] == 2 and "BRK.B" in E.index and il.project_ticker("BRK.B") == "BRK-B"


def test_an_amendment_repeating_its_original_is_counted_at_its_first_report():
    hist = _history("O", (2023, 2024, 2025), 3)
    orig = row("O", "AAA", "2026-10-02", "2026-10-01", shares=1000, price=20.0)
    amend = dict(row("O", "AAA", "2026-10-07", "2026-10-01", shares=1000, price=20.0), form="4/A")
    R = rows(*hist, orig, amend)
    assert il.events_from_rows(R, PREV, PREV, DAY).empty                     # the 4/A repeats a reported purchase


def test_a_row_filed_on_or_after_the_session_is_never_read():
    hist = _history("O", (2023, 2024, 2025), 3)
    R = rows(*hist, row("O", "AAA", "2026-10-08", "2026-10-07"))             # filed on the session itself
    assert il.events_from_rows(R, PREV, DAY, DAY).empty


def test_the_filing_window_runs_from_the_session_before_to_the_eve():
    assert il.filing_window(DAY) == (PREV, PREV)
    mon = date(2026, 10, 12)                                                 # a Monday (Columbus Day: NYSE open)
    assert il.filing_window(mon) == (date(2026, 10, 9), date(2026, 10, 11))
    assert il.filing_window(date(2026, 4, 6)) == (date(2026, 4, 2), date(2026, 4, 5))   # after Good Friday


# ── the bars and the rule ────────────────────────────────────────────────────

def bars_frame(closes, end=PREV, vol=1e6, skip=()):
    days = _sessions_before(end + timedelta(days=1), len(closes))
    keep = [k for k in range(len(days)) if k not in set(skip)]
    return pd.DataFrame({"o": [closes[k] for k in keep], "h": [closes[k] * 1.01 for k in keep],
                         "l": [closes[k] * 0.99 for k in keep], "c": [closes[k] for k in keep],
                         "v": [vol] * len(keep)}, index=pd.DatetimeIndex([days[k] for k in keep]))


def test_the_stocks_numbers_are_its_runs_last_close_dollar_volume_and_length():
    st = il.bar_stats(bars_frame([10.0] * 80), DAY)
    assert st["bars"] == 80 and st["close"] == 10.0 and st["dv20"] == pytest.approx(1e7) and st["session"] == "2026-10-07"
    gap = bars_frame([10.0] * 120, skip=range(40, 70))                      # 30 sessions without a bar: > 30 days
    assert il.bar_stats(gap, DAY)["bars"] == 50
    stale = bars_frame([10.0] * 80, end=date(2026, 8, 20))                  # its last bar 48 days before
    assert il.bar_stats(stale, DAY)["bars"] == 0                             # a bar on D would start a new run


def test_the_rule_needs_an_opportunistic_event_5_dollars_5_million_and_61_bars():
    ev = {"opp": True, "n_opp": 1, "n_rout": 0}
    ok = {"bars": 61, "close": 5.0, "dv20": 5e6}
    assert il.qualifies(ev, ok) == (True, "ok")
    assert il.qualifies(ev, dict(ok, bars=60))[1] == "history"
    assert il.qualifies(ev, dict(ok, close=4.99))[1] == "price"
    assert il.qualifies(ev, dict(ok, dv20=4.9e6))[1] == "liquidity"
    assert il.qualifies(dict(ev, opp=False, n_rout=1), ok)[1] == "routine"
    assert il.qualifies(ev, None)[1] == "no_bars"


def test_the_day_buys_the_largest_purchase_first_only_on_a_gate_session():
    E = pd.DataFrame([{"sym": "AAA", "fd": pd.Timestamp(PREV), "issuer": "1", "notional": 1e5, "buyers": 1,
                       "td": pd.Timestamp(PREV), "n_opp": 1, "n_rout": 0, "opp": True},
                      {"sym": "BBB", "fd": pd.Timestamp(PREV), "issuer": "2", "notional": 5e6, "buyers": 2,
                       "td": pd.Timestamp(PREV), "n_opp": 1, "n_rout": 0, "opp": True},
                      {"sym": "CCC", "fd": pd.Timestamp(PREV), "issuer": "3", "notional": 9e6, "buyers": 1,
                       "td": pd.Timestamp(PREV), "n_opp": 1, "n_rout": 0, "opp": True}])
    b = {"AAA": bars_frame([20.0] * 70), "BBB": bars_frame([30.0] * 70), "CCC": bars_frame([3.0] * 70)}
    sig, rej = il.decide(DAY, {"on": True}, E, b)
    assert [s["ticker"] for s in sig] == ["BBB", "AAA"] and [s["rank"] for s in sig] == [1, 2]
    assert rej == {"price": ["CCC"]}
    assert il.decide(DAY, {"on": False}, E, b) == ([], {})


def test_the_long_leaves_at_the_open_after_60_sessions():
    from src.signals.sel_short import session_after
    t = {"ticker": "AAA", "ins_entry_day": "2026-07-14"}
    sell_day = session_after(date(2026, 7, 14), 60)
    assert il.exit_check(t, sell_day)[0] == "ins_time" and il.exit_check(t, sell_day)[1]["held"] == 60
    assert il.exit_check(t, dl.prev_session(sell_day))[0] is None


# ── the account ──────────────────────────────────────────────────────────────

@pytest.fixture
def acct(monkeypatch):
    for k, v in {"enable_sel_short": True, "enable_sel_short_account_sizing": True,
                 "enable_sel_short_ibkr_whatif": False, "enable_live_account": True,
                 "sel_short_account_initial": 10_000.0, "dip_long_account_slices": 2,
                 "cap_long_account_slices": 6.0, "cap_long_max_open": 3, "crash_long_account_slices": 3.0,
                 "crash_long_margin_rate": 0.75, "live_account_delever_trigger": 0.7,
                 "live_account_delever_target": 0.5, "sel_short_account_house_maint": 2.0,
                 "sel_short_account_house_init": 2.86, "insider_long_account_slices": 10.0,
                 "insider_long_max_open": 5}.items():
        monkeypatch.setattr(settings, k, v)


def long_(mech, tk, n=10, px=100.0, at_iso="2026-10-07T13:31:00+00:00", ret=0.0):
    pre = {"dip_long": "dip", "cap_long": "cap", "insider_long": "ins"}[mech]
    return {"ticker": tk, "entry_mechanism": mech, "action": "BUY", "type": "STOCK", "status": "OPEN",
            f"{pre}_account_shares": n, "entry_price": px, "current_price": px, "return_pct": ret,
            "entry_datetime": at_iso, "entry_date": at_iso[:10], "entry_session": "rth", "position_size_multiplier": 1.0,
            "recommendation_id": f"{pre}-{tk}", "current_price_datetime": OPEN_TICK.isoformat()}


def ins(tk, **kw):
    return long_("insider_long", tk, **kw)


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


def test_the_insider_longs_count_in_the_account_with_their_own_slots(acct):
    st = la.state([ins("I1"), cap("C1")], OPEN_TICK)
    assert st["ins_open"] == 1 and st["cap_open"] == 1 and st["ins_gross"] == pytest.approx(1_000.0)
    assert st["init_req"] == pytest.approx(1_000.0) and st["maint"] == pytest.approx(500.0)   # Reg T on both
    assert la.long_book(ins("I1")) == "ins" and la.ins_funded(ins("I1")) and not la.ins_funded(cap("C1"))


def test_each_long_is_a_tenth_at_most_five_open_and_one_percent_of_its_dollar_volume(acct):
    assert la.size_insider([], 50.0, 1e9, OPEN_TICK)[:2] == (20, "ok")      # $10,000 / 10 / $50
    full = [ins(f"I{k}") for k in range(5)]
    assert la.size_insider(full, 50.0, 1e9, OPEN_TICK)[:2] == (0, "no_slot")
    assert la.size_insider([], 50.0, 50_000.0, OPEN_TICK)[:2] == (10, "ok")  # 1% of $50,000 = $500
    assert la.size_insider([], 50.0, 4_000.0, OPEN_TICK)[:2] == (0, "volume_cap")


def test_the_insider_longs_give_way_first_to_a_short_the_latest_first(acct):
    held = [dip("D1", n=40), cap("C1", n=10, at_iso="2026-10-07T13:32:00+00:00"),
            ins("I1", n=10, at_iso="2026-10-07T13:33:00+00:00"), ins("I2", n=10, at_iso="2026-10-07T13:34:00+00:00")]
    order = [t["ticker"] for t in la._give_way_longs(held)]
    assert order == ["I2", "I1", "C1", "D1"]                                 # insider, capitulation, dip
    plan, want, _ = la.give_way_plan(held + [short("HOT", 150)], 10.0, None, OPEN_TICK, slices=1)
    assert [t["ticker"] for t in plan][:2] == ["I2", "I1"]


def test_the_crash_buy_sells_the_insider_longs_first(acct):
    held = [dip("D1", n=40), cap("C1", n=20), ins("I1", n=20)]
    plan, want, _ = la.crash_give_way_plan(held, 50.0, OPEN_TICK)
    assert want > 0 and [t["ticker"] for t in plan][0] == "I1"


def test_a_capitulation_order_is_sized_as_if_the_insider_longs_were_sold_and_its_fill_sells_them(acct):
    held = [ins("I1", n=50, at_iso="2026-10-07T13:31:00+00:00"), ins("I2", n=40, at_iso="2026-10-07T13:32:00+00:00")]
    assert la.size_cap(held, 100.0, OPEN_TICK)[:2] == (16, "ok")            # the 1/6 slice as if they were sold
    assert la.fill_cap(held, 100.0, OPEN_TICK)[0] == 10                      # the room as it stands
    assert la.fill_cap(held, 100.0, OPEN_TICK, free_cap=True, book="cap")[0] == 100
    plan, _ = la.fill_give_way_plan(held, 100.0, 16, OPEN_TICK, book="cap")
    assert [t["ticker"] for t in plan] == ["I2"]                             # the latest, whole
    # a capitulation fill never sells a dip long, nor another capitulation long
    assert la.fill_give_way_plan([dip("D1", n=90), cap("C1", n=5)], 100.0, 50, OPEN_TICK, book="cap")[0] == []


def test_a_dip_fill_sells_the_insider_longs_before_the_capitulation_longs(acct):
    held = [cap("C1", n=40, at_iso="2026-10-07T13:35:00+00:00"), ins("I1", n=50, at_iso="2026-10-07T13:31:00+00:00")]
    plan, _ = la.fill_give_way_plan(held, 100.0, 50, OPEN_TICK)
    assert [t["ticker"] for t in plan] == ["I1"]
    plan, _ = la.fill_give_way_plan(held, 100.0, 90, OPEN_TICK)
    assert [t["ticker"] for t in plan] == ["I1", "C1"]
    assert la.size_long(held, 100.0, OPEN_TICK)[:2] == (50, "ok")            # sized as if both were freed


def test_deleveraging_sells_the_insider_and_capitulation_longs_together_the_latest_first(acct):
    held = [dip("D1", n=40, at_iso="2026-10-07T13:40:00+00:00"), cap("C1", n=20, at_iso="2026-10-07T13:35:00+00:00"),
            ins("I1", n=20, at_iso="2026-10-07T13:31:00+00:00"), short("HOT", 300, px=10.0)]
    assert la.state(held, OPEN_TICK)["maint"] > 0.7 * la.state(held, OPEN_TICK)["equity"]
    names = [t["ticker"] for t in la.delever_plan(held, OPEN_TICK)[0]]
    assert names[:2] == ["C1", "I1"]                                         # the engine's delever_order 0 for both


def test_a_margin_call_counts_the_insider_longs(acct):
    called, st = la.margin_call([ins("I1", n=300, px=100.0, ret=-50.0)], OPEN_TICK)
    assert called and st["ins_open"] == 1


# ── the ledger steps ─────────────────────────────────────────────────────────

def _sig(tk, notional, fd="2026-10-07", dv20=1e8):
    return {"ticker": tk, "sym": tk, "fd": fd, "issuer": "1", "notional": notional, "buyers": 1, "n_opp": 1,
            "n_rout": 0, "td": fd, "session": "2026-10-07", "close": 50.0, "dv20": dv20, "bars": 252}


@pytest.fixture
def book(monkeypatch, tmp_path, acct):
    for k, v in {"enable_insider_long": True, "insider_long_dir": str(tmp_path / "insider_long")}.items():
        monkeypatch.setattr(settings, k, v)
    monkeypatch.setattr(tracker, "_reference_close", lambda t: None)
    monkeypatch.setattr(tracker, "_execution_iso", lambda: OPEN_TICK.isoformat())
    day = {"day": DAY.isoformat(), "signal_session": "2026-10-07",
           "gate": {"on": True, "drawdown": 0.23, "symbol": "SPY"},
           "signals": [_sig("AAA", 2e6), _sig("BBB", 9e6)]}
    monkeypatch.setattr(il, "signals_for", lambda d: day if d == DAY else None)
    prices = {"AAA": 50.0, "BBB": 40.0}
    monkeypatch.setattr(tracker, "_fetch_price", lambda tk: prices.get(tk))
    return {"day": day, "prices": prices}


def insiders():
    return {t["ticker"]: t for t in tracker._load_trades() if t.get("entry_mechanism") == books.INSIDER_LONG}


def test_the_open_buys_each_signal_largest_first_at_a_tenth(book):
    assert tracker.record_insider_long_trades(now=OPEN_TICK) == 2
    t = insiders()
    assert t["BBB"]["ins_account_shares"] == 10_000 // 10 // 40 and t["AAA"]["ins_account_shares"] == 20
    assert [r["ticker"] for r in dl._read_jsonl(il.entries_path(DAY))] == ["BBB", "AAA"]   # the largest first
    assert t["AAA"]["ins_filing_date"] == "2026-10-07" and t["AAA"]["ins_entry_day"] == "2026-10-08"
    assert t["AAA"]["target_horizon"] == "60d" and books.is_live(t["AAA"])
    assert tracker.record_insider_long_trades(now=at(DAY, 10, 0)) == 0      # settled once a day
    assert {r["outcome"] for r in dl._read_jsonl(il.entries_path(DAY))} == {"opened"}


def test_a_missing_price_is_retried_and_another_books_name_is_held(book):
    tracker._save_trades([dict(cap("AAA"), current_price=50.0)])
    book["prices"]["BBB"] = None
    assert tracker.record_insider_long_trades(now=OPEN_TICK) == 0
    out = {r["ticker"]: r["outcome"] for r in dl._read_jsonl(il.entries_path(DAY))}
    assert out == {"AAA": "held", "BBB": "no_price"}
    book["prices"]["BBB"] = 40.0
    assert tracker.record_insider_long_trades(now=at(DAY, 10, 0)) == 1      # BBB retried, AAA never again
    assert set(insiders()) == {"BBB"}


def test_no_buy_outside_the_window_or_without_the_switch(book, monkeypatch):
    assert tracker.record_insider_long_trades(now=at(DAY, 11, 0)) == 0
    monkeypatch.setattr(settings, "enable_insider_long", False)
    assert tracker.record_insider_long_trades(now=OPEN_TICK) == 0


def test_the_open_sells_the_long_held_60_sessions(book, monkeypatch):
    from src.signals.sel_short import session_after
    entry = date(2026, 7, 14)
    tracker._save_trades([dict(ins("AAA"), ins_entry_day=entry.isoformat(), entry_date=entry.isoformat())])
    sell_day = session_after(entry, 60)
    monkeypatch.setattr(tracker, "_fetch_price", lambda tk: 61.0)
    assert tracker.monitor_insider_long_positions(now=at(sell_day, 9, 30)) == 1
    t = insiders()["AAA"]
    assert t["status"] == "CLOSED" and t["exit_reason"] == "ins_time" and t["exit_price"] == 61.0


def test_a_short_sells_the_insider_long_as_ins_gave_way_journaled_in_its_book(book):
    assert tracker._long_give_way_reason(ins("I1"), "gave_way") == ("ins_gave_way", "insider_long")
    assert tracker._long_give_way_reason(ins("I1"), "margin_call") == ("ins_margin_call", "insider_long")
    t = dict(ins("I1"), status="CLOSED", exit_price=99.0)
    tracker._journal_long_exit(DAY, t, "ins_gave_way", {"price": 99.0})
    assert dl._read_jsonl(il.exits_path(DAY))[0]["reason"] == "ins_gave_way"


def test_deleveraging_closes_the_insider_longs_as_ins_delever(book, monkeypatch):
    tracker._save_trades([ins("I1", n=30), dip("D1", n=50, at_iso="2026-10-07T13:40:00+00:00"),
                          short("HOT", 300, px=10.0)])                         # maintenance $8,000 > 70% of $10,000
    monkeypatch.setattr(tracker, "_execution_iso", lambda: at(DAY, 10, 0).isoformat())
    assert tracker.monitor_live_account(now=at(DAY, 10, 0)) >= 1
    assert {x["ticker"]: x for x in tracker._load_trades()}["I1"]["exit_reason"] == "ins_delever"


def test_the_broker_orders_its_share_count_and_the_book_is_live():
    from src.broker import reconcile as rec
    assert rec._entry_qty(ins("I1", n=7), 100.0, 1e6, 1.0) == 7
    assert books.INSIDER_LONG in books.LIVE_MECHANISMS and books.is_live(ins("I1"))
    assert "ins_account_shares" in tracker.SPLIT_SHARE_FIELDS and "ins_close" in tracker.SPLIT_PRICE_FIELDS
    assert "ins_account_shares" in tracker._ACCOUNT_SHARE_KEYS


def test_the_second_account_never_trades_it_unless_named(monkeypatch):
    from src.accounts import profiles
    monkeypatch.setattr(settings, "second_account_books", "dip,fall,crash")
    assert profiles.env_for("second")["ENABLE_INSIDER_LONG"] == "false"
    monkeypatch.setattr(settings, "second_account_books", "dip,ins")
    assert profiles.env_for("second")["ENABLE_INSIDER_LONG"] == "true"


# ── the simulator ────────────────────────────────────────────────────────────

def test_the_simulator_runs_the_book_as_production_does(monkeypatch):
    from src.backtest import mimic as M
    monkeypatch.setattr(settings, "enable_insider_long", True)
    m = M.Mimic()
    names = {s.name: s for s in M.live_strategies(m)}
    s = names["ins"]
    assert s.side == 1 and s.slices == 10.0 and s.max_open == 5 and s.quality == 0.05 and s.adv_cap == 0.01
    assert s.preemptible and s.yield_whole and s.yield_first == "latest" and s.delever_order == 0
    assert s.orders is None and s.house_init == 0.0 and s.quality < M.QUALITY["cap"] < M.QUALITY["dip"]
    assert M.live_rules(m).fills_make_room and M.live_rules(m).fill_room_exact and M.BOOK["ins"] == books.INSIDER_LONG
    assert "ins" not in {x.name for x in M.live_strategies(M.Mimic(**M.PRE205))} and M.Mimic.off().ins is False
    monkeypatch.setattr(settings, "enable_insider_long", False)
    assert "ins" not in {x.name for x in M.live_strategies(M.Mimic())}


def test_the_adapter_prices_each_long_from_the_open_to_the_open_after_60_sessions_in_entry_terms(monkeypatch):
    from src.backtest.strategies import insider as INS
    from src.data import intraday_store
    from src.signals import cap_long
    from src.signals.sel_short import session_after
    d = date(2026, 3, 2)
    sess = [d] + [session_after(d, k) for k in range(1, 61)]
    split_day = sess[30]

    def frame(s, fetch=True):
        if s not in sess:
            return None
        k = sess.index(s)
        px = 20.0 + k * 0.1
        if s >= split_day:                                                   # a 2-for-1 split: quoted at half
            px /= 2.0
        return pd.DataFrame({"sym": ["XYZ"], "o": [px], "h": [px * 1.02], "l": [px * 0.98], "c": [px + 0.05],
                             "v": [1e6]})
    monkeypatch.setattr(cap_long, "_session_frame", frame)
    monkeypatch.setattr(intraday_store, "split_rows", lambda: {"XYZ": [(split_day, 1.0, 2.0)]})
    p = INS.piece(d, {"ticker": "XYZ", "notional": 1e6, "dv20": 2e7, "fd": "2026-02-27"}, INS._Bars(fetch=False))
    assert p["e"] == pytest.approx(20.0) and p["x"] == pytest.approx(20.0 + 60 * 0.1)   # entry terms throughout
    assert p["ens"] == INS._ns(d, "09:30") and p["xns"] == INS._ns(sess[60], "09:30")
    assert np.all(np.diff(p["pt"]) > 0) and p["pt"][-1] <= p["xns"] and p["open_marks"]
    assert np.max(np.abs(np.diff(p["pc"]))) < 1.0                            # no fake halving at the split
    assert p["rank"] == -1e6
