"""THE FEAR BOOK (PREREG219-226; user 2026-10-27: "Deploy to production the $1B, 1/6 each, 4 slots.";
`src/signals/fear_long.py`).

What must hold: a name's IVP at the signal close is PREREG219's percentile of its implied volatility among its own 252
sessions through that close (126 values at least, the bar within 5 days), reading no later bar; the day's fear signals
are the dip book's $1B+ signals (kept, not-top and filtered alike) whose IVP is at least 0.8, never the dip book's own
order of the day, IVP highest first, and a signal without a value is never ordered; they wait for the day's table (or
its window's end) and for the dip book's own order; each order is the capitulation book's — 1/6 of the live account at
its limit, at most 4 open with the working orders, filled at the open or at the end of the first bar reaching the limit
within the room, lapsing after its window, sold at the open after the first close above its 3-session average or after
5 sessions; its capital gives way FIRST (before the insider, capitulation and dip books: a short's, the crash book's
and every other book's fill's room; de-leveraging with the capitulation and insider books), a margin call closes its
longs and cancels its orders; it is ONE BOOK WITH THE DIP BOOK for one book per name — a fear long never blocks a dip
order and a dip long never blocks a fear order, while the other books still do; the broker rests its orders, orders
its share count and keeps a fear long and a dip long in one name apart; the pre-market fetch merges IBKR's bars into
the store and writes each candidate's IVP; the simulator builds the same book with PREREG223's account strategy.
"""
from __future__ import annotations

import gzip
import json
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import pytest

from config.settings import settings
from src.performance import books
from src.performance import live_account as la
from src.performance import tracker
from src.signals import dip_long as dl
from src.signals import fear_long as fl

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


def _iv(values, end=PREV):
    days = _sessions(end, len(values))
    return pd.Series(np.asarray(values, float), index=pd.DatetimeIndex([pd.Timestamp(x) for x in days]))


# ── the percentile ───────────────────────────────────────────────────────────

def test_the_ivp_is_the_values_percentile_among_its_own_252_sessions():
    s = _iv(list(np.linspace(0.2, 0.4, 300)) + [0.5])                         # the last value the highest
    v, x = fl.ivp_at(s, pd.Timestamp(PREV))
    assert x == 0.5 and v == pytest.approx((251 + 0.5) / 252)                 # 251 under it, itself half
    s2 = _iv(list(np.linspace(0.2, 0.4, 300)) + [0.1])
    assert fl.ivp_at(s2, pd.Timestamp(PREV))[0] == pytest.approx(0.5 / 252)  # the lowest of the window
    assert fl.ivp_at(_iv([0.3] * 125), pd.Timestamp(PREV)) is None           # fewer than 126 values
    assert fl.ivp_at(_iv([0.3] * 200, end=PREV - timedelta(days=9)), pd.Timestamp(PREV)) is None   # stale: > 5 days


def test_the_ivp_equals_prereg219s_percentile():
    """`fear_long.ivp_at` is PREREG219's `h219.pct_at` line for line (the research's construction, which the simulator
    and the live book share): a series with ties and gaps, every session."""
    rng = np.random.default_rng(219)
    vals = np.round(rng.lognormal(-1.2, 0.3, 600), 3)
    s = _iv(list(vals))
    s = s.drop(s.index[[50, 51, 52, 300]])                                    # a few missing bars

    def h219_pct_at(ser, S, win=252, min_n=126):                             # PREREG219's, copied verbatim
        if ser is None or not len(ser):
            return None
        i = int(ser.index.searchsorted(S, side="right")) - 1
        if i < 0 or (S - ser.index[i]).days > 5:
            return None
        lo = max(0, i - win + 1)
        if (ser.index[i] - ser.index[lo]).days > win * 1.6 / 252 * 365.25:
            return None
        w = ser.values[lo: i + 1]
        w = w[np.isfinite(w)]
        if len(w) < min_n:
            return None
        x = float(ser.values[i])
        return float((np.sum(w < x) + 0.5 * np.sum(w == x)) / len(w)), x
    for S in s.index[100::7]:
        assert fl.ivp_at(s, S) == h219_pct_at(s, S)


def test_the_store_merges_ibkrs_bars_a_fetched_bar_replacing_a_stored_one(monkeypatch, tmp_path):
    monkeypatch.setattr(settings, "fear_long_iv_dir", str(tmp_path / "iv"))
    assert fl.merge_rows("AAA", [["2026-10-01", 0.3, 0.31, 0.29, 0.30], ["2026-10-02", 0.3, 0.33, 0.3, 0.32]]) == (2, 2)
    assert fl.merge_rows("AAA", [["2026-10-02", 0.3, 0.35, 0.3, 0.34], ["2026-10-05", 0.3, 0.36, 0.3, 0.36]]) == (1, 3)
    s = fl.iv_series("AAA")
    assert list(s.round(2)) == [0.30, 0.34, 0.36]                             # the newer fetch of 10-02 kept
    rows = json.loads(gzip.decompress((tmp_path / "iv" / "AAA.json.gz").read_bytes()))
    assert rows[0][0] == "2026-10-01" and len(rows[0]) == 5


def test_every_stored_bar_an_answer_revises_or_no_longer_serves_is_journaled(monkeypatch, tmp_path):
    """P7: the store keeps IBKR's latest values, the journal the ones they replaced."""
    monkeypatch.setattr(settings, "fear_long_iv_dir", str(tmp_path / "iv"))
    fl.merge_rows("AAA", [["2026-10-01", 0.3, 0.31, 0.29, 0.30], ["2026-10-02", 0.3, 0.33, 0.3, 0.32],
                          ["2026-10-05", 0.3, 0.3, 0.3, 0.31]], day=DAY)
    assert fl.load_revisions(DAY) == []                                       # a name new to the store: nothing
    cnt: dict = {}
    # IBKR's answer: 10-01 as stored, 10-02 revised, 10-05 missing inside its span, 10-06 new
    assert fl.merge_rows("AAA", [["2026-10-01", 0.3, 0.31, 0.29, 0.30], ["2026-10-02", 0.3, 0.35, 0.3, 0.34],
                                 ["2026-10-06", 0.3, 0.3, 0.3, 0.36]], day=DAY, at="2026-10-08T12:40:00+00:00",
                         counts=cnt) == (1, 4)
    assert cnt == {"answered": 3, "revised": 1, "absent": 1}
    js = fl.load_revisions(DAY)
    assert [(x["kind"], x["date"], x["sym"]) for x in js] == [("revised", "2026-10-02", "AAA"),
                                                              ("absent", "2026-10-05", "AAA")]
    assert js[0]["old"] == [0.3, 0.33, 0.3, 0.32] and js[0]["new"] == [0.3, 0.35, 0.3, 0.34]
    assert js[0]["at"] == "2026-10-08T12:40:00+00:00" and js[0]["day"] == DAY.isoformat() and js[1]["old"][3] == 0.31
    assert list(fl.iv_series("AAA").round(2)) == [0.30, 0.34, 0.31, 0.36]    # IBKR's latest; the absent bar kept
    fl.merge_rows("AAA", [["2026-10-02", 0.3, 0.35, 0.3, 0.34]], day=DAY)    # the same answer again: nothing new
    assert len(fl.load_revisions(DAY)) == 2
    assert not fl.differs(["d", 0.3, None, 0.3, 0.3], ["d", 0.3 + 1e-12, None, 0.3, 0.3])
    assert fl.differs(["d", 0.3, None, 0.3, 0.3], ["d", 0.3, 0.31, 0.3, 0.3])     # a value in one only
    assert fl.differs(["d", 0.3, 0.3, 0.3, 0.3], ["d", 0.3, 0.3, 0.3, 0.3001])


# ── the day's signals ────────────────────────────────────────────────────────

def _dsig(tk, rsi, close, atr, dv=3e9):
    return {"ticker": tk, "session": "2026-10-07", "close": close, "sma_trend": close * 0.9, "rsi2": rsi, "dv20": dv,
            "atr14": atr, "limit": dl.limit_price(close, atr)}


@pytest.fixture
def day(monkeypatch, tmp_path):
    monkeypatch.setattr(settings, "enable_fear_long", True)
    monkeypatch.setattr(settings, "fear_long_dir", str(tmp_path / "fear_long"))
    monkeypatch.setattr(settings, "enable_dip_long", True)
    monkeypatch.setattr(settings, "dip_long_dir", str(tmp_path / "dip_long"))
    dip_day = {"day": DAY.isoformat(), "signal_session": "2026-10-07",
               "signals": [_dsig("PICK", 2.0, 100.0, 4.0)],                    # the dip book's daily top pick
               "not_top": [_dsig("FEAR1", 6.0, 50.0, 2.0), _dsig("CALM", 3.0, 80.0, 2.0),
                           _dsig("SMAL", 4.0, 40.0, 1.0, dv=1.2e9), _dsig("TINY", 4.0, 40.0, 1.0, dv=0.9e9),
                           _dsig("NOIV", 5.0, 60.0, 2.0)],
               "filtered": [_dsig("FEAR2", 8.0, 70.0, 2.0)]}                    # the meta-model's cut: still a candidate
    monkeypatch.setattr(dl, "signals_for", lambda d: dip_day)
    dl.save_orders(DAY, [{"oid": "PICK|2026-10-08", "ticker": "PICK", "day": "2026-10-08", "status": "pending",
                          "dip_top": 1},
                         {"oid": "FEAR1|2026-10-08", "ticker": "FEAR1", "day": "2026-10-08", "status": "not_top"}])
    tab = {"day": DAY.isoformat(), "status": "done", "names": {
        "PICK": {"status": "ok", "ivp": 0.95}, "FEAR1": {"status": "ok", "ivp": 0.82, "iv": 0.4},
        "FEAR2": {"status": "ok", "ivp": 0.91, "iv": 0.5}, "CALM": {"status": "ok", "ivp": 0.79},
        "SMAL": {"status": "ok", "ivp": 0.99}, "TINY": {"status": "ok", "ivp": 0.99},
        "NOIV": {"status": "no_contract"}}}
    dl._write_json(fl.table_path(DAY), tab)
    return dip_day


def test_the_fear_signals_are_the_fearful_dip_signals_highest_ivp_first(day):
    out = fl.compute_signals(DAY)
    assert [s["ticker"] for s in out["signals"]] == ["SMAL", "FEAR2", "FEAR1"]      # 0.99, 0.91, 0.82
    f = out["fates"]
    assert f["PICK"] == "dip_pick" and f["CALM"] == "calm" and f["NOIV"] == "no_iv" and f["TINY"] == "below_floor"
    s = {x["ticker"]: x for x in out["signals"]}
    assert s["FEAR1"]["limit"] == fl.limit_price(50.0, 2.0) == 49.0 and s["FEAR1"]["ivp"] == 0.82
    assert fl.signals_path(DAY).exists() and out["signal_session"] == "2026-10-07"


def test_the_signals_wait_for_the_days_table_and_the_dip_books_order(day):
    tab = fl.load_table(DAY)
    dl._write_json(fl.table_path(DAY), dict(tab, status="running"))
    assert fl.signals_for(DAY, at(DAY, 9, 10)) is None and not fl.signals_path(DAY).exists()   # still fetching
    assert fl.signals_for(DAY, at(DAY, 9, 28)) is not None                   # its window ended: final as it is
    p = dl.orders_path(DAY)
    p.unlink()
    assert fl.dip_ready(DAY) is False                                         # a kept pick, no order yet
    assert fl.dip_ready(DAY + timedelta(days=1)) is False                     # no dip signals at all yet


def test_without_a_table_nothing_is_ordered_and_the_banner_says_so(day, monkeypatch):
    fl.table_path(DAY).unlink()
    out = fl.compute_signals(DAY)
    assert out["signals"] == [] and out["table"] is None and set(out["fates"].values()) >= {"no_iv"}
    monkeypatch.setattr(dl, "in_rth", lambda now=None: True)
    h = fl.health(at(DAY, 10, 45))
    assert any("no implied-volatility table" in p for p in h["problems"])


def test_a_signal_the_fetch_missed_is_a_problem_one_ibkr_has_no_iv_for_a_note(day, monkeypatch):
    tab = fl.load_table(DAY)
    names = sorted(t for t, f in fl.compute_signals(DAY, write=False)["fates"].items() if f != "dip_pick")
    assert len(names) >= 2
    a, b = names[0], names[1]
    tab["names"][a] = {"status": "no_data", "errors": [[162, "HMDS query returned no data"]]}
    tab["names"][b] = {"status": "error", "errors": [[-1, "TimeoutError: "]]}
    dl._write_json(fl.table_path(DAY), tab)
    out = fl.compute_signals(DAY)
    assert out["fates"][a] == out["fates"][b] == "no_iv"
    assert out["iv_status"][a] == "no_data" and out["iv_status"][b] == "error"
    monkeypatch.setattr(dl, "in_rth", lambda now=None: True)
    h = fl.health(at(DAY, 10, 45))
    assert any(b + " error" in p for p in h["problems"]) and not any(a + " no_data" in p for p in h["problems"])
    assert any(a + " no_data" in n for n in h["notes"])


def test_before_its_first_session_the_book_fetches_orders_and_checks_nothing(day, monkeypatch, tmp_path):
    monkeypatch.setattr(settings, "fear_long_first_session", (DAY + timedelta(days=1)).isoformat())
    monkeypatch.setattr(dl, "in_rth", lambda now=None: True)
    fl.table_path(DAY).unlink()                                               # the day never fetched
    launched = []
    monkeypatch.setattr(fl.subprocess, "Popen", lambda *a, **k: launched.append(a[0]))
    monkeypatch.chdir(tmp_path)
    assert fl.prepare(at(DAY, 9, 0)) is None and not launched
    assert tracker.record_fear_long_trades(now=at(DAY, 9, 31)) == 0 and not fl.signals_path(DAY).exists()
    assert fl.health(at(DAY, 12, 45)) == {"problems": [], "notes": []}
    monkeypatch.setattr(settings, "fear_long_first_session", DAY.isoformat())   # its first session: checked
    assert any("no fear signals decided" in p for p in fl.health(at(DAY, 12, 45))["problems"])
    monkeypatch.setattr(settings, "fear_long_first_session", "not a date")       # unreadable: no first session
    assert fl.live_on(DAY) and fl.first_session() is None


def test_the_signals_preview_writes_nothing(day):
    assert not fl.signals_path(DAY).exists()
    out = fl.compute_signals(DAY, write=False)
    assert out["signals"] and not fl.signals_path(DAY).exists()
    fl.main(["--signals", "--day", DAY.isoformat()])
    assert not fl.signals_path(DAY).exists()                                  # the hand run is a preview


# ── the live account ─────────────────────────────────────────────────────────

@pytest.fixture
def acct(monkeypatch):
    for k, v in {"enable_sel_short": True, "enable_sel_short_account_sizing": True,
                 "enable_sel_short_ibkr_whatif": False, "enable_live_account": True,
                 "sel_short_account_initial": 10_000.0, "dip_long_account_slices": 2,
                 "cap_long_account_slices": 6.0, "cap_long_max_open": 3, "crash_long_account_slices": 3.0,
                 "crash_long_margin_rate": 0.75, "live_account_delever_trigger": 0.7,
                 "live_account_delever_target": 0.5, "sel_short_account_house_maint": 2.0,
                 "sel_short_account_house_init": 2.86, "fear_long_account_slices": 6.0, "fear_long_max_open": 4,
                 "insider_long_account_slices": 10.0, "insider_long_max_open": 5}.items():
        monkeypatch.setattr(settings, k, v)


def long_(mech, tk, n=10, px=100.0, at_iso="2026-10-07T13:31:00+00:00", ret=0.0):
    pre = {"dip_long": "dip", "cap_long": "cap", "fear_long": "fear", "insider_long": "ins"}[mech]
    return {"ticker": tk, "entry_mechanism": mech, "action": "BUY", "type": "STOCK", "status": "OPEN",
            f"{pre}_account_shares": n, "entry_price": px, "current_price": px, "return_pct": ret,
            "entry_datetime": at_iso, "entry_date": at_iso[:10], "entry_session": "rth", "position_size_multiplier": 1.0,
            "recommendation_id": f"{pre}-{tk}", "current_price_datetime": NOW.isoformat()}


def fear(tk, **kw):
    return long_("fear_long", tk, **kw)


def dip(tk, **kw):
    return long_("dip_long", tk, **kw)


def cap(tk, **kw):
    return long_("cap_long", tk, **kw)


def ins(tk, **kw):
    return long_("insider_long", tk, **kw)


def short(tk, n, px=10.0, at_iso="2026-10-08T14:00:00+00:00"):
    return {"ticker": tk, "entry_mechanism": "sel_short", "sel_arm": "vol", "action": "SELL", "type": "STOCK",
            "status": "OPEN", "sel_account_shares": n, "entry_price": px, "current_price": px, "return_pct": 0.0,
            "sel_house_maint": 2.0, "sel_house_init": 2.86, "entry_datetime": at_iso, "entry_date": at_iso[:10],
            "entry_session": "rth", "position_size_multiplier": 1.0, "recommendation_id": f"sel-{tk}",
            "current_price_datetime": at_iso}


def test_the_fear_longs_count_in_the_account_with_their_own_slots(acct):
    st = la.state([fear("F1"), dip("D1"), cap("C1")], NOW)
    assert st["fear_open"] == 1 and st["fear_gross"] == pytest.approx(1_000.0) and st["long_open"] == 1
    assert st["init_req"] == pytest.approx(1_500.0) and la.long_book(fear("F1")) == "fear"
    called, st2 = la.margin_call([fear("F1", n=300, px=100.0, ret=-50.0)], NOW)
    assert called and st2["fear_open"] == 1


def test_each_order_is_a_sixth_at_most_four_open_the_working_orders_counted(acct):
    assert la.size_fear([], 100.0, NOW)[:2] == (16, "ok")                    # $10,000 / 6 / $100
    held = [fear("F1"), fear("F2"), fear("F3")]
    assert la.size_fear(held, 100.0, NOW, working=1)[:2] == (0, "no_slot")
    assert la.size_fear(held, 100.0, NOW, working=0)[0] > 0


def test_the_fear_book_gives_way_first_to_a_short_then_insider_capitulation_dip(acct):
    held = [dip("D1", n=40, at_iso="2026-10-07T13:31:00+00:00"), cap("C1", n=10, at_iso="2026-10-07T13:32:00+00:00"),
            ins("I1", n=5, at_iso="2026-10-07T13:36:00+00:00"),
            fear("F1", n=10, at_iso="2026-10-07T13:33:00+00:00"), fear("F2", n=10, at_iso="2026-10-07T13:34:00+00:00")]
    plan, _, _ = la.give_way_plan(held + [short("HOT", 150)], 10.0, None, NOW, slices=1)
    order = [t["ticker"] for t in plan]
    assert order[:2] == ["F2", "F1"]                                          # the fear book first, the latest first
    assert order[2:4] in (["I1", "C1"], ["I1"], [])                           # then the insider book, then the cap


def test_deleveraging_sells_the_fear_longs_with_the_capitulation_and_insider_books(acct):
    held = [dip("D1", n=50, at_iso="2026-10-07T13:40:00+00:00"), cap("C1", n=20, at_iso="2026-10-07T13:30:00+00:00"),
            fear("F1", n=20, at_iso="2026-10-07T13:35:00+00:00"), short("HOT", 250, px=10.0)]
    assert la.state(held, NOW)["maint"] > 0.7 * la.state(held, NOW)["equity"]
    names = [t["ticker"] for t in la.delever_plan(held, NOW)[0]]
    assert names[:2] == ["F1", "C1"]                                          # together, the latest first, before the dip


def test_every_other_books_order_is_sized_as_if_the_fear_longs_were_sold_and_a_fear_fill_frees_nothing(acct):
    held = [fear("F1", n=50, at_iso="2026-10-07T13:31:00+00:00"), fear("F2", n=40, at_iso="2026-10-07T13:32:00+00:00")]
    assert la.size_long(held, 100.0, NOW)[:2] == (50, "ok")                   # the dip slice as if they were sold
    assert la.size_cap(held, 100.0, NOW)[:2] == (16, "ok")                    # the capitulation slice too
    assert [t["ticker"] for t in la.fill_give_way_plan(held, 100.0, 50, NOW, book="dip")[0]] == ["F2"]
    assert [t["ticker"] for t in la.fill_give_way_plan(held, 100.0, 50, NOW, book="cap")[0]] == ["F2"]
    assert la.fill_give_way_plan(held + [ins("I1", n=5)], 100.0, 50, NOW, book="fear")[0] == []   # nobody under it
    assert la.fill_cap(held, 100.0, NOW, book="fear")[0] == la.fill_cap(held, 100.0, NOW)[0]


# ── the ledger steps ─────────────────────────────────────────────────────────

def bars(d, rows):
    idx = [(datetime(d.year, d.month, d.day, 9, 30, tzinfo=ET) + timedelta(minutes=30 * k)).astimezone(timezone.utc)
           .replace(tzinfo=None) for k in range(len(rows))]
    return pd.DataFrame(rows, columns=["Open", "High", "Low", "Close"], index=pd.DatetimeIndex(idx)).assign(Volume=1e6)


def _fsig(tk, ivp, rsi, close, atr):
    return {"ticker": tk, "close": close, "atr14": atr, "rsi2": rsi, "dv20": 1.5e9, "sma_trend": close * 0.9,
            "ivp": ivp, "iv": 0.4, "limit": fl.limit_price(close, atr)}


@pytest.fixture
def book(monkeypatch, tmp_path, acct):
    for k, v in {"enable_fear_long": True, "fear_long_dir": str(tmp_path / "fear_long"), "fear_long_limit_atr": 0.5,
                 "fear_long_limit_sessions": 3, "enable_dip_long": True, "dip_long_dir": str(tmp_path / "dip_long"),
                 "dip_long_entry_mode": "limit", "dip_long_limit_atr": 0.5, "dip_long_limit_sessions": 3,
                 "enable_cap_long": True, "cap_long_dir": str(tmp_path / "cap_long"),
                 "enable_sel_short_vol": True, "sel_short_traded_arms": "vol,etf,fall"}.items():
        monkeypatch.setattr(settings, k, v)
    monkeypatch.setattr(dl, "_extend", lambda tks, through: None)
    monkeypatch.setattr(tracker, "_reference_close", lambda t: None)
    day = {"day": DAY.isoformat(), "signal_session": "2026-10-07",
           "signals": [_fsig("BBB", 0.85, 6.0, 50.0, 2.0), _fsig("AAA", 0.97, 2.0, 100.0, 4.0)]}   # AAA first: 0.97
    monkeypatch.setattr(fl, "signals_for", lambda d, now=None: day if d == DAY else {"signals": []})
    monkeypatch.setattr(fl, "dip_ready", lambda d: True)
    monkeypatch.setattr(dl, "signals_for", lambda d: {"signals": []})
    feed = {}
    monkeypatch.setattr(dl, "session_bars", lambda tk, d: feed.get((tk, d), bars(d, [])))
    return feed


def fears():
    return {t["ticker"]: t for t in tracker._load_trades() if t.get("entry_mechanism") == "fear_long"}


def orders_of(d=DAY):
    return {o["ticker"]: o for o in fl.load_orders(d)}


def test_the_open_places_every_order_ivp_first_at_a_sixth_and_fills_one_opening_under_its_limit(book):
    book[("AAA", DAY)] = bars(DAY, [(97.5, 99.0, 97.0, 98.5)])              # opened under 98.00: filled at the open
    book[("BBB", DAY)] = bars(DAY, [(51.0, 51.2, 50.5, 51.0)])
    assert tracker.record_fear_long_trades(now=OPEN_TICK) == 3              # 2 placed + 1 bought at the open
    o = orders_of()
    assert o["AAA"]["rank"] == 2 and fl.order_key(o["AAA"]) < fl.order_key(o["BBB"])    # IVP 0.97 placed first
    assert o["AAA"]["status"] == "filled" and o["BBB"]["status"] == "working"
    assert o["AAA"]["qty"] == int(10_000 / 6 // 98.0) and o["BBB"]["qty"] == int(10_000 / 6 // 49.0)
    assert o["AAA"]["window"] == ["2026-10-08", "2026-10-09", "2026-10-12"]
    t = fears()["AAA"]
    assert t["entry_price"] == 97.5 and t["fear_fill_kind"] == "open" and t["fear_account_shares"] == o["AAA"]["qty"]
    assert t["fear_ivp"] == 0.97 and t["fear_entry_day"] == "2026-10-08" and books.is_live(t)
    assert tracker.record_fear_long_trades(now=OPEN_TICK + timedelta(minutes=30)) == 0   # one placement a session


def test_a_touch_fills_at_the_bar_end_and_an_order_lapses_after_its_window(book):
    book[("AAA", DAY)] = bars(DAY, [(101.0, 101.5, 100.5, 101.0)])
    book[("BBB", DAY)] = bars(DAY, [(51.0, 51.2, 50.5, 51.0)])
    tracker.record_fear_long_trades(now=OPEN_TICK)
    book[("AAA", DAY)] = bars(DAY, [(101.0, 101.5, 97.8, 98.6), (98.6, 99.0, 98.2, 98.9)])
    assert tracker.monitor_fear_long_orders(now=at(DAY, 10, 0)) == 1
    t = fears()["AAA"]
    assert t["entry_price"] == 98.0 and t["fear_fill_kind"] == "touch" and t["entry_datetime"].startswith("2026-10-08T14:00")
    tracker.monitor_fear_long_orders(now=at(date(2026, 10, 12), 16, 1))
    assert orders_of()["BBB"]["status"] == "lapsed"


def test_the_dip_book_is_not_another_book_but_the_capitulation_book_is(book):
    tracker._save_trades([dip("AAA"), cap("BBB")])
    book[("AAA", DAY)] = bars(DAY, [(97.0, 99.0, 96.0, 98.5)])
    book[("BBB", DAY)] = bars(DAY, [(48.0, 51.2, 47.5, 51.0)])
    tracker.record_fear_long_trades(now=OPEN_TICK)
    assert "AAA" in fears() and fears()["AAA"]["fear_stack_n"] == 1          # bought beside the dip long
    assert orders_of()["BBB"]["status"] == "pending" and "BBB" not in fears()   # the capitulation book's name: held
    assert "AAA" not in tracker._dip_other_book(tracker._load_trades())       # nor does it block the dip book
    assert "BBB" in tracker._dip_other_book(tracker._load_trades())


def test_a_dip_fill_sells_the_fear_long_for_its_room(book, monkeypatch):
    tracker._save_trades([fear("F1", n=90, px=100.0)])                        # $9,000 of the $10,000 long side
    monkeypatch.setattr(tracker, "_execution_iso", lambda: at(DAY, 10, 0).isoformat())
    o = {"oid": "DIP1|2026-10-08", "ticker": "DIP1", "day": "2026-10-08", "signal_session": "2026-10-07", "rank": 1,
         "rsi2": 3.0, "close": 100.0, "atr14": 4.0, "limit": 98.0, "dv20": 3e9, "sma_trend": 90.0,
         "window": ["2026-10-08", "2026-10-09", "2026-10-12"], "status": "working", "qty": 50, "events": [],
         "session_armed_at": {"2026-10-08": OPEN_TICK.isoformat()}, "ibkr_qty": 50, "ibkr": None, "dip_top": 1}
    dl.save_orders(DAY, [o])
    book[("DIP1", DAY)] = bars(DAY, [(101.0, 101.5, 97.8, 98.6)])
    assert tracker.monitor_dip_long_orders(now=at(DAY, 10, 0)) == 1
    tr = {t["ticker"]: t for t in tracker._load_trades()}
    assert tr["F1"]["status"] == "CLOSED" and tr["F1"]["exit_reason"] == "fear_gave_way"
    assert dl._read_jsonl(fl.exits_path(DAY))[0]["reason"] == "fear_gave_way"   # its own book's journal


def test_a_margin_call_closes_the_fear_longs_and_cancels_its_orders(book, monkeypatch):
    book[("AAA", DAY)] = bars(DAY, [(101.0, 101.5, 100.5, 101.0)])
    book[("BBB", DAY)] = bars(DAY, [(51.0, 51.2, 50.5, 51.0)])
    tracker.record_fear_long_trades(now=OPEN_TICK)
    tracker._save_trades(tracker._load_trades() + [fear("LOS", n=300, px=100.0, ret=-50.0)])
    monkeypatch.setattr(tracker, "_execution_iso", lambda: at(DAY, 10, 0).isoformat())
    assert tracker.monitor_live_account(now=at(DAY, 10, 0)) == 1
    t = fears()["LOS"]
    assert t["status"] == "CLOSED" and t["exit_reason"] == "fear_margin_call"
    assert {x["status"] for x in orders_of().values()} == {"pending"}


def test_deleveraging_closes_the_fear_longs_as_fear_delever(book, monkeypatch):
    tracker._save_trades([fear("F1", n=30), dip("D1", n=50, at_iso="2026-10-07T13:40:00+00:00"),
                          short("HOT", 300, px=10.0)])
    monkeypatch.setattr(tracker, "_execution_iso", lambda: at(DAY, 10, 0).isoformat())
    assert tracker.monitor_live_account(now=at(DAY, 10, 0)) >= 1
    assert {x["ticker"]: x for x in tracker._load_trades()}["F1"]["exit_reason"] == "fear_delever"


def _daily(closes, end=PREV):
    days = _sessions(end, len(closes))
    c = np.asarray(closes, float)
    return pd.DataFrame({"o": c, "h": c * 1.01, "l": c * 0.99, "c": c, "v": np.full(len(c), 1e7)},
                        index=pd.DatetimeIndex([pd.Timestamp(x) for x in days]))


def test_the_long_leaves_after_the_first_close_above_its_3_session_average_or_5_sessions(book, monkeypatch):
    up = _daily(list(np.linspace(100, 80, 230)) + [78.0, 76.0, 80.0])
    monkeypatch.setattr(dl, "_daily", lambda tk, through=None: up)
    reason, info = fl.exit_check({"ticker": "AAA", "fear_entry_day": _sessions(PREV, 3)[0].isoformat()}, DAY)
    assert reason == "fear_rebound" and info["rebound_session"] == "2026-10-07"
    down = _daily(list(np.linspace(100, 60, 240)))
    monkeypatch.setattr(dl, "_daily", lambda tk, through=None: down)
    assert fl.exit_check({"ticker": "AAA", "fear_entry_day": _sessions(PREV, 5)[0].isoformat()}, DAY)[0] == "fear_time"
    assert fl.exit_check({"ticker": "AAA", "fear_entry_day": _sessions(PREV, 4)[0].isoformat()}, DAY)[0] is None
    tracker._save_trades([fear("AAA")])
    monkeypatch.setattr(fl, "exit_check", lambda t, d: ("fear_rebound", {"held": 2, "close": 105.0, "sma_exit": 101.0,
                                                                         "rebound_session": "2026-10-07"}))
    monkeypatch.setattr(tracker, "_fetch_price", lambda t: 106.0)
    monkeypatch.setattr(tracker, "_execution_iso", lambda: OPEN_TICK.isoformat())
    assert tracker.monitor_fear_long_positions(now=OPEN_TICK) == 1
    t = fears()["AAA"]
    assert t["status"] == "CLOSED" and t["exit_reason"] == "fear_rebound" and t["exit_price"] == 106.0


def test_the_give_way_reasons_name_the_longs_own_book():
    assert tracker._long_give_way_reason(fear("F1"), "gave_way") == ("fear_gave_way", "fear_long")
    assert tracker._long_give_way_reason(fear("F1"), "margin_call") == ("fear_margin_call", "fear_long")
    assert "fear_account_shares" in tracker.SPLIT_SHARE_FIELDS and "fear_limit" in tracker.SPLIT_PRICE_FIELDS


# ── the broker ───────────────────────────────────────────────────────────────

def test_the_broker_orders_its_share_count_rests_its_orders_and_books_an_open_fill_at_ibkrs_price(book, monkeypatch):
    from src.broker import reconcile as rec
    from tests.test_dip_limit import _broker
    assert rec._entry_qty(fear("F1", n=7), 100.0, 1e6, 1.0) == 7
    t = dict(fear("F1"), fear_fill_kind="open", broker_fill_price=97.62, broker_fill_qty=10, entry_price=97.5)
    rec._book_dip_open_fill(t)
    assert t["entry_price"] == 97.62 and t["fear_open_print"] == 97.5
    book[("AAA", DAY)] = bars(DAY, [(101.0, 101.5, 100.5, 101.0)])
    book[("BBB", DAY)] = bars(DAY, [(51.0, 51.2, 50.5, 51.0)])
    tracker._save_trades([dip("BBB", n=5)])                                   # the dip book holds BBB: still rests
    tracker.record_fear_long_trades(now=OPEN_TICK)
    store = {"trades": tracker._load_trades()}
    r, b = _broker(monkeypatch, store)
    r.sync(broker=b)
    rest = sorted((o.ticker, o.side, o.quantity, o.limit_price, o.rest, o.client_ref[:6]) for o in b.orders
                  if getattr(o, "rest", False))
    assert ("AAA", "BUY", orders_of()["AAA"]["qty"], 98.0, True, "fearw-") in rest
    assert ("BBB", "BUY", orders_of()["BBB"]["qty"], 49.0, True, "fearw-") in rest
    assert orders_of()["AAA"]["ibkr"]["ref"] in r._dip_resting_refs()        # owned: the orphan sweep keeps it


def test_a_fear_long_and_a_dip_long_in_one_name_each_order_and_close_their_own_shares(monkeypatch):
    """Two live books' long trades in one symbol (the one-book rule's consequence): the second orders its OWN shares
    against the holding the first owns, and closing one sells only its shares."""
    import src.broker.reconcile as rec
    from src.broker.base import Position
    from tests.test_broker_reconcile import FakeBroker
    store = {"trades": []}
    monkeypatch.setattr(rec.repo, "load_trades", lambda: store["trades"])
    monkeypatch.setattr(rec.repo, "save_trades", lambda t: store.update(trades=t))
    monkeypatch.setattr(settings, "broker_mode", "ibkr_paper")
    monkeypatch.setattr(settings, "broker_settle_seconds", 0)
    monkeypatch.setattr(settings, "enable_legacy_entries", False)
    monkeypatch.setattr(rec, "usd_per_unit", lambda ccy: 1.0)
    from src.performance import market_calendar
    monkeypatch.setattr(market_calendar, "current_session", lambda now=None: "rth")
    f = dict(fear("XYZ", n=10), broker_order_id="o-f", broker_status="Filled", broker_fill_qty=10,
             broker_client_ref="fear-XYZ")
    d = dip("XYZ", n=6)
    store["trades"] = [f, d]
    brk = FakeBroker(positions=[Position("XYZ", 10, 100.0)])                  # the fear long's 10 shares held
    rec.sync(broker=brk)
    assert [(o.client_ref, o.side, o.quantity) for o in brk.orders] == [("dip-XYZ", "BUY", 6)]
    d2 = dict(store["trades"][1], broker_status="Filled", broker_fill_qty=6)
    f2 = dict(f, status="CLOSED", exit_price=101.0, exit_reason="fear_rebound")
    store["trades"] = [f2, d2]
    brk2 = FakeBroker(positions=[Position("XYZ", 16, 100.0)])
    rec.sync(broker=brk2)
    assert [(o.side, o.quantity) for o in brk2.orders] == [("SELL", 10)]      # the fear long's shares only


# ── the pre-market fetch ─────────────────────────────────────────────────────

class _Event(list):
    """eventkit's Event as the fetch uses it: ``+=`` connects a handler, ``emit`` calls them."""

    def __iadd__(self, h):
        self.append(h)
        return self

    def emit(self, *a):
        for h in self:
            h(*a)


class _FakeIB:
    """ib_async's IB for the fetch: qualifies every symbol but NOCON, answers DATA bars; a name starting NODATA gets
    IBKR's 162 on its error stream and no bar, SLOW a timeout."""

    def __init__(self, bars_by_sym):
        self.bars, self.asked, self.RequestTimeout = bars_by_sym, [], 0
        self.errorEvent = _Event()

    def isConnected(self):
        return True

    def qualifyContracts(self, c):
        if c.symbol == "NOCON":
            return []
        c.conId = 1
        return [c]

    def reqHistoricalData(self, c, **kw):
        self.asked.append((c.symbol, kw["durationStr"], kw["whatToShow"]))
        if c.symbol.startswith("NODATA"):
            self.errorEvent.emit(7, 162, "Historical Market Data Service error message:HMDS query returned no data", c)
            return []
        if c.symbol.startswith("SLOW"):
            raise TimeoutError()
        return [SimpleNamespace(date=pd.Timestamp(d).date(), open=v, high=v, low=v, close=v)
                for d, v in self.bars.get(c.symbol, [])]


def test_the_fetch_asks_each_candidate_merges_its_bars_and_writes_its_ivp(monkeypatch, tmp_path):
    monkeypatch.setattr(settings, "enable_fear_long", True)
    monkeypatch.setattr(settings, "fear_long_dir", str(tmp_path / "fear_long"))
    monkeypatch.setattr(settings, "fear_long_iv_dir", str(tmp_path / "iv"))
    monkeypatch.setattr(fl.time, "sleep", lambda s: None)
    monkeypatch.setattr(fl, "near_slot", lambda now: False)
    st = {"AAA": {"session": PREV, "close": 100.0, "sma_trend": 90.0, "rsi2": 4.0, "dv20": 3e9},
          "BBB": {"session": PREV, "close": 100.0, "sma_trend": 90.0, "rsi2": 25.0, "dv20": 3e9},   # not a dip
          "NOCON": {"session": PREV, "close": 50.0, "sma_trend": 40.0, "rsi2": 2.0, "dv20": 2e9}}
    monkeypatch.setattr(dl, "candidates", lambda d, min_dv=None: (sorted(st), "test"))
    monkeypatch.setattr(dl, "name_stats", lambda tk, through: st[tk])
    days = _sessions(PREV, 400)
    vals = list(np.linspace(0.2, 0.3, 399)) + [0.6]
    ib = _FakeIB({"AAA": list(zip([str(x) for x in days], vals))})
    tab = fl.fetch(DAY, until="09:27", connect=lambda: ib, clock=lambda: at(DAY, 8, 45))
    assert tab["status"] == "done" and tab["candidates"] == ["AAA", "NOCON"]
    assert ib.asked == [("AAA", "2 Y", "OPTION_IMPLIED_VOLATILITY")]          # NOCON: no contract, never a bar asked
    assert tab["names"]["AAA"]["ivp"] == pytest.approx((251 + 0.5) / 252) and tab["names"]["NOCON"]["status"] == "no_contract"
    assert fl.load_table(DAY)["status"] == "done" and fl.iv_series("AAA").iloc[-1] == 0.6
    ib2 = _FakeIB({"AAA": list(zip([str(x) for x in days[-20:]], vals[-20:]))})
    dl._write_json(fl.table_path(DAY), {})
    fl.fetch(DAY, until="09:27", connect=lambda: ib2, clock=lambda: at(DAY, 8, 45))
    assert ib2.asked[0][1] == "1 M"                                           # a year held: only a month asked


def test_a_fetch_journals_the_bars_ibkr_revised_and_counts_them_in_the_days_table(monkeypatch, tmp_path):
    """P7: the month a fetch re-reads is compared with the store, bar by bar."""
    monkeypatch.setattr(settings, "enable_fear_long", True)
    monkeypatch.setattr(settings, "fear_long_dir", str(tmp_path / "fear_long"))
    monkeypatch.setattr(settings, "fear_long_iv_dir", str(tmp_path / "iv"))
    monkeypatch.setattr(fl.time, "sleep", lambda s: None)
    monkeypatch.setattr(fl, "near_slot", lambda now: False)
    st = {"AAA": {"session": PREV, "close": 100.0, "sma_trend": 90.0, "rsi2": 4.0, "dv20": 3e9}}
    monkeypatch.setattr(dl, "candidates", lambda d, min_dv=None: (sorted(st), "test"))
    monkeypatch.setattr(dl, "name_stats", lambda tk, through: st[tk])
    days = _sessions(PREV, 400)
    vals = list(np.linspace(0.2, 0.3, 399)) + [0.6]
    fl.merge_rows("AAA", [[str(x), v, v, v, v] for x, v in zip(days, vals)], day=PREV)    # the store holds a history
    revised = list(vals)
    revised[-3] = 0.9                                                         # IBKR revised the bar 3 sessions back
    ib = _FakeIB({"AAA": list(zip([str(x) for x in days[-20:]], revised[-20:]))})
    tab = fl.fetch(DAY, until="09:27", connect=lambda: ib, clock=lambda: at(DAY, 8, 45))
    r = tab["names"]["AAA"]
    assert ib.asked[0][1] == "1 M" and r["status"] == "ok"
    assert (r["answered"], r["added"], r["revised"], r["absent"]) == (20, 0, 1, 0)
    assert tab["revisions"] == {"AAA": {"revised": 1, "absent": 0}}
    js = fl.load_revisions(DAY)
    assert len(js) == 1 and js[0]["date"] == str(days[-3]) and js[0]["new"][3] == 0.9 and js[0]["at"] == r["at"]
    assert js[0]["old"][3] == pytest.approx(vals[-3]) and fl.iv_series("AAA").loc[pd.Timestamp(days[-3])] == 0.9


def test_ibkrs_no_data_answers_are_not_timeouts_and_three_timeouts_stop_the_fetch(monkeypatch, tmp_path):
    monkeypatch.setattr(settings, "enable_fear_long", True)
    monkeypatch.setattr(settings, "fear_long_dir", str(tmp_path / "fear_long"))
    monkeypatch.setattr(settings, "fear_long_iv_dir", str(tmp_path / "iv"))
    monkeypatch.setattr(fl.time, "sleep", lambda s: None)
    monkeypatch.setattr(fl, "near_slot", lambda now: False)
    base = {"session": PREV, "close": 100.0, "sma_trend": 90.0, "dv20": 3e9}
    st = {f"NODATA{i}": dict(base, rsi2=1.0 + i) for i in range(3)} | {"AAA": dict(base, rsi2=8.0)}
    monkeypatch.setattr(dl, "candidates", lambda d, min_dv=None: (sorted(st), "test"))
    monkeypatch.setattr(dl, "name_stats", lambda tk, through: st[tk])
    days = _sessions(PREV, 400)
    ib = _FakeIB({"AAA": list(zip([str(x) for x in days], list(np.linspace(0.2, 0.3, 399)) + [0.6]))})
    tab = fl.fetch(DAY, until="09:27", connect=lambda: ib, clock=lambda: at(DAY, 8, 45))
    assert [a[0] for a in ib.asked] == ["NODATA0", "NODATA1", "NODATA2", "AAA"]     # three no-data answers go on
    assert tab["status"] == "done" and tab["names"]["AAA"]["status"] == "ok"
    assert {tab["names"][f"NODATA{i}"]["status"] for i in range(3)} == {"no_data"}
    assert tab["names"]["NODATA0"]["errors"][0][0] == 162
    st2 = {f"SLOW{i}": dict(base, rsi2=1.0 + i) for i in range(3)} | {"BBB": dict(base, rsi2=8.0)}
    monkeypatch.setattr(dl, "candidates", lambda d, min_dv=None: (sorted(st2), "test"))
    monkeypatch.setattr(dl, "name_stats", lambda tk, through: st2[tk])
    d2 = DAY + timedelta(days=1)
    while not dl.is_session(d2):
        d2 += timedelta(days=1)
    monkeypatch.setattr(dl, "prev_session", lambda d: PREV)
    ib2 = _FakeIB({})
    tab2 = fl.fetch(d2, until="09:27", connect=lambda: ib2, clock=lambda: at(d2, 8, 45))
    assert tab2["status"] == "stopped" and tab2["stopped"] == "three timeouts in a row"
    assert [a[0] for a in ib2.asked] == ["SLOW0", "SLOW1", "SLOW2"]
    assert {tab2["names"][f"SLOW{i}"]["status"] for i in range(3)} == {"error"}


def test_the_launch_runs_in_its_window_once_and_never_in_an_extra_account(monkeypatch, tmp_path):
    monkeypatch.setattr(settings, "enable_fear_long", True)
    monkeypatch.setattr(settings, "fear_long_dir", str(tmp_path / "fear_long"))
    launched = []

    class P:
        pid = 4242

    monkeypatch.setattr(fl.subprocess, "Popen", lambda *a, **k: launched.append(a[0]) or P())
    monkeypatch.chdir(tmp_path)
    assert fl.prepare(at(DAY, 8, 10)) is None and not launched                # before its window
    assert fl.prepare(at(DAY, 9, 0))["launched"] == 4242
    assert launched[0][-3:] == ["--fetch", "--day", DAY.isoformat()]
    dl._write_json(fl.table_path(DAY), {"status": "done"})
    assert fl.prepare(at(DAY, 9, 5)) is None and len(launched) == 1           # the day's table done
    assert fl.prepare(at(DAY, 9, 31)) is None                                 # after its window


# ── the simulator ────────────────────────────────────────────────────────────

def test_the_simulator_runs_the_book_as_production_does(monkeypatch):
    from src.backtest import mimic as M
    monkeypatch.setattr(settings, "enable_fear_long", True)
    m = M.Mimic()
    names = {s.name: s for s in M.live_strategies(m)}
    f = names["fear"]
    assert f.side == 1 and f.slices == 6.0 and f.max_open == 4 and f.priority == 9 and f.quality == 0.04
    assert f.preemptible and f.yield_whole and f.yield_first == "latest" and f.delever_order == 0
    assert f.orders == "fill" and f.book == names["dip"].book == M.BOOK["dip"]     # one book with the dip book
    assert M.BOOK["fear"] == books.FEAR_LONG and "fear" in m.ext_longs
    assert M.live_rules(m).fills_make_room and M.live_rules(m).fill_room_exact
    assert list(M.live_strategies(m))[-1].name == "fear"                           # appended last (PREREG225's order)
    assert "fear" not in {s.name for s in M.live_strategies(M.Mimic(**M.PRE226))}
    assert M.Mimic.off().fear is False
    monkeypatch.setattr(settings, "enable_fear_long", False)
    assert "fear" not in {s.name for s in M.live_strategies(M.Mimic())}


def test_the_adapter_orders_only_the_fearful_dips_with_their_ivp(monkeypatch, tmp_path):
    from src.backtest.strategies import dip as DIP
    from src.backtest.strategies import fear as FEAR
    monkeypatch.setattr(settings, "fear_long_iv_dir", str(tmp_path / "iv"))
    n = 300
    days = pd.DatetimeIndex([pd.Timestamp(x) for x in _sessions(PREV, n)])
    c = np.r_[np.linspace(50.0, 100.0, n - 3), [99.0, 97.0, 95.0]]           # an uptrend, a 3-day dip at the end
    frame = pd.DataFrame({"o": c, "h": c * 1.01, "l": c * 0.99, "c": c, "v": np.full(n, 3e7)}, index=days)
    panel = {"FEAR": frame, "CALM": frame.copy(), "GONE@D": frame.copy()}
    monkeypatch.setattr(DIP, "eligible_names", lambda p: sorted(p))
    for tk, last in (("FEAR", 0.9), ("CALM", 0.1)):
        rows = [[str(d.date()), 0.3, 0.3, 0.3, 0.3] for d in days[:-1]] + [[str(days[-1].date()), last, last, last, last]]
        fl.merge_rows(tk, rows)
    iv = FEAR.IVP()
    assert iv.name("GONE@D", False) is None and iv.name("CALM", False) == "CALM"
    store: dict = {}
    sig = FEAR.signals_fn(iv, False, store)(panel, FEAR.params(), days[-1], days[-1])
    # the dip is the panel's last session: no exit inside the data, so no signal row — the IVP is read all the same
    assert store.get("cand", 0) >= 0 and set(sig["tk"]) <= {"FEAR"}
    assert FEAR.rank_of(0.9, 3.0) < FEAR.rank_of(0.85, 1.0)                   # IVP highest first


def test_the_simulator_reads_one_frozen_copy_of_the_store_named_by_its_digest(monkeypatch, tmp_path):
    """P7: a fetch merging IBKR's bars while the simulator runs moves nothing it reads; a store that differs in the
    window gets another digest, a bar after the window does not."""
    from src.backtest.strategies import fear as FEAR
    monkeypatch.setattr(settings, "fear_long_iv_dir", str(tmp_path / "iv"))
    monkeypatch.setattr(FEAR, "snapshot_root", lambda: tmp_path / "snaps")
    days = _sessions(PREV, 300)
    fl.merge_rows("AAA", [[str(d), 0.3, 0.3, 0.3, 0.3 + i * 1e-4] for i, d in enumerate(days)])     # rising: FEAR
    fl.merge_rows("BBB", [[str(d), 0.5, 0.5, 0.5, 0.5] for d in days[:200]])
    S, cut = days[200], days[250]
    with FEAR.freeze(cut) as st:
        assert FEAR.IV_ROOT == st["path"] and (st["names"], st["bars"], st["through"]) == (2, 251 + 200, str(cut))
        iv = FEAR.IVP()
        assert Path(iv.root) == Path(st["path"]) and (Path(st["path"]) / FEAR.STATE_FILE).exists()
        fl.merge_rows("AAA", [[str(S), 0.3, 0.3, 0.3, 0.01]])                     # a fetch revises a bar meanwhile
        assert iv.at("AAA", S, False) == pytest.approx((200 + 0.5) / 201)         # the frozen close: still the top
        assert FEAR.IVP(tmp_path / "iv").at("AAA", S, False) == pytest.approx(0.5 / 201)    # the live store's: calm
        assert max(r[0] for r in FEAR.store_rows(None, st["path"])["AAA"]) == str(cut)      # nothing after the window
    assert FEAR.IV_ROOT is None and Path(FEAR.IVP().root) == tmp_path / "iv"
    st2 = FEAR.snapshot(cut)                                                      # the revision is in the window
    assert st2["digest"] != st["digest"]
    d = FEAR.diff(st["path"], st2["path"])
    assert d["names_changed"] == {"AAA": {"added": 0, "removed": 0, "revised": 1}} and d["bars_revised"] == 1
    fl.merge_rows("AAA", [[str(days[280]), 0.3, 0.3, 0.3, 0.77]])                 # after the window: the same store
    fl.merge_rows("CCC", [[str(days[290]), 0.4, 0.4, 0.4, 0.4]])                  # a name with no bar in it: the same
    assert FEAR.snapshot(cut)["digest"] == st2["digest"]
    assert sorted(p.name for p in (tmp_path / "snaps").iterdir()) == sorted([st["digest"], st2["digest"]])
    fl.merge_rows("DDD", [[str(days[10]), 0.4, 0.4, 0.4, 0.4]])                   # a new name inside the window
    st3 = FEAR.snapshot(cut)
    assert st3["digest"] != st2["digest"] and FEAR.diff(st2["path"], st3["path"])["names_added"] == ["DDD"]


def test_the_baseline_keeps_every_frozen_copy_in_the_tests_own_dir(tmp_path):
    """The bank's ``iv_store/`` is production's cache (a worktree's ``cache`` is a link to it): never a test's."""
    from src.backtest.strategies import fear as FEAR
    assert FEAR.snapshot_root() == tmp_path / "iv_store" and FEAR.IV_ROOT is None


def test_a_card_build_reads_one_frozen_store_and_records_it(monkeypatch, tmp_path):
    """P7: `reliability.build` runs its whole body on one snapshot through the pooled history's end."""
    from src.backtest import periods
    from src.backtest import reliability as REL
    from src.backtest.strategies import fear as FEAR
    monkeypatch.setattr(settings, "fear_long_iv_dir", str(tmp_path / "iv"))
    monkeypatch.setattr(FEAR, "snapshot_root", lambda: tmp_path / "snaps")
    fl.merge_rows("AAA", [["2026-09-24", 0.3, 0.3, 0.3, 0.3], ["2026-10-01", 0.3, 0.3, 0.3, 0.4]])
    seen = {}

    def fake(sets, sensitivities, seed, cost_snapshot, extra_bars, day, log, iv):
        seen.update(root=FEAR.IV_ROOT, iv=iv, sets=sets)
        return tmp_path / "cards"
    monkeypatch.setattr(REL, "_build", fake)
    assert REL.build(("A", "B")) == tmp_path / "cards"
    hi = periods.pool_span(periods.POOLED)[1]
    assert seen["root"] == seen["iv"]["path"] and seen["iv"]["through"] == hi == REL.iv_through(("A0",))
    assert seen["iv"]["bars"] == 1 and FEAR.IV_ROOT is None                    # the bar after the pool: not read
