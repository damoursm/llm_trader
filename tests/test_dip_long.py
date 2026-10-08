"""The mega-cap dip long book (user directive 2026-10-08: "Deploy the mega-cap dip buying strategy to live
production"; `src/signals/dip_long.py`): the study's construction and rule reproduced exactly, its own cash
account, the entry step (deepest dip first, settled once a day, never on another book's name), the exits at
the open, the broker sending its orders at the account's share count, and the legacy exits leaving it alone.
The look-ahead guards are in `tests/test_no_lookahead.py`."""
from __future__ import annotations

from datetime import date, datetime, timedelta, timezone
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import pytest

from config.settings import settings
from src.performance import tracker
from src.signals import dip_long as dl
from src.signals import sel_short as ss

ET = ZoneInfo("America/New_York")
DAY = date(2026, 10, 8)                       # a Thursday session
OPEN_TICK = datetime(2026, 10, 8, 9, 30, 20, tzinfo=ET).astimezone(timezone.utc)


@pytest.fixture
def on(monkeypatch, tmp_path):
    monkeypatch.setattr(settings, "enable_dip_long", True)
    monkeypatch.setattr(settings, "dip_long_dir", str(tmp_path / "dip_long"))
    monkeypatch.setattr(settings, "dip_long_account_initial", 10_000.0)
    monkeypatch.setattr(settings, "dip_long_account_slices", 10)
    monkeypatch.setattr(dl, "_extend", lambda tks, through: None)
    return tmp_path


# ── the study's construction ─────────────────────────────────────────────────

def _study_rsi2(c):
    """lg_mega.rsi2, verbatim."""
    d = np.diff(c, prepend=np.nan)
    up, dn = np.where(d > 0, d, 0.0), np.where(d < 0, -d, 0.0)
    ru = pd.Series(up).ewm(alpha=0.5, adjust=False).mean().to_numpy()
    rd = pd.Series(dn).ewm(alpha=0.5, adjust=False).mean().to_numpy()
    with np.errstate(divide="ignore", invalid="ignore"):
        r = 100 - 100 / (1 + ru / rd)
    return np.where(rd == 0, 100.0, r)


def _bars(days, closes, volume=1_000_000.0):
    """Flat 30-minute regular-hours bars at each session's close (naive-UTC bar starts)."""
    idx, rows = [], []
    for d, c in zip(days, closes):
        for k in range(13):
            idx.append(pd.Timestamp(datetime(d.year, d.month, d.day, 9, 30, tzinfo=ET) + timedelta(minutes=30 * k))
                       .tz_convert("UTC").tz_localize(None))
            rows.append((c, c * 1.001, c * 0.999, c, volume))
    return pd.DataFrame(rows, columns=["Open", "High", "Low", "Close", "Volume"], index=pd.DatetimeIndex(idx))


def test_daily_bars_are_the_studys_regular_hours_sessions():
    days = [date(2026, 10, 6), date(2026, 10, 7)]
    df = _bars(days, [10.0, 11.0], volume=5.0)
    df.iloc[3, df.columns.get_loc("High")] = 12.5                  # an intraday high
    df.iloc[0, df.columns.get_loc("Open")] = 9.5                   # the session's first bar opens lower
    out = dl.daily_bars(df)
    assert list(out.index.date) == days
    assert out.loc["2026-10-06", ["o", "h", "c", "v"]].tolist() == [9.5, 12.5, 10.0, 65.0]
    assert out.loc["2026-10-07", "l"] == pytest.approx(11.0 * 0.999)


def test_rsi2_is_the_studys():
    c = 100 * np.exp(np.cumsum(np.random.default_rng(3).normal(0, 0.02, 400)))
    np.testing.assert_allclose(dl.rsi2(c), _study_rsi2(c))


def test_the_signal_rule_matches_the_study_day_by_day(monkeypatch):
    """Every session of a random walk: the study's M1_1B condition (close > SMA200, RSI2 < 10, 20-session
    mean close x volume >= $1B, 220 bars) equals `qualifies(stats_at(...))` cut at that session."""
    rng = np.random.default_rng(11)
    days = ss.sessions_before(DAY, 600)
    c = 50 * np.exp(np.cumsum(rng.normal(0.0008, 0.02, len(days))))
    v = rng.uniform(2e6, 6e7, len(days)) / 13                      # some sessions over $1B, some under
    daily = dl.daily_bars(pd.concat([_bars([d], [x], volume=y) for d, x, y in zip(days, c, v)]))
    s = pd.Series(c)
    sma200 = s.rolling(200).mean().to_numpy()
    dv20 = pd.Series(c * v * 13).rolling(20).mean().to_numpy()
    r2 = _study_rsi2(c)
    hits = 0
    for i in range(200, len(days)):
        study = bool(c[i] > sma200[i] and r2[i] < 10 and dv20[i] >= 1e9) and i + 1 >= 220
        live = dl.qualifies(dl.stats_at(daily, days[i]))
        assert study == live, days[i]
        hits += study
    assert hits >= 3                                               # the walk does produce signals


# ── exits ────────────────────────────────────────────────────────────────────

def _exit_case(monkeypatch, closes, entry_idx):
    days = ss.sessions_before(DAY, len(closes))
    df = _bars(days, closes)
    from src.data import intraday_store
    monkeypatch.setattr(intraday_store, "load_deep_30m", lambda tk: df)
    return days, {"ticker": "SYN", "dip_entry_day": days[entry_idx].isoformat()}


def test_a_close_above_the_five_session_average_sells_at_the_next_open(monkeypatch, on):
    closes = [100.0] * 30 + [97.0, 94.0, 92.0, 96.5]               # the last close rebounds above its average
    days, trade = _exit_case(monkeypatch, closes, len(closes) - 2)
    reason, info = dl.exit_check(trade, DAY)
    assert reason == "dip_rebound" and info["rebound_session"] == days[-1].isoformat() and info["held"] == 2
    assert dl.exit_check(trade, days[-1])[0] is None               # decided only once that close exists


def test_the_entry_sessions_own_close_counts(monkeypatch, on):
    closes = [100.0] * 30 + [97.0, 94.0, 99.0]                     # bought at the open of the last session
    days, trade = _exit_case(monkeypatch, closes, len(closes) - 1)
    assert dl.exit_check(trade, DAY)[0] == "dip_rebound"


def test_ten_sessions_without_a_rebound_sell_on_time(monkeypatch, on):
    closes = [100.0] * 30 + [100.0 - 2 * k for k in range(1, 12)]  # falling every session
    days, trade = _exit_case(monkeypatch, closes, len(closes) - 10)
    reason, info = dl.exit_check(trade, DAY)
    assert reason == "dip_time" and info["held"] == 10
    days9, trade9 = _exit_case(monkeypatch, closes, len(closes) - 9)
    assert dl.exit_check(trade9, DAY)[0] is None                    # nine sessions: still held


def test_a_missed_day_still_sells_on_the_first_rebound(monkeypatch, on):
    """The rebound happened two sessions ago and the book missed that open: it sells now, not never."""
    closes = [100.0] * 30 + [95.0, 99.0, 97.0]
    days, trade = _exit_case(monkeypatch, closes, len(closes) - 3)
    reason, info = dl.exit_check(trade, DAY)
    assert reason == "dip_rebound" and info["rebound_session"] == days[-2].isoformat()


# ── the account ──────────────────────────────────────────────────────────────

def _dip_trade(tk, shares, entry, ret=0.0, status="OPEN"):
    return {"ticker": tk, "entry_mechanism": "dip_long", "action": "BUY", "status": status,
            "dip_account_shares": shares, "entry_price": entry, "return_pct": ret}


def test_the_account_is_the_studys_cash_account(on):
    trades = [_dip_trade("A", 10, 100.0, ret=10.0, status="CLOSED"),       # +$100 realized
              _dip_trade("B", 5, 200.0, ret=-5.0),                         # -$50 open, $1,000 at work
              {"ticker": "S", "entry_mechanism": "sel_short", "sel_account_shares": 9, "entry_price": 50.0,
               "return_pct": 50.0, "status": "OPEN", "action": "SELL"}]   # another book: not counted
    st = dl.account_state(trades)
    assert st["equity"] == pytest.approx(10_050.0) and st["cash"] == pytest.approx(9_100.0) and st["open"] == 1
    n, why, _ = dl.size(trades, 50.0)
    assert why == "ok" and n == int(10_050.0 / 10 // (50.0 * 1.0001 + 0.01))   # 1/10 of the equity


def test_the_account_stops_at_ten_slices_and_at_its_cash(on, monkeypatch):
    ten = [_dip_trade(f"T{i}", 1, 10.0) for i in range(10)]
    assert dl.size(ten, 50.0)[1] == "no_slot"
    monkeypatch.setattr(settings, "dip_long_account_initial", 1_500.0)
    spent = [_dip_trade("B", 14, 100.0)]                                  # $1,400 at work, $100 cash left
    n, why, st = dl.size(spent, 30.0)
    assert why == "ok" and n * 30.0 * 1.0001 + max(1.0, 0.005 * n) <= st["cash"] and n == 3
    assert dl.size(spent, 150.0)[1] == "account_too_small"


# ── the entry step ───────────────────────────────────────────────────────────

def _sig(tk, rsi, close=100.0):
    return {"ticker": tk, "session": "2026-10-07", "close": close, "sma_trend": 90.0, "sma_exit": 104.0,
            "rsi2": rsi, "dv20": 2e9, "bars": 1400}


@pytest.fixture
def entry(monkeypatch, on):
    day = {"day": DAY.isoformat(), "signal_session": "2026-10-07", "signals": [_sig("AAA", 4.0), _sig("BBB", 8.0)]}
    monkeypatch.setattr(dl, "signals_for", lambda d: day)
    monkeypatch.setattr(tracker, "_fetch_price", lambda t: {"AAA": 101.0, "BBB": 52.0}.get(t))
    monkeypatch.setattr(tracker, "_reference_close", lambda t: None)
    monkeypatch.setattr(tracker, "_execution_iso", lambda: OPEN_TICK.isoformat())
    return day


def test_the_open_buys_the_deepest_dip_first_at_the_accounts_size(entry):
    assert tracker.record_dip_long_trades(run_id="r1", now=OPEN_TICK) == 2
    t = {x["ticker"]: x for x in tracker._load_trades() if x.get("entry_mechanism") == "dip_long"}
    assert t["AAA"]["dip_rank"] == 1 and t["BBB"]["dip_rank"] == 2
    assert t["AAA"]["action"] == "BUY" and t["AAA"]["dip_account_shares"] == int(1_000 // (101.0 * 1.0001 + 0.01))
    assert t["AAA"]["dip_entry_day"] == DAY.isoformat() and t["AAA"]["dip_signal_day"] == "2026-10-07"
    assert dl.settled(DAY) == {"AAA": "opened", "BBB": "opened"}
    assert tracker.record_dip_long_trades(run_id="r2", now=OPEN_TICK + timedelta(minutes=30)) == 0   # settled


def test_no_entry_outside_the_opening_window_or_regular_hours(entry):
    late = datetime(2026, 10, 8, 10, 31, tzinfo=ET).astimezone(timezone.utc)
    pre = datetime(2026, 10, 8, 9, 0, tzinfo=ET).astimezone(timezone.utc)
    sat = datetime(2026, 10, 10, 9, 45, tzinfo=ET).astimezone(timezone.utc)
    for now in (late, pre, sat):
        assert not dl.in_entry_window(now) and tracker.record_dip_long_trades(now=now) == 0
    assert dl.in_entry_window(OPEN_TICK)


def test_a_name_another_book_holds_is_settled_held(entry):
    tracker._save_trades([{"ticker": "AAA", "entry_mechanism": "sel_short", "action": "SELL", "status": "OPEN",
                           "entry_price": 100.0, "entry_date": "2026-10-07", "recommendation_id": "s1"}])
    assert tracker.record_dip_long_trades(now=OPEN_TICK) == 1
    assert dl.settled(DAY) == {"AAA": "held", "BBB": "opened"}


def test_a_repeat_signal_stacks_on_the_books_own_long(entry):
    tracker._save_trades([dict(_dip_trade("AAA", 9, 110.0), entry_date="2026-10-07", recommendation_id="d0",
                               dip_entry_day="2026-10-07")])
    assert tracker.record_dip_long_trades(now=OPEN_TICK) == 2
    aaa = [x for x in tracker._load_trades() if x["ticker"] == "AAA" and x["status"] == "OPEN"]
    assert sorted(x.get("dip_stack_n") or 0 for x in aaa) == [0, 2]


def test_a_missing_price_is_retried_next_tick(entry, monkeypatch):
    monkeypatch.setattr(tracker, "_fetch_price", lambda t: None if t == "AAA" else 52.0)
    assert tracker.record_dip_long_trades(now=OPEN_TICK) == 1
    assert "AAA" not in dl.settled(DAY)
    monkeypatch.setattr(tracker, "_fetch_price", lambda t: 101.0)
    assert tracker.record_dip_long_trades(now=OPEN_TICK + timedelta(minutes=30)) == 1


def test_a_full_account_settles_no_slot(entry):
    tracker._save_trades([dict(_dip_trade(f"T{i}", 1, 10.0), recommendation_id=f"t{i}", entry_date="2026-10-07",
                               dip_entry_day="2026-10-07") for i in range(10)])
    assert tracker.record_dip_long_trades(now=OPEN_TICK) == 0
    assert dl.settled(DAY) == {"AAA": "no_slot", "BBB": "no_slot"}


# ── the exit step ────────────────────────────────────────────────────────────

def test_the_open_sells_a_long_whose_exit_is_due(on, monkeypatch):
    tracker._save_trades([dict(_dip_trade("AAA", 9, 100.0), recommendation_id="d1", entry_date="2026-10-06",
                               dip_entry_day="2026-10-06", type="STOCK", position_size_multiplier=1.0,
                               current_price=103.0, entry_session="rth")])
    monkeypatch.setattr(dl, "exit_check", lambda t, d: ("dip_rebound", {"held": 2, "close": 103.0, "sma_exit": 101.0,
                                                                        "rebound_session": "2026-10-07"}))
    monkeypatch.setattr(tracker, "_fetch_price", lambda t: 104.0)
    monkeypatch.setattr(tracker, "_reference_close", lambda t: None)
    monkeypatch.setattr(tracker, "_execution_iso", lambda: OPEN_TICK.isoformat())
    assert tracker.monitor_dip_long_positions(now=OPEN_TICK) == 1
    t = tracker._load_trades()[0]
    assert t["status"] == "CLOSED" and t["exit_reason"] == "dip_rebound" and t["exit_price"] == 104.0
    assert dl._read_jsonl(dl.exits_path(DAY))[0]["reason"] == "dip_rebound"
    after = datetime(2026, 10, 8, 17, 0, tzinfo=ET).astimezone(timezone.utc)
    assert tracker.monitor_dip_long_positions(now=after) == 0       # extended hours: never


def test_the_legacy_exits_and_the_signal_reversal_leave_a_dip_long_alone(on):
    from src.models import Recommendation
    tracker._save_trades([dict(_dip_trade("AAA", 9, 100.0), recommendation_id="d1", entry_date="2026-10-06",
                               type="STOCK", position_size_multiplier=1.0, current_price=80.0,
                               entry_session="rth", confidence=0.9)])
    rec = Recommendation.model_construct(ticker="AAA", action="SELL")
    tracker.close_trades_on_signal_reversal([rec], hold_prompt_active=False)
    tracker.monitor_open_positions(signals_by_ticker={}, hold_prompt_active=False)
    assert tracker._load_trades()[0]["status"] == "OPEN"


# ── the broker ───────────────────────────────────────────────────────────────

def test_the_broker_sends_a_dip_entry_at_the_accounts_share_count(monkeypatch):
    import src.broker.reconcile as rec
    from tests.test_broker_reconcile import FakeBroker, _open_trade
    store = {"trades": []}
    monkeypatch.setattr(rec.repo, "load_trades", lambda: store["trades"])
    monkeypatch.setattr(rec.repo, "save_trades", lambda t: store.update(trades=t))
    monkeypatch.setattr(settings, "broker_mode", "ibkr_paper")
    monkeypatch.setattr(settings, "broker_settle_seconds", 0)
    monkeypatch.setattr(settings, "enable_sel_short", True)             # the live books trade ...
    monkeypatch.setattr(settings, "enable_dip_long", True)
    monkeypatch.setattr(settings, "enable_legacy_entries", False)       # ... and the legacy books are shadow
    monkeypatch.setattr(rec, "usd_per_unit", lambda ccy: 1.0)
    from src.performance import market_calendar
    monkeypatch.setattr(market_calendar, "current_session", lambda now=None: "rth")
    dip = _open_trade("AAPL")
    dip.update(entry_mechanism="dip_long", dip_account_shares=7, recommendation_id="dip-1")
    legacy = _open_trade("MSFT")
    store["trades"] = [dip, legacy]
    b = FakeBroker()
    rec.sync(broker=b)
    assert [(o.ticker, o.side, o.quantity) for o in b.orders] == [("AAPL", "BUY", 7)]
    assert legacy["broker_status"] == "LEGACY_ENTRY_SHADOWED"


# ── health ───────────────────────────────────────────────────────────────────

def test_health_flags_a_day_the_book_never_looked_at_the_open(on, tmp_path):
    noon = datetime(2026, 10, 8, 12, 0, tzinfo=ET).astimezone(timezone.utc)
    assert any("no signal computation" in p for p in dl.health(noon)["problems"])
    dl._write_json(dl.signals_path(DAY), {"signals": [], "stale": [], "inactive": ["WBD"],
                                          "signal_session": "2026-10-07"})
    h = dl.health(noon)
    assert h["problems"] == [] and h["notes"]
    dl._write_json(dl.signals_path(DAY), {"signals": [], "stale": ["JPM"], "signal_session": "2026-10-07"})
    assert any("JPM" in p for p in dl.health(noon)["problems"])
    early = datetime(2026, 10, 8, 9, 45, tzinfo=ET).astimezone(timezone.utc)
    assert dl.health(early)["problems"] == [] or not dl.signals_path(DAY).exists()


# ── the tick ─────────────────────────────────────────────────────────────────

def test_the_dip_orders_are_synced_before_the_scorer_wait(monkeypatch):
    """At the open the book's buys (and the sells of the tick's first half) go to the broker at once —
    then the selection short's scorer wait, its entries and the usual sync."""
    import src.pipeline as pl
    calls = []
    monkeypatch.setattr(tracker, "record_dip_long_trades", lambda run_id=None: calls.append(("dip", run_id)) or 2)
    monkeypatch.setattr(ss, "wait", lambda h, timeout=None: calls.append(("wait", h)))
    monkeypatch.setattr(tracker, "record_sel_short_trades", lambda run_id=None: calls.append(("record", run_id)))
    monkeypatch.setattr(pl, "_broker_sync_watchdogged",
                        lambda run_id, a: calls.append(("sync", a)) or {"ok": True, "entries_submitted": 1})
    rep = pl._live_entries_and_sync("r1", {"proc": None})
    assert [c[0] for c in calls] == ["dip", "sync", "wait", "record", "sync"]
    assert rep["entries_submitted"] == 2                       # the two syncs' reports merged
    calls.clear()
    monkeypatch.setattr(tracker, "record_dip_long_trades", lambda run_id=None: 0)
    pl._live_entries_and_sync("r2", {"proc": None}, dip_closed=1)          # a sale alone also syncs at once
    assert [c[0] for c in calls] == ["sync", "wait", "record", "sync"]
    calls.clear()
    pl._live_entries_and_sync("r3", {"proc": None})
    assert [c[0] for c in calls] == ["wait", "record", "sync"]             # nothing of the book: as before
