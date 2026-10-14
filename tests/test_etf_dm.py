"""The ETF dual-momentum book (`src/signals/etf_dm.py`, PREREG46-51): the month's decision (the best mean 3/6/12-month
return, held only above T-bills), the rotation on the month's first session inside the opening window (the old fund
sold, the pick bought with the whole simulated account), the switch stopping buys but never the rule's sale, the broker
ordering the account's share count, the health banner, and no look-ahead (the decision reads closes through the
previous month's last session only)."""
from __future__ import annotations

from datetime import date, datetime, timedelta, timezone
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import pytest

from config.settings import settings
from src.performance import tracker
from src.signals import etf_dm as em

ET = ZoneInfo("America/New_York")
DAY = date(2026, 10, 1)                        # October's first session; September's last is 2026-09-30
OPEN_TICK = datetime(2026, 10, 1, 9, 30, 20, tzinfo=ET).astimezone(timezone.utc)


@pytest.fixture
def on(monkeypatch, tmp_path):
    monkeypatch.setattr(settings, "enable_etf_dm", True)
    monkeypatch.setattr(settings, "etf_dm_dir", str(tmp_path / "etf_dm"))
    monkeypatch.setattr(settings, "etf_dm_account_initial", 10_000.0)
    monkeypatch.setattr(em, "_extend", lambda tks, through: None)
    return tmp_path


def _series(end: date, n: int, growth: float) -> pd.Series:
    """``n`` daily closes ending on ``end`` growing ``growth`` per session."""
    idx = pd.bdate_range(end=pd.Timestamp(end), periods=n)
    return pd.Series(100.0 * (1 + growth) ** np.arange(n), index=idx)


# ── the decision ─────────────────────────────────────────────────────────────

def test_fund_stats_scores_the_three_lookbacks_and_needs_the_month_end_close():
    me = date(2026, 9, 30)
    s = _series(me, 300, 0.001)
    st = em.fund_stats(s, me)
    for n in (63, 126, 252):
        assert st[f"r{n}"] == pytest.approx(1.001 ** n - 1)
    assert st["score"] == pytest.approx(np.mean([1.001 ** n - 1 for n in (63, 126, 252)]))
    assert em.fund_stats(s.iloc[:-1], me) is None               # the month-end close missing
    assert em.fund_stats(_series(me, 200, 0.001), me) is None     # under 253 closes


def test_the_decision_holds_the_top_fund_when_it_beats_tbills(on, monkeypatch):
    me = date(2026, 9, 30)
    g = {"QQQ": 0.002, "SPY": 0.001, "IWM": 0.0, "TLT": -0.001, "GLD": 0.0015}
    seen = []
    monkeypatch.setattr(em, "closes", lambda tk, through: seen.append(through) or _series(me, 300, g[tk]))
    monkeypatch.setattr(em, "tbill_growth", lambda a, b: 0.04)
    dec = em.compute_decision(DAY)
    assert dec["top"] == "QQQ" and dec["pick"] == "QQQ" and dec["decided_at_close"] == "2026-09-30"
    assert set(seen) == {me}                                       # closes cut at the previous month's last session
    assert em.decision_for(DAY)["pick"] == "QQQ"                   # written once, read back


def test_the_decision_is_cash_when_the_top_fund_trails_tbills(on, monkeypatch):
    me = date(2026, 9, 30)
    monkeypatch.setattr(em, "closes", lambda tk, through: _series(me, 300, 0.0001))   # +2.6% over 12 months
    monkeypatch.setattr(em, "tbill_growth", lambda a, b: 0.05)
    dec = em.compute_decision(DAY)
    assert dec["top"] is not None and dec["pick"] is None


# ── the rotation ─────────────────────────────────────────────────────────────

@pytest.fixture
def rot(monkeypatch, on):
    box = {"pick": "QQQ"}
    monkeypatch.setattr(em, "decision_for", lambda d: {"pick": box["pick"], "decided_at_close": "2026-09-30",
                                                       "funds": {}, "tbill_12m": 0.04})
    monkeypatch.setattr(tracker, "_fetch_price", lambda t: {"QQQ": 400.0, "TLT": 90.0, "SPY": 500.0}.get(t))
    monkeypatch.setattr(tracker, "_reference_close", lambda t: None)
    monkeypatch.setattr(tracker, "_execution_iso", lambda: OPEN_TICK.isoformat())
    return box


def _book():
    return [t for t in tracker._load_trades() if t.get("entry_mechanism") == "etf_dm"]


def test_the_rotation_buys_the_pick_with_the_whole_account(rot):
    assert tracker.rebalance_etf_dm(run_id="r1", now=OPEN_TICK) == 1
    (t,) = _book()
    assert t["ticker"] == "QQQ" and t["action"] == "BUY" and t["status"] == "OPEN"
    assert t["etf_dm_account_shares"] == int(10_000 // (400.0 * 1.0001 + 0.01))
    assert tracker.rebalance_etf_dm(now=OPEN_TICK + timedelta(minutes=30)) == 0      # held: nothing to do


def test_a_new_pick_sells_the_old_fund_and_buys_the_new_one(rot):
    tracker.rebalance_etf_dm(now=OPEN_TICK)
    rot["pick"] = "TLT"
    nov = datetime(2026, 11, 2, 9, 31, tzinfo=ET).astimezone(timezone.utc)
    assert tracker.rebalance_etf_dm(now=nov) == 2
    b = {t["ticker"]: t for t in _book()}
    assert b["QQQ"]["status"] == "CLOSED" and b["QQQ"]["exit_reason"] == "etf_dm_rotation"
    assert b["TLT"]["status"] == "OPEN" and b["TLT"]["etf_dm_account_shares"] > 0


def test_a_cash_month_sells_and_buys_nothing(rot):
    tracker.rebalance_etf_dm(now=OPEN_TICK)
    rot["pick"] = None
    nov = datetime(2026, 11, 2, 9, 31, tzinfo=ET).astimezone(timezone.utc)
    assert tracker.rebalance_etf_dm(now=nov) == 1
    assert [t["status"] for t in _book()] == ["CLOSED"]


def test_no_rotation_outside_the_opening_window(rot):
    for now in (datetime(2026, 10, 1, 10, 31, tzinfo=ET), datetime(2026, 10, 1, 9, 0, tzinfo=ET),
                datetime(2026, 10, 3, 9, 45, tzinfo=ET)):
        assert tracker.rebalance_etf_dm(now=now.astimezone(timezone.utc)) == 0
    assert _book() == []


def test_off_the_book_still_sells_on_its_rule_but_never_buys(rot, monkeypatch):
    tracker.rebalance_etf_dm(now=OPEN_TICK)
    monkeypatch.setattr(settings, "enable_etf_dm", False)
    rot["pick"] = "TLT"
    nov = datetime(2026, 11, 2, 9, 31, tzinfo=ET).astimezone(timezone.utc)
    assert tracker.rebalance_etf_dm(now=nov) == 1
    assert [(t["ticker"], t["status"]) for t in _book()] == [("QQQ", "CLOSED")]


def test_a_pick_another_book_holds_is_not_bought_beside_it(rot):
    short = {"ticker": "QQQ", "status": "OPEN", "action": "SELL", "entry_mechanism": "sel_short",
             "recommendation_id": "sel-1", "entry_date": "2026-09-30", "entry_price": 400.0}
    tracker._save_trades([short])
    assert tracker.rebalance_etf_dm(now=OPEN_TICK) == 0
    assert _book() == []
    assert '"held"' in em.rebalances_path(DAY).read_text(encoding="utf-8")


def test_a_missing_sale_price_holds_the_rotation_for_the_next_tick(rot, monkeypatch):
    tracker.rebalance_etf_dm(now=OPEN_TICK)
    rot["pick"] = "TLT"
    nov = datetime(2026, 11, 2, 9, 31, tzinfo=ET).astimezone(timezone.utc)
    monkeypatch.setattr(tracker, "_fetch_price", lambda t: None)
    assert tracker.rebalance_etf_dm(now=nov) == 0
    assert [(t["ticker"], t["status"]) for t in _book()] == [("QQQ", "OPEN")]


# ── the broker and the banner ────────────────────────────────────────────────

def test_the_broker_sends_the_etf_entry_at_the_accounts_share_count(monkeypatch):
    import src.broker.reconcile as rec
    from tests.test_broker_reconcile import FakeBroker, _open_trade
    store = {"trades": []}
    monkeypatch.setattr(rec.repo, "load_trades", lambda: store["trades"])
    monkeypatch.setattr(rec.repo, "save_trades", lambda t: store.update(trades=t))
    monkeypatch.setattr(settings, "broker_mode", "ibkr_paper")
    monkeypatch.setattr(settings, "broker_settle_seconds", 0)
    monkeypatch.setattr(settings, "enable_sel_short", False)
    monkeypatch.setattr(settings, "enable_dip_long", False)
    monkeypatch.setattr(settings, "enable_etf_dm", True)
    monkeypatch.setattr(settings, "enable_legacy_entries", False)
    monkeypatch.setattr(rec, "usd_per_unit", lambda ccy: 1.0)
    from src.performance import market_calendar
    monkeypatch.setattr(market_calendar, "current_session", lambda now=None: "rth")
    t = _open_trade("QQQ")
    t.update(entry_mechanism="etf_dm", etf_dm_account_shares=24, recommendation_id="etf-1")
    legacy = _open_trade("MSFT")
    store["trades"] = [t, legacy]
    b = FakeBroker()
    rec.sync(broker=b)
    assert [(o.ticker, o.side, o.quantity) for o in b.orders] == [("QQQ", "BUY", 24)]
    assert legacy["broker_status"] == "LEGACY_ENTRY_SHADOWED"


def test_health_flags_a_holding_that_is_not_the_pick(on):
    noon = datetime(2026, 10, 1, 12, 0, tzinfo=ET).astimezone(timezone.utc)
    assert any("no decision" in p for p in em.health(noon, trades=[])["problems"])
    em.DL._write_json(em.decision_path(DAY), {"pick": "QQQ"})
    assert any("cash" in p for p in em.health(noon, trades=[])["problems"])
    held = [{"ticker": "QQQ", "status": "OPEN", "entry_mechanism": "etf_dm"}]
    h = em.health(noon, trades=held)
    assert h["problems"] == [] and h["notes"]
    early = datetime(2026, 10, 1, 9, 45, tzinfo=ET).astimezone(timezone.utc)
    assert em.health(early, trades=[])["problems"] == []
