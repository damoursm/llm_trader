"""The simulated account the vol arm is sized from (user directive 2026-10-05: "Have the account
based sizing considering the simulated 5000$+1000$ every two weeks").

What must hold: the money paid in is $5,000 plus $1,000 every 14 calendar days from the start, never a
deposit dated after the tick; equity adds every funded trade's dollar P&L from the ledger (shares x
entry x net return; open trades at their mark); a new short is the audited engine's size — a tenth
of equity in whole shares, at most equity minus the open shorts' value, at most 1% of the stock's
dollar volume, within the Reg T initial-margin room, none under $2,000 of equity; the vol arm's picks
are sized from it (the model's are not) and the broker orders exactly that share count; equity below
the open shorts' maintenance buys back every funded short.
"""
from __future__ import annotations

from datetime import date, datetime, timedelta
from zoneinfo import ZoneInfo

import pytest

from config.settings import settings
from src.performance import sim_account as sa

ET = ZoneInfo("America/New_York")


@pytest.fixture
def on(monkeypatch):
    monkeypatch.setattr(settings, "enable_sel_short_account_sizing", True)
    monkeypatch.setattr(settings, "sel_short_account_start", "2026-10-05")
    monkeypatch.setattr(settings, "sel_short_account_initial", 5000.0)
    monkeypatch.setattr(settings, "sel_short_account_deposit", 1000.0)
    monkeypatch.setattr(settings, "sel_short_account_deposit_days", 14)
    monkeypatch.setattr(settings, "sel_short_account_slices", 10)
    monkeypatch.setattr(settings, "sel_short_account_max_dollar_volume_share", 0.01)
    monkeypatch.setattr(settings, "sel_short_account_min_equity", 2000.0)
    return settings


def at(y, m, d, hh=11, mm=0):
    return datetime(y, m, d, hh, mm, tzinfo=ET)


def _t(n, entry, ret, status="OPEN", mark=None, **kw):
    return dict({"entry_mechanism": "sel_short", "sel_arm": "vol", "status": status, "sel_account_shares": n,
                 "entry_price": entry, "current_price": mark if mark is not None else entry, "return_pct": ret}, **kw)


def test_the_money_paid_in_follows_the_deposit_plan_and_never_a_later_deposit(on):
    assert sa.paid_in(at(2026, 10, 5)) == 5000.0
    assert sa.paid_in(at(2026, 10, 18, 23, 59)) == 5000.0                  # the first deposit is due Oct 19
    assert sa.paid_in(at(2026, 10, 19, 0, 1)) == 6000.0
    assert sa.paid_in(at(2026, 11, 2)) == 7000.0
    assert sa.paid_in(at(2026, 9, 1)) == 5000.0                            # before the start: the start balance


def test_equity_adds_every_funded_trades_dollar_pnl(on):
    trades = [_t(10, 50.0, 20.0, status="CLOSED"),                        # +$100 realized
              _t(5, 40.0, 24.0, mark=30.0),                               # +$48 open, worth $150 at the mark
              _t(3, 40.0, 50.0, sel_account_shares=None),                 # not funded (flat-size trade)
              dict(_t(4, 10.0, 10.0), entry_mechanism="legacy")]          # another book
    st = sa.state(trades, at(2026, 10, 6))
    assert st["realized"] == pytest.approx(100.0) and st["unrealized"] == pytest.approx(48.0)
    assert st["equity"] == pytest.approx(5148.0) and st["gross"] == pytest.approx(150.0)
    assert st["maint"] == pytest.approx(max(5 * 5.0, 0.30 * 150.0))       # $5 a share vs 30%: 45
    assert st["open"] == 1 and st["funded"] == 2


def test_a_new_short_is_a_tenth_of_equity_in_whole_shares_within_every_cap(on):
    now = at(2026, 10, 6)
    assert sa.size([], 25.0, 3e7, now)[:2] == (20, "ok")                   # $500 / $25
    assert sa.size([], 24.0, 3e7, now)[:2] == (20, "ok")                   # floor(20.8)
    assert sa.size([], 25.0, 30_000.0, now)[:2] == (12, "ok")              # 1% of $30k a day = $300
    assert sa.size([], 25.0, float("nan"), now)[:2] == (20, "ok")          # unknown volume: no volume cap
    assert sa.size([], 600.0, 3e7, now)[:2] == (0, "account_too_small")    # one share is more than a slice
    full = [_t(100, 49.0, 0.0, mark=49.0)]                                # $4,900 short of $5,000
    assert sa.size(full, 25.0, 3e7, now)[:2] == (4, "ok")                  # only $100 of room left
    fuller = [_t(102, 49.0, 0.0, mark=49.0)]
    assert sa.size(fuller, 25.0, 3e7, now)[:2] == (0, "account_full")
    poor = [_t(10, 100.0, -300.0, status="CLOSED")]                       # -$3,000: $2,000 left... and less
    assert sa.size(poor + [_t(1, 10.0, -1.0, status="CLOSED")], 25.0, 3e7, now)[:2] == (0, "account_below_minimum")


def test_the_initial_margin_room_caps_a_short_when_it_binds_first(on, monkeypatch):
    """Half the account in a $1 stock: the slice buys 2,500 shares, but Reg T's initial margin
    under $5 is $2.50 a share, so the margin room ($5,000 less the entry's fees) buys 1,995."""
    monkeypatch.setattr(settings, "sel_short_account_slices", 2)
    n, why, st = sa.size([], 1.0, 3e7, at(2026, 10, 6))
    assert why == "ok" and n == 1995 and n < 2500
    assert sa.size([], 25.0, 1_000.0, at(2026, 10, 6))[:2] == (0, "volume_cap")   # 1% of $1,000 a day = $10


def test_a_margin_call_is_equity_below_the_open_shorts_maintenance(on):
    now = at(2026, 10, 6)
    calm = [_t(10, 50.0, -10.0, mark=55.0)]
    assert sa.margin_call(calm, now)[0] is False
    squeezed = [_t(100, 10.0, -600.0, mark=70.0)]                         # -$6,000, maintenance $2,100
    called, st = sa.margin_call(squeezed, now)
    assert called and st["equity"] < st["maint"]
    assert sa.margin_call([], now)[0] is False


def test_only_the_vol_arm_is_funded(on, monkeypatch):
    assert sa.arm_funded("vol") and not sa.arm_funded("model") and not sa.arm_funded("etf")
    monkeypatch.setattr(settings, "enable_sel_short_account_sizing", False)
    assert not sa.arm_funded("vol")


def test_the_entry_step_sizes_vol_picks_from_the_account_and_the_broker_orders_that_count(on, monkeypatch):
    from src.broker import reconcile
    from tests.test_sel_short import _pick, _tracker_env
    picks = [dict(_pick("ABC"), arm="vol", score=4.2, target=9.5, dv20=3e7), _pick("XYZ")]
    tracker, consumed, _ = _tracker_env(monkeypatch, picks)              # live price 10.20
    assert tracker.record_sel_short_trades(run_id="r1") == 2
    t = {x["ticker"]: x for x in tracker._load_trades() if x.get("entry_mechanism") == "sel_short"}
    assert t["ABC"]["sel_account_shares"] == 49                            # floor($500 / $10.20)
    assert t["ABC"]["sel_account_notional"] == pytest.approx(49 * 10.2)
    assert t["ABC"]["sel_account_equity"] == pytest.approx(5000.0)
    assert "1/10 of the simulated account" in t["ABC"]["rationale"]
    assert "sel_account_shares" not in t["XYZ"]                           # the model arm: the flat size
    assert reconcile._entry_qty(t["ABC"], 12.0, 1e6, 0.73) == 49           # whatever price it is resent at
    assert reconcile._entry_qty(t["XYZ"], 10.2, 1e6, 0.73) > 0
    # a second vol pick in the next pass sees the first one's exposure
    tracker2, _, _ = _tracker_env(monkeypatch, [dict(_pick("DEF", bar=2), arm="vol", target=9.5, dv20=3e7)])
    assert tracker2.record_sel_short_trades(run_id="r2") == 1
    d = [x for x in tracker2._load_trades() if x["ticker"] == "DEF"][0]
    assert d["sel_account_shares"] == 49 and d["sel_account_equity"] == pytest.approx(5000.0)


def test_a_vol_pick_the_account_cannot_fund_is_journaled_with_the_reason(on, monkeypatch):
    from tests.test_sel_short import _pick, _tracker_env
    monkeypatch.setattr(settings, "sel_short_account_initial", 1500.0)    # under FINRA's $2,000
    tracker, consumed, _ = _tracker_env(monkeypatch, [dict(_pick("ABC"), arm="vol", target=9.5, dv20=3e7)])
    assert tracker.record_sel_short_trades(run_id="r1") == 0
    assert list(consumed.values()) == ["account_below_minimum"]


def test_the_monitor_buys_back_every_funded_short_on_a_margin_call(on, monkeypatch):
    from tests.test_sel_short import _seed_trades
    from src.performance import tracker
    monkeypatch.setattr(settings, "enable_sel_short", True)
    now = datetime.now(ET)
    marked = (now - timedelta(minutes=5)).isoformat()
    deadline = (now + timedelta(days=20)).isoformat()
    base = {"action": "SELL", "entry_mechanism": "sel_short", "sel_arm": "vol", "current_price_datetime": marked,
            "sel_target_price": 1.0, "sel_deadline": deadline, "entry_date": date.today().isoformat()}
    _seed_trades(tracker, [dict(base, ticker="SQZ", recommendation_id="a", sel_account_shares=100, entry_price=10.0,
                                current_price=70.0, return_pct=-600.0),
                           dict(base, ticker="CALM", recommendation_id="b", sel_account_shares=10, entry_price=20.0,
                                current_price=19.0, return_pct=5.0),
                           dict(base, ticker="FLAT", recommendation_id="c", sel_arm="model", entry_price=20.0,
                                current_price=21.0, return_pct=-5.0)])
    tracker.monitor_sel_short_positions(now=now)
    st = {t["ticker"]: t for t in tracker._load_trades()}
    assert st["SQZ"]["status"] != "OPEN" and st["SQZ"]["exit_reason"] == "sel_margin_call"
    assert st["CALM"]["status"] != "OPEN" and st["CALM"]["exit_reason"] == "sel_margin_call"
    assert st["FLAT"]["status"] == "OPEN"                                  # not the account's
