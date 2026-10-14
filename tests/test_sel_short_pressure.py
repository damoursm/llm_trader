"""PREREG71's V6 in the entry step (user directive 2026-10-10: "Deploy the narrow part that is working in live"):
on a bar where the vol arm has a live pick, each of the bar's other candidates that pass the whole rule (the
journaled backups) and are under borrow PRESSURE (HIGH, `sel_pressure`) is shorted too — its own trade.

What must hold: a HIGH backup opens its own vol trade beside the pick (stamped ``sel_pressure_extra_of`` / its
pressure) and a LOW one does not; when IBKR cannot lend the pick, the fallback takes the first lendable backup and
the extras are the OTHER HIGH backups — never the same name twice; one IBKR cannot lend now is re-checked next tick
(never consumed); only the vol arm takes extras; the switch turns it off; the entry journal names the pick an extra
stood beside.
"""
from __future__ import annotations

import json
from datetime import date, datetime
from zoneinfo import ZoneInfo

import pytest

from config.settings import settings
from src.signals import sel_pressure as SP
from src.signals import sel_short as ss

ET = ZoneInfo("America/New_York")
DAY = date(2026, 9, 28)


@pytest.fixture
def on(monkeypatch):
    for k, v in {"enable_sel_short": True, "enable_sel_short_vol": True, "enable_sel_short_vol2": True,
                 "enable_sel_short_vol_fallback": True, "enable_sel_short_vol_pressure_extras": True,
                 "enable_sel_short_account_sizing": True, "enable_sel_short_ibkr_whatif": False,
                 "enable_sel_short_borrow_retry": True, "sel_short_traded_arms": "vol,vol2"}.items():
        monkeypatch.setattr(settings, k, v)
    return settings


def _backup(tk, rank):
    return {"ticker": tk, "rank": rank, "score": 4.0, "px": 10.0, "pre5": 8.0, "runup_pct": 25.0, "target": 8.0,
            "deadline": ss.bar_end_et(ss.session_after(DAY, 15), 1).isoformat(), "days_to_cover": 0.5, "rvol": 2.0,
            "atr_pct": 4.0, "dv20": 5e7}


def _pick(tk="AAA", arm="vol", backups=()):
    return {"day": DAY.isoformat(), "bar_of_day": 1, "bar_end": ss.bar_end_et(DAY, 1).isoformat(), "ticker": tk,
            "decision": "short", "px": 10.0, "pre5": 8.0, "target": 8.0, "arm": arm,
            "deadline": ss.bar_end_et(ss.session_after(DAY, 15), 1).isoformat(), "score": 4.2, "runup_pct": 25.0,
            "days_to_cover": 0.5, "rvol": 2.0, "atr_pct": 4.2, "dv20": 5e7, "backups": list(backups)}


def _env(monkeypatch, picks, lendable=lambda t: True, high=("BBB",)):
    from src.data import ibkr_borrow
    from src.performance import tracker
    monkeypatch.setattr(ss, "pending_entries", lambda now=None: list(picks))
    monkeypatch.setattr(tracker, "_fetch_price", lambda t: 10.2)
    monkeypatch.setattr(tracker, "_reference_close", lambda t: None)
    monkeypatch.setattr(tracker, "_execution_iso", lambda: "2026-09-28T14:40:00+00:00")
    monkeypatch.setattr(SP, "pressure_of", lambda t, d, build_missing=True: (0.4, True) if t in high else (0.05, False))
    stamp = datetime(2026, 9, 28, 9, 0, tzinfo=ET)

    def block(ticker, price, base=None, now=None, max_fee_pct="default"):
        if lendable(ticker):
            return None, ibkr_borrow.Borrow(ticker, 12.0, -5.0, 900_000, stamp)
        return "no_borrow", None
    monkeypatch.setattr(ibkr_borrow, "short_block", block)
    return tracker


def _opened(tracker):
    return sorted(((t["ticker"], t.get("sel_pressure_extra_of"), t.get("sel_fallback_of"))
                   for t in tracker._load_trades() if t.get("entry_mechanism") == "sel_short"), key=str)


def test_a_high_backup_is_shorted_beside_the_pick_and_a_low_one_is_not(monkeypatch, on):
    tracker = _env(monkeypatch, [_pick(backups=[_backup("BBB", 2), _backup("CCC", 3)])])
    assert tracker.record_sel_short_trades(run_id="r1") == 2
    assert _opened(tracker) == [("AAA", None, None), ("BBB", "AAA", None)]
    x = next(t for t in tracker._load_trades() if t["ticker"] == "BBB")
    assert x["sel_arm"] == "vol" and x["sel_pressure"] == pytest.approx(0.4) and "PREREG71 V6" in x["rationale"]
    assert x["sel_account_shares"] > 0
    lines = [json.loads(s) for s in ss.entries_path(DAY).read_text(encoding="utf-8").splitlines()]
    assert any(r["ticker"] == "BBB" and r.get("pressure_extra_of") == "AAA" for r in lines)


def test_the_fallback_comes_first_then_the_other_high_backups(monkeypatch, on):
    tracker = _env(monkeypatch, [_pick(backups=[_backup("BBB", 2), _backup("CCC", 3)])],
                   lendable=lambda t: t != "AAA", high=("BBB", "CCC"))
    assert tracker.record_sel_short_trades(run_id="r1") == 2
    assert _opened(tracker) == [("BBB", None, "AAA"), ("CCC", "AAA", None)]


def test_an_extra_ibkr_cannot_lend_is_rechecked_next_tick(monkeypatch, on):
    tracker = _env(monkeypatch, [_pick(backups=[_backup("BBB", 2)])], lendable=lambda t: t != "BBB")
    assert tracker.record_sel_short_trades(run_id="r1") == 1
    assert _opened(tracker) == [("AAA", None, None)]
    bkey = ss.pick_key({"day": DAY.isoformat(), "bar_of_day": 1, "ticker": "BBB", "arm": "vol"})
    assert bkey not in ss.consumed(DAY)                    # left for the next tick, inside the entry window
    lines = [json.loads(s) for s in ss.entries_path(DAY).read_text(encoding="utf-8").splitlines()]
    assert any(r["ticker"] == "BBB" and r["outcome"] == "no_borrow_retry" for r in lines)


def test_only_the_vol_arm_takes_extras_and_the_switch_turns_them_off(monkeypatch, on):
    tracker = _env(monkeypatch, [_pick(arm="vol2", backups=[_backup("BBB", 2)])])
    assert tracker.record_sel_short_trades(run_id="r1") == 1
    assert _opened(tracker) == [("AAA", None, None)]


def test_the_switch_turns_the_extras_off(monkeypatch, on):
    monkeypatch.setattr(settings, "enable_sel_short_vol_pressure_extras", False)
    tracker = _env(monkeypatch, [_pick(backups=[_backup("BBB", 2)])])
    assert tracker.record_sel_short_trades(run_id="r1") == 1
    assert _opened(tracker) == [("AAA", None, None)]


def test_a_name_the_arm_already_shorted_today_is_not_an_extra(monkeypatch, on):
    picks = [_pick("BBB", backups=[]), {**_pick(backups=[_backup("BBB", 2)]), "bar_of_day": 2,
                                        "bar_end": ss.bar_end_et(DAY, 2).isoformat()}]
    tracker = _env(monkeypatch, picks)
    assert tracker.record_sel_short_trades(run_id="r1") == 2
    assert _opened(tracker) == [("AAA", None, None), ("BBB", None, None)]
