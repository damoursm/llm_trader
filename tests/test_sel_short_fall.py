"""FALL — the faller shorts (user directive 2026-10-09: "Yes add the faller short arm to live"; PREREG38's FALLERS_L,
PREREG57): the vol arm's own pick, shorted on the bars where it did NOT rise over the 5 sessions before.

What must hold: the arm runs only with the vol arm and its own switch; it ranks exactly the vol arm's rows with its
own history (`scores_fall`); on one bar it shorts the pick the vol arm skips as "not a riser" and skips the riser
the vol arm shorts; a pick without a valid 5-session base is no faller; the vol arm's freshness, first-pick-of-the-day,
crowding and relative-volume rules apply; it has NO target and NO same-bar fallback; its trades leave on the
volatility exit, the squeeze cover and the time limit; the simulated account funds it; a later bar's record never
changes a bar's decision.
"""
from __future__ import annotations

import json
from datetime import date, datetime, timedelta, timezone
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import pytest

from config.settings import settings
from src.signals import sel_short as ss

ET = ZoneInfo("America/New_York")
D = date(2026, 10, 8)


@pytest.fixture
def on(monkeypatch):
    monkeypatch.setattr(settings, "enable_sel_short", True)
    monkeypatch.setattr(settings, "enable_sel_short_vol", True)
    monkeypatch.setattr(settings, "enable_sel_short_fall", True)
    monkeypatch.setattr(settings, "enable_sel_short_vol2", False)
    monkeypatch.setattr(settings, "enable_sel_short_thin", False)
    monkeypatch.setattr(settings, "enable_sel_short_etf", False)
    monkeypatch.setattr(settings, "enable_sel_short_dtc_filter", True)
    monkeypatch.setattr(settings, "enable_sel_short_vol_rvol_filter", True)
    monkeypatch.setattr(settings, "enable_sel_short_vol_fallback", True)
    monkeypatch.setattr(settings, "sel_short_min_run_rows", 1)
    return settings


def _res(px=(8.0, 20.0, 20.0), pre5=(10.0, 10.0, 10.0), dtc=(0.5, 0.5, 0.5), rvol=(2.0, 2.0, 2.0)):
    """AAA: the most volatile; by default it FELL (8 vs 10 five sessions ago), clean filters; BBB, CCC rose."""
    return pd.DataFrame({"ticker": ["AAA", "BBB", "CCC"], "status": "OK", "score": [0.5, 0.4, 0.3],
                         "vol": [9.0, 8.0, 7.0], "px": list(px), "dv20": 1e7, "pre5": list(pre5),
                         "dtc": list(dtc), "rvol": list(rvol)})


def _stand(res):
    return pd.DataFrame({"n_prior": 0.0, "prior_max": np.nan, "prior_min": np.nan},
                        index=pd.Index(res.ticker, name="ticker"))


def test_fall_runs_with_the_vol_arm_and_its_own_switch(monkeypatch, on):
    assert ss.live_arms() == ["model", "vol", "fall"]
    monkeypatch.setattr(settings, "enable_sel_short_fall", False)
    assert ss.live_arms() == ["model", "vol"]
    monkeypatch.setattr(settings, "enable_sel_short_fall", True)
    monkeypatch.setattr(settings, "enable_sel_short_vol", False)
    assert ss.live_arms() == ["model"]
    assert "fall" in ss.ARMS and "fall" in ss.VOL_RULE_ARMS and ss.FALLER_ARMS == ("fall",)
    assert ss.scores_path(D, "fall").parent.name == "scores_fall"
    assert ss.own_window("fall") == ss.own_window("vol")
    assert ss.volnorm_exit("fall")                                         # it keeps the volatility exit


def test_fall_ranks_exactly_the_vol_arms_rows(monkeypatch, on):
    monkeypatch.setattr(ss, "load_model", lambda: (None, {"tickers": ["AAA", "BBB"]}))
    p = ss.listings_path()
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps({"CCC": {"added": "2026-10-05", "type": "CS"}}), encoding="utf-8")
    res = _res().assign(ca_gap=[False, False, True])
    assert ss.arm_rows(res, "fall")["ticker"].tolist() == ss.arm_rows(res, "vol")["ticker"].tolist() == ["AAA", "BBB"]
    assert ss.arm_rows(res, "fall", ca_gaps=True)["ticker"].tolist() == ["CCC"]   # the corporate-action gap left out


def test_fall_shorts_the_pick_the_vol_arm_skips_and_skips_the_riser(on):
    fell = _res()
    vol = ss.select(fell, D, 2, _stand(fell), [], arm="vol")
    fall = ss.select(fell, D, 2, _stand(fell), [], arm="fall")
    assert (vol["ticker"], vol["decision"]) == ("AAA", "not_a_riser")
    assert (fall["ticker"], fall["decision"]) == ("AAA", "short")
    assert fall["target"] is None                                          # no run-up to give back
    assert fall["deadline"] == ss.bar_end_et(ss.session_after(D, ss.max_hold("fall")), 2).isoformat()
    assert fall["runup_pct"] == pytest.approx(-20.0) and fall["rose"] is False
    rose = _res(px=(20.0, 20.0, 20.0))
    assert ss.select(rose, D, 2, _stand(rose), [], arm="vol")["decision"] == "short"
    assert ss.select(rose, D, 2, _stand(rose), [], arm="fall")["decision"] == "a_riser"
    flat = _res(px=(10.0, 20.0, 20.0))                                     # at the base: did not rise = a faller
    assert ss.select(flat, D, 2, _stand(flat), [], arm="fall")["decision"] == "short"
    nobase = _res(pre5=(np.nan, 10.0, 10.0))
    assert ss.select(nobase, D, 2, _stand(nobase), [], arm="fall")["decision"] == "no_base"
    zero = _res(pre5=(0.0, 10.0, 10.0))
    assert ss.select(zero, D, 2, _stand(zero), [], arm="fall")["decision"] == "no_base"


def test_fall_keeps_the_vol_arms_filters_freshness_and_no_fallback(on):
    crowded = _res(dtc=(3.0, 0.5, 0.5))
    assert ss.select(crowded, D, 2, _stand(crowded), [], arm="fall")["decision"] == "crowded"
    quiet = _res(rvol=(1.0, 2.0, 2.0))
    assert ss.select(quiet, D, 2, _stand(quiet), [], arm="fall")["decision"] == "low_rvol"
    fell = _res()
    rec = ss.select(fell, D, 2, _stand(fell), [], arm="fall")
    assert "backups" not in rec and "fresh_pool" not in rec                # no same-bar fallback
    stale = pd.DataFrame({"n_prior": 20.0, "prior_max": 12.0, "prior_min": 1.0}, index=pd.Index(fell.ticker, name="ticker"))
    assert ss.select(fell, D, 2, stale, [], arm="fall")["decision"] == "not_fresh"
    earlier = [{"arm": "fall", "ticker": "AAA", "fresh": True, "bar_of_day": 1}]
    assert ss.select(fell, D, 2, _stand(fell), earlier, arm="fall")["decision"] == "not_first_today"
    # another arm's earlier pick of the name does not count against the faller arm
    other = [{"arm": "vol", "ticker": "AAA", "fresh": True, "bar_of_day": 1}]
    assert ss.select(fell, D, 2, _stand(fell), other, arm="fall")["decision"] == "short"


def test_decide_keeps_the_fall_history_a_copy_of_the_vol_arms(monkeypatch, on):
    monkeypatch.setattr(ss, "load_model", lambda: (None, {"tickers": ["AAA", "BBB", "CCC"]}))
    run = ss.decide(_res(), D, 2)
    assert run["arms"]["vol"]["decision"] == "not_a_riser" and run["arms"]["fall"]["decision"] == "short"
    h2, h1 = pd.read_pickle(ss.scores_path(D, "fall")), pd.read_pickle(ss.scores_path(D, "vol"))
    assert h2.sort_values("ticker").reset_index(drop=True).equals(h1.sort_values("ticker").reset_index(drop=True))
    assert {r["arm"] for r in ss.read_picks(D)} == {"model", "vol", "fall"}
    # the faller pick waits for its turn (PREREG76, tests/test_sel_short_turn.py); switched off, it is handed over
    # at once, as before
    monkeypatch.setattr(ss, "_turn_bars", lambda rec, now: None)
    assert ss.pending_entries(ss.bar_end_et(D, 2) + timedelta(minutes=5)) == []
    monkeypatch.setattr(settings, "enable_sel_short_fall_turn", False)
    assert [r["arm"] for r in ss.pending_entries(ss.bar_end_et(D, 2) + timedelta(minutes=5))] == ["fall"]


def _trade(entry=10.0, mark=8.0, atr0=4.0):
    now = datetime(2026, 10, 9, 15, 0, tzinfo=timezone.utc)
    return {"ticker": "AAA", "entry_mechanism": "sel_short", "sel_arm": "fall", "status": "OPEN", "action": "SELL",
            "entry_price": entry, "entry_date": "2026-10-08", "current_price": mark,
            "current_price_datetime": now.isoformat(), "sel_target_price": None, "sel_atr_pct": atr0,
            "sel_deadline": (now + timedelta(days=10)).isoformat()}, now


def test_fall_trades_leave_on_the_volatility_exit_the_cover_and_the_time_limit(monkeypatch, on):
    from src.data import intraday_store
    from src.performance import tracker
    monkeypatch.setattr(settings, "enable_sel_short_volnorm_exit", True)
    monkeypatch.setattr(settings, "enable_sel_short_squeeze_cover", True)
    monkeypatch.setattr(settings, "sel_short_fall_volnorm_ratio", 0.5)      # the mechanism at 0.5 (0.7 from 2026-10-20)
    monkeypatch.setattr(intraday_store, "split_factor_between", lambda t, a, b: 1.0)
    t, now = _trade()
    assert tracker._sel_short_exit_reason(dict(t, current_price=0.5), now) is None   # no target to reach
    assert tracker._sel_short_exit_reason(t, now, atr_now=1.9) == "sel_volnorm"       # in profit, ATR% halved
    assert tracker._sel_short_exit_reason(t, now, atr_now=2.5) is None
    assert tracker._sel_short_exit_reason(dict(t, current_price=12.0), now, atr_now=1.9) is None   # not in profit
    assert tracker._sel_short_exit_reason(dict(t, current_price=60.0), now) == "sel_cover"
    late = dict(t, sel_deadline=(now - timedelta(minutes=1)).isoformat())
    assert tracker._sel_short_exit_reason(late, now) == "sel_time"


def _pick(tk, arm="fall", day=date(2026, 9, 28), bar=1, **kw):
    r = {"day": day.isoformat(), "bar_of_day": bar, "bar_end": ss.bar_end_et(day, bar).isoformat(),
         "ticker": tk, "decision": "short", "px": 10.0, "pre5": 12.0, "target": None, "arm": arm,
         "deadline": ss.bar_end_et(ss.session_after(day, 15), bar).isoformat(), "score": 6.5,
         "runup_pct": -16.7, "days_to_cover": 0.5, "rvol": 2.0, "atr_pct": 6.5, "dv20": 5e7}
    r.update(kw)
    return r


def _entry_env(monkeypatch, picks, lendable=lambda t: True):
    from src.data import ibkr_borrow
    from src.performance import tracker
    monkeypatch.setattr(ss, "pending_entries", lambda now=None: list(picks))
    monkeypatch.setattr(tracker, "_fetch_price", lambda t: 9.8)
    monkeypatch.setattr(tracker, "_reference_close", lambda t: None)
    monkeypatch.setattr(tracker, "_execution_iso", lambda: "2026-09-28T14:40:00+00:00")
    monkeypatch.setattr(settings, "enable_sel_short_account_sizing", True)
    monkeypatch.setattr(settings, "enable_sel_short_ibkr_whatif", False)
    monkeypatch.setattr(settings, "enable_sel_short_borrow_retry", True)
    monkeypatch.setattr(settings, "enable_sel_short_volnorm_exit", True)
    stamp = datetime(2026, 9, 28, 9, 0, tzinfo=ET)

    def block(ticker, price, base=None, now=None, max_fee_pct="default"):
        if lendable(ticker):
            return None, ibkr_borrow.Borrow(ticker, 12.0, -5.0, 900_000, stamp)
        return "no_borrow", None
    monkeypatch.setattr(ibkr_borrow, "short_block", block)
    return tracker


def test_the_entry_step_opens_and_funds_a_faller_short(monkeypatch, on):
    tracker = _entry_env(monkeypatch, [_pick("AAA")])
    assert tracker.record_sel_short_trades(run_id="r1") == 1
    t = next(x for x in tracker._load_trades() if x.get("entry_mechanism") == "sel_short")
    assert t["sel_arm"] == "fall" and t["sel_account_shares"] > 0          # the simulated account funds it
    assert t["sel_target_price"] is None and t["sel_vol_score"] == 6.5 and t["sel_atr_pct"] == 6.5
    r = t["rationale"]
    assert "FELL" in r and "squeeze cover" in r and "N/A" not in r and "given back" not in r
    assert f"ATR% falls to {settings.sel_short_fall_volnorm_ratio:.0%} of 6.50%" in r   # the volatility exit it leaves on


def test_a_faller_pick_ibkr_cannot_lend_takes_no_fallback(monkeypatch, on):
    backups = [{"ticker": "BBB", "rank": 2, "score": 6.0, "px": 10.0, "pre5": 12.0, "runup_pct": -16.7,
                "target": None, "deadline": _pick("AAA")["deadline"], "days_to_cover": 0.5, "rvol": 2.0,
                "atr_pct": 6.0, "dv20": 5e7}]
    tracker = _entry_env(monkeypatch, [_pick("AAA", backups=backups)], lendable=lambda t: t != "AAA")
    assert tracker.record_sel_short_trades(run_id="r1") == 0
    assert not [x for x in tracker._load_trades() if x.get("entry_mechanism") == "sel_short"]


def test_the_vol_listings_history_seed_reaches_the_faller_history(monkeypatch, on):
    """A stock the prepare adds to the vol arm gets its history in every copy of the vol history."""
    from src.data import deep as deep_pkg
    from src.data import intraday_store
    monkeypatch.setattr(intraday_store, "extend_deep_30m", lambda *a, **k: {})
    monkeypatch.setattr(intraday_store, "load_deep_30m", lambda tk: pd.DataFrame({"Close": [1.0]}))
    monkeypatch.setattr(deep_pkg, "deep_universe", lambda refresh=False: None, raising=False)
    seen = {}
    monkeypatch.setattr(ss, "backfill_days", lambda days, tickers=None, arms=(), **k: seen.setdefault("arms", arms) and {})
    out = ss.add_listings(D, [{"ticker": "NEW", "type": "CS"}])
    assert out["added"] == ["NEW"] and tuple(seen["arms"]) == ("vol", "fall")
    monkeypatch.setattr(settings, "enable_sel_short_vol2", True)
    seen.clear()
    ss.add_listings(D, [{"ticker": "NEW2", "type": "CS"}])
    assert tuple(seen["arms"]) == ("vol", "vol2", "fall")


def test_a_later_bars_record_never_changes_a_faller_decision(monkeypatch, on):
    """The faller decision at a bar reads only the EARLIER bars' records: a later bar's record of the same
    name journaled first (a rerun out of order) must not make it "not first today"; fed to the rule (the
    injected leak), it does — so the guard can fail."""
    monkeypatch.setattr(ss, "load_model", lambda: (None, {"tickers": ["AAA", "BBB", "CCC"]}))
    res = _res()
    assert ss.decide(res, D, 3)["arms"]["fall"]["decision"] == "short"
    later = {"day": D.isoformat(), "bar_of_day": 5, "arm": "fall", "ticker": "AAA", "fresh": True, "decision": "short"}
    ss._journal(later)
    assert ss.decide(res, D, 3)["arms"]["fall"]["decision"] == "short"
    assert ss.select(res, D, 3, _stand(res), [later], arm="fall")["decision"] == "not_first_today"


def test_only_the_live_accounts_arms_trade(monkeypatch, on):
    """User directive 2026-10-09: only the best combination trades on the $10,000 account — the model's and
    vol2's picks are journaled and settled `shadow_arm`, never opened; the vol, ETF and fall arms trade."""
    monkeypatch.setattr(settings, "sel_short_traded_arms", "vol,etf,fall")
    assert ss.traded_arms() == frozenset({"vol", "etf", "fall"})
    picks = [_pick("MMM", arm="model", target=8.0, runup_pct=25.0, pre5=8.0),
             _pick("VVV", arm="vol2", target=8.0, runup_pct=25.0, pre5=8.0), _pick("FFF")]
    tracker = _entry_env(monkeypatch, picks)
    settled = []
    monkeypatch.setattr(ss, "mark_consumed", lambda d, key, outcome: settled.append((key.split("|")[2], outcome)))
    monkeypatch.setattr(ss, "journal_entry", lambda rec, outcome, **kw: None)
    assert tracker.record_sel_short_trades(run_id="r1") == 1
    assert [x["ticker"] for x in tracker._load_trades() if x.get("entry_mechanism") == "sel_short"] == ["FFF"]
    assert ("MMM", "shadow_arm") in settled and ("VVV", "shadow_arm") in settled and ("FFF", "opened") in settled
    monkeypatch.setattr(settings, "sel_short_traded_arms", "")
    assert ss.traded_arms() is None                                        # empty = every arm trades
