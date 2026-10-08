"""VOL2 — the vol rule without the crowding filter, the relative-volume filter and the volatility exit (user directive
2026-10-07: "Deploy it as another version that will also make real trades in the paper account"; PREREG37), and one
trade per arm when several arms pick the same name ("if multiple arms pick the same stock, still order for each one
of them").

What must hold: vol2 runs only with the vol arm and its own switch; it ranks exactly the vol arm's rows (the model's
names + the vol listings, never a thin row) with its own history (`scores_vol2`); a pick the crowding or the
relative-volume filter stops for the vol arm is a "short" for vol2, and its same-bar backups skip neither filter; its
target gives back the whole run-up; its trades never take the volatility exit (the target, the time limit and the
6x cover still apply); the simulated account funds it; a name vol and vol2 both pick opens two trades with their own
order references; the fallback's one-trade-per-name-per-day rule counts the arm's OWN trades only; the broker orders
each arm's trade on one name in full.
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


@pytest.fixture
def on(monkeypatch):
    monkeypatch.setattr(settings, "enable_sel_short", True)
    monkeypatch.setattr(settings, "enable_sel_short_vol", True)
    monkeypatch.setattr(settings, "enable_sel_short_vol2", True)
    monkeypatch.setattr(settings, "enable_sel_short_thin", False)
    monkeypatch.setattr(settings, "enable_sel_short_etf", False)
    monkeypatch.setattr(settings, "enable_sel_short_dtc_filter", True)
    monkeypatch.setattr(settings, "enable_sel_short_vol_rvol_filter", True)
    monkeypatch.setattr(settings, "enable_sel_short_vol_fallback", True)
    monkeypatch.setattr(settings, "sel_short_min_run_rows", 1)
    return settings


def _res():
    """AAA: the most volatile, CROWDED (days to cover 3) and LOW relative volume; BBB: crowded; CCC: clean."""
    return pd.DataFrame({"ticker": ["AAA", "BBB", "CCC", "TH1"],
                         "status": ["OK", "OK", "OK", "VOL_ONLY"],
                         "score": [0.5, 0.4, 0.3, np.nan],
                         "vol": [9.0, 8.0, 7.0, 12.0],
                         "px": [20.0, 20.0, 20.0, 12.0],
                         "dv20": [1e7, 1e7, 1e7, 2e6],
                         "pre5": [10.0, 10.0, 10.0, 8.0],
                         "dtc": [3.0, 2.5, 0.5, 0.5],
                         "rvol": [1.0, 2.0, 2.0, 2.0],
                         "thin": [False, False, False, True]})


def _stand(res):
    return pd.DataFrame({"n_prior": 0.0, "prior_max": np.nan, "prior_min": np.nan},
                        index=pd.Index(res.ticker, name="ticker"))


def test_vol2_runs_with_the_vol_arm_and_its_own_switch(monkeypatch, on):
    assert ss.live_arms() == ["model", "vol", "vol2"]
    monkeypatch.setattr(settings, "enable_sel_short_vol2", False)
    assert ss.live_arms() == ["model", "vol"]
    monkeypatch.setattr(settings, "enable_sel_short_vol2", True)
    monkeypatch.setattr(settings, "enable_sel_short_vol", False)
    assert ss.live_arms() == ["model"]
    assert "vol2" in ss.ARMS and "vol2" in ss.VOL_RULE_ARMS
    assert ss.scores_path(date(2026, 10, 8), "vol2").parent.name == "scores_vol2"
    assert ss.own_window("vol2") == ss.own_window("vol")


def test_vol2_ranks_exactly_the_vol_arms_rows(monkeypatch, on):
    monkeypatch.setattr(ss, "load_model", lambda: (None, {"tickers": ["AAA", "BBB"]}))
    p = ss.listings_path()
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps({"CCC": {"added": "2026-10-05", "type": "CS"}}), encoding="utf-8")
    res = _res()
    assert ss.arm_rows(res, "vol2")["ticker"].tolist() == ss.arm_rows(res, "vol")["ticker"].tolist() \
        == ["AAA", "BBB", "CCC"]                                            # the listing in, the thin row out


def test_vol2_takes_what_the_crowding_and_volume_filters_stop(monkeypatch, on):
    d = date(2026, 10, 8)
    res = _res()[lambda x: ~x.thin]
    vol = ss.select(res, d, 2, _stand(res), [], arm="vol")
    vol2 = ss.select(res, d, 2, _stand(res), [], arm="vol2")
    assert (vol["ticker"], vol["decision"]) == ("AAA", "crowded")
    assert (vol2["ticker"], vol2["decision"]) == ("AAA", "short")
    # the whole run-up given back, the vol arm's target and deadline
    assert ss.give_back("vol2") == ss.give_back("vol") == float(settings.sel_short_vol_give_back)
    assert vol2["target"] == pytest.approx(vol["target"]) == pytest.approx(20.0 - ss.give_back("vol") * 10.0)
    assert vol2["deadline"] == vol["deadline"]
    # its same-bar backups pass no crowding / volume filter; the vol arm's do
    assert [b["ticker"] for b in vol2["backups"]] == ["BBB", "CCC"]
    assert [b["ticker"] for b in vol["backups"]] == ["CCC"]
    # the relative-volume filter alone (an uncrowded, low-volume pick): low_rvol for vol, short for vol2
    clean = res.assign(dtc=0.5)
    assert ss.select(clean, d, 2, _stand(clean), [], arm="vol")["decision"] == "low_rvol"
    assert ss.select(clean, d, 2, _stand(clean), [], arm="vol2")["decision"] == "short"


def test_decide_keeps_the_vol2_history_its_own(monkeypatch, on):
    monkeypatch.setattr(ss, "load_model", lambda: (None, {"tickers": ["AAA", "BBB", "CCC"]}))
    d = date(2026, 10, 8)
    run = ss.decide(_res(), d, 2)
    assert run["arms"]["vol"]["decision"] == "crowded" and run["arms"]["vol2"]["decision"] == "short"
    assert run["arms"]["vol2"]["ticker"] == run["arms"]["vol"]["ticker"] == "AAA"
    h2, h1 = pd.read_pickle(ss.scores_path(d, "vol2")), pd.read_pickle(ss.scores_path(d, "vol"))
    assert h2.sort_values("ticker").reset_index(drop=True).equals(h1.sort_values("ticker").reset_index(drop=True))
    assert {r["arm"] for r in ss.read_picks(d)} == {"model", "vol", "vol2"}
    assert [r["arm"] for r in ss.pending_entries(ss.bar_end_et(d, 2) + timedelta(minutes=5))] == ["vol2"]


def _trade(arm, entry=10.0, mark=8.0, atr0=4.0):
    now = datetime(2026, 10, 9, 15, 0, tzinfo=timezone.utc)
    return {"ticker": "AAA", "entry_mechanism": "sel_short", "sel_arm": arm, "status": "OPEN", "action": "SELL",
            "entry_price": entry, "entry_date": "2026-10-08", "current_price": mark,
            "current_price_datetime": now.isoformat(), "sel_target_price": 5.0, "sel_atr_pct": atr0,
            "sel_deadline": (now + timedelta(days=10)).isoformat()}, now


def test_vol2_trades_never_take_the_volatility_exit(monkeypatch, on):
    from src.performance import tracker
    monkeypatch.setattr(settings, "enable_sel_short_volnorm_exit", True)
    monkeypatch.setattr(settings, "enable_sel_short_squeeze_cover", True)
    from src.data import intraday_store
    monkeypatch.setattr(intraday_store, "split_factor_between", lambda t, a, b: 1.0)
    t_vol, now = _trade("vol")
    t_vol2, _ = _trade("vol2")
    # in profit, ATR% halved: the vol arm's trade covers, vol2's holds
    assert tracker._sel_short_exit_reason(t_vol, now, atr_now=1.9) == "sel_volnorm"
    assert tracker._sel_short_exit_reason(t_vol2, now, atr_now=1.9) is None
    # the target, the squeeze cover and the time limit still apply to vol2
    assert tracker._sel_short_exit_reason(dict(t_vol2, current_price=4.9), now, atr_now=1.9) == "sel_target"
    assert tracker._sel_short_exit_reason(dict(t_vol2, current_price=60.0), now) == "sel_cover"
    late = dict(t_vol2, sel_deadline=(now - timedelta(minutes=1)).isoformat())
    assert tracker._sel_short_exit_reason(late, now) == "sel_time"
    # the live ATR% is never fetched for a vol2 trade
    asked = []
    monkeypatch.setattr(ss, "live_atr", lambda names, now_: asked.extend(names) or {})
    tracker._sel_short_live_atr([t_vol, t_vol2], now)
    assert asked == ["AAA"] and len(asked) == 1


def _pick(tk, arm, day=date(2026, 9, 28), bar=1, **kw):
    r = {"day": day.isoformat(), "bar_of_day": bar, "bar_end": ss.bar_end_et(day, bar).isoformat(),
         "ticker": tk, "decision": "short", "px": 10.0, "pre5": 8.0, "target": 8.0, "arm": arm,
         "deadline": ss.bar_end_et(ss.session_after(day, 15), bar).isoformat(), "score": 4.2,
         "runup_pct": 25.0, "days_to_cover": 3.0, "rvol": 1.0, "atr_pct": 4.2, "dv20": 5e7}
    r.update(kw)
    return r


def _entry_env(monkeypatch, picks, lendable=lambda t: True):
    from src.data import ibkr_borrow
    from src.performance import tracker
    monkeypatch.setattr(ss, "pending_entries", lambda now=None: list(picks))
    monkeypatch.setattr(tracker, "_fetch_price", lambda t: 10.2)
    monkeypatch.setattr(tracker, "_reference_close", lambda t: None)
    monkeypatch.setattr(tracker, "_execution_iso", lambda: "2026-09-28T14:40:00+00:00")
    monkeypatch.setattr(settings, "enable_sel_short_account_sizing", True)
    monkeypatch.setattr(settings, "enable_sel_short_ibkr_whatif", False)
    monkeypatch.setattr(settings, "enable_sel_short_borrow_retry", True)
    stamp = datetime(2026, 9, 28, 9, 0, tzinfo=ET)

    def block(ticker, price, base=None, now=None, max_fee_pct="default"):
        if lendable(ticker):
            return None, ibkr_borrow.Borrow(ticker, 12.0, -5.0, 900_000, stamp)
        return "no_borrow", None
    monkeypatch.setattr(ibkr_borrow, "short_block", block)
    return tracker


def test_a_name_vol_and_vol2_both_pick_opens_one_trade_per_arm(monkeypatch, on):
    tracker = _entry_env(monkeypatch, [_pick("AAA", "vol"), _pick("AAA", "vol2")])
    assert tracker.record_sel_short_trades(run_id="r1") == 2
    t = [x for x in tracker._load_trades() if x.get("entry_mechanism") == "sel_short"]
    assert sorted(x["sel_arm"] for x in t) == ["vol", "vol2"]
    assert len({x["recommendation_id"] for x in t}) == 2                  # two IBKR order references
    assert all(x["sel_account_shares"] > 0 for x in t)                     # the account funds both
    v2 = next(x for x in t if x["sel_arm"] == "vol2")
    assert v2["sel_vol_score"] == 4.2 and v2["sel_atr_pct"] == 4.2 and "VOL2 rule" in v2["rationale"]
    assert "volatility" not in v2["rationale"].split("—")[-1]              # no volatility exit in its exit line


def test_the_fallback_counts_only_the_arms_own_trades_today(monkeypatch, on):
    """vol2 shorted BBB at bar 1; at bar 2 IBKR cannot lend the vol arm's pick AAA and its backup is BBB —
    another arm's trade on BBB never stops the vol arm's own."""
    backups = [{"ticker": "BBB", "rank": 2, "score": 4.0, "px": 10.0, "pre5": 8.0, "runup_pct": 25.0,
                "target": 8.0, "deadline": ss.bar_end_et(ss.session_after(date(2026, 9, 28), 15), 2).isoformat(),
                "days_to_cover": 0.5, "rvol": 2.0, "atr_pct": 4.0, "dv20": 5e7}]
    picks = [_pick("BBB", "vol2", bar=1), _pick("AAA", "vol", bar=2, days_to_cover=0.5, rvol=2.0, backups=backups)]
    tracker = _entry_env(monkeypatch, picks, lendable=lambda t: t != "AAA")
    assert tracker.record_sel_short_trades(run_id="r1") == 2
    t = sorted((x["ticker"], x["sel_arm"]) for x in tracker._load_trades() if x.get("entry_mechanism") == "sel_short")
    assert t == [("BBB", "vol"), ("BBB", "vol2")]
    fb = next(x for x in tracker._load_trades() if x.get("sel_arm") == "vol")
    assert fb["sel_fallback_of"] == "AAA" and fb["sel_stack_n"] == 2


def test_the_account_funds_vol2():
    from src.performance import sim_account as sa
    assert "vol2" in sa.FUNDED_ARMS
