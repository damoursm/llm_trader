"""The live record against the simulator's assumptions (2026-10-12, `src/backtest/live_calibration.py`)."""
from __future__ import annotations

import json
from datetime import date

import pytest

from config.settings import settings
from src.backtest import live_calibration as LC


@pytest.fixture(autouse=True)
def _dir(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "live_calibration_dir", str(tmp_path / "live_cal"))


def test_wilson_range_brackets_the_share():
    lo, hi = LC.wilson(7, 100)
    assert lo < 0.07 < hi and 0 <= lo and hi <= 1
    assert LC.wilson(0, 0) == (None, None)
    lo, hi = LC.wilson(0, 10)
    assert lo == 0.0 and hi > 0.2


def test_the_medians_range_is_its_order_statistics():
    assert LC.median_range([1, 2, 3, 4, 5]) == (None, None)                   # n = 5: no 95% range exists
    assert LC.median_range([6, 1, 5, 2, 4, 3]) == (1.0, 6.0)                  # n = 6: the extremes (96.9%)
    assert LC.median_range(list(range(1, 21))) == (6.0, 15.0)                 # n = 20: x(6) .. x(15)


def _e(key, arm, outcome, at, ticker="AAA", checked=True, margin=None):
    return {"key": key, "arm": arm, "outcome": outcome, "at": at, "ticker": ticker, "borrow_checked": checked,
            "ibkr_margin": margin}


def test_entry_facts_take_one_decision_per_pick(monkeypatch):
    from src.signals import sel_short
    monkeypatch.setattr(sel_short, "etf_names", lambda names, model_names=None: [n for n in names if n == "ETFX"])
    E = [_e("k1", "vol", "no_borrow_retry", "2026-10-12T14:01:00"),           # first check: unlendable
         _e("k1", "vol", "no_borrow_retry", "2026-10-12T14:01:40"),           # re-journaled each pass
         _e("k1", "vol", "opened", "2026-10-12T14:31:00",                     # lendable later: not the first check
            margin={"init_rate": 3.5, "maint_rate": 2.5}),
         _e("k2", "fall", "opened", "2026-10-12T15:01:00", ticker="BBB", margin={"init_rate": 1.0, "maint_rate": 0.9}),
         _e("k3", "vol", "ibkr_refused", "2026-10-12T15:31:00", ticker="CCC"),
         _e("k4", "etf", "opened", "2026-10-12T16:01:00", ticker="ETFX", margin={"init_rate": 0.8, "maint_rate": 0.7}),
         _e("k5", "model", "shadow_arm", "2026-10-12T16:01:00", checked=False),   # a shadow arm: not counted
         _e("k6", "vol", "target_reached", "2026-10-12T16:31:00", checked=False),  # never reached the borrow check
         _e("k7", "vol", "opened", "2026-10-12T16:35:00", ticker="DDD",            # the what-if timed out: neither
            margin={"error": "timeout"})]
    f = LC.entry_facts(E, ["vol", "etf", "fall"])
    lf = f["lendable_first_check"]
    assert (lf["k"], lf["n"]) == (4, 5)                       # k1 no; k2, k3 (refused AFTER the borrow), k4, k7 yes
    assert f["lendable_by_arm"]["vol"]["n"] == 3 and f["lendable_by_arm"]["vol"]["k"] == 2
    assert (f["ibkr_refused_stock"]["k"], f["ibkr_refused_stock"]["n"]) == (1, 3)   # CCC of CCC, AAA, BBB
    assert f["house_init"]["n"] == 2 and f["house_init"]["median"] == pytest.approx(2.25)
    assert f["house_init_above_2_86"]["k"] == 1
    assert f["picks_settled"] == 6                            # the live arms' distinct picks
    assert f["outcomes"]["opened"] == 4 and "no_borrow_retry" not in f["outcomes"]   # each pick's LAST outcome


def _short(**kw):
    t = {"entry_mechanism": "sel_short", "sel_arm": "vol", "action": "SELL", "status": "OPEN",
         "entry_date": "2026-10-12", "entry_datetime": "2026-10-12T14:01:30+00:00",
         "sel_bar_end": "2026-10-12T14:00:00+00:00", "sel_bar_close": 10.0, "entry_price": 10.05}
    t.update(kw)
    return t


def test_trade_facts_measure_each_short_from_its_decision_bar():
    T = [_short(broker_status="Filled", broker_order_id=1, broker_fill_qty=100, broker_fill_price=10.04),
         # a faller entered on its TURN two sessions later: measured from the turning bar, not the pick bar
         _short(sel_arm="fall", entry_datetime="2026-10-14T15:02:00+00:00", entry_date="2026-10-14",
                sel_turn_bar_end="2026-10-14T11:00:00-04:00", sel_turn_close=8.0, entry_price=7.96),
         {"entry_mechanism": "dip_long", "action": "BUY", "status": "CLOSED", "entry_date": "2026-10-12",
          "exit_date": "2026-10-13", "exit_reason": "dip_sma5", "exit_session": "rth", "entry_price": 100.0,
          "broker_status": "Filled", "broker_order_id": 2, "broker_fill_qty": 5, "broker_fill_price": 100.1},
         {"entry_mechanism": "rank", "status": "OPEN", "entry_date": "2026-10-12", "entry_price": 5.0}]   # legacy
    f = LC.trade_facts(T, date(2026, 10, 12), date(2026, 10, 14))
    assert f["opened"] == 3 and f["opened_by_book"] == {"sel_short": 2, "dip_long": 1}
    assert f["short_entry_delay_min"]["median"] == pytest.approx(1.5)
    assert f["short_entry_drift_bp"]["mean"] == pytest.approx(50.0)          # entered 0.5% above the pick close
    assert f["turn_entry_delay_min"]["median"] == pytest.approx(2.0)          # 15:02 UTC = 11:02 ET
    assert f["turn_entry_drift_bp"]["mean"] == pytest.approx(-50.0)
    assert f["broker_entry_legs"]["k"] == 2 and f["broker_entry_slippage_bp"]["n"] == 2
    # + = worse than the ledger's price: the short sold 9.95 bp lower, the long bought 10 bp higher
    assert f["broker_entry_slippage_bp"]["median"] == pytest.approx((9.950248756 + 10.0) / 2, rel=1e-3)
    assert f["closed"] == 1 and f["exits"] == {"dip_long:dip_sma5": 1}


def test_fee_facts_test_ibkrs_charges_against_the_formula_not_the_ledgers_copy(monkeypatch):
    from src.db import repo
    from src.performance import borrow_fees as BF
    rows = [{"ticker": "AAA", "value_date": "2026-10-12", "quantity": -100, "fee": -2.0, "fee_rate": 50.0},
            {"ticker": "AAA", "value_date": "2026-10-13", "quantity": -100, "fee": -3.0, "fee_rate": 50.0},
            {"ticker": "AAA", "value_date": "2026-11-30", "quantity": -100, "fee": -9.0, "fee_rate": 50.0}]  # outside
    monkeypatch.setattr(repo, "load_broker_borrow_fees", lambda: rows)
    monkeypatch.setattr(BF, "day_rate", lambda tk, d, now, trade=None: (36.0, "ibkr_api"))
    monkeypatch.setattr(BF, "prior_close", lambda tk, d, closes=None: 19.5)       # collateral roundup(1.02 x) = 20
    T = [{"entry_mechanism": "sel_short", "ticker": "AAA", "sel_account_shares": 100,
          # the ledger's stored schedule already holds IBKR's charges — never the comparison
          "borrow_days": [["2026-10-12", "ibkr_charged", 50.0, 20.0, 0.02],
                          ["2026-10-13", "ibkr_charged", 50.0, 20.0, 0.03]]}]
    f = LC.fee_facts(T, date(2026, 10, 12), date(2026, 10, 13))
    formula = 100 * BF.collateral_price(19.5) * 36.0 / 100.0 / 360.0
    assert f["charged_rows"] == 2 and f["charged_usd"] == pytest.approx(5.0)
    assert f["ratio"]["median"] == pytest.approx((2.0 / formula + 3.0 / formula) / 2)
    assert f["qty_ratio"]["median"] == pytest.approx(1.0)


def _six(x):
    return [x] * 6


def test_notes_name_every_gap_beyond_its_sampling_error():
    assume = {"refusal_rate": 0.07, "house_init_median": 2.86}
    rec = {"entries": {"ibkr_refused_stock": LC._share(5, 10), "house_init": LC._summary(_six(8.0))},
           "trades": {"short_entry_delay_min": LC._summary(_six(7.0)), "short_entry_drift_bp": LC._summary(_six(-80.0)),
                      "turn_entry_delay_min": LC._summary(_six(1.0)), "broker_entry_legs": LC._share(1, 10)},
           "fees": {"ratio": LC._summary(_six(2.0)), "qty_ratio": LC._summary(_six(1.0))}}
    notes = LC.compare(rec, assume)
    assert len(notes) == 6
    assert any("refused 5/10" in n for n in notes) and any("2.00x the borrow formula" in n for n in notes)
    assert not any("turn shorts entered" in n for n in notes)               # 1 min: inside the assumption
    small = {"entries": {"ibkr_refused_stock": LC._share(2, 3)},
             "trades": {"short_entry_delay_min": LC._summary([9.0] * 5)}, "fees": {}}
    assert LC.compare(small, assume) == []                                     # too few to judge
    ok = {"entries": {"ibkr_refused_stock": LC._share(1, 12)},
          "trades": {"broker_entry_legs": LC._share(3, 5),                    # 60%, but its range reaches 80%
                     "short_entry_drift_bp": LC._summary([-200, -150, -90, 10, 20, 30])}, "fees": {}}
    assert LC.compare(ok, assume) == []                                        # within their sampling error


def _quiet_sections(monkeypatch):
    from src.signals import sel_short as ss
    monkeypatch.setattr(ss, "traded_arms", lambda: frozenset({"vol"}))
    monkeypatch.setattr(LC, "entry_facts", lambda e, a: {})
    monkeypatch.setattr(LC, "trade_facts", lambda t, lo, hi: {})
    monkeypatch.setattr(LC, "fee_facts", lambda t, lo, hi: {})
    monkeypatch.setattr(LC, "dip_order_facts", lambda ds: {})


def test_the_last_sessions_gap_is_noted_when_the_whole_record_hides_it(monkeypatch):
    """2026-10-19: the record since 10-09 holds 7 sessions, the window the last 5."""
    _quiet_sections(monkeypatch)
    monkeypatch.setattr(LC, "_notes", lambda rec, a: [("fills", "fills gap")] if rec["sessions"] == 5 else [])
    assert LC.build(date(2026, 10, 19), [])["notes"] == ["last 5 sessions: fills gap"]
    monkeypatch.setattr(LC, "_notes", lambda rec, a: [("fills", "fills gap")])
    assert LC.build(date(2026, 10, 19), [])["notes"] == ["fills gap"]                # once, the whole record's


def test_health_reads_the_latest_record_as_notes_never_problems():
    LC.out_dir().mkdir(parents=True, exist_ok=True)
    rec = {"day": "2026-10-12", "live_since": "2026-10-09",
           "since": {"sessions": 2, "entries": {"lendable_first_check": LC._share(3, 4),
                                                "ibkr_refused_stock": LC._share(1, 3)},
                     "trades": {"short_entry_delay_min": {"n": 2, "median": 1.2}, "broker_entry_legs": LC._share(2, 2)}},
           "notes": ["IBKR refused 1/3 stock picks"]}
    (LC.out_dir() / "2026-10-12.json").write_text(json.dumps(rec), encoding="utf-8")
    (LC.out_dir() / "2026-10-13.123.tmp").write_text("{", encoding="utf-8")   # a write in progress: never read
    h = LC.health()
    assert h["problems"] == []
    assert "lendable at entry 3/4" in h["notes"][0] and "entry 1.2 min after the pick bar" in h["notes"][0]
    assert h["notes"][1] == "live vs simulator: IBKR refused 1/3 stock picks"


def test_assumptions_are_the_simulators():
    from src.backtest import mimic as M
    a = LC.assumptions()
    assert "error" not in a
    assert a["refusal_rate"] == M.REFUSAL_RATE and a["entry_delay_min"] == 0.0
    assert a["house_names"] == len(M.house_mix())


def test_the_eod_chain_runs_it_last():
    from src.scheduler import eod
    assert eod.STEPS[-1] == "live_calibration" and "live_calibration" in eod.STEP_FNS
    assert type(settings).model_fields["enable_eod_live_calibration"].default is True


def test_build_write_health_end_to_end(monkeypatch):
    """The record as the EOD step writes it and the banner reads it: the ISO day survives the sections."""
    from src.db import repo
    from src.signals import sel_short as ss
    monkeypatch.setattr(repo, "load_broker_borrow_fees", lambda: [])
    monkeypatch.setattr(ss, "etf_names", lambda names, model_names=None: [])
    d = date(2026, 10, 12)
    p = ss.entries_path(d)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(_e("k1", "vol", "opened", "2026-10-12T14:01:00",
                               margin={"init_rate": 2.0, "maint_rate": 1.5})) + "\n", encoding="utf-8")
    T = [_short(ticker="AAA", entry_price=10.0)]
    path = LC.write(d, T)
    rec = json.loads(path.read_text(encoding="utf-8"))
    assert path.name == "2026-10-12.json" and rec["day"] == "2026-10-12"
    assert not list(path.parent.glob("*.tmp"))
    assert rec["session"]["sessions"] == 1 and rec["since"]["sessions"] == 2          # 10-09 and 10-12
    assert rec["since"]["entries"]["lendable_first_check"]["n"] == 1
    assert rec["since"]["trades"]["opened"] == 1 and rec["window"]["trades"]["opened"] == 1
    h = LC.health()
    assert h["problems"] == []
    assert h["notes"][0].startswith("live vs simulator since 2026-10-09 (2 sessions, as of 2026-10-12)")
    assert "{" not in h["notes"][0]


# ── each arm's reliability: the live trades against the simulator's reference (item 6, 2026-10-12) ──────────────
# The trade dicts carry the ledger's own field names: the entries stamp `entry_mechanism`, `sel_arm`, `action`,
# `status`, `entry_date`, `entry_datetime`, `entry_price` (tracker.record_sel_short_trades / _dip_trade /
# record_crash_long_trades); a close stamps `status="CLOSED"`, `exit_date`, `exit_datetime`, `exit_price`,
# `return_pct`, `exit_reason` (tracker._close_trade_now and the monitors' close blocks).

def _closed(arm=None, mech="sel_short", action="SELL", entry="2026-10-12T14:00:00+00:00",
            exit_="2026-10-14T14:00:00+00:00", entry_px=10.0, exit_px=9.0, ret=9.0, **kw):
    t = {"ticker": kw.pop("ticker", "AAA"), "entry_mechanism": mech, "action": action, "status": "CLOSED",
         "entry_date": entry[:10], "entry_datetime": entry, "exit_date": (exit_ or "")[:10] or None,
         "exit_datetime": exit_, "entry_price": entry_px, "exit_price": exit_px, "return_pct": ret,
         "exit_reason": "sel_target"}
    if arm is not None:
        t["sel_arm"] = arm
    t.update(kw)
    return t


def _ref(**arms):
    """Write the simulator's reference file under the (isolated) record directory."""
    LC.out_dir().mkdir(parents=True, exist_ok=True)
    js = {"as_of": "2026-10-11", "source": "test", "arms": arms}
    (LC.out_dir() / LC.REFERENCE_FILE).write_text(json.dumps(js), encoding="utf-8")
    return js


def test_arm_keys_and_books_are_the_simulators():
    """The arm -> book map is books.py's constants under mimic.BOOK's keys — two sites pinned equal."""
    from src.backtest import mimic as M
    from src.performance import books
    assert LC.arm_books() == M.BOOK and tuple(LC.ARM_KEYS) == tuple(M.BOOK)
    assert set(LC.arm_books().values()) <= set(books.LIVE_MECHANISMS)


def test_arms_split_by_sel_arm_and_by_mechanism():
    from src.performance import books
    T = [_closed("vol"), _closed("etf", ticker="ETFX"), _closed("fall", ticker="BBB"),
         _closed("vol", ticker="CCC"),
         _closed(mech=books.DIP_LONG, action="BUY", entry_px=100.0, exit_px=101.0, ret=0.9, ticker="DDD"),
         _closed(mech=books.CRASH_LONG, action="BUY", entry_px=50.0, exit_px=52.0, ret=3.8, ticker="TQQQ"),
         _closed("model"), _closed("vol2"), _closed("thin"),                        # the shadow arms: not counted
         _closed("model+vol", ticker="EEE"),                                         # a pre-2026-10-04 shared stamp
         _closed(mech=books.ETF_DM, action="BUY", ticker="QQQ"),                     # a live book, not an arm
         _closed(mech="rank", action="BUY", ticker="FFF"),                           # legacy
         _closed("vol", entry="2026-10-08T14:00:00+00:00", ticker="GGG")]           # entered before the window
    f = LC.arm_facts(T, date(2026, 10, 9), date(2026, 10, 14))
    assert {a: f[a]["n"] for a in LC.ARM_KEYS} == {"vol": 3, "etf": 1, "fall": 1, "dip": 1, "crash": 1}
    assert LC.arm_of(_closed("model+vol")) == "vol" and LC.arm_of(_closed("model")) is None
    assert LC.arm_of({"entry_mechanism": books.SEL_SHORT}) is None                 # no arm stamped at all


def test_return_per_day_and_gross_win_rate_follow_the_standing_definitions():
    """(1 + the average net trade)^(1 / the average 24-hour hold) - 1; a win = the right direction on raw prices."""
    from src.performance import books
    T = [  # a short that fell 10% over exactly 2.0 days, +9.0% net; one that rose 5% over 0.5 days, -6.0% net
         _closed("vol", exit_="2026-10-14T14:00:00+00:00", entry_px=10.0, exit_px=9.0, ret=9.0),
         _closed("vol", exit_="2026-10-13T02:00:00+00:00", entry_px=10.0, exit_px=10.5, ret=-6.0, ticker="BBB"),
         _closed("vol", status="OPEN", exit_=None, exit_px=None, ret=2.0, ticker="CCC",                 # still open
                 entry="2026-10-13T14:00:00+00:00", current_price=9.8),
         # a date-only legacy row: the hold counts from midnight to midnight (3 days)
         _closed(mech=books.DIP_LONG, action="BUY", entry="2026-10-12", exit_="2026-10-15", entry_px=100.0,
                 exit_px=101.0, ret=0.9, ticker="DDD", entry_datetime=None, exit_datetime=None),
         # a closed trade without a usable price pair: in n and the net statistics, in neither gross count
         _closed("etf", entry_px=None, exit_px=9.0, ret=0.0, ticker="ETFX")]
    f = LC.arm_facts(T, date(2026, 10, 9), date(2026, 10, 14))
    v = f["vol"]
    assert v["n"] == 2 and v["open_n"] == 1
    assert v["avg_trade"] == pytest.approx(1.5) and v["avg_days"] == pytest.approx(1.25)
    assert v["return_per_day"] == pytest.approx(((1 + 1.5 / 100) ** (1 / 1.25) - 1) * 100)   # ~1.198 %/day
    assert (v["gross_win"]["k"], v["gross_win"]["n"], v["gross_win"]["share"]) == (1, 2, 0.5)
    assert v["gross_win"]["lo95"] == pytest.approx(0.0945, abs=1e-3)
    assert v["gross_win"]["hi95"] == pytest.approx(0.9055, abs=1e-3)
    assert v["gross_win_rate"] == 50.0                                              # tracker.gross_win_rate's %
    assert v["net"]["n"] == 2 and v["net"]["median"] == pytest.approx(1.5) and v["net"]["median_lo95"] is None
    d = f["dip"]
    assert d["avg_days"] == pytest.approx(3.0)
    assert d["return_per_day"] == pytest.approx(((1.009) ** (1 / 3) - 1) * 100)
    e = f["etf"]
    assert e["n"] == 1 and e["gross_win"]["n"] == 0 and e["gross_win_rate"] is None and e["net"]["n"] == 1
    assert LC.hold_days({"entry_datetime": "2026-10-12T14:00:00+00:00"}) is None
    assert LC.return_per_day(1.5, 0.0) is None and LC.return_per_day(None, 2.0) is None


def test_too_few_trades_is_the_verdict_under_five_and_open_trades_do_not_count():
    _ref(vol={"return_per_day": 1.3, "gross_win_rate": 0.68, "avg_trade": 3.0, "avg_days": 2.5, "n": 400, "set": "A"})
    T = [_closed("vol", ticker=f"T{i}") for i in range(4)] + [_closed("vol", status="OPEN", ticker="OPEN1")]
    rec = LC.arm_records(T, date(2026, 10, 9), date(2026, 10, 14))
    assert rec["vol"]["n"] == 4 and rec["vol"]["open_n"] == 1 and rec["vol"]["verdict"] == "too few trades"
    assert rec["vol"]["reference"]["set"] == "A" and rec["vol"]["checks"]["return_per_day"]["reference"] == 1.3
    assert rec["etf"]["verdict"] == "no reference"                                    # absent from the file
    assert LC.compare({"arms": rec}, {}) == []                                        # never a note under five


def test_a_gap_is_noted_when_the_reference_sits_outside_the_live_ranges():
    _ref(vol={"return_per_day": 1.3, "gross_win_rate": 0.68, "avg_trade": 3.0, "avg_days": 2.5, "n": 400, "set": "A"},
         fall={"return_per_day": 1.5, "gross_win_rate": 0.1, "avg_trade": -9.0, "avg_days": 2.0, "n": 80, "set": "A"})
    # eight shorts that all ROSE: gross win 0/8 (Wilson 0 .. 0.32), the net median -10.5 (range -10.5 .. -10.5)
    T = [_closed("vol", exit_px=11.0, ret=-10.5, ticker=f"V{i}") for i in range(8)]
    # eight fallers, every one a loser too, but the reference lies inside both live ranges
    rets = [-12.0, -11.0, -10.5, -10.0, -9.0, -8.0, -7.0, -6.0]
    T += [_closed("fall", exit_px=11.0, ret=r, ticker=f"F{i}") for i, r in enumerate(rets)]
    rec = LC.arm_records(T, date(2026, 10, 9), date(2026, 10, 14))
    v = rec["vol"]
    assert v["verdict"] == "gap"
    assert v["checks"]["gross_win_rate"]["inside"] is False and v["checks"]["avg_trade"]["inside"] is False
    assert v["checks"]["return_per_day"]["diff"] == pytest.approx(v["return_per_day"] - 1.3)
    assert rec["fall"]["verdict"] == "consistent"                                     # 0.1 in 0 .. 0.32; -9 in -12 .. -6
    notes = LC.compare({"arms": rec}, {})
    assert len(notes) == 1
    assert notes[0].startswith("the vol arm's 8 live trades: gross win 0/8 (0%; the simulator 68%)")
    assert "net median -10.50% a trade (the simulator's average +3.00%)" in notes[0]
    assert "vs the simulator's 1.30 %/day" in notes[0]
    # the same gap on four trades is "too few trades" and silent
    few = LC.arm_records(T[:4], date(2026, 10, 9), date(2026, 10, 14))
    assert few["vol"]["verdict"] == "too few trades" and LC.compare({"arms": few}, {}) == []


def test_a_missing_reference_file_compares_nothing():
    assert LC.reference() == {}
    T = [_closed("vol", ticker=f"V{i}") for i in range(8)]
    rec = LC.arm_records(T, date(2026, 10, 9), date(2026, 10, 14))
    assert rec["vol"]["n"] == 8 and rec["vol"]["verdict"] == "no reference" and "reference" not in rec["vol"]
    assert all(rec[a]["verdict"] == "no reference" for a in LC.ARM_KEYS)
    assert LC.compare({"arms": rec}, {}) == []
    LC.out_dir().mkdir(parents=True, exist_ok=True)
    (LC.out_dir() / LC.REFERENCE_FILE).write_text("{", encoding="utf-8")            # unreadable: the same
    assert LC.reference() == {}


def test_the_record_carries_the_arms_and_their_gap_rides_through_health(monkeypatch):
    """2026-10-19: eight vol shorts entered 10-12 sit in the record since 10-09, not in the last five sessions."""
    _quiet_sections(monkeypatch)
    _ref(vol={"return_per_day": 1.3, "gross_win_rate": 0.68, "avg_trade": 3.0, "avg_days": 2.5, "n": 400, "set": "A"})
    T = [_closed("vol", exit_px=11.0, ret=-10.5, ticker=f"V{i}") for i in range(8)]
    rec = LC.build(date(2026, 10, 19), T)
    assert rec["reference"]["as_of"] == "2026-10-11" and rec["reference"]["arms"] == ["vol"]
    assert rec["since"]["arms"]["vol"]["n"] == 8 and rec["since"]["arms"]["vol"]["verdict"] == "gap"
    assert rec["window"]["arms"]["vol"]["n"] == 0 and rec["window"]["arms"]["vol"]["verdict"] == "too few trades"
    assert rec["session"]["arms"]["dip"]["verdict"] == "no reference"
    assert len(rec["notes"]) == 1 and rec["notes"][0].startswith("the vol arm's 8 live trades")
    path = LC.write(date(2026, 10, 19), T)
    assert json.loads(path.read_text(encoding="utf-8"))["since"]["arms"]["vol"]["return_per_day"] is not None
    h = LC.health()
    assert h["problems"] == [] and h["notes"][1].startswith("live vs simulator: the vol arm's 8 live trades")
    (LC.out_dir() / LC.REFERENCE_FILE).unlink()                                      # the file gone: nothing compared
    missing = LC.build(date(2026, 10, 19), T)
    assert missing["reference"] == {"path": str(LC.out_dir() / LC.REFERENCE_FILE), "missing": True}
    assert missing["since"]["arms"]["vol"]["verdict"] == "no reference" and missing["notes"] == []
