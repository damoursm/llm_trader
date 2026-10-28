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


def test_assumptions_are_the_simulators(monkeypatch):
    """The simulator refuses BY NAME from 2026-10-24 (PREREG193): no flat rate to judge the live share against (the
    refusal model's own check judges it); under `Mimic(**PRE193)`'s rule the 7% comes back."""
    from src.backtest import mimic as M
    a = LC.assumptions()
    assert "error" not in a
    assert a["refusal_rate"] is None and a["refusal"] == f"by name: table {M.REFUSAL_TABLE}"
    assert a["entry_delay_min"] == 0.0 and a["house_names"] == len(M.house_mix())
    class _Rate:                                                                   # a simulator on PRE193's rule
        refusal, refusal_table, lend_intra = "rate", M.REFUSAL_TABLE, M.LEND_INTRA
    monkeypatch.setattr(M, "Mimic", lambda *a, **k: _Rate())
    b = LC.assumptions()
    assert b["refusal_rate"] == M.REFUSAL_RATE and b["refusal"] == "rate"


def test_the_eod_chain_runs_it_last():
    from src.scheduler import eod                      # the forward check (2026-10-23) runs after it
    assert eod.STEPS[-2:] == ("live_calibration", "forward_check") and "live_calibration" in eod.STEP_FNS
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


def _ref(current=False, **arms):
    """Write the simulator's reference file under the (isolated) record directory — ``current``: stamped with the
    deployed configuration's fingerprints (else it carries none, as every reference before 2026-10-21: stale)."""
    LC.out_dir().mkdir(parents=True, exist_ok=True)
    js = {"as_of": "2026-10-11", "source": "test", "arms": arms}
    if current:
        fps = LC.current_fingerprints()
        js.update(fingerprints={a: fps["arms"][a] for a in arms}, simulator=fps["simulator"])
    (LC.out_dir() / LC.REFERENCE_FILE).write_text(json.dumps(js), encoding="utf-8")
    return js


def _apart(arm, n, ret=-10.5, exit_px=11.0, start=date(2026, 10, 12), **kw):
    """``n`` closed trades of ``arm`` entered and closed on distinct sessions (each its own episode): entered
    ``start`` + i days, closed two days later."""
    from datetime import timedelta
    out = []
    for i in range(n):
        e = start + timedelta(days=i)
        x = e + timedelta(days=2)
        r = ret[i] if isinstance(ret, (list, tuple)) else ret
        out.append(_closed(arm, entry=f"{e.isoformat()}T14:00:00+00:00", exit_=f"{x.isoformat()}T14:00:00+00:00",
                           exit_px=exit_px, ret=r, ticker=f"{(arm or 'x').upper()}{i}", **kw))
    return out


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
         _closed(mech=books.CAP_LONG, action="BUY", entry_px=80.0, exit_px=83.0, ret=3.6, ticker="KKK"),
         _closed(mech=books.INSIDER_LONG, action="BUY", entry_px=20.0, exit_px=24.0, ret=19.5, ticker="III"),
         _closed(mech=books.FEAR_LONG, action="BUY", entry_px=60.0, exit_px=62.0, ret=3.2, ticker="JJJ"),
         _closed("model"), _closed("vol2"), _closed("thin"),                        # the shadow arms: not counted
         _closed("model+vol", ticker="EEE"),                                         # a pre-2026-10-04 shared stamp
         _closed(mech=books.ETF_DM, action="BUY", ticker="QQQ"),                     # a live book, not an arm
         _closed(mech="rank", action="BUY", ticker="FFF"),                           # legacy
         _closed("vol", entry="2026-10-08T14:00:00+00:00", ticker="GGG")]           # entered before the window
    f = LC.arm_facts(T, date(2026, 10, 9), date(2026, 10, 14))
    assert {a: f[a]["n"] for a in LC.ARM_KEYS} == {"vol": 3, "etf": 1, "fall": 1, "dip": 1, "crash": 1, "cap": 1,
                                                   "ins": 1, "fear": 1}
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
    # eight shorts that all ROSE, each its own episode: gross win 0/8 (Wilson 0 .. 0.32), the median -10.5
    T = _apart("vol", 8)
    # eight fallers, every one a loser too, but the reference lies inside both live ranges
    T += _apart("fall", 8, ret=[-12.0, -11.0, -10.5, -10.0, -9.0, -8.0, -7.0, -6.0])
    rec = LC.arm_records(T, date(2026, 10, 9), date(2026, 10, 31))
    v = rec["vol"]
    assert v["verdict"] == "gap" and v["episodes"] == 8
    assert v["checks"]["gross_win_rate"]["inside"] is False and v["checks"]["avg_trade"]["inside"] is False
    assert v["checks"]["return_per_day"]["diff"] == pytest.approx(v["return_per_day"] - 1.3)
    assert rec["fall"]["verdict"] == "consistent"                                     # 0.1 in 0 .. 0.32; -9 in -12 .. -6
    notes = LC.compare({"arms": rec}, {})
    assert len(notes) == 1
    assert notes[0].startswith("the vol arm's 8 live trades in 8 episodes since 2026-10-09: gross win 0/8 (0% over "
                               "the episodes; the simulator 68%)")
    assert "the episodes' median net -10.50% a trade (the simulator's average +3.00%)" in notes[0]
    assert "vs the simulator's 1.30 %/day" in notes[0]
    # the same gap on four trades is "too few trades" and silent
    few = LC.arm_records(T[:4], date(2026, 10, 9), date(2026, 10, 31))
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


def _floor(monkeypatch, iso="2026-10-09T09:30:00-04:00"):
    """Move the arms' floor (the live account's fresh start) to ``iso``."""
    monkeypatch.setattr(settings, "live_account_fresh_start", iso)


def test_the_record_carries_the_arms_and_their_gap_rides_through_health(monkeypatch):
    """Eight vol shorts entered 10-09 .. 10-16 (each its own episode) sit in the record since 10-09, five of them in
    the last five sessions' window; a CURRENT reference judges them, the gap rides through the banner, and the live
    arms without a reference make the reference's own note."""
    _quiet_sections(monkeypatch)
    _floor(monkeypatch)
    _ref(current=True,
         vol={"return_per_day": 1.3, "gross_win_rate": 0.68, "avg_trade": 3.0, "avg_days": 2.5, "n": 400, "set": "A"})
    T = _apart("vol", 8, start=date(2026, 10, 9))
    rec = LC.build(date(2026, 10, 19), T)
    assert rec["reference"]["as_of"] == "2026-10-11" and rec["reference"]["arms"] == ["vol"]
    assert rec["reference"]["stale"] == {}
    assert rec["since"]["arms"]["vol"]["n"] == 8 and rec["since"]["arms"]["vol"]["verdict"] == "gap"
    assert rec["session"]["arms"]["dip"]["verdict"] == "no reference"
    assert rec["notes"][0].startswith("the simulator's reference (as of 2026-10-11) is out of date — none for etf, ")
    assert rec["notes"][1].startswith("the vol arm's 8 live trades in 8 episodes since 2026-10-09")
    path = LC.write(date(2026, 10, 19), T)
    assert json.loads(path.read_text(encoding="utf-8"))["since"]["arms"]["vol"]["return_per_day"] is not None
    h = LC.health()
    assert h["problems"] == [] and h["notes"][2].startswith("live vs simulator: the vol arm's 8 live trades")
    (LC.out_dir() / LC.REFERENCE_FILE).unlink()                                      # the file gone: nothing compared
    missing = LC.build(date(2026, 10, 19), T)
    assert missing["reference"] == {"path": str(LC.out_dir() / LC.REFERENCE_FILE), "missing": True}
    assert missing["since"]["arms"]["vol"]["verdict"] == "no reference" and missing["notes"] == []


# ── 2026-10-21: the arm's own exits, episodes, configuration epochs, a stale reference (the user's "Fix all") ─────

def test_forced_exits_are_reported_apart_never_scored():
    """2026-10-20: the fresh start's nine closes of five chip stocks read as a dip "gap" — the account's and the
    operator's closes are not the rule's."""
    from src.performance import books
    dip = dict(mech=books.DIP_LONG, action="BUY", entry_px=100.0)
    T = _apart(None, 3, ret=2.0, exit_px=102.0, exit_reason="dip_rebound", **dip)
    T += _apart(None, 2, ret=-15.0, exit_px=85.0, exit_reason="fresh_start", start=date(2026, 10, 20), **dip)
    T += _apart(None, 1, ret=-3.0, exit_px=97.0, exit_reason="dip_gave_way", start=date(2026, 10, 25), **dip)
    T += _apart("vol", 2, ret=-40.0, exit_px=14.0, exit_reason="sel_margin_call", start=date(2026, 10, 26))
    f = LC.arm_facts(T, date(2026, 10, 9), date(2026, 10, 31))
    d = f["dip"]
    assert d["n"] == 3 and d["avg_trade"] == pytest.approx(2.0) and d["gross_win"]["k"] == 3
    assert d["forced"]["n"] == 3 and d["forced"]["by_reason"] == {"fresh_start": 2, "dip_gave_way": 1}
    assert d["forced"]["avg_trade"] == pytest.approx(-11.0) and d["forced"]["gross_win"]["k"] == 0
    assert f["vol"]["n"] == 0 and f["vol"]["forced"]["by_reason"] == {"sel_margin_call": 2}
    for r in ("fresh_start", "cap_delever", "crash_margin_call", "cap_gave_way"):
        assert LC.forced_exit({"exit_reason": r})
    for r in ("dip_rebound", "sel_target", "sel_cover", "crash_pm_exit", "cap_time", None):
        assert not LC.forced_exit({"exit_reason": r})


def test_episodes_join_trades_entered_or_closed_in_the_same_session():
    T = [{"entry_date": "2026-10-12", "exit_date": "2026-10-14"},
         {"entry_date": "2026-10-13", "exit_date": "2026-10-14"},       # closed with the first
         {"entry_date": "2026-10-13", "exit_date": "2026-10-15"},       # entered with the second: the same episode
         {"entry_date": "2026-10-14", "exit_date": "2026-10-16"},       # entered the day the first two closed: apart
         {"entry_date": "2026-10-20", "exit_date": None}]
    assert LC.episodes(T) == [[0, 1, 2], [3], [4]]
    assert LC.episodes([]) == []


def test_correlated_trades_count_once_in_the_verdict():
    """Eight losers of one basket — bought in one session, sold in one — are ONE episode: too few to judge; the same
    eight on eight sessions are a gap."""
    _ref(current=True,
         vol={"return_per_day": 1.3, "gross_win_rate": 0.68, "avg_trade": 3.0, "avg_days": 2.5, "n": 400, "set": "A"})
    basket = [_closed("vol", exit_px=11.0, ret=-10.5, ticker=f"V{i}") for i in range(8)]
    rec = LC.arm_records(basket, date(2026, 10, 9), date(2026, 10, 31))
    assert rec["vol"]["n"] == 8 and rec["vol"]["episodes"] == 1 and rec["vol"]["verdict"] == "too few episodes"
    assert LC.compare({"arms": rec}, {}) == []
    apart = LC.arm_records(_apart("vol", 8), date(2026, 10, 9), date(2026, 10, 31))
    assert apart["vol"]["episodes"] == 8 and apart["vol"]["verdict"] == "gap"


def test_the_arms_start_at_the_fresh_start_and_at_their_epoch(monkeypatch):
    """Trades entered before the live account's fresh start (or before an arm's settings changed) are never scored
    against the deployed rules' reference."""
    _floor(monkeypatch, "2026-10-20T09:30:00-04:00")
    fps = {"vol": "aaa", "dip": "ddd"}
    ep = LC.epochs(date(2026, 10, 21), fps)                              # nothing stored: every arm from the floor
    assert ep["vol"] == {"since": "2026-10-20", "fingerprint": "aaa", "changed": False}
    LC._keep_epochs(date(2026, 10, 21), ep)
    same = LC.epochs(date(2026, 10, 22), fps)
    assert same["vol"]["since"] == "2026-10-20" and not same["vol"]["changed"]
    moved = LC.epochs(date(2026, 10, 23), {"vol": "aaa", "dip": "eee"})    # the dip book's settings changed
    assert moved["dip"] == {"since": "2026-10-23", "fingerprint": "eee", "changed": True}
    assert moved["vol"]["since"] == "2026-10-20"
    LC._keep_epochs(date(2026, 10, 23), moved)
    LC._keep_epochs(date(2026, 10, 22), same)                            # a rerun of an earlier day moves nothing
    assert LC.epochs(date(2026, 10, 24), {"vol": "aaa", "dip": "eee"})["dip"]["since"] == "2026-10-23"
    T = _apart("vol", 3, ret=1.0, exit_px=9.9, start=date(2026, 10, 17))   # entered 10-17, 10-18, 10-19
    T += _apart("vol", 2, ret=2.0, exit_px=9.8, start=date(2026, 10, 20))  # entered 10-20, 10-21
    f = LC.arm_facts(T, date(2026, 10, 9), date(2026, 10, 31), {"vol": date(2026, 10, 20)})
    assert f["vol"]["n"] == 2 and f["vol"]["since"] == "2026-10-20" and f["vol"]["avg_trade"] == pytest.approx(2.0)


def test_a_reference_built_on_other_rules_is_stale_and_judges_nothing():
    ref = {"return_per_day": 1.3, "gross_win_rate": 0.68, "avg_trade": 3.0, "avg_days": 2.5, "n": 400, "set": "A"}
    js = _ref(vol=ref)                                                   # no fingerprint: a reference before them
    fps = LC.current_fingerprints()
    assert "error" not in fps and set(fps["arms"]) == set(LC.ARM_KEYS)
    assert LC.stale_arms(js, fps) == {"vol": "no fingerprint; no simulator fingerprint"}
    rec = LC.arm_records(_apart("vol", 8), date(2026, 10, 9), date(2026, 10, 31), js["arms"],
                         stale=LC.stale_arms(js, fps))
    assert rec["vol"]["verdict"] == "stale reference" and LC.compare({"arms": rec}, {}) == []
    cur = _ref(current=True, vol=ref)
    assert LC.stale_arms(cur, fps) == {}
    changed = dict(fps, arms=dict(fps["arms"], vol="changed"))           # a deploy since the build
    assert LC.stale_arms(cur, changed) == {"vol": "its settings changed since"}
    note = LC._reference_note(cur, LC.stale_arms(cur, changed), changed)
    assert "built on other rules than the deployed vol" in note and "none for etf, fall, dip, crash, cap" in note
    assert "python -m src.backtest.reliability --build" in note
    assert LC._reference_note(cur, {}, {"arms": {"vol": "x"}}) is None


def test_the_fingerprint_moves_with_an_arms_setting_and_only_its_own(monkeypatch):
    from src.backtest import reliability as REL
    before = {a: REL.arm_fingerprint(a) for a in REL.ARMS}
    monkeypatch.setattr(settings, "dip_long_exit_sma_sessions", int(settings.dip_long_exit_sma_sessions) + 1)
    after = {a: REL.arm_fingerprint(a) for a in REL.ARMS}
    assert after["dip"] != before["dip"] and all(after[a] == before[a] for a in REL.ARMS if a != "dip")
    assert REL.simulator_fingerprint() == REL.simulator_fingerprint()


def _wi(tk, outcome, answer):
    return {"key": f"2026-10-20|{tk}|fall", "day": "2026-10-20", "arm": "fall", "ticker": tk, "outcome": outcome,
            "at": "2026-10-20T15:00:00+00:00", "price": 10.0, "ibkr_margin": answer}


def test_the_refusal_model_is_scored_on_ibkrs_own_answers(monkeypatch):
    """PREREG190's model against IBKR's what-if answers: one per pick, a timeout counts in neither, the expected count
    and the AUC (journal-only)."""
    from src.signals import sel_refusal as SR
    probs = {"A": 0.5, "B": 0.5, "C": 0.01, "D": 0.01, "E": 0.01, "F": 0.01, "G": 0.9}
    monkeypatch.setattr(SR, "model", lambda: {"source": "test"})
    monkeypatch.setattr(SR, "probability", lambda tk, d, price=None: probs[tk])
    monkeypatch.setattr("src.signals.sel_short.etf_names", lambda names: [])
    priced, refused = {"init_rate": 2.9, "maint_rate": 2.0}, {"refused": "closing-only"}
    entries = [_wi("A", "ibkr_refused", refused), _wi("B", "opened", priced), _wi("C", "opened", priced),
               _wi("D", "opened", priced), _wi("E", "opened", priced), _wi("F", "ibkr_refused", refused),
               _wi("F", "ibkr_refused", refused),                       # the same pick journaled twice: one answer
               _wi("G", "opened", {"error": "timeout"})]                # no answer: in neither
    f = LC.refusal_model_facts(entries, ["fall"])
    assert f["n"] == 6 and f["refused"] == 2 and abs(f["expected"] - 1.04) < 1e-9
    assert abs(f["auc"] - 0.625) < 1e-12 and f["range95"][0] <= 2 <= f["range95"][1]
    assert LC.pb_range([0.5, 0.5]) == (0, 2) and LC.pb_range([]) == (0, 0)


def test_the_refusal_models_notes(monkeypatch):
    from datetime import timedelta
    from src.data import ibkr_microcaps
    monkeypatch.setattr(ibkr_microcaps, "latest_day", lambda before=None: date.today() - timedelta(days=1))
    assert LC._refusal_notes({"n": 6, "refused": 2, "expected": 1.0, "range95": [0, 3]}) == []
    off = LC._refusal_notes({"n": 6, "refused": 5, "expected": 0.3, "range95": [0, 1]})
    assert len(off) == 1 and "its level is off" in off[0]
    monkeypatch.setattr(settings, "refusal_model_refit_refusals", 20)
    assert LC._refusal_notes({"n": 400, "refused": 20, "expected": 19.0, "range95": [11, 28]}) == []   # _refit_notes
    monkeypatch.setattr(ibkr_microcaps, "latest_day", lambda before=None: None)
    assert any("not archived" in n for n in LC._refusal_notes({"n": 0, "refused": 0, "range95": [0, 0]}))


def test_a_whatif_only_answer_is_a_label_never_an_entry_outcome(monkeypatch):
    """PREREG191's change 3: a pick settled before its what-if (``already_open``) and asked after the live path — its
    answer counts as IBKR's, its last OUTCOME stays the entry step's."""
    from src.signals import sel_refusal as SR
    monkeypatch.setattr("src.signals.sel_short.etf_names", lambda names: [])
    entries = [_wi("A", "already_open", None),
               {**_wi("A", "whatif_only", {"refused": "Small Cap"}), "at": "2026-10-20T15:01:00+00:00"},
               _wi("B", "opened", {"init_rate": 2.9, "maint_rate": 2.0})]
    f = LC.entry_facts(entries, ["fall"])
    assert f["outcomes"] == {"already_open": 1, "opened": 1} and f["picks_settled"] == 2
    assert f["ibkr_refused_stock"]["k"] == 1 and f["ibkr_refused_stock"]["n"] == 2
    monkeypatch.setattr(SR, "model", lambda: {"source": "test"})
    monkeypatch.setattr(SR, "probability", lambda tk, d, price=None: 0.2)
    g = LC.refusal_model_facts(entries, ["fall"])
    assert g["n"] == 2 and g["refused"] == 1


def test_the_evening_probes_answers_are_scored_by_source(monkeypatch, tmp_path):
    """PREREG191's evening probe against the refusal model: the arms' candidates (picks, top names) apart, an
    unanswered ask nowhere, the re-asks still refused, the latest run's failure a note."""
    from src.signals import refusal_probe as RP
    monkeypatch.setattr(settings, "sel_short_dir", str(tmp_path / "sel"))
    d = RP.probes_dir()
    d.mkdir(parents=True)
    rows = [{"ticker": "A", "source": "pick", "refused": "Small Cap", "p_model": 0.6},
            {"ticker": "B", "source": "top50", "init_rate": 2.9, "maint_rate": 2.0, "p_model": 0.1},
            {"ticker": "C", "source": "top50", "error": "timeout", "p_model": 0.1},
            {"ticker": "D", "source": "reask", "refused": "close-only", "p_model": 0.5},
            {"ticker": "E", "source": "queue", "init_rate": 3.0, "maint_rate": 2.0, "p_model": 0.05}]
    (d / "2026-10-26.jsonl").write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")
    (d / "2026-10-27.jsonl").write_text(json.dumps({"ticker": "Z", "source": "pick", "refused": "x", "p_model": 0.1}),
                                        encoding="utf-8")                    # after the day: not read
    (d / "2026-10-26.state.json").write_text(json.dumps({"day": "2026-10-26", "stopped": "failed: boom"}),
                                             encoding="utf-8")
    f = LC.refusal_probe_facts(date(2026, 10, 26))
    assert f["answers"] == 4 and f["arms"]["n"] == 2 and f["arms"]["refused"] == 1
    assert abs(f["arms"]["expected"] - 0.7) < 1e-9 and f["arms"]["auc"] == 1.0
    assert f["reasks"] == {"n": 1, "still_refused": 1} and set(f["by_source"]) == {"pick", "top50", "reask", "queue"}
    notes = LC._probe_notes(f)
    assert len(notes) == 1 and "failed" in notes[0]
    off = LC._probe_notes({"arms": {"n": 6, "refused": 5, "expected": 0.4, "range95": [0, 2]}})
    assert len(off) == 1 and "evening probe refused 5/6" in off[0]


def test_a_changed_answer_when_asked_again_is_recorded(monkeypatch, tmp_path):
    """PREREG192's watch: every consecutive pair of IBKR's answers about one name at least an hour apart, the changes
    either way listed — the studies' imports included, a later day's answers never."""
    from src.signals import refusal_probe as RP
    monkeypatch.setattr(settings, "sel_short_dir", str(tmp_path / "sel"))
    d = RP.probes_dir()
    d.mkdir(parents=True)
    (d / "import_prereg190.jsonl").write_text("\n".join(json.dumps(r) for r in [
        {"ticker": "A", "init_rate": 2.9, "maint_rate": 2.0, "at": "2026-10-23T14:00:00-04:00"},
        {"ticker": "B", "refused": "Small Cap", "at": "2026-10-23T14:00:00-04:00"},
        {"ticker": "C", "init_rate": 2.9, "maint_rate": 2.0, "at": "2026-10-23T14:00:00-04:00"}]), encoding="utf-8")
    (d / "2026-10-30.jsonl").write_text("\n".join(json.dumps(r) for r in [
        {"ticker": "A", "source": "watch", "refused": "closing-only", "at": "2026-10-30T17:00:00-04:00"},
        {"ticker": "B", "source": "reask", "init_rate": 3.1, "maint_rate": 2.2, "at": "2026-10-30T17:01:00-04:00"},
        {"ticker": "C", "source": "watch", "init_rate": 2.9, "maint_rate": 2.0, "at": "2026-10-30T17:02:00-04:00"},
        {"ticker": "C", "source": "watch", "refused": "x", "at": "2026-10-30T17:30:00-04:00"}]),   # 28 min later: skipped
        encoding="utf-8")
    (d / "2026-11-02.jsonl").write_text(json.dumps({"ticker": "C", "source": "watch", "refused": "later",
                                                    "at": "2026-11-02T17:00:00-05:00"}), encoding="utf-8")
    f = LC.refusal_probe_facts(date(2026, 10, 30))
    tr = f["transitions"]
    assert tr["pairs"] == 3
    assert [x["ticker"] for x in tr["to_refused"]] == ["A"] and tr["to_refused"][0]["reason"] == "closing-only"
    assert [x["ticker"] for x in tr["to_accepted"]] == ["B"]


def test_a_refit_is_due_on_new_refusals_since_the_install_and_new_daily_lists(monkeypatch, tmp_path):
    """The user 2026-10-24: "still refit/recalibrate the model to make it better in the future" — the refusal model's
    refit is due once `refusal_model_refit_refusals` names are refused after its install (the probe, the live record,
    the studies' imports; an answer before the install never counts), the listing model's once IBKR's daily list is
    archived `refusal_listing_refit_sessions` sessions after its table's last list."""
    from src.backtest import mimic as M
    from src.signals import refusal_probe as RP
    from src.signals import sel_refusal as SR
    monkeypatch.setattr(settings, "sel_short_dir", str(tmp_path / "sel"))
    monkeypatch.setattr(settings, "ibkr_microcap_dir", str(tmp_path / "micro"))
    monkeypatch.setattr(SR, "model", lambda: {"installed": "2026-10-23T21:00:00-04:00"})
    d = RP.probes_dir()
    d.mkdir(parents=True)
    rows = [{"ticker": f"R{i}", "source": "watch", "refused": "Small Cap", "at": "2026-10-27T17:00:00-04:00"} for i in range(3)]
    rows += [{"ticker": "OLD", "refused": "Small Cap", "at": "2026-10-23T19:00:00-04:00"},       # before the install
             {"ticker": "P1", "init_rate": 2.9, "maint_rate": 2.0, "at": "2026-10-27T17:05:00-04:00"}]
    (d / "2026-10-27.jsonl").write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")
    (tmp_path / "micro").mkdir()
    for day in ("2026-10-22", "2026-10-23", "2026-10-26", "2026-10-27"):
        (tmp_path / "micro" / f"{day}.csv.gz").write_bytes(b"")
    monkeypatch.setattr(M, "REFUSAL_LISTS_THROUGH", {M.REFUSAL_TABLE: "2026-10-23"})
    st = LC.refit_status(date(2026, 10, 27))
    assert st["refusal_model"] == {"installed": "2026-10-23T21:00:00-04:00", "names_answered_since": 4, "refused_since": 3}
    assert st["listing_model"]["lists_since"] == 2
    monkeypatch.setattr(settings, "refusal_model_refit_refusals", 3)
    monkeypatch.setattr(settings, "refusal_listing_refit_sessions", 20)
    notes = LC._refit_notes(st)
    assert len(notes) == 1 and "refit due" in notes[0] and "3 new refusals" in notes[0]
    monkeypatch.setattr(settings, "refusal_listing_refit_sessions", 2)
    assert len(LC._refit_notes(st)) == 2


def test_the_record_counts_the_implied_volatility_bars_ibkr_revised_and_the_fear_calls_they_flip(monkeypatch, tmp_path):
    """P7: the fetches' journal read against the store; a fetch before the journal (no ``revised`` field) not counted."""
    from datetime import timedelta

    import numpy as np

    from src.signals import dip_long as dl
    from src.signals import fear_long as fl
    monkeypatch.setattr(settings, "enable_fear_long", True)
    monkeypatch.setattr(settings, "fear_long_dir", str(tmp_path / "fear_long"))
    monkeypatch.setattr(settings, "fear_long_iv_dir", str(tmp_path / "iv"))
    monkeypatch.setattr(settings, "fear_long_first_session", "2026-10-28")
    d1, d2 = date(2026, 10, 28), date(2026, 10, 29)
    days, x = [], date(2026, 10, 27)
    while len(days) < 300:
        if dl.is_session(x):
            days.append(x)
        x -= timedelta(days=1)
    days = days[::-1]
    vals = list(np.linspace(0.2, 0.3, 300))                                 # rising: the last bar the year's top
    fl.merge_rows("AAA", [[str(d), v, v, v, v] for d, v in zip(days, vals)], day=d1)
    dl._write_json(fl.table_path(d1), {"names": {"AAA": {"status": "ok", "added": 300, "bars": 300}}})   # no journal
    fl.merge_rows("AAA", [[str(days[-1]), 0.05, 0.05, 0.05, 0.05]], day=d2)  # IBKR revised it: calm now
    dl._write_json(fl.table_path(d2), {"names": {"AAA": {"status": "ok", "answered": 21, "added": 1, "revised": 1,
                                                          "absent": 0}}})
    f = LC.iv_revision_facts(d2)
    assert (f["fetches"], f["reread"], f["revised"], f["absent"], f["names"]) == (1, 20, 1, 0, ["AAA"])
    assert f["max_close_move"] == pytest.approx(1 - 0.05 / 0.3)
    assert f["flips"] == [{"sym": "AAA", "date": str(days[-1]), "ivp_old": round((251 + 0.5) / 252, 4),
                           "ivp_new": round(0.5 / 252, 4)}]
    assert f["session"]["revised"] == 1 and f["session"]["journal"].endswith("2026-10-29.jsonl")
    notes = LC._iv_notes(f)
    assert len(notes) == 2 and notes[0].startswith("IBKR revised 1 of the 20 stored implied-volatility bars")
    assert notes[1].startswith("1 fear call flipped") and "AAA" in notes[1]
    f2 = LC.iv_revision_facts(date(2026, 10, 30))                            # a later session: the record's flip only
    assert f2["session"] is None and f2["fetches"] == 1 and len(LC._iv_notes(f2)) == 1
