"""THE TURN for the faller shorts (PREREG76; user directive 2026-10-10: "Deploy the 'Turn, fallers shorts', even if
it's not significant it is better and not hurting"): a faller pick is not shorted at once — it WAITS, and is shorted
at the live price once a completed regular-hours 30-minute bar closes at least ½ x the pick's ATR$ (its close x its
30-minute ATR%) under the highest close since the pick (the pick close included), within 5 sessions counting the
pick's own.

What must hold: no turn yet = the pick waits (never expired at 75 minutes, never consumed); the FIRST bar that turns
triggers, with the running maximum including later highs; only bars COMPLETED by now count; the pick bar's close comes
from the bars (their terms — a split during the wait cannot misplace the level); no turn by the end of the window
(plus the entry freshness) = `turn_lapsed`; a turn first seen after the entry freshness = `turn_expired`; picks of
earlier sessions keep waiting; the vol arm and the switch-off keep the old 75-minute rule; the trade carries the turn
and the pick's own deadline.
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
D = date(2026, 9, 28)                  # a Monday: the window is Mon 09-28 .. Fri 10-02


@pytest.fixture
def on(monkeypatch):
    for k, v in {"enable_sel_short": True, "enable_sel_short_vol": True, "enable_sel_short_fall": True,
                 "enable_sel_short_fall_turn": True, "sel_short_turn_atr": 0.5,
                 "sel_short_turn_sessions": 5, "sel_short_entry_max_age_minutes": 75.0,
                 "sel_short_traded_arms": "vol,etf,fall"}.items():
        monkeypatch.setattr(settings, k, v)
    return settings


def _pick(tk="AAA", arm="fall", day=D, bar=1, px=10.0, atr=4.0, **kw):
    r = {"day": day.isoformat(), "bar_of_day": bar, "bar_end": ss.bar_end_et(day, bar).isoformat(), "ticker": tk,
         "decision": "short", "px": px, "pre5": 12.0, "target": None, "arm": arm, "score": atr, "atr_pct": atr,
         "deadline": ss.bar_end_et(ss.session_after(day, 15), bar).isoformat(), "runup_pct": -16.7,
         "days_to_cover": 0.5, "rvol": 2.0, "dv20": 5e7, "fresh": True}
    r.update(kw)
    return r


def _starts(day, bar, n):
    """``n`` consecutive regular-hours bar starts (naive UTC) from bar ``bar`` of ``day``, across sessions."""
    out, d, k = [], day, bar
    while len(out) < n:
        t = ss.bar_end_et(d, k) - timedelta(minutes=30)
        out.append(pd.Timestamp(t).tz_convert("UTC").tz_localize(None))
        k += 1
        if k >= ss.BARS_PER_SESSION:
            d, k = ss.session_after(d, 1), 0
    return pd.DatetimeIndex(out)


def _hlc(closes, day=D, bar=1):
    """The `series` tuple for the pick bar (``closes[0]``) and the bars after it."""
    idx = _starts(day, bar, len(closes))
    c = pd.Series(np.asarray(closes, float))
    return idx, c.copy(), c.copy(), c, pd.Series(np.ones(len(closes)))


def _end(i, day=D, bar=1):
    """The END of the ``i``-th bar of `_hlc` (0 = the pick bar), ET."""
    t = _starts(day, bar, i + 1)[-1] + pd.Timedelta(minutes=30)
    return t.tz_localize("UTC").tz_convert(ET).to_pydatetime()


# ── the trigger ──────────────────────────────────────────────────────────────

def test_no_turn_yet_waits(on):
    st = ss.turn_state(_pick(), _end(3), hlc=_hlc([10.0, 10.1, 10.15, 9.96]))  # 9.96 > 10.15 - 0.2
    assert st["state"] == "waiting"


def test_the_first_bar_closing_half_an_atr_under_the_highest_close_triggers(on):
    # ATR$ = 10 x 4% = 0.40, half = 0.20; the highest close since the pick is 10.8 -> the level 10.6
    closes = [10.0, 10.5, 10.8, 10.7, 10.55, 10.2]
    st = ss.turn_state(_pick(), _end(5), hlc=_hlc(closes))
    assert st["state"] == "triggered"
    assert st["close"] == pytest.approx(10.55) and st["max_close"] == pytest.approx(10.8)
    assert st["level"] == pytest.approx(10.6) and st["k"] == pytest.approx(0.5)
    assert datetime.fromisoformat(st["bar_end"]) == _end(4)


def test_an_immediate_faller_triggers_on_its_first_bar(on):
    st = ss.turn_state(_pick(), _end(2), hlc=_hlc([10.0, 9.75, 9.5]))           # 9.75 <= 10 - 0.2
    assert st["state"] == "triggered" and datetime.fromisoformat(st["bar_end"]) == _end(1)


def test_only_bars_completed_by_now_count(on):
    h = _hlc([10.0, 10.1, 9.7])
    assert ss.turn_state(_pick(), _end(2) - timedelta(minutes=1), hlc=h)["state"] == "waiting"
    assert ss.turn_state(_pick(), _end(2), hlc=h)["state"] == "triggered"


def test_the_level_is_judged_in_the_bars_terms(on):
    """A split during the wait: the bars come back adjusted (x0.5) while the journal kept the pick close 10 —
    the pick bar's close is read from the bars (5.0; ATR$ 0.20, half 0.10, level 4.90)."""
    h = _hlc([5.0, 4.95, 4.88])
    st = ss.turn_state(_pick(px=10.0), _end(2), hlc=h)
    assert st["state"] == "triggered" and st["close"] == pytest.approx(4.88) and st["level"] == pytest.approx(4.9)


def test_no_turn_in_the_window_lapses_after_its_last_bar_and_the_entry_freshness(on):
    # 5 sessions counting the pick's own: Mon .. Fri; a turn in the 6th session (Mon 10-05) never counts
    n = 11 + 13 * 4                                                              # every window bar after the pick
    closes = [10.0] + [10.1] * n + [9.0]
    h = _hlc(closes)
    end = ss.turn_window_end(_pick())
    assert end == ss.bar_end_et(date(2026, 10, 2), ss.BARS_PER_SESSION - 1)
    assert ss.turn_state(_pick(), end + timedelta(minutes=30), hlc=h)["state"] == "waiting"      # the last bar's grace
    assert ss.turn_state(_pick(), end + timedelta(minutes=76), hlc=h)["state"] == "lapsed"
    assert ss.turn_state(_pick(), _end(n + 1), hlc=h)["state"] == "lapsed"     # the 6th session's fall is too late


def test_without_the_pick_bar_the_turn_cannot_be_judged(on):
    h = _hlc([10.0, 9.7], day=D, bar=3)                                          # bars of another bar of the day
    assert ss.turn_state(_pick(bar=1), _end(4), hlc=h)["state"] == "no_bars"


# ── the entry step's queue ───────────────────────────────────────────────────

def _queue(monkeypatch, picks_by_day, bars):
    """The journal holds ``picks_by_day``; `_turn_bars` serves ``bars[ticker]``."""
    for day, picks in picks_by_day.items():
        for r in picks:
            ss._journal(r)
    monkeypatch.setattr(ss, "_turn_bars", lambda rec, now: bars.get(rec["ticker"]))


def test_a_waiting_faller_is_never_expired_at_75_minutes(monkeypatch, on):
    _queue(monkeypatch, {D: [_pick("AAA")]}, {"AAA": _hlc([10.0, 10.1, 10.15, 10.2, 10.25, 10.3])})
    now = _end(5)                                                                # 2 h after the pick bar
    assert ss.pending_entries(now) == []
    assert ss.consumed(D) == {}                                                  # still waiting, never expired


def test_a_fresh_turn_is_handed_to_the_entry_step_with_its_turn(monkeypatch, on):
    _queue(monkeypatch, {D: [_pick("AAA")]}, {"AAA": _hlc([10.0, 10.5, 10.8, 10.55])})
    out = ss.pending_entries(_end(3) + timedelta(minutes=3))
    assert [r["ticker"] for r in out] == ["AAA"]
    assert out[0]["turn"]["state"] == "triggered" and out[0]["turn"]["close"] == pytest.approx(10.55)


def test_a_turn_first_seen_after_the_entry_freshness_expires(monkeypatch, on):
    _queue(monkeypatch, {D: [_pick("AAA")]}, {"AAA": _hlc([10.0, 9.7, 9.6, 9.5, 9.4])})
    assert ss.pending_entries(_end(1) + timedelta(minutes=80)) == []
    assert ss.consumed(D) == {ss.pick_key(_pick("AAA")): "turn_expired"}
    lines = [json.loads(x) for x in ss.entries_path(D).read_text(encoding="utf-8").splitlines()]
    assert lines[-1]["outcome"] == "turn_expired" and lines[-1]["turn"]["close"] == pytest.approx(9.7)


def test_no_turn_by_the_end_of_the_window_lapses(monkeypatch, on):
    n = 11 + 13 * 4
    _queue(monkeypatch, {D: [_pick("AAA")]}, {"AAA": _hlc([10.0] + [10.1] * n)})
    now = ss.turn_window_end(_pick()) + timedelta(minutes=90)
    assert ss.pending_entries(now) == []
    assert ss.consumed(D) == {ss.pick_key(_pick("AAA")): "turn_lapsed"}


def test_an_earlier_sessions_pick_keeps_waiting_and_enters_on_its_turn(monkeypatch, on):
    # picked Monday (bar 1); it turns at Wednesday's bar 3 (11:00-11:30): 10.15 <= 10.4 - 0.2
    closes = [10.0] + [10.2] * 12 + [10.3] * 13 + [10.4, 10.35, 10.15]
    _queue(monkeypatch, {D: [_pick("AAA")]}, {"AAA": _hlc(closes)})
    wed = date(2026, 9, 30)
    assert ss.pending_entries(datetime(2026, 9, 29, 15, 0, tzinfo=ET)) == []
    assert ss.pending_entries(ss.bar_end_et(wed, 2) + timedelta(minutes=3)) == []          # bar 3 not over yet
    out = ss.pending_entries(ss.bar_end_et(wed, 3) + timedelta(minutes=3))
    assert [r["ticker"] for r in out] == ["AAA"] and out[0]["turn"]["close"] == pytest.approx(10.15)


def test_the_vol_arm_keeps_the_75_minute_rule(monkeypatch, on):
    vol = _pick("VVV", arm="vol", target=8.0, pre5=8.0, runup_pct=25.0)
    _queue(monkeypatch, {D: [vol]}, {})
    assert [r["ticker"] for r in ss.pending_entries(_end(1) + timedelta(minutes=5))] == ["VVV"]
    assert ss.pending_entries(_end(1) + timedelta(minutes=80)) == []
    assert ss.consumed(D) == {ss.pick_key(vol): "expired"}


def test_the_switch_off_shorts_a_faller_at_once(monkeypatch, on):
    monkeypatch.setattr(settings, "enable_sel_short_fall_turn", False)
    _queue(monkeypatch, {D: [_pick("AAA")]}, {})
    out = ss.pending_entries(_end(1) + timedelta(minutes=5))
    assert [r["ticker"] for r in out] == ["AAA"] and "turn" not in out[0]


def test_a_shadow_arm_is_settled_without_waiting(monkeypatch, on):
    monkeypatch.setattr(settings, "sel_short_traded_arms", "vol,etf")
    _queue(monkeypatch, {D: [_pick("AAA")]}, {})
    out = ss.pending_entries(_end(1) + timedelta(minutes=5))
    assert [r["ticker"] for r in out] == ["AAA"]                                 # the entry step settles it shadow_arm


# ── the trade ────────────────────────────────────────────────────────────────

def test_the_trade_carries_the_turn_and_the_picks_deadline(monkeypatch, on):
    from src.data import ibkr_borrow
    from src.performance import tracker
    rec = _pick("AAA")
    rec["turn"] = ss.turn_state(rec, _end(3), hlc=_hlc([10.0, 10.5, 10.8, 10.55]))
    monkeypatch.setattr(ss, "pending_entries", lambda now=None: [rec])
    monkeypatch.setattr(tracker, "_fetch_price", lambda t: 10.5)
    monkeypatch.setattr(tracker, "_reference_close", lambda t: None)
    monkeypatch.setattr(tracker, "_execution_iso", lambda: "2026-09-28T15:03:00+00:00")
    monkeypatch.setattr(settings, "enable_sel_short_account_sizing", True)
    monkeypatch.setattr(settings, "enable_sel_short_ibkr_whatif", False)
    stamp = datetime(2026, 9, 28, 9, 0, tzinfo=ET)
    monkeypatch.setattr(ibkr_borrow, "short_block",
                        lambda ticker, price, base=None, now=None, max_fee_pct="default":
                        (None, ibkr_borrow.Borrow(ticker, 12.0, -5.0, 900_000, stamp)))
    assert tracker.record_sel_short_trades(run_id="r1") == 1
    t = next(x for x in tracker._load_trades() if x.get("entry_mechanism") == "sel_short")
    assert t["sel_arm"] == "fall" and t["entry_price"] == pytest.approx(10.5)
    assert t["sel_turn_close"] == pytest.approx(10.55) and t["sel_turn_max_close"] == pytest.approx(10.8)
    assert t["sel_turn_level"] == pytest.approx(10.6) and t["sel_turn_bar_end"] == rec["turn"]["bar_end"]
    assert t["sel_deadline"] == rec["deadline"] and t["sel_bar_close"] == pytest.approx(10.0)
    assert "on the turn" in t["rationale"] and "PREREG76" in t["rationale"]


# ── the ETF arm on the turn (PREREG77): its target cancels a pick that reaches it first ──

def _etf(tk="EEE", target=9.0, **kw):
    return _pick(tk, arm="etf", target=target, pre5=9.0, runup_pct=11.1, **kw)


def test_the_etf_arm_turns_too(on):
    st = ss.turn_state(_etf(), _end(3), hlc=_hlc([10.0, 10.5, 10.8, 10.55]))
    assert st["state"] == "triggered" and st["close"] == pytest.approx(10.55)


def test_an_etf_pick_reaching_its_target_first_is_cancelled(on):
    # ATR$ 0.40, a turn = 0.20 under the highest close: 9.9 is no turn (10.05 - 0.2 = 9.85) but reaches the target
    st = ss.turn_state(_etf(target=9.9), _end(2), hlc=_hlc([10.0, 10.05, 9.9]))
    assert st["state"] == "cancelled" and st["close"] == pytest.approx(9.9)
    st = ss.turn_state(_etf(target=9.8), _end(2), hlc=_hlc([10.0, 10.05, 9.8]))
    assert st["state"] == "cancelled"                                             # the turning bar reaches it too
    st = ss.turn_state(_etf(target=9.95), _end(3), hlc=_hlc([10.0, 9.95, 10.4, 10.1]))
    assert st["state"] == "cancelled" and datetime.fromisoformat(st["bar_end"]) == _end(1)   # the target came first
    st = ss.turn_state(_etf(target=9.0), _end(2), hlc=_hlc([10.0, 10.05, 9.7]))
    assert st["state"] == "triggered"                                             # the turn first, the target later


def test_the_etf_target_is_scaled_to_the_bars_terms(on):
    """A split during the wait: the journal kept the pick close 10 and target 9.6; the bars are x0.5."""
    st = ss.turn_state(_etf(target=9.6, px=10.0), _end(2), hlc=_hlc([5.0, 5.02, 4.8]))
    assert st["state"] == "cancelled" and st["target"] == pytest.approx(4.8)


def test_a_cancelled_etf_pick_is_settled_target_reached(monkeypatch, on):
    _queue(monkeypatch, {D: [_etf("EEE", target=9.95)]}, {"EEE": _hlc([10.0, 9.95, 10.4])})
    assert ss.pending_entries(_end(2) + timedelta(minutes=3)) == []
    assert ss.consumed(D) == {ss.pick_key(_etf("EEE")): "target_reached"}


def test_the_etf_switch_off_shorts_its_picks_at_once(monkeypatch, on):
    monkeypatch.setattr(settings, "enable_sel_short_etf_turn", False)
    _queue(monkeypatch, {D: [_etf("EEE")]}, {})
    out = ss.pending_entries(_end(1) + timedelta(minutes=5))
    assert [r["ticker"] for r in out] == ["EEE"] and "turn" not in out[0]


# ── the per-arm exits of PREREG77 ──

def test_the_vol_arm_covers_at_4x_without_its_volatility_exit_and_the_others_keep_theirs(monkeypatch, on):
    monkeypatch.setattr(settings, "sel_short_vol_cover_multiple", 4.0)
    monkeypatch.setattr(settings, "sel_short_cover_multiple", 6.0)
    monkeypatch.setattr(settings, "enable_sel_short_vol_volnorm_exit", False)
    assert ss.cover_multiple("vol") == 4.0
    assert ss.cover_multiple("fall") == ss.cover_multiple("etf") == ss.cover_multiple("vol2") == 6.0
    assert not ss.volnorm_exit("vol") and not ss.volnorm_exit("vol2")
    assert ss.volnorm_exit("fall") and ss.volnorm_exit("etf") and ss.volnorm_exit("model")
    monkeypatch.setattr(settings, "enable_sel_short_vol_volnorm_exit", True)
    assert ss.volnorm_exit("vol")


def test_the_etf_arm_gives_back_the_whole_run_up(monkeypatch, on):
    monkeypatch.setattr(settings, "sel_short_etf_give_back", 1.0)
    monkeypatch.setattr(settings, "sel_short_give_back", 0.5)
    monkeypatch.setattr(settings, "sel_short_vol_give_back", 1.25)         # the vol arm keeps its own (G1, 2026-10-14)
    assert ss.give_back("etf") == 1.0 and ss.give_back("vol") == 1.25 and ss.give_back("model") == 0.5


def test_the_squeeze_cover_level_is_the_arms(monkeypatch, on):
    from src.data import intraday_store
    from src.performance import tracker
    monkeypatch.setattr(settings, "enable_sel_short_squeeze_cover", True)
    monkeypatch.setattr(settings, "sel_short_vol_cover_multiple", 4.0)
    monkeypatch.setattr(settings, "sel_short_cover_multiple", 6.0)
    monkeypatch.setattr(intraday_store, "split_factor_between", lambda t, a, b: 1.0)
    now = datetime(2026, 10, 13, 15, 0, tzinfo=timezone.utc)
    base = {"ticker": "AAA", "entry_mechanism": "sel_short", "entry_price": 10.0, "entry_date": "2026-10-12",
            "entry_datetime": "2026-10-12T14:00:00+00:00"}
    assert tracker._sel_short_cover_level(dict(base, sel_arm="vol"), now) == pytest.approx(40.0)
    assert tracker._sel_short_cover_level(dict(base, sel_arm="fall"), now) == pytest.approx(60.0)
    assert tracker._sel_short_cover_level(dict(base, sel_arm="etf"), now) == pytest.approx(60.0)


# ── a split during the wait: the pick comes over in today's terms (2026-10-11 review) ──

def _split_fake(monkeypatch, factor_from=None, factor=2.0):
    """The split data: a split executed on ``factor_from`` multiplies older prices by ``factor``."""
    from src.data import intraday_store
    monkeypatch.setattr(intraday_store, "split_factor_between",
                        lambda tk, a, b, rows=None: factor if factor_from and a < factor_from <= b else 1.0)


def test_a_split_during_the_wait_hands_the_pick_over_in_todays_terms(monkeypatch, on):
    """A 1-for-2 reverse split on 09-29, between an ETF pick on 09-28 (close 10, target 9, base 9) and its turn on
    09-29: the bars are in today's terms (x2), the turn is judged on them, and the journaled closes and target come
    over x2 — the entry step's target check and the trade's target in the live price's terms."""
    _split_fake(monkeypatch, date(2026, 9, 29))
    closes = [20.0] + [20.2] * 11 + [20.4, 19.9]                        # the pick bar, the rest of 09-28, 2 of 09-29
    _queue(monkeypatch, {D: [_etf("EEE", target=9.0)]}, {"EEE": _hlc(closes)})
    out = ss.pending_entries(_end(13) + timedelta(minutes=3))
    assert [r["ticker"] for r in out] == ["EEE"] and out[0]["turn"]["state"] == "triggered"
    r = out[0]
    assert r["target"] == pytest.approx(18.0) and r["px"] == pytest.approx(20.0) and r["pre5"] == pytest.approx(18.0)
    assert r["pick_split_factor"] == pytest.approx(2.0)
    assert ss.read_picks(D)[0]["target"] == pytest.approx(9.0)          # the journal keeps the pick's own terms


def test_no_split_during_the_wait_leaves_the_pick_as_journaled(monkeypatch, on):
    _split_fake(monkeypatch, None)
    _queue(monkeypatch, {D: [_etf("EEE", target=9.0, px=20.0)]},
           {"EEE": _hlc([20.0] + [20.2] * 11 + [20.4, 19.9])})
    r = ss.pending_entries(_end(13) + timedelta(minutes=3))[0]
    assert r["target"] == 9.0 and r["px"] == 20.0 and "pick_split_factor" not in r


def test_the_trade_stamps_the_turns_split_factor_and_rescales_the_turn_after_a_split(monkeypatch, on):
    from src.performance import tracker
    assert {"sel_turn_close", "sel_turn_max_close", "sel_turn_level"} <= set(tracker.SPLIT_PRICE_FIELDS)


def test_a_pick_journaled_twice_is_handed_over_once(monkeypatch, on):
    """A rerun after a crash between the journal and the live marker can journal a pick twice: one trade."""
    v = _pick("VVV", arm="vol", target=9.0, pre5=9.0)
    f = _pick("AAA")
    _queue(monkeypatch, {D: [v, dict(v), f, dict(f)]}, {"AAA": _hlc([10.0, 10.5, 10.8, 10.55])})
    out = ss.pending_entries(_end(0) + timedelta(minutes=3))
    assert [r["ticker"] for r in out] == ["VVV"]                       # the faller still waits for its turn
    out = ss.pending_entries(_end(3) + timedelta(minutes=3))
    assert [r["ticker"] for r in out] == ["AAA"]                       # its turn; the vol pick is past 75 min
