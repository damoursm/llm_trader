"""PREREG99 — trading halts in the simulator (src/backtest/halts.py): NYSE's halt file read in ET with its duplicates
merged, production's entry passes (two per slot, 75 minutes), a halt in force from its start to its resume (without a
resume: until a post-halt 1-minute bar has ended, at the latest the next open), the 1-minute price at a pass in the
candidate's dollars, V0 (production's guard) and the variants H1a / H1b / H2 / H3, the vol arm's pass loop (fallback,
pressure extras), NO LOOK-AHEAD (rewriting everything after a pass changes nothing at it — and a leak is caught), the
ETF arm's delayed piece, and `Mimic(halts=None)` building exactly what it built before. Synthetic data only — no
research store, no network."""
from __future__ import annotations

import types

import numpy as np
import pandas as pd
import pytest

from config import settings
from src.backtest import halts as H
from src.backtest import mimic as M

ET = "America/New_York"
# production's slots (rth every 30 minutes 09:30-16:00 + the extended windows), minutes after midnight ET
SLOTS = (240, 300, 360, 420, 450, 480, 510, 540) + tuple(range(570, 961, 30)) + (990, 1020, 1050, 1080, 1140, 1190)
WINDOWS = "04:00-07:00@60,07:00-09:30,16:00-17:30,18:00-19:00@60,19:50-19:50"


def ns(day, hhmmss):
    return pd.Timestamp(f"{day} {hhmmss}", tz=ET).value


def dn(day):
    return int((pd.Timestamp(day) - pd.Timestamp("1970-01-01")).days)


def et(t):
    return pd.Timestamp(int(t), unit="ns", tz="UTC").tz_convert(ET).strftime("%Y-%m-%d %H:%M:%S")


CAL = np.array([dn(d) for d in pd.bdate_range("2021-03-01", "2021-03-31")], np.int64)


def halt_csv(tmp_path, rows, name="nyse_halts_2021.csv"):
    """rows: (halt date, halt time, symbol, reason, resume date, resume time) — "" = blank."""
    cols = ["Halt Date", "Halt Time", "Symbol", "Name", "Exchange", "Reason", "Resume Date", "NYSE Resume Time"]
    df = pd.DataFrame([(a, b, s, f"{s} Inc", "Nasdaq", r, c, d) for a, b, s, r, c, d in rows], columns=cols)
    p = tmp_path / name
    df.to_csv(p, index=False)
    return p


def minute_source(bars):
    """bars: {(sym, "YYYY-MM-DD"): [("HH:MM", close, volume), ...]} -> a MinuteBars source."""
    def src(sym, day):
        iso = H.iso_day(day)
        rows = bars.get((sym, iso))
        if rows is None:
            return {"s": np.zeros(0, np.int64), "c": np.zeros(0), "v": np.zeros(0)}
        rows = sorted(rows)
        return {"s": np.array([ns(iso, f"{h}:00") for h, _, _ in rows], np.int64),
                "c": np.array([float(c) for _, c, _ in rows]), "v": np.array([float(v) for _, _, v in rows])}
    return src


def model(tmp_path, rows, bars=None, cal=CAL):
    book = H.HaltBook(H.read_halts([halt_csv(tmp_path, rows)]))
    return H.HaltModel(book, H.MinuteBars(source=minute_source(bars or {})), H.Schedule(cal, SLOTS, 75.0))


# ── the halt file ────────────────────────────────────────────────────────────

def test_the_halt_file_is_read_in_et_and_a_halt_listed_twice_is_one_halt(tmp_path):
    rows = [("2021-03-10", "12:22:06", "GME", "LULD pause", "2021-03-10", "12:27:12"),
            ("2021-03-10", "12:22:06", "GME", "LULD Pause", "", ""),               # its notice, without a resume
            ("2021-03-10", "12:29:53", "GME", "LULD pause", "2021-03-10", "12:34:58"),
            ("2021-03-10", "12:29:53", "GME", "LULD pause", "2021-03-10", "12:34:58"),  # an exact duplicate
            ("2021-07-01", "10:00:00", "BRK B", "News Pending", "", ""),             # no resume at all (EDT)
            ("2021-03-11", "03:58:00", "ZZZ", "News pending", "2021-03-11", "03:58:00")]
    t = H.read_halts([halt_csv(tmp_path, rows)])
    assert len(t) == 4
    g = t[t.sym == "GME"].sort_values("start")
    assert list(g.start) == [ns("2021-03-10", "12:22:06"), ns("2021-03-10", "12:29:53")]
    assert list(g.resume) == [ns("2021-03-10", "12:27:12"), ns("2021-03-10", "12:34:58")]
    assert pd.Timestamp(int(g.start.iloc[0]), unit="ns", tz="UTC").hour == 17          # ET (EST) -> UTC
    assert set(g.reason) == {"luld pause"}
    b = t[t.sym == "BRK B"].iloc[0]
    assert b.resume == -1 and pd.Timestamp(int(b.start), unit="ns", tz="UTC").hour == 14   # EDT
    assert H.halt_symbol("BRK-B") == "BRK B" and H.halt_symbol("XYZ@D") == "XYZ"


# ── production's passes ──────────────────────────────────────────────────────

def test_the_passes_are_two_per_slot_from_the_bars_end_while_at_most_75_minutes_old():
    S = H.Schedule(CAL, SLOTS, 75.0)
    got = [et(t)[11:] for t in S.passes(ns("2021-03-10", "10:30:00"))]
    assert got == ["10:31:30", "10:41:00", "11:01:30", "11:11:00", "11:31:30", "11:41:00"]
    got = [et(t)[11:] for t in S.passes(ns("2021-03-10", "16:00:00"))]           # the 16:00 bar: the post-market
    assert got == ["16:01:30", "16:11:00", "16:31:30", "16:41:00", "17:01:30", "17:11:00"]
    got = [et(t)[11:] for t in S.passes(ns("2021-03-10", "15:30:00"))]
    assert got == ["15:31:30", "15:41:00", "16:01:30", "16:11:00", "16:31:30", "16:41:00"]


def test_the_wait_runs_to_the_next_sessions_close_over_market_days_only():
    S = H.Schedule(CAL, SLOTS, 75.0)
    T = ns("2021-03-12", "15:00:00")                                                 # a Friday
    end = S.wait_end(T)
    assert et(end) == "2021-03-15 16:00:00"
    V = [et(t) for t in S.passes(T, until=end)]
    assert V[0] == "2021-03-12 15:01:30" and V[-1] == "2021-03-15 15:41:00"
    assert not any(v.startswith(("2021-03-13", "2021-03-14")) for v in V)           # no weekend pass
    assert "2021-03-12 20:01:00" in V and "2021-03-15 04:01:30" in V               # 19:50's second pass, 04:00's
    assert S.next_open(ns("2021-03-12", "19:00:00")) == ns("2021-03-15", "09:30:00")
    assert S.next_open(ns("2021-03-15", "08:00:00")) == ns("2021-03-15", "09:30:00")


def test_the_slots_are_the_schedulers_regular_and_extended_ones(monkeypatch):
    monkeypatch.setattr(settings, "extended_hours_mode", "trade", raising=False)
    monkeypatch.setattr(settings, "overnight_hours_mode", "trade", raising=False)
    monkeypatch.setattr(settings, "extended_windows", WINDOWS, raising=False)
    monkeypatch.setattr(settings, "intraday_session_start", "09:30", raising=False)
    monkeypatch.setattr(settings, "intraday_session_end", "16:00", raising=False)
    monkeypatch.setattr(settings, "extended_tick_minutes", 30, raising=False)
    H.production_slots.cache_clear()
    try:
        assert H.production_slots() == SLOTS                                         # the overnight venue's left out
    finally:
        H.production_slots.cache_clear()


# ── halts in force, prices ───────────────────────────────────────────────────

def test_a_halt_is_in_force_from_its_start_until_its_resume(tmp_path):
    m = model(tmp_path, [("2021-03-10", "12:22:06", "GME", "LULD pause", "2021-03-10", "12:27:12"),
                         ("2021-03-10", "12:00:00", "ABC", "News pending", "2021-03-10", "14:00:00"),
                         ("2021-03-10", "12:30:00", "ABC", "LULD pause", "2021-03-10", "12:35:00"),
                         ("2021-03-10", "03:58:00", "ZZZ", "News pending", "2021-03-10", "03:58:00")])
    assert m.halt_at("GME", ns("2021-03-10", "12:22:05")) is None
    assert m.halt_at("GME", ns("2021-03-10", "12:22:06"))["reason"] == "luld pause"
    assert m.halt_at("GME", ns("2021-03-10", "12:27:11")) is not None
    assert m.halt_at("GME", ns("2021-03-10", "12:27:12")) is None
    assert m.halt_at("GME@D", ns("2021-03-10", "12:25:00")) is not None              # the traded symbol decides
    h = m.halt_at("ABC", ns("2021-03-10", "12:31:00"))
    assert h["reason"] == "luld pause"                                               # the latest-starting one
    assert m.halt_at("ABC", ns("2021-03-10", "12:40:00"))["reason"] == "news pending"
    assert m.halt_at("ZZZ", ns("2021-03-10", "03:58:00")) is None                    # zero length: never
    assert m.halt_at("NONE", ns("2021-03-10", "12:25:00")) is None


def test_a_halt_without_a_resume_ends_once_a_post_halt_minute_has_ended_or_at_the_next_open(tmp_path):
    rows = [("2021-03-10", "10:00:30", "NEWS", "News pending", "", ""),
            ("2021-03-10", "15:30:00", "LATE", "News pending", "", "")]
    bars = {("NEWS", "2021-03-10"): [("09:59", 10.0, 500), ("10:00", 10.1, 300), ("10:07", 9.0, 800)]}
    m = model(tmp_path, rows, bars)
    assert m.halt_at("NEWS", ns("2021-03-10", "10:05:00")) is not None               # the 10:00 bar began before it
    assert m.halt_at("NEWS", ns("2021-03-10", "10:07:59")) is not None               # the 10:07 bar not ended yet
    assert m.halt_at("NEWS", ns("2021-03-10", "10:08:00")) is None
    assert m.halt_at("LATE", ns("2021-03-10", "19:59:00")) is not None               # no bar after it ...
    assert m.halt_at("LATE", ns("2021-03-11", "09:29:59")) is not None
    assert m.halt_at("LATE", ns("2021-03-11", "09:30:00")) is None                   # ... over at the next open
    ev = m.event("NEWS", ns("2021-03-10", "10:00:00"), ns("2021-03-10", "10:01:30"),
                 m.halt_at("NEWS", ns("2021-03-10", "10:01:30")))
    assert ev["reopen"].startswith("2021-03-10T10:08") and ev["missing_resume"] is True


def test_the_price_at_a_pass_is_the_last_minute_ended_by_then_in_the_candidates_dollars(tmp_path):
    # split-adjusted minute bars (a later 4:1 split): the candidate's bar closed at 302.68 traded dollars
    bars = {("GME", "2021-03-10"): [("12:20", 82.92, 1), ("12:21", 75.67, 1), ("12:27", 69.30, 1), ("12:28", 70.67, 1)],
            ("GME", "2021-03-11"): [("04:00", 60.0, 1), ("09:30", 65.0, 1)]}
    m = model(tmp_path, [], bars)
    T = ns("2021-03-10", "12:22:00")                                                 # the last bar ending by T: 12:21
    px, scale = m.price("GME", ns("2021-03-10", "12:28:00"), T, 302.68)
    assert scale == pytest.approx(4.0) and px == pytest.approx(69.30 * 4.0)         # 12:27 ended at 12:28: it counts
    assert m.price("GME", ns("2021-03-10", "12:28:59"), T, 302.68)[0] == pytest.approx(69.30 * 4.0)   # 12:28 not yet
    assert m.price("GME", ns("2021-03-10", "18:00:00"), T, 302.68)[0] == pytest.approx(70.67 * 4.0)
    assert m.price("GME", ns("2021-03-11", "04:00:30"), T, 302.68)[0] == pytest.approx(70.67 * 4.0)   # yesterday's
    assert m.price("GME", ns("2021-03-11", "09:31:30"), T, 302.68)[0] == pytest.approx(65.0 * 4.0)
    assert m.price("NOPE", ns("2021-03-10", "12:28:00"), T, 10.0) is None


# ── V0 and the variants on one candidate (a fall or ETF turn entry) ─────────

def _gme_model(tmp_path, resume="12:52:00", extra_rows=(), bars=None):
    rows = [("2021-03-10", "12:30:30", "GME", "LULD pause", "2021-03-10" if resume else "", resume or "")]
    rows += list(extra_rows)
    b = {("GME", "2021-03-10"): [("12:29", 10.0, 1), ("12:59", 9.0, 1), ("13:05", 8.5, 1)],
         ("GME", "2021-03-11"): [("09:30", 8.0, 1), ("09:45", 7.0, 1)]}
    b.update(bars or {})
    return model(tmp_path, rows, b)


def test_v0_a_candidate_not_halted_at_its_first_pass_is_unchanged(tmp_path):
    m = _gme_model(tmp_path)
    d = H.decide(m, "guard", "GME", ns("2021-03-10", "12:00:00"), 10.0, 8.0, 10.0)   # first pass 12:01:30: free
    assert d.kind == "first"


def test_v0_a_halted_candidate_enters_at_its_first_free_pass_at_the_minute_price(tmp_path):
    m = _gme_model(tmp_path)                                                         # halted 12:30:30 .. 12:52:00
    d = H.decide(m, "guard", "GME", ns("2021-03-10", "12:30:00"), 10.0, 5.0, 10.0)
    assert d.kind == "delayed" and et(d.at) == "2021-03-10 13:01:30" and d.k == 2
    assert d.price == pytest.approx(9.0) and d.event["reason"] == "luld pause"
    assert d.event["reopen"].startswith("2021-03-10T12:52") and d.event["free_min"] == pytest.approx(30.0)
    d = H.decide(m, "guard", "GME", ns("2021-03-10", "12:30:00"), 10.0, 9.0, 10.0)   # at/under the target then
    assert d.kind == "lost" and d.why == "target_reached"


def test_v0_halted_at_every_pass_is_lost_and_h1_waits_to_the_next_close(tmp_path):
    T = ns("2021-03-10", "12:30:00")                         # window 12:31:30 .. 13:41, all inside 12:30:30 .. 13:59
    m2 = _gme_model(tmp_path, resume="13:59:00")
    assert H.decide(m2, "guard", "GME", T, 10.0, 5.0, 10.0).why == "halted_expired"
    d = H.decide(m2, "wait", "GME", T, 10.0, 5.0, 10.0)
    assert d.kind == "delayed" and et(d.at) == "2021-03-10 14:01:30" and d.price == pytest.approx(8.5)
    assert H.decide(m2, "wait", "GME", T, 10.0, 8.5, 10.0).why == "wait_target"     # not above the target
    # H1b: at least half the run-up left: 8.5 >= 10 - (10 - 5) / 2 = 7.5 enters; a target of 7.5 needs 8.75
    assert H.decide(m2, "wait_half", "GME", T, 10.0, 5.0, 10.0).kind == "delayed"
    assert H.decide(m2, "wait_half", "GME", T, 10.0, 7.5, 10.0).why == "wait_half"
    assert H.decide(m2, "both", "GME", T, 10.0, 8.0, 10.0).kind == "delayed"         # H3 = H1a here
    # the 12:30:30 halt without a resume is over once the 12:59 bar has ended (13:00): the 13:01:30 pass enters
    m = _gme_model(tmp_path, resume=None, extra_rows=[("2021-03-10", "13:40:00", "GME", "LULD pause", "2021-03-11",
                                                       "09:40:00")])
    assert m.halt_at("GME", ns("2021-03-10", "12:59:59")) is not None
    assert m.halt_at("GME", ns("2021-03-10", "13:20:00")) is None
    d = H.decide(m, "guard", "GME", T, 10.0, 5.0, 10.0)
    assert d.kind == "delayed" and et(d.at) == "2021-03-10 13:01:30" and d.price == pytest.approx(9.0)
    assert d.event["missing_resume"] is True and d.event["reopen"].startswith("2021-03-10T13:00")


def test_the_fall_arm_has_no_target_and_waits_alike(tmp_path):
    m = _gme_model(tmp_path, resume="13:59:00")
    T = ns("2021-03-10", "12:30:00")
    assert H.decide(m, "wait", "GME", T, 10.0, None, 10.0).kind == "delayed"
    assert H.decide(m, "wait_half", "GME", T, 10.0, None, 10.0).kind == "delayed"


# ── the vol arm's pass loop ──────────────────────────────────────────────────

def cand(sym, T, close=10.0, target=8.0, lend=True, hi=False, tk=None, src="P", row=None):
    c = H.Cand(src=src, j=0, pos=0, row=row if row is not None else abs(hash(sym)) % 10**6, sym=sym,
               tk=tk if tk is not None else abs(hash(sym)) % 10**6, T=T, close=close, target=target, lend0=lend,
               day=H.day_of(T), hi=hi)
    return c


def run_vol(m, mode, runs, pool=True, extras=True, later=lambda c, at, px: c.lend0):
    return H.VolMachine(m, mode, pool, extras, later).run(runs)


def trades(vm):
    return sorted((t.cand.sym, et(t.at), round(t.price, 6), t.role) for t in vm.trades)


def _vol_model(tmp_path, rows):
    bars = {}
    for s in ("PICK", "B1", "B2", "B3"):                     # 11:41 reads 9.6, 12:01:30 and later 9.4
        bars[(s, "2021-03-10")] = [("11:29", 10.0, 1), ("11:40", 9.6, 1), ("12:00", 9.4, 1)]
        bars[(s, "2021-03-11")] = [("09:30", 9.2, 1)]
    return model(tmp_path, rows, bars)


T0 = ns("2021-03-10", "11:30:00")                            # the bar's end: passes 11:31:30, 11:41, 12:01:30, ...


def test_vol_a_lendable_pick_halted_first_enters_at_its_first_free_pass(tmp_path):
    m = _vol_model(tmp_path, [("2021-03-10", "11:31:00", "PICK", "LULD pause", "2021-03-10", "11:50:00")])
    vm = run_vol(m, "guard", [H.Run(cand("PICK", T0, src="L"), [])])
    assert trades(vm) == [("PICK", "2021-03-10 12:01:30", 9.4, "pick")]
    ev = [e for e in vm.events if e["role"] == "pick"][0]
    assert ev["halted_at_first"] and ev["outcome"] == "entered" and ev["at_k"] == 2


def test_vol_without_halts_the_loop_is_the_simulators_selection(tmp_path):
    m = H.NoHalts(H.Schedule(CAL, SLOTS, 75.0))
    T1 = ns("2021-03-10", "12:00:00")
    runs = [H.Run(cand("PICK", T0, src="L", lend=True), [cand("B1", T0, hi=True), cand("B2", T0)]),
            H.Run(cand("P2", T1, src="L", lend=False), [cand("B1", T1, lend=True, tk=7),     # B1: lendable, not hi
                                                        cand("B3", T1, lend=False, hi=True), cand("B2", T1, hi=True)])]
    vm = run_vol(m, "guard", runs)
    # run 1: the lendable pick + its HIGH backup B1; run 2: unlendable -> the first lendable backup B1 (another name
    # id: not traded today), the HIGH B3 unlendable, the HIGH B2 an extra
    assert [(t.cand.sym, t.role) for t in vm.trades] == [("PICK", "pick"), ("B1", "extra"), ("B1", "fallback"),
                                                          ("B2", "extra")]
    assert all(t.k == 0 and t.at == t.cand.T for t in vm.trades) and not vm.events


def test_vol_the_fallback_skips_a_halted_backup_and_never_returns_to_it(tmp_path):
    m = _vol_model(tmp_path, [("2021-03-10", "11:00:00", "B1", "LULD pause", "2021-03-10", "11:35:00")])
    vm = run_vol(m, "guard", [H.Run(cand("PICK", T0, src="L", lend=False), [cand("B1", T0), cand("B2", T0)])])
    assert trades(vm) == [("B2", "2021-03-10 11:30:00", 10.0, "fallback")]
    assert [e["outcome"] for e in vm.events] == ["not_retried"]


def test_vol_a_halt_never_triggers_the_fallback_in_v0_but_it_runs_once_the_pick_trades(tmp_path):
    rows = [("2021-03-10", "11:31:00", "PICK", "LULD pause", "2021-03-10", "11:45:00")]
    m = _vol_model(tmp_path, rows)
    vm = run_vol(m, "guard", [H.Run(cand("PICK", T0, src="L", lend=False), [cand("B1", T0)])])
    # halted at 11:31:30 and 11:41: nothing; free at 12:01:30, unlendable -> the fallback there, at the minute price
    assert trades(vm) == [("B1", "2021-03-10 12:01:30", 9.4, "fallback")]
    # H2: the halt is a fallback trigger at the first pass: B1 at its bar's close, the pick settled
    vm = run_vol(m, "backup", [H.Run(cand("PICK", T0, src="L", lend=True), [cand("B1", T0)])])
    assert trades(vm) == [("B1", "2021-03-10 11:30:00", 10.0, "halted_fallback")]
    assert [e["outcome"] for e in vm.events if e["role"] == "pick"] == ["halted_fallback"]
    # H3: the backup at once AND the pick when it trades
    vm = run_vol(m, "both", [H.Run(cand("PICK", T0, src="L", lend=True), [cand("B1", T0)])])
    assert trades(vm) == [("B1", "2021-03-10 11:30:00", 10.0, "halted_fallback"),
                          ("PICK", "2021-03-10 12:01:30", 9.4, "pick")]
    # H2 without a qualifying backup (halted too): V0
    m2 = _vol_model(tmp_path, rows + [("2021-03-10", "11:00:00", "B1", "LULD pause", "2021-03-10", "11:50:00")])
    vm = run_vol(m2, "backup", [H.Run(cand("PICK", T0, src="L", lend=True), [cand("B1", T0)])])
    assert trades(vm) == [("PICK", "2021-03-10 12:01:30", 9.4, "pick")]


def test_vol_extras_are_retried_only_while_the_pick_is_pending(tmp_path):
    rows = [("2021-03-10", "11:00:00", "B1", "LULD pause", "2021-03-10", "11:38:00")]
    m = _vol_model(tmp_path, rows)
    vm = run_vol(m, "guard", [H.Run(cand("PICK", T0, src="L"), [cand("B1", T0, hi=True)])])
    assert trades(vm) == [("PICK", "2021-03-10 11:30:00", 10.0, "pick")]             # B1 halted: never retried
    m2 = _vol_model(tmp_path, rows + [("2021-03-10", "11:31:00", "PICK", "LULD pause", "2021-03-10", "11:55:00")])
    vm = run_vol(m2, "guard", [H.Run(cand("PICK", T0, src="L"), [cand("B1", T0, hi=True)])])
    assert trades(vm) == [("B1", "2021-03-10 11:41:00", 9.6, "extra"),               # retried while the pick waits
                          ("PICK", "2021-03-10 12:01:30", 9.4, "pick")]


def test_vol_a_delayed_pick_at_its_target_is_not_entered_and_h1_waits_alone(tmp_path):
    rows = [("2021-03-10", "11:31:00", "PICK", "LULD pause", "2021-03-10", "13:00:00")]
    m = _vol_model(tmp_path, rows)
    runs = [H.Run(cand("PICK", T0, src="L", target=9.5), [cand("B1", T0, hi=True)])]
    vm = run_vol(m, "guard", runs)
    assert trades(vm) == [("B1", "2021-03-10 11:30:00", 10.0, "extra")]
    assert [e["outcome"] for e in vm.events] == ["halted_expired"]
    vm = run_vol(m, "wait", [H.Run(cand("PICK", T0, src="L", target=9.0), [])])     # 13:01:30 at 9.4 > 9.0
    assert trades(vm) == [("PICK", "2021-03-10 13:01:30", 9.4, "wait")]
    vm = run_vol(m, "wait", [H.Run(cand("PICK", T0, src="L", target=9.5), [])])     # 9.4 <= 9.5
    assert trades(vm) == [] and [e["outcome"] for e in vm.events] == ["wait_target"]
    vm = run_vol(m, "wait_half", [H.Run(cand("PICK", T0, src="L", target=8.0), [])])  # needs >= 9.0
    assert trades(vm) == [("PICK", "2021-03-10 13:01:30", 9.4, "wait")]
    vm = run_vol(m, "wait", [H.Run(cand("PICK", T0, src="L", target=8.0), [])],
                 later=lambda c, at, px: False)                                      # not lendable at the reopen
    assert trades(vm) == [] and [e["outcome"] for e in vm.events] == ["wait_unlendable"]


# ── no look-ahead ────────────────────────────────────────────────────────────

def _random_world(rng, day="2021-03-10"):
    """Random halts (half without a resume) and sparse minute bars (some without volume) for one symbol over two
    sessions."""
    rows, bars = [], {}
    for d in (day, "2021-03-11"):
        t = pd.Timestamp(f"{d} 09:35", tz=ET)
        for _ in range(8):
            t = t + pd.Timedelta(minutes=int(rng.integers(5, 60)), seconds=int(rng.integers(0, 60)))
            r = t + pd.Timedelta(minutes=int(rng.integers(1, 40)))
            no_resume = rng.random() < 0.5
            rows.append((t.strftime("%Y-%m-%d"), t.strftime("%H:%M:%S"), "SYM", "LULD pause",
                         "" if no_resume else r.strftime("%Y-%m-%d"), "" if no_resume else r.strftime("%H:%M:%S")))
        bars[("SYM", d)] = [(f"{m // 60:02d}:{m % 60:02d}", float(rng.uniform(5, 15)), float(rng.integers(0, 3)))
                            for m in range(240, 1200) if rng.random() < 0.08]
    return rows, bars


def _after(rows, bars, P, rng, drop):
    """The same world with everything after the pass ``P`` rewritten: the halts that start after it moved (their
    resumes erased) and one more added, the resumes later than it moved to other later times, the minute bars ending
    after it redrawn (``drop``: removed)."""
    def later(span):
        t = pd.Timestamp(P + int(rng.integers(1, span)) * 10**9, unit="ns", tz="UTC").tz_convert(ET)
        return t.strftime("%Y-%m-%d"), t.strftime("%H:%M:%S")
    out_rows = []
    for a, b, s, r, c, d in rows:
        if ns(a, b) > P:
            out_rows.append((*later(3 * 3600), s, r, "", ""))
            continue
        if c and ns(c, d) > P:
            c, d = later(5 * 3600)
        out_rows.append((a, b, s, r, c, d))
    out_rows.append((*later(600), "SYM", "News pending", "", ""))
    out_bars = {}
    for k, v in bars.items():
        keep = [x for x in v if ns(k[1], f"{x[0]}:00") + 60 * 10**9 <= P]
        new = [] if drop else [(x[0], float(rng.uniform(1, 50)), float(rng.integers(0, 5))) for x in v
                               if ns(k[1], f"{x[0]}:00") + 60 * 10**9 > P]
        out_bars[k] = keep + new
    return out_rows, out_bars


def _violations(tmp_path, make=H.HaltModel, n_worlds=8, n_passes=30):
    rng = np.random.default_rng(99)
    bad = 0
    T = ns("2021-03-10", "09:40:00")
    for w in range(n_worlds):
        rows, bars = _random_world(rng)
        S = H.Schedule(CAL, SLOTS, 75.0)
        passes = S.passes(T, until=S.wait_end(T))
        for q, P in enumerate(rng.choice(passes, n_passes, replace=False)):
            P = int(P)
            rows2, bars2 = _after(rows, bars, P, rng, drop=q % 2 == 0)
            a = make(H.HaltBook(H.read_halts([halt_csv(tmp_path, rows, f"a{w}.csv")])),
                     H.MinuteBars(source=minute_source(bars)), S)
            b = make(H.HaltBook(H.read_halts([halt_csv(tmp_path, rows2, f"b{w}.csv")])),
                     H.MinuteBars(source=minute_source(bars2)), S)
            ha, hb = a.halt_at("SYM", P), b.halt_at("SYM", P)
            pa, pb = a.price("SYM", P, T, 10.0), b.price("SYM", P, T, 10.0)
            if (ha is None) != (hb is None) or (ha and (ha["start"], ha["reason"]) != (hb["start"], hb["reason"])):
                bad += 1
            if (pa is None) != (pb is None) or (pa and abs(pa[0] - pb[0]) > 1e-12):
                bad += 1
    return bad


def test_no_look_ahead_rewriting_everything_after_a_pass_changes_nothing_at_it(tmp_path):
    assert _violations(tmp_path) == 0


def test_the_look_ahead_guard_catches_a_leak(tmp_path):
    class LeakyPrice(H.HaltModel):                    # reads the bar that STARTED by the pass (not yet ended)
        def price(self, sym, t, ref_t, ref_close):
            return super().price(sym, int(t) + 60 * 10**9, ref_t, ref_close)

    class LeakyHalt(H.HaltModel):                     # a halt without a resume: over if a bar EVER traded after it
        def traded_between(self, sym, s, t):
            return super().traded_between(sym, s, int(t) + 3 * 86400 * 10**9)
    assert _violations(tmp_path, LeakyPrice) > 0
    assert _violations(tmp_path, LeakyHalt) > 0


# ── the ETF arm's delayed piece ──────────────────────────────────────────────

def _etf_stub():
    from src.backtest import account as A
    from src.backtest.strategies.etf import EtfArm

    class _CM:
        @staticmethod
        def predict(price, dv, atr, bod, year):
            return np.zeros(len(price))

    class _RF:
        @staticmethod
        def daily(tkn, days, fb):
            return np.zeros(len(days))
    arm = object.__new__(EtfArm)
    arm.cal = np.array([20000 + i for i in range(30)], np.int64)
    arm.cm, arm.rf, arm.settle = _CM(), _RF(), A.default_settle()
    return arm


def _etf_bars(closes, per=13):
    n = len(closes)
    sday = np.array([20000 + i // per for i in range(n)], np.int64)
    bod = np.array([i % per for i in range(n)], np.int64)
    ns_ = np.array([(int(d) * 86400 + 14 * 3600 + 30 * 60 * (b + 1)) * 10**9 for d, b in zip(sday, bod)], np.int64)
    c = np.asarray(closes, float)
    return {"c": c, "h": c * 1.001, "l": c * 0.999, "atr": np.full(n, 4.0), "ns": ns_, "sday": sday, "bod": bod,
            "pos": bod}


def test_the_etf_entry_piece_reproduces_the_turn_and_judges_a_delayed_entry_from_the_next_bar():
    arm = _etf_stub()
    r = types.SimpleNamespace(tkn="EEE@E", dv20=5e7, year=2022, fee_file=float("nan"), dn=20000)
    a = _etf_bars([10.0, 10.5, 10.8, 10.55, 10.3, 10.2, 10.1, 10.0, 9.8, 9.7])
    q = arm._turn_piece(r, a, 0, 0, 9, False, 10.0, 9.0, 1.0, (0.5, 5), 6.0, "etf", True)
    q2 = H.etf_entry_piece(arm, r, a, 0, 9, False, 10.0, 9.0, 1.0, (0.5, 5), 6.0, "etf", True, int(a["ns"][3]),
                           float(a["c"][3]))
    assert not H._same(q, q2) and q2["turn_k"] == 0.5
    at = int(a["ns"][4]) + 11 * 60 * 10**9                       # a pass inside the bar after the next
    q3 = H.etf_entry_piece(arm, r, a, 0, 9, False, 10.0, 9.0, 1.0, (0.5, 5), 6.0, "etf", True, at, 10.25)
    assert q3["ens"] == at and q3["e"] == pytest.approx(10.25) and q3["pt"][0] == int(a["ns"][5])
    assert q3["xns"] == int(a["ns"][9]) and q3["why"] == "time"
    assert H.etf_entry_piece(arm, r, a, 0, 9, False, 10.0, 9.0, 1.0, (0.5, 5), 6.0, "etf", True,
                             int(a["ns"][9]) + 60 * 10**9, 9.7) is None       # past the deadline bar


# ── the Mimic switch ─────────────────────────────────────────────────────────

def test_the_mimic_switch_is_productions_guard_by_default_and_refuses_nonsense():
    """AUDIT151 (2026-10-19): production's halt guard is the default; the earlier simulator (`Mimic.off()`) and the
    construction before AUDIT151 (`PRE151`) keep halts unmodelled."""
    assert M.Mimic().halts == "guard" and M.Mimic.borrow_v2().halts == "guard"
    assert M.Mimic.off().halts is None and M.Mimic(**M.PRE151).halts is None
    assert H.MODES == ("guard", "wait", "wait_half", "backup", "both")
    for mode in H.MODES:
        assert M.Mimic(halts=mode).halts == mode
    with pytest.raises(ValueError):
        M.Mimic(halts="nope")


def _stub_case():
    """A tiny synthetic live combination for `mimic.live_books`: a store of six 2023 rows, the vol / fall / ETF
    adapters' pieces, one dip order piece, the ledger's costs — everything it reads stubbed."""
    def piece(strategy, tkn, day, hhmm, e, x, out_day, out_hhmm, **kw):
        t0, t1 = ns(day, hhmm), ns(out_day, out_hhmm)
        pt = np.array([t0 + 30 * 60 * 10**9 * k for k in range(1, 4)], np.int64)
        p = {"strategy": strategy, "tkn": tkn, "ens": t0, "xns": t1, "e": float(e), "x": float(x), "pick_day": dn(day),
             "pt": pt, "pc": np.array([e * 0.99, e * 0.97, e * 0.95]), "ph": np.array([e, e * 0.99, e * 0.97]),
             "hs_in": 10.0, "hs_out": 12.0, "dv20": 2e7, "lvl": float(e) * 0.8, "apk": 3.0,
             "pa": np.array([3.0, 2.9, 2.8]), "vn": None, "cover_mult": 4.0, "why": "time", "f": 1.0,
             "d0": dn(day) + 2, "cum": np.array([1.0, 2.0, 3.0]), "bfac": 1.0 / 100.0 / 360.0, "d_out": dn(out_day)}
        p.update(kw)
        return p
    days = [d.date().isoformat() for d in pd.bdate_range("2023-03-01", "2023-03-31")]
    cal = np.array([dn(d) for d in days], np.int64)
    C = pd.DataFrame({"dn": [dn("2023-03-06")] * 3 + [dn("2023-03-07")] * 3,
                      "tkn": ["AAA", "BBB", "CCC@D", "DDD", "EEE@X", "FFF"],
                      "bod": [3, 3, 3, 5, 5, 5], "actual": [12.0, 30.0, 8.0, 15.0, 22.0, 40.0]})
    vol = [piece("vol", "BBB", "2023-03-06", "11:30", 30.0, 27.0, "2023-03-08", "10:00", R=1, xi=5),
           piece("vol", "AAA", "2023-03-06", "11:30", 12.0, 13.0, "2023-03-09", "15:00", R=0, xi=9),
           piece("vol", "EEE@X", "2023-03-07", "12:30", 22.0, 20.0, "2023-03-10", "12:00", R=4, xi=7)]
    fall = [piece("fall", "DDD", "2023-03-07", "14:00", 15.0, 14.0, "2023-03-13", "11:00", R=3, xi=12, turn_bar=2,
                  lvl=None, vn=0.5, cover_mult=6.0)]
    etf = [piece("etf", "QQQX@E", "2023-03-06", "13:00", 25.0, 24.0, "2023-03-09", "10:30", pick_px=26.0)]
    dip = [{"strategy": "dip", "tkn": "MEGA", "ens": ns("2023-03-06", "10:00"), "xns": ns("2023-03-10", "09:30"),
            "e": 100.0, "x": 103.0, "pick_day": dn("2023-03-03"), "pt": np.array([ns("2023-03-06", "16:00")], np.int64),
            "pc": np.array([101.0]), "hs_in": 1.0, "hs_out": 1.0, "dv20": 2e9, "ord_k": 0}]

    class Eng:
        def __init__(self):
            self.st = types.SimpleNamespace(C=C)
            self.C4 = types.SimpleNamespace(CAL=cal)
            self._fee_pred, self._fee_pred_loaded, self.fee_fallback = np.zeros(len(C)), "v4", None
            self.calls = []

        def lend_mask(self, params=None, at="next"):
            return np.ones(len(C), bool)

        def pieces(self, params, lo, hi, strategy="vol", lend_all=None, extra=None, ctx=False):
            self.calls.append(("pieces", strategy, dict(extra or {}), int(np.sum(lend_all))))
            return [dict(p) for p in vol]

        def turn_pieces(self, params, lo, hi, k=0.5, W=5, strategy="vol", lend_all=None, fall=False, ctx=True,
                        lend_at=None):
            self.calls.append(("turn", strategy, float(k), int(W), bool(fall), lend_at is not None))
            return [dict(p) for p in fall_]
    fall_ = fall

    class Etf:
        def __init__(self):
            self.calls = []

        def pieces(self, lo, hi, **kw):
            self.calls.append(sorted(kw))
            return [dict(p) for p in etf]

    class Costs:
        def apply(self, p, side, legs=True, adv=None):
            q = dict(p)
            if legs:
                q["hs_in"], q["hs_out"] = 5.0 + side, 6.0 - side
            q["hs_mark"] = 7.0 + side
            return q
    return Eng(), Etf(), Costs(), np.full(len(C), 0.5), dip


def _fields(books):
    return {k: [(str(p["tkn"]), int(p["ens"]), int(p["xns"]), round(float(p["e"]), 9), round(float(p["x"]), 9),
                 None if p.get("hm") is None else round(float(p["hm"]), 9),
                 None if p.get("hi") is None else round(float(p["hi"]), 9),
                 None if p.get("rank") is None else float(p["rank"]),
                 round(float(p["hs_in"]), 9), round(float(p["hs_out"]), 9),
                 None if p.get("hs_mark") is None else round(float(p["hs_mark"]), 9)) for p in books[k]]
            for k in sorted(books)}


def _live_books(monkeypatch, mim):
    monkeypatch.setattr(settings, "enable_crash_long", False, raising=False)
    monkeypatch.setattr(M, "house_mix", lambda: (("X", 2.0, 2.86), ("Y", 41.1, 45.2), ("Z", 3.5, 4.0)))
    eng, etf, costs, p_lend, dip = _stub_case()
    bars = M.ExtBars()
    for t in ("AAA", "BBB", "CCC@D", "DDD", "EEE@X", "FFF", "QQQX@E", "MEGA"):
        bars._m[t] = None
    log = {}
    books = M.live_books("2023-03-01", "2023-03-31", mim, eng=eng, etf=etf, panel=None, bars=bars, costs=costs,
                         p_lend=p_lend, settle=lambda d: int(d) + 2, log=log, dip=dip)
    return books, log, eng, etf


# `live_books` on `_stub_case` under `Mimic(seed=3)`, captured from the simulator BEFORE PREREG99's change
BEFORE = {
    "dip": [("MEGA", 1678114800000000000, 1678458600000000000, 100.0, 103.0, None, None, None, 1.0, 1.0, 8.0)],
    "etf": [("QQQX@E", 1678125600000000000, 1678375800000000000, 25.0, 24.0, None, None, 42.0, 4.0, 7.0, 6.0)],
    "fall": [("DDD", 1678215600000000000, 1678719600000000000, 15.0, 14.0, 2.0, 2.86, 25.0, 4.0, 7.0, 6.0)],
    "vol": [("BBB", 1678120200000000000, 1678287600000000000, 30.0, 27.0, 2.0, 2.86, 11.0, 4.0, 7.0, 6.0),
            ("AAA", 1678120200000000000, 1678392000000000000, 12.0, 13.0, 2.0, 2.86, 1.0, 4.0, 7.0, 6.0),
            ("EEE@X", 1678210200000000000, 1678467600000000000, 22.0, 20.0, 2.0, 2.86, 31.0, 4.0, 7.0, 6.0)],
}
BEFORE_LOG = {"ext_dip_no_bars": 1, "ext_etf_no_bars": 1, "ext_fall_no_bars": 1, "ext_vol_no_bars": 3}


def test_mimic_without_halts_builds_exactly_what_it_built_before(monkeypatch):
    class Boom:
        def __init__(self, *a, **k):
            raise AssertionError("the halt model must not run with halts=None")
    monkeypatch.setattr(H, "Applier", Boom)
    books, log, eng, etf = _live_books(monkeypatch, M.Mimic(seed=3, **M.PRE151))
    assert _fields(books) == BEFORE and log == BEFORE_LOG
    assert eng.calls == [("turn", "fall", 0.5, 5, True, True),
                         ("pieces", "vol", {"entry_at_pick": True, "rule201": False}, 6)]
    # PREREG102: the default follows production's VIX block (`sel_short_etf_exclude_vix`, from 2026-10-14) — the ETF
    # arm's picks re-ranked without the VIX-linked products (``exclude``)
    base = ["cagap", "ctx", "entry", "give", "hmax", "lend_after", "limit", "through", "turn"]   # hmax: 2026-10-20
    base += ["cover", "max_dtc", "min_rvol"]           # PREREG227's S7: production's numbers, the adapter's defaults
    assert etf.calls == [sorted(base + ([] if M.Mimic(seed=3, **M.PRE151).etf_vix else ["exclude"]))]


def test_mimic_with_halts_routes_the_etf_entries_and_the_vol_and_fall_books_through_the_model(monkeypatch):
    seen = {}

    class Fake:
        def __init__(self, mim, eng, L):
            seen["init"] = (mim.halts, len(L))

        def etf_pieces(self, etf):
            def call(lo, hi, **kw):
                seen["etf"] = sorted(kw)
                return etf.pieces(lo, hi, **kw)
            return call

        def apply(self, books, lo, hi, extra, log):
            seen["apply"] = (sorted(books), lo, hi, dict(extra))
            books["vol"] = books["vol"][:1]
            log["halts"] = {"mode": "guard"}
    monkeypatch.setattr(H, "Applier", Fake)
    books, log, _, _ = _live_books(monkeypatch, M.Mimic(seed=3, **dict(M.PRE151, halts="guard")))
    assert seen["init"] == ("guard", 6) and seen["etf"][0] == "cagap"
    assert seen["apply"] == (["dip", "etf", "fall", "vol"], "2023-03-01", "2023-03-31",
                             {"entry_at_pick": True, "rule201": False})
    assert [p["tkn"] for p in books["vol"]] == ["BBB"] and log["halts"]["mode"] == "guard"
    assert _fields(books)["vol"][0][:5] == BEFORE["vol"][0][:5]                       # the post-processing still runs


def test_the_pressure_extras_need_v6_installed():
    """The plain evaluator (the reliability cards' construction) adds no pressure extras even when the engine carries
    pressure flags (set A0's ``pressure_hi``); V6's ``Real6`` evaluator (a study harness's `R75.Applied`) does — the
    halt model's pass loop follows (2026-10-19: the A0 cards failed its own selection check without it)."""
    import types

    class Real:
        pass

    class Real6(Real):
        pass
    eng = types.SimpleNamespace(Real=Real, pressure_hi=np.ones(3, bool))
    assert H.pressure_flags(eng) is not None and not H.v6_installed(eng)
    eng.Real = Real6
    assert H.v6_installed(eng)
