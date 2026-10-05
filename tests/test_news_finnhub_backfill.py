"""The one-year news backfill (`src/analysis/news_finnhub_backfill.py`, 2026-09-25).

What must hold: the plan is the model arrays' tradeable universe, ranked by
liquidity; the pull goes oldest week first, re-asks a capped week per day, and
never runs beside the live refresher or a live fetch; an answer serves only the
instants it was fetched after; the scorer waits for the pull's frontier (the
events source never does), resumes by run id, and writes a (day, band) whole or
not at all; and every event leg reads only what was knowable at 08:30 ET.
"""
from __future__ import annotations

import json
from datetime import date, datetime, timedelta, timezone
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import pytest

from config.settings import settings
from src.analysis import news_finnhub_backfill as fb

ET = ZoneInfo("America/New_York")


@pytest.fixture
def root(tmp_path, monkeypatch):
    monkeypatch.setattr(fb, "ROOT", tmp_path / "fb")
    monkeypatch.setattr(fb, "refresher_active", lambda: False)
    monkeypatch.setattr(fb, "live_phase", lambda: None)
    return tmp_path / "fb"


class _NoGate:
    def wait(self):
        return None


# ── the window and the gate ──────────────────────────────────────────────────

@pytest.mark.parametrize("when,expected", [
    (datetime(2026, 9, 25, 20, 10, tzinfo=ET), False),   # Friday evening: not yet
    (datetime(2026, 9, 25, 23, 59, tzinfo=ET), False),
    (datetime(2026, 9, 26, 0, 59, tzinfo=ET), False),    # Saturday, before 01:00
    (datetime(2026, 9, 26, 1, 0, tzinfo=ET), True),
    (datetime(2026, 9, 26, 15, 0, tzinfo=ET), True),     # Saturday afternoon
    (datetime(2026, 9, 27, 19, 54, tzinfo=ET), True),    # Sunday, before the refresher wakes
    (datetime(2026, 9, 27, 19, 55, tzinfo=ET), False),
    (datetime(2026, 9, 28, 12, 0, tzinfo=ET), False),    # Monday
])
def test_the_weekend_window(when, expected):
    assert fb.in_window(when, "weekend") is expected
    assert fb.in_window(when, "any") is True


def test_the_acquire_gate_waits_for_the_refresher_and_a_live_fetch(monkeypatch):
    states = iter([True, True, False, False])
    phases = iter(["fetch", None])
    monkeypatch.setattr(fb, "refresher_active", lambda: next(states))
    monkeypatch.setattr(fb, "live_phase", lambda: next(phases))
    slept = []
    g = fb.Gate("acquire", "any", sleep=slept.append)
    g.wait()
    assert slept == [60, 60, 15]


def test_the_score_gate_waits_only_for_live_sentiment(monkeypatch):
    phases = iter(["fetch", "sentiment", None])
    monkeypatch.setattr(fb, "refresher_active", lambda: True)   # irrelevant to the LLM
    monkeypatch.setattr(fb, "live_phase", lambda: next(phases))
    slept = []
    g = fb.Gate("score", "any", sleep=slept.append)
    g.wait()                                                    # a live FETCH does not hold the LLM
    assert slept == []
    g.wait()
    assert slept == [15]


def test_a_live_pause_is_logged_once_with_its_length(monkeypatch):
    from loguru import logger
    phases = iter(["sentiment", "sentiment", "sentiment", None])
    monkeypatch.setattr(fb, "live_phase", lambda: next(phases))
    lines = []
    sink = logger.add(lambda m: lines.append(str(m)), level="INFO",
                      filter=lambda r: "[finnhub-backfill]" in r["message"])
    try:
        fb.Gate("score", "any", sleep=lambda s: None).wait()
    finally:
        logger.remove(sink)
    assert sum("paused" in x for x in lines) == 1 and sum("resumed after" in x for x in lines) == 1


def test_the_gate_sleeps_outside_the_window_and_stops_at_the_deadline(monkeypatch):
    monkeypatch.setattr(fb, "live_phase", lambda: None)
    clock = iter([datetime(2026, 9, 28, 9, 0, tzinfo=ET), datetime(2026, 9, 26, 9, 0, tzinfo=ET)])
    slept = []
    fb.Gate("score", "weekend", sleep=slept.append, now=lambda: next(clock)).wait()
    assert slept == [600]
    late = datetime(2026, 9, 26, 9, 0, tzinfo=ET)
    with pytest.raises(fb.Stop):
        fb.Gate("score", "any", until=late - timedelta(minutes=1), now=lambda: late).wait()


def _line(ts: str, msg: str) -> str:
    return f"2026-09-25 {ts}.000 | INFO     | x:y:1 - {msg}\n"


def test_the_live_phase_is_read_incrementally_past_a_large_tick(tmp_path):
    """An RTH tick logs more than news_replay's fixed 3 MB tail: the start line
    must still be found, and later calls read only the appended bytes."""
    log = tmp_path / "llm_trader_2026-09-25.log"
    now = datetime(2026, 9, 25, 12, 10)
    with open(log, "w", encoding="utf-8") as fh:
        fh.write(_line("11:50:23", "[db] Persisted run 2026-09-25_153023: 10 rec"))
        fh.write(_line("12:00:01", "[scheduler] tick for 12:00 ET (rth, email=True, 1s after slot)"))
        fh.write(_line("12:02:09", "Steps 1–3: 3052 total articles assembled"))
        fh.write(_line("12:03:16", "Signal weights [NEUTRAL] — news=9%"))
        filler = _line("12:05:00", "DEBUG chatter " + "x" * 200)
        fh.write(filler * (6_000_000 // len(filler)))           # 6 MB of DEBUG lines
    ph = fb.LivePhase(tmp_path)
    assert ph(now=now) == "sentiment"
    with open(log, "a", encoding="utf-8") as fh:
        fh.write(_line("12:10:06", "[aggregator] rank pool: 372/405 tradeable names"))
    assert ph(now=now) == "post"
    assert ph.pos == log.stat().st_size                         # nothing re-read
    with open(log, "a", encoding="utf-8") as fh:
        fh.write(_line("12:16:00", "[db] Persisted run 2026-09-25_160001: 10 rec"))
    assert ph(now=now) is None
    with open(log, "a", encoding="utf-8") as fh:
        fh.write(_line("12:30:02", "[scheduler] tick for 12:30 ET (rth, email=True, 2s after slot)"))
        fh.write("2026-09-25 12:31:00.000 | INFO | partial line with no newline yet")
    assert ph(now=now.replace(minute=31)) == "fetch"
    # a tick that never persisted (the watchdog killed it) stops counting after 50 min
    assert ph(now=datetime(2026, 9, 25, 13, 25)) is None


def test_the_live_phase_follows_the_newest_log_and_caches(tmp_path):
    old = tmp_path / "llm_trader_2026-09-24.log"
    old.write_text(_line("23:30:00", "[scheduler] tick for 23:30 ET (overnight)"), encoding="utf-8")
    ph = fb.LivePhase(tmp_path, ttl=60)
    assert ph(now=datetime(2026, 9, 24, 23, 31)) == "fetch"
    import os
    import time as _t
    new = tmp_path / "llm_trader_2026-09-25.log"
    new.write_text(_line("00:10:00", "[db] Persisted run x"), encoding="utf-8")
    later = _t.time() + 10
    os.utime(new, (later, later))
    assert ph() == "fetch"                                      # cached for the TTL
    assert ph(now=datetime(2026, 9, 25, 0, 11)) is None         # an explicit read: the new file
    assert fb.LivePhase(tmp_path / "none")(now=datetime(2026, 9, 25)) is None


# ── the plan ─────────────────────────────────────────────────────────────────

def _arrays(tmp_path, rows):
    """rows: (day, ticker, price, dv20)."""
    names = sorted({r[1] for r in rows})
    code = {t: i for i, t in enumerate(names)}
    d = tmp_path / "sel30"
    d.mkdir()
    (d / "meta.json").write_text(json.dumps({"tickers": names}), encoding="utf-8")
    np.save(d / "dn.npy", np.array([np.datetime64(r[0], "D").astype(np.int64) for r in rows]))
    np.save(d / "tk.npy", np.array([code[r[1]] for r in rows], dtype=np.int32))
    np.save(d / "px.npy", np.array([r[2] for r in rows], dtype=np.float32))
    np.save(d / "dv20.npy", np.array([r[3] for r in rows], dtype=np.float32))
    return d


def test_the_plan_is_the_tradeable_universe_ranked_by_liquidity(tmp_path):
    rows = [("2025-10-06", "BIG", 50, 9e8), ("2025-10-06", "MID", 20, 5e7),
            ("2025-10-06", "PENNY", 2, 9e8),                    # under the $5 floor
            ("2025-10-06", "THIN", 30, 1e6),                    # under $5M
            ("2025-10-07", "BIG", 50, 9e8), ("2025-10-07", "MID", 20, 5e7),
            ("2025-10-07", "THIN", 30, 6e6),                    # tradeable this day only
            ("2025-10-07", "MID", 20, 5e7),                     # a second bar that day
            ("2025-09-30", "OLD", 50, 9e8)]                     # before the start
    plan = fb.build_plan(start=date(2025, 10, 6), arrays_dir=_arrays(tmp_path, rows), band_size=2)
    assert plan["days"] == ["2025-10-06", "2025-10-07"]
    assert plan["universe"]["2025-10-06"] == ["BIG", "MID"]
    assert plan["universe"]["2025-10-07"] == ["BIG", "MID", "THIN"]
    assert plan["rank"] == {"BIG": 0, "MID": 1, "THIN": 2}
    assert [fb.band_of(plan, t) for t in ("BIG", "MID", "THIN")] == [0, 0, 1]


def test_a_thin_tail_takes_the_last_full_universe():
    uni = {"2026-09-10": ["A", "B", "C", "D", "E"], "2026-09-11": ["A", "B", "C", "D"],
           "2026-09-14": ["A", "F"], "2026-09-15": ["G"]}
    added = fb.fill_thin_days(uni)
    assert uni["2026-09-11"] == ["A", "B", "C", "D"]            # 4/5 = 0.8: a full day
    assert uni["2026-09-14"] == ["A", "B", "C", "D", "F"]       # merged with 09-11, not 09-10
    assert uni["2026-09-15"] == ["A", "B", "C", "D", "G"]
    assert added == {"2026-09-14": 3, "2026-09-15": 4}


def test_units_go_oldest_week_first_most_liquid_first():
    plan = {"universe": {"2025-10-13": ["A", "B"], "2025-10-06": ["B", "A"], "2025-10-07": ["A"]},
            "rank": {"A": 1, "B": 0}}
    us = fb.units(plan)
    assert [(str(wk), tk) for wk, tk, _ in us] == [("2025-10-06", "B"), ("2025-10-06", "A"),
                                                   ("2025-10-13", "B"), ("2025-10-13", "A")]
    assert us[1][2] == [date(2025, 10, 6), date(2025, 10, 7)]


# ── raw answers ──────────────────────────────────────────────────────────────

def _meta(n, fetched_at=None, items=None):
    return {"from": "x", "to": "y", "n": n, "items": items or [],
            "fetched_at": (fetched_at or datetime.now(timezone.utc)).isoformat()}


def test_an_answer_serves_only_instants_it_was_fetched_after(root):
    d, wk = date(2025, 10, 7), date(2025, 10, 6)
    assert fb.items_for("AAA", d) is None                      # not acquired
    assert fb.unit_tasks("AAA", wk, [d]) == [("week", "AAA", wk)]
    early = fb.cutoff(d) + timedelta(minutes=30)               # inside the 1 h margin
    fb.write_raw(fb.week_path("AAA", wk), _meta(3, early, [{"headline": "h"}]))
    assert fb.items_for("AAA", d) is None
    assert fb.unit_tasks("AAA", wk, [d]) == [("week", "AAA", wk)]
    fb.write_raw(fb.week_path("AAA", wk), _meta(3, items=[{"headline": "h"}]))
    assert fb.items_for("AAA", d) == [{"headline": "h"}]
    assert fb.unit_tasks("AAA", wk, [d]) == []


def test_a_capped_week_is_answered_by_the_day_requests(root):
    d1, d2, wk = date(2025, 10, 6), date(2025, 10, 7), date(2025, 10, 6)
    fb.write_raw(fb.week_path("AAA", wk), _meta(fb.CAP, items=[{"headline": "w"}]))
    assert fb.unit_tasks("AAA", wk, [d1, d2]) == [("day", "AAA", d1), ("day", "AAA", d2)]
    assert fb.items_for("AAA", d1) is None                     # the truncated week never serves
    fb.write_raw(fb.day_path("AAA", d1), _meta(40, items=[{"headline": "d"}]))
    assert fb.items_for("AAA", d1) == [{"headline": "d"}]
    assert fb.unit_tasks("AAA", wk, [d1, d2]) == [("day", "AAA", d2)]


def test_trim_keeps_what_the_rule_reads_and_cuts_summaries_like_live():
    out = fb.trim([{"datetime": 1, "headline": "h", "url": "u", "source": "s", "summary": "x" * 3000,
                    "image": "big", "related": "AAA", "id": 7}])
    assert out == [{"datetime": 1, "headline": "h", "url": "u", "source": "s", "summary": "x" * 1000}]


# ── the pull ─────────────────────────────────────────────────────────────────

def _plan(days_by_ticker, rank=None):
    uni = {}
    for tk, days in days_by_ticker.items():
        for d in days:
            uni.setdefault(d, []).append(tk)
    rank = rank or {tk: i for i, tk in enumerate(days_by_ticker)}
    return {"universe": uni, "rank": rank, "days": sorted(uni), "band_size": 500,
            "start": min(uni), "end": max(uni)}


def test_acquire_pulls_every_unit_reasks_capped_weeks_and_records_the_frontier(root, monkeypatch):
    monkeypatch.setattr(settings, "finnhub_api_key", "k")
    plan = _plan({"AAA": ["2025-10-06", "2025-10-07"], "BBB": ["2025-10-14"]})
    calls = []

    def fetch(tk, frm, to):
        calls.append((tk, frm, to))
        if (frm, to) == fb.week_window(date(2025, 10, 6)) and tk == "AAA":
            return 200, [{"datetime": 1, "headline": str(i)} for i in range(fb.CAP + 5)], 50, None
        return 200, [{"datetime": 1, "headline": "x"}], 50, None
    slept = []
    out = fb.acquire(plan, window="any", fetch=fetch, sleep=slept.append, gate=_NoGate())
    assert out["ok"] == 4 and out["errors"] == {}
    assert calls == [("AAA", date(2025, 10, 3), date(2025, 10, 12)),     # the week, capped
                     ("AAA", date(2025, 10, 3), date(2025, 10, 6)),      # ... so live's own requests
                     ("AAA", date(2025, 10, 4), date(2025, 10, 7)),
                     ("BBB", date(2025, 10, 10), date(2025, 10, 19))]
    assert fb.read_progress()["pull_complete"] is True
    assert fb.items_for("AAA", date(2025, 10, 7)) == [{"datetime": 1, "headline": "x",
                                                       "url": None, "source": None, "summary": ""}]
    # resumable: a second pass asks nothing
    calls.clear()
    fb.acquire(plan, window="any", fetch=fetch, sleep=slept.append, gate=_NoGate())
    assert calls == []


def test_acquire_waits_out_a_429_and_the_remaining_floor(root, monkeypatch):
    monkeypatch.setattr(settings, "finnhub_api_key", "k")
    plan = _plan({"AAA": ["2025-10-06"]})
    answers = iter([(429, [], 0, None), (200, [], 3, None)])
    slept = []
    out = fb.acquire(plan, window="any", fetch=lambda *a: next(answers), sleep=slept.append,
                     gate=_NoGate())
    assert out["ok"] == 1
    assert 65 in slept                                          # the 429
    assert any(1.0 <= s <= 61.0 for s in slept if s != 65)      # remaining 3 < the floor


def test_acquire_stops_at_max_requests_and_records_errors(root, monkeypatch):
    monkeypatch.setattr(settings, "finnhub_api_key", "k")
    plan = _plan({"AAA": ["2025-10-06"], "BBB": ["2025-10-06"], "CCC": ["2025-10-06"]})
    out = fb.acquire(plan, window="any", fetch=lambda *a: (403, [], 50, None), sleep=lambda s: None,
                     gate=_NoGate(), max_requests=2)
    assert out["stopped"] == "max_requests"
    assert out["errors"] == {403: 2}          # every answer is recorded before the stop
    assert fb.read_progress() == {}           # no week completed


def test_the_last_answer_before_a_stop_is_kept(root, monkeypatch):
    monkeypatch.setattr(settings, "finnhub_api_key", "k")
    plan = _plan({"AAA": ["2025-10-06"], "BBB": ["2025-10-06"]})
    out = fb.acquire(plan, window="any", fetch=lambda *a: (200, [{"headline": "x"}], 50, None),
                     sleep=lambda s: None, gate=_NoGate(), max_requests=1)
    assert out["stopped"] == "max_requests" and out["ok"] == 1
    assert fb.items_for("AAA", date(2025, 10, 6)) is not None


def test_acquire_needs_a_key(root, monkeypatch):
    monkeypatch.setattr(settings, "finnhub_api_key", "")
    assert "skipped" in fb.acquire(_plan({"AAA": ["2025-10-06"]}), gate=_NoGate())


# ── the events source ────────────────────────────────────────────────────────

def _write_parquet(path, df):
    import duckdb
    con = duckdb.connect()
    try:
        con.register("df_", df)
        con.execute(f"COPY (SELECT * FROM df_) TO '{path.as_posix()}' (FORMAT PARQUET)")
    finally:
        con.close()


@pytest.fixture
def deep(tmp_path):
    """A deep store with the REAL column names and types of each family."""
    d = tmp_path / "deep"
    d.mkdir()
    ts = pd.Timestamp
    _write_parquet(d / "yf_analyst.parquet", pd.DataFrame({
        "ticker": ["AAA", "AAA", "AAA"],
        "grade_date": [ts("2025-10-01 14:00"), ts("2025-10-07 10:00"), ts("2025-08-01 14:00")],
        "firm": ["F1", "F2", "F3"], "to_grade": ["Buy", "Sell", "Buy"],
        "from_grade": ["Hold", "Hold", "Hold"], "action": ["up", "down", "up"],
        "pt_action": ["Raises", "Lowers", "Raises"], "pt_current": [120.0, 80.0, 110.0],
        "pt_prior": [100.0, 100.0, 100.0]}))
    _write_parquet(d / "yf_earnings.parquet", pd.DataFrame({
        "ticker": ["AAA", "AAA", "BBB"],
        "event_ts": [ts("2025-07-30 20:00"), ts("2025-10-06 20:00"), ts("2025-10-07 12:00")],
        "event_et": ["2025-07-30 16:00", "2025-10-06 16:00", "2025-10-07 08:00"],
        "eps_estimate": [1.0, 1.0, 1.0], "eps_reported": [1.5, 1.2, 0.5],
        "surprise_pct": [50.0, 20.0, -50.0]}))
    _write_parquet(d / "short_interest.parquet", pd.DataFrame({
        "settlement_date": ["2025-09-12", "2025-09-15", "2025-09-30"],
        "ticker": ["AAA"] * 3, "short_interest": [100, 100, 200],
        "avg_daily_volume": [50, 50, 50], "days_to_cover": [20.0, 20.0, 20.0]}))
    _write_parquet(d / "short_volume.parquet", pd.DataFrame({
        "ticker": ["AAA", "AAA"], "date": ["2025-10-06", "2025-10-07"],
        "total_volume": [10.0, 10.0], "short_volume": [6.0, 9.0], "exempt_volume": [0.0, 0.0],
        "non_exempt_volume": [6.0, 9.0], "short_volume_ratio": [60.0, 90.0]}))
    _write_parquet(d / "yf_shares.parquet", pd.DataFrame({
        "ticker": ["AAA", "AAA"], "date": ["2025-01-01", "2025-11-01"], "shares": [1000, 10]}))
    days = pd.bdate_range("2025-09-01", "2025-10-07")
    dpi = [0.40] * (len(days) - 1) + [1.00]                    # the shift lands on 10-07
    _write_parquet(d / "quiver_dpi.parquet", pd.DataFrame({
        "ticker": ["AAA"] * len(days), "Date": [x.date().isoformat() for x in days],
        "OTC_Short": [1] * len(days), "OTC_Total": [2] * len(days), "DPI": dpi}))
    _write_parquet(d / "sec_filings.parquet", pd.DataFrame({
        "ticker": ["AAA", "AAA"], "cik": ["1", "1"], "accession": ["a1", "a2"],
        "filing_date": ["2025-10-06", "2025-10-07"], "report_date": ["", ""],
        "acceptance": [ts("2025-10-06 21:00"), ts("2025-10-07 13:00")], "act": ["", ""],
        "form": ["8-K", "8-K"], "file_number": ["", ""], "items": ["2.02", "2.02"],
        "size": [1.0, 1.0], "is_xbrl": [0, 0], "is_inline_xbrl": [0, 0],
        "primary_doc": ["x.htm", "y.htm"]}))
    return d


def test_event_tables_load_the_plan_names_and_window(deep):
    t = fb.EventTables(["aaa", "ZZZ"], date(2025, 10, 6), date(2025, 10, 7), deep_dir=deep)
    assert set(t.by["analyst"]) == {"AAA"}
    assert t.rows("analyst", "aaa") is not None and t.rows("analyst", "ZZZ") is None
    assert t.rows("earnings", "BBB") is None                   # not a plan name


def test_the_analyst_leg_reads_the_30_days_before_the_session(deep):
    t = fb.EventTables(["AAA"], date(2025, 10, 6), date(2025, 10, 8), deep_dir=deep)
    a = fb._analyst_article(t, "AAA", date(2025, 10, 7))
    assert a is not None and "1 upgrade(s)" in a.title        # 10-07 10:00 is the session itself
    assert a.published_at == fb.first_tick(date(2025, 10, 7))
    a8 = fb._analyst_article(t, "AAA", date(2025, 10, 8))
    assert "1 upgrade(s) and 1 downgrade(s)" in a8.title       # visible once the day is over


def test_the_eps_leg_reads_the_latest_release_out_by_the_first_tick(deep):
    t = fb.EventTables(["AAA"], date(2025, 10, 6), date(2025, 10, 8), deep_dir=deep)
    a6 = fb._eps_article(t, "AAA", date(2025, 10, 6))          # the 10-06 release is after the close
    assert a6 is not None and "50.0%" in a6.title
    a7 = fb._eps_article(t, "AAA", date(2025, 10, 7))
    assert "20.0%" in a7.title and a7.published_at == fb.first_tick(date(2025, 10, 7))


def test_the_short_leg_uses_published_settlements_and_live_thresholds(deep):
    from src.data import short_interest as si
    t = fb.EventTables(["AAA"], date(2025, 10, 1), date(2025, 10, 20), deep_dir=deep)
    # 09-30 is published 10 business days later (10-14): on 10-14 only 09-12/09-15 are known
    assert fb._short_article(t, "AAA", date(2025, 10, 14)) is None      # 100/1000 = 10% < 15%
    a = fb._short_article(t, "AAA", date(2025, 10, 15))             # 200/1000 = 20%, +100% vs 09-12
    assert a is not None and 200 / 1000 >= si._MIN_SHORT_PCT
    assert "rises" in a.title and "90%" in a.summary                # FINRA ratio for 10-07 (< D)
    assert a.published_at == fb.first_tick(date(2025, 10, 15))


def test_the_dark_pool_leg_reads_rows_two_sessions_old(deep):
    t = fb.EventTables(["AAA"], date(2025, 10, 1), date(2025, 10, 9), deep_dir=deep)
    # the 10-07 bar that makes the shift is visible only from 10-09 (D-2)
    assert fb._dpi_articles(t, ["AAA"], date(2025, 10, 8)).get("AAA", []) == []
    arts = fb._dpi_articles(t, ["AAA"], date(2025, 10, 9))["AAA"]
    assert len(arts) == 1 and "accumulation" in arts[0].title


@pytest.fixture
def named(monkeypatch):
    from src.data import company_names
    monkeypatch.setattr(company_names, "company_name", lambda tk: "Alpha Corp")


def test_the_8k_leg_reads_filings_accepted_by_0830_and_restores_the_frame(deep, named, monkeypatch):
    from src.analysis import news_history as nh
    sentinel = object()
    monkeypatch.setattr(nh, "_SEC_8K", sentinel)
    t = fb.EventTables(["AAA"], date(2025, 10, 6), date(2025, 10, 8), deep_dir=deep)
    a7 = fb._eight_k_articles(t, "AAA", date(2025, 10, 7))
    assert len(a7) == 1                                         # 10-07 13:00 UTC is after 08:30 ET
    assert len(fb._eight_k_articles(t, "AAA", date(2025, 10, 8))) == 2
    assert nh._SEC_8K is sentinel


def test_events_pools_merge_every_leg_per_name(deep, named):
    t = fb.EventTables(["AAA", "CCC"], date(2025, 10, 1), date(2025, 10, 15), deep_dir=deep)
    pools = fb.events_pools(t, ["AAA", "CCC"], date(2025, 10, 15))
    assert pools["CCC"] == []
    # live's merge order; the two 8-Ks are past the 5-day look-back by 10-15
    assert [a.source for a in pools["AAA"]] == ["Analyst Ratings", "Earnings/EPS", "Short Interest",
                                                "Quiver Dark Pool"]
    with_8k = fb.events_pools(t, ["AAA"], date(2025, 10, 8))["AAA"]
    assert "SEC 8-K" in with_8k[0].title and len([a for a in with_8k if "8-K" in a.title]) == 2


# ── scoring ──────────────────────────────────────────────────────────────────

def test_the_verdict_cache_keys_on_what_the_model_reads(monkeypatch):
    """An EPS beat re-stamped at each day's first tick reads "8h ago" at every
    08:30: one question, asked once."""
    from src.analysis import sentiment
    from src.models import NewsArticle
    live_key = sentiment._sentiment_cache_key
    monkeypatch.setattr(fb, "_LIVE_KEY", live_key)
    d1, d2 = date(2025, 10, 7), date(2025, 10, 8)

    def art(d, summary="s"):
        return NewsArticle(title="AAA beat EPS", summary=summary, url="u", source="Earnings/EPS",
                           published_at=fb.first_tick(d))
    monkeypatch.setattr(fb, "_KEY_AS_OF", fb.cutoff(d1))
    k1 = fb._prompt_key("AAA", "local", [art(d1)], "hdr")
    monkeypatch.setattr(fb, "_KEY_AS_OF", fb.cutoff(d2))
    k2 = fb._prompt_key("AAA", "local", [art(d2)], "hdr")
    assert k1 == k2
    assert live_key("AAA", "local", [art(d1)], "hdr") != live_key("AAA", "local", [art(d2)], "hdr")
    assert fb._prompt_key("AAA", "local", [art(d2, "other")], "hdr") != k2
    assert fb._prompt_key("AAA", "local", [art(d1)], "hdr") != k2      # a day older reads "32h ago"
    assert fb._prompt_key("BBB", "local", [art(d2)], "hdr") != k2
    assert fb._prompt_key("AAA", "local", [art(d2)], "other header") != k2
    monkeypatch.setattr(fb, "_KEY_AS_OF", None)
    assert fb._prompt_key("AAA", "local", [art(d2)], "hdr") == live_key("AAA", "local", [art(d2)], "hdr")


def test_the_prompt_key_is_installed_once_in_the_backfill_process(monkeypatch):
    from src.analysis import news_replay as nr
    from src.analysis import sentiment
    live_key = sentiment._sentiment_cache_key
    monkeypatch.setattr(sentiment, "_sentiment_cache_key", live_key)
    monkeypatch.setattr(sentiment, "_SENT_CACHE_FLUSH_EVERY_S", sentiment._SENT_CACHE_FLUSH_EVERY_S)
    monkeypatch.setattr(nr, "_isolate_sentiment_cache", lambda: None)
    monkeypatch.setattr(fb, "_LIVE_KEY", None)
    fb._bound_verdict_cache()
    fb._bound_verdict_cache()
    assert sentiment._sentiment_cache_key is fb._prompt_key and fb._LIVE_KEY is live_key
    assert sentiment._SENT_CACHE_FLUSH_EVERY_S >= 1e12


def test_score_goes_band_by_band_waits_for_the_pull_and_resumes(root, monkeypatch):
    plan = _plan({"AAA": ["2025-10-06", "2025-10-13"], "BBB": ["2025-10-06"]}, rank={"AAA": 0, "BBB": 1})
    plan["band_size"] = 1
    monkeypatch.setattr(fb, "_bound_verdict_cache", lambda: None)
    monkeypatch.setattr(fb, "scored_runs", lambda spec=fb.SPEC: {"pre-2025-10-06-b1"})
    seen = []

    def fake(plan_, d, band, **kw):
        seen.append((str(d), band, kw["source"]))
        if str(d) == "2025-10-06" and band == 0:
            fb._write_progress(week_done="2025-10-13")          # the pull moves on meanwhile
        return {"status": "stored"}
    monkeypatch.setattr(fb, "score_day_band", fake)
    fb._write_progress(week_done="2025-10-06")
    slept = []
    out = fb.score(plan, window="any", sleep=slept.append, poll_s=1.0)
    # band 0 over every day, then band 1; (10-06, b1) was stored by an earlier run
    assert seen == [("2025-10-06", 0, "finnhub"), ("2025-10-13", 0, "finnhub"),
                    ("2025-10-13", 1, "finnhub")]
    assert out == {"stored": 3} and slept == []


def test_the_scorer_waits_while_a_week_is_not_pulled(root, monkeypatch):
    plan = _plan({"AAA": ["2025-10-13"]})
    monkeypatch.setattr(fb, "_bound_verdict_cache", lambda: None)
    monkeypatch.setattr(fb, "scored_runs", lambda spec=fb.SPEC: set())
    monkeypatch.setattr(fb, "score_day_band", lambda *a, **k: {"status": "stored"})
    fb._write_progress(week_done="2025-10-06")
    naps = []

    def sleep(s):
        naps.append(s)
        fb._write_progress(pull_complete=True)
    assert fb.score(plan, window="any", sleep=sleep, poll_s=7.0) == {"stored": 1}
    assert naps == [7.0]


def test_the_events_source_never_waits_for_the_pull(root, monkeypatch):
    plan = _plan({"AAA": ["2025-10-13"]})
    monkeypatch.setattr(fb, "_bound_verdict_cache", lambda: None)
    specs = []
    monkeypatch.setattr(fb, "scored_runs", lambda spec=fb.SPEC: specs.append(spec) or set())
    got = []
    monkeypatch.setattr(fb, "score_day_band", lambda *a, **k: got.append(k) or {"status": "stored"})
    naps = []
    out = fb.score(plan, window="any", sleep=naps.append, source="events", tables=object())
    assert out == {"stored": 1} and naps == [] and specs == [fb.EVENTS_SPEC]
    assert got[0]["source"] == "events"


def _stub_scoring(monkeypatch, fail=()):
    from src.analysis import news_clustering, news_replay as nr
    from src.analysis import sentiment
    monkeypatch.setattr(news_clustering, "set_corpus", lambda arts: None)
    inserted = []
    monkeypatch.setattr(fb, "_insert_rows", lambda rows, **k: inserted.append(rows))
    monkeypatch.setattr(fb, "prev_closes", lambda tks, d: {t: 10.0 for t in tks})
    monkeypatch.setattr(nr, "baselines_as_of", lambda d, spec=None: {})
    monkeypatch.setattr(sentiment, "analyse_sentiment", lambda *a, **k: None)
    monkeypatch.setattr(sentiment, "filter_relevant_articles", lambda tk, arts: list(arts))
    calls = []

    def replay_one(tk, pool, info, engine, prov, baselines):
        calls.append((tk, len(pool), prov["pool_spec"], info["run_id"], info["when"]))
        if tk in fail:
            return {"ticker": tk, "scorer_failed": True}
        return {"ticker": tk, "news": 0.1 if pool else 0.0, "pool_spec": prov["pool_spec"]}
    monkeypatch.setattr(nr, "_replay_one", replay_one)
    return inserted, calls


def test_a_day_band_is_scored_from_its_own_source_and_written_whole(root, monkeypatch):
    inserted, calls = _stub_scoring(monkeypatch)
    plan = _plan({"AAA": ["2025-10-07"], "BBB": ["2025-10-07"]})
    monkeypatch.setattr(fb, "events_pools",
                        lambda t, tks, d: {"AAA": ["a1", "a2"], "BBB": []})
    res = fb.score_day_band(plan, date(2025, 10, 7), 0, source="events", tables=object())
    assert res["status"] == "stored" and res["views"] == 1
    assert calls == [("AAA", 2, "pre:events", "pre-2025-10-07-b0", fb.cutoff(date(2025, 10, 7))),
                     ("BBB", 0, "pre:events", "pre-2025-10-07-b0", fb.cutoff(date(2025, 10, 7)))]
    assert len(inserted) == 1 and {r["replay_version"] for r in inserted[0]} == {"events-preopen-v1"}


def test_finnhub_names_not_acquired_get_no_row(root, monkeypatch):
    inserted, calls = _stub_scoring(monkeypatch)
    d = date(2025, 10, 7)
    fb.write_raw(fb.week_path("AAA", fb.week_start(d)), _meta(1, items=[
        {"datetime": int(datetime(2025, 10, 6, 15, tzinfo=timezone.utc).timestamp()),
         "headline": "AAA wins a contract", "url": "http://x", "source": "Reuters", "summary": ""}]))
    plan = _plan({"AAA": ["2025-10-07"], "BBB": ["2025-10-07"]})
    res = fb.score_day_band(plan, d, 0)
    assert res["missing"] == 1 and [c[0] for c in calls] == ["AAA"]
    assert calls[0][1] == 1 and calls[0][2] == "pre:finnhub"


def test_a_scorer_failure_writes_nothing(root, monkeypatch):
    inserted, _ = _stub_scoring(monkeypatch, fail=("AAA",))
    from src.analysis import news_history as nh
    monkeypatch.setattr(nh, "content_failures_only", lambda rows: False)
    plan = _plan({"AAA": ["2025-10-07"]})
    monkeypatch.setattr(fb, "events_pools", lambda t, tks, d: {"AAA": ["a"]})
    res = fb.score_day_band(plan, date(2025, 10, 7), 0, source="events", tables=object())
    assert res["status"] == "scorer failures" and inserted == []


def test_the_events_source_needs_its_tables(root):
    with pytest.raises(ValueError):
        fb.score_day_band(_plan({"AAA": ["2025-10-07"]}), date(2025, 10, 7), 0, source="events")


def test_the_live_pipeline_never_imports_the_backfill():
    import pathlib
    import re
    src = pathlib.Path("src")
    pat = re.compile(r"^\s*(from|import)\s+[\w.]*news_finnhub_backfill|"
                     r"^\s*from\s+src\.analysis\s+import\s+[^\n]*\bnews_finnhub_backfill\b", re.M)
    users = [p for p in src.rglob("*.py") if p.name != "news_finnhub_backfill.py"
             and pat.search(p.read_text(encoding="utf-8", errors="ignore"))]
    assert users == []
