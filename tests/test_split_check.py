"""The verified split table (2026-10-16, `src/data/split_check.py`): split records checked against traded prices, the
false ones left out and undone, confirmed missing ones added, and every store and the ledger following the effective
set. Synthetic prices only — no network."""
from __future__ import annotations

import json
from datetime import date

import numpy as np
import pandas as pd
import pytest

from src.data import split_check as SC


@pytest.fixture(autouse=True)
def checks_file(tmp_path, monkeypatch):
    p = tmp_path / "split_checks.json"
    monkeypatch.setattr(SC, "CHECKS_PATH", p)
    SC._CACHE.clear()
    yield p
    SC._CACHE.clear()


def raw_frame(days, opens, closes):
    return pd.DataFrame({"o": opens, "c": closes}, index=pd.DatetimeIndex(pd.to_datetime(days)))


def yf_series(when=None, ratio=None):
    if when is None:
        return pd.Series(dtype=float)
    return pd.Series([ratio], index=pd.DatetimeIndex([pd.Timestamp(when, tz="America/New_York")]))


# ── the verdicts ──────────────────────────────────────────────────────────────

def test_judge_tells_a_split_from_no_split():
    assert SC.judge(10.0, 1.0, 10.2, 9.8)["verdict"] == "verified"            # a 1-for-10 reverse split traded
    assert SC.judge(10.0, 1.0, 1.03, 0.98)["verdict"] == "nosplit"            # the price did not move like it
    assert SC.judge(10.0, 1.0, 4.0, 4.5)["verdict"] == "ambiguous"            # a move that fits neither
    assert SC.judge(2 / 3, 1.5, 2.21, 2.10)["verdict"] == "ambiguous"         # XRX: +42% on a "2-for-3"... not a 2/3x
    assert SC.judge(1.05, 10.0, 10.6, 10.5)["verdict"] in ("tiny", "verified")  # too small to tell from a day's move


def test_yfinance_confirms_by_date_and_ratio():
    d = date(2024, 6, 10)
    assert SC.yf_confirms(yf_series("2024-06-10", 10.0), d, 0.1) is True      # NVDA's 10-for-1: price x0.1
    assert SC.yf_confirms(yf_series("2024-06-10", 2.0), d, 0.1) is False      # another ratio
    assert SC.yf_confirms(yf_series("2024-05-01", 10.0), d, 0.1) is False     # another date
    assert SC.yf_confirms(yf_series(), d, 0.1) is False
    assert SC.yf_confirms(None, d, 0.1) is None


@pytest.mark.parametrize("raw_verdict,yf,want", [
    ("verified", None, "verified"), ("nosplit", False, "false"), ("nosplit", None, "suspect"),
    ("nosplit", True, "verified"), ("ambiguous", False, "ambiguous"), ("ambiguous", True, "verified"),
    ("unverifiable", None, "unverifiable")])
def test_decide(raw_verdict, yf, want):
    assert SC.decide({"verdict": raw_verdict}, yf) == want


def test_a_record_the_traded_prices_contradict_is_false_xrx():
    """XRX 2026-02-09 "2-for-3" (factor 2/3): the raw prices continuous at ~$2.2, yfinance holds no split."""
    raw = lambda t, a, b: raw_frame(["2026-02-05", "2026-02-06", "2026-02-09", "2026-02-10"],
                                    [2.28, 2.16, 2.21, 2.11], [2.12, 2.34, 2.11, 2.11])
    rec = SC.verify("XRX", date(2026, 2, 9), 2.0, 3.0, today=date(2026, 10, 16), raw=raw, yf=lambda t: yf_series())
    assert rec["status"] == "false" and rec["raw_verdict"] == "nosplit"
    rec2 = SC.verify("XRX", date(2026, 2, 9), 2.0, 3.0, today=date(2026, 10, 16), raw=raw, yf=lambda t: None)
    assert rec2["status"] == "suspect"                                        # no second source: applied, alerted


def test_a_symbol_yfinance_does_not_know_is_no_evidence(monkeypatch):
    """CBCY (OTC): yfinance answers an empty split list AND no price history -> unreachable, not "no split"."""
    class T:
        def __init__(self, sym):
            self.splits = pd.Series(dtype=float)
            self.sym = sym

        def history(self, period="1mo", timeout=30):
            return pd.DataFrame() if self.sym == "CBCY" else pd.DataFrame({"Close": [1.0]})
    import yfinance
    monkeypatch.setattr(yfinance, "Ticker", T)
    assert SC.yf_split_list("CBCY") is None                                   # unknown symbol
    assert SC.yf_split_list("XRX").empty                                      # known, no split
    raw = lambda t, a, b: raw_frame(["2025-10-17", "2025-10-20"], [1.0, 1.0], [1.0, 1.0])
    assert SC.verify("CBCY", date(2025, 10, 20), 1.0, 51.0, today=date(2026, 10, 16), raw=raw)["status"] == "suspect"


def test_a_split_that_traded_is_verified_without_asking_yfinance():
    raw = lambda t, a, b: raw_frame(["2024-06-06", "2024-06-07", "2024-06-10"], [1200.0, 1210.0, 120.5],
                                    [1209.0, 1208.9, 121.8])
    asked = []
    rec = SC.verify("NVDA", date(2024, 6, 10), 1.0, 10.0, today=date(2026, 10, 16), raw=raw,
                    yf=lambda t: asked.append(t))
    assert rec["status"] == "verified" and not asked


def test_future_and_untraded_records_wait():
    raw = lambda t, a, b: raw_frame(["2026-10-15"], [10.0], [10.0])
    assert SC.verify("ABC", date(2026, 10, 20), 1, 2, today=date(2026, 10, 16), raw=raw)["status"] == "pending"
    assert SC.verify("ABC", date(2026, 10, 16), 1, 2, today=date(2026, 10, 16), raw=raw)["status"] == "pending"
    assert SC.verify("ABC", date(2026, 1, 5), 1, 2, today=date(2026, 10, 16),
                     raw=lambda *a: None)["status"] == "unverifiable"


# ── the effective set and the corrections ─────────────────────────────────────

ROWS = {"XRX": [(date(2026, 2, 9), 2.0, 3.0)], "NVDA": [(date(2024, 6, 10), 1.0, 10.0)]}


def test_the_effective_set_leaves_out_false_records_and_adds_missing_ones():
    SC.save_checks({"XRX|2026-02-09": {"status": "false", "from": 2.0, "to": 3.0},
                    "NVDA|2024-06-10": {"status": "verified", "from": 1.0, "to": 10.0},
                    "ABC|2025-03-03": {"status": "missing", "from": 10.0, "to": 1.0},
                    "DEF|2025-03-03": {"status": "candidate", "from": 10.0, "to": 1.0}})
    eff = SC.effective_rows(ROWS)
    assert "XRX" not in eff                                                    # contradicted: out
    assert eff["NVDA"] == [(date(2024, 6, 10), 1.0, 10.0)]                     # verified: kept
    assert "ABC" not in eff                     # confirmed missing: NOT the ledger's (a spin-off can look like a split)
    assert SC.effective_rows(ROWS, with_missing=True)["ABC"] == [(date(2025, 3, 3), 10.0, 1.0)]
    assert "DEF" not in SC.effective_rows(ROWS, with_missing=True)             # unconfirmed: never applied
    assert SC.effective_rows(ROWS, checks={}) == ROWS                          # no table: as recorded


def test_corrections_undo_a_false_record_and_apply_a_missing_split():
    SC.save_checks({"XRX|2026-02-09": {"status": "false", "from": 2.0, "to": 3.0},
                    "ABC|2025-03-03": {"status": "missing", "from": 10.0, "to": 1.0}})
    assert SC.corrections("XRX") == [(date(2026, 2, 9), 1.5)]                  # Polygon's x2/3 undone
    assert SC.corrections("ABC") == [(date(2025, 3, 3), 10.0)]                 # the 1-for-10 applied
    # a 30-minute frame (UTC-naive bar starts): only the bars of sessions before the date move
    idx = pd.DatetimeIndex(["2026-02-06 20:30", "2026-02-09 14:30", "2026-02-09 15:00"])
    df = pd.DataFrame({"Open": [1.44, 2.21, 2.2], "Close": [1.56, 2.2, 2.15], "Volume": [300, 100, 100]}, index=idx)
    out = SC.correct_frame("XRX", df, utc_naive=True)
    assert out["Close"].round(3).tolist() == [2.34, 2.2, 2.15] and out["Volume"].tolist() == [200, 100, 100]
    # a daily frame (naive session dates) and a deep part (a ts column)
    daily = pd.DataFrame({"Close": [1.0, 10.0], "Volume": [1000, 100]},
                         index=pd.DatetimeIndex(["2025-02-28", "2025-03-03"]))
    assert SC.correct_frame("ABC", daily, utc_naive=False)["Close"].tolist() == [10.0, 10.0]
    part = pd.DataFrame({"ts": pd.to_datetime(["2025-02-28 20:00", "2025-03-03 14:30"]), "close": [1.0, 10.0],
                         "volume": [1000, 100]})
    assert SC.correct_frame("ABC", part, ts_col="ts")["close"].tolist() == [10.0, 10.0]
    assert SC.correct_frame("NONE", daily, utc_naive=False) is daily           # nothing to correct: untouched


def test_the_store_and_ledger_readers_get_the_effective_set(tmp_path, monkeypatch):
    """`intraday_store.split_rows` / `latest_splits` / `split_factor_between` — every reader — follow the table."""
    import duckdb
    from src.data import intraday_store as IS
    sp = tmp_path / "splits.parquet"
    df = pd.DataFrame({"ticker": ["XRX", "NVDA"], "execution_date": ["2026-02-09", "2024-06-10"],
                       "split_from": [2.0, 1.0], "split_to": [3.0, 10.0]})
    con = duckdb.connect()
    con.register("df_", df)
    con.execute(f"COPY (SELECT * FROM df_) TO '{sp.as_posix()}' (FORMAT PARQUET)")
    con.close()
    monkeypatch.setattr(IS, "SPLITS_PATH", sp)
    IS._SPLIT_CACHE.clear()
    assert IS.split_factor_between("XRX", date(2026, 1, 1), date(2026, 3, 1)) == pytest.approx(2 / 3)
    SC.save_checks({"XRX|2026-02-09": {"status": "false", "from": 2.0, "to": 3.0}})
    assert IS.split_factor_between("XRX", date(2026, 1, 1), date(2026, 3, 1)) == 1.0     # left out at once
    assert "XRX" not in IS.latest_splits(date(2026, 10, 16)) and "NVDA" in IS.latest_splits(date(2026, 10, 16))
    assert IS.raw_split_rows()["XRX"] == [(date(2026, 2, 9), 2.0, 3.0)]                  # the records themselves
    IS._SPLIT_CACHE.clear()


# ── missing splits ────────────────────────────────────────────────────────────

def _sessions(closes, opens=None, start="2025-02-24"):
    idx = pd.bdate_range(start, periods=len(closes))
    return pd.Series(opens if opens is not None else closes, index=idx), pd.Series(closes, index=idx)


def test_a_jump_of_a_split_ratio_no_record_explains_is_a_candidate():
    o, c = _sessions([1.0, 1.02, 1.01, 10.1, 10.2, 10.0, 10.1], [1.0, 1.0, 1.0, 10.05, 10.1, 10.1, 10.0])
    got = SC.jump_candidates(o, c, known=[])
    assert len(got) == 1 and got[0]["factor"] == 10.0 and got[0]["day"] == c.index[3].date()
    assert SC.jump_candidates(o, c, known=[c.index[4].date()]) == []                     # a record explains it
    spike_o, spike_c = _sessions([1.0, 1.0, 1.0, 1.0, 1.0], [1.0, 1.0, 2.0, 1.0, 1.0])  # a one-day spike reverts
    assert SC.jump_candidates(spike_o, spike_c, known=[]) == []


def test_find_missing_skips_warrants_and_illiquid_names():
    idx = pd.date_range("2025-02-24 14:30", periods=7, freq="1D")
    prices = [1.0, 1.0, 1.0, 10.0, 10.0, 10.0, 10.0]
    thin = pd.DataFrame({"Open": prices, "Close": prices, "Volume": [100.0] * 7}, index=idx)     # ~$1k a day
    asked = []
    got = SC.find_missing(["ABCDW", "ABC"], {}, load=lambda t: thin, yf=lambda t: asked.append(t))
    assert got == {} and asked == []
    assert SC._not_a_common_listing("FOXXW") and SC._not_a_common_listing("MLACU") and not SC._not_a_common_listing("ABC")


def test_find_missing_adds_only_what_yfinance_confirms():
    idx = pd.date_range("2025-02-24 14:30", periods=7, freq="1D")
    prices = [1.0, 1.0, 1.0, 10.0, 10.0, 10.0, 10.0]
    frame = pd.DataFrame({"Open": prices, "Close": prices, "Volume": [1e7] * 7}, index=idx)
    got = SC.find_missing(["ABC", "DEF"], {}, load=lambda t: frame,
                          yf=lambda t: yf_series(str(idx[3].date()), 0.1) if t == "ABC" else yf_series())
    assert got["ABC|2025-02-27"]["status"] == "missing" and got["ABC|2025-02-27"]["from"] == 10.0
    assert got["DEF|2025-02-27"]["status"] == "candidate"
    assert SC.corrections("ABC") == [(date(2025, 2, 27), 10.0)] and SC.corrections("DEF") == []


# ── the verdict's window: a contradicted record is false only once its sessions have traded ─────────────────

def test_a_contradicted_record_is_false_only_once_its_sessions_have_traded():
    raw = lambda t, a, b: raw_frame(["2026-10-15", "2026-10-16", "2026-10-19", "2026-10-20", "2026-10-21"],
                                    [2.28, 2.16, 2.21, 2.11, 2.15], [2.12, 2.34, 2.11, 2.11, 2.13])
    rec = SC.verify("XRX", date(2026, 10, 19), 2.0, 3.0, today=date(2026, 10, 21), raw=raw, yf=lambda t: yf_series())
    assert rec["status"] == "suspect" and rec["provisional"] is True           # 3 of its 6 sessions traded
    days = list(pd.bdate_range("2026-10-15", periods=9).strftime("%Y-%m-%d"))  # 7 sessions from 10-19
    full = lambda t, a, b: raw_frame(days, [2.2] * 9, [2.2] * 9)
    assert SC.verify("XRX", date(2026, 10, 19), 2.0, 3.0, today=date(2026, 10, 27), raw=full,
                     yf=lambda t: yf_series())["status"] == "false"


def test_a_record_dated_early_is_kept_once_yfinance_lists_the_split():
    """Polygon dates a 1-for-10 on 10-19 and it trades on 10-20: on its date the prices say no split (suspect, still
    applied); once it has traded yfinance lists it within its window -> verified, never false."""
    on_date = lambda t, a, b: raw_frame(["2026-10-15", "2026-10-16", "2026-10-19"], [1.0, 1.0, 1.01], [1.0, 1.0, 1.0])
    assert SC.verify("ABC", date(2026, 10, 19), 10.0, 1.0, today=date(2026, 10, 19), raw=on_date,
                     yf=lambda t: yf_series())["status"] == "suspect"
    later = lambda t, a, b: raw_frame(["2026-10-15", "2026-10-16", "2026-10-19", "2026-10-20"],
                                      [1.0, 1.0, 1.01, 10.1], [1.0, 1.0, 1.0, 9.9])
    assert SC.verify("ABC", date(2026, 10, 19), 10.0, 1.0, today=date(2026, 10, 21), raw=later,
                     yf=lambda t: yf_series("2026-10-20", 0.1))["status"] == "verified"


def test_an_old_record_with_no_session_after_it_is_unverifiable_not_pending():
    """A name that stopped trading at its split: never "pending" (the nightly backfill re-checked those forever)."""
    raw = lambda t, a, b: raw_frame(["2026-01-01", "2026-01-02"], [5.0, 5.0], [5.0, 5.0])
    assert SC.verify("GONE", date(2026, 1, 5), 1, 10, today=date(2026, 10, 16), raw=raw)["status"] == "unverifiable"
    assert SC.verify("SOON", date(2026, 10, 14), 1, 10, today=date(2026, 10, 16), raw=raw)["status"] == "pending"


# ── the table's robustness ────────────────────────────────────────────────────

def test_a_malformed_table_never_breaks_a_reader(checks_file):
    import os
    SC.save_checks({"XRX|2026-02-09": {"status": "false", "from": 2.0, "to": 3.0}})
    assert SC.status_of("XRX", date(2026, 2, 9)) == "false"
    checks_file.write_text(json.dumps(["not", "a", "table"]))                  # a hand edit of the wrong shape
    st = checks_file.stat()
    os.utime(checks_file, (st.st_atime, st.st_mtime + 5))
    assert SC.status_of("XRX", date(2026, 2, 9)) == "false"                    # the last good version is kept
    SC._CACHE.clear()
    assert SC.load_checks() == {} and SC.status_of("XRX", date(2026, 2, 9)) == "unchecked"   # none read: as recorded
    checks_file.write_text(json.dumps({"BAD|2026-01-01": "x", "OK|2026-01-02": {"status": "false"}}))
    SC._CACHE.clear()
    assert list(SC.load_checks()) == ["OK|2026-01-02"]                        # a record of another shape left out
    assert SC.corrections("OK") == []                                          # no ratio: never applied
    assert SC.status_of("X", date(2026, 1, 1), checks={"X|2026-01-01": "junk"}) == "unchecked"


def test_writes_survive_a_busy_target_and_a_dead_writers_lock(checks_file, monkeypatch):
    import os
    import time
    calls = []
    real = os.replace

    def flaky(a, b):
        calls.append(1)
        if len(calls) == 1:
            raise PermissionError(5, "Access is denied")                       # another process holds the file
        return real(a, b)
    monkeypatch.setattr(SC.os, "replace", flaky)
    lock = checks_file.with_name(checks_file.name + ".lock")
    lock.write_text("123")
    old = time.time() - 600
    os.utime(lock, (old, old))                                                 # a dead writer's lock
    SC.save_checks({"A|2026-01-01": {"status": "verified", "from": 1.0, "to": 2.0}})
    assert SC.load_checks()["A|2026-01-01"]["status"] == "verified" and len(calls) == 2 and not lock.exists()
    lock.write_text("456")                                                     # a live writer: wait, then go on
    assert SC._lock_file(checks_file, timeout=0.2) is None
    lock.unlink()


# ── the live guard ────────────────────────────────────────────────────────────

def at_et(day, hh, mm=0):
    return pd.Timestamp(f"{day} {hh:02d}:{mm:02d}", tz="America/New_York").to_pydatetime()


def held(tk, px=1.02):
    return [{"status": "OPEN", "ticker": tk, "current_price": px}]


def test_the_guard_never_judges_a_split_before_0400_et_of_its_date(monkeypatch):
    """The 01:00-03:30 overnight ticks fall on the split's date but quote the old terms: a real 1-for-10 judged then
    reads "no split" (2026-10-16's review: the record turned false, the held short unrescaled at 04:00 — a fake 900%
    loss). Nothing is read or written before 04:00 ET of its date."""
    from src.data import intraday_store as IS
    monkeypatch.setattr(IS, "split_rows", lambda: {"ABCD": [(date(2026, 10, 19), 10.0, 1.0)]})
    asked = []
    raw = lambda t, a, b: asked.append(t)
    for hh, mm in ((1, 0), (2, 0), (3, 0), (3, 30), (3, 59)):
        assert SC.verify_held(held("ABCD"), now=at_et("2026-10-19", hh, mm), raw=raw, yf=lambda t: yf_series()) == {}
    assert SC.verify_held(held("ABCD"), now=at_et("2026-10-18", 22, 30), raw=raw) == {}   # the evening before
    assert asked == [] and SC.load_checks() == {}


def test_the_guard_reads_the_raw_prices_of_the_split_date_never_a_mark(monkeypatch):
    from src.data import intraday_store as IS
    monkeypatch.setattr(IS, "split_rows", lambda: {"ABCD": [(date(2026, 10, 19), 10.0, 1.0)]})
    raw = lambda t, a, b: raw_frame(["2026-10-15", "2026-10-16", "2026-10-19"], [1.01, 1.0, 10.1], [1.0, 1.0, 9.9])
    asked = []
    up = SC.verify_held(held("ABCD", px=1.02), now=at_et("2026-10-19", 4, 30), raw=raw,
                        yf=lambda t: asked.append(t))
    assert up["ABCD|2026-10-19"]["status"] == "verified" and not asked         # the stale $1.02 mark never read
    assert SC.load_checks()["ABCD|2026-10-19"]["how"].startswith("held")


def test_no_trade_on_the_split_date_yet_stays_pending(monkeypatch):
    from src.data import intraday_store as IS
    monkeypatch.setattr(IS, "split_rows", lambda: {"ABCD": [(date(2026, 10, 19), 10.0, 1.0)]})
    raw = lambda t, a, b: raw_frame(["2026-10-15", "2026-10-16"], [1.01, 1.0], [1.0, 1.0])
    assert SC.verify_held(held("ABCD"), now=at_et("2026-10-19", 4, 30), raw=raw) == {}
    assert SC.load_checks() == {}


def test_a_contradicted_record_is_alerted_and_kept_never_dropped_by_the_guard(monkeypatch):
    """The raw prices of its date continuous, yfinance holds no split: SUSPECT (provisional), still in the effective
    set, no store corrected — the guard never drops a record (one dated a day early looks the same on its date)."""
    from src.data import intraday_store as IS
    rows = {"XRX": [(date(2026, 10, 19), 2.0, 3.0)]}
    monkeypatch.setattr(IS, "split_rows", lambda: rows)
    n = []

    def raw(t, a, b):
        n.append(t)
        return raw_frame(["2026-10-15", "2026-10-16", "2026-10-19"], [2.28, 2.16, 2.21], [2.12, 2.34, 2.11])
    up = SC.verify_held(held("XRX", 2.21), now=at_et("2026-10-19", 10, 0), raw=raw, yf=lambda t: yf_series())
    rec = up["XRX|2026-10-19"]
    assert rec["status"] == "suspect" and rec["provisional"] is True
    assert SC.effective_rows(rows) == rows and SC.corrections("XRX") == []      # still applied, nothing undone
    assert SC.verify_held(held("XRX", 2.21), now=at_et("2026-10-19", 10, 30), raw=raw,
                          yf=lambda t: yf_series()) == {}                       # re-checked at most hourly
    assert len(n) == 1
    up = SC.verify_held(held("XRX", 2.21), now=at_et("2026-10-19", 11, 30), raw=raw, yf=lambda t: yf_series())
    assert up["XRX|2026-10-19"]["status"] == "suspect" and len(n) == 2
    # outside the window, or no open trade: nothing checked
    assert SC.verify_held(held("XRX"), now=at_et("2026-10-30", 10, 0), raw=raw) == {}
    assert SC.verify_held([{"status": "CLOSED", "ticker": "XRX"}], now=at_et("2026-10-19", 12, 0), raw=raw) == {}


def test_the_guard_turns_any_false_verdict_into_an_alert(monkeypatch):
    """Belt and braces: whatever `verify` answers inside the guard's window, the guard never writes "false"."""
    from src.data import intraday_store as IS
    monkeypatch.setattr(IS, "split_rows", lambda: {"ABCD": [(date(2026, 10, 19), 10.0, 1.0)]})
    monkeypatch.setattr(SC, "verify", lambda *a, **k: {"status": "false", "factor": 10.0})
    up = SC.verify_held(held("ABCD"), now=at_et("2026-10-19", 10, 0))
    assert up["ABCD|2026-10-19"]["status"] == "suspect" and up["ABCD|2026-10-19"]["provisional"] is True
    assert SC.status_of("ABCD", date(2026, 10, 19)) == "suspect"
