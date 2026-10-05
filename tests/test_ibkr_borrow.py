"""IBKR's borrow book, archived every tick, and the borrow gate on every new
short (2026-09-25, `src/data/ibkr_borrow.py`).

What must hold: the file parses (the capped ">10000000", class shares written
with a space); each distinct file is archived once, RAW; a later evaluation
reads the snapshot at or before its instant, never a newer one; a short IBKR
cannot lend in size, or lends above the fee cap, is skipped at the point a short
is OPENED (both entry paths) while a BUY is never checked; the quoted fee rides
the trade into the borrow carry; and no current file means "not checked",
never "blocked".
"""
from __future__ import annotations

import gzip
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from zoneinfo import ZoneInfo

import pytest

from config.settings import settings
from src.data import ibkr_borrow as ib

ET = ZoneInfo("America/New_York")


def _file(stamp: str, rows: list) -> str:
    """rows: (sym, rebate, fee, available)."""
    d, t = stamp.split(" ")
    lines = [f"#BOF|{d.replace('-', '.')}|{t}", "#SYM|CUR|NAME|CON|ISIN|REBATERATE|FEERATE|AVAILABLE|FIGI|"]
    for sym, reb, fee, av in rows:
        lines.append(f"{sym}|USD|{sym} INC|1|US0|{reb}|{fee}|{av}|BBG0|")
    lines.append(f"#EOF|{len(rows)}")
    return "\n".join(lines) + "\n"


ROWS = [("AAPL", 3.6, 0.25, ">10000000"), ("HOT", -80.0, 496.15, 3000), ("NONE", 0.0, 12.0, 0),
        ("BRK B", 3.5, 0.3, 500000), ("MID", 1.0, 30.0, 200000)]


@pytest.fixture
def on(monkeypatch):
    monkeypatch.setattr(settings, "enable_ibkr_borrow_snapshot", True)
    monkeypatch.setattr(settings, "enable_short_borrow_gate", True)
    monkeypatch.setattr(settings, "short_borrow_max_fee_pct", 50.0)
    monkeypatch.setattr(settings, "short_borrow_min_available_usd", 10_000.0)
    monkeypatch.setattr(settings, "ibkr_borrow_max_age_minutes", 120.0)
    ib.reset()
    yield
    ib.reset()


def test_the_file_parses_with_capped_counts_and_class_shares():
    ts, t = ib.parse(_file("2026-09-25 16:27:30", ROWS))
    assert ts == datetime(2026, 9, 25, 16, 27, 30, tzinfo=ET)
    assert t["AAPL"].available == 10_000_001 and t["AAPL"].fee_pct == pytest.approx(0.25)
    assert t["HOT"].fee_pct == pytest.approx(496.15) and t["HOT"].rebate_pct == pytest.approx(-80.0)
    assert ib.lookup(t, "brk-b").symbol == "BRK B"          # this project writes BRK-B
    assert ib.lookup(t, "XXXX") is None


def test_each_file_is_archived_once_raw(on, tmp_path, monkeypatch):
    text = _file("2026-09-25 16:27:30", ROWS)
    monkeypatch.setattr(ib, "_download", lambda timeout=30.0: text.encode())
    t = ib.snapshot()
    assert len(t) == 5
    path = ib.archive_path(datetime(2026, 9, 25, 16, 27, 30, tzinfo=ET))
    assert path.exists() and path.parent.name == "2026-09-25" and path.name == "162730.txt.gz"
    with gzip.open(path, "rt", encoding="utf-8") as fh:
        assert fh.read() == text                          # the raw file, byte for byte
    ib.snapshot()                                         # same file again: nothing new
    assert len(ib.archive_index()) == 1
    monkeypatch.setattr(ib, "_download", lambda timeout=30.0: _file("2026-09-25 16:42:30", ROWS).encode())
    ib.snapshot()
    assert [p.name for _, p in ib.archive_index()] == ["162730.txt.gz", "164230.txt.gz"]


def test_a_later_evaluation_reads_the_snapshot_at_or_before_its_instant(on, monkeypatch):
    for stamp, fee in (("2026-09-25 10:00:00", 5.0), ("2026-09-25 14:00:00", 90.0)):
        monkeypatch.setattr(ib, "_download",
                            lambda timeout=30.0, s=stamp, f=fee: _file(s, [("HOT", 0, f, 5000)]).encode())
        ib.snapshot()
    at = lambda h, m=0: datetime(2026, 9, 25, h, m, tzinfo=ET)       # noqa: E731
    assert ib.borrow_at("HOT", at(9, 59)) is None                    # before any snapshot
    assert ib.borrow_at("HOT", at(10, 0)).fee_pct == 5.0
    assert ib.borrow_at("HOT", at(13, 59)).fee_pct == 5.0            # never the newer one
    assert ib.borrow_at("HOT", at(14, 0)).fee_pct == 90.0
    naive_utc = at(15).astimezone(timezone.utc).replace(tzinfo=None)
    assert ib.borrow_at("HOT", naive_utc).fee_pct == 90.0            # naive = UTC
    assert ib.borrow_at("AAPL", at(15)) is None                      # not listed then


def test_the_gate(on, monkeypatch):
    now = datetime(2026, 9, 25, 16, 30, tzinfo=ET)
    monkeypatch.setattr(ib, "_download", lambda timeout=30.0: _file("2026-09-25 16:27:30", ROWS).encode())
    ib.snapshot()
    assert ib.short_block("AAPL", 250.0, now=now)[0] is None
    assert ib.short_block("MID", 20.0, now=now)[0] is None           # 30%/yr is under the cap
    assert ib.short_block("HOT", 9.0, now=now)[0] == "borrow_fee"    # 496%/yr
    assert ib.short_block("NONE", 9.0, now=now)[0] == "no_borrow"    # listed, zero shares
    assert ib.short_block("GONE", 9.0, now=now)[0] == "no_borrow"    # IBKR does not list it
    monkeypatch.setattr(settings, "short_borrow_max_fee_pct", 1000.0)
    assert ib.short_block("HOT", 9.0, now=now)[0] is None            # 3,000 × $9 = $27,000 clears $10,000
    assert ib.short_block("HOT", 3.0, now=now)[0] == "no_borrow"     # 3,000 × $3 = $9,000 does not
    assert ib.short_block("HOT", 3.0, now=now)[1].available == 3000
    # futures, indices, FX and crypto borrow no stock: never judged by the stock file
    for sym in ("CL=F", "YM=F", "^NSEI", "EURUSD=X", "BTC-USD"):
        assert ib.short_block(sym, 70.0, now=now) == (None, None)
    assert ib.is_equity_symbol("BRK-B") and ib.is_equity_symbol("SPY")
    monkeypatch.setattr(settings, "enable_short_borrow_gate", False)
    assert ib.short_block("NONE", 9.0, now=now) == (None, None)


def test_no_current_file_means_not_checked(on, monkeypatch):
    assert ib.short_block("NONE", 9.0) == (None, None)               # nothing downloaded or archived
    monkeypatch.setattr(ib, "_download", lambda timeout=30.0: _file("2026-09-25 10:00:00", ROWS).encode())
    ib.snapshot()
    late = datetime(2026, 9, 25, 13, 0, tzinfo=ET)                   # 3 h later > the 120-min cap
    assert ib.short_block("NONE", 9.0, now=late) == (None, None)
    ib.reset()                                                       # a fresh process reads the archive
    assert ib.short_block("NONE", 9.0, now=datetime(2026, 9, 25, 10, 30, tzinfo=ET))[0] == "no_borrow"


def test_a_failed_download_raises_for_the_source_log(on, monkeypatch):
    def boom(timeout=30.0):
        raise RuntimeError("down")
    monkeypatch.setattr(ib, "_download", boom)
    with pytest.raises(RuntimeError):
        ib.snapshot()


# ── the two entry paths ──────────────────────────────────────────────────────

def _rec(ticker, action):
    from src.models import Recommendation
    return Recommendation(ticker=ticker, type="STOCK",
                          direction="BULLISH" if action == "BUY" else "BEARISH",
                          confidence=0.9, action=action, time_horizon="SWING",
                          rationale="test", generated_at=datetime.now(timezone.utc))


def _entry_env(monkeypatch):
    from src.performance import tracker
    monkeypatch.setattr(settings, "enable_intraday_timing", False)
    monkeypatch.setattr(settings, "enable_correlation_sizing", False)
    monkeypatch.setattr(tracker, "_execution_iso", lambda: "2026-06-10T15:00:00+00:00")
    monkeypatch.setattr(tracker, "_fetch_price", lambda t: 9.0)
    monkeypatch.setattr(tracker, "_reference_close", lambda t: None)
    return tracker


def _blocking(monkeypatch):
    calls = []

    def block(ticker, price, base=None, now=None):
        calls.append(ticker)
        if ticker == "HOT":
            return "borrow_fee", ib.Borrow("HOT", 496.0, -80.0, 3000, None)
        if ticker == "NONE":
            return "no_borrow", None
        return None, ib.Borrow(ticker, 0.3, 3.5, 900_000,
                               datetime(2026, 9, 25, 16, 27, 30, tzinfo=ET))
    monkeypatch.setattr(ib, "short_block", block)
    return calls


def test_the_rank_path_skips_unborrowable_shorts_and_stamps_the_fee(monkeypatch):
    tracker = _entry_env(monkeypatch)
    calls = _blocking(monkeypatch)
    diag = tracker.record_new_trades([_rec("HOT", "SELL"), _rec("NONE", "SELL"), _rec("OKS", "SELL"),
                                      _rec("HOT2", "BUY")], run_id="bw1")
    assert diag["skipped_borrow_fee"] == 1 and diag["skipped_no_borrow"] == 1
    assert diag["opened"] == 2
    assert calls == ["HOT", "NONE", "OKS"]                           # a BUY is never checked
    opened = {t["ticker"]: t for t in tracker._load_trades() if t.get("run_id") == "bw1"}
    assert set(opened) == {"OKS", "HOT2"}
    assert opened["OKS"]["borrow_fee_pct"] == pytest.approx(0.3)
    assert opened["OKS"]["borrow_available_at_entry"] == 900_000
    assert opened["OKS"]["borrow_file_ts"].startswith("2026-09-25T16:27:30")
    assert "borrow_fee_pct" not in opened["HOT2"]
    from src.performance.spread import borrow_annual_pct
    assert borrow_annual_pct(opened["OKS"]) == pytest.approx(0.3)   # the carry charges IBKR's fee


def test_the_follow_through_path_checks_its_shorts(monkeypatch):
    tracker = _entry_env(monkeypatch)
    calls = _blocking(monkeypatch)
    monkeypatch.setattr(settings, "enable_follow_through_trading", True)
    monkeypatch.setattr(settings, "ft_max_entries_per_day", 10)
    ft = {"HOT": {"selected": True, "score": -0.9, "dir": -1.0},
          "OKS": {"selected": True, "score": -0.8, "dir": -1.0},
          "LNG": {"selected": True, "score": -0.7, "dir": 1.0}}
    n = tracker.record_follow_through_trades(ft, signals_by_ticker=None, run_id="ft1")
    assert n == 2 and calls == ["HOT", "OKS"]                        # the long is not checked
    opened = {t["ticker"]: t for t in tracker._load_trades() if t.get("entry_mechanism") == "follow_through"}
    assert set(opened) == {"OKS", "LNG"}
    assert opened["OKS"]["borrow_fee_pct"] == pytest.approx(0.3)
