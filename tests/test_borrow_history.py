"""The backtests' borrow history (`src/data/deep/borrow_history.py`, user directive 2026-10-06: "continue building
in our database all the borrow fees, and now the availability of lending").

What must hold: the community archive's files parse in IBKR's own format (">10000000" capped and flagged, `BRK B`
read as `BRK-B`, IBKR's ET stamp stored as naive UTC across a DST change, a weekend re-download of the same file
dropped); a reader at an instant only sees a file stamped at or before it, never one older than 4 days ("unknown"),
and a name the file in force does not list is NOT lendable (IBKR drops what it cannot lend); the lendable rule is the
live gate's ($10,000 at the price); the daily fee prefers our archive, then IBKR's API, then the community archive;
the API pull consolidates; the fetcher asks only names without a history and never runs in regular hours or beside
another pull.
"""
from __future__ import annotations

import io
import json
import zipfile
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import pytest

from src.data import deep as _deep
from src.data.deep import borrow_history as bh

HEADER = "#SYM\tCUR\tNAME\tCON\tISIN\tREBATERATE\tFEERATE\tAVAILABLE"


def _file(stamp, rows):
    lines = [f"#BOF\t{stamp}"] + [HEADER] + ["\t".join(r) for r in rows] + ["#EOF\t3"]
    return "\n".join(lines) + "\n"


@pytest.fixture
def store(tmp_path, monkeypatch):
    monkeypatch.setattr(_deep, "DEEP_DIR", tmp_path / "deep")
    bh._reset_cache()
    zp = tmp_path / "ib_shorting.zip"
    with zipfile.ZipFile(zp, "w") as z:
        # Friday 2021-03-12 07:30 EST: AAA capped, BRK B 5,000 shares at 1.5 %/yr, XYZ
        z.writestr("210313_shorting.tsv", _file("2021.03.12\t07:30:00", [
            ("AAA", "USD", "AAA INC", "101", "US0001", "-0.30", "0.25", ">10000000"),
            ("BRK B", "USD", "BERKSHIRE B", "102", "US0002", "-1.00", "1.50", "5000"),
            ("XYZ", "USD", "XYZ CORP", "103", "US0003", "-50.0", "51.0", "200")]))
        # Sunday's re-download of the same IBKR file: dropped
        z.writestr("210314_shorting.tsv", _file("2021.03.12\t07:30:00", [
            ("AAA", "USD", "AAA INC", "101", "US0001", "-0.30", "0.25", ">10000000")]))
        # Monday 2021-03-15 07:30 EDT: BRK B no longer listed (nothing to lend)
        z.writestr("210316_shorting.tsv", _file("2021.03.15\t07:30:00", [
            ("AAA", "USD", "AAA INC", "101", "US0001", "-0.30", "0.30", "9000000")]))
    r = bh.build_archive(zp)
    yield r
    bh._reset_cache()


def test_the_archive_is_read_in_ibkrs_own_format(store):
    assert store == {"rows": 4, "files": 2}                                  # the re-download dropped
    out = bh.family_dir(bh.FAMILY)
    S = _deep.read_parquet(out / bh.ARCHIVE_FILE)
    a = S[(S.ticker == "AAA")].sort_values("ts")
    assert list(a.available) == [10_000_000.0, 9_000_000.0] and list(a.available_capped) == [True, False]
    assert set(S.ticker) == {"AAA", "BRK-B", "XYZ"} and "BRK B" in set(S.sym)
    F = _deep.read_parquet(out / bh.FILES_FILE)
    assert [str(x) for x in pd.to_datetime(F.ts)] == ["2021-03-12 12:30:00", "2021-03-15 11:30:00"]   # EST, then EDT
    m = json.loads((out / "manifest.json").read_text())
    assert m["archive"]["duplicate_files_dropped"] == 1


def test_a_reader_sees_only_the_file_in_force_at_the_instant(store):
    et = "America/New_York"
    before = bh.snapshot_at("BRK-B", pd.Timestamp("2021-03-12 07:00", tz=et))
    assert before["status"] == "unknown"                                     # the first file is not out yet
    fri = bh.snapshot_at("BRK-B", pd.Timestamp("2021-03-12 15:00", tz=et))
    assert (fri["status"], fri["available"], fri["fee"], fri["source"]) == ("listed", 5000.0, 1.5, "community_archive")
    mon = bh.snapshot_at("BRK-B", pd.Timestamp("2021-03-15 10:00", tz=et))
    assert mon["status"] == "not_listed"                                     # Monday's file dropped it
    sat = bh.snapshot_at("BRK-B", pd.Timestamp("2021-03-13 12:00", tz=et))
    assert sat["status"] == "listed"                                         # Friday's file still in force
    stale = bh.snapshot_at("AAA", pd.Timestamp("2021-03-20 10:00", tz=et))
    assert stale["status"] == "unknown"                                      # 5 days old: no file in force
    assert bh.archive_rows(["BRK-B"], window=("2021-03-13", "2021-03-31")).empty   # outside a name's own dates


def test_lendable_is_the_live_gates_rule(store):
    t = pd.Timestamp("2021-03-12 15:00", tz="America/New_York")
    assert bh.lendable_at("BRK-B", t, price=1.0, min_usd=10_000)[0] is False     # $5,000 of shares
    assert bh.lendable_at("BRK-B", t, price=3.0, min_usd=10_000)[0] is True      # $15,000
    assert bh.lendable_at("BRK-B", pd.Timestamp("2021-03-15 10:00", tz="America/New_York"), 3.0, min_usd=10_000)[0] is False
    assert bh.lendable_at("AAA", pd.Timestamp("2021-03-25", tz="America/New_York"), 50.0)[0] is None


def test_the_daily_fee_prefers_our_archive_then_ibkrs_api_then_the_community_archive(store, tmp_path):
    own = pd.DataFrame({"ticker": ["AAA"], "date": ["2021-03-15"], "source": ["ibkr"], "available": [1.0], "fee": [5.0],
                        "rebate": [0.0], "open_available": [1.0], "high_available": [1.0], "low_available": [1.0],
                        "open_fee": [5.0], "high_fee": [5.0], "low_fee": [5.0]})
    _deep.write_parquet(own, _deep.DEEP_DIR / "borrow_daily.parquet")
    api = pd.DataFrame({"ticker": ["AAA", "AAA"], "date": pd.to_datetime(["2021-03-15", "2021-03-11"]).date,
                        "open_fee": [6.0, 7.0], "high_fee": [6.0, 7.0], "low_fee": [6.0, 7.0], "fee": [6.0, 7.0]})
    _deep.write_parquet(api, bh.family_dir(bh.FAMILY) / bh.API_FILE)
    D = bh.fee_daily("AAA")
    got = {str(d): (f, s) for d, f, s in zip(D.date, D.fee, D.source)}
    assert got["2021-03-15"] == (5.0, "own_archive")
    assert got["2021-03-11"] == (7.0, "ibkr_api")
    assert got["2021-03-12"] == (0.25, "community_archive")


def test_the_api_pull_consolidates_and_only_missing_names_are_asked(tmp_path, monkeypatch):
    monkeypatch.setattr(_deep, "DEEP_DIR", tmp_path / "deep")
    api = tmp_path / "api"
    (api / "parts").mkdir(parents=True)
    (api / "parts" / "AAA.csv").write_text("date,open_fee,high_fee,low_fee,fee\n2026-10-01,0.25,0.3,0.25,0.3\n")
    (api / "parts" / "BRK-B.csv").write_text("date,open_fee,high_fee,low_fee,fee\n2026-10-01,0.25,0.25,0.25,0.25\n"
                                            "2026-10-02,0.25,0.25,0.25,0.26\n")
    (api / "status.json").write_text(json.dumps({"AAA": {"status": "ok", "conid": 1}, "GONE": {"status": "no_contract"},
                                                 "ERR": {"status": "error"}}))
    r = bh.build_fee_api(api_dir=api)
    assert r["rows"] == 3 and r["names"] == 3
    D = _deep.read_parquet(bh.family_dir(bh.FAMILY) / bh.API_FILE)
    assert sorted(set(D.ticker)) == ["AAA", "BRK-B"]
    monkeypatch.setattr(_deep, "deep_universe", lambda refresh=False: ["AAA", "BRK-B", "GONE", "ERR", "NEW"])
    assert bh.missing_names(api) == ["ERR", "NEW"]                        # an error is retried, a new name asked


def test_the_fetcher_never_runs_beside_another_pull_nor_through_ibcs_restart(tmp_path, monkeypatch):
    """Day or night (user 2026-10-06: "we can continue during the day"), but never while another pull writes parts
    and never through IBC's daily gateway restart — without ever dialing IBKR in either case."""
    api = tmp_path / "api"
    (api / "parts").mkdir(parents=True)
    (api / "parts" / "AAA.csv").write_text("date,open_fee,high_fee,low_fee,fee\n")   # written just now
    monkeypatch.setattr(bh, "missing_names", lambda api_dir=None, names=None: pytest.fail("must not look for names"))
    assert bh.fetch_missing(api_dir=api)["skipped"] == "another pull is writing"
    import os
    import time as _t
    old = _t.time() - 3600
    os.utime(api / "parts" / "AAA.csv", (old, old))
    monkeypatch.setattr(bh, "missing_names", lambda api_dir=None, names=None: ["NEW"])
    monkeypatch.setattr(bh, "_in_ibc_restart", lambda now=None: True)
    monkeypatch.setattr(bh.time, "sleep", lambda s: None)
    out = bh.fetch_missing(budget_seconds=0.0, api_dir=api)
    assert out["skipped"] == "IBC restart window" and out["fetched"] == 0
