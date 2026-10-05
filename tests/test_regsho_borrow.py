"""Reg SHO threshold lists (`src/data/deep/regsho.py`) and IBKR borrow history
(`src/data/deep/borrow.py`) — the parsing, the calendar and the archive summary.

The features' point-in-time rules are pinned in `tests/test_deep_features.py`;
these pin what lands in the store: every market's file format, that a day is
asked only once every market has published it (NYSE answers an unpublished day
like an empty list), and that the borrow table is OUR OWN archive of IBKR's file
alone, each finished day summarised once (user directive 2026-10-02: "just ingest
our own from ibkr borrow data").
"""
from __future__ import annotations

import gzip
from datetime import date, datetime
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import pytest

from src.data import deep
from src.data.deep import borrow as bw
from src.data.deep import regsho as rs

ET = ZoneInfo("America/New_York")


@pytest.fixture(autouse=True)
def _tmp_store(tmp_path, monkeypatch):
    monkeypatch.setattr(deep, "DEEP_DIR", tmp_path / "deep")
    (tmp_path / "deep").mkdir()
    yield


# ── Reg SHO ──────────────────────────────────────────────────────────────────

NASDAQ = ("Symbol|Security Name|Market Category|Reg SHO Threshold Flag|Rule 3210|Filler\n"
          "AAPD|DIREXION SHS ETF TR DAILY AAPL|G|Y|N|\nXYZ|NOT FLAGGED|G|N|N|\n20261001230024\n")
NYSE = ("Symbol|Security Name|Market Category|Reg SHO Threshold Flag|Filler|Filler\n"
        "BRK.B|Berkshire B|NYSE|Y||\n20261001220201\n")
CBOE = "Symbol|CompanyName\nAMPU|Defiance Daily Target 2X Long AMPX ETF\n20261001030418\n"


def test_each_markets_file_format_parses_to_our_spelling():
    d = date(2026, 10, 1)
    a = rs.parse_list(NASDAQ, d, "nasdaq")
    assert a["symbol"].tolist() == ["AAPD"]                      # a row whose flag is not Y is dropped
    b = rs.parse_list(NYSE, d, "nyse")
    assert b["symbol"].tolist() == ["BRK-B"] and b["symbol_raw"].tolist() == ["BRK.B"]
    c = rs.parse_list(CBOE, d, "bzx")
    assert c["symbol"].tolist() == ["AMPU"] and (c["date"] == "2026-10-01").all()
    empty = rs.parse_list("Symbol|Security Name|Market Category|Reg SHO Threshold Flag|Filler|Filler\n"
                          "20261001210500\n", d, "arca")
    assert empty.empty and list(empty.columns) == ["date", "market", "symbol", "symbol_raw", "name"]


def test_a_page_without_the_list_header_is_refused():
    with pytest.raises(RuntimeError):
        rs.parse_list("<html><body>blocked</body></html>", date(2026, 10, 1), "nasdaq")


def test_only_completed_days_are_asked_never_today(monkeypatch):
    sess = [date(2026, 9, 28), date(2026, 9, 29), date(2026, 9, 30)]
    monkeypatch.setattr(rs, "store_sessions", lambda: sess)
    at = lambda h: datetime(2026, 10, 2, h, 0, tzinfo=ET)          # noqa: E731
    got = rs.completed_days(date(2026, 9, 28), now=at(8))
    assert got == sess + [date(2026, 10, 1)]                     # the store's sessions, then the calendar's
    assert date(2026, 10, 2) not in got
    assert rs.completed_days(date(2026, 9, 28), now=at(5))[-1] == date(2026, 9, 30)   # before Cboe's 03:05 + margin


class _Resp:
    def __init__(self, status, text="", headers=None):
        self.status_code, self.text, self.headers = status, text, headers or {}


def _session(resp):
    class S:
        headers = {}

        def get(self, *a, **k):
            return resp
    return S()


@pytest.mark.parametrize("status", [302, 403, 404])
def test_an_unpublished_file_fails_the_key_instead_of_reading_empty(monkeypatch, status):
    monkeypatch.setitem(rs._SESSIONS, "bzx", _session(_Resp(status)))
    with pytest.raises(RuntimeError):
        rs.fetch(rs.key_of(date(2026, 10, 1), "bzx"))


def test_a_rate_limit_past_the_cap_fails_fast(monkeypatch):
    monkeypatch.setitem(rs._SESSIONS, "nyse", _session(_Resp(429, "", {"Retry-After": "2538"})))
    monkeypatch.setattr(rs, "MAX_WAIT_S", 120.0)
    with pytest.raises(RuntimeError, match="asked to wait"):
        rs.fetch(rs.key_of(date(2026, 10, 1), "nyse"))


def test_a_published_empty_list_is_a_valid_answer(monkeypatch):
    monkeypatch.setitem(rs._SESSIONS, "bzx", _session(_Resp(200, "Symbol|CompanyName\n20261001030418\n")))
    df = rs.fetch(rs.key_of(date(2026, 10, 1), "bzx"))
    assert df.empty


# ── borrow ───────────────────────────────────────────────────────────────────

def _archive_file(root, day, hhmmss, rows):
    d = root / day
    d.mkdir(parents=True, exist_ok=True)
    body = f"#BOF|{day.replace('-', '.')}|{hhmmss[:2]}:{hhmmss[2:4]}:{hhmmss[4:]}\n" \
           "#SYM|CUR|NAME|CON|ISIN|REBATERATE|FEERATE|AVAILABLE|FIGI|\n" + \
           "".join(f"{s}|USD|N|1|X|{reb}|{fee}|{av}|F|\n" for s, reb, fee, av in rows)
    with gzip.open(d / f"{hhmmss}.txt.gz", "wt", encoding="utf-8") as fh:
        fh.write(body)


def test_archive_day_summarises_the_whole_day_and_closes_on_its_last_file(tmp_path):
    root = tmp_path / "ibkr_borrow"
    day = "2026-10-01"
    _archive_file(root, day, "004725", [("UVIX", -9.2, 13.13, 30000)])
    _archive_file(root, day, "085357", [("UVIX", -9.2, 13.13, 95000), ("BRK B", 3.6, 0.25, ">10000000")])
    _archive_file(root, day, "121737", [("UVIX", -9.2, 12.97, 15000), ("BRK B", 3.6, 0.25, ">10000000")])
    _archive_file(root, day, "212641", [("UVIX", -9.3, 13.17, 100000), ("BRK B", 3.6, 0.25, ">10000000")])
    _archive_file(root, "2026-10-02", "004724", [("UVIX", -9.3, 13.17, 7)])      # the next day: not this row
    out = bw.archive_daily(date(2026, 10, 1), base=root).set_index("ticker")
    u = out.loc["UVIX"]
    assert (u["available"], u["fee"], u["rebate"]) == (100000.0, 13.17, -9.3)          # the day's LAST file
    assert (u["open_available"], u["high_available"], u["low_available"]) == (30000.0, 100000.0, 15000.0)
    assert out.loc["BRK-B", "available"] == bw.CAP                                    # '>10000000' capped, BRK B -> BRK-B
    assert (out["source"] == "ibkr").all() and (out["date"] == day).all()


def test_archive_day_with_too_few_files_is_not_summarised(tmp_path):
    root = tmp_path / "ibkr_borrow"
    _archive_file(root, "2026-09-25", "123300", [("AAA", 3.6, 0.25, 100)])
    assert bw.archive_daily(date(2026, 9, 25), base=root).empty


def test_daily_table_is_our_archive_alone_and_never_today(tmp_path):
    """Only our archive's summaries make the table — a third-party file left in the
    store is never read — and the day in progress is never summarised."""
    deep.write_parquet(pd.DataFrame({"ticker": ["AAA"], "date": ["2026-09-25"], "source": ["ibd"],
                                     "available": [1.0], "fee": [1.0]}),
                       deep.family_dir("iborrowdesk") / "parts" / "AAA.parquet")
    root = tmp_path / "ibkr_borrow"
    for d_ in ("2026-09-28", "2026-10-02"):
        for t in ("090000", "120000", "200000"):
            _archive_file(root, d_, t, [("AAA", 3.0, 9.0, 900)])
    bw.build_daily(base=root, today=date(2026, 10, 2))
    df = deep.read_parquet(deep.DEEP_DIR / bw.DAILY_FILE)
    assert df["date"].tolist() == ["2026-09-28"] and df["source"].tolist() == ["ibkr"]
    assert df["available"].tolist() == [900.0]
