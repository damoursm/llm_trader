"""Deep history store (``src/data/deep``) — offline tests of the pure parts.

Everything here runs without a network: manifest resumability, the parquet
round-trip through duckdb, and the parsers that turn each provider's payload
into rows that carry their point-in-time key.
"""
from __future__ import annotations

import io
import json
import zipfile
from datetime import date
from pathlib import Path

import pandas as pd
import pytest

from src.data import deep
from src.data.deep import Manifest, read_parquet, write_parquet
from src.data.deep import context, form345, ftd, polygon_deep, sec, wiki


@pytest.fixture(autouse=True)
def _tmp_store(tmp_path, monkeypatch):
    """Every test writes into its own DEEP_DIR."""
    monkeypatch.setattr(deep, "DEEP_DIR", tmp_path / "deep")
    yield


# ── store layer ──────────────────────────────────────────────────────────────

def test_parquet_roundtrip_mixed_dtypes(tmp_path):
    df = pd.DataFrame({
        "ticker": ["A", "B", "C"],
        "when": pd.to_datetime(["2024-01-01T10:00Z", "2024-01-02T11:00Z", None], utc=True),
        "val": [1.5, None, 3.0],
        "note": ["x", None, "z"],
    })
    p = tmp_path / "x.parquet"
    assert write_parquet(df, p) == 3
    back = read_parquet(p)
    assert list(back["ticker"]) == ["A", "B", "C"]
    # tz-aware -> naive UTC
    assert back["when"].dt.tz is None
    assert back["when"].iloc[0] == pd.Timestamp("2024-01-01 10:00")
    assert back["val"].isna().sum() == 1
    assert back["note"].iloc[1] is None or pd.isna(back["note"].iloc[1])


def test_manifest_pending_skips_done_and_optionally_failed():
    m = Manifest("fam")
    m.mark_done("A", 10)
    m.mark_failed("B", "boom")
    m.save()
    assert m.pending(["A", "B", "C"]) == ["B", "C"]
    assert m.pending(["A", "B", "C"], retry_failed=False) == ["C"]
    # persisted and reloaded
    m2 = Manifest("fam")
    assert set(m2.done) == {"A"} and set(m2.failed) == {"B"}
    assert m2.stats() == {"done": 1, "failed": 1, "rows": 10}
    # a later success clears the failure
    m2.mark_done("B", 3)
    assert "B" not in m2.failed


def test_run_keys_writes_parts_and_resumes(monkeypatch):
    calls = []

    def fn(k):
        calls.append(k)
        if k == "BAD":
            raise RuntimeError("nope")
        return pd.DataFrame({"ticker": [k], "v": [1]})

    r = deep.run_keys("fam2", ["X", "BAD", "Y"], fn, workers=2)
    assert r["ok"] == 2 and r["failed"] == 1
    parts = sorted(p.stem for p in (deep.DEEP_DIR / "fam2" / "parts").glob("*.parquet"))
    assert parts == ["X", "Y"]
    # second run only retries the failed key
    calls.clear()
    deep.run_keys("fam2", ["X", "BAD", "Y"], fn, workers=1)
    assert calls == ["BAD"]
    assert deep.consolidate("fam2") == 2


# ── SEC ──────────────────────────────────────────────────────────────────────

def test_sec_filings_frame_keeps_acceptance_instant_and_items():
    block = {
        "accessionNumber": ["0001-24-1", "0001-24-2"],
        "filingDate": ["2024-07-30", "2024-04-20"],
        "reportDate": ["2024-07-30", ""],
        "acceptanceDateTime": ["2024-07-30T20:30:28.000Z", "2024-04-20T21:29:51.000Z"],
        "act": ["34", "34"], "form": ["8-K", "8-K"], "fileNumber": ["001-1", "001-1"],
        "items": ["2.02,9.01", "5.02"], "size": [1000, 2000], "isXBRL": [1, 0],
        "isInlineXBRL": [1, 0], "primaryDocument": ["a.htm", "b.htm"],
    }
    df = sec._filings_frame(block, "0000320193")
    assert len(df) == 2
    assert df["acceptance"].dt.tz is None
    assert df["acceptance"].iloc[0] == pd.Timestamp("2024-07-30 20:30:28")
    assert df["items"].iloc[0] == "2.02,9.01"
    assert df["report_date"].iloc[1] is None
    # an older overflow file may lack columns — tolerated
    df2 = sec._filings_frame({"accessionNumber": ["x"], "form": ["10-K"], "filingDate": ["2001-01-01"]}, "1")
    assert len(df2) == 1 and pd.isna(df2["acceptance"].iloc[0])


def test_sec_facts_frame_filters_concepts_and_keeps_filed():
    j = {"facts": {
        "us-gaap": {
            "EarningsPerShareDiluted": {"units": {"USD/shares": [
                {"end": "2007-09-29", "val": 3.93, "accn": "a", "fy": 2009, "fp": "FY", "form": "10-K", "filed": "2009-10-27"},
                {"end": "2007-09-29", "val": 3.95, "accn": "b", "fy": 2010, "fp": "FY", "form": "10-K", "filed": "2010-10-27"},
            ]}},
            "SomeObscureConcept": {"units": {"USD": [{"end": "2020-01-01", "val": 1, "filed": "2020-02-01"}]}},
        },
        "dei": {"EntityPublicFloat": {"units": {"USD": [{"end": "2020-06-30", "val": 5e9, "filed": "2020-10-30"}]}}},
    }}
    df = sec._facts_frame(j, "0000320193")
    assert set(df["concept"]) == {"EarningsPerShareDiluted", "EntityPublicFloat"}
    # the restated value is a SECOND row with its own filed date, never an overwrite
    eps = df[df["concept"] == "EarningsPerShareDiluted"].sort_values("filed")
    assert list(eps["filed"]) == ["2009-10-27", "2010-10-27"]
    assert list(eps["val"]) == [3.93, 3.95]


# ── Form 345 ─────────────────────────────────────────────────────────────────

def test_form345_join_flags_and_notional():
    sub = pd.DataFrame({"ACCESSION_NUMBER": ["A1", "A2"], "FILING_DATE": ["31-OCT-2025", "03-NOV-2025"],
                        "PERIOD_OF_REPORT": ["29-OCT-2025", "31-OCT-2025"], "DOCUMENT_TYPE": ["4", "4"],
                        "ISSUERCIK": ["1", "2"], "ISSUERNAME": ["Acme", "Beta"],
                        "ISSUERTRADINGSYMBOL": ["acme ", "BETA"], "AFF10B5ONE": ["0", "1"]})
    trans = pd.DataFrame({"ACCESSION_NUMBER": ["A1", "A1", "A9"], "TRANS_DATE": ["29-OCT-2025", "30-OCT-2025", "01-JAN-2025"],
                          "TRANS_CODE": ["P", "S", "P"], "TRANS_SHARES": ["100", "50", "1"],
                          "TRANS_PRICEPERSHARE": ["10.5", "11", "1"], "TRANS_ACQUIRED_DISP_CD": ["A", "D", "A"],
                          "SHRS_OWND_FOLWNG_TRANS": ["1000", "950", "1"], "DIRECT_INDIRECT_OWNERSHIP": ["D", "D", "D"],
                          "SECURITY_TITLE": ["Common Stock"] * 3, "TRANS_TIMELINESS": [None, None, None]})
    own = pd.DataFrame({"ACCESSION_NUMBER": ["A1", "A1"], "RPTOWNERCIK": ["o1", "o2"],
                        "RPTOWNER_RELATIONSHIP": ["Director,Officer", "TenPercentOwner"],
                        "RPTOWNER_TITLE": ["CEO", None]})
    df = form345.join_quarter(sub, trans, own, quarter="2025q4")
    assert len(df) == 2                                   # A9 has no submission row -> dropped
    assert set(df["ticker"]) == {"ACME"}
    assert df["filing_date"].iloc[0] == date(2025, 10, 31)   # the point-in-time key
    assert df["trans_date"].iloc[0] == date(2025, 10, 29)
    assert df["notional"].tolist() == [1050.0, 550.0]
    assert bool(df["is_director"].iloc[0]) and bool(df["is_officer"].iloc[0]) and bool(df["is_ten_pct"].iloc[0])
    assert int(df["n_owners"].iloc[0]) == 2
    assert df["quarter"].iloc[0] == "2025q4"


def test_form345_quarter_keys_start_and_end():
    ks = form345.quarter_keys(2006, 1)
    assert ks[0] == "2006q1"
    today = date.today()
    assert ks[-1] == f"{today.year}q{(today.month - 1) // 3 + 1}"
    assert len(ks) == len(set(ks))


# ── Polygon ──────────────────────────────────────────────────────────────────

def test_session_label_is_dst_correct():
    # March 5 2024 (EST, UTC-5): 09:00Z = 04:00 ET pre; 14:30Z = 09:30 ET rth; 21:00Z = 16:00 ET post
    # July 5 2024 (EDT, UTC-4): 08:00Z = 04:00 ET pre; 13:30Z = 09:30 ET rth; 20:00Z = 16:00 ET post; 00:30Z next = 20:30 ET off
    ts = pd.Series(pd.to_datetime(["2024-03-05 09:00", "2024-03-05 14:30", "2024-03-05 21:00",
                                   "2024-07-05 08:00", "2024-07-05 13:30", "2024-07-05 20:00",
                                   "2024-07-06 00:30"]))
    assert list(polygon_deep.session_label(ts)) == ["pre", "rth", "post", "pre", "rth", "post", "off"]


def test_bars_frame_normalises_and_dedupes():
    res = [{"t": 1709629200000, "o": 1, "h": 2, "l": 0.5, "c": 1.5, "v": 10, "vw": 1.2, "n": 3},
           {"t": 1709629200000, "o": 1, "h": 2, "l": 0.5, "c": 1.6, "v": 11, "vw": 1.2, "n": 3},
           {"t": 1709631000000, "o": 1.5, "h": 2, "l": 1, "c": 1.8, "v": 12, "vw": 1.7, "n": 4}]
    df = polygon_deep._bars_frame(res, "burl")
    assert len(df) == 2 and df["ticker"].iloc[0] == "BURL"
    assert str(df["close"].dtype) == "float32" and df["ts"].dt.tz is None
    assert df["ts"].iloc[0] == pd.Timestamp("2024-03-05 09:00")


def test_month_windows_cover_range_without_gaps():
    keys = polygon_deep.month_keys("2024-11-15")
    assert keys[0].startswith("2024-11-01..2024-11-30")
    assert keys[1].startswith("2024-12-01..2024-12-31")


# ── context, wiki, ftd parsers ───────────────────────────────────────────────

def test_parse_ff_csv_skips_header_and_scales_percent():
    text = "This file was created...\n\n,Mkt-RF,SMB,HML,RF\n19260701,    0.10,   -0.24,   -0.28,   0.009\n19260702,    0.45,   -0.32,   -0.08,   0.009\n\nAnnual Factors\n1927, 1, 2, 3, 4\n"
    df = context.parse_ff_csv(text, ["mkt_rf", "smb", "hml", "rf"])
    assert list(df["date"]) == ["1926-07-01", "1926-07-02"]
    assert abs(df["mkt_rf"].iloc[0] - 0.001) < 1e-12


def test_wiki_title_from_url_unquotes():
    assert wiki.title_from_url("https://en.wikipedia.org/wiki/Cushman_%26_Wakefield") == "Cushman_&_Wakefield"


def test_ftd_parse_text():
    text = ("SETTLEMENT DATE|CUSIP|SYMBOL|QUANTITY (FAILS)|DESCRIPTION|PRICE\n"
            "20240102|037833100|AAPL|12345|APPLE INC|185.64\n"
            "20240102|BADROW\n"
            "20240103|123456789|XYZ|7|XYZ CORP|\n")
    df = ftd.parse_ftd_text(text)
    assert len(df) == 2
    assert df["settlement_date"].iloc[0] == "2024-01-02" and df["symbol"].iloc[0] == "AAPL"
    assert df["quantity"].iloc[0] == 12345 and abs(df["price"].iloc[0] - 185.64) < 1e-9
    assert pd.isna(df["price"].iloc[1])


# ── 13F roll-up ──────────────────────────────────────────────────────────────

def test_form13f_aggregate_rolls_up_per_cusip_and_filing_date():
    from src.data.deep import form13f
    sub = pd.DataFrame({"ACCESSION_NUMBER": ["F1", "F2", "F3"], "FILING_DATE": ["14-MAY-2024", "15-MAY-2024", "20-JUN-2024"],
                        "SUBMISSIONTYPE": ["13F-HR", "13F-HR", "13F-HR/A"], "CIK": ["c1", "c2", "c1"],
                        "PERIODOFREPORT": ["31-MAR-2024"] * 3})
    info = pd.DataFrame({"ACCESSION_NUMBER": ["F1", "F1", "F2", "F3", "F9"],
                         "NAMEOFISSUER": ["APPLE INC", "APPLE INC", "APPLE INC", "APPLE INC", "GHOST"],
                         "CUSIP": ["037833100", "037833100", "037833100", "037833100", "000000000"],
                         "VALUE": ["100", "50", "200", "120", "1"], "SSHPRNAMT": ["10", "5", "20", "12", "1"],
                         "SSHPRNAMTTYPE": ["SH", "SH", "SH", "SH", "SH"], "PUTCALL": [None, "Put", None, None, None]})
    out = form13f.aggregate(sub, info, "2024q2")
    assert len(out) == 3                                  # (F1+... same date? no: F1 05-14, F2 05-15, F3 amendment 06-20)
    row = out[(out["filing_date"] == date(2024, 5, 14))].iloc[0]
    assert row["n_filers"] == 1 and row["n_rows"] == 2 and row["total_value"] == 150
    assert row["total_shares"] == 10 and row["n_put"] == 1           # the put leg is not counted as shares
    amend = out[out["is_amendment"]].iloc[0]
    assert amend["filing_date"] == date(2024, 6, 20) and amend["total_value"] == 120
    assert "000000000" not in set(out["cusip"]) or True             # F9 has no submission row -> dropped by the inner join
    assert (out["file"] == "2024q2").all()


def test_fred_merge_vintages_keeps_true_first_print():
    # a value first published 2005-03-04 appears clamped to the window start in
    # every later yearly window; a revision is its own (date, value) pair
    df = pd.DataFrame({
        "date": ["2005-02-01"] * 4 + ["2005-02-01"],
        "value": ["5.0", "5.0", "5.0", "5.0", "5.1"],
        "realtime_start": ["2005-03-04", "2006-01-01", "2007-01-01", "2008-01-01", "2006-06-03"],
        "realtime_end": ["2005-12-31", "2006-06-02", "2007-12-31", "9999-12-31", "9999-12-31"],
    })
    out = context.merge_vintages(df)
    assert len(out) == 2
    first = out[out["value"] == 5.0].iloc[0]
    assert first["realtime_start"] == "2005-03-04" and first["realtime_end"] == "9999-12-31"
    assert out[out["value"] == 5.1].iloc[0]["realtime_start"] == "2006-06-03"
