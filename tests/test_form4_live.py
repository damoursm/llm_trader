"""`src/data/deep/form4_live.py` — offline tests of the live Form 3/4/5 path.

The live tail exists because the SEC's quarterly bulk insider sets lag by a
quarter or more while the filings themselves are public within minutes. Its
whole claim is that it reproduces the bulk schema and values, so these tests pin
the parser, the rounding that makes the two eras comparable, and the daily-index
de-duplication. No network.
"""
from __future__ import annotations

from datetime import date

import pandas as pd
import pytest

from src.data.deep import form4_live as F
from src.data.deep.form345 import join_quarter

_SUBMISSION = """<SEC-DOCUMENT>0001628280-26-062712.txt : 20260918
<SEC-HEADER>0001628280-26-062712.hdr.sgml : 20260918
<ACCEPTANCE-DATETIME>20260918160430
ACCESSION NUMBER:\t\t0001628280-26-062712
CONFORMED SUBMISSION TYPE:\t4
CONFORMED PERIOD OF REPORT:\t20260917
FILED AS OF DATE:\t\t20260918
</SEC-HEADER>
<XML>
<ownershipDocument>
    <schemaVersion>X0609</schemaVersion>
    <documentType>4</documentType>
    <periodOfReport>2026-09-17</periodOfReport>
    <issuer>
        <issuerCik>0000910638</issuerCik>
        <issuerName>3D SYSTEMS CORP</issuerName>
        <issuerTradingSymbol>ddd</issuerTradingSymbol>
    </issuer>
    <reportingOwner>
        <reportingOwnerId>
            <rptOwnerCik>0001251036</rptOwnerCik>
            <rptOwnerName>GRAVES JEFFREY A</rptOwnerName>
        </reportingOwnerId>
        <reportingOwnerRelationship>
            <isDirector>1</isDirector>
            <isOfficer>1</isOfficer>
            <isTenPercentOwner>0</isTenPercentOwner>
            <isOther>0</isOther>
            <officerTitle>President and CEO</officerTitle>
        </reportingOwnerRelationship>
    </reportingOwner>
    <reportingOwner>
        <reportingOwnerId>
            <rptOwnerCik>0009999999</rptOwnerCik>
            <rptOwnerName>SOME TRUST</rptOwnerName>
        </reportingOwnerId>
        <reportingOwnerRelationship>
            <isDirector>0</isDirector>
            <isOfficer>0</isOfficer>
            <isTenPercentOwner>1</isTenPercentOwner>
            <isOther>0</isOther>
        </reportingOwnerRelationship>
    </reportingOwner>
    <aff10b5One>1</aff10b5One>
    <nonDerivativeTable>
        <nonDerivativeTransaction>
            <securityTitle><value>Common Stock</value></securityTitle>
            <transactionDate><value>2026-09-17</value></transactionDate>
            <transactionCoding>
                <transactionFormType>4</transactionFormType>
                <transactionCode>S</transactionCode>
            </transactionCoding>
            <transactionAmounts>
                <transactionShares><value>1378</value></transactionShares>
                <transactionPricePerShare><value>156.9250</value></transactionPricePerShare>
                <transactionAcquiredDisposedCode><value>D</value></transactionAcquiredDisposedCode>
            </transactionAmounts>
            <postTransactionAmounts>
                <sharesOwnedFollowingTransaction><value>16050.206</value></sharesOwnedFollowingTransaction>
            </postTransactionAmounts>
            <ownershipNature>
                <directOrIndirectOwnership><value>D</value></directOrIndirectOwnership>
            </ownershipNature>
        </nonDerivativeTransaction>
        <nonDerivativeTransaction>
            <securityTitle><value>Common Stock</value></securityTitle>
            <transactionDate><value>2026-09-17</value></transactionDate>
            <transactionCoding><transactionCode>A</transactionCode></transactionCoding>
            <transactionAmounts>
                <transactionShares><value>500</value></transactionShares>
                <transactionPricePerShare><footnoteId id="F1"/></transactionPricePerShare>
                <transactionAcquiredDisposedCode><value>A</value></transactionAcquiredDisposedCode>
            </transactionAmounts>
            <ownershipNature>
                <directOrIndirectOwnership><value>I</value></directOrIndirectOwnership>
            </ownershipNature>
        </nonDerivativeTransaction>
        <nonDerivativeHolding>
            <securityTitle><value>Common Stock</value></securityTitle>
            <postTransactionAmounts>
                <sharesOwnedFollowingTransaction><value>999</value></sharesOwnedFollowingTransaction>
            </postTransactionAmounts>
        </nonDerivativeHolding>
    </nonDerivativeTable>
    <derivativeTable>
        <derivativeTransaction>
            <securityTitle><value>Stock Option</value></securityTitle>
            <transactionDate><value>2026-09-17</value></transactionDate>
            <transactionCoding><transactionCode>M</transactionCode></transactionCoding>
            <transactionAmounts>
                <transactionShares><value>7777</value></transactionShares>
            </transactionAmounts>
        </derivativeTransaction>
    </derivativeTable>
</ownershipDocument>
</XML>
</SEC-DOCUMENT>
"""


def test_parse_submission_fields_and_scope():
    df = F.parse_submission(_SUBMISSION)
    # non-derivative TRANSACTIONS only: the holding and the derivative leg are excluded,
    # matching what `form345.join_quarter` reads out of the bulk zips
    assert len(df) == 2
    assert 7777 not in set(df["shares"]) and 999 not in set(df["shares_owned_after"].dropna())

    r = df.iloc[0]
    assert r["ticker"] == "DDD"                       # upper-cased
    assert r["issuer_cik"] == "0000910638"
    assert r["accession"] == "0001628280-26-062712"
    assert r["form"] == "4"
    assert r["filing_date"] == date(2026, 9, 18)
    assert r["period_of_report"] == date(2026, 9, 17)
    assert r["trans_date"] == date(2026, 9, 17)
    assert r["trans_code"] == "S" and r["acq_disp"] == "D"
    assert r["direct_indirect"] == "D" and r["security_title"] == "Common Stock"
    assert r["aff_10b5_1"] == "1"
    # the acceptance INSTANT — what the bulk set does not carry
    assert r["acceptance"] == pd.Timestamp("2026-09-18 16:04:30")

    # flags are OR-ed across reporting owners; n_owners counts distinct CIKs
    assert bool(r["is_director"]) and bool(r["is_officer"]) and bool(r["is_ten_pct"])
    assert not bool(r["is_other"])
    assert int(r["n_owners"]) == 2
    assert r["owner_cik"] == "0001251036" and r["owner_title"] == "President and CEO"

    # a footnoted price carries no value -> None, and notional falls back to 0
    g = df.iloc[1]
    assert g["price"] is None or pd.isna(g["price"])
    assert g["notional"] == 0.0 and g["direct_indirect"] == "I"


def test_values_are_rounded_to_the_bulk_precision_half_up():
    df = F.parse_submission(_SUBMISSION)
    r = df.iloc[0]
    # XML carries 156.9250 / 16050.206; the bulk set stores 2 decimals
    assert r["price"] == 156.93
    assert r["shares_owned_after"] == 16050.21
    assert r["notional"] == pytest.approx(1378 * 156.93)
    # HALF-UP, not python's half-to-even: .545 -> .55 (the SEC's direction)
    assert F.round2(1910567.545) == 1910567.55
    assert F.round2(2104008.545) == 2104008.55
    assert F.round2(None) is None and F.round2(float("nan")) is None


def test_live_rows_match_the_bulk_schema_exactly():
    """A live row must concatenate with `form345.parquet` unchanged — only
    `acceptance` is added."""
    live = F.parse_submission(_SUBMISSION)
    sub = pd.DataFrame({"ACCESSION_NUMBER": ["A1"], "FILING_DATE": ["18-SEP-2026"],
                        "PERIOD_OF_REPORT": ["17-SEP-2026"], "DOCUMENT_TYPE": ["4"],
                        "ISSUERCIK": ["1"], "ISSUERNAME": ["X"], "ISSUERTRADINGSYMBOL": ["X"],
                        "AFF10B5ONE": ["0"]})
    trans = pd.DataFrame({"ACCESSION_NUMBER": ["A1"], "TRANS_DATE": ["17-SEP-2026"],
                          "TRANS_CODE": ["S"], "TRANS_SHARES": ["1"], "TRANS_PRICEPERSHARE": ["1"],
                          "TRANS_ACQUIRED_DISP_CD": ["D"], "SHRS_OWND_FOLWNG_TRANS": ["1"],
                          "DIRECT_INDIRECT_OWNERSHIP": ["D"], "SECURITY_TITLE": ["Common Stock"],
                          "TRANS_TIMELINESS": [None]})
    own = pd.DataFrame({"ACCESSION_NUMBER": ["A1"], "RPTOWNERCIK": ["o1"],
                        "RPTOWNER_RELATIONSHIP": ["Officer"], "RPTOWNER_TITLE": ["CEO"]})
    bulk = join_quarter(sub, trans, own, quarter="2026q3")
    assert list(bulk.columns) == F.BULK_COLUMNS
    assert list(live.columns) == F.BULK_COLUMNS + ["acceptance"]
    assert pd.concat([bulk, live], ignore_index=True).shape[0] == 3


def test_parse_submission_is_fail_soft_on_junk():
    for junk in ("", "not a filing", "<ownershipDocument><unclosed>"):
        df = F.parse_submission(junk)
        assert df.empty and list(df.columns) == F.LIVE_COLUMNS


def test_daily_index_parsing_dedupes_issuer_and_owner_rows(monkeypatch):
    """EDGAR indexes a Form 4 under the issuer AND every reporting owner, so the
    same accession appears several times; one filing must yield one row."""
    idx_text = (
        "Form Type   Company Name                    CIK         Date Filed  File Name\n"
        "---------------------------------------------------------------------------\n"
        "4           3D SYSTEMS CORP                 910638      20260918    edgar/data/910638/0001628280-26-062712.txt\n"
        "4           GRAVES JEFFREY A                1251036     20260918    edgar/data/1251036/0001628280-26-062712.txt\n"
        "3           ABATE ANTHONY                   1199933     20260918    edgar/data/1199933/0001749723-26-000153.txt\n"
        "4/A         SOME CORP                       111         20260918    edgar/data/111/0000000000-26-000001.txt\n"
        "8-K         IGNORE ME                       222         20260918    edgar/data/222/0000000000-26-000002.txt\n"
        "10-Q        ALSO IGNORE                     333         20260918    edgar/data/333/0000000000-26-000003.txt\n"
    )

    class _R:
        status_code = 200
        text = idx_text

    monkeypatch.setattr(F, "http_get", lambda *a, **k: _R())
    df = F.daily_index(date(2026, 9, 18))
    assert len(df) == 3                                   # the duplicate accession collapses
    assert set(df["form"]) == {"4", "3", "4/A"}           # 8-K / 10-Q filtered out
    assert set(df["accession"]) == {"0001628280-26-062712", "0001749723-26-000153",
                                    "0000000000-26-000001"}
    assert df["cik"].iloc[0] == "0000910638"              # zero-padded to 10
    assert df["filing_date"].iloc[0] == date(2026, 9, 18)


def test_daily_index_empty_on_weekend(monkeypatch):
    class _R:
        status_code = 404
        text = ""

    monkeypatch.setattr(F, "http_get", lambda *a, **k: _R())
    df = F.daily_index(date(2026, 9, 19))
    assert df.empty and "accession" in df.columns


def test_compare_rows_reports_per_field_agreement():
    bulk = pd.DataFrame({"accession": ["A", "A"], "trans_date": ["2026-01-02"] * 2,
                         "trans_code": ["P", "S"], "shares": [10.0, 20.0],
                         "price": [1.0, 2.0], "ticker": ["X", "X"]})
    live = bulk.copy()
    live.loc[1, "price"] = 2.5                            # one field, one row wrong
    res = F.compare_rows(bulk, live)
    assert res["matched_rows"] == 2 and res["bulk_only"] == 0 and res["live_only"] == 0
    assert res["fields"]["price"] == 0.5
    assert res["fields"]["ticker"] == 1.0


def test_form3_with_only_holdings_yields_no_rows():
    """A Form 3 reports HOLDINGS, not transactions, so it contributes nothing —
    the bulk set omits it too. 218 of 1,259 filings on a sampled day were like
    this, so "fewer rows than filings" must never read as a fetch failure."""
    doc = _SUBMISSION.split("<nonDerivativeTable>")[0] + """<nonDerivativeTable>
        <nonDerivativeHolding>
            <securityTitle><value>Common Stock</value></securityTitle>
            <postTransactionAmounts>
                <sharesOwnedFollowingTransaction><value>5000</value></sharesOwnedFollowingTransaction>
            </postTransactionAmounts>
        </nonDerivativeHolding>
    </nonDerivativeTable>
</ownershipDocument>
"""
    df = F.parse_submission(doc)
    assert df.empty and list(df.columns) == F.LIVE_COLUMNS
