"""SEC bulk Form 3/4/5 data sets — every insider transaction since 2006 Q1.

Quarterly zips from ``sec.gov/files/structureddata/data/insider-transactions-data-sets``
(``<yyyy>q<n>_form345.zip``, 8–17 MB each). Three tables are joined here:
SUBMISSION (one row per filing: FILING_DATE, issuer CIK / symbol, form type),
NONDERIV_TRANS (one row per non-derivative transaction: date, code, shares,
price, acquired/disposed) and REPORTINGOWNER (who filed, with the
Director / Officer / TenPercentOwner relationship). One part per quarter,
raw zips kept under ``form345/raw/``.

Point-in-time key: ``filing_date`` (the day the Form 4 hit EDGAR), never
``trans_date`` — an insider has two business days to file, and a feature keyed
on the trade date would know the trade before the market did.

Transaction codes worth knowing: P open-market purchase, S open-market sale,
A grant/award, M option exercise, F tax withholding, G gift, D disposition to
issuer, J other. The classic signal is P vs S by officers/directors.
"""
from __future__ import annotations

import csv
import io
import zipfile
from datetime import date
from pathlib import Path
from typing import List

import pandas as pd
from loguru import logger

from src.data.deep import SEC_HEADERS, RateLimiter, family_dir, http_get

_URL = "https://www.sec.gov/files/structureddata/data/insider-transactions-data-sets/{q}_form345.zip"
_LIMITER = RateLimiter(0.5)


def quarter_keys(start_year: int = 2006, start_q: int = 1) -> List[str]:
    """'2006q1' .. the current quarter (whose zip may not exist yet — a 404 is
    recorded as failed and retried on the next run)."""
    today = date.today()
    out = []
    y, q = start_year, start_q
    while (y, q) <= (today.year, (today.month - 1) // 3 + 1):
        out.append(f"{y}q{q}")
        q += 1
        if q == 5:
            y, q = y + 1, 1
    return out


def _read_tsv(z: zipfile.ZipFile, name: str, usecols: List[str]) -> pd.DataFrame:
    with z.open(name) as fh:
        raw = fh.read()
    return pd.read_csv(io.BytesIO(raw), sep="\t", dtype=str, quoting=csv.QUOTE_NONE,
                       encoding="utf-8", encoding_errors="replace", on_bad_lines="skip",
                       usecols=lambda c: c in usecols)


def _to_date(s: pd.Series) -> pd.Series:
    d = pd.to_datetime(s, format="%d-%b-%Y", errors="coerce")
    d2 = pd.to_datetime(s.where(d.isna()), errors="coerce")
    return d.fillna(d2).dt.date.astype(object).where(lambda x: x.notna(), None)


def join_quarter(sub: pd.DataFrame, trans: pd.DataFrame, own: pd.DataFrame,
                 quarter: str = "") -> pd.DataFrame:
    """Join the three tables into one row per non-derivative transaction with
    the filing's date and the filers' relationship flags. Pure — testable."""
    if trans is None or not len(trans) or sub is None or not len(sub):
        return pd.DataFrame()
    s = sub.rename(columns={"ACCESSION_NUMBER": "accession", "FILING_DATE": "filing_date",
                            "PERIOD_OF_REPORT": "period_of_report", "DOCUMENT_TYPE": "form",
                            "ISSUERCIK": "issuer_cik", "ISSUERNAME": "issuer_name",
                            "ISSUERTRADINGSYMBOL": "ticker", "AFF10B5ONE": "aff_10b5_1"})
    keep_s = [c for c in ("accession", "filing_date", "period_of_report", "form", "issuer_cik",
                          "issuer_name", "ticker", "aff_10b5_1") if c in s.columns]
    s = s[keep_s].drop_duplicates("accession")
    t = trans.rename(columns={"ACCESSION_NUMBER": "accession", "TRANS_DATE": "trans_date",
                              "TRANS_CODE": "trans_code", "TRANS_SHARES": "shares",
                              "TRANS_PRICEPERSHARE": "price", "TRANS_ACQUIRED_DISP_CD": "acq_disp",
                              "SHRS_OWND_FOLWNG_TRANS": "shares_owned_after",
                              "DIRECT_INDIRECT_OWNERSHIP": "direct_indirect",
                              "SECURITY_TITLE": "security_title", "TRANS_TIMELINESS": "timeliness"})
    keep_t = [c for c in ("accession", "trans_date", "trans_code", "shares", "price", "acq_disp",
                          "shares_owned_after", "direct_indirect", "security_title", "timeliness")
              if c in t.columns]
    t = t[keep_t]
    if own is not None and len(own):
        o = own.rename(columns={"ACCESSION_NUMBER": "accession", "RPTOWNERCIK": "owner_cik",
                                "RPTOWNER_RELATIONSHIP": "rel", "RPTOWNER_TITLE": "owner_title"})
        rel = o["rel"].fillna("").astype(str)
        o = o.assign(is_director=rel.str.contains("Director", case=False),
                     is_officer=rel.str.contains("Officer", case=False),
                     is_ten_pct=rel.str.contains("TenPercent", case=False),
                     is_other=rel.str.contains("Other", case=False))
        agg = o.groupby("accession").agg(
            is_director=("is_director", "max"), is_officer=("is_officer", "max"),
            is_ten_pct=("is_ten_pct", "max"), is_other=("is_other", "max"),
            n_owners=("owner_cik", "nunique"),
            owner_cik=("owner_cik", "first"), owner_title=("owner_title", "first"),
        ).reset_index()
        t = t.merge(agg, on="accession", how="left")
    df = t.merge(s, on="accession", how="inner")
    if not len(df):
        return pd.DataFrame()
    df["ticker"] = df["ticker"].fillna("").astype(str).str.strip().str.upper()
    df["filing_date"] = _to_date(df["filing_date"])
    df["trans_date"] = _to_date(df["trans_date"])
    if "period_of_report" in df.columns:
        df["period_of_report"] = _to_date(df["period_of_report"])
    for c in ("shares", "price", "shares_owned_after"):
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    df["notional"] = (df["shares"].fillna(0.0) * df["price"].fillna(0.0)).astype(float)
    for c in ("is_director", "is_officer", "is_ten_pct", "is_other"):
        if c in df.columns:
            df[c] = df[c].fillna(False).astype(bool)
    if "n_owners" in df.columns:
        df["n_owners"] = pd.to_numeric(df["n_owners"], errors="coerce").fillna(0).astype(int)
    df["quarter"] = quarter
    df = df[df["filing_date"].notna()].reset_index(drop=True)
    cols = ["ticker", "issuer_cik", "issuer_name", "accession", "form", "filing_date",
            "period_of_report", "trans_date", "trans_code", "acq_disp", "shares", "price",
            "notional", "shares_owned_after", "direct_indirect", "security_title", "timeliness",
            "is_director", "is_officer", "is_ten_pct", "is_other", "n_owners", "owner_cik",
            "owner_title", "aff_10b5_1", "quarter"]
    return df[[c for c in cols if c in df.columns]]


def fetch_quarter(q: str) -> pd.DataFrame:
    """Download (or reuse the raw zip of) one quarter and return the joined rows."""
    raw_dir = family_dir("form345") / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    p = raw_dir / f"{q}_form345.zip"
    if not p.exists() or p.stat().st_size < 1000:
        r = http_get(_URL.format(q=q), headers=SEC_HEADERS, timeout=300, limiter=_LIMITER)
        if r is None:
            raise RuntimeError("no response")
        if r.status_code == 404:
            raise RuntimeError("404 (quarter not published yet)")
        if r.status_code != 200:
            raise RuntimeError(f"HTTP {r.status_code}")
        tmp = p.with_suffix(".zip.tmp")
        tmp.write_bytes(r.content)
        tmp.replace(p)
    with zipfile.ZipFile(p) as z:
        names = {n.rsplit("/", 1)[-1].upper(): n for n in z.namelist() if not n.endswith("/")}
        sub = _read_tsv(z, names["SUBMISSION.TSV"],
                        ["ACCESSION_NUMBER", "FILING_DATE", "PERIOD_OF_REPORT", "DOCUMENT_TYPE",
                         "ISSUERCIK", "ISSUERNAME", "ISSUERTRADINGSYMBOL", "AFF10B5ONE"])
        trans = _read_tsv(z, names["NONDERIV_TRANS.TSV"],
                          ["ACCESSION_NUMBER", "TRANS_DATE", "TRANS_CODE", "TRANS_SHARES",
                           "TRANS_PRICEPERSHARE", "TRANS_ACQUIRED_DISP_CD", "SHRS_OWND_FOLWNG_TRANS",
                           "DIRECT_INDIRECT_OWNERSHIP", "SECURITY_TITLE", "TRANS_TIMELINESS"])
        own = _read_tsv(z, names["REPORTINGOWNER.TSV"],
                        ["ACCESSION_NUMBER", "RPTOWNERCIK", "RPTOWNER_RELATIONSHIP", "RPTOWNER_TITLE"])
    df = join_quarter(sub, trans, own, quarter=q)
    logger.info(f"[deep.form345] {q}: {len(df):,} transactions from {len(sub):,} filings")
    return df
