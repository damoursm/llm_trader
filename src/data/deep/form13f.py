"""SEC Form 13F data sets — institutional holdings since 2013 Q2, aggregated
per security per filing date.

The raw INFOTABLE is one row per (filer, holding) — millions of rows a
quarter — so what is stored is the per-CUSIP roll-up that a feature needs:
for every (cusip, period_of_report, filing_date, amendment flag) the number of
filers, total reported value, total shares and the put/call counts. A
feature at date D sums the rows with ``filing_date <= D`` for the latest
period — the holdings became public on the filing date (45 days after
quarter end for on-time filers, later for amendments), never on the period.

CUSIP, not ticker: the FTD family's rows carry (CUSIP, SYMBOL) pairs, which is
the free join key. ``VALUE`` is in THOUSANDS of dollars through 2022 and in
DOLLARS from 2023 (the SEC changed the form) — normalise on shares × price
rather than trusting the unit.

File names vary by era (``2013q2_form13f.zip`` … ``01dec2024-28feb2025_form13f.zip``),
so the index page is parsed rather than guessed, like the FTD family.
"""
from __future__ import annotations

import csv
import io
import re
import zipfile
from typing import Dict, List

import pandas as pd
from loguru import logger

from src.data.deep import SEC_HEADERS, RateLimiter, family_dir, http_get

_INDEX = "https://www.sec.gov/data-research/sec-markets-data/form-13f-data-sets"
_LIMITER = RateLimiter(1.0)
_links: Dict[str, str] = {}


def file_links() -> Dict[str, str]:
    global _links
    if _links:
        return _links
    r = http_get(_INDEX, headers=SEC_HEADERS, timeout=60, limiter=_LIMITER)
    if r is None or r.status_code != 200:
        raise RuntimeError(f"index HTTP {getattr(r, 'status_code', None)}")
    out = {}
    for href in re.findall(r'href="([^"]*13f[^"]*\.zip)"', r.text, flags=re.I):
        url = href if href.startswith("http") else f"https://www.sec.gov{href}"
        stem = href.rsplit("/", 1)[-1].replace(".zip", "")
        out[stem] = url
    _links = out
    return out


def _read_tsv(z: zipfile.ZipFile, name: str, usecols: List[str]) -> pd.DataFrame:
    with z.open(name) as fh:
        raw = fh.read()
    return pd.read_csv(io.BytesIO(raw), sep="\t", dtype=str, quoting=csv.QUOTE_NONE,
                       encoding="utf-8", encoding_errors="replace", on_bad_lines="skip",
                       usecols=lambda c: c in usecols)


def aggregate(sub: pd.DataFrame, info: pd.DataFrame, stem: str = "") -> pd.DataFrame:
    """Per (cusip, period_of_report, filing_date, is_amendment) roll-up. Pure."""
    if sub is None or not len(sub) or info is None or not len(info):
        return pd.DataFrame()
    s = sub.rename(columns={"ACCESSION_NUMBER": "accession", "FILING_DATE": "filing_date",
                            "SUBMISSIONTYPE": "form", "CIK": "filer_cik",
                            "PERIODOFREPORT": "period_of_report"})
    s = s[[c for c in ("accession", "filing_date", "form", "filer_cik", "period_of_report") if c in s.columns]]
    s = s.drop_duplicates("accession")
    i = info.rename(columns={"ACCESSION_NUMBER": "accession", "NAMEOFISSUER": "issuer",
                             "CUSIP": "cusip", "VALUE": "value", "SSHPRNAMT": "amount",
                             "SSHPRNAMTTYPE": "amount_type", "PUTCALL": "putcall"})
    i = i[[c for c in ("accession", "issuer", "cusip", "value", "amount", "amount_type", "putcall") if c in i.columns]]
    df = i.merge(s, on="accession", how="inner")
    if not len(df):
        return pd.DataFrame()
    df["cusip"] = df["cusip"].fillna("").astype(str).str.strip().str.upper()
    df = df[df["cusip"].str.len() >= 8]
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    df["amount"] = pd.to_numeric(df["amount"], errors="coerce")
    df["is_amendment"] = df["form"].fillna("").astype(str).str.contains("/A")
    df["is_put"] = df["putcall"].fillna("").astype(str).str.upper().str.startswith("PUT")
    df["is_call"] = df["putcall"].fillna("").astype(str).str.upper().str.startswith("CALL")
    df["is_sh"] = df["amount_type"].fillna("SH").astype(str).str.upper().str.startswith("SH")
    df["shares"] = df["amount"].where(df["is_sh"] & ~df["is_put"] & ~df["is_call"])
    for c in ("filing_date", "period_of_report"):
        d = pd.to_datetime(df[c], format="%d-%b-%Y", errors="coerce")
        d = d.fillna(pd.to_datetime(df[c].where(d.isna()), errors="coerce"))
        df[c] = d.dt.date.astype(object).where(d.notna(), None)
    df = df[df["filing_date"].notna()]
    g = df.groupby(["cusip", "period_of_report", "filing_date", "is_amendment"], dropna=False)
    out = g.agg(issuer=("issuer", "first"), n_filers=("filer_cik", "nunique"),
                n_rows=("accession", "size"), total_value=("value", "sum"),
                total_shares=("shares", "sum"), n_put=("is_put", "sum"),
                n_call=("is_call", "sum")).reset_index()
    out["n_put"] = out["n_put"].astype(int)
    out["n_call"] = out["n_call"].astype(int)
    out["file"] = stem
    return out


def fetch_quarter(stem: str) -> pd.DataFrame:
    url = file_links().get(stem)
    if not url:
        raise RuntimeError("unknown file")
    raw_dir = family_dir("form13f") / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    p = raw_dir / f"{stem}.zip"
    if not p.exists() or p.stat().st_size < 1000:
        r = http_get(url, headers=SEC_HEADERS, timeout=900, limiter=_LIMITER)
        if r is None or r.status_code != 200:
            raise RuntimeError(f"HTTP {getattr(r, 'status_code', None)}")
        tmp = p.with_suffix(".zip.tmp")
        tmp.write_bytes(r.content)
        tmp.replace(p)
    with zipfile.ZipFile(p) as z:
        # members sit at the root in the quarterly-era zips and inside a folder
        # in the 2024+ monthly-range zips — match on the basename
        names = {n.rsplit("/", 1)[-1].upper(): n for n in z.namelist() if not n.endswith("/")}
        sub = _read_tsv(z, names["SUBMISSION.TSV"],
                        ["ACCESSION_NUMBER", "FILING_DATE", "SUBMISSIONTYPE", "CIK", "PERIODOFREPORT"])
        info = _read_tsv(z, names["INFOTABLE.TSV"],
                         ["ACCESSION_NUMBER", "NAMEOFISSUER", "CUSIP", "VALUE", "SSHPRNAMT",
                          "SSHPRNAMTTYPE", "PUTCALL"])
    df = aggregate(sub, info, stem)
    logger.info(f"[deep.form13f] {stem}: {len(info):,} holdings -> {len(df):,} cusip-filing rows")
    return df
