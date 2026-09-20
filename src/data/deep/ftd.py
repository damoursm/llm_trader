"""SEC fails-to-deliver — bi-monthly files since 2004 (434 zips on the index
page, paths vary by era so the page is parsed rather than the name guessed).

Each row: settlement date, CUSIP, symbol, fails quantity, description, price.
Two uses: a settlement-stress feature per name, and a free CUSIP ↔ symbol map
(13F holdings are keyed by CUSIP). Publication lags the period by ~2–4 weeks;
the feature builder applies that lag on the settlement date.
"""
from __future__ import annotations

import io
import re
import zipfile
from typing import Dict, List

import pandas as pd
from loguru import logger

from src.data.deep import SEC_HEADERS, RateLimiter, family_dir, http_get

_INDEX = "https://www.sec.gov/data-research/sec-markets-data/fails-deliver-data"
_LIMITER = RateLimiter(0.3)
_links: Dict[str, str] = {}


def file_links() -> Dict[str, str]:
    """{stem: absolute url} for every fails zip on the index page."""
    global _links
    if _links:
        return _links
    r = http_get(_INDEX, headers=SEC_HEADERS, timeout=60, limiter=_LIMITER)
    if r is None or r.status_code != 200:
        raise RuntimeError(f"index HTTP {getattr(r, 'status_code', None)}")
    out = {}
    for href in re.findall(r'href="([^"]*fails[^"]*\.zip)"', r.text):
        url = href if href.startswith("http") else f"https://www.sec.gov{href}"
        stem = href.rsplit("/", 1)[-1].replace(".zip", "")
        out[stem] = url
    _links = out
    return out


def parse_ftd_text(text: str) -> pd.DataFrame:
    rows: List[list] = []
    for line in text.splitlines():
        parts = line.split("|")
        if len(parts) < 5 or not parts[0].strip().isdigit():
            continue
        rows.append(parts[:6] + [""] * (6 - len(parts[:6])))
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows, columns=["settlement_date", "cusip", "symbol", "quantity", "description", "price"])
    df["settlement_date"] = pd.to_datetime(df["settlement_date"].str.strip(), format="%Y%m%d",
                                           errors="coerce").dt.date.astype(str)
    df["cusip"] = df["cusip"].str.strip()
    df["symbol"] = df["symbol"].str.strip().str.upper()
    df["quantity"] = pd.to_numeric(df["quantity"].str.strip(), errors="coerce")
    df["price"] = pd.to_numeric(df["price"].str.strip(), errors="coerce")
    df["description"] = df["description"].str.strip()
    return df[df["settlement_date"] != "NaT"].reset_index(drop=True)


def fetch_file(stem: str) -> pd.DataFrame:
    url = file_links().get(stem)
    if not url:
        raise RuntimeError("unknown file")
    raw_dir = family_dir("ftd") / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    p = raw_dir / f"{stem}.zip"
    if not p.exists() or p.stat().st_size < 200:
        r = http_get(url, headers=SEC_HEADERS, timeout=180, limiter=_LIMITER)
        if r is None or r.status_code != 200:
            raise RuntimeError(f"HTTP {getattr(r, 'status_code', None)}")
        tmp = p.with_suffix(".zip.tmp")
        tmp.write_bytes(r.content)
        tmp.replace(p)
    with zipfile.ZipFile(p) as z:
        name = next((n for n in z.namelist() if not n.endswith("/")), None)
        text = z.read(name).decode("latin-1", "replace") if name else ""
    df = parse_ftd_text(text)
    if len(df):
        df["file"] = stem
    logger.info(f"[deep.ftd] {stem}: {len(df):,} rows")
    return df
