"""Form 3/4/5 from EDGAR directly — the LIVE tail of the `form345` bulk family.

**Why this exists.** The SEC's quarterly bulk insider data sets stop at
2026-03-31 while today is 2026-09-20: the bulk distribution lags roughly one
full quarter plus the running one. The FILINGS do not lag — a Form 4 is on
EDGAR within minutes of acceptance — so the bulk set is a convenience, not the
source. This module reads the source, emitting rows with the SAME SCHEMA as
``form345.parquet`` so history and live are one table.

**One generator, both directions** (the standing rule, `memory/news-backfill-fidelity-2026-09`):
``parse_submission`` is the only parser, and it is validated field-by-field against
the bulk set on accessions both cover (`tests/test_form4_live.py`, and the
``--validate`` CLI reports the agreement rate on a fresh sample). If the two ever
disagree, the bulk set is the thing to re-derive, not this.

**Point-in-time is BETTER here than in the bulk set.** The bulk carries
FILING_DATE only (a date); the submission header carries
``<ACCEPTANCE-DATETIME>`` to the second, which is when the filing actually became
public. Both are emitted: ``filing_date`` for continuity with the bulk rows and
``acceptance`` for anything that needs the instant.

Enumeration is the EDGAR daily index (one request lists every filing that day,
market-wide); each filing is then one ~5 KB request for the complete submission
text. A Form 4 is indexed under BOTH the issuer's and each reporting owner's
CIK, so index rows are de-duplicated on the accession.

CLI: ``python -m src.data.deep.form4_live --days 5 [--workers 4]``,
``--validate [--n 300]``, ``--catch-up`` (from the last bulk quarter to today).
"""
from __future__ import annotations

import re
from datetime import date, datetime, timedelta
from decimal import Decimal, ROUND_HALF_UP
from typing import Dict, List, Optional, Sequence
from xml.etree import ElementTree as ET

import pandas as pd
from loguru import logger

from src.data.deep import SEC_HEADERS, RateLimiter, http_get

_DAILY_IDX = "https://www.sec.gov/Archives/edgar/daily-index/{year}/QTR{qtr}/form.{ymd}.idx"
_ARCHIVE = "https://www.sec.gov/Archives/{path}"
_LIMITER = RateLimiter(0.12)                 # SEC fair access: 10 req/s, shared
FORMS = ("3", "4", "5", "3/A", "4/A", "5/A")

# The column list `form345.join_quarter` emits, so live rows concatenate with
# the bulk table unchanged. `acceptance` is appended (bulk has no such column).
BULK_COLUMNS: List[str] = [
    "ticker", "issuer_cik", "issuer_name", "accession", "form", "filing_date",
    "period_of_report", "trans_date", "trans_code", "acq_disp", "shares", "price",
    "notional", "shares_owned_after", "direct_indirect", "security_title", "timeliness",
    "is_director", "is_officer", "is_ten_pct", "is_other", "n_owners", "owner_cik",
    "owner_title", "aff_10b5_1", "quarter",
]
LIVE_COLUMNS: List[str] = BULK_COLUMNS + ["acceptance"]

_IDX_RE = re.compile(r"^(\S+(?:/A)?)\s{2,}(.*?)\s{2,}(\d+)\s+(\d{8})\s+(edgar/\S+)\s*$")


# ── enumeration ──────────────────────────────────────────────────────────────

def daily_index(d: date, forms: Sequence[str] = FORMS) -> pd.DataFrame:
    """Every ``forms`` filing EDGAR indexed on ``d``, de-duplicated on accession.

    Returns columns ``form, company, cik, filing_date, path, accession``. An
    empty frame for a weekend/holiday (EDGAR serves no index) — the caller
    cannot distinguish that from a quiet day and does not need to."""
    qtr = (d.month - 1) // 3 + 1
    url = _DAILY_IDX.format(year=d.year, qtr=qtr, ymd=d.strftime("%Y%m%d"))
    r = http_get(url, headers=SEC_HEADERS, timeout=120, limiter=_LIMITER)
    if r is None or r.status_code != 200:
        return pd.DataFrame(columns=["form", "company", "cik", "filing_date", "path", "accession"])
    want = {f.upper() for f in forms}
    rows = []
    for line in r.text.splitlines():
        m = _IDX_RE.match(line)
        if not m:
            continue
        form, company, cik, ymd, path = m.groups()
        if form.upper() not in want:
            continue
        accn = path.rsplit("/", 1)[-1].replace(".txt", "")
        rows.append({"form": form, "company": company.strip(), "cik": str(int(cik)).zfill(10),
                     "filing_date": datetime.strptime(ymd, "%Y%m%d").date(), "path": path,
                     "accession": accn})
    if not rows:
        return pd.DataFrame(columns=["form", "company", "cik", "filing_date", "path", "accession"])
    # a Form 4 is indexed under the issuer AND every reporting owner — one row per filing
    return pd.DataFrame(rows).drop_duplicates("accession").reset_index(drop=True)


# ── parsing ──────────────────────────────────────────────────────────────────

def _txt(el: Optional[ET.Element]) -> Optional[str]:
    if el is None:
        return None
    v = el.findtext("value")
    s = v if v is not None else (el.text or "")
    s = (s or "").strip()
    return s or None


def round2(v: Optional[float]) -> Optional[float]:
    """Two decimals, HALF-UP — the precision and the rounding direction the SEC's
    bulk data sets store.

    The XML carries full precision (a price prints as 156.9250, the bulk set as
    156.93), so live rows would otherwise be a hair finer than the 7.2M rows of
    history, and `round()` alone would still disagree on exact half-cents
    because Python rounds half-to-EVEN (1910567.545 -> .54, the SEC -> .55).
    Validated: with this, 399 of 400 sampled 2026q1 filings reproduce the bulk
    rows as an identical multiset; without it, price agrees on 83.7%."""
    if v is None:
        return None
    try:
        if pd.isna(v):
            return None
    except (TypeError, ValueError):
        pass
    return float(Decimal(repr(float(v))).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP))


def _num(el: Optional[ET.Element]) -> Optional[float]:
    s = _txt(el)
    if s is None:
        return None
    try:
        return round2(float(s.replace(",", "")))
    except ValueError:
        return None


def _flag(el: Optional[ET.Element], tag: str) -> bool:
    if el is None:
        return False
    s = (el.findtext(tag) or "").strip().lower()
    return s in ("1", "true")


def _iso(s: Optional[str]):
    if not s:
        return None
    try:
        return datetime.strptime(s[:10], "%Y-%m-%d").date()
    except ValueError:
        return None


def parse_submission(text: str, accession: str = "", filing_date=None) -> pd.DataFrame:
    """A complete EDGAR submission text file -> one row per NON-DERIVATIVE
    transaction, in the bulk schema. Pure: no network, testable.

    Holdings (``nonDerivativeHolding``) and the derivative tables are excluded,
    matching what `form345.join_quarter` reads (NONDERIV_TRANS only). **A Form 3
    or a derivative-only Form 4 therefore yields ZERO rows, and that is correct** —
    the bulk set omits them too. On 2026-05-21, 218 of 1,259 indexed filings
    carried no non-derivative transaction, so "fewer rows than filings" is the
    normal case, never evidence of a fetch problem. Owner
    flags are OR-ed across every reporting owner and ``n_owners`` counts the
    distinct owner CIKs — the same aggregation the bulk join performs."""
    m = re.search(r"<ownershipDocument>.*?</ownershipDocument>", text, re.S)
    if not m:
        return pd.DataFrame(columns=LIVE_COLUMNS)
    try:
        doc = ET.fromstring(m.group(0))
    except ET.ParseError as e:                           # noqa: BLE001
        logger.debug(f"[form4_live] {accession}: unparseable XML ({e})")
        return pd.DataFrame(columns=LIVE_COLUMNS)

    acc_m = re.search(r"<ACCEPTANCE-DATETIME>\s*(\d{14})", text)
    acceptance = (pd.Timestamp(datetime.strptime(acc_m.group(1), "%Y%m%d%H%M%S"))
                  if acc_m else pd.NaT)
    if filing_date is None:
        fd = re.search(r"FILED AS OF DATE:\s*(\d{8})", text)
        filing_date = datetime.strptime(fd.group(1), "%Y%m%d").date() if fd else None
    if not accession:
        am = re.search(r"ACCESSION NUMBER:\s*(\S+)", text)
        accession = am.group(1) if am else ""

    issuer = doc.find("issuer")
    issuer_cik = (issuer.findtext("issuerCik") or "").strip() if issuer is not None else ""
    issuer_name = (issuer.findtext("issuerName") or "").strip() if issuer is not None else ""
    ticker = ((issuer.findtext("issuerTradingSymbol") or "").strip().upper()
              if issuer is not None else "")

    owners = doc.findall("reportingOwner")
    is_dir = any(_flag(o.find("reportingOwnerRelationship"), "isDirector") for o in owners)
    is_off = any(_flag(o.find("reportingOwnerRelationship"), "isOfficer") for o in owners)
    is_ten = any(_flag(o.find("reportingOwnerRelationship"), "isTenPercentOwner") for o in owners)
    is_oth = any(_flag(o.find("reportingOwnerRelationship"), "isOther") for o in owners)
    owner_ciks = [(o.findtext("reportingOwnerId/rptOwnerCik") or "").strip() for o in owners]
    owner_ciks = [c for c in owner_ciks if c]
    # `owner_cik` / `owner_title` are the FIRST reporting owner, matching the bulk
    # join's `first` aggregation. On a multi-owner filing "first" is arbitrary on
    # both sides and the two orderings disagree (8 of 755 sampled rows) — read
    # `n_owners` and the OR-ed flags, never this, when n_owners > 1.
    first_rel = owners[0].find("reportingOwnerRelationship") if owners else None
    owner_title = (first_rel.findtext("officerTitle") or "").strip() if first_rel is not None else ""

    form = (doc.findtext("documentType") or "").strip()
    period = _iso(doc.findtext("periodOfReport"))
    aff = (doc.findtext("aff10b5One") or "").strip()

    rows = []
    for t in doc.findall("nonDerivativeTable/nonDerivativeTransaction"):
        amounts = t.find("transactionAmounts")
        coding = t.find("transactionCoding")
        shares = _num(amounts.find("transactionShares")) if amounts is not None else None
        price = _num(amounts.find("transactionPricePerShare")) if amounts is not None else None
        rows.append({
            "ticker": ticker,
            "issuer_cik": issuer_cik,
            "issuer_name": issuer_name,
            "accession": accession,
            "form": form,
            "filing_date": filing_date,
            "period_of_report": period,
            "trans_date": _iso(_txt(t.find("transactionDate"))),
            "trans_code": (coding.findtext("transactionCode") or "").strip() if coding is not None else None,
            "acq_disp": _txt(amounts.find("transactionAcquiredDisposedCode")) if amounts is not None else None,
            "shares": shares,
            "price": price,
            "notional": float((shares or 0.0) * (price or 0.0)),   # from the rounded parts, as the bulk join does
            "shares_owned_after": _num(t.find("postTransactionAmounts/sharesOwnedFollowingTransaction")),
            "direct_indirect": _txt(t.find("ownershipNature/directOrIndirectOwnership")),
            "security_title": _txt(t.find("securityTitle")),
            "timeliness": _txt(t.find("transactionTimeliness")),
            "is_director": is_dir, "is_officer": is_off, "is_ten_pct": is_ten, "is_other": is_oth,
            "n_owners": len(set(owner_ciks)),
            "owner_cik": owner_ciks[0] if owner_ciks else None,
            "owner_title": owner_title or None,
            "aff_10b5_1": aff or None,
            "quarter": "live",
            "acceptance": acceptance,
        })
    if not rows:
        return pd.DataFrame(columns=LIVE_COLUMNS)
    return pd.DataFrame(rows)[LIVE_COLUMNS]


# ── fetching ─────────────────────────────────────────────────────────────────

_RATE_LIMITED = re.compile(r"request rate|automated tool|exceeded", re.I)


def _sec_get(url: str, tries: int = 4):
    """GET with an explicit retry on the SEC's RATE-LIMIT 403.

    `http_get` deliberately returns a 403 as-is (for entitlement errors, where a
    retry is pointless), but the SEC answers an over-rate requester with a 403
    whose BODY says so. Left alone that would drop filings silently — the
    failure mode this repo refuses — so it is distinguished from a real 403 by
    the body and backed off."""
    import time as _t
    for attempt in range(tries):
        r = http_get(url, headers=SEC_HEADERS, timeout=120, limiter=_LIMITER)
        if r is None:
            continue
        if r.status_code == 403 and _RATE_LIMITED.search(r.text[:2000] or ""):
            wait = 10.0 * (attempt + 1)
            logger.warning(f"[form4_live] SEC rate-limited; backing off {wait:.0f}s")
            _t.sleep(wait)
            continue
        return r
    return None


def fetch_filing(path: str, accession: str = "", filing_date=None) -> pd.DataFrame:
    """One filing's transactions. ``path`` is the daily index's
    ``edgar/data/<cik>/<accession>.txt``."""
    r = _sec_get(_ARCHIVE.format(path=path.lstrip("/")))
    if r is None or r.status_code != 200:
        raise RuntimeError(f"HTTP {getattr(r, 'status_code', None)}")
    return parse_submission(r.text, accession=accession, filing_date=filing_date)


def fetch_accession(cik: str, accession: str) -> pd.DataFrame:
    """One filing by (cik, accession) — the path the daily index would give."""
    cik_int = str(int(cik))
    return fetch_filing(f"edgar/data/{cik_int}/{accession}.txt", accession=accession)


def fetch_day(d: date, ciks: Optional[set] = None, workers: int = 4) -> pd.DataFrame:
    """Every Form 3/4/5 filed on ``d``; ``ciks`` (10-digit, zero-padded) limits
    the fetch to an issuer universe — the index row's CIK is the issuer's for
    the issuer-side row, so filtering before fetching is what keeps a tick cheap."""
    from concurrent.futures import ThreadPoolExecutor, as_completed
    idx = daily_index(d)
    if not len(idx):
        return pd.DataFrame(columns=LIVE_COLUMNS)
    if ciks is not None:
        idx = idx[idx["cik"].isin(ciks)]
        if not len(idx):
            return pd.DataFrame(columns=LIVE_COLUMNS)
    frames = []
    n_fail = n_gone = 0
    with ThreadPoolExecutor(max_workers=max(1, workers)) as ex:
        futs = {ex.submit(fetch_filing, r.path, r.accession, r.filing_date): r.accession
                for r in idx.itertuples()}
        for f in as_completed(futs):
            try:
                df = f.result()
            except Exception as e:                       # noqa: BLE001
                # A 404 on a document the index itself named means the filing was
                # WITHDRAWN after being indexed — verified 2026-09-20 on all three
                # that occurred in the Apr-Sep catch-up: gone from the directory
                # listing AND from the issuer's submissions history. That is correct
                # data, not a gap, so it is counted apart from real failures.
                if "404" in str(e):
                    n_gone += 1
                else:
                    n_fail += 1
                    logger.debug(f"[form4_live] {futs[f]}: {e}")
                continue
            if len(df):
                frames.append(df)
    if n_gone:
        logger.info(f"[form4_live] {d}: {n_gone} indexed filing(s) withdrawn from EDGAR (404) — expected")
    if n_fail:
        # a silently-short day is indistinguishable from a quiet one, so say it
        logger.warning(f"[form4_live] {d}: {n_fail} of {len(idx)} filings FAILED to fetch")
    if not frames:
        return pd.DataFrame(columns=LIVE_COLUMNS)
    return pd.concat(frames, ignore_index=True).sort_values(["filing_date", "accession"]).reset_index(drop=True)


# ── validation against the bulk set ──────────────────────────────────────────

_CMP_COLS = ["ticker", "issuer_cik", "form", "filing_date", "period_of_report", "trans_date",
             "trans_code", "acq_disp", "shares", "price", "shares_owned_after",
             "direct_indirect", "security_title", "is_director", "is_officer", "is_ten_pct",
             "is_other", "n_owners", "owner_cik", "aff_10b5_1"]


def compare_rows(bulk: pd.DataFrame, live: pd.DataFrame) -> dict:
    """Field-by-field agreement between bulk rows and re-parsed live rows for
    the SAME accessions. Pure. Rows are keyed on (accession, trans_date,
    trans_code, shares) because a filing can carry several transactions."""
    def key(df):
        return (df["accession"].astype(str) + "|" + df["trans_date"].astype(str) + "|"
                + df["trans_code"].astype(str) + "|" + df["shares"].round(4).astype(str))
    b = bulk.copy(); l = live.copy()
    b["_k"] = key(b); l["_k"] = key(l)
    b = b.drop_duplicates("_k").set_index("_k")
    l = l.drop_duplicates("_k").set_index("_k")
    common = b.index.intersection(l.index)
    out = {"bulk_rows": len(b), "live_rows": len(l), "matched_rows": len(common),
           "bulk_only": int(len(b) - len(common)), "live_only": int(len(l) - len(common)),
           "fields": {}}
    for c in _CMP_COLS:
        if c not in b.columns or c not in l.columns:
            continue
        bv, lv = b.loc[common, c], l.loc[common, c]
        if pd.api.types.is_bool_dtype(bv) or pd.api.types.is_bool_dtype(lv):
            same = bv.fillna(False).astype(bool) == lv.fillna(False).astype(bool)
        elif pd.api.types.is_numeric_dtype(bv) and pd.api.types.is_numeric_dtype(lv):
            same = ((bv - lv).abs() <= 1e-6) | (bv.isna() & lv.isna())
        else:
            bs = bv.astype(object).where(bv.notna(), None).astype(str).str.strip().str.upper()
            ls = lv.astype(object).where(lv.notna(), None).astype(str).str.strip().str.upper()
            same = (bs == ls) | (bv.isna() & lv.isna())
        out["fields"][c] = round(float(same.mean()), 6) if len(common) else None
    return out


def validate(n: int = 300, quarter: str = "2026q1", workers: int = 6) -> dict:
    """Re-fetch a random sample of accessions the BULK set already covers and
    compare. The whole live path rests on this number."""
    from concurrent.futures import ThreadPoolExecutor, as_completed
    import duckdb
    pq = "cache/ml/deep/form345.parquet"
    con = duckdb.connect()
    try:
        # sample AFTER the quarter filter and per ACCESSION: `USING SAMPLE` on the
        # scan would draw n rows out of the whole 7.2M-row table and then filter,
        # which silently returns a handful. hash() ordering keeps it reproducible.
        accs = con.sql(f"""SELECT accession, min(issuer_cik) AS issuer_cik
                           FROM '{pq}' WHERE quarter = '{quarter}'
                           GROUP BY accession ORDER BY hash(accession) LIMIT {int(n)}""").df()
        con.register("acc_", accs[["accession"]])
        bulk = con.sql(f"SELECT b.* FROM '{pq}' b JOIN acc_ a USING (accession) "
                       f"WHERE b.quarter = '{quarter}'").df()
    finally:
        con.close()
    frames = []
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(fetch_accession, r.issuer_cik, r.accession): r.accession
                for r in accs.itertuples()}
        for f in as_completed(futs):
            try:
                df = f.result()
            except Exception as e:                       # noqa: BLE001
                logger.warning(f"[form4_live] validate {futs[f]}: {e}")
                continue
            if len(df):
                frames.append(df)
    live = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=LIVE_COLUMNS)
    res = compare_rows(bulk, live)
    res["accessions_sampled"] = int(len(accs))
    res["accessions_fetched"] = int(len(frames))
    return res


# ── CLI ──────────────────────────────────────────────────────────────────────

def run_days(start: date, end: date, *, universe_only: bool = False, workers: int = 4,
             refetch: bool = False, deadline: Optional[float] = None) -> dict:
    """Fetch every weekday in ``[start, end]`` whose day part is missing (all
    of them with ``refetch``) and consolidate when anything landed. The unit
    the CLI catch-up and the nightly refresh share.

    A day with no index leaves NO part so it is asked again (a holiday costs
    one request per run until a newer part exists; an EDGAR outage is
    recovered). ``deadline`` (unix time) stops before the next day."""
    import time as _t
    from src.data.deep import consolidate, family_dir, write_parquet
    ciks = None
    if universe_only:
        from src.data.deep import deep_universe
        from src.data.deep.sec import cik_map
        m = cik_map()
        ciks = {m[t] for t in deep_universe() if t in m}
        logger.info(f"[form4_live] issuer filter: {len(ciks)} CIKs")
    parts = family_dir("form345_live") / "parts"
    total = skipped = fetched = empty = 0
    stopped = False
    t0 = datetime.now()
    d = start
    while d <= end:
        if deadline is not None and _t.time() > deadline:
            stopped = True
            break
        if d.weekday() < 5:
            part = parts / f"{d.isoformat()}.parquet"
            # resumable: a finished day is never re-fetched unless asked
            if part.exists() and not refetch:
                skipped += 1
            else:
                df = fetch_day(d, ciks=ciks, workers=workers)
                if len(df):
                    write_parquet(df, part)
                    total += len(df)
                    fetched += 1
                else:                       # a holiday has no index; leave no part so it retries
                    empty += 1
                    logger.info(f"[form4_live] {d}: no filings (holiday?)")
                el = (datetime.now() - t0).total_seconds()
                logger.info(f"[form4_live] {d}: {len(df):,} transactions "
                            f"(total {total:,}, {skipped} days already on disk, {el:.0f}s)")
        d += timedelta(days=1)
    if fetched:
        consolidate("form345_live")
    return {"days": fetched, "skipped": skipped, "empty_days": empty, "transactions": total,
            "seconds": round((datetime.now() - t0).total_seconds(), 1), "budget_stop": stopped}


def main(argv=None) -> None:
    import argparse
    import json
    from src.data.deep import DEEP_DIR, consolidate, family_dir, write_parquet

    ap = argparse.ArgumentParser(description="Form 3/4/5 live tail from EDGAR")
    ap.add_argument("--days", type=int, default=0, help="fetch the last N calendar days")
    ap.add_argument("--since", default="", help="ISO date to fetch from (overrides --days)")
    ap.add_argument("--catch-up", action="store_true",
                    help="from the day after the bulk table's last filing_date to today")
    ap.add_argument("--universe-only", action="store_true",
                    help="fetch only issuers in the deep universe (cheap; default is every filer)")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--rate", type=float, default=0.0,
                    help="seconds between SEC requests (default 0.12 = 10/s, the SEC ceiling). "
                         "Raise it to leave headroom for the live pipeline's own EDGAR calls.")
    ap.add_argument("--refetch", action="store_true", help="re-fetch days whose part already exists")
    ap.add_argument("--include-today", action="store_true",
                    help="also ask for today's index (default stops at yesterday: a day part is "
                         "never re-fetched, and today's index is final only once the day is over)")
    ap.add_argument("--validate", action="store_true")
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--quarter", default="2026q1")
    a = ap.parse_args(argv)

    logger.add("logs/deep_form4_live.log", rotation="1 day", retention="14 days", level="INFO")
    if a.rate and a.rate > 0:
        # the limiter is module-global and shared across worker threads, so one
        # value bounds the whole sweep regardless of --workers
        _LIMITER.min_interval = float(a.rate)
        logger.info(f"[form4_live] SEC request spacing {a.rate}s (~{1/a.rate:.1f}/s)")

    if a.validate:
        print(json.dumps(validate(n=a.n, quarter=a.quarter, workers=a.workers), indent=2, default=str))
        return

    start = None
    if a.catch_up:
        import duckdb
        con = duckdb.connect()
        try:
            last = con.sql("SELECT max(filing_date) FROM 'cache/ml/deep/form345.parquet' "
                           "WHERE quarter <> 'live'").fetchone()[0]
        finally:
            con.close()
        # the bulk parquet stores dates as VARCHAR (write_parquet stringifies
        # datetime.date objects), so accept either
        if isinstance(last, str):
            last = date.fromisoformat(last[:10])
        start = last + timedelta(days=1)
        logger.info(f"[form4_live] bulk ends {last}; catching up from {start}")
    elif a.since:
        start = date.fromisoformat(a.since)
    elif a.days:
        start = date.today() - timedelta(days=a.days)
    else:
        raise SystemExit("one of --days / --since / --catch-up / --validate is required")

    end = date.today() if a.include_today else date.today() - timedelta(days=1)
    r = run_days(start, end, universe_only=a.universe_only, workers=a.workers, refetch=a.refetch)
    logger.info(f"[form4_live] {start} .. {end}: {r}")
    if not r["days"] and not (DEEP_DIR / "form345_live.parquet").exists() and any(
            (family_dir("form345_live") / "parts").glob("*.parquet")):
        consolidate("form345_live")


if __name__ == "__main__":
    main()
