"""SEC EDGAR families — the two free per-company feeds with the deepest
point-in-time history in the system.

* ``sec_filings`` — every filing a company ever made (``data.sec.gov/submissions``,
  the recent block plus the older overflow files, back to the 1990s): form type,
  filing date, ACCEPTANCE instant (UTC) and, for 8-Ks, the item codes. Item 2.02
  acceptance is the true earnings-release instant; 5.02 officer departures, 1.01
  material agreements, SC 13D/G activist stakes, 424B offerings, 10-Q/10-K
  filing instants all live here.
* ``companyfacts`` — every XBRL fact for a whitelist of ~50 standard concepts
  (``data.sec.gov/api/xbrl/companyfacts``) with the ``filed`` date per value, so a
  fundamental ratio can be computed as it was KNOWN on a date, restatements
  included as later rows rather than overwrites.

SEC fair-access policy: 10 requests/s with a descriptive User-Agent — one
shared limiter at 0.12 s across all worker threads.
"""
from __future__ import annotations

from typing import Dict, List, Optional

import pandas as pd
from loguru import logger

from src.data.deep import SEC_HEADERS, RateLimiter, http_get

_SUBM_URL = "https://data.sec.gov/submissions/{name}"
_FACTS_URL = "https://data.sec.gov/api/xbrl/companyfacts/CIK{cik10}.json"
_LIMITER = RateLimiter(0.12)

# XBRL concepts kept from companyfacts (us-gaap unless prefixed). Broad enough
# for value / quality / growth / accrual / payout / leverage / size factors.
CONCEPTS: List[str] = [
    "Revenues", "RevenueFromContractWithCustomerExcludingAssessedTax", "SalesRevenueNet",
    "CostOfRevenue", "GrossProfit", "OperatingIncomeLoss", "OperatingExpenses",
    "NetIncomeLoss", "ProfitLoss", "ComprehensiveIncomeNetOfTax",
    "IncomeLossFromContinuingOperationsBeforeIncomeTaxesExtraordinaryItemsNoncontrollingInterest",
    "EarningsPerShareBasic", "EarningsPerShareDiluted",
    "WeightedAverageNumberOfSharesOutstandingBasic", "WeightedAverageNumberOfDilutedSharesOutstanding",
    "Assets", "AssetsCurrent", "Liabilities", "LiabilitiesCurrent",
    "StockholdersEquity", "StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest",
    "CashAndCashEquivalentsAtCarryingValue", "LongTermDebt", "LongTermDebtNoncurrent",
    "LongTermDebtCurrent", "DebtCurrent", "ShortTermBorrowings",
    "NetCashProvidedByUsedInOperatingActivities", "NetCashProvidedByUsedInInvestingActivities",
    "NetCashProvidedByUsedInFinancingActivities", "PaymentsToAcquirePropertyPlantAndEquipment",
    "PaymentsForRepurchaseOfCommonStock", "PaymentsOfDividends", "PaymentsOfDividendsCommonStock",
    "DepreciationDepletionAndAmortization", "ResearchAndDevelopmentExpense",
    "SellingGeneralAndAdministrativeExpense", "InterestExpense", "IncomeTaxExpenseBenefit",
    "InventoryNet", "AccountsReceivableNetCurrent", "AccountsPayableCurrent", "Goodwill",
    "IntangibleAssetsNetExcludingGoodwill", "PropertyPlantAndEquipmentNet",
    "CommonStockSharesOutstanding", "CommonStockSharesIssued", "TreasuryStockValue",
    "RetainedEarningsAccumulatedDeficit", "StockIssuedDuringPeriodValueNewIssues",
    "ShareBasedCompensation", "DeferredRevenueCurrent",
    "dei:EntityCommonStockSharesOutstanding", "dei:EntityPublicFloat",
]
_CONCEPT_SET = {c.split(":", 1)[-1] for c in CONCEPTS if not c.startswith("dei:")}
_DEI_SET = {c.split(":", 1)[-1] for c in CONCEPTS if c.startswith("dei:")}


def cik_map() -> Dict[str, str]:
    """ticker -> 10-digit CIK from the SEC's current registrant list (the
    repo's existing loader; delisted / renamed names are not in it)."""
    from src.data.eight_k import _load_ticker_cik_map
    return _load_ticker_cik_map()


# ── filings ──────────────────────────────────────────────────────────────────

_FILING_COLS = ["accessionNumber", "filingDate", "reportDate", "acceptanceDateTime", "act",
                "form", "fileNumber", "items", "size", "isXBRL", "isInlineXBRL", "primaryDocument"]


def _filings_frame(block: dict, cik10: str) -> pd.DataFrame:
    """One filings block (the ``recent`` dict-of-arrays, or an older overflow
    file) -> tidy frame. Missing columns are tolerated (older files lack some)."""
    if not block or not block.get("accessionNumber"):
        return pd.DataFrame()
    n = len(block["accessionNumber"])
    data = {c: (block.get(c) if isinstance(block.get(c), list) and len(block.get(c)) == n
                else [None] * n) for c in _FILING_COLS}
    df = pd.DataFrame(data)
    df = df.rename(columns={"accessionNumber": "accession", "filingDate": "filing_date",
                            "reportDate": "report_date", "acceptanceDateTime": "acceptance",
                            "fileNumber": "file_number", "isXBRL": "is_xbrl",
                            "isInlineXBRL": "is_inline_xbrl", "primaryDocument": "primary_doc"})
    df.insert(0, "cik", cik10)
    df["filing_date"] = pd.to_datetime(df["filing_date"], errors="coerce").dt.date.astype(str)
    df["report_date"] = df["report_date"].replace("", None)
    df["acceptance"] = pd.to_datetime(df["acceptance"], errors="coerce", utc=True).dt.tz_localize(None)
    df["items"] = df["items"].fillna("").astype(str)
    df["form"] = df["form"].fillna("").astype(str)
    for c in ("size", "is_xbrl", "is_inline_xbrl"):
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def fetch_filings(cik10: str, ticker: str = "") -> pd.DataFrame:
    """Every filing for a CIK: the recent block + each older overflow file."""
    r = http_get(_SUBM_URL.format(name=f"CIK{cik10}.json"), headers=SEC_HEADERS,
                 timeout=60, limiter=_LIMITER)
    if r is None or r.status_code != 200:
        raise RuntimeError(f"submissions HTTP {getattr(r, 'status_code', None)}")
    j = r.json()
    frames = [_filings_frame((j.get("filings") or {}).get("recent") or {}, cik10)]
    for f in (j.get("filings") or {}).get("files") or []:
        name = f.get("name")
        if not name:
            continue
        r2 = http_get(_SUBM_URL.format(name=name), headers=SEC_HEADERS, timeout=60, limiter=_LIMITER)
        if r2 is None or r2.status_code != 200:
            logger.debug(f"[deep.sec] {cik10}: older file {name} HTTP {getattr(r2, 'status_code', None)}")
            continue
        frames.append(_filings_frame(r2.json(), cik10))
    df = pd.concat([x for x in frames if len(x)], ignore_index=True) if any(len(x) for x in frames) else pd.DataFrame()
    if len(df):
        df.insert(0, "ticker", ticker.upper())
        df = df.drop_duplicates("accession").sort_values("filing_date").reset_index(drop=True)
    return df


# ── company facts ────────────────────────────────────────────────────────────

def _facts_frame(j: dict, cik10: str) -> pd.DataFrame:
    """companyfacts JSON -> long frame over the concept whitelist, one row per
    (concept, unit, end, filed, accn). Keeps ``filed`` — the point-in-time key."""
    facts = (j or {}).get("facts") or {}
    rows: List[dict] = []
    for taxonomy, concepts in facts.items():
        keep = _DEI_SET if taxonomy == "dei" else (_CONCEPT_SET if taxonomy == "us-gaap" else set())
        if not keep:
            continue
        for concept, body in (concepts or {}).items():
            if concept not in keep:
                continue
            for unit, vals in ((body or {}).get("units") or {}).items():
                for v in vals or []:
                    rows.append({
                        "cik": cik10, "taxonomy": taxonomy, "concept": concept, "unit": unit,
                        "start": v.get("start"), "end": v.get("end"), "val": v.get("val"),
                        "accn": v.get("accn"), "fy": v.get("fy"), "fp": v.get("fp"),
                        "form": v.get("form"), "filed": v.get("filed"), "frame": v.get("frame"),
                    })
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    df["val"] = pd.to_numeric(df["val"], errors="coerce")
    df["fy"] = pd.to_numeric(df["fy"], errors="coerce")
    for c in ("start", "end", "filed", "accn", "fp", "form", "frame"):
        df[c] = df[c].astype(object).where(df[c].notna(), None)
    return df


def fetch_companyfacts(cik10: str, ticker: str = "") -> pd.DataFrame:
    r = http_get(_FACTS_URL.format(cik10=cik10), headers=SEC_HEADERS, timeout=120, limiter=_LIMITER)
    if r is None:
        raise RuntimeError("companyfacts: no response")
    if r.status_code == 404:                             # no XBRL facts (funds, some ADRs)
        return pd.DataFrame()
    if r.status_code != 200:
        raise RuntimeError(f"companyfacts HTTP {r.status_code}")
    df = _facts_frame(r.json(), cik10)
    if len(df):
        df.insert(0, "ticker", ticker.upper())
    return df


# ── per-ticker entry points for the runner ───────────────────────────────────

def filings_for_ticker(ticker: str) -> Optional[pd.DataFrame]:
    cik = cik_map().get(ticker.upper())
    if not cik:
        return pd.DataFrame()                            # no registrant: done with 0 rows
    return fetch_filings(cik, ticker)


def facts_for_ticker(ticker: str) -> Optional[pd.DataFrame]:
    cik = cik_map().get(ticker.upper())
    if not cik:
        return pd.DataFrame()
    return fetch_companyfacts(cik, ticker)
