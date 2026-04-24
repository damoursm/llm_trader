"""
SEC EDGAR S-1 / S-11 IPO pipeline intelligence.

Tracks registration statements filed with the SEC:
  S-1   — IPO registration for general companies
  S-1/A — amendment (company advancing toward IPO date)
  S-11  — REIT / real estate trust IPO registration
  S-11/A — REIT amendment

Signal interpretation:
  - High filing count in a sector → institutional underwriters see strong demand
    → positive macro read for that sector's ETF
  - Amendment wave → IPO window opening, risk appetite is elevated
  - Cold IPO market (few filings) → institutional caution, risk-off signal
  - Sector concentration of S-1s often leads sector rotation by 4–12 weeks

Tickers are NOT the IPO companies (not yet listed). The signal is about
which sectors are attracting institutional underwriter conviction.

Results cached daily. No API key required.
"""
from __future__ import annotations

import json
from collections import Counter
from datetime import date, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import httpx
from loguru import logger

from config import settings
from src.models import IPOFiling, IPOContext

_EDGAR_SEARCH = "https://efts.sec.gov/LATEST/search-index"
_HEADERS      = {"User-Agent": "llm-trader research@example.com"}
_TIMEOUT      = 20
_CACHE_DIR    = Path("cache")


# ---------------------------------------------------------------------------
# Sector classification
# ---------------------------------------------------------------------------

# SIC code → investment sector (checked first when SIC is available)
def _sic_to_sector(sic: int) -> str:
    if 2830 <= sic <= 2836 or 5910 <= sic <= 5912:
        return "Healthcare/Biotech"
    if 8000 <= sic <= 8099:
        return "Healthcare/Biotech"
    if 7370 <= sic <= 7379:
        return "Technology"
    if 3570 <= sic <= 3579 or 3670 <= sic <= 3679 or 3812 <= sic <= 3812:
        return "Technology"
    if 4800 <= sic <= 4899:
        return "Telecom"
    if 6000 <= sic <= 6411:
        return "Financials"
    if 6500 <= sic <= 6552 or 6726 <= sic <= 6726:
        return "Real Estate"
    if 1311 <= sic <= 1389 or 2900 <= sic <= 2999 or 4910 <= sic <= 4939:
        return "Energy"
    if 5000 <= sic <= 5999:
        return "Consumer/Retail"
    if 2000 <= sic <= 2111 or 5140 <= sic <= 5149:
        return "Food & Beverage"
    if 3000 <= sic <= 3569 or 3580 <= sic <= 3669 or 3700 <= sic <= 3812:
        return "Industrials"
    if 1000 <= sic <= 1499 or 2800 <= sic <= 2829:
        return "Materials/Chemicals"
    return "Other"


# Company name keyword classification (fallback when SIC absent)
_NAME_KEYWORDS: List[Tuple[str, List[str]]] = [
    ("Technology",         ["tech", "software", " ai", "ai ", "data", "cloud",
                            "cyber", "digital", "saas", "platform", "intelligence",
                            "compute", "analytic", "semiconductor", "chip"]),
    ("Healthcare/Biotech", ["pharma", "bio", "therapeutic", "health", "medical",
                            "oncol", "genomic", "clinical", "gene", "life science",
                            "medtech", "diagnostics"]),
    ("Financials",         ["financial", " bank", "capital", "payment", "fintech",
                            "insurance", "invest", "credit", "lending", "mortgage",
                            "asset management", "wealth"]),
    ("Energy",             ["energy", " oil", " gas", "solar", "wind", "renewable",
                            "power", "petroleum", "clean energy", "battery"]),
    ("Real Estate",        ["real estate", "realty", "reit", "properties", "property"]),
    ("Consumer/Retail",    ["brands", "consumer", "retail", "food", "beverage",
                            "apparel", "restaurant", "hospitality"]),
    ("Industrials",        ["industrial", "manufactur", "logistics", "transport",
                            "aerospace", "defense", "construction"]),
    ("Materials",          ["mining", " metal", "chemical", "materials", "resource"]),
    ("Telecom",            ["telecom", "wireless", "broadband", "fiber", "spectrum"]),
]


def _infer_sector(company: str, form_type: str, sic: Optional[int]) -> str:
    # S-11 is always a real estate trust
    if "S-11" in form_type:
        return "Real Estate"

    # SIC code takes priority when available
    if sic is not None:
        s = _sic_to_sector(sic)
        if s != "Other":
            return s

    # Keyword match on lowercased company name
    name_lower = company.lower()
    for sector, keywords in _NAME_KEYWORDS:
        if any(kw in name_lower for kw in keywords):
            return sector

    return "Other"


# ---------------------------------------------------------------------------
# Cache helpers
# ---------------------------------------------------------------------------

def _cache_path(today: date) -> Path:
    return _CACHE_DIR / f"ipo_{today.isoformat()}.json"


def _load_cache(today: date) -> Optional[dict]:
    path = _cache_path(today)
    if path.exists():
        try:
            return json.loads(path.read_text())
        except Exception:
            pass
    return None


def _save_cache(today: date, data: dict) -> None:
    _CACHE_DIR.mkdir(exist_ok=True)
    _cache_path(today).write_text(json.dumps(data, default=str))


# ---------------------------------------------------------------------------
# EDGAR fetch
# ---------------------------------------------------------------------------

def _fetch_filings(cutoff: date, today: date) -> List[dict]:
    """
    Fetch recent S-1/S-11 filings from EDGAR EFTS.
    Returns raw list of _source dicts from the search hits.
    """
    try:
        resp = httpx.get(
            _EDGAR_SEARCH,
            params={
                "forms":     "S-1,S-1/A,S-11,S-11/A",
                "dateRange": "custom",
                "startdt":   cutoff.isoformat(),
                "enddt":     today.isoformat(),
            },
            headers=_HEADERS,
            timeout=_TIMEOUT,
        )
        resp.raise_for_status()
        hits = resp.json().get("hits", {}).get("hits", [])
        return [hit.get("_source", {}) for hit in hits]
    except Exception as e:
        logger.warning(f"[ipo] EDGAR S-1/S-11 fetch failed: {e}")
        return []


def _parse_sic(src: dict) -> Optional[int]:
    """Extract SIC code from a filing source dict if present."""
    for key in ("sic", "sic_code", "sicCode"):
        val = src.get(key)
        if val is not None:
            try:
                return int(str(val))
            except (ValueError, TypeError):
                pass
    return None


def _parse_date(s: str) -> Optional[date]:
    try:
        return date.fromisoformat(s.strip())
    except (ValueError, AttributeError):
        return None


# ---------------------------------------------------------------------------
# Context assembly
# ---------------------------------------------------------------------------

def _build_context(filings: List[IPOFiling], amendments: List[IPOFiling], lookback_days: int) -> IPOContext:
    today = date.today()

    # Sector counts from initial filings only
    sector_counts: Dict[str, int] = dict(
        Counter(f.sector for f in filings).most_common()
    )
    hot_sectors = list(sector_counts.keys())[:3]

    # Summary
    total_new  = len(filings)
    total_amnd = len(amendments)

    if total_new == 0:
        activity = "The IPO pipeline is quiet"
    elif total_new >= 20:
        activity = "The IPO pipeline is highly active"
    elif total_new >= 10:
        activity = "The IPO pipeline is moderately active"
    else:
        activity = "The IPO pipeline shows modest activity"

    parts = [
        f"{activity} — {total_new} new S-1/S-11 registration(s) and "
        f"{total_amnd} amendment(s) filed in the last {lookback_days} days."
    ]

    if hot_sectors:
        top = ", ".join(
            f"{s} ({sector_counts[s]})" for s in hot_sectors
        )
        parts.append(f"Top sectors by filing count: {top}.")

    if total_amnd >= 5:
        parts.append(
            f"{total_amnd} amendments suggest an active pipeline maturing toward public offerings — "
            f"risk appetite among institutional underwriters is elevated."
        )

    summary = " ".join(parts)

    return IPOContext(
        filings         = filings,
        amendments      = amendments,
        sector_counts   = sector_counts,
        hot_sectors     = hot_sectors,
        total_new       = total_new,
        total_amendments= total_amnd,
        lookback_days   = lookback_days,
        report_date     = today,
        summary         = summary,
    )


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def fetch_ipo_context(lookback_days: int = 30) -> Optional[IPOContext]:
    """
    Fetch recent S-1/S-11 IPO filings from SEC EDGAR.
    Returns an IPOContext with sector-level intelligence.
    Results are cached daily — re-runs on the same day are instant.
    Returns None if the fetch fails entirely.
    """
    today  = date.today()
    cutoff = today - timedelta(days=lookback_days)

    # Cache check
    cached = _load_cache(today)
    if cached is not None:
        logger.info(f"[ipo] Using cached IPO pipeline data ({today})")
        try:
            filings    = [IPOFiling(**f) for f in cached.get("filings",    [])]
            amendments = [IPOFiling(**a) for a in cached.get("amendments", [])]
            return _build_context(filings, amendments, lookback_days)
        except Exception as e:
            logger.warning(f"[ipo] Cache parse error: {e}")

    logger.info(f"[ipo] Fetching S-1/S-11 filings from EDGAR (last {lookback_days} days)...")
    raw_sources = _fetch_filings(cutoff, today)

    if not raw_sources:
        logger.warning("[ipo] No S-1/S-11 filings returned from EDGAR")
        return None

    filings:    List[IPOFiling] = []
    amendments: List[IPOFiling] = []
    seen: set = set()   # deduplicate by (company, date, form)

    for src in raw_sources:
        form_type = src.get("form_type", "").strip()
        if not form_type:
            continue

        filed = _parse_date(src.get("file_date", ""))
        if filed is None:
            continue

        company = src.get("entity_name", "").strip()
        if not company:
            display = src.get("display_names") or []
            company = (display[0] if isinstance(display, list) and display
                       else str(display) if display else "Unknown")

        key = (company.lower(), filed, form_type)
        if key in seen:
            continue
        seen.add(key)

        sic    = _parse_sic(src)
        sector = _infer_sector(company, form_type, sic)
        is_amendment = form_type.endswith("/A")

        filing = IPOFiling(
            company      = company[:120],
            filing_date  = filed,
            form_type    = form_type,
            sector       = sector,
            is_amendment = is_amendment,
        )

        if is_amendment:
            amendments.append(filing)
        else:
            filings.append(filing)

    # Sort newest first
    filings.sort(key=lambda f: f.filing_date, reverse=True)
    amendments.sort(key=lambda f: f.filing_date, reverse=True)

    logger.info(
        f"[ipo] {len(filings)} new registration(s), {len(amendments)} amendment(s) | "
        f"top sectors: {list(Counter(f.sector for f in filings).most_common(3))}"
    )

    # Cache raw data
    _save_cache(today, {
        "filings":    [f.model_dump() for f in filings],
        "amendments": [a.model_dump() for a in amendments],
    })

    return _build_context(filings, amendments, lookback_days)
