"""
Fetch recent SEC 8-K material event filings for tracked tickers.

8-K filings are submitted within 4 business days of a triggering event and surface
material catalysts — earnings, M&A, leadership changes, restatements, cybersecurity
incidents — faster than RSS feeds pick them up.

Results are returned as NewsArticle objects and injected directly into the pipeline's
article list so the existing DeepSeek sentiment analysis scores them automatically.

No API key required. SEC rate limit: 10 requests/second (we use 0.12 s delay).
"""
from __future__ import annotations

import time
from datetime import date, datetime, timedelta, timezone
from typing import Dict, List, Optional

import httpx
from loguru import logger

from src.models import NewsArticle

_HEADERS      = {"User-Agent": "llm-trader research@example.com"}
_SUBM_URL     = "https://data.sec.gov/submissions/CIK{cik10}.json"
_TICKERS_URL  = "https://www.sec.gov/files/company_tickers.json"
_ARCHIVE_BASE = "https://www.sec.gov/Archives/edgar/data/{cik_int}/{accn_nodash}/{doc}"
_TIMEOUT      = 20
_REQUEST_DELAY = 0.12   # stay well within SEC's 10 req/s limit

# Cached across the pipeline run — built once from the SEC master list
_ticker_cik: Dict[str, str] = {}   # "AAPL" → "0000320193"


# ---------------------------------------------------------------------------
# Item code metadata
# ---------------------------------------------------------------------------

_ITEM_DESCRIPTIONS: Dict[str, str] = {
    "1.01": "Entry into Material Definitive Agreement",
    "1.02": "Termination of Material Definitive Agreement",
    "1.03": "Bankruptcy or Receivership",
    "1.04": "Mine Safety Reporting",
    "1.05": "Material Cybersecurity Incident",
    "2.01": "Completion of Acquisition or Disposition",
    "2.02": "Results of Operations and Financial Condition (Earnings)",
    "2.03": "Creation of Direct Financial Obligation",
    "2.04": "Triggering Events — Accelerated or Increased Obligation",
    "2.05": "Costs Associated with Exit or Disposal Activities (Restructuring/Layoffs)",
    "2.06": "Material Impairment",
    "3.01": "Notice of Delisting or Failure to Satisfy Listing Rule",
    "3.02": "Unregistered Sales of Securities (Dilution)",
    "3.03": "Material Modification to Rights of Security Holders",
    "4.01": "Changes in Registrant's Certifying Accountant",
    "4.02": "Non-Reliance on Previously Issued Financial Statements (Restatement)",
    "5.01": "Changes in Control of Registrant",
    "5.02": "Departure or Appointment of Principal Officers/Directors",
    "5.03": "Amendments to Articles or Bylaws",
    "5.08": "Shareholder Director Nominations",
    "7.01": "Regulation FD Disclosure",
    "8.01": "Other Events",
    "9.01": "Financial Statements and Exhibits",
}

# Lower number = higher priority when choosing which item to use as the title
_ITEM_PRIORITY: Dict[str, int] = {
    "1.03": 1,    # bankruptcy
    "4.02": 2,    # restatement
    "3.01": 3,    # delisting notice
    "2.04": 4,    # accelerated obligation/default
    "1.05": 5,    # cybersecurity incident
    "2.06": 6,    # material impairment
    "2.05": 7,    # restructuring/layoffs
    "5.01": 8,    # change of control
    "1.02": 9,    # material agreement termination
    "4.01": 10,   # auditor change (red flag)
    "3.02": 11,   # dilutive securities
    "2.01": 12,   # acquisition/disposition complete
    "2.02": 13,   # earnings release
    "1.01": 14,   # new material agreement
    "5.02": 15,   # leadership departure/appointment
    "2.03": 16,   # new financial obligation (debt)
    "7.01": 17,   # reg FD
    "8.01": 18,   # other events
    "9.01": 99,   # exhibits only — never use as title
}

# Items that warrant inclusion. 9.01 (exhibits attachment) is always skipped.
_SKIP_ITEMS = {"9.01"}

# Items with no meaningful directional signal (pure compliance filings)
_NOISE_ITEMS = {"5.03", "5.08", "1.04"}


# ---------------------------------------------------------------------------
# CIK lookup
# ---------------------------------------------------------------------------

def _load_ticker_cik_map() -> Dict[str, str]:
    """Build ticker → zero-padded CIK string. Cached for the process lifetime."""
    global _ticker_cik
    if _ticker_cik:
        return _ticker_cik
    try:
        resp = httpx.get(_TICKERS_URL, headers=_HEADERS, timeout=30)
        resp.raise_for_status()
        for entry in resp.json().values():
            ticker  = entry.get("ticker", "").strip().upper()
            cik_int = entry.get("cik_str", 0)
            if ticker and cik_int:
                _ticker_cik[ticker] = str(cik_int).zfill(10)
        logger.info(f"[8k] Loaded {len(_ticker_cik):,} ticker→CIK mappings from SEC")
    except Exception as e:
        logger.warning(f"[8k] Could not load ticker→CIK map: {e}")
    return _ticker_cik


# ---------------------------------------------------------------------------
# Submissions fetch and parse
# ---------------------------------------------------------------------------

def _fetch_submissions(cik10: str) -> Optional[dict]:
    try:
        resp = httpx.get(
            _SUBM_URL.format(cik10=cik10),
            headers=_HEADERS,
            timeout=_TIMEOUT,
        )
        resp.raise_for_status()
        return resp.json()
    except Exception as e:
        logger.debug(f"[8k] Submissions fetch failed for CIK {cik10}: {e}")
        return None


def _parse_items(raw_items: str) -> List[str]:
    """Parse a comma-separated item string like '2.02,9.01' into a filtered list."""
    return [
        i.strip()
        for i in raw_items.split(",")
        if i.strip() and i.strip() not in _SKIP_ITEMS and i.strip() not in _NOISE_ITEMS
    ]


def _primary_item(items: List[str]) -> Optional[str]:
    """Return the highest-priority item code from a list."""
    if not items:
        return None
    return min(items, key=lambda i: _ITEM_PRIORITY.get(i, 50))


def _build_article(
    ticker: str,
    company: str,
    cik_int: int,
    filing_date: date,
    items: List[str],
    accn: str,
    primary_doc: str,
) -> Optional[NewsArticle]:
    """Construct a NewsArticle from a single 8-K filing."""
    if not items:
        return None

    lead_item = _primary_item(items)
    if lead_item is None:
        return None

    lead_desc = _ITEM_DESCRIPTIONS.get(lead_item, f"Item {lead_item}")

    # Title: "{TICKER} 8-K — {Lead Event Description}"
    title = f"{ticker} SEC 8-K — {lead_desc}"

    # Summary: structured description of all material items
    all_descs = [
        f"Item {i}: {_ITEM_DESCRIPTIONS.get(i, i)}"
        for i in items
    ]
    summary = (
        f"{company} ({ticker}) filed SEC Form 8-K on {filing_date}. "
        f"Material events disclosed: {'; '.join(all_descs)}. "
        f"This filing was submitted within 4 business days of the triggering event "
        f"and may precede coverage in news feeds."
    )

    # Direct URL to the primary filing document when available
    if primary_doc:
        accn_nodash = accn.replace("-", "")
        url = _ARCHIVE_BASE.format(
            cik_int=cik_int,
            accn_nodash=accn_nodash,
            doc=primary_doc,
        )
    else:
        accn_nodash = accn.replace("-", "")
        url = f"https://www.sec.gov/Archives/edgar/data/{cik_int}/{accn_nodash}/"

    published_at = datetime(
        filing_date.year, filing_date.month, filing_date.day,
        tzinfo=timezone.utc,
    )

    return NewsArticle(
        title=title,
        summary=summary,
        url=url,
        source="SEC 8-K Filing",
        published_at=published_at,
    )


def _get_recent_8ks(
    ticker: str,
    cik10: str,
    lookback_days: int,
) -> List[NewsArticle]:
    """Return 8-K NewsArticle objects for a single ticker within the lookback window."""
    data = _fetch_submissions(cik10)
    if not data:
        return []

    company  = data.get("name", ticker)
    cik_int  = int(cik10)
    recent   = data.get("filings", {}).get("recent", {})
    cutoff   = date.today() - timedelta(days=lookback_days)
    articles: List[NewsArticle] = []

    forms       = recent.get("form", [])
    dates       = recent.get("filingDate", [])
    accns       = recent.get("accessionNumber", [])
    items_list  = recent.get("items", [])
    primary_docs= recent.get("primaryDocument", [])

    for i, form in enumerate(forms):
        if form not in ("8-K", "8-K/A"):
            continue

        raw_date = dates[i] if i < len(dates) else ""
        try:
            filing_date = date.fromisoformat(raw_date)
        except ValueError:
            continue

        if filing_date < cutoff:
            break   # submissions are newest-first; once past cutoff, stop

        raw_items = items_list[i] if i < len(items_list) else ""
        items     = _parse_items(raw_items)
        if not items:
            continue   # no material items (e.g., exhibit-only amendment)

        accn       = accns[i] if i < len(accns) else ""
        prim_doc   = primary_docs[i] if i < len(primary_docs) else ""

        article = _build_article(ticker, company, cik_int, filing_date, items, accn, prim_doc)
        if article:
            articles.append(article)

    return articles


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def fetch_8k_articles(tickers: List[str], lookback_days: int = 5) -> List[NewsArticle]:
    """
    Fetch recent 8-K filings for all tickers in the universe.
    Returns a list of NewsArticle objects ready to be appended to the
    pipeline's article list for sentiment scoring.

    Tickers with no known CIK (e.g. some ETFs, indices) are skipped silently.
    Rate-limited to stay within SEC's 10 req/s guideline.
    """
    cik_map  = _load_ticker_cik_map()
    articles: List[NewsArticle] = []
    seen_urls: set = set()     # deduplicate across tickers (8-K/A may duplicate)
    found = 0
    skipped = 0

    for ticker in tickers:
        cik10 = cik_map.get(ticker.upper())
        if not cik10:
            skipped += 1
            continue

        try:
            new_articles = _get_recent_8ks(ticker, cik10, lookback_days)
        except Exception as e:
            logger.debug(f"[8k] {ticker}: error — {e}")
            new_articles = []

        for a in new_articles:
            if a.url not in seen_urls:
                seen_urls.add(a.url)
                articles.append(a)
                found += 1

        time.sleep(_REQUEST_DELAY)

    logger.info(
        f"[8k] {found} 8-K article(s) from {len(tickers) - skipped} tickers "
        f"(lookback: {lookback_days}d, {skipped} tickers had no CIK)"
    )
    return articles
