"""
Smart money signals from SEC EDGAR public filings — three strategies:

  1. SC 13D / SC 13G  — activist and large institutional position disclosures (>5% ownership)
  2. Form 144          — planned insider sales filed before selling restricted shares
  3. Form 13F-HR       — quarterly holdings for tracked superinvestors (Buffett, Ackman, etc.)

Tickers are discovered directly from the filings using the SEC's own company tickers
index (company_tickers.json) — not limited to any predefined watchlist.

All data is public. No API key required.
"""
from __future__ import annotations

import json
import re
import time
import xml.etree.ElementTree as ET
from datetime import date, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import httpx
from loguru import logger

from config import settings
from src.models import InsiderTrade
from src.data.insider_trades import _parse_date, _notional_to_amount_range


_EDGAR_SEARCH        = "https://efts.sec.gov/LATEST/search-index"
_EDGAR_SUBM          = "https://data.sec.gov/submissions/CIK{cik10}.json"
_EDGAR_ARCHIVE       = "https://www.sec.gov/Archives/edgar/data/{cik}/{accn_nodash}/{accn_nodash}-index.json"
_COMPANY_TICKERS_URL = "https://www.sec.gov/files/company_tickers.json"
_HEADERS             = {"User-Agent": "llm-trader research@example.com"}
_REQUEST_DELAY       = 0.15   # seconds between EDGAR requests (rate limit)

# Module-level cache — populated once per process, reused across strategies
_ticker_index: Dict[str, str] = {}   # normalized_company_name → ticker


# ---------------------------------------------------------------------------
# Company name → ticker index  (SEC master list, ~13 k entries)
# ---------------------------------------------------------------------------

def _load_ticker_index() -> Dict[str, str]:
    """
    Download the SEC company tickers file and build a normalized name→ticker map.
    Cached in-process so it is only fetched once per pipeline run.
    """
    global _ticker_index
    if _ticker_index:
        return _ticker_index

    try:
        resp = httpx.get(_COMPANY_TICKERS_URL, headers=_HEADERS, timeout=30)
        resp.raise_for_status()
        for entry in resp.json().values():
            ticker = entry.get("ticker", "").strip().upper()
            title  = entry.get("title",  "").strip()
            if ticker and title:
                _ticker_index[_normalize_name(title)] = ticker
        logger.info(f"[sec] Loaded {len(_ticker_index):,} company name→ticker mappings from SEC")
    except Exception as e:
        logger.warning(f"[sec] Could not load company ticker index: {e}")

    return _ticker_index


def _normalize_name(name: str) -> str:
    """Strip legal suffixes, punctuation and whitespace for fuzzy matching."""
    name = name.upper()
    for suffix in [
        " INC", " CORP", " LTD", " LLC", " L.P.", " LP", " PLC",
        " N.V.", " NV", " S.A.", " SA", " HOLDINGS", " GROUP",
        " INTERNATIONAL", " TECHNOLOGIES", " TECHNOLOGY",
        " COMPANY", " CO", " CLASS A", " CLASS B", " COMMON",
    ]:
        name = name.replace(suffix, "")
    return re.sub(r"[.,]", "", name).strip()


def _ticker_from_name(name: str, index: Dict[str, str]) -> Optional[str]:
    """
    Map a company name to a ticker via the SEC index.
    Tries exact match first, then leading-token prefix match for truncated names.
    """
    if not name:
        return None
    norm = _normalize_name(name)
    if norm in index:
        return index[norm]
    # Prefix match — handles 13F truncated names like "ALPHABET INC CL A" → "GOOGL"
    for mapped, ticker in index.items():
        n = min(len(norm), len(mapped), 15)
        if n >= 5 and norm[:n] == mapped[:n]:
            return ticker
    return None


# ---------------------------------------------------------------------------
# Strategy 1: SC 13D / SC 13G  (activist and large passive stakes)
# ---------------------------------------------------------------------------

def fetch_activist_stakes() -> List[InsiderTrade]:
    """
    Fetch ALL recent SC 13D/13G filings from EDGAR EFTS.
    Discovers tickers directly from the filings — no predefined watchlist needed.

    In EDGAR, 13D/13G filings are indexed under the subject company (the entity
    whose shares are being accumulated), so entity_name maps to the stock ticker.
    """
    cutoff       = date.today() - timedelta(days=settings.sec_filings_lookback_days)
    ticker_index = _load_ticker_index()
    trades: List[InsiderTrade] = []

    try:
        resp = httpx.get(
            _EDGAR_SEARCH,
            params={
                "forms":     "SC 13D,SC 13G,SC 13D/A,SC 13G/A",
                "dateRange": "custom",
                "startdt":   cutoff.isoformat(),
                "enddt":     date.today().isoformat(),
            },
            headers=_HEADERS,
            timeout=20,
        )
        resp.raise_for_status()
        hits = resp.json().get("hits", {}).get("hits", [])

        for hit in hits:
            src       = hit.get("_source", {})
            form_type = src.get("form_type", "")
            filed     = _parse_date(src.get("file_date", ""))
            if not filed:
                continue

            entity = src.get("entity_name", "").strip()

            # entity_name = subject company (the stock being accumulated)
            ticker = _ticker_from_name(entity, ticker_index)

            # display_names may hold the filer (activist fund) name or an alternate company name
            display_names = src.get("display_names") or []
            if isinstance(display_names, str):
                display_names = [display_names]

            if not ticker:
                for dn in display_names:
                    ticker = _ticker_from_name(dn, ticker_index)
                    if ticker:
                        break

            if not ticker:
                continue

            # Best guess at the activist/institution name
            filer = display_names[0] if display_names else entity
            is_activist = "13D" in form_type

            trades.append(InsiderTrade(
                ticker=ticker,
                trader_name=filer[:80],
                trader_type="institutional",
                role="Activist Investor" if is_activist else "Passive Institutional",
                transaction_type="13d_activist_stake" if is_activist else "13g_passive_stake",
                amount_range=">5% ownership",
                transaction_date=filed,
                disclosure_date=filed,
                notes=f"SEC {form_type} — {entity}",
            ))
    except Exception as e:
        logger.warning(f"[sec] 13D/13G fetch failed: {e}")

    unique_tickers = len({t.ticker for t in trades})
    logger.info(f"[sec] 13D/13G: {len(trades)} filing(s) across {unique_tickers} tickers")
    return trades


# ---------------------------------------------------------------------------
# Strategy 2: Form 144  (planned insider sales)
# ---------------------------------------------------------------------------

def fetch_form144_sales() -> List[InsiderTrade]:
    """
    Fetch ALL recent Form 144 filings from EDGAR EFTS.
    Discovers tickers from the filings — no predefined watchlist needed.

    Form 144 is filed by an officer/director before selling restricted shares.
    EDGAR indexes these under the issuer company (the subject stock).
    """
    cutoff       = date.today() - timedelta(days=settings.sec_filings_lookback_days)
    ticker_index = _load_ticker_index()
    trades: List[InsiderTrade] = []

    try:
        resp = httpx.get(
            _EDGAR_SEARCH,
            params={
                "forms":     "144",
                "dateRange": "custom",
                "startdt":   cutoff.isoformat(),
                "enddt":     date.today().isoformat(),
            },
            headers=_HEADERS,
            timeout=20,
        )
        resp.raise_for_status()
        hits = resp.json().get("hits", {}).get("hits", [])

        for hit in hits:
            src   = hit.get("_source", {})
            filed = _parse_date(src.get("file_date", ""))
            if not filed:
                continue

            entity = src.get("entity_name", "").strip()
            ticker = _ticker_from_name(entity, ticker_index)

            display_names = src.get("display_names") or []
            if isinstance(display_names, str):
                display_names = [display_names]

            if not ticker:
                for dn in display_names:
                    ticker = _ticker_from_name(dn, ticker_index)
                    if ticker:
                        break

            if not ticker:
                continue

            filer = display_names[0] if display_names else "Insider"
            trades.append(InsiderTrade(
                ticker=ticker,
                trader_name=filer[:80],
                trader_type="corporate_insider",
                role="Officer/Director (Form 144)",
                transaction_type="planned_sale_144",
                amount_range="see filing",
                transaction_date=filed,
                disclosure_date=filed,
                notes=f"Form 144 — planned sale | {entity}",
            ))
    except Exception as e:
        logger.warning(f"[sec] Form 144 fetch failed: {e}")

    unique_tickers = len({t.ticker for t in trades})
    logger.info(f"[sec] Form 144: {len(trades)} filing(s) across {unique_tickers} tickers")
    return trades


# ---------------------------------------------------------------------------
# Strategy 3: Form 13F-HR  (superinvestor quarterly holdings)
# ---------------------------------------------------------------------------

def _lookup_institution_cik(name: str) -> Optional[str]:
    """
    Resolve the CIK for a tracked institution by finding their own 13F filings.
    The accession number prefix IS the filer's CIK:
      "0001067983-24-000042"  →  CIK = "0001067983"
    """
    try:
        resp = httpx.get(
            _EDGAR_SEARCH,
            params={
                "q":         f'"{name}"',
                "forms":     "13F-HR",
                "dateRange": "custom",
                "startdt":   "2020-01-01",
            },
            headers=_HEADERS,
            timeout=15,
        )
        resp.raise_for_status()
        hits = resp.json().get("hits", {}).get("hits", [])

        for hit in hits:
            entity    = hit.get("_source", {}).get("entity_name", "")
            accession = hit.get("_id", "")
            if name.lower() in entity.lower() and accession:
                parts = accession.split("-")
                if parts and parts[0].isdigit():
                    return parts[0]   # 10-digit zero-padded CIK
    except Exception as e:
        logger.debug(f"[13f] CIK lookup failed for '{name}': {e}")
    return None


def _get_recent_13f_accessions(cik: str, n: int = 2) -> List[Tuple[str, date]]:
    """Get the N most recent 13F-HR accession numbers for a CIK."""
    cik10 = cik.zfill(10)
    try:
        resp = httpx.get(
            _EDGAR_SUBM.format(cik10=cik10),
            headers=_HEADERS,
            timeout=20,
        )
        resp.raise_for_status()
        recent = resp.json().get("filings", {}).get("recent", {})
        results: List[Tuple[str, date]] = []
        for form, accn, dt in zip(
            recent.get("form", []),
            recent.get("accessionNumber", []),
            recent.get("filingDate", []),
        ):
            if form == "13F-HR" and len(results) < n:
                filed = _parse_date(dt)
                if filed:
                    results.append((accn, filed))
        return results
    except Exception as e:
        logger.debug(f"[13f] Accession fetch failed for CIK {cik}: {e}")
        return []


def _get_infotable_url(cik: str, accession: str) -> Optional[str]:
    """Locate the infotable XML document URL within a 13F-HR filing."""
    cik_int     = int(cik)
    accn_nodash = accession.replace("-", "")
    try:
        resp = httpx.get(
            _EDGAR_ARCHIVE.format(cik=cik_int, accn_nodash=accn_nodash),
            headers=_HEADERS,
            timeout=15,
        )
        resp.raise_for_status()
        items = resp.json().get("directory", {}).get("item", [])
        base  = f"https://www.sec.gov/Archives/edgar/data/{cik_int}/{accn_nodash}/"

        for item in items:
            n = item.get("name", "").lower()
            if "infotable" in n and n.endswith(".xml"):
                return base + item["name"]
        for item in items:
            n = item.get("name", "")
            if n.lower().endswith(".xml") and "primary" not in n.lower():
                return base + n
    except Exception as e:
        logger.debug(f"[13f] Index fetch failed for {accession}: {e}")
    return None


def _parse_infotable(xml_text: str) -> Dict[str, Dict]:
    """
    Parse a 13F infotable XML.
    Returns {cusip: {name, value_usd, shares}}.
    value_usd is in dollars (XML stores thousands; we multiply by 1000).
    Handles both namespaced and non-namespaced variants.
    """
    holdings: Dict[str, Dict] = {}
    try:
        clean = re.sub(r"<(/?)[\w-]+:", r"<\1", xml_text)
        clean = re.sub(r'\sxmlns[^"]*"[^"]*"', "", clean)
        root  = ET.fromstring(clean)

        for table in root.iter("infoTable"):
            name  = (table.findtext("nameOfIssuer") or "").strip().upper()
            cusip = (table.findtext("cusip") or name).strip()
            if not cusip:
                continue

            try:
                value_usd = int((table.findtext("value") or "0").replace(",", "")) * 1000
            except ValueError:
                value_usd = 0

            shr_el = table.find("shrsOrPrnAmt")
            try:
                shares = int((shr_el.findtext("sshPrnamt") if shr_el is not None else "0" or "0").replace(",", ""))
            except ValueError:
                shares = 0

            holdings[cusip] = {"name": name, "value_usd": value_usd, "shares": shares}
    except Exception as e:
        logger.debug(f"[13f] XML parse error: {e}")
    return holdings


def _diff_holdings(
    prev: Dict[str, Dict],
    curr: Dict[str, Dict],
    threshold: float = 0.10,
) -> Dict[str, str]:
    """
    Diff two 13F holdings snapshots.
    Returns {cusip: change_type} for meaningful changes only (>10% move).
    """
    changes: Dict[str, str] = {}
    for cusip in set(prev) | set(curr):
        prev_s = prev.get(cusip, {}).get("shares", 0)
        curr_s = curr.get(cusip, {}).get("shares", 0)

        if prev_s == 0 and curr_s > 0:
            changes[cusip] = "13f_new_position"
        elif curr_s == 0 and prev_s > 0:
            changes[cusip] = "13f_exit"
        elif curr_s > 0 and prev_s > 0:
            ratio = curr_s / prev_s
            if ratio >= 1 + threshold:
                changes[cusip] = "13f_increase"
            elif ratio <= 1 - threshold:
                changes[cusip] = "13f_decrease"
    return changes


def fetch_13f_positions() -> List[InsiderTrade]:
    """
    Fetch the two most recent 13F-HR filings for each tracked institution,
    diff holdings, and emit InsiderTrade signals for ALL position changes.

    Tickers are discovered from the 13F holdings themselves using the SEC
    company tickers index — not limited to any predefined watchlist.
    """
    institutions = settings.tracked_institutions_list
    if not institutions:
        return []

    ticker_index = _load_ticker_index()
    results: List[InsiderTrade] = []

    for inst_name in institutions:
        logger.info(f"[13f] Processing: {inst_name}")

        cik = _lookup_institution_cik(inst_name)
        if not cik:
            logger.warning(f"[13f] Could not resolve CIK for '{inst_name}' — skipping")
            continue
        time.sleep(_REQUEST_DELAY)

        accessions = _get_recent_13f_accessions(cik)
        if len(accessions) < 2:
            logger.debug(f"[13f] Need ≥2 13F filings for diff — found {len(accessions)} for {inst_name}")
            continue
        time.sleep(_REQUEST_DELAY)

        curr_accn, curr_date = accessions[0]
        prev_accn, _         = accessions[1]

        curr_url = _get_infotable_url(cik, curr_accn)
        prev_url = _get_infotable_url(cik, prev_accn)
        if not curr_url or not prev_url:
            logger.debug(f"[13f] Could not find infotable XML for {inst_name}")
            continue
        time.sleep(_REQUEST_DELAY)

        try:
            curr_xml = httpx.get(curr_url, headers=_HEADERS, timeout=30).text
            time.sleep(_REQUEST_DELAY)
            prev_xml = httpx.get(prev_url, headers=_HEADERS, timeout=30).text
        except Exception as e:
            logger.debug(f"[13f] XML fetch failed for {inst_name}: {e}")
            continue

        curr_h = _parse_infotable(curr_xml)
        prev_h = _parse_infotable(prev_xml)
        if not curr_h:
            logger.debug(f"[13f] No holdings parsed for {inst_name}")
            continue

        changes   = _diff_holdings(prev_h, curr_h)
        new_count = sum(1 for c in changes.values() if "new" in c)
        logger.info(f"[13f] {inst_name}: {len(changes)} changes, {new_count} new positions")

        for cusip, change_type in changes.items():
            info      = curr_h.get(cusip) or prev_h.get(cusip, {})
            hold_name = info.get("name", cusip)
            value_usd = info.get("value_usd", 0)
            curr_s    = curr_h.get(cusip, {}).get("shares", 0)
            prev_s    = prev_h.get(cusip, {}).get("shares", 0)

            # Discover ticker from the holding name — not limited to existing universe
            ticker = _ticker_from_name(hold_name, ticker_index)
            if not ticker:
                continue

            results.append(InsiderTrade(
                ticker=ticker,
                trader_name=inst_name,
                trader_type="institutional",
                role="Superinvestor (13F)",
                transaction_type=change_type,
                amount_range=_notional_to_amount_range(value_usd),
                transaction_date=curr_date,
                disclosure_date=curr_date,
                notes=(
                    f"{hold_name} | "
                    f"{prev_s:,} → {curr_s:,} shares | "
                    f"Value ~${value_usd:,.0f}"
                ),
            ))

    unique_tickers = len({t.ticker for t in results})
    logger.info(f"[13f] Total: {len(results)} 13F signal(s) across {unique_tickers} tickers")
    return results


# ---------------------------------------------------------------------------
# Strategy 4: Market-wide Form 4 open-market BUYS (transaction code "P")
# ---------------------------------------------------------------------------

_CACHE_DIR = Path("cache")


def _form4_cache_path(today: date) -> Path:
    return _CACHE_DIR / f"form4_buys_{today.isoformat()}.json"


def _load_form4_cache(today: date) -> Optional[List[InsiderTrade]]:
    p = _form4_cache_path(today)
    if not p.exists():
        return None
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
        return [InsiderTrade.model_validate(d) for d in data]
    except Exception:
        return None


def _save_form4_cache(today: date, trades: List[InsiderTrade]) -> None:
    try:
        _CACHE_DIR.mkdir(exist_ok=True)
        _form4_cache_path(today).write_text(
            json.dumps([t.model_dump(mode="json") for t in trades], indent=2, default=str),
            encoding="utf-8",
        )
    except Exception as e:
        logger.debug(f"[form4] cache save failed: {e}")


def _fetch_form4_xml(hit: dict) -> Optional[str]:
    """Fetch a Form 4 filing's XML document. Tries each associated CIK for the archive path."""
    _id = hit.get("_id", "")
    if ":" not in _id:
        return None
    src   = hit.get("_source", {})
    accn  = src.get("adsh") or _id.split(":")[0]
    fname = _id.split(":", 1)[1]
    accn_nodash = accn.replace("-", "")
    for cik in src.get("ciks", []):
        try:
            url = f"https://www.sec.gov/Archives/edgar/data/{int(cik)}/{accn_nodash}/{fname}"
            resp = httpx.get(url, headers=_HEADERS, timeout=15)
            if resp.status_code == 200 and "<ownershipDocument" in resp.text:
                return resp.text
        except Exception:
            continue
    return None


def _parse_form4_buy(xml_text: str, ticker_index: Dict[str, str]) -> Optional[dict]:
    """Return {ticker, owner, role, notional, date} if the Form 4 has an open-market BUY (code P)."""
    try:
        clean = re.sub(r'\sxmlns[^"]*"[^"]*"', "", xml_text)
        root  = ET.fromstring(clean)
    except Exception:
        return None

    total_shares = 0.0
    total_notional = 0.0
    tx_date: Optional[date] = None
    for tx in root.iter("nonDerivativeTransaction"):
        code = (tx.findtext(".//transactionCoding/transactionCode") or "").strip()
        aord = (tx.findtext(".//transactionAcquiredDisposedCode/value") or "").strip()
        if code != "P" or aord != "A":          # P = open-market purchase, A = acquired
            continue
        try:
            shares = float((tx.findtext(".//transactionShares/value") or "0").replace(",", ""))
            price  = float((tx.findtext(".//transactionPricePerShare/value") or "0").replace(",", ""))
        except ValueError:
            shares = price = 0.0
        total_shares   += shares
        total_notional += shares * price
        d = _parse_date(tx.findtext(".//transactionDate/value") or "")
        if d and (tx_date is None or d > tx_date):
            tx_date = d

    if total_shares <= 0:
        return None

    ticker = (root.findtext(".//issuerTradingSymbol") or "").strip().upper()
    if not ticker:
        ticker = _ticker_from_name(root.findtext(".//issuerName") or "", ticker_index) or ""
    if not ticker:
        return None

    owner = (root.findtext(".//reportingOwner/reportingOwnerId/rptOwnerName") or "Insider").strip()
    rel   = root.find(".//reportingOwner/reportingOwnerRelationship")
    role  = "Insider"
    if rel is not None:
        if (rel.findtext("isOfficer") or "0") in ("1", "true", "True"):
            role = (rel.findtext("officerTitle") or "Officer").strip() or "Officer"
        elif (rel.findtext("isDirector") or "0") in ("1", "true", "True"):
            role = "Director"
        elif (rel.findtext("isTenPercentOwner") or "0") in ("1", "true", "True"):
            role = "10% Owner"

    return {
        "ticker": ticker, "owner": owner[:80], "role": role[:40],
        "notional": total_notional, "date": tx_date or date.today(),
    }


def fetch_form4_open_market_buys() -> List[InsiderTrade]:
    """
    Market-wide scan of recent Form 4 filings for OPEN-MARKET PURCHASES (code "P").

    Replaces the old watchlist-capped Form 4 scan: surfaces insider accumulation across the
    WHOLE market, and — unlike the previous non-directional "form_4_filing" records — emits
    real ``purchase`` records that feed the insider cluster + persistence detectors and
    universe discovery. Open-market buys are only ~1-2% of all Form 4s, so it parses a bounded
    number of the most recent filings; cached daily.
    """
    if not settings.enable_form4_scan:
        return []

    today  = date.today()
    cached = _load_form4_cache(today)
    if cached is not None:
        logger.info(f"[form4] Loaded {len(cached)} open-market buy(s) from cache")
        return cached

    ticker_index = _load_ticker_index()
    start = (today - timedelta(days=max(1, settings.form4_scan_lookback_days))).isoformat()
    end   = today.isoformat()
    max_filings = max(0, settings.form4_scan_max_filings)

    # Collect the most recent Form 4 hits (paginated, deduped by accession+doc id)
    hits: List[dict] = []
    seen_ids: set = set()
    frm = 0
    for _ in range(10):  # hard page cap
        if len(hits) >= max_filings:
            break
        try:
            resp = httpx.get(
                _EDGAR_SEARCH,
                params={"forms": "4", "dateRange": "custom", "startdt": start, "enddt": end, "from": frm},
                headers=_HEADERS, timeout=20,
            )
            resp.raise_for_status()
            page = resp.json().get("hits", {}).get("hits", [])
        except Exception as e:
            logger.warning(f"[form4] efts search failed: {e}")
            break
        new = [h for h in page if h.get("_id") not in seen_ids]
        for h in new:
            seen_ids.add(h.get("_id"))
        hits.extend(new)
        time.sleep(_REQUEST_DELAY)
        if len(page) < 10 or not new:   # last/repeating page
            break
        frm += len(page)
    hits = hits[:max_filings]

    buys: List[InsiderTrade] = []
    seen: set = set()
    for hit in hits:
        xml_text = _fetch_form4_xml(hit)
        time.sleep(_REQUEST_DELAY)
        if not xml_text:
            continue
        parsed = _parse_form4_buy(xml_text, ticker_index)
        if not parsed:
            continue
        key = (parsed["ticker"], parsed["owner"].lower(), parsed["date"])
        if key in seen:
            continue
        seen.add(key)
        buys.append(InsiderTrade(
            ticker=parsed["ticker"],
            trader_name=parsed["owner"],
            trader_type="corporate_insider",
            role=parsed["role"],
            transaction_type="purchase",
            amount_range=_notional_to_amount_range(parsed["notional"]),
            transaction_date=parsed["date"],
            disclosure_date=parsed["date"],
            notes=f"Form 4 open-market purchase (~${parsed['notional']:,.0f})",
        ))

    logger.info(
        f"[form4] Market-wide scan: {len(buys)} open-market buy(s) across "
        f"{len({t.ticker for t in buys})} ticker(s) (parsed {len(hits)} filings)"
    )
    _save_form4_cache(today, buys)
    return buys


# ---------------------------------------------------------------------------
# Public aggregate function
# ---------------------------------------------------------------------------

def fetch_sec_filings() -> List[InsiderTrade]:
    """
    Aggregate all SEC filing-based smart money signals.
    Discovers tickers from the filings themselves — adds new stocks to the
    analysis universe beyond the static watchlist.
    """
    all_trades: List[InsiderTrade] = []
    all_trades.extend(fetch_activist_stakes())
    all_trades.extend(fetch_form144_sales())
    all_trades.extend(fetch_13f_positions())
    all_trades.extend(fetch_form4_open_market_buys())

    seen:   set                 = set()
    unique: List[InsiderTrade]  = []
    for t in all_trades:
        key = (t.ticker, t.trader_name.lower(), t.transaction_date, t.transaction_type)
        if key not in seen:
            seen.add(key)
            unique.append(t)

    unique.sort(key=lambda t: t.transaction_date, reverse=True)
    unique_tickers = len({t.ticker for t in unique})
    logger.info(f"[sec] Total: {len(unique)} unique SEC signal(s) across {unique_tickers} tickers")
    return unique
