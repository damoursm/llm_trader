"""
Fetch insider and politician trades from two public sources:

  1. House Stock Watcher  — congressional House disclosures
     https://house-stock-watcher-data.s3-us-west-2.amazonaws.com/data/all_transactions.json

  2. Senate Stock Watcher — congressional Senate disclosures
     https://senate-stock-watcher-data.s3-us-west-2.amazonaws.com/aggregate/all_transactions.json

  3. SEC EDGAR Form 4     — corporate officer/director insider trades
     https://efts.sec.gov/LATEST/search-index (free, no key required)

All data is public. No API key required for any source.
"""

from __future__ import annotations

import re
from datetime import date, timedelta
from typing import List

import httpx
from loguru import logger

from config import settings
from src.models import InsiderTrade


# ---------------------------------------------------------------------------
# Amount helpers
# ---------------------------------------------------------------------------

_AMOUNT_ORDER = {
    "$1,001 - $15,000":     1,
    "$15,001 - $50,000":    2,
    "$50,001 - $100,000":   3,
    "$100,001 - $250,000":  4,
    "$250,001 - $500,000":  5,
    "$500,001 - $1,000,000": 6,
    "$1,000,001 - $5,000,000": 7,
    "Over $5,000,000":      8,
}

def _amount_weight(amount_range: str) -> float:
    """Higher dollar amount → higher weight (0.1 – 1.0)."""
    val = _AMOUNT_ORDER.get(amount_range, 0)
    return round(val / 8, 2) if val else 0.1


def _notional_to_amount_range(notional: float) -> str:
    """Map a dollar notional value to the nearest _AMOUNT_ORDER tier string."""
    if notional >= 1_000_000:
        return "$1,000,001 - $5,000,000"
    if notional >= 500_000:
        return "$500,001 - $1,000,000"
    if notional >= 250_000:
        return "$250,001 - $500,000"
    if notional >= 100_000:
        return "$100,001 - $250,000"
    if notional >= 50_000:
        return "$50,001 - $100,000"
    if notional >= 15_000:
        return "$15,001 - $50,000"
    return "$1,001 - $15,000"


# ---------------------------------------------------------------------------
# Source 1: House Stock Watcher
# ---------------------------------------------------------------------------

_HOUSE_URL = (
    "https://house-stock-watcher-data.s3-us-west-2.amazonaws.com"
    "/data/all_transactions.json"
)


def _fetch_house_trades(cutoff: date, tracked: List[str]) -> List[InsiderTrade]:
    trades: List[InsiderTrade] = []
    try:
        resp = httpx.get(_HOUSE_URL, timeout=30,
                         headers={"User-Agent": "llm-trader/1.0"})
        resp.raise_for_status()
        data = resp.json()
    except Exception as e:
        logger.warning(f"[insider] House watcher fetch failed: {e}")
        return trades

    for item in data:
        try:
            ticker = (item.get("ticker") or "").strip().upper()
            if not ticker or ticker in ("N/A", "--", ""):
                continue

            tx_date = _parse_date(item.get("transaction_date", ""))
            if tx_date is None or tx_date < cutoff:
                continue

            rep = (item.get("representative") or "").strip()
            if tracked and not any(t.lower() in rep.lower() for t in tracked):
                continue

            tx_type = (item.get("type") or "").lower().replace(" ", "_")
            trades.append(InsiderTrade(
                ticker=ticker,
                trader_name=rep,
                trader_type="politician",
                role="Representative",
                transaction_type=tx_type,
                amount_range=item.get("amount") or "unknown",
                transaction_date=tx_date,
                disclosure_date=_parse_date(item.get("disclosure_date", "")) or tx_date,
                notes=item.get("asset_description") or "",
            ))
        except Exception:
            continue

    logger.info(f"[insider] House watcher: {len(trades)} trades (cutoff={cutoff})")
    return trades


# ---------------------------------------------------------------------------
# Source 2: Senate Stock Watcher
# ---------------------------------------------------------------------------

_SENATE_URL = (
    "https://senate-stock-watcher-data.s3-us-west-2.amazonaws.com"
    "/aggregate/all_transactions.json"
)


def _fetch_senate_trades(cutoff: date, tracked: List[str]) -> List[InsiderTrade]:
    trades: List[InsiderTrade] = []
    try:
        resp = httpx.get(_SENATE_URL, timeout=30,
                         headers={"User-Agent": "llm-trader/1.0"})
        resp.raise_for_status()
        data = resp.json()
    except Exception as e:
        logger.warning(f"[insider] Senate watcher fetch failed: {e}")
        return trades

    for item in data:
        try:
            ticker = (item.get("ticker") or "").strip().upper()
            if not ticker or ticker in ("N/A", "--", ""):
                continue

            tx_date = _parse_date(item.get("transaction_date", ""))
            if tx_date is None or tx_date < cutoff:
                continue

            senator = (item.get("senator") or "").strip()
            if tracked and not any(t.lower() in senator.lower() for t in tracked):
                continue

            tx_type = (item.get("type") or "").lower().replace(" ", "_")
            trades.append(InsiderTrade(
                ticker=ticker,
                trader_name=senator,
                trader_type="politician",
                role="Senator",
                transaction_type=tx_type,
                amount_range=item.get("amount") or "unknown",
                transaction_date=tx_date,
                disclosure_date=_parse_date(item.get("disclosure_date", "")) or tx_date,
                notes=item.get("asset_description") or "",
            ))
        except Exception:
            continue

    logger.info(f"[insider] Senate watcher: {len(trades)} trades (cutoff={cutoff})")
    return trades


# ---------------------------------------------------------------------------
# Source 3: SEC EDGAR Form 4 (corporate insiders)
# ---------------------------------------------------------------------------
# The old per-ticker Form 4 scan was capped to the first 20 watchlist tickers and produced
# only non-directional "form_4_filing" records. It has been replaced by a MARKET-WIDE
# open-market-BUY scan in sec_filings.fetch_form4_open_market_buys(), which parses Form 4 XML
# for transaction code "P" and emits real corporate-insider ``purchase`` records (feeding the
# insider cluster + persistence detectors). Those flow into smart_money via the SEC path.


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def fetch_insider_trades(tickers: List[str]) -> List[InsiderTrade]:
    """
    Aggregate insider/politician trades from all sources.
    Returns trades sorted by transaction_date descending.
    """
    lookback  = settings.insider_lookback_days
    cutoff    = date.today() - timedelta(days=lookback)
    tracked   = settings.tracked_politicians_list

    all_trades: List[InsiderTrade] = []
    all_trades.extend(_fetch_house_trades(cutoff, tracked))
    all_trades.extend(_fetch_senate_trades(cutoff, tracked))
    # Corporate-insider Form 4 buys are captured market-wide by
    # sec_filings.fetch_form4_open_market_buys() and merged via the smart_money / SEC path.

    # Deduplicate by (ticker, trader, date, type)
    seen: set = set()
    unique: List[InsiderTrade] = []
    for t in all_trades:
        key = (t.ticker, t.trader_name.lower(), t.transaction_date, t.transaction_type)
        if key not in seen:
            seen.add(key)
            unique.append(t)

    unique.sort(key=lambda t: t.transaction_date, reverse=True)
    logger.info(
        f"[insider] Total: {len(unique)} unique trades across "
        f"{len({t.ticker for t in unique})} tickers (last {lookback} days)"
    )
    return unique


def build_insider_summary(ticker: str, trades: List[InsiderTrade]) -> str:
    """
    Build a compact human-readable summary of insider trades for one ticker.
    Returns an empty string if there are no trades.
    """
    relevant = [t for t in trades if t.ticker.upper() == ticker.upper()]
    if not relevant:
        return ""

    lines = []
    for t in relevant[:8]:   # cap at 8 most recent
        lines.append(
            f"  {t.trader_name} ({t.role}) {t.action_label} {t.amount_range} "
            f"on {t.transaction_date} [disclosed {t.disclosure_date}]"
        )

    buys  = sum(1 for t in relevant if t.is_bullish)
    sells = sum(1 for t in relevant if not t.is_bullish)
    header = (
        f"{len(relevant)} insider trade(s): {buys} buy(s), {sells} sell(s) "
        f"in last {settings.insider_lookback_days} days"
    )
    return header + "\n" + "\n".join(lines)


def get_tickers_from_smart_money(trades: List[InsiderTrade]) -> List[str]:
    """
    Return all tickers discovered from smart money signals for universe expansion.

    - Activist 13D/13G, 13F new positions, unusual options sweeps → added immediately
      (these are discovered from EDGAR/options and are inherently new)
    - Politician buys → added only if ≥2 different buy signals (noise filter)
    - Form 144 planned sales → added (bearish signal worth monitoring)
    """
    from collections import Counter
    discovered: set       = set()
    pol_buys: Counter     = Counter()
    insider_buys: Counter = Counter()   # distinct corporate-insider open-market buys per ticker

    # High-conviction source types that are always added
    _instant_types = {
        "13d_activist_stake", "13g_passive_stake",
        "13f_new_position", "13f_increase", "13f_exit", "13f_decrease",
        "unusual_call", "unusual_put",
        "planned_sale_144",
    }

    for t in trades:
        if t.transaction_type in _instant_types:
            discovered.add(t.ticker)
        elif t.trader_type == "politician" and t.is_bullish:
            pol_buys[t.ticker] += 1
        elif t.trader_type == "corporate_insider" and "purchase" in t.transaction_type:
            insider_buys[t.ticker] += 1

    # Congressional buys: a CURATED politician list means each buy is signal (≥1);
    # an open list (all politicians) needs corroboration (≥2) to filter noise.
    pol_threshold = 1 if settings.tracked_politicians_list else 2
    for ticker, count in pol_buys.items():
        if count >= pol_threshold:
            discovered.add(ticker)

    # Corporate-insider OPEN-MARKET buys (from the market-wide Form 4 scan): each is a genuine
    # code-P purchase, so surface on ≥1 — "insider accumulation everywhere" feeds discovery.
    for ticker in insider_buys:
        discovered.add(ticker)

    return list(discovered)


# Keep old name as alias for any external callers
def get_tickers_from_insider_trades(trades: List[InsiderTrade]) -> List[str]:
    return get_tickers_from_smart_money(trades)


# ---------------------------------------------------------------------------
# Date helper
# ---------------------------------------------------------------------------

def _parse_date(s: str) -> date | None:
    if not s:
        return None
    s = s.strip()
    for fmt in ("%Y-%m-%d", "%m/%d/%Y", "%Y/%m/%d", "%d-%m-%Y"):
        try:
            from datetime import datetime as _dt
            return _dt.strptime(s, fmt).date()
        except ValueError:
            continue
    return None
