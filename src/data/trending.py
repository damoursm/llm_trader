"""Discover trending / hot tickers from multiple online sources."""

import re
import httpx
from loguru import logger
from typing import List
from config import settings
from src.data.ticker_extract import extract_candidate_tickers
from src.data.reddit_sentiment import discover_wsb_tickers


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_TICKER_RE = re.compile(r'\b([A-Z]{1,5})\b')

# Curated broad candidates to extract from headlines — avoids false positives
# from all-caps words that aren't tickers (e.g. "GDP", "IPO" are fine context words)
_NOISE_WORDS = {
    "A", "I", "AM", "PM", "ET", "US", "UK", "EU", "ON", "IN", "AT", "BE",
    "DO", "IF", "OF", "OR", "TO", "UP", "VS", "BY", "AS", "IT", "IS",
    "THE", "AND", "FOR", "NOT", "BUT", "CEO", "CFO", "COO", "IPO", "ETF",
    "GDP", "CPI", "FED", "SEC", "FDA", "DOJ", "NYC", "API", "AI", "EV",
    "S&P", "DOW", "NYSE", "NASDAQ", "IMF", "WTO", "OPEC", "LLC", "INC",
    "LTD", "PLC", "AG", "SA", "NV",
}


def _clean_tickers(raw: List[str]) -> List[str]:
    """Normalise a list of raw ticker strings and remove noise."""
    seen, out = set(), []
    for t in raw:
        t = t.strip().upper().replace(".", "").replace(",", "")
        if t and t not in _NOISE_WORDS and 1 <= len(t) <= 5:
            if t not in seen:
                seen.add(t)
                out.append(t)
    return out


# ---------------------------------------------------------------------------
# Source 1: Yahoo Finance trending tickers (public JSON endpoint)
# ---------------------------------------------------------------------------

def _yahoo_trending() -> List[str]:
    """Return tickers currently trending on Yahoo Finance."""
    url = "https://query1.finance.yahoo.com/v1/finance/trending/US"
    params = {"count": 20, "lang": "en-US", "region": "US"}
    headers = {"User-Agent": "Mozilla/5.0 (compatible; llm-trader/1.0)"}
    try:
        resp = httpx.get(url, params=params, headers=headers, timeout=10)
        resp.raise_for_status()
        quotes = resp.json()["finance"]["result"][0]["quotes"]
        tickers = [q["symbol"] for q in quotes if "symbol" in q]
        logger.info(f"Yahoo trending: {tickers}")
        return tickers
    except Exception as e:
        logger.warning(f"Yahoo trending failed: {e}")
        return []


# ---------------------------------------------------------------------------
# Source 2: Alpha Vantage top gainers / losers
# ---------------------------------------------------------------------------

def _alpha_vantage_movers() -> List[str]:
    """Return today's top gainers and losers from Alpha Vantage."""
    if not settings.alpha_vantage_key:
        return []
    url = "https://www.alphavantage.co/query"
    params = {"function": "TOP_GAINERS_LOSERS", "apikey": settings.alpha_vantage_key}
    try:
        resp = httpx.get(url, params=params, timeout=15)
        resp.raise_for_status()
        data = resp.json()
        tickers = []
        for section in ("top_gainers", "top_losers", "most_actively_traded"):
            for item in data.get(section, [])[:5]:
                sym = item.get("ticker", "")
                if sym and "." not in sym:          # skip preferred / OTC
                    tickers.append(sym)
        logger.info(f"Alpha Vantage movers: {tickers}")
        return tickers
    except Exception as e:
        logger.warning(f"Alpha Vantage movers failed: {e}")
        return []


# ---------------------------------------------------------------------------
# Source 3: NewsAPI — OPEN-VOCABULARY ticker discovery from recent headlines
# ---------------------------------------------------------------------------

def _newsapi_discovered_tickers() -> List[str]:
    """
    Query NewsAPI business headlines and mine tickers OPEN-VOCABULARY: every
    $cashtag / capitalized token is resolved against the full SEC ticker universe
    (via ticker_extract), not a hardcoded allowlist — so ANY listed name can be
    discovered, not just a curated set of mega-caps. Returns tickers appearing in
    ≥2 distinct articles.
    """
    if not settings.newsapi_key:
        return []

    url = "https://newsapi.org/v2/top-headlines"
    params = {
        "category": "business",
        "language": "en",
        "country": "us",
        "pageSize": 100,
        "apiKey": settings.newsapi_key,
    }
    try:
        resp = httpx.get(url, params=params, timeout=10)
        resp.raise_for_status()
        articles = resp.json().get("articles", [])
        counts: dict[str, int] = {}
        for art in articles:
            text = f"{art.get('title', '')} {art.get('description', '')}"
            for t in extract_candidate_tickers(text):
                counts[t] = counts.get(t, 0) + 1
        # Keep those mentioned in ≥2 distinct articles
        hot = [t for t, c in sorted(counts.items(), key=lambda x: -x[1]) if c >= 2]
        logger.info(f"NewsAPI discovered tickers (open-vocab): {hot[:20]}")
        return hot[:20]
    except Exception as e:
        logger.warning(f"NewsAPI ticker extraction failed: {e}")
        return []


# ---------------------------------------------------------------------------
# Source 5: StockTwits trending symbols (public JSON, no key required)
# ---------------------------------------------------------------------------

def _stocktwits_trending() -> List[str]:
    """Return symbols currently trending on StockTwits (social-momentum signal).

    Note: the StockTwits public API is Cloudflare-gated and typically returns 403
    to non-browser clients, so this source is OFF by default. Kept (and fail-graceful)
    in case access is available via a token or allowlisted IP.
    """
    url = "https://api.stocktwits.com/api/2/trending/symbols.json"
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                      "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        "Accept": "application/json, text/plain, */*",
    }
    try:
        resp = httpx.get(url, headers=headers, timeout=10)
        resp.raise_for_status()
        symbols = resp.json().get("symbols", [])
        tickers = [s.get("symbol", "") for s in symbols if s.get("symbol")]
        logger.info(f"StockTwits trending: {tickers}")
        return tickers
    except Exception as e:
        logger.warning(f"StockTwits trending unavailable (Cloudflare-gated): {e}")
        return []


# ---------------------------------------------------------------------------
# Source 4: Alpha Vantage news sentiment — tickers with most news today
# ---------------------------------------------------------------------------

def _alpha_vantage_news_tickers() -> List[str]:
    """Return tickers with the most news activity today via AV news feed."""
    if not settings.alpha_vantage_key:
        return []
    url = "https://www.alphavantage.co/query"
    params = {
        "function": "NEWS_SENTIMENT",
        "sort": "LATEST",
        "limit": 50,
        "apikey": settings.alpha_vantage_key,
    }
    try:
        resp = httpx.get(url, params=params, timeout=15)
        resp.raise_for_status()
        feed = resp.json().get("feed", [])
        counts: dict[str, int] = {}
        for item in feed:
            for ts in item.get("ticker_sentiment", []):
                sym = ts.get("ticker", "")
                if sym and "." not in sym and len(sym) <= 5:
                    counts[sym] = counts.get(sym, 0) + 1
        hot = [t for t, c in sorted(counts.items(), key=lambda x: -x[1]) if c >= 2]
        logger.info(f"AV news-active tickers: {hot[:20]}")
        return hot[:20]
    except Exception as e:
        logger.warning(f"AV news sentiment tickers failed: {e}")
        return []


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def get_trending_tickers(base_tickers: List[str], base_sectors: List[str]) -> List[str]:
    """
    Aggregate trending tickers from all sources, merge with the base watchlist,
    and return a deduplicated list ready for analysis.
    """
    discovered: List[str] = []
    discovered.extend(_yahoo_trending())
    discovered.extend(_alpha_vantage_movers())
    discovered.extend(_newsapi_discovered_tickers())
    discovered.extend(_alpha_vantage_news_tickers())

    # Social-momentum discovery (fail-graceful, gated by flags)
    if settings.enable_stocktwits_discovery:
        discovered.extend(_stocktwits_trending())
    if settings.enable_wsb_discovery:
        try:
            discovered.extend(discover_wsb_tickers(
                user_agent=settings.reddit_user_agent,
                min_mentions=settings.wsb_discovery_min_mentions,
            ))
        except Exception as e:
            logger.warning(f"WSB discovery failed: {e}")

    discovered = _clean_tickers(discovered)

    # Merge: base watchlist first (priority), then discovered extras
    seen = set(base_tickers) | set(base_sectors)
    extras = [t for t in discovered if t not in seen]

    # Cap extras to avoid blowing up the prompt
    extras = extras[:30]

    combined = base_tickers + base_sectors + extras
    logger.info(
        f"Ticker universe: {len(base_tickers)} watchlist + "
        f"{len(base_sectors)} sectors + {len(extras)} trending = {len(combined)} total"
    )
    return combined