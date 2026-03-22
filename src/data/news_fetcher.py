"""Fetch financial news from multiple sources."""

import feedparser
import httpx
from datetime import datetime, timedelta, timezone
from loguru import logger
from typing import List
from src.models import NewsArticle
from config import settings


SECTOR_ETF_NAMES = {
    "XLK": "Technology",
    "XLF": "Financials",
    "XLE": "Energy",
    "XLV": "Health Care",
    "XLY": "Consumer Discretionary",
    "XLP": "Consumer Staples",
    "XLI": "Industrials",
    "XLB": "Materials",
    "XLU": "Utilities",
    "XLRE": "Real Estate",
    "XLC": "Communication Services",
}

# Public RSS feeds (no API key needed)
RSS_FEEDS = {
    "reuters_markets": "https://feeds.reuters.com/reuters/businessNews",
    "cnbc_markets": "https://www.cnbc.com/id/20910258/device/rss/rss.html",
    "marketwatch": "https://feeds.content.dowjones.io/public/rss/mw_realtimeheadlines",
    "seeking_alpha": "https://seekingalpha.com/market_currents.xml",
    "yahoo_finance": "https://finance.yahoo.com/news/rssindex",
    "wsj_markets": "https://feeds.a.dj.com/rss/RSSMarketsMain.xml",
}


def fetch_rss_news(max_age_hours: int = 24) -> List[NewsArticle]:
    """Fetch news from all RSS feeds, filtered to recent articles."""
    articles: List[NewsArticle] = []
    cutoff = datetime.now(timezone.utc) - timedelta(hours=max_age_hours)

    for source, url in RSS_FEEDS.items():
        try:
            feed = feedparser.parse(url)
            for entry in feed.entries:
                pub = _parse_feed_date(entry)
                if pub and pub < cutoff:
                    continue
                articles.append(NewsArticle(
                    title=entry.get("title", ""),
                    summary=entry.get("summary", ""),
                    url=entry.get("link", ""),
                    source=source,
                    published_at=pub or datetime.now(timezone.utc),
                ))
        except Exception as e:
            logger.warning(f"RSS feed {source} failed: {e}")

    logger.info(f"RSS: fetched {len(articles)} articles")
    return articles


def fetch_newsapi(query: str, max_age_hours: int = 24) -> List[NewsArticle]:
    """Fetch news from NewsAPI for a given query string."""
    if not settings.newsapi_key:
        return []

    from_date = (datetime.now(timezone.utc) - timedelta(hours=max_age_hours)).strftime("%Y-%m-%dT%H:%M:%S")
    url = "https://newsapi.org/v2/everything"
    params = {
        "q": query,
        "from": from_date,
        "sortBy": "publishedAt",
        "language": "en",
        "pageSize": 50,
        "apiKey": settings.newsapi_key,
    }

    try:
        resp = httpx.get(url, params=params, timeout=10)
        resp.raise_for_status()
        data = resp.json()
        articles = []
        for art in data.get("articles", []):
            articles.append(NewsArticle(
                title=art.get("title") or "",
                summary=art.get("description") or "",
                url=art.get("url") or "",
                source=art.get("source", {}).get("name", "newsapi"),
                published_at=_parse_iso(art.get("publishedAt")),
            ))
        logger.info(f"NewsAPI '{query}': fetched {len(articles)} articles")
        return articles
    except Exception as e:
        logger.warning(f"NewsAPI failed for '{query}': {e}")
        return []


def fetch_all_news(tickers: List[str], sectors: List[str]) -> List[NewsArticle]:
    """Aggregate news from all sources for watchlist tickers and sectors."""
    all_articles: List[NewsArticle] = []

    # RSS feeds (always available)
    all_articles.extend(fetch_rss_news())

    # NewsAPI targeted queries
    ticker_query = " OR ".join(tickers[:10])  # API limit
    all_articles.extend(fetch_newsapi(ticker_query))

    sector_names = [SECTOR_ETF_NAMES.get(s, s) for s in sectors[:5]]
    sector_query = " OR ".join(sector_names)
    all_articles.extend(fetch_newsapi(sector_query))

    # Deduplicate by URL
    seen = set()
    unique = []
    for art in all_articles:
        if art.url not in seen:
            seen.add(art.url)
            unique.append(art)

    logger.info(f"Total unique articles: {len(unique)}")
    return unique


def _parse_feed_date(entry) -> datetime | None:
    parsed = entry.get("published_parsed")
    if parsed:
        import calendar
        return datetime.fromtimestamp(calendar.timegm(parsed), tz=timezone.utc)
    return None


def _parse_iso(s: str | None) -> datetime:
    if not s:
        return datetime.now(timezone.utc)
    try:
        return datetime.fromisoformat(s.replace("Z", "+00:00"))
    except Exception:
        return datetime.now(timezone.utc)
