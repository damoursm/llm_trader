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

# Public RSS feeds (no API key needed). NOTE: Reuters retired its public RSS
# feeds, so that endpoint is intentionally omitted (it returned 0 entries /
# bozo errors). These feeds are general-market — per-ticker relevance comes
# from fetch_ticker_news below, not from these.
RSS_FEEDS = {
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


def fetch_ticker_news(tickers: List[str], max_age_hours: int = 168) -> List[NewsArticle]:
    """Per-ticker news via yfinance ``Ticker.news`` — the core per-symbol feed.

    Unlike the general-market RSS/NewsAPI pools, every article here is already
    associated with its ticker by Yahoo, so it maps to the symbol directly
    (``NewsArticle.tickers``) — no fuzzy title/summary keyword matching, and it
    covers EVERY symbol, not just the ~30 mega-caps the keyword aliases know.
    This is what feeds the per-ticker news + sentiment-velocity scores (which
    were ~0% populated before, starved by the keyword-only mapping).

    Handles both the current yfinance schema (``item['content']`` with
    ``title``/``summary``/``pubDate``/``provider``/``canonicalUrl``) and the
    legacy flat schema (``title``/``link``/``providerPublishTime``). Fails soft
    per ticker so one bad symbol never aborts the batch.
    """
    if not settings.enable_ticker_news:
        return []
    import yfinance as yf

    cutoff = datetime.now(timezone.utc) - timedelta(hours=max_age_hours)
    out: List[NewsArticle] = []
    covered = 0
    for tk in tickers:
        try:
            items = yf.Ticker(tk).news or []
        except Exception as e:
            logger.debug(f"[news] yfinance news failed for {tk}: {e}")
            continue
        n_before = len(out)
        for it in items:
            c = it.get("content") if isinstance(it.get("content"), dict) else it
            title = (c.get("title") or "").strip()
            summary = (c.get("summary") or c.get("description") or "").strip()
            if not title and not summary:
                continue
            url = ((c.get("canonicalUrl") or {}).get("url")
                   or (c.get("clickThroughUrl") or {}).get("url")
                   or c.get("link") or it.get("link") or "")
            provider = ((c.get("provider") or {}).get("displayName")
                        or c.get("publisher") or it.get("publisher") or "yahoo")
            pub = _parse_news_time(c.get("pubDate") or c.get("displayTime")
                                   or it.get("providerPublishTime"))
            if pub and pub < cutoff:
                continue
            out.append(NewsArticle(
                title=title, summary=summary, url=url, source=str(provider),
                published_at=pub or datetime.now(timezone.utc), tickers=[tk.upper()],
            ))
        if len(out) > n_before:
            covered += 1
    logger.info(f"yfinance per-ticker news: {len(out)} articles across {covered}/{len(tickers)} tickers")
    return out


def fetch_all_news(tickers: List[str], sectors: List[str]) -> List[NewsArticle]:
    """Aggregate news from all sources for watchlist tickers and sectors."""
    all_articles: List[NewsArticle] = []

    # Per-ticker news (the primary per-symbol feed — ticker-tagged, covers all).
    all_articles.extend(fetch_ticker_news(tickers))

    # RSS feeds (general-market context, always available)
    all_articles.extend(fetch_rss_news())

    # NewsAPI targeted queries. Window must exceed the free tier's ~24h embargo
    # (a from<24h query returns 0 results on the free plan), so look back 7 days;
    # the sentiment scorer weights by recency and drops anything >7d as stale.
    ticker_query = " OR ".join(tickers[:10])  # API limit
    all_articles.extend(fetch_newsapi(ticker_query, max_age_hours=168))

    sector_names = [SECTOR_ETF_NAMES.get(s, s) for s in sectors[:5]]
    sector_query = " OR ".join(sector_names)
    all_articles.extend(fetch_newsapi(sector_query, max_age_hours=168))

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


def _parse_news_time(v) -> datetime | None:
    """Parse a yfinance news timestamp — ISO string (current schema's ``pubDate``)
    or a Unix epoch int/float (legacy ``providerPublishTime``). None when absent
    or unparseable so the caller can fall back to 'now'."""
    if v is None or v == "":
        return None
    if isinstance(v, (int, float)):
        try:
            return datetime.fromtimestamp(float(v), tz=timezone.utc)
        except (OverflowError, OSError, ValueError):
            return None
    try:
        dt = datetime.fromisoformat(str(v).replace("Z", "+00:00"))
        return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)
    except Exception:
        return None
