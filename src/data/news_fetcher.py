"""Fetch financial news from multiple sources."""

import feedparser
import httpx
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta, timezone
from urllib.parse import quote_plus
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

# Press-release wires — where primary catalysts (earnings, guidance, M&A, FDA,
# contracts) break in real time, minutes before the secondary RSS pickup. Free
# public RSS, no key. URLs verified live (2026-06-17); Business Wire's token feed
# and Accesswire's endpoint returned 0/bozo and are intentionally omitted. Fetched
# on the same fast path as RSS_FEEDS (fresh every tick — never behind the news
# cache), so a wire crossing between hourly refreshes is seen at the next tick.
PR_WIRE_FEEDS = {
    "globenewswire_pubco": "https://www.globenewswire.com/RssFeed/orgclass/1/feedTitle/GlobeNewswire%20-%20Public%20Companies",
    "prnewswire_financial": "https://www.prnewswire.com/rss/financial-services-latest-news/financial-services-latest-news-list.rss",
    "prnewswire_ma": "https://www.prnewswire.com/rss/financial-services-latest-news/acquisitions-mergers-and-takeovers-list.rss",
}

# Regulatory catalyst feeds — FDA approvals/CRLs and MedWatch recalls are binary,
# market-moving events for drug/device names that the general wires sometimes lag.
# Market-wide (mapped to tickers downstream via the keyword aliases, like the other
# RSS feeds); gated by enable_fda_news. URLs verified live 2026-06-19. On the same
# fresh-every-tick fast lane as RSS_FEEDS.
FDA_FEEDS = {
    "fda_press": "https://www.fda.gov/about-fda/contact-fda/stay-informed/rss-feeds/press-releases/rss.xml",
    "fda_medwatch": "https://www.fda.gov/about-fda/contact-fda/stay-informed/rss-feeds/medwatch/rss.xml",
}


def fetch_rss_news(max_age_hours: int = 24) -> List[NewsArticle]:
    """Fetch news from all RSS + press-release-wire feeds, filtered to recent
    articles. Real-time by nature — the pipeline fetches this FRESH every tick
    (never cached) so breaking catalysts aren't hidden behind the hourly cache."""
    articles: List[NewsArticle] = []
    cutoff = datetime.now(timezone.utc) - timedelta(hours=max_age_hours)

    feeds = {**RSS_FEEDS, **PR_WIRE_FEEDS}
    if settings.enable_fda_news:
        feeds.update(FDA_FEEDS)          # regulatory catalysts on the fresh fast lane
    for source, url in feeds.items():
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


# Google News RSS search — free, no key, near-real-time, and far broader than the
# 5 fixed market feeds: a per-ticker query surfaces Reuters / Bloomberg / Barron's /
# FT / Investing.com AND Business Wire (the one wire our direct feeds miss). Each
# article is tied to its ticker by the query, so it maps directly — no fuzzy
# title matching (the same reason fetch_ticker_news beats the keyword pools).
_GOOGLE_NEWS_RSS = "https://news.google.com/rss/search?q={q}&hl=en-US&gl=US&ceid=US:en"
_GOOGLE_NEWS_WORKERS = 6   # modest fan-out so a 40-name universe doesn't burst-trip Google


def _google_entry_source(entry) -> str:
    """Real publisher name from a Google News entry → 'google_news/<Publisher>'
    (keeps source-diversity scoring meaningful), falling back to 'google_news'."""
    try:
        src = entry.get("source")
        title = src.get("title") if isinstance(src, dict) else getattr(src, "title", None)
        if title:
            return f"google_news/{title}"
    except Exception:
        pass
    return "google_news"


def _fetch_google_news_for_ticker(ticker: str, cutoff: datetime, with_bw: bool) -> List[NewsArticle]:
    """Per-ticker Google News: a general query + (optionally) a Business Wire
    site-query. Fails soft so one bad symbol never aborts the batch."""
    queries = [f'"{ticker}" stock']
    if with_bw:
        queries.append(f'"{ticker}" site:businesswire.com')
    out: List[NewsArticle] = []
    for q in queries:
        try:
            feed = feedparser.parse(_GOOGLE_NEWS_RSS.format(q=quote_plus(q)))
        except Exception as e:
            logger.debug(f"[google_news] {ticker} query failed: {e}")
            continue
        for entry in feed.entries:
            pub = _parse_feed_date(entry)
            if pub and pub < cutoff:
                continue
            title = (entry.get("title") or "").strip()
            if not title:
                continue
            out.append(NewsArticle(
                title=title,
                summary=entry.get("summary", ""),
                url=entry.get("link", ""),
                source=_google_entry_source(entry),
                published_at=pub or datetime.now(timezone.utc),
                tickers=[ticker.upper()],
            ))
    return out


def fetch_google_news(tickers: List[str], max_age_hours: int = 24) -> List[NewsArticle]:
    """Per-ticker Google News RSS (general + Business Wire), ticker-tagged.

    Free and near-real-time, so the pipeline fetches it FRESH every tick (the
    reactivity fast-lane). Bounded by ``google_news_max_tickers`` and fanned out
    over a small thread pool to keep the per-tick request burst reasonable;
    fail-soft per ticker. Skips non-equity symbols (futures ``=``, indices ``^``)
    that produce junk queries. Deduped by URL within this source.
    """
    if not settings.enable_google_news:
        return []
    eligible = [t for t in tickers if t and "=" not in t and not t.startswith("^")]
    cap = int(settings.google_news_max_tickers)
    if cap > 0:
        eligible = eligible[:cap]
    if not eligible:
        return []
    cutoff = datetime.now(timezone.utc) - timedelta(hours=max_age_hours)
    with_bw = bool(settings.google_news_business_wire)
    out: List[NewsArticle] = []
    workers = max(1, min(_GOOGLE_NEWS_WORKERS, len(eligible)))
    with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="gnews") as ex:
        futs = [ex.submit(_fetch_google_news_for_ticker, t, cutoff, with_bw) for t in eligible]
        for f in as_completed(futs):
            try:
                out.extend(f.result() or [])
            except Exception as e:
                logger.debug(f"[google_news] worker failed: {e}")
    out = _dedupe_by_url(out)
    logger.info(f"Google News: {len(out)} articles across {len(eligible)} tickers (BW={with_bw})")
    return out


def _dedupe_by_url(articles: List[NewsArticle]) -> List[NewsArticle]:
    """Drop duplicate URLs, first occurrence wins (preserves source ordering)."""
    seen = set()
    unique: List[NewsArticle] = []
    for art in articles:
        if art.url not in seen:
            seen.add(art.url)
            unique.append(art)
    return unique


def fetch_cached_news(tickers: List[str], sectors: List[str]) -> List[NewsArticle]:
    """The CACHE-WORTHY news bundle: per-ticker yfinance + NewsAPI.

    These are the rate-limited (yfinance 429s) / quota-bound (NewsAPI free tier)
    sources, so the pipeline caches them hourly. RSS + press-release wires are
    deliberately NOT included here — they're fetched fresh every tick via
    ``fetch_rss_news`` so a breaking headline is never hidden behind this cache
    (the reactivity fast-lane). ``fetch_all_news`` recombines the two.
    """
    out: List[NewsArticle] = []

    # Per-ticker news (the primary per-symbol feed — ticker-tagged, covers all).
    out.extend(fetch_ticker_news(tickers))

    # NewsAPI targeted queries. Window must exceed the free tier's ~24h embargo
    # (a from<24h query returns 0 results on the free plan), so look back 7 days;
    # the sentiment scorer weights by recency and drops anything >7d as stale.
    ticker_query = " OR ".join(tickers[:10])  # API limit
    out.extend(fetch_newsapi(ticker_query, max_age_hours=168))

    sector_names = [SECTOR_ETF_NAMES.get(s, s) for s in sectors[:5]]
    sector_query = " OR ".join(sector_names)
    out.extend(fetch_newsapi(sector_query, max_age_hours=168))

    return _dedupe_by_url(out)


def fetch_all_news(tickers: List[str], sectors: List[str]) -> List[NewsArticle]:
    """Full FRESH bundle (cache-worthy sources + live RSS/wires), deduped.

    Direct callers (e.g. the hold-review refetch) get everything fresh in one
    call — including per-ticker Google News, so held positions are re-judged on
    the broadest coverage. The SCHEDULED pipeline instead splits these —
    ``fetch_cached_news`` hourly-cached + ``fetch_rss_news`` + ``fetch_google_news``
    every tick — so RSS stays real-time without re-hammering the rate-limited
    per-ticker feed."""
    unique = _dedupe_by_url(
        fetch_cached_news(tickers, sectors) + fetch_rss_news() + fetch_google_news(tickers)
    )
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
