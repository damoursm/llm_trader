"""Provider news feeds that can carry PRE-COMPUTED sentiment.

Two sources, both flag-gated and fail-soft (return [] on any error so the
existing RSS/NewsAPI/LLM path is unaffected):

* **Polygon** (``enable_polygon_news``) — the ``/v2/reference/news`` endpoint is
  Benzinga-sourced and adds Polygon's own per-article sentiment ``insights``
  ({ticker, sentiment, reasoning}). Mapped onto ``NewsArticle.provider_insights``
  so the provider-sentiment hybrid (sentiment.py) can score those tickers WITHOUT
  an LLM call. One market-wide call, filtered to the universe — cheap on the
  free tier. This is the verified sentiment-skip source.

* **Finnhub** (``enable_finnhub_news``) — real-time ``company-news`` per ticker.
  The free tier has NO per-article sentiment, so it adds news COVERAGE only
  (``provider_insights`` stays empty → those articles still go through the LLM
  scorer). Requires ``finnhub_api_key``.
"""
from __future__ import annotations

import time
from datetime import datetime, date, timedelta, timezone
from typing import List, Optional

import httpx
from loguru import logger

from config import settings
from src.models import NewsArticle


def _parse_iso(s: str) -> Optional[datetime]:
    if not s:
        return None
    try:
        dt = datetime.fromisoformat(str(s).replace("Z", "+00:00"))
        return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)
    except (ValueError, TypeError):
        return None


def _dedupe(articles: List[NewsArticle]) -> List[NewsArticle]:
    seen, out = set(), []
    for a in articles:
        if a.url and a.url not in seen:
            seen.add(a.url)
            out.append(a)
    return out


# ── Polygon news + sentiment insights ────────────────────────────────────────

# Polygon insight labels → the NewsArticle.provider_insights value the hybrid
# reads. Kept as the raw label; sentiment.py maps label → numeric score.
def fetch_polygon_news(tickers: List[str]) -> List[NewsArticle]:
    """Recent Polygon news (one market-wide call) filtered to *tickers*, with
    per-ticker sentiment ``insights`` attached. [] when disabled or unavailable."""
    if not settings.enable_polygon_news:
        return []
    try:
        from src.data import polygon_client
        raw = polygon_client.get_news(limit=1000)
    except Exception as e:
        logger.warning(f"[polygon_news] fetch failed: {e}")
        return []

    universe = {t.upper() for t in (tickers or [])}
    out: List[NewsArticle] = []
    for item in raw or []:
        art_tickers = [t.upper() for t in (item.get("tickers") or [])]
        if universe and not (universe & set(art_tickers)):
            continue   # keep only news touching our universe
        published = _parse_iso(item.get("published_utc"))
        title = (item.get("title") or "").strip()
        url = (item.get("article_url") or item.get("amp_url") or "").strip()
        if not title or not url or published is None:
            continue
        insights = {}
        for ins in (item.get("insights") or []):
            tk = (ins.get("ticker") or "").upper()
            sent = (ins.get("sentiment") or "").strip().lower()
            if tk and sent:
                insights[tk] = sent
        out.append(NewsArticle(
            title=title,
            summary=(item.get("description") or "")[:1000],
            url=url,
            source=((item.get("publisher") or {}).get("name") or "Polygon"),
            published_at=published,
            tickers=art_tickers,
            provider_insights=insights,
            provider_sentiment_source="polygon" if insights else None,
        ))
    out = _dedupe(out)
    scored = sum(1 for a in out if a.provider_insights)
    logger.info(f"[polygon_news] {len(out)} article(s) for universe ({scored} with sentiment insights)")
    return out


# ── Finnhub company news (coverage only; free tier has no sentiment) ──────────

_FINNHUB_NEWS = "https://finnhub.io/api/v1/company-news"

# Finnhub's per-ticker feed is flooded with generic market-summary "roundups"
# (not ticker catalysts) that dilute the sentiment digest. Drop them by:
#   - SOURCE: pure aggregators that only publish auto-generated roundups, and
#   - TITLE: unambiguous movers/session/roundup phrasing.
# Confirmed live: ChartMill "Which S&P500 stocks are moving", "Thursday's
# session:", "most active stock…". Hard catalysts ("Apple beats Q3…", "FDA
# approves…", "Nvidia announces…") match none of these, so they're never dropped.
_FINNHUB_NOISE_SOURCES = {"chartmill"}
_FINNHUB_NOISE_TITLE_PATTERNS = (
    # market-breadth / movers roundups
    "stocks are moving", "stocks moving", "most active", "top movers",
    "biggest movers", "movers within", "making the most noise", "stocks to watch",
    "what to watch", "trending stocks", "premarket movers", "after-hours movers",
    "after hours movers", "market wrap", "closing bell", "opening bell",
    "session:", "stocks that are making", "movers and shakers",
    "stock market today", "wall street today", "market today:", "s&p 500 ",
    "dow jones", "nasdaq today",
    # retail listicle / SEO hooks (Motley Fool / Yahoo syndication) — these
    # phrasings never appear in an institutional catalyst headline.
    "hand over fist", "got $", "1 stock", "2 stocks", "3 stocks", "4 stocks",
    "5 stocks", "stock to buy", "stocks to buy", "stock to sell", "stocks to sell",
    "best stock", "best stocks", "top stock", "top stocks", "stocks for",
    "should you buy", "better buy", "is it too late", "no-brainer",
    "screaming buy", "millionaire", "billionaire", "if you'd invested",
    "if you invested", "could make you", "would have", "magnificent seven",
    "where will", "prediction", "reasons to buy", "stock split",
    "to buy and hold", "dividend stock", "for retirees", "retirement",
    "smartest", "no brainer", "here are", "vs.",
)


def _is_finnhub_noise(title: str, source: str) -> bool:
    """True for aggregator roundups / market summaries (not a ticker catalyst)."""
    if (source or "").strip().lower() in _FINNHUB_NOISE_SOURCES:
        return True
    t = (title or "").lower()
    return any(p in t for p in _FINNHUB_NOISE_TITLE_PATTERNS)


def fetch_finnhub_news(tickers: List[str], lookback_days: int = 3,
                       max_tickers: int = 60, max_per_ticker: int = 15) -> List[NewsArticle]:
    """Real-time Finnhub company-news for each ticker (last ``lookback_days``),
    noise-filtered and capped to the ``max_per_ticker`` most-recent real articles.
    No per-article sentiment on the free tier, so ``provider_insights`` stays
    empty (these articles still go through the LLM scorer). [] when disabled or
    no key."""
    if not settings.enable_finnhub_news or not settings.finnhub_api_key:
        return []
    frm = (date.today() - timedelta(days=lookback_days)).isoformat()
    to = date.today().isoformat()
    out: List[NewsArticle] = []
    dropped = 0
    for tk in (tickers or [])[:max_tickers]:
        try:
            r = httpx.get(_FINNHUB_NEWS, params={
                "symbol": tk.upper(), "from": frm, "to": to,
                "token": settings.finnhub_api_key}, timeout=15)
            if r.status_code == 429:
                logger.warning("[finnhub_news] rate limited — stopping early")
                break
            r.raise_for_status()
            items = r.json() or []
        except Exception as e:
            logger.debug(f"[finnhub_news] {tk} failed: {e}")
            continue
        # Newest first, then keep only the top max_per_ticker non-noise articles —
        # bounds the volume of low-signal back-catalogue Finnhub returns per name.
        items = sorted(items, key=lambda x: x.get("datetime") or 0, reverse=True)
        kept_tk = 0
        for item in items:
            if kept_tk >= max_per_ticker:
                break
            ts = item.get("datetime")
            headline = (item.get("headline") or "").strip()
            url = (item.get("url") or "").strip()
            if not ts or not headline or not url:
                continue
            source = item.get("source") or "Finnhub"
            if _is_finnhub_noise(headline, source):
                dropped += 1
                continue   # generic roundup / listicle — not a ticker catalyst
            try:
                published = datetime.fromtimestamp(int(ts), tz=timezone.utc)
            except (ValueError, OSError, TypeError):
                continue
            out.append(NewsArticle(
                title=headline,
                summary=(item.get("summary") or "")[:1000],
                url=url,
                source=source,
                published_at=published,
                tickers=[tk.upper()],
                provider_sentiment_source="finnhub",
            ))
            kept_tk += 1
        time.sleep(0.1)   # gentle pacing for the free 60/min tier
    out = _dedupe(out)
    n_tk = min(len(tickers or []), max_tickers)
    if out:
        newest = max(a.published_at for a in out)
        age_min = (datetime.now(timezone.utc) - newest).total_seconds() / 60.0
        # Empirical freshness check: how old is the most recent article Finnhub
        # returned? Lets you SEE whether the feed is <15 min fresh in production.
        logger.info(f"[finnhub_news] {len(out)} article(s) across {n_tk} ticker(s) "
                    f"({dropped} noise dropped); newest is {age_min:.0f} min old")
    else:
        logger.info(f"[finnhub_news] 0 article(s) across {n_tk} ticker(s) ({dropped} noise dropped)")
    return out
