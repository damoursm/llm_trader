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


def fetch_finnhub_news(tickers: List[str], lookback_days: int = 3,
                       max_tickers: int = 60) -> List[NewsArticle]:
    """Real-time Finnhub company-news for each ticker (last ``lookback_days``).
    No per-article sentiment on the free tier, so ``provider_insights`` stays
    empty (these articles still go through the LLM scorer). [] when disabled or
    no key."""
    if not settings.enable_finnhub_news or not settings.finnhub_api_key:
        return []
    frm = (date.today() - timedelta(days=lookback_days)).isoformat()
    to = date.today().isoformat()
    out: List[NewsArticle] = []
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
        for item in items:
            ts = item.get("datetime")
            headline = (item.get("headline") or "").strip()
            url = (item.get("url") or "").strip()
            if not ts or not headline or not url:
                continue
            try:
                published = datetime.fromtimestamp(int(ts), tz=timezone.utc)
            except (ValueError, OSError, TypeError):
                continue
            out.append(NewsArticle(
                title=headline,
                summary=(item.get("summary") or "")[:1000],
                url=url,
                source=(item.get("source") or "Finnhub"),
                published_at=published,
                tickers=[tk.upper()],
                provider_sentiment_source="finnhub",
            ))
        time.sleep(0.1)   # gentle pacing for the free 60/min tier
    out = _dedupe(out)
    logger.info(f"[finnhub_news] {len(out)} article(s) across {min(len(tickers or []), max_tickers)} ticker(s)")
    return out
