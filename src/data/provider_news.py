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

import json
import time
from datetime import datetime, date, timedelta, timezone
from pathlib import Path
from typing import List, Optional, Tuple

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


# ── Alpha Vantage NEWS_SENTIMENT (pre-scored — feeds the LLM-skip hybrid) ─────
#
# AV returns per-article, per-ticker sentiment labels, so (like Polygon insights)
# it can score tickers WITHOUT an LLM call. The catch is the FREE tier's ~25
# requests/DAY, shared with discovery (trending.py) + the earnings calendar — so
# this is ONE batched multi-ticker call, hourly-cached, and OFF by default
# (enable_alpha_vantage_news). Turn it on only on a paid AV tier, or if you don't
# rely on AV elsewhere. URL/schema are AV's stable documented NEWS_SENTIMENT shape.
_AV_NEWS_URL = "https://www.alphavantage.co/query"
_AV_CACHE_DIR = Path("cache")


def _av_label(lbl: Optional[str]) -> Optional[str]:
    """Map an AV ticker_sentiment_label (Bullish / Somewhat-Bullish / Neutral /
    Somewhat-Bearish / Bearish) to the hybrid's bullish|bearish|neutral."""
    l = (lbl or "").strip().lower()
    if "bull" in l:
        return "bullish"
    if "bear" in l:
        return "bearish"
    if "neutral" in l:
        return "neutral"
    return None


def _parse_av_time(s: Optional[str]) -> Optional[datetime]:
    """AV time_published is 'YYYYMMDDTHHMMSS' (UTC)."""
    try:
        return datetime.strptime(str(s), "%Y%m%dT%H%M%S").replace(tzinfo=timezone.utc)
    except (ValueError, TypeError):
        return None


def _av_cache_path() -> Path:
    key = datetime.now(timezone.utc).strftime("%Y-%m-%d_%H")
    return _AV_CACHE_DIR / f"av_news_{key}.json"


def _load_av_cache() -> Optional[List[NewsArticle]]:
    path = _av_cache_path()
    if not path.exists():
        return None
    try:
        return [NewsArticle.model_validate(a) for a in json.loads(path.read_text(encoding="utf-8"))]
    except Exception as e:
        logger.warning(f"[av_news] cache load failed: {e}")
        return None


def _save_av_cache(articles: List[NewsArticle]) -> None:
    try:
        _AV_CACHE_DIR.mkdir(exist_ok=True)
        _av_cache_path().write_text(
            json.dumps([a.model_dump(mode="json") for a in articles], default=str),
            encoding="utf-8",
        )
    except Exception as e:
        logger.warning(f"[av_news] cache save failed: {e}")


def fetch_alpha_vantage_news(tickers: List[str]) -> List[NewsArticle]:
    """ONE batched AV NEWS_SENTIMENT call for the universe, with per-ticker
    sentiment attached (provider_insights → LLM-skip). Hourly-cached to stay
    inside the free 25/day budget. [] when disabled / no key / quota hit."""
    if not settings.enable_alpha_vantage_news or not settings.alpha_vantage_key:
        return []
    cached = _load_av_cache()
    if cached is not None:
        logger.info(f"[av_news] {len(cached)} cached article(s) (hourly)")
        return cached

    universe = [t.upper() for t in (tickers or []) if t]
    cap = int(settings.alpha_vantage_news_max_tickers)
    subset = universe[:cap] if cap > 0 else universe
    if not subset:
        return []
    try:
        r = httpx.get(_AV_NEWS_URL, params={
            "function": "NEWS_SENTIMENT", "tickers": ",".join(subset),
            "apikey": settings.alpha_vantage_key, "limit": 200, "sort": "LATEST",
        }, timeout=20)
        r.raise_for_status()
        data = r.json()
    except Exception as e:
        logger.warning(f"[av_news] fetch failed: {e}")
        return []

    feed = data.get("feed")
    if not isinstance(feed, list):
        # AV signals quota/rate problems via an Information/Note string (HTTP 200).
        note = data.get("Information") or data.get("Note") or data.get("Error Message")
        if note:
            logger.warning(f"[av_news] {str(note)[:140]}")
        return []

    uni = set(subset)
    out: List[NewsArticle] = []
    for item in feed:
        title = (item.get("title") or "").strip()
        url = (item.get("url") or "").strip()
        if not title or not url:
            continue
        insights, tks = {}, []
        for ts in (item.get("ticker_sentiment") or []):
            tk = (ts.get("ticker") or "").upper()
            if tk not in uni:
                continue
            tks.append(tk)
            lbl = _av_label(ts.get("ticker_sentiment_label"))
            if lbl:
                insights[tk] = lbl
        if not tks:
            continue   # article didn't actually touch our universe
        out.append(NewsArticle(
            title=title,
            summary=(item.get("summary") or "")[:1000],
            url=url,
            source=(item.get("source") or "AlphaVantage"),
            published_at=_parse_av_time(item.get("time_published")) or datetime.now(timezone.utc),
            tickers=tks,
            provider_insights=insights,
            provider_sentiment_source="alphavantage" if insights else None,
        ))
    out = _dedupe(out)
    _save_av_cache(out)
    scored = sum(1 for a in out if a.provider_insights)
    logger.info(f"[av_news] {len(out)} article(s) for universe ({scored} pre-scored); 1 AV request this hour")
    return out


class FinnhubRateLimited(RuntimeError):
    """Finnhub answered 429 — the free tier's 60 calls/min are spent."""


def finnhub_company_news(ticker: str, lookback_days: int = 3,
                         max_per_ticker: int = 15) -> Tuple[List[dict], int]:
    """ONE live Finnhub company-news request for *ticker* — the leg's exact
    request (``from`` = local today - ``lookback_days``, ``to`` = today) —
    processed the way the leg always did: newest first, aggregator roundups
    dropped (`_is_finnhub_noise`), the ``max_per_ticker`` newest real articles
    kept. Returns ``(items, n_noise_dropped)`` with items as plain dicts
    (``datetime`` epoch s, ``headline``, ``url``, ``source``, ``summary``) so the
    background refresher can cache them (`finnhub_refresher`).

    Raises `FinnhubRateLimited` on a 429 and lets any other transport / HTTP
    error propagate — the caller decides whether a failure is a skip or a stop.
    """
    frm = (date.today() - timedelta(days=lookback_days)).isoformat()
    to = date.today().isoformat()
    r = httpx.get(_FINNHUB_NEWS, params={
        "symbol": ticker.upper(), "from": frm, "to": to,
        "token": settings.finnhub_api_key}, timeout=15)
    if r.status_code == 429:
        raise FinnhubRateLimited(ticker)
    r.raise_for_status()
    raw = r.json() or []
    # Newest first, then keep only the top max_per_ticker non-noise articles —
    # bounds the volume of low-signal back-catalogue Finnhub returns per name.
    raw = sorted(raw, key=lambda x: x.get("datetime") or 0, reverse=True)
    kept: List[dict] = []
    dropped = 0
    for item in raw:
        if len(kept) >= max_per_ticker:
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
            datetime.fromtimestamp(int(ts), tz=timezone.utc)
        except (ValueError, OSError, TypeError, OverflowError):
            continue
        kept.append({"datetime": int(ts), "headline": headline, "url": url, "source": source,
                     "summary": (item.get("summary") or "")[:1000]})
    return kept, dropped


def finnhub_articles(ticker: str, items: List[dict]) -> List[NewsArticle]:
    """`finnhub_company_news` items → the leg's NewsArticles (tagged with the
    queried symbol; no provider sentiment on the free tier)."""
    return [NewsArticle(
        title=it["headline"],
        summary=it.get("summary") or "",
        url=it["url"],
        source=it.get("source") or "Finnhub",
        published_at=datetime.fromtimestamp(int(it["datetime"]), tz=timezone.utc),
        tickers=[ticker.upper()],
        provider_sentiment_source="finnhub",
    ) for it in items or []]


def fetch_finnhub_news(tickers: List[str], lookback_days: int = 3,
                       max_tickers: int = 0, max_per_ticker: int = 15) -> List[NewsArticle]:
    """Real-time Finnhub company-news for each ticker (last ``lookback_days``),
    noise-filtered and capped to the ``max_per_ticker`` most-recent real articles.
    No per-article sentiment on the free tier, so ``provider_insights`` stays
    empty (these articles still go through the LLM scorer). [] when disabled or
    no key.

    ALL-SOURCE (2026-09-25): every ticker is covered — ``max_tickers`` 0 means no
    cap (it was the first 60) — and most of them are served from the background
    refresher's cache (`finnhub_refresher`), because the free tier's 60
    calls/min would hold Step 1 for ~7 minutes on a ~400-name universe. A name
    with no usable cache entry (never fetched, fetched on another day, or older
    than ``finnhub_cache_max_age_seconds``) is fetched inline through the SAME
    rate limiter, up to ``finnhub_inline_budget`` names per call in the
    universe's order — which is exactly the old behaviour when no refresher runs
    (a one-off ``main.py`` run, a cold start)."""
    if not settings.enable_finnhub_news or not settings.finnhub_api_key:
        return []
    from src.data import finnhub_refresher as fr
    names = list(dict.fromkeys(str(t).strip().upper() for t in (tickers or []) if str(t).strip()))
    if max_tickers and max_tickers > 0:
        names = names[:max_tickers]
    budget = max(0, int(getattr(settings, "finnhub_inline_budget", 60)))
    max_age = float(getattr(settings, "finnhub_cache_max_age_seconds", 1800))
    out: List[NewsArticle] = []
    ages: List[float] = []
    n_cache = n_inline = n_missing = 0
    for tk in names:
        hit = fr.cached_items(tk, max_age)
        if hit is not None:
            items, age = hit
            n_cache += 1
            ages.append(age)
        elif n_inline < budget:
            items = fr.fetch_now(tk, lookback_days=lookback_days, max_per_ticker=max_per_ticker,
                                 max_wait=float(getattr(settings, "finnhub_inline_max_wait_seconds", 15)))
            if items is None:
                n_missing += 1
                continue
            n_inline += 1
        else:
            n_missing += 1
            continue
        out.extend(finnhub_articles(tk, items))
    out = _dedupe(out)
    med_age = f"{sorted(ages)[len(ages) // 2] / 60.0:.0f} min" if ages else "n/a"
    if out:
        newest = max(a.published_at for a in out)
        age_min = (datetime.now(timezone.utc) - newest).total_seconds() / 60.0
        # Empirical freshness check: how old is the most recent article Finnhub
        # returned? Lets you SEE whether the feed is <15 min fresh in production.
        logger.info(f"[finnhub_news] {len(out)} article(s) across {n_cache + n_inline}/{len(names)} "
                    f"ticker(s) (cache {n_cache}, median age {med_age}; inline {n_inline}; "
                    f"missing {n_missing}); newest is {age_min:.0f} min old")
    else:
        logger.info(f"[finnhub_news] 0 article(s) across {n_cache + n_inline}/{len(names)} ticker(s) "
                    f"(cache {n_cache}; inline {n_inline}; missing {n_missing})")
    return out


