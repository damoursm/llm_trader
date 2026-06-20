"""StockTwits per-ticker crowd sentiment — real-time trader chatter.

Complements Reddit (reddit_sentiment.py): each ticker gets ONE synthetic
``NewsArticle`` summarising recent StockTwits message volume and the bull/bear
mix, which the normal sentiment pipeline scores (same idiom as Reddit / Google
Trends / short-interest) — NOT a hard provider-sentiment label, since crowd
chatter is noisy.

Access: the public ``streams/symbol`` endpoint now returns **403 without
authentication**, so this is token-gated (``stocktwits_access_token``) and OFF by
default. Register a free StockTwits API app to obtain a token. Fail-soft: any
error (no token, 403, rate limit) returns ``[]`` so the rest of the pipeline is
unaffected. Bounded by ``stocktwits_max_tickers`` and hourly-cached.
"""
from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional

import httpx
from loguru import logger

from config import settings
from src.models import NewsArticle

_STREAM_URL = "https://api.stocktwits.com/api/2/streams/symbol/{sym}.json"
_CACHE_DIR = Path("cache")
_MIN_MESSAGES = 5          # too few tagged messages → no reliable read, skip the ticker


def _cache_path() -> Path:
    key = datetime.now(timezone.utc).strftime("%Y-%m-%d_%H")
    return _CACHE_DIR / f"stocktwits_{key}.json"


def _load_cache() -> Optional[List[NewsArticle]]:
    path = _cache_path()
    if not path.exists():
        return None
    try:
        return [NewsArticle.model_validate(a) for a in json.loads(path.read_text(encoding="utf-8"))]
    except Exception as e:
        logger.warning(f"[stocktwits] cache load failed: {e}")
        return None


def _save_cache(articles: List[NewsArticle]) -> None:
    try:
        _CACHE_DIR.mkdir(exist_ok=True)
        _cache_path().write_text(
            json.dumps([a.model_dump(mode="json") for a in articles], default=str),
            encoding="utf-8",
        )
    except Exception as e:
        logger.warning(f"[stocktwits] cache save failed: {e}")


def _summarise_ticker(ticker: str, messages: list) -> Optional[NewsArticle]:
    """Build one sentiment-summary article from a ticker's recent messages."""
    bull = bear = 0
    for m in messages:
        basic = ((m.get("entities") or {}).get("sentiment") or {}).get("basic")
        if basic == "Bullish":
            bull += 1
        elif basic == "Bearish":
            bear += 1
    tagged = bull + bear
    if tagged < _MIN_MESSAGES:
        return None
    bull_pct = round(100.0 * bull / tagged)
    lean = "bullish" if bull > bear else "bearish" if bear > bull else "mixed"
    title = (f"StockTwits chatter on {ticker}: {lean} — {bull_pct}% bullish "
             f"({bull} bull / {bear} bear of {tagged} tagged, {len(messages)} recent msgs)")
    return NewsArticle(
        title=title,
        summary=(f"Recent StockTwits trader sentiment for {ticker} skews {lean}: {bull} bullish vs "
                 f"{bear} bearish tagged messages out of {len(messages)} recent posts."),
        url=f"https://stocktwits.com/symbol/{ticker}",
        source="StockTwits",
        published_at=datetime.now(timezone.utc),
        tickers=[ticker.upper()],
    )


def fetch_stocktwits_sentiment(tickers: List[str]) -> List[NewsArticle]:
    """One crowd-sentiment summary article per ticker. [] when disabled / no token."""
    if not settings.enable_stocktwits or not settings.stocktwits_access_token:
        return []
    cached = _load_cache()
    if cached is not None:
        logger.info(f"[stocktwits] {len(cached)} cached article(s) (hourly)")
        return cached

    cap = int(settings.stocktwits_max_tickers)
    subset = [t for t in (tickers or []) if t and "=" not in t and not t.startswith("^")]
    if cap > 0:
        subset = subset[:cap]
    out: List[NewsArticle] = []
    for tk in subset:
        try:
            r = httpx.get(
                _STREAM_URL.format(sym=tk.upper()),
                params={"access_token": settings.stocktwits_access_token},
                headers={"User-Agent": "Mozilla/5.0"},
                timeout=15,
            )
            if r.status_code == 429:
                logger.warning("[stocktwits] rate limited — stopping early")
                break
            if r.status_code in (403, 401):
                logger.warning(f"[stocktwits] auth rejected ({r.status_code}) — check stocktwits_access_token")
                break
            r.raise_for_status()
            messages = r.json().get("messages", []) or []
        except Exception as e:
            logger.debug(f"[stocktwits] {tk} failed: {e}")
            continue
        art = _summarise_ticker(tk.upper(), messages)
        if art:
            out.append(art)
        time.sleep(0.2)   # gentle pacing

    _save_cache(out)
    logger.info(f"[stocktwits] {len(out)} sentiment article(s) across {len(subset)} ticker(s)")
    return out
