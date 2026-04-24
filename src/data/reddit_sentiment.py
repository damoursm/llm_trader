"""
Reddit social sentiment from r/wallstreetbets, r/stocks, and r/investing.
Uses Reddit OAuth2 application-only auth (client_id + client_secret — free Reddit API).

For each ticker with ≥5 post mentions in the last 24h, generates a NewsArticle
with mention count and keyword-based bullish/bearish sentiment from post titles.

Cached hourly — same cadence as the main news feed.
"""

import json
import re
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import httpx
from loguru import logger

from src.models import NewsArticle

CACHE_DIR = Path("cache")
_REQUEST_DELAY = 0.5   # Reddit rate limit: 60 req/min for OAuth apps

# Subreddits to scan
_SUBREDDITS = ["wallstreetbets", "stocks", "investing"]

# Minimum mentions before generating an article
_MIN_MENTIONS = 5

# Lookback window for posts (24 hours in seconds)
_LOOKBACK_SECONDS = 86_400

# Bullish/bearish keyword sets for title-based sentiment
_BULLISH_KEYWORDS = frozenset({
    "buy", "bull", "bullish", "calls", "moon", "mooning", "long", "going up",
    "upside", "breakout", "squeeze", "short squeeze", "catalyst", "strong",
    "beat", "beats", "earnings beat", "upgrade", "outperform", "buying",
    "bought", "yolo", "to the moon", "💎", "🚀", "🐂",
})
_BEARISH_KEYWORDS = frozenset({
    "sell", "bear", "bearish", "puts", "short", "shorting", "drop", "crash",
    "tank", "tanking", "downside", "breakdown", "overvalued", "miss", "misses",
    "earnings miss", "downgrade", "underperform", "selling", "sold", "inverse",
    "🐻", "📉", "💀",
})


def _cache_path() -> Path:
    key = datetime.now(timezone.utc).strftime("%Y-%m-%d_%H")
    return CACHE_DIR / f"reddit_{key}.json"


def _load_cache() -> Optional[List[NewsArticle]]:
    path = _cache_path()
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        articles = [NewsArticle.model_validate(a) for a in data]
        logger.info(f"[reddit] Loaded {len(articles)} cached Reddit articles")
        return articles
    except Exception as e:
        logger.warning(f"[reddit] Cache load failed: {e}")
        return None


def _save_cache(articles: List[NewsArticle]) -> None:
    CACHE_DIR.mkdir(exist_ok=True)
    path = _cache_path()
    try:
        data = [a.model_dump(mode="json") for a in articles]
        path.write_text(json.dumps(data, indent=2, default=str), encoding="utf-8")
        logger.info(f"[reddit] Saved {len(articles)} Reddit articles → {path.name}")
    except Exception as e:
        logger.warning(f"[reddit] Cache save failed: {e}")


def _get_access_token(client_id: str, client_secret: str, user_agent: str) -> Optional[str]:
    """Obtain OAuth2 bearer token via application-only flow."""
    try:
        resp = httpx.post(
            "https://www.reddit.com/api/v1/access_token",
            data={"grant_type": "client_credentials"},
            auth=(client_id, client_secret),
            headers={"User-Agent": user_agent},
            timeout=15,
        )
        resp.raise_for_status()
        token = resp.json().get("access_token")
        if not token:
            logger.warning("[reddit] No access_token in OAuth response")
        return token
    except Exception as e:
        logger.warning(f"[reddit] OAuth token fetch failed: {e}")
        return None


def _score_title(title: str) -> float:
    """Return sentiment score in [-1, +1] based on keyword matching in a post title."""
    words = set(re.findall(r"[\w🚀🐻🐂💎📉💀]+", title.lower()))
    bull_hits = len(words & _BULLISH_KEYWORDS)
    bear_hits = len(words & _BEARISH_KEYWORDS)
    total = bull_hits + bear_hits
    if total == 0:
        return 0.0
    return (bull_hits - bear_hits) / total


def _search_subreddit(
    client: httpx.Client,
    subreddit: str,
    ticker: str,
    after_utc: float,
    user_agent: str,
) -> List[Dict]:
    """
    Search a subreddit for posts mentioning a ticker in the last 24 hours.
    Returns list of post dicts with title, score, and created_utc.
    """
    posts = []
    try:
        # Use the search endpoint with `new` sort so we capture recent posts
        resp = client.get(
            f"https://oauth.reddit.com/r/{subreddit}/search",
            params={
                "q": ticker,
                "restrict_sr": "true",
                "sort": "new",
                "limit": 100,
                "t": "day",
            },
            headers={"User-Agent": user_agent},
            timeout=15,
        )
        resp.raise_for_status()
        children = resp.json().get("data", {}).get("children", [])
        for child in children:
            post = child.get("data", {})
            created = post.get("created_utc", 0)
            if created >= after_utc:
                posts.append({
                    "title": post.get("title", ""),
                    "score": max(post.get("score", 1), 1),  # floor at 1 for weighting
                    "created_utc": created,
                    "permalink": post.get("permalink", ""),
                })
        time.sleep(_REQUEST_DELAY)
    except Exception as e:
        logger.debug(f"[reddit] Search r/{subreddit} for {ticker} failed: {e}")
    return posts


def _build_article(ticker: str, mention_count: int, weighted_score: float, subreddits: List[str]) -> NewsArticle:
    now = datetime.now(timezone.utc)
    direction = "bullish" if weighted_score > 0.1 else ("bearish" if weighted_score < -0.1 else "mixed")
    sentiment_desc = {
        "bullish": "predominantly bullish",
        "bearish": "predominantly bearish",
        "mixed": "mixed",
    }[direction]

    title = (
        f"Reddit Buzz: {ticker} mentioned {mention_count}x in "
        f"{', '.join(f'r/{s}' for s in subreddits)} — {sentiment_desc} sentiment"
    )
    summary = (
        f"{ticker} appeared {mention_count} times across "
        f"{', '.join(f'r/{s}' for s in subreddits)} in the past 24 hours. "
        f"Weighted sentiment score: {weighted_score:+.2f} (range: -1 bearish to +1 bullish). "
        f"Community tone is {sentiment_desc}. "
        f"High mention count with {'positive' if weighted_score > 0 else 'negative'} sentiment "
        f"can signal {'near-term retail buying pressure' if weighted_score > 0 else 'short-term selling pressure or negative momentum'}."
    )
    url = f"https://www.reddit.com/search/?q={ticker}&sort=new&t=day"
    return NewsArticle(
        title=title,
        summary=summary,
        url=url,
        source="Reddit/WSB",
        published_at=now,
    )


def fetch_reddit_sentiment(
    tickers: List[str],
    client_id: str,
    client_secret: str,
    user_agent: str = "llm_trader/1.0 (stock analysis bot)",
) -> List[NewsArticle]:
    """
    Scan Reddit for ticker mentions and return NewsArticle objects for tickers
    with significant discussion activity.

    Args:
        tickers: list of ticker symbols to scan
        client_id: Reddit OAuth2 client_id
        client_secret: Reddit OAuth2 client_secret
        user_agent: Reddit API user agent string

    Returns:
        List[NewsArticle] — one per ticker with ≥5 mentions in last 24h.
    """
    cached = _load_cache()
    if cached is not None:
        return cached

    if not client_id or not client_secret:
        logger.info("[reddit] No Reddit credentials configured — skipping")
        return []

    token = _get_access_token(client_id, client_secret, user_agent)
    if not token:
        logger.warning("[reddit] Could not obtain OAuth token — skipping Reddit sentiment")
        return []

    after_utc = datetime.now(timezone.utc).timestamp() - _LOOKBACK_SECONDS
    articles: List[NewsArticle] = []

    headers = {
        "Authorization": f"Bearer {token}",
        "User-Agent": user_agent,
    }

    with httpx.Client(headers=headers, timeout=20) as client:
        for ticker in tickers:
            # Aggregate posts across all subreddits
            all_posts: List[Dict] = []
            active_subreddits: List[str] = []

            for subreddit in _SUBREDDITS:
                posts = _search_subreddit(client, subreddit, ticker, after_utc, user_agent)
                if posts:
                    all_posts.extend(posts)
                    active_subreddits.append(subreddit)

            mention_count = len(all_posts)
            if mention_count < _MIN_MENTIONS:
                logger.debug(f"[reddit] {ticker}: {mention_count} mentions — below threshold, skipping")
                continue

            # Weighted sentiment: each post's score weighted by its upvote count
            total_weight = sum(p["score"] for p in all_posts)
            weighted_score = sum(
                _score_title(p["title"]) * p["score"] for p in all_posts
            ) / total_weight if total_weight > 0 else 0.0

            article = _build_article(ticker, mention_count, weighted_score, active_subreddits)
            articles.append(article)
            logger.info(
                f"[reddit] {ticker}: {mention_count} mentions, "
                f"sentiment={weighted_score:+.2f}, "
                f"subreddits={active_subreddits}"
            )

    logger.info(f"[reddit] Generated {len(articles)} articles from {len(tickers)} tickers")
    _save_cache(articles)
    return articles
