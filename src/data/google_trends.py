"""
Google Trends search interest as a leading indicator of retail/consumer attention.
Uses pytrends (unofficial Google Trends API — no key required).

For each ticker in the static watchlist, compares the last 7-day average search
interest to the prior 21-day baseline. Generates a NewsArticle only when the
deviation is meaningful (>25% change AND current interest >10/100).

Best signal for consumer-facing stocks (TSLA, AMZN, AAPL, NVDA).
Cached daily — Google Trends data updates at most once per day.
"""

import json
import time
from datetime import datetime, timezone, date
from pathlib import Path
from typing import List, Optional

from loguru import logger

from src.models import NewsArticle

CACHE_DIR = Path("cache")
_REQUEST_DELAY = 1.5   # pytrends uses unofficial API; throttle aggressively


def _cache_path(today: Optional[str] = None) -> Path:
    key = today or date.today().isoformat()
    return CACHE_DIR / f"trends_{key}.json"


def _load_cache() -> Optional[List[NewsArticle]]:
    path = _cache_path()
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        articles = [NewsArticle.model_validate(a) for a in data]
        logger.info(f"[google_trends] Loaded {len(articles)} cached trend articles")
        return articles
    except Exception as e:
        logger.warning(f"[google_trends] Cache load failed: {e}")
        return None


def _save_cache(articles: List[NewsArticle]) -> None:
    CACHE_DIR.mkdir(exist_ok=True)
    path = _cache_path()
    try:
        data = [a.model_dump(mode="json") for a in articles]
        path.write_text(json.dumps(data, indent=2, default=str), encoding="utf-8")
        logger.info(f"[google_trends] Saved {len(articles)} trend articles → {path.name}")
    except Exception as e:
        logger.warning(f"[google_trends] Cache save failed: {e}")


def _build_article(ticker: str, current_avg: float, baseline_avg: float, pct_change: float) -> NewsArticle:
    direction = "surge" if pct_change > 0 else "drop"
    sentiment = "bullish" if pct_change > 0 else "bearish"
    now = datetime.now(timezone.utc)

    title = (
        f"Google Trends: {ticker} search interest {direction}s "
        f"{abs(pct_change):.0f}% vs recent baseline"
    )
    summary = (
        f"Google Trends shows {ticker} search interest averaged {current_avg:.1f}/100 "
        f"over the past 7 days, compared to a {baseline_avg:.1f}/100 baseline over the prior 21 days "
        f"— a {abs(pct_change):.0f}% {direction}. "
        f"Rising search interest typically precedes retail buying pressure; "
        f"a sharp drop may indicate fading momentum. "
        f"This is a {sentiment} signal for {ticker}."
    )
    url = f"https://trends.google.com/trends/explore?q={ticker}&geo=US"
    return NewsArticle(
        title=title,
        summary=summary,
        url=url,
        source="Google Trends",
        published_at=now,
    )


def fetch_google_trends(tickers: List[str]) -> List[NewsArticle]:
    """
    Fetch Google Trends search interest for each ticker and return
    a List[NewsArticle] for tickers with meaningful deviations.

    Args:
        tickers: watchlist tickers to check (max ~25 recommended)

    Returns:
        List of NewsArticle objects — one per ticker with notable trend change.
    """
    cached = _load_cache()
    if cached is not None:
        return cached

    try:
        from pytrends.request import TrendReq
    except ImportError:
        logger.warning("[google_trends] pytrends not installed — skipping. Run: pip install pytrends")
        return []

    articles: List[NewsArticle] = []
    # pytrends allows max 5 keywords per request
    batch_size = 5
    batches = [tickers[i:i + batch_size] for i in range(0, len(tickers), batch_size)]

    try:
        pt = TrendReq(hl="en-US", tz=-300)  # tz=-300 = EST
    except Exception as e:
        logger.warning(f"[google_trends] Failed to initialize pytrends: {e}")
        return []

    for batch in batches:
        try:
            # 90 days gives us 7-day recent + 21-day baseline + buffer
            pt.build_payload(batch, cat=0, timeframe="today 3-m", geo="US")
            df = pt.interest_over_time()

            if df is None or df.empty:
                logger.debug(f"[google_trends] No data for batch {batch}")
                time.sleep(_REQUEST_DELAY)
                continue

            # Drop the isPartial column if present
            if "isPartial" in df.columns:
                df = df.drop(columns=["isPartial"])

            for ticker in batch:
                if ticker not in df.columns:
                    continue

                series = df[ticker].dropna()
                if len(series) < 28:
                    # Not enough history for a meaningful comparison
                    continue

                # Last 7 data points = recent week; prior 21 = baseline
                recent = series.iloc[-7:]
                baseline = series.iloc[-28:-7]

                current_avg = float(recent.mean())
                baseline_avg = float(baseline.mean())

                if baseline_avg < 1:
                    # Avoid division by zero / meaningless signal
                    continue

                pct_change = (current_avg - baseline_avg) / baseline_avg * 100

                # Only generate an article if the signal is meaningful
                if current_avg > 10 and abs(pct_change) > 25:
                    article = _build_article(ticker, current_avg, baseline_avg, pct_change)
                    articles.append(article)
                    logger.info(
                        f"[google_trends] {ticker}: interest={current_avg:.1f}, "
                        f"baseline={baseline_avg:.1f}, Δ={pct_change:+.0f}%"
                    )

            time.sleep(_REQUEST_DELAY)

        except Exception as e:
            logger.warning(f"[google_trends] Batch {batch} failed: {e}")
            time.sleep(_REQUEST_DELAY * 2)

    logger.info(f"[google_trends] Generated {len(articles)} trend articles from {len(tickers)} tickers")
    _save_cache(articles)
    return articles
