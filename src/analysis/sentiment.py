"""LLM-based sentiment analysis of news articles using Claude."""

import anthropic
from loguru import logger
from typing import List
from config import settings
from src.models import NewsArticle


_client = None


def _get_client() -> anthropic.Anthropic:
    global _client
    if _client is None:
        _client = anthropic.Anthropic(api_key=settings.anthropic_api_key)
    return _client


def analyse_sentiment(ticker: str, articles: List[NewsArticle]) -> tuple[float, str]:
    """
    Use Claude to score sentiment for a ticker given a list of news articles.

    Returns:
        (score, rationale)
        score: float in [-1.0, +1.0], positive = bullish
        rationale: brief explanation
    """
    if not articles:
        return 0.0, "No recent news articles found."

    # Build news digest (cap at 15 most recent articles)
    digest = "\n\n".join(
        f"[{a.source}] {a.title}\n{a.summary[:300]}"
        for a in sorted(articles, key=lambda x: x.published_at, reverse=True)[:15]
    )

    prompt = f"""You are a professional equity analyst. Analyse the following recent news headlines and summaries for **{ticker}** and determine the likely short-term (1–5 day) directional impact on the stock price.

<news>
{digest}
</news>

Respond with a JSON object with exactly these fields:
- "score": float between -1.0 (very bearish) and +1.0 (very bullish), 0.0 is neutral
- "rationale": one to three sentences explaining the key drivers

Example: {{"score": 0.6, "rationale": "Strong earnings beat and raised guidance dominate headlines."}}

Respond with JSON only, no markdown."""

    try:
        client = _get_client()
        message = client.messages.create(
            model="claude-opus-4-6",
            max_tokens=256,
            messages=[{"role": "user", "content": prompt}],
        )
        import json
        raw = message.content[0].text.strip()
        data = json.loads(raw)
        score = float(data["score"])
        rationale = str(data["rationale"])
        score = max(-1.0, min(1.0, score))
        logger.info(f"{ticker} sentiment score: {score:.2f}")
        return score, rationale
    except Exception as e:
        logger.error(f"Sentiment analysis failed for {ticker}: {e}")
        return 0.0, f"Analysis error: {e}"


def filter_relevant_articles(ticker: str, articles: List[NewsArticle]) -> List[NewsArticle]:
    """Simple keyword filter to keep articles likely relevant to a ticker."""
    keywords = {ticker.lower()}
    # Add common variations (e.g. AAPL → apple)
    ticker_aliases = {
        "AAPL": ["apple"], "MSFT": ["microsoft"], "NVDA": ["nvidia"],
        "TSLA": ["tesla"], "AMZN": ["amazon"], "META": ["meta", "facebook"],
        "GOOGL": ["google", "alphabet"], "GOOG": ["google", "alphabet"],
        "XLK": ["technology", "tech sector"], "XLF": ["financials", "banks"],
        "XLE": ["energy", "oil"], "XLV": ["health", "biotech"],
    }
    keywords.update(ticker_aliases.get(ticker, []))

    relevant = [
        a for a in articles
        if any(kw in (a.title + a.summary).lower() for kw in keywords)
    ]
    # Fall back to all articles if filter is too aggressive
    return relevant if len(relevant) >= 3 else articles
