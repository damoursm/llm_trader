"""Aggregate news sentiment scores into a unified TickerSignal."""

from loguru import logger
from typing import List
from src.models import NewsArticle, TickerSignal
from src.analysis.sentiment import analyse_sentiment, filter_relevant_articles


def build_signals(
    tickers: List[str],
    articles: List[NewsArticle],
) -> List[TickerSignal]:
    """Build a TickerSignal for each ticker based purely on news sentiment."""
    signals = []

    for ticker in tickers:
        relevant = filter_relevant_articles(ticker, articles)
        sentiment_score, rationale = analyse_sentiment(ticker, relevant)

        if sentiment_score >= 0.15:
            direction = "BULLISH"
        elif sentiment_score <= -0.15:
            direction = "BEARISH"
        else:
            direction = "NEUTRAL"

        confidence = min(1.0, abs(sentiment_score) / 0.5)

        signals.append(TickerSignal(
            ticker=ticker,
            direction=direction,
            confidence=round(confidence, 2),
            sentiment_score=round(sentiment_score, 3),
            technical_score=0.0,
            rationale=rationale,
        ))

        logger.info(
            f"{ticker}: {direction} (conf={confidence:.0%}) | "
            f"news_sentiment={sentiment_score:+.2f} | articles={len(relevant)}"
        )

    return signals
