"""Aggregate sentiment and technical scores into a unified TickerSignal."""

from loguru import logger
from typing import List
from src.models import NewsArticle, TickerSnapshot, TickerSignal
from src.analysis.sentiment import analyse_sentiment, filter_relevant_articles
from src.analysis.technical import compute_technical_score


# Weighting between sentiment and technical signals
SENTIMENT_WEIGHT = 0.6
TECHNICAL_WEIGHT = 0.4


def build_signals(
    snapshots: List[TickerSnapshot],
    articles: List[NewsArticle],
) -> List[TickerSignal]:
    """Build a TickerSignal for each ticker by combining sentiment + technical."""
    signals = []

    for snap in snapshots:
        ticker = snap.ticker

        # Sentiment analysis
        relevant = filter_relevant_articles(ticker, articles)
        sentiment_score, rationale = analyse_sentiment(ticker, relevant)

        # Technical analysis
        technical_score = compute_technical_score(ticker)

        # Composite score
        composite = (
            SENTIMENT_WEIGHT * sentiment_score +
            TECHNICAL_WEIGHT * technical_score
        )

        # Direction
        if composite >= 0.15:
            direction = "BULLISH"
        elif composite <= -0.15:
            direction = "BEARISH"
        else:
            direction = "NEUTRAL"

        # Confidence: how strong is the composite signal
        confidence = min(1.0, abs(composite) / 0.5)

        signals.append(TickerSignal(
            ticker=ticker,
            direction=direction,
            confidence=round(confidence, 2),
            sentiment_score=round(sentiment_score, 3),
            technical_score=round(technical_score, 3),
            rationale=rationale,
        ))

        logger.info(
            f"{ticker}: {direction} (conf={confidence:.0%}) | "
            f"sentiment={sentiment_score:+.2f} technical={technical_score:+.2f}"
        )

    return signals
