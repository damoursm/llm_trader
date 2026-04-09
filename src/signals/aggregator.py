"""
Build a TickerSignal for each ticker by combining all enabled analysis methods:

  Method 1 — News sentiment        (enable_news_sentiment)
  Method 2 — Technical analysis    (enable_technical_analysis)
  Method 3 — Smart money / insider (enable_insider_trades / options_flow / sec_filings)

Score combination weights when multiple methods are enabled:
  News + Technical + Insider  →  40% news, 30% technical, 30% insider score
  News + Technical            →  60% news, 40% technical
  News + Insider              →  55% news, 45% insider
  Technical + Insider         →  50% technical, 50% insider
  Single method               → 100% that method

Convergence multiplier (applied when ≥2 methods are enabled):
  ≥2 methods agree directionally  → ×1.25  (reward corroborated signals)
  Only 1 method drives the signal → ×0.60  (penalise uncorroborated signals)
  This ensures a single strong news print cannot alone produce a BUY/SELL.
"""

from loguru import logger
from typing import List, Optional

from config import settings
from src.models import NewsArticle, TickerSignal, InsiderTrade
from src.analysis.sentiment import analyse_sentiment, filter_relevant_articles
from src.data.insider_trades import build_insider_summary


# Direction and strength multipliers for each smart money signal type.
# Multiplier encodes conviction beyond raw dollar amount.
_TX_SCORE: dict = {
    # Bullish
    "unusual_call":       (+1.0, 1.5),   # (direction, multiplier)
    "13d_activist_stake": (+1.0, 2.0),   # activist = very strong catalyst signal
    "13g_passive_stake":  (+1.0, 0.8),
    "13f_new_position":   (+1.0, 1.0),
    "13f_increase":       (+1.0, 0.8),
    # Bearish
    "unusual_put":        (-1.0, 1.5),
    "planned_sale_144":   (-1.0, 0.9),
    "13f_exit":           (-1.0, 1.2),   # full exit = stronger bearish signal
    "13f_decrease":       (-1.0, 0.7),
}


def _insider_score(ticker: str, trades: List[InsiderTrade]) -> float:
    """
    Derive a [-1, +1] score from all smart money signals for a ticker.
    Handles politician/corporate insider trades (purchase/sale),
    options sweeps, activist stakes, and 13F position changes.
    """
    relevant = [t for t in trades if t.ticker.upper() == ticker.upper()]
    if not relevant:
        return 0.0

    from src.data.insider_trades import _amount_weight
    total = 0.0
    for t in relevant:
        w = _amount_weight(t.amount_range)
        tx = t.transaction_type

        if tx in _TX_SCORE:
            direction, multiplier = _TX_SCORE[tx]
            total += direction * w * multiplier
        elif "purchase" in tx:
            total += w
        elif "sale" in tx:
            total -= w
        # else: neutral / unknown type — skip

    # Normalise to [-1, +1]
    return round(max(-1.0, min(1.0, total / 3.0)), 3)


def build_signals(
    tickers: List[str],
    articles: List[NewsArticle],
    insider_trades: Optional[List[InsiderTrade]] = None,
) -> List[TickerSignal]:
    """Build a TickerSignal for each ticker using all enabled methods."""

    use_news    = settings.enable_news_sentiment
    use_tech    = settings.enable_technical_analysis and settings.enable_market_data
    use_insider = (
        (settings.enable_insider_trades or
         settings.enable_options_flow or
         settings.enable_sec_filings)
        and insider_trades is not None
    )

    if not use_news and not use_tech and not use_insider:
        logger.warning("All analysis methods disabled — no signals will be generated.")
        return []

    # Determine combination weights
    active = sum([use_news, use_tech, use_insider])
    if active == 3:
        w_news, w_tech, w_insider = 0.40, 0.30, 0.30   # insider given equal weight to tech
    elif use_news and use_tech:
        w_news, w_tech, w_insider = 0.60, 0.40, 0.00
    elif use_news and use_insider:
        w_news, w_tech, w_insider = 0.55, 0.00, 0.45
    elif use_tech and use_insider:
        w_news, w_tech, w_insider = 0.00, 0.50, 0.50
    elif use_news:
        w_news, w_tech, w_insider = 1.00, 0.00, 0.00
    elif use_tech:
        w_news, w_tech, w_insider = 0.00, 1.00, 0.00
    else:
        w_news, w_tech, w_insider = 0.00, 0.00, 1.00

    logger.info(
        f"Signal weights — news={w_news:.0%}  tech={w_tech:.0%}  insider={w_insider:.0%}"
    )

    # Lazy-import technical scorer only when needed (avoids yfinance calls otherwise)
    if use_tech:
        from src.analysis.technical import compute_technical_score

    signals = []

    for ticker in tickers:
        # --- Method 1: News sentiment ---
        sentiment_score = 0.0
        news_rationale  = "News sentiment disabled."
        if use_news:
            relevant = filter_relevant_articles(ticker, articles)
            sentiment_score, news_rationale = analyse_sentiment(ticker, relevant)

        # --- Method 2: Technical analysis ---
        technical_score = 0.0
        tech_rationale  = ""
        if use_tech:
            technical_score = compute_technical_score(ticker)
            tech_rationale  = f"Technical score: {technical_score:+.2f}"

        # --- Method 3: Insider trades ---
        insider_score   = 0.0
        insider_summary = ""
        if use_insider:
            insider_score   = _insider_score(ticker, insider_trades)
            insider_summary = build_insider_summary(ticker, insider_trades)

        # --- Combine ---
        combined = (
            w_news    * sentiment_score +
            w_tech    * technical_score +
            w_insider * insider_score
        )

        if combined >= 0.15:
            direction = "BULLISH"
        elif combined <= -0.15:
            direction = "BEARISH"
        else:
            direction = "NEUTRAL"

        # --- Convergence-aware confidence ---
        # Count how many enabled methods have a meaningful score that agrees
        # with the combined direction (threshold 0.10 avoids noise).
        _AGREE_THRESHOLD = 0.10
        sources_agreeing = 0
        if use_news and abs(sentiment_score) >= _AGREE_THRESHOLD:
            if (combined > 0 and sentiment_score > 0) or (combined < 0 and sentiment_score < 0):
                sources_agreeing += 1
        if use_tech and abs(technical_score) >= _AGREE_THRESHOLD:
            if (combined > 0 and technical_score > 0) or (combined < 0 and technical_score < 0):
                sources_agreeing += 1
        if use_insider and abs(insider_score) >= _AGREE_THRESHOLD:
            if (combined > 0 and insider_score > 0) or (combined < 0 and insider_score < 0):
                sources_agreeing += 1

        # Convergence multiplier — only applied when ≥2 methods are enabled.
        # Prevents a single strong signal from crossing the BUY/SELL threshold alone.
        if active >= 2:
            convergence_factor = 1.25 if sources_agreeing >= 2 else 0.60
        else:
            convergence_factor = 1.0   # single-method mode: no adjustment

        raw_confidence = min(1.0, abs(combined) / 0.5)
        confidence = round(min(1.0, raw_confidence * convergence_factor), 2)

        # Build rationale from active methods
        rationale_parts = []
        if use_news:
            rationale_parts.append(news_rationale)
        if use_tech and tech_rationale:
            rationale_parts.append(tech_rationale)
        rationale = " | ".join(rationale_parts) if rationale_parts else "No rationale available."

        signals.append(TickerSignal(
            ticker=ticker,
            direction=direction,
            confidence=confidence,
            sentiment_score=round(sentiment_score, 3),
            technical_score=round(technical_score, 3),
            insider_score=round(insider_score, 3),
            rationale=rationale,
            insider_summary=insider_summary,
            sources_agreeing=sources_agreeing,
        ))

        logger.info(
            f"{ticker}: {direction} (conf={confidence:.0%}, {sources_agreeing}/{active} sources agree) | "
            f"news={sentiment_score:+.2f}  tech={technical_score:+.2f}  "
            f"insider={insider_score:+.2f}  combined={combined:+.2f}  "
            f"convergence={convergence_factor:.2f}x"
        )

    return signals
