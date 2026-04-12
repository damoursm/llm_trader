"""
Build a TickerSignal for each ticker by combining all enabled analysis methods:

  Method 1 — News sentiment        (enable_news_sentiment)
  Method 2 — Technical analysis    (enable_technical_analysis)
  Method 3 — Smart money / insider (enable_insider_trades / options_flow / sec_filings)
  Method 4 — Put/call ratio        (enable_put_call, per-ticker only)

Base weights (normalised to 1.0 across active methods):
  news=0.40  technical=0.30  insider=0.30  put_call=0.15

Convergence multiplier (applied when ≥2 methods are enabled):
  ≥2 methods agree directionally  → ×1.25  (reward corroborated signals)
  Only 1 method drives the signal → ×0.60  (penalise uncorroborated signals)
  This ensures a single strong signal source cannot alone produce a BUY/SELL.
"""

from loguru import logger
from typing import List, Optional

from config import settings
from src.models import NewsArticle, TickerSignal, InsiderTrade
from src.analysis.sentiment import analyse_sentiment, filter_relevant_articles
from src.data.insider_trades import build_insider_summary


# Direction and strength multipliers for each smart money signal type.
_TX_SCORE: dict = {
    # Bullish
    "unusual_call":       (+1.0, 1.5),
    "13d_activist_stake": (+1.0, 2.0),
    "13g_passive_stake":  (+1.0, 0.8),
    "13f_new_position":   (+1.0, 1.0),
    "13f_increase":       (+1.0, 0.8),
    # Bearish
    "unusual_put":        (-1.0, 1.5),
    "planned_sale_144":   (-1.0, 0.9),
    "13f_exit":           (-1.0, 1.2),
    "13f_decrease":       (-1.0, 0.7),
}

# Base weights — normalised across active methods at runtime
_BASE_WEIGHTS = {
    "news":     0.40,
    "tech":     0.30,
    "insider":  0.30,
    "put_call": 0.15,
}

# Put/call signal → directional score mapping
_PC_SIGNAL_SCORE = {
    "EXTREME_PUTS":  +0.70,   # contrarian bullish
    "PUTS_HEAVY":    +0.35,
    "BALANCED":       0.00,
    "CALLS_HEAVY":   -0.35,
    "EXTREME_CALLS": -0.70,   # contrarian bearish
}


def _insider_score(ticker: str, trades: List[InsiderTrade]) -> float:
    """Derive a [-1, +1] score from all smart money signals for a ticker."""
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

    return round(max(-1.0, min(1.0, total / 3.0)), 3)


def _put_call_score_for(ticker: str, put_call_context) -> float:
    """
    Extract a [-1, +1] directional score from per-ticker put/call data.
    High put/call → bullish (contrarian), high call/put → bearish (contrarian).
    Returns 0.0 if no data for this ticker or put/call is disabled.
    """
    if put_call_context is None:
        return 0.0
    for sig in put_call_context.ticker_signals:
        if sig.ticker.upper() == ticker.upper():
            return _PC_SIGNAL_SCORE.get(sig.signal, 0.0)
    return 0.0


# Stock → sector ETF mapping for alignment check
# When a stock signal contradicts its sector ETF signal, reduce confidence.
_SECTOR_MAP: dict = {
    # Technology
    "AAPL": "XLK", "MSFT": "XLK", "NVDA": "XLK", "AMD": "XLK",
    "INTC": "XLK", "ORCL": "XLK", "ADBE": "XLK", "CRM": "XLK",
    # Consumer Discretionary
    "AMZN": "XLY", "TSLA": "XLY", "UBER": "XLY", "LYFT": "XLY",
    # Communication Services
    "META": "XLC", "GOOGL": "XLC", "GOOG": "XLC", "NFLX": "XLC",
    # Financials
    "JPM": "XLF", "BAC": "XLF", "GS": "XLF", "MS": "XLF",
    # Payment networks overlap tech/financials
    "PYPL": "XLF",
}

# Sector alignment confidence adjustments
_SECTOR_ALIGN_BOOST   = 1.10   # stock and sector agree → mild boost
_SECTOR_ALIGN_PENALTY = 0.75   # stock contradicts sector → meaningful penalty


def _sector_alignment_factor(ticker: str, combined: float, signals_by_ticker: dict) -> float:
    """
    Compare this ticker's signal direction to its sector ETF signal.
    Returns a multiplier applied to confidence.

    Only applied to individual stocks (not ETFs — they ARE the sector).
    Requires the sector ETF to itself have a meaningful signal (|combined| > 0.10).
    """
    sector_etf = _SECTOR_MAP.get(ticker.upper())
    if sector_etf is None:
        return 1.0   # no sector mapping — no adjustment

    sector_sig = signals_by_ticker.get(sector_etf)
    if sector_sig is None:
        return 1.0   # sector ETF not in this run

    sector_combined = (
        sector_sig.sentiment_score * 0.40 +
        sector_sig.technical_score * 0.30 +
        sector_sig.insider_score   * 0.30
    )
    if abs(sector_combined) < 0.10:
        return 1.0   # sector signal too weak to inform alignment

    stock_bull  = combined > 0
    sector_bull = sector_combined > 0
    if stock_bull == sector_bull:
        logger.debug(f"{ticker} ↔ {sector_etf} ALIGNED (sector={sector_combined:+.2f}) → +boost")
        return _SECTOR_ALIGN_BOOST
    else:
        logger.debug(f"{ticker} ↔ {sector_etf} CONTRADICTS (sector={sector_combined:+.2f}) → penalty")
        return _SECTOR_ALIGN_PENALTY


def _normalised_weights(active_flags: dict) -> dict:
    """
    Given a dict of {method: bool}, return normalised weights summing to 1.0
    using each method's base weight. Inactive methods get weight 0.0.
    All keys are always present in the returned dict.
    """
    raw = {m: (_BASE_WEIGHTS[m] if on else 0.0) for m, on in active_flags.items()}
    total = sum(raw.values())
    if total == 0:
        return {m: 0.0 for m in active_flags}
    return {m: w / total for m, w in raw.items()}


def build_signals(
    tickers: List[str],
    articles: List[NewsArticle],
    insider_trades: Optional[List[InsiderTrade]] = None,
    put_call_context=None,   # Optional[PutCallContext]
) -> List[TickerSignal]:
    """Build a TickerSignal for each ticker using all enabled methods."""

    use_news     = settings.enable_news_sentiment
    use_tech     = settings.enable_technical_analysis and settings.enable_market_data
    use_insider  = (
        (settings.enable_insider_trades or
         settings.enable_options_flow or
         settings.enable_sec_filings)
        and insider_trades is not None
    )
    use_put_call = settings.enable_put_call and put_call_context is not None

    if not use_news and not use_tech and not use_insider and not use_put_call:
        logger.warning("All analysis methods disabled — no signals will be generated.")
        return []

    active_flags = {
        "news":     use_news,
        "tech":     use_tech,
        "insider":  use_insider,
        "put_call": use_put_call,
    }
    weights = _normalised_weights(active_flags)
    active_count = sum(active_flags.values())

    logger.info(
        f"Signal weights — "
        f"news={weights['news']:.0%}  tech={weights['tech']:.0%}  "
        f"insider={weights['insider']:.0%}  put_call={weights['put_call']:.0%}"
    )

    if use_tech:
        from src.analysis.technical import compute_technical_score

    signals = []
    combined_scores: dict = {}   # ticker → raw combined score (for sector alignment)

    for ticker in tickers:
        # --- Method 1: News sentiment (all article-based sources) ---
        sentiment_score = 0.0
        news_rationale  = "News sentiment disabled."
        if use_news:
            relevant = filter_relevant_articles(ticker, articles)
            sentiment_score, news_rationale = analyse_sentiment(ticker, relevant)

        # --- Method 2: Technical analysis ---
        technical_score = 0.0
        if use_tech:
            technical_score = compute_technical_score(ticker)

        # --- Method 3: Smart money (insider trades, options flow, SEC filings) ---
        insider_sc   = 0.0
        insider_summary = ""
        if use_insider:
            insider_sc      = _insider_score(ticker, insider_trades)
            insider_summary = build_insider_summary(ticker, insider_trades)

        # --- Method 4: Put/call ratio (per-ticker, contrarian) ---
        pc_score = 0.0
        if use_put_call:
            pc_score = _put_call_score_for(ticker, put_call_context)

        # --- Combine with normalised weights ---
        combined = (
            weights["news"]     * sentiment_score +
            weights["tech"]     * technical_score +
            weights["insider"]  * insider_sc +
            weights["put_call"] * pc_score
        )

        if combined >= 0.15:
            direction = "BULLISH"
        elif combined <= -0.15:
            direction = "BEARISH"
        else:
            direction = "NEUTRAL"

        # --- Convergence: count methods that agree with combined direction ---
        _AGREE_THRESHOLD = 0.10
        sources_agreeing = 0
        method_scores = [
            (use_news,     sentiment_score),
            (use_tech,     technical_score),
            (use_insider,  insider_sc),
            (use_put_call, pc_score),
        ]
        for enabled, score in method_scores:
            if not enabled or abs(score) < _AGREE_THRESHOLD:
                continue
            if (combined > 0 and score > 0) or (combined < 0 and score < 0):
                sources_agreeing += 1

        if active_count >= 2:
            convergence_factor = 1.25 if sources_agreeing >= 2 else 0.60
        else:
            convergence_factor = 1.0

        raw_confidence = min(1.0, abs(combined) / 0.5)
        confidence = round(min(1.0, raw_confidence * convergence_factor), 2)

        # Build rationale
        rationale_parts = []
        if use_news:
            rationale_parts.append(news_rationale)
        if use_tech and technical_score != 0:
            rationale_parts.append(f"Technical score: {technical_score:+.2f}")
        if use_put_call and pc_score != 0:
            rationale_parts.append(f"Put/call signal: {pc_score:+.2f}")
        rationale = " | ".join(rationale_parts) if rationale_parts else "No rationale available."

        combined_scores[ticker] = combined
        signals.append(TickerSignal(
            ticker=ticker,
            direction=direction,
            confidence=confidence,
            sentiment_score=round(sentiment_score, 3),
            technical_score=round(technical_score, 3),
            insider_score=round(insider_sc, 3),
            put_call_score=round(pc_score, 3),
            rationale=rationale,
            insider_summary=insider_summary,
            sources_agreeing=sources_agreeing,
        ))

        logger.info(
            f"{ticker}: {direction} (conf={confidence:.0%}, {sources_agreeing}/{active_count} sources agree) | "
            f"news={sentiment_score:+.2f}  tech={technical_score:+.2f}  "
            f"insider={insider_sc:+.2f}  pc={pc_score:+.2f}  "
            f"combined={combined:+.2f}  convergence={convergence_factor:.2f}x"
        )

    # ── Second pass: sector alignment ────────────────────────────────────────
    # Cross-reference each stock against its sector ETF signal.
    # ETFs and tickers with no sector mapping are unchanged.
    signals_by_ticker = {s.ticker: s for s in signals}
    final_signals = []
    for sig in signals:
        raw_combined = combined_scores.get(sig.ticker, 0.0)
        sector_factor = _sector_alignment_factor(sig.ticker, raw_combined, signals_by_ticker)
        if sector_factor != 1.0:
            adjusted = round(min(1.0, sig.confidence * sector_factor), 2)
            sig = sig.model_copy(update={"confidence": adjusted})
        final_signals.append(sig)

    return final_signals
