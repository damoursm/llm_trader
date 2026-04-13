"""
Build a TickerSignal for each ticker by combining all enabled analysis methods.

Methods
───────
  1  News sentiment        (enable_news_sentiment)
  2  Technical analysis    (enable_technical_analysis)
  3  Smart money / insider (enable_insider_trades / options_flow / sec_filings)
  4  Put/call ratio        (enable_put_call, per-ticker contrarian)

Confidence formula
──────────────────
  confidence = raw_conf × coherence_factor × movement_factor × volume_factor

  raw_conf         — abs(combined_score) normalised to [0, 1].
  coherence_factor — continuous [0.45, 1.35]: how strongly do all methods agree?
                     Replaces the old binary 1.25×/0.60× toggle. Magnitude-weighted,
                     so a weak outlier costs less than a strong one pointing opposite.
  movement_factor  — [0.80, 1.20]: is the stock actually likely to move?
                     Derived from ATR% + BB-width%. Low-volatility stocks are
                     unlikely to deliver on a signal; high-volatility ones are "in play".
  volume_factor    — [0.90, 1.15]: does recent volume confirm the aggregate direction?
                     Separate from the vol multiplier already baked into technical_score;
                     this operates at the cross-method level.

Interaction adjustments (additive, applied before confidence, capped ±0.15)
────────────────────────────────────────────────────────────────────────────
  • Insider accumulation at technical support — insiders buying while price is
    technically weak signals contrarian value accumulation.
  • Options extreme + technical aligned — extreme put/call skew at a price
    inflection is a classic squeeze/unwind setup.
  • News catalyst confirmed by volume — a strong sentiment signal backed by
    above-average volume is more likely to be a real market-moving event.

Sector alignment (second pass, unchanged from before)
──────────────────────────────────────────────────────
  Individual stocks are cross-referenced against their sector ETF direction.
  Alignment → mild boost (1.10×). Contradiction → meaningful penalty (0.75×).
"""

from datetime import date
from loguru import logger
from typing import List, Optional

from config import settings
from src.models import NewsArticle, Direction, TickerSignal, InsiderTrade
from src.analysis.sentiment import analyse_sentiment, filter_relevant_articles
from src.data.insider_trades import build_insider_summary
from src.analysis.technical import TechnicalResult, EMPTY_RESULT, compute_technical_score
from src.signals.vwap import compute_vwap_score


# ── Smart-money signal direction + strength ──────────────────────────────────

_TX_SCORE: dict = {
    "unusual_call":       (+1.0, 1.5),
    "13d_activist_stake": (+1.0, 2.0),
    "13g_passive_stake":  (+1.0, 0.8),
    "13f_new_position":   (+1.0, 1.0),
    "13f_increase":       (+1.0, 0.8),
    "unusual_put":        (-1.0, 1.5),
    "planned_sale_144":   (-1.0, 0.9),
    "13f_exit":           (-1.0, 1.2),
    "13f_decrease":       (-1.0, 0.7),
}

# Base weights — normalised across active methods at runtime
_BASE_WEIGHTS = {
    "news":      0.40,
    "tech":      0.30,
    "insider":   0.30,
    "put_call":  0.15,
    "max_pain":  0.12,   # options-expiry gravity; weight fades automatically via expiry_factor
    "oi_skew":   0.15,   # OI-weighted call/put directional positioning
    "vwap":      0.12,   # mean-reversion: price vs rolling 20-day VWAP
}

# Put/call contrarian score mapping
_PC_SIGNAL_SCORE = {
    "EXTREME_PUTS":  +0.70,
    "PUTS_HEAVY":    +0.35,
    "BALANCED":       0.00,
    "CALLS_HEAVY":   -0.35,
    "EXTREME_CALLS": -0.70,
}

# Sector → ETF mapping for sector-alignment second pass
_SECTOR_MAP: dict = {
    "AAPL": "XLK", "MSFT": "XLK", "NVDA": "XLK", "AMD": "XLK",
    "INTC": "XLK", "ORCL": "XLK", "ADBE": "XLK", "CRM": "XLK",
    "AMZN": "XLY", "TSLA": "XLY", "UBER": "XLY", "LYFT": "XLY",
    "META": "XLC", "GOOGL": "XLC", "GOOG": "XLC", "NFLX": "XLC",
    "JPM":  "XLF", "BAC":  "XLF", "GS":   "XLF", "MS":   "XLF",
    "PYPL": "XLF",
}
_SECTOR_ALIGN_BOOST   = 1.10
_SECTOR_ALIGN_PENALTY = 0.75

_AGREE_THRESHOLD = 0.10   # minimum |score| to count a method as having a view


# ── Per-method score helpers ──────────────────────────────────────────────────

def _insider_score(ticker: str, trades: List[InsiderTrade]) -> float:
    """Derive a [-1, +1] score from all smart money signals for a ticker."""
    relevant = [t for t in trades if t.ticker.upper() == ticker.upper()]
    if not relevant:
        return 0.0
    from src.data.insider_trades import _amount_weight
    total = 0.0
    for t in relevant:
        w  = _amount_weight(t.amount_range)
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
    if put_call_context is None:
        return 0.0
    for sig in put_call_context.ticker_signals:
        if sig.ticker.upper() == ticker.upper():
            return _PC_SIGNAL_SCORE.get(sig.signal, 0.0)
    return 0.0


def _normalised_weights(active_flags: dict) -> dict:
    raw   = {m: (_BASE_WEIGHTS[m] if on else 0.0) for m, on in active_flags.items()}
    total = sum(raw.values())
    if total == 0:
        return {m: 0.0 for m in active_flags}
    return {m: w / total for m, w in raw.items()}


# ── Coherence ─────────────────────────────────────────────────────────────────

def _coherence_factor(combined: float, method_scores: list) -> tuple[float, float]:
    """
    Return (coherence_ratio, coherence_factor).

    coherence_ratio ∈ [0, 1]:
        Magnitude-weighted fraction of active methods that agree with the
        combined direction. A weak outlier pointing opposite hurts less than
        a strong one; methods with |score| < AGREE_THRESHOLD are ignored.

    coherence_factor ∈ [0.45, 1.35]:
        Continuous multiplier — replaces the old binary 1.25×/0.60× toggle.
        Maps coherence_ratio linearly: 0 → 0.45, 1 → 1.35.
    """
    agree_w = 0.0
    total_w = 0.0
    for enabled, score in method_scores:
        if not enabled or abs(score) < _AGREE_THRESHOLD:
            continue
        total_w += abs(score)
        if (combined > 0 and score > 0) or (combined < 0 and score < 0):
            agree_w += abs(score)

    if total_w == 0:
        return 0.0, 0.60   # no method has a view — moderate penalty

    ratio  = agree_w / total_w                  # ∈ [0, 1]
    factor = round(0.45 + ratio * 0.90, 3)      # ∈ [0.45, 1.35]
    return ratio, factor


# ── Movement potential ────────────────────────────────────────────────────────

def _movement_factor(atr_pct: float, bb_width_pct: float) -> float:
    """
    Answers: "if a signal fires, how likely is this stock to actually move?"

    Uses ATR% (typical daily range / price) and BB-width% (Bollinger Band
    squeeze vs expansion) as complementary volatility regime measures.

    Thresholds are calibrated for typical large-cap stocks:
      ATR% ~0.8–2% = normal, > 2.5% = high vol, < 0.5% = very quiet.
      BB-width ~4–8% = normal, > 10% = wide/trending, < 2% = squeeze.

    A stock in a vol squeeze (narrow BB) may be about to move — but the
    direction is unknown, so we don't boost; we leave it neutral. The boost
    only applies when ATR confirms the stock already moves meaningfully.
    """
    # ATR-based score: 5 bands → [0.80, 1.20]
    if atr_pct >= 0.030:
        atr_score = 1.20
    elif atr_pct >= 0.020:
        atr_score = 1.10
    elif atr_pct >= 0.010:
        atr_score = 1.00
    elif atr_pct >= 0.005:
        atr_score = 0.90
    else:
        atr_score = 0.80

    # BB-width score: wide band means stock is already in motion
    if bb_width_pct >= 0.12:
        bb_score = 1.10
    elif bb_width_pct >= 0.06:
        bb_score = 1.00
    elif bb_width_pct >= 0.03:
        bb_score = 0.95
    else:
        bb_score = 0.88    # very tight BB (squeeze) — movement imminent but direction unclear

    # Average the two, clamp to [0.80, 1.20]
    raw = (atr_score + bb_score) / 2
    return round(max(0.80, min(1.20, raw)), 3)


# ── Volume cross-method factor ────────────────────────────────────────────────

def _volume_factor(vol_ratio: float, abs_combined: float, coherence_ratio: float) -> float:
    """
    Cross-method volume confirmation: does current volume support the aggregate signal?

    This is distinct from the volume multiplier already baked into technical_score.
    That one shapes the *technical sub-score*. This one asks: given that multiple
    methods are pointing in the same direction, is the market actually trading on it?

    Only applies a meaningful boost when:
      - Volume is elevated (vol_ratio >= 1.5)
      - The combined signal is non-trivial (abs_combined >= 0.25)
      - Methods are at least moderately coherent (coherence_ratio >= 0.5)
    Otherwise applies a mild dampening for low-volume signals.
    """
    if vol_ratio >= 2.0 and abs_combined >= 0.30 and coherence_ratio >= 0.5:
        return 1.15
    if vol_ratio >= 1.5 and abs_combined >= 0.25 and coherence_ratio >= 0.5:
        return 1.08
    if vol_ratio <= 0.5:
        return 0.92   # suspiciously low volume — signal may not be market-confirmed
    return 1.00


# ── Interaction adjustments ───────────────────────────────────────────────────

def _interaction_adjustment(
    combined: float,
    sentiment_score: float,
    technical_score: float,
    insider_sc: float,
    pc_score: float,
    vol_ratio: float,
) -> float:
    """
    Detect three specific market setups and return a small additive correction
    to the combined score (capped to ±0.15 in total).

    These are cases where two methods used together are more informative than
    their individual contributions to the weighted sum suggest.

    1. Smart money accumulation at technical support
       Insiders buying while the stock looks technically weak is a contrarian
       value-accumulation setup — institutions building a position before a
       technical breakout. The combined score may be artificially low because
       the technical drag cancels the insider signal.

    2. Options extreme + technical alignment
       Extreme put/call skew (heavy puts) at a technical support is a classic
       short-squeeze or capitulation setup. The contrarian put/call signal and
       the technical support are reinforcing.

    3. Strong news + volume confirmation
       A high-magnitude sentiment signal backed by 1.5× normal volume is more
       likely to be a genuine market-moving event than the same news on thin
       volume. Boost when both are present and aligned.
    """
    direction = 1 if combined >= 0 else -1
    adj = 0.0

    # 1. Insider accumulation at technical support
    #    Criteria: insiders bullish (>0.30), technical score weak/bearish (<-0.20),
    #              but combined still leans bullish — insiders are more right.
    if insider_sc > 0.30 and technical_score < -0.20 and combined > 0:
        adj += 0.10
        logger.debug("Interaction: insider accumulation at technical support (+0.10)")
    elif insider_sc < -0.30 and technical_score > 0.20 and combined < 0:
        adj -= 0.10
        logger.debug("Interaction: insider distribution at technical peak (-0.10)")

    # 2. Options extreme aligned with combined direction
    #    pc_score is contrarian: large positive = extreme puts → bullish bias.
    #    If options market is at an extreme AND it agrees with combined direction,
    #    the setup is more compelling.
    if abs(pc_score) >= 0.50 and (pc_score * direction) > 0:
        adj += 0.07 * direction
        logger.debug(f"Interaction: options extreme aligned with signal ({adj:+.2f})")

    # 3. News catalyst confirmed by volume
    #    High-magnitude sentiment AND elevated volume: news is being acted on.
    if abs(sentiment_score) >= 0.40 and vol_ratio >= 1.5 and (sentiment_score * direction) > 0:
        adj += 0.06 * direction
        logger.debug(f"Interaction: news catalyst confirmed by volume spike ({adj:+.2f})")

    return max(-0.15, min(0.15, adj))


# ── Sector alignment (second pass) ───────────────────────────────────────────

def _sector_alignment_factor(ticker: str, combined: float, signals_by_ticker: dict) -> float:
    sector_etf = _SECTOR_MAP.get(ticker.upper())
    if sector_etf is None:
        return 1.0
    sector_sig = signals_by_ticker.get(sector_etf)
    if sector_sig is None:
        return 1.0
    sector_combined = (
        sector_sig.sentiment_score * 0.40 +
        sector_sig.technical_score * 0.30 +
        sector_sig.insider_score   * 0.30
    )
    if abs(sector_combined) < 0.10:
        return 1.0
    if (combined > 0) == (sector_combined > 0):
        logger.debug(f"{ticker} ↔ {sector_etf} ALIGNED (sector={sector_combined:+.2f}) → +boost")
        return _SECTOR_ALIGN_BOOST
    logger.debug(f"{ticker} ↔ {sector_etf} CONTRADICTS (sector={sector_combined:+.2f}) → penalty")
    return _SECTOR_ALIGN_PENALTY


# ── Main entry point ──────────────────────────────────────────────────────────

def _gex_for(ticker: str, gex_context) -> tuple[str, Optional[float], str, float, float]:
    """
    Look up GEX data for a ticker.
    Returns (gex_signal, gamma_flip, max_pain_bias, expected_move_pct, oi_skew).
    """
    if gex_context is None:
        return "", None, "", 0.0, 0.0
    for sig in gex_context.signals:
        if sig.ticker.upper() == ticker.upper():
            return sig.gex_signal, sig.gamma_flip, sig.max_pain_bias, sig.expected_move_pct, sig.oi_skew
    return "", None, "", 0.0, 0.0


def _max_pain_score_for(ticker: str, gex_context) -> float:
    """
    Convert max pain distance into a [-1, +1] signal score.

    Logic
    ─────
    Max pain is the strike price where the aggregate dollar loss across all
    open options is minimised.  Market makers with net-short options books
    have an incentive to pin price near this level into expiration, creating
    a weak but statistically observable gravitational pull.

    Score formula
    ─────────────
      distance  = (max_pain - spot) / spot        signed, positive → bullish pull
      expiry_factor ∈ [0.05, 1.0]                 falls quickly beyond 7 days
      raw_score = distance × expiry_factor × 5    scale to make typical values ~±0.2–0.5

    The 5× scale compensates for typical distances (0.5–5%).  A 2% deviation
    3 days before expiry → 0.02 × 0.90 × 5 = 0.09 — a modest but real input.
    Beyond 14 days the weight is < 0.20, making this a near-zero contributor
    until the final two weeks of the cycle.
    """
    if gex_context is None:
        return 0.0
    for sig in gex_context.signals:
        if sig.ticker.upper() != ticker.upper():
            continue
        if sig.max_pain is None or sig.spot_price <= 0:
            return 0.0

        # Days until nearest expiry (use dominant_expiry stored on the signal)
        try:
            days = (date.fromisoformat(sig.dominant_expiry) - date.today()).days
        except (ValueError, AttributeError):
            return 0.0

        if days <= 0:
            return 0.0

        # Expiry-decay factor — strongest in the final week
        if days <= 2:
            expiry_factor = 1.00
        elif days <= 4:
            expiry_factor = 0.85
        elif days <= 7:
            expiry_factor = 0.65
        elif days <= 10:
            expiry_factor = 0.40
        elif days <= 14:
            expiry_factor = 0.22
        elif days <= 21:
            expiry_factor = 0.10
        else:
            expiry_factor = 0.05

        distance = (sig.max_pain - sig.spot_price) / sig.spot_price
        raw = distance * expiry_factor * 5.0
        score = round(max(-1.0, min(1.0, raw)), 3)

        if abs(score) >= 0.05:
            logger.debug(
                f"[max_pain] {ticker}: pain=${sig.max_pain:.2f} spot=${sig.spot_price:.2f} "
                f"dist={distance:+.2%} expiry={sig.dominant_expiry}({days}d) "
                f"factor={expiry_factor} → score={score:+.3f}"
            )
        return score
    return 0.0


def _gex_movement_modifier(gex_signal: str) -> float:
    """
    Adjust the movement_factor based on dealer gamma positioning.

    PINNED:    dealers are long gamma → they suppress moves → signal less likely to fire
               quickly → dampen movement factor.
    AMPLIFIED: dealers are short gamma → moves will be amplified → signal more likely
               to materialise and do so fast → boost movement factor.
    """
    if gex_signal == "PINNED":
        return 0.85
    if gex_signal == "AMPLIFIED":
        return 1.15
    return 1.0


def build_signals(
    tickers: List[str],
    articles: List[NewsArticle],
    insider_trades: Optional[List[InsiderTrade]] = None,
    put_call_context=None,
    gex_context=None,      # Optional[GEXContext]
) -> List[TickerSignal]:
    """Build a TickerSignal for each ticker using all enabled methods."""

    use_news      = settings.enable_news_sentiment
    use_tech      = settings.enable_technical_analysis and settings.enable_fetch_data
    use_insider   = (
        (settings.enable_insider_trades or
         settings.enable_options_flow or
         settings.enable_sec_filings)
        and insider_trades is not None
    )
    use_put_call  = settings.enable_put_call and put_call_context is not None
    use_max_pain  = settings.enable_gex and gex_context is not None
    use_oi_skew   = settings.enable_gex and gex_context is not None
    # VWAP uses the OHLCV chart cache first, so it works even with ENABLE_FETCH_DATA=false.
    use_vwap      = settings.enable_vwap

    if not use_news and not use_tech and not use_insider and not use_put_call:
        logger.warning("All analysis methods disabled — no signals will be generated.")
        return []

    active_flags = {
        "news":      use_news,
        "tech":      use_tech,
        "insider":   use_insider,
        "put_call":  use_put_call,
        "max_pain":  use_max_pain,
        "oi_skew":   use_oi_skew,
        "vwap":      use_vwap,
    }
    weights      = _normalised_weights(active_flags)
    active_count = sum(active_flags.values())

    logger.info(
        f"Signal weights — "
        f"news={weights['news']:.0%}  tech={weights['tech']:.0%}  "
        f"insider={weights['insider']:.0%}  put_call={weights['put_call']:.0%}  "
        f"max_pain={weights['max_pain']:.0%}  oi_skew={weights['oi_skew']:.0%}  "
        f"vwap={weights['vwap']:.0%}"
    )

    signals        = []
    combined_scores: dict = {}

    for ticker in tickers:

        # ── Method 1: News sentiment ──────────────────────────────────────
        sentiment_score = 0.0
        news_rationale  = "News sentiment disabled."
        if use_news:
            relevant = filter_relevant_articles(ticker, articles)
            sentiment_score, news_rationale = analyse_sentiment(ticker, relevant)

        # ── Method 2: Technical analysis ─────────────────────────────────
        tech_result: TechnicalResult = EMPTY_RESULT
        if use_tech:
            tech_result = compute_technical_score(ticker)
        technical_score = tech_result.score
        vol_ratio       = tech_result.vol_ratio
        atr_pct         = tech_result.atr_pct
        bb_width_pct    = tech_result.bb_width_pct

        # ── Method 3: Smart money ─────────────────────────────────────────
        insider_sc      = 0.0
        insider_summary = ""
        if use_insider:
            insider_sc      = _insider_score(ticker, insider_trades)
            insider_summary = build_insider_summary(ticker, insider_trades)

        # ── Method 4: Put/call ratio (contrarian) ─────────────────────────
        pc_score = 0.0
        if use_put_call:
            pc_score = _put_call_score_for(ticker, put_call_context)

        # ── Method 5: Max pain gravity (expiry-weighted) ──────────────────
        mp_score = 0.0
        if use_max_pain:
            mp_score = _max_pain_score_for(ticker, gex_context)

        # ── GEX lookup (used by Method 6, movement factor, and signal fields) ──
        gex_sig, gamma_flip, max_pain_bias, expected_move_pct, oi_skew = _gex_for(ticker, gex_context)

        # ── Method 6: OI-weighted directional skew ────────────────────────
        # Positive = call OI concentrated above spot (bullish positioning)
        # Negative = put OI concentrated below spot (bearish positioning)
        oi_skew_score = round(oi_skew, 3) if use_oi_skew else 0.0

        # ── Method 7: VWAP distance (mean-reversion) ─────────────────────
        # Score > 0: price below VWAP → institutional buyers expected (bullish)
        # Score < 0: price above VWAP → institutional sellers expected (bearish)
        vwap_score    = 0.0
        vwap_dist_pct = 0.0
        if use_vwap:
            vwap_score, vwap_dist_pct = compute_vwap_score(ticker)

        # ── Weighted combination ──────────────────────────────────────────
        combined = (
            weights["news"]      * sentiment_score +
            weights["tech"]      * technical_score +
            weights["insider"]   * insider_sc +
            weights["put_call"]  * pc_score +
            weights["max_pain"]  * mp_score +
            weights["oi_skew"]   * oi_skew_score +
            weights["vwap"]      * vwap_score
        )

        # ── Interaction adjustments ───────────────────────────────────────
        # Small additive corrections for setups where two methods together
        # are more informative than their linear contributions suggest.
        if active_count >= 2:
            combined += _interaction_adjustment(
                combined, sentiment_score, technical_score,
                insider_sc, pc_score, vol_ratio,
            )

        # ── Direction ─────────────────────────────────────────────────────
        direction: Direction
        if combined >= 0.15:
            direction = "BULLISH"
        elif combined <= -0.15:
            direction = "BEARISH"
        else:
            direction = "NEUTRAL"

        # ── Coherence factor (continuous, replaces binary 1.25×/0.60×) ───
        method_scores = [
            (use_news,      sentiment_score),
            (use_tech,      technical_score),
            (use_insider,   insider_sc),
            (use_put_call,  pc_score),
            (use_max_pain,  mp_score),
            (use_oi_skew,   oi_skew_score),
            (use_vwap,      vwap_score),
        ]
        coherence_ratio, coherence_factor = _coherence_factor(combined, method_scores)

        # sources_agreeing kept for backward-compat with Claude's prompt context
        sources_agreeing = sum(
            1 for enabled, score in method_scores
            if enabled and abs(score) >= _AGREE_THRESHOLD
            and ((combined > 0 and score > 0) or (combined < 0 and score < 0))
        )

        # ── Movement potential (ATR + BB width + GEX) ────────────────────
        movement_factor = _movement_factor(atr_pct, bb_width_pct) * _gex_movement_modifier(gex_sig)
        movement_factor = round(max(0.70, min(1.30, movement_factor)), 3)

        # ── Cross-method volume confirmation ──────────────────────────────
        volume_factor = _volume_factor(vol_ratio, abs(combined), coherence_ratio)

        # ── Final confidence ──────────────────────────────────────────────
        raw_confidence = min(1.0, abs(combined) / 0.5)
        confidence = round(
            min(1.0, raw_confidence * coherence_factor * movement_factor * volume_factor),
            2,
        )

        # ── Rationale ─────────────────────────────────────────────────────
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
            max_pain_score=round(mp_score, 3),
            oi_skew_score=round(oi_skew_score, 3),
            vwap_score=round(vwap_score, 3),
            vwap_distance_pct=round(vwap_dist_pct, 2),
            rationale=rationale,
            insider_summary=insider_summary,
            sources_agreeing=sources_agreeing,
            gex_signal=gex_sig,
            gamma_flip=gamma_flip,
            max_pain_bias=max_pain_bias,
            expected_move_pct=expected_move_pct,
        ))

        gex_str = f"  gex={gex_sig}({gamma_flip})" if gex_sig else ""
        mp_str   = f"  mp={mp_score:+.2f}" if mp_score != 0.0 else ""
        skew_str = f"  skew={oi_skew_score:+.2f}" if oi_skew_score != 0.0 else ""
        vwap_str = f"  vwap={vwap_score:+.2f}({vwap_dist_pct:+.1f}%)" if vwap_score != 0.0 else ""
        logger.info(
            f"{ticker}: {direction} (conf={confidence:.0%}, {sources_agreeing}/{active_count} agree) | "
            f"news={sentiment_score:+.2f}  tech={technical_score:+.2f}  "
            f"insider={insider_sc:+.2f}  pc={pc_score:+.2f}{mp_str}{skew_str}{vwap_str}  combined={combined:+.2f} | "
            f"coherence={coherence_ratio:.2f}({coherence_factor:.2f}x)  "
            f"movement={movement_factor:.2f}x  volume={volume_factor:.2f}x  "
            f"atr={atr_pct:.3f}  vol_ratio={vol_ratio:.2f}x{gex_str}"
        )

    # ── Second pass: sector alignment ────────────────────────────────────────
    signals_by_ticker = {s.ticker: s for s in signals}
    final_signals = []
    for sig in signals:
        raw_combined  = combined_scores.get(sig.ticker, 0.0)
        sector_factor = _sector_alignment_factor(sig.ticker, raw_combined, signals_by_ticker)
        if sector_factor != 1.0:
            adjusted = round(min(1.0, sig.confidence * sector_factor), 2)
            sig = sig.model_copy(update={"confidence": adjusted})
        final_signals.append(sig)

    return final_signals
