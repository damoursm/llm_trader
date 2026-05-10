"""Catalyst Timing — event-driven guards and amplifiers applied around known binary events.

Three mechanisms
────────────────
1. Earnings Blackout
   Any ticker with an earnings report within 2 calendar days is removed from the
   actionable BUY/SELL set.  IV crush, gap risk, and binary outcomes make directional
   trades around earnings highly unreliable.  The blackout covers both sides: we
   don't enter new longs or shorts into a catalyst that erases the edge.

2. OpEx Max-Pain Amplifier
   During OpEx week (especially Triple Witching: Mar/Jun/Sep/Dec), market-makers
   with net-short options books exert demonstrably stronger gravitational pull toward
   the max-pain strike.  The `max_pain` signal weight in the aggregator is boosted from
   its 0.12 baseline:
     Normal OpEx week     → 0.20  (+67%)
     Triple Witching week → 0.28  (+133%)
   This is applied as a per-run override in build_signals() before normalisation,
   so it combines cleanly with the market-mode weight profile.

3. 8-K + Insider Buy → WATCH Elevation
   The combination of a freshly filed SEC 8-K (material catalyst) and an insider
   purchase by a corporate officer or politician is among the highest-predictive
   pre-signal setups in the system.  When BOTH are present for the same ticker:
     - If the ticker is already HOLD in the top-10 recommendations → upgraded to WATCH.
     - If not yet in the top-10 → injected as a new WATCH recommendation.
   Volume confirmation (via TickerSignal technical/VWAP score) is noted when available
   but is not required to trigger the elevation.
"""

from datetime import datetime, timezone
from typing import Dict, List, Optional
from loguru import logger

from src.models import CatalystSetup, CatalystTimingContext, Recommendation


# ── Weights ───────────────────────────────────────────────────────────────────

OPEX_WEEK_MAX_PAIN_WEIGHT         = 0.20   # normal OpEx week boost
OPEX_TRIPLE_WITCHING_MAX_PAIN_WEIGHT = 0.28  # Triple Witching boost


# ── 8-K ticker extraction ─────────────────────────────────────────────────────

def _ticker_from_8k_title(title: str) -> Optional[str]:
    """Extract the ticker from an 8-K article title (format: '{TICKER} SEC 8-K — ...')."""
    if " SEC 8-K" in title:
        candidate = title.split(" SEC 8-K")[0].strip()
        # Sanity: only uppercase alpha + common suffixes, ≤8 chars
        if candidate and candidate.isupper() and len(candidate) <= 8:
            return candidate
    return None


# ── Main computation ──────────────────────────────────────────────────────────

def compute_catalyst_context(
    earnings_context=None,
    opex_context=None,
    articles=None,
    insider_trades=None,
    signals_by_ticker: Optional[dict] = None,
    sectors_list: Optional[List[str]] = None,
    commodities_list: Optional[List[str]] = None,
) -> CatalystTimingContext:
    """
    Compute all three catalyst timing signals from already-fetched contexts.

    Parameters
    ----------
    earnings_context   : EarningsContext   — upstream earnings calendar
    opex_context       : OpExContext       — options expiration calendar
    articles           : List[NewsArticle] — all articles including 8-K filings
    insider_trades     : List[InsiderTrade]
    signals_by_ticker  : Dict[str, TickerSignal] — aggregated signals (for vol check)
    sectors_list       : sector ETF tickers (for Recommendation.type assignment)
    commodities_list   : commodity tickers (for Recommendation.type assignment)
    """

    # ── 1. Earnings Blackout ──────────────────────────────────────────────────
    blackout_tickers: List[str] = []
    blackout_details: Dict[str, int] = {}
    if earnings_context is not None:
        for event in earnings_context.upcoming:
            if 0 <= event.days_until <= 2:
                blackout_tickers.append(event.ticker.upper())
                blackout_details[event.ticker.upper()] = event.days_until

    if blackout_tickers:
        logger.info(
            f"[catalyst] Earnings blackout: {len(blackout_tickers)} ticker(s) blocked "
            f"(within 2 days of earnings) — {blackout_tickers}"
        )

    # ── 2. OpEx Max-Pain Amplifier ────────────────────────────────────────────
    opex_boost_active        = False
    opex_is_triple_witching  = False
    opex_max_pain_weight     = 0.12   # default _BASE_WEIGHTS["max_pain"]
    opex_signal              = "NEUTRAL"

    if opex_context is not None and opex_context.in_opex_week:
        opex_boost_active       = True
        opex_is_triple_witching = opex_context.is_triple_witching
        opex_signal             = opex_context.signal
        opex_max_pain_weight    = (
            OPEX_TRIPLE_WITCHING_MAX_PAIN_WEIGHT
            if opex_is_triple_witching
            else OPEX_WEEK_MAX_PAIN_WEIGHT
        )
        logger.info(
            f"[catalyst] OpEx boost active ({opex_signal}): "
            f"max_pain weight 0.12 → {opex_max_pain_weight:.2f}"
            + (" [TRIPLE WITCHING]" if opex_is_triple_witching else "")
        )

    # ── 3. 8-K + Insider Buy Catalyst Setups ─────────────────────────────────
    # Collect tickers with recent 8-K filings
    eight_k_tickers: set[str] = set()
    eight_k_descriptions: Dict[str, str] = {}
    for art in (articles or []):
        if art.source == "SEC 8-K Filing":
            ticker = _ticker_from_8k_title(art.title)
            if ticker:
                eight_k_tickers.add(ticker)
                # Keep the first title as description if multiple filings
                eight_k_descriptions.setdefault(ticker, art.title)

    # Collect tickers with insider buys (corporate insider or politician only)
    insider_buy_tickers: set[str] = set()
    insider_buy_descriptions: Dict[str, str] = {}
    for trade in (insider_trades or []):
        if (trade.is_bullish
                and trade.trader_type in ("corporate_insider", "politician")):
            t = trade.ticker.upper()
            insider_buy_tickers.add(t)
            insider_buy_descriptions.setdefault(
                t, f"{trade.trader_name} ({trade.role}) — {trade.amount_range}"
            )

    # Intersection: tickers with BOTH
    overlap = eight_k_tickers & insider_buy_tickers
    catalyst_setups: List[CatalystSetup] = []
    watch_elevation_tickers: List[str] = []

    for ticker in sorted(overlap):
        sig = (signals_by_ticker or {}).get(ticker)
        has_vol = (
            sig is not None
            and (abs(sig.technical_score) > 0.10 or abs(sig.vwap_score) > 0.10)
        )

        reason_parts = ["recent 8-K filing"]
        if has_vol:
            reason_parts.append("volume-confirmed technical signal")
        reason_parts.append(f"insider buy ({insider_buy_descriptions.get(ticker, 'disclosed')})")

        catalyst_setups.append(CatalystSetup(
            ticker=ticker,
            has_8k=True,
            has_insider_buy=True,
            has_vol_spike=has_vol,
            catalyst_reason=" + ".join(reason_parts),
        ))
        watch_elevation_tickers.append(ticker)
        logger.info(
            f"[catalyst] WATCH candidate: {ticker} — "
            + ", ".join(reason_parts)
        )

    # ── Summary ───────────────────────────────────────────────────────────────
    parts = []
    if blackout_tickers:
        parts.append(f"Earnings blackout: {blackout_tickers}")
    if opex_boost_active:
        parts.append(f"OpEx boost: max_pain → {opex_max_pain_weight:.2f}")
    if catalyst_setups:
        parts.append(f"WATCH candidates: {watch_elevation_tickers}")
    summary = " | ".join(parts) if parts else "No active catalyst timing signals."

    return CatalystTimingContext(
        earnings_blackout_tickers=blackout_tickers,
        earnings_blackout_details=blackout_details,
        opex_max_pain_weight=opex_max_pain_weight,
        opex_boost_active=opex_boost_active,
        opex_is_triple_witching=opex_is_triple_witching,
        opex_signal=opex_signal,
        catalyst_setups=catalyst_setups,
        watch_elevation_tickers=watch_elevation_tickers,
        summary=summary,
    )


# ── WATCH injection helper ────────────────────────────────────────────────────

def apply_watch_elevation(
    recommendations: List[Recommendation],
    catalyst_context: CatalystTimingContext,
    signals_by_ticker: Optional[dict] = None,
    sectors_list: Optional[List[str]] = None,
    commodities_list: Optional[List[str]] = None,
) -> List[Recommendation]:
    """
    For tickers in catalyst_context.watch_elevation_tickers:
      • HOLD in top-10 → upgrade to WATCH (in-place).
      • Not yet in top-10 → inject a new WATCH Recommendation.

    Returns the (possibly extended) recommendations list.
    """
    if not catalyst_context.watch_elevation_tickers:
        return recommendations

    recs_by_ticker = {r.ticker: r for r in recommendations}
    sectors_set    = set(sectors_list or [])
    commodities_set = set(commodities_list or [])
    injected = []

    for ticker in catalyst_context.watch_elevation_tickers:
        setup = next((s for s in catalyst_context.catalyst_setups if s.ticker == ticker), None)
        rationale = (
            f"Catalyst alert: {setup.catalyst_reason}."
            if setup else "Catalyst alert: 8-K filing + insider buy detected."
        )

        existing = recs_by_ticker.get(ticker)
        if existing is not None:
            if existing.action == "HOLD":
                # In-place upgrade: replace the action on this recommendation
                recs_by_ticker[ticker] = existing.model_copy(
                    update={"action": "WATCH", "rationale": rationale}
                )
                logger.info(f"[catalyst] {ticker}: upgraded HOLD → WATCH (catalyst setup detected)")
        else:
            # Determine type
            t_upper = ticker.upper()
            rec_type = (
                "COMMODITY" if t_upper in commodities_set
                else "ETF"   if t_upper in sectors_set
                else "STOCK"
            )
            sig = (signals_by_ticker or {}).get(ticker)
            direction = sig.direction if sig else "NEUTRAL"
            confidence = sig.confidence if sig else 0.10
            injected.append(Recommendation(
                ticker=ticker,
                type=rec_type,
                direction=direction,
                confidence=confidence,
                action="WATCH",
                time_horizon="N/A",
                rationale=rationale,
                generated_at=datetime.now(timezone.utc),
            ))
            logger.info(f"[catalyst] {ticker}: injected new WATCH recommendation (catalyst setup detected)")

    # Reconstruct the recommendations list from the potentially updated dict
    updated = [recs_by_ticker.get(r.ticker, r) for r in recommendations]
    return updated + injected
