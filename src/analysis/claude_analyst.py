"""Final synthesis: Claude generates a high-level market report and recommendations."""

import anthropic
import json
from loguru import logger
from typing import List, Optional, TYPE_CHECKING
from datetime import datetime, timezone
from src.utils import now_et, fmt_et
from config import settings
from src.models import TickerSignal, Recommendation, InsiderTrade, MacroContext
from src.data.insider_trades import build_insider_summary


_client = None


def _get_client() -> anthropic.Anthropic:
    global _client
    if _client is None:
        _client = anthropic.Anthropic(api_key=settings.anthropic_api_key)
    return _client


def generate_recommendations(
    signals: List[TickerSignal],
    insider_trades: Optional[List["InsiderTrade"]] = None,
    macro_context: Optional["MacroContext"] = None,
) -> List[Recommendation]:
    """
    Feed all ticker signals to Claude and get final actionable recommendations.
    Includes sections for whichever analysis methods are enabled.
    """
    if not signals:
        return []

    use_news    = settings.enable_news_sentiment
    use_tech    = settings.enable_technical_analysis and settings.enable_market_data
    use_insider = (
        (settings.enable_insider_trades or
         settings.enable_options_flow or
         settings.enable_sec_filings)
        and insider_trades is not None
    )

    # Cap the signal list sent to Claude:
    # - Always include tickers with any meaningful signal (confidence > 10% or insider activity)
    # - Fill up to 40 tickers sorted by confidence descending
    # This prevents JSON truncation when the universe is large.
    _MAX_SIGNALS = 40
    meaningful      = [s for s in signals if s.confidence > 0.10 or s.insider_summary]
    noise           = [s for s in signals if s not in meaningful]
    ranked          = sorted(meaningful, key=lambda s: s.confidence, reverse=True)
    if len(ranked) < _MAX_SIGNALS:
        ranked     += sorted(noise, key=lambda s: s.confidence, reverse=True)[:_MAX_SIGNALS - len(ranked)]
    signals_for_claude = ranked[:_MAX_SIGNALS]
    skipped            = len(signals) - len(signals_for_claude)
    if skipped:
        logger.info(f"[claude] Sending {len(signals_for_claude)}/{len(signals)} signals (skipped {skipped} near-zero tickers)")

    # Build the signals block
    signal_lines = []
    for s in signals_for_claude:
        parts = [f"- {s.ticker}: direction={s.direction}, combined_confidence={s.confidence:.0%}, sources_agreeing={s.sources_agreeing}"]
        if use_news:
            parts.append(f"  News sentiment={s.sentiment_score:+.2f} | {s.rationale}")
        if use_tech:
            parts.append(f"  Technical score={s.technical_score:+.2f}")
        if use_insider and s.insider_summary:
            parts.append(f"  Insider activity: {s.insider_summary}")
        signal_lines.append("\n".join(parts))

    signals_text = "\n\n".join(signal_lines)
    # Append a note so Claude knows the full universe size
    if skipped:
        signals_text += f"\n\n[{skipped} additional tickers omitted — all had near-zero signals]"

    # Build the active-methods description for the prompt
    active_methods = []
    if use_news:
        active_methods.append("news/sentiment (LLM-scored headlines, RSS, social)")
    if use_tech:
        active_methods.append("technical analysis (RSI, MACD, SMA50, SMA200, Bollinger Bands)")
    if use_insider:
        active_methods.append(
            "smart money signals (politician disclosures, SEC Form 4, "
            "13D/13G activist stakes, Form 144 planned sales, "
            "13F superinvestor positions, unusual options sweeps)"
        )
    methods_desc = ", ".join(active_methods) if active_methods else "combined signals"

    # Insider-specific instructions
    insider_instructions = ""
    if use_insider:
        insider_instructions = """
5. Smart money signal weighting:
   - Congressional buys from multiple politicians: STRONG signal — politicians have proven information advantages. Multiple politicians buying the same ticker = high-conviction.
   - Large-amount buys ($500k+) from known market-beating politicians (e.g. Pelosi) should be weighted heavily. Net insider selling weakens a BUY thesis.
   - Activist 13D stakes: an activist crossing 5% ownership is a VERY STRONG bullish catalyst — they are forcing change (buyback, sale of company, board turnover).
   - Passive 13G stakes: large institutional accumulation is mildly bullish with no immediate catalyst implied.
   - Form 144 planned sales: an insider pre-announcing a sale reduces BUY confidence — they expect the stock to be lower or are reducing risk.
   - 13F superinvestor new positions (Buffett, Ackman, etc.): high-conviction long signals with a ~45-day reporting lag. Still meaningful as a structural thesis.
   - 13F superinvestor exits: notable de-risking — weigh against current BUY thesis.
   - Unusual options sweeps (CALL/PUT): institutional directional bets. Multiple sweeps on the same ticker before an event are a strong signal. OTM calls = bullish; OTM puts = bearish."""

    tech_instructions = ""
    if use_tech:
        tech_instructions = """
   - Factor in the technical score: a score above +0.3 adds conviction to a BUY; below -0.3 adds conviction to a SELL.
   - Require alignment between news catalyst and technical picture for highest-confidence calls."""

    # Build macro context block for the prompt
    macro_block = ""
    macro_instructions = ""
    if macro_context and macro_context.summary:
        regime_color = {
            "RECESSION":  "DANGER — recession risk is elevated",
            "LATE_CYCLE": "CAUTION — late-cycle dynamics, reduce risk exposure",
            "SLOWDOWN":   "CAUTION — growth is decelerating",
            "EXPANSION":  "CONSTRUCTIVE — macro tailwind supports risk assets",
        }.get(macro_context.regime, "UNCERTAIN")

        macro_block = f"""
<macro_context>
FRED Macro Regime: {macro_context.regime} ({regime_color})
{macro_context.summary}

Key indicators:
- Yield curve (10Y-2Y): {f"{macro_context.yield_spread_10y2y:+.2f}%" if macro_context.yield_spread_10y2y is not None else "N/A"} — {macro_context.yield_curve_signal}
- Fed Funds Rate: {f"{macro_context.fed_funds_rate:.2f}%" if macro_context.fed_funds_rate is not None else "N/A"}
- CPI YoY: {f"{macro_context.cpi_yoy:+.1f}%" if macro_context.cpi_yoy is not None else "N/A"} — {macro_context.inflation_signal}
- Unemployment: {f"{macro_context.unemployment_rate:.1f}%" if macro_context.unemployment_rate is not None else "N/A"} ({macro_context.unemployment_trend})
- HY Credit Spread: {f"{macro_context.hy_spread:.2f}%" if macro_context.hy_spread is not None else "N/A"} — {macro_context.credit_signal}
- IG Credit Spread: {f"{macro_context.ig_spread:.2f}%" if macro_context.ig_spread is not None else "N/A"}
- M2 Growth YoY: {f"{macro_context.m2_growth_yoy:+.1f}%" if macro_context.m2_growth_yoy is not None else "N/A"}
</macro_context>
"""
        macro_instructions = f"""
6. Macro regime overlay (from FRED data — apply to ALL recommendations):
   - Regime is {macro_context.regime}. Calibrate conviction accordingly:
     * RECESSION: strongly prefer HOLD/WATCH for longs; shorts become higher-conviction. Avoid POSITION-horizon BUYs.
     * LATE_CYCLE: be selective — only BUY names with recession-resistant fundamentals. Favor SWING over POSITION horizons.
     * SLOWDOWN: tilt bearish on cyclicals, constructive on defensives (staples, utilities, gold). Shorten time horizons.
     * EXPANSION: macro tailwind — conviction on longs is higher. Still require signal convergence.
   - Inverted yield curve (current: {macro_context.yield_curve_signal}): historically predicts recession 6-18 months out.
     Do NOT extend time horizons on speculative longs if curve is inverted.
   - Credit spreads ({macro_context.credit_signal}): widening HY spreads signal institutional risk-off.
     When credit is STRESSED or ELEVATED, be more conservative on all BUY calls.
   - Inflation ({macro_context.inflation_signal}): high inflation → Fed stays restrictive → pressure on rate-sensitive sectors (tech, real estate).
   - Unemployment trend ({macro_context.unemployment_trend}): rising unemployment is a leading recession indicator.
     Downgrade POSITION-horizon BUY calls to SWING or HOLD when unemployment is rising."""

    commodity_tickers = ", ".join(settings.commodities_list) or "GLD, SLV, IAU, GDX, PPLT, PALL, CPER"

    prompt = f"""You are an elite portfolio manager with a verified 30-year track record of market-beating returns. You combine the analytical precision of a quant, the pattern recognition of a seasoned discretionary trader, and the macro intuition of a global macro fund manager. You have studied every major market cycle since 1990 and have an exceptional ability to identify when multiple independent evidence layers converge on the same directional call — these are the moments of highest expected value.

Your defining edge: you are ruthlessly disciplined about false positives. You understand that a wrong BUY or SELL costs capital that cannot be recovered. You output HOLD or WATCH whenever the evidence is mixed, incomplete, or driven by a single source. When you do issue a BUY or SELL, it is because the convergence of evidence makes the directional call highly reliable — and you explain precisely why.

Signal sources available today: {methods_desc}

Today's date: {fmt_et(now_et())}
{macro_block}
INPUT — multi-method ticker signals:
<signals>
{signals_text}
</signals>

YOUR TASK:
1. Identify the BEST opportunities across the full list — both longs (BUY) and shorts (SELL).
   - Apply the same discipline that made your career: only act when multiple independent layers of evidence converge. A genuine BUY/SELL signal is rare and valuable — treat it as such.
   - If no ticker clears the bar today, output HOLD/WATCH for all. Markets offer high-probability setups infrequently. Patience is the highest-conviction trade.
   - Strongly prefer tickers where sources_agreeing ≥ 2: when news sentiment, technical momentum, AND smart money all point the same direction, the probability of being right is substantially higher than any single source alone.
   - A single strong news print or a single options sweep is NEVER sufficient for BUY/SELL. It may be positioning, it may be noise. Require corroboration.
   - Do NOT ignore trending/discovered tickers just because they are not mega-caps. Small caps with strong smart money conviction and technical breakouts are often your best risk/reward setups.

2. Distinguish time horizons:
   - "SWING" (2-10 days): catalyst-driven move not yet priced in.
   - "SHORT-TERM" (1-4 weeks): sector rotation, earnings run-up/fade, macro shift.
   - "POSITION" (1-3 months): structural change — regulatory, competitive, macro theme.
{tech_instructions}
3. Conviction rules:
   - confidence ≥ 0.78 AND sources_agreeing ≥ 2 → eligible for BUY / SELL.
   - confidence ≥ 0.78 but sources_agreeing = 1 → HOLD maximum (single-source signals are noise).
   - confidence 0.55-0.77 → HOLD (monitor closely).
   - confidence < 0.55 → WATCH only.
   - Do NOT inflate confidence. A 90%+ call requires multiple converging signals with clear price catalyst.
   - When in doubt, HOLD is the correct output — a wrong BUY/SELL destroys capital.

4. Short-selling discipline:
   - SELL means initiating a short position (or buying an inverse ETF).
   - Only short when: (a) clearly negative catalyst, (b) no counter-narrative, (c) broad market not in capitulation.
{insider_instructions}{macro_instructions}
Commodity tickers always present in the list: {commodity_tickers}
— Label these as type "COMMODITY". Apply your macro expertise:
  - Precious metals (GLD, SLV, IAU, GDX, PPLT, PALL): driven by real rates, USD strength/weakness, geopolitical risk, and central bank policy expectations. A falling real rate environment or rising macro uncertainty is structurally bullish for gold and silver.
  - Industrial metals (CPER): driven by global growth expectations, China PMI, and supply disruptions.
  - Give each commodity a standalone BUY/HOLD/SELL view with a rationale grounded in the current macro environment as reflected in the news signals. Do not default to HOLD for commodities — they have directional macro drivers that are often identifiable even when equity signals are mixed.

Output a JSON array where each element has:
- "ticker": string
- "type": "STOCK" | "ETF" | "COMMODITY"
- "direction": "BULLISH" | "BEARISH" | "NEUTRAL"
- "action": "BUY" | "SELL" | "HOLD" | "WATCH"
- "time_horizon": "SWING" | "SHORT-TERM" | "POSITION" | "N/A"
- "confidence": float 0.0-1.0
- "rationale": 2-3 sentences — cite the specific catalysts from ALL active signal layers, explain the price mechanism, state expected time horizon and key risk.

Return ALL tickers from the input. No markdown, JSON only."""

    try:
        client = _get_client()
        logger.info(f"[claude] Using model: {settings.analyst_model}")
        # Haiku max output = 8096; Sonnet/Opus support higher limits
        _max_tokens = 8096 if "haiku" in settings.analyst_model else 16000
        message = client.messages.create(
            model=settings.analyst_model,
            max_tokens=_max_tokens,
            messages=[{"role": "user", "content": prompt}],
        )
        raw = message.content[0].text.strip()
        # Strip markdown fences if the model wrapped the response
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
            raw = raw.strip()
        logger.debug(f"Raw Claude response ({len(raw)} chars): {raw[:200]}")
        data = json.loads(raw)
        now = now_et()
        recommendations = [
            Recommendation(
                ticker=r["ticker"],
                type=r.get("type", "STOCK"),
                direction=r["direction"],
                action=r["action"],
                confidence=float(r["confidence"]),
                time_horizon=r.get("time_horizon", "N/A"),
                rationale=r["rationale"],
                generated_at=now,
            )
            for r in data
        ]
        logger.info(f"Generated {len(recommendations)} recommendations")
        return recommendations
    except Exception as e:
        logger.error(f"Claude recommendations failed: {e}")
        # Fallback: convert signals directly to recommendations
        return _fallback_recommendations(signals)


def _fallback_recommendations(signals: List[TickerSignal]) -> List[Recommendation]:
    now = datetime.now(timezone.utc)
    recs = []
    for s in signals:
        if s.direction == "BULLISH" and s.confidence >= 0.6:
            action = "BUY"
        elif s.direction == "BEARISH" and s.confidence >= 0.6:
            action = "SELL"
        elif s.confidence < 0.4:
            action = "WATCH"
        else:
            action = "HOLD"
        recs.append(Recommendation(
            ticker=s.ticker,
            direction=s.direction,
            action=action,
            confidence=s.confidence,
            rationale=s.rationale,
            generated_at=now,
        ))
    return recs
