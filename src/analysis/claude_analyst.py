"""Final synthesis: Claude generates a high-level market report and recommendations."""

import anthropic
import json
from loguru import logger
from typing import List, Optional, TYPE_CHECKING
from datetime import datetime, timezone
from src.utils import now_et, fmt_et
from config import settings
from src.models import TickerSignal, Recommendation, InsiderTrade, MacroContext, COTContext, IPOContext, VIXContext, PutCallContext, EarningsContext
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
    cot_context: Optional["COTContext"] = None,
    ipo_context: Optional["IPOContext"] = None,
    vix_context: Optional["VIXContext"] = None,
    put_call_context: Optional["PutCallContext"] = None,
    earnings_context: Optional["EarningsContext"] = None,
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

    # Build COT context block for the prompt
    cot_block        = ""
    cot_instructions = ""
    if cot_context and cot_context.signals:
        rows = []
        for s in cot_context.signals:
            tickers_str = "/".join(s.tickers)
            direction_icon = "▲" if s.direction == "BULLISH" else ("▼" if s.direction == "BEARISH" else "→")
            rows.append(
                f"  {s.contract:<14} ({tickers_str:<14}) "
                f"net={s.net_speculator_pct:+6.1f}%  WoW={s.net_change_wow:+5.1f}%  "
                f"pct={s.percentile_52w:4.0f}th  {s.signal:<14}  {direction_icon} {s.direction}"
            )
        table = "\n".join(rows)

        cot_block = f"""
<cot_context>
CFTC Commitment of Traders — as of {cot_context.report_date} (cached weekly)
{cot_context.summary}

Contract positioning (net speculator % of OI, 52-week percentile, contrarian direction applied):
{table}

Interpretation guide:
  EXTREME_LONG  (≥80th pct) → contrarian BEARISH — specs crowded long, reversal risk high
  BULLISH_TREND (60-79th)   → BULLISH momentum — specs still adding longs
  NEUTRAL       (40-59th)   → no clear COT signal
  BEARISH_TREND (20-39th)   → BEARISH momentum — specs reducing exposure
  EXTREME_SHORT (≤20th pct) → contrarian BULLISH — specs max short, coiled for squeeze
</cot_context>
"""

        cot_instructions = """
7. COT positioning overlay (apply to commodity and index ETFs):
   - COT is a MEDIUM-TERM signal (weeks to months), best used to confirm or fade news-driven moves.
   - EXTREME_LONG: when specs are at 52-week max longs, upside is limited and reversal risk is high.
     Treat as a ceiling — avoid new BUY calls; elevate conviction on SELL if news also negative.
   - EXTREME_SHORT: when specs are at 52-week max shorts, downside is likely exhausted.
     Treat as a floor — avoid new SELL calls; elevate conviction on BUY if news also positive.
   - BULLISH_TREND: specs are accumulating longs → confirms BUY, weakens SELL thesis.
   - BEARISH_TREND: specs reducing exposure → weakens BUY, confirms SELL thesis.
   - COT alone is never sufficient for BUY/SELL — it must converge with at least one other signal.
   - For commodities (GLD, SLV, CPER, etc.): COT is especially high-signal given futures market depth.
   - For index ETFs (SPY, QQQ): extreme spec positioning is a useful sentiment extreme indicator."""

    # Build IPO pipeline context block for the prompt
    ipo_block        = ""
    ipo_instructions = ""
    if ipo_context and ipo_context.total_new > 0:
        # Sector breakdown table
        sector_rows = "\n".join(
            f"  {sector:<25} {count:>3} new registration(s)"
            for sector, count in ipo_context.sector_counts.items()
        )
        # Recent initial filings (up to 10)
        recent_filings = "\n".join(
            f"  {f.filing_date}  {f.form_type:<6}  {f.sector:<25}  {f.company}"
            for f in ipo_context.filings[:10]
        )

        ipo_block = f"""
<ipo_pipeline>
SEC IPO Pipeline — S-1/S-11 filings (last {ipo_context.lookback_days} days, as of {ipo_context.report_date})
{ipo_context.summary}

Sector breakdown (initial registrations only):
{sector_rows}

  Total amendments (S-1/A, S-11/A): {ipo_context.total_amendments}
  — amendments indicate companies advancing toward a listing date

Recent initial filings:
{recent_filings}
</ipo_pipeline>
"""

        hot = ", ".join(ipo_context.hot_sectors) if ipo_context.hot_sectors else "none identified"
        ipo_instructions = f"""
8. IPO pipeline overlay (sector-level institutional demand signal):
   - Hot IPO sectors (most S-1 filings): {hot}.
     Institutional underwriters only open the IPO window when they have conviction in demand.
     A high-activity sector → their real-money clients are willing buyers → bullish for the sector ETF.
   - Use this to CONFIRM, not originate, a sector-level BUY: if news is already bullish on XLK
     and Technology dominates the S-1 pipeline, that convergence raises conviction.
   - Cold IPO market (few filings overall): institutional caution → temper aggressive BUY calls
     on growth sectors even if news is positive; market may lack the risk appetite to follow through.
   - Amendment wave ({ipo_context.total_amendments} amendments): a large amendment count signals
     multiple companies are actively preparing to price — implies underwriters see a viable window.
   - Do NOT use IPO data as a standalone BUY/SELL trigger. It is a secondary confirming layer only."""

    # Build VIX context block for the prompt
    vix_block        = ""
    vix_instructions = ""
    if vix_context and vix_context.vix:
        v = vix_context
        ts_color = {"BACKWARDATION": "⚠ BACKWARDATION", "CONTANGO": "CONTANGO", "FLAT": "FLAT"}.get(v.term_structure, v.term_structure)

        def _fmt(val):
            return f"{val:.1f}" if val is not None else "N/A"

        vix_block = f"""
<vix_context>
VIX Volatility Regime: {v.vix:.1f} — {v.vix_signal}  (contrarian direction: {v.vix_direction})
{v.summary}

Term structure (vol curve shape):
  VIX9D={_fmt(v.vix9d)}  VIX={_fmt(v.vix)}  VIX3M={_fmt(v.vix3m)}  VIX6M={_fmt(v.vix6m)}
  Slope (VIX3M − VIX) = {f"{v.slope_1m_3m:+.1f}pt" if v.slope_1m_3m is not None else "N/A"}  →  {ts_color}
  VXN (Nasdaq vol) = {_fmt(v.vxn)}   |   VVIX (vol-of-vol) = {_fmt(v.vvix)}

VIX signal guide:
  > 45 PANIC          → very strong contrarian BUY — capitulation; forced selling near exhaustion
  35–45 EXTREME_FEAR  → strong contrarian BUY bias; fade SELL signals on quality names
  25–35 HIGH          → elevated risk; start watching for reversal; prefer quality longs
  20–25 ELEVATED      → selective; macro headwind present; no override
  15–20 NORMAL        → standard regime; no VIX override
  12–15 LOW           → mild complacency; reduce aggressive BUY exposure
  < 12  COMPLACENCY   → crowd is not hedging; contrarian BEARISH risk

Term structure guide:
  BACKWARDATION (VIX > VIX3M): near-term panic spike; often marks a short-term bottom
  FLAT: transitional; watch for direction of next move
  CONTANGO (VIX3M > VIX): normal; market expects future uncertainty > current; calm regime
</vix_context>
"""
        vix_instructions = f"""
11. VIX & volatility regime overlay (apply to ALL recommendations):
    Current VIX={v.vix:.1f} ({v.vix_signal}), term structure={v.term_structure}:
    - PANIC / EXTREME_FEAR (VIX > 35): This is a capitulation zone. The crowd is panic-selling.
      * Strongly fade SELL signals on diversified/quality names.
      * Upgrade BUY conviction by +0.05-0.10 when news and technical signals also positive.
      * Do NOT open new SELL positions at VIX extremes — mean reversion risk is very high.
      * Exception: if a specific company has a company-specific negative catalyst (fraud, bankruptcy), VIX does not override that SELL.
    - HIGH (VIX 25–35): Elevated fear. Market is pricing significant downside.
      * Be selective on new BUY calls. Require stronger signal convergence (sources_agreeing ≥ 3).
      * Shorts are riskier — crowded shorts can get squeezed on any relief rally.
    - ELEVATED/NORMAL (VIX 15–25): Standard operating range. No VIX override.
    - LOW / COMPLACENCY (VIX < 15): The market is not pricing risk. Complacency can persist,
      but any shock hits harder. Reduce confidence on aggressive POSITION-length BUY calls.

    Term structure:
    - BACKWARDATION: near-term vol > long-term vol → classic panic/capitulation shape.
      When VIX is also in EXTREME_FEAR, BACKWARDATION is one of the strongest contrarian BUY signals.
    - CONTANGO: normal; no override needed.

    VVIX (vol-of-vol) = {_fmt(v.vvix)}:
    - VVIX > 120: VIX itself is oscillating wildly — extreme uncertainty. Reduce confidence on all calls.
    - VVIX > 100: elevated — heightened tail-risk environment; prefer shorter time horizons (SWING).

    VXN vs VIX spread = {f"{v.vxn - v.vix:+.1f}pt" if v.vxn and v.vix else "N/A"}:
    - VXN significantly above VIX (>5pt): tech sector is experiencing disproportionate fear.
      Tech names with otherwise positive signals may have an oversold bounce setup."""

    # Build put/call ratio context block for the prompt
    pc_block        = ""
    pc_instructions = ""
    if put_call_context:
        # Market-wide row
        mkt_pc_str = f"{put_call_context.market_pc_ratio:.2f}" if put_call_context.market_pc_ratio else "N/A"
        dir_icon = {"BULLISH": "▲", "BEARISH": "▼", "NEUTRAL": "→"}.get(put_call_context.market_direction, "→")

        # Per-ticker table
        ticker_rows = ""
        if put_call_context.ticker_signals:
            rows = []
            for s in put_call_context.ticker_signals:
                icon = "▼" if s.direction == "BEARISH" else "▲"
                rows.append(
                    f"  {s.ticker:<6}  P/C={s.put_call_ratio:>5.2f}  "
                    f"puts={s.put_volume:>7,}  calls={s.call_volume:>7,}  "
                    f"{s.signal:<14}  {icon} {s.direction}"
                )
            ticker_rows = "\nPer-ticker extremes (balanced readings omitted):\n" + "\n".join(rows)

        pc_block = f"""
<put_call_context>
CBOE Equity P/C Ratio: {mkt_pc_str}  →  {put_call_context.market_signal}  {dir_icon} {put_call_context.market_direction} (contrarian)
{put_call_context.summary}
{ticker_rows}

Market-wide interpretation guide (contrarian — crowd is usually wrong at extremes):
  EXTREME_GREED (<0.60) → too many calls → contrarian BEARISH warning
  GREED (0.60–0.80)     → mild complacency → slight caution
  NEUTRAL (0.80–1.00)   → balanced activity → no regime signal
  FEAR (1.00–1.20)      → elevated puts → mild contrarian BULLISH
  EXTREME_FEAR (>1.20)  → panic hedging → strong contrarian BULLISH signal

Per-ticker interpretation (directional — follows positioning):
  EXTREME_PUTS / PUTS_HEAVY → bearish institutional positioning
  CALLS_HEAVY / EXTREME_CALLS → bullish institutional positioning
</put_call_context>
"""

        pc_instructions = f"""
10. Put/Call ratio overlay:
    Market-wide P/C is {mkt_pc_str} ({put_call_context.market_signal}):
    - EXTREME_FEAR / FEAR: the crowd is panicking. Reduce SELL conviction on broad market ETFs (SPY, QQQ).
      Upgrade confidence on BUY calls that converge with positive news + technical signals.
    - EXTREME_GREED / GREED: the crowd is complacent. Reduce BUY conviction — the risk is a sentiment unwind.
      Upgrade confidence on SELL calls that converge with negative news.
    - NEUTRAL: P/C provides no incremental market-wide signal today.

    Per-ticker P/C extremes:
    - EXTREME_PUTS / PUTS_HEAVY on a ticker: institutional players are hedging or speculating bearishly.
      This CONFIRMS a SELL thesis. It WEAKENS a BUY thesis — note "heavy put positioning" in rationale.
    - CALLS_HEAVY / EXTREME_CALLS on a ticker: institutional players are speculating bullishly.
      This CONFIRMS a BUY thesis. It WEAKENS a SELL thesis — note "heavy call positioning" in rationale.
    - P/C alone never justifies a BUY/SELL. It is a confirming layer — it adds conviction when it agrees
      with the direction already supported by news + technicals + smart money."""

    # Build earnings calendar context block for the prompt
    earnings_block        = ""
    earnings_instructions = ""
    if earnings_context and earnings_context.upcoming:
        rows = []
        for ev in earnings_context.upcoming:
            eps_str = f"${ev.estimated_eps:.2f}" if ev.estimated_eps is not None else "N/A"
            urgency = "⚠️ IMMINENT" if ev.days_until <= 3 else ("⚡ THIS WEEK" if ev.days_until <= 7 else "")
            rows.append(
                f"  {ev.ticker:<6}  {ev.earnings_date}  ({ev.days_until:>2}d)  "
                f"EPS est: {eps_str:>8}  {urgency}"
            )
        table = "\n".join(rows)

        earnings_block = f"""
<earnings_calendar>
Upcoming Earnings — next {(earnings_context.upcoming[-1].earnings_date - earnings_context.report_date).days} days (as of {earnings_context.report_date})
{earnings_context.summary}

Ticker  | Report Date  | Days | EPS Est
{table}
</earnings_calendar>
"""
        imminent = [e for e in earnings_context.upcoming if e.days_until <= 3]
        this_week = [e for e in earnings_context.upcoming if 3 < e.days_until <= 7]
        imminent_str  = ", ".join(e.ticker for e in imminent)  or "none"
        this_week_str = ", ".join(e.ticker for e in this_week) or "none"

        earnings_instructions = f"""
9. Earnings calendar overlay (apply to tickers in the earnings calendar above):
   - Imminent reporters ({imminent_str} — ≤3 days): this is a BINARY EVENT. Do NOT open POSITION-horizon longs or shorts.
     If the signal is strong, label time_horizon as "SWING" at most and explicitly note the earnings risk in the rationale.
     Exception: if the consensus estimate is very low and sentiment signals are strongly positive, a SWING BUY into earnings can be flagged as a higher-risk play.
   - This-week reporters ({this_week_str} — 4-7 days): shorten time horizon to SWING or SHORT-TERM.
     Flag "pre-earnings IV expansion" as a reason to consider call options rather than stock.
     Confidence cap: max 0.85 for any BUY/SELL on these tickers regardless of signal strength.
   - EPS surprise articles in the news signal: treat these as HIGH-WEIGHT catalysts.
     A beat of >10% is a strong bullish catalyst; a miss of >10% is a strong bearish catalyst.
     Combine with technical and insider signals — an earnings beat confirmed by insider buying is a very high-conviction setup.
   - Post-earnings tickers (surprise already in the news signal): the initial gap is partially priced in.
     Focus on whether there is further follow-through vs. a fade pattern — check technical score direction."""

    commodity_tickers = ", ".join(settings.commodities_list) or "GLD, SLV, IAU, GDX, PPLT, PALL, CPER"

    prompt = f"""You are an elite portfolio manager with a verified 30-year track record of market-beating returns. You combine the analytical precision of a quant, the pattern recognition of a seasoned discretionary trader, and the macro intuition of a global macro fund manager. You have studied every major market cycle since 1990 and have an exceptional ability to identify when multiple independent evidence layers converge on the same directional call — these are the moments of highest expected value.

Your defining edge: you are ruthlessly disciplined about false positives. You understand that a wrong BUY or SELL costs capital that cannot be recovered. You output HOLD or WATCH whenever the evidence is mixed, incomplete, or driven by a single source. When you do issue a BUY or SELL, it is because the convergence of evidence makes the directional call highly reliable — and you explain precisely why.

Signal sources available today: {methods_desc}

Today's date: {fmt_et(now_et())}
{macro_block}{cot_block}{ipo_block}{vix_block}{pc_block}{earnings_block}
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
   - The pre-computed confidence already reflects: recency-weighted sentiment, article count (thin news = lower score), source diversity, volume conviction on technicals, and sector ETF alignment. Trust it — do not override upward without explicit multi-source justification.

4. Short-selling discipline:
   - SELL means initiating a short position (or buying an inverse ETF).
   - Only short when: (a) clearly negative catalyst, (b) no counter-narrative, (c) broad market not in capitulation.
{insider_instructions}{macro_instructions}{cot_instructions}{ipo_instructions}{vix_instructions}{pc_instructions}{earnings_instructions}
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
