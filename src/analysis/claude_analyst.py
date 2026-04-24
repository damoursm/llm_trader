"""Final synthesis: Claude generates a high-level market report and recommendations."""

import anthropic
import json
from loguru import logger
from typing import List, Optional, TYPE_CHECKING
from datetime import datetime, timezone
from src.utils import now_et, fmt_et
from config import settings
from src.models import TickerSignal, Recommendation, InsiderTrade, MacroContext, COTContext, IPOContext, VIXContext, PutCallContext, EarningsContext, BreadthContext, HighsLowsContext, McClellanContext, MacroSurpriseContext, FedWatchContext, RevisionMomentumContext, WhisperContext, OpExContext, SeasonalityContext, BondInternalsContext, MOVEContext, GlobalMacroContext
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
    credit_context=None,  # Optional[CreditContext]
    put_call_context: Optional["PutCallContext"] = None,
    tick_context=None,   # Optional[TICKContext]
    breadth_context: Optional["BreadthContext"] = None,
    highs_lows_context: Optional["HighsLowsContext"] = None,
    mcclellan_context: Optional["McClellanContext"] = None,
    macro_surprise_context: Optional["MacroSurpriseContext"] = None,
    fedwatch_context: Optional["FedWatchContext"] = None,
    revision_momentum_context: Optional["RevisionMomentumContext"] = None,
    whisper_context: Optional["WhisperContext"] = None,
    earnings_context: Optional["EarningsContext"] = None,
    gex_context=None,    # Optional[GEXContext]
    opex_context: Optional["OpExContext"] = None,
    seasonality_context: Optional["SeasonalityContext"] = None,
    bond_internals_context: Optional["BondInternalsContext"] = None,
    move_context: Optional["MOVEContext"] = None,
    global_macro_context: Optional["GlobalMacroContext"] = None,
) -> List[Recommendation]:
    """
    Feed all ticker signals to Claude and get final actionable recommendations.
    Includes sections for whichever analysis methods are enabled.
    """
    if not signals:
        return []

    use_news         = settings.enable_news_sentiment
    use_tech         = settings.enable_technical_analysis and settings.enable_fetch_data
    use_insider      = (
        (settings.enable_insider_trades or
         settings.enable_options_flow or
         settings.enable_sec_filings)
        and insider_trades is not None
    )
    use_put_call_signal = settings.enable_put_call and put_call_context is not None
    use_vwap_signal     = settings.enable_vwap
    use_gex_signal      = settings.enable_gex and gex_context is not None

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
        if use_insider and s.insider_cluster_detected:
            parts.append(
                f"  *** INSIDER CLUSTER: {s.insider_cluster_size} different insiders bought within 5 days "
                f"(insider_score already amplified 1.75×) ***"
            )
        if use_insider and s.insider_summary:
            parts.append(f"  Insider activity: {s.insider_summary}")
        if use_put_call_signal and s.put_call_score:
            parts.append(f"  Put/call score={s.put_call_score:+.2f} (contrarian; >0=extreme puts=bullish bias, <0=extreme calls=bearish bias)")
        if s.vwap_score:
            dist = f" ({s.vwap_distance_pct:+.1f}% from VWAP)" if s.vwap_distance_pct else ""
            parts.append(f"  VWAP_score={s.vwap_score:+.2f}{dist}")
        if s.gex_signal:
            flip = f", gamma_flip=${s.gamma_flip:.2f}" if s.gamma_flip else ""
            em   = f", exp_move=±{s.expected_move_pct:.1f}%" if s.expected_move_pct else ""
            mp   = f", max_pain_score={s.max_pain_score:+.2f}" if s.max_pain_score else ""
            sk   = f", oi_skew={s.oi_skew_score:+.2f}" if s.oi_skew_score else ""
            parts.append(f"  GEX={s.gex_signal}{flip}, max_pain_bias={s.max_pain_bias}{mp}{sk}{em}")
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
    if use_put_call_signal:
        active_methods.append("put/call ratio (per-ticker contrarian positioning)")
    if use_vwap_signal:
        active_methods.append("VWAP distance (mean-reversion vs 20-day VWAP)")
    if use_gex_signal:
        active_methods.append("gamma exposure / GEX (max pain, OI skew, dealer positioning)")
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

    # Build MOVE Index context block for the prompt
    move_block        = ""
    move_instructions = ""
    if move_context and move_context.move is not None:
        m = move_context
        dir_arrow = {"BULLISH": "▲", "BEARISH": "▼", "NEUTRAL": "→"}.get(m.direction, "→")
        spike_line = ""
        if m.spike_5d is not None:
            spike_word = "SPIKE ↑" if m.spike_5d > 0 else "DROP ↓"
            spike_line = f"  5-day change: {m.spike_5d:+.1f}pt  {'⚠ SPIKING' if m.is_spiking else ''}\n"
        ratio_line = ""
        if m.move_vix_ratio is not None:
            ratio_line = f"  MOVE/VIX ratio: {m.move_vix_ratio:.1f}×  {'⚠ DIVERGENCE: bond fear >> equity fear' if m.move_vix_ratio > 8 else ''}\n"
        avg_line = f"  20d avg: {m.move_20d_avg:.1f}" if m.move_20d_avg else ""

        move_block = f"""
<move_context>
MOVE Index (bond market VIX): {m.move:.1f}  →  {m.signal}  {dir_arrow} {m.direction}  [source: {m.source}]
{m.summary}

{spike_line}{ratio_line}{avg_line}
Signal guide:
  < 60   CALM       → unusually quiet bond market; no equity signal
  60–80  LOW        → below-average vol; mild constructive backdrop
  80–100 NORMAL     → typical regime; no override
  100–120 ELEVATED  → above-average; rising rate vol; watch for equity spillover
  120–150 HIGH      → significant stress; often precedes equity weakness 1–5 days
  150–200 EXTREME   → major disruption; BEARISH for equities
  > 200  PANIC      → crisis-level bond market stress

Spike (>20pt in 5d): even from a low base, a sharp spike is an early warning.
MOVE/VIX ratio >8×: bond market pricing stress that equities have not yet priced in → equities likely to follow.
</move_context>
"""

        level_clause = ""
        if m.signal in ("HIGH", "EXTREME", "PANIC"):
            level_clause = (
                f"\n    MOVE={m.move:.1f} ({m.signal}): Bond market is in significant stress. "
                f"Apply a −0.05 confidence haircut on POSITION-horizon BUY calls across the board. "
                f"Rate-sensitive sectors (XLK, XLRE, high-P/E growth) face amplified headwinds. "
                f"Do NOT open new aggressive longs until MOVE shows signs of mean-reverting."
            )
        elif m.signal == "ELEVATED":
            level_clause = (
                f"\n    MOVE={m.move:.1f} (ELEVATED): Bond vol above average — apply mild caution on "
                f"duration-sensitive names (growth tech, REITs). Prefer SWING over POSITION horizons."
            )

        spike_clause = ""
        if m.is_spiking and m.spike_5d is not None and m.spike_5d > 0:
            spike_clause = (
                f"\n    ⚠ BOND VOL SPIKE (+{m.spike_5d:.1f}pt in 5d): Treasury market stress is escalating. "
                f"Equity markets typically lag bond vol by 1–5 days. Reduce BUY conviction; "
                f"raise SELL conviction on credit-sensitive and rate-sensitive names."
            )
        elif m.is_spiking and m.spike_5d is not None and m.spike_5d < 0:
            spike_clause = (
                f"\n    MOVE dropping sharply ({m.spike_5d:+.1f}pt in 5d): Bond market stress de-escalating. "
                f"This is a mild tailwind — equity risk appetite tends to recover as MOVE normalises."
            )

        ratio_clause = ""
        if m.move_vix_ratio and m.move_vix_ratio > 8:
            ratio_clause = (
                f"\n    MOVE/VIX divergence ({m.move_vix_ratio:.1f}×): Bond market significantly more fearful "
                f"than equity market. Historical precedent: equities catch down within 1–5 days. "
                f"Treat this as a latent BEARISH signal — avoid aggressive new BUY calls on broad market ETFs."
            )

        move_instructions = f"""
11b. MOVE Index overlay (bond market volatility — Treasury VIX):
    Current: MOVE={m.move:.1f} ({m.signal}), direction = {m.direction}.{level_clause}{spike_clause}{ratio_clause}

    Core principles:
    - MOVE is a LEADING indicator for equities: Treasury options traders reprice risk earlier than equity options traders.
      A MOVE spike while VIX is still calm is one of the most reliable early-warning setups available.
    - MOVE > 100 consistently correlates with equity market stress in the weeks that follow.
    - Mean reversion from high MOVE levels (dropping sharply from >130) is a genuine BULLISH tail-risk fade signal.
    - MOVE/VIX > 8×: unusual divergence. Bond market sees something equity market hasn't priced yet.
      This divergence resolves either by equities selling off OR bond vol dropping — equities catching down is historically more common.
    - MOVE is NOT a per-ticker signal. It is a broad-market / regime overlay:
      * Affects ALL rate-sensitive names (XLK, XLRE, high-P/E growth, XLF from curve steepening)
      * Does NOT override company-specific catalysts (M&A, blowout earnings, fraud) — those are idiosyncratic
    - Combine MOVE with VIX for the full picture:
      * Both elevated: confirmed fear across all asset classes → strongest BEARISH regime backdrop
      * MOVE elevated, VIX calm: divergence → equities complacent; increase BEARISH weight
      * MOVE calm, VIX elevated: equity-specific fear (not systemic) → easier to fade individual stock SELLs"""

    # Build global macro context block (DXY + Copper/Gold ratio)
    global_macro_block        = ""
    global_macro_instructions = ""
    if global_macro_context:
        gm = global_macro_context
        dir_arrow_gm = {"BULLISH": "▲", "BEARISH": "▼", "NEUTRAL": "→"}.get(gm.composite_direction, "→")

        dxy_line = ""
        if gm.dxy is not None:
            dxy_line = (
                f"  DXY (US Dollar Index): {gm.dxy:.2f}"
                f"  5d={gm.dxy_return_5d:+.2f}%  20d={gm.dxy_return_20d:+.2f}%"
                f"  →  {gm.dxy_signal}  ({gm.dxy_direction} for equities)\n"
                if gm.dxy_return_5d is not None and gm.dxy_return_20d is not None
                else f"  DXY: {gm.dxy:.2f}  →  {gm.dxy_signal}\n"
            )

        cg_line = ""
        if gm.copper_gold_ratio is not None:
            cg_line = f"  Copper/Gold ratio: {gm.copper_gold_ratio:.5f}"
            if gm.copper_gold_change_5d is not None:
                cg_line += f"  5d={gm.copper_gold_change_5d:+.1f}%"
            if gm.copper_gold_change_20d is not None:
                cg_line += f"  20d={gm.copper_gold_change_20d:+.1f}%"
            cg_line += f"  →  {gm.copper_gold_signal}  ({gm.copper_gold_direction})\n"
            if gm.copper_price and gm.gold_price:
                cg_line += f"  (Copper={gm.copper_price:.4f}  Gold=${gm.gold_price:.0f})\n"

        ob_line = ""
        if gm.oil_price is not None:
            ob_line = f"  Oil (WTI CL=F): ${gm.oil_price:.1f}/bbl"
            if gm.oil_return_5d is not None:
                ob_line += f"  5d={gm.oil_return_5d:+.1f}%"
            if gm.oil_return_20d is not None:
                ob_line += f"  20d={gm.oil_return_20d:+.1f}%"
            ob_line += "\n"
        if gm.oil_bond_signal != "NEUTRAL" and gm.oil_return_5d is not None and gm.tlt_return_5d_ob is not None:
            ob_line += (
                f"  Oil/Bond divergence: Oil 5d={gm.oil_return_5d:+.1f}%  TLT 5d={gm.tlt_return_5d_ob:+.1f}%"
                f"  →  {gm.oil_bond_signal}  ({gm.oil_bond_direction})\n"
            )

        global_macro_block = f"""
<global_macro_context>
Global Macro (DXY + Copper/Gold + Oil/Bond): {gm.composite_signal}  {dir_arrow_gm} {gm.composite_direction}
{gm.summary}

{dxy_line}{cg_line}{ob_line}
DXY signal guide (5-day return):
  STRONG_BULL (>+1.5%): significant dollar strength → headwind for EM equities, commodities (GLD/SLV/CPER), multinationals
  BULL        (+0.5–1.5%): mild dollar strength → mild headwind
  NEUTRAL     (±0.5%): no dollar signal
  BEAR        (−0.5 to −1.5%): mild dollar weakness → tailwind for commodities/EM
  STRONG_BEAR (<−1.5%): significant dollar weakness → strong tailwind for commodities, EM ETFs (EEM/VWO)

Copper/Gold ratio signal guide (20-day % change):
  RISK_ON_SURGE   (>+5%):  Dr. Copper strongly bullish → global growth accelerating; favour cyclicals
  RISK_ON         (+2–5%): mild growth optimism → mild cyclical/EM tailwind
  NEUTRAL         (±2%):   no directional signal
  RISK_OFF        (−2–5%): mild contraction signal → cautious on cyclicals
  RISK_OFF_CRASH  (<−5%):  Dr. Copper signalling recession risk → strong BEARISH regime signal

Oil/Bond divergence signal guide (5-day co-movement — oil and bonds normally inverse):
  POLICY_PIVOT_SIGNAL  (oil>+2.5%, TLT>+1.5%): UNUSUAL co-rally → market pricing Fed policy pivot despite oil
                         BULLISH for equities: growth support expected to outweigh inflation concern
  STAGFLATION_RISK     (oil>+2.5%, TLT<−1.5%): rising costs + rising rates = margin compression + multiple contraction
                         BEARISH: worst regime for equity valuations; especially bad for XLK, XLRE, consumer discretionary
  GROWTH_FEAR_RISK_OFF (oil<−2.5%, TLT>+1.5%): demand destruction + flight to safety
                         BEARISH for cyclicals; classic pre-recession signal; favour GLD and defensives
  DEFLATION_SHOCK      (oil<−2.5%, TLT<−1.5%): both assets selling off → broad de-risking / liquidity squeeze
                         BEARISH: avoid risk assets; favour cash/T-bills until signal stabilises
  NEUTRAL: oil and bonds not in a significant divergence setup

Composite regime guide (DXY + Copper/Gold):
  RISK_ON:      strong dollar weakness + rising Cu/Au → full risk-on; favour cyclicals, commodities, EM
  CONSTRUCTIVE: one bullish signal dominant
  NEUTRAL:      mixed or no signal
  DEFENSIVE:    one bearish signal dominant
  RISK_OFF:     strong dollar + falling Cu/Au → full risk-off; avoid cyclicals; favour defensives/gold
</global_macro_context>
"""

        # Build instruction clauses
        dxy_clause = ""
        if gm.dxy_signal == "STRONG_BULL":
            dxy_clause = (
                f"\n    DXY STRONG_BULL ({gm.dxy_return_5d:+.2f}% 5d, level={gm.dxy:.2f}): "
                f"Dollar surging. Apply a moderate confidence haircut (−0.05) on: "
                f"(1) emerging market ETFs (EEM, VWO, MCHI), "
                f"(2) commodity names (GLD, SLV, CPER, energy), "
                f"(3) US multinationals with >50% overseas revenue. "
                f"US domestic small-caps (IWM, XLP) are relatively insulated."
            )
        elif gm.dxy_signal == "BULL":
            dxy_clause = (
                f"\n    DXY BULL ({gm.dxy_return_5d:+.2f}% 5d): "
                f"Mild dollar strength. Apply a slight haircut (−0.02) on commodity and EM BUY calls."
            )
        elif gm.dxy_signal == "STRONG_BEAR":
            dxy_clause = (
                f"\n    DXY STRONG_BEAR ({gm.dxy_return_5d:+.2f}% 5d, level={gm.dxy:.2f}): "
                f"Dollar falling sharply. Apply a mild confidence boost (+0.03) on: "
                f"commodity names (GLD, SLV, CPER), EM ETFs (EEM, VWO), "
                f"and US multinationals with large overseas revenue."
            )
        elif gm.dxy_signal == "BEAR":
            dxy_clause = (
                f"\n    DXY BEAR ({gm.dxy_return_5d:+.2f}% 5d): "
                f"Mild dollar weakness. Mild constructive lean on commodities and EM."
            )

        cg_clause = ""
        if gm.copper_gold_signal == "RISK_OFF_CRASH":
            cg_clause = (
                f"\n    Cu/Au RISK_OFF_CRASH ({gm.copper_gold_change_20d:+.1f}% 20d): "
                f"Dr. Copper in freefall vs gold — market pricing global recession risk. "
                f"Apply a strong confidence haircut (−0.08) on cyclical BUY calls (XLI, XLB, XLE, IWM). "
                f"Favour defensives (XLP, XLV, XLU) and gold (GLD). "
                f"Do NOT issue POSITION-horizon BUY calls on cyclicals or EM while Cu/Au is collapsing."
            )
        elif gm.copper_gold_signal == "RISK_OFF":
            cg_clause = (
                f"\n    Cu/Au RISK_OFF ({gm.copper_gold_change_20d:+.1f}% 20d): "
                f"Copper underperforming gold — mild contraction signal. "
                f"Apply mild haircut (−0.03) on cyclical longs; prefer defensives on POSITION horizon."
            )
        elif gm.copper_gold_signal == "RISK_ON_SURGE":
            cg_clause = (
                f"\n    Cu/Au RISK_ON_SURGE ({gm.copper_gold_change_20d:+.1f}% 20d): "
                f"Copper surging vs gold — Dr. Copper strongly bullish on global growth. "
                f"Apply a mild confidence boost (+0.04) on cyclicals (XLI, XLB), EM ETFs, and commodities. "
                f"Risk-on environment: growth / cyclical rotation confirmed."
            )
        elif gm.copper_gold_signal == "RISK_ON":
            cg_clause = (
                f"\n    Cu/Au RISK_ON ({gm.copper_gold_change_20d:+.1f}% 20d): "
                f"Copper outperforming gold — mild growth-positive signal. "
                f"Mild confidence boost (+0.02) on cyclicals with other confirming signals."
            )

        ob_clause = ""
        if gm.oil_bond_signal == "POLICY_PIVOT_SIGNAL":
            ob_clause = (
                f"\n    OIL/BOND POLICY_PIVOT_SIGNAL "
                f"(Oil 5d={gm.oil_return_5d:+.1f}%, TLT 5d={gm.tlt_return_5d_ob:+.1f}%): "
                f"Highly unusual: oil and bonds rallying simultaneously. "
                f"Market is pricing a Fed policy pivot — growth support expected to outweigh inflation concern. "
                f"Apply a mild confidence boost (+0.03) on broad-market and rate-sensitive BUY calls "
                f"(especially XLK, XLRE, growth tech with other confirming signals). "
                f"This signal is rare and short-lived — prioritise SWING horizon. "
                f"Discount if MOVE is also elevated (conflicting signals = uncertainty, not clarity)."
            )
        elif gm.oil_bond_signal == "STAGFLATION_RISK":
            ob_clause = (
                f"\n    OIL/BOND STAGFLATION_RISK "
                f"(Oil 5d={gm.oil_return_5d:+.1f}%, TLT 5d={gm.tlt_return_5d_ob:+.1f}%): "
                f"Worst regime for equities: rising costs (oil) + tightening rates (TLT falling). "
                f"Apply a strong confidence haircut (−0.07) on ALL new POSITION-horizon BUY calls. "
                f"Especially avoid: XLK (high multiple, rate sensitive), XLRE (rate sensitive), "
                f"consumer discretionary (margin compression from higher input costs + financing costs). "
                f"Favour: energy sector (XLE — direct oil beneficiary), short-duration value, cash."
            )
        elif gm.oil_bond_signal == "GROWTH_FEAR_RISK_OFF":
            ob_clause = (
                f"\n    OIL/BOND GROWTH_FEAR_RISK_OFF "
                f"(Oil 5d={gm.oil_return_5d:+.1f}%, TLT 5d={gm.tlt_return_5d_ob:+.1f}%): "
                f"Classic pre-recession signal: oil falling (demand destruction) + bonds rallying (flight to safety). "
                f"Apply a moderate confidence haircut (−0.05) on cyclical BUY calls (XLI, XLB, XLE, IWM). "
                f"Favour gold (GLD), defensives (XLP, XLV, XLU), and long-duration assets (TLT, XLRE). "
                f"Confirm with Cu/Au RISK_OFF and rising VIX for highest conviction."
            )
        elif gm.oil_bond_signal == "DEFLATION_SHOCK":
            ob_clause = (
                f"\n    OIL/BOND DEFLATION_SHOCK "
                f"(Oil 5d={gm.oil_return_5d:+.1f}%, TLT 5d={gm.tlt_return_5d_ob:+.1f}%): "
                f"Both oil and bonds selling off — broad de-risking or liquidity squeeze underway. "
                f"Avoid ALL new long positions. Prioritise SELL/HOLD calls. "
                f"This typically signals a disorderly market environment where correlations break down."
            )

        global_macro_instructions = f"""
26. Global macro cross-asset overlay (DXY + Copper/Gold + Oil/Bond divergence):
    Current regime: {gm.composite_signal} ({gm.composite_direction}).{dxy_clause}{cg_clause}{ob_clause}

    Core principles:
    - DXY is a GLOBAL TIGHTENING indicator: a rising dollar drains liquidity from EM economies
      (USD-denominated debt becomes more expensive) and compresses commodity prices (priced in USD).
      A falling dollar does the reverse. This affects SECTORS, not individual stocks.
    - Copper/Gold ratio is the most reliable non-Fed cross-asset growth barometer:
      • Copper = global industrial demand proxy ("Dr. Copper" because it predicts GDP turns)
      • Gold = safe-haven / real-rate inverse proxy
      • The RATIO removes the inflation component and isolates the growth vs. fear dimension.
      • When the ratio is falling AND VIX is rising AND credit spreads widen = triple confirmation
        of risk-off → highest conviction DEFENSIVE/SELL calls.
    - Oil/Bond divergence is the most powerful cross-asset policy signal:
      • Oil and bonds are NORMALLY inversely correlated (oil up = inflation → yields rise → TLT falls).
      • When they BOTH rally simultaneously, the normal macro logic is suspended — a POLICY PIVOT
        is being priced: the Fed is expected to cut DESPITE oil (growth fear > inflation fear).
      • When oil is up AND bonds are falling: STAGFLATION — avoid equities broadly.
      • The co-movement signal is 5-day, so it is a short-to-medium term signal (1–3 weeks).
    - SECTOR implications of a strong DXY (STRONG_BULL):
      • EM (EEM, VWO, MCHI): direct headwind — dollar cost of USD debt rises
      • Commodities (GLD, SLV, CPER, oil): direct headwind — priced in USD
      • US multinationals (AAPL, MSFT, GOOGL): earnings translation headwind
      • US domestic small-caps (IWM), US consumer staples: relatively insulated
    - Global macro is a MEDIUM-TERM signal (weeks to months). Do not override
      near-term individual stock catalysts (earnings, M&A, guidance changes).
    - Convergence check: DXY STRONG_BULL + Cu/Au RISK_OFF + STAGFLATION_RISK oil/bond + MOVE elevated
      = maximum bearish regime confidence. Avoid all new cyclical/long-duration longs."""

    # Build credit market context block for the prompt
    credit_block        = ""
    credit_instructions = ""
    if credit_context and credit_context.divergence_5d is not None:
        c = credit_context
        dir_arrow = {"BULLISH": "▲", "BEARISH": "▼", "NEUTRAL": "→"}.get(c.direction, "→")
        credit_block = f"""
<credit_context>
Credit Market Leading Indicator (HYG vs SPY):
  HYG 5d return: {c.hyg_return_5d:+.2f}%   SPY 5d return: {c.spy_return_5d:+.2f}%
  Divergence (HYG − SPY, 5d): {c.divergence_5d:+.2f}%  →  {c.signal}  {dir_arrow} {c.direction}

{c.summary}

Divergence guide (negative = HYG lagging SPY = credit stress warning):
  < −3.0%   CREDIT_STRESS   BEARISH  — credit significantly underperforming; equity weakness likely 1–3d
  < −1.5%   CREDIT_CAUTION  BEARISH  — mild credit underperformance; watch for follow-through
  −1.5–1.5% NEUTRAL         NEUTRAL  — HYG/SPY in sync; no divergence signal
  > +1.5%   CREDIT_STRONG   BULLISH  — credit leading equities; risk-on confirmation
  > +3.0%   CREDIT_SURGE    BULLISH  — strong credit outperformance; equity rally likely to follow
</credit_context>
"""
        credit_instructions = f"""
12. Credit market leading indicator (HYG vs SPY):
    Current: {c.signal} ({c.divergence_5d:+.2f}% divergence, 5d), direction = {c.direction}
    - CREDIT_STRESS: HYG is lagging SPY by >3% over 5 days. This is a serious early warning.
      * Do NOT open new POSITION-length BUY calls on broad market or cyclical names.
      * Existing BUY recommendations on rate-sensitive sectors (financials, real estate) → downgrade to WATCH.
      * Credit stress often resolves by equities catching down. Raise conviction on defensive SELL setups.
    - CREDIT_CAUTION: mild underperformance. Apply a 0.05 confidence haircut to new BUY calls.
    - NEUTRAL: no override from credit. Standard analysis applies.
    - CREDIT_STRONG / CREDIT_SURGE: credit is LEADING equities higher.
      * This is a bullish confirmation signal. Mild confidence boost (+0.03–0.05) on BUY calls for
        risk-on names (growth tech, cyclicals, small caps) when other signals also positive.
      * Do NOT use as standalone BUY trigger — it confirms but does not initiate."""

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

    # Build NYSE TICK breadth context block for the prompt
    tick_block        = ""
    tick_instructions = ""
    if tick_context and tick_context.signal not in ("UNKNOWN",):
        t = tick_context
        tick_block = f"""
<tick_context>
NYSE TICK Index (^TICK) — breadth exhaustion signal  ({t.session_date})
High: {t.tick_high or 'N/A'}   Low: {t.tick_low or 'N/A'}   Close: {t.tick_close or 'N/A'}
Signal: {t.signal}  →  contrarian direction: {t.direction}
{t.summary}

Lookback (last 5 sessions): {t.extreme_high_count} session(s) with TICK > +1000 | {t.extreme_low_count} session(s) with TICK < −1000

Interpretation guide (contrarian reversal signal):
  EXTREME_BULLS (high > +1000): institutions buying en masse → exhaustion; contrarian BEARISH
  EXTREME_BEARS (low  < −1000): panic selling cascade → capitulation; contrarian BULLISH
  WHIPSAW (both extremes hit): no directional edge; high-noise session
  NEUTRAL: breadth orderly; no reversal signal
</tick_context>
"""
        tick_instructions = f"""
13. NYSE TICK overlay — breadth exhaustion / reversal signal:
    Today's TICK: high={t.tick_high or 'N/A'}, low={t.tick_low or 'N/A'} ({t.signal} → contrarian: {t.direction}).
    This is a SHORT-TERM intraday reversal signal. Rules:
    - EXTREME_BEARS (low < −1000): panic selling is exhausted. Contrarian BULLISH.
      * For broad market ETFs (SPY, QQQ, IWM): reduce SELL conviction; upgrade HOLD/WATCH calls.
      * For individual tickers with otherwise bullish signals: note "breadth capitulation → reversal setup" in rationale.
      * Persistent EXTREME_BEARS over 3+ sessions ({t.extreme_low_count} observed): forced institutional selling / distribution.
        In this case the contrarian edge weakens — prefer "WATCH for stabilisation" over outright BUY.
    - EXTREME_BULLS (high > +1000): broad buying climax. Contrarian BEARISH.
      * Fade aggressive long calls on broad market ETFs. Tag as SWING only — don't extend horizon.
      * Persistent EXTREME_BULLS over 3+ sessions ({t.extreme_high_count} observed): late-cycle institutional accumulation.
        Momentum may continue but reward/risk deteriorates; trim conviction slightly.
    - WHIPSAW: institutions active on both sides; no edge. Do not use TICK as a modifier today.
    - NEUTRAL: TICK provides no incremental signal today. No override."""

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

    # Build GEX context block for the prompt
    gex_block        = ""
    gex_instructions = ""
    if gex_context and gex_context.signals:
        idx_sigs = {s.ticker: s for s in gex_context.signals if s.ticker in ("SPY", "QQQ", "IWM")}
        indiv    = [s for s in gex_context.signals if s.ticker not in idx_sigs]

        def _gex_row(s):
            flip = f"${s.gamma_flip:.2f}" if s.gamma_flip else "N/A"
            pain = f"${s.max_pain:.2f} ({s.max_pain_bias})" if s.max_pain else "N/A"
            em   = f"±{s.expected_move_pct:.1f}%" if s.expected_move_pct else "N/A"
            return (
                f"  {s.ticker:<6} {s.gex_signal:<10} norm={s.gex_normalized:+.2f}  "
                f"flip={flip:<10} max_pain={pain:<20} exp_move={em}"
            )

        idx_rows   = "\n".join(_gex_row(s) for s in idx_sigs.values())
        indiv_rows = "\n".join(_gex_row(s) for s in indiv[:15])  # cap to 15 individual tickers

        gex_block = f"""
<gex_context>
Gamma Exposure (GEX) — dealer positioning as of {gex_context.report_date}
{gex_context.summary}

Index ETFs (most reliable GEX data):
  Ticker GEX_Signal  Norm       Gamma_Flip  Max_Pain(Bias)       Exp_Move
{idx_rows}

Individual tickers (top {min(15, len(indiv))} by options liquidity):
{indiv_rows}

Legend:
  PINNED    — dealers long gamma → stabilising; price suppressed near gamma flip
  AMPLIFIED — dealers short gamma → destabilising; directional moves will accelerate
  NEUTRAL   — no meaningful dealer gamma influence
  Gamma flip: price below which GEX turns negative and moves accelerate
  Max pain:   expiry gravitational level (strongest pull in final 3-5 days before expiry)
  Exp move:   ATM straddle / spot = market-implied ±1σ range to nearest expiry
</gex_context>
"""

        gex_instructions = """
12. GEX (Gamma Exposure) overlay — options market structure:
    Each ticker's GEX signal, gamma flip level, max pain, and expected move are shown in the
    signals block above AND in the <gex_context> table.  Use them as follows:

    PINNED (positive GEX):
    - Dealers are net long gamma → they SELL rallies and BUY dips to delta-hedge.
    - This dampens volatility and keeps price near the gamma flip level.
    - Do NOT set short time_horizons (SWING) unless a hard catalyst (earnings, news) is imminent.
    - Require additional signal convergence before issuing a BUY/SELL — signals are slower to
      materialise in a pinned regime.

    AMPLIFIED (negative GEX):
    - Dealers are net short gamma → they BUY into rallies and SELL into drops (momentum feeding).
    - Moves will be larger and faster than normal.
    - A BUY or SELL signal with AMPLIFIED GEX can set time_horizon = "SWING" with higher conviction.
    - Note: it cuts both ways — a bad entry in an AMPLIFIED regime loses faster too.

    Gamma flip:
    - The gamma flip is the line between a stabilising and destabilising dealer regime.
    - If the current price is near (within 1%) or below the gamma flip, treat it as a key breakdown
      trigger: breaking below = accelerated move; holding above = potential pin/bounce.
    - Mention the gamma flip level explicitly in the rationale for any SWING BUY/SELL.

    Max pain (gravity score):
    - A numeric max_pain_score ∈ [-1, +1] is now pre-computed and included in each ticker's signal.
      It combines the direction of the pull (spot vs max_pain) with an expiry-decay factor:
        • > 0: spot is below max pain → bullish gravitational pull toward max pain into expiry.
        • < 0: spot is above max pain → bearish gravitational pull toward max pain into expiry.
        • Score fades to near-zero beyond 14 days; strongest within 2-4 days of expiry.
    - Treat max_pain_score as a weak corroborating signal: ≥ +0.30 or ≤ -0.30 is notable.
    - If max_pain_score reinforces the combined direction, mildly boost confidence.
    - If max_pain_score contradicts the combined direction, note the conflict; avoid aggressive
      BUY/SELL targets where price must move through max pain to reach the target.
    - Only meaningful within the final week before expiry; ignore for POSITION horizons.

    Expected move:
    - The options market is pricing ±X% by the nearest expiry.
    - A BUY recommendation with 0.85 confidence on a name with ±0.5% expected move is inconsistent —
      the market doesn't believe it will move much. Reduce confidence or extend horizon.
    - A name with ±3% expected move + AMPLIFIED GEX + directional signal convergence = high-conviction
      setup; acknowledge the expected-move range in the rationale."""

    # Build market breadth context block for the prompt
    breadth_block        = ""
    breadth_instructions = ""
    if breadth_context:
        b = breadth_context
        spy_str = ""
        if b.spy_above_200d is not None:
            pos = "ABOVE" if b.spy_above_200d else "BELOW"
            dist = f" ({b.spy_200d_distance_pct:+.1f}% from 200d SMA)" if b.spy_200d_distance_pct is not None else ""
            spy_str = f"  SPY: {pos} its 200d SMA{dist}\n"

        delta_line = ""
        if b.pct_above_200d_5d_ago is not None:
            delta = b.pct_above_200d - b.pct_above_200d_5d_ago
            delta_line = f"  5d ago: {b.pct_above_200d_5d_ago:.0f}%  →  now: {b.pct_above_200d:.0f}%  (Δ{delta:+.0f}pp)\n"

        thrust_line = ""
        if b.is_breadth_thrust:
            rise = b.pct_above_200d - (b.pct_above_200d_5d_ago or b.pct_above_200d)
            thrust_line = f"\n⚡ BREADTH THRUST CONFIRMED: +{rise:.0f}pp rise from oversold base — historically one of the strongest multi-month bullish setups.\n"

        breadth_block = f"""
<breadth_context>
Market Breadth — Sector ETFs above 200-day SMA: {b.pct_above_200d:.0f}%  ({b.etfs_above}/{b.etf_count} ETFs)
Signal: {b.signal}  →  direction: {b.direction}
{spy_str}{delta_line}{thrust_line}
{b.summary}

Signal guide:
  ≥ 85%   BREADTH_EXTENDED  → contrarian BEARISH: market over-extended; limited upside
  70–84%  BREADTH_HEALTHY   → BULLISH: broad participation confirms trend
  50–69%  BREADTH_MIXED     → NEUTRAL: stock-picking environment; no broad override
  30–49%  BREADTH_WEAK      → BEARISH: more sectors below 200d SMA; caution on longs
  < 30%   BREADTH_COLLAPSE  → BEARISH; rising from this level = breadth thrust (very bullish)
</breadth_context>
"""

        delta = b.pct_above_200d - (b.pct_above_200d_5d_ago or b.pct_above_200d)
        thrust_clause = ""
        if b.is_breadth_thrust:
            thrust_clause = (
                f"\n    ⚡ BREADTH THRUST ACTIVE: breadth rose +{delta:.0f}pp from an oversold base. "
                f"Historically, breadth thrusts generate sustained 3-6 month rallies with very high win rates. "
                f"Upgrade BUY conviction on broad market ETFs (SPY, QQQ, IWM) and sector leaders by +0.05-0.08 "
                f"when other signals also support the direction."
            )

        breadth_instructions = f"""
15. Market breadth overlay (% of sector ETFs above 200-day SMA):
    Current: {b.pct_above_200d:.0f}% above 200d SMA ({b.signal}, {b.direction} direction).{thrust_clause}
    - BREADTH_COLLAPSE (<30%): Market broadly oversold across sectors — forced selling, not fundamentals.
      * Do NOT open aggressive new SELL positions: broad capitulation creates violent mean-reversion risk.
      * Watch for the breadth thrust signal (rising 8+ pp from this level) — it would be one of the
        highest-conviction BUY setups available, overriding other bearish overlays on broad market ETFs.
    - BREADTH_WEAK (30-50%): More sectors below 200d than above — market breadth deteriorating.
      * Apply a mild confidence haircut (-0.03 to -0.05) on POSITION-horizon BUY calls.
      * Prefer SWING trades with clear near-term catalysts. Avoid speculative or small-cap longs.
    - BREADTH_MIXED (50-70%): No breadth override — standard signal analysis applies.
    - BREADTH_HEALTHY (70-85%): Broad participation confirms the prevailing uptrend.
      * Mild confidence boost (+0.03) on BUY calls aligned with the trend.
      * Short-selling is structurally harder — require a very strong specific catalyst for SELL calls.
    - BREADTH_EXTENDED (≥85%): Nearly all sectors above 200d SMA — complacency risk.
      * Do NOT open new POSITION-horizon longs. Risk/reward deteriorates at breadth extremes.
      * Prefer SWING over POSITION horizons on any new BUY. Explicitly note the extended breadth."""

    # Build 52-week Highs / Lows context block for the prompt
    highs_lows_block        = ""
    highs_lows_instructions = ""
    if highs_lows_context:
        hl = highs_lows_context
        dir_arrow = {"BULLISH": "▲", "BEARISH": "▼", "NEUTRAL": "→"}.get(hl.direction, "→")

        div_line = ""
        if hl.is_bearish_divergence:
            div_line = "\n⚠ BEARISH DIVERGENCE: SPY near 52-week high but HL spread declining — rally led by fewer names.\n"
        elif hl.is_bullish_divergence:
            div_line = "\n⚡ BULLISH DIVERGENCE: SPY near 52-week low but new lows contracting — selling exhaustion.\n"

        trend_line = ""
        if hl.hl_spread_5d_ago is not None:
            delta = hl.hl_spread - hl.hl_spread_5d_ago
            trend_line = f"  5d ago: {hl.hl_spread_5d_ago:+.0f}pp  →  now: {hl.hl_spread:+.0f}pp  (Δ{delta:+.0f}pp)\n"

        spy_line = ""
        if hl.spy_pct_from_52w_high is not None and hl.spy_pct_from_52w_low is not None:
            spy_line = (
                f"  SPY: {hl.spy_pct_from_52w_high:+.1f}% from 52w high  |  "
                f"{hl.spy_pct_from_52w_low:+.1f}% from 52w low\n"
            )

        highs_lows_block = f"""
<highs_lows_context>
New 52-Week Highs vs. Lows — HL Spread: {hl.hl_spread:+.0f}pp  →  {hl.signal}  {dir_arrow} {hl.direction}
{div_line}
Basket counts ({hl.total_count} tickers checked):
  Near 52w high (within 5%): {hl.highs_count}  ({hl.pct_near_highs:.0f}%)
  Near 52w low  (within 5%): {hl.lows_count}  ({hl.pct_near_lows:.0f}%)
  Neutral:                   {hl.neutral_count}
{trend_line}{spy_line}
{hl.summary}

HL Spread signal guide:
  ≥ +50   STRONG_HIGHS    → BULLISH:  most names near cycle highs; broad-based strength
  +20–50  HIGHS_DOMINATE  → BULLISH:  more highs than lows; trend has wide participation
  −20–20  BALANCED        → NEUTRAL:  no breadth edge today
  −50– −20 LOWS_DOMINATE  → BEARISH:  more lows than highs; underlying weakness building
  ≤ −50   STRONG_LOWS     → BEARISH:  broad deterioration; most names near cycle lows

Divergence interpretation (best signal, precedes reversals 1–2 weeks):
  Bearish: SPY at/near 52w high + HL spread declining → narrowing leadership = distribution
  Bullish: SPY at/near 52w low  + HL spread rising   → lows contracting = capitulation exhaustion
</highs_lows_context>
"""

        div_clause = ""
        if hl.is_bearish_divergence:
            div_clause = (
                "\n    ⚠ BEARISH DIVERGENCE ACTIVE: SPY is near its 52-week high but the HL spread has "
                "deteriorated over the past 5 sessions. This is a classic distribution signal — the index "
                "is levitating on shrinking leadership. Reduce BUY conviction on broad market ETFs; "
                "upgrade SELL conviction on broad market when other signals also bearish. "
                "Time-horizon: cap BUY calls at SWING; no POSITION-length longs on broad ETFs."
            )
        elif hl.is_bullish_divergence:
            div_clause = (
                "\n    ⚡ BULLISH DIVERGENCE ACTIVE: SPY is near its 52-week low but new lows are contracting. "
                "Selling is becoming exhausted across the broad market — a classic capitulation signal. "
                "Upgrade BUY conviction on broad market ETFs by +0.05 when other signals also bullish. "
                "Time-horizon: SWING entries have a historically high win rate in this setup."
            )

        highs_lows_instructions = f"""
17. New 52-Week Highs vs. Lows overlay — breadth participation and divergence:
    Current: HL Spread={hl.hl_spread:+.0f}pp ({hl.signal}), direction={hl.direction}.{div_clause}
    - STRONG_HIGHS (spread ≥ +50): near-unanimous strength across the basket.
      * Mildly contrarian at extremes: if hl_spread_5d_ago was lower, trend is accelerating → BUY confirm.
      * If hl_spread_5d_ago was also ≥ +50 and spread is now flat/declining → breadth peaking; reduce new longs.
    - HIGHS_DOMINATE (+20 to +50): healthy participation. Mild confidence boost (+0.03) on BUY calls.
    - BALANCED (-20 to +20): no breadth edge. Do not use as a modifier today.
    - LOWS_DOMINATE (-50 to -20): more names hitting lows than highs. Apply -0.03 haircut on new BUY calls.
      Prefer individual names with strong specific catalysts over broad market index ETFs.
    - STRONG_LOWS (spread ≤ -50): broad deterioration. Do NOT chase new POSITION-horizon longs.
      This level can persist during bear markets; wait for a confirmed spread improvement before BUY calls.
    - Highs/Lows is a MARKET-WIDE overlay. For individual stocks with strong specific catalysts, it is
      a secondary modifier only — it never overrides a strong company-level thesis.
    - Use in conjunction with McClellan and breadth: all three confirming the same direction = highest conviction."""

    # Build McClellan Oscillator context block for the prompt
    mcclellan_block        = ""
    mcclellan_instructions = ""
    if mcclellan_context:
        m = mcclellan_context

        cross_line = ""
        if m.is_bullish_cross:
            cross_line = "\n⚡ BULLISH ZERO CROSS: oscillator crossed above 0 — momentum shift confirmed.\n"
        elif m.is_bearish_cross:
            cross_line = "\n⚠ BEARISH ZERO CROSS: oscillator crossed below 0 — momentum shift confirmed.\n"

        osc_delta_str = ""
        if m.oscillator_5d_ago is not None:
            osc_delta_str = f"  (5d ago: {m.oscillator_5d_ago:+.1f}, Δ{m.oscillator - m.oscillator_5d_ago:+.1f})"

        si_delta_str = ""
        if m.summation_5d_ago is not None:
            si_delta = m.summation - m.summation_5d_ago
            si_delta_str = f"  (5d Δ{si_delta:+.0f})"

        mcclellan_block = f"""
<mcclellan_context>
McClellan Oscillator: {m.oscillator:+.1f}{osc_delta_str}  →  {m.osc_signal}  (direction: {m.direction})
McClellan Summation:  {m.summation:+.0f}{si_delta_str}  →  {m.sum_signal}
{cross_line}
EMA19 (fast): {m.ema19:+.1f}   EMA39 (slow): {m.ema39:+.1f}

{m.summary}

Oscillator signal guide:
  > +100   OVERBOUGHT        → contrarian BEARISH: momentum stretched; breadth exhaustion
  > +50    BULLISH_MOMENTUM  → BULLISH: net advances accelerating above trend
  -50–50   NEUTRAL           → no breadth momentum signal
  < -50    BEARISH_MOMENTUM  → BEARISH: net declines accelerating
  < -100   OVERSOLD          → contrarian BULLISH: capitulation; coiling for reversal

Summation Index guide:
  > +500   EXTENDED_BULL → overstretched; limit new longs
  > 0      BULL_TREND    → breadth trend positive; buy pullbacks
  < 0      BEAR_TREND    → breadth trend negative; sell rallies
  < -500   EXTENDED_BEAR → approaching major reversal zone; watch for turn

Zero-line crossings (most reliable swing signal):
  Oscillator crosses above 0 → EMA19 > EMA39: bullish momentum shift; high-probability buy timing
  Oscillator crosses below 0 → EMA19 < EMA39: bearish momentum shift; high-probability sell timing
</mcclellan_context>
"""

        cross_clause = ""
        if m.is_bullish_cross:
            cross_clause = (
                "\n    ⚡ BULLISH ZERO CROSS ACTIVE: Oscillator just crossed above 0. This is the highest-conviction "
                "swing-trade timing signal from the McClellan. Upgrade BUY confidence on broad market ETFs (SPY, QQQ, IWM) "
                "by +0.05 when other signals agree. Time-horizon: SWING or SHORT-TERM."
            )
        elif m.is_bearish_cross:
            cross_clause = (
                "\n    ⚠ BEARISH ZERO CROSS ACTIVE: Oscillator just crossed below 0. Bearish momentum shift confirmed. "
                "Reduce BUY confidence on broad market ETFs; upgrade SELL confidence when other signals also bearish."
            )

        mcclellan_instructions = f"""
16. McClellan Oscillator & Summation overlay — breadth momentum and swing timing:
    Current: Oscillator={m.oscillator:+.1f} ({m.osc_signal}), Summation={m.summation:+.0f} ({m.sum_signal}), direction={m.direction}.{cross_clause}
    - OVERBOUGHT (osc > +100): Net advances over-extended. Breadth exhaustion is near.
      * Apply a confidence haircut (-0.05) on new POSITION-horizon BUY calls.
      * If osc is also turning lower from this level, it often marks an intermediate top.
    - BULLISH_MOMENTUM (osc +50 to +100): Breadth accelerating positively.
      * Mild confidence boost (+0.03) on BUY calls when osc is also still rising.
      * Summation in BULL_TREND zone adds further confirmation.
    - NEUTRAL (osc -50 to +50): No A/D directional edge. Do not use as a modifier.
    - BEARISH_MOMENTUM (osc -50 to -100): Net declines accelerating.
      * Apply -0.03 confidence haircut on new BUY calls on broad market names.
      * Prefer individual names with strong specific catalysts over broad market longs.
    - OVERSOLD (osc < -100): Breadth in capitulation. Mean-reversion setup coiling.
      * Do NOT chase new SELLs on broad market ETFs — capitulation creates violent snapbacks.
      * When osc is also turning higher from this level (5d-ago lower), upgrade BUY conviction
        on broad market ETFs by +0.05. This combination (oversold + turning) has high win rate.
    - Summation Index direction:
      * BULL_TREND (SI > 0): breadth trend positive; buy dips, avoid aggressive shorts.
      * BEAR_TREND (SI < 0): breadth trend negative; sell rallies, avoid aggressive longs.
      * EXTENDED levels (|SI| > 500): trend overstretched; risk of reversal increases.
    - McClellan is a MARKET-WIDE overlay, not per-ticker. Apply to broad market ETFs and
      sector ETFs. For individual stocks with strong specific catalysts, McClellan is a
      secondary modifier only — it never overrides a strong company-level thesis."""

    # Build Macro Surprise context block for the prompt
    macro_surprise_block        = ""
    macro_surprise_instructions = ""
    if macro_surprise_context:
        ms = macro_surprise_context
        dir_arrow = {"BULLISH": "▲", "BEARISH": "▼", "NEUTRAL": "→"}.get(ms.direction, "→")

        ind_rows = "\n".join(
            f"  {ind.name:<24}  actual={ind.actual:>8.3f} {ind.unit:<8}  "
            f"expected={ind.expected:>8.3f}  z={ind.z_score:>+5.2f}  {ind.signal}  [{ind.release_date}]"
            for ind in ms.indicators
        )

        macro_surprise_block = f"""
<macro_surprise_context>
Macro Economic Surprise Index (CESI-style): score={ms.score:+.2f}  →  {ms.signal}  {dir_arrow} {ms.direction}
{ms.beats} beats / {ms.in_line} in-line / {ms.misses} misses across {len(ms.indicators)} FRED indicators

{ms.summary}

Per-indicator breakdown (z-score = sign-adjusted; positive = upside surprise):
{ind_rows}

Score guide:
  > +0.40  STRONG_BEAT  BULLISH  — economy accelerating well above recent trend
  +0.15–0.40 MILD_BEAT  BULLISH  — modest positive momentum
  ±0.15    NEUTRAL      NEUTRAL  — in line with recent trend
  -0.40– -0.15 MILD_MISS BEARISH — modest negative momentum
  < -0.40  STRONG_MISS  BEARISH  — economy decelerating well below trend

Indicator sign convention: CPI and Unemployment are sign-flipped (lower = positive surprise).
</macro_surprise_context>
"""

        surprise_verb = "beating" if ms.direction == "BULLISH" else ("missing" if ms.direction == "BEARISH" else "meeting")
        macro_surprise_instructions = f"""
18. Macro Economic Surprise overlay (CESI-style — apply to ALL recommendations):
    Current score: {ms.score:+.2f} ({ms.signal}), direction={ms.direction}.
    Economic data is broadly {surprise_verb} recent trend expectations.
    - STRONG_BEAT (score > +0.40): Cyclical tailwind. Consistent upside surprises accelerate growth expectations.
      * Boost BUY conviction on cyclical sectors (industrials, materials, consumer discretionary, financials) by +0.03-0.05.
      * Rate-sensitive sectors may see pressure if beats suggest Fed stays higher for longer — note the trade-off.
      * Commodities: industrial metals (CPER) get a mild tailwind; precious metals may face headwind (less easing needed).
    - MILD_BEAT (+0.15 to +0.40): Modest positive momentum. Mildly constructive for risk assets.
      * No broad override — use as a marginal confidence boost (+0.02) for cyclical BUY calls with other support.
    - NEUTRAL (±0.15): Data in line with trend. No surprise overlay today.
    - MILD_MISS (-0.40 to -0.15): Data softening relative to trend. Minor headwind.
      * Apply a mild confidence haircut (-0.02) on new POSITION-horizon BUY calls in cyclical sectors.
      * Defensives (XLV, XLP, XLU) and safe-haven assets (GLD) see mild tailwind.
    - STRONG_MISS (score < -0.40): Economy decelerating meaningfully below trend.
      * Significantly reduce conviction on POSITION-horizon longs in cyclicals.
      * Defensives and gold are meaningfully more attractive. Note deceleration risk in rationale.
      * If also in late-cycle macro regime, this convergence is a strong signal to cut broad market exposure.
    - Surprise score is based on FRED releases and lags realtime events by weeks. Use as MEDIUM-TERM context.
      Do NOT use it as a short-term swing signal — it does not override near-term news catalysts.
    - Convergence check: macro surprise + FRED regime + credit + VIX all pointing same direction = highest conviction
      for broad-market-level calls. Single indicator alone is confirming only."""

    # Build FedWatch context block for the prompt
    fedwatch_block        = ""
    fedwatch_instructions = ""
    if fedwatch_context:
        fw = fedwatch_context
        dir_arrow = {"BULLISH": "▲", "BEARISH": "▼", "NEUTRAL": "→"}.get(fw.direction, "→")
        trend_icon = {"DOVISH_SHIFT": "⬇ DOVISH SHIFT", "HAWKISH_SHIFT": "⬆ HAWKISH SHIFT", "NEUTRAL": "→ NEUTRAL"}.get(fw.rate_trend, "→")

        tbill_line = ""
        if fw.tbill_3m is not None:
            tbill_line = (
                f"  3m T-bill={fw.tbill_3m:.3f}%  "
                f"({'↓'+f'{abs(fw.tbill_3m-fw.tbill_3m_5d_ago)*100:.1f}bp' if fw.tbill_3m_5d_ago and fw.tbill_3m < fw.tbill_3m_5d_ago else '↑'+f'{abs(fw.tbill_3m-fw.tbill_3m_5d_ago)*100:.1f}bp' if fw.tbill_3m_5d_ago else '—'})  "
                f"6m T-bill={fw.tbill_6m:.3f}%  12m T-bill={fw.tbill_12m:.3f}%\n"
                if fw.tbill_6m and fw.tbill_12m else ""
            )

        next_line = ""
        if fw.next_meeting:
            next_line = (
                f"  Next FOMC: {fw.next_meeting}  ({fw.days_to_next_meeting}d)  "
                f"P(cut)={fw.p_cut_next:.0%}  P(hold)={fw.p_hold_next:.0%}  P(hike)={fw.p_hike_next:.0%}\n"
            )

        fedwatch_block = f"""
<fedwatch_context>
Fed Rate Expectations: FF {fw.ff_lower:.2f}–{fw.ff_upper:.2f}%  →  {fw.signal}  {dir_arrow} {fw.direction}
Weekly trend: {trend_icon}
{fw.summary}

Implied rate changes (T-bill spread = market's expected average FF rate minus current target):
  3m horizon:  {fw.implied_cuts_3m_bp:+.1f}bp  (cuts/hikes expected over next 90 days)
  6m horizon:  {fw.implied_cuts_6m_bp:+.1f}bp
  12m horizon: {fw.implied_cuts_12m_bp:+.1f}bp  ← primary signal
{tbill_line}{next_line}
Signal guide (based on 12m implied cuts):
  ≥ +75bp  STRONGLY_DOVISH   BULLISH  — 3+ cuts priced in; major equity tailwind
  ≥ +25bp  DOVISH            BULLISH  — 1–3 cuts priced in; supports risk assets
  ≥  +8bp  MILDLY_DOVISH     NEUTRAL  — partial cut bias; mild easing tailwind
   ±  8bp  NEUTRAL                    — no rate-change expectation; no override
  ≤  −8bp  MILDLY_HAWKISH    BEARISH  — slight hike risk; mild headwind
  ≤ −25bp  HAWKISH           BEARISH  — 1+ hikes priced in; pressure on rate-sensitive names
  ≤ −75bp  STRONGLY_HAWKISH  BEARISH  — major tightening; significant headwind for growth/tech
</fedwatch_context>
"""

        dovish_shift_clause = ""
        hawkish_shift_clause = ""
        if fw.rate_trend == "DOVISH_SHIFT":
            shift_bp = abs(fw.tbill_3m - fw.tbill_3m_5d_ago) * 100 if fw.tbill_3m and fw.tbill_3m_5d_ago else 0
            dovish_shift_clause = (
                f"\n    ⬇ DOVISH SHIFT: T-bills fell {shift_bp:.1f}bp this week — "
                f"the market is repricing for more rate cuts. This is a BULLISH impulse, especially for: "
                f"growth tech (rate-sensitive), REITs (financing cost relief), small caps (floating-rate debt). "
                f"Apply a mild confidence boost (+0.03) on BUY calls for rate-sensitive names when other signals also positive."
            )
        elif fw.rate_trend == "HAWKISH_SHIFT":
            shift_bp = abs(fw.tbill_3m - fw.tbill_3m_5d_ago) * 100 if fw.tbill_3m and fw.tbill_3m_5d_ago else 0
            hawkish_shift_clause = (
                f"\n    ⬆ HAWKISH SHIFT: T-bills rose {shift_bp:.1f}bp this week — "
                f"the market is repricing for fewer cuts (or potential hikes). This is a BEARISH impulse for: "
                f"long-duration growth tech, REITs, bonds. Apply a mild confidence haircut (-0.03) on BUY calls "
                f"for rate-sensitive names."
            )

        next_meeting_clause = ""
        if fw.next_meeting and fw.days_to_next_meeting is not None and fw.days_to_next_meeting <= 7:
            next_meeting_clause = (
                f"\n    ⚠ FOMC MEETING IMMINENT ({fw.days_to_next_meeting}d): "
                f"P(cut)={fw.p_cut_next:.0%}, P(hold)={fw.p_hold_next:.0%}, P(hike)={fw.p_hike_next:.0%}. "
                f"An imminent FOMC is a binary event for rate-sensitive tickers (growth tech, REITs, financials). "
                f"Do NOT extend to POSITION horizons for rate-sensitive names this close to the meeting. "
                f"A surprise cut (if P(cut) > 60%) would be highly bullish for equities broadly; a surprise hike would be sharply bearish."
            )

        fedwatch_instructions = f"""
19. Fed Rate Expectations overlay (market-implied monetary policy path):
    Current: {fw.signal} — {fw.implied_cuts_12m_bp:+.0f}bp priced in over 12 months.{dovish_shift_clause}{hawkish_shift_clause}{next_meeting_clause}
    - STRONGLY_DOVISH / DOVISH (12m cuts ≥ +25bp): Easing expectations are a structural tailwind for equities.
      * Boost BUY conviction on rate-sensitive sectors: growth tech (XLK), REITs (XLRE), small caps (IWM) by +0.03-0.05.
      * Gold (GLD) benefits from a falling real-rate outlook — confirm with COT and technical signals.
      * Financials (XLF) face net interest margin pressure when cuts are deep — don't conflate dovish with "buy banks."
    - MILDLY_DOVISH (8-25bp): Mild easing bias. Use as a marginal confirming factor only (+0.02 on rate-sensitive BUYs).
    - NEUTRAL (±8bp): Rate expectations provide no directional edge today. Do not use as a modifier.
    - MILDLY_HAWKISH / HAWKISH (cuts ≤ -8bp): Tighter-for-longer expectations.
      * Apply a mild confidence haircut (-0.03 to -0.05) on POSITION-horizon BUY calls for growth tech and REITs.
      * Financials (XLF) benefit from higher-for-longer rates in the short run (NIM expansion).
      * Short-duration value stocks are more resilient — prefer cyclicals with near-term earnings catalysts.
    - STRONGLY_HAWKISH (cuts ≤ -75bp): Major tightening priced in. Significant headwind for high-multiple names.
      * Do NOT open POSITION-horizon longs on growth/tech without a very compelling company-specific thesis.
    - Rate expectations are a MEDIUM-TERM overlay. They do not override near-term earnings catalysts or strong news flow.
      A strong earnings beat on a growth tech name can still be BUY even in a hawkish rate environment — just shorten horizon."""

    # Build Revision Momentum context block for the prompt
    revision_block        = ""
    revision_instructions = ""
    if revision_momentum_context and revision_momentum_context.tickers:
        rm = revision_momentum_context
        dir_arrow = {"BULLISH": "▲", "BEARISH": "▼", "NEUTRAL": "→"}.get(rm.direction, "→")

        # Per-ticker table (sort by momentum score descending)
        sorted_tickers = sorted(rm.tickers, key=lambda t: t.momentum_score, reverse=True)
        ticker_rows = "\n".join(
            f"  {t.ticker:<6}  score={t.momentum_score:+.3f}  {t.direction:<13}  "
            f"recent: {t.recent_upgrades}↑{t.recent_downgrades}↓ {t.recent_pt_raises}PT↑{t.recent_pt_cuts}PT↓  "
            f"prior: {t.prior_upgrades}↑{t.prior_downgrades}↓ {t.prior_pt_raises}PT↑{t.prior_pt_cuts}PT↓"
            + (f"  PT chg: {t.pt_change_pct:+.1f}%" if t.pt_change_pct is not None else "")
            for t in sorted_tickers
        )

        improving_str    = ", ".join(rm.top_improving)    or "none"
        deteriorating_str = ", ".join(rm.top_deteriorating) or "none"

        revision_block = f"""
<revision_momentum_context>
Estimate Revision Momentum (analyst consensus trend — 30d recent vs 31-60d prior):
Breadth score={rm.breadth_score:+.2f}  →  {rm.signal}  {dir_arrow} {rm.direction}
{rm.summary}

Rising revisions (top improving):     {improving_str}
Falling revisions (top deteriorating): {deteriorating_str}

Per-ticker breakdown (score ∈ [-1,+1]; IMPROVING ≥ +0.25, DETERIORATING ≤ -0.25):
{ticker_rows}

Interpretation:
  IMPROVING     → analyst consensus is accelerating; recent upgrades/PT-raises > prior period
  STABLE        → revision activity similar between windows; no momentum signal
  DETERIORATING → analyst consensus is decelerating; recent downgrades/PT-cuts > prior period
  breadth_score summarises the cross-watchlist average; positive = more tickers improving than not
</revision_momentum_context>
"""

        revision_instructions = f"""
20. Estimate Revision Momentum overlay (analyst consensus trend):
    Breadth score={rm.breadth_score:+.2f} ({rm.signal}), direction={rm.direction}.
    Top improving: {improving_str}.  Top deteriorating: {deteriorating_str}.

    Key principle: the TREND in revisions matters more than any single upgrade.
    A stream of rising price targets and upgrades over 30 days = earnings momentum factor.
    Analyst estimate cuts accelerating = deteriorating earnings visibility.

    - STRONG_IMPROVING / IMPROVING: Analyst consensus is in an upgrade cycle.
      * For tickers in the top_improving list with otherwise positive signals: mild confidence boost (+0.03).
        The upgrade cycle confirms the thesis and suggests institutional coverage rotation is underway.
      * Rising avg PT across windows (pt_change_pct > 0): institutions are raising their targets —
        implies expanding earnings expectations. Note the PT trend in the rationale.
      * Do NOT initiate a BUY on revision momentum alone — it must converge with news + technical signals.
    - NEUTRAL: No revision momentum edge today. Do not use as a modifier.
    - DETERIORATING / STRONG_DETERIORATING: Analysts are cutting estimates or downgrading.
      * For tickers in the top_deteriorating list: apply a mild confidence haircut (-0.03) on BUY calls.
      * Falling avg PT (pt_change_pct < 0): consensus is reducing earnings expectations.
        Note "analyst estimate cuts building" in any BUY rationale as a risk factor.
      * Downgrade cycles often persist for 2-3 quarters — do NOT chase bottoms on names in active downgrade cycles.
    - Revision momentum is a MEDIUM-TERM signal (weeks to months). It does not override near-term
      catalysts like earnings beats or surprise news. Use as a confirming or cautionary overlay only.
    - Breadth score > 0 with multiple improving tickers: constructive for the broad watchlist.
      Breadth score < 0 with multiple deteriorating tickers: analyst community broadly reducing expectations —
      a headwind for the overall portfolio even if individual names look bullish on news."""

    # Build Earnings Whisper context block for the prompt
    whisper_block        = ""
    whisper_instructions = ""
    if whisper_context and whisper_context.signals:
        wc = whisper_context

        # Only include tickers with upcoming earnings OR strong historical signal
        relevant = [
            s for s in wc.signals
            if s.days_until_earnings is not None
            or s.signal in ("BEAT_LIKELY", "MISS_LIKELY")
            or abs(s.avg_eps_surprise_pct) >= 2.0
        ]
        if not relevant:
            relevant = wc.signals  # fallback: include all

        rows = []
        for s in sorted(relevant, key=lambda x: (x.days_until_earnings or 999, -abs(s.avg_eps_surprise_pct))):
            date_str = f" [{s.earnings_date} ({s.days_until_earnings}d)]" if s.earnings_date else ""
            whisper_str = (
                f"  implied_whisper=${s.implied_whisper:.2f} ({s.whisper_gap_pct:+.1f}%)"
                if s.implied_whisper else ""
            )
            trend_str = (
                f"  trend={s.eps_trend_direction}"
                + (f"({s.eps_trend_current:.2f} vs {s.eps_trend_30d:.2f} 30d ago)"
                   if s.eps_trend_current and s.eps_trend_30d else "")
            ) if s.eps_trend_direction != "STABLE" else ""
            rev_str = f"  rev:{s.revisions_up_30d}↑{s.revisions_down_30d}↓" if (s.revisions_up_30d + s.revisions_down_30d) > 0 else ""
            rows.append(
                f"  {s.ticker:<6}{date_str}  {s.signal:<14}  "
                f"beat_rate={s.beat_rate_pct:.0f}%({s.quarters_analyzed}q)  "
                f"avg_surprise={s.avg_eps_surprise_pct:+.1f}%"
                f"{whisper_str}{trend_str}{rev_str}"
            )
        table = "\n".join(rows)

        whisper_block = f"""
<whisper_context>
Earnings Whisper vs. Consensus Gap (historical beat pattern + consensus revision trend):
{wc.summary}

Per-ticker whisper signals ({len(relevant)} shown):
  (beat_rate = historical % quarters beating consensus; avg_surprise = mean EPS surprise %;
   implied_whisper = consensus × (1 + avg_surprise); trend = consensus revision direction)
{table}

Signal classification:
  BEAT_LIKELY:   beat_rate ≥ 75% + avg_surprise ≥ 3.0% + consensus not being revised down
  BEAT_POSSIBLE: beat_rate ≥ 60% or avg_surprise ≥ 1.5% or consensus REVISING_UP
  NEUTRAL:       mixed signals, insufficient data, or first-time reporters
  MISS_POSSIBLE: beat_rate < 45% or avg_surprise ≤ -1.0% or consensus REVISING_DOWN
  MISS_LIKELY:   beat_rate < 30% or avg_surprise ≤ -3.0%

Key principle: "beats consensus but misses the whisper" → stock sells off.
Avoid chasing pre-earnings longs on NEUTRAL/MISS tickers even if news is bullish.
</whisper_context>
"""

        upcoming_with_whisper = [
            s for s in wc.signals if s.days_until_earnings is not None and s.days_until_earnings <= 7
        ]
        beat_likely_str    = ", ".join(s.ticker for s in wc.signals if s.signal == "BEAT_LIKELY")   or "none"
        miss_likely_str    = ", ".join(s.ticker for s in wc.signals if s.signal == "MISS_LIKELY")   or "none"
        imminent_str       = ", ".join(
            f"{s.ticker}({s.days_until_earnings}d, {s.signal})"
            for s in upcoming_with_whisper
        ) or "none"

        whisper_instructions = f"""
21. Earnings Whisper overlay (implied whisper vs. consensus — use with earnings calendar):
    BEAT_LIKELY tickers: {beat_likely_str}.
    MISS_LIKELY tickers: {miss_likely_str}.
    Imminent earnings (≤7d) with whisper signal: {imminent_str}.

    Core principle: the whisper IS the bar. The market prices the whisper, not the consensus.
    A stock that beats the printed consensus by $0.02 but misses the whisper by $0.05 will sell off.
    Use the whisper signal to adjust pre-earnings confidence:

    - BEAT_LIKELY (beat_rate ≥ 75%, avg_surprise ≥ 3%, not being revised down):
      * This company has a well-established pattern of exceeding estimates.
      * If other signals (technical, news, insider) are positive → mild confidence boost (+0.03) for pre-earnings BUY.
      * BUT this pattern is also PRICED IN — the stock must beat the implied whisper, not just consensus.
        Tag as SWING only (earnings day gap risk), note "whisper is ${avg_surprise}% above consensus."
    - BEAT_POSSIBLE: constructive setup. No confidence change vs. base case, but note "beat history" in rationale.
    - NEUTRAL: no whisper edge. Treat as binary event; follow earnings_calendar caution rules above.
    - MISS_POSSIBLE (beat_rate < 45% or consensus being revised down):
      * Apply a confidence haircut (-0.03) on pre-earnings BUY calls.
      * Consensus may already be pessimistic, but if the company's beat rate is low, the market doesn't trust it.
    - MISS_LIKELY: significant caution flag pre-earnings.
      * Do NOT open new pre-earnings longs on a MISS_LIKELY ticker. The company has a pattern of disappointing.
      * If already bearish on the name for other reasons: note "low historical beat rate" as a confirming factor.
    - POST-EARNINGS (ticker reported, Surprise(%) now visible in news signal):
      * The post-earnings price reaction is more important than the EPS beat/miss.
        If a BEAT_LIKELY stock beats and still falls → whisper was higher → distribute, don't add.
        If a MISS_POSSIBLE stock misses but holds → bottom-fishing signal; wait for stabilisation.
    - Whisper signal is a CONFIRMING layer for pre-earnings setups. Never the only reason to BUY/SELL.
      Combine with: earnings calendar timing, news sentiment, technical, insider buying."""

    # Build insider cluster instruction
    cluster_instruction = ""
    if use_insider:
        cluster_tickers = [s.ticker for s in signals_for_claude if s.insider_cluster_detected]
        if cluster_tickers:
            cluster_str = ", ".join(cluster_tickers)
            cluster_instruction = f"""
22. Insider cluster detection (VERY HIGH CONVICTION signal):
    Cluster detected on: {cluster_str}.
    A cluster = ≥3 DIFFERENT corporate insiders or politicians independently buying the same
    stock within a 5-day window. This is the highest-conviction insider signal available:
    - Single insider buy: informative but could reflect personal diversification or scheduled plan.
    - Cluster of 3+ insiders: independent simultaneous conviction from multiple senior executives,
      directors, and/or politicians — each with full visibility into the business.
      Insiders cannot legally coordinate; simultaneous buying is genuinely independent signal.
    - Historically, insider clusters precede significant positive re-ratings (earnings beats, M&A).

    How to apply:
    - Cluster ticker + positive news/technical signals → STRONG conviction BUY candidate.
      Upgrade confidence by +0.05-0.08. These are among the highest win-rate historical setups.
    - Cluster ticker + mixed/neutral signals → WATCH with high priority; flag "insider cluster —
      await technical/news confirmation" in the rationale.
    - Cluster ticker + negative external signals → Note the conflict explicitly.
      Insiders may know something the market does not. Flag as
      "insider cluster vs. negative news — monitor for insider thesis to play out."
      Do NOT let negative news alone override the cluster signal.
    - The insider_score for cluster tickers already has the 1.75× amplifier applied.
    - A cluster on its own, with no other confirming signal, is NOT sufficient for a BUY —
      it is a very strong WATCH / monitor signal. Two or more methods agreeing (cluster + news
      or cluster + technical) qualifies for BUY."""

    # Build OpEx calendar context block and instruction
    opex_block        = ""
    opex_instructions = ""
    if opex_context:
        ox = opex_context
        tc_str = "TRIPLE WITCHING" if ox.is_triple_witching else "standard"
        signal_colors = {
            "OPEX_DAY":             "CRITICAL — peak pinning",
            "OPEX_IMMINENT":        "HIGH — peak pinning tomorrow",
            "TRIPLE_WITCHING_WEEK": "HIGH — highest-intensity OpEx week",
            "OPEX_WEEK":            "MODERATE — pinning in effect",
            "POST_OPEX":            "NOTE — pin released, directional moves expected",
            "NEUTRAL":              "LOW — no OpEx effect",
        }
        effect_label = signal_colors.get(ox.signal, "")

        opex_block = f"""
<opex_context>
Options Expiration Calendar ({tc_str}):
Signal: {ox.signal}  —  {effect_label}
{ox.summary}

Key dates:
  Previous OpEx: {ox.prev_opex}  ({ox.days_since_prev_opex}d ago)
  Next OpEx:     {ox.next_opex}  ({ox.days_to_opex}d from today)
  OpEx week:     {ox.opex_week_monday} → {ox.next_opex}
  Is Triple Witching (Mar/Jun/Sep/Dec): {ox.is_triple_witching}

Effect on max pain gravity:
  OPEX_DAY / OPEX_IMMINENT:        max_pain_score is at maximum reliability — strongest pin force
  TRIPLE_WITCHING_WEEK:             max_pain_score reliability elevated — stock + index options + futures all expiring
  OPEX_WEEK:                        max_pain_score moderately elevated vs. non-OpEx weeks
  POST_OPEX:                        max_pain gravity weak — new OI cycle just starting; discount max_pain_score
  NEUTRAL (>5d from OpEx):          max_pain gravity at baseline — follow normal signal weighting
</opex_context>
"""

        opex_instructions = f"""
23. Options Expiration (OpEx) calendar overlay:
    Current signal: {ox.signal} — next OpEx {ox.next_opex} ({ox.days_to_opex}d away, {tc_str}).
    {ox.summary}

    OpEx creates a structural force that temporarily overrides normal supply/demand:
    Market makers delta-hedge their short-gamma books, which mathematically pulls price
    toward the strike that minimises their payout (max pain). This is a well-documented
    mechanical effect, strongest in the final 1-3 days before expiry.

    How to apply (modifies max_pain_score confidence and time horizon):
    - OPEX_DAY (today is expiry):
      * Max pain gravity is at peak strength. Do NOT fight it.
      * If price is near max pain: the pin will likely hold through close; avoid directional BUY/SELL into the pin.
      * If price is far from max pain: strong pull toward max pain into close.
      * Tag any directional call as "into OpEx — post-close resolution expected."
    - OPEX_IMMINENT (1 day away):
      * Near-peak pinning pressure. Same logic as OPEX_DAY but with slightly less force.
      * Upgrade max_pain_score weight by +0.05 when making directional calls.
    - TRIPLE_WITCHING_WEEK / OPEX_WEEK:
      * Max pain gravity is elevated vs. non-OpEx weeks — mildly upgrade max_pain_score weight (+0.03).
      * Expect suppressed intraday moves on broad market ETFs (SPY, QQQ, IWM) toward max pain.
      * TRIPLE_WITCHING adds stock index futures and index options — vol/volume is significantly higher.
        For broad market SWING calls during triple witching, note the vol spike risk explicitly.
    - POST_OPEX (1–5 days after expiry):
      * The pin has been released. Open interest is being rebuilt — max pain is not yet meaningful.
      * Discount max_pain_score contributions for this period. Directional moves are more reliable.
      * A large post-OpEx move that was suppressed during OpEx week often completes now.
      * This is a period where technical and news signals are MORE reliable relative to options-structure.
    - NEUTRAL (no OpEx effect):
      * Standard analysis. No OpEx modifier. Follow normal signal weighting.

    Important: OpEx is a TIMING and MAGNITUDE modifier, not a directional signal.
    It does not tell you which way to trade — it tells you HOW MUCH to trust max pain gravity.
    Never use OpEx alone to initiate a BUY or SELL. It amplifies or discounts the max_pain_score
    sub-component only."""

    # Build seasonality context block and instruction
    seasonality_block        = ""
    seasonality_instructions = ""
    if seasonality_context:
        sc = seasonality_context
        dir_arrow = {"BULLISH": "▲", "BEARISH": "▼", "NEUTRAL": "→"}.get(sc.monthly_bias, "→")

        effects_str = ""
        if sc.active_effects:
            effect_lines = "\n".join(
                f"  + {e.name} ({e.direction}) — {e.assets_affected}: {e.description}"
                for e in sc.active_effects
            )
            effects_str = f"\nActive calendar windows:\n{effect_lines}\n"

        seasonality_block = f"""
<seasonality_context>
Seasonal Calendar — {sc.month_name} {sc.today.year}  (Q{sc.quarter})
Monthly bias: {sc.monthly_bias} ({sc.monthly_signal})  {dir_arrow}
{sc.monthly_description}
{effects_str}
Composite seasonal signal: {sc.composite_signal}  (direction: {sc.composite_direction})
{sc.summary}

Seasonal signal guide:
  STRONG_TAILWIND  — monthly bias + 2+ active window effects all bullish
  TAILWIND         — monthly bias or active effect net bullish
  NEUTRAL          — seasonal forces balanced or absent
  HEADWIND         — monthly bias or active effect net bearish
  STRONG_HEADWIND  — monthly bias + 2+ active window effects all bearish

Common seasonal patterns:
  End-of-month / Start-of-month (last/first 3 calendar days): pension + 401k rebalancing bid
  Quarter-end window dressing  (last 5 calendar days of Mar/Jun/Sep/Dec): fund managers buy winners
  January effect               (Jan 1–15): small-caps (IWM) outperform large-caps (SPY)
  April: historically strongest month (+2–3% avg S&P 500)
  September: historically weakest month (−1% to −1.5% avg)
  May–October: underperforms November–April by ~6–7pp annualised ("Sell in May")
</seasonality_context>
"""

        headwind_clause = ""
        tailwind_clause = ""
        if sc.composite_direction == "BULLISH":
            tailwind_clause = (
                f"\n    Seasonal {sc.composite_signal}: {sc.summary} "
                f"Use this as a mild confirming tail-wind for BUY calls — never as a standalone trigger."
            )
        elif sc.composite_direction == "BEARISH":
            headwind_clause = (
                f"\n    Seasonal {sc.composite_signal}: {sc.summary} "
                f"Apply a mild confidence haircut (−0.02 to −0.03) on POSITION-horizon BUY calls when "
                f"seasonality and other signals disagree. Strong individual catalysts still override seasonal bias."
            )

        effect_tickers = ""
        if sc.in_january_effect:
            effect_tickers = "\n    January Effect active (day 1–15): IWM / small-cap ETFs historically outperform SPY. Mildly upgrade BUY conviction on small-cap names with otherwise positive signals."
        if sc.in_quarter_end_window:
            effect_tickers += f"\n    Quarter-end window dressing (Q{sc.quarter} ends in {sc.today.strftime('%B')}): expect cosmetic buying of YTD winners (broad ETFs, large-cap leaders). Avoid reading this as fundamental momentum — it reverses shortly after quarter-close."
        if sc.in_quarter_start_window:
            effect_tickers += f"\n    New-quarter institutional flows (Q{sc.quarter} started recently): fresh pension/fund allocations being deployed. Mild systematic bid on broad market ETFs."
        if sc.in_month_end_window or sc.in_month_start_window:
            effect_tickers += "\n    Month-end/start rebalancing window: pension and 401k systematic flows active. Broad equity bid likely — not fundamental; fades quickly."

        seasonality_instructions = f"""
24. Seasonal calendar overlay (pure date-math, no API — apply to broad market and sector ETFs):
    Today: {sc.month_name} {sc.today.day}, {sc.today.year}  |  Monthly bias: {sc.monthly_bias} ({sc.monthly_signal})  |  Composite: {sc.composite_signal}.{tailwind_clause}{headwind_clause}{effect_tickers}

    Core rules for using seasonality:
    - Seasonality is a WEAK secondary overlay. It shifts probability slightly but NEVER overrides a
      strong company-level or macro-level thesis. A MISS_LIKELY whisper + negative news + bearish
      technical is still a SELL even in April (historically bullish month).
    - Use it as a tie-breaker: when two similarly-scored tickers compete for a slot, prefer the one
      with seasonal tailwind. When a BUY is borderline (confidence 0.75-0.80), a TAILWIND is a mild
      additional reason to cross the threshold.
    - When seasonality CONTRADICTS the signal direction, note it explicitly in the rationale as a risk
      factor: "seasonal headwind (September) — monitor for follow-through."
    - Monthly bias is a population-level average across ~95 years. Any single year can deviate sharply.
      Do not cite seasonal statistics as a primary catalyst — always pair with a current specific reason.
    - Quarter-end window dressing and month-end rebalancing are short-lived mechanical effects (1-5 days).
      Do not use them to justify POSITION-horizon calls — only SWING or SHORT-TERM at most."""

    # Build bond market internals context block and instruction
    bond_block        = ""
    bond_instructions = ""
    if bond_internals_context:
        bi = bond_internals_context
        dir_arrow = {"BULLISH": "▲", "BEARISH": "▼", "NEUTRAL": "→"}.get(bi.direction, "→")

        def _bfmt(v):
            return f"{v:+.2f}%" if v is not None else "N/A"

        def _yfmt(v):
            return f"{v:.2f}%" if v is not None else "N/A"

        curve_line = ""
        if bi.spread_10y_3m is not None:
            curve_line = (
                f"  10Y-3M spread: {bi.spread_10y_3m:+.2f}pp  →  {bi.curve_signal}\n"
                f"  (10Y={_yfmt(bi.yield_10y)}  3M={_yfmt(bi.yield_3m)}  "
                f"5Y={_yfmt(bi.yield_5y)}  30Y={_yfmt(bi.yield_30y)})\n"
            )

        tlt_line = ""
        if bi.tlt_return_20d is not None:
            tlt_line = (
                f"  TLT 1w={_bfmt(bi.tlt_return_5d)}  4w={_bfmt(bi.tlt_return_20d)}  "
                f"8w={_bfmt(bi.tlt_return_40d)}  →  {bi.tlt_signal}\n"
            )

        dur_line = ""
        if bi.tlt_ief_spread_5d is not None:
            dur_line = f"  Duration (TLT−IEF, 5d): {bi.tlt_ief_spread_5d:+.2f}pp  →  {bi.tlt_ief_signal}\n"

        real_line = ""
        if bi.tip_ief_spread_5d is not None:
            real_line = f"  Real yield (TIP−IEF, 5d): {bi.tip_ief_spread_5d:+.2f}pp  →  {bi.real_yield_signal}\n"

        ig_line = ""
        if bi.lqd_tlt_spread_5d is not None:
            ig_line = f"  IG credit (LQD−TLT, 5d): {bi.lqd_tlt_spread_5d:+.2f}pp  →  {bi.ig_credit_signal}\n"

        be_line = ""
        if bi.tlt_spy_div_5d is not None and bi.bond_equity_signal != "NEUTRAL":
            ief_conf = ""
            if bi.ief_spy_div_5d is not None:
                ief_conf = f" (IEF−SPY={bi.ief_spy_div_5d:+.2f}pp confirms)"
            be_line = (
                f"  Bond-equity divergence (TLT−SPY, 5d): {bi.tlt_spy_div_5d:+.2f}pp{ief_conf}"
                f"  →  {bi.bond_equity_signal}  ({bi.bond_equity_direction})\n"
            )

        bond_block = f"""
<bond_internals_context>
Bond Market Internals: {bi.regime}  {dir_arrow} {bi.direction}
{bi.summary}

{curve_line}{tlt_line}{dur_line}{real_line}{ig_line}{be_line}
Yield curve signal guide (10Y-3M — best recession predictor):
  DEEPLY_INVERTED (<−0.75pp): strong recession warning (6–18 months lead time, ~95% historical accuracy)
  INVERTED (−0.75 to −0.10pp): recession risk elevated; avoid POSITION-horizon cyclical longs
  FLAT (−0.10 to +0.50pp): late-cycle; reduced conviction on duration-sensitive names
  NORMAL (+0.50 to +1.50pp): standard macro backdrop
  STEEP (>+1.50pp): early expansion or reflation; constructive for cyclicals/banks

TLT signal guide (20-day return):
  RALLYING_STRONG (>+3%):   rates falling sharply — tailwind for growth tech, REITs, long-duration
  RALLYING        (+1–3%):  mild rate tailwind
  FLAT            (±1%):    no rate pressure either way
  FALLING         (−1–3%):  mild headwind for rate-sensitive names
  FALLING_STRONG  (<−3%):   rates rising sharply — significant headwind for high-multiple/duration assets

IG credit signal guide (LQD vs TLT, 5-day):
  IG_STRESS   (<−0.30pp): corporate spreads widening vs Treasuries → leads equity weakness 1–5 days
  IG_CAUTION  (<−0.10pp): mild corporate spread widening → slight caution
  NEUTRAL     (±0.10pp):  no credit signal
  IG_STRONG   (>+0.10pp): corporate spreads tightening → risk-on confirmation

Bond-equity divergence guide (TLT−SPY, 5-day):
  EQUITY_CATCHUP_LIKELY   (TLT≥+2.5%, SPY flat): bonds pricing rate cuts; equities lagging — catch-up rally expected
  EQUITY_CATCHUP_POSSIBLE (TLT≥+1.5%, SPY flat): milder version of the above — modest bullish lean for equities
  SYNCHRONIZED_RISK_ON    (both TLT and SPY rallying): unusual but occurs at dovish pivots — constructive
  NEUTRAL                 : bonds and equities broadly in sync — no actionable divergence
  EQUITY_SELLOFF_RISK     (TLT≤−2.0%, SPY flat): rate headwind not yet priced into stocks — weakness likely
  SYNCHRONIZED_RISK_OFF   (both TLT and SPY selling off): broad market de-risking underway

Regime guide:
  RISK_ON:       bond market broadly constructive for equities
  CONSTRUCTIVE:  mild bond tailwind
  NEUTRAL:       no bond market override
  DEFENSIVE:     some headwinds; mild caution on POSITION-horizon longs
  RISK_OFF:      multiple bond warning signals; avoid aggressive new longs
  REFLATIONARY:  TLT falling + inflation expectations rising → commodities/cyclicals over growth tech
</bond_internals_context>
"""

        # Build the instruction
        regime_clause = ""
        if bi.regime == "RISK_OFF":
            regime_clause = (
                f"\n    RISK_OFF: Multiple bond market warning signals active. "
                f"Apply a confidence haircut (−0.05) on POSITION-horizon BUY calls across the board. "
                f"Particularly avoid long-duration names (high-growth tech, REITs) and cyclicals."
            )
        elif bi.regime == "DEFENSIVE":
            regime_clause = (
                f"\n    DEFENSIVE: Bond headwinds present. Apply a mild confidence haircut (−0.03) "
                f"on new POSITION-horizon BUY calls. Prefer SWING or SHORT-TERM horizons."
            )
        elif bi.regime in ("RISK_ON", "CONSTRUCTIVE"):
            regime_clause = (
                f"\n    {bi.regime}: Bond market constructive. Mild confidence boost (+0.02–0.03) "
                f"on BUY calls with other confirming signals. Risk-on environment supports equities."
            )
        elif bi.regime == "REFLATIONARY":
            regime_clause = (
                f"\n    REFLATIONARY: TLT falling + inflation expectations rising. "
                f"Favour commodities (GLD, CPER, energy ETFs) and cyclicals over long-duration tech. "
                f"Apply mild confidence haircut on high-multiple growth tech and REIT BUY calls."
            )

        curve_clause = ""
        if bi.curve_signal == "DEEPLY_INVERTED":
            curve_clause = (
                f"\n    DEEPLY INVERTED 10Y-3M ({bi.spread_10y_3m:+.2f}pp): "
                f"Historical recession signal with ~95% accuracy. Avoid POSITION-horizon longs on "
                f"cyclical sectors. Prefer defensives (XLP, XLV, XLU) and gold. "
                f"Do NOT extend time horizons on speculative names."
            )
        elif bi.curve_signal == "INVERTED":
            curve_clause = (
                f"\n    INVERTED 10Y-3M ({bi.spread_10y_3m:+.2f}pp): "
                f"Recession risk elevated. Cap time horizons at SHORT-TERM for cyclical longs."
            )

        ig_clause = ""
        if bi.ig_credit_signal == "IG_STRESS":
            ig_clause = (
                f"\n    IG_STRESS (LQD−TLT 5d={bi.lqd_tlt_spread_5d:+.2f}pp): "
                f"Investment-grade spreads widening vs Treasuries — corporate stress leading equities. "
                f"Reduce BUY conviction on broad market ETFs; upgrade SELL conviction when other signals align. "
                f"Credit leads equities by 1–5 days — this is an early warning, act ahead of equities."
            )

        tlt_clause = ""
        if bi.tlt_signal == "FALLING_STRONG":
            tlt_clause = (
                f"\n    TLT FALLING STRONG (4w={bi.tlt_return_20d:+.1f}%): "
                f"Long rates rising sharply. Significant headwind for: growth tech (XLK), REITs (XLRE), "
                f"high P/E names. Boost BUY conviction on financials (XLF — NIM expansion) and "
                f"short-duration value stocks. Discount BUY calls on rate-sensitive names."
            )
        elif bi.tlt_signal == "RALLYING_STRONG":
            tlt_clause = (
                f"\n    TLT RALLYING STRONG (4w={bi.tlt_return_20d:+.1f}%): "
                f"Long rates falling sharply. Tailwind for: growth tech, REITs, long-duration assets. "
                f"Mild confidence boost (+0.03) on BUY calls for rate-sensitive names with other support."
            )

        be_clause = ""
        if bi.bond_equity_signal == "EQUITY_CATCHUP_LIKELY":
            be_clause = (
                f"\n    EQUITY_CATCHUP_LIKELY (TLT 5d={bi.tlt_return_5d:+.1f}%, "
                f"SPY 5d={bi.spy_return_5d:+.1f}%, TLT−SPY div={bi.tlt_spy_div_5d:+.2f}pp): "
                f"Bonds have rallied hard while equities held. Bond market is pricing in rate cuts or "
                f"growth deceleration that equities have not yet reflected. Apply a mild confidence "
                f"boost (+0.03) on broad-market ETF and rate-sensitive BUY calls. "
                f"This is a 1–2 week leading signal — prioritise SWING over POSITION horizon."
            )
        elif bi.bond_equity_signal == "EQUITY_CATCHUP_POSSIBLE":
            be_clause = (
                f"\n    EQUITY_CATCHUP_POSSIBLE (TLT 5d={bi.tlt_return_5d:+.1f}%, "
                f"SPY 5d={bi.spy_return_5d:+.1f}%): "
                f"Mild bond outperformance with equities flat — modest bullish lean. "
                f"Not strong enough to upgrade conviction on its own; use as a confirming signal."
            )
        elif bi.bond_equity_signal == "EQUITY_SELLOFF_RISK":
            be_clause = (
                f"\n    EQUITY_SELLOFF_RISK (TLT 5d={bi.tlt_return_5d:+.1f}%, "
                f"SPY 5d={bi.spy_return_5d:+.1f}%, TLT−SPY div={bi.tlt_spy_div_5d:+.2f}pp): "
                f"Bonds selling off while equities hold — rising rate headwind not yet priced into stocks. "
                f"Apply a mild confidence haircut (−0.03) on BUY calls for rate-sensitive names. "
                f"Discount POSITION-horizon longs on XLK, XLRE, and high-multiple names. "
                f"Watch for equity weakness to follow within 1–2 weeks."
            )
        elif bi.bond_equity_signal == "SYNCHRONIZED_RISK_OFF":
            be_clause = (
                f"\n    SYNCHRONIZED_RISK_OFF (TLT 5d={bi.tlt_return_5d:+.1f}%, "
                f"SPY 5d={bi.spy_return_5d:+.1f}%): "
                f"Both bonds and equities selling off — broad de-risking underway. "
                f"Avoid new long positions; prioritise SELL/HOLD calls."
            )

        bond_instructions = f"""
25. Bond market internals overlay (1–8 week macro regime from Treasury/credit ETFs):
    Current regime: {bi.regime} ({bi.direction}).{regime_clause}{curve_clause}{ig_clause}{tlt_clause}{be_clause}

    Core principles:
    - Bond market internals COMPLEMENT FRED data (which gives levels) with price momentum.
      FRED tells you the IG spread is 1.2%; bond internals tell you it widened 20bp this week.
      Both are needed: levels for calibration, momentum for timing.
    - The 10Y-3M spread is a BETTER recession predictor than the 2Y-10Y (which FRED provides).
      The 3M bill prices the expected average Fed funds rate over the next 90 days —
      inverting 10Y-3M means the market expects cuts soon = the economy is already slowing.
    - IG credit leads equities by 1–5 days. When IG_STRESS or IG_CAUTION: watch for equity
      weakness to follow even if equities haven't moved yet.
    - Bond-equity divergence is a 1–2 week LEADING signal: bonds typically price macro shifts
      before equities react. EQUITY_CATCHUP_LIKELY/POSSIBLE = equities lagging bonds = bullish lag.
      EQUITY_SELLOFF_RISK = bonds already pricing rate pain = equity weakness building.
    - TLT momentum affects specific SECTORS differentially:
      • Falling TLT (rising rates): HEADWIND for XLK/XLRE/high-P/E; TAILWIND for XLF
      • Rallying TLT (falling rates): TAILWIND for XLK/XLRE; HEADWIND for XLF
    - Reflationary regime: when TLT is falling AND inflation expectations are rising
      (TIP outperforming IEF), commodities and real assets benefit; long-duration tech faces
      both a rate headwind AND an inflation headwind (from multiple compression + cost pressures).
    - Bond internals is a MEDIUM-TERM signal (weeks). It does not override near-term catalysts
      like earnings beats, M&A announcements, or strong earnings surprises.
    - Convergence check: bond internals + FRED regime + credit (HYG/SPY) + VIX all pointing
      the same direction = highest conviction for broad-market-level calls."""

    # Build VWAP instructions
    vwap_instructions = ""
    if settings.enable_vwap and settings.enable_fetch_data:
        vwap_instructions = """
14. VWAP distance overlay — mean-reversion signal:
    Each ticker's VWAP_score ∈ [-1, +1] measures how far price sits from its rolling 20-day VWAP,
    normalised by the security's own typical deviation (z-score / 2). Interpretation:
    - VWAP_score > 0: price is BELOW VWAP → institutions seeking VWAP see value; mean-reversion BUY bias.
    - VWAP_score < 0: price is ABOVE VWAP → institutions selling to unwind into VWAP; mean-reversion SELL bias.
    - Raw distance shown as e.g. "+3.2% from VWAP" — how far price has stretched.

    Using VWAP as a modifier (not a standalone signal):
    - A strong VWAP_score (|score| ≥ 0.5) in the SAME direction as the combined signal: mild confidence boost.
      Example: BULLISH combined + VWAP_score = +0.70 → price stretched below VWAP; institutions likely to buy.
    - A strong VWAP_score CONTRADICTING the combined direction: note the conflict.
      Example: BULLISH combined + VWAP_score = -0.80 → price already far above VWAP; a rally chase from here
      has poor risk/reward. Prefer WATCH or shorten horizon to SWING.
    - VWAP pull is weaker in strong trending markets (momentum can override mean-reversion gravity).
      If other signals strongly point to a breakout (AMPLIFIED GEX + positive news + insider buying),
      de-emphasise the VWAP headwind and acknowledge in the rationale.
    - Near-zero VWAP_score (|score| < 0.15): price trading near VWAP; no reversion pressure. No VWAP override."""

    commodity_tickers = ", ".join(settings.commodities_list) or "GLD, SLV, IAU, GDX, PPLT, PALL, CPER"

    prompt = f"""You are an elite portfolio manager with a verified 30-year track record of market-beating returns. You combine the analytical precision of a quant, the pattern recognition of a seasoned discretionary trader, and the macro intuition of a global macro fund manager. You have studied every major market cycle since 1990 and have an exceptional ability to identify when multiple independent evidence layers converge on the same directional call — these are the moments of highest expected value.

Your defining edge: you are ruthlessly disciplined about false positives. You understand that a wrong BUY or SELL costs capital that cannot be recovered. You output HOLD or WATCH whenever the evidence is mixed, incomplete, or driven by a single source. When you do issue a BUY or SELL, it is because the convergence of evidence makes the directional call highly reliable — and you explain precisely why.

Signal sources available today: {methods_desc}

Today's date: {fmt_et(now_et())}
{macro_block}{macro_surprise_block}{fedwatch_block}{bond_block}{revision_block}{cot_block}{ipo_block}{vix_block}{move_block}{global_macro_block}{credit_block}{pc_block}{tick_block}{breadth_block}{highs_lows_block}{mcclellan_block}{whisper_block}{earnings_block}{gex_block}{opex_block}{seasonality_block}
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
{insider_instructions}{macro_instructions}{macro_surprise_instructions}{fedwatch_instructions}{bond_instructions}{revision_instructions}{cot_instructions}{ipo_instructions}{vix_instructions}{move_instructions}{global_macro_instructions}{credit_instructions}{pc_instructions}{tick_instructions}{breadth_instructions}{highs_lows_instructions}{mcclellan_instructions}{whisper_instructions}{earnings_instructions}{gex_instructions}{vwap_instructions}{cluster_instruction}{opex_instructions}{seasonality_instructions}
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
