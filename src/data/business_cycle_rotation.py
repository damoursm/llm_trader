"""Business Cycle Sector Rotation — structural economic phase → sector leadership biases.

Derives the current business cycle phase from already-fetched FRED macro context
(no new API calls), then maps it to per-sector expected leadership scores using the
Fidelity-style historical rotation model.

Business cycle phases (and typical sector leadership):
  EARLY_EXPANSION  — rate cuts just passed, unemployment falling, inflation low
                     Leaders: XLF, XLRE, XLY, XLK
  MID_EXPANSION    — unemployment near cycle lows, inflation moderate, curve normal
                     Leaders: XLK, XLI, XLY, XLC
  LATE_EXPANSION   — inflation rising, curve flattening, momentum still positive
                     Leaders: XLE, XLB, XLI
  LATE_CYCLE       — curve flat/inverted, credit widening, defensives rotate in
                     Leaders: XLV, XLP, XLU
  CONTRACTION      — recession in progress, unemployment rising
                     Leaders: XLV, XLP, XLU (maximum defensiveness)
  UNKNOWN          — insufficient macro data to classify

Why this complements the other rotation layers:
  sector_rotation.py   — reactive: WHERE money IS moving NOW (1w/1m/3m momentum)
  rotation_drivers.py  — monetary: WHAT the Fed is doing (rate trajectory + CPI)
  business_cycle_rotation.py (this) — structural: WHERE we are in the economic cycle
                                      (historically repeating 7-10y patterns)

The three layers are distinct but often converge. When all three agree, conviction is
highest. When they diverge (e.g., Ebb-and-Flow shows money flowing to defensives while
cycle says EARLY_EXPANSION), it flags a potential anomaly worth noting.
"""

from datetime import date
from loguru import logger

from src.models import BusinessCycleContext, SectorCycleBias


# ── Sector map: ETF → human-readable name ────────────────────────────────────
_SECTOR_NAMES = {
    "XLK":  "Technology",
    "XLF":  "Financials",
    "XLY":  "Consumer Disc.",
    "XLC":  "Comms Services",
    "XLI":  "Industrials",
    "XLB":  "Materials",
    "XLE":  "Energy",
    "XLRE": "Real Estate",
    "XLU":  "Utilities",
    "XLV":  "Healthcare",
    "XLP":  "Consumer Staples",
}

# ── Per-phase leadership scores ∈ [-1, +1] ───────────────────────────────────
# Based on Fidelity / PIMCO business cycle sector rotation research.
# Positive = historically outperforms SPY in this phase.
# Negative = historically underperforms.
_PHASE_SCORES: dict[str, dict[str, float]] = {
    "EARLY_EXPANSION": {
        "XLF":  +0.80,  # financials recover fastest — credit normalises
        "XLRE": +0.70,  # real estate: rate cuts boost valuations
        "XLY":  +0.65,  # consumer discretionary: pent-up demand unleashed
        "XLK":  +0.50,  # tech benefits from falling rates (multiple expansion)
        "XLC":  +0.40,  # comms services: growth re-rating
        "XLI":  +0.30,  # industrials: early capex pickup
        "XLB":  +0.20,  # materials: nascent demand recovery
        "XLE":  +0.10,  # energy: demand starting to recover
        "XLV":  +0.00,  # healthcare: neutral — defensive but less exciting
        "XLP":  -0.20,  # staples: lags in risk-on environment
        "XLU":  -0.30,  # utilities: rate-sensitive but now rates are falling — held back
    },
    "MID_EXPANSION": {
        "XLK":  +0.80,  # tech: earnings leverage to growth, moderate rates
        "XLI":  +0.65,  # industrials: capex and hiring in full swing
        "XLY":  +0.50,  # consumer discretionary: peak employment = peak spending
        "XLC":  +0.45,  # comms: growth monetisation
        "XLF":  +0.35,  # financials: steady — curve normal, credit healthy
        "XLB":  +0.25,  # materials: construction + manufacturing demand
        "XLE":  +0.20,  # energy: moderate demand, not yet late-cycle surge
        "XLRE": +0.10,  # real estate: positive but rates normalising
        "XLV":  +0.05,  # healthcare: stable, not a leader here
        "XLP":  -0.25,  # staples: lags risk-on
        "XLU":  -0.35,  # utilities: rate headwinds as curve normalises
    },
    "LATE_EXPANSION": {
        "XLE":  +0.80,  # energy: commodity inflation + peak demand
        "XLB":  +0.65,  # materials: inflation leaders; capacity tightening
        "XLI":  +0.45,  # industrials: still building, but slowing momentum
        "XLV":  +0.30,  # healthcare: early defensive rotation begins
        "XLC":  +0.10,  # comms: slowing growth
        "XLF":  +0.00,  # financials: curve flattening = margin pressure
        "XLP":  +0.10,  # staples: early rotation to defensives
        "XLK":  -0.15,  # tech: rate headwinds, multiple compression begins
        "XLY":  -0.20,  # consumer disc.: consumers starting to feel inflation
        "XLU":  +0.20,  # utilities: safety bid begins
        "XLRE": -0.30,  # real estate: worst of rate pressure
    },
    "LATE_CYCLE": {
        "XLV":  +0.80,  # healthcare: recession-resistant earnings
        "XLP":  +0.70,  # staples: demand inelastic; dividend yield safety
        "XLU":  +0.60,  # utilities: regulated earnings; yield > bonds narrows
        "XLE":  +0.25,  # energy: still elevated but plateauing
        "XLB":  +0.10,  # materials: mixed — supply still tight
        "XLC":  -0.10,  # comms: ad spend slowing
        "XLI":  -0.20,  # industrials: leading indicator for slowdown
        "XLF":  -0.35,  # financials: inverted curve = NIM compression
        "XLRE": -0.45,  # real estate: high rates = severe pressure
        "XLY":  -0.55,  # consumer disc.: demand destruction; credit stress
        "XLK":  -0.45,  # tech: elevated rates = multiple contraction
    },
    "CONTRACTION": {
        "XLV":  +0.65,  # healthcare: demand inelastic; budget cuts slow
        "XLP":  +0.75,  # staples: maximum defensiveness
        "XLU":  +0.60,  # utilities: dividends as bond proxy when rates fall
        "XLE":  -0.30,  # energy: demand destruction
        "XLB":  -0.45,  # materials: industrial demand collapses
        "XLC":  -0.40,  # comms: ad budgets cut first
        "XLI":  -0.55,  # industrials: capex halted; layoffs
        "XLF":  -0.50,  # financials: credit losses, NPA rise
        "XLRE": -0.70,  # real estate: forced selling + vacancy rise
        "XLY":  -0.65,  # consumer disc.: biggest income-elastic collapse
        "XLK":  -0.50,  # tech: growth premium crashes fastest in recession
    },
    "UNKNOWN": {k: 0.0 for k in _SECTOR_NAMES},
}

# ── Narrative summary per phase ────────────────────────────────────────────────
_PHASE_NARRATIVES = {
    "EARLY_EXPANSION": (
        "Macro regime signals early-cycle recovery: rate cuts are working, unemployment "
        "is declining, and inflation is subdued. Historically this phase rewards financials, "
        "real estate, consumer discretionary, and tech (multiple expansion on falling rates)."
    ),
    "MID_EXPANSION": (
        "Mid-cycle expansion: employment near full, inflation moderate, yield curve healthy. "
        "Technology and industrials historically lead as earnings growth accelerates. "
        "Consumer discretionary benefits from peak employment and consumer confidence."
    ),
    "LATE_EXPANSION": (
        "Late-cycle dynamics: inflation rising, curve flattening, unemployment at lows. "
        "Energy and materials historically outperform on commodity price momentum. "
        "Defensives (healthcare, staples) begin to attract early rotation capital."
    ),
    "LATE_CYCLE": (
        "Late-cycle regime: yield curve flat/inverted, credit spreads widening, leading "
        "indicators softening. Defensives (healthcare, staples, utilities) historically "
        "outperform as risk appetite wanes and growth expectations reset lower."
    ),
    "CONTRACTION": (
        "Recession / contraction: recession risk elevated, unemployment rising, credit "
        "stress building. Consumer staples, healthcare, and utilities provide maximum "
        "defensiveness. Cyclicals (consumer disc., materials, real estate) face the "
        "largest earnings revision risk."
    ),
    "UNKNOWN": (
        "Insufficient macro data to classify the business cycle phase. "
        "No sector rotation bias applied — use other rotation signals."
    ),
}

_PHASE_DIRECTIONS = {
    "EARLY_EXPANSION": "BULLISH",
    "MID_EXPANSION":   "BULLISH",
    "LATE_EXPANSION":  "NEUTRAL",
    "LATE_CYCLE":      "BEARISH",
    "CONTRACTION":     "BEARISH",
    "UNKNOWN":         "NEUTRAL",
}


def _classify_phase(macro_context) -> tuple[str, str]:
    """Derive business cycle phase from FRED macro context.

    Returns (phase, evidence_string).
    """
    if macro_context is None:
        return "UNKNOWN", "no FRED macro context available"

    regime            = getattr(macro_context, "regime",            "UNKNOWN")
    yield_curve       = getattr(macro_context, "yield_curve_signal", "UNKNOWN")
    inflation_signal  = getattr(macro_context, "inflation_signal",   "UNKNOWN")
    unemployment_trend = getattr(macro_context, "unemployment_trend", "STABLE")
    cpi_yoy           = getattr(macro_context, "cpi_yoy",            None)

    evidence_parts = [
        f"regime={regime}",
        f"yield_curve={yield_curve}",
        f"inflation={inflation_signal}",
        f"unemployment={unemployment_trend}",
    ]
    if cpi_yoy is not None:
        evidence_parts.append(f"cpi_yoy={cpi_yoy:+.1f}%")
    evidence = "  |  ".join(evidence_parts)

    # CONTRACTION: FRED explicitly classifies as RECESSION
    if regime == "RECESSION":
        return "CONTRACTION", evidence

    # LATE_CYCLE (FRED label) → Late Cycle unless yield curve steepening = early recovery
    if regime == "LATE_CYCLE":
        if yield_curve == "INVERTED":
            return "LATE_CYCLE", evidence
        # Flat curve but not inverted — could be deep late-expansion
        return "LATE_CYCLE", evidence

    # SLOWDOWN → heading into Late Cycle; treat as Late Expansion unless curve inverted
    if regime == "SLOWDOWN":
        if yield_curve == "INVERTED":
            return "LATE_CYCLE", evidence
        return "LATE_EXPANSION", evidence

    # EXPANSION — sub-classify by inflation + unemployment
    if regime == "EXPANSION":
        is_high_inflation = inflation_signal in ("HIGH", "ELEVATED")
        is_low_inflation  = inflation_signal in ("LOW", "MODERATE")

        if is_high_inflation or unemployment_trend == "RISING":
            # Inflation elevated in expansion → Late Expansion pressure
            return "LATE_EXPANSION", evidence
        if is_low_inflation and unemployment_trend == "FALLING":
            return "EARLY_EXPANSION", evidence
        # Default mid-expansion: stable employment, moderate inflation
        return "MID_EXPANSION", evidence

    return "UNKNOWN", evidence


def _score_to_signal(score: float) -> str:
    if score >= 0.55:
        return "STRONG_LEADER"
    if score >= 0.25:
        return "LEADER"
    if score <= -0.55:
        return "STRONG_LAGGARD"
    if score <= -0.25:
        return "LAGGARD"
    return "NEUTRAL"


def _build_convergence_notes(sector_biases, sector_rotation_context) -> str:
    """Compare cycle leaders vs real-time Ebb-and-Flow inflows."""
    if sector_rotation_context is None:
        return ""

    leaders  = {b.etf for b in sector_biases if b.cycle_score >= 0.40}
    laggards = {b.etf for b in sector_biases if b.cycle_score <= -0.40}

    inflow_etfs  = set(sector_rotation_context.top_inflow[:3])
    outflow_etfs = set(sector_rotation_context.top_outflow[:3])

    agree_bull = leaders  & inflow_etfs
    agree_bear = laggards & outflow_etfs
    contradict = (leaders & outflow_etfs) | (laggards & inflow_etfs)

    parts = []
    if agree_bull:
        parts.append(f"Confirming: real-time inflows into cycle leaders {sorted(agree_bull)}")
    if agree_bear:
        parts.append(f"Confirming: real-time outflows from cycle laggards {sorted(agree_bear)}")
    if contradict:
        parts.append(f"Divergence: cycle model and Ebb-and-Flow disagree on {sorted(contradict)} — flag for review")

    return "  |  ".join(parts) if parts else "Cycle model and Ebb-and-Flow signals are inconclusive"


def compute_business_cycle_context(
    macro_context=None,
    sector_rotation_context=None,
) -> BusinessCycleContext:
    """Derive business cycle phase and sector biases from FRED macro context.

    Pure synthesis — no I/O, no cache, instant.
    Call after the parallel fetch resolves, same timing as compute_market_mode().
    """
    phase, evidence = _classify_phase(macro_context)
    direction       = _PHASE_DIRECTIONS.get(phase, "NEUTRAL")
    score_map       = _PHASE_SCORES.get(phase, _PHASE_SCORES["UNKNOWN"])

    sector_biases = [
        SectorCycleBias(
            etf=etf,
            name=_SECTOR_NAMES[etf],
            cycle_score=round(score_map.get(etf, 0.0), 2),
            cycle_signal=_score_to_signal(score_map.get(etf, 0.0)),
        )
        for etf in _SECTOR_NAMES
    ]
    # Sort descending by score so email / prompt can iterate in order
    sector_biases.sort(key=lambda b: b.cycle_score, reverse=True)

    top_cycle_leaders  = [b.etf for b in sector_biases if b.cycle_score >= 0.40]
    weak_cycle_sectors = [b.etf for b in sector_biases if b.cycle_score <= -0.40]

    convergence_notes = _build_convergence_notes(sector_biases, sector_rotation_context)

    summary = _PHASE_NARRATIVES.get(phase, "Unknown phase.")
    if convergence_notes:
        summary += f"  {convergence_notes}."

    ctx = BusinessCycleContext(
        cycle_phase=phase,
        cycle_direction=direction,
        evidence=evidence,
        sector_biases=sector_biases,
        top_cycle_leaders=top_cycle_leaders,
        weak_cycle_sectors=weak_cycle_sectors,
        convergence_notes=convergence_notes,
        report_date=date.today(),
        summary=summary,
    )

    logger.info(
        f"[business_cycle] Phase={phase}  Direction={direction}  "
        f"Leaders={top_cycle_leaders[:3]}  Laggards={weak_cycle_sectors[:3]}"
    )
    logger.debug(f"[business_cycle] Evidence: {evidence}")

    return ctx
