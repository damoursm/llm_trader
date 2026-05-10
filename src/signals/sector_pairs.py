"""Sector Pairs / Relative Value — market-neutral pair trades from sector divergence.

When a sector ETF is BULLISH but one of its constituent stocks is BEARISH (or vice
versa), two sources of alpha combine:
  1. The ETF captures the sector tailwind (or absorbs the headwind).
  2. The stock captures the idiosyncratic weakness (or resilience).

Going LONG the bullish leg and SHORT the bearish leg strips out broad-market beta
and isolates the relative mispricing between the two.

Setup types
───────────
  ETF_BULL_STOCK_BEAR  — sector trend is up, stock is lagging or deteriorating.
      Example: XLK BULLISH + INTC BEARISH → Long XLK / Short INTC.
      Thesis: sector tailwind not lifting INTC; it likely has a company-specific problem.

  ETF_BEAR_STOCK_BULL  — sector trend is down, stock is holding up or improving.
      Example: XLF BEARISH + GS BULLISH → Long GS / Short XLF.
      Thesis: GS is outperforming in a difficult sector; idiosyncratic catalyst or
      fundamental improvement. Short XLF hedges broad financials weakness.

Signal quality
──────────────
  Both legs must have a non-NEUTRAL direction AND confidence ≥ MIN_LEG_CONFIDENCE.
  The pair_score = (stock_confidence + etf_confidence) / 2 — the harmonic mean
  would be more conservative but average is sufficient given the confidence gate.
  Pairs sorted by pair_score descending; only the top MAX_PAIRS reported.
"""

from typing import Dict, List
from loguru import logger

from src.models import SectorPair, SectorPairsContext, TickerSignal
from src.signals.aggregator import _SECTOR_MAP

MIN_LEG_CONFIDENCE = 0.35   # both legs need at least this confidence to form a pair
MAX_PAIRS          = 10     # cap output to avoid email bloat


def find_sector_pairs(
    signals_by_ticker: Dict[str, TickerSignal],
) -> SectorPairsContext:
    """
    Scan all stocks in _SECTOR_MAP for opposing-direction signals vs their sector ETF.
    Returns a SectorPairsContext with the best divergence pairs sorted by conviction.
    """
    pairs: List[SectorPair] = []

    seen: set = set()  # deduplicate (stock, etf) combinations

    for stock, etf in _SECTOR_MAP.items():
        key = (stock, etf)
        if key in seen:
            continue
        seen.add(key)

        stock_sig = signals_by_ticker.get(stock)
        etf_sig   = signals_by_ticker.get(etf)

        if stock_sig is None or etf_sig is None:
            continue

        # Both legs need a directional view — skip NEUTRAL legs
        if stock_sig.direction == "NEUTRAL" or etf_sig.direction == "NEUTRAL":
            continue

        # Both legs must agree in opposite directions to form a pair
        if stock_sig.direction == etf_sig.direction:
            continue   # aligned — the sector_alignment_factor already handles this

        # Minimum conviction on both legs
        if stock_sig.confidence < MIN_LEG_CONFIDENCE or etf_sig.confidence < MIN_LEG_CONFIDENCE:
            continue

        # Determine which leg is long, which is short
        if etf_sig.direction == "BULLISH":
            # ETF bull + Stock bear → Long ETF / Short Stock
            long_leg, short_leg = etf, stock
            setup_type = "ETF_BULL_STOCK_BEAR"
            rationale = (
                f"Sector {etf} is BULLISH (conf={etf_sig.confidence:.0%}) while "
                f"{stock} lags BEARISH (conf={stock_sig.confidence:.0%}). "
                f"Long {etf} captures sector momentum; Short {stock} captures idiosyncratic weakness. "
                f"Market-neutral: removes {etf} beta from the {stock} short."
            )
        else:
            # ETF bear + Stock bull → Long Stock / Short ETF
            long_leg, short_leg = stock, etf
            setup_type = "ETF_BEAR_STOCK_BULL"
            rationale = (
                f"{stock} is BULLISH (conf={stock_sig.confidence:.0%}) despite "
                f"sector {etf} being BEARISH (conf={etf_sig.confidence:.0%}). "
                f"Long {stock} captures idiosyncratic strength; Short {etf} hedges sector headwind. "
                f"Market-neutral: {etf} short isolates {stock} alpha from sector drag."
            )

        pair_score = round((stock_sig.confidence + etf_sig.confidence) / 2, 3)

        pairs.append(SectorPair(
            stock=stock,
            etf=etf,
            long_leg=long_leg,
            short_leg=short_leg,
            stock_direction=stock_sig.direction,
            etf_direction=etf_sig.direction,
            stock_confidence=stock_sig.confidence,
            etf_confidence=etf_sig.confidence,
            pair_score=pair_score,
            setup_type=setup_type,
            rationale=rationale,
        ))
        logger.info(
            f"[pairs] {setup_type}: Long {long_leg} / Short {short_leg} "
            f"| pair_score={pair_score:.0%}"
        )

    # Sort by conviction descending, cap output
    pairs.sort(key=lambda p: p.pair_score, reverse=True)
    pairs = pairs[:MAX_PAIRS]

    if not pairs:
        summary = "No sector divergence pairs detected — ETFs and constituents are directionally aligned."
    else:
        entries = [f"L {p.long_leg}/S {p.short_leg} ({p.pair_score:.0%})" for p in pairs]
        summary = f"{len(pairs)} pair(s): " + " | ".join(entries)

    logger.info(f"[pairs] {summary}")
    return SectorPairsContext(pairs=pairs, summary=summary)
