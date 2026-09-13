"""MECHANICAL entry selection — the cross-sectional rank rule (2026-09-04).

This replaces the LLM synthesis call as the thing that decides what to trade.

**Why (measured, pre-registered, house standard `/evaluate`).** Over 2026-08-13
→ 09-04 (the current rank-combine era), 1,245 actionable LLM BUY/SELL decisions
across 16 days, per-run and per-SIDE count-matched against the same selection
made by the within-run rank of ``combined_score``, judged on the H/L pivot label
(live-price-extended, 88% settled), over the Gate-4 tradeable pool:

    LLM funnel                       +0.19 % / decision
    rank rule (whole tradeable pool) +1.56 %
    rank rule (inside the shortlist) +1.10 %
    RANDOM K from the same pool      -0.04 %      <- the control

The LLM's selection is not distinguishable from RANDOM (+0.23, day-clustered
t +0.56). The rank rule beats random by +1.60 (t +2.11, halves +0.57/+2.63) and
beats the LLM on the LONG side by 1.51 pp/decision (t -2.20, halves
-0.35/-2.67). Neither result is a tail artifact (winsorized 5/95: rank +0.83,
LLM -0.06, A-B1 t -2.01) nor a volatility artifact (mean |pivot move| of the
names picked: rank 5.18 % vs LLM 5.32 %). Pick overlap was 18.8 %.

**What is kept from the old path, deliberately.** Selection is the INTERSECTION
of two rules, not the rank alone:

  1. the direction BAND must have fired (`aggregator._direction_bands` —
     `rank_diff_threshold_long/_short`, the top ~10 % / bottom ~5 % of the
     tradeable cross-section). This is the ABSTENTION half: on a run where
     nothing clears the band, nothing trades. Dropping it would make this a
     pure rank gate, and a pure rank gate was measured WORSE when the same
     question was asked of Gate 1c (-0.36..-0.50 %/day) precisely because it
     forces trades on runs the book judged weak;
  2. within that set, only the top-K by `combined_score` on the BUY side and
     the bottom-K on the SELL side, K = `gate1_rank_cap` per side — the same
     per-side cap Gate 1c applied to the LLM's calls, and close to the ~2.6
     decisions/run-side the LLM funnel actually produced.

Everything downstream is unchanged: these are ordinary `Recommendation` objects,
so Gates 2/3/4/4b/5, the earnings blackout, the regime BUY block, sizing, the
ledger, the broker sync, the email and the panel all see exactly what they saw
from the LLM. What no longer exists is Gate 1's absolute confidence floor: it
was a threshold on the LLM's STATED confidence, a quantity this path does not
produce. The regime still acts through the PANIC BUY block and the RISK_OFF
size haircut — its threshold arm is inert here, which is stated in `gate_diag`
rather than left to be inferred from a gate that always passes.

`confidence` on the emitted rows is the AGGREGATOR's own confidence (the same
value the panel persists). It is not a gate any more; it still drives the sizing
ramp, which is what it was measured to be usable for.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Dict, Iterable, List, Optional

from loguru import logger

from config import settings
from src.models import Recommendation, TickerSignal

# The rank rule's provenance stamp. It rides `runs.llm_synthesis_provider`, the
# per-recommendation engine column and the per-trade stamp, so every eval can
# split the LLM era from this one — the same discipline that keeps a
# self-hosted model from wearing a hosted model's name.
RANK_PROVIDER = "rank"
RANK_MODEL = "rank-v1"


def _cap() -> int:
    return max(0, int(getattr(settings, "gate1_rank_cap", 3) or 0))


def build_rank_recommendations(
    signals: List[TickerSignal],
    *,
    tradeable: Optional[Iterable[str]] = None,
    max_per_side: Optional[int] = None,
) -> List[Recommendation]:
    """One `Recommendation` per signal; BUY/SELL only for the selected extremes.

    ``tradeable`` is the Gate-4-eligible ticker set the ranking is computed
    over — the same pool the combine's rank transform uses. Names outside it
    are still emitted (HOLD/WATCH) so every downstream surface keeps a row per
    ticker, but they can never be selected: ranking an observe-only name
    against the tradeable cross-section is what Gate 4 exists to prevent.
    """
    now = datetime.now(timezone.utc)
    pool = set(tradeable) if tradeable is not None else {s.ticker for s in signals}
    k = _cap() if max_per_side is None else max(0, int(max_per_side))

    ranked = sorted(
        [s for s in signals if s.ticker in pool],
        key=lambda s: float(s.combined_score or 0.0),
        reverse=True,
    )
    n = len(ranked)
    pct: Dict[str, float] = {
        s.ticker: (100.0 * (n - i) / n if n else 0.0) for i, s in enumerate(ranked)
    }

    buys, sells = [], []
    if k and n:
        # The band decides IF (abstention), the rank decides WHICH (selection).
        buys = [s for s in ranked if s.direction == "BULLISH"][:k]
        sells = [s for s in reversed(ranked) if s.direction == "BEARISH"][:k]
    picked = {s.ticker: ("BUY", i + 1) for i, s in enumerate(buys)}
    picked.update({s.ticker: ("SELL", i + 1) for i, s in enumerate(sells)})

    recs: List[Recommendation] = []
    for s in signals:
        hit = picked.get(s.ticker)
        action = hit[0] if hit else None
        if action is None:
            # Not selected: a fired band with no slot is a HOLD (it had a view),
            # everything else is a WATCH. Neither reaches the ledger.
            action = "HOLD" if s.direction in ("BULLISH", "BEARISH") else "WATCH"
            rationale = s.rationale
        else:
            rationale = (
                f"Rank {'top' if action == 'BUY' else 'bottom'} "
                f"#{hit[1]} of {n} tradeable names: "
                f"combined_score {float(s.combined_score or 0.0):+.3f} "
                f"(p{pct.get(s.ticker, 0.0):.0f}), {s.sources_agreeing} sources agreeing, "
                f"direction {s.direction}. Mechanical rank selection (no LLM)."
            )
        recs.append(Recommendation(
            ticker=s.ticker,
            direction=s.direction,
            action=action,
            confidence=float(s.confidence or 0.0),
            time_horizon="SWING" if action in ("BUY", "SELL") else "N/A",
            rationale=rationale,
            generated_at=now,
        ))

    logger.info(
        f"[rank_entry] {len(buys)} BUY / {len(sells)} SELL selected from "
        f"{n} tradeable names (cap {k}/side, band-gated); "
        f"{len(recs) - len(buys) - len(sells)} HOLD/WATCH"
    )
    return recs
