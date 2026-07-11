"""Exit-conviction consensus — the entry-breadth analog on the exit side.

The entry decision blends many methods (agreement breadth is its strongest
signal); the exit decision, until now, used a single LLM-scalar hold-review and
IGNORED the 27-method exit-conviction panel it already collects every tick.
This module adds the missing analog: a breadth-of-method EXIT CONSENSUS that
gently modulates the close bar — evidence-throttled, so with no data it is a
NO-OP and the live behavior is exactly today's, and it can only strengthen as
the exit panel accrues (never assumed to work).

How it feeds the decision (``tracker._evaluate_decay``): the same-direction
close test fires when the hold-review conviction falls below
``_confidence_floor``. Here the EFFECTIVE floor is nudged:

    effective_floor = base_floor − span_eff × exit_consensus

where ``exit_consensus`` ∈ [−1, +1] is the mean of the underlying SIGNAL
methods' position-oriented exit scores (+ = hold, − = exit; the LLM review and
the aggregator combined-score are excluded so this is a pure breadth-of-raw-
methods read, and so it does NOT reintroduce the aggregator trigger-happiness
Fix #2 removed). Consensus says EXIT (negative) → floor RAISED → the position
closes more readily; consensus says HOLD (positive) → floor LOWERED → it holds
more readily (which also makes a confident LLM hold stickier). ``span_eff`` is
bounded and ramps with the closed-trade sample toward a small cap, so a
confident LLM review (conv well above the floor) is never overridden — only
borderline convictions get tipped by method consensus, preserving Fix #2.

The whole layer is validated OFFLINE, before it earns real weight, by
``exit_policy_eval`` (the exit-breadth close rule there is the same idea scored
counterfactually). Deterministic given the ledger; registry-reported.
"""

from __future__ import annotations

from statistics import mean
from typing import Dict, List, Optional

from config.settings import settings

# Method scores that are NOT part of the raw-method breadth consensus: the LLM
# review (that IS the conviction the floor tests), the aggregator combined score
# (Fix #2 distrusts it for exits), the non-signal decision/excursion layers, and
# the PANEL-FIRST classic anomalies (hi52 / 12-1 momentum / short-term reversal,
# 2026-07-08) — measured in the entry AND exit panels but acting on neither
# until their IC earns it (st_reversal in particular scores every winner's
# up-week as exit pressure, which is exactly the hypothesis to TEST, not assume).
_CONSENSUS_SKIP = frozenset({"llm_review", "aggregator", "horizon", "edge_decay",
                             "macro_regime", "mfe", "mae",
                             "hi52", "mom_12_1", "st_reversal",
                             "squeeze", "iv_term", "avwap",
                             "resid_mom", "vol_profile"})


def exit_method_consensus(scores: Dict[str, float]) -> Optional[float]:
    """Mean of the underlying SIGNAL methods' position-oriented exit scores
    (+ = hold, − = exit) — the breadth-and-magnitude consensus. ``None`` when no
    signal method scored (→ no adjustment). Bounded to [−1, +1] by construction."""
    vals: List[float] = [float(v) for m, v in (scores or {}).items()
                         if m not in _CONSENSUS_SKIP and v is not None]
    if not vals:
        return None
    return max(-1.0, min(1.0, mean(vals)))


def exit_conviction_calibration(trades: Optional[List[dict]] = None) -> dict:
    """The evidence-throttled strength of the exit-consensus nudge.

    ``span_eff`` ramps with the CLOSED-trade sample from a small prior toward a
    bounded cap — trust grows with evidence, and stays gentle:

        ramp     = n_closed / (n_closed + exit_conviction_prior_n)
        span_eff = span_prior + (span_max − span_prior) × ramp

    Deliberately NOT a positive-correctness prior (unlike entry breadth, which
    had a measured effect): the exit consensus is UNVALIDATED, so it starts as a
    small, bounded nudge to be confirmed or denied by ``exit_policy_eval`` over
    the coming weeks. Registry-reported. ``trades`` supplied by the caller (the
    monitor passes its loaded ledger); ``None`` loads it."""
    if not settings.enable_exit_conviction:
        return {"span_eff": 0.0, "n_closed": 0}
    if trades is None:
        from src.performance.tracker import _load_trades
        trades = _load_trades()
    n_closed = sum(1 for t in trades if t.get("status") == "CLOSED")
    prior = float(settings.exit_conviction_span_prior)
    cap = float(settings.exit_conviction_span_max)
    prior_n = max(1, int(settings.exit_conviction_prior_n))
    ramp = n_closed / (n_closed + prior_n)
    span_eff = round(prior + (cap - prior) * ramp, 4)
    try:
        from src.performance.calibration import report_calibration
        report_calibration("exit_conviction_span", value=span_eff, prior=prior,
                           n_evidence=n_closed, unit="floor Δ",
                           note="exit method-consensus nudge strength "
                                "(UNVALIDATED — gentle, ramps with closes; confirm via exit_policy_eval)")
    except Exception:
        pass
    return {"span_eff": span_eff, "n_closed": n_closed}


def exit_floor_adjustment(scores: Dict[str, float], cal: dict) -> float:
    """Bounded nudge to the same-direction close floor from the method consensus.
    ``+`` raises the floor (close more readily — methods corroborate an exit);
    ``−`` lowers it (hold more readily). Clamped to ±``span_eff`` and 0 when
    there is no consensus / the layer is off."""
    if not settings.enable_exit_conviction:
        return 0.0
    span = float(cal.get("span_eff") or 0.0)
    if span <= 0:
        return 0.0
    consensus = exit_method_consensus(scores)
    if consensus is None:
        return 0.0
    adj = -span * consensus                        # consensus<0 (exit) → adj>0 (raise floor)
    return round(max(-span, min(span, adj)), 4)
