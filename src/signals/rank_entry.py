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

import hashlib
import math
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


# ── The selection RULE (2026-09-18) ──────────────────────────────────────────
# `topk` is everything described above. `gap_cluster` replaces the "band ∧ top-K"
# intersection with a shape test on the run's own score distribution, and carries
# its own abstention: a run with no cluster buys nothing, which is why it does
# NOT consult the direction band. Both rules can be intersected with the
# own-history freshness filter (`src/signals/score_history.py`).


def _rule() -> str:
    return str(getattr(settings, "rank_entry_rule", "topk") or "topk").lower()


def _rule_b() -> str:
    return str(getattr(settings, "rank_entry_rule_b", "top_pct") or "top_pct").lower()


def resolve_arm(run_id: Optional[str]) -> str:
    """Which selection RULE decides this run.

    A per-RUN flip, so each arm accrues whole-run samples and no run is half one
    rule and half the other. Deterministic on the run id rather than a draw: a
    retried run has to land on the same arm, or the sample over-represents
    exactly the runs that failed once — the same reason the sentiment shadow
    hashes instead of calling `random`.
    """
    share = float(getattr(settings, "rank_entry_ab_share", 0.0) or 0.0)
    if share <= 0.0 or not run_id:
        return _rule()
    if share >= 1.0:
        return _rule_b()
    h = int(hashlib.sha1(str(run_id).encode("utf-8")).hexdigest()[:8], 16) / 0xFFFFFFFF
    return _rule_b() if h < share else _rule()


def union_on() -> bool:
    return bool(getattr(settings, "rank_entry_union", False))


def active_rules(run_id: Optional[str] = None) -> List[str]:
    """Every rule that DECIDES this run.

    Union routing returns both, so each accrues at 100% of its natural rate
    instead of the 50% an A/B split would give it. Otherwise one arm decides.
    """
    if union_on():
        a, b = _rule(), _rule_b()
        return [a] if a == b else [a, b]
    return [resolve_arm(run_id)]


_RULE_MODEL = {"gap_cluster": "rank-gap10", "top_pct": "rank-top5", "topk": "rank-v1"}


def model_stamp(rule=None) -> str:
    """The provenance string for the rule(s) that decided a run. It rides
    `runs.llm_synthesis_provider`, every recommendation row and every new trade,
    so an eval can split the arms exactly as it splits engines. Under union
    routing both names are joined, because both decided."""
    if rule is None:
        rule = active_rules()
    if not isinstance(rule, str):
        parts = [model_stamp(r) for r in rule]
        return parts[0] if len(parts) == 1 else "+".join(parts)
    base = _RULE_MODEL.get(str(rule).lower(), RANK_MODEL)
    return base + "f" if (_freshness_on() and base != RANK_MODEL) else base


def _score_column() -> str:
    """Which quantity the rule ranks on. `combined` is the aggregator's combine;
    `ml_ohlcv` ranks on that method's score directly, which is what the 5%-pivot
    rule study measured (E30: per-day IC +0.229 against the combine's +0.125 on
    the same rows, post-retrain)."""
    return str(getattr(settings, "rank_entry_score", "combined") or "combined").lower()


def _panel_column() -> str:
    """The `signals` column holding the same quantity, for the own-history window."""
    return "ml_ohlcv" if _score_column() == "ml_ohlcv" else "combined_score"


def _score_of(s: TickerSignal) -> float:
    if _score_column() == "ml_ohlcv":
        return float(getattr(s, "ml_ohlcv_score", 0.0) or 0.0)
    return float(getattr(s, "combined_score", 0.0) or 0.0)


def _gap_cluster(vals_desc: List[float]) -> int:
    """How many names sit above the run's top cluster hole, 0 when there is none.

    ``vals_desc`` is the run's scores sorted DESCENDING. Inside the top
    `rank_entry_gap_max_frac` of ranks, find the largest gap between adjacent
    scores measured against the LOCAL spacing (the mean of the neighbouring gaps
    within ±`rank_entry_gap_window`, excluding itself). A gap at least
    `rank_entry_gap_multiple` times that spacing is a cluster boundary. Mirrored
    at the bottom by the caller, which negates and reverses the list.

    Identical arithmetic to the offline study (`scratchpad/final_strategies.py`),
    which is what the measured +3.82%/entry refers to.
    """
    n = len(vals_desc)
    if n < 3:
        return 0
    mult = float(getattr(settings, "rank_entry_gap_multiple", 10.0) or 10.0)
    win = max(1, int(getattr(settings, "rank_entry_gap_window", 5) or 5))
    frac = float(getattr(settings, "rank_entry_gap_max_frac", 0.20) or 0.20)
    depth = max(1, int(math.ceil(frac * n)))
    gaps = [vals_desc[i] - vals_desc[i + 1] for i in range(n - 1)]
    best, best_ratio = 0, 0.0
    for i in range(min(depth, len(gaps))):
        lo, hi = max(0, i - win), min(len(gaps), i + win + 1)
        neigh = gaps[lo:i] + gaps[i + 1:hi]
        if not neigh:
            continue
        local = sum(neigh) / len(neigh)
        if not local > 0:
            continue
        ratio = gaps[i] / local
        if ratio >= mult and ratio > best_ratio:
            best, best_ratio = i + 1, ratio
    return best


def _freshness_on() -> bool:
    return bool(getattr(settings, "enable_rank_entry_freshness", False))


def _apply_freshness(picks: List[TickerSignal], side: str, standings) -> List[TickerSignal]:
    """Keep only names whose score is also a new extreme against their own recent
    history, plus names with no standing (see `score_history` for why those stay)."""
    if not _freshness_on() or not picks:
        return picks
    from src.signals import score_history
    col = _panel_column()
    return [s for s in picks
            if score_history.is_fresh(s.ticker, _score_of(s), side, standings, col)]


def _select(ranked: List[TickerSignal], k: int, rule: str) -> tuple:
    """``(buys, sells)`` before the freshness filter. ``ranked`` is descending."""
    if not ranked:
        return [], []
    if rule == "gap_cluster":
        vals = [_score_of(s) for s in ranked]
        kt = _gap_cluster(vals)
        kb = _gap_cluster([-v for v in reversed(vals)])
        # The cluster IS the abstention: no cluster on a side, nothing bought on
        # it. The direction band is deliberately not consulted — the measured
        # rule never saw one.
        return ranked[:kt], list(reversed(ranked))[:kb]
    if rule == "top_pct":
        pct = float(getattr(settings, "rank_entry_shadow_pct", 0.05) or 0.05)
        kk = max(1, int(round(pct * len(ranked))))
        return ranked[:kk], list(reversed(ranked))[:kk]
    if not k:
        return [], []
    # `topk`: the band decides IF (abstention), the rank decides WHICH.
    return ([s for s in ranked if s.direction == "BULLISH"][:k],
            [s for s in reversed(ranked) if s.direction == "BEARISH"][:k])


def build_rank_recommendations(
    signals: List[TickerSignal],
    *,
    tradeable: Optional[Iterable[str]] = None,
    max_per_side: Optional[int] = None,
    run_id: Optional[str] = None,
) -> List[Recommendation]:
    """One `Recommendation` per signal; BUY/SELL only for the selected extremes.

    ``tradeable`` is the Gate-4-eligible ticker set the ranking is computed
    over — the same pool the combine's rank transform uses. Names outside it
    are still emitted (HOLD/WATCH) so every downstream surface keeps a row per
    ticker, but they can never be selected: ranking an observe-only name
    against the tradeable cross-section is what Gate 4 exists to prevent.

    ``run_id`` picks the A/B arm (`resolve_arm`). Without one the configured
    arm A decides, which is what every caller predating the A/B expects.
    """
    now = datetime.now(timezone.utc)
    pool = set(tradeable) if tradeable is not None else {s.ticker for s in signals}
    k = _cap() if max_per_side is None else max(0, int(max_per_side))

    return _build(signals, pool=pool, k=k, rule=active_rules(run_id), now=now,
                  tag="rank_entry")


def shadow_rule_for(run_id: Optional[str] = None) -> str:
    """The rule the shadow records. "auto" = whichever arm did NOT decide this
    run, so every run yields a matched pair on one cross-section."""
    want = str(getattr(settings, "rank_entry_shadow_rule", "auto") or "auto").lower()
    if want != "auto":
        return want
    live = resolve_arm(run_id)
    return _rule_b() if live == _rule() else _rule()


def build_shadow_recommendations(
    signals: List[TickerSignal],
    *,
    tradeable: Optional[Iterable[str]] = None,
    run_id: Optional[str] = None,
) -> List[Recommendation]:
    """The SHADOW selection rule's picks — computed on the same cross-section,
    persisted as its own arm in `engine_recommendations`, traded by nobody.

    It exists so a promotion is decided on paired data rather than on a fresh
    backtest: `python -m src.analysis.engine_eval` scores the live arm against
    this one on the ticker-runs where they DISAGREE, which is the only subset
    that carries information about either.
    """
    now = datetime.now(timezone.utc)
    pool = set(tradeable) if tradeable is not None else {s.ticker for s in signals}
    rule = shadow_rule_for(run_id)
    return [r for r in _build(signals, pool=pool, k=_cap(), rule=rule, now=now,
                              tag="rank_entry:shadow")
            if r.action in ("BUY", "SELL")]


def _build(signals: List[TickerSignal], *, pool: set, k: int, rule, now: datetime,
           tag: str) -> List[Recommendation]:
    """``rule`` is one rule name, or a list of them — the UNION of their picks.

    Under union routing a ticker can be chosen by both rules; it is still ONE
    recommendation, and the rationale names every rule that chose it so the
    ledger says why. Per-RULE attribution does not live here: it is written to
    `engine_recommendations`, one arm per rule, because a name both rules picked
    is a single position and the ledger cannot split it.
    """
    rules = [rule] if isinstance(rule, str) else list(rule)
    ranked = sorted(
        [s for s in signals if s.ticker in pool],
        key=_score_of,
        reverse=True,
    )
    n = len(ranked)
    pct: Dict[str, float] = {
        s.ticker: (100.0 * (n - i) / n if n else 0.0) for i, s in enumerate(ranked)
    }
    col = _score_column()

    standings = None
    if _freshness_on() and ranked:
        from src.signals import score_history
        standings = score_history.load_standings(_panel_column())

    picked: Dict[str, tuple] = {}
    by_rule: Dict[str, List[str]] = {}
    n_cut = 0
    for r in rules:
        b, s_ = _select(ranked, k, r)
        n_cut += len(b) + len(s_)
        b = _apply_freshness(b, "L", standings)
        s_ = _apply_freshness(s_, "S", standings)
        for act, group in (("BUY", b), ("SELL", s_)):
            for i, sig in enumerate(group):
                by_rule.setdefault(sig.ticker, []).append(r)
                # First rule to claim a ticker sets its rank; a later rule only
                # adds provenance. Ranks are per rule and not comparable anyway.
                picked.setdefault(sig.ticker, (act, i + 1))
    buys = [t for t, (a, _) in picked.items() if a == "BUY"]
    sells = [t for t, (a, _) in picked.items() if a == "SELL"]

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
            why = ""
            if _freshness_on():
                from src.signals import score_history
                why = (f", {score_history.describe(s.ticker, _score_of(s), action, standings, _panel_column())}")
            chose = by_rule.get(s.ticker) or rules
            rationale = (
                f"{' + '.join(_RULE_LABEL.get(r, r) for r in chose)} "
                f"{'top' if action == 'BUY' else 'bottom'} "
                f"#{hit[1]} of {n} tradeable names: "
                f"{col} {_score_of(s):+.3f} "
                f"(p{pct.get(s.ticker, 0.0):.0f}), {s.sources_agreeing} sources agreeing, "
                f"direction {s.direction}{why}. Mechanical selection (no LLM)."
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

    gate = "band-gated" if rules == ["topk"] else "self-abstaining"
    drop = f", {n_cut - len(picked)} dropped by freshness or shared" if _freshness_on() else ""
    both = sum(1 for v in by_rule.values() if len(v) > 1)
    overlap = f", {both} picked by both" if len(rules) > 1 else ""
    logger.info(
        f"[{tag}] rule={'+'.join(rules)} score={col} — {len(buys)} BUY / {len(sells)} SELL "
        f"selected from {n} tradeable names ({gate}{drop}{overlap}); "
        f"{len(recs) - len(picked)} HOLD/WATCH"
    )
    return recs


def picks_for_rule(signals: List[TickerSignal], *, tradeable: Optional[Iterable[str]] = None,
                   rule: str) -> List[Recommendation]:
    """One rule's actionable picks, for the per-arm record. Used to write each
    rule to `engine_recommendations` separately under union routing, which is
    where per-rule evaluation comes from — the ledger holds one position for a
    name both rules chose and cannot attribute it."""
    pool = set(tradeable) if tradeable is not None else {s.ticker for s in signals}
    return [r for r in _build(signals, pool=pool, k=_cap(), rule=rule,
                              now=datetime.now(timezone.utc), tag="rank_entry:arm")
            if r.action in ("BUY", "SELL")]


_RULE_LABEL = {"topk": "Rank", "gap_cluster": "Gap cluster", "top_pct": "Top percentile"}
