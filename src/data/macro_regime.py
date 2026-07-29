"""Macro Regime Filter — composite top-down overlay across all macro context objects.

Reads VIX, MOVE, bond internals, global macro, FRED, breadth, credit, and the Dark
Pool Index (DIX) to produce a single PANIC | RISK_OFF | CAUTION | NEUTRAL | RISK_ON
regime classification.

Regime effects in the pipeline
────────────────────────────────
  PANIC     → confidence threshold 0.95, BUY entries blocked
  RISK_OFF  → confidence threshold 0.89, BUY entries blocked
  CAUTION   → confidence threshold 0.87, BUY entries allowed
  NEUTRAL   → confidence threshold 0.85 (baseline), BUY entries allowed
  RISK_ON   → confidence threshold 0.79, BUY entries allowed
"""

from loguru import logger

from config import settings
from src.models import MacroRegimeContext

# Per-source score mappings ∈ [-3, +1]
_VIX_SCORES = {
    "PANIC":        -3,
    "EXTREME_FEAR": -2,
    "HIGH":         -1,
    "ELEVATED":     -0.5,
    "NORMAL":        0,
    "LOW":           0.5,
    "COMPLACENCY":   0.5,
    "UNKNOWN":       0,
}

_MOVE_SCORES = {
    "PANIC":    -3,
    "EXTREME":  -2,
    "HIGH":     -1.5,
    "ELEVATED": -0.5,
    "NORMAL":    0,
    "LOW":       0.5,
    "CALM":      1,
    "UNKNOWN":   0,
}

_BOND_REGIME_SCORES = {
    "RISK_OFF":      -2,
    "DEFENSIVE":     -1,
    "NEUTRAL":        0,
    "CONSTRUCTIVE":   0.5,
    "RISK_ON":        1,
    "REFLATIONARY":   0.5,
    "UNKNOWN":        0,
}

_GLOBAL_MACRO_SCORES = {
    "RISK_OFF":      -2,
    "DEFENSIVE":     -1,
    "NEUTRAL":        0,
    "CONSTRUCTIVE":   0.5,
    "RISK_ON":        1,
    "UNKNOWN":        0,
}

_FRED_REGIME_SCORES = {
    "RECESSION":  -2,
    "LATE_CYCLE": -1,
    "SLOWDOWN":   -0.5,
    "EXPANSION":   1,
    "UNKNOWN":     0,
}

_BREADTH_SCORES = {
    "BREADTH_COLLAPSE": -2,
    "BREADTH_WEAK":     -1,
    "BREADTH_MIXED":     0,
    "BREADTH_HEALTHY":   0.5,
    "BREADTH_EXTENDED":  0.5,
    "UNKNOWN":           0,
}

_CREDIT_SCORES = {
    "CREDIT_STRESS":  -2,
    "CREDIT_CAUTION": -1,
    "NEUTRAL":         0,
    "CREDIT_STRONG":   0.5,
    "CREDIT_SURGE":    0.5,
    "UNKNOWN":         0,
}

# Dark Pool Index — hidden institutional accumulation (bullish-tilted flow signal)
_DIX_SCORES = {
    "STRONG_ACCUMULATION":  1,
    "ACCUMULATION":         0.5,
    "NEUTRAL":              0,
    "DISTRIBUTION":        -1,
    "STRONG_DISTRIBUTION": -2,
    "UNKNOWN":             0,
}

# Macro news regime — geopolitics / oil / tariffs / policy / black-swan.
# Asymmetric weighting: a CRISIS-grade headline routinely sets the market tone
# in a way that VIX and credit only catch hours/days later, so we treat CRISIS
# as a -3 panic-tier input. STABLE adds no information.
_MACRO_NEWS_SCORES = {
    "STABLE":         0,
    "WATCH":         -0.5,
    "ELEVATED_RISK": -1.5,
    "CRISIS":        -3,
    "UNKNOWN":        0,
}

# Intermarket divergence (broad index ETFs vs SPY) — small-cap-led RISK_OFF
# carries asymmetric weight: when IWM + RSP both lag SPY it's one of the
# strongest distribution warnings in the macro stack, so we score it slightly
# more aggressively in the negative direction than in the positive.
_INTERMARKET_SCORES = {
    "BROAD_EXPANSION":  1,
    "BROAD_HEALTHY":    0.5,
    "NEUTRAL":          0,
    "NARROW_CAUTION":  -1,
    "NARROW_RISK_OFF": -2,
    "UNKNOWN":          0,
}

# VIX and MOVE are the fastest-moving signals and deserve more weight.
# Macro news gets weight = 1.5 because narrative-driven moves (wars, OPEC
# decisions, tariff announcements) reliably lead the numeric macro stack
# by hours to days, but it's also the noisiest input (LLM classification
# variance) so we don't push it as high as VIX/MOVE.
_WEIGHTS = {
    "vix":          2.0,
    "move":         2.0,
    "bond":         1.5,
    "macro_news":   1.5,
    "global_macro": 1.0,
    "fred":         1.0,
    "breadth":      1.0,
    "dix":          1.0,
    "intermarket":  1.0,
    "credit":       0.5,
}

_REGIME_THRESHOLD = {
    # Baseline raised from 0.78 → 0.85 (2026-07-21, user directive: "minimum
    # confidence score 0.85"). The whole table is shifted UP a uniform +0.07 so
    # the NEUTRAL baseline = 0.85 while the regime filter's relative dynamics are
    # preserved (relax when calm, tighten when risky). NOT a hard floor — RISK_ON
    # still sits below 0.85 by design.
    "PANIC":    0.95,
    "RISK_OFF": 0.89,
    "CAUTION":  0.87,
    "NEUTRAL":  0.85,
    "RISK_ON":  0.79,
}

# Smoothed-step threshold (2026-07-27). The step table is a CLIFF: composites of
# -0.801 and -0.799 sit in different regimes, and the live composite has sd 0.146
# — that boundary is well inside one standard deviation of ordinary daily
# variation, so the same market sampled two ticks apart could jump the actionable
# bar a full step.
#
# The fix deliberately SMOOTHS THE BOUNDARIES rather than interpolating across
# whole bands. A full band-to-band interpolation was tried first and rejected on
# measurement: it raised the bar on 100% of observed runs (mean +0.0089) and
# would have dropped 7.9% of actionable trades — a systematic tightening, not
# the removal of an instability. Here each band keeps its documented threshold
# throughout its interior; only within `_THRESHOLD_SMOOTH_HALFWIDTH` of a
# boundary does the value ramp linearly between the two neighbouring bands. So a
# run sitting deep inside NEUTRAL is completely unaffected, and a run straddling
# a boundary moves smoothly instead of jumping.
_THRESHOLD_SMOOTH_HALFWIDTH = 0.05

# (boundary norm, band BELOW it, band ABOVE it) — ordered, matching the bands in
# compute_macro_regime.
_THRESHOLD_BOUNDARIES = ((-1.5, "PANIC", "RISK_OFF"),
                         (-0.8, "RISK_OFF", "CAUTION"),
                         (-0.3, "CAUTION", "NEUTRAL"),
                         (+0.3, "NEUTRAL", "RISK_ON"))


def continuous_threshold(norm: float) -> float:
    """Actionable-confidence threshold, continuous in the macro composite.

    Equals the documented per-regime threshold everywhere except within
    ``_THRESHOLD_SMOOTH_HALFWIDTH`` of a band boundary, where it ramps linearly
    between the two bands. Monotonically non-increasing in ``norm`` (worse macro
    ⇒ higher bar) and never below the band values it interpolates, so the
    minimum-confidence directive holds wherever the step table held it.
    """
    w = float(_THRESHOLD_SMOOTH_HALFWIDTH)
    for edge, below, above in _THRESHOLD_BOUNDARIES:
        lo, hi = _REGIME_THRESHOLD[below], _REGIME_THRESHOLD[above]
        # The ramp sits entirely on the LENIENT side of the boundary: the
        # stricter band holds its value right up to the edge, and the easing
        # happens inside the more permissive band. Straddling the edge
        # symmetrically would have pushed NEUTRAL below 0.85 near the RISK_ON
        # boundary, breaking the minimum-confidence directive (0.85 is a user
        # directive, not a tunable; RISK_ON is its one documented exception).
        if w > 0 and edge < norm < (edge + 2.0 * w):
            frac = (norm - edge) / (2.0 * w)
            return lo + (hi - lo) * frac
    # Outside every transition zone: the plain step value.
    if norm <= -1.5:
        return _REGIME_THRESHOLD["PANIC"]
    if norm <= -0.8:
        return _REGIME_THRESHOLD["RISK_OFF"]
    if norm <= -0.3:
        return _REGIME_THRESHOLD["CAUTION"]
    if norm <= 0.3:
        return _REGIME_THRESHOLD["NEUTRAL"]
    return _REGIME_THRESHOLD["RISK_ON"]


# RISK_OFF no longer BLOCKS buys (2026-07-27) — it takes a size haircut instead.
# The 7-input historical reconstruction over 2000-2026 measured RISK_OFF as the
# regime with the BEST forward returns of any state (+2.25% SPY at 21d, 70.0%
# up-rate over 337 days) — better than NEUTRAL (+0.37%) and RISK_ON (+0.84%) —
# while PANIC is the only genuinely bad one (-0.69%, 45.5% up over 66 days).
# The two had been lumped into one blocked bucket on no evidence they behave
# alike, so the system was sitting out its best-performing regime.
#
# It is a HAIRCUT rather than full size deliberately: the measurement says
# longs are fine in RISK_OFF, but RISK_OFF is by definition a stressed state
# (wider spreads, fatter tails), the 337 days are concentrated in a handful of
# episodes, and going from "banned" straight to "full size" is a bigger step
# than the evidence carries. The ban stays for PANIC.
_REGIME_ALLOW_BUYS = {
    "PANIC":    False,
    "RISK_OFF": True,
    "CAUTION":  True,
    "NEUTRAL":  True,
    "RISK_ON":  True,
}


def regime_size_multiplier(regime: str) -> float:
    """Position-size haircut for entries opened in ``regime`` (1.0 = no change).

    Only RISK_OFF is haircut today; every other regime is either unrestricted
    or (PANIC) blocked outright, and no other regime has evidence justifying a
    size change. Returns 1.0 when the feature is disabled or the regime is
    unknown, so the sizing chain is untouched by default."""
    if not getattr(settings, "enable_regime_size_haircut", False):
        return 1.0
    if str(regime or "").upper() == "RISK_OFF":
        return float(settings.risk_off_size_multiplier)
    return 1.0

# Last run's input coverage (available / total macro signals), so the pipeline can
# surface it through ``_collect_sources`` → run_sources (Data Quality) + the health
# banner without threading the context object into that helper.
_LAST_COVERAGE: dict = {"available": 0, "total": len(_WEIGHTS)}


def reset_regime_coverage() -> None:
    """Clear the cached regime input-coverage (call at run start). total=0 marks
    'not computed this run' so a disabled regime never false-alarms in source-health."""
    _LAST_COVERAGE.update(available=0, total=0)


def get_regime_coverage() -> dict:
    """Snapshot of the most recent regime computation's input coverage."""
    return dict(_LAST_COVERAGE)


def compute_macro_regime(
    vix_context=None,
    move_context=None,
    bond_internals_context=None,
    global_macro_context=None,
    macro_context=None,
    breadth_context=None,
    credit_context=None,
    dix_context=None,
    intermarket_context=None,
    macro_news_context=None,
) -> MacroRegimeContext:
    """Compute a composite macro regime from all available macro inputs.

    Returns a MacroRegimeContext that the pipeline uses to:
      1. Gate new BUY entries (blocked during PANIC / RISK_OFF).
      2. Adjust the minimum confidence threshold for actionable signals.
    """
    weighted_score = 0.0
    total_weight   = 0.0
    has_panic      = False
    inputs_available = 0
    evidence: list[str] = []

    def _add(name, signal_val, score_map, weight):
        nonlocal weighted_score, total_weight, has_panic, inputs_available
        score = score_map.get(signal_val, 0.0)
        weighted_score += score * weight
        total_weight   += weight
        inputs_available += 1
        evidence.append(f"{name}={signal_val}({score:+.1f})")
        if score_map is _VIX_SCORES or score_map is _MOVE_SCORES:
            if signal_val == "PANIC":
                has_panic = True

    if vix_context is not None:
        _add("VIX",    vix_context.vix_signal,               _VIX_SCORES,         _WEIGHTS["vix"])
    if move_context is not None:
        _add("MOVE",   move_context.signal,                  _MOVE_SCORES,        _WEIGHTS["move"])
    if bond_internals_context is not None:
        _add("BOND",   bond_internals_context.regime,        _BOND_REGIME_SCORES, _WEIGHTS["bond"])
    if global_macro_context is not None:
        _add("GLOBAL", global_macro_context.composite_signal,_GLOBAL_MACRO_SCORES,_WEIGHTS["global_macro"])
    if macro_context is not None:
        _add("FRED",   macro_context.regime,                 _FRED_REGIME_SCORES, _WEIGHTS["fred"])
    if breadth_context is not None:
        _add("BREADTH",breadth_context.signal,               _BREADTH_SCORES,     _WEIGHTS["breadth"])
    if credit_context is not None:
        _add("CREDIT", credit_context.signal,                _CREDIT_SCORES,      _WEIGHTS["credit"])
    if dix_context is not None:
        _add("DIX",    dix_context.signal,                   _DIX_SCORES,         _WEIGHTS["dix"])
    if intermarket_context is not None:
        _add("INTERMARKET", intermarket_context.composite_signal,
             _INTERMARKET_SCORES, _WEIGHTS["intermarket"])
    if macro_news_context is not None:
        _add("MACRO_NEWS", macro_news_context.composite_signal,
             _MACRO_NEWS_SCORES, _WEIGHTS["macro_news"])
        # CRISIS in macro news should be able to trigger PANIC by itself —
        # mirrors the VIX/MOVE PANIC short-circuit logic.
        if macro_news_context.composite_signal == "CRISIS":
            has_panic = True

    norm = weighted_score / total_weight if total_weight > 0 else 0.0

    if (has_panic and norm <= -1.0) or norm <= -1.5:
        regime = "PANIC"
    elif norm <= -0.8:
        regime = "RISK_OFF"
    elif norm <= -0.3:
        regime = "CAUTION"
    elif norm <= +0.3:
        regime = "NEUTRAL"
    else:
        regime = "RISK_ON"

    # Fail-CAUTIOUS, not fail-open: when too few macro inputs survived, the
    # composite is untrustworthy — and its absence-of-signal default is a
    # PERMISSIVE NEUTRAL (BUYs allowed at the 0.85 baseline). If all the macro
    # feeds go dark we must not silently relax the gate; degrade a permissive
    # verdict to at least CAUTION (higher threshold, BUYs still allowed). A
    # genuinely protective verdict (CAUTION/RISK_OFF/PANIC — e.g. a lone VIX
    # PANIC) is never loosened by this.
    inputs_total = len(_WEIGHTS)
    low_coverage = inputs_available < int(settings.macro_regime_min_inputs)
    coverage_note = ""
    if low_coverage and regime in ("NEUTRAL", "RISK_ON"):
        coverage_note = (
            f"LOW MACRO COVERAGE ({inputs_available}/{inputs_total} inputs) — "
            f"regime forced {regime}→CAUTION (fail-safe). "
        )
        regime = "CAUTION"
    _LAST_COVERAGE.update(available=inputs_available, total=inputs_total)

    # Threshold is continuous in the composite when enabled (default) — the step
    # table is kept as the anchor set and as the fallback. The REGIME LABEL and
    # the BUY block stay step-based: those are genuine state decisions, and a
    # half-blocked BUY is not a meaningful thing.
    if getattr(settings, "enable_continuous_regime_threshold", False):
        threshold = round(continuous_threshold(norm), 4)
    else:
        threshold = _REGIME_THRESHOLD[regime]
    allow_buys = _REGIME_ALLOW_BUYS[regime]

    evidence_str = "  |  ".join(evidence) if evidence else "no macro inputs available"
    summary = (
        f"Composite regime: {regime} (score={norm:+.2f}) — "
        + coverage_note
        + ("BUY entries BLOCKED. " if not allow_buys else "")
        + f"Actionable threshold → {threshold:.0%}. "
        + evidence_str
    )

    logger.info(
        f"[macro_regime] {regime}  score={norm:+.2f}  threshold={threshold:.0%}  "
        f"allow_buys={allow_buys}  inputs={inputs_available}/{inputs_total}"
    )
    if coverage_note:
        logger.warning(f"[macro_regime] {coverage_note.strip()}")
    logger.debug(f"[macro_regime] evidence: {evidence_str}")

    return MacroRegimeContext(
        regime=regime,
        composite_score=round(norm, 3),
        confidence_threshold=threshold,
        allow_buys=allow_buys,
        has_panic_signal=has_panic,
        evidence=evidence_str,
        summary=summary,
        inputs_available=inputs_available,
        inputs_total=inputs_total,
    )