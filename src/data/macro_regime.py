"""Macro Regime Filter — composite top-down overlay across all macro context objects.

Reads VIX, MOVE, bond internals, global macro, FRED, breadth, credit, and the Dark
Pool Index (DIX) to produce a single PANIC | RISK_OFF | CAUTION | NEUTRAL | RISK_ON
regime classification.

Regime effects in the pipeline
────────────────────────────────
  PANIC     → confidence threshold 0.88, BUY entries blocked
  RISK_OFF  → confidence threshold 0.82, BUY entries blocked
  CAUTION   → confidence threshold 0.80, BUY entries allowed
  NEUTRAL   → confidence threshold 0.78 (baseline), BUY entries allowed
  RISK_ON   → confidence threshold 0.72, BUY entries allowed
"""

from loguru import logger

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

# VIX and MOVE are the fastest-moving signals and deserve more weight
_WEIGHTS = {
    "vix":          2.0,
    "move":         2.0,
    "bond":         1.5,
    "global_macro": 1.0,
    "fred":         1.0,
    "breadth":      1.0,
    "dix":          1.0,
    "credit":       0.5,
}

_REGIME_THRESHOLD = {
    "PANIC":    0.88,
    "RISK_OFF": 0.82,
    "CAUTION":  0.80,
    "NEUTRAL":  0.78,
    "RISK_ON":  0.72,
}

_REGIME_ALLOW_BUYS = {
    "PANIC":    False,
    "RISK_OFF": False,
    "CAUTION":  True,
    "NEUTRAL":  True,
    "RISK_ON":  True,
}


def compute_macro_regime(
    vix_context=None,
    move_context=None,
    bond_internals_context=None,
    global_macro_context=None,
    macro_context=None,
    breadth_context=None,
    credit_context=None,
    dix_context=None,
) -> MacroRegimeContext:
    """Compute a composite macro regime from all available macro inputs.

    Returns a MacroRegimeContext that the pipeline uses to:
      1. Gate new BUY entries (blocked during PANIC / RISK_OFF).
      2. Adjust the minimum confidence threshold for actionable signals.
    """
    weighted_score = 0.0
    total_weight   = 0.0
    has_panic      = False
    evidence: list[str] = []

    def _add(name, signal_val, score_map, weight):
        nonlocal weighted_score, total_weight, has_panic
        score = score_map.get(signal_val, 0.0)
        weighted_score += score * weight
        total_weight   += weight
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

    threshold  = _REGIME_THRESHOLD[regime]
    allow_buys = _REGIME_ALLOW_BUYS[regime]

    evidence_str = "  |  ".join(evidence) if evidence else "no macro inputs available"
    summary = (
        f"Composite regime: {regime} (score={norm:+.2f}) — "
        + ("BUY entries BLOCKED. " if not allow_buys else "")
        + f"Actionable threshold → {threshold:.0%}. "
        + evidence_str
    )

    logger.info(
        f"[macro_regime] {regime}  score={norm:+.2f}  threshold={threshold:.0%}  allow_buys={allow_buys}"
    )
    logger.debug(f"[macro_regime] evidence: {evidence_str}")

    return MacroRegimeContext(
        regime=regime,
        composite_score=round(norm, 3),
        confidence_threshold=threshold,
        allow_buys=allow_buys,
        has_panic_signal=has_panic,
        evidence=evidence_str,
        summary=summary,
    )