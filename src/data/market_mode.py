"""Market Mode Classifier — dynamic signal weight switching.

Reads market-wide regime signals (VIX, breadth, highs/lows, McClellan) to classify
the current market environment as TRENDING, NEUTRAL, or CHOPPY, then returns the
corresponding signal weight profile for use in the per-ticker aggregator.

Why it matters
──────────────
Two market strategies are fundamentally opposed:
  • Momentum / trend-following  — buy breakouts, ride moves (works in low-VIX, directional markets)
  • Mean-reversion / contrarian — fade extremes, buy dips (works in high-VIX, range-bound markets)

Blending both with fixed weights gives mediocre results in either regime. Mode-switching
concentrates weight on the strategy best suited to current conditions.

Weight profiles
───────────────
  TRENDING  — low VIX, broad participation, highs dominating:
              up-weight tech/news (momentum); down-weight vwap/put_call (contrarian)

  NEUTRAL   — baseline _BASE_WEIGHTS; no adjustment

  CHOPPY    — elevated VIX, mixed breadth:
              up-weight vwap/put_call (mean-reversion + sentiment extremes);
              down-weight tech (momentum less reliable)

Scoring
───────
  Composite score drawn from 4 market-wide signals, each with a calibrated weight:
    VIX       2.0×  — fastest-moving; most directly reflects trending vs. choppy regimes
    Breadth   1.5×  — % of sector ETFs above 200d SMA; wide participation = trending
    Highs/Lows 1.0× — HL spread = %near-highs − %near-lows
    McClellan  1.0× — A/D EMA momentum; cross above 0 = trending begins

  Normalised composite → TRENDING (>+0.5) | NEUTRAL (±0.5) | CHOPPY (<-0.5)
"""

from loguru import logger

from src.models import MarketModeContext

# VIX → mode score ∈ [-3, +2]
_VIX_MODE_SCORES = {
    "COMPLACENCY":   +2.0,   # calm, controlled uptrend
    "LOW":           +1.5,
    "NORMAL":        +0.5,
    "ELEVATED":      -0.5,
    "HIGH":          -1.5,
    "EXTREME_FEAR":  -2.0,
    "PANIC":         -3.0,
    "UNKNOWN":        0.0,
}

# Breadth signal → mode score
_BREADTH_MODE_SCORES = {
    "BREADTH_EXTENDED":  +1.5,
    "BREADTH_HEALTHY":   +1.0,
    "BREADTH_MIXED":      0.0,
    "BREADTH_WEAK":      -1.0,
    "BREADTH_COLLAPSE":  -2.0,
    "UNKNOWN":            0.0,
}

# Highs/Lows signal → mode score
_HL_MODE_SCORES = {
    "STRONG_HIGHS":    +1.5,
    "HIGHS_DOMINATE":  +0.75,
    "BALANCED":         0.0,
    "LOWS_DOMINATE":   -0.75,
    "STRONG_LOWS":     -1.5,
    "UNKNOWN":          0.0,
}

# McClellan osc_signal → mode score
_MCL_MODE_SCORES = {
    "OVERBOUGHT":       +1.5,
    "BULLISH_MOMENTUM": +0.75,
    "NEUTRAL":           0.0,
    "BEARISH_MOMENTUM": -0.75,
    "OVERSOLD":         -1.5,
    "UNKNOWN":           0.0,
}

_MODE_WEIGHTS = {
    "vix":       2.0,   # most direct, fastest-moving
    "breadth":   1.5,
    "hl":        1.0,
    "mcclellan": 1.0,
}

# Last run's input coverage (available / total market-structure signals), surfaced
# by the pipeline through ``_collect_sources``. Unlike the regime, low mode coverage
# needs no behaviour guard — NEUTRAL is already the safe default (no weight tilt).
_LAST_COVERAGE: dict = {"available": 0, "total": len(_MODE_WEIGHTS)}


def reset_mode_coverage() -> None:
    """Clear the cached market-mode input-coverage (call at run start). total=0 marks
    'not computed this run' so it never false-alarms in source-health."""
    _LAST_COVERAGE.update(available=0, total=0)


def get_mode_coverage() -> dict:
    """Snapshot of the most recent market-mode computation's input coverage."""
    return dict(_LAST_COVERAGE)

# ── Weight profiles for each mode ────────────────────────────────────────────
# Raw (unnormalised) weights — the aggregator's _normalised_weights() will
# divide by total so the final fractions depend only on relative ratios.

WEIGHT_PROFILES = {
    "TRENDING": {
        "news":      0.40,   # momentum is news-driven; keep full weight
        "tech":      0.45,   # boost: momentum signals more reliable in trends
        "insider":   0.30,   # unchanged: smart money always relevant
        "put_call":  0.08,   # cut: contrarian overshoots less in steady trends
        "max_pain":  0.12,   # unchanged
        "oi_skew":   0.15,   # unchanged
        "vwap":      0.04,   # cut heavily: mean-reversion fights the trend
    },
    "NEUTRAL": {
        "news":      0.40,   # baseline _BASE_WEIGHTS (exact copy)
        "tech":      0.30,
        "insider":   0.30,
        "put_call":  0.15,
        "max_pain":  0.12,
        "oi_skew":   0.15,
        "vwap":      0.12,
    },
    "CHOPPY": {
        "news":      0.30,   # slight cut: hard to act on news in choppy tape
        "tech":      0.15,   # cut: momentum signals produce whipsaws in ranges
        "insider":   0.30,   # unchanged
        "put_call":  0.28,   # boost: sentiment extremes are the primary edge
        "max_pain":  0.12,   # unchanged
        "oi_skew":   0.15,   # unchanged
        "vwap":      0.28,   # boost: mean-reversion is the primary edge
    },
}


def compute_market_mode(
    vix_context=None,
    breadth_context=None,
    highs_lows_context=None,
    mcclellan_context=None,
) -> MarketModeContext:
    """Classify the market as TRENDING, NEUTRAL, or CHOPPY and return the weight profile.

    Any missing context simply contributes zero weight to the composite.
    When no context is available at all, NEUTRAL is returned (no adjustment).
    """
    weighted_score = 0.0
    total_weight   = 0.0
    inputs_available = 0
    evidence: list[str] = []

    def _add(name, signal_val, score_map, weight):
        nonlocal weighted_score, total_weight, inputs_available
        score = score_map.get(signal_val, 0.0)
        weighted_score += score * weight
        total_weight   += weight
        inputs_available += 1
        evidence.append(f"{name}={signal_val}({score:+.2f})")

    if vix_context is not None:
        _add("VIX",       vix_context.vix_signal,   _VIX_MODE_SCORES,     _MODE_WEIGHTS["vix"])
    if breadth_context is not None:
        _add("BREADTH",   breadth_context.signal,    _BREADTH_MODE_SCORES, _MODE_WEIGHTS["breadth"])
    if highs_lows_context is not None:
        _add("HL",        highs_lows_context.signal, _HL_MODE_SCORES,      _MODE_WEIGHTS["hl"])
    if mcclellan_context is not None:
        _add("MCLELLAN",  mcclellan_context.osc_signal, _MCL_MODE_SCORES,  _MODE_WEIGHTS["mcclellan"])

    norm = weighted_score / total_weight if total_weight > 0 else 0.0

    if norm > +0.5:
        mode = "TRENDING"
    elif norm < -0.5:
        mode = "CHOPPY"
    else:
        mode = "NEUTRAL"

    inputs_total = len(_MODE_WEIGHTS)
    _LAST_COVERAGE.update(available=inputs_available, total=inputs_total)
    weight_profile = WEIGHT_PROFILES[mode]
    evidence_str   = "  |  ".join(evidence) if evidence else "no market-structure inputs available"

    # Build a human-readable description of the weight adjustments
    neutral = WEIGHT_PROFILES["NEUTRAL"]
    changes = []
    for k in ("tech", "news", "vwap", "put_call"):
        delta = weight_profile[k] - neutral[k]
        if abs(delta) >= 0.01:
            direction = "▲" if delta > 0 else "▼"
            changes.append(f"{k} {direction}{abs(delta):.2f}")
    weight_summary = ", ".join(changes) if changes else "no adjustment"

    summary = (
        f"Market mode: {mode} (score={norm:+.2f}).  "
        f"Weight adjustment: {weight_summary}.  "
        + evidence_str
    )

    logger.info(
        f"[market_mode] {mode}  score={norm:+.2f}  adjustments: {weight_summary}"
    )
    logger.debug(f"[market_mode] evidence: {evidence_str}")

    return MarketModeContext(
        mode=mode,
        composite_score=round(norm, 3),
        weight_profile=weight_profile,
        evidence=evidence_str,
        weight_summary=weight_summary,
        summary=summary,
        inputs_available=inputs_available,
        inputs_total=inputs_total,
    )
