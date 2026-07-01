"""Exit-method scoring — signed hold-conviction per held position.

The exit side of the aggregator's sign convention (see ``aggregator.py``). Every
exit method emits, for one OPEN position at one tick, a signed **hold-conviction**
score in ``[-1, +1]``:

* ``+`` = the position should KEEP running in its direction (hold),
* ``−`` = the position should REVERSE (exit),
* ``|score|`` = confidence.

Crucially every score is **position-oriented** — already multiplied by the
position's direction (``+1`` for a BUY/long, ``−1`` for a SELL/short) — so the
exit panel (``exit_panel.compute_exit_method_perf``) can correlate it directly
against the position's *direction-oriented* forward return. A persistently
POSITIVE IC then means the method correctly holds winners / exits losers, exactly
as a positive entry IC means the method correctly picks direction.

Two families of exit method:

* **Exit-decision overlays** — the exit-specific inputs to
  ``tracker._evaluate_decay``: the synthesized ``llm_review`` (the method that
  actually decides), the ``aggregator`` combined score, the ``macro_regime``
  risk overlay, and the ``horizon`` time-stop.
* **Signal methods** — the same per-ticker entry methods (news, tech, momentum,
  …), re-read on the held ticker and oriented to the position, so we learn which
  entry signals are also good EXIT predictors.

Only NON-zero scores are returned — a ``0`` means "no exit view" and is excluded
from the panel, exactly like a ``0`` entry score is excluded from entry IC.
"""

from __future__ import annotations

from typing import Dict

# The exit-specific overlay methods (distinct from the entry signal methods,
# which are ALSO re-scored as exit signals on a held position).
EXIT_DECISION_METHODS = ("llm_review", "aggregator", "macro_regime", "horizon")

# Dashboard Exit-IC table grouping (mirrors signal_panel.IC_CATEGORY_ORDER).
EXIT_CATEGORY_DECISION = "Exit decision (synthesized review + overlays)"
EXIT_CATEGORY_SIGNAL = "Signal methods (re-scored as exit signals)"
EXIT_CATEGORY_ORDER = (EXIT_CATEGORY_DECISION, EXIT_CATEGORY_SIGNAL)

# Human labels for the exit-specific methods (signal-method labels come from
# tracker.METHOD_LABELS).
EXIT_METHOD_LABELS: Dict[str, str] = {
    "llm_review":   "LLM hold-review (synthesized decider)",
    "aggregator":   "Aggregator combined score",
    "macro_regime": "Macro regime overlay",
    "horizon":      "Horizon time-stop",
}

# Regime → hold-pressure for a LONG position (× the position's dir_sign). Only
# the regimes the exit rule actually reacts to are non-zero; NEUTRAL/CAUTION → 0
# (no exit view). Mirrors tracker._evaluate_decay's PANIC/RISK_OFF long-exit.
_REGIME_PRESSURE = {"RISK_ON": 0.5, "RISK_OFF": -0.7, "PANIC": -1.0}


def exit_category_for(method: str) -> str:
    """Map an exit method to its Exit-IC table category."""
    return EXIT_CATEGORY_DECISION if method in EXIT_DECISION_METHODS else EXIT_CATEGORY_SIGNAL


def _dir_sign(trade: dict) -> int:
    """+1 for a long (BUY) position, −1 for a short (SELL)."""
    return 1 if (trade.get("action") or "").upper() == "BUY" else -1


def _horizon_pressure(trade: dict) -> float:
    """One-sided hold-conviction from the horizon time-stop: ``0`` while the
    position is within its target-horizon window, then increasingly negative once
    it has outlived that window (the ``horizon_expired`` exit rule, as a signal).
    ``0`` when horizon synthesis produced no target or its duration is unknown."""
    target_h = trade.get("target_horizon")
    if not target_h:
        return 0.0
    from src.signals.edge_curve import horizon_hours
    from src.performance.tracker import _held_hours
    window = horizon_hours(target_h)
    held = _held_hours(trade)
    if not window or held is None or held < window:
        return 0.0
    return -min(1.0, held / window - 1.0)


def build_exit_scores(trade: dict, hold_review, signals_by_ticker, macro_regime_context) -> Dict[str, float]:
    """Signed hold-conviction score per exit method for one held position.

    ``trade`` is the open-trade dict; ``hold_review`` its opener-pinned
    ``Recommendation`` this tick (or ``None``); ``signals_by_ticker`` the run's
    aggregator cross-section; ``macro_regime_context`` the run regime. Returns
    ``{method: score}`` for the NON-zero scores only (0 = no exit view). See the
    module docstring for the sign convention.
    """
    scores: Dict[str, float] = {}
    dir_sign = _dir_sign(trade)
    entry_action = (trade.get("action") or "").upper()
    ticker = trade.get("ticker")
    today_signal = (signals_by_ticker or {}).get(ticker)

    # 1. Synthesized LLM hold-review — the method that actually decides. +conf when
    #    it reaffirms the position's direction, −conf when it flips; HOLD/WATCH is
    #    no directional view (0 → omitted).
    if hold_review is not None:
        rev_action = (getattr(hold_review, "action", "") or "").upper()
        conf = float(getattr(hold_review, "confidence", 0.0) or 0.0)
        if rev_action == entry_action:
            scores["llm_review"] = conf
        elif rev_action in ("BUY", "SELL"):
            scores["llm_review"] = -conf

    # 2. Aggregator combined score, oriented to the position.
    if today_signal is not None:
        scores["aggregator"] = float(getattr(today_signal, "combined_score", 0.0) or 0.0) * dir_sign

    # 3. Macro regime risk overlay, oriented to the position.
    regime = (getattr(macro_regime_context, "regime", "") or "").upper()
    scores["macro_regime"] = _REGIME_PRESSURE.get(regime, 0.0) * dir_sign

    # 4. Horizon time-stop pressure (one-sided: 0 within window, negative past it).
    scores["horizon"] = _horizon_pressure(trade)

    # 5. The entry signal methods, re-scored on the held ticker and oriented.
    if today_signal is not None:
        from src.performance.tracker import _method_scores_from_signal
        raw = _method_scores_from_signal(ticker, trade.get("direction"), signals_by_ticker)
        for m, v in raw.items():
            scores[m] = float(v or 0.0) * dir_sign

    # Persist only non-zero scores (0 = no view, mirrors the entry panel).
    return {m: round(s, 6) for m, s in scores.items() if s}
