"""Agreement floor (actionable-filter Gate 1b, 2026-07-20).

CLAUDE.md documents "a single strong signal source never produces a BUY/SELL
regardless of score" as a baseline invariant, but a 2026-07-20 audit found NO
pipeline gate actually enforced it — sources_agreeing >= 2 was only a PROMPT
INSTRUCTION the LLM was trusted to self-apply. This gate makes it mechanical:
pipeline._passes_agreement_gate drops a BUY/SELL whose OWN direction matches the
aggregator's sig.direction (the common "echo" case) when sig.sources_agreeing is
below the configured floor. A genuine LLM override (direction != sig.direction)
or a missing signal passes unchecked — sources_agreeing isn't attributable to a
call the aggregator didn't itself make.
"""

from config.settings import settings
from src.pipeline import _passes_agreement_gate
from src.models import TickerSignal


def _sig(direction="BULLISH", sources_agreeing=0):
    return TickerSignal(
        ticker="AAA", direction=direction, confidence=0.8,
        sentiment_score=0.0, technical_score=0.0, rationale="r",
        sources_agreeing=sources_agreeing,
    )


def test_gate_off_always_passes(monkeypatch):
    monkeypatch.setattr(settings, "enable_agreement_gate", False)
    monkeypatch.setattr(settings, "min_sources_agreeing_gate", 2)
    assert _passes_agreement_gate("BULLISH", _sig("BULLISH", sources_agreeing=0)) is True


def test_missing_signal_passes_unchecked(monkeypatch):
    monkeypatch.setattr(settings, "enable_agreement_gate", True)
    monkeypatch.setattr(settings, "min_sources_agreeing_gate", 2)
    assert _passes_agreement_gate("BULLISH", None) is True


def test_override_direction_passes_unchecked(monkeypatch):
    # LLM's own call (BEARISH) disagrees with the aggregator's sig.direction
    # (BULLISH) — sources_agreeing (computed relative to sig.direction) isn't a
    # meaningful count for the call actually being made, so the gate doesn't fire
    # even though sources_agreeing is 0.
    monkeypatch.setattr(settings, "enable_agreement_gate", True)
    monkeypatch.setattr(settings, "min_sources_agreeing_gate", 2)
    assert _passes_agreement_gate("BEARISH", _sig("BULLISH", sources_agreeing=0)) is True


def test_echo_call_below_floor_blocked(monkeypatch):
    monkeypatch.setattr(settings, "enable_agreement_gate", True)
    monkeypatch.setattr(settings, "min_sources_agreeing_gate", 2)
    assert _passes_agreement_gate("BULLISH", _sig("BULLISH", sources_agreeing=1)) is False
    assert _passes_agreement_gate("BULLISH", _sig("BULLISH", sources_agreeing=0)) is False


def test_echo_call_at_or_above_floor_passes(monkeypatch):
    monkeypatch.setattr(settings, "enable_agreement_gate", True)
    monkeypatch.setattr(settings, "min_sources_agreeing_gate", 2)
    assert _passes_agreement_gate("BULLISH", _sig("BULLISH", sources_agreeing=2)) is True
    assert _passes_agreement_gate("BULLISH", _sig("BULLISH", sources_agreeing=5)) is True


def test_threshold_is_configurable(monkeypatch):
    monkeypatch.setattr(settings, "enable_agreement_gate", True)
    monkeypatch.setattr(settings, "min_sources_agreeing_gate", 3)
    assert _passes_agreement_gate("BULLISH", _sig("BULLISH", sources_agreeing=2)) is False
    assert _passes_agreement_gate("BULLISH", _sig("BULLISH", sources_agreeing=3)) is True


def test_sell_direction_matches_bearish_sig(monkeypatch):
    # The recommendation's own "direction" field (BULLISH/BEARISH), not the
    # action string (BUY/SELL) — SELL recs carry direction="BEARISH".
    monkeypatch.setattr(settings, "enable_agreement_gate", True)
    monkeypatch.setattr(settings, "min_sources_agreeing_gate", 2)
    assert _passes_agreement_gate("BEARISH", _sig("BEARISH", sources_agreeing=1)) is False
    assert _passes_agreement_gate("BEARISH", _sig("BEARISH", sources_agreeing=2)) is True


# ── wiring into the actionable-filter loop (gate_diag + gate_outcomes stamp) ──

def test_gate_diag_has_dropped_low_agreement_key():
    # gate_diag initializes the counter even when the gate never fires — the
    # dashboard/log line reads gate_diag['dropped_low_agreement'] unconditionally.
    import inspect
    import src.pipeline as pipeline
    src = inspect.getsource(pipeline.run_pipeline)
    assert '"dropped_low_agreement":' in src
    assert 'gate_diag["dropped_low_agreement"] += 1' in src
    assert '_gate_outcomes[r.ticker] = "low_agreement"' in src
