"""Dual-case synthesis prompt (2026-07-25).

Presents the BULL CASE and BEAR CASE side by side, each built from its OWN
camp's vetted method set, and lets the model weigh them.

Why it exists (measured, not assumed):
  * the per-side win-rate filter gives the two camps DIFFERENT valid method
    sets, but the prompt filtered methods GLOBALLY only -- so the model saw
    `news` while considering a BUY, on a method the system had decided must not
    inform buying. This is the same leak class fixed in the combine/coherence.
  * it protects the cohort where the LLM demonstrably adds value: when the
    aggregator is NEUTRAL and the model supplies the direction, its BUY calls
    returned +0.21% / 66% win vs -3.19% / 33% when echoing a bullish aggregator
    (same side, so the long/short asymmetry is controlled for).
It is NOT expected to move the echo rate -- the blind A/B moved it only
94.5% -> 91.4%, i.e. the model echoes because it reads the same scores.

All fakes, no network.
"""

import json

import pytest

from config.settings import settings
import src.analysis.claude_analyst as ca
import src.signals.aggregator as agg
from src.models import TickerSignal


@pytest.fixture
def captured(monkeypatch):
    """Capture the prompt and stub every synthesis engine."""
    box = {}

    def fake(prompt, **kw):
        box["prompt"] = prompt
        return json.dumps([{"ticker": "AAA", "type": "STOCK", "direction": "BULLISH",
                            "action": "WATCH", "confidence": 0.5, "rationale": "x"}])

    for name in ("_call_claude_analyst", "_call_deepseek_analyst", "_call_qwen_analyst"):
        monkeypatch.setattr(ca, name, fake)
    monkeypatch.setattr(settings, "llm_ab_synthesis_models", "deepseek-v4-flash")
    return box


def _sig(**kw):
    base = dict(ticker="AAA", direction="BULLISH", confidence=0.91,
                combined_score=0.42, combined_buy_score=0.55,
                combined_sell_score=0.13, sources_agreeing=3,
                sentiment_score=0.0, technical_score=0.0, rationale="coverage")
    base.update(kw)
    return TickerSignal(**base)


def _no_side_filter(monkeypatch):
    monkeypatch.setattr(agg, "side_filtered_methods", lambda side: frozenset())
    monkeypatch.setattr(agg, "winrate_filtered_methods", lambda: frozenset())


# ── the two-case presentation ───────────────────────────────────────────────

def test_dual_case_shows_both_camps_and_no_verdict(captured, monkeypatch):
    _no_side_filter(monkeypatch)
    ca.generate_recommendations([_sig()], dual_case=True)
    p = captured["prompt"]
    assert "BULL CASE conviction=0.55" in p
    assert "BEAR CASE conviction=0.13" in p
    # The aggregator's own verdict must NOT be handed over — that is the whole
    # point of presenting both cases symmetrically.
    assert "direction=BULLISH" not in p
    assert "combined_confidence" not in p
    assert "neither is a recommendation" in p


def test_sighted_still_shows_the_verdict(captured, monkeypatch):
    """Baseline integrity: the existing arm is unchanged."""
    _no_side_filter(monkeypatch)
    ca.generate_recommendations([_sig()], dual_case=False, blind_synthesis=False)
    p = captured["prompt"]
    assert "direction=BULLISH" in p and "combined_confidence" in p
    assert "BULL CASE conviction" not in p


def test_blind_arm_is_unaffected(captured, monkeypatch):
    _no_side_filter(monkeypatch)
    ca.generate_recommendations([_sig()], dual_case=False, blind_synthesis=True)
    p = captured["prompt"]
    assert "direction=BULLISH" not in p
    assert "BULL CASE conviction" not in p


# ── the per-side method gate (the incoherence this fixes) ───────────────────

def test_dual_case_hides_a_method_on_the_side_it_is_filtered_from(captured, monkeypatch):
    """news is excluded from the BULL camp: its BULLISH score must not be shown
    (the model would otherwise lean on it for a BUY), while its BEARISH score
    must still be, because its sell side is kept."""
    monkeypatch.setattr(agg, "winrate_filtered_methods", lambda: frozenset())
    monkeypatch.setattr(agg, "side_filtered_methods",
                        lambda side: frozenset({"news"}) if side == "buy" else frozenset())

    ca.generate_recommendations([_sig(sentiment_score=0.60)], dual_case=True)
    assert "News sentiment" not in captured["prompt"], (
        "a bullish score from a buy-filtered method must be hidden")

    ca.generate_recommendations([_sig(sentiment_score=-0.60)], dual_case=True)
    assert "News sentiment" in captured["prompt"], (
        "the same method's BEARISH score must still be shown — its sell side is kept")


def test_other_arms_ignore_the_per_side_filter(captured, monkeypatch):
    """Only the dual-case arm applies it: in the sighted/blind arms the model
    picks the direction AFTER seeing the evidence, so hiding a method by the
    side it happens to point today would be wrong."""
    monkeypatch.setattr(agg, "winrate_filtered_methods", lambda: frozenset())
    monkeypatch.setattr(agg, "side_filtered_methods",
                        lambda side: frozenset({"news"}) if side == "buy" else frozenset())
    ca.generate_recommendations([_sig(sentiment_score=0.60)], dual_case=False)
    assert "News sentiment" in captured["prompt"]


def test_global_filter_still_hides_a_method_in_dual_case(captured, monkeypatch):
    """The global sub-50% filter is unconditional — it outranks the side gate."""
    monkeypatch.setattr(agg, "winrate_filtered_methods", lambda: frozenset({"news"}))
    monkeypatch.setattr(agg, "side_filtered_methods", lambda side: frozenset())
    ca.generate_recommendations([_sig(sentiment_score=-0.60)], dual_case=True)
    assert "News sentiment" not in captured["prompt"]


# ── instructions ────────────────────────────────────────────────────────────

def test_dual_case_instructions_allow_declining_a_direction(captured, monkeypatch):
    """The both-weak / contested outcome must be expressible — it is the state
    the single-verdict format could not represent."""
    _no_side_filter(monkeypatch)
    ca.generate_recommendations([_sig()], dual_case=True)
    p = captured["prompt"]
    assert "CONTESTED" in p
    assert "Declining to call a direction is a correct and expected output" in p
    # And it must NOT tell the model to trust a pre-computed number.
    assert "Trust it" not in p
