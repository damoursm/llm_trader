"""Confidence PLACEMENT rubric (synthesis prompt v2026-08-19).

The bug: the conviction rules stated the bands as bare thresholds, and the
THRESHOLD VALUES BECAME THE ANSWERS — distinct confidence values collapsed
28 -> 8 over the three days after the 2026-08-17 deploy, with 20-35% of BUY/SELL
calls at exactly 1.00 (vs 8.7% before). Because the ~1.00 bucket is precisely
the cohort that clears the ~0.89 regime-adjusted Gate-1 bar, that inflation
roughly DOUBLED the trading rate without any signal change.

These tests pin the three properties of the repair: the rubric reaches every arm
variant from ONE definition, it does not itself plant a numeric attractor, and
the collapse is detected mechanically if the model ignores it.
"""

from __future__ import annotations

import re

import pytest

import src.analysis.claude_analyst as ca


# ── the rubric itself ────────────────────────────────────────────────────────

def test_rubric_bans_the_saturating_value():
    r = ca._CONFIDENCE_PLACEMENT
    assert "NEVER output 1.00" in r
    assert "0.99" in r
    # and says WHY, so the next editor does not "simplify" it away
    assert "saturates" in r.lower()


def test_rubric_is_band_then_placement_not_a_bare_mandate():
    """v2 of the sentiment prompt proved that ordering the model to 'be precise'
    produces precise-LOOKING copies of the examples. What worked was an explicit
    procedure, so the rubric must carry one."""
    r = ca._CONFIDENCE_PLACEMENT
    assert "TWO DECIMALS" in r
    assert "lower edge" in r.lower()   # the placement start point
    for modifier in ("INDEPENDENT family", "tape structure", "catalyst",
                     "independent reporting"):
        assert modifier in r, f"named modifier missing: {modifier}"


def test_rubric_plants_no_numeric_attractor():
    """The rubric must not DEMONSTRATE a placement value — any number written
    into a prompt is a candidate modal output (the exact v2 failure).

    Two classes of numeral are allowed because neither is a worked example:
    the values the rubric FORBIDS (1.00 / 0.99 / 0.95), and the BAND EDGES
    (0.85 / 0.84 / 0.55), which are structural, already stated in the
    conviction rules above, and explicitly labelled "BOUNDARIES, not answers"
    right here. Naming the actionable band's floor is load-bearing — omitting
    it is what let the model drift to 0.76-0.82 and halt the book. Anything
    ELSE would be a demonstrated placement."""
    body = ca._CONFIDENCE_PLACEMENT
    allowed = {"1.00", "0.99", "0.95", "0.85", "0.84", "0.55"}
    nums = set(re.findall(r"\d\.\d\d", body))
    assert nums <= allowed, (
        f"rubric demonstrates confidence value(s) {nums - allowed} — these "
        "become the next modal output")
    assert "BOUNDARIES, not answers" in body, (
        "band edges are named, so the anti-anchoring counter-instruction must "
        "stay next to them")


def test_rubric_does_not_argue_from_the_true_hit_rate():
    """THE OVER-CORRECTION REGRESSION. The first version justified the 1.00 ban
    with the honest base rate ("realised hit rate is close to a coin flip") and
    the next live run put every call at 0.76-0.82 — 10 of 10 candidates dropped
    by Gate 1, zero survivors, book halted. This confidence is a CONVICTION
    SCORE gated at 0.85, not a calibrated probability: tell a model the true
    base rate and it correctly answers ~0.5, which is below every gate. The ban
    must rest on the MECHANICAL argument only."""
    r = ca._CONFIDENCE_PLACEMENT.lower()
    for forbidden in ("coin flip", "hit rate", "base rate", "realised directional"):
        assert forbidden not in r, (
            f"rubric argues from '{forbidden}' — that reasoning collapses the "
            "level below Gate 1 and halts trading")
    assert "no ranking information" in r or "carries no ranking" in r


def test_rubric_ties_the_action_to_the_band():
    """The other half of the repair: without this the model emitted BUY/SELL
    carrying monitor-band numbers, which Gate 1 then dropped wholesale."""
    r = ca._CONFIDENCE_PLACEMENT
    assert "MUST AGREE" in r
    assert "0.85 or above" in r
    assert "the action is HOLD" in r


def test_rubric_explains_the_number_ranks_calls():
    r = ca._CONFIDENCE_PLACEMENT
    assert "RANK" in r
    assert "SIZE" in r                # it drives sizing, not just the gate
    assert "must" in r and "NOT receive the same confidence" in r


# ── it reaches every arm, from one definition ────────────────────────────────

def test_every_arm_variant_uses_the_shared_rubric():
    """Three arm variants (dual-case / blind / sighted) carried three COPIES of
    the anchoring lines. A revived dead arm must not resurrect the bug, so all
    three interpolate the one constant."""
    import inspect
    src = inspect.getsource(ca.generate_recommendations)
    assert src.count("_CONFIDENCE_PLACEMENT") >= 3, (
        "each conviction-rules variant must interpolate the shared rubric")


def test_the_old_anchoring_line_is_gone_everywhere():
    import inspect
    src = inspect.getsource(ca.generate_recommendations)
    assert "A 90%+ call requires" not in src, (
        "the bare-threshold line that became the modal output is still present")


def test_bands_are_unchanged_so_gate_1_keeps_its_level():
    """The repair changes the GRANULARITY, not the LEVEL: shifting the centre of
    the distribution would move what clears Gate 1 and silently re-rate the whole
    book. The action bands must still be stated."""
    import inspect
    src = inspect.getsource(ca.generate_recommendations)
    assert "confidence ≥ 0.85" in src
    assert "confidence 0.55-0.84" in src


def test_prompt_version_marks_the_change():
    assert ca.SYNTHESIS_PROMPT_VERSION == "2026-08-19-confidence-placement"


def test_schema_emits_rationale_before_confidence():
    """Thinking is off, so JSON field order IS generation order — the number must
    come after the reasoning that justifies it (the sentiment v3 repair). The
    parser reads by KEY, so the order is free to change."""
    import inspect
    src = inspect.getsource(ca.generate_recommendations)
    i_rat = src.find('- "rationale"')
    i_conf = src.find('- "confidence"')
    assert i_rat != -1 and i_conf != -1
    assert i_rat < i_conf, "confidence is still emitted before the rationale"
    assert "strictly below 1.00" in src
    assert "float 0.0-1.0" not in src, "the schema still permits 1.0"


def test_parser_is_order_agnostic():
    """Reordering the schema is only safe because parsing is by key."""
    from datetime import datetime, timezone
    now = datetime.now(timezone.utc)
    payload = [{"rationale": "r", "confidence": 0.87, "ticker": "AAA",
                "direction": "BULLISH", "action": "BUY"}]
    out = ca._recommendations_from_data(payload, "test", now)
    assert len(out) == 1 and out[0].confidence == pytest.approx(0.87)


# ── the detector ─────────────────────────────────────────────────────────────

class _R:
    def __init__(self, c, action="BUY"):
        self.confidence, self.action = c, action


def _caplog_texts(records):
    return " ".join(r.getMessage() if hasattr(r, "getMessage") else str(r) for r in records)


def test_detector_fires_on_a_collapsed_grid(monkeypatch):
    seen = []
    monkeypatch.setattr(ca.logger, "warning", lambda m, *a, **k: seen.append(str(m)))
    # 30 calls, all at 1.00 — the exact regression signature
    ca._warn_on_degenerate_confidence([_R(1.0) for _ in range(30)], "test")
    assert seen and "DEGENERATE" in seen[0]
    assert "100%" in seen[0] or "1.0" in seen[0]


def test_detector_fires_on_too_few_distinct_values(monkeypatch):
    seen = []
    monkeypatch.setattr(ca.logger, "warning", lambda m, *a, **k: seen.append(str(m)))
    vals = [0.85, 0.86, 0.87] * 10        # only 3 distinct, none at 1.00
    ca._warn_on_degenerate_confidence([_R(v) for v in vals], "test")
    assert seen and "DEGENERATE" in seen[0]


def test_detector_is_quiet_on_a_healthy_grid(monkeypatch):
    seen = []
    monkeypatch.setattr(ca.logger, "warning", lambda m, *a, **k: seen.append(str(m)))
    vals = [0.80 + i * 0.007 for i in range(30)]      # 30 distinct, none at 1.00
    ca._warn_on_degenerate_confidence([_R(v) for v in vals], "test")
    assert not seen


def test_detector_ignores_small_samples(monkeypatch):
    seen = []
    monkeypatch.setattr(ca.logger, "warning", lambda m, *a, **k: seen.append(str(m)))
    ca._warn_on_degenerate_confidence([_R(1.0) for _ in range(5)], "test")
    assert not seen, "a handful of calls says nothing about the grid's shape"


def test_detector_only_judges_actionable_calls(monkeypatch):
    seen = []
    monkeypatch.setattr(ca.logger, "warning", lambda m, *a, **k: seen.append(str(m)))
    # HOLD/WATCH rows are not what Gate 1 or sizing consume
    ca._warn_on_degenerate_confidence([_R(1.0, "HOLD") for _ in range(50)], "test")
    assert not seen


def test_detector_never_breaks_the_tick(monkeypatch):
    seen = []
    monkeypatch.setattr(ca.logger, "warning", lambda m, *a, **k: seen.append(str(m)))

    class _Bad:
        action = "BUY"
        @property
        def confidence(self):
            raise RuntimeError("boom")
    ca._warn_on_degenerate_confidence([_Bad() for _ in range(30)], "test")  # must not raise


def test_detector_thresholds_are_constants_not_settings():
    """A knob that silences a detector is a knob that eventually gets turned."""
    from config.settings import Settings
    for name in ("conf_max_ones_frac", "conf_min_distinct", "conf_min_calls"):
        assert name not in Settings.model_fields
    assert isinstance(ca._CONF_MAX_ONES_FRAC, float)
    assert isinstance(ca._CONF_MIN_DISTINCT, int)
