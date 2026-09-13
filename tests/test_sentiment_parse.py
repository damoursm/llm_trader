"""Sentiment response parsing — salvages a truncated/malformed LLM reply.

A DeepSeek reply that hits the 256-token cap mid-rationale produces invalid JSON
(observed for XBI). The score is the only field the aggregator consumes, so it is
recovered by regex rather than lost to a 0.0 fallback — which matters most when
the other engine is rate-limited and can't be retried.
"""

import json

import pytest

import src.analysis.sentiment as sent
from src.analysis.sentiment import _parse_response


def test_valid_json():
    # 3-tuple since 2026-08-15 (catalyst field); a reply without one → None.
    assert _parse_response('{"score": 0.4, "rationale": "ok"}') == (0.4, "ok", None)


def test_markdown_fenced_json():
    assert _parse_response('```json\n{"score": -0.2, "rationale": "x"}\n```') == (-0.2, "x", None)


def test_score_clamped_to_unit_range():
    assert _parse_response('{"score": 2.5, "rationale": "y"}')[0] == 1.0
    assert _parse_response('{"score": -9, "rationale": "y"}')[0] == -1.0


def test_truncated_response_salvages_score():
    trunc = ('{\n  "score": 0.35,\n  "rationale": "Strong biotech momentum with '
             'several positive catalysts including FDA approvals that could drive')
    score, rationale, _catalyst = _parse_response(trunc)
    assert score == 0.35
    assert "biotech momentum" in rationale


def test_unrecoverable_response_raises():
    # No score anywhere → fall through to the engine fallback / 0.0 as before.
    with pytest.raises((json.JSONDecodeError, ValueError)):
        _parse_response("totally not json at all")


# ── non-finite scores (2026-09-11) ──────────────────────────────────────────

def test_a_NaN_score_is_REFUSED_not_read_as_maximum_conviction():
    """`json.loads` accepts bare `NaN` and `Infinity` by default, and the
    obvious clamp is silently wrong on both: every comparison against NaN is
    False, so `max(-1.0, min(1.0, nan))` returns **+1.0**.

    That is a malformed answer arriving as the strongest possible BULLISH
    verdict in the 0.40-weight `news` method — and in a rank-consumed combine a
    +1.0 puts the ticker at the top of the cross-section, where it can open a
    trade. Refusing hands the ticker to the next engine, which is what every
    other unreadable response already does."""
    for bad in ("NaN", "Infinity", "-Infinity"):
        with pytest.raises(Exception):
            sent._parse_response(
                '{"rationale": "r", "catalyst": "guidance", "score": %s}' % bad)


def test_the_clamp_still_works_on_ordinary_out_of_range_numbers():
    """The refusal must be narrow: a model that overshoots the scale is giving a
    readable answer and is still clamped, not thrown away."""
    assert sent._parse_response(
        '{"rationale": "r", "catalyst": "guidance", "score": 4.2}')[0] == pytest.approx(1.0)
    assert sent._parse_response(
        '{"rationale": "r", "catalyst": "guidance", "score": -4.2}')[0] == pytest.approx(-1.0)


def test_the_salvage_path_cannot_readmit_a_non_finite_score():
    r"""The JSON branch raises INSIDE the try, so a NaN falls through to the
    salvage regex — which must not rescue it. `[-+]?\d+` cannot match `NaN`,
    so the bare `raise` fires; this pins that the regex is never loosened into
    matching one."""
    with pytest.raises(Exception):
        sent._parse_response('{"rationale": "cut, "catalyst": "analyst", "score": NaN')
