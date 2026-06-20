"""Sentiment response parsing — salvages a truncated/malformed LLM reply.

A DeepSeek reply that hits the 256-token cap mid-rationale produces invalid JSON
(observed for XBI). The score is the only field the aggregator consumes, so it is
recovered by regex rather than lost to a 0.0 fallback — which matters most when
the other engine is rate-limited and can't be retried.
"""

import json

import pytest

from src.analysis.sentiment import _parse_response


def test_valid_json():
    assert _parse_response('{"score": 0.4, "rationale": "ok"}') == (0.4, "ok")


def test_markdown_fenced_json():
    assert _parse_response('```json\n{"score": -0.2, "rationale": "x"}\n```') == (-0.2, "x")


def test_score_clamped_to_unit_range():
    assert _parse_response('{"score": 2.5, "rationale": "y"}')[0] == 1.0
    assert _parse_response('{"score": -9, "rationale": "y"}')[0] == -1.0


def test_truncated_response_salvages_score():
    trunc = ('{\n  "score": 0.35,\n  "rationale": "Strong biotech momentum with '
             'several positive catalysts including FDA approvals that could drive')
    score, rationale = _parse_response(trunc)
    assert score == 0.35
    assert "biotech momentum" in rationale


def test_unrecoverable_response_raises():
    # No score anywhere → fall through to the engine fallback / 0.0 as before.
    with pytest.raises((json.JSONDecodeError, ValueError)):
        _parse_response("totally not json at all")
