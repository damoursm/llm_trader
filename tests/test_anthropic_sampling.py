"""Guard against the 'temperature is deprecated for this model' 400 (2026-06-17).

Opus 4.7+ and the Fable/Mythos 5 families removed temperature/top_p/top_k;
sending temperature 400s and silently kicks synthesis to the DeepSeek fallback.
"""

import pytest

from src.analysis.claude_analyst import _anthropic_sampling_kwargs


@pytest.mark.parametrize("model", [
    "claude-opus-4-7",
    "claude-opus-4-8",
    "claude-opus-4-8-20260101",
    "claude-opus-5-0",        # future Opus — still no sampling params
    "claude-fable-5",
    "claude-mythos-5",
])
def test_newer_models_omit_temperature(model):
    assert _anthropic_sampling_kwargs(model) == {}


@pytest.mark.parametrize("model", [
    "claude-haiku-4-5-20251001",   # the default analyst model
    "claude-sonnet-4-6",
    "claude-opus-4-6",
    "claude-opus-4-5",
    "claude-opus-4-1-20250805",
    "claude-sonnet-4-5",
])
def test_older_models_keep_temperature(model):
    assert _anthropic_sampling_kwargs(model) == {"temperature": 0}


def test_empty_or_unknown_defaults_to_temperature():
    # Unknown/empty → keep temperature (older behavior); a genuine 400 still
    # degrades safely via the engine fallback.
    assert _anthropic_sampling_kwargs("") == {"temperature": 0}
    assert _anthropic_sampling_kwargs("some-other-model") == {"temperature": 0}
