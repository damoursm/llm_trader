"""Synthesis prompt caching: cache_control split + model-aware minimum + flag."""

import src.analysis.claude_analyst as ca


def test_splits_prefix_when_enabled_and_large(monkeypatch):
    monkeypatch.setattr(ca.settings, "enable_prompt_caching", True)
    prefix = "X" * 5000                      # > 1024 tokens (~4 chars/tok)
    prompt = prefix + ca._CACHE_SENTINEL + "VARIABLE SUFFIX"
    content = ca._analyst_content(prompt, "claude-opus-4-8")
    assert isinstance(content, list) and len(content) == 2
    assert content[0]["text"] == prefix
    assert content[0]["cache_control"] == {"type": "ephemeral"}
    assert content[1]["text"] == "VARIABLE SUFFIX"
    # sentinel never reaches the model
    assert ca._CACHE_SENTINEL not in (content[0]["text"] + content[1]["text"])


def test_haiku_requires_2048_token_prefix(monkeypatch):
    monkeypatch.setattr(ca.settings, "enable_prompt_caching", True)
    prefix = "X" * 6000                       # ~1500 tok: above Opus 1024, below Haiku 2048
    prompt = prefix + ca._CACHE_SENTINEL + "SUF"
    assert isinstance(ca._analyst_content(prompt, "claude-opus-4-8"), list)      # opus caches
    out = ca._analyst_content(prompt, "claude-haiku-4-5-20251001")              # haiku: too small
    assert isinstance(out, str) and ca._CACHE_SENTINEL not in out


def test_disabled_returns_single_stripped_string(monkeypatch):
    monkeypatch.setattr(ca.settings, "enable_prompt_caching", False)
    prompt = ("X" * 5000) + ca._CACHE_SENTINEL + "SUF"
    out = ca._analyst_content(prompt, "claude-opus-4-8")
    assert out == ("X" * 5000) + "SUF"        # sentinel stripped, one block, identical content


def test_no_sentinel_passes_through(monkeypatch):
    monkeypatch.setattr(ca.settings, "enable_prompt_caching", True)
    assert ca._analyst_content("plain prompt", "claude-opus-4-8") == "plain prompt"
