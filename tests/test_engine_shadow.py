"""Paired synthesis engines and the A/B that decides which one trades (2026-09-04).

The final synthesis is the call that actually opens trades, and until now it
had exactly one engine per run. This module pins the machinery that makes it
two: every run's LIVE decision is routed by a per-run flip
(`synthesis_local_share`), the OTHER engine is asked for its own decision on
the same signals in the background (`src/analysis/engine_shadow.py`), and both
sets of per-ticker decisions are persisted to `engine_recommendations` so the
two can be compared as DECIDERS on the pivot basis later.

Four properties carry the design, and three of them are silent when they break:

  * with `enable_local_llm` off the routing is byte-identical to before the
    local engine existed — even at `synthesis_local_share=1.0`. A new engine
    must not perturb production until it is deliberately switched on;
  * the local call sends the LOCAL server's no-reasoning dialect, never the
    hosted one, and is REFUSED when the server silently truncated the prompt.
    Ollama drops the OLDEST tokens (the persona and the process block) without
    an error when a request exceeds its context, so a verdict produced that way
    is not this engine's decision and must never enter the paired dataset as
    one — the refusal is what makes an under-sized server context loud;
  * the shadow arms never touch the live decision: they run on a background
    thread, a failure records nothing (rather than a fabricated call), and the
    rows they queue are drained non-blocking;
  * `engine_recommendations` is keyed on (run, engine, prompt_variant, ticker).
    The variant is part of the key because a `deepseek:compact` decomposition
    arm runs BESIDE the live `deepseek:full` row, and a key without it would
    make the shadow drain delete the live decision it is supposed to be
    compared against.

All synthetic: no server, no network, no model.
"""

import json
import threading
from types import SimpleNamespace

import pandas as pd
import pytest

from config.settings import settings
import src.analysis.claude_analyst as ca
import src.analysis.engine_shadow as es
import src.analysis.local_llm as local_llm
import src.analysis.engine_eval as ee
import src.signals.aggregator as agg
import src.pipeline as pl
from src.db import repo
from src.models import TickerSignal

_VALID = json.dumps([{"ticker": "AAA", "type": "STOCK", "direction": "BULLISH",
                      "action": "WATCH", "confidence": 0.5, "rationale": "x"}])


@pytest.fixture(autouse=True)
def _reset_shadow_state(monkeypatch):
    """Module globals: the queued rows and the in-flight branch count."""
    monkeypatch.setattr(es, "_ROWS", [])
    monkeypatch.setattr(es, "_PENDING", 0)
    monkeypatch.setattr(ca, "_local_analyst_client", None)
    monkeypatch.setattr(settings, "synthesis_shadow_engine", "auto")
    monkeypatch.setattr(settings, "synthesis_shadow_extra", "")
    monkeypatch.setattr(settings, "synthesis_shadow_max_pending", 2)


def _sig(ticker="AAA", **kw):
    base = dict(ticker=ticker, direction="BULLISH", confidence=0.91,
                combined_score=0.42, combined_buy_score=0.55,
                combined_sell_score=0.13, sources_agreeing=3,
                sentiment_score=0.0, technical_score=0.0, rationale="coverage",
                price=100.0)
    base.update(kw)
    return TickerSignal(**base)


def _no_side_filter(monkeypatch):
    monkeypatch.setattr(agg, "side_filtered_methods", lambda side: frozenset())
    monkeypatch.setattr(agg, "winrate_filtered_methods", lambda: frozenset())


@pytest.fixture
def engines(monkeypatch):
    """Stub every synthesis engine; record the prompt each one was sent."""
    seen = {}

    def _mk(name):
        def fake(prompt, **kw):
            seen.setdefault(name, []).append(prompt)
            return _VALID
        return fake

    monkeypatch.setattr(ca, "_call_claude_analyst", _mk("anthropic"))
    monkeypatch.setattr(ca, "_call_deepseek_analyst", _mk("deepseek"))
    monkeypatch.setattr(ca, "_call_qwen_analyst", _mk("qwen"))
    monkeypatch.setattr(ca, "_call_local_analyst", _mk("local"))
    monkeypatch.setattr(settings, "llm_ab_synthesis_models", "deepseek-v4-flash")
    return seen


def _enable_local(monkeypatch):
    monkeypatch.setattr(settings, "enable_local_llm", True)
    monkeypatch.setattr(settings, "local_sentiment_base_url", "http://127.0.0.1:11434/v1")
    monkeypatch.setattr(settings, "local_sentiment_model", "qwen3:8b")


# ── 1. default-off is a provable no-op ────────────────────────────────────────

def test_local_synthesis_disabled_never_routes_even_at_full_share(engines, monkeypatch):
    """The share is not the switch — `enable_local_llm` is. A disabled engine
    must not be reachable by a setting that looks like it should reach it."""
    _no_side_filter(monkeypatch)
    monkeypatch.setattr(settings, "enable_local_llm", False)
    monkeypatch.setattr(settings, "synthesis_local_share", 1.0)

    ca.generate_recommendations([_sig()])

    assert "local" not in engines
    assert engines["deepseek"]
    assert ca._engine_of(ca.get_last_synthesis_meta()["model"]) == "deepseek"


def test_disabled_local_is_absent_from_every_attempt_chain(monkeypatch):
    monkeypatch.setattr(settings, "enable_local_llm", False)
    assert not ca._local_synthesis_available()
    chain = ca._synthesis_attempts_for("deepseek-v4-flash", "claude-x", "deepseek-v4-flash", "qwen-x")
    assert [e for e, _ in chain] == ["deepseek", "qwen", "anthropic"]
    assert pl.hold_review_fallbacks("deepseek") == ["qwen", "anthropic"]


def test_enabled_local_joins_the_chains_last(monkeypatch):
    """As a FALLBACK the local engine goes after every hosted one: it is the
    tier that answers when the hosted accounts go unfunded together (the
    2026-09-01 outage), not a preferred engine."""
    _enable_local(monkeypatch)
    chain = ca._synthesis_attempts_for("deepseek-v4-flash", "claude-x", "deepseek-v4-flash", "qwen-x")
    assert [e for e, _ in chain] == ["deepseek", "qwen", "anthropic", "local"]
    assert pl.hold_review_fallbacks("deepseek") == ["qwen", "anthropic", "local"]
    # As the CHOSEN engine it leads and the hosted engines follow it.
    chain = ca._synthesis_attempts_for("local/qwen3:8b", "claude-x", "deepseek-v4-flash", "qwen-x")
    assert [e for e, _ in chain] == ["local", "deepseek", "qwen", "anthropic"]


def test_local_model_id_is_its_own_namespace(monkeypatch):
    """`local/qwen3:8b` contains 'qwen': the prefix must resolve FIRST or every
    provenance column would label the self-hosted model as the hosted one."""
    _enable_local(monkeypatch)
    assert ca._local_synthesis_model() == "local/qwen3:8b"
    assert ca._engine_of("local/qwen3:8b") == "local"
    assert ca.forced_synthesis_model("local") == "local/qwen3:8b"
    assert ca.forced_synthesis_model("deepseek") == ca._DEEPSEEK_ANALYST_MODEL
    assert ca.forced_synthesis_model("anthropic") == settings.analyst_model
    assert ca._engine_of(ca.forced_synthesis_model("qwen")) == "qwen"


# ── 2. the per-run A/B routes the LIVE decision ───────────────────────────────

def test_local_share_routes_the_live_decision_to_local(engines, monkeypatch):
    _no_side_filter(monkeypatch)
    _enable_local(monkeypatch)
    monkeypatch.setattr(settings, "synthesis_local_share", 1.0)
    monkeypatch.setattr(ca.random, "random", lambda: 0.0)

    recs = ca.generate_recommendations([_sig()])

    assert recs
    assert engines["local"] and "deepseek" not in engines
    meta = ca.get_last_synthesis_meta()
    assert meta["model"] == "local/qwen3:8b"
    assert ca._engine_of(meta["model"]) == "local"


def test_local_share_zero_keeps_the_hosted_pool(engines, monkeypatch):
    _no_side_filter(monkeypatch)
    _enable_local(monkeypatch)
    monkeypatch.setattr(settings, "synthesis_local_share", 0.0)
    monkeypatch.setattr(ca.random, "random", lambda: 0.0)   # below 0.0 is impossible

    ca.generate_recommendations([_sig()])

    assert "local" not in engines
    assert engines["deepseek"]


def test_local_gets_the_compact_prompt_and_hosted_gets_the_full_one(engines, monkeypatch):
    """The whole point of the compact variant: the full prompt (p50 ~60k tokens
    measured on DeepSeek's usage) cannot fit an 8B model's context on this GPU."""
    _no_side_filter(monkeypatch)
    _enable_local(monkeypatch)
    monkeypatch.setattr(settings, "synthesis_local_share", 1.0)
    monkeypatch.setattr(ca.random, "random", lambda: 0.0)

    ca.generate_recommendations([_sig()])
    local_prompt = engines["local"][0]

    monkeypatch.setattr(settings, "synthesis_local_share", 0.0)
    ca.generate_recommendations([_sig()])
    hosted_prompt = engines["deepseek"][0]

    assert len(local_prompt) < len(hosted_prompt)
    assert ca._COMPACT_PERSONA.splitlines()[0] in local_prompt
    # Same decision framing, same output contract — only the macro/per-method
    # blocks are gone, so the two decisions stay comparable.
    for field in ('"ticker"', '"action"', '"confidence"', '"time_horizon"', '"rationale"'):
        assert field in local_prompt and field in hosted_prompt
    assert "<selection_process>" in local_prompt


def test_prompt_variant_overrides_the_engine_default(engines, monkeypatch):
    """`prompt_variant` is what decomposes engine-vs-prompt: a `deepseek:compact`
    shadow arm answers whether a difference came from the model or the prompt."""
    _no_side_filter(monkeypatch)
    _enable_local(monkeypatch)
    monkeypatch.setattr(settings, "synthesis_local_share", 0.0)

    ca.generate_recommendations([_sig()], prompt_variant="compact")
    assert ca._COMPACT_PERSONA.splitlines()[0] in engines["deepseek"][0]

    ca.generate_recommendations([_sig()], force_engine="local", prompt_variant="full")
    assert ca._COMPACT_PERSONA.splitlines()[0] not in engines["local"][0]


# ── 3. the pinned (hold-review) branch is unchanged by the new engine ─────────

def test_pinned_local_is_one_attempt_and_never_fabricates(monkeypatch):
    """A pinned engine gets ONE shot, no cross-engine fallback and no
    rule-based fill — the caller reads [] as 'no review this tick' (hold)."""
    _no_side_filter(monkeypatch)
    _enable_local(monkeypatch)
    calls = []

    def boom(prompt, **kw):
        calls.append("local")
        raise RuntimeError("local box is cold")

    def hosted(prompt, **kw):
        calls.append("hosted")
        return _VALID

    monkeypatch.setattr(ca, "_call_local_analyst", boom)
    monkeypatch.setattr(ca, "_call_deepseek_analyst", hosted)
    monkeypatch.setattr(ca, "_call_qwen_analyst", hosted)
    monkeypatch.setattr(ca, "_call_claude_analyst", hosted)
    monkeypatch.setattr(settings, "llm_transient_retries", 0)
    ca._set_synthesis_meta("deepseek", "deepseek-v4-flash")

    assert ca.generate_recommendations([_sig()], force_engine="local") == []
    assert calls == ["local"]
    # The failed pin must not clobber the run's own synthesis provenance.
    assert ca.get_last_synthesis_meta()["model"] == "deepseek-v4-flash"


def test_pinned_local_is_not_coerced_to_qwen(engines, monkeypatch):
    """The legacy 'coerce every pin to qwen' path folds the hosted Qwen route
    under one provider name; the local engine is its own provider by design."""
    _no_side_filter(monkeypatch)
    _enable_local(monkeypatch)
    monkeypatch.setattr(settings, "llm_primary_provider", "qwen")

    ca.generate_recommendations([_sig()], force_engine="local")
    assert "local" in engines and "qwen" not in engines


# ── 4. the local call itself ──────────────────────────────────────────────────

class _FakeCompletions:
    def __init__(self, sink, content, prompt_tokens):
        self._sink, self._content, self._pt = sink, content, prompt_tokens

    def create(self, **kw):
        self._sink.append(kw)
        pt = self._pt
        if pt is None:                      # a HEALTHY server: it read the whole prompt
            prompt = kw["messages"][0]["content"]
            pt = local_llm.estimate_tokens(prompt)
        return SimpleNamespace(
            usage=SimpleNamespace(prompt_tokens=pt, completion_tokens=10),
            choices=[SimpleNamespace(message=SimpleNamespace(content=self._content))],
        )


# `prompt_tokens=None` = what a server that did NOT truncate would report, derived
# from the prompt it was actually sent. A fixed number cannot play that role: the
# truncation guards compare the count against both the prompt estimate and the
# server context, so a constant is either a truncation or an over-count for some
# prompt size, and the tests that are not about truncation would fail for a reason
# they do not test. Pass an explicit count to simulate a truncating server.
def _fake_client(sink, content, prompt_tokens=None):
    return SimpleNamespace(chat=SimpleNamespace(
        completions=_FakeCompletions(sink, content, prompt_tokens)))


def test_local_call_sends_the_local_dialect_only(monkeypatch):
    """A hosted reasoning knob is either a 400 or — worse — silently ignored,
    and with reasoning left ON the model spends its whole answer budget on the
    chain and returns content="" (a lost decision)."""
    import src.analysis.sentiment as sent
    _enable_local(monkeypatch)
    sink = []
    monkeypatch.setattr(ca, "_local_analyst_client",
                        _fake_client(sink, "<think>weighing</think>" + _VALID))

    out = ca._call_local_analyst("PROMPT", model="local/qwen3:8b")

    assert out == _VALID                       # the leaked think block is stripped
    kw = sink[0]
    assert kw["model"] == "qwen3:8b"           # the local/ prefix is an internal id
    assert kw["temperature"] == 0
    assert kw["max_tokens"] == int(settings.local_synthesis_max_output_tokens)
    assert kw["extra_body"] == sent._local_extra_body()
    for hosted_knob in ("thinking", "enable_thinking", "reasoning"):
        assert hosted_knob not in kw


def test_local_call_refuses_a_silently_truncated_prompt(monkeypatch):
    """Ollama truncates the OLDEST tokens (the persona, the process block)
    without an error when the request exceeds its context. A decision made
    without ever seeing the instructions is not this engine's decision."""
    _enable_local(monkeypatch)
    sink = []
    prompt = "x" * 60_000                       # ~18.7k estimated tokens
    monkeypatch.setattr(ca, "_local_analyst_client",
                        _fake_client(sink, _VALID, prompt_tokens=2050))
    # The pre-flight is switched off here on purpose: this is the SECOND net,
    # the one that catches a server context smaller than the setting claims (or
    # a chars-per-token estimate that is simply wrong).
    monkeypatch.setattr(settings, "local_sentiment_context_tokens", 0)

    with pytest.raises(RuntimeError, match="truncated"):
        ca._call_local_analyst(prompt)


def test_local_call_refuses_an_empty_answer(monkeypatch):
    _enable_local(monkeypatch)
    monkeypatch.setattr(ca, "_local_analyst_client", _fake_client([], "<think>...</think>"))
    with pytest.raises(RuntimeError):
        ca._call_local_analyst("PROMPT")


def test_local_call_raises_when_the_engine_is_disabled(monkeypatch):
    monkeypatch.setattr(settings, "enable_local_llm", False)
    monkeypatch.setattr(ca, "_local_analyst_client", None)
    with pytest.raises(RuntimeError):
        ca._call_local_analyst("PROMPT")


def test_a_truncated_local_run_falls_through_to_a_hosted_engine(monkeypatch):
    """Fail-closed, not fail-quiet: the refusal must cost the local ARM its
    row, never cost the run its decision."""
    _no_side_filter(monkeypatch)
    _enable_local(monkeypatch)
    monkeypatch.setattr(settings, "synthesis_local_share", 1.0)
    monkeypatch.setattr(ca.random, "random", lambda: 0.0)
    monkeypatch.setattr(settings, "llm_transient_retries", 0)
    monkeypatch.setattr(ca, "_local_analyst_client",
                        _fake_client([], _VALID, prompt_tokens=2050))
    hosted = []
    monkeypatch.setattr(ca, "_call_deepseek_analyst",
                        lambda prompt, **kw: hosted.append(prompt) or _VALID)

    recs = ca.generate_recommendations([_sig()])

    assert recs
    assert hosted and ca._COMPACT_PERSONA.splitlines()[0] not in hosted[0]
    assert ca._engine_of(ca.get_last_synthesis_meta()["model"]) == "deepseek"


def test_a_dead_local_box_falls_through(engines, monkeypatch):
    _no_side_filter(monkeypatch)
    _enable_local(monkeypatch)
    monkeypatch.setattr(settings, "synthesis_local_share", 1.0)
    monkeypatch.setattr(ca.random, "random", lambda: 0.0)
    monkeypatch.setattr(settings, "llm_transient_retries", 0)

    def dead(prompt, **kw):
        raise ConnectionError("connection refused")

    monkeypatch.setattr(ca, "_call_local_analyst", dead)
    assert ca.generate_recommendations([_sig()])
    assert engines["deepseek"]


def test_a_prompt_larger_than_the_server_context_is_refused_before_the_call(monkeypatch):
    """The post-call check already refuses a truncated answer, but only after
    minutes of prefill+decode — once per tick, for a row that is discarded. The
    pre-flight makes the refusal free, which is what stops the local arm burning
    a generation every tick while the server context is too small."""
    _enable_local(monkeypatch)
    sink = []
    monkeypatch.setattr(ca, "_local_analyst_client", _fake_client(sink, _VALID))
    monkeypatch.setattr(settings, "local_sentiment_context_tokens", 8192)

    with pytest.raises(RuntimeError, match="server context"):
        ca._call_local_analyst("x" * 40_000)     # ~12.5k estimated tokens
    assert sink == []                            # nothing was sent

    # Raise the server context and the same prompt goes through.
    monkeypatch.setattr(settings, "local_sentiment_context_tokens", 32768)
    assert ca._call_local_analyst("x" * 40_000) == _VALID
    assert len(sink) == 1


def test_the_context_preflight_is_disabled_by_zero(monkeypatch):
    _enable_local(monkeypatch)
    sink = []
    monkeypatch.setattr(ca, "_local_analyst_client", _fake_client(sink, _VALID))
    monkeypatch.setattr(settings, "local_sentiment_context_tokens", 0)
    assert ca._call_local_analyst("x" * 40_000) == _VALID
    assert len(sink) == 1


def test_the_synthesis_route_has_its_own_local_config(monkeypatch):
    """Ollama's context, parallelism and KV type are SERVER-WIDE and the two
    jobs want opposite settings — ~68 short sentiment calls a tick that want
    parallel slots, one very long synthesis call that wants context. So the
    synthesis route can name its own endpoint, model, dialect and context, and
    inherits the sentiment ones only while they are unset."""
    _enable_local(monkeypatch)
    # Inheritance: the single-server default in force today.
    assert ca._local_synthesis_base_url() == settings.local_sentiment_base_url
    assert ca._local_synthesis_model() == "local/qwen3:8b"
    assert ca._local_synthesis_context_tokens() == settings.local_sentiment_context_tokens

    monkeypatch.setattr(settings, "local_synthesis_base_url", "http://127.0.0.1:11435/v1")
    monkeypatch.setattr(settings, "local_synthesis_model", "qwen3-synth")
    monkeypatch.setattr(settings, "local_synthesis_context_tokens", 32768)
    assert ca._local_synthesis_base_url() == "http://127.0.0.1:11435/v1"
    # The id is provenance: `runs.llm_synthesis_provider` and the trade stamp
    # must show WHICH local model decided, not just "local".
    assert ca._local_synthesis_model() == "local/qwen3-synth"
    assert ca.forced_synthesis_model("local") == "local/qwen3-synth"
    assert ca._local_synthesis_context_tokens() == 32768


def test_the_synthesis_dialect_inherits_then_overrides_and_fails_soft(monkeypatch):
    import src.analysis.sentiment as sent
    _enable_local(monkeypatch)
    assert ca._local_synthesis_extra_body() == sent._local_extra_body()

    monkeypatch.setattr(settings, "local_synthesis_extra_body", '{"reasoning_effort": "low"}')
    assert ca._local_synthesis_extra_body() == {"reasoning_effort": "low"}

    # A malformed knob must not take the engine down with it.
    monkeypatch.setattr(settings, "local_synthesis_extra_body", "{not json")
    assert ca._local_synthesis_extra_body() == {}


def test_a_second_endpoint_is_what_the_client_connects_to(monkeypatch):
    _enable_local(monkeypatch)
    monkeypatch.setattr(settings, "local_synthesis_base_url", "http://127.0.0.1:11435/v1")
    monkeypatch.setattr(ca, "_local_analyst_client", None)

    client = ca._get_local_analyst_client()
    assert str(client.base_url).rstrip("/") == "http://127.0.0.1:11435/v1"
    # And the sentiment route is untouched by that: separate client, separate URL.
    assert settings.local_sentiment_base_url == "http://127.0.0.1:11434/v1"


def test_the_synthesis_context_is_read_from_its_own_setting(monkeypatch):
    """The pre-flight must judge the SYNTHESIS server, not the sentiment one —
    pointing synthesis at a 32k endpoint while sentiment stays at 8192 is the
    whole reason the two configs are separate."""
    _enable_local(monkeypatch)
    sink = []
    monkeypatch.setattr(ca, "_local_analyst_client", _fake_client(sink, _VALID))
    monkeypatch.setattr(settings, "local_sentiment_context_tokens", 8192)
    monkeypatch.setattr(settings, "local_synthesis_context_tokens", 32768)

    assert ca._call_local_analyst("x" * 40_000) == _VALID     # ~12.5k tok: fits 32768
    assert len(sink) == 1

# ── 5. the compact prompt's token budget ─────────────────────────────────────

def _compact(max_tokens, n=200):
    lines = [f"[{i:03d}] TICK{i:03d} — combined_score=0.4 confidence=0.9" for i in range(n)]
    return ca._compact_synthesis_prompt(
        methods_desc="news, tech", session_block="", date_line="Today's date: 2026-09-04",
        open_positions_block="", open_positions_instructions="",
        signal_lines=lines, skipped=3, agreement_instruction="",
        conviction_rules="3. Conviction rules: place the confidence.",
        tech_instructions="", commodity_tickers="GLD", max_tokens=max_tokens), lines


def test_compact_prompt_drops_the_lowest_ranked_blocks_not_the_framing():
    """`signal_lines` is in shortlist-key order, so the tail is what the full
    prompt would have listed last. The framing must never be what goes — Ollama
    would otherwise drop the persona itself."""
    framing, _ = _compact(None, n=0)
    floor = ca._estimate_local_tokens(framing[0])
    budget = floor + 400
    (text, dropped), lines = _compact(budget)

    assert 0 < dropped < 200                      # some kept, the tail gone
    assert ca._estimate_local_tokens(text) <= budget
    assert lines[0] in text and lines[-1] not in text
    assert ca._COMPACT_PERSONA.splitlines()[0] in text
    assert "<selection_process>" in text
    assert '"confidence"' in text
    assert f"[{3 + dropped} additional tickers omitted" in text


def test_an_impossible_budget_still_keeps_the_framing():
    """The floor is the framing itself. Dropping INTO it would send the model a
    ticker list with no instructions — the exact state the truncation refusal
    in `_call_local_analyst` exists to make loud."""
    (text, dropped), lines = _compact(200)

    assert dropped == 200
    assert ca._COMPACT_PERSONA.splitlines()[0] in text
    assert '"confidence"' in text


def test_compact_prompt_keeps_everything_under_a_generous_budget():
    (text, dropped), lines = _compact(100_000)
    assert dropped == 0
    assert lines[-1] in text


# ── 6. the shadow branch ─────────────────────────────────────────────────────

def _rec(ticker="AAA", action="BUY", conf=0.9, rule_filled=False):
    return SimpleNamespace(ticker=ticker, action=action,
                           direction=SimpleNamespace(value="BULLISH"),
                           confidence=conf, time_horizon="SWING",
                           rationale="because", rule_filled=rule_filled)


def _boom(signals, **kw):
    raise RuntimeError("boom")


def _none(signals, **kw):
    return []


def test_resolve_variant_defaults_compact_for_local_only():
    assert es.resolve_variant("local") == "compact"
    assert es.resolve_variant("deepseek") == "full"
    assert es.resolve_variant("local", "full") == "full"
    assert es.resolve_variant("deepseek", "compact") == "compact"


def test_rows_for_flattens_a_decision_set():
    rows = es.rows_for(engine="local", model="local/qwen3:8b", prompt_variant="compact",
                       live=False, recs=[_rec("AAA"), _rec("BBB", "SELL", rule_filled=True),
                                         SimpleNamespace(ticker=None)],
                       prices={"AAA": 12.5}, latency_s=41.0, n_signals=7,
                       run_id="R1", generated_at="2026-09-04T14:00:00+00:00",
                       signal_date="2026-09-04")

    assert len(rows) == 2                       # the ticker-less rec is skipped
    a, b = rows
    assert a["ticker"] == "AAA" and a["snap_price"] == 12.5 and a["rule_filled"] is False
    assert a["direction"] == "BULLISH"          # the enum is unwrapped for the DB
    assert a["live"] is False and a["engine"] == "local" and a["prompt_variant"] == "compact"
    assert a["n_recs"] == 2 and a["n_signals"] == 7 and a["latency_s"] == 41.0   # n_recs counts what was persisted
    assert b["rule_filled"] is True and b["snap_price"] is None


def test_shadow_arms_auto_pairs_the_two_engines_of_the_ab(monkeypatch):
    _enable_local(monkeypatch)
    assert es.shadow_arms("deepseek") == [("local", "auto")]
    assert es.shadow_arms("local") == [("deepseek", "auto")]
    # A rule-based run had no engine at all — still pair it against local.
    assert es.shadow_arms(None) == [("local", "auto")]


def test_shadow_arms_drops_local_while_the_engine_is_off(monkeypatch):
    monkeypatch.setattr(settings, "enable_local_llm", False)
    assert es.shadow_arms("deepseek") == []


def test_shadow_arms_never_re_runs_the_live_arm(monkeypatch):
    """The live rows already carry that answer; paying for it twice would also
    make the pair a self-comparison."""
    _enable_local(monkeypatch)
    monkeypatch.setattr(settings, "synthesis_shadow_engine", "deepseek")
    assert es.shadow_arms("deepseek") == []
    # Same engine, DIFFERENT prompt: that is a real decomposition arm, kept.
    monkeypatch.setattr(settings, "synthesis_shadow_extra", "deepseek:compact")
    assert es.shadow_arms("deepseek") == [("deepseek", "compact")]


def test_shadow_arms_extra_specs_and_dedupe(monkeypatch):
    _enable_local(monkeypatch)
    monkeypatch.setattr(settings, "synthesis_shadow_extra",
                        "deepseek:compact, local:compact , qwen")
    arms = es.shadow_arms("deepseek")
    # local:auto already resolves to local:compact — the duplicate is dropped.
    assert arms == [("local", "auto"), ("deepseek", "compact"), ("qwen", "auto")]


def test_shadow_arms_ignores_a_bad_spec_rather_than_raising(monkeypatch):
    _enable_local(monkeypatch)
    monkeypatch.setattr(settings, "synthesis_shadow_engine", "gpt5")
    monkeypatch.setattr(settings, "synthesis_shadow_extra", "qwen:tiny,notanengine,local")
    assert es.shadow_arms("deepseek") == [("local", "auto")]


def test_maybe_start_queues_the_live_rows_and_runs_the_arm(monkeypatch):
    _enable_local(monkeypatch)
    monkeypatch.setattr(settings, "enable_synthesis_shadow", True)
    seen = []

    def generate(signals, **kw):
        seen.append(kw)
        return [_rec("AAA", "SELL")]

    branch = es.maybe_start(
        signals=[_sig()], live_engine="DeepSeek", live_model="deepseek-v4-flash",
        live_recs=[_rec("AAA")], live_latency_s=12.0, synth_kwargs={"sectors": None},
        generate=generate, run_id="R1", generated_at="2026-09-04T14:00:00+00:00",
        signal_date="2026-09-04", arm_kwargs={"dual_case": False}, wait_for=lambda: 0)

    assert branch is not None and branch.join(timeout=10)
    assert seen[0]["force_engine"] == "local"
    assert seen[0]["prompt_variant"] == "auto"
    assert seen[0]["sectors"] is None and seen[0]["dual_case"] is False

    rows = es.pop_engine_shadow_rows()
    assert es.pop_engine_shadow_rows() == []          # the drain is one-shot
    live = [r for r in rows if r["live"]]
    shadow = [r for r in rows if not r["live"]]
    assert len(live) == 1 and live[0]["engine"] == "deepseek" and live[0]["prompt_variant"] == "full"
    assert live[0]["latency_s"] == 12.0
    assert len(shadow) == 1 and shadow[0]["engine"] == "local"
    assert shadow[0]["prompt_variant"] == "compact"   # RESOLVED, not "auto"
    assert shadow[0]["model"] == "local/qwen3:8b" and shadow[0]["action"] == "SELL"
    assert es.engine_shadow_pending() == 0


@pytest.mark.parametrize("generate", [_boom, _none])
def test_a_failing_shadow_arm_records_nothing_and_never_raises(generate, monkeypatch):
    """A shadow arm that errors — or that returns the pinned branch's empty
    list — must leave no row: a fabricated decision would poison the pair."""
    _enable_local(monkeypatch)
    monkeypatch.setattr(settings, "enable_synthesis_shadow", True)

    branch = es.maybe_start(
        signals=[_sig()], live_engine="deepseek", live_model="deepseek-v4-flash",
        live_recs=[_rec("AAA")], live_latency_s=1.0, synth_kwargs={},
        generate=generate, run_id="R1", generated_at="t", signal_date="2026-09-04")

    assert branch.join(timeout=10)
    assert [r["live"] for r in es.pop_engine_shadow_rows()] == [True]
    assert es.engine_shadow_pending() == 0


def test_maybe_start_is_single_flight(monkeypatch):
    """A local synthesis runs minutes; without the cap a slow server would pile
    up one branch per tick until the box thrashes."""
    _enable_local(monkeypatch)
    monkeypatch.setattr(settings, "enable_synthesis_shadow", True)
    monkeypatch.setattr(settings, "synthesis_shadow_max_pending", 1)
    monkeypatch.setattr(es, "_PENDING", 1)

    branch = es.maybe_start(
        signals=[_sig()], live_engine="deepseek", live_model="m", live_recs=[_rec("AAA")],
        live_latency_s=1.0, synth_kwargs={}, generate=_none,
        run_id="R1", generated_at="t", signal_date="2026-09-04")

    assert branch is None
    # The LIVE decision is still recorded — only the shadow work is skipped.
    assert [r["live"] for r in es.pop_engine_shadow_rows()] == [True]


def test_maybe_start_is_off_by_setting_and_needs_something_to_pair(monkeypatch):
    _enable_local(monkeypatch)
    common = dict(live_model="m", live_latency_s=1.0, synth_kwargs={}, generate=_none,
                  run_id="R1", generated_at="t", signal_date="2026-09-04")

    monkeypatch.setattr(settings, "enable_synthesis_shadow", False)
    assert es.maybe_start(signals=[_sig()], live_engine="deepseek",
                          live_recs=[_rec("AAA")], **common) is None

    monkeypatch.setattr(settings, "enable_synthesis_shadow", True)
    assert es.maybe_start(signals=[], live_engine="deepseek",
                          live_recs=[_rec("AAA")], **common) is None
    assert es.maybe_start(signals=[_sig()], live_engine="deepseek",
                          live_recs=[], **common) is None
    assert es.pop_engine_shadow_rows() == []


def test_the_live_pass_is_never_blocked_by_the_shadow(monkeypatch):
    """The arm runs on a background thread: `maybe_start` must return while the
    arm is still working, or a minutes-long local call would sit on the tick."""
    _enable_local(monkeypatch)
    monkeypatch.setattr(settings, "enable_synthesis_shadow", True)
    release = threading.Event()

    def slow(signals, **kw):
        release.wait(10)
        return [_rec("AAA")]

    branch = es.maybe_start(
        signals=[_sig()], live_engine="deepseek", live_model="m", live_recs=[_rec("AAA")],
        live_latency_s=1.0, synth_kwargs={}, generate=slow,
        run_id="R1", generated_at="t", signal_date="2026-09-04")

    assert es.engine_shadow_pending() == 1
    # The live rows are drainable NOW, while the arm is still running.
    assert [r["live"] for r in es.pop_engine_shadow_rows()] == [True]
    release.set()
    assert branch.join(timeout=10)
    assert [r["live"] for r in es.pop_engine_shadow_rows()] == [False]


# ── 7. persistence ───────────────────────────────────────────────────────────

def _row(engine, ticker, variant="full", live=True, run="R1", action="BUY", conf=0.9):
    return {"run_id": run, "generated_at": "2026-09-04T14:00:00+00:00",
            "signal_date": "2026-09-04", "engine": engine, "model": f"{engine}-m",
            "prompt_variant": variant, "live": live, "ticker": ticker, "action": action,
            "direction": "BULLISH", "confidence": conf, "time_horizon": "SWING",
            "rationale": "because", "snap_price": 10.0, "rule_filled": False,
            "latency_s": 3.0, "n_signals": 2, "n_recs": 1}


def _read_engine_rows():
    return repo.fetch_df(
        "SELECT engine, prompt_variant, ticker, live, confidence FROM engine_recommendations "
        "ORDER BY engine, prompt_variant, ticker")


def test_engine_rows_are_idempotent_per_key():
    """The shadow arms land on a LATER tick's write than the live rows they
    pair with (a local synthesis outruns the run that started it), so the write
    can only ever be per-key — a run-wide DELETE would erase the live decision
    the arm is meant to be compared against."""
    repo.insert_engine_recommendations([_row("deepseek", "AAA"), _row("deepseek", "BBB")])
    repo.insert_engine_recommendations([_row("deepseek", "AAA", conf=0.42)])

    df = _read_engine_rows()
    assert len(df) == 2
    assert float(df[df["ticker"] == "AAA"]["confidence"].iloc[0]) == pytest.approx(0.42)


def test_the_prompt_variant_is_part_of_the_key():
    """`deepseek:compact` is a decomposition arm running BESIDE the live
    `deepseek:full` row. Keyed without the variant, the shadow drain would
    delete the live decision."""
    repo.insert_engine_recommendations([_row("deepseek", "AAA", variant="full", live=True)])
    repo.insert_engine_recommendations([_row("deepseek", "AAA", variant="compact", live=False)])
    repo.insert_engine_recommendations([_row("local", "AAA", variant="compact", live=False)])

    df = _read_engine_rows()
    assert len(df) == 3
    assert set(zip(df["engine"], df["prompt_variant"])) == {
        ("deepseek", "full"), ("deepseek", "compact"), ("local", "compact")}
    assert list(df[df["prompt_variant"] == "full"]["live"]) == [True]


def test_a_later_run_never_disturbs_an_earlier_one():
    repo.insert_engine_recommendations([_row("local", "AAA", run="R1")])
    repo.insert_engine_recommendations([_row("local", "AAA", run="R2")])
    assert len(_read_engine_rows()) == 2


# ── 8. the paired evaluation ─────────────────────────────────────────────────

def _panel():
    """Two arms, five days, same ticker-runs — B is right whenever they differ."""
    rows = []
    for d in range(5):
        for tic in ("AAA", "BBB"):
            a_action = "BUY" if tic == "AAA" else "SELL"
            b_action = "BUY"
            # Oriented pivot return of each arm's own call.
            rows.append(dict(run_id=f"R{d}", ticker=tic, signal_date=f"2026-09-0{d + 1}",
                             engine="local", prompt_variant="compact", arm="local:compact",
                             action=a_action, _side=1 if a_action == "BUY" else -1,
                             confidence=0.90, rule_filled=False, live=False,
                             latency_s=120.0, ret_pv=-1.0 if tic == "BBB" else 2.0,
                             fwd_ret_pv=2.0))
            rows.append(dict(run_id=f"R{d}", ticker=tic, signal_date=f"2026-09-0{d + 1}",
                             engine="deepseek", prompt_variant="full", arm="deepseek:full",
                             action=b_action, _side=1, confidence=0.88, rule_filled=False,
                             live=True, latency_s=30.0, ret_pv=2.0, fwd_ret_pv=2.0))
    return pd.DataFrame(rows)


def test_day_clustered_t_counts_days_not_rows():
    """Forty disagreements on one day are one observation — the alternative
    reports a t-stat driven by how many names an engine happened to cover."""
    dates = ["d1"] * 40 + ["d2"] * 40
    out = ee.day_clustered_t([1.0] * 40 + [-1.0] * 40, dates)
    assert out["n_days"] == 2 and out["n_rows"] == 80
    assert out["mean"] == pytest.approx(0.0)
    assert out["t"] != out["t"]                      # below 3 days: NaN, never 0.0
    assert ee.day_clustered_t([], []) == {"mean": None, "t": None, "n_days": 0, "n_rows": 0}

    steady = ee.day_clustered_t([1.0, 1.1, 0.9, 1.0], ["d1", "d2", "d3", "d4"])
    assert steady["n_days"] == 4 and steady["t"] > 2


def test_engine_summary_describes_each_arm():
    rows = {r["arm"]: r for r in ee.engine_summary(_panel())}
    assert set(rows) == {"local:compact", "deepseek:full"}
    assert rows["local:compact"]["calls"] == 10 and rows["local:compact"]["runs"] == 5
    assert rows["local:compact"]["live_runs"] == 0
    assert rows["deepseek:full"]["live_runs"] == 5
    assert rows["local:compact"]["buy_pct"] == 50.0
    assert rows["local:compact"]["latency_s_p50"] == 120.0
    assert rows["deepseek:full"]["ret_pv"] == pytest.approx(2.0)


def test_engine_pairs_reads_only_the_disagreements():
    """A shared call is not evidence about either engine — both would have
    taken it. Only the rows where the arms took different sides carry any."""
    pairs = ee.engine_pairs(_panel())
    assert len(pairs) == 1
    p = pairs[0]
    assert p["common"] == 10
    assert p["side_agree_pct"] == 50.0
    assert p["disagree"] == 5                        # the BBB rows, one per day
    a_first = p["a"] == "deepseek:full"
    # A minus B on the disagreement subset: deepseek (+2.0) beats local (−1.0).
    assert p["edge"] == pytest.approx(3.0 if a_first else -3.0)
    assert p["edge_days"] == 5


def test_engine_pairs_excludes_rule_filled_rows():
    df = _panel()
    df.loc[df["arm"] == "local:compact", "rule_filled"] = True
    assert ee.engine_pairs(df) == []                 # a fill is not a decision
