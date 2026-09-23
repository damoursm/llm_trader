"""Local (self-hosted) sentiment engine — 2026-09-03.

Why this engine exists: on 2026-09-01 the DeepSeek AND OpenRouter accounts went
unfunded together, and because every sentiment engine was a HOSTED one, the
0.40-weight `news` method — the highest weight in `_BASE_WEIGHTS` — read exactly
0.0000 for a full day while synthesis fell through to `_fallback_recommendations`.
No amount of cross-hosted-engine ordering can cover that; only an engine with no
billing relationship can.

Three properties are pinned here, and the second and third are the subtle ones:

  * with `enable_local_llm=False` the module is byte-identical to before this
    engine existed (the try-order, the pins, the fallbacks) — a new engine must
    not perturb production until it is deliberately switched on;
  * the local engine has its OWN provider name and its OWN model id. Repointing
    the qwen route would have been a 3-line .env change, but `runs.llm_sentiment_provider`,
    the per-rec engine stamp and the per-LLM eval all key on the provider name,
    so a self-hosted qwen3:8b would have been labelled as the hosted qwen3.7-plus
    forever — and `memory/llm-eval-data-bugs-2026-07` is exactly that failure;
  * the local call sends the LOCAL server's own no-reasoning dialect and never
    the HOSTED one. Probed 2026-09-03 against Ollama 0.33.2 / qwen3:8b: only
    `reasoning_effort: "none"` works; `think: false`, `chat_template_kwargs`
    and the in-prompt `/no_think` are all IGNORED, and with reasoning left on
    the model spent its entire 256-token answer budget on the chain and
    returned content="" — a LOST verdict that `analyse_sentiment` would have
    scored as a neutral 0.0 into the 0.40-weight news method.

All synthetic: no server, no network, no model.
"""

from types import SimpleNamespace

import pytest

import src.analysis.sentiment as sent
from config.settings import settings


@pytest.fixture(autouse=True)
def _reset_local_client(monkeypatch):
    """The client is a module global memoised across calls."""
    monkeypatch.setattr(sent, "_local_client", None)


# ── default-off is a provable no-op ────────────────────────────────────────

def test_disabled_local_is_invisible_to_every_route(monkeypatch):
    monkeypatch.setattr(settings, "enable_local_llm", False)
    monkeypatch.setattr(settings, "enable_claude_sentiment", False)
    monkeypatch.setattr(sent, "_PRIMARY_SENTIMENT_ENGINE", "deepseek")
    assert "local" not in sent._fallback_order()
    assert sent._sentiment_engine_order(None) == ["deepseek", "qwen"]
    # Even an explicit pin cannot reach a switched-off engine (it would dead-end
    # on a client that is always None and score a fabricated neutral 0.0).
    assert "local" not in sent._sentiment_engine_order("local")
    assert sent._get_local() is None


def test_enabled_local_joins_the_chain_without_displacing_deepseek(monkeypatch):
    monkeypatch.setattr(settings, "enable_local_llm", True)
    monkeypatch.setattr(settings, "enable_claude_sentiment", False)
    monkeypatch.setattr(sent, "_PRIMARY_SENTIMENT_ENGINE", "deepseek")
    # DeepSeek still leads; local is a fallback, not a takeover.
    assert sent._sentiment_engine_order(None) == ["deepseek", "qwen", "local"]
    assert sent._sentiment_engine_order("local")[0] == "local"
    # A pin is a preference, not a suicide pact (the 2026-07-22 rule).
    assert "deepseek" in sent._sentiment_engine_order("local")


def test_local_primary_run_keeps_a_hosted_error_fallback(monkeypatch):
    monkeypatch.setattr(settings, "enable_local_llm", True)
    monkeypatch.setattr(settings, "enable_claude_sentiment", False)
    monkeypatch.setattr(sent, "_PRIMARY_SENTIMENT_ENGINE", "local")
    order = sent._sentiment_engine_order(None)
    assert order[0] == "local" and "deepseek" in order


def test_run_flip_routes_by_share(monkeypatch):
    monkeypatch.setattr(settings, "enable_local_llm", True)
    monkeypatch.setattr(settings, "enable_claude_sentiment", False)
    monkeypatch.setattr(settings, "sentiment_local_share", 1.0)
    sent.reset_sentiment_providers()
    assert sent._PRIMARY_SENTIMENT_ENGINE == "local"
    monkeypatch.setattr(settings, "sentiment_local_share", 0.0)
    monkeypatch.setattr(settings, "sentiment_qwen_share", 0.0)
    sent.reset_sentiment_providers()
    assert sent._PRIMARY_SENTIMENT_ENGINE == "deepseek"


# ── provenance: a self-hosted model is not the hosted one ──────────────────

def test_local_carries_its_own_provider_name_and_model_id(monkeypatch):
    monkeypatch.setattr(settings, "local_sentiment_model", "qwen3:8b")
    monkeypatch.setattr(settings, "qwen_model", "qwen/qwen3.7-plus")
    local_id = sent.sentiment_model_for("local")
    assert local_id == "local/qwen3:8b"
    assert local_id != sent.sentiment_model_for("qwen"),         "a self-hosted qwen must never share a model id with the hosted one"
    assert local_id.startswith("local/"), "the ledger must show this was self-hosted"

    # Resolved at CALL time: swapping the model must not keep stamping the old id
    # (and, because the cache key salts on this string, must invalidate its cache).
    monkeypatch.setattr(settings, "local_sentiment_model", "llama3.1:8b")
    assert sent.sentiment_model_for("local") == "local/llama3.1:8b"


def test_changing_the_local_model_invalidates_its_cached_verdicts(monkeypatch):
    """The cache key salts on the model id, so a model swap must not keep
    serving the previous model's answers for the TTL — the same discipline
    _SENT_PROMPT_VERSION enforces for a prompt edit."""
    arts = _articles()
    monkeypatch.setattr(settings, "local_sentiment_model", "qwen3:8b")
    k1 = sent._sentiment_cache_key("AAA", "local", arts)
    monkeypatch.setattr(settings, "local_sentiment_model", "llama3.1:8b")
    k2 = sent._sentiment_cache_key("AAA", "local", arts)
    assert k1 != k2
    # ...and a local key can never collide with the hosted qwen's.
    assert sent._sentiment_cache_key("AAA", "qwen", arts) not in (k1, k2)


def test_dominant_model_reports_the_local_id(monkeypatch):
    monkeypatch.setattr(sent, "_PROVIDER_COUNTS", {"local": 40, "deepseek": 2})
    assert sent.get_dominant_sentiment_model() == sent.sentiment_model_for("local")
    assert sent.get_dominant_sentiment_model().startswith("local/")


# ── the call itself ────────────────────────────────────────────────────────

class _FakeCompletions:
    def __init__(self, sink, content, prompt_tokens=None):
        self._sink, self._content, self._pt = sink, content, prompt_tokens

    def create(self, **kw):
        self._sink.append(kw)
        return SimpleNamespace(
            usage=(SimpleNamespace(prompt_tokens=self._pt) if self._pt else None),
            choices=[SimpleNamespace(message=SimpleNamespace(content=self._content))])


def _fake_client(sink, content, prompt_tokens=None):
    return SimpleNamespace(chat=SimpleNamespace(
        completions=_FakeCompletions(sink, content, prompt_tokens)))


def _articles():
    from datetime import datetime, timezone
    from src.models import NewsArticle
    return [NewsArticle(title="Contract awarded", summary="A $500M contract.",
                        url="http://x/1", source="Reuters",
                        published_at=datetime.now(timezone.utc))]


def test_local_call_sends_the_local_dialect_never_the_hosted_one(monkeypatch):
    """`enable_thinking` (DashScope) / `reasoning` (OpenRouter) are HOSTED
    parameters that a local server ignores SILENTLY — indistinguishable from
    them working. The local route sends its own dialect instead."""
    monkeypatch.setattr(settings, "enable_local_llm", True)
    monkeypatch.setattr(settings, "enable_claude_sentiment", False)
    monkeypatch.setattr(settings, "local_sentiment_model", "qwen3:8b")
    monkeypatch.setattr(settings, "local_sentiment_extra_body", '{"reasoning_effort": "none"}')
    monkeypatch.setattr(sent, "_PRIMARY_SENTIMENT_ENGINE", "local")
    sink = []
    monkeypatch.setattr(sent, "_get_local", lambda: _fake_client(
        sink, '{"rationale":"r","catalyst":"contract_partnership","score":0.5}'))

    score, rationale, meta = sent.analyse_sentiment("AAA", _articles())

    assert len(sink) == 1
    kw = sink[0]
    assert kw["extra_body"] == {"reasoning_effort": "none"}
    assert "enable_thinking" not in kw["extra_body"], "DashScope dialect leaked"
    assert "reasoning" not in kw["extra_body"], "OpenRouter dialect leaked"
    assert kw["model"] == "qwen3:8b"
    assert kw["temperature"] == 0
    assert meta.get("catalyst") == "contract_partnership"
    assert score != 0.0


def test_local_extra_body_is_configurable_and_fails_soft(monkeypatch):
    """A different server wants a different dialect (or none). A malformed knob
    must not take the engine down — the <think> strip still covers the inline
    case."""
    monkeypatch.setattr(settings, "local_sentiment_extra_body", '{"think": false}')
    assert sent._local_extra_body() == {"think": False}
    monkeypatch.setattr(settings, "local_sentiment_extra_body", "{}")
    assert sent._local_extra_body() == {}
    monkeypatch.setattr(settings, "local_sentiment_extra_body", "")
    assert sent._local_extra_body() == {}
    monkeypatch.setattr(settings, "local_sentiment_extra_body", "not json at all")
    assert sent._local_extra_body() == {}
    monkeypatch.setattr(settings, "local_sentiment_extra_body", '["a","list"]')
    assert sent._local_extra_body() == {}


def test_an_empty_local_verdict_is_logged_as_a_config_error(monkeypatch):
    """THE 2026-09-03 failure: reasoning on, the chain ate all 256 answer
    tokens, content="" with finish_reason="length". The verdict is lost and
    scored as a neutral 0.0 — so it must be LOUD, and it must name the cause."""
    seen = []
    monkeypatch.setattr(sent.logger, "error", lambda msg, *a, **k: seen.append(str(msg)))
    empty = SimpleNamespace(choices=[SimpleNamespace(
        finish_reason="length",
        message=SimpleNamespace(content="", model_extra={"reasoning": "Okay, the user..."}))])
    sent._warn_if_empty_local("AAA", empty)
    assert len(seen) == 1
    assert "EMPTY verdict" in seen[0] and "AAA" in seen[0]
    assert "REASONING" in seen[0], "the diagnosis must name the actual cause"
    assert "local_sentiment_extra_body" in seen[0], "...and the knob that fixes it"

    # A good response says nothing.
    seen.clear()
    ok = SimpleNamespace(choices=[SimpleNamespace(
        finish_reason="stop",
        message=SimpleNamespace(content='{"score":0.1}', model_extra={}))])
    sent._warn_if_empty_local("AAA", ok)
    assert seen == []



def test_a_dead_local_server_falls_through_to_a_hosted_engine(monkeypatch):
    """The whole point of the fallback chain: a local box that is off, wedged or
    mid-model-load must cost a retry, never a fabricated neutral 0.0."""
    monkeypatch.setattr(settings, "enable_local_llm", True)
    monkeypatch.setattr(settings, "enable_claude_sentiment", False)
    monkeypatch.setattr(sent, "_PRIMARY_SENTIMENT_ENGINE", "local")
    monkeypatch.setattr(sent, "_sentiment_cache_get", lambda k: None)
    monkeypatch.setattr(sent, "_sentiment_cache_put", lambda *a, **k: None)

    class _Dead:
        def create(self, **kw):
            raise ConnectionError("connection refused")

    monkeypatch.setattr(sent, "_get_local", lambda: SimpleNamespace(
        chat=SimpleNamespace(completions=_Dead())))
    sink = []
    monkeypatch.setattr(sent, "_get_deepseek", lambda: _fake_client(
        sink, '{"rationale":"r","catalyst":"contract_partnership","score":0.8}'))

    score, _, _ = sent.analyse_sentiment("AAA", _articles())
    assert len(sink) == 1, "DeepSeek was never tried after the local server died"
    assert score != 0.0


def test_a_prompt_the_server_cannot_hold_never_leaves_the_process(monkeypatch):
    """The sentiment prompt measures 4.4k-5.7k tokens at 14-20 articles (a
    fixed 2,176-token rubric prefix plus the digest), so it lives close to
    this server's per-request context —
    and Ollama truncates the OLDEST tokens (the scoring rubric) with no error,
    then answers. A verdict scored without its rubric is not an abstention and
    not a verdict; it is noise entering a 0.40-weight method, so the route
    refuses BEFORE the call and a hosted engine answers instead."""
    monkeypatch.setattr(settings, "enable_local_llm", True)
    monkeypatch.setattr(settings, "enable_claude_sentiment", False)
    monkeypatch.setattr(settings, "local_sentiment_context_tokens", 8)      # smaller than any prompt
    monkeypatch.setattr(sent, "_PRIMARY_SENTIMENT_ENGINE", "local")
    monkeypatch.setattr(sent, "_sentiment_cache_get", lambda k: None)
    monkeypatch.setattr(sent, "_sentiment_cache_put", lambda *a, **k: None)
    local_sink, hosted_sink = [], []
    monkeypatch.setattr(sent, "_get_local", lambda: _fake_client(local_sink, "{}"))
    monkeypatch.setattr(sent, "_get_deepseek", lambda: _fake_client(
        hosted_sink, '{"rationale":"r","catalyst":"contract_partnership","score":0.8}'))

    score, _, _ = sent.analyse_sentiment("AAA", _articles())

    assert local_sink == [], "an over-long prompt was still sent to the local server"
    assert len(hosted_sink) == 1 and score != 0.0


def test_a_silently_truncated_local_verdict_is_refused(monkeypatch):
    """Second net, for a server context smaller than the setting claims: the
    server's OWN reported prompt_tokens contradicts the estimate."""
    monkeypatch.setattr(settings, "enable_local_llm", True)
    monkeypatch.setattr(settings, "enable_claude_sentiment", False)
    monkeypatch.setattr(settings, "local_sentiment_context_tokens", 0)      # pre-flight off
    monkeypatch.setattr(sent, "_PRIMARY_SENTIMENT_ENGINE", "local")
    monkeypatch.setattr(sent, "_sentiment_cache_get", lambda k: None)
    monkeypatch.setattr(sent, "_sentiment_cache_put", lambda *a, **k: None)
    local_sink, hosted_sink = [], []
    # A perfectly parseable answer — the ONLY tell is the token count.
    monkeypatch.setattr(sent, "_get_local", lambda: _fake_client(
        local_sink, '{"rationale":"r","catalyst":"contract_partnership","score":0.5}',
        prompt_tokens=20))
    monkeypatch.setattr(sent, "_get_deepseek", lambda: _fake_client(
        hosted_sink, '{"rationale":"r","catalyst":"contract_partnership","score":0.8}'))

    score, _, meta = sent.analyse_sentiment("AAA", _articles())

    assert len(local_sink) == 1                    # it was called
    assert len(hosted_sink) == 1                   # and its answer was thrown away
    assert score != 0.0


def test_the_two_truncation_tests_cover_each_other(monkeypatch):
    """Why there are TWO post-call tests rather than one ratio.

    Truncation to an effective context E starts as soon as the REAL count
    passes E, but a ratio test only sees it once the ESTIMATE passes
    E/ratio — so a prompt just over the context reports a healthy-looking
    fraction and slips through. That window is what the exact test closes for
    the case that matters most in practice (the configured context IS the
    effective one): a prompt that fit reports strictly fewer tokens than the
    context it fit in, so a count landing AT the context is the server saying
    it filled the window.

    The ratio test is still needed for the case the exact one cannot see: an
    effective context SMALLER than the configured one (a server reconfigured
    behind the setting, or a runtime dividing the window across slots), where
    the count pins well below the setting and only the estimate reveals it.
    """
    from src.analysis import local_llm

    # A healthy call: reported below both the estimate's tolerance and the ctx.
    local_llm.check_reported(5_667, estimate=6_206, context_tokens=8_192, label="ok")

    # Pinned AT the configured context — caught with no reference to the estimate,
    # so a wrong CHARS_PER_TOKEN cannot defeat it.
    with pytest.raises(RuntimeError, match="prompt that fit would report FEWER"):
        local_llm.check_reported(8_192, estimate=9_000, context_tokens=8_192, label="pinned")

    # Effective context BELOW the setting (the divided-window case): the count is
    # legal against the setting and only the estimate contradicts it.
    with pytest.raises(RuntimeError, match="server context is smaller"):
        local_llm.check_reported(4_096, estimate=6_206, context_tokens=8_192, label="divided")

    # The constants are a PAIR: the ratio only separates healthy from truncated
    # because the estimate is calibrated. At the old 3.2 chars/token a healthy
    # call reported ~0.73 of the estimate and a truncated one ~0.66 — closer
    # together than any threshold could split.
    assert local_llm.CHARS_PER_TOKEN == 4.0
    assert local_llm.TRUNCATION_RATIO == 0.85
    healthy = 1 / (4.38 / local_llm.CHARS_PER_TOKEN)       # measured tokenizer vs estimate
    assert healthy > local_llm.TRUNCATION_RATIO, "a healthy call would now be refused"


def test_inline_reasoning_block_is_stripped(monkeypatch):
    """A hybrid Qwen3 checkpoint that ignores the server-side reasoning switch
    (`local_sentiment_extra_body`) emits <think>…</think> ahead of the JSON.
    Inert for the hosted engines, which never inline it."""
    assert sent._parse_response(
        '<think>weighing the catalyst</think>\n'
        '{"rationale":"r","catalyst":"guidance","score":-0.3}') == (-0.3, "r", "guidance")
    # Multi-line and with a fenced block after it.
    assert sent._parse_response(
        '<think>\nline one\nline two\n</think>\n```json\n'
        '{"rationale":"r2","catalyst":"none","score":0.0}\n```')[1] == "r2"
    # No think block → byte-identical to before.
    assert sent._parse_response(
        '{"rationale":"r3","catalyst":"none","score":0.1}') == (0.1, "r3", "none")


# ── the local engine must count as "an LLM ran" ────────────────────────────

def test_local_counts_as_a_real_llm_engine():
    """`_LLM_ENGINES` is the "a model answered" test. Omitting `local` would
    make a HEALTHY local-only sentiment run report as sentiment DOWN — a
    CRITICAL log, an email banner and the 🔔 subject tag on a run that worked
    perfectly — and would mark it a non-LLM run in the hold-review grouping.
    The list is about whether a model answered, not about who invoiced."""
    from src.performance.tracker import _LLM_ENGINES
    assert "local" in _LLM_ENGINES
    # Every engine the sentiment router can select must be recognised, or that
    # engine's healthy runs are alarms.
    assert set(sent.SENTIMENT_PROVIDER_MODELS) <= set(_LLM_ENGINES)


def test_a_local_only_run_is_not_reported_as_sentiment_down(monkeypatch):
    import src.pipeline as pl
    monkeypatch.setattr(pl, "get_last_synthesis_meta", lambda: {"provider": "deepseek"})
    monkeypatch.setattr(pl, "get_sentiment_provider_summary", lambda: "local\u00d768")
    health = pl._assess_llm_health()
    assert not health["sentiment_down"], "a healthy local sentiment run was flagged as an outage"
    assert not health["down"]

    # ...while a genuinely dead layer still alarms.
    monkeypatch.setattr(pl, "get_sentiment_provider_summary", lambda: "none\u00d768")
    assert pl._assess_llm_health()["sentiment_down"]


def test_local_model_id_resolves_to_local_not_qwen():
    """`_provider_of_synth_model` is SUBSTRING matching, and the self-hosted ids
    are `local/<model>` — so `local/qwen3:8b` matched the `qwen` branch and
    resolved to the HOSTED engine. Consequence: every position opened on a local
    run would have had its hold review pinned to hosted Qwen (Fix #2 re-judges
    with the OPENING engines), and the cohort would carry the wrong label. Order
    is load-bearing; the `local/` prefix must be checked first."""
    from src.performance.tracker import _provider_of_synth_model as pv
    assert pv("local/qwen3:8b") == "local"
    assert pv("local/llama3.1:8b") == "local"
    assert pv("local/deepseek-r1:7b") == "local", "a local DeepSeek is still local"
    # ...without breaking the hosted engines.
    assert pv("qwen/qwen3.7-plus") == "qwen"
    assert pv("deepseek-v4-flash") == "deepseek"
    assert pv("claude-haiku-4-5-20251001") == "anthropic"
    assert pv("rule-based (no LLM)") == "rule-based"
    assert pv("") is None and pv(None) is None


def test_a_locally_opened_trade_is_pinned_to_local_for_its_hold_review():
    """End-to-end of the above: the opener-pinned grouping must put a
    local-opened position in a `local` cohort, not a `qwen` one."""
    import src.pipeline as pl
    trades = [{"ticker": "AAA", "llm_sentiment_model": "local/qwen3:8b",
               "llm_synthesis_model": "deepseek-v4-flash"}]
    groups, legacy = pl._hold_review_groups(trades, run_sent="deepseek", run_synth="deepseek")
    assert groups == {("local", "deepseek"): ["AAA"]}, groups
    assert legacy == []


# ── LOCAL-ONLY MODE (2026-09-18, user directive) ─────────────────────────────

def test_hosted_engines_are_stripped_when_disabled(monkeypatch):
    """`enable_hosted_sentiment_engines=False` must remove deepseek/qwen from
    EVERY try-order. A tier that returns HTTP 402 on every call is not a
    fallback, it is two dead round trips per failure."""
    from config.settings import settings
    from src.analysis import sentiment as sent
    monkeypatch.setattr(settings, "enable_local_llm", True, raising=False)
    monkeypatch.setattr(settings, "enable_hosted_sentiment_engines", False, raising=False)
    monkeypatch.setattr(sent, "_PRIMARY_SENTIMENT_ENGINE", "local", raising=False)
    assert sent._fallback_order() == ("local",)
    assert sent._sentiment_engine_order(None) == ["local"]


def test_a_hosted_pin_coerces_to_local_rather_than_dead_ending(monkeypatch):
    """A pin is a preference, not a suicide pact (2026-07-22): a hold review or
    shadow arm pinned to deepseek must fall to local, not to a fabricated 0.0 in
    the 0.40-weight `news` method."""
    from config.settings import settings
    from src.analysis import sentiment as sent
    monkeypatch.setattr(settings, "enable_local_llm", True, raising=False)
    monkeypatch.setattr(settings, "enable_hosted_sentiment_engines", False, raising=False)
    monkeypatch.setattr(sent, "_PRIMARY_SENTIMENT_ENGINE", "local", raising=False)
    for pin in ("deepseek", "qwen", "anthropic"):
        assert sent._sentiment_engine_order(pin) == ["local"], pin


def test_the_try_order_is_never_empty(monkeypatch):
    """Belt and braces: local off AND hosted off must not empty the order, or
    every ticker scores a fabricated neutral 0.0."""
    from config.settings import settings
    from src.analysis import sentiment as sent
    monkeypatch.setattr(settings, "enable_local_llm", False, raising=False)
    monkeypatch.setattr(settings, "enable_hosted_sentiment_engines", False, raising=False)
    assert sent._fallback_order()
    assert sent._sentiment_engine_order(None)


def test_hosted_engines_on_is_the_shipped_order(monkeypatch):
    from config.settings import settings
    from src.analysis import sentiment as sent
    monkeypatch.setattr(settings, "enable_local_llm", True, raising=False)
    monkeypatch.setattr(settings, "enable_hosted_sentiment_engines", True, raising=False)
    monkeypatch.setattr(sent, "_PRIMARY_SENTIMENT_ENGINE", "local", raising=False)
    assert sent._sentiment_engine_order(None)[:2] == ["local", "deepseek"]


def test_a_dead_local_only_tier_records_why_for_the_health_alert(monkeypatch):
    """The provider tally says THAT every call fell to "none"; the engine-error
    tally says WHY. It is what lets the health alert point at the local server
    instead of at hosted credits — 2026-09-21, a reboot left the local-only tier
    dead for 20 ticks under an alert that blamed the hosted engines."""
    monkeypatch.setattr(settings, "enable_local_llm", True)
    monkeypatch.setattr(settings, "enable_hosted_sentiment_engines", False)
    monkeypatch.setattr(settings, "enable_claude_sentiment", False)
    monkeypatch.setattr(sent, "_PRIMARY_SENTIMENT_ENGINE", "local")
    monkeypatch.setattr(sent, "_sentiment_cache_get", lambda k: None)
    monkeypatch.setattr(sent, "_sentiment_cache_put", lambda *a, **k: None)
    monkeypatch.setattr(sent, "_PROVIDER_COUNTS", {})
    monkeypatch.setattr(sent, "_ENGINE_ERRORS", {})

    class _Dead:
        def create(self, **kw):
            raise ConnectionError("Connection error.")

    monkeypatch.setattr(sent, "_get_local", lambda: SimpleNamespace(
        chat=SimpleNamespace(completions=_Dead())))

    score, _, _ = sent.analyse_sentiment("AAA", _articles())
    assert score == 0.0
    assert sent.get_sentiment_provider_summary() == "none\u00d71"
    assert sent.get_sentiment_engine_errors() == {"local": (1, "Connection error.")}

    sent.reset_sentiment_providers()
    assert sent.get_sentiment_engine_errors() == {}, "one run's errors leaked into the next"
