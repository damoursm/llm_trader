"""Shadow sentiment — the local engine runs on EVERY tick, beside the primary.

2026-09-04, user directive: "have the local Qwen always run so that we can
compare ticker per ticker the difference in direction and score value". Both
engines score the SAME article digest and both verdicts land on one
``sentiment_shadow`` row.

Four properties are pinned here, and every one of them is a way this could go
wrong INVISIBLY:

  * the shadow never changes the live verdict. The score, rationale and meta a
    caller receives must be identical with the shadow on and off — a shadow that
    could alter the 0.40-weight `news` method is not a shadow;
  * the shadow never blocks. Submission is fire-and-forget and the drain is
    non-blocking, because the local engine's measured ceiling (~0.54 calls/s)
    is ~325 s for a tick's calls — that must sit behind the tick, not in it;
  * it scores the OTHER engine on the SAME article set. If the shadow re-derived
    its own digest, an input difference would be reported as a model difference,
    which is exactly the confound that makes a cached head-to-head impossible;
  * it never tallies into the run's provider counts, and a forced (hold-review)
    call spawns no shadow — otherwise `runs.llm_sentiment_provider` and the
    per-LLM eval would attribute the run to an engine that only measured it.

All synthetic: no server, no network, no database.
"""

from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import pytest

import src.analysis.sentiment as sent
from config.settings import settings
from src.models import NewsArticle


def _article(i: int = 0, source: str = "Reuters") -> NewsArticle:
    return NewsArticle(
        title=f"Company reports something specific number {i}",
        summary="Body text with enough substance to look like a real digest entry.",
        url=f"https://example.com/{i}", source=source,
        published_at=datetime.now(timezone.utc) - timedelta(hours=1),
        tickers=["AAPL"],
    )


@pytest.fixture(autouse=True)
def _clean_shadow_state(monkeypatch):
    """Drain the module's shadow buffers so tests cannot leak into each other."""
    sent.pop_sentiment_shadow_rows()
    monkeypatch.setattr(sent, "_SHADOW_PENDING", 0)
    monkeypatch.setattr(settings, "enable_sentiment_shadow", True)
    monkeypatch.setattr(settings, "sentiment_shadow_engine", "auto")
    monkeypatch.setattr(settings, "enable_local_llm", True)
    monkeypatch.setattr(settings, "enable_claude_sentiment", False)
    monkeypatch.setattr(sent, "_PRIMARY_SENTIMENT_ENGINE", "deepseek")
    yield
    sent.pop_sentiment_shadow_rows()


def _run_inline(monkeypatch):
    """Run submitted shadows synchronously so a test can assert on the rows."""
    class _Inline:
        def submit(self, fn, *a, **kw):
            fn(*a, **kw)
            return None
    monkeypatch.setattr(sent, "_shadow_pool", lambda: _Inline())


# ── which engine shadows which ─────────────────────────────────────────────

def test_shadow_is_always_the_other_engine(monkeypatch):
    assert sent._shadow_engine_for("deepseek") == "local"
    assert sent._shadow_engine_for("local") == "deepseek"
    # ... so a ticker accrues BOTH verdicts whichever way the per-run flip landed.


def test_shadow_is_skippable_and_pinnable(monkeypatch):
    monkeypatch.setattr(settings, "enable_sentiment_shadow", False)
    assert sent._shadow_engine_for("deepseek") is None
    monkeypatch.setattr(settings, "enable_sentiment_shadow", True)
    monkeypatch.setattr(settings, "sentiment_shadow_engine", "local")
    assert sent._shadow_engine_for("deepseek") == "local"
    # A pin that names the primary is a no-op, never a self-comparison.
    assert sent._shadow_engine_for("local") is None


def test_shadow_never_names_a_switched_off_engine(monkeypatch):
    monkeypatch.setattr(settings, "enable_local_llm", False)
    assert sent._shadow_engine_for("deepseek") is None


# ── the pairing itself ─────────────────────────────────────────────────────

def test_shadow_scores_the_same_digest_and_never_tallies(monkeypatch):
    _run_inline(monkeypatch)
    seen = []

    def fake_analyse(ticker, articles, force_engine=None):
        seen.append((ticker, force_engine, tuple(sorted(a.url for a in articles))))
        return -0.08, "local rationale", {"catalyst": "macro_sector", "raw_score": -0.10}

    monkeypatch.setattr(sent, "analyse_sentiment", fake_analyse)
    sent.reset_sentiment_providers()
    articles = [_article(i) for i in range(4)]
    sent._submit_shadow("AAPL", articles, "deepseek", 0.42, 0.38, "guidance", "digest-hash")

    rows = sent.pop_sentiment_shadow_rows()
    assert len(rows) == 1 and len(seen) == 1
    ticker, forced, urls = seen[0]
    assert forced == "local", "the shadow must pin the other engine, never fall back"
    assert urls == tuple(sorted(a.url for a in articles)),         "a different digest would report an INPUT difference as a MODEL difference"

    row = rows[0]
    assert (row["primary_engine"], row["shadow_engine"]) == ("deepseek", "local")
    assert (row["primary_raw"], row["shadow_raw"]) == (0.42, -0.10)
    assert row["primary_catalyst"] == "guidance" and row["shadow_catalyst"] == "macro_sector"
    assert row["digest_hash"] == "digest-hash" and row["n_articles"] == 4
    assert row["shadow_model"] == sent.sentiment_model_for("local")
    # The run's provider tally is the PRIMARY engine's record; a measurement call
    # that showed up there would mis-attribute the run in the per-LLM eval.
    assert "local" not in (sent.get_sentiment_provider_summary() or "")


def test_drain_is_one_shot_and_stamps_the_run(monkeypatch):
    _run_inline(monkeypatch)
    monkeypatch.setattr(sent, "analyse_sentiment",
                        lambda t, a, force_engine=None: (0.1, "r", {"catalyst": None, "raw_score": 0.1}))
    sent.set_current_run("2026-09-04_120000")
    sent._submit_shadow("AAPL", [_article()], "deepseek", 0.2, 0.2, None, "h")
    rows = sent.pop_sentiment_shadow_rows()
    assert [r["run_id"] for r in rows] == ["2026-09-04_120000"]
    assert sent.pop_sentiment_shadow_rows() == [], "a drained row must not be written twice"
    sent.set_current_run(None)


def test_a_shadow_that_abstains_or_fails_costs_only_a_row(monkeypatch):
    _run_inline(monkeypatch)

    def boom(ticker, articles, force_engine=None):
        raise RuntimeError("local server wedged")

    monkeypatch.setattr(sent, "analyse_sentiment", boom)
    sent._submit_shadow("AAPL", [_article()], "deepseek", 0.2, 0.2, None, "h")
    assert sent.pop_sentiment_shadow_rows() == []
    assert sent.sentiment_shadow_pending() == 0, "a failed shadow must release its slot"

    # A degraded verdict (no raw score) is not a comparison either.
    monkeypatch.setattr(sent, "analyse_sentiment",
                        lambda t, a, force_engine=None: (0.0, "error", {}))
    sent._submit_shadow("AAPL", [_article()], "deepseek", 0.2, 0.2, None, "h")
    assert sent.pop_sentiment_shadow_rows() == []


def test_shadow_row_is_attributed_to_the_engine_that_answered(monkeypatch):
    """The forced engine LEADS its order but the fallbacks are appended, so a
    dead local server makes DeepSeek answer a "local" shadow. That row must say
    DeepSeek - and a pair of the primary against its own engine is dropped
    outright (cost with no comparison in it). Fakes without an `engine` key
    keep the forced engine, as every older test relies on."""
    _run_inline(monkeypatch)
    from src.analysis import catalyst_repair as cr
    offered = []
    monkeypatch.setattr(cr, "maybe_submit", lambda **kw: offered.append(kw) or None)
    sent.set_current_run("2026-09-06_150000")
    try:
        # Fell through to the primary's own engine: no row, slot released.
        monkeypatch.setattr(sent, "analyse_sentiment", lambda t, a, force_engine=None: (
            0.1, "r", {"catalyst": "none", "raw_score": 0.1, "engine": "deepseek", "digest": "d"}))
        sent._submit_shadow("AAPL", [_article()], "deepseek", 0.2, 0.2, None, "h", digest_id="dg-1")
        assert sent.pop_sentiment_shadow_rows() == [] and offered == []
        assert sent.sentiment_shadow_pending() == 0
        # Answered by the forced engine: attributed to it, repair offered under the shadow role.
        monkeypatch.setattr(sent, "analyse_sentiment", lambda t, a, force_engine=None: (
            -0.2, "local rationale", {"catalyst": "macro_sector", "catalyst_raw": "product",
                                      "raw_score": -0.25, "engine": "local", "digest": "body"}))
        sent._submit_shadow("AAPL", [_article()], "deepseek", 0.2, 0.2, "guidance", "h", digest_id="dg-2")
        rows = sent.pop_sentiment_shadow_rows()
        assert len(rows) == 1 and rows[0]["shadow_engine"] == "local"
        assert rows[0]["shadow_model"] == sent.sentiment_model_for("local")
        assert rows[0]["shadow_catalyst"] == "product", "the engine's OWN label, pre-override"
        assert len(offered) == 1
        o = offered[0]
        assert (o["role"], o["engine"], o["digest_id"], o["digest_text"]) == ("shadow", "local", "dg-2", "body")
        assert (o["first_pass"], o["final_label"], o["rationale"]) == ("product", "macro_sector", "local rationale")
        assert o["run_id"] == "2026-09-06_150000"
        # No `engine` in the meta: the forced engine stands.
        monkeypatch.setattr(sent, "analyse_sentiment", lambda t, a, force_engine=None: (
            0.1, "r", {"catalyst": None, "raw_score": 0.1}))
        sent._submit_shadow("AAPL", [_article()], "deepseek", 0.2, 0.2, None, "h", digest_id="dg-3")
        rows = sent.pop_sentiment_shadow_rows()
        assert len(rows) == 1 and rows[0]["shadow_engine"] == "local"
    finally:
        sent.set_current_run(None)


def test_backlog_cap_stops_submitting(monkeypatch):
    monkeypatch.setattr(settings, "sentiment_shadow_max_pending", 2)
    submitted = []

    class _Pool:
        def submit(self, fn, *a, **kw):
            submitted.append(a[1])            # ticker
            return None                       # never runs: pending stays high

    monkeypatch.setattr(sent, "_shadow_pool", lambda: _Pool())
    for i in range(5):
        sent._submit_shadow(f"T{i}", [_article()], "deepseek", 0.2, 0.2, None, "h")
    assert len(submitted) == 2,         "a wedged shadow engine must not accumulate a tick of work per tick"


def test_primary_verdict_is_identical_with_and_without_the_shadow(monkeypatch):
    """The shadow is measurement: the live path must not move by one digit."""
    monkeypatch.setattr(sent, "_provider_sentiment_score", lambda *a, **kw: None)
    monkeypatch.setattr(sent, "_sentiment_cache_get", lambda key: None)
    monkeypatch.setattr(sent, "_sentiment_cache_put", lambda *a, **kw: None)

    class _Msg:
        def __init__(self, txt): self.content = txt
    class _Choice:
        def __init__(self, txt): self.message = _Msg(txt)
    class _Resp:
        def __init__(self, txt): self.choices = [_Choice(txt)]; self.usage = None

    payload = '{"rationale": "specific catalyst", "catalyst": "earnings", "score": 0.33}'

    class _Client:
        class chat:
            class completions:
                @staticmethod
                def create(**kw):
                    return _Resp(payload)

    monkeypatch.setattr(sent, "_get_deepseek", lambda: _Client())
    monkeypatch.setattr(sent, "_get_local", lambda: _Client())
    articles = [_article(i) for i in range(3)]

    monkeypatch.setattr(settings, "enable_sentiment_shadow", False)
    off = sent.analyse_sentiment("AAPL", articles)

    _run_inline(monkeypatch)
    monkeypatch.setattr(settings, "enable_sentiment_shadow", True)
    on = sent.analyse_sentiment("AAPL", articles)

    assert on == off, "the shadow pass must not perturb the live verdict"
    assert len(sent.pop_sentiment_shadow_rows()) == 1


def test_forced_calls_spawn_no_shadow(monkeypatch):
    """A hold review is pinned to the opener's engine on purpose; shadowing it
    would double the local server's load for a comparison the entry rows carry."""
    _run_inline(monkeypatch)
    monkeypatch.setattr(sent, "_provider_sentiment_score", lambda *a, **kw: None)
    monkeypatch.setattr(sent, "_sentiment_cache_get", lambda key: None)
    monkeypatch.setattr(sent, "_sentiment_cache_put", lambda *a, **kw: None)

    class _Resp:
        def __init__(self):
            self.choices = [SimpleNamespace(message=SimpleNamespace(
                content='{"rationale": "r", "catalyst": "none", "score": 0.12}'))]
            self.usage = None

    class _Client:
        class chat:
            class completions:
                @staticmethod
                def create(**kw): return _Resp()

    monkeypatch.setattr(sent, "_get_deepseek", lambda: _Client())
    sent.analyse_sentiment("AAPL", [_article()], force_engine="deepseek")
    assert sent.pop_sentiment_shadow_rows() == []


# ── the comparison surface ─────────────────────────────────────────────────

def test_compare_reports_direction_and_value():
    import pandas as pd
    from src.analysis import sentiment_shadow as shadow

    rows = [
        # (primary, shadow) raw verdicts: agree, oppose, one abstains
        ("r1", "AAPL", "deepseek", 0.40, "local", 0.20),
        ("r1", "MSFT", "deepseek", -0.30, "local", -0.10),
        ("r1", "NVDA", "deepseek", 0.10, "local", -0.25),
        ("r1", "TSLA", "deepseek", 0.05, "local", 0.00),
        # a local-primary run: the engines swap columns and must still line up
        ("r2", "AAPL", "local", 0.15, "deepseek", 0.35),
    ]
    df = pd.DataFrame([{
        "run_id": r, "generated_at": "2026-09-04T12:00:00+00:00", "ticker": t,
        "digest_hash": "h", "n_articles": 5,
        "primary_engine": pe, "primary_model": pe, "primary_raw": pr, "primary_score": pr,
        "primary_catalyst": "earnings",
        "shadow_engine": se, "shadow_model": se, "shadow_raw": sr, "shadow_score": sr,
        "shadow_catalyst": "earnings", "shadow_latency_s": 2.5,
    } for r, t, pe, pr, se, sr in rows])

    st = shadow.compare(df)
    assert st["engine_a"] == "deepseek" and st["engine_b"] == "local"
    assert st["n_pairs"] == 5 and st["n_runs"] == 2
    # AAPL/r2 was scored with local as PRIMARY: 0.15 must land in the local
    # column, not in deepseek's, or the flip would silently mix the engines.
    d = shadow._orient(df)
    r2 = d[d["run_id"] == "r2"].iloc[0]
    assert (r2["a_raw"], r2["b_raw"]) == (0.35, 0.15)
    # One of five abstained on the local side; four pairs have both views.
    assert st["abstain_local"] == pytest.approx(0.2)
    assert st["both_scored"] == pytest.approx(0.8)
    assert st["sign_agree_both_scored"] == pytest.approx(0.75)
    assert st["opposite_sides"] == pytest.approx(0.25)
    assert st["catalyst_agree"] == pytest.approx(1.0)


# ── sampling the shadow (2026-09-05 directive) ───────────────────────────────

def test_shadow_share_samples_deterministically(monkeypatch):
    """`sentiment_shadow_share` halves the shadow LOAD without biasing the pair
    sample: the draw is a hash of (run_id, ticker), so the same ticker gets the
    same answer if it is scored twice in a run — a per-call random draw would
    resample a retried ticker and over-represent exactly the calls that failed
    once."""
    monkeypatch.setattr(sent, "_CURRENT_RUN_ID", "2026-09-05_120000")
    monkeypatch.setattr(settings, "sentiment_shadow_share", 0.5)
    names = [f"TK{i}" for i in range(400)]

    hits = [t for t in names if sent._shadow_sampled(t)]
    assert 0.4 < len(hits) / len(names) < 0.6          # ~half, not all-or-nothing
    assert all(sent._shadow_sampled(t) for t in hits)  # stable within the run

    # A different run re-draws, so no ticker is permanently excluded.
    monkeypatch.setattr(sent, "_CURRENT_RUN_ID", "2026-09-05_123000")
    assert [t for t in names if sent._shadow_sampled(t)] != hits


def test_shadow_share_bounds_are_the_old_behaviour_and_off(monkeypatch):
    monkeypatch.setattr(sent, "_CURRENT_RUN_ID", "R")
    monkeypatch.setattr(settings, "sentiment_shadow_share", 1.0)
    assert all(sent._shadow_sampled(f"T{i}") for i in range(50))
    monkeypatch.setattr(settings, "sentiment_shadow_share", 0.0)
    assert not any(sent._shadow_sampled(f"T{i}") for i in range(50))


def test_a_pinned_shadow_engine_never_shadows_its_own_runs(monkeypatch):
    """"DeepSeek only, and doesn't shadow the rest": pinning the shadow engine
    means a run whose PRIMARY is that engine gets no shadow at all — pairing
    DeepSeek against DeepSeek would be a cost with no comparison in it."""
    monkeypatch.setattr(settings, "enable_sentiment_shadow", True)
    monkeypatch.setattr(settings, "sentiment_shadow_engine", "deepseek")
    assert sent._shadow_engine_for("local") == "deepseek"
    assert sent._shadow_engine_for("deepseek") is None
