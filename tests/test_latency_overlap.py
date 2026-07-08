"""Tick→order latency levers (2026-07-08).

Covers the four latency changes:
  1. `_HoldReviewBranch` — the opener-pinned hold-review running CONCURRENTLY
     with pipeline Steps 4–5. Invariants: every held stock still gets its own
     signal build (Step-4 equivalent) AND pinned synthesis (Step-5 equivalent)
     on the SAME synthesis context as the main pass; the signal build starts
     before the synthesis context exists; legacy positions wait for the run's
     engines; un-supplied futures fail soft (positions hold).
  2. Provenance isolation — forced-engine synthesis/sentiment calls must not
     touch the process-global last-synthesis meta / sentiment tallies (they run
     concurrently with the main pass now).
  3. Sentiment LLM cache — identical (ticker, engine, article set) reuses the
     raw verdict; any new article forces a fresh call; forced calls share the
     cache but never tally.
  4. Liquidity-gate cold pre-warm + options-flow bounded concurrency plumbing.
"""

import threading
import time
from datetime import datetime, timezone
from types import SimpleNamespace

import pandas as pd
import pytest

from config.settings import settings
from src.models import NewsArticle, Recommendation


def _rec(ticker, action="BUY", confidence=0.85):
    return Recommendation(
        ticker=ticker, type="ETF",
        direction="BULLISH" if action == "BUY" else "BEARISH",
        action=action, confidence=confidence, time_horizon="1w",
        rationale="r", generated_at=datetime.now(timezone.utc),
    )


def _claude_trade(ticker="XLE"):
    return {"ticker": ticker, "status": "OPEN",
            "llm_synthesis_model": "claude-haiku-4-5-20251001",
            "llm_sentiment_model": "claude-haiku-4-5-20251001"}


def _deepseek_trade(ticker="USO"):
    return {"ticker": ticker, "status": "OPEN",
            "llm_synthesis_model": "deepseek-v4-flash",
            "llm_sentiment_model": "deepseek-v4-flash"}


def _legacy_trade(ticker="GLD"):
    return {"ticker": ticker, "status": "OPEN",
            "llm_synthesis_model": "rule-based (no LLM)", "llm_sentiment_model": "none"}


# ── 1. _HoldReviewBranch orchestration ────────────────────────────────────────

def test_branch_runs_full_signal_build_and_pinned_synthesis_per_held_stock(monkeypatch):
    """The load-bearing invariant of the overlap: each held stock still gets its
    own Step-4 equivalent (build_signals on fresh data, pinned sentiment engine)
    AND Step-5 equivalent (pinned synthesis) — and the review synthesis receives
    the SAME synth_kwargs the main Step 5 uses."""
    import src.pipeline as pl
    monkeypatch.setattr(settings, "enable_pinned_hold_review", True)

    seen = {"news": [], "build": [], "synth": [], "synth_ctx": []}
    monkeypatch.setattr(pl, "fetch_all_news",
                        lambda tks, sectors: seen["news"].append(sorted(tks)) or [])
    monkeypatch.setattr(pl, "get_snapshots", lambda tks: [])

    def fake_build(tickers, articles, snapshots=None, session=None,
                   force_sentiment_engine=None, **kw):
        seen["build"].append((tuple(tickers), force_sentiment_engine))
        return [SimpleNamespace(ticker=t) for t in tickers]
    monkeypatch.setattr(pl, "build_signals", fake_build)

    def fake_synth(signals, session=None, force_engine=None, **kw):
        seen["synth"].append((tuple(s.ticker for s in signals), force_engine))
        seen["synth_ctx"].append(kw.get("macro_context"))
        return [_rec(s.ticker) for s in signals]
    monkeypatch.setattr(pl, "generate_recommendations", fake_synth)

    branch = pl._HoldReviewBranch(
        [_claude_trade("XLE"), _deepseek_trade("USO")], ["Tech"], {}, "rth")
    branch.supply_synth_kwargs({"macro_context": "MACRO-CTX"})
    branch.supply_run_engines("deepseek", "deepseek")
    reviews = branch.result()

    assert set(reviews) == {"XLE", "USO"}
    assert seen["news"] == [["USO", "XLE"]]                       # ONE fresh fetch for the held set
    assert dict(seen["build"]) == {("XLE",): "anthropic", ("USO",): "deepseek"}
    assert dict(seen["synth"]) == {("XLE",): "anthropic", ("USO",): "deepseek"}
    assert seen["synth_ctx"] == ["MACRO-CTX", "MACRO-CTX"]        # main-pass context reached the reviews


def test_branch_signal_build_starts_before_synthesis_context_exists(monkeypatch):
    """Overlap semantics: build_signals (Step-4 work) runs while the synthesis
    context future is still unresolved; the pinned synthesis only fires after
    supply_synth_kwargs."""
    import src.pipeline as pl
    monkeypatch.setattr(settings, "enable_pinned_hold_review", True)

    built = threading.Event()
    synthed = threading.Event()
    monkeypatch.setattr(pl, "fetch_all_news", lambda tks, sectors: [])
    monkeypatch.setattr(pl, "get_snapshots", lambda tks: [])

    def fake_build(tickers, *a, **k):
        built.set()
        return [SimpleNamespace(ticker=t) for t in tickers]
    monkeypatch.setattr(pl, "build_signals", fake_build)

    def fake_synth(signals, session=None, force_engine=None, **kw):
        synthed.set()
        return [_rec(s.ticker) for s in signals]
    monkeypatch.setattr(pl, "generate_recommendations", fake_synth)

    branch = pl._HoldReviewBranch([_deepseek_trade("USO")], [], {}, "rth")
    assert built.wait(10)                    # Step-4 work ran with no context supplied
    assert not synthed.wait(0.3)             # synthesis is genuinely blocked on the context
    branch.supply_synth_kwargs({})
    reviews = branch.result()
    assert synthed.is_set()
    assert set(reviews) == {"USO"}


def test_branch_legacy_positions_wait_for_run_engines(monkeypatch):
    """A position with no opener stamps is reviewed with THIS run's engines,
    which only exist after Step 5 — the branch must block that group on
    supply_run_engines and pin it to the supplied engines."""
    import src.pipeline as pl
    monkeypatch.setattr(settings, "enable_pinned_hold_review", True)

    synth_engines = []
    synthed = threading.Event()
    monkeypatch.setattr(pl, "fetch_all_news", lambda tks, sectors: [])
    monkeypatch.setattr(pl, "get_snapshots", lambda tks: [])
    monkeypatch.setattr(pl, "build_signals",
                        lambda tickers, *a, **k: [SimpleNamespace(ticker=t) for t in tickers])

    def fake_synth(signals, session=None, force_engine=None, **kw):
        synth_engines.append(force_engine)
        synthed.set()
        return [_rec(s.ticker) for s in signals]
    monkeypatch.setattr(pl, "generate_recommendations", fake_synth)

    branch = pl._HoldReviewBranch([_legacy_trade("GLD")], [], {}, "rth")
    branch.supply_synth_kwargs({})
    assert not synthed.wait(0.3)             # legacy group still waiting on the engines
    branch.supply_run_engines("deepseek", "deepseek")
    reviews = branch.result()
    assert set(reviews) == {"GLD"}
    assert synth_engines == ["deepseek"]


def test_branch_fails_soft_when_pipeline_never_supplies_context(monkeypatch):
    """If the pipeline dies before Step 5, the branch must not hang or raise —
    the waits time out and the positions simply hold this tick."""
    import src.pipeline as pl
    monkeypatch.setattr(settings, "enable_pinned_hold_review", True)
    monkeypatch.setattr(pl._HoldReviewBranch, "_WAIT_TIMEOUT_S", 0.2)

    monkeypatch.setattr(pl, "fetch_all_news", lambda tks, sectors: [])
    monkeypatch.setattr(pl, "get_snapshots", lambda tks: [])
    monkeypatch.setattr(pl, "build_signals",
                        lambda tickers, *a, **k: [SimpleNamespace(ticker=t) for t in tickers])
    monkeypatch.setattr(pl, "generate_recommendations",
                        lambda *a, **k: (_ for _ in ()).throw(
                            AssertionError("synthesis must not run without context")))

    branch = pl._HoldReviewBranch([_deepseek_trade("USO")], [], {}, "rth")
    reviews = branch.result()
    assert reviews == {}                     # no review ⇒ hold, same as a failed refetch


def test_branch_and_sequential_paths_produce_identical_reviews(monkeypatch):
    """The overlap is a scheduling change, not a semantics change: with the same
    fakes, the branch and the legacy sequential `_build_hold_reviews` produce
    the same reviews with the same per-ticker engines."""
    import src.pipeline as pl
    monkeypatch.setattr(settings, "enable_pinned_hold_review", True)

    monkeypatch.setattr(pl, "fetch_all_news", lambda tks, sectors: [])
    monkeypatch.setattr(pl, "get_snapshots", lambda tks: [])
    monkeypatch.setattr(pl, "build_signals",
                        lambda tickers, *a, **k: [SimpleNamespace(ticker=t) for t in tickers])
    monkeypatch.setattr(pl, "generate_recommendations",
                        lambda signals, session=None, force_engine=None, **kw:
                        [_rec(s.ticker) for s in signals])

    trades = [_claude_trade("XLE"), _deepseek_trade("USO"), _legacy_trade("GLD")]

    sequential = pl._build_hold_reviews(
        trades, "deepseek", "deepseek", [], [], {}, {}, "rth")

    branch = pl._HoldReviewBranch(trades, [], {}, "rth")
    branch.supply_synth_kwargs({})
    branch.supply_run_engines("deepseek", "deepseek")
    overlapped = branch.result()

    assert set(sequential) == set(overlapped) == {"XLE", "USO", "GLD"}
    assert getattr(sequential, "engines", {}) == getattr(overlapped, "engines", {})


# ── 2. Provenance isolation under concurrency ─────────────────────────────────

_VALID_SYNTH_JSON = (
    '[{"ticker": "USO", "type": "ETF", "direction": "BULLISH", "action": "BUY", '
    '"time_horizon": "SWING", "confidence": 0.8, "rationale": "r"}]'
)


def _ticker_signal(ticker="USO"):
    from src.models import TickerSignal
    return TickerSignal(
        ticker=ticker, direction="BULLISH", confidence=0.8,
        action_suggestion="BUY", news_sentiment_score=0.2, sentiment_score=0.2,
        insider_score=0.0, technical_score=0.2, sources_agreeing=2,
        key_reasons=["r"], rationale="r", price=10.0,
    )


def test_forced_synthesis_does_not_stamp_last_meta(monkeypatch):
    import src.analysis.claude_analyst as ca
    monkeypatch.setattr(ca, "_call_deepseek_analyst", lambda p, **k: _VALID_SYNTH_JSON)
    ca._set_synthesis_meta("anthropic", "main-pass-model")
    out = ca.generate_recommendations([_ticker_signal("USO")], force_engine="deepseek")
    assert [r.ticker for r in out] == ["USO"]
    # The forced (hold-review) call must NOT have overwritten the main pass's meta.
    assert ca.get_last_synthesis_meta() == {"provider": "anthropic", "model": "main-pass-model"}


def test_forced_synthesis_unparseable_yields_no_review_and_no_meta_clobber(monkeypatch):
    import src.analysis.claude_analyst as ca
    monkeypatch.setattr(ca, "_call_deepseek_analyst", lambda p, **k: "utter garbage, no json here")
    ca._set_synthesis_meta("anthropic", "main-pass-model")
    out = ca.generate_recommendations([_ticker_signal("USO")], force_engine="deepseek")
    assert out == []                                    # never a fabricated rule-based review
    assert ca.get_last_synthesis_meta() == {"provider": "anthropic", "model": "main-pass-model"}


def test_forced_sentiment_does_not_tally_provider(monkeypatch):
    import src.analysis.sentiment as sent
    calls = []

    class _Completions:
        def create(self, **kw):
            calls.append(1)
            return SimpleNamespace(choices=[SimpleNamespace(
                message=SimpleNamespace(content='{"score": 0.6, "rationale": "r"}'))])

    monkeypatch.setattr(sent, "_get_deepseek",
                        lambda: SimpleNamespace(chat=SimpleNamespace(completions=_Completions())))
    sent.reset_sentiment_providers()
    art = NewsArticle(title="t", summary="material catalyst " * 5, url="u1",
                      source="Reuters", published_at=datetime.now(timezone.utc))
    sent.analyse_sentiment("AAA", [art], force_engine="deepseek")
    assert calls == [1]
    # Forced (hold-review) scoring must leave the run's provider tallies untouched.
    assert sent.get_sentiment_provider_summary() is None
    assert sent.get_dominant_sentiment_model() is None


# ── 3. Sentiment LLM cache ────────────────────────────────────────────────────

def _stub_deepseek(calls):
    class _Completions:
        def create(self, **kw):
            calls.append(1)
            return SimpleNamespace(choices=[SimpleNamespace(
                message=SimpleNamespace(content='{"score": 0.6, "rationale": "r"}'))])
    return SimpleNamespace(chat=SimpleNamespace(completions=_Completions()))


def _art(url, title="headline"):
    return NewsArticle(title=title, summary="material catalyst " * 5, url=url,
                       source="Reuters", published_at=datetime.now(timezone.utc))


def test_sentiment_cache_skips_llm_on_identical_article_set(monkeypatch):
    import src.analysis.sentiment as sent
    calls = []
    monkeypatch.setattr(sent, "_get_deepseek", lambda: _stub_deepseek(calls))
    sent.reset_sentiment_providers()
    arts = [_art("u1"), _art("u2", "second headline")]

    s1, _ = sent.analyse_sentiment("AAA", arts)
    s2, _ = sent.analyse_sentiment("AAA", arts)
    assert len(calls) == 1                     # second scoring served from cache
    assert s1 == s2                            # identical adjusted score
    # Cache hits still tally (the verdict IS that engine's) so per-LLM
    # attribution / the dominant-model stamp keep working on cached runs.
    assert sent.get_sentiment_provider_summary() == "deepseek×2"


def test_sentiment_cache_new_article_forces_fresh_call(monkeypatch):
    import src.analysis.sentiment as sent
    calls = []
    monkeypatch.setattr(sent, "_get_deepseek", lambda: _stub_deepseek(calls))
    arts = [_art("u1")]
    sent.analyse_sentiment("AAA", arts)
    sent.analyse_sentiment("AAA", arts + [_art("u-new", "breaking catalyst")])
    assert len(calls) == 2                     # changed article set ⇒ new key ⇒ re-score


def test_sentiment_cache_is_per_ticker(monkeypatch):
    import src.analysis.sentiment as sent
    calls = []
    monkeypatch.setattr(sent, "_get_deepseek", lambda: _stub_deepseek(calls))
    arts = [_art("u1")]
    sent.analyse_sentiment("AAA", arts)
    sent.analyse_sentiment("BBB", arts)
    assert len(calls) == 2                     # same articles, different ticker ⇒ no reuse


def test_sentiment_cache_shared_between_main_and_forced_review(monkeypatch):
    """The big latency win: the hold-review re-scoring a held ticker minutes
    after the main pass (same engine, same articles) is a cache hit."""
    import src.analysis.sentiment as sent
    calls = []
    monkeypatch.setattr(sent, "_get_deepseek", lambda: _stub_deepseek(calls))
    sent.reset_sentiment_providers()
    arts = [_art("u1")]
    s_main, _ = sent.analyse_sentiment("AAA", arts)                          # main pass (miss)
    s_review, _ = sent.analyse_sentiment("AAA", arts, force_engine="deepseek")  # review (hit)
    assert len(calls) == 1
    assert s_main == s_review
    assert sent.get_sentiment_provider_summary() == "deepseek×1"   # forced hit never tallies


def test_sentiment_cache_ttl_expiry_rescores(monkeypatch):
    import src.analysis.sentiment as sent
    calls = []
    monkeypatch.setattr(sent, "_get_deepseek", lambda: _stub_deepseek(calls))
    arts = [_art("u1")]
    sent.analyse_sentiment("AAA", arts)
    with sent._SENT_CACHE_LOCK:                # age the sole entry past the TTL
        for entry in sent._SENT_CACHE.values():
            entry["ts"] -= sent._sent_cache_ttl_seconds() + 10
    sent.analyse_sentiment("AAA", arts)
    assert len(calls) == 2


def test_sentiment_cache_disabled_flag(monkeypatch):
    import src.analysis.sentiment as sent
    monkeypatch.setattr(settings, "enable_sentiment_cache", False)
    calls = []
    monkeypatch.setattr(sent, "_get_deepseek", lambda: _stub_deepseek(calls))
    arts = [_art("u1")]
    sent.analyse_sentiment("AAA", arts)
    sent.analyse_sentiment("AAA", arts)
    assert len(calls) == 2


# ── 4. Liquidity-gate cold pre-warm ───────────────────────────────────────────

def test_prewarm_cold_fetches_within_budget_and_marks_attempted(monkeypatch):
    import src.data.liquidity as liq
    fetched = []
    monkeypatch.setattr(liq, "load_ohlcv", lambda t: None)          # everything cold
    monkeypatch.setattr(liq, "get_history",
                        lambda t, period="3mo": fetched.append(t) or pd.DataFrame())
    monkeypatch.setattr(settings, "liquidity_gate_fetch_workers", 4)
    monkeypatch.setattr(settings, "enable_fetch_data", True)

    budget = {"n": 3}
    liq._prewarm_cold(["A", "B", "C", "D", "E"], budget)
    assert sorted(fetched) == ["A", "B", "C"]                       # capped by the budget
    assert budget["n"] == 0
    assert budget["attempted"] == {"A", "B", "C"}


def test_load_never_refetches_an_attempted_ticker(monkeypatch):
    import src.data.liquidity as liq
    monkeypatch.setattr(liq, "load_ohlcv", lambda t: None)
    monkeypatch.setattr(liq, "get_history",
                        lambda *a, **k: (_ for _ in ()).throw(
                            AssertionError("attempted ticker must not refetch")))
    monkeypatch.setattr(settings, "enable_fetch_data", True)
    budget = {"n": 5, "attempted": {"A"}}
    assert liq._load("A", budget) is None                           # fail-closed, no fetch
    assert budget["n"] == 5                                          # budget untouched


def test_gate_cold_candidates_fetched_exactly_once(monkeypatch):
    """End-to-end through apply_liquidity_gate: the pre-warm fetches each cold
    candidate once; the sequential verdict loop must not fetch again."""
    import src.data.liquidity as liq
    from collections import Counter
    counts: Counter = Counter()
    monkeypatch.setattr(liq, "load_ohlcv", lambda t: None)
    monkeypatch.setattr(liq, "get_history",
                        lambda t, period="3mo": counts.update([t]) or pd.DataFrame())
    monkeypatch.setattr(settings, "enable_fetch_data", True)
    monkeypatch.setattr(settings, "enable_discovery_liquidity_gate", True)
    monkeypatch.setattr(settings, "liquidity_gate_fetch_workers", 4)

    kept = liq.apply_liquidity_gate(["AAA", "BBB"], budget={"n": 10})
    assert kept == []                                                # no data ⇒ fail closed
    assert counts == Counter({"AAA": 1, "BBB": 1})


def test_prewarm_noop_when_workers_1(monkeypatch):
    import src.data.liquidity as liq
    monkeypatch.setattr(liq, "load_ohlcv", lambda t: None)
    monkeypatch.setattr(liq, "get_history",
                        lambda *a, **k: (_ for _ in ()).throw(AssertionError("must not prefetch")))
    monkeypatch.setattr(settings, "liquidity_gate_fetch_workers", 1)
    monkeypatch.setattr(settings, "enable_fetch_data", True)
    budget = {"n": 5}
    liq._prewarm_cold(["A", "B"], budget)                            # legacy path: no-op
    assert budget == {"n": 5}


# ── 5. Options-flow bounded concurrency plumbing ──────────────────────────────

def _stub_yfinance(created):
    def _ticker(t):
        created.append(t)
        return SimpleNamespace(options=[])          # no chains ⇒ scan returns nothing
    return SimpleNamespace(Ticker=_ticker)


def test_options_flow_parallel_scans_every_ticker(monkeypatch):
    import sys
    import src.data.options_flow as of
    created = []
    monkeypatch.setitem(sys.modules, "yfinance", _stub_yfinance(created))
    monkeypatch.setattr(settings, "enable_fetch_data", True)
    monkeypatch.setattr(settings, "options_flow_max_workers", 2)
    out = of.fetch_options_flow(["A", "B", "C"])
    assert out == []
    assert sorted(created) == ["A", "B", "C"]


def test_options_flow_sequential_when_workers_1(monkeypatch):
    import sys
    import src.data.options_flow as of
    created = []
    monkeypatch.setitem(sys.modules, "yfinance", _stub_yfinance(created))
    monkeypatch.setattr(settings, "enable_fetch_data", True)
    monkeypatch.setattr(settings, "options_flow_max_workers", 1)
    out = of.fetch_options_flow(["A", "B"])
    assert out == []
    assert created == ["A", "B"]                    # strict input order (legacy path)
