"""Fix #2 — every-tick, opener-pinned hold-review (2026-06-14).

Entry is gated by the LLM synthesis confidence; so is the hold/close decision —
and on EVERY tick a held position is re-judged by the SAME synthesis AND sentiment
engines that opened it, on freshly refetched news + prices. The pipeline's
``_build_hold_reviews`` guarantees each ``hold_reviews`` entry is opener-pinned, so
``_evaluate_decay`` simply trusts a present review. The aggregator path survives only
as the backstop for legacy / rule-based-opened trades.
"""

from datetime import datetime, timezone
from types import SimpleNamespace

import pytest

from config.settings import settings
from src.models import NewsArticle, Recommendation, TickerSignal, TickerSnapshot
from src.performance import tracker

# Methods disabled to keep the build_signals overlay-gating test cheap + LLM/
# network-free (insider stays on with an empty list to pass the "all methods
# disabled" guard; ext_gap is enabled explicitly in the test).
_HEAVY_METHOD_FLAGS = [
    "enable_news_sentiment", "enable_sentiment_velocity", "enable_technical_analysis",
    "enable_options_flow", "enable_sec_filings", "enable_put_call", "enable_gex",
    "enable_vwap", "enable_pattern_recognition", "enable_price_momentum",
    "enable_sector_relative_momentum", "enable_market_relative_momentum",
    "enable_money_flow", "enable_trend_strength", "enable_pead", "enable_iv_rank",
    "enable_iv_expr", "enable_cointegration", "enable_cross_sectional",
    "enable_adaptive_weights", "enable_market_mode_switching", "enable_catalyst_timing",
]


def _rec(ticker="XLE", action="BUY", confidence=0.85, direction=None):
    return Recommendation(
        ticker=ticker, type="ETF",
        direction=direction or ("BULLISH" if action == "BUY" else "BEARISH"),
        action=action, confidence=confidence, time_horizon="1w",
        rationale="r", generated_at=datetime.now(timezone.utc),
    )


def _claude_trade(action="BUY", entry_conf=0.85,
                  synth_model="claude-haiku-4-5-20251001",
                  sent_model="claude-haiku-4-5-20251001"):
    """A position opened by LLM engines whose *aggregator* signal was weak —
    the dead-on-arrival case fix #2 targets (entry_conf is the LLM number)."""
    return {
        "ticker": "XLE", "type": "ETF", "action": action, "status": "OPEN",
        "confidence": entry_conf,
        "llm_synthesis_model": synth_model, "llm_sentiment_model": sent_model,
        "entry_date": "2026-06-13", "entry_datetime": "2026-06-13T15:00:00+00:00",
        "entry_price": 57.0, "current_price": 57.5,
        "current_price_datetime": "2026-06-14T15:00:00+00:00",
        # Aggregator was near zero at entry — it must NOT drive the exit now.
        "signal_at_entry": {"combined_score": 0.05, "confidence": 0.06},
    }


def _hostile_aggregator():
    """An aggregator signal the OLD logic would have exited on immediately."""
    return SimpleNamespace(combined_score=-0.6, confidence=0.10)


# ── _evaluate_decay: the opener's pinned review governs ───────────────────────

def test_weak_aggregator_does_not_close_llm_position():
    # Opener review still bullish + confident → hold, despite a hostile aggregator.
    assert tracker._evaluate_decay(
        _claude_trade(), today_signal=_hostile_aggregator(), macro_regime_context=None,
        hold_review=_rec(action="BUY", confidence=0.85),
    ) is None


def test_flip_closes():
    assert tracker._evaluate_decay(
        _claude_trade(action="BUY"), today_signal=None, macro_regime_context=None,
        hold_review=_rec(action="SELL", confidence=0.8),
    ) == "llm_signal_flipped"


def test_hold_keeps_position():
    # A neutral HOLD is NOT a close (the engine isn't calling the other way).
    for conf in (0.9, 0.5, 0.2):
        assert tracker._evaluate_decay(
            _claude_trade(action="BUY"), today_signal=None, macro_regime_context=None,
            hold_review=_rec(action="HOLD", confidence=conf),
        ) is None


def test_same_direction_conviction_collapse_closes():
    # Still a BUY, but conviction fell below floor(0.85)=0.5525 → close.
    assert tracker._evaluate_decay(
        _claude_trade(entry_conf=0.85), today_signal=None, macro_regime_context=None,
        hold_review=_rec(action="BUY", confidence=0.40),
    ) == "llm_confidence_loss"


def test_confidence_floor_boundary():
    floor = max(0.45, 0.65 * 0.85)   # 0.5525
    assert tracker._evaluate_decay(
        _claude_trade(entry_conf=0.85), today_signal=None, macro_regime_context=None,
        hold_review=_rec(action="BUY", confidence=floor - 0.01)) == "llm_confidence_loss"
    assert tracker._evaluate_decay(
        _claude_trade(entry_conf=0.85), today_signal=None, macro_regime_context=None,
        hold_review=_rec(action="BUY", confidence=floor + 0.01)) is None


def test_no_review_holds():
    # The opener's engine produced no review this tick → hold (not the aggregator).
    assert tracker._evaluate_decay(
        _claude_trade(action="BUY"), today_signal=_hostile_aggregator(),
        macro_regime_context=None, hold_review=None,
    ) is None


def test_macro_regime_exit_overrides_everything():
    assert tracker._evaluate_decay(
        _claude_trade(action="BUY"), today_signal=None,
        macro_regime_context=SimpleNamespace(regime="PANIC"),
        hold_review=_rec(action="BUY", confidence=0.99),
    ) == "macro_regime_exit"


# ── aggregator backstop still governs legacy / rule-based trades ──────────────

def test_legacy_trade_uses_aggregator_backstop(monkeypatch):
    monkeypatch.setattr(settings, "enable_signal_decay_exits", True)
    legacy = {                                   # no llm_*_model
        "ticker": "XLE", "type": "ETF", "action": "BUY", "status": "OPEN",
        "confidence": 0.85, "entry_price": 57.0, "current_price": 57.5,
        "signal_at_entry": {"combined_score": 0.5, "confidence": 0.85},
    }
    assert tracker._evaluate_decay(
        legacy, today_signal=_hostile_aggregator(), macro_regime_context=None,
        hold_review=None) == "signal_flipped"


def test_rule_based_without_review_uses_aggregator_backstop(monkeypatch):
    # Rule-based / legacy-opened trade with NO LLM review this tick → the
    # aggregator backstop is the true last resort (#4). It no longer fires when a
    # review IS available — see test_rule_based_with_review_uses_llm_path.
    monkeypatch.setattr(settings, "enable_signal_decay_exits", True)
    trade = _claude_trade(synth_model="rule-based (no LLM)", sent_model="none")
    trade["signal_at_entry"] = {"combined_score": 0.5, "confidence": 0.85}
    assert tracker._evaluate_decay(
        trade, today_signal=_hostile_aggregator(), macro_regime_context=None,
        hold_review=None) == "signal_flipped"


def test_rule_based_with_review_uses_llm_path(monkeypatch):
    # #4 hardening: a rule-based / legacy-opened trade now gets an LLM review
    # (built with THIS run's engine — pipeline._build_hold_reviews), and that
    # review GOVERNS the exit. The poor aggregator-decay backstop (30%-win
    # historically) no longer fires when a review exists.
    monkeypatch.setattr(settings, "enable_signal_decay_exits", True)
    trade = _claude_trade(synth_model="rule-based (no LLM)", sent_model="none")
    trade["signal_at_entry"] = {"combined_score": 0.5, "confidence": 0.85}
    # Review flips direction → LLM flip close (NOT the aggregator's signal_flipped).
    assert tracker._evaluate_decay(
        trade, today_signal=_hostile_aggregator(), macro_regime_context=None,
        hold_review=_rec(action="SELL", confidence=0.8)) == "llm_signal_flipped"
    # Same-direction confident review → HOLD, despite the hostile aggregator.
    assert tracker._evaluate_decay(
        trade, today_signal=_hostile_aggregator(), macro_regime_context=None,
        hold_review=_rec(action="BUY", confidence=0.85)) is None


def test_disabled_review_falls_back_to_aggregator(monkeypatch):
    monkeypatch.setattr(settings, "enable_llm_hold_review", False)
    monkeypatch.setattr(settings, "enable_signal_decay_exits", True)
    assert tracker._evaluate_decay(
        _claude_trade(action="BUY"), today_signal=_hostile_aggregator(),
        macro_regime_context=None, hold_review=_rec(action="BUY", confidence=0.9),
    ) == "signal_flipped"


@pytest.mark.parametrize("model,expected", [
    ("claude-haiku-4-5-20251001", "anthropic"),
    ("claude-sonnet-4-6", "anthropic"),
    ("deepseek-v4-flash", "deepseek"),
    ("rule-based (no LLM)", "rule-based"),
    ("none", None), ("", None), (None, None),
])
def test_provider_of_synth_model(model, expected):
    assert tracker._provider_of_synth_model(model) == expected


# ── monitor end-to-end: opener review acts regardless of the run's A/B engine ─

def test_monitor_closes_on_opener_review_regardless_of_run_engine(tmp_path, monkeypatch):
    monkeypatch.setattr(tracker, "TRADES_FILE", tmp_path / "no-legacy.json")
    monkeypatch.setattr(settings, "enable_intraday_exit", False)
    tracker._save_trades([_claude_trade(action="BUY")])     # Claude-opened
    closed = tracker.monitor_open_positions(
        signals_by_ticker={"XLE": _hostile_aggregator()},   # hostile — must be ignored
        hold_reviews={"XLE": _rec(action="SELL", confidence=0.8)},  # opener (Claude) flip
        run_synthesis_provider="deepseek",                  # run engine differs — no longer matters
        hold_prompt_active=False,
    )
    assert closed == 1
    assert tracker._load_trades()[0]["exit_reason"] == "llm_signal_flipped"


def test_monitor_holds_when_no_review(tmp_path, monkeypatch):
    monkeypatch.setattr(tracker, "TRADES_FILE", tmp_path / "no-legacy.json")
    monkeypatch.setattr(settings, "enable_intraday_exit", False)
    tracker._save_trades([_claude_trade(action="BUY")])
    closed = tracker.monitor_open_positions(
        signals_by_ticker={"XLE": _hostile_aggregator()},
        hold_reviews={},                                    # opener produced no review this tick
        run_synthesis_provider="anthropic", hold_prompt_active=False,
    )
    assert closed == 0
    assert tracker._load_trades()[0]["status"] == "OPEN"


# ── pipeline _build_hold_reviews: fresh data + per-engine pinning ─────────────

def _fake_signals(tickers):
    return [SimpleNamespace(ticker=t) for t in tickers]


def test_build_hold_reviews_pins_each_position_to_its_engines(monkeypatch):
    import src.pipeline as pl
    monkeypatch.setattr(settings, "enable_pinned_hold_review", True)
    open_trades = [
        _claude_trade(),  # XLE — Claude / Claude
        {"ticker": "USO", "status": "OPEN",
         "llm_synthesis_model": "deepseek-v4-flash", "llm_sentiment_model": "deepseek-v4-flash"},
    ]
    seen = {"news": [], "snaps": [], "build": [], "synth": []}
    monkeypatch.setattr(pl, "fetch_all_news", lambda tks, sectors: seen["news"].append(sorted(tks)) or [])
    monkeypatch.setattr(pl, "get_snapshots", lambda tks: seen["snaps"].append(sorted(tks)) or [])

    def fake_build(tickers, articles, snapshots=None, session=None, force_sentiment_engine=None, **kw):
        seen["build"].append((tuple(tickers), force_sentiment_engine))
        return _fake_signals(tickers)
    monkeypatch.setattr(pl, "build_signals", fake_build)

    def fake_synth(signals, session=None, force_engine=None, **kw):
        seen["synth"].append((tuple(s.ticker for s in signals), force_engine))
        return [_rec(s.ticker, "BUY", 0.8) for s in signals]
    monkeypatch.setattr(pl, "generate_recommendations", fake_synth)

    reviews = pl._build_hold_reviews(
        open_trades, run_sent="anthropic", run_synth="anthropic", full_recs=[],
        sectors=["Tech"], build_kwargs={}, synth_kwargs={}, session="rth")

    assert set(reviews) == {"XLE", "USO"}
    assert seen["news"] == [["USO", "XLE"]]            # ONE fresh news fetch for the held set
    assert seen["snaps"] == [["USO", "XLE"]]           # ONE fresh price fetch
    assert dict(seen["build"]) == {("XLE",): "anthropic", ("USO",): "deepseek"}
    assert dict(seen["synth"]) == {("XLE",): "anthropic", ("USO",): "deepseek"}


def test_build_hold_reviews_failed_engine_yields_no_review(monkeypatch):
    import src.pipeline as pl
    monkeypatch.setattr(settings, "enable_pinned_hold_review", True)
    open_trades = [{"ticker": "USO", "status": "OPEN",
                    "llm_synthesis_model": "deepseek-v4-flash",
                    "llm_sentiment_model": "deepseek-v4-flash"}]
    monkeypatch.setattr(pl, "fetch_all_news", lambda tks, sectors: [])
    monkeypatch.setattr(pl, "get_snapshots", lambda tks: [])
    monkeypatch.setattr(pl, "build_signals", lambda *a, **k: _fake_signals(["USO"]))
    monkeypatch.setattr(pl, "generate_recommendations", lambda *a, **k: [])  # forced engine failed
    reviews = pl._build_hold_reviews(
        open_trades, "deepseek", "deepseek", [], [], {}, {}, "rth")
    assert reviews == {}


def test_build_hold_reviews_off_mode_reuses_matching_engine_only(monkeypatch):
    import src.pipeline as pl
    monkeypatch.setattr(settings, "enable_pinned_hold_review", False)
    open_trades = [
        _claude_trade(),  # XLE — Claude / Claude (matches the run)
        {"ticker": "USO", "status": "OPEN",
         "llm_synthesis_model": "deepseek-v4-flash", "llm_sentiment_model": "deepseek-v4-flash"},
    ]
    full = [_rec("XLE", "BUY", 0.8), _rec("USO", "BUY", 0.7)]
    monkeypatch.setattr(pl, "fetch_all_news",
                        lambda *a, **k: (_ for _ in ()).throw(AssertionError("off mode must not fetch")))
    reviews = pl._build_hold_reviews(
        open_trades, run_sent="anthropic", run_synth="anthropic", full_recs=full,
        sectors=[], build_kwargs={}, synth_kwargs={}, session="rth")
    assert set(reviews) == {"XLE"}                     # only the engine-matched position


def test_build_hold_reviews_covers_rule_based_with_run_engine(monkeypatch):
    # #4: a rule-based / legacy-opened position (no LLM opener to pin) is reviewed
    # with THIS run's engine, so it still gets an LLM hold-review instead of being
    # left to the aggregator-decay backstop. An LLM-opened position stays pinned.
    import src.pipeline as pl
    monkeypatch.setattr(settings, "enable_pinned_hold_review", True)
    open_trades = [
        _claude_trade(),                                  # XLE — Claude-opened → pinned anthropic
        {"ticker": "USO", "status": "OPEN",               # rule-based-opened
         "llm_synthesis_model": "rule-based (no LLM)", "llm_sentiment_model": "none"},
    ]
    seen = {"synth": []}
    monkeypatch.setattr(pl, "fetch_all_news", lambda tks, sectors: [])
    monkeypatch.setattr(pl, "get_snapshots", lambda tks: [])
    monkeypatch.setattr(pl, "build_signals",
                        lambda tickers, *a, **k: _fake_signals(tickers))

    def fake_synth(signals, session=None, force_engine=None, **kw):
        seen["synth"].append((tuple(s.ticker for s in signals), force_engine))
        return [_rec(s.ticker, "BUY", 0.8) for s in signals]
    monkeypatch.setattr(pl, "generate_recommendations", fake_synth)

    reviews = pl._build_hold_reviews(
        open_trades, run_sent="deepseek", run_synth="deepseek", full_recs=[],
        sectors=[], build_kwargs={}, synth_kwargs={}, session="rth")

    assert set(reviews) == {"XLE", "USO"}                 # BOTH reviewed
    # XLE pinned to its opener (anthropic); USO (rule-based) uses the run engine (deepseek).
    assert dict(seen["synth"]) == {("XLE",): "anthropic", ("USO",): "deepseek"}


def test_build_hold_reviews_skips_rule_based_when_run_not_llm(monkeypatch):
    # If THIS run has no LLM engine either, there is nothing to review a
    # rule-based trade with → left to the aggregator backstop (no fetch happens).
    import src.pipeline as pl
    monkeypatch.setattr(settings, "enable_pinned_hold_review", True)
    open_trades = [{"ticker": "USO", "status": "OPEN",
                    "llm_synthesis_model": "rule-based (no LLM)", "llm_sentiment_model": "none"}]
    monkeypatch.setattr(pl, "fetch_all_news",
                        lambda *a, **k: (_ for _ in ()).throw(AssertionError("must not fetch")))
    reviews = pl._build_hold_reviews(
        open_trades, run_sent="rule-based", run_synth="rule-based", full_recs=[],
        sectors=[], build_kwargs={}, synth_kwargs={}, session="rth")
    assert reviews == {}


# ── force_engine plumbing (synthesis + sentiment) ────────────────────────────

def _ticker_signal(ticker="AAA"):
    return TickerSignal(
        ticker=ticker, direction="BULLISH", confidence=0.8,
        action_suggestion="BUY", news_sentiment_score=0.2, sentiment_score=0.2,
        insider_score=0.0, technical_score=0.2, sources_agreeing=2,
        key_reasons=["r"], rationale="r", price=10.0,
    )


def test_generate_recommendations_force_engine_no_fallback(monkeypatch):
    import src.analysis.claude_analyst as ca
    called = []

    def boom(_p, model=None):   # _call_claude_analyst now takes an optional model
        called.append("anthropic")
        raise RuntimeError("forced engine down")

    def ds(_p):
        called.append("deepseek")
        return "[]"

    monkeypatch.setattr(ca, "_call_claude_analyst", boom)
    monkeypatch.setattr(ca, "_call_deepseek_analyst", ds)
    out = ca.generate_recommendations([_ticker_signal()], force_engine="anthropic")
    assert out == []                 # no rule-based fill, no cross-engine fallback
    assert called == ["anthropic"]   # deepseek never tried


def test_analyse_sentiment_force_engine_pins_to_anthropic(monkeypatch):
    import src.analysis.sentiment as sent
    # An anthropic pin is only honored when Claude sentiment is enabled; with it
    # off (the default) the pin coerces to deepseek (Claude never scores sentiment).
    monkeypatch.setattr(sent.settings, "enable_claude_sentiment", True)
    calls = []

    class _Msgs:
        def create(self, **kw):
            calls.append("anthropic")
            return SimpleNamespace(
                content=[SimpleNamespace(text='{"score": 0.5, "rationale": "r"}')])

    monkeypatch.setattr(sent, "_get_haiku", lambda: SimpleNamespace(messages=_Msgs()))
    monkeypatch.setattr(sent, "_get_deepseek",
                        lambda: (_ for _ in ()).throw(AssertionError("deepseek used under force anthropic")))
    art = NewsArticle(title="FDA approval", summary="material catalyst " * 5,
                      url="u", source="Reuters", published_at=datetime.now(timezone.utc))
    score, _ = sent.analyse_sentiment("X", [art], force_engine="anthropic")
    assert calls == ["anthropic"]


# ── review trajectory: DB table + persistence + plot (the analytics path) ─────

@pytest.fixture(autouse=True)
def _restore_repo_rw():
    """Importing dashboard.data flips repo to read-only globally; reset after each
    test so DB writes in other tests aren't affected."""
    yield
    from src.db import repo
    repo.set_read_only(False)


def test_confidence_floor_matches_evaluate_decay():
    # The persisted floor must equal what the exit gate uses.
    assert tracker._confidence_floor(0.85) == max(0.45, 0.65 * 0.85)
    assert tracker._confidence_floor(None) == 0.45


def test_trade_reviews_table_roundtrip():
    from src.db import repo
    repo.insert_trade_reviews([{
        "run_id": "r1", "reviewed_at": "2026-06-13T15:00:00+00:00", "ticker": "AAA",
        "position_id": "p1", "entry_datetime": "2026-06-13T14:00:00+00:00",
        "confidence": 0.70, "action": "BUY", "direction": "BULLISH", "conf_floor": 0.55,
        "entry_confidence": 0.85, "entry_action": "BUY", "price": 10.0, "return_pct": 1.0,
        "synthesis_model": "claude-haiku-4-5-20251001", "sentiment_model": "deepseek-v4-flash",
    }])
    df = repo.fetch_df("SELECT ticker, action, confidence FROM trade_reviews WHERE ticker = 'AAA'")
    assert list(df["ticker"]) == ["AAA"]
    assert df.iloc[0]["action"] == "BUY"
    assert abs(float(df.iloc[0]["confidence"]) - 0.70) < 1e-9


def test_persist_trade_reviews_builds_rows():
    import src.pipeline as pl
    from src.db import repo
    trade = _claude_trade(action="BUY", entry_conf=0.85)
    trade["recommendation_id"] = "pos-xle-1"
    trade["return_pct"] = -1.9
    pl._persist_trade_reviews("run1", {"XLE": _rec("XLE", "SELL", 0.48)}, [trade])
    df = repo.fetch_df("SELECT * FROM trade_reviews WHERE ticker = 'XLE'")
    assert len(df) == 1
    row = df.iloc[0]
    assert row["action"] == "SELL" and row["entry_action"] == "BUY"
    assert abs(float(row["confidence"]) - 0.48) < 1e-9
    assert abs(float(row["entry_confidence"]) - 0.85) < 1e-9
    assert abs(float(row["conf_floor"]) - max(0.45, 0.65 * 0.85)) < 1e-9
    assert row["position_id"] == "pos-xle-1"
    assert abs(float(row["price"]) - 57.5) < 1e-9


def test_persist_trade_reviews_empty_is_noop():
    import src.pipeline as pl
    pl._persist_trade_reviews("run1", {}, [])   # must not raise


def test_confidence_timeline_fig_builds():
    import pandas as pd
    from dashboard import figures as fig_mod
    df = pd.DataFrame({
        "reviewed_at": ["2026-06-13T15:00:00+00:00", "2026-06-13T15:30:00+00:00",
                        "2026-06-13T16:00:00+00:00"],
        "confidence": [0.85, 0.70, 0.48], "action": ["BUY", "BUY", "SELL"],
        "direction": ["BULLISH", "BULLISH", "BEARISH"], "conf_floor": [0.55] * 3,
        "entry_confidence": [0.85] * 3, "price": [57.0, 56.4, 55.9],
        "return_pct": [0.0, -1.1, -1.9],
    })
    trades = [{"ticker": "XLE", "action": "BUY", "status": "CLOSED",
               "entry_datetime": "2026-06-13T15:00:00+00:00", "entry_price": 57.0,
               "exit_datetime": "2026-06-13T16:00:00+00:00", "exit_price": 55.9,
               "exit_reason": "llm_signal_flipped"}]
    names = [t.name for t in fig_mod.confidence_timeline_fig(df, trades).data]
    assert "Review confidence" in names and "Price" in names and "close floor" in names
    assert len(fig_mod.confidence_timeline_fig(pd.DataFrame(), []).data) == 0   # empty placeholder


def test_review_timeline_section_no_history_message():
    from dashboard import app as dash_app
    import dash
    out = dash_app._review_timeline_section("ZZZZ")   # nothing recorded → friendly message
    assert isinstance(out, dash.html.Div)


# ── session-invariant scoring (enable_extended_signal_profile, default OFF) ───

def test_session_profile_off_removes_synthesis_block(monkeypatch):
    import src.analysis.claude_analyst as ca
    monkeypatch.setattr(settings, "enable_extended_signal_profile", False)
    cap = {}

    def fake(p, **kwargs):   # tolerate model= / thinking= kwargs from the bake-off
        cap["p"] = p
        return "[]"

    monkeypatch.setattr(ca, "_call_claude_analyst", fake)
    monkeypatch.setattr(ca, "_call_deepseek_analyst", fake)
    ca.generate_recommendations([_ticker_signal()], session="extended")
    assert "SESSION CONTEXT" not in cap["p"]          # prompt is session-invariant


def test_session_profile_on_adds_synthesis_block(monkeypatch):
    import src.analysis.claude_analyst as ca
    monkeypatch.setattr(settings, "enable_extended_signal_profile", True)
    cap = {}

    def fake(p, **kwargs):   # tolerate model= / thinking= kwargs from the bake-off
        cap["p"] = p
        return "[]"

    monkeypatch.setattr(ca, "_call_claude_analyst", fake)
    monkeypatch.setattr(ca, "_call_deepseek_analyst", fake)
    ca.generate_recommendations([_ticker_signal()], session="extended")
    assert "SESSION CONTEXT" in cap["p"]              # legacy session-adaptive prompt


def test_overlay_gated_off_by_default(monkeypatch):
    # With the profile OFF (default), build_signals must NOT apply the weight
    # overlay even in an extended session.
    import src.signals.aggregator as agg
    monkeypatch.setattr(settings, "enable_extended_signal_profile", False)
    monkeypatch.setattr(agg, "_extended_session_weight_overlay",
                        lambda *_a, **_k: (_ for _ in ()).throw(AssertionError("overlay applied with profile OFF")))
    # Disable every method except insider (kept on with an empty list to pass the
    # "all methods disabled" guard) + ext_gap, so the build is cheap and LLM-free.
    for flag in _HEAVY_METHOD_FLAGS:
        monkeypatch.setattr(settings, flag, False)
    monkeypatch.setattr(settings, "enable_insider_trades", True)
    monkeypatch.setattr(settings, "enable_extended_gap", True)
    called = {"rth": False, "ext": False}

    def fake_gap(ticker, price, session=None, now=None):
        called["ext" if session and session != "rth" else "rth"] = True
        return (0.4 if session and session != "rth" else 0.0), 0.0
    monkeypatch.setattr(agg, "compute_extended_gap_score", fake_gap)

    snap = TickerSnapshot(ticker="AAA", price=10.0, pct_change_1d=0.0, pct_change_5d=0.0, volume=1000)
    # session="extended" must not raise (overlay gated off); session="rth" must
    # still call ext_gap (method set identical across sessions).
    agg.build_signals(["AAA"], [], insider_trades=[], snapshots=[snap], session="extended")
    agg.build_signals(["AAA"], [], insider_trades=[], snapshots=[snap], session="rth")
    assert called["ext"] and called["rth"]


def test_review_timeline_section_renders_graph_with_data():
    import dash
    import src.pipeline as pl
    from dashboard import app as dash_app
    trade = _claude_trade(action="BUY", entry_conf=0.85)
    trade["recommendation_id"] = "p1"
    trade["return_pct"] = -1.0
    pl._persist_trade_reviews("run1", {"XLE": _rec("XLE", "SELL", 0.45)}, [trade])

    out = dash_app._review_timeline_section("XLE")     # click-through render path
    graphs = []

    def walk(n):
        if isinstance(n, dash.dcc.Graph):
            graphs.append(n)
        ch = getattr(n, "children", None)
        if isinstance(ch, (list, tuple)):
            for c in ch:
                walk(c)
        elif ch is not None:
            walk(ch)

    walk(out)
    assert len(graphs) == 1                            # the confidence-over-time chart rendered
