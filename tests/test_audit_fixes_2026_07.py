"""Tests for the 2026-07-01 audit fixes.

Covers: the save_trades wipe guard, fail-closed _load_trades, the re-entry
cooldown, the default horizon window (zombie-loser flush), the real-fill cost
override price floor, the look-ahead-free LLM pseudo-trade anchor, the
hold-review engine fallback, run-persist provenance (synthesis meta captured
before the hold-reviews clobber the global), the went-dark source detector,
and the scheduler missed-slot helper.
"""

from datetime import datetime, time, timedelta, timezone
from types import SimpleNamespace

import pandas as pd
import pytest

from config.settings import settings


# ── repo.save_trades: wipe guard ──────────────────────────────────────────────

def _mini_trade(ticker, status="CLOSED", **kw):
    t = {"ticker": ticker, "action": "BUY", "status": status,
         "entry_date": "2026-06-01", "entry_price": 10.0}
    t.update(kw)
    return t


def test_save_trades_refuses_drastic_shrink():
    from src.db import repo
    repo.save_trades([_mini_trade(f"T{i}") for i in range(6)])
    with pytest.raises(RuntimeError, match="shrink"):
        repo.save_trades([_mini_trade("T0"), _mini_trade("T1")])
    # The refused save must not have touched the table.
    assert len(repo.load_trades()) == 6


def test_save_trades_allow_shrink_and_small_ledgers():
    from src.db import repo
    # Fresh/small ledgers are never blocked (tests, first runs).
    repo.save_trades([_mini_trade("A")])
    repo.save_trades([_mini_trade("B")])
    assert [t["ticker"] for t in repo.load_trades()] == ["B"]
    # Deliberate maintenance path.
    repo.save_trades([_mini_trade(f"T{i}") for i in range(6)])
    repo.save_trades([_mini_trade("KEEP")], allow_shrink=True)
    assert [t["ticker"] for t in repo.load_trades()] == ["KEEP"]


def test_save_trades_normal_growth_and_modest_shrink_pass():
    from src.db import repo
    repo.save_trades([_mini_trade(f"T{i}") for i in range(6)])
    repo.save_trades([_mini_trade(f"T{i}") for i in range(5)])   # sanitize-drop scale
    repo.save_trades([_mini_trade(f"T{i}") for i in range(9)])   # growth
    assert len(repo.load_trades()) == 9


# ── tracker._load_trades: fail-closed on DB errors ────────────────────────────

def test_load_trades_propagates_db_errors(monkeypatch):
    from src.performance import tracker

    def boom():
        raise RuntimeError("IO Error: file is locked by another process")

    monkeypatch.setattr(tracker.repo, "load_trades", boom)
    with pytest.raises(RuntimeError, match="locked"):
        tracker._load_trades()


# ── record_new_trades: re-entry cooldown ─────────────────────────────────────

def _mk_rec(ticker="TEST", action="BUY", confidence=0.78):
    from src.models import Recommendation
    return Recommendation(
        ticker=ticker, type="STOCK",
        direction="BULLISH" if action == "BUY" else "BEARISH",
        confidence=confidence, action=action, time_horizon="SWING",
        rationale="test", generated_at=datetime.now(timezone.utc),
    )


def _seed_closed(monkeypatch, ticker="TEST", action="BUY", hours_ago=1.0):
    from src.performance import tracker
    exit_dt = (datetime.now(timezone.utc) - timedelta(hours=hours_ago)).isoformat()
    tracker._save_trades([_mini_trade(
        ticker, status="CLOSED", action=action,
        exit_date=exit_dt[:10], exit_datetime=exit_dt, exit_price=11.0, return_pct=1.0,
    )])


def _entry_env(monkeypatch):
    from src.performance import tracker
    monkeypatch.setattr(settings, "enable_intraday_timing", False)
    monkeypatch.setattr(settings, "enable_correlation_sizing", False)
    monkeypatch.setattr(tracker, "_execution_iso", lambda: "2026-06-10T15:00:00+00:00")
    monkeypatch.setattr(tracker, "_fetch_price", lambda t: 50.0)
    monkeypatch.setattr(tracker, "_reference_close", lambda t: None)
    return tracker


def test_reentry_cooldown_blocks_same_direction(monkeypatch):
    tracker = _entry_env(monkeypatch)
    monkeypatch.setattr(settings, "reentry_cooldown_hours", 4.0)
    _seed_closed(monkeypatch, hours_ago=1.0)
    diag = tracker.record_new_trades([_mk_rec()], run_id="cool1")
    assert diag["skipped_reentry_cooldown"] == 1
    assert diag["opened"] == 0


def test_reentry_cooldown_allows_flip_and_expiry(monkeypatch):
    tracker = _entry_env(monkeypatch)
    monkeypatch.setattr(settings, "reentry_cooldown_hours", 4.0)
    _seed_closed(monkeypatch, hours_ago=1.0)
    # Opposite direction (a genuine flip) is never blocked.
    diag = tracker.record_new_trades([_mk_rec(action="SELL", confidence=0.80)], run_id="cool2")
    assert diag["skipped_reentry_cooldown"] == 0 and diag["opened"] == 1
    # Same direction after the window is fine too.
    _seed_closed(monkeypatch, ticker="OLD", hours_ago=9.0)
    diag = tracker.record_new_trades([_mk_rec(ticker="OLD")], run_id="cool3")
    assert diag["skipped_reentry_cooldown"] == 0 and diag["opened"] == 1


def test_reentry_cooldown_disabled(monkeypatch):
    tracker = _entry_env(monkeypatch)
    monkeypatch.setattr(settings, "reentry_cooldown_hours", 0.0)
    _seed_closed(monkeypatch, hours_ago=0.01)
    diag = tracker.record_new_trades([_mk_rec()], run_id="cool4")
    assert diag["opened"] == 1


# ── _horizon_expiry: default window for horizon-less positions ───────────────

def test_horizon_default_window_gives_legacy_trades_a_time_stop(monkeypatch):
    from src.performance import tracker
    monkeypatch.setattr(settings, "horizon_default_window", "1w")
    old = {"ticker": "FUN", "action": "BUY", "confidence": 0.78,
           "entry_datetime": (datetime.now(timezone.utc) - timedelta(days=15)).isoformat()}
    st = tracker._horizon_expiry(old)
    assert st is not None and st["ramp"] == pytest.approx(1.0)


def test_horizon_default_window_empty_restores_legacy_behavior(monkeypatch):
    from src.performance import tracker
    monkeypatch.setattr(settings, "horizon_default_window", "")
    old = {"ticker": "FUN", "action": "BUY", "confidence": 0.78,
           "entry_datetime": (datetime.now(timezone.utc) - timedelta(days=15)).isoformat()}
    assert tracker._horizon_expiry(old) is None


def test_horizon_explicit_target_still_wins(monkeypatch):
    from src.performance import tracker
    monkeypatch.setattr(settings, "horizon_default_window", "1w")
    t = {"ticker": "X", "action": "BUY", "confidence": 0.8, "target_horizon": "1m",
         "entry_datetime": (datetime.now(timezone.utc) - timedelta(days=15)).isoformat()}
    # 15 days held < the trade's OWN 1m (30d) window → still inside, no expiry state.
    assert tracker._horizon_expiry(t) is None


# ── spread: real-fill override respects the price floor ──────────────────────

def test_real_cost_override_skips_sub_dollar_legs(monkeypatch):
    from src.performance import spread
    monkeypatch.setattr(settings, "sim_real_fill_min_price", 1.0)
    spread.set_real_cost_override(0.0008)
    try:
        assert spread._one_side_cost(110.0, "STOCK") == pytest.approx(0.0008)
        # $0.054 warrant: the flat 8 bp would be fantasy — the modeled penny
        # tier (250 bp, commission_model pinned to 'none' by conftest) applies.
        assert spread._one_side_cost(0.054, "STOCK") == pytest.approx(0.0250)
    finally:
        spread.set_real_cost_override(None)


# ── tracker._generated_after_rth_close: pseudo-trade anchor guard ─────────────

def test_generated_after_rth_close():
    from src.performance.tracker import _generated_after_rth_close
    assert _generated_after_rth_close("2026-07-01T20:05:00+00:00") is True    # 16:05 ET
    assert _generated_after_rth_close("2026-07-01T18:30:00+00:00") is False   # 14:30 ET
    assert _generated_after_rth_close("2026-07-01T08:00:00+00:00") is False   # 04:00 ET pre-market
    assert _generated_after_rth_close("2026-07-01") is False                  # date-only
    assert _generated_after_rth_close(None) is False
    assert _generated_after_rth_close("garbage") is False


# ── pipeline._build_hold_reviews: engine fallback ─────────────────────────────

def _fake_signals(tickers):
    return [SimpleNamespace(ticker=t) for t in tickers]


def _rec_obj(ticker, action="BUY", confidence=0.8):
    return SimpleNamespace(ticker=ticker, action=action, confidence=confidence)


def _claude_trade(ticker="XLE"):
    return {"ticker": ticker, "status": "OPEN",
            "llm_synthesis_model": "claude-haiku-4-5-20251001",
            "llm_sentiment_model": "claude-haiku-4-5-20251001"}


def test_hold_review_fallback_covers_dead_pinned_engine(monkeypatch):
    import src.pipeline as pl
    monkeypatch.setattr(settings, "enable_pinned_hold_review", True)
    monkeypatch.setattr(settings, "hold_review_engine_fallback", True)
    monkeypatch.setattr(pl, "fetch_all_news", lambda tks, sectors: [])
    monkeypatch.setattr(pl, "get_snapshots", lambda tks: [])
    monkeypatch.setattr(pl, "build_signals", lambda tickers, *a, **k: _fake_signals(tickers))

    def fake_synth(signals, session=None, force_engine=None, **kw):
        if force_engine == "anthropic":
            return []   # credits exhausted — the pinned engine yields nothing
        return [_rec_obj(s.ticker) for s in signals]
    monkeypatch.setattr(pl, "generate_recommendations", fake_synth)

    reviews = pl._build_hold_reviews(
        [_claude_trade()], run_sent="deepseek", run_synth="deepseek",
        full_recs=[], sectors=[], build_kwargs={}, synth_kwargs={}, session="rth")

    assert set(reviews) == {"XLE"}                       # judged despite the dead opener
    assert getattr(reviews, "engines", {}).get("XLE") == "deepseek"


def test_hold_review_fallback_off_preserves_strict_pinning(monkeypatch):
    import src.pipeline as pl
    monkeypatch.setattr(settings, "enable_pinned_hold_review", True)
    monkeypatch.setattr(settings, "hold_review_engine_fallback", False)
    monkeypatch.setattr(pl, "fetch_all_news", lambda tks, sectors: [])
    monkeypatch.setattr(pl, "get_snapshots", lambda tks: [])
    monkeypatch.setattr(pl, "build_signals", lambda tickers, *a, **k: _fake_signals(tickers))
    monkeypatch.setattr(pl, "generate_recommendations",
                        lambda signals, session=None, force_engine=None, **kw:
                        [] if force_engine == "anthropic" else [_rec_obj(s.ticker) for s in signals])
    reviews = pl._build_hold_reviews(
        [_claude_trade()], "deepseek", "deepseek", [], [], {}, {}, "rth")
    assert reviews == {}                                  # no review ⇒ hold (original Fix #2)


def test_persist_trade_reviews_stamps_fallback_reviewer(monkeypatch):
    import src.pipeline as pl
    captured = {}
    monkeypatch.setattr(pl.repo, "insert_trade_reviews", lambda rows: captured.update(rows={r["ticker"]: r for r in rows}))
    reviews = pl._ReviewMap({"XLE": _rec_obj("XLE"), "USO": _rec_obj("USO")})
    reviews.engines = {"XLE": "deepseek", "USO": "deepseek"}
    open_trades = [
        dict(_claude_trade("XLE"), recommendation_id="r1", confidence=0.8),      # fallback case
        {"ticker": "USO", "status": "OPEN", "recommendation_id": "r2", "confidence": 0.8,
         "llm_synthesis_model": "deepseek-v4-flash", "llm_sentiment_model": "deepseek-v4-flash"},
    ]
    pl._persist_trade_reviews("run1", reviews, open_trades)
    rows = captured["rows"]
    assert rows["XLE"]["synthesis_model"] == "deepseek (fallback)"   # opener anthropic, reviewer deepseek
    assert rows["USO"]["synthesis_model"] == "deepseek-v4-flash"     # pinned — opener model kept


# ── pipeline._persist_run: provenance survives the hold-review clobber ───────

def test_persist_run_uses_snapshotted_synthesis_meta(monkeypatch):
    import src.pipeline as pl
    from src.analysis import claude_analyst as ca

    captured = {}
    monkeypatch.setattr(pl.repo, "insert_run", lambda row: captured.update(run=row))
    monkeypatch.setattr(pl.repo, "insert_run_sources", lambda *a, **k: None)
    monkeypatch.setattr(pl.repo, "insert_recommendations", lambda rows: captured.update(recs=rows))
    monkeypatch.setattr(pl, "_collect_sources", lambda: [])

    # Simulate the hold-review clobber: the process-global now holds the LAST
    # review's engine, NOT the engine that produced the run's recommendations.
    ca._set_synthesis_meta("deepseek", "deepseek-v4-flash")

    rec = _mk_rec("HNGE", "BUY", 0.9)
    now = datetime.now(timezone.utc)
    pl._persist_run(
        "runX", now, now, ["HNGE"], [rec], [rec],
        {}, None, None, 0.78, True, {},
        synthesis_meta={"provider": "deepseek", "model": "deepseek-v4-flash-thinking"},
        sentiment_summary="deepseek×5",
    )
    assert captured["run"]["llm_synthesis_provider"] == "deepseek"
    assert captured["run"]["llm_sentiment_provider"] == "deepseek×5"
    assert captured["recs"][0]["llm_provider"] == "deepseek-v4-flash-thinking"


# ── data_quality.compute_dark_sources ─────────────────────────────────────────

def _source_rows(label, empties, start="2026-06-20"):
    t0 = datetime.fromisoformat(start)
    return [{"source_label": label, "ok": True, "empty": e, "error": None,
             "duration_s": 1.0, "started_at": (t0 + timedelta(hours=i)).isoformat()}
            for i, e in enumerate(empties)]


def test_compute_dark_sources_flags_zero_to_hundred_flip():
    from src.analysis.data_quality import compute_dark_sources
    rows = (
        _source_rows("quiver_congress", [False] * 15 + [True] * 10)   # went dark
        + _source_rows("ticker_events", [True] * 25)                  # always empty — not a flip
        + _source_rows("analyst", [False] * 25)                       # healthy
        + _source_rows("insider", [False] * 15 + [True] * 10)         # known-dead — excluded
        + _source_rows("newfeed", [False] * 3 + [True] * 6)           # too little history
    )
    dark = compute_dark_sources(pd.DataFrame(rows))
    assert [d["source"] for d in dark] == ["quiver_congress"]
    assert dark[0]["prior_empty_pct"] == 0.0


# ── scheduler: missed-slot helper ─────────────────────────────────────────────

def test_missed_slots_between_skips_weekend():
    from src.scheduler.runner import _missed_slots_between
    slots = [(time(9, 30), "rth"), (time(10, 0), "rth")]
    # Friday 2026-06-26 09:00 → Monday 2026-06-29 09:00 (grace 60 s): only
    # Friday's two slots fall in the gap; the weekend contributes nothing and
    # Monday's slots are after the window end.
    missed = _missed_slots_between(
        datetime(2026, 6, 26, 9, 0), datetime(2026, 6, 29, 9, 0), slots, grace=60)
    assert missed == [datetime(2026, 6, 26, 9, 30), datetime(2026, 6, 26, 10, 0)]
