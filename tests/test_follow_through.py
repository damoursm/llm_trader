"""Follow-through mechanism (2026-08-25): selection rules, the mechanical
entry recorder, and the one-session exit. Each guard here is one that would
fail INVISIBLY (a silently-empty candidate set, a swing exit closing an ft
trade, a re-entered episode) — so each is pinned mechanically."""

from __future__ import annotations

from datetime import date, timedelta
from types import SimpleNamespace

import pytest

from config.settings import settings
import src.signals.follow_through as ft
from src.performance import tracker


# ── selection: tail + level + first-episode-day ─────────────────────────────

def _wire(monkeypatch, scores, prev_selected=frozenset()):
    """Wire compute_follow_through's collaborators to synthetic data. `scores`:
    ticker -> (exit_score, cohort_ds). All tickers pass Gate-4."""
    monkeypatch.setattr(settings, "enable_follow_through", True)
    monkeypatch.setattr(ft, "_gate4_ok", lambda *a, **k: True)
    monkeypatch.setattr(ft, "_prev_selected_tickers", lambda: set(prev_selected))
    hist = {t: {"2026-08-20": {"ds": ds, "c_abs": 0.1,
                               "scores": {"news": 0.1}}}
            for t, (sc, ds) in scores.items()}
    monkeypatch.setattr(ft, "_panel_history", lambda n, methods=(): hist)

    import src.analysis.ml_exit_dataset as me
    monkeypatch.setattr(me, "_load_exit_artifact",
                        lambda: {"model": object(), "features": ["days_held"]})
    import numpy as np
    monkeypatch.setattr(ft, "_score_rows",
                        lambda art, order, pend: np.array(
                            [scores[r["ticker"]][0] for r in pend], dtype=float))
    import src.analysis.exit_methods as em
    monkeypatch.setattr(em, "method_horizon_days", lambda t: 0.0)

    import pandas as pd
    idx = pd.to_datetime(["2026-08-20"])
    fake = pd.DataFrame({"Close": [100.0], "Volume": [1e7]}, index=idx)
    import src.data.cache as cache
    monkeypatch.setattr(cache, "load_ohlcv", lambda tk, interval="1d": fake)
    # _method_scores_from_signal reads a subset of fields by DIRECT attribute
    # access (the documented contract in test_exit_uses_current_strategy) —
    # real TickerSignals always carry them; the stand-ins must too.
    base = dict(price=100.0, combined_score=0.0, direction="BULLISH",
                sentiment_score=0.0, technical_score=0.0, insider_score=0.0,
                put_call_score=0.0, max_pain_score=0.0, oi_skew_score=0.0,
                vwap_score=0.0, pattern_score=0.0, momentum_score=0.0,
                sector_momentum_score=0.0, money_flow_score=0.0,
                trend_strength_score=0.0, pead_score=0.0, iv_rank_score=0.0)
    sigs = {t: SimpleNamespace(ticker=t, **base) for t in scores}
    return sigs


def test_tail_level_and_direction(monkeypatch):
    # 40 tickers: one deep-negative (tail+level), one negative-but-shallow
    # (tail w/o level), the rest mildly positive. dir = MINUS the cohort ds.
    scores = {f"T{i:02d}": (0.3, 1.0) for i in range(38)}
    scores["AAA"] = (-0.9, 1.0)       # long cohort dying -> SHORT follow-through
    scores["BBB"] = (-0.3, -1.0)      # inside tail but above ft_score_max
    monkeypatch.setattr(settings, "ft_tail_pct", 0.05)
    monkeypatch.setattr(settings, "ft_score_max", -0.50)
    sigs = _wire(monkeypatch, scores)
    out = ft.compute_follow_through(sigs)
    assert out["AAA"]["selected"] is True
    assert out["AAA"]["dir"] == -1.0
    assert out["BBB"]["selected"] is False          # level guard (abstention)
    assert sum(1 for r in out.values() if r["selected"]) == 1


def test_first_episode_day_guard(monkeypatch):
    scores = {f"T{i:02d}": (0.3, 1.0) for i in range(39)}
    scores["AAA"] = (-0.9, -1.0)
    sigs = _wire(monkeypatch, scores, prev_selected={"AAA"})
    out = ft.compute_follow_through(sigs)
    assert out["AAA"]["selected"] is False          # episode continuation


def test_fail_soft_without_artifact(monkeypatch):
    monkeypatch.setattr(settings, "enable_follow_through", True)
    import src.analysis.ml_exit_dataset as me
    monkeypatch.setattr(me, "_load_exit_artifact", lambda: None)
    assert ft.compute_follow_through({"AAA": SimpleNamespace(price=100.0)}) == {}


def test_disabled_is_empty(monkeypatch):
    monkeypatch.setattr(settings, "enable_follow_through", False)
    assert ft.compute_follow_through({"AAA": SimpleNamespace(price=100.0)}) == {}


# ── the one-session exit rule ───────────────────────────────────────────────

def _ft_trade(days_ago: int) -> dict:
    d = date.today() - timedelta(days=days_ago)
    return {"ticker": "AAA", "action": "BUY", "direction": "BULLISH",
            "entry_mechanism": "follow_through", "status": "OPEN",
            "entry_date": d.isoformat(), "entry_price": 100.0,
            "current_price": 101.0, "position_size_multiplier": 0.5}


def test_ft_exit_holds_day_zero(monkeypatch):
    monkeypatch.setattr(tracker, "_adverse_stop_triggered", lambda t: False)
    assert tracker._ft_exit_reason(_ft_trade(0), None) is None


def test_ft_exit_fires_next_session_after_cutoff(monkeypatch):
    monkeypatch.setattr(tracker, "_adverse_stop_triggered", lambda t: False)
    monkeypatch.setattr(tracker, "_trading_days_held", lambda ed: 1)
    monkeypatch.setattr(tracker, "_session_of_iso", lambda x: "rth")

    class _T:
        _hm = "15:31"

        def strftime(self, fmt):
            return self._hm

        def isoformat(self, timespec=None):
            return "2026-08-25T15:31:00+00:00"

    class _FakeDT:
        @staticmethod
        def now(tz=None):
            return _T()
    monkeypatch.setattr(tracker, "datetime", _FakeDT)
    assert tracker._ft_exit_reason(_ft_trade(1), None) == "ft_horizon"


def test_ft_exit_waits_before_cutoff_and_off_rth(monkeypatch):
    monkeypatch.setattr(tracker, "_adverse_stop_triggered", lambda t: False)
    monkeypatch.setattr(tracker, "_trading_days_held", lambda ed: 1)

    class _T:
        _hm = "10:00"

        def strftime(self, fmt):
            return self._hm

        def isoformat(self, timespec=None):
            return "2026-08-25T10:00:00+00:00"

    class _FakeDT:
        @staticmethod
        def now(tz=None):
            return _T()
    monkeypatch.setattr(tracker, "datetime", _FakeDT)
    monkeypatch.setattr(tracker, "_session_of_iso", lambda x: "rth")
    assert tracker._ft_exit_reason(_ft_trade(1), None) is None       # before 15:30
    monkeypatch.setattr(tracker, "_session_of_iso", lambda x: "overnight")
    assert tracker._ft_exit_reason(_ft_trade(1), None) is None       # off-RTH


def test_ft_exit_hard_stop_and_safety(monkeypatch):
    monkeypatch.setattr(tracker, "_adverse_stop_triggered", lambda t: False)
    monkeypatch.setattr(tracker, "_trading_days_held",
                        lambda ed: int(settings.ft_max_hold_days))
    assert tracker._ft_exit_reason(_ft_trade(5), None) == "ft_horizon"
    # adverse stop takes precedence over everything
    monkeypatch.setattr(tracker, "_adverse_stop_triggered", lambda t: True)
    assert tracker._ft_exit_reason(_ft_trade(0), None) == "adverse_stop"
    # PANIC closes a LONG (safety), never triggered for the short side
    monkeypatch.setattr(tracker, "_adverse_stop_triggered", lambda t: False)
    monkeypatch.setattr(tracker, "_trading_days_held", lambda ed: 0)
    panic = SimpleNamespace(regime="PANIC")
    assert tracker._ft_exit_reason(_ft_trade(0), panic) == "macro_regime_exit"
    short = dict(_ft_trade(0), action="SELL", direction="BEARISH")
    assert tracker._ft_exit_reason(short, panic) is None


# ── the mechanical entry recorder ───────────────────────────────────────────

def test_record_follow_through_trades_guards(monkeypatch):
    monkeypatch.setattr(settings, "enable_follow_through_trading", True)
    monkeypatch.setattr(settings, "ft_max_entries_per_day", 2)
    book = [{"ticker": "OPEN1", "status": "OPEN", "entry_date": "2026-08-01"}]
    saved = {}
    monkeypatch.setattr(tracker, "_load_trades", lambda: book)
    monkeypatch.setattr(tracker, "_save_trades", lambda t: saved.update(t=list(t)))
    monkeypatch.setattr(tracker, "_fetch_price", lambda t: 50.0)
    monkeypatch.setattr(tracker, "_session_of_iso", lambda x: "rth")
    ft_map = {
        "OPEN1": {"score": -0.9, "dir": 1.0, "selected": True,
                  "cohort_entry_date": "2026-08-20"},   # already open -> skipped
        "NEW1": {"score": -0.8, "dir": -1.0, "selected": True,
                 "cohort_entry_date": "2026-08-20"},
        "NEW2": {"score": -0.7, "dir": 1.0, "selected": True,
                 "cohort_entry_date": "2026-08-20"},
        "NEW3": {"score": -0.6, "dir": 1.0, "selected": True,
                 "cohort_entry_date": "2026-08-20"},    # over the daily cap
        "NOSEL": {"score": -0.9, "dir": 1.0, "selected": False,
                  "cohort_entry_date": "2026-08-20"},
    }
    n = tracker.record_follow_through_trades(ft_map, None, run_id="r1")
    assert n == 2
    new = [t for t in saved["t"] if t.get("entry_mechanism") == "follow_through"]
    assert {t["ticker"] for t in new} == {"NEW1", "NEW2"}
    t1 = next(t for t in new if t["ticker"] == "NEW1")
    assert t1["action"] == "SELL" and t1["direction"] == "BEARISH"
    assert t1["universe_source"] == "follow_through"
    assert t1["position_size_multiplier"] == pytest.approx(settings.ft_size_multiplier)
    assert t1["target_horizon"] == "1d" and t1["status"] == "OPEN"


def test_record_disabled_or_empty(monkeypatch):
    monkeypatch.setattr(settings, "enable_follow_through_trading", False)
    assert tracker.record_follow_through_trades(
        {"A": {"score": -1, "dir": 1, "selected": True}}, None) == 0
    monkeypatch.setattr(settings, "enable_follow_through_trading", True)
    assert tracker.record_follow_through_trades({}, None) == 0


# ── the monitor branch: ft trades never touch the swing machinery ───────────

def test_monitor_branch_closes_and_skips_llm(monkeypatch):
    monkeypatch.setattr(settings, "enable_llm_hold_review", True)
    trade = dict(_ft_trade(1), current_price=103.0)
    monkeypatch.setattr(tracker, "_load_trades", lambda: [trade])
    saved = {}
    monkeypatch.setattr(tracker, "_save_trades", lambda t: saved.update(t=list(t)))
    monkeypatch.setattr(tracker, "_ft_exit_reason", lambda t, m: "ft_horizon")
    # the LLM/consensus machinery must never be consulted for an ft trade —
    # poison it so any touch fails the test loudly.
    import src.analysis.exit_methods as em
    monkeypatch.setattr(em, "build_exit_scores",
                        lambda *a, **k: (_ for _ in ()).throw(AssertionError(
                            "swing exit machinery touched a follow-through trade")))
    closed = tracker.monitor_open_positions(signals_by_ticker={}, hold_reviews={})
    assert closed == 1
    assert trade["status"] == "CLOSED"
    assert trade["exit_reason"] == "ft_horizon"
    assert trade["return_pct"] != 0.0


# ── Gate 4 parity with the LLM funnel (2026-09-03) ──────────────────────────

def test_ft_gate4_is_the_funnel_gate4(monkeypatch):
    """`_gate4_ok` must delegate to `liquidity.is_liquid` with the TRADE floors,
    the tick's live price and the already-loaded frame — until 2026-09-03 it
    carried its own median-60 formula, so the two entry paths traded two
    different liquidity populations under one NBBO sizing rule. A missing
    snapshot price is a refusal, and any exception fails CLOSED."""
    import src.data.liquidity as liq
    monkeypatch.setattr(settings, "trade_min_price", 5.0)
    monkeypatch.setattr(settings, "trade_min_dollar_volume", 5e6)
    seen = []
    sentinel = object()

    def fake(ticker, budget, mp, mdv, price=None, df=None):
        seen.append((ticker, dict(budget), mp, mdv, price, df is sentinel))
        return ticker == "OK"

    monkeypatch.setattr(liq, "is_liquid", fake)
    assert ft._gate4_ok("OK", 10.0, sentinel) is True
    assert ft._gate4_ok("THIN", 10.0, sentinel) is False
    assert seen == [("OK", {"n": 0}, 5.0, 5e6, 10.0, True),
                    ("THIN", {"n": 0}, 5.0, 5e6, 10.0, True)]
    assert ft._gate4_ok("OK", None, sentinel) is False          # no snapshot price → refuse
    assert len(seen) == 2                                        # ...without consulting the gate

    def boom(*a, **k):
        raise RuntimeError("cache unreadable")

    monkeypatch.setattr(liq, "is_liquid", boom)
    assert ft._gate4_ok("OK", 10.0, sentinel) is False


# ── the cohort-anchor queries against the REAL `signals` schema (2026-09-23) ─
# Every test above replaces `_panel_history` with a fake, which is how its SQL
# went dark unnoticed: from 2026-09-07 it named 21 stacker-only features that
# are not `signals` columns, DuckDB refused it, and follow-through scored and
# traded nothing for 16 days while failing soft. These run the real queries on
# the isolated test DB, which carries the production schema (conftest).

def _insert_panel(rows_by_date):
    """`signals` rows through the production writer:
    {date: [(ticker, direction, news, ft_selected)]}."""
    from src.db import repo
    from src.db.schema import SIGNAL_METHOD_COLUMNS
    for d, rows in rows_by_date.items():
        out = []
        for tk, direction, news, sel in rows:
            scores = {m: 0.0 for m in SIGNAL_METHOD_COLUMNS}
            scores["news"] = news
            out.append({"ticker": tk, "type": "STOCK", "direction": direction,
                        "combined_score": 0.3, "price": 100.0, "scores": scores,
                        "ft_score": (-0.9 if sel else None), "ft_selected": sel})
        repo.insert_signals(f"run-{d}", f"{d}T15:00:00+00:00", d, out)


def test_panel_history_runs_on_the_real_signals_schema():
    """Names `signals` does not carry are DROPPED from the SELECT, never sent."""
    _insert_panel({"2026-09-21": [("AAA", "BULLISH", 0.4, None),
                                  ("BBB", "NEUTRAL", 0.1, None)],
                   "2026-09-22": [("AAA", "BEARISH", -0.2, None)]})
    methods = ["news", "tech", "ml_ohlcv", "tape_score",   # tape_score: a column since 09-25
               "cat_earnings", "cat_analyst"]              # the one-hots are not columns
    hist = ft._panel_history(5, methods)
    assert set(hist) == {"AAA"}                        # a NEUTRAL row anchors no cohort
    assert hist["AAA"]["2026-09-21"]["ds"] == 1.0
    assert hist["AAA"]["2026-09-22"]["ds"] == -1.0
    assert hist["AAA"]["2026-09-21"]["c_abs"] == pytest.approx(0.3)
    assert hist["AAA"]["2026-09-21"]["scores"]["news"] == pytest.approx(0.4)
    assert set(hist["AAA"]["2026-09-21"]["scores"]) == {"news", "tech", "ml_ohlcv", "tape_score"}


def test_entry_scores_come_from_the_artifact_not_the_stacker_constants():
    """The entry-day columns are the LOADED artifact's own `ex_<method>` names
    (what its training handed `method_horizon_days`) — never the state block,
    never today's stacker list — and every WEIGHTED one must be a `signals`
    column, or the served horizon clock (`ex_elapsed_ratio`) silently differs
    from the one the model was trained on."""
    from src.analysis.ml_exit_dataset import EXIT_FEATURE_COLUMNS, EXIT_STATE_FEATURES
    from src.db import repo
    from src.signals.aggregator import _BASE_WEIGHTS
    art = ["days_held", "ex_ret", "ex_news", "ex_tape_score", "ex_elapsed_ratio"]
    assert ft._artifact_methods(art) == ["news", "tape_score"]
    methods = ft._artifact_methods(EXIT_FEATURE_COLUMNS)
    assert not set(methods) & {f[3:] for f in EXIT_STATE_FEATURES}
    have = set(repo.fetch_df("SELECT column_name FROM information_schema.columns "
                             "WHERE table_name = 'signals'",
                             read_only=False)["column_name"].astype(str))
    missing_weighted = [m for m in methods if m not in have and _BASE_WEIGHTS.get(m)]
    assert missing_weighted == []


def test_episode_guard_reads_the_prior_panel_date_not_the_last_scored_one():
    """After an outage the last date follow-through SCORED is weeks old; its
    selections must not block today's candidates. A prior date follow-through
    did not score blocks nothing; one it did score blocks its selections."""
    today = date.today()
    old = (today - timedelta(days=19)).isoformat()
    prior = (today - timedelta(days=1)).isoformat()
    _insert_panel({old: [("OLD", "BULLISH", 0.1, 1.0)],
                   prior: [("AAA", "BULLISH", 0.1, None)]})
    assert ft._prev_selected_tickers() == set()
    _insert_panel({prior: [("AAA", "BULLISH", 0.1, None), ("NEW", "BEARISH", -0.1, 1.0)]})
    assert ft._prev_selected_tickers() == {"NEW"}
