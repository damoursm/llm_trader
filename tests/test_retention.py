"""Table retention — collapse simulated_trades intraday bloat + age-prune.

Verifies: the collapse keeps only the last row per (day, ticker, method) in the
OLD partition while leaving the recent RAW window untouched; that it is
behavior-NEUTRAL for the deduped analysis (compute_method_perf output unchanged
over the collapsed window); the age-prune; and the disabled/no-op paths.
"""

from datetime import date, timedelta

import pandas as pd
import pytest

from config.settings import settings
from src.db import repo
from src.db import retention


def _sim(run_id, gen, sig_date, ticker, method, score):
    return {"ticker": ticker, "method": method, "score": score,
            "direction": "BUY" if score > 0 else "SELL", "entry_price": 100.0}


def _insert(run_id, gen, sig_date, rows):
    repo.insert_simulated_trades(run_id, gen, sig_date,
                                 [_sim(run_id, gen, sig_date, *r) for r in rows])


def _count(where="1=1", params=None):
    return int(repo.fetch_df(f"SELECT COUNT(*) n FROM simulated_trades WHERE {where}",
                             params or []).iloc[0]["n"])


# ── collapse ──────────────────────────────────────────────────────────────────

def test_collapse_keeps_last_per_group_in_old_partition():
    old = (date.today() - timedelta(days=30)).isoformat()
    recent = (date.today() - timedelta(days=2)).isoformat()
    # OLD day: same (ticker, method) scored in THREE intraday runs → collapse to 1.
    _insert("r1", f"{old}T14:00:00+00:00", old, [("AAA", "news", 0.5)])
    _insert("r2", f"{old}T14:30:00+00:00", old, [("AAA", "news", 0.6)])
    _insert("r3", f"{old}T15:00:00+00:00", old, [("AAA", "news", 0.7)])   # latest → kept
    # RECENT day: two runs — must be LEFT RAW.
    _insert("r4", f"{recent}T14:00:00+00:00", recent, [("BBB", "tech", 0.3)])
    _insert("r5", f"{recent}T14:30:00+00:00", recent, [("BBB", "tech", 0.4)])
    assert _count() == 5

    removed = retention.collapse_simulated_trades(raw_days=14)
    assert removed == 2                                      # two old duplicates dropped
    assert _count(where="signal_date = ?", params=[old]) == 1
    kept = repo.fetch_df("SELECT score FROM simulated_trades WHERE signal_date = ?", [old])
    assert float(kept.iloc[0]["score"]) == pytest.approx(0.7)   # the LATEST-generated row
    assert _count(where="signal_date = ?", params=[recent]) == 2  # recent window untouched


def test_collapse_is_behavior_neutral_for_deduped_analysis(monkeypatch):
    # The analysis dedupes to last-per-(day,ticker,method); collapsing to exactly
    # that set must leave compute_method_perf's output identical.
    from src.analysis import simulated_trades as st
    d0, d1 = date(2026, 6, 1), date(2026, 6, 2)
    monkeypatch.setattr(st, "_daily_series", lambda tk: ([d0, d1], {d0: 100.0, d1: 110.0}))
    monkeypatch.setattr(st, "_intraday_series", lambda tk: [])
    old = (date.today() - timedelta(days=40)).isoformat()
    _insert("r1", f"{old}T14:00:00+00:00", old, [("A", "news", 0.2)])
    _insert("r2", f"{old}T15:00:00+00:00", old, [("A", "news", 0.9)])     # latest

    before = st.compute_method_perf(min_n=1, dedupe="last")
    retention.collapse_simulated_trades(raw_days=14)
    after = st.compute_method_perf(min_n=1, dedupe="last")
    # Same single deduped observation → identical views + win/return.
    b, a = before[before.method == "news"].iloc[0], after[after.method == "news"].iloc[0]
    assert int(b["views"]) == int(a["views"]) == 1
    assert b["win_1d"] == a["win_1d"]


def test_collapse_noop_when_nothing_old():
    recent = (date.today() - timedelta(days=2)).isoformat()
    _insert("r1", f"{recent}T14:00:00+00:00", recent, [("A", "news", 0.5)])
    assert retention.collapse_simulated_trades(raw_days=14) == 0
    assert _count() == 1


# ── age prune ─────────────────────────────────────────────────────────────────

def test_prune_beyond_deletes_old_only():
    ancient = (date.today() - timedelta(days=200)).isoformat()
    recent = (date.today() - timedelta(days=10)).isoformat()
    _insert("r1", f"{ancient}T14:00:00+00:00", ancient, [("A", "news", 0.5)])
    _insert("r2", f"{recent}T14:00:00+00:00", recent, [("B", "tech", 0.5)])
    removed = retention.prune_beyond("simulated_trades", "signal_date", keep_days=150)
    assert removed == 1
    assert _count() == 1 and _count(where="signal_date = ?", params=[recent]) == 1


# ── orchestration + disabled ─────────────────────────────────────────────────

def test_run_retention_disabled_is_noop(monkeypatch):
    monkeypatch.setattr(settings, "enable_sim_retention", False)
    old = (date.today() - timedelta(days=40)).isoformat()
    _insert("r1", f"{old}T14:00:00+00:00", old, [("A", "news", 0.5)])
    _insert("r2", f"{old}T15:00:00+00:00", old, [("A", "news", 0.6)])
    assert retention.run_retention() == {}
    assert _count() == 2                                     # untouched


def test_run_retention_reports_counts(monkeypatch):
    monkeypatch.setattr(settings, "enable_sim_retention", True)
    monkeypatch.setattr(settings, "sim_retention_raw_days", 14)
    monkeypatch.setattr(settings, "sim_retention_keep_days", 150)
    monkeypatch.setattr(settings, "exit_signals_keep_days", 150)
    old = (date.today() - timedelta(days=40)).isoformat()
    _insert("r1", f"{old}T14:00:00+00:00", old, [("A", "news", 0.5)])
    _insert("r2", f"{old}T15:00:00+00:00", old, [("A", "news", 0.6)])
    res = retention.run_retention()
    assert res["sim_collapsed"] == 1
    assert "sim_pruned" in res and "exit_signals_pruned" in res
