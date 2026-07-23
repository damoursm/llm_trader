"""Monte Carlo overfitting evaluation (2026-07-19, src/analysis/monte_carlo.py):
per-method luck-vs-skill (bootstrap + permutation null on the gross solo win
rate), the win-rate filter's selection-bias null, and the exit-timing-vs-random
MC. Deterministic given the seed; fail-soft on thin data."""

from datetime import date, timedelta

import numpy as np
import pytest

import src.analysis.monte_carlo as mc


# ── method luck-vs-skill ─────────────────────────────────────────────────────

def test_perfect_method_is_significant_coinflip_is_not():
    rng = np.random.default_rng(7)
    moves = list(rng.normal(0, 2.0, size=40))
    perfect = [abs(m) + 0.1 for m in moves]            # always right, never zero
    coin = list(rng.choice([-1, 1], size=40) * np.abs(moves))
    rows = mc.method_overfit_mc({"perfect": perfect, "coin": coin},
                                n_sims=1000, seed=1)
    by = {r["method"]: r for r in rows}
    assert by["perfect"]["p_luck"] < 0.05
    assert "SKILL" in by["perfect"]["verdict"]
    assert 0.05 < by["coin"]["p_luck"] < 0.95          # indistinguishable band
    assert "luck" in by["coin"]["verdict"]


def test_always_wrong_method_reads_worse_than_chance():
    rows = mc.method_overfit_mc({"bad": [-1.0] * 30}, n_sims=1000, seed=1)
    r = rows[0]
    assert r["win_rate"] == 0.0
    assert r["p_luck"] > 0.95
    assert r["verdict"] == "worse than chance"


def test_bootstrap_ci_brackets_observed():
    rets = [1.0] * 12 + [-1.0] * 8                     # 60% WR
    r = mc.method_overfit_mc({"m": rets}, n_sims=1000, seed=2)[0]
    assert r["wr_lo"] <= r["win_rate"] <= r["wr_hi"]
    assert r["ret_lo"] <= r["mean_ret"] <= r["ret_hi"]
    assert r["wr_lo"] < r["wr_hi"]                     # non-degenerate CI


def test_min_trades_floor_skips_thin_methods():
    rows = mc.method_overfit_mc({"thin": [1.0, -1.0]}, n_sims=200, seed=1, min_trades=5)
    assert rows == []


def test_determinism_same_seed_same_rows():
    rets = {"m": [0.5, -1.2, 2.0, -0.3, 1.1, -0.8, 0.9, 1.4, -2.0, 0.2]}
    a = mc.method_overfit_mc(rets, n_sims=500, seed=9)
    b = mc.method_overfit_mc(rets, n_sims=500, seed=9)
    assert a == b


# ── filter selection-bias null ───────────────────────────────────────────────

def test_selection_null_centers_near_half():
    # 10 coin-flip methods at n=100: chance keeps ~half (WR ≥ 50% has p ≈ 0.54
    # at even n due to the ≥ boundary).
    counts = {f"m{i}": 100 for i in range(10)}
    out = mc.filter_selection_mc(counts, kept_actual=5, n_sims=2000, seed=3)
    assert out["n_judgeable"] == 10
    assert 4.0 <= out["kept_null_mean"] <= 6.5
    assert "chance" in out["verdict"] or "provisional" in out["verdict"]


def test_selection_null_flags_excess_keeps():
    # Keeping ALL 10 judgeable methods is far beyond the chance distribution.
    counts = {f"m{i}": 100 for i in range(10)}
    out = mc.filter_selection_mc(counts, kept_actual=10, n_sims=2000, seed=3)
    assert out["p_ge_actual"] < 0.05
    assert "LARGER than chance" in out["verdict"]


def test_selection_null_empty_when_nothing_judgeable():
    out = mc.filter_selection_mc({"m": 3}, kept_actual=0, min_trades=10, n_sims=100, seed=1)
    assert out["n_judgeable"] == 0


# ── exit-timing MC ───────────────────────────────────────────────────────────

def _mk_series(prices, start=date(2026, 1, 5)):
    """Business-day close series {date: price}."""
    out, d, i = {}, start, 0
    while i < len(prices):
        if d.weekday() < 5:
            out[d] = float(prices[i])
            i += 1
        d += timedelta(days=1)
    return out


def _trade(ticker, entry_d, exit_d, action="BUY", entry_price=100.0, reason="rule"):
    return {"status": "CLOSED", "ticker": ticker, "action": action,
            "entry_price": entry_price, "entry_date": entry_d.isoformat(),
            "exit_date": exit_d.isoformat(), "exit_reason": reason}


def test_exit_at_peak_beats_random(monkeypatch):
    # Price path rises to a peak at session 5 then collapses; the rule exits AT
    # the peak → its mean beats ~every random draw → percentile ~100.
    prices = [100, 102, 104, 106, 108, 110, 90, 80, 70, 60, 50, 45, 40, 35, 30, 28, 26, 24, 22, 20]
    series = _mk_series(prices)
    dates = sorted(series.keys())
    monkeypatch.setattr(mc, "_close_series", lambda t: series)
    trades = [_trade("X", dates[0], dates[5], reason="peak_rule")]
    rep = mc.exit_timing_mc(trades, n_sims=500, seed=4)
    row = next(r for r in rep["rows"] if r["reason"] == "peak_rule")
    assert row["percentile"] >= 95.0
    assert "BETTER" in row["verdict"]


def test_exit_at_trough_loses_to_random(monkeypatch):
    # Price rises steadily; the rule exits on session 1 (the worst feasible exit
    # in a monotone up-path is the earliest) → random beats it.
    prices = [100 + 2 * i for i in range(20)]
    series = _mk_series(prices)
    dates = sorted(series.keys())
    monkeypatch.setattr(mc, "_close_series", lambda t: series)
    trades = [_trade("X", dates[0], dates[1], reason="early_rule")]
    rep = mc.exit_timing_mc(trades, n_sims=500, seed=4)
    row = next(r for r in rep["rows"] if r["reason"] == "early_rule")
    assert row["percentile"] <= 5.0
    assert "BEATEN" in row["verdict"]


def test_exit_mc_short_orientation(monkeypatch):
    # SHORT into a monotone falling tape, exiting at the LAST session — the LOW
    # of the feasible window → best feasible exit → high percentile (orientation
    # sign applied: a short profits as price falls).
    prices = [100 - 2 * i for i in range(15)]
    series = _mk_series(prices)
    dates = sorted(series.keys())
    monkeypatch.setattr(mc, "_close_series", lambda t: series)
    trades = [_trade("X", dates[0], dates[-1], action="SELL", reason="short_rule")]
    rep = mc.exit_timing_mc(trades, n_sims=500, seed=4)
    row = next(r for r in rep["rows"] if r["reason"] == "short_rule")
    assert row["actual_mean"] > 0                       # short profited
    assert row["percentile"] >= 90.0


def test_exit_mc_skips_unanchorable(monkeypatch):
    monkeypatch.setattr(mc, "_close_series", lambda t: {})   # no cache
    trades = [_trade("X", date(2026, 1, 5), date(2026, 1, 9))]
    rep = mc.exit_timing_mc(trades, n_sims=100, seed=1)
    assert rep["n"] == 0 and rep["n_skipped"] == 1
    assert rep["rows"] == []


def test_exit_mc_includes_all_exits_row(monkeypatch):
    prices = [100 + i for i in range(20)]
    series = _mk_series(prices)
    dates = sorted(series.keys())
    monkeypatch.setattr(mc, "_close_series", lambda t: series)
    trades = [_trade("X", dates[0], dates[4], reason="a"),
              _trade("Y", dates[0], dates[6], reason="b")]
    rep = mc.exit_timing_mc(trades, n_sims=200, seed=1)
    reasons = [r["reason"] for r in rep["rows"]]
    assert "a" in reasons and "b" in reasons and "ALL exits" in reasons
    assert reasons[-1] == "ALL exits"                   # summary row last


# ── ledger wrappers (empty DB fail-soft) ─────────────────────────────────────

def test_reports_fail_soft_on_empty_db():
    # conftest points settings.db_path at an empty throwaway DB.
    rep = mc.compute_method_overfit_report(n_sims=100)
    assert rep["rows"] == []
    ex = mc.compute_exit_timing_report(n_sims=100)
    assert ex["n"] == 0
