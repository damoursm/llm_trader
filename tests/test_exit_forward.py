"""Post-exit forward returns (analysis/exit_forward.py) — held-longer counterfactual.

Anchored at the ACTUAL exit fill, oriented by position side, forward closes from
the OHLCV cache; exits without forward bars are pending, never guessed. Plus the
cache-warm extension that keeps recently-exited tickers accruing forward bars.
"""

from datetime import date, timedelta

import pandas as pd
import pytest

import src.analysis.exit_forward as xf


def _series(start: str, closes, include=None):
    """{date: close} over consecutive business days starting at ``start``."""
    days = pd.bdate_range(start=start, periods=len(closes))
    out = {d.date(): float(c) for d, c in zip(days, closes)}
    if include:
        out.update({date.fromisoformat(k): float(v) for k, v in include.items()})
    return out


def _trade(**kw):
    base = {"status": "CLOSED", "ticker": "AAPL", "action": "BUY",
            "exit_price": 100.0, "exit_date": "2026-06-10",
            "exit_reason": "llm_signal_flipped", "return_pct": 2.5}
    base.update(kw)
    return base


def _patch_series(monkeypatch, by_ticker):
    monkeypatch.setattr(xf, "_close_series", lambda tk: by_ticker.get(tk, {}))


# ── orientation + anchoring ──────────────────────────────────────────────────

def test_long_forward_returns_from_exit_fill(monkeypatch):
    # Exit Wed 2026-06-10 at 100; the NEXT sessions close 101, 102, … 112.
    # The exit day's own close (100.5) must NOT be the +1d bar (bisect_right).
    _patch_series(monkeypatch, {"AAPL": _series("2026-06-11", range(101, 113),
                                                include={"2026-06-10": 100.5})})
    rep = xf.compute_exit_forward([_trade()])
    r = rep["per_trade"][0]
    assert (r["fwd_1d"], r["fwd_3d"], r["fwd_5d"], r["fwd_10d"]) == (1.0, 3.0, 5.0, 10.0)
    assert rep["overall"]["mean_5d"] == 5.0
    assert rep["overall"]["pct_pos_5d"] == 100.0


def test_short_orientation_flips_sign(monkeypatch):
    # A short exited at 100: price FALLING further = positive forward (the short
    # would have kept earning); price rising = negative (good exit).
    _patch_series(monkeypatch, {"DN": _series("2026-06-11", [98, 96, 94, 92, 90]),
                                "UP": _series("2026-06-11", [102, 104, 106, 108, 110])})
    rep = xf.compute_exit_forward([
        _trade(ticker="DN", action="SELL"),
        _trade(ticker="UP", action="SELL"),
    ], horizons=(1, 3, 5))
    by = {r["ticker"]: r for r in rep["per_trade"]}
    assert by["DN"]["fwd_1d"] == pytest.approx(2.0)
    assert by["DN"]["fwd_5d"] == pytest.approx(10.0)
    assert by["UP"]["fwd_1d"] == pytest.approx(-2.0)
    assert by["UP"]["fwd_5d"] == pytest.approx(-10.0)


def test_weekend_exit_anchors_to_next_session(monkeypatch):
    # Exited Saturday (overnight/weekend stamp): +1d = Monday's close.
    _patch_series(monkeypatch, {"AAPL": _series("2026-06-15", [103, 104, 105])})
    rep = xf.compute_exit_forward([_trade(exit_date="2026-06-13")], horizons=(1, 3))
    r = rep["per_trade"][0]
    assert r["fwd_1d"] == pytest.approx(3.0)
    assert r["fwd_3d"] == pytest.approx(5.0)


# ── missing data stays honest ────────────────────────────────────────────────

def test_no_forward_bars_is_pending_not_guessed(monkeypatch):
    _patch_series(monkeypatch, {"AAPL": {}})
    rep = xf.compute_exit_forward([_trade()])
    assert rep["n"] == 0
    assert rep["n_pending"] == 1
    assert rep["per_trade"] == []


def test_partial_forward_window_counts_per_horizon(monkeypatch):
    # Only 2 sessions after exit: fwd_1d exists, fwd_5d/10d are None (not 0).
    _patch_series(monkeypatch, {"AAPL": _series("2026-06-11", [101, 102])})
    rep = xf.compute_exit_forward([_trade()])
    r = rep["per_trade"][0]
    assert r["fwd_1d"] == 1.0
    assert r["fwd_5d"] is None and r["fwd_10d"] is None
    assert rep["overall"]["n_1d"] == 1
    assert rep["overall"]["n_5d"] == 0
    assert rep["overall"]["mean_5d"] is None


def test_open_and_unanchorable_trades_skipped(monkeypatch):
    _patch_series(monkeypatch, {"AAPL": _series("2026-06-11", range(101, 113))})
    rep = xf.compute_exit_forward([
        _trade(status="OPEN"),
        _trade(exit_price=0.0),
        _trade(exit_date=""),
        _trade(action="", direction=""),          # unorientable
        _trade(),                                  # the one good row
    ])
    assert rep["n"] == 1


# ── aggregation by exit rule ─────────────────────────────────────────────────

def test_reason_grouping_and_pct_pos(monkeypatch):
    _patch_series(monkeypatch, {
        "W1": _series("2026-06-11", [102] * 5),    # +2% forward — exited too early
        "W2": _series("2026-06-11", [104] * 5),    # +4%
        "W3": _series("2026-06-11", [106] * 5),    # +6%
        "L1": _series("2026-06-11", [95] * 5),     # −5% — good exit
    })
    rep = xf.compute_exit_forward([
        _trade(ticker="W1", exit_reason="llm_confidence_loss"),
        _trade(ticker="W2", exit_reason="llm_confidence_loss"),
        _trade(ticker="W3", exit_reason="llm_confidence_loss"),
        _trade(ticker="L1", exit_reason="trailing_stop"),
    ], horizons=(1, 5))
    assert [r["exit_reason"] for r in rep["by_reason"]] == ["llm_confidence_loss", "trailing_stop"]
    conf = rep["by_reason"][0]
    assert conf["trades"] == 3
    assert conf["mean_5d"] == pytest.approx(4.0)
    assert conf["pct_pos_5d"] == 100.0
    assert rep["by_reason"][1]["mean_5d"] == pytest.approx(-5.0)
    assert rep["overall"]["pct_pos_5d"] == 75.0
    # Overall mean +1.75% @5d with the worst rule at n=3 → the verdict names it.
    assert "llm_confidence_loss" in rep["verdict"]


# ── cache warm keeps exited tickers accruing forward bars ────────────────────

def test_cache_warm_includes_recently_closed_tickers(monkeypatch):
    from src.data.cache_warm import _panel_tickers
    from src.db import repo

    monkeypatch.setattr(repo, "fetch_df", lambda *a, **k: pd.DataFrame())
    recent = (date.today() - timedelta(days=5)).isoformat()
    monkeypatch.setattr(repo, "load_trades", lambda: [
        {"status": "OPEN", "ticker": "OPN"},
        {"status": "CLOSED", "ticker": "NEW", "exit_date": recent},
        {"status": "CLOSED", "ticker": "OLD", "exit_date": "2020-01-01"},
    ])
    out = _panel_tickers(days=120)
    assert "OPN" in out and "NEW" in out
    assert "OLD" not in out
