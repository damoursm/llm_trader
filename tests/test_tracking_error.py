"""Tests for src.analysis.tracking_error (sim ledger vs IBKR fills)."""

import pytest

from src.analysis.tracking_error import compute_tracking_error


def _trade_with_fill():
    """A sim trade whose broker entry+exit actually filled, with a deliberate
    50 bp entry-price gap and a known return gap."""
    return {
        "trade_id": "t1", "ticker": "AAA", "action": "BUY", "status": "CLOSED",
        "entry_date": "2026-06-12", "entry_session": "rth",
        "entry_price": 100.0, "exit_price": 105.0, "return_pct": 5.0,
        "position_size_multiplier": 1.0,
        # broker view inputs
        "broker_fill_qty": 10, "broker_fill_price": 100.5, "broker_commission": 1.0,
        "broker_exit_fill_qty": 10, "broker_exit_fill_price": 104.5, "broker_exit_commission": 1.0,
    }


def test_matched_trade_metrics():
    rep = compute_tracking_error([_trade_with_fill()])
    assert rep["n_matched"] == 1
    r = rep["per_trade"][0]
    assert r["entry_bps"] == 50.0                       # (100.5-100)/100 * 1e4
    # broker return = (104.5-100.5)/100.5*100 - 2/(10*100.5)*100 ≈ 3.781
    assert r["broker_return"] == pytest.approx(3.78, abs=0.02)
    assert r["d_return"] == pytest.approx(5.0 - 3.78, abs=0.02)


def test_unfilled_trade_not_matched():
    t = {"trade_id": "t2", "ticker": "BBB", "action": "BUY", "status": "OPEN",
         "entry_price": 50.0, "return_pct": 1.0, "broker_fill_qty": 0}
    rep = compute_tracking_error([t])
    assert rep["n_matched"] == 0
    assert "needs filled" in rep["verdict"]


def test_one_sided_gap_flagged():
    rep = compute_tracking_error([_trade_with_fill()])
    # sim is ~1.2% optimistic vs the real fill → bias warning fires.
    assert "⚠" in rep["verdict"]
    assert rep["overall"]["mean_entry_bps"] == 50.0


def test_by_session_breakdown():
    rep = compute_tracking_error([_trade_with_fill()])
    sessions = {s["session"] for s in rep["by_session"]}
    assert sessions == {"rth"}
