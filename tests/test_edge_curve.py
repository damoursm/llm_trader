"""Tests for the term-structure-of-edge synthesis (src/signals/edge_curve.py):
sign-aware IC weighting, cost-aware horizon selection, and the never-lengthen cap."""

import pytest

from src.signals import edge_curve as ec


def test_edge_is_sign_aware():
    """A positive-IC method counts in its score's direction; a NEGATIVE-IC method
    is flipped (its bullish score pushes edge DOWN)."""
    ic_matrix = {
        "tech": {"1d": (0.5, 1.0)},     # +IC, +ret
        "vwap": {"1d": (-0.5, -0.8)},   # −IC, −ret (a reversal/contrarian tell)
    }
    scores = {"tech": 0.8, "vwap": 0.6}
    curve = ec.compute_edge_curve(scores, ic_matrix, cost_hurdle_pct=0.0)
    # edge = (0.5·0.8 + (−0.5)·0.6) / (0.5+0.5) = (0.4 − 0.3)/1.0 = 0.10
    assert curve["1d"]["edge"] == pytest.approx(0.10)
    # exp_gross: tech 0.4·(+1.0)=0.40 ; vwap 0.3·(−(−0.8))=0.3·0.8=0.24 → 0.64/0.7
    assert curve["1d"]["exp_gross"] == pytest.approx(0.64 / 0.7, abs=1e-3)
    # a horizon with no IC cells abstains entirely
    assert curve["30m"]["edge"] == 0.0
    assert curve["30m"]["exp_gross"] == 0.0


def test_net_subtracts_cost_hurdle():
    ic_matrix = {"tech": {"1d": (0.5, 1.0)}}
    curve = ec.compute_edge_curve({"tech": 1.0}, ic_matrix, cost_hurdle_pct=0.4)
    assert curve["1d"]["exp_gross"] == pytest.approx(1.0)      # single method, ret 1.0
    assert curve["1d"]["net"] == pytest.approx(0.6)            # 1.0 − 0.4


def test_select_horizon_argmax_net_with_conviction_floor(monkeypatch):
    from config.settings import settings
    monkeypatch.setattr(settings, "horizon_min_conviction", 0.05)
    curve = {
        "30m": {"edge": 0.02, "exp_gross": 0.5, "net": 0.10},   # net>0 but |edge|<floor → not a candidate
        "1d":  {"edge": 0.30, "exp_gross": 0.6, "net": 0.20},   # candidate
        "1w":  {"edge": 0.40, "exp_gross": 0.9, "net": 0.50},   # candidate, best net
        "1m":  {"edge": 0.10, "exp_gross": 0.2, "net": -0.20},  # fails cost
    }
    sel = ec.select_horizon(curve)
    assert sel["target_horizon"] == "1w"
    assert sel["horizon_label"] == "POSITION"
    assert sel["direction"] == "BULLISH"
    assert sel["tradeable"] is True
    assert sel["conviction"] == pytest.approx(0.40)


def test_select_horizon_none_tradeable(monkeypatch):
    from config.settings import settings
    monkeypatch.setattr(settings, "horizon_min_conviction", 0.05)
    curve = {
        "30m": {"edge": -0.30, "exp_gross": 0.1, "net": -0.30},
        "1d":  {"edge": -0.20, "exp_gross": 0.2, "net": -0.20},
    }
    sel = ec.select_horizon(curve)
    assert sel["tradeable"] is False
    assert sel["target_horizon"] == "1d"          # best (least negative) net, still reported
    assert sel["direction"] == "BEARISH"


def test_cap_horizon_shortens_never_lengthens():
    # LLM bucket SHORT-TERM (rep "6h") shortens a mechanical "1w"
    assert ec.cap_horizon("1w", "SHORT-TERM") == "6h"
    # LLM bucket POSITION cannot lengthen a mechanical "6h"
    assert ec.cap_horizon("6h", "POSITION") == "6h"
    # SWING (rep "3d") confirms a "1d" (1d is shorter → keep mechanical)
    assert ec.cap_horizon("1d", "SWING") == "1d"
    # no mechanical view → unchanged
    assert ec.cap_horizon("", "SWING") == ""


def test_horizon_hours_known():
    assert ec.horizon_hours("6h") == 6.0
    assert ec.horizon_hours("1w") == 168.0
    assert ec.horizon_hours("nonsense") is None
