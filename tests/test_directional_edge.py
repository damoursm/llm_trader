"""Direction-aware, market-neutral shadow edge curve:
compute_directional_perf (per-side, SPY-relative skill) + compute_directional_edge_curve
(per-side skill weighting, shrink toward both-sides, sign-aware flip)."""

from datetime import date

import pandas as pd
import pytest

import src.analysis.simulated_trades as st
from src.signals import edge_curve as ec


def _patch(monkeypatch, series):
    monkeypatch.setattr(st, "_daily_series", lambda tk: series[tk])
    monkeypatch.setattr(st, "_intraday_series", lambda tk: [])   # daily-only test


def test_directional_perf_is_market_relative(monkeypatch):
    d1, d2 = date(2026, 6, 1), date(2026, 6, 2)
    series = {
        "TKR":  ([d1, d2], {d1: 100.0, d2: 105.0}),   # +5%
        "TKR2": ([d1, d2], {d1: 100.0, d2: 99.0}),    # −1%
        "SPY":  ([d1, d2], {d1: 400.0, d2: 408.0}),   # +2% (the market leg)
    }
    _patch(monkeypatch, series)
    sim = pd.DataFrame([
        {"generated_at": "t1", "signal_date": "2026-06-01", "ticker": "TKR",
         "method": "tech", "score": 0.8, "direction": "BUY"},
        {"generated_at": "t1", "signal_date": "2026-06-01", "ticker": "TKR2",
         "method": "tech", "score": -0.5, "direction": "SELL"},
    ])
    perf = st.compute_directional_perf(sim_df=sim, min_n=1, benchmark="SPY")

    bull = perf[(perf.method == "tech") & (perf.side == "bull")].iloc[0]
    # TKR +5% − SPY +2% = +3% market-relative; BUY → correct
    assert bull["yield_1d"] == pytest.approx(3.0)
    assert bull["hit_1d"] == pytest.approx(100.0)

    bear = perf[(perf.method == "tech") & (perf.side == "bear")].iloc[0]
    # TKR2 −1% − SPY +2% = −3%; SELL → signed = +3% (correctly shorted an underperformer)
    assert bear["yield_1d"] == pytest.approx(3.0)
    assert bear["hit_1d"] == pytest.approx(100.0)

    both = perf[(perf.method == "tech") & (perf.side == "both")].iloc[0]
    assert both["n_1d"] == 2
    assert both["yield_1d"] == pytest.approx(3.0)   # mean of +3, +3


def test_directional_edge_shrinks_side_toward_both():
    # bull skill 0.6 / yield 1.0 (n=100); both 0.4 / 0.5 — shrink with prior_n=30
    dmatrix = {"m1": {"1d": {"bull": (0.6, 1.0, 100), "both": (0.4, 0.5, 200)}}}
    curve = ec.compute_directional_edge_curve({"m1": 1.0}, dmatrix, cost_hurdle_pct=0.0, prior_n=30)
    yld = (100 * 1.0 + 30 * 0.5) / 130
    assert curve["1d"]["edge"] == pytest.approx(1.0)        # single +skill method → full conviction
    assert curve["1d"]["exp_gross"] == pytest.approx(round(yld, 4), abs=1e-3)


def test_directional_edge_flips_anti_predictive_side():
    # A BULLISH score on a method whose BULL calls LOSE to the market (skill<0) →
    # the contribution flips bearish, and the (flipped) market-relative yield is +.
    dmatrix = {"m2": {"1d": {"bull": (-0.4, -0.5, 100), "both": (-0.2, -0.3, 200)}}}
    curve = ec.compute_directional_edge_curve({"m2": 1.0}, dmatrix, cost_hurdle_pct=0.0, prior_n=0)
    assert curve["1d"]["edge"] == pytest.approx(-1.0)       # flipped: bullish score → bearish edge
    assert curve["1d"]["exp_gross"] == pytest.approx(0.5)   # sign-corrected: −(−0.5)


def test_directional_edge_picks_side_by_score_sign():
    # Same method, opposite skills per side. A negative (bearish) score must read
    # the BEAR cell, not the bull cell.
    dmatrix = {"m": {"1d": {"bull": (0.6, 1.0, 100), "bear": (-0.6, -1.0, 100),
                             "both": (0.0, 0.0, 200)}}}
    bull = ec.compute_directional_edge_curve({"m": 1.0}, dmatrix, cost_hurdle_pct=0.0, prior_n=0)
    bear = ec.compute_directional_edge_curve({"m": -1.0}, dmatrix, cost_hurdle_pct=0.0, prior_n=0)
    assert bull["1d"]["edge"] == pytest.approx(1.0)         # +bull skill → confirm the long
    # bearish score on a method whose bear calls LOSE to market (skill −0.6) → flip
    # to a bullish edge: (−0.6)·(−1)/0.6 = +1.0
    assert bear["1d"]["edge"] == pytest.approx(1.0)


def test_model_has_shadow_fields_defaulting():
    from src.models import TickerSignal
    s = TickerSignal(ticker="AAA", direction="BULLISH", confidence=0.8,
                     sentiment_score=0.0, technical_score=0.0, rationale="t")
    assert s.shadow_target_horizon == ""
    assert s.shadow_direction == ""
    assert s.shadow_horizon_net_edge_pct == 0.0
    assert s.target_horizon == ""        # live + shadow are independent fields
