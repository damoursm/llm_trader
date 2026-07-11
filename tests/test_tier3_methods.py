"""Tier-3 panel-first methods (2026-07-08) — resid_mom / vol_profile.

Same contract as the earlier panel-first batches: scored on every ticker,
panel + trade attribution wired (drift test guards the schema sync), zero
weight in the combine, excluded from coherence / sources_agreeing / the exit
consensus.
"""

import numpy as np
import pandas as pd
import pytest

from config.settings import settings
from src.signals.residual_momentum import compute_residual_momentum_score
from src.signals.volume_profile import compute_volume_profile_score


def _frame(closes, volumes=None, highs=None, lows=None, end="2026-07-07"):
    n = len(closes)
    idx = pd.bdate_range(end=end, periods=n)
    c = pd.Series(closes, dtype=float)
    return pd.DataFrame({
        "Open":   c.values,
        "High":   (pd.Series(highs, dtype=float) if highs is not None else c * 1.01).values,
        "Low":    (pd.Series(lows, dtype=float) if lows is not None else c * 0.99).values,
        "Close":  c.values,
        "Volume": (np.full(n, 1_000_000.0) if volumes is None else np.asarray(volumes, dtype=float)),
    }, index=idx)


# ── Residual momentum ─────────────────────────────────────────────────────────

def _mkt(n=300, daily=0.0008, sigma=0.008, seed=7):
    # A market with REAL return variance — a constant-drift series is degenerate
    # (zero variance ⇒ beta undefined) and correctly fails closed in the scorer.
    rng = np.random.default_rng(seed)
    return _frame(100 * np.cumprod(1 + rng.normal(daily, sigma, n)))


def test_resid_mom_pure_beta_ride_scores_flat():
    # A stock that is 2× the market every day (plus a whiff of idio noise so
    # residual vol exists) — raw momentum would love the ride; residual
    # momentum must strip the beta leg and see ~nothing left.
    n = 300
    mkt = _mkt(n)
    mr = mkt["Close"].pct_change().fillna(0.0008).to_numpy()
    rng = np.random.default_rng(13)                # a zero-mean idio draw (Σnoise ≈ 0)
    stock = _frame(50 * np.cumprod(1 + 2 * mr + rng.normal(0.0, 0.001, n)))
    score, resid_pct, beta = compute_residual_momentum_score("BETA2", df=stock, mkt_df=mkt)
    assert beta == pytest.approx(2.0, abs=0.1)
    assert abs(resid_pct) < 2.0                    # vs a ~+60% raw 12-1 ride
    assert abs(score) < 0.2                        # essentially no residual view


def test_resid_mom_true_alpha_scores_positive():
    # Market flat; the stock grinds up on its own — pure idiosyncratic drift
    # (with real daily vol, so the residual z-score is well-defined).
    n = 300
    rng = np.random.default_rng(5)
    mkt = _frame(100 * np.cumprod(1 + rng.normal(0, 0.002, n)))
    stock = _frame(50 * np.cumprod(1 + rng.normal(0.002, 0.004, n)))
    score, resid_pct, _ = compute_residual_momentum_score("ALPHA", df=stock, mkt_df=mkt)
    assert resid_pct > 30.0
    assert score > 0.5


def test_resid_mom_negative_alpha_scores_negative():
    n = 300
    rng = np.random.default_rng(6)
    mkt = _frame(100 * np.cumprod(1 + rng.normal(0.0005, 0.002, n)))
    stock = _frame(200 * np.cumprod(1 + rng.normal(-0.002, 0.004, n)))
    score, resid_pct, _ = compute_residual_momentum_score("DECAY", df=stock, mkt_df=mkt)
    assert resid_pct < -20.0
    assert score < -0.5


def test_resid_mom_benchmark_and_short_history_are_no_view():
    assert compute_residual_momentum_score("SPY", df=_mkt(), mkt_df=_mkt()) == (0.0, 0.0, 1.0)
    short = _frame(np.linspace(50, 60, 120))
    assert compute_residual_momentum_score("NEW", df=short, mkt_df=_mkt())[0] == 0.0


def test_resid_mom_misaligned_dates_fail_soft():
    # Stock and market histories that barely overlap → no view, not a crash.
    stock = _frame(np.linspace(50, 60, 260), end="2024-01-05")
    score, _, _ = compute_residual_momentum_score("OLD", df=stock, mkt_df=_mkt())
    assert score == 0.0


# ── Volume profile ────────────────────────────────────────────────────────────

def _profile_frame(last_close, node_price=100.0, n=80):
    """Heavy volume node at ``node_price`` for most of the window, then the
    price walks to ``last_close`` on thin volume — POC/value stay at the node."""
    n_node = n - 10
    node = node_price * (1 + 0.004 * np.sin(np.arange(n_node)))   # tight rotation at the node
    walk = np.linspace(node_price, last_close, 10)
    closes = np.concatenate([node, walk])
    volumes = np.concatenate([np.full(n_node, 5_000_000.0), np.full(10, 200_000.0)])
    return _frame(closes, volumes=volumes)


def test_vol_profile_acceptance_above_value_is_bullish():
    score, label, poc_dist = compute_volume_profile_score("UP", df=_profile_frame(118.0))
    assert label == "ABOVE_VALUE"
    assert score > 0.3
    assert poc_dist > 10.0


def test_vol_profile_acceptance_below_value_is_bearish():
    score, label, poc_dist = compute_volume_profile_score("DN", df=_profile_frame(84.0))
    assert label == "BELOW_VALUE"
    assert score < -0.3
    assert poc_dist < -10.0


def test_vol_profile_in_value_gravitates_to_poc_with_small_score():
    score, label, _ = compute_volume_profile_score("BAL", df=_profile_frame(100.4))
    assert label in ("IN_VALUE",)
    assert abs(score) <= 0.35 + 1e-9               # gravity leg is capped


def test_vol_profile_missing_or_zero_volume_fails_closed():
    df = _profile_frame(118.0).drop(columns=["Volume"])
    assert compute_volume_profile_score("NOVOL", df=df) == (0.0, "NO_DATA", 0.0)
    zero = _frame(np.linspace(90, 110, 80), volumes=np.zeros(80))
    assert compute_volume_profile_score("ZV", df=zero)[1] == "NO_DATA"


def test_tier3_zero_price_rows_never_emit_nan():
    closes = np.linspace(50, 100, 300)
    closes[-6] = 0.0
    df = _frame(closes)
    s1, p1, b1 = compute_residual_momentum_score("BADPX", df=df, mkt_df=_mkt())
    s2, _, p2 = compute_volume_profile_score("BADPX", df=df)
    assert all(np.isfinite(x) for x in (s1, p1, b1, s2, p2))


# ── Zero-impact invariant + plumbing ─────────────────────────────────────────

_OFF = [
    "enable_news_sentiment", "enable_sentiment_velocity", "enable_technical_analysis",
    "enable_options_flow", "enable_sec_filings", "enable_put_call", "enable_gex",
    "enable_vwap", "enable_pattern_recognition", "enable_price_momentum",
    "enable_sector_relative_momentum", "enable_market_relative_momentum",
    "enable_money_flow", "enable_trend_strength", "enable_pead", "enable_iv_rank",
    "enable_iv_expr", "enable_cointegration", "enable_cross_sectional",
    "enable_adaptive_weights", "enable_market_mode_switching", "enable_catalyst_timing",
    "enable_multi_timeframe_signals", "enable_extended_gap", "enable_massive_tech",
    "enable_trend_predictability_methods",
    "enable_high_52w", "enable_momentum_12_1", "enable_st_reversal",
    "enable_ttm_squeeze", "enable_iv_term_structure", "enable_anchored_vwap",
]

_T3_FLAGS = ("enable_residual_momentum", "enable_volume_profile")


def test_zero_impact_on_combine_confidence_and_agreement(monkeypatch):
    import src.signals.aggregator as agg
    for flag in _OFF:
        monkeypatch.setattr(settings, flag, False)
    monkeypatch.setattr(settings, "enable_news_sentiment", True)
    monkeypatch.setattr(agg, "analyse_sentiment",
                        lambda t, articles, force_engine=None: (0.4, f"news {t}"))
    monkeypatch.setattr(agg, "compute_residual_momentum_score",
                        lambda t, df=None, mkt_df=None: (0.9, 42.0, 1.3))
    monkeypatch.setattr(agg, "compute_volume_profile_score",
                        lambda t, df=None: (-0.8, "BELOW_VALUE", -12.0))

    for flag in _T3_FLAGS:
        monkeypatch.setattr(settings, flag, False)
    off = agg.build_signals(["AAPL"], [])[0]

    for flag in _T3_FLAGS:
        monkeypatch.setattr(settings, flag, True)
    on = agg.build_signals(["AAPL"], [])[0]

    assert on.combined_score == off.combined_score
    assert on.confidence == off.confidence
    assert on.sources_agreeing == off.sources_agreeing
    assert on.direction == off.direction
    assert (on.resid_mom_score, on.vol_profile_score) == (0.9, -0.8)
    assert (on.resid_mom_beta, on.vol_profile_label) == (1.3, "BELOW_VALUE")
    assert (off.resid_mom_score, off.vol_profile_score) == (0.0, 0.0)


def test_methods_reach_attribution_bridge():
    from src.performance.tracker import _ALL_METHODS, _method_scores_from_signal
    from src.models import TickerSignal
    for m in ("resid_mom", "vol_profile"):
        assert m in _ALL_METHODS
    sig = TickerSignal(ticker="AAPL", direction="BULLISH", confidence=0.8,
                       sentiment_score=0.0, technical_score=0.0, rationale="test")
    sig.resid_mom_score = 0.9
    sig.vol_profile_score = -0.8
    scores = _method_scores_from_signal("AAPL", "BULLISH", {"AAPL": sig})
    assert (scores["resid_mom"], scores["vol_profile"]) == (0.9, -0.8)


def test_excluded_from_exit_consensus():
    from src.analysis.exit_conviction import exit_method_consensus
    base = {"money_flow": 0.2, "max_pain": -0.1}
    with_t3 = dict(base, resid_mom=-1.0, vol_profile=-1.0)
    assert exit_method_consensus(with_t3) == exit_method_consensus(base)


def test_labels_and_categories_registered():
    from src.performance.tracker import METHOD_CATEGORIES, METHOD_LABELS
    for m in ("resid_mom", "vol_profile"):
        assert m in METHOD_LABELS
        assert m in METHOD_CATEGORIES["Technical"]
