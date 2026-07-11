"""Classic cross-sectional anomalies (2026-07-08) — hi52 / mom_12_1 / st_reversal.

Panel-first contract: the three methods are scored on every ticker, flow into
the signals panel + trade attribution (`_ALL_METHODS` / `_method_scores_from_signal`
/ schema columns — the drift test guards the sync), but carry ZERO weight in
`combined_score`, are excluded from coherence / sources_agreeing, and sit in the
exit-consensus skip set — so entries, confidence, and live exits are
bit-identical with the flags on or off until the IC panel earns them a weight.
"""

import numpy as np
import pandas as pd
import pytest

from config.settings import settings
from src.signals.classic_anomalies import (compute_high_52w_score,
                                           compute_momentum_12_1_score,
                                           compute_st_reversal_score)


def _frame(closes, volumes=None, highs=None):
    n = len(closes)
    idx = pd.bdate_range(end="2026-07-07", periods=n)
    closes = pd.Series(closes, dtype=float)
    return pd.DataFrame({
        "Open":   closes.values,
        "High":   (pd.Series(highs, dtype=float) if highs is not None else closes * 1.01).values,
        "Low":    (closes * 0.99).values,
        "Close":  closes.values,
        "Volume": (np.full(n, 1_000_000.0) if volumes is None else np.asarray(volumes, dtype=float)),
    }, index=idx)


# ── hi52: 52-week-high proximity ─────────────────────────────────────────────

def test_hi52_at_the_high_is_max_bullish():
    closes = np.linspace(50, 100, 260)                    # steady climb, ends at the high
    score, ratio_pct = compute_high_52w_score("UP", df=_frame(closes))
    assert ratio_pct >= 98.0
    assert score >= 0.85                                  # near the +1 end of the map


def test_hi52_far_below_high_is_bearish():
    closes = np.concatenate([np.linspace(60, 100, 130),   # peaked at 100...
                             np.linspace(100, 55, 130)])  # ...now at 55% of the high
    score, ratio_pct = compute_high_52w_score("DN", df=_frame(closes))
    assert ratio_pct < 60.0
    assert score == -1.0                                  # clipped at the −1 end


def test_hi52_insufficient_history_is_no_view():
    closes = np.linspace(10, 20, 120)                     # < 200 bars — truncated year
    assert compute_high_52w_score("IPO", df=_frame(closes)) == (0.0, 0.0)


# ── mom_12_1: skip-month momentum ────────────────────────────────────────────

def test_mom_12_1_skips_the_most_recent_month():
    # Strong 11-month rise, then a −30% crash in the FINAL month: 12-1 momentum
    # must stay POSITIVE (the crash belongs to the reversal regime, not to it).
    rise  = np.linspace(50, 150, 279)
    crash = np.linspace(150, 105, 21)
    score, ret_pct = compute_momentum_12_1_score("SKIP", df=_frame(np.concatenate([rise, crash])))
    assert ret_pct > 50.0                                 # ≈ +67% measured t−252 → t−21
    assert score > 0.5


def test_mom_12_1_downtrend_is_bearish():
    closes = np.concatenate([np.linspace(150, 60, 279), np.linspace(60, 62, 21)])
    score, ret_pct = compute_momentum_12_1_score("DN", df=_frame(closes))
    assert ret_pct < -40.0
    assert score < -0.5


def test_mom_12_1_requires_a_full_year():
    closes = np.linspace(50, 100, 200)                    # < 252 bars
    assert compute_momentum_12_1_score("NEW", df=_frame(closes)) == (0.0, 0.0)


# ── st_reversal: 1-week reversal, liquidity-gated ────────────────────────────

def _reversal_frame(last_week_ret, price=100.0, volume=2_000_000.0):
    # A year of gentle noise, then a decisive final week.
    rng = np.random.default_rng(7)
    base = price * np.cumprod(1 + rng.normal(0, 0.01, 300))
    week = base[-6] * np.linspace(1, 1 + last_week_ret, 6)[1:]
    closes = np.concatenate([base[:-5], week])
    return _frame(closes, volumes=np.full(len(closes), volume))


def test_st_reversal_flips_the_sign_of_a_big_up_week():
    score, ret_pct = compute_st_reversal_score("HOT", df=_reversal_frame(+0.15))
    assert ret_pct > 10.0
    assert score < -0.5                                   # reversal baked into the sign


def test_st_reversal_bounce_call_after_a_big_down_week():
    score, ret_pct = compute_st_reversal_score("COLD", df=_reversal_frame(-0.15))
    assert ret_pct < -10.0
    assert score > 0.5


def test_st_reversal_below_liquidity_floor_is_no_view(monkeypatch):
    monkeypatch.setattr(settings, "st_reversal_min_dollar_volume", 50_000_000)
    df = _reversal_frame(+0.15, price=10.0, volume=1_000.0)   # ~$10k/day — bid-ask bounce land
    score, ret_pct = compute_st_reversal_score("THIN", df=df)
    assert score == 0.0
    assert ret_pct > 10.0                                 # the raw move is still reported


def test_st_reversal_missing_volume_fails_closed():
    df = _reversal_frame(+0.15).drop(columns=["Volume"])
    assert compute_st_reversal_score("NOVOL", df=df)[0] == 0.0


def test_st_reversal_quiet_week_is_no_view():
    score, _ = compute_st_reversal_score("FLAT", df=_reversal_frame(+0.0005))
    assert score == 0.0                                   # inside the z deadband


def test_zero_price_rows_never_emit_nan():
    # Regression (SOLS, 2026-07-08 offline run): a zero close inside the window
    # made z NaN, which slipped past the `std < 1e-8` guard (NaN < x is False)
    # and persisted a NaN score to the panel. All three scorers must stay finite.
    closes = np.linspace(50, 100, 300)
    closes[-6] = 0.0                                      # garbage row right at the anchor
    df = _frame(closes)
    for fn in (compute_st_reversal_score, compute_momentum_12_1_score, compute_high_52w_score):
        score, ctx = fn("BADPX", df=df)
        assert np.isfinite(score) and np.isfinite(ctx)


# ── Zero-impact invariant + panel plumbing ───────────────────────────────────

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
    "enable_ttm_squeeze", "enable_iv_term_structure", "enable_anchored_vwap",
    "enable_residual_momentum", "enable_volume_profile",
]


def _minimal_build(monkeypatch):
    import src.signals.aggregator as agg
    for flag in _OFF:
        monkeypatch.setattr(settings, flag, False)
    monkeypatch.setattr(settings, "enable_news_sentiment", True)
    monkeypatch.setattr(agg, "analyse_sentiment",
                        lambda t, articles, force_engine=None: (0.4, f"news {t}"))
    return agg


def test_zero_impact_on_combine_confidence_and_agreement(monkeypatch):
    """The load-bearing panel-first guarantee: strong anomaly scores must leave
    combined_score / confidence / sources_agreeing bit-identical, while the
    scores themselves land on the signal for the panel."""
    agg = _minimal_build(monkeypatch)
    monkeypatch.setattr(agg, "compute_high_52w_score", lambda t, df=None: (0.9, 99.0))
    monkeypatch.setattr(agg, "compute_momentum_12_1_score", lambda t, df=None: (-0.8, -35.0))
    monkeypatch.setattr(agg, "compute_st_reversal_score", lambda t, df=None: (-0.7, 12.0))

    for flag in ("enable_high_52w", "enable_momentum_12_1", "enable_st_reversal"):
        monkeypatch.setattr(settings, flag, False)
    off = agg.build_signals(["AAPL"], [])[0]

    for flag in ("enable_high_52w", "enable_momentum_12_1", "enable_st_reversal"):
        monkeypatch.setattr(settings, flag, True)
    on = agg.build_signals(["AAPL"], [])[0]

    assert on.combined_score == off.combined_score
    assert on.confidence == off.confidence
    assert on.sources_agreeing == off.sources_agreeing
    assert on.direction == off.direction
    # ...and the panel payload is populated only on the ON run.
    assert (on.high_52w_score, on.momentum_12_1_score, on.st_reversal_score) == (0.9, -0.8, -0.7)
    assert (off.high_52w_score, off.momentum_12_1_score, off.st_reversal_score) == (0.0, 0.0, 0.0)


def test_methods_reach_attribution_bridge():
    from src.performance.tracker import _ALL_METHODS, _method_scores_from_signal
    from src.models import TickerSignal
    for m in ("hi52", "mom_12_1", "st_reversal"):
        assert m in _ALL_METHODS
    sig = TickerSignal(ticker="AAPL", direction="BULLISH", confidence=0.8,
                       sentiment_score=0.0, technical_score=0.0, rationale="test")
    sig.high_52w_score = 0.9
    sig.momentum_12_1_score = -0.8
    sig.st_reversal_score = -0.7
    scores = _method_scores_from_signal("AAPL", "BULLISH", {"AAPL": sig})
    assert scores["hi52"] == 0.9
    assert scores["mom_12_1"] == -0.8
    assert scores["st_reversal"] == -0.7


def test_excluded_from_exit_consensus():
    """Unvalidated methods must not move the mechanical_exit trigger or the
    confidence-loss floor nudge: a screaming st_reversal exit view leaves the
    consensus exactly where the validated methods put it."""
    from src.analysis.exit_conviction import exit_method_consensus
    base = {"money_flow": 0.2, "max_pain": -0.1}
    with_anomalies = dict(base, hi52=-1.0, mom_12_1=-1.0, st_reversal=-1.0)
    assert exit_method_consensus(with_anomalies) == exit_method_consensus(base)


def test_labels_and_categories_registered():
    from src.performance.tracker import METHOD_CATEGORIES, METHOD_LABELS
    for m in ("hi52", "mom_12_1", "st_reversal"):
        assert m in METHOD_LABELS
        assert m in METHOD_CATEGORIES["Technical"]
