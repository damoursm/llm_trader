"""Tier-2 panel-first methods (2026-07-08) — squeeze / iv_term / avwap.

Same contract as the classic anomalies: scored on every ticker, panel + trade
attribution wired (drift test guards the schema sync), zero weight in the
combine, excluded from coherence / sources_agreeing / the exit consensus.
"""

from datetime import date, timedelta
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from config.settings import settings
from src.signals.anchored_vwap import compute_anchored_vwap_score
from src.signals.iv_term_structure import compute_iv_term_score
from src.signals.ttm_squeeze import compute_ttm_squeeze_score


def _frame(closes, volumes=None, highs=None, lows=None):
    n = len(closes)
    idx = pd.bdate_range(end="2026-07-07", periods=n)
    c = pd.Series(closes, dtype=float)
    return pd.DataFrame({
        "Open":   c.values,
        "High":   (pd.Series(highs, dtype=float) if highs is not None else c * 1.01).values,
        "Low":    (pd.Series(lows, dtype=float) if lows is not None else c * 0.99).values,
        "Close":  c.values,
        "Volume": (np.full(n, 1_000_000.0) if volumes is None else np.asarray(volumes, dtype=float)),
    }, index=idx)


# ── TTM Squeeze ───────────────────────────────────────────────────────────────

def _coil_then_breakout(release_bars_ago: int):
    """~200 bars of normal vol, then a dead-flat coil, then an upside breakout
    ``release_bars_ago`` bars before the end (0 = still coiling)."""
    rng = np.random.default_rng(11)
    normal = 100 * np.cumprod(1 + rng.normal(0, 0.02, 200))
    anchor = normal[-1]
    coil = anchor * (1 + rng.normal(0, 0.0005, 30))       # vol collapses — BB inside KC
    if release_bars_ago == 0:
        closes = np.concatenate([normal, coil])
    else:
        burst = anchor * (1 + np.linspace(0.02, 0.10, release_bars_ago))  # hard breakout up
        closes = np.concatenate([normal, coil, burst])
    return _frame(closes)


def test_squeeze_fires_up_on_breakout_after_coil():
    score, label, bars = compute_ttm_squeeze_score("BRK", df=_coil_then_breakout(release_bars_ago=2))
    assert label == "FIRED_UP"
    assert score > 0.3
    assert 1 <= bars <= 3


def test_squeeze_on_gives_small_anticipatory_score():
    score, label, bars = compute_ttm_squeeze_score("COIL", df=_coil_then_breakout(release_bars_ago=0))
    assert label in ("SQUEEZE_ON", "NONE")                # deadband may absorb a flat coil
    assert abs(score) <= 0.35 + 1e-9                      # never more than the anticipatory cap


def test_squeeze_normal_vol_is_no_view():
    rng = np.random.default_rng(3)
    closes = 100 * np.cumprod(1 + rng.normal(0, 0.02, 260))   # never coils
    score, label, _ = compute_ttm_squeeze_score("WIDE", df=_frame(closes))
    assert (score, label) == (0.0, "NONE")


def test_squeeze_insufficient_data_is_no_view():
    assert compute_ttm_squeeze_score("NEW", df=_frame(np.linspace(10, 12, 30))) == (0.0, "NONE", 0)


# ── IV term structure ─────────────────────────────────────────────────────────

def _gex_ctx(**kw):
    sig = SimpleNamespace(ticker="AAPL", atm_iv_front=None, atm_iv_back=None,
                          front_dte=None, back_dte=None)
    for k, v in kw.items():
        setattr(sig, k, v)
    return SimpleNamespace(signals=[sig])


def test_iv_term_backwardation_is_bearish():
    ctx = _gex_ctx(atm_iv_front=0.60, atm_iv_back=0.45, front_dte=7, back_dte=28)
    score, slope_pts, label = compute_iv_term_score("AAPL", ctx)
    assert label == "BACKWARDATION"
    assert slope_pts == pytest.approx(15.0)
    assert score < -0.5


def test_iv_term_contango_is_mildly_bullish():
    ctx = _gex_ctx(atm_iv_front=0.30, atm_iv_back=0.38, front_dte=7, back_dte=28)
    score, _, label = compute_iv_term_score("AAPL", ctx)
    assert label == "CONTANGO"
    assert 0.0 < score < 1.0


def test_iv_term_flat_curve_and_missing_data_are_no_view():
    flat = _gex_ctx(atm_iv_front=0.40, atm_iv_back=0.395, front_dte=7, back_dte=28)
    assert compute_iv_term_score("AAPL", flat)[0] == 0.0
    assert compute_iv_term_score("AAPL", _gex_ctx())[2] == "NO_DATA"          # no IVs captured
    assert compute_iv_term_score("MSFT", _gex_ctx(atm_iv_front=0.4))[2] == "NO_DATA"  # not covered
    assert compute_iv_term_score("AAPL", None)[2] == "NO_DATA"


def test_iv_term_needs_a_real_dte_gap_and_sane_ivs():
    narrow = _gex_ctx(atm_iv_front=0.60, atm_iv_back=0.45, front_dte=14, back_dte=18)
    assert compute_iv_term_score("AAPL", narrow)[2] == "NO_DATA"              # <7d gap
    junk = _gex_ctx(atm_iv_front=0.001, atm_iv_back=0.45, front_dte=7, back_dte=28)
    assert compute_iv_term_score("AAPL", junk)[2] == "NO_DATA"                # junk yfinance IV


# ── Anchored VWAP ─────────────────────────────────────────────────────────────

def test_avwap_above_both_anchors_is_bullish():
    # Year-long climb ending at the top: price sits above BOTH anchored VWAPs.
    closes = np.linspace(50, 120, 260)
    score, hi_pct, lo_pct = compute_anchored_vwap_score("UP", df=_frame(closes))
    assert lo_pct > 0                                      # above the low-anchored VWAP
    assert score > 0.2


def test_avwap_below_both_anchors_is_bearish():
    # Peak, crash to the bottom, dead-cat bounce, then fade back BELOW the
    # post-bottom VWAP — price ends under both anchored VWAPs. (Ending AT the
    # bottom would make the low anchor the last bar → a legitimately ~0 leg.)
    closes = np.concatenate([np.linspace(80, 120, 120), np.linspace(120, 50, 100),
                             np.linspace(50, 60, 20), np.linspace(60, 51, 20)])
    score, hi_pct, lo_pct = compute_anchored_vwap_score("DN", df=_frame(closes))
    assert hi_pct < 0 and lo_pct < 0                       # below both anchors
    assert score < -0.2


def test_avwap_missing_volume_fails_closed():
    df = _frame(np.linspace(50, 120, 260)).drop(columns=["Volume"])
    assert compute_anchored_vwap_score("NOVOL", df=df) == (0.0, 0.0, 0.0)


def test_avwap_zero_volume_and_short_history_are_no_view():
    zero_vol = _frame(np.linspace(50, 120, 260), volumes=np.zeros(260))
    assert compute_anchored_vwap_score("ZV", df=zero_vol)[0] == 0.0
    short = _frame(np.linspace(50, 120, 120))
    assert compute_anchored_vwap_score("NEW", df=short) == (0.0, 0.0, 0.0)


def test_tier2_zero_price_rows_never_emit_nan():
    closes = np.linspace(50, 100, 260)
    closes[-6] = 0.0
    df = _frame(closes)
    s1, _, _ = compute_ttm_squeeze_score("BADPX", df=df)
    s2, h, lo = compute_anchored_vwap_score("BADPX", df=df)
    assert np.isfinite(s1) and np.isfinite(s2) and np.isfinite(h) and np.isfinite(lo)


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
    "enable_residual_momentum", "enable_volume_profile",
]

_T2_FLAGS = ("enable_ttm_squeeze", "enable_iv_term_structure", "enable_anchored_vwap")


def test_zero_impact_on_combine_confidence_and_agreement(monkeypatch):
    import src.signals.aggregator as agg
    for flag in _OFF:
        monkeypatch.setattr(settings, flag, False)
    monkeypatch.setattr(settings, "enable_news_sentiment", True)
    monkeypatch.setattr(agg, "analyse_sentiment",
                        lambda t, articles, force_engine=None: (0.4, f"news {t}"))
    monkeypatch.setattr(agg, "compute_ttm_squeeze_score", lambda t, df=None: (0.9, "FIRED_UP", 2))
    monkeypatch.setattr(agg, "compute_iv_term_score", lambda t, ctx: (-0.8, 12.0, "BACKWARDATION"))
    monkeypatch.setattr(agg, "compute_anchored_vwap_score", lambda t, df=None: (0.7, 5.0, 20.0))

    for flag in _T2_FLAGS:
        monkeypatch.setattr(settings, flag, False)
    off = agg.build_signals(["AAPL"], [])[0]

    for flag in _T2_FLAGS:
        monkeypatch.setattr(settings, flag, True)
    on = agg.build_signals(["AAPL"], [])[0]

    assert on.combined_score == off.combined_score
    assert on.confidence == off.confidence
    assert on.sources_agreeing == off.sources_agreeing
    assert on.direction == off.direction
    assert (on.squeeze_score, on.iv_term_score, on.avwap_score) == (0.9, -0.8, 0.7)
    assert (on.squeeze_label, on.iv_term_label) == ("FIRED_UP", "BACKWARDATION")
    assert (off.squeeze_score, off.iv_term_score, off.avwap_score) == (0.0, 0.0, 0.0)


def test_methods_reach_attribution_bridge():
    from src.performance.tracker import _ALL_METHODS, _method_scores_from_signal
    from src.models import TickerSignal
    for m in ("squeeze", "iv_term", "avwap"):
        assert m in _ALL_METHODS
    sig = TickerSignal(ticker="AAPL", direction="BULLISH", confidence=0.8,
                       sentiment_score=0.0, technical_score=0.0, rationale="test")
    sig.squeeze_score = 0.9
    sig.iv_term_score = -0.8
    sig.avwap_score = 0.7
    scores = _method_scores_from_signal("AAPL", "BULLISH", {"AAPL": sig})
    assert (scores["squeeze"], scores["iv_term"], scores["avwap"]) == (0.9, -0.8, 0.7)


def test_excluded_from_exit_consensus():
    from src.analysis.exit_conviction import exit_method_consensus
    base = {"money_flow": 0.2, "max_pain": -0.1}
    with_t2 = dict(base, squeeze=-1.0, iv_term=-1.0, avwap=-1.0)
    assert exit_method_consensus(with_t2) == exit_method_consensus(base)


def test_labels_and_categories_registered():
    from src.performance.tracker import METHOD_CATEGORIES, METHOD_LABELS
    for m in ("squeeze", "iv_term", "avwap"):
        assert m in METHOD_LABELS
    assert "squeeze" in METHOD_CATEGORIES["Technical"]
    assert "avwap" in METHOD_CATEGORIES["Technical"]
    assert "iv_term" in METHOD_CATEGORIES["Options"]


def test_gexsignal_iv_fields_optional_for_old_caches():
    # An old cache row (no ATM-IV fields) must deserialize with the defaults —
    # the scorer then reports NO_DATA instead of crashing.
    from src.models import GEXSignal
    sig = GEXSignal(ticker="SPY", spot_price=500.0, net_gex_bn=1.0, gex_normalized=0.5,
                    gex_signal="PINNED", gamma_flip=None, max_pain=None,
                    expected_move_pct=1.0, max_pain_bias="NEUTRAL",
                    dominant_expiry="2026-07-17", report_date=date(2026, 7, 8), summary="s")
    assert sig.atm_iv_front is None and sig.back_dte is None
    assert compute_iv_term_score("SPY", SimpleNamespace(signals=[sig]))[2] == "NO_DATA"
