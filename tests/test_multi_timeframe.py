"""Multi-timeframe orchestration: the timeframe blend, flag/cap gating, and the
scorer df-parameter parity that guarantees the refactor didn't change the daily
output."""

import importlib

import numpy as np
import pandas as pd
import pytest

import src.signals.multi_timeframe as mtf


# ── blend_timeframes ────────────────────────────────────────────────────────

def test_blend_daily_only_is_passthrough(monkeypatch):
    monkeypatch.setattr(mtf.settings, "tf_blend_30m", 0.2)
    monkeypatch.setattr(mtf.settings, "tf_blend_1d", 0.6)
    monkeypatch.setattr(mtf.settings, "tf_blend_1w", 0.2)
    # Only the daily view present → blend == daily (legacy behaviour preserved).
    assert mtf.blend_timeframes({"30m": None, "1d": 0.8, "1w": None}) == pytest.approx(0.8)


def test_blend_weighted_average(monkeypatch):
    monkeypatch.setattr(mtf.settings, "tf_blend_30m", 0.2)
    monkeypatch.setattr(mtf.settings, "tf_blend_1d", 0.6)
    monkeypatch.setattr(mtf.settings, "tf_blend_1w", 0.2)
    # (0.2*1.0 + 0.6*0.5 + 0.2*(-1.0)) / 1.0 = 0.3
    assert mtf.blend_timeframes({"30m": 1.0, "1d": 0.5, "1w": -1.0}) == pytest.approx(0.3)


def test_blend_excludes_abstaining_timeframe(monkeypatch):
    monkeypatch.setattr(mtf.settings, "tf_blend_30m", 0.2)
    monkeypatch.setattr(mtf.settings, "tf_blend_1d", 0.6)
    monkeypatch.setattr(mtf.settings, "tf_blend_1w", 0.2)
    # 30m abstains (0.0) → renormalise over 1d + 1w only.
    assert mtf.blend_timeframes({"30m": 0.0, "1d": 0.5, "1w": 0.5}) == pytest.approx(0.5)


def test_blend_no_views_is_zero():
    assert mtf.blend_timeframes({"30m": 0.0, "1d": 0.0, "1w": None}) == 0.0


def test_blend_lone_timeframe_full_magnitude(monkeypatch):
    monkeypatch.setattr(mtf.settings, "tf_blend_30m", 0.2)
    monkeypatch.setattr(mtf.settings, "tf_blend_1d", 0.6)
    monkeypatch.setattr(mtf.settings, "tf_blend_1w", 0.2)
    # Only 30m speaks → renormalised to its own value (not diluted by its weight).
    assert mtf.blend_timeframes({"30m": -0.7, "1d": 0.0, "1w": None}) == pytest.approx(-0.7)


# ── compute_timeframe_scores: flag + cap gating ─────────────────────────────

def _small_frame(n=80):
    idx = pd.bdate_range("2026-01-01", periods=n)
    c = np.linspace(100, 110, n)
    return pd.DataFrame({"Open": c, "High": c + 1, "Low": c - 1, "Close": c,
                         "Volume": [1_000_000] * n}, index=idx)


def test_compute_timeframe_scores_master_off(monkeypatch):
    monkeypatch.setattr(mtf.settings, "enable_multi_timeframe_signals", False)
    assert mtf.compute_timeframe_scores("AAA") == {}


def test_compute_timeframe_scores_produces_both_timeframes(monkeypatch):
    fx = _small_frame()
    monkeypatch.setattr(mtf.settings, "enable_multi_timeframe_signals", True)
    monkeypatch.setattr(mtf.settings, "enable_intraday_30m", True)
    monkeypatch.setattr(mtf.settings, "enable_weekly_signals", True)
    monkeypatch.setattr(mtf, "get_history", lambda t, interval="30m": fx)
    monkeypatch.setattr(mtf, "get_weekly_history", lambda t: fx)
    out = mtf.compute_timeframe_scores("AAA", allow_30m=True)
    expected = {f"{m}_{tf}" for m in mtf.TECHNICAL_METHODS for tf in mtf.NON_DAILY_TIMEFRAMES}
    assert set(out) == expected               # every method × {30m, 1w}


def test_compute_timeframe_scores_cap_skips_30m_fetch(monkeypatch):
    fx = _small_frame()
    monkeypatch.setattr(mtf.settings, "enable_multi_timeframe_signals", True)
    monkeypatch.setattr(mtf.settings, "enable_intraday_30m", True)
    monkeypatch.setattr(mtf.settings, "enable_weekly_signals", True)
    intraday_calls = []
    monkeypatch.setattr(mtf, "get_history",
                        lambda t, interval="30m": intraday_calls.append(t) or fx)
    monkeypatch.setattr(mtf, "get_weekly_history", lambda t: fx)
    out = mtf.compute_timeframe_scores("AAA", allow_30m=False)   # over the cap
    assert intraday_calls == []                                  # no 30m fetch attempted
    assert not any(k.endswith("_30m") for k in out)
    assert any(k.endswith("_1w") for k in out)                   # weekly still runs (free)


# ── Scorer df-parameter parity (df=None must reproduce passing the frame) ────

def _fixture_daily(n=260):
    idx = pd.bdate_range("2025-01-01", periods=n)
    rng = np.random.default_rng(42)
    close = np.abs(100 + np.cumsum(rng.normal(0, 1, n))) + 20.0
    return pd.DataFrame(
        {"Open": close, "High": close + 1.5, "Low": close - 1.5, "Close": close,
         "Volume": rng.integers(1_000_000, 5_000_000, n).astype(float)},
        index=idx,
    )


@pytest.mark.parametrize("modpath, func_name, patch_attr, get_score", [
    ("src.analysis.technical", "compute_technical_score", "get_history", lambda r: r.score),
    ("src.signals.vwap", "compute_vwap_score", "_get_ohlcv", lambda r: r[0]),
    ("src.signals.price_momentum", "compute_price_momentum_score", "_get_ohlcv", lambda r: r[0]),
    ("src.signals.money_flow", "compute_money_flow_score", "_get_ohlcv", lambda r: r[0]),
    ("src.signals.trend_strength", "compute_trend_strength_score", "_get_ohlcv", lambda r: r[0]),
    ("src.signals.iv_rank", "compute_iv_rank_score", "_get_ohlcv", lambda r: r[0]),
])
def test_scorer_df_param_parity(monkeypatch, modpath, func_name, patch_attr, get_score):
    """compute_*_score(t, df=frame) must equal compute_*_score(t) when the no-df
    fetch returns that same frame — guards that the refactor is behaviour-neutral."""
    mod = importlib.import_module(modpath)
    fx = _fixture_daily()
    fn = getattr(mod, func_name)
    with_df = get_score(fn("TEST", df=fx))
    monkeypatch.setattr(mod, patch_attr, lambda *a, **k: fx)   # df=None path returns the same frame
    no_df = get_score(fn("TEST"))
    assert with_df == pytest.approx(no_df)
