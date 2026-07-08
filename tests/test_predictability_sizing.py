"""Predictability sizing (Tier 1): the self-calibrating trend-predictability tilt.

Guards the properties that make it safe and self-improving: the score blends
trend efficiency + ADX and renormalises when one is missing; the multiplier is
bounded, neutral when unknown, and neutral while evidence is thin; the
calibration measures the panel hit-gap, NEVER tilts on an anti-predictive edge,
and — the whole point — STRENGTHENS as more signal-days of confirming evidence
accrue.
"""

import pandas as pd
import pytest

import src.performance.predictability_sizing as ps
from config.settings import settings


@pytest.fixture(autouse=True)
def _reset():
    ps.reset_cache()
    yield
    ps.reset_cache()


# ── score blend ──────────────────────────────────────────────────────────────

def test_blend_score_weights_and_normalises(monkeypatch):
    monkeypatch.setattr(settings, "predictability_er_weight", 0.5)
    monkeypatch.setattr(settings, "predictability_adx_weight", 0.5)
    monkeypatch.setattr(settings, "predictability_adx_cap", 40.0)
    # ER 0.5, ADX 20 → 20/40 = 0.5 → blend 0.5.
    assert ps.blend_score(0.5, 20.0) == pytest.approx(0.5)
    # ADX missing → ER-only (renormalised), not diluted toward 0.
    assert ps.blend_score(0.8, None) == pytest.approx(0.8)
    # ER missing → ADX-only.
    assert ps.blend_score(None, 40.0) == pytest.approx(1.0)
    # Both missing → None.
    assert ps.blend_score(None, float("nan")) is None
    # Components clamp into [0,1] (ADX above cap saturates).
    assert ps.blend_score(1.5, 80.0) == pytest.approx(1.0)


# ── multiplier geometry ──────────────────────────────────────────────────────

_CAL = {"center": 0.3, "half_width": 0.2, "span_eff": 0.10, "edge": 1.0,
        "d_obs": 0.1, "n_days": 30, "n_rows": 300, "horizon": 5}


def test_multiplier_sizes_up_above_center_down_below():
    up = ps.predictability_multiplier(0.7, _CAL)     # well above center → +span (clamped)
    dn = ps.predictability_multiplier(0.0, _CAL)     # well below center → −span (clamped)
    mid = ps.predictability_multiplier(0.3, _CAL)    # at center → neutral
    assert up == pytest.approx(1.10)
    assert dn == pytest.approx(0.90)
    assert mid == pytest.approx(1.0)


def test_multiplier_neutral_when_unknown_or_inert(monkeypatch):
    assert ps.predictability_multiplier(None, _CAL) == 1.0          # no score
    inert = {**_CAL, "span_eff": 0.0}
    assert ps.predictability_multiplier(0.9, inert) == 1.0          # thin evidence
    monkeypatch.setattr(settings, "enable_predictability_sizing", False)
    assert ps.predictability_multiplier(0.9, _CAL) == 1.0           # disabled


def test_multiplier_is_bounded():
    m_hi = ps.predictability_multiplier(10.0, _CAL)   # far past the ramp edge
    m_lo = ps.predictability_multiplier(-10.0, _CAL)
    assert m_hi == pytest.approx(1.10) and m_lo == pytest.approx(0.90)


# ── calibration from the panel ───────────────────────────────────────────────

def _panel(n_days, per=4, hi_fwd=(2, 2, 2, -2), lo_fwd=(2, -2, -2, -2), start=1):
    """Feature panel: each day has `per` high-score rows (eff 0.5 / adx 30) and
    `per` low-score rows (eff 0.05 / adx 5), combined_score always +0.5, forward
    returns per the given patterns (sign vs +score = correct/incorrect)."""
    rows = []
    for d in range(n_days):
        day = f"2026-02-{start + d:02d}"
        for f in hi_fwd[:per]:
            rows.append({"signal_date": day, "combined_score": 0.5,
                         "eff_ratio": 0.5, "adx": 30.0, "fwd_ret_5d": float(f)})
        for f in lo_fwd[:per]:
            rows.append({"signal_date": day, "combined_score": 0.5,
                         "eff_ratio": 0.05, "adx": 5.0, "fwd_ret_5d": float(f)})
    return pd.DataFrame(rows)


def test_calibration_measures_positive_edge(monkeypatch):
    monkeypatch.setattr(settings, "predictability_cal_min_rows", 16)
    # High-score cohort ALL correct, low-score cohort ALL wrong → big hit-gap.
    fp = _panel(8, hi_fwd=(2, 2, 2, 2), lo_fwd=(-2, -2, -2, -2))
    cal = ps.calibrate_predictability(fp)
    assert cal["d_obs"] > 0.5
    assert cal["span_eff"] > 0.0
    assert cal["n_days"] == 8
    # center sits between the two score modes (~0.09 low, ~0.62 high).
    assert 0.1 < cal["center"] < 0.62


def test_calibration_never_tilts_on_anti_predictive_edge(monkeypatch):
    monkeypatch.setattr(settings, "predictability_cal_min_rows", 16)
    # INVERTED: high-score cohort wrong, low-score cohort right → negative gap.
    fp = _panel(8, hi_fwd=(-2, -2, -2, -2), lo_fwd=(2, 2, 2, 2))
    cal = ps.calibrate_predictability(fp)
    assert cal["d_obs"] < 0
    assert cal["span_eff"] == 0.0        # edge floored at 0 — never inverts sizing


def test_span_eff_strengthens_with_more_signal_days(monkeypatch):
    # THE self-improvement property: same moderate per-day edge, more signal-days
    # ⇒ the prior is outweighed ⇒ a stronger tilt. This is what makes the layer
    # improve over weeks/months as data accrues.
    monkeypatch.setattr(settings, "predictability_cal_min_rows", 16)
    few = ps.calibrate_predictability(_panel(3))
    many = ps.calibrate_predictability(_panel(8))
    assert few["span_eff"] < many["span_eff"]
    assert few["d_obs"] == pytest.approx(many["d_obs"], abs=0.08)   # same-shape edge


def test_calibration_inert_below_min_rows(monkeypatch):
    monkeypatch.setattr(settings, "predictability_cal_min_rows", 200)
    cal = ps.calibrate_predictability(_panel(3))     # only 24 rows
    assert cal["span_eff"] == 0.0
    assert cal == {**ps._INERT}


def test_disabled_returns_inert(monkeypatch):
    monkeypatch.setattr(settings, "enable_predictability_sizing", False)
    assert ps.calibrate_predictability(_panel(8)) == {**ps._INERT}


# ── end-to-end: the tilt actually changes position size in record_new_trades ──

def _pinned_env(monkeypatch, score_fn):
    """Isolate the predictability tilt inside record_new_trades: no network, RTH,
    breadth/edge/correlation off, and a pinned non-inert calibration + score."""
    from src.performance import tracker
    monkeypatch.setattr(settings, "enable_intraday_timing", False)
    monkeypatch.setattr(settings, "enable_correlation_sizing", False)
    monkeypatch.setattr(settings, "breadth_sizing_enabled", False)
    monkeypatch.setattr(settings, "edge_sizing_enabled", False)
    monkeypatch.setattr(settings, "enable_predictability_sizing", True)
    monkeypatch.setattr(tracker, "_execution_iso", lambda: "2026-06-10T15:00:00+00:00")
    monkeypatch.setattr(tracker, "_fetch_price", lambda t: 50.0)
    monkeypatch.setattr(tracker, "_reference_close", lambda t: None)
    # record_new_trades does `from ...predictability_sizing import ...` at call time,
    # so patching the module attributes here is picked up.
    pinned = {"center": 0.3, "half_width": 0.2, "span_eff": 0.10, "edge": 1.0,
              "d_obs": 0.1, "n_days": 30, "n_rows": 300, "horizon": 5}
    monkeypatch.setattr(ps, "calibrate_predictability", lambda *a, **k: dict(pinned))
    monkeypatch.setattr(ps, "predictability_score", score_fn)
    return tracker


def _rec(ticker):
    from datetime import datetime, timezone
    from src.models import Recommendation
    return Recommendation(ticker=ticker, type="STOCK", direction="BULLISH", confidence=0.78,
                          action="BUY", time_horizon="SWING", rationale="t",
                          generated_at=datetime.now(timezone.utc))


def test_record_new_trades_sizes_up_clean_trend(monkeypatch):
    tracker = _pinned_env(monkeypatch, lambda tk, asof=None: 0.7)   # 0.7 > center 0.3 → +span
    diag = tracker.record_new_trades([_rec("CLN")], signals_by_ticker=None, run_id="p1")
    assert diag["opened"] == 1 and diag["predictability_tilt_applied"] == 1
    t = next(t for t in tracker._load_trades() if t["ticker"] == "CLN")
    assert t["predictability_size_multiplier"] == pytest.approx(1.10)   # clamped +span
    assert t["position_size_multiplier"] == pytest.approx(1.10)         # conf 0.78 base = 1.0
    assert t["predictability_score_at_entry"] == pytest.approx(0.7)
    assert t["predictability_span_eff"] == pytest.approx(0.10)


def test_record_new_trades_sizes_down_chop(monkeypatch):
    tracker = _pinned_env(monkeypatch, lambda tk, asof=None: 0.0)   # 0.0 < center → −span
    tracker.record_new_trades([_rec("CHP")], signals_by_ticker=None, run_id="p2")
    t = next(t for t in tracker._load_trades() if t["ticker"] == "CHP")
    assert t["predictability_size_multiplier"] == pytest.approx(0.90)
    assert t["position_size_multiplier"] == pytest.approx(0.90)


def test_record_new_trades_neutral_when_score_unknown(monkeypatch):
    # No OHLCV history → score None → tilt exactly 1.0 (absence of evidence, not bearish).
    tracker = _pinned_env(monkeypatch, lambda tk, asof=None: None)
    diag = tracker.record_new_trades([_rec("UNK")], signals_by_ticker=None, run_id="p3")
    assert diag["predictability_tilt_applied"] == 0
    t = next(t for t in tracker._load_trades() if t["ticker"] == "UNK")
    assert t["predictability_size_multiplier"] == pytest.approx(1.0)
    assert t["position_size_multiplier"] == pytest.approx(1.0)
