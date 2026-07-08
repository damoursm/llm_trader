"""Trend-predictability methods: the signed Kaufman/ADX scorer split long/short,
and its wiring into the per-method attribution bridge.

Guards the sign convention (uptrend → bullish long score, downtrend → bearish
short score, mutually exclusive), the no-data path, and that the four methods
reach the scores dict the signals panel / trade attribution consume.
"""

import numpy as np
import pandas as pd
import pytest

import src.signals.trend_predictability as tp
from config.settings import settings
from src.signals.trend_predictability import (TREND_PREDICT_METHODS,
                                              calibrate_trend_orientation,
                                              compute_trend_predictability_scores)


@pytest.fixture(autouse=True)
def _reset():
    tp.reset_cache()
    yield
    tp.reset_cache()


def _ohlcv(closes):
    close = pd.Series([float(c) for c in closes])
    return pd.DataFrame({"High": close + 0.5, "Low": close - 0.5, "Close": close})


def test_uptrend_fires_long_only():
    scores = compute_trend_predictability_scores("X", df=_ohlcv(range(100, 160)))
    assert scores["kaufman_long"] > 0        # efficient uptrend → bullish long score
    assert scores["kaufman_short"] == 0.0    # short side abstains
    assert scores["adx_long"] > 0            # strong uptrend via ADX·DMI
    assert scores["adx_short"] == 0.0
    # A clean monotone climb is maximally efficient.
    assert scores["kaufman_long"] == pytest.approx(1.0, abs=1e-6)


def test_downtrend_fires_short_only():
    scores = compute_trend_predictability_scores("X", df=_ohlcv(range(160, 100, -1)))
    assert scores["kaufman_short"] < 0       # efficient downtrend → bearish short score
    assert scores["kaufman_long"] == 0.0
    assert scores["adx_short"] < 0
    assert scores["adx_long"] == 0.0
    assert scores["kaufman_short"] == pytest.approx(-1.0, abs=1e-6)


def test_chop_gives_near_zero_kaufman():
    # Net-flat oscillation: no efficient trend either way → both Kaufman sides ~0.
    closes = [100 + (i % 2) for i in range(61)]   # ends at 100 → net 0 over the window
    scores = compute_trend_predictability_scores("X", df=_ohlcv(closes))
    assert abs(scores["kaufman_long"]) < 0.2
    assert abs(scores["kaufman_short"]) < 0.2


def test_insufficient_data_all_zero():
    scores = compute_trend_predictability_scores("X", df=_ohlcv(range(100, 110)))  # < _MIN_ROWS
    assert scores == {m: 0.0 for m in TREND_PREDICT_METHODS}


def test_scores_are_bounded():
    up = compute_trend_predictability_scores("X", df=_ohlcv(range(100, 170)))
    for v in up.values():
        assert -1.0 <= v <= 1.0


def test_default_orientation_is_pure_continuation():
    # With no orientation passed (all +1), the score keeps the trend's sign.
    up = compute_trend_predictability_scores("X", df=_ohlcv(range(100, 160)))
    assert up["kaufman_long"] == pytest.approx(1.0, abs=1e-6)   # continuation, bullish
    down = compute_trend_predictability_scores("X", df=_ohlcv(range(160, 100, -1)))
    assert down["kaufman_short"] == pytest.approx(-1.0, abs=1e-6)  # continuation, bearish


def test_reversal_orientation_flips_the_score():
    # A −1 (reversal) orientation flips a downtrend-context method to a BULLISH
    # score — predicting the bounce — and an uptrend method to bearish.
    orient = {"kaufman_short": -1.0, "adx_short": -1.0, "kaufman_long": -1.0, "adx_long": -1.0}
    down = compute_trend_predictability_scores("X", df=_ohlcv(range(160, 100, -1)), orientation=orient)
    assert down["kaufman_short"] > 0        # downtrend + reversal → predict up (bounce)
    assert down["adx_short"] > 0
    assert down["kaufman_long"] == 0.0      # not an uptrend → no view
    up = compute_trend_predictability_scores("X", df=_ohlcv(range(100, 160)), orientation=orient)
    assert up["kaufman_long"] < 0           # uptrend + reversal → predict down


# ── orientation calibration ──────────────────────────────────────────────────

def _orient_panel(n_days, per=8, side="up", forward="continue"):
    """Panel of one trend context: `side` up/down sets er_signed/adx_signed sign;
    `forward` continue/reverse sets whether the forward move follows the trend."""
    rows = []
    for d in range(n_days):
        day = f"2026-03-{d + 1:02d}"
        er = adx = (0.5 if side == "up" else -0.5)
        trend_up = side == "up"
        fwd = (2.0 if trend_up else -2.0) if forward == "continue" else (-2.0 if trend_up else 2.0)
        for _ in range(per):
            rows.append({"signal_date": day, "er_signed": er, "adx_signed": adx,
                         "fwd_ret_5d": float(fwd)})
    return pd.DataFrame(rows)


def test_orientation_continuation_stays_positive(monkeypatch):
    monkeypatch.setattr(settings, "trend_orientation_cal_min_rows", 10)
    o = calibrate_trend_orientation(_orient_panel(15, side="up", forward="continue"))
    assert o["kaufman_long"] == pytest.approx(1.0, abs=1e-6)   # uptrends continued → +1
    assert o["kaufman_short"] == pytest.approx(1.0)            # no downtrend rows → prior +1


def test_orientation_learns_reversal(monkeypatch):
    # Downtrends that BOUNCE (forward up) → the short-context orientation flips
    # toward reversal (negative), so the method will predict the bounce.
    monkeypatch.setattr(settings, "trend_orientation_cal_min_rows", 10)
    monkeypatch.setattr(settings, "trend_orientation_prior_n", 5)   # let evidence win faster
    o = calibrate_trend_orientation(_orient_panel(20, side="down", forward="reverse"))
    assert o["kaufman_short"] < 0        # downtrend context → predict reversal (up)
    assert o["adx_short"] < 0
    assert o["kaufman_long"] == pytest.approx(1.0)   # untouched → continuation prior


def test_orientation_thin_data_holds_continuation(monkeypatch):
    monkeypatch.setattr(settings, "trend_orientation_cal_min_rows", 500)  # nothing clears it
    o = calibrate_trend_orientation(_orient_panel(3, side="down", forward="reverse"))
    assert o == {m: 1.0 for m in TREND_PREDICT_METHODS}       # all stay at the +1 prior


def test_orientation_disabled_returns_continuation(monkeypatch):
    monkeypatch.setattr(settings, "enable_trend_predictability_methods", False)
    assert calibrate_trend_orientation(_orient_panel(20)) == {m: 1.0 for m in TREND_PREDICT_METHODS}


def test_methods_reach_attribution_bridge():
    # The four scores must flow from the TickerSignal through _method_scores_from_signal
    # into the scores dict the signals panel + trade attribution consume.
    from src.performance.tracker import _method_scores_from_signal, _ALL_METHODS
    from src.models import TickerSignal
    sig = TickerSignal(ticker="AAPL", direction="BULLISH", confidence=0.8,
                       sentiment_score=0.0, technical_score=0.0, rationale="test")
    sig.kaufman_long_score = 0.7
    sig.adx_short_score = -0.4
    scores = _method_scores_from_signal("AAPL", "BULLISH", {"AAPL": sig})
    assert set(scores) == set(_ALL_METHODS)          # every attributed method present
    assert scores["kaufman_long"] == pytest.approx(0.7)
    assert scores["adx_short"] == pytest.approx(-0.4)
    assert scores["kaufman_short"] == 0.0            # unset → 0 (no view)
    assert scores["adx_long"] == 0.0
