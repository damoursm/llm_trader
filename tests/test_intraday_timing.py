"""Intraday 30-min timing overlay (`src/signals/intraday_timing.py`).

TIMING only — it never changes the daily signal's direction. It can defer an
entry (`enable_intraday_timing`, on) and, opt-in, close a position
(`enable_intraday_exit`). So the properties that matter are the geometry of
"opposed" (a sign error here defers exactly the entries it should let through
and lets through the ones it should defer) and fail-graceful behaviour — a
missing 30-min feed must never block the pipeline or, worse, silently oppose
everything.

The one behaviour with a real-world trap is the extended-session fetch: outside
RTH the bars are pulled with `prepost=True`, because a 4 AM entry gate reading
yesterday's 16:00 momentum is measuring nothing.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from config.settings import settings
from src.signals import intraday_timing as it


class _FakeTicker:
    """Records the kwargs `history()` was called with, returns a canned frame."""
    last_kwargs: dict = {}

    def __init__(self, ticker):
        self.ticker = ticker

    def history(self, **kw):
        _FakeTicker.last_kwargs = kw
        return _FakeTicker.frame


def _frame(closes) -> pd.DataFrame:
    idx = pd.date_range("2026-08-14 09:30", periods=len(closes), freq="30min")
    return pd.DataFrame({"Close": np.asarray(closes, dtype=float)}, index=idx)


@pytest.fixture
def _yf(monkeypatch):
    monkeypatch.setattr(settings, "enable_fetch_data", True)
    monkeypatch.setattr(it, "current_session", lambda: "rth")
    monkeypatch.setattr(it.yf, "Ticker", _FakeTicker)
    _FakeTicker.last_kwargs = {}
    return _FakeTicker


# ── the momentum read ───────────────────────────────────────────────────────

def test_a_rising_series_reads_rising(_yf):
    _yf.frame = _frame([100.0 + i * 0.5 for i in range(40)])
    out = it.compute_intraday_timing("AAPL")
    assert out["classification"] == "RISING"
    assert out["score"] > 0.15
    assert out["last_price"] == pytest.approx(119.5)
    assert out["ret_30m"] > 0


def test_a_falling_series_reads_falling(_yf):
    _yf.frame = _frame([100.0 - i * 0.5 for i in range(40)])
    out = it.compute_intraday_timing("AAPL")
    assert out["classification"] == "FALLING"
    assert out["score"] < -0.15
    assert out["ret_30m"] < 0


def test_a_flat_series_reads_flat(_yf):
    _yf.frame = _frame([100.0] * 40)
    out = it.compute_intraday_timing("AAPL")
    assert out["classification"] == "FLAT"
    assert out["score"] == pytest.approx(0.0, abs=1e-9)
    assert out["ret_30m"] == 0.0


def test_the_score_is_bounded(_yf):
    """tanh squashes it — a violent move must not emit a score outside [-1, 1]
    that would then be compared against a threshold in that range."""
    _yf.frame = _frame([100.0] * 25 + [1000.0] * 15)
    assert -1.0 <= it.compute_intraday_timing("AAPL")["score"] <= 1.0
    _yf.frame = _frame([1000.0] * 25 + [1.0] * 15)
    assert -1.0 <= it.compute_intraday_timing("AAPL")["score"] <= 1.0


def test_classification_matches_the_sign_of_the_score(_yf):
    for closes in ([100 + i for i in range(40)], [100 - i * 0.1 for i in range(40)],
                   [100.0] * 40):
        _yf.frame = _frame(closes)
        out = it.compute_intraday_timing("AAPL")
        if out["classification"] == "RISING":
            assert out["score"] > 0
        elif out["classification"] == "FALLING":
            assert out["score"] < 0
        else:
            assert abs(out["score"]) <= 0.15


# ── the extended-session fetch ──────────────────────────────────────────────

def test_rth_fetches_regular_bars_only(_yf, monkeypatch):
    monkeypatch.setattr(it, "current_session", lambda: "rth")
    _yf.frame = _frame([100.0] * 40)
    it.compute_intraday_timing("AAPL")
    assert _yf.last_kwargs["prepost"] is False
    assert _yf.last_kwargs["interval"] == "30m"


@pytest.mark.parametrize("session", ["extended", "overnight"])
def test_off_hours_fetches_prepost_bars(_yf, monkeypatch, session):
    """Otherwise a 4 AM entry gate reads yesterday's 16:00 momentum — a stale
    number that still passes every threshold test."""
    monkeypatch.setattr(it, "current_session", lambda: session)
    _yf.frame = _frame([100.0] * 40)
    it.compute_intraday_timing("AAPL")
    assert _yf.last_kwargs["prepost"] is True


# ── fail-graceful ───────────────────────────────────────────────────────────

def test_disabled_fetching_returns_none(monkeypatch):
    monkeypatch.setattr(settings, "enable_fetch_data", False)
    monkeypatch.setattr(it.yf, "Ticker",
                        lambda t: pytest.fail("network hit while disabled"))
    assert it.compute_intraday_timing("AAPL") is None


def test_a_raising_feed_returns_none(_yf, monkeypatch):
    class _Boom:
        def __init__(self, t): pass
        def history(self, **kw): raise RuntimeError("yfinance 429")
    monkeypatch.setattr(it.yf, "Ticker", _Boom)
    assert it.compute_intraday_timing("AAPL") is None


@pytest.mark.parametrize("frame", [
    None,
    pd.DataFrame(),
    pd.DataFrame({"Open": [1.0] * 40}),          # no Close column
])
def test_unusable_frames_return_none(_yf, frame):
    _yf.frame = frame
    assert it.compute_intraday_timing("AAPL") is None


def test_too_few_bars_returns_none(_yf):
    """The slow EMA needs a real window; computing it on 5 bars would emit a
    confident-looking score built from almost nothing."""
    _yf.frame = _frame([100.0] * it._SLOW)
    assert it.compute_intraday_timing("AAPL") is None
    _yf.frame = _frame([100.0] * (it._SLOW + 1))
    assert it.compute_intraday_timing("AAPL") is not None


def test_nan_closes_are_dropped_before_the_length_check(_yf):
    closes = [100.0] * 40
    df = _frame(closes)
    df.iloc[:35, 0] = np.nan            # only 5 usable bars remain
    _yf.frame = df
    assert it.compute_intraday_timing("AAPL") is None


def test_non_positive_prices_return_none(_yf):
    _yf.frame = _frame([0.0] * 40)
    assert it.compute_intraday_timing("AAPL") is None


# ── the entry gate geometry ─────────────────────────────────────────────────

def test_a_buy_is_opposed_only_by_falling_momentum():
    assert it.opposes_entry("BUY", {"score": -0.8}, 0.5) is True
    assert it.opposes_entry("BUY", {"score": +0.8}, 0.5) is False
    assert it.opposes_entry("BUY", {"score": -0.2}, 0.5) is False   # below threshold


def test_a_sell_is_opposed_only_by_rising_momentum():
    assert it.opposes_entry("SELL", {"score": +0.8}, 0.5) is True
    assert it.opposes_entry("SELL", {"score": -0.8}, 0.5) is False
    assert it.opposes_entry("SELL", {"score": +0.2}, 0.5) is False


def test_the_threshold_boundary_is_inclusive():
    assert it.opposes_entry("BUY", {"score": -0.5}, 0.5) is True
    assert it.opposes_entry("SELL", {"score": 0.5}, 0.5) is True


def test_a_negative_threshold_is_read_as_a_magnitude():
    """The setting is documented as a magnitude; a sign slip in config must not
    invert the gate into opposing every entry."""
    assert it.opposes_entry("BUY", {"score": -0.8}, -0.5) is True
    assert it.opposes_entry("BUY", {"score": +0.8}, -0.5) is False


@pytest.mark.parametrize("timing", [None, {}, {"score": None}])
def test_missing_timing_never_opposes(timing):
    """Fail OPEN. The overlay is a nicety; an unavailable 30-min feed must not
    silently stop the book from trading."""
    assert it.opposes_entry("BUY", timing, 0.5) is False
    assert it.opposes_entry("SELL", timing, 0.5) is False
    assert it.reverses_position("BUY", timing, 0.5) is False


def test_an_unknown_action_never_opposes():
    assert it.opposes_entry("HOLD", {"score": -0.9}, 0.5) is False
    assert it.opposes_entry("", {"score": -0.9}, 0.5) is False


def test_exit_accelerator_shares_the_entry_geometry():
    """`reverses_position` is deliberately the same test — a long is hurt by the
    same falling tape that would have deferred opening it."""
    for action in ("BUY", "SELL"):
        for score in (-0.9, -0.4, 0.0, 0.4, 0.9):
            t = {"score": score}
            assert it.reverses_position(action, t, 0.5) == it.opposes_entry(action, t, 0.5)
