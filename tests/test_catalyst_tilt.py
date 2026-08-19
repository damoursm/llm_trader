"""catalyst_tilt (2026-08-15, panel-first weight 0): fit math, scorer contract,
fail-soft calibration, asof registration, and the add-method checklist."""

import numpy as np
import pandas as pd
import pytest

from config.settings import settings
import src.signals.catalyst_tilt as ct


@pytest.fixture(autouse=True)
def _fresh_cache():
    ct.reset_cache()
    yield
    ct.reset_cache()


def _events(n, catalyst, news, fwd):
    return pd.DataFrame({
        "ticker": [f"T{i % 7}" for i in range(n)],
        "signal_date": ["2026-07-15"] * n,
        "news": [news] * n,
        "catalyst": [catalyst] * n,
        "fwd_ret_pivot": [fwd] * n,
    })


# ── fit math ────────────────────────────────────────────────────────────────

def test_fit_orientation_signs_and_shrinkage():
    ev = pd.concat([
        _events(200, "contract_partnership", +0.5, +4.0),   # bull reads WIN → +tilt
        _events(200, "analyst", +0.5, -4.0),                # bull reads LOSE → −tilt
        _events(10, "ma_deal", +0.5, +4.0),                 # thin cell → shrunk small
    ])
    tilts = ct._fit(ev)
    assert tilts[("contract_partnership", "bull")] > 0.5
    assert tilts[("analyst", "bull")] < -0.5
    thin = tilts[("ma_deal", "bull")]
    assert 0 < thin < 0.35                                  # 10/(10+40) × clip(...)
    # bear side keyed separately
    ev2 = _events(200, "legal_regulatory", -0.5, +4.0)      # bear reads LOSE (stock rose)
    assert ct._fit(ev2)[("legal_regulatory", "bear")] < -0.5


# ── scorer contract ─────────────────────────────────────────────────────────

def test_score_keeps_flips_and_abstains():
    tilts = {("contract_partnership", "bull"): 0.8,
             ("analyst", "bull"): -0.6,
             ("guidance", "bull"): 0.03}
    keep = ct.compute_catalyst_tilt_score(0.5, "contract_partnership", tilts)
    assert keep == pytest.approx(0.4)
    flip = ct.compute_catalyst_tilt_score(0.5, "analyst", tilts)
    assert flip == pytest.approx(-0.3)                      # bullish read FLIPPED bearish
    assert ct.compute_catalyst_tilt_score(0.5, "guidance", tilts) == 0.0   # |tilt|<0.05
    assert ct.compute_catalyst_tilt_score(0.5, "ma_deal", tilts) == 0.0    # unknown cell
    assert ct.compute_catalyst_tilt_score(0.5, None, tilts) == 0.0         # no capture
    assert ct.compute_catalyst_tilt_score(0.0, "analyst", tilts) == 0.0    # no news
    assert ct.compute_catalyst_tilt_score(None, "analyst", tilts) == 0.0
    assert ct.compute_catalyst_tilt_score(0.5, "analyst", {}) == 0.0       # no calibration


def test_score_uses_the_side_of_the_live_read():
    tilts = {("earnings", "bull"): -0.5, ("earnings", "bear"): 0.5}
    assert ct.compute_catalyst_tilt_score(0.4, "earnings", tilts) == pytest.approx(-0.2)
    assert ct.compute_catalyst_tilt_score(-0.4, "earnings", tilts) == pytest.approx(-0.2)


# ── calibration: fail-soft + evidence floor + gate ──────────────────────────

def _ohlcv(price=10.0, bars=30):
    idx = pd.date_range("2026-06-01", periods=bars, freq="B")
    return pd.DataFrame({"Open": price, "High": price, "Low": price,
                         "Close": [price] * bars, "Volume": [1_000_000] * bars},
                        index=idx)


def test_calibration_failure_means_abstain(monkeypatch):
    import src.analysis.news_events as ne
    monkeypatch.setattr(ne, "load_news_events",
                        lambda *a, **k: (_ for _ in ()).throw(RuntimeError("db down")))
    assert ct.calibrate_catalyst_tilt(force=True) == {}


def test_calibration_evidence_floor(monkeypatch):
    import src.analysis.news_events as ne
    import src.data.cache as cache
    monkeypatch.setattr(ne, "load_news_events", lambda *a, **k: _events(50, "earnings", 0.5, 1.0))
    monkeypatch.setattr(cache, "load_ohlcv", lambda t, interval="1d": _ohlcv())
    assert ct.calibrate_catalyst_tilt(force=True) == {}       # 50 < _MIN_EVENTS


def test_calibration_fits_gated_events(monkeypatch):
    import src.analysis.news_events as ne
    import src.data.cache as cache
    ev = pd.concat([_events(250, "analyst", 0.5, -4.0),
                    _events(250, "product", 0.5, +4.0)])
    ev["signal_date"] = "2026-07-10"
    monkeypatch.setattr(ne, "load_news_events", lambda *a, **k: ev)
    monkeypatch.setattr(cache, "load_ohlcv", lambda t, interval="1d": _ohlcv(price=10.0))
    tilts = ct.calibrate_catalyst_tilt(force=True)
    assert tilts[("analyst", "bull")] < -0.5
    assert tilts[("product", "bull")] > 0.5
    # sub-$5 names are gated out → below the floor → {}
    ct.reset_cache()
    monkeypatch.setattr(cache, "load_ohlcv", lambda t, interval="1d": _ohlcv(price=2.0))
    assert ct.calibrate_catalyst_tilt(force=True) == {}


# ── asof + wiring ───────────────────────────────────────────────────────────

def test_asof_flush_registration():
    import inspect
    import src.analysis.asof as asof
    assert "src.signals.catalyst_tilt" in inspect.getsource(asof)
    assert callable(ct.reset_cache)


def test_catalyst_tilt_wiring_complete():
    from src.analysis.code_version import METHOD_SOURCES, unmapped_methods
    from src.db.schema import SIGNAL_BASE_METHOD_COLUMNS, _ADD_COLUMNS
    from src.models import TickerSignal
    from src.performance.tracker import _ALL_METHODS, METHOD_CATEGORIES, METHOD_LABELS
    from src.signals.agreement import FAMILY_OF
    from src.signals.aggregator import _BASE_WEIGHTS

    assert "catalyst_tilt" in _ALL_METHODS
    assert "catalyst_tilt" in SIGNAL_BASE_METHOD_COLUMNS
    assert "catalyst_tilt" in METHOD_CATEGORIES["Sentiment"]
    assert "catalyst_tilt" in METHOD_LABELS
    assert "catalyst_tilt" in METHOD_SOURCES and not unmapped_methods()
    assert ("signals", "catalyst_tilt") in {(t, c) for t, c, _ in _ADD_COLUMNS}
    assert "catalyst_tilt_score" in TickerSignal.model_fields
    # PANEL-FIRST: not weighted, not a family voter.
    assert "catalyst_tilt" not in _BASE_WEIGHTS
    assert "catalyst_tilt" not in FAMILY_OF


def test_score_flows_to_the_signal(monkeypatch):
    import src.signals.aggregator as agg
    from tests.test_news_events import _OFF
    for flag in _OFF:
        monkeypatch.setattr(settings, flag, False)
    monkeypatch.setattr(settings, "enable_news_sentiment", True)
    monkeypatch.setattr(settings, "enable_massive_tech", False)
    monkeypatch.setattr(settings, "enable_catalyst_tilt", True)
    monkeypatch.setattr(settings, "signal_scoring_max_workers", 2)
    monkeypatch.setattr(settings, "inverted_methods", "")
    monkeypatch.setattr(agg, "analyse_sentiment",
                        lambda t, a, force_engine=None:
                        (0.5, "r", {"catalyst": "analyst", "raw_score": 0.55}))
    monkeypatch.setattr(ct, "calibrate_catalyst_tilt",
                        lambda force=False: {("analyst", "bull"): -0.6})
    s = agg.build_signals(["TST"], articles=[], snapshots=[])[0]
    assert s.catalyst_tilt_score == pytest.approx(-0.3)       # bullish read flipped
