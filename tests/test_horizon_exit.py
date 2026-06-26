"""Horizon-matched exit: a position held past its target-horizon window must be
STRONGLY re-confirmed by its opener to survive (else ``horizon_expired``); within
the window the normal opener-pinned decay gate applies unchanged."""

from datetime import datetime, timedelta
from types import SimpleNamespace

import pytest

from config.settings import settings
from src.utils import ET
from src.performance import tracker


@pytest.fixture
def _floors(monkeypatch):
    monkeypatch.setattr(settings, "enable_horizon_synthesis", True)
    monkeypatch.setattr(settings, "enable_horizon_matched_exit", True)
    monkeypatch.setattr(settings, "enable_llm_hold_review", True)
    monkeypatch.setattr(settings, "signal_decay_confidence_floor", 0.30)
    monkeypatch.setattr(settings, "signal_decay_confidence_floor_relative", 0.0)
    monkeypatch.setattr(settings, "horizon_expiry_floor_mult", 2.0)  # raised floor = 0.60


def _trade(hours_ago, **kw):
    dt = (datetime.now(ET) - timedelta(hours=hours_ago)).isoformat()
    t = {"action": "BUY", "confidence": 0.8, "target_horizon": "6h", "entry_datetime": dt}
    t.update(kw)
    return t


def _rev(action="BUY", conf=0.5):
    return SimpleNamespace(action=action, confidence=conf)


def test_raised_floor_only_past_window(_floors):
    assert tracker._horizon_expired_floor(_trade(8)) == pytest.approx(0.60)   # 0.30 × 2.0
    assert tracker._horizon_expired_floor(_trade(1)) is None                  # still inside 6h
    assert tracker._horizon_expired_floor(_trade(8, target_horizon="")) is None  # no horizon


def test_past_horizon_low_conviction_closes(_floors):
    assert tracker._evaluate_decay(_trade(8), None, None,
                                   hold_review=_rev("BUY", 0.50)) == "horizon_expired"


def test_past_horizon_strong_conviction_holds(_floors):
    # 0.70 ≥ raised floor 0.60 → a still-conviction winner survives past its window
    assert tracker._evaluate_decay(_trade(8), None, None,
                                   hold_review=_rev("BUY", 0.70)) is None


def test_past_horizon_neutral_review_closes(_floors):
    # A neutral HOLD/WATCH is not strong re-confirmation → close past the window
    assert tracker._evaluate_decay(_trade(8), None, None,
                                   hold_review=_rev("HOLD", 0.95)) == "horizon_expired"


def test_within_horizon_uses_normal_floor(_floors):
    t = _trade(1)
    assert tracker._evaluate_decay(t, None, None, hold_review=_rev("BUY", 0.50)) is None
    assert tracker._evaluate_decay(t, None, None, hold_review=_rev("BUY", 0.20)) == "llm_confidence_loss"


def test_flip_closes_immediately_regardless_of_horizon(_floors):
    assert tracker._evaluate_decay(_trade(8), None, None,
                                   hold_review=_rev("SELL", 0.99)) == "llm_signal_flipped"


def test_matched_exit_disabled_falls_back(_floors, monkeypatch):
    monkeypatch.setattr(settings, "enable_horizon_matched_exit", False)
    # past window but matched exit off → no raised floor; normal gate holds a
    # same-direction conviction above the base floor
    assert tracker._horizon_expired_floor(_trade(8)) is None
    assert tracker._evaluate_decay(_trade(8), None, None, hold_review=_rev("BUY", 0.50)) is None
