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


def test_floor_ramps_continuously_past_window(_floors):
    # No cliff: AT the window the floor equals the base floor (0.30), then ramps
    # toward base × mult (0.60) over one window past expiry (ramp_windows=1.0).
    assert tracker._horizon_expired_floor(_trade(6)) == pytest.approx(0.30)          # ramp 0 → base
    assert tracker._horizon_expired_floor(_trade(8)) == pytest.approx(0.40, abs=0.01)  # 8/6 → ramp .33
    assert tracker._horizon_expired_floor(_trade(12)) == pytest.approx(0.60)         # 2× window → full
    assert tracker._horizon_expired_floor(_trade(1)) is None                         # inside the window
    assert tracker._horizon_expired_floor(_trade(8, target_horizon="")) is None      # no horizon


def test_past_horizon_faded_conviction_closes(_floors):
    # 8h past a 6h window → ramped floor 0.40. Below it closes; the SAME 0.50 that
    # the old hard 0.60 cliff would have cut now SURVIVES (the softening).
    assert tracker._evaluate_decay(_trade(8), None, None,
                                   hold_review=_rev("BUY", 0.35)) == "horizon_expired"
    assert tracker._evaluate_decay(_trade(8), None, None,
                                   hold_review=_rev("BUY", 0.50)) is None


def test_fully_expired_uses_full_floor(_floors):
    # 12h = 2× the 6h window → ramp saturated, floor = base × mult = 0.60.
    assert tracker._evaluate_decay(_trade(12), None, None,
                                   hold_review=_rev("BUY", 0.70)) is None                # strong survives
    assert tracker._evaluate_decay(_trade(12), None, None,
                                   hold_review=_rev("BUY", 0.55)) == "horizon_expired"   # below 0.60


def test_neutral_hold_flushed_only_when_fully_expired(_floors):
    # A neutral HOLD is NOT cut at the boundary / mid-ramp (the premature-cut fix)…
    assert tracker._evaluate_decay(_trade(8), None, None,
                                   hold_review=_rev("HOLD", 0.95)) is None
    # …but a persistent neutral position FULLY past its window (ramp ≥ 1) is flushed.
    assert tracker._evaluate_decay(_trade(12), None, None,
                                   hold_review=_rev("HOLD", 0.95)) == "horizon_expired"


def test_neutral_flush_can_be_disabled(_floors, monkeypatch):
    monkeypatch.setattr(settings, "horizon_expiry_flush_neutral", False)
    # Even fully past the window, a neutral HOLD is never force-closed when off…
    assert tracker._evaluate_decay(_trade(12), None, None,
                                   hold_review=_rev("HOLD", 0.95)) is None
    # …while the ramped conviction bar still cuts a faded same-direction re-affirmation.
    assert tracker._evaluate_decay(_trade(12), None, None,
                                   hold_review=_rev("BUY", 0.50)) == "horizon_expired"


def test_ramp_windows_zero_restores_cliff(_floors, monkeypatch):
    monkeypatch.setattr(settings, "horizon_expiry_ramp_windows", 0.0)
    # ramp_windows → 0: any overage saturates the ramp immediately → the old hard
    # base × mult cliff the moment the window passes.
    assert tracker._horizon_expired_floor(_trade(8)) == pytest.approx(0.60)


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
