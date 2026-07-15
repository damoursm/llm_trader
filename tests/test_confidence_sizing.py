"""Confidence recalibration for sizing (2026-07-12) — the tilt input is the
LEDGER's empirical win rate for the trade's stated-confidence band (shrunk
toward the pooled win rate), not the stated number. House calibration idiom:
prior → evidence → shrinkage → clamp → fail-soft → registry-reported.
"""

import pytest

from config.settings import settings
from src.performance.confidence_sizing import (
    _INERT, calibrate_confidence_sizing, confidence_sizing_multiplier)


def _trade(conf, ret, status="CLOSED"):
    return {"status": status, "confidence": conf, "return_pct": ret}


def _ledger(band_specs):
    """band_specs: list of (confidence, n_wins, n_losses)."""
    out = []
    for conf, wins, losses in band_specs:
        out += [_trade(conf, 1.0)] * wins + [_trade(conf, -1.0)] * losses
    return out


@pytest.fixture(autouse=True)
def _on(monkeypatch):
    monkeypatch.setattr(settings, "enable_confidence_recal_sizing", True)
    monkeypatch.setattr(settings, "confidence_recal_span", 0.25)
    monkeypatch.setattr(settings, "confidence_recal_half_width", 0.10)
    monkeypatch.setattr(settings, "confidence_recal_prior_n", 20)
    monkeypatch.setattr(settings, "confidence_recal_min_trades", 20)


# ── calibration ─────────────────────────────────────────────────────────────

def test_inert_below_min_trades():
    cal = calibrate_confidence_sizing(_ledger([(0.80, 5, 5)]))   # 10 < 20
    assert cal == _INERT
    assert confidence_sizing_multiplier(0.80, cal) == 1.0


def test_open_and_unstamped_trades_excluded():
    trades = _ledger([(0.80, 15, 15)])
    trades += [_trade(0.9, 5.0, status="OPEN")] * 50             # open: ignored
    trades += [{"status": "CLOSED", "return_pct": 1.0}] * 50     # no confidence: ignored
    cal = calibrate_confidence_sizing(trades)
    assert cal["n"] == 30                                        # only the closed+stamped


def test_bands_shrink_toward_pool():
    # 0.80-band: 20/40 = 50% raw; 0.93-band: 16/20 = 80% raw. Pool = 36/60 = 60%.
    cal = calibrate_confidence_sizing(_ledger([(0.80, 20, 20), (0.93, 16, 4)]))
    assert cal["pool_win"] == 0.6
    b80 = next(b for b in cal["bands"] if b["lo"] == 0.78)
    b93 = next(b for b in cal["bands"] if b["lo"] == 0.92)
    # shrunk values sit BETWEEN raw and pool (Bayesian pull, k=20)
    assert 0.50 < b80["p_shrunk"] < 0.60
    assert 0.60 < b93["p_shrunk"] < 0.80
    # exact: (20 + 20*0.6)/(40+20)=0.5333 ; (16 + 20*0.6)/(20+20)=0.70
    assert b80["p_shrunk"] == pytest.approx(0.5333, abs=1e-3)
    assert b93["p_shrunk"] == pytest.approx(0.70, abs=1e-3)


def test_small_band_says_almost_nothing():
    """Same raw win rate, tiny n → shrunk ≈ pool ≈ neutral multiplier."""
    cal = calibrate_confidence_sizing(_ledger([(0.80, 15, 15), (0.93, 2, 0)]))
    b93 = next(b for b in cal["bands"] if b["lo"] == 0.92)
    assert abs(b93["p_shrunk"] - cal["pool_win"]) < 0.05          # pulled to pool
    assert abs(confidence_sizing_multiplier(0.93, cal) - 1.0) < 0.12


# ── multiplier direction (the point of the layer) ───────────────────────────

def test_band_above_pool_sizes_up_and_below_sizes_down():
    cal = calibrate_confidence_sizing(_ledger([(0.80, 10, 30), (0.93, 30, 10)]))
    up = confidence_sizing_multiplier(0.93, cal)
    down = confidence_sizing_multiplier(0.80, cal)
    assert up > 1.0 > down


def test_inversion_high_stated_confidence_that_loses_is_sized_down():
    """The headline behavior the raw ramp can never express: the ≥0.92 band
    empirically LOSES → sized DOWN despite the high stated number."""
    cal = calibrate_confidence_sizing(_ledger([(0.80, 30, 10), (0.95, 8, 32)]))
    assert confidence_sizing_multiplier(0.95, cal) < 1.0
    assert confidence_sizing_multiplier(0.80, cal) > 1.0


def test_clamped_at_span():
    # Extreme separation saturates the ramp at exactly 1 ± span.
    cal = calibrate_confidence_sizing(_ledger([(0.80, 0, 100), (0.93, 100, 0)]))
    assert confidence_sizing_multiplier(0.93, cal) == pytest.approx(1.25)
    assert confidence_sizing_multiplier(0.80, cal) == pytest.approx(0.75)


# ── fail-soft ───────────────────────────────────────────────────────────────

def test_neutral_when_disabled(monkeypatch):
    cal = calibrate_confidence_sizing(_ledger([(0.80, 0, 30), (0.93, 30, 0)]))
    monkeypatch.setattr(settings, "enable_confidence_recal_sizing", False)
    assert confidence_sizing_multiplier(0.93, cal) == 1.0
    assert calibrate_confidence_sizing(_ledger([(0.8, 30, 30)])) == _INERT


def test_neutral_on_missing_inputs():
    cal = calibrate_confidence_sizing(_ledger([(0.80, 20, 20), (0.93, 16, 4)]))
    assert confidence_sizing_multiplier(None, cal) == 1.0        # no confidence
    assert confidence_sizing_multiplier(0.93, None) == 1.0       # no calibration
    assert confidence_sizing_multiplier(0.93, {}) == 1.0
    assert confidence_sizing_multiplier("junk", cal) == 1.0      # unparseable


def test_neutral_for_unpopulated_band():
    """A band with zero closes is absence of evidence, not a verdict."""
    cal = calibrate_confidence_sizing(_ledger([(0.80, 25, 15)]))  # only 0.78-0.85 populated
    assert confidence_sizing_multiplier(0.95, cal) == 1.0


def test_registry_reported(monkeypatch):
    from src.performance import calibration as reg
    seen = {}
    monkeypatch.setattr(reg, "report_calibration",
                        lambda name, **kw: seen.update(name=name, **kw))
    calibrate_confidence_sizing(_ledger([(0.80, 20, 20), (0.93, 16, 4)]))
    assert seen["name"] == "confidence_recal_spread"
    assert seen["n_evidence"] == 60
    assert seen["value"] > 0
