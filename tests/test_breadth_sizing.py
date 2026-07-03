"""Evidence-based conviction sizing (2026-07-02/03 ledger study, n=44).

Two changes: (1) the confidence->size ramp's span above 1.0x is compressed by
``confidence_size_span`` (confidence was ~uninformative for outcomes); (2)
agreement BREADTH tilts size via a CONTINUOUS, SELF-CALIBRATING ramp — breadth
is normalized by the attribution-set size (the set already grew 19->28 once;
absolute thresholds would break, and normalizing by "methods that voted"
flips the signal negative), the ramp's center/width re-derive from the
ledger's own recent breadth distribution, and its strength is Bayesian-shrunk
by realized (closed-trade) evidence — decaying to neutral if the edge fades,
never auto-inverting.
"""

from datetime import datetime, timezone
from types import SimpleNamespace

import pytest

from config.settings import settings


@pytest.fixture(autouse=True)
def _fresh_breadth_cache():
    """The calibration cache is a process global — isolate it per test."""
    from src.performance import tracker
    tracker._breadth_cal_cache.update(ts=0.0, cal=None)
    yield
    tracker._breadth_cal_cache.update(ts=0.0, cal=None)


# ── _position_multiplier: span compression ────────────────────────────────────

def test_confidence_span_compression(monkeypatch):
    from src.performance import tracker
    # Legacy shape at span=1.0 (the old 1.0 → 2.0 ramp, anchor points intact).
    monkeypatch.setattr(settings, "confidence_size_span", 1.0)
    assert tracker._position_multiplier(0.78) == pytest.approx(1.00)
    assert tracker._position_multiplier(0.85) == pytest.approx(1.50)
    assert tracker._position_multiplier(0.92) == pytest.approx(1.85)
    assert tracker._position_multiplier(0.95) == pytest.approx(2.00)
    # Default compression (0.5): the same anchors at half the span above 1.0.
    monkeypatch.setattr(settings, "confidence_size_span", 0.5)
    assert tracker._position_multiplier(0.78) == pytest.approx(1.00)
    assert tracker._position_multiplier(0.85) == pytest.approx(1.25)
    assert tracker._position_multiplier(0.92) == pytest.approx(1.43, abs=0.011)  # 1+0.85*0.5, rounded
    assert tracker._position_multiplier(0.95) == pytest.approx(1.50)
    # Confidence-blind at span=0.
    monkeypatch.setattr(settings, "confidence_size_span", 0.0)
    assert tracker._position_multiplier(0.95) == pytest.approx(1.00)


# ── _breadth_multiplier: continuous ramp over an explicit calibration ─────────

_CAL = {"center": 0.46, "half_width": 0.10, "span_eff": 0.16, "n_cal": 40, "n_closed": 26}


def test_breadth_ramp_is_continuous_monotone_and_bounded(monkeypatch):
    from src.performance import tracker
    monkeypatch.setattr(settings, "breadth_sizing_enabled", True)
    m = lambda f: tracker._breadth_multiplier(f, _CAL)
    assert m(0.46) == pytest.approx(1.0)                      # neutral at the center
    assert m(0.56) == pytest.approx(1.16)                     # saturates at center+half
    assert m(0.36) == pytest.approx(0.84)                     # ... and center−half
    assert m(0.90) == pytest.approx(1.16)                     # bounded past saturation
    assert m(0.05) == pytest.approx(0.84)
    assert m(0.51) == pytest.approx(1.08)                     # halfway → half the tilt
    # Monotone non-decreasing across the whole domain.
    grid = [m(f / 100) for f in range(0, 101, 2)]
    assert all(b >= a for a, b in zip(grid, grid[1:]))
    # Unknown breadth is NEUTRAL, never a penalty; disabled → neutral.
    assert tracker._breadth_multiplier(None, _CAL) == pytest.approx(1.0)
    monkeypatch.setattr(settings, "breadth_sizing_enabled", False)
    assert tracker._breadth_multiplier(0.9, _CAL) == pytest.approx(1.0)


# ── _breadth_calibration: self-calibration + evidence shrinkage ───────────────

def _attr_trade(n_agree, n_set=28, status="CLOSED", ret=1.0, action="BUY"):
    ms = {f"m{i}": (0.5 if i < n_agree else 0.0) for i in range(n_set)}
    return {"ticker": "T", "action": action, "status": status,
            "return_pct": ret, "method_scores": ms}


def test_calibration_priors_hold_on_thin_ledger(monkeypatch):
    from src.performance import tracker
    monkeypatch.setattr(settings, "breadth_adaptive_min_trades", 10)
    cal = tracker._breadth_calibration([_attr_trade(10) for _ in range(3)])
    assert cal["center"] == pytest.approx(settings.breadth_center_prior)
    assert cal["half_width"] == pytest.approx(settings.breadth_halfwidth_floor)


def test_calibration_center_adapts_to_the_ledger(monkeypatch):
    from src.performance import tracker
    monkeypatch.setattr(settings, "breadth_adaptive_min_trades", 10)
    # A ledger whose breadth lives around 20/28 ≈ 0.714 recenters the ramp there:
    trades = [_attr_trade(20 + (i % 3) - 1) for i in range(30)]     # 19..21 of 28
    cal = tracker._breadth_calibration(trades)
    assert cal["center"] == pytest.approx(20 / 28, abs=0.02)
    assert cal["half_width"] >= settings.breadth_halfwidth_floor


def test_method_set_growth_recenters_automatically(monkeypatch):
    """The future-proofing property: the SAME relative breadth expressed over a
    doubled method set still sizes ~neutral once the ledger reflects the new
    set — no threshold retuning."""
    from src.performance import tracker
    monkeypatch.setattr(settings, "breadth_adaptive_min_trades", 10)
    monkeypatch.setattr(settings, "breadth_sizing_enabled", True)
    # Ledger after a hypothetical expansion to 56 methods, typical agree ~24.
    trades = [_attr_trade(24 + (i % 5) - 2, n_set=56) for i in range(40)]
    cal = tracker._breadth_calibration(trades)
    typical = tracker._breadth_multiplier(24 / 56, cal)
    assert typical == pytest.approx(1.0, abs=0.03)            # typical entry ≈ neutral
    broad = tracker._breadth_multiplier(34 / 56, cal)
    assert broad > typical                                     # breadth still rewarded


def test_evidence_shrinkage_and_neutral_decay(monkeypatch):
    from src.performance import tracker
    monkeypatch.setattr(settings, "breadth_adaptive_min_trades", 10)
    monkeypatch.setattr(settings, "breadth_size_span", 0.2)
    monkeypatch.setattr(settings, "breadth_edge_prior", 0.20)
    monkeypatch.setattr(settings, "breadth_edge_prior_n", 30)
    monkeypatch.setattr(settings, "breadth_edge_ref", 0.25)
    # No closed trades at all → prior-only edge: span = 0.2 × (0.20/0.25) = 0.16.
    open_only = [_attr_trade(10, status="OPEN") for _ in range(20)]
    cal = tracker._breadth_calibration(open_only)
    assert cal["span_eff"] == pytest.approx(0.16, abs=1e-3)
    # A long anti-predictive record drags d_post negative → clamp → NEUTRAL (0),
    # never an inversion: broad losers + narrow winners, 100 closed trades.
    bad = ([_attr_trade(24, ret=-2.0) for _ in range(50)]      # broad → losses
           + [_attr_trade(6, ret=+2.0) for _ in range(50)])    # narrow → wins
    cal = tracker._breadth_calibration(bad)
    assert cal["span_eff"] == pytest.approx(0.0, abs=1e-6)
    # Confirming evidence STRENGTHENS toward the full span.
    good = ([_attr_trade(24, ret=+2.0) for _ in range(60)]
            + [_attr_trade(6, ret=-2.0) for _ in range(60)])
    cal = tracker._breadth_calibration(good)
    assert cal["span_eff"] > 0.16                              # stronger than prior-only
    assert cal["span_eff"] <= 0.2 + 1e-9                       # never past the cap


# ── record_new_trades: end-to-end with a pinned calibration ──────────────────

def _mk_rec(ticker="BRD", action="BUY", confidence=0.78):
    from src.models import Recommendation
    return Recommendation(
        ticker=ticker, type="STOCK",
        direction="BULLISH" if action == "BUY" else "BEARISH",
        confidence=confidence, action=action, time_horizon="SWING",
        rationale="test", generated_at=datetime.now(timezone.utc),
    )


def _entry_env(monkeypatch):
    from src.performance import tracker
    monkeypatch.setattr(settings, "enable_intraday_timing", False)
    monkeypatch.setattr(settings, "enable_correlation_sizing", False)
    monkeypatch.setattr(tracker, "_execution_iso", lambda: "2026-06-10T15:00:00+00:00")  # RTH
    monkeypatch.setattr(tracker, "_fetch_price", lambda t: 50.0)
    monkeypatch.setattr(tracker, "_reference_close", lambda t: None)
    monkeypatch.setattr(tracker, "_breadth_calibration", lambda trades=None: dict(_CAL))
    return tracker


def _signal():
    return SimpleNamespace(direction="BULLISH", combined_score=0.4, confidence=0.9,
                           pattern_name="")


def test_record_new_trades_breadth_tilt(monkeypatch):
    tracker = _entry_env(monkeypatch)
    monkeypatch.setattr(settings, "breadth_sizing_enabled", True)

    # 12 of 16 methods agreeing → frac 0.75, saturated above center → ×1.16.
    monkeypatch.setattr(tracker, "_method_scores_from_signal",
                        lambda tk, d, s: {f"m{i}": (0.5 if i < 12 else 0.0) for i in range(16)})
    diag = tracker.record_new_trades([_mk_rec("BRD")], signals_by_ticker={"BRD": _signal()},
                                     run_id="b1")
    assert diag["opened"] == 1 and diag["breadth_tier_applied"] == 1
    t = next(t for t in tracker._load_trades() if t["ticker"] == "BRD")
    assert t["position_size_multiplier"] == pytest.approx(1.16)   # conf 0.78 → 1.0 base
    assert t["breadth_size_multiplier"] == pytest.approx(1.16)
    assert t["breadth_at_entry"] == 12
    assert t["breadth_frac_at_entry"] == pytest.approx(0.75)
    assert t["breadth_center_at_entry"] == pytest.approx(_CAL["center"])

    # 3 of 16 agreeing → frac 0.1875, saturated below → ×0.84.
    monkeypatch.setattr(tracker, "_method_scores_from_signal",
                        lambda tk, d, s: {f"m{i}": (0.5 if i < 3 else 0.0) for i in range(16)})
    tracker.record_new_trades([_mk_rec("NRW")], signals_by_ticker={"NRW": _signal()}, run_id="b2")
    t = next(t for t in tracker._load_trades() if t["ticker"] == "NRW")
    assert t["position_size_multiplier"] == pytest.approx(0.84)
    assert t["breadth_at_entry"] == 3


def test_record_new_trades_unknown_breadth_is_neutral(monkeypatch):
    # No signals_by_ticker at all → breadth unknown → ×1.0, never a penalty.
    tracker = _entry_env(monkeypatch)
    monkeypatch.setattr(settings, "breadth_sizing_enabled", True)
    diag = tracker.record_new_trades([_mk_rec("UNK")], run_id="b3")
    assert diag["opened"] == 1 and diag["breadth_tier_applied"] == 0
    t = next(t for t in tracker._load_trades() if t["ticker"] == "UNK")
    assert t["position_size_multiplier"] == pytest.approx(1.0)
    assert t["breadth_size_multiplier"] == pytest.approx(1.0)
    assert t["breadth_at_entry"] is None
    assert t["breadth_frac_at_entry"] is None
