"""Exit-conviction consensus — the entry-breadth analog on the exit side.

Verifies the raw-method consensus, the evidence-throttled + bounded span, the
floor-adjustment sign/clamp, and the wiring into _evaluate_decay (raises the
close bar on exit consensus, lowers it on hold consensus, no-op with no data
so pre-2026-07-03 behavior is preserved).
"""

import pytest

from config.settings import settings
import src.analysis.exit_conviction as ec


# ── consensus ─────────────────────────────────────────────────────────────────

def test_consensus_excludes_llm_and_aggregator_and_layers():
    # llm_review, aggregator, mfe/mae/horizon/macro_regime are excluded; the mean
    # is over the raw signal methods only.
    scores = {"llm_review": +0.9, "aggregator": -0.9, "mfe": -1.0, "macro_regime": -1.0,
              "news": -0.4, "momentum": -0.6, "tech": -0.2}
    assert ec.exit_method_consensus(scores) == pytest.approx((-0.4 - 0.6 - 0.2) / 3)
    assert ec.exit_method_consensus({"llm_review": 0.5}) is None    # nothing raw scored
    assert ec.exit_method_consensus({}) is None
    # bounded to [-1, 1]
    assert ec.exit_method_consensus({"a": -3.0, "b": -3.0}) == pytest.approx(-1.0)


# ── calibration: evidence ramp + bounds + registry ───────────────────────────

def _closed(n):
    return [{"status": "CLOSED"} for _ in range(n)] + [{"status": "OPEN"}]


def test_span_ramps_with_closed_sample(monkeypatch):
    monkeypatch.setattr(settings, "enable_exit_conviction", True)
    monkeypatch.setattr(settings, "exit_conviction_span_prior", 0.03)
    monkeypatch.setattr(settings, "exit_conviction_span_max", 0.10)
    monkeypatch.setattr(settings, "exit_conviction_prior_n", 40)
    # No closes → prior span exactly.
    assert ec.exit_conviction_calibration([])["span_eff"] == pytest.approx(0.03)
    # 40 closes → halfway to the cap: 0.03 + (0.10−0.03)·0.5 = 0.065.
    assert ec.exit_conviction_calibration(_closed(40))["span_eff"] == pytest.approx(0.065, abs=1e-3)
    # Large sample → approaches (never exceeds) the cap.
    big = ec.exit_conviction_calibration(_closed(4000))["span_eff"]
    assert 0.099 <= big <= 0.10
    # Disabled → 0.
    monkeypatch.setattr(settings, "enable_exit_conviction", False)
    assert ec.exit_conviction_calibration(_closed(40))["span_eff"] == 0.0


def test_calibration_reports_to_registry(monkeypatch):
    monkeypatch.setattr(settings, "enable_exit_conviction", True)
    from src.performance.calibration import get_calibrations, reset_calibrations
    reset_calibrations()
    ec.exit_conviction_calibration(_closed(10))
    names = {c["name"] for c in get_calibrations()}
    assert "exit_conviction_span" in names


# ── floor adjustment: sign + clamp ───────────────────────────────────────────

def test_adjustment_sign_and_clamp(monkeypatch):
    monkeypatch.setattr(settings, "enable_exit_conviction", True)
    cal = {"span_eff": 0.10}
    # Consensus says EXIT (negative) → positive adj (raise floor, close readier).
    assert ec.exit_floor_adjustment({"news": -0.5, "tech": -0.5}, cal) == pytest.approx(0.05)
    # Consensus says HOLD (positive) → negative adj (lower floor, hold readier).
    assert ec.exit_floor_adjustment({"news": +0.5, "tech": +0.5}, cal) == pytest.approx(-0.05)
    # Clamped to ±span_eff even on a saturated consensus.
    assert ec.exit_floor_adjustment({"news": -1.0, "tech": -1.0}, cal) == pytest.approx(0.10)
    # No consensus / zero span / disabled → 0.
    assert ec.exit_floor_adjustment({"llm_review": -0.9}, cal) == pytest.approx(0.0)
    assert ec.exit_floor_adjustment({"news": -0.5}, {"span_eff": 0.0}) == pytest.approx(0.0)
    monkeypatch.setattr(settings, "enable_exit_conviction", False)
    assert ec.exit_floor_adjustment({"news": -0.5}, cal) == pytest.approx(0.0)


# ── wiring into _evaluate_decay ──────────────────────────────────────────────

def _held(action="BUY", entry_conf=0.85):
    return {"ticker": "XLE", "type": "ETF", "action": action, "status": "OPEN",
            "confidence": entry_conf,
            "llm_synthesis_model": "claude-haiku-4-5-20251001",
            "llm_sentiment_model": "claude-haiku-4-5-20251001",
            "entry_date": "2026-06-13", "entry_datetime": "2026-06-13T15:00:00+00:00"}


def _rev(action="BUY", confidence=0.85):
    from src.models import Recommendation
    from datetime import datetime, timezone
    return Recommendation(ticker="XLE", type="ETF",
                          direction="BULLISH" if action == "BUY" else "BEARISH",
                          action=action, confidence=confidence, time_horizon="1w",
                          rationale="r", generated_at=datetime.now(timezone.utc))


@pytest.fixture(autouse=True)
def _no_default_time_stop(monkeypatch):
    monkeypatch.setattr(settings, "horizon_default_window", "")   # isolate the floor test
    monkeypatch.setattr(settings, "enable_exit_floor_calibration", True) if hasattr(
        settings, "enable_exit_floor_calibration") else None


def test_exit_consensus_nudge_flips_a_borderline_close(monkeypatch):
    from src.performance import tracker
    monkeypatch.setattr(settings, "exit_floor_calibration_enabled", False)  # base floor = 0.45/rel
    # entry_conf 0.85 → base floor = max(0.45, 0.55·0.85) = 0.4675 (2026-07-11
    # recalibration). A same-dir reaffirm at conv 0.49 HOLDS at the base floor…
    trade, rev = _held(entry_conf=0.85), _rev("BUY", 0.49)
    assert tracker._evaluate_decay(trade, None, None, rev, exit_conviction_adj=0.0) is None
    # …but with the methods corroborating an EXIT (+0.05 raises the floor to
    # 0.5175), 0.49 < 0.5175 → close.
    assert tracker._evaluate_decay(trade, None, None, rev,
                                   exit_conviction_adj=+0.05) == "llm_confidence_loss"
    # Methods corroborating a HOLD (−0.05 lowers the floor) keeps it open, and a
    # CONFIDENT reaffirm (0.80) is never tipped by the bounded nudge.
    assert tracker._evaluate_decay(trade, None, None, rev,
                                   exit_conviction_adj=-0.05) is None
    assert tracker._evaluate_decay(trade, None, None, _rev("BUY", 0.80),
                                   exit_conviction_adj=+0.10) is None


def test_zero_adjustment_is_exact_legacy_behavior(monkeypatch):
    from src.performance import tracker
    monkeypatch.setattr(settings, "exit_floor_calibration_enabled", False)
    trade = _held(entry_conf=0.85)
    # At adj=0 the effective floor == base floor for a range of convictions.
    for conv in (0.40, 0.55, 0.5525, 0.60, 0.90):
        legacy = tracker._evaluate_decay(trade, None, None, _rev("BUY", conv))
        explicit0 = tracker._evaluate_decay(trade, None, None, _rev("BUY", conv),
                                            exit_conviction_adj=0.0)
        assert legacy == explicit0
