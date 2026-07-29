"""Continuous (smoothed-step) regime threshold (2026-07-27).

The step table is a cliff: composites of -0.801 and -0.799 sit in different
regimes, and the live composite has sd 0.146 — so that boundary is well inside
one standard deviation of ordinary daily variation, and the same market sampled
two ticks apart could jump the actionable bar a full step (0.89 -> 0.87).

The fix smooths BOUNDARIES, not whole bands. A full band-to-band interpolation
was implemented first and rejected on measurement: it raised the bar on 100% of
observed runs (mean +0.0089) and would have dropped 7.9% of actionable trades —
a systematic tightening rather than the removal of an instability. These tests
pin the distinction, because it is easy to "fix the cliff" and silently move the
gate.
"""

import pytest

from config.settings import settings
import src.data.macro_regime as mr


BOUNDARIES = (-1.5, -0.8, -0.3, 0.3)


def test_band_interiors_are_unchanged():
    """The whole point: a run that is not near a boundary must get EXACTLY the
    threshold the step table gave it."""
    for norm, regime in ((-2.0, "PANIC"), (-1.0, "RISK_OFF"), (-0.5, "CAUTION"),
                         (0.0, "NEUTRAL"), (0.1, "NEUTRAL"), (0.6, "RISK_ON")):
        assert mr.continuous_threshold(norm) == pytest.approx(
            mr._REGIME_THRESHOLD[regime]), f"{norm} drifted from {regime}"


def test_no_cliff_at_any_boundary():
    """Two composites either side of a boundary must give near-identical bars.
    Before this, the RISK_OFF/CAUTION edge jumped 0.89 -> 0.87 on a 0.002 move."""
    for edge in BOUNDARIES:
        lo = mr.continuous_threshold(edge - 0.001)
        hi = mr.continuous_threshold(edge + 0.001)
        assert abs(lo - hi) < 0.005, (
            f"cliff of {abs(lo-hi):.4f} remains at norm={edge}")


def test_monotone_non_increasing():
    """Worse macro must never lower the bar."""
    xs = [i / 200.0 for i in range(-400, 200)]
    vals = [mr.continuous_threshold(x) for x in xs]
    for a, b in zip(vals, vals[1:]):
        assert b <= a + 1e-9, "threshold rose as conditions improved"


def test_never_leaves_the_band_values_it_interpolates():
    """A smoothed value must sit BETWEEN the two neighbouring band thresholds —
    never overshoot into a stricter or looser regime than either."""
    lo_all = min(mr._REGIME_THRESHOLD.values())
    hi_all = max(mr._REGIME_THRESHOLD.values())
    for i in range(-400, 200):
        v = mr.continuous_threshold(i / 200.0)
        assert lo_all - 1e-9 <= v <= hi_all + 1e-9


def test_neutral_never_drops_below_the_minimum_confidence_directive():
    """0.85 is a user directive, not a tunable. RISK_ON is deliberately below it;
    NEUTRAL must not be, including inside a transition zone."""
    for i in range(-30, 31):
        norm = i / 100.0                    # -0.30 .. +0.30, the NEUTRAL band
        assert mr.continuous_threshold(norm) >= 0.85 - 1e-9


def test_anchors_match_the_documented_table_at_band_centres():
    assert mr.continuous_threshold(-2.0) == pytest.approx(0.95)
    assert mr.continuous_threshold(-1.0) == pytest.approx(0.89)
    assert mr.continuous_threshold(-0.5) == pytest.approx(0.87)
    assert mr.continuous_threshold(0.0) == pytest.approx(0.85)
    assert mr.continuous_threshold(0.6) == pytest.approx(0.79)


def test_typical_live_composite_is_untouched():
    """Observed live composites cluster near +0.08 (sd 0.146). The typical run
    must be completely unaffected — this change is about boundaries, not level."""
    assert mr.continuous_threshold(0.08) == pytest.approx(mr._REGIME_THRESHOLD["NEUTRAL"])


def test_smoothing_is_narrow_relative_to_normal_variation():
    """The transition zone must be small against the composite's own sd (0.146),
    or the 'smoothing' becomes a level change in disguise."""
    assert mr._THRESHOLD_SMOOTH_HALFWIDTH <= 0.05


def test_flag_off_reproduces_the_exact_step_table(monkeypatch):
    monkeypatch.setattr(settings, "enable_continuous_regime_threshold", False)
    # The step path is what compute_macro_regime uses when disabled; assert the
    # table itself is untouched so a revert is a true revert.
    assert mr._REGIME_THRESHOLD == {"PANIC": 0.95, "RISK_OFF": 0.89,
                                    "CAUTION": 0.87, "NEUTRAL": 0.85,
                                    "RISK_ON": 0.79}


def test_regime_label_and_buy_block_stay_step_based():
    """Only the THRESHOLD is continuous. The label drives the BUY block, the
    exits and every analysis — and a half-blocked BUY is not a meaningful thing."""
    assert mr._REGIME_ALLOW_BUYS["PANIC"] is False
    assert mr._REGIME_ALLOW_BUYS["CAUTION"] is True
    # RISK_OFF stopped blocking on 2026-07-27 — it measured as the BEST regime
    # (+2.25% SPY at 21d over 337 days) and now takes a size haircut instead.
    # See tests/test_regime_haircut.py.
    assert mr._REGIME_ALLOW_BUYS["RISK_OFF"] is True
