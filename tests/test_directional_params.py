"""Direction-split parameters (2026-07-25).

Four changes, each justified differently:
  * short borrow carry - a real cost longs never pay (structural, no sample)
  * directional() - the resolver; inert until an override is set
  * per-side horizon ramp - MEASURED per direction, shrunk, clamped
  * adverse stop - split by measurement, and the measurement REVERSED the
    structural prior (a stop helps longs, hurts shorts)
All pure/fakes, no network.
"""

import pytest

from config.settings import settings, directional
from src.performance.spread import _pct_return, borrow_cost_fraction, borrow_annual_pct


# -- the resolver -----------------------------------------------------------

def test_directional_falls_back_to_the_shared_value(monkeypatch):
    monkeypatch.setattr(settings, "adverse_stop_pct", 10.0)
    monkeypatch.setattr(settings, "adverse_stop_pct_long", None)
    monkeypatch.setattr(settings, "adverse_stop_pct_short", None)
    assert directional("adverse_stop_pct", "BUY") == 10.0
    assert directional("adverse_stop_pct", "SELL") == 10.0
    assert directional("adverse_stop_pct", None) == 10.0


def test_directional_uses_the_side_override(monkeypatch):
    monkeypatch.setattr(settings, "adverse_stop_pct", 10.0)
    monkeypatch.setattr(settings, "adverse_stop_pct_long", 8.0)
    monkeypatch.setattr(settings, "adverse_stop_pct_short", 20.0)
    assert directional("adverse_stop_pct", "BUY") == 8.0
    assert directional("adverse_stop_pct", "SELL") == 20.0
    # Direction aliases used across the codebase all resolve.
    assert directional("adverse_stop_pct", "LONG") == 8.0
    assert directional("adverse_stop_pct", "BEARISH") == 20.0
    # An unknown direction can only mean "no side" -> shared value.
    assert directional("adverse_stop_pct", "sideways") == 10.0


# -- short borrow carry -----------------------------------------------------

def test_borrow_is_charged_to_shorts_only(monkeypatch):
    monkeypatch.setattr(settings, "enable_short_borrow_cost", True)
    monkeypatch.setattr(settings, "short_borrow_annual_pct", 3.0)
    assert borrow_cost_fraction("BUY", 30) == 0.0, "longs never pay borrow"
    assert borrow_cost_fraction("SELL", 0) == 0.0
    assert borrow_cost_fraction("SELL", 365) == pytest.approx(0.03)
    # Linear in holding days (calendar, not trading).
    assert borrow_cost_fraction("SELL", 30) == pytest.approx(6 * borrow_cost_fraction("SELL", 5))


def test_borrow_prefers_the_real_broker_rate(monkeypatch):
    monkeypatch.setattr(settings, "enable_short_borrow_cost", True)
    monkeypatch.setattr(settings, "short_borrow_annual_pct", 3.0)
    htb = {"borrow_fee_pct": 25.0}
    assert borrow_annual_pct(htb) == 25.0
    assert borrow_cost_fraction("SELL", 365, htb) == pytest.approx(0.25)
    # A malformed stored rate must not crash or win over the default.
    assert borrow_annual_pct({"borrow_fee_pct": "n/a"}) == 3.0


def test_borrow_reduces_a_short_return_but_not_a_long(monkeypatch):
    monkeypatch.setattr(settings, "enable_short_borrow_cost", True)
    monkeypatch.setattr(settings, "short_borrow_annual_pct", 3.0)
    b = borrow_cost_fraction("SELL", 365)          # a full year = 3%
    short_with = _pct_return("SELL", 100.0, 95.0, entry_cost=0.0, exit_cost=0.0, borrow_cost=b)
    short_without = _pct_return("SELL", 100.0, 95.0, entry_cost=0.0, exit_cost=0.0)
    assert short_without - short_with == pytest.approx(3.0, abs=1e-6)
    long_with = _pct_return("BUY", 100.0, 105.0, entry_cost=0.0, exit_cost=0.0,
                            borrow_cost=borrow_cost_fraction("BUY", 365))
    long_without = _pct_return("BUY", 100.0, 105.0, entry_cost=0.0, exit_cost=0.0)
    assert long_with == long_without, "a long's return must be untouched"


def test_borrow_off_is_a_noop(monkeypatch):
    monkeypatch.setattr(settings, "enable_short_borrow_cost", False)
    assert borrow_cost_fraction("SELL", 365) == 0.0


# -- adverse stop -----------------------------------------------------------

def _t(action, ret):
    return {"action": action, "return_pct": ret}


def test_adverse_stop_uses_the_per_side_threshold(monkeypatch):
    from src.performance.tracker import _adverse_stop_triggered
    monkeypatch.setattr(settings, "enable_adverse_stop", True)
    monkeypatch.setattr(settings, "adverse_stop_pct_long", 8.0)
    monkeypatch.setattr(settings, "adverse_stop_pct_short", 20.0)
    # The measured split: longs are cut at -8%, shorts ride to -20%.
    assert _adverse_stop_triggered(_t("BUY", -8.5)) is True
    assert _adverse_stop_triggered(_t("BUY", -7.5)) is False
    assert _adverse_stop_triggered(_t("SELL", -8.5)) is False, "a short at -8% must NOT be cut"
    assert _adverse_stop_triggered(_t("SELL", -20.5)) is True


def test_adverse_stop_ignores_winners_and_can_be_disabled(monkeypatch):
    from src.performance.tracker import _adverse_stop_triggered
    monkeypatch.setattr(settings, "enable_adverse_stop", True)
    monkeypatch.setattr(settings, "adverse_stop_pct_long", 8.0)
    assert _adverse_stop_triggered(_t("BUY", +12.0)) is False
    monkeypatch.setattr(settings, "enable_adverse_stop", False)
    assert _adverse_stop_triggered(_t("BUY", -50.0)) is False
    # A zero threshold disables that side only.
    monkeypatch.setattr(settings, "enable_adverse_stop", True)
    monkeypatch.setattr(settings, "adverse_stop_pct_long", 0.0)
    assert _adverse_stop_triggered(_t("BUY", -50.0)) is False


# -- per-side horizon ramp --------------------------------------------------

def test_horizon_ramp_tightens_when_timeouts_did_worse(monkeypatch):
    """Negative gap (horizon_expired underperformed the side's other exits)
    means the time-stop let losers bleed -> raise the multiplier."""
    import src.performance.tracker as tracker
    monkeypatch.setattr(settings, "enable_horizon_ramp_calibration", True)
    monkeypatch.setattr(settings, "horizon_expiry_floor_mult", 1.5)
    monkeypatch.setattr(settings, "horizon_expiry_floor_mult_long", None)
    monkeypatch.setattr(settings, "horizon_ramp_prior_n", 0)      # full strength
    trades = ([{"status": "CLOSED", "action": "BUY", "exit_reason": "horizon_expired",
                "return_pct": -6.0} for _ in range(20)]
              + [{"status": "CLOSED", "action": "BUY", "exit_reason": "trailing_stop",
                  "return_pct": 0.0} for _ in range(20)])
    monkeypatch.setattr(tracker, "_load_trades", lambda: trades)
    tracker.reset_horizon_ramp_cache()
    assert tracker.calibrate_horizon_ramp("BUY") > 1.5


def test_horizon_ramp_loosens_when_timeouts_did_better(monkeypatch):
    import src.performance.tracker as tracker
    monkeypatch.setattr(settings, "enable_horizon_ramp_calibration", True)
    monkeypatch.setattr(settings, "horizon_expiry_floor_mult", 1.5)
    monkeypatch.setattr(settings, "horizon_expiry_floor_mult_short", None)
    monkeypatch.setattr(settings, "horizon_ramp_prior_n", 0)
    trades = ([{"status": "CLOSED", "action": "SELL", "exit_reason": "horizon_expired",
                "return_pct": +6.0} for _ in range(20)]
              + [{"status": "CLOSED", "action": "SELL", "exit_reason": "trailing_stop",
                  "return_pct": 0.0} for _ in range(20)])
    monkeypatch.setattr(tracker, "_load_trades", lambda: trades)
    tracker.reset_horizon_ramp_cache()
    val = tracker.calibrate_horizon_ramp("SELL")
    assert 1.0 <= val < 1.5, f"a paying time-stop should loosen, got {val}"


def test_horizon_ramp_is_inert_on_thin_evidence(monkeypatch):
    """Two observations must barely move it - the whole point of the shrinkage."""
    import src.performance.tracker as tracker
    monkeypatch.setattr(settings, "enable_horizon_ramp_calibration", True)
    monkeypatch.setattr(settings, "horizon_expiry_floor_mult", 1.5)
    monkeypatch.setattr(settings, "horizon_expiry_floor_mult_long", None)
    monkeypatch.setattr(settings, "horizon_ramp_prior_n", 30)
    trades = [{"status": "CLOSED", "action": "BUY", "exit_reason": "horizon_expired",
               "return_pct": -20.0},
              {"status": "CLOSED", "action": "BUY", "exit_reason": "trailing_stop",
               "return_pct": 0.0}]
    monkeypatch.setattr(tracker, "_load_trades", lambda: trades)
    tracker.reset_horizon_ramp_cache()
    assert tracker.calibrate_horizon_ramp("BUY") == pytest.approx(1.5, abs=0.05)


def test_horizon_ramp_respects_an_explicit_override(monkeypatch):
    import src.performance.tracker as tracker
    monkeypatch.setattr(settings, "enable_horizon_ramp_calibration", False)
    monkeypatch.setattr(settings, "horizon_expiry_floor_mult", 1.5)
    monkeypatch.setattr(settings, "horizon_expiry_floor_mult_short", 1.1)
    tracker.reset_horizon_ramp_cache()
    assert tracker.calibrate_horizon_ramp("SELL") == 1.1


# -- per-side actionable threshold ------------------------------------------
#
# Gate 1 is the highest-leverage parameter in the system, so this split is the
# most conservative of the four: it can only TIGHTEN, it is capped small, and
# it keys on whether confidence DISCRIMINATES on that side -- raising a bar
# only helps if the bar sorts good calls from bad.

def _ct(action, conf, ret):
    return {"status": "CLOSED", "action": action, "confidence": conf, "return_pct": ret}


def test_side_threshold_raises_when_confidence_discriminates(monkeypatch):
    import src.performance.tracker as tracker
    monkeypatch.setattr(settings, "enable_side_threshold_calibration", True)
    monkeypatch.setattr(settings, "actionable_threshold_adj_short", None)
    monkeypatch.setattr(settings, "side_threshold_prior_n", 0)     # full strength
    monkeypatch.setattr(settings, "side_threshold_max_adjust", 0.04)
    monkeypatch.setattr(settings, "side_threshold_rho_ref", 0.30)
    # Monotonic: higher confidence -> better return.
    trades = [_ct("SELL", 0.70 + i * 0.01, -5.0 + i) for i in range(20)]
    monkeypatch.setattr(tracker, "_load_trades", lambda: trades)
    tracker.reset_side_threshold_cache()
    assert tracker.calibrate_side_threshold("SELL") == pytest.approx(0.04, abs=1e-6)


def test_side_threshold_stays_put_when_confidence_is_uninformative(monkeypatch):
    """The long-side case: rho ~ 0 means cutting low-confidence calls would
    remove volume at random, so the bar must NOT move."""
    import src.performance.tracker as tracker
    monkeypatch.setattr(settings, "enable_side_threshold_calibration", True)
    monkeypatch.setattr(settings, "actionable_threshold_adj_long", None)
    monkeypatch.setattr(settings, "side_threshold_prior_n", 0)
    # A tent: returns rise then fall as confidence rises -> rho exactly 0.
    # (An ALTERNATING series is not neutral, it scores +0.087 — which is how the
    # tie-handling bug in the first _spearman_conf_return was caught.)
    trades = [_ct("BUY", 0.70 + i * 0.01, float(i if i < 10 else 19 - i)) for i in range(20)]
    monkeypatch.setattr(tracker, "_load_trades", lambda: trades)
    tracker.reset_side_threshold_cache()
    assert tracker.calibrate_side_threshold("BUY") == 0.0


def test_spearman_uses_average_tie_ranks(monkeypatch):
    """Regression: a ranker that breaks ties by INPUT order manufactures
    correlation, because trades load chronologically and confidence has many
    ties. An alternating +/-5% series must score ~0.09, not 0.57."""
    import src.performance.tracker as tracker
    rows = [{"confidence": 0.70 + i * 0.01, "return_pct": (5.0 if i % 2 else -5.0)}
            for i in range(20)]
    rho = tracker._spearman_conf_return(rows)
    assert rho == pytest.approx(0.0867, abs=0.005), f"tie handling regressed: {rho}"


def test_side_threshold_never_loosens(monkeypatch):
    """A NEGATIVE rho (confidence anti-predictive) must not lower the bar --
    loosening a risk gate is not something a calibration may do."""
    import src.performance.tracker as tracker
    monkeypatch.setattr(settings, "enable_side_threshold_calibration", True)
    monkeypatch.setattr(settings, "actionable_threshold_adj_long", None)
    monkeypatch.setattr(settings, "side_threshold_prior_n", 0)
    trades = [_ct("BUY", 0.70 + i * 0.01, 5.0 - i) for i in range(20)]   # rho strongly negative
    monkeypatch.setattr(tracker, "_load_trades", lambda: trades)
    tracker.reset_side_threshold_cache()
    assert tracker.calibrate_side_threshold("BUY") == 0.0


def test_side_threshold_is_inert_on_thin_evidence(monkeypatch):
    import src.performance.tracker as tracker
    monkeypatch.setattr(settings, "enable_side_threshold_calibration", True)
    monkeypatch.setattr(settings, "actionable_threshold_adj_short", None)
    monkeypatch.setattr(settings, "side_threshold_prior_n", 40)
    trades = [_ct("SELL", 0.80 + i * 0.05, -5.0 + 5 * i) for i in range(3)]
    monkeypatch.setattr(tracker, "_load_trades", lambda: trades)
    tracker.reset_side_threshold_cache()
    assert tracker.calibrate_side_threshold("SELL") < 0.005


def test_side_threshold_respects_a_pin_and_the_flag(monkeypatch):
    import src.performance.tracker as tracker
    monkeypatch.setattr(settings, "enable_side_threshold_calibration", True)
    monkeypatch.setattr(settings, "actionable_threshold_adj_long", 0.03)
    tracker.reset_side_threshold_cache()
    assert tracker.calibrate_side_threshold("BUY") == 0.03
    # A pinned NEGATIVE value is still clamped to no-loosening.
    monkeypatch.setattr(settings, "actionable_threshold_adj_long", -0.05)
    assert tracker.calibrate_side_threshold("BUY") == 0.0
    monkeypatch.setattr(settings, "actionable_threshold_adj_long", None)
    monkeypatch.setattr(settings, "enable_side_threshold_calibration", False)
    tracker.reset_side_threshold_cache()
    assert tracker.calibrate_side_threshold("BUY") == 0.0
