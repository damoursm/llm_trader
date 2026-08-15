"""Tests for the ML combine arm — the ml_stacker as combined_buy_score plus
the min-hold exit gate. All the live pieces are off by default; these pin the
mechanism (fail-soft, correct feature set, the swap arm flag, the min-hold gate)."""

from __future__ import annotations

from datetime import date, timedelta

import pandas as pd
import pytest

from src.analysis import ml_stacker as ms
from src.db.schema import SIGNAL_BASE_METHOD_COLUMNS


# ── feature set + circularity ────────────────────────────────────────────────

def test_live_features_are_21_weighted_methods_no_combine():
    # The live stacker trains/infers on exactly the weighted methods the combine
    # uses — a subset of the persisted base columns, and NEVER a weight-derived
    # aggregate (the circularity guard).
    assert len(ms.STACKER_LIVE_FEATURES) == 21
    assert set(ms.STACKER_LIVE_FEATURES).issubset(set(SIGNAL_BASE_METHOD_COLUMNS))
    for banned in ("combined_score", "combined_buy_score", "combined_sell_score", "confidence"):
        assert banned not in ms.STACKER_LIVE_FEATURES


def test_live_features_match_the_aggregator_method_map():
    # Drift guard: the live feature set must equal the aggregator's weighted
    # method_score_map keys (the scores actually available at the combine point).
    import inspect
    import src.signals.aggregator as agg
    src = inspect.getsource(agg._score_ticker) if hasattr(agg, "_score_ticker") else inspect.getsource(agg)
    # every live feature appears as a method_score_map key in the aggregator
    for m in ms.STACKER_LIVE_FEATURES:
        assert f'"{m}":' in src, f"{m} not found as a method_score_map key in aggregator"


# ── the centered conviction map ──────────────────────────────────────────────

def test_buy_conviction_from_proba_centering():
    assert ms.buy_conviction_from_proba(0.5) == 0.0        # neutral -> no buy view
    assert ms.buy_conviction_from_proba(0.4) == 0.0        # bearish -> no buy view
    assert ms.buy_conviction_from_proba(1.0) == 1.0        # certain -> full
    assert ms.buy_conviction_from_proba(0.75) == pytest.approx(0.5)


def test_compute_buy_conviction_fail_soft_without_artifact(tmp_path, monkeypatch):
    # No artifact / lightgbm missing => None, so the caller keeps the weighted
    # combine. An invisible degradation to a broken model is impossible.
    monkeypatch.setattr(ms, "_BUY_MODEL_PATH", tmp_path / "nope.pkl")
    ms.reset_buy_caches()
    scores = {f: 0.3 for f in ms.STACKER_LIVE_FEATURES}
    assert ms.compute_buy_conviction(scores) is None


def test_compute_sell_conviction_fail_soft_without_artifact(tmp_path, monkeypatch):
    # The sell side is symmetric: no artifact => None => keep the weighted combine.
    monkeypatch.setattr(ms, "_SELL_MODEL_PATH", tmp_path / "nope.pkl")
    ms.reset_sell_caches()
    scores = {f: -0.3 for f in ms.STACKER_LIVE_FEATURES}
    assert ms.compute_sell_conviction(scores) is None


# ── probability calibration ──────────────────────────────────────────────────

def test_isotonic_is_monotone_and_bounded():
    from src.analysis.ml_train import IsotonicCalibrator
    rng = __import__("numpy").random.default_rng(0)
    p = rng.random(500)
    y = (rng.random(500) < p).astype(int)          # genuinely informative
    c = IsotonicCalibrator().fit(p, y)
    xs = [0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0]
    out = c.transform(xs)
    assert all(out[i] <= out[i + 1] + 1e-12 for i in range(len(out) - 1)), "not monotone"
    assert out.min() >= 0.0 and out.max() <= 1.0


def test_isotonic_improves_a_miscalibrated_model():
    # An over-spread (overconfident) forecaster — the boosted-tree failure mode.
    import numpy as np
    from src.analysis.ml_train import IsotonicCalibrator, brier, brier_skill
    rng = np.random.default_rng(1)
    n = 4000
    truth = rng.uniform(0.3, 0.7, n)
    y = (rng.random(n) < truth).astype(int)
    raw = np.clip((truth - 0.5) * 3.0 + 0.5, 0, 1)
    cal = IsotonicCalibrator().fit(raw, y).transform(raw)
    assert brier(cal, y) < brier(raw, y)
    assert brier_skill(raw, y) < 0 < brier_skill(cal, y)


def test_isotonic_collapses_a_no_signal_model_to_the_base_rate():
    # THE SAFETY PROPERTY: a model with no ranking power must ABSTAIN, not assert.
    # Uncalibrated, ml_buy was emitting conviction up to 0.86 on coin-flip names.
    import numpy as np
    from src.analysis.ml_train import IsotonicCalibrator
    rng = np.random.default_rng(2)
    n = 6000
    raw = rng.random(n)                             # pure noise, no relation to y
    y = (rng.random(n) < 0.47).astype(int)
    cal = IsotonicCalibrator().fit(raw, y).transform(raw)
    assert abs(cal.mean() - y.mean()) < 0.02        # centred on the base rate
    # …and the resulting BUY conviction is ~0 for essentially the whole universe.
    conv = np.clip(2 * cal - 1, 0, 1)
    assert (conv > 0.15).mean() < 0.02


def test_isotonic_unfitted_and_thin_data_are_identity():
    # Fail-soft: never fabricate a mapping from too little evidence.
    from src.analysis.ml_train import IsotonicCalibrator
    assert list(IsotonicCalibrator().transform([0.2, 0.8])) == [0.2, 0.8]
    thin = IsotonicCalibrator().fit([0.1, 0.9], [0, 1])          # < 20 rows
    assert list(thin.transform([0.2, 0.8])) == [0.2, 0.8]


def test_calibrate_helper_is_fail_soft(monkeypatch):
    from config.settings import settings
    from src.analysis.ml_train import IsotonicCalibrator
    monkeypatch.setattr(settings, "enable_ml_probability_calibration", True)
    assert ms._calibrate({}, 0.9) == 0.9                          # no calibrator on artifact
    assert ms._calibrate({"calibrator": None}, 0.9) == 0.9
    cal = IsotonicCalibrator().fit([i / 100 for i in range(100)],
                                   [0] * 50 + [1] * 50)
    art = {"calibrator": cal}
    monkeypatch.setattr(settings, "enable_ml_probability_calibration", False)
    assert ms._calibrate(art, 0.9) == 0.9                          # setting off → raw
    monkeypatch.setattr(settings, "enable_ml_probability_calibration", True)
    assert ms._calibrate(art, 0.9) != 0.9                          # on → mapped


def test_fit_calibrator_declines_on_thin_panel():
    # Too few OOF rows ⇒ no calibrator (caller then stores none and uses raw).
    import pandas as pd
    cal, diag = ms._fit_calibrator(pd.DataFrame(), 5, "rel", ms.STACKER_LIVE_FEATURES, 0.0)
    assert cal is None and diag == {}


# ── the aggregator arm flag ──────────────────────────────────────────────────

def test_ml_combine_arm_override_and_settings():
    import src.signals.aggregator as agg
    agg.set_ml_combine_arm(None)                               # no override -> settings (default off)
    assert agg.ml_combine_arm_active() is False
    agg.set_ml_combine_arm(True)
    assert agg.ml_combine_arm_active() is True
    agg.set_ml_combine_arm(False)
    assert agg.ml_combine_arm_active() is False
    agg.set_ml_combine_arm(None)                               # restore


# ── the min-hold exit gate ───────────────────────────────────────────────────

def _arm_trade(days_ago: int, ml_arm: bool = True) -> dict:
    return {"ticker": "AAA", "ml_arm": ml_arm,
            "entry_date": (date.today() - timedelta(days=days_ago)).isoformat()}


def test_arm_suppresses_time_exit_within_min_hold(monkeypatch):
    # The min-hold MECHANISM (default 0 / inert since it was measured harmful —
    # see tests/test_ml_exit.py::test_min_hold_is_off_by_default; set explicitly here).
    from config.settings import settings
    from src.performance import tracker
    monkeypatch.setattr(settings, "ml_arm_min_hold_days", 5)
    t = _arm_trade(days_ago=0)                             # 0 trading days held < 5
    assert tracker._arm_suppresses_exit(t, "horizon_expired") is True
    assert tracker._arm_suppresses_exit(t, "llm_confidence_loss") is True
    assert tracker._arm_suppresses_exit(t, "mechanical_exit") is True


def test_arm_never_suppresses_safety_exits(monkeypatch):
    from config.settings import settings
    from src.performance import tracker
    monkeypatch.setattr(settings, "ml_arm_min_hold_days", 5)
    t = _arm_trade(days_ago=0)
    for safe in ("llm_signal_flipped", "macro_regime_exit", "adverse_stop"):
        assert tracker._arm_suppresses_exit(t, safe) is False


def test_arm_does_not_touch_non_arm_trades_or_past_min_hold(monkeypatch):
    from config.settings import settings
    from src.performance import tracker
    monkeypatch.setattr(settings, "ml_arm_min_hold_days", 5)
    # non-arm trade: never suppressed
    assert tracker._arm_suppresses_exit(_arm_trade(0, ml_arm=False), "horizon_expired") is False
    # past the min-hold window (well over 5 trading days): normal exits resume
    assert tracker._arm_suppresses_exit(_arm_trade(21), "horizon_expired") is False
    # no reason: nothing to suppress
    assert tracker._arm_suppresses_exit(_arm_trade(0), None) is False


def test_new_trades_stamped_with_arm_flag():
    # set_ml_arm toggles the module flag that record_new_trades reads.
    from src.performance import tracker
    tracker.set_ml_arm(True)
    assert tracker._ML_ARM is True
    tracker.set_ml_arm(False)
    assert tracker._ML_ARM is False


def test_ml_arm_stamp_requires_actual_stacker_use():
    """The stacker-entry <=> ML-exit invariant (2026-08-11): the stamp — which is
    what routes ml_exit closes and the llm_confidence_loss suppression — requires
    BOTH the run's coin AND the stacker having actually driven THIS trade's side
    (combine_source is fail-soft per side)."""
    from src.performance import tracker
    st = tracker._ml_arm_stamp
    tracker.set_ml_arm(False)
    # coin off -> never stamped, whatever the source says
    assert st("BUY", "ml") is False and st("SELL", "ml") is False
    tracker.set_ml_arm(True)
    try:
        # full swap drove both sides
        assert st("BUY", "ml") is True and st("SELL", "ml") is True
        # partial swap: only the side the stacker actually decided
        assert st("BUY", "ml_buy") is True and st("SELL", "ml_buy") is False
        assert st("SELL", "ml_sell") is True and st("BUY", "ml_sell") is False
        # weighted / missing signal -> no proof of use -> no ML exit coupling
        assert st("BUY", "weighted") is False and st("SELL", "weighted") is False
        assert st("BUY", None) is False and st("SELL", "") is False
    finally:
        tracker.set_ml_arm(False)


# ── the arm INSIDE build_signals (integration) ───────────────────────────────
#
# Everything above pins the pieces in isolation. The conftest pins
# `enable_ml_combine=False` / `ml_combine_arm_share=0.0` suite-wide and until
# now nothing opted back in, so the arm — live on ~50% of production runs — was
# never executed inside `build_signals`. In particular the four-way
# `combine_source` mapping was untested, and that string is what decides which
# direction bands and which confidence divisor a row gets: the 2026-08-14
# repair hangs off it, and mislabelling a partial swap as a full one would apply
# the ML scale to a difference that is half weighted.

_ARM_FIXTURE = ["AAPL", "MSFT", "GLD"]


def _build_with_arm(monkeypatch, buy, sell):
    """build_signals with the stacker convictions forced (no artifact needed).
    `buy`/`sell` are the conviction values, or None to make that side fail soft."""
    from config.settings import settings
    import src.analysis.ml_stacker as st
    import src.signals.aggregator as agg

    monkeypatch.setattr(settings, "enable_ml_combine", True)
    monkeypatch.setattr(settings, "enable_rank_shaping", False)
    monkeypatch.setattr(st, "compute_buy_conviction", lambda scores: buy)
    monkeypatch.setattr(st, "compute_sell_conviction", lambda scores: sell)
    return {s.ticker: s for s in agg.build_signals(list(_ARM_FIXTURE), [])}


def test_full_swap_replaces_both_camps_and_is_labelled_ml(monkeypatch):
    sigs = _build_with_arm(monkeypatch, 0.05, 0.01)
    assert sigs
    for tk, s in sigs.items():
        assert s.combine_source == "ml", tk
        assert s.combined_buy_score == pytest.approx(0.05), tk
        assert s.combined_sell_score == pytest.approx(0.01), tk


def test_partial_swap_is_labelled_per_side(monkeypatch):
    """Fail-soft is PER SIDE: a missing sell model leaves the weighted sell camp
    in place, and the row must say so — stamping it "ml" would hand a
    half-weighted difference the ML scale, and would couple its exits to
    ml_exit on the strength of a side the stacker never decided."""
    buy_only = _build_with_arm(monkeypatch, 0.05, None)
    for tk, s in buy_only.items():
        assert s.combine_source == "ml_buy", tk
        assert s.combined_buy_score == pytest.approx(0.05), tk
        assert s.combined_sell_score != pytest.approx(0.05), tk   # still weighted

    sell_only = _build_with_arm(monkeypatch, None, 0.02)
    for tk, s in sell_only.items():
        assert s.combine_source == "ml_sell", tk
        assert s.combined_sell_score == pytest.approx(0.02), tk


def test_both_sides_failing_soft_stays_weighted(monkeypatch):
    """No artifact at all is the common case (fresh checkout, lightgbm absent):
    the arm coin may be ON and the combine must be untouched AND unlabelled."""
    sigs = _build_with_arm(monkeypatch, None, None)
    for tk, s in sigs.items():
        assert s.combine_source == "weighted", tk


def test_ml_rows_get_the_ml_confidence_divisor(monkeypatch):
    """The 2026-08-14 repair at its real call site. The stacker conviction lives
    on a ~5x smaller scale, so an ML row divided by the weighted 0.5 sat below
    Gate 1 essentially always (0.2% pass rate) and the arm could not act."""
    from config.settings import settings
    monkeypatch.setattr(settings, "enable_cross_sectional", False)
    sigs = _build_with_arm(monkeypatch, 0.05, 0.01)
    ml_scale = float(settings.ml_raw_confidence_scale)
    discriminating = 0
    for tk, s in sigs.items():
        assert s.raw_confidence == pytest.approx(
            min(1.0, abs(s.combined_score) / ml_scale), abs=1e-3), tk
        # ...and the weighted divisor would have given a DIFFERENT answer here,
        # so the assertion above is not satisfied by both scales at once.
        if abs(min(1.0, abs(s.combined_score) / 0.5) - s.raw_confidence) > 0.01:
            discriminating += 1
    assert discriminating, "fixture cannot tell the two divisors apart"


def test_partial_swap_keeps_the_weighted_divisor(monkeypatch):
    """Two scales in one difference -> the conservative (weighted) read. Pinned
    end to end because the resolver's `combine_source == "ml"` check and the
    aggregator's four-way label have to agree about what "full swap" means."""
    from config.settings import settings
    from src.signals.aggregator import _raw_confidence_scale
    monkeypatch.setattr(settings, "enable_cross_sectional", False)
    sigs = _build_with_arm(monkeypatch, 0.05, None)
    scale = _raw_confidence_scale("weighted")
    for tk, s in sigs.items():
        assert s.raw_confidence == pytest.approx(
            min(1.0, abs(s.combined_score) / scale), abs=1e-3), tk
