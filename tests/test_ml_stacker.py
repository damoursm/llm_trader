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

def test_live_features_split_into_signed_scores_and_unsigned_context():
    # 29 SIGNED scores (the 22 weighted methods + tape_score + the six news
    # additions of 2026-09-07) and 23 UNSIGNED context features (the evidence
    # counts, the pre-combine market state, and the 18-class catalyst one-hot).
    assert len(ms.STACKER_LIVE_FEATURES) == 54   # +news_quiet 2026-09-11
    assert ms.STACKER_LIVE_FEATURES == ms.STACKER_SIGNED_FEATURES + ms.STACKER_CONTEXT_FEATURES
    assert len(ms.STACKER_SIGNED_FEATURES) == 31   # +news_quiet 2026-09-11
    assert len(ms.STACKER_CONTEXT_FEATURES) == 23
    for f in ("tape_score", "ml_ohlcv", "news_shock", "news_bear_fresh",
              "catalyst_tilt", "news_raw_score", "news_unpriced", "news_unpriced_all"):
        assert f in ms.STACKER_SIGNED_FEATURES
    # the method columns are still a subset of the panel's method set; the four
    # carve-outs are panel columns that are not METHOD scores.
    carve = {"tape_score", "news_raw_score"}
    methods_only = [f for f in ms.STACKER_SIGNED_FEATURES if f not in carve]
    assert set(methods_only).issubset(set(SIGNAL_BASE_METHOD_COLUMNS))


def test_the_combine_and_everything_derived_from_it_stay_out():
    """The circularity guard, widened 2026-09-07. `confidence` and
    `raw_confidence` are functions of the combine this model PRODUCES, and the
    other confidence factors are computed downstream of it — at serving time
    they do not exist yet when the stacker runs, so training on them would be a
    train/serve skew on top of the circularity. What goes in instead is the
    market state UNDERNEATH them, which is available before the combine."""
    for banned in ("combined_score", "combined_buy_score", "combined_sell_score",
                   "confidence", "raw_confidence", "coherence_factor",
                   "volume_factor", "family_conf_factor", "tape_conf_factor",
                   "sector_conf_factor", "movement_factor"):
        assert banned not in ms.STACKER_LIVE_FEATURES
    for kept in ("atr_pct", "bb_width_pct", "vol_ratio"):
        assert kept in ms.STACKER_CONTEXT_FEATURES


def test_catalyst_onehot_covers_the_fixed_taxonomy_and_never_invents_a_class():
    from src.analysis.sentiment import NEWS_CATALYST_TYPES
    row = ms.catalyst_onehot("earnings")
    assert set(row) == set(ms.CATALYST_ONEHOT_FEATURES)
    assert len(row) == len(NEWS_CATALYST_TYPES) == 18
    assert row["cat_earnings"] == 1.0 and sum(row.values()) == 1.0
    # an unknown or missing class is ALL-ZERO, not a fabricated `none`: `none`
    # is a real verdict meaning "read it, nothing there".
    assert sum(ms.catalyst_onehot(None).values()) == 0.0
    assert sum(ms.catalyst_onehot("alien_invasion").values()) == 0.0
    assert ms.catalyst_onehot(" MA_Deal ")["cat_ma_deal"] == 1.0


def test_exit_model_takes_the_signed_features_only():
    """Every exit feature is oriented by the position's direction, and orienting
    a count, a width or a one-hot multiplies a magnitude by a meaningless
    sign."""
    import src.analysis.ml_exit_dataset as me
    assert me.EXIT_METHODS == list(ms.STACKER_SIGNED_FEATURES)
    for unsigned in ("news_article_count", "news_recency_mass", "atr_pct",
                     "cat_earnings"):
        assert unsigned not in me.EXIT_METHODS
        assert f"ex_{unsigned}" not in me.EXIT_FEATURE_COLUMNS


def test_model_factory_dispatches_on_the_setting(monkeypatch):
    # "logistic" (the 2026-08-22 measured default) -> SoftmaxLogistic;
    # "gbm" reverts to the small-data LightGBM classifier; junk falls back to
    # logistic rather than erroring (operator knob, fail-soft).
    from config.settings import settings
    from src.analysis.ml_train import SoftmaxLogistic
    monkeypatch.setattr(settings, "stacker_model_class", "logistic")
    assert isinstance(ms._stacker_model_factory(), SoftmaxLogistic)
    monkeypatch.setattr(settings, "stacker_model_class", "GBM")
    assert type(ms._stacker_model_factory()).__name__ == "LightGBMModel"
    monkeypatch.setattr(settings, "stacker_model_class", "typo")
    assert isinstance(ms._stacker_model_factory(), SoftmaxLogistic)


def test_softmax_logistic_serves_through_the_conviction_path(tmp_path, monkeypatch):
    # End-to-end on the NEW model class: train a tiny SoftmaxLogistic artifact
    # through train-time plumbing stand-ins and serve it via
    # compute_buy_conviction — bull_bear + calibrate + centering all apply, and
    # a missing tape_score key imputes (train median) instead of erroring.
    import pickle

    import numpy as np

    from src.analysis.ml_train import SoftmaxLogistic
    rng = np.random.default_rng(0)
    X = rng.normal(size=(400, len(ms.STACKER_LIVE_FEATURES)))
    y = (X[:, 0] + 0.5 * rng.normal(size=400) > 0).astype(int) * 2   # classes {0, 2}
    model = SoftmaxLogistic().fit(X, y)
    art = {"model": model, "features": list(ms.STACKER_LIVE_FEATURES),
           "config": {"horizon": 5, "basis": "rank_pv", "deadband": 0.0},
           "model_class": "logistic", "calibrator": None, "calibration": {},
           "trained_at": "2026-08-22T00:00:00+00:00", "n_train": 400,
           "train_max_date": "2026-08-21"}
    p = tmp_path / "ml_buy_model.pkl"
    with open(p, "wb") as fh:
        pickle.dump(art, fh)
    monkeypatch.setattr(ms, "_BUY_MODEL_PATH", p)
    ms.reset_buy_caches()
    try:
        scores = {f: 0.4 for f in ms.STACKER_LIVE_FEATURES}
        conv = ms.compute_buy_conviction(scores)
        assert conv is not None and 0.0 <= conv <= 1.0
        # tape_score absent from the dict -> NaN -> imputed, never an error
        scores.pop("tape_score")
        conv2 = ms.compute_buy_conviction(scores)
        assert conv2 is not None and 0.0 <= conv2 <= 1.0
    finally:
        ms.reset_buy_caches()


def test_live_features_match_the_aggregator_method_map():
    # Drift guard: the live feature set must equal the aggregator's weighted
    # method_score_map keys (the scores actually available at the combine point).
    import inspect
    import src.signals.aggregator as agg
    src = inspect.getsource(agg._score_ticker) if hasattr(agg, "_score_ticker") else inspect.getsource(agg)
    # Every live feature must be supplied at serving. Most appear as literal
    # dict keys; the catalyst block is spread in from `catalyst_onehot`, whose
    # key set is pinned against the feature list above — the two together are
    # the parity guarantee.
    for m in ms.STACKER_LIVE_FEATURES:
        if m in ms.CATALYST_ONEHOT_FEATURES:
            continue
        assert f'"{m}":' in src, f"{m} not found as a method_score_map key in aggregator"
    assert "**catalyst_onehot(" in src, "the catalyst one-hot is not spread into the serving dict"


def test_serving_supplies_missing_news_inputs_as_nan_not_zero():
    """A 0.0 is a REAL verdict here ("read it, nothing there"), so a run that
    captured no raw verdict must serve NaN and let the model impute — the panel
    column is NULL on exactly those rows."""
    import math

    import src.signals.aggregator as agg
    assert math.isnan(agg._news_raw_feature(None))
    assert math.isnan(agg._news_raw_feature({}))
    assert math.isnan(agg._news_raw_feature({"raw_score": None}))
    assert agg._news_raw_feature({"raw_score": -0.42}) == -0.42
    assert math.isnan(agg._f_or_nan(None)) and math.isnan(agg._f_or_nan(float("nan")))
    assert agg._f_or_nan("1.5") == 1.5


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
    # The doubles must mirror the real signature, `ranked=` included — the
    # serving path passes the run's centered-rank news map (2026-09-07).
    monkeypatch.setattr(st, "compute_buy_conviction",
                        lambda scores, ranked=None: buy)
    monkeypatch.setattr(st, "compute_sell_conviction",
                        lambda scores, ranked=None: sell)
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


# ── news context features on the panel (2026-09-07) ─────────────────────────

def _panel_frame():
    import pandas as pd
    return pd.DataFrame({
        "ticker": ["AAA", "BBB"],
        "signal_date": ["2026-08-20", "2026-09-06"],
        "news_catalyst": ["earnings", "guidance"],
        "news_raw_score": [0.4, -0.3],
        "news_recency_mass": [2.0, 3.0],
        "news_article_count": [4, 6],
    })


def test_news_epoch_mask_hides_the_pre_epoch_news_columns(monkeypatch):
    """`build_panel` masks METHOD scores; the raw verdict, the catalyst and the
    evidence counts are not method columns, so the same scorer-epoch rule is
    applied here — otherwise the training set mixes two news eras."""
    from datetime import date
    import src.analysis.ml_stacker as m
    monkeypatch.setattr(m, "_news_epoch_day", lambda: date(2026, 9, 5).isoformat())
    out = m.add_news_features(_panel_frame())
    pre, post = out.iloc[0], out.iloc[1]
    assert pre["news_raw_score"] != pre["news_raw_score"]          # NaN
    assert pre["news_article_count"] != pre["news_article_count"]
    assert pre["cat_earnings"] != pre["cat_earnings"]
    assert post["news_raw_score"] == -0.3                          # post-epoch kept
    assert post["cat_guidance"] == 1.0


def test_news_epoch_mask_can_be_switched_off_for_a_measurement_arm(monkeypatch):
    from datetime import date
    from config.settings import settings
    import src.analysis.ml_stacker as m
    monkeypatch.setattr(m, "_news_epoch_day", lambda: date(2026, 9, 5).isoformat())
    monkeypatch.setattr(settings, "enable_stacker_news_epoch_mask", False, raising=False)
    out = m.add_news_features(_panel_frame())
    assert out.iloc[0]["news_raw_score"] == 0.4
    assert out.iloc[0]["cat_earnings"] == 1.0


def test_catalyst_backfill_fills_only_untyped_rows(monkeypatch, tmp_path):
    """The backfill classifier is a different provenance from the live scorer,
    so it fills a NULL and never overwrites a label the scorer emitted."""
    import pandas as pd
    from config.settings import settings
    import src.analysis.ml_stacker as m
    monkeypatch.setattr(settings, "db_path", str(tmp_path / "t.db"), raising=False)
    monkeypatch.setattr(m, "_news_epoch_day", lambda: None)
    from src.db import repo
    repo.insert_news_event_backfill([
        {"ticker": "AAA", "signal_date": "2026-08-20", "catalyst": "ma_deal",
         "headline_count": 2, "top_headline": "h", "classifier_version": "bf1",
         "classified_at": "2026-08-21T00:00:00+00:00"},
        {"ticker": "BBB", "signal_date": "2026-09-06", "catalyst": "distress",
         "headline_count": 2, "top_headline": "h", "classifier_version": "bf1",
         "classified_at": "2026-09-06T00:00:00+00:00"},
    ])
    df = _panel_frame()
    df.loc[0, "news_catalyst"] = None                       # never typed live
    out = m.add_news_features(df)
    assert out.iloc[0]["cat_ma_deal"] == 1.0                # filled from backfill
    assert out.iloc[1]["cat_guidance"] == 1.0               # live label kept
    assert out.iloc[1]["cat_distress"] == 0.0


# ── news basis: rank, and the artifact decides (2026-09-07) ─────────────────

def test_centered_rank_conventions():
    """Zeros abstain, NaN survives, ties share a rank, a thin cross-section is
    neutral rather than an invented extreme."""
    import math
    r = ms.centered_rank([0.1, 0.5, 0.3, 0.0, None, 0.5])
    assert r[0] == -1.0 and r[2] == pytest.approx(-1 / 3)
    assert r[1] == r[5] == pytest.approx(2 / 3)          # tie -> shared rank
    assert r[3] == 0.0                                   # zero abstains
    assert math.isnan(r[4])                              # missing stays missing
    assert ms.centered_rank([0.4]) == [0.0]              # no cross-section
    assert ms.centered_rank([]) == []
    # invariant to any positive rescale — the whole point (engines differ 2.7x)
    a = ms.centered_rank([0.1, 0.2, 0.4])
    b = ms.centered_rank([0.27, 0.54, 1.08])
    assert a == b


def test_training_and_serving_share_one_transform():
    """A rank the two sides compute differently is worse than no rank: the
    dataset builder and the aggregator must call the SAME function."""
    import inspect

    import src.signals.aggregator as agg
    assert "centered_rank_pooled(" in inspect.getsource(ms.apply_news_basis)
    assert "centered_rank_pooled(" in inspect.getsource(ms.rank_news_features)
    assert "rank_news_features" in inspect.getsource(agg.build_signals)
    # and both sides must rank over the Gate-4 pool, not the whole universe
    assert "tradeable=_tradeable" in inspect.getsource(agg.build_signals)
    assert "_tradeable_pools(" in inspect.getsource(ms.apply_news_basis)


def test_serving_dispatches_on_the_artifacts_own_stamp():
    """An artifact trained on ABSOLUTE values must keep receiving them even
    while the setting says rank — otherwise flipping the setting silently
    changes what a frozen model is fed."""
    absolute = {"config": {}}
    ranked_art = {"config": {"news_basis": "rank"}}
    assert ms.news_basis_of(absolute) == "absolute"
    assert ms.news_basis_of(ranked_art) == "rank"
    assert ms.news_basis_of(None) == "absolute"
    raw = {"news": 0.9, "tech": 0.2}
    rnk = {"news": -1.0}
    # absolute artifact -> the raw value, even when a rank map is supplied
    assert ms._feature_value("news", raw, rnk, absolute) == 0.9
    # rank artifact -> the ranked value
    assert ms._feature_value("news", raw, rnk, ranked_art) == -1.0
    # a non-news feature is never re-routed
    assert ms._feature_value("tech", raw, {"tech": -1.0}, ranked_art) == 0.2
    # no rank map (fail-soft) -> raw
    assert ms._feature_value("news", raw, None, ranked_art) == 0.9


def test_the_live_frozen_artifact_is_not_silently_rebased():
    """The shipped stackers were trained 2026-08-24 on absolute news values.
    Until they are retrained, serving must feed them absolute values."""
    art = ms._load_buy_artifact()
    if art is None:
        pytest.skip("no artifact on this machine")
    if str((art.get("config") or {}).get("news_basis") or "") != "rank":
        assert ms.news_basis_of(art) == "absolute"


def test_apply_news_basis_ranks_within_the_day_only(monkeypatch):
    """Ranking across days would mix regimes and could not be reproduced at
    serve time, where only the current run's cross-section exists."""
    import pandas as pd
    from config.settings import settings
    monkeypatch.setattr(settings, "stacker_news_basis", "rank", raising=False)
    df = pd.DataFrame({"signal_date": ["d1"] * 3 + ["d2"] * 3,
                       "ticker": list("ABCABC"),
                       "news": [0.9, 0.1, 0.0, 0.02, 0.01, 0.03]})
    out = ms.apply_news_basis(df)
    assert list(out.news) == [1.0, -1.0, 0.0, 0.0, -1.0, 1.0]
    monkeypatch.setattr(settings, "stacker_news_basis", "absolute", raising=False)
    assert list(ms.apply_news_basis(df).news) == [0.9, 0.1, 0.0, 0.02, 0.01, 0.03]


def test_pooled_news_rank_matches_the_combines_own_transform():
    """The news features must be ranked EXACTLY as every other method is
    (2026-09-09 directive) — same tradeable pool, same tie-averaging, same
    observe-only interpolation. A second implementation of the house rank is
    how the two silently drift, so this pins `centered_rank_pooled` against
    `aggregator._rank_transform_run` itself.

    Shaping is deliberately switched OFF on the aggregator side (`shapes={}`):
    the shaped curves are fitted on the same forward returns the stacker trains
    against, so they belong in the combine and not inside a model feature — that
    is the ONE intended difference between the two.
    """
    import random

    from src.signals.aggregator import _rank_transform_run
    random.seed(11)
    tickers = [f"T{i}" for i in range(40)]
    vals = {t: random.choice([0.0, round(random.uniform(-1, 1), 3)]) for t in tickers}
    for t in tickers[:6]:
        vals[t] = 0.25                                  # a real tie group
    pool = set(tickers[:22])                            # the rest are observe-only
    combine, _ = _rank_transform_run({t: {"news": (True, vals[t])} for t in tickers},
                                     tradeable=pool, shapes={})
    mine = ms.centered_rank_pooled([(t, vals[t]) for t in tickers], pool)
    for t in tickers:
        if vals[t] == 0.0:
            assert mine[t] == 0.0                       # zeros abstain on both sides
            continue
        assert round(mine[t], 4) == combine[t]["news"][1]
    # observe-only names land INSIDE the tradeable range, never past its ends
    assert max(abs(mine[t]) for t in tickers if t not in pool and vals[t]) <= 1.0
    # tradeable=None is the documented fail-soft: the plain full-universe rank
    plain = ms.centered_rank([vals[t] for t in tickers])
    none_pool = ms.centered_rank_pooled([(t, vals[t]) for t in tickers], None)
    assert [round(none_pool[t], 9) for t in tickers] == [round(v, 9) for v in plain]


def test_pooled_news_rank_fails_soft_on_a_thin_pool():
    """A pool of one cannot order anything; the transform must return a neutral
    0.0 rather than a fabricated extreme, and must never raise."""
    pairs = [("A", 0.9), ("B", 0.1), ("C", -0.4)]
    assert set(ms.centered_rank_pooled(pairs, {"A"}).values()) == {0.0}
    assert set(ms.centered_rank_pooled(pairs, set()).values()) == {0.0}


def test_training_pool_is_point_in_time(monkeypatch):
    """The training-side pool judges liquidity on the bars visible ON the signal
    date. A name that only became liquid LATER must not be in an earlier day's
    pool — a pool built from future liquidity leaks, and the live path has no
    such option."""
    import pandas as pd
    from config.settings import settings

    monkeypatch.setattr(settings, "trade_min_price", 5.0, raising=False)
    monkeypatch.setattr(settings, "trade_min_dollar_volume", 5e6, raising=False)

    def fake_ohlcv(tk, *a, **k):
        idx = pd.to_datetime(["2026-01-01", "2026-01-02", "2026-01-03"])
        vol = [1e3, 1e3, 1e9] if tk == "LATE" else [1e9, 1e9, 1e9]
        return pd.DataFrame({"Close": [10.0] * 3, "Volume": vol}, index=idx)

    monkeypatch.setattr("src.data.cache.load_ohlcv", fake_ohlcv)
    rows = [{"signal_date": d, "ticker": t, "price": 10.0}
            for d in ("2026-01-02", "2026-01-03")
            for t in ["LATE"] + [f"OK{i}" for i in range(40)]]
    pools = ms._tradeable_pools(pd.DataFrame(rows))
    assert "LATE" not in (pools["2026-01-02"] or set())     # not yet liquid
    assert "LATE" in (pools["2026-01-03"] or set())         # liquid that day
    assert "OK0" in (pools["2026-01-02"] or set())
