"""Self-calibrating ML confidence divisor (`src/signals/ml_scale.py`, 2026-08-18).

What these guard: the divisor is a SATURATION control, and the stackers retrain
weekly onto a new conviction scale, so the thing that must stay fixed is the
saturation SHARE, not the number. The decisive test is
`test_solved_divisor_reproduces_the_target_share` — it re-derives the divisor
from a synthetic cross-section and checks that applying it actually clips the
intended fraction. The rest pin the fail-soft paths, because every one of them
silently re-scales the live book if it breaks.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from config.settings import Settings, settings
from src.signals import ml_scale


def _frame(ml_abs, wt_saturated_frac=0.09, n_wt=20000, seed=0):
    """A signals-shaped frame: ml rows carrying |combined_score| = ml_abs, plus
    weighted rows whose raw_confidence saturates at the given rate."""
    rng = np.random.default_rng(seed)
    ml = pd.DataFrame({
        "combine_source": ["ml"] * len(ml_abs),
        "combined_score": np.asarray(ml_abs, dtype=float) * rng.choice([-1.0, 1.0], len(ml_abs)),
        # raw_confidence on ml rows is deliberately NOT what the solve reads
        "raw_confidence": rng.random(len(ml_abs)),
    })
    n_sat = int(round(n_wt * wt_saturated_frac))
    wt = pd.DataFrame({
        "combine_source": ["weighted"] * n_wt,
        "combined_score": rng.random(n_wt),
        "raw_confidence": np.r_[np.ones(n_sat), rng.random(n_wt - n_sat) * 0.9],
    })
    return pd.concat([ml, wt], ignore_index=True)


@pytest.fixture(autouse=True)
def _fresh(monkeypatch):
    ml_scale.reset_cache()
    monkeypatch.setattr(settings, "enable_ml_scale_calibration", True)
    monkeypatch.setattr(settings, "ml_raw_confidence_scale", 0.12)
    monkeypatch.setattr(settings, "ml_scale_min_rows", 2000)
    monkeypatch.setattr(settings, "ml_scale_prior_n", 4000)
    monkeypatch.setattr(settings, "ml_saturation_target", 0.09)
    yield
    ml_scale.reset_cache()


def _patch_db(monkeypatch, df):
    import src.db.repo as repo
    monkeypatch.setattr(repo, "fetch_df", lambda *a, **k: df)


# ── the property the whole design rests on ───────────────────────────────────

def test_solved_divisor_reproduces_the_target_share(monkeypatch):
    """Solve on a known distribution, then APPLY the result: the fraction of ml
    rows that saturate must come out at the target. This is the contract — the
    share is pinned, the divisor is whatever delivers it."""
    rng = np.random.default_rng(7)
    ml_abs = np.abs(rng.normal(0.06, 0.05, 30000))
    # weighted saturates at 15% -> target 15% (its 20k rows swamp the 0.09 prior)
    _patch_db(monkeypatch, _frame(ml_abs, wt_saturated_frac=0.15))
    m = ml_scale.measure()
    assert m["target_share"] == pytest.approx(0.15, abs=0.012)
    realized = float((ml_abs >= m["solved_scale"]).mean())
    assert realized == pytest.approx(m["target_share"], abs=0.01), (
        f"solved divisor {m['solved_scale']:.4f} clips {realized:.3f}, "
        f"target {m['target_share']:.3f}")


def test_quantile_is_not_censored_by_the_divisor_in_force(monkeypatch):
    """`combined_score` is persisted UNCLIPPED, so rows far above the current
    divisor still move the answer. If the solve ever read `raw_confidence`
    instead, every such row would read 1.0 and the estimate would be pinned at
    whatever divisor produced the history — a calibration that can only ever
    confirm itself."""
    base = np.full(20000, 0.05)
    hot = np.r_[base, np.full(10000, 5.0)]          # a third far above any divisor
    cold = np.r_[base, np.full(10000, 0.051)]
    _patch_db(monkeypatch, _frame(hot))
    hot_scale = ml_scale.measure()["solved_scale"]
    _patch_db(monkeypatch, _frame(cold))
    cold_scale = ml_scale.measure()["solved_scale"]
    assert hot_scale > cold_scale * 5, (
        "the extreme tail did not move the solved divisor — the solve is reading "
        "a clipped quantity")


def test_tracks_a_retrain_that_rescales_conviction(monkeypatch):
    """The reason this exists: a retrain that doubles the stacker's conviction
    scale must roughly double the divisor, holding saturation constant."""
    rng = np.random.default_rng(11)
    before = np.abs(rng.normal(0.06, 0.04, 30000))
    _patch_db(monkeypatch, _frame(before))
    s1 = ml_scale.measure()["solved_scale"]
    _patch_db(monkeypatch, _frame(before * 2.0))
    s2 = ml_scale.measure()["solved_scale"]
    assert s2 == pytest.approx(s1 * 2.0, rel=0.05)


# ── fail-soft paths (each one silently re-scales the book if it breaks) ──────

def test_disabled_returns_the_static_setting(monkeypatch):
    monkeypatch.setattr(settings, "enable_ml_scale_calibration", False)
    monkeypatch.setattr(settings, "ml_raw_confidence_scale", 0.0777)

    def _boom(*a, **k):
        pytest.fail("disabled calibration must not touch the database")
    import src.db.repo as repo
    monkeypatch.setattr(repo, "fetch_df", _boom)
    assert ml_scale.calibrate_ml_confidence_scale() == pytest.approx(0.0777)


def test_db_failure_falls_back_to_static(monkeypatch):
    import src.db.repo as repo
    monkeypatch.setattr(repo, "fetch_df",
                        lambda *a, **k: (_ for _ in ()).throw(RuntimeError("db down")))
    assert ml_scale.calibrate_ml_confidence_scale() == pytest.approx(0.12)


def test_thin_evidence_holds_the_static_value(monkeypatch):
    rng = np.random.default_rng(3)
    _patch_db(monkeypatch, _frame(np.abs(rng.normal(0.06, 0.04, 50))))
    assert ml_scale.calibrate_ml_confidence_scale() == pytest.approx(0.12)


def test_no_ml_rows_holds_the_static_value(monkeypatch):
    _patch_db(monkeypatch, _frame(np.array([]), n_wt=5000))
    assert ml_scale.calibrate_ml_confidence_scale() == pytest.approx(0.12)


def test_empty_panel_holds_the_static_value(monkeypatch):
    _patch_db(monkeypatch, pd.DataFrame())
    assert ml_scale.calibrate_ml_confidence_scale() == pytest.approx(0.12)


# ── guardrails ───────────────────────────────────────────────────────────────

def test_runaway_solution_is_clamped_to_the_band(monkeypatch):
    """A degenerate cross-section (every |combined| enormous) must not hand the
    live book a 100x divisor. The band is a module constant, not a setting."""
    _patch_db(monkeypatch, _frame(np.full(30000, 50.0)))
    v = ml_scale.calibrate_ml_confidence_scale()
    assert v == pytest.approx(0.12 * ml_scale._BAND_HI)
    ml_scale.reset_cache()
    _patch_db(monkeypatch, _frame(np.full(30000, 1e-6)))
    v = ml_scale.calibrate_ml_confidence_scale()
    assert v == pytest.approx(0.12 * ml_scale._BAND_LO)


def test_target_share_is_clamped(monkeypatch):
    """A weighted arm that saturated ~everything must not drag ML with it."""
    rng = np.random.default_rng(5)
    _patch_db(monkeypatch, _frame(np.abs(rng.normal(0.06, 0.04, 30000)),
                                  wt_saturated_frac=0.99))
    assert ml_scale.measure()["target_share"] <= ml_scale._TARGET_HI + 1e-9


def test_shrinkage_pulls_toward_the_static_prior(monkeypatch):
    """With evidence comparable to the prior weight the result sits BETWEEN the
    solved value and the static one, not on either."""
    monkeypatch.setattr(settings, "ml_scale_prior_n", 30000)
    rng = np.random.default_rng(9)
    _patch_db(monkeypatch, _frame(np.abs(rng.normal(0.30, 0.05, 30000))))
    solved = ml_scale.measure()["solved_scale"]
    v = ml_scale.calibrate_ml_confidence_scale()
    assert 0.12 < v < solved, f"expected a blend of 0.12 and {solved}, got {v}"


# ── wiring ───────────────────────────────────────────────────────────────────

def test_resolver_uses_the_calibrated_value_for_ml_only(monkeypatch):
    import src.signals.aggregator as agg
    monkeypatch.setattr(ml_scale, "calibrate_ml_confidence_scale", lambda: 0.4242)
    assert agg._raw_confidence_scale("ml") == pytest.approx(0.4242)
    # every other combine keeps its own divisor — the ML calibration must not
    # leak into the weighted book
    monkeypatch.setattr(settings, "method_score_basis", "rank")
    assert agg._raw_confidence_scale("weighted") == pytest.approx(
        settings.rank_raw_confidence_scale)
    monkeypatch.setattr(settings, "method_score_basis", "absolute")
    assert agg._raw_confidence_scale("weighted") == pytest.approx(0.5)


def test_cache_is_registered_with_the_asof_flush():
    """A walk-forward step must not reuse a divisor solved on later rows."""
    import inspect

    from src.analysis import asof
    src = inspect.getsource(asof.reset_all_calibration_caches)
    assert "src.signals.ml_scale" in src
    assert hasattr(ml_scale, "reset_cache")


def test_calibration_is_reported_to_the_registry(monkeypatch):
    from src.performance.calibration import get_calibrations, reset_calibrations
    reset_calibrations()
    rng = np.random.default_rng(13)
    _patch_db(monkeypatch, _frame(np.abs(rng.normal(0.06, 0.04, 30000))))
    ml_scale.calibrate_ml_confidence_scale()
    names = {c["name"] for c in get_calibrations()}
    assert "ml_confidence_scale" in names
    row = [c for c in get_calibrations() if c["name"] == "ml_confidence_scale"][0]
    assert row["prior"] == pytest.approx(0.12)
    assert row["n_evidence"] == 30000


def test_defaults_are_on_and_documented():
    """The static constant stays the documented prior even though the live
    value is now solved — a fresh environment must fail soft to it."""
    assert Settings.model_fields["enable_ml_scale_calibration"].default is True
    assert Settings.model_fields["ml_raw_confidence_scale"].default == pytest.approx(0.12)
