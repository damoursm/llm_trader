"""``compute_ml_score`` v2 (pivot_rank artifact) — serving-branch contract.

What must hold: the scorer dispatches on the ARTIFACT's config (not settings);
the v2 branch applies NO clean-trend/liquidity conditioning; the score is
``clip(2·pred, −1, 1)``; feature vectors are assembled in the artifact's own
order with leg features preferred from ``latest_leg_features`` and NaN for
whatever is missing; and a ticker with no leg state (<50 bars) is NO_DATA, not
a guess. All exercised through a real pickled LightGBMRankRegressor artifact so
unpickling in the production venv is part of what's tested.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

lgb = pytest.importorskip("lightgbm")

from src.analysis.ml_train import LightGBMRankRegressor
from src.signals import ml_model


@pytest.fixture()
def pivot_artifact(tmp_path, monkeypatch):
    """A tiny real v2 artifact on disk + a canned feature/leg serving path."""
    rng = np.random.default_rng(0)
    feats = ["f_a", "f_b", "leg_dir", "leg_age"]
    X = rng.normal(size=(400, len(feats)))
    y = (0.8 * X[:, 2] + 0.2 * rng.normal(size=400)).astype(float)  # leg_dir drives it
    model = LightGBMRankRegressor(n_estimators=20, min_child_samples=5,
                                  num_threads=1).fit(X, y)
    from src.analysis.pivot_target import pivot_basis
    cfg = dict(ml_model.TRAIN_CONFIG_PIVOT)
    cfg["pivot_basis"] = pivot_basis()        # a CURRENT-basis artifact
    art = {"model": model, "features": feats,
           "config": cfg,
           "trained_at": "2026-08-08T00:00:00+00:00", "n_train": 400,
           "train_max_date": "2026-08-07"}
    path = tmp_path / "ml_ohlcv_model.pkl"
    import pickle
    with open(path, "wb") as fh:
        pickle.dump(art, fh)
    monkeypatch.setattr(ml_model, "_MODEL_PATH", path)
    ml_model.reset_caches()

    # canned serving inputs — the frame carries f_a/f_b, the leg call leg_dir/leg_age
    frame = pd.DataFrame({"f_a": [0.1, 0.2], "f_b": [1.0, -0.5]})
    monkeypatch.setattr("src.analysis.ml_dataset.ticker_feature_frame",
                        lambda tk: frame)
    monkeypatch.setattr("src.data.cache._ohlcv_path",
                        lambda tk, interval="1d": path)   # any stat()-able file
    return dict(art=art, model=model, monkeypatch=monkeypatch, path=path)


def _predict_net(model, vec):
    p = float(model.predict(np.asarray([vec], dtype=float))[0])
    return round(max(-1.0, min(1.0, 2.0 * p)), 4)


def test_v2_scores_without_conditioning_and_scales_by_two(pivot_artifact, monkeypatch):
    leg = {"leg_dir": 1.0, "leg_age": 3.0}
    monkeypatch.setattr("src.analysis.pivot_target.latest_leg_features",
                        lambda tk: dict(leg))
    net, label = ml_model.compute_ml_score("TEST")
    assert label == "OK"                       # no NO_VIEW gate on the v2 path
    expected = _predict_net(pivot_artifact["model"], [0.2, -0.5, 1.0, 3.0])
    assert net == pytest.approx(expected)
    assert -1.0 <= net <= 1.0


def test_v2_leg_features_win_over_frame_and_missing_go_nan(pivot_artifact, monkeypatch):
    # leg dict misses leg_age -> that slot must be NaN (lgb handles it), not 0.
    monkeypatch.setattr("src.analysis.pivot_target.latest_leg_features",
                        lambda tk: {"leg_dir": -1.0})
    net, label = ml_model.compute_ml_score("TEST2")
    assert label == "OK"
    expected = _predict_net(pivot_artifact["model"], [0.2, -0.5, -1.0, np.nan])
    assert net == pytest.approx(expected)


def test_v2_no_leg_state_is_no_data(pivot_artifact, monkeypatch):
    monkeypatch.setattr("src.analysis.pivot_target.latest_leg_features",
                        lambda tk: None)
    net, label = ml_model.compute_ml_score("TEST3")
    assert (net, label) == (0.0, "NO_DATA")


def test_v2_direction_tracks_the_planted_signal(pivot_artifact, monkeypatch):
    """leg_dir drove the synthetic label, so flipping it must flip the score —
    the assembled vector really reaches the booster in the right order."""
    monkeypatch.setattr("src.analysis.pivot_target.latest_leg_features",
                        lambda tk: {"leg_dir": 1.0, "leg_age": 3.0})
    up, _ = ml_model.compute_ml_score("UP")
    monkeypatch.setattr("src.analysis.pivot_target.latest_leg_features",
                        lambda tk: {"leg_dir": -1.0, "leg_age": 3.0})
    dn, _ = ml_model.compute_ml_score("DOWN")
    assert up > 0 > dn


def test_v2_stale_pivot_basis_abstains(pivot_artifact, monkeypatch):
    """An artifact trained on the RETIRED close basis (no/old pivot_basis in
    its config) must ABSTAIN — its leg features no longer mean what it learned.
    The 2026-08-12 H/L refactor guard: degraded, never wrong."""
    import pickle

    from src.signals import ml_model

    art = dict(pivot_artifact["art"])
    cfg = dict(art["config"])
    cfg.pop("pivot_basis", None)              # close-era artifact
    art["config"] = cfg
    with open(pivot_artifact["path"], "wb") as fh:
        pickle.dump(art, fh)
    ml_model._ART_CACHE.update(mtime=None, art=None)
    ml_model._SCORE_CACHE.clear()
    ml_model._BASIS_WARNED = False
    score, status = ml_model.compute_ml_score("AAPL")
    assert score == 0.0
    assert status == "BASIS_STALE"
