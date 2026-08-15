"""ml_ohlcv SERVING path (`src/signals/ml_model.py`).

Training is exercised offline; what runs on every tick is `compute_ml_score`,
and since 2026-08-11 the method is WEIGHTED (0.12) in the combine — so a wrong
answer here moves real trades, and a silent 0.0 removes a weighted method
without anything reporting it.

The behaviours worth pinning are all abstentions, because that is how this
method fails:

* **BASIS_STALE** — the 2026-08-12 close→H/L pivot change. An artifact trained
  on the old basis would be handed leg features it never saw. It must abstain
  (degraded) rather than predict (wrong), and the label has to be distinct from
  the other zeros or the "method reads 0.0" diagnosis is unfalsifiable;
* **serving dispatches on the ARTIFACT's own config**, not on the setting — the
  documented rule that stops `ml_ohlcv_target` from silently reinterpreting a
  v1 pickle as v2;
* the artifact memo is keyed on the pickle's MTIME so a fresh EOD train is
  picked up without a restart.

Nothing here needs lightgbm: the artifact is a plain dict with a stub model.
"""

from __future__ import annotations

import pickle
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd
import pytest

from src.signals import ml_model as mm


class _StubRank:
    """Stands in for LightGBMRankRegressor — returns a fixed centred rank."""
    def __init__(self, pred=0.25):
        self.pred = pred
        self.seen = None

    def predict(self, X):
        self.seen = X
        return np.array([self.pred])


class _StubClassifier:
    def __init__(self, bull=0.7, bear=0.2):
        self._b, self._s = bull, bear

    def bull_bear(self, X):
        return np.array([self._b]), np.array([self._s])


def _pivot_art(model=None, basis="hl1", features=("f1", "f2")):
    return {"config": {"target": "pivot_rank", "pivot_basis": basis},
            "features": list(features), "model": model or _StubRank(),
            "trained_at": datetime.now(timezone.utc).isoformat()}


def _classic_art(model=None, features=("eff_ratio", "dollar_vol_log")):
    return {"config": dict(mm.TRAIN_CONFIG), "features": list(features),
            "model": model or _StubClassifier(),
            "trained_at": datetime.now(timezone.utc).isoformat()}


@pytest.fixture(autouse=True)
def _isolated(tmp_path, monkeypatch):
    """Point the artifact at tmp_path, drop both memos, and make the OHLCV
    mtime probe deterministic."""
    monkeypatch.setattr(mm, "_MODEL_PATH", tmp_path / "ml_ohlcv_model.pkl")
    monkeypatch.setattr(mm, "_BASIS_WARNED", False)
    mm.reset_caches()
    import src.data.cache as cache
    probe = tmp_path / "AAA.json"
    probe.write_text("{}", encoding="utf-8")
    monkeypatch.setattr(cache, "_ohlcv_path", lambda tk, interval="1d": probe)
    yield
    mm.reset_caches()


def _write_art(art):
    mm._MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(mm._MODEL_PATH, "wb") as fh:
        pickle.dump(art, fh)
    mm.reset_caches()


def _serve_art(monkeypatch, art):
    """Install the artifact WITHOUT round-tripping it through pickle.

    `_write_art` is the right tool for the load/mtime paths, but pickling hands
    the module a COPY of the stub model — so a test that inspects what the model
    was CALLED with must keep object identity."""
    monkeypatch.setattr(mm, "_load_artifact", lambda: art)
    mm._SCORE_CACHE.clear()


def _stub_features(monkeypatch, row: dict):
    import src.analysis.ml_dataset as ds
    monkeypatch.setattr(ds, "ticker_feature_frame",
                        lambda tk: pd.DataFrame([row]))


def _stub_legs(monkeypatch, legs, basis="hl1"):
    import src.analysis.pivot_target as pt
    monkeypatch.setattr(pt, "latest_leg_features", lambda tk: legs)
    monkeypatch.setattr(pt, "pivot_basis", lambda: basis)


# ── abstention labels ───────────────────────────────────────────────────────

def test_no_artifact_abstains_with_NO_MODEL():
    """A fresh checkout, a failed train, or a missing lightgbm — the method
    contributes nothing and says which."""
    assert mm.compute_ml_score("AAA") == (0.0, "NO_MODEL")


def test_a_corrupt_artifact_abstains_rather_than_raising():
    mm._MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    mm._MODEL_PATH.write_bytes(b"not a pickle")
    mm.reset_caches()
    assert mm.compute_ml_score("AAA") == (0.0, "NO_MODEL")


def test_missing_ohlcv_abstains_with_NO_DATA(monkeypatch, tmp_path):
    _write_art(_pivot_art())
    import src.data.cache as cache
    monkeypatch.setattr(cache, "_ohlcv_path",
                        lambda tk, interval="1d": tmp_path / "absent.json")
    assert mm.compute_ml_score("AAA") == (0.0, "NO_DATA")


def test_an_empty_feature_frame_abstains_with_NO_DATA(monkeypatch):
    _write_art(_pivot_art())
    import src.analysis.ml_dataset as ds
    monkeypatch.setattr(ds, "ticker_feature_frame", lambda tk: pd.DataFrame())
    assert mm.compute_ml_score("AAA") == (0.0, "NO_DATA")


def test_missing_leg_features_abstain_with_NO_DATA(monkeypatch):
    _write_art(_pivot_art())
    _stub_features(monkeypatch, {"f1": 1.0, "f2": 2.0})
    _stub_legs(monkeypatch, None)
    assert mm.compute_ml_score("AAA") == (0.0, "NO_DATA")


def test_a_raising_scorer_abstains_with_ERROR(monkeypatch):
    """Fail-soft is the contract — a weighted method must never take a tick
    down, but the label has to say it broke rather than that it had no view."""
    _write_art(_pivot_art())
    import src.analysis.ml_dataset as ds
    monkeypatch.setattr(ds, "ticker_feature_frame",
                        lambda tk: (_ for _ in ()).throw(RuntimeError("boom")))
    assert mm.compute_ml_score("AAA") == (0.0, "ERROR")


# ── the pivot-basis guard ───────────────────────────────────────────────────

def test_a_stale_pivot_basis_abstains(monkeypatch):
    """The 2026-08-12 guard. An artifact trained on the close basis would be fed
    H/L leg features it has never seen — degraded beats wrong."""
    _write_art(_pivot_art(basis="close1"))
    _stub_features(monkeypatch, {"f1": 1.0, "f2": 2.0})
    _stub_legs(monkeypatch, {"f1": 1.0, "f2": 2.0}, basis="hl1")
    assert mm.compute_ml_score("AAA") == (0.0, "BASIS_STALE")


def test_a_matching_basis_scores(monkeypatch):
    _write_art(_pivot_art(basis="hl1", model=_StubRank(pred=0.25)))
    _stub_features(monkeypatch, {"f1": 1.0, "f2": 2.0})
    _stub_legs(monkeypatch, {"f1": 1.0, "f2": 2.0}, basis="hl1")
    net, label = mm.compute_ml_score("AAA")
    assert label == "OK"
    assert net == pytest.approx(0.5)          # 2 x the centred rank


# ── the pivot head ──────────────────────────────────────────────────────────

def test_the_prediction_is_doubled_and_clipped(monkeypatch):
    """The head predicts a centred within-day rank in [-0.5, 0.5]; x2 maps it
    onto the method-score convention, and the clip guards a model that
    extrapolates past its own label range."""
    _stub_features(monkeypatch, {"f1": 1.0, "f2": 2.0})
    _stub_legs(monkeypatch, {"f1": 1.0, "f2": 2.0})
    for pred, expected in [(0.5, 1.0), (-0.5, -1.0), (0.0, 0.0),
                           (5.0, 1.0), (-5.0, -1.0)]:
        _write_art(_pivot_art(model=_StubRank(pred=pred)))
        assert mm.compute_ml_score("AAA")[0] == pytest.approx(expected)


def test_leg_features_override_the_frame_row(monkeypatch):
    """Leg state comes from `latest_leg_features`; the frame supplies the rest.
    Reading the frame first would serve a stale or absent leg value."""
    model = _StubRank()
    _serve_art(monkeypatch, _pivot_art(model=model, features=("f1", "legf")))
    _stub_features(monkeypatch, {"f1": 1.0, "legf": -99.0})
    _stub_legs(monkeypatch, {"legf": 7.0})
    mm.compute_ml_score("AAA")
    assert model.seen[0][0] == pytest.approx(1.0)
    assert model.seen[0][1] == pytest.approx(7.0), "frame value shadowed the leg"


def test_missing_features_become_nan_not_zero(monkeypatch):
    """LightGBM handles NaN as 'missing' natively; substituting 0.0 would be a
    real (and usually extreme) feature value."""
    model = _StubRank()
    _serve_art(monkeypatch, _pivot_art(model=model, features=("f1", "absent")))
    _stub_features(monkeypatch, {"f1": 1.0})
    _stub_legs(monkeypatch, {})
    mm.compute_ml_score("AAA")
    assert np.isnan(model.seen[0][1])


def test_features_are_passed_in_the_artifacts_own_order(monkeypatch):
    """Column order is part of the trained model; reordering silently feeds
    every feature into the wrong split."""
    model = _StubRank()
    _serve_art(monkeypatch, _pivot_art(model=model, features=("b", "a")))
    _stub_features(monkeypatch, {"a": 1.0, "b": 2.0})
    _stub_legs(monkeypatch, {})
    mm.compute_ml_score("AAA")
    assert list(model.seen[0]) == [2.0, 1.0]


# ── the v1 classic head ─────────────────────────────────────────────────────

def test_serving_dispatches_on_the_artifact_not_the_setting(monkeypatch):
    """The documented rule: `ml_ohlcv_target` picks what to TRAIN; serving reads
    the artifact's own config. Otherwise flipping the setting silently
    reinterprets an existing v1 pickle as a v2 rank model."""
    monkeypatch.setattr(
        __import__("config.settings", fromlist=["settings"]).settings,
        "ml_ohlcv_target", "pivot_rank")
    _write_art(_classic_art(model=_StubClassifier(bull=0.8, bear=0.1)))
    _stub_features(monkeypatch, {"eff_ratio": 0.9, "dollar_vol_log": 8.0})
    net, label = mm.compute_ml_score("AAA")
    assert label == "OK"
    assert net == pytest.approx(0.7)          # bull - bear, the v1 convention


def test_v1_abstains_outside_its_validated_subset(monkeypatch):
    """v1 was validated only on clean-trend liquid names; scoring outside that
    is extrapolation, so it declines to have a view."""
    _write_art(_classic_art())
    cfg = mm.TRAIN_CONFIG
    _stub_features(monkeypatch, {"eff_ratio": cfg["min_eff_ratio"] - 0.1,
                                 "dollar_vol_log": 9.0})
    assert mm.compute_ml_score("AAA") == (0.0, "NO_VIEW")
    _stub_features(monkeypatch, {"eff_ratio": 0.9,
                                 "dollar_vol_log": cfg["min_dollar_vol_log"] - 1})
    mm.reset_caches()
    assert mm.compute_ml_score("AAA") == (0.0, "NO_VIEW")


def test_v1_abstains_on_missing_conditioning_features(monkeypatch):
    _write_art(_classic_art())
    _stub_features(monkeypatch, {"eff_ratio": np.nan, "dollar_vol_log": 9.0})
    assert mm.compute_ml_score("AAA") == (0.0, "NO_VIEW")


# ── caching ─────────────────────────────────────────────────────────────────

def test_the_score_is_memoised_within_a_tick(monkeypatch):
    """`build_signals` calls this many times per tick against an unchanged
    frame; recomputing would repeat the whole feature build each time."""
    calls = {"n": 0}
    _write_art(_pivot_art())
    import src.analysis.ml_dataset as ds
    monkeypatch.setattr(ds, "ticker_feature_frame",
                        lambda tk: (calls.__setitem__("n", calls["n"] + 1)
                                    or pd.DataFrame([{"f1": 1.0, "f2": 2.0}])))
    _stub_legs(monkeypatch, {"f1": 1.0, "f2": 2.0})
    first = mm.compute_ml_score("AAA")
    second = mm.compute_ml_score("AAA")
    assert first == second and calls["n"] == 1


def test_a_retrained_artifact_is_picked_up_without_a_restart(monkeypatch):
    """The memo is keyed on the pickle's mtime, so the EOD train takes effect on
    the next tick rather than the next process."""
    _stub_features(monkeypatch, {"f1": 1.0, "f2": 2.0})
    _stub_legs(monkeypatch, {"f1": 1.0, "f2": 2.0})
    _write_art(_pivot_art(model=_StubRank(pred=0.1)))
    assert mm.compute_ml_score("AAA")[0] == pytest.approx(0.2)

    import os, time
    _write_art(_pivot_art(model=_StubRank(pred=0.4)))
    os.utime(mm._MODEL_PATH, (time.time() + 10, time.time() + 10))
    mm._ART_CACHE.update(mtime=None, art=None)      # simulate the mtime change
    assert mm.compute_ml_score("AAA")[0] == pytest.approx(0.8)


def test_reset_caches_clears_both_memos():
    _write_art(_pivot_art())
    mm._SCORE_CACHE[("AAA", 1)] = (0.5, "OK")
    mm.reset_caches()
    assert mm._SCORE_CACHE == {} and mm._ART_CACHE["art"] is None


# ── the training config ─────────────────────────────────────────────────────

def test_the_pivot_config_is_the_measured_winning_setup():
    """Full universe, unconditioned, uniform day-equal weights — conditioning
    and every weighting scheme were measured WORSE (memory/pivot-horizon-target)."""
    cfg = mm.TRAIN_CONFIG_PIVOT
    assert cfg["target"] == "pivot_rank" and cfg["model"] == "gbm_rank"
    assert "min_eff_ratio" not in cfg and "min_dollar_vol_log" not in cfg


def test_the_classic_config_keeps_its_conditioning():
    cfg = mm.TRAIN_CONFIG
    assert cfg["basis"] == "rel" and cfg["horizon"] == 10
    assert cfg["min_eff_ratio"] > 0 and cfg["min_dollar_vol_log"] > 0


def test_eod_train_dispatches_on_the_setting(monkeypatch):
    """The CLI/EOD entry point is the one place the SETTING decides, and the
    2026-08-12 fix routes it here so `python -m src.signals.ml_model` can no
    longer silently train v1 over a v2 artifact."""
    from config.settings import settings
    called = {}
    monkeypatch.setattr(mm, "train_and_persist_pivot",
                        lambda **kw: called.setdefault("pivot", kw))
    monkeypatch.setattr(mm, "train_and_persist", lambda **kw: called.setdefault("classic", kw))
    monkeypatch.setattr(settings, "ml_ohlcv_target", "pivot_rank")
    mm.eod_train(force=True)
    assert "pivot" in called and "classic" not in called


def test_pivot_retrain_is_throttled_unless_forced(monkeypatch):
    """Unforced, a fresh artifact skips the heavy rebuild and returns None; the
    weekly caller passes force=True so a human edit is never silently ignored."""
    from config.settings import settings
    monkeypatch.setattr(settings, "ml_ohlcv_target", "pivot_rank")
    monkeypatch.setattr(settings, "ml_pivot_retrain_days", 7)
    calls = {"n": 0}
    monkeypatch.setattr(mm, "train_and_persist_pivot",
                        lambda **kw: calls.__setitem__("n", calls["n"] + 1))
    _write_art(_pivot_art())                       # trained just now
    assert mm.eod_train() is None and calls["n"] == 0
    mm.eod_train(force=True)
    assert calls["n"] == 1


def test_an_aged_artifact_retrains(monkeypatch):
    from config.settings import settings
    monkeypatch.setattr(settings, "ml_ohlcv_target", "pivot_rank")
    monkeypatch.setattr(settings, "ml_pivot_retrain_days", 7)
    calls = {"n": 0}
    monkeypatch.setattr(mm, "train_and_persist_pivot",
                        lambda **kw: calls.__setitem__("n", calls["n"] + 1))
    art = _pivot_art()
    art["trained_at"] = (datetime.now(timezone.utc) - timedelta(days=30)).isoformat()
    _write_art(art)
    mm.eod_train()
    assert calls["n"] == 1
