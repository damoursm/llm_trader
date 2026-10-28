"""PREREG190's refusal model in production (src/signals/sel_refusal.py, journal-only): the installed model's logistic
on the pick's features, a missing feature at the model's stand-in, a fund never refused, no model no probability."""
import json
import math
from datetime import date

from config.settings import settings
from src.signals import sel_refusal as SR

MODEL = {"source": "test", "model": {"name": "M2c", "features": ["micro", "fpi", "age", "mcap"],
                                     "mean": [0.3, 0.2, 1.5, 8.6], "sd": [0.5, 0.4, 1.3, 1.0],
                                     "w": [-4.7, 0.6, 0.6, -0.8, -0.5]},
         "fill": {"micro": 0.3, "fpi": 0.0, "age": 1.7, "mcap": 8.8}, "state": True}


def _install(monkeypatch, tmp_path, feats):
    monkeypatch.setattr(settings, "sel_short_dir", str(tmp_path / "sel"))
    p = SR.model_path()
    p.parent.mkdir(parents=True)
    p.write_text(json.dumps(MODEL), encoding="utf-8")
    monkeypatch.setattr(SR, "features", lambda tk, d, price=None: dict(feats))


def _expected(x):
    m = MODEL["model"]
    z = m["w"][0] + sum(w * (v - mu) / sd for w, v, mu, sd in zip(m["w"][1:], x, m["mean"], m["sd"]))
    return 1 / (1 + math.exp(-z))


def test_the_probability_is_the_installed_models_logistic(monkeypatch, tmp_path):
    _install(monkeypatch, tmp_path, {"kind": "stock", "micro": 1.0, "fpi": 1.0, "age": 0.0, "mcap": 7.0})
    assert abs(SR.probability("NCT", date(2026, 10, 26)) - _expected([1.0, 1.0, 0.0, 7.0])) < 1e-12


def test_a_missing_feature_takes_the_stand_in_and_a_fund_is_never_refused(monkeypatch, tmp_path):
    _install(monkeypatch, tmp_path, {"kind": "stock", "micro": None, "fpi": 0.0, "age": None, "mcap": 9.0})
    assert abs(SR.probability("X", date(2026, 10, 26)) - _expected([0.3, 0.0, 1.7, 9.0])) < 1e-12
    monkeypatch.setattr(SR, "features", lambda tk, d, price=None: {"kind": "etf"})
    assert SR.probability("SPY", date(2026, 10, 26)) == 0.0


def test_no_model_no_probability(monkeypatch, tmp_path):
    monkeypatch.setattr(settings, "sel_short_dir", str(tmp_path / "none"))
    assert SR.model() is None and SR.probability("NCT", date(2026, 10, 26)) is None
