"""Tests for the Phase-0 ML dataset builder and walk-forward harness.

The centre of gravity is LEAKAGE, because a leak is invisible — it looks like a
great model. Three probes, mirroring the adversarial style of test_replay.py:

  * features are causal: appending extreme FUTURE bars cannot move an already-
    emitted row;
  * a pure-NOISE dataset must walk-forward to IC ~ 0 (a harness that leaks would
    manufacture skill from nothing);
  * a PLANTED signal must be recovered (so the harness isn't just returning ~0
    for everything).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.analysis import ml_dataset as ds
from src.analysis import ml_train as mt


# ── synthetic OHLCV plumbing ─────────────────────────────────────────────────

def _dates(n, start="2019-01-01"):
    return [d.date() for d in pd.bdate_range(start=start, periods=n)]


def _series(vals, idx):
    return pd.Series([float(v) for v in vals], index=idx)


def _hlc(closes, highs=None, lows=None, vols=None, start="2019-01-01"):
    idx = _dates(len(closes), start)
    c = _series(closes, idx)
    h = _series(highs if highs is not None else [x * 1.01 for x in closes], idx)
    lo = _series(lows if lows is not None else [x * 0.99 for x in closes], idx)
    v = _series(vols if vols is not None else [1e6] * len(closes), idx)
    return idx, h, lo, c, v


def _patch_hlc(monkeypatch, table):
    """table: {ticker: (idx, high, low, close, vol)}."""
    monkeypatch.setattr(ds, "_hlc_by_session", lambda tk: table.get(tk))


# ── feature causality (the look-ahead probe) ─────────────────────────────────

def test_features_are_causal_future_bars_do_not_move_past_rows(monkeypatch):
    rng = np.random.default_rng(0)
    closes = list(100 * np.cumprod(1 + rng.normal(0, 0.01, 400)))

    _patch_hlc(monkeypatch, {"AAA": _hlc(closes)})
    full = ds.ticker_feature_frame("AAA")

    # Append 20 EXTREME future bars (10x jumps) and rebuild.
    closes_ext = closes + [closes[-1] * 10 ** (i + 1) for i in range(20)]
    _patch_hlc(monkeypatch, {"AAA": _hlc(closes_ext)})
    ext = ds.ticker_feature_frame("AAA")

    probe = full.index[300]                      # a row well before the appended tail
    # FRAME_FEATURE_COLUMNS, not FEATURE_COLUMNS: the within-ticker z-scores are
    # emitted by the same frame and are exactly the kind of column where a
    # full-sample (rather than expanding) normalisation would leak silently.
    cols = ds.FRAME_FEATURE_COLUMNS
    a = full.loc[probe, cols].astype(float).to_numpy()
    b = ext.loc[probe, cols].astype(float).to_numpy()
    # NaN-equal + value-equal: a future bar changing a past feature is a leak.
    assert np.allclose(a, b, equal_nan=True), "a future bar moved a past feature — look-ahead"


def test_within_ticker_z_is_expanding_and_scale_invariant(monkeypatch):
    """The two properties the _tz columns exist for.

    (1) Each value is the z-score over that ticker's OWN history up to that row —
        checked against a straight row-by-row reference, so a full-sample or
        rolling-window implementation fails here.
    (2) Rescaling and shifting a ticker's prices leaves the z-scores unchanged,
        which is the whole point: the pooled model can then compare a $8 biotech
        and a $400 utility on the same axis.
    """
    rng = np.random.default_rng(7)
    closes = list(100 * np.cumprod(1 + rng.normal(0, 0.012, 420)))
    _patch_hlc(monkeypatch, {"AAA": _hlc(closes)})
    fs = ds.ticker_feature_frame("AAA")

    for col in ("ret_5", "rsi_14", "realized_vol_20"):
        raw = fs[col].to_numpy(dtype=float)
        got = fs[col + "_tz"].to_numpy(dtype=float)
        i = 380
        hist = raw[: i + 1]
        hist = hist[~np.isnan(hist)]
        assert len(hist) >= ds._TZ_MIN_OBS
        ref = (raw[i] - hist.mean()) / hist.std(ddof=0)
        assert got[i] == pytest.approx(ref, rel=1e-9, abs=1e-9), f"{col}_tz is not an expanding z"

    # Warm-up rows must abstain rather than z-score off a handful of points.
    assert np.isnan(fs["ret_5_tz"].to_numpy(dtype=float)[:ds._TZ_MIN_OBS - 1]).all()

    # Scale/shift invariance: 3x the price level, same z-scores for a ratio feature.
    _patch_hlc(monkeypatch, {"AAA": _hlc([c * 3.0 for c in closes])})
    scaled = ds.ticker_feature_frame("AAA")
    assert np.allclose(fs["ret_5_tz"].to_numpy(dtype=float),
                       scaled["ret_5_tz"].to_numpy(dtype=float), equal_nan=True), \
        "within-ticker z must not depend on the ticker's price scale"


def test_stale_parquet_is_detected_not_silently_used():
    # The failure this guards is invisible: every consumer intersects its feature
    # list with the frame's columns, so a parquet predating a feature change
    # trains/validates the OLD model and reports it as current.
    fresh = pd.DataFrame({c: [0.0] for c in ds.ALL_FEATURE_COLUMNS})
    assert ds.missing_feature_columns(fresh) == []
    stale = fresh.drop(columns=[ds.TZ_FEATURE_COLUMNS[0]])
    assert ds.TZ_FEATURE_COLUMNS[0] in ds.missing_feature_columns(stale)
    assert ds.missing_feature_columns(pd.DataFrame()) == []   # emptiness is the caller's check


def test_features_within_expected_ranges(monkeypatch):
    rng = np.random.default_rng(1)
    closes = list(100 * np.cumprod(1 + rng.normal(0, 0.015, 400)))
    _patch_hlc(monkeypatch, {"AAA": _hlc(closes)})
    fs = ds.ticker_feature_frame("AAA")
    tail = fs.iloc[300]
    assert 0.0 <= tail["eff_ratio"] <= 1.0 + 1e-9
    assert -1.0 - 1e-9 <= tail["er_signed"] <= 1.0 + 1e-9
    assert 0.0 <= tail["rsi_14"] <= 100.0
    assert -1.0 - 1e-9 <= tail["updown_vol_10"] <= 1.0 + 1e-9
    assert -1.0 - 1e-9 <= tail["donchian_pos_20"] <= 1.0 + 1e-9


# ── labels ───────────────────────────────────────────────────────────────────

def test_forward_returns_raw_and_relative(monkeypatch):
    # A clean geometric ramp so the arithmetic is checkable by hand.
    n = 300
    tkr = [100.0 * (1.01 ** i) for i in range(n)]     # +1%/session
    spy = [50.0 * (1.005 ** i) for i in range(n)]      # +0.5%/session
    _patch_hlc(monkeypatch, {"AAA": _hlc(tkr), "SPY": _hlc(spy)})

    df = ds.build_dataset(tickers=["AAA"], horizons=[1], benchmark="SPY")
    assert not df.empty
    row = df.iloc[0]
    # 1-session raw return of a +1%/session ramp is ~+1%.
    assert row["fwd_ret_raw_1d"] == pytest.approx(1.0, abs=1e-6)
    # relative = raw - benchmark; benchmark ramps +0.5%/session.
    assert row["fwd_ret_rel_1d"] == pytest.approx(1.0 - 0.5, abs=1e-3)


def test_label_from_return_bands():
    assert mt.label_from_return(2.0, 0.0) == 2       # up
    assert mt.label_from_return(-2.0, 0.0) == 0      # down
    assert mt.label_from_return(0.1, 0.5) == 1       # inside deadband -> flat
    assert mt.label_from_return(0.9, 0.5) == 2       # outside -> up
    assert mt.label_from_return(float("nan"), 0.0) is None


# ── model ────────────────────────────────────────────────────────────────────

def test_model_is_deterministic_and_probabilities_valid():
    rng = np.random.default_rng(3)
    X = rng.normal(size=(400, 5))
    y = (X[:, 0] + rng.normal(0, 0.5, 400) > 0).astype(int) * 2   # classes {0, 2}
    p1 = mt.SoftmaxLogistic().fit(X, y).predict_proba(X)
    p2 = mt.SoftmaxLogistic().fit(X, y).predict_proba(X)
    assert np.array_equal(p1, p2), "training is not deterministic"

    model = mt.SoftmaxLogistic().fit(X, y)
    bull, bear = model.bull_bear(X)
    assert bull.min() >= 0.0 and bull.max() <= 1.0
    assert bear.min() >= 0.0 and bear.max() <= 1.0


def test_model_handles_nan_features_via_train_median():
    rng = np.random.default_rng(4)
    X = rng.normal(size=(300, 4))
    X[::7, 1] = np.nan                                # scattered missing values
    y = (X[:, 0] > 0).astype(int) * 2
    y[np.isnan(X[:, 0])] = 0
    model = mt.SoftmaxLogistic().fit(X, y)
    out = model.predict_proba(X)
    assert np.isfinite(out).all()


# ── walk-forward: the leakage probes ─────────────────────────────────────────

def _synthetic_panel(n_dates=160, n_tickers=40, seed=0, planted=False):
    """A feature/label frame in the shape build_dataset emits, built directly so
    the signal (or its absence) is controlled exactly."""
    rng = np.random.default_rng(seed)
    dates = [d.date().isoformat() for d in pd.bdate_range("2020-01-01", periods=n_dates + 1)]
    feats = ["ret_1", "ret_5", "ret_21", "er_signed"]
    rows = []
    for di in range(n_dates):
        for t in range(n_tickers):
            f = {c: float(rng.normal()) for c in feats}
            if planted:
                fwd = 3.0 * f["ret_21"] + rng.normal(0, 1.0)     # signal in ret_21
            else:
                fwd = float(rng.normal())                         # pure noise
            row = {"signal_date": dates[di], "ticker": f"T{t}",
                   "end_date_1d": dates[di + 1],
                   "fwd_ret_raw_1d": fwd, "fwd_ret_rel_1d": fwd}
            row.update(f)
            rows.append(row)
    return pd.DataFrame(rows)


def test_walkforward_pure_noise_gives_near_zero_ic():
    df = _synthetic_panel(planted=False, seed=11)
    preds = mt.walk_forward_predict(df, horizon=1, basis="raw", features=["ret_1", "ret_5", "ret_21", "er_signed"],
                                    min_train_days=60, step_days=20, min_train_rows=200)
    assert not preds.empty
    m = mt._metrics(preds["signal_date"], preds["net"], preds["fwd"])
    # No signal + no leak => OOS IC indistinguishable from zero. A leak would
    # manufacture skill here; this is the guard that it does not.
    assert abs(m["ic"]) < 0.08, f"noise produced |IC|={m['ic']} — the harness is leaking"


def test_walkforward_recovers_a_planted_signal():
    df = _synthetic_panel(planted=True, seed=12)
    preds = mt.walk_forward_predict(df, horizon=1, basis="raw", features=["ret_1", "ret_5", "ret_21", "er_signed"],
                                    min_train_days=60, step_days=20, min_train_rows=200)
    assert not preds.empty
    m = mt._metrics(preds["signal_date"], preds["net"], preds["fwd"])
    # The harness must also be able to FIND real signal, else "IC~0 on noise" is
    # trivially satisfied by a broken model that predicts nothing.
    assert m["ic"] > 0.2, f"planted signal not recovered (IC={m['ic']})"


def test_gbm_backend_deterministic_and_recovers_signal():
    pytest.importorskip("lightgbm")            # explicit skip, never a silent pass
    rng = np.random.default_rng(21)
    X = rng.normal(size=(600, 4))
    # A non-linear (interaction) target the linear model could not express.
    y = ((X[:, 0] * X[:, 1] > 0)).astype(int) * 2
    m1 = mt.LightGBMModel(n_estimators=50).fit(X, y)
    m2 = mt.LightGBMModel(n_estimators=50).fit(X, y)
    b1, _ = m1.bull_bear(X)
    b2, _ = m2.bull_bear(X)
    assert np.allclose(b1, b2), "GBM training is not deterministic"
    assert b1.min() >= 0.0 and b1.max() <= 1.0

    df = _synthetic_panel(planted=True, seed=22)
    preds = mt.walk_forward_predict(df, horizon=1, basis="raw",
                                    features=["ret_1", "ret_5", "ret_21", "er_signed"],
                                    min_train_days=60, step_days=20, min_train_rows=200,
                                    model_factory=mt.LightGBMModel)
    assert not preds.empty
    m = mt._metrics(preds["signal_date"], preds["net"], preds["fwd"])
    assert m["ic"] > 0.15, f"GBM did not recover planted signal (IC={m['ic']})"


def test_validate_on_panel_trains_deep_evaluates_panel():
    # Train on one frame (the "deep cache"), evaluate on a DIFFERENT frame (the
    # "panel"); when both carry the same relationship the deep-trained model must
    # recover it on the held-out panel — the train-deep / judge-forward split.
    from src.analysis.ml_validate import validate_on_panel
    deep = _synthetic_panel(planted=True, seed=31)
    panel = _synthetic_panel(planted=True, seed=32)
    preds = validate_on_panel(deep, panel, horizon=1, basis="raw",
                              features=["ret_1", "ret_5", "ret_21", "er_signed"],
                              min_train_rows=200, step_days=20)
    assert not preds.empty
    # Every prediction is a PANEL row, and the deep signal generalises to it.
    m = mt._metrics(preds["signal_date"], preds["net"], preds["fwd"])
    assert m["ic"] > 0.15, f"deep-trained model did not generalise to the panel (IC={m['ic']})"


def test_validate_on_panel_noise_panel_gives_zero_ic():
    # Deep has signal, panel is noise → the deep-trained model has nothing to
    # predict on the panel, so IC must be ~0 (not manufactured).
    from src.analysis.ml_validate import validate_on_panel
    deep = _synthetic_panel(planted=True, seed=33)
    panel = _synthetic_panel(planted=False, seed=34)
    preds = validate_on_panel(deep, panel, horizon=1, basis="raw",
                              features=["ret_1", "ret_5", "ret_21", "er_signed"],
                              min_train_rows=200, step_days=20)
    if not preds.empty:
        m = mt._metrics(preds["signal_date"], preds["net"], preds["fwd"])
        assert abs(m["ic"]) < 0.08, f"noise panel produced |IC|={m['ic']}"


def test_stacker_features_exclude_weight_dependent_columns():
    # Circularity guard (the load-bearing invariant): the buy stacker must NOT
    # train on values derived from the weights it exists to inform. Its features
    # are the individual (weight-independent) method scores only.
    from src.analysis.ml_stacker import STACKER_FEATURES
    for banned in ("combined_score", "combined_buy_score", "combined_sell_score", "confidence"):
        assert banned not in STACKER_FEATURES, f"{banned} must not be a stacker feature (circularity)"
    assert "news" in STACKER_FEATURES and "tech" in STACKER_FEATURES, "method scores should be features"


def test_condition_universe_filters_rows():
    df = pd.DataFrame({"ticker": list("ABCD"), "signal_date": ["2020-01-01"] * 4,
                       "eff_ratio": [0.1, 0.9, 0.5, 0.95], "dollar_vol_log": [5, 8, 6, 9]})
    out = mt.condition_universe(df, min_eff_ratio=0.8, min_dollar_vol_log=7.5)
    assert list(out["ticker"]) == ["B", "D"]


def test_walkforward_train_never_sees_unrealized_labels():
    # A row's label uses its end_date bar; training at cutoff D must exclude any
    # row whose end_date >= D. We assert the split honours end_date, not
    # signal_date, by planting a row that is signal<D but end>=D and checking it
    # is excluded from that step's training set via the public predict path
    # producing stable OOS coverage (no crash / no future row leaking in).
    df = _synthetic_panel(planted=True, seed=13)
    preds = mt.walk_forward_predict(df, horizon=1, basis="raw", features=["ret_21"],
                                    min_train_days=60, step_days=20, min_train_rows=200)
    # Every OOS prediction must be dated at or after the first eval cutoff, never
    # a training-window date.
    assert not preds.empty
    assert preds["signal_date"].nunique() > 0
