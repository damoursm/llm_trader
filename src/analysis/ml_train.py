"""ML training + walk-forward harness (Phase 0 go/no-go; see
``memory/ml-methods-plan-2026-07.md``).

Model classes (all deterministic, all loadable without sklearn/scipy):
``SoftmaxLogistic`` (pure-numpy baseline), ``LightGBMModel`` (native-Booster
classifier — the v1 ``ml_ohlcv`` config, kept as the fallback path), and
``LightGBMRankRegressor`` (the PRODUCTION ``ml_ohlcv`` v2 model since
2026-08-08: L2 on the within-day-rank signed-pivot label; see
``analysis/pivot_target.py``). Plus the walk-forward harness below.

1. ``SoftmaxLogistic`` — a dependency-free (pure-numpy) multinomial logistic
   regression, the elastic-net-style baseline the plan calls for and the model a
   later LightGBM backend must beat. Deterministic: zero-initialised, fixed
   iterations, no RNG — so inference is reproducible given the stored weights,
   the same contract the rest of the system holds (``daily_nav`` is bit-identical
   from the DB). Standardisation and median imputation are fit on TRAIN ONLY and
   baked into the object, so applying it to test data cannot leak test statistics.
   Not scikit/scipy on purpose: this project reimplements small numerics in-tree
   (``analysis/stats.py``) rather than depend on an uninstalled package that would
   silently take the fallback branch.

2. The **walk-forward harness** — the honest measurement. At each step it trains
   only on rows whose LABEL had already printed (``end_date < cutoff``) and
   predicts the next block, so no out-of-sample prediction can see its own future.
   This is the ML analog of ``backtest.run_backtest(walk_forward=True)``; getting
   it wrong is invisible (it looks like a great model), so the point-in-time split
   is verified adversarially in ``tests/test_ml_dataset.py`` — a pure-noise
   dataset must produce IC ~ 0.

**Output convention (user directive):** each method emits a BULLISH score and a
BEARISH score, both in [0, 1] — matching the buy/sell split combine — not a
single signed scalar. A 3-class {down, flat, up} target gives ``bull = P(up)``,
``bear = P(down)``, and both low = a genuine no-view (which a scalar cannot
express). Evaluation reduces to ``net = bull - bear`` for the Spearman IC (the
same convention as ``cmb_buy``/``cmb_sell``), plus per-side hit/return.

**Read the IC with the survivorship caveat.** This walks the deep OHLCV cache,
which is today's universe — optimistically biased. It answers "can an OHLCV model
learn ANYTHING out-of-sample", not "what will it earn live". The survivorship-free
promotion check is the forward ``signals`` panel, in Phase 1.

CLI:  python -m src.analysis.ml_train [--horizons 1,3] [--limit-tickers 400]
                                      [--step-days 10] [--min-train-days 120]
"""

from __future__ import annotations

import argparse
from datetime import date
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from loguru import logger

from src.analysis.ml_dataset import ALL_FEATURE_COLUMNS, build_dataset, load_materialized
from src.analysis.signal_panel import _spearman, periodic_ic_stats

_EPS = 1e-12


# ── labels ───────────────────────────────────────────────────────────────────

def label_from_return(r: float, deadband: float) -> Optional[int]:
    """{0=down, 1=flat, 2=up} from a forward return (%). ``deadband`` (>=0) sets
    the flat band; 0 collapses flat to the exact-zero measure (effectively
    two-class). None for a NaN return (no label)."""
    if r is None or r != r:
        return None
    if r > deadband:
        return 2
    if r < -deadband:
        return 0
    return 1


# ── pure-numpy multinomial logistic ──────────────────────────────────────────

class SoftmaxLogistic:
    """L2-regularised multinomial logistic regression, gradient descent, numpy.

    Deterministic (zero init, fixed iterations). ``fit`` learns and stores the
    imputation medians and standardisation stats from TRAIN, so ``predict_proba``
    on unseen rows never consults their statistics.
    """

    def __init__(self, l2: float = 1.0, lr: float = 0.5, iters: int = 300):
        self.l2 = float(l2)
        self.lr = float(lr)
        self.iters = int(iters)
        self.classes_: np.ndarray = np.array([])
        self._median: Optional[np.ndarray] = None
        self._mean: Optional[np.ndarray] = None
        self._std: Optional[np.ndarray] = None
        self._W: Optional[np.ndarray] = None          # (n_classes, d+1) incl. bias

    def _prep(self, X: np.ndarray, fit: bool) -> np.ndarray:
        X = np.asarray(X, dtype=float)
        if fit:
            med = np.nanmedian(X, axis=0)
            med = np.where(np.isnan(med), 0.0, med)     # all-NaN column -> 0
            self._median = med
        X = np.where(np.isnan(X), self._median, X)
        if fit:
            self._mean = X.mean(axis=0)
            std = X.std(axis=0)
            self._std = np.where(std < _EPS, 1.0, std)  # constant column -> unit scale
        X = (X - self._mean) / self._std
        return np.hstack([X, np.ones((X.shape[0], 1))])  # bias column

    @staticmethod
    def _softmax(Z: np.ndarray) -> np.ndarray:
        Z = Z - Z.max(axis=1, keepdims=True)
        E = np.exp(Z)
        return E / E.sum(axis=1, keepdims=True)

    def fit(self, X: np.ndarray, y: np.ndarray) -> "SoftmaxLogistic":
        Xb = self._prep(X, fit=True)
        self.classes_ = np.unique(y)
        k = len(self.classes_)
        idx = {c: i for i, c in enumerate(self.classes_)}
        Y = np.zeros((len(y), k))
        for r, c in enumerate(y):
            Y[r, idx[c]] = 1.0
        n, d = Xb.shape
        W = np.zeros((k, d))
        reg = np.ones(d); reg[-1] = 0.0                 # don't regularise the bias
        for _ in range(self.iters):
            P = self._softmax(Xb @ W.T)                 # (n, k)
            grad = (P - Y).T @ Xb / n + self.l2 * (W * reg) / n
            W -= self.lr * grad
        self._W = W
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        Xb = self._prep(X, fit=False)
        return self._softmax(Xb @ self._W.T)

    def bull_bear(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """``(bull, bear)`` each in [0, 1]: P(up) and P(down). A class absent from
        TRAIN contributes 0 on that side — an honest "the model never saw it"."""
        return _bull_bear_from_proba(self.predict_proba(X), self.classes_)


def _bull_bear_from_proba(P: np.ndarray, classes: Sequence) -> Tuple[np.ndarray, np.ndarray]:
    """Map class probabilities (columns ordered by sorted ``classes``) to
    ``bull = P(up=2)`` / ``bear = P(down=0)``, 0 for a class absent from train."""
    cls = list(classes)
    bull = P[:, cls.index(2)] if 2 in cls else np.zeros(len(P))
    bear = P[:, cls.index(0)] if 0 in cls else np.zeros(len(P))
    return bull, bear


class LightGBMModel:
    """Gradient-boosted trees behind the SAME ``fit`` / ``bull_bear`` interface as
    ``SoftmaxLogistic`` — the real test of the non-linear-interaction thesis the
    linear model provably cannot express (momentum working only inside clean
    trends, etc.). NaN is handled natively (no imputation) and no standardisation
    is needed. Deterministic: single-threaded with ``deterministic=True`` and a
    fixed seed, so inference is reproducible given the fitted trees.

    Uses lightgbm's NATIVE ``Booster`` API, NOT the ``LGBMClassifier`` sklearn
    wrapper — the wrapper would drag scikit-learn in as a hard dependency (it is
    needed even to UNPICKLE a saved model), and a Booster artifact is portable
    across environments without it. Found the hard way: an LGBMClassifier artifact
    trained where sklearn happened to be present failed to load in the production
    .venv (no sklearn) and fell soft to inactive.

    Imported lazily and required LOUDLY: if ``--model gbm`` is asked for and
    lightgbm is missing, the constructor raises rather than silently falling back
    to the linear model — an invisible downgrade is exactly the failure mode this
    project guards against.
    """

    def __init__(self, n_estimators: int = 300, learning_rate: float = 0.03,
                 num_leaves: int = 31, min_child_samples: int = 200,
                 reg_lambda: float = 5.0, colsample_bytree: float = 0.8):
        try:
            import lightgbm  # noqa: F401
        except ImportError as e:
            raise ImportError("LightGBMModel needs lightgbm — `pip install lightgbm`") from e
        self.n_estimators = int(n_estimators)
        # Native-API param names (the sklearn wrapper's aliases map to these):
        # reg_lambda→lambda_l2, colsample_bytree→feature_fraction, n_jobs→num_threads.
        self.params = dict(learning_rate=learning_rate, num_leaves=num_leaves,
                           min_child_samples=min_child_samples, lambda_l2=reg_lambda,
                           feature_fraction=colsample_bytree)
        self.classes_: np.ndarray = np.array([])
        self._booster = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> "LightGBMModel":
        import lightgbm as lgb
        self.classes_ = np.unique(y)
        k = len(self.classes_)
        remap = {c: i for i, c in enumerate(self.classes_)}
        y2 = np.array([remap[v] for v in y], dtype=int)
        params = dict(self.params)
        params.update(objective="multiclass", num_class=k, seed=0, num_threads=1,
                      deterministic=True, force_col_wise=True, verbosity=-1)
        dtrain = lgb.Dataset(np.asarray(X, dtype=float), label=y2)
        self._booster = lgb.train(params, dtrain, num_boost_round=self.n_estimators)
        return self

    def bull_bear(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        P = self._booster.predict(np.asarray(X, dtype=float))
        # multiclass predict returns (n, k) ordered by the remapped 0..k-1 labels,
        # which correspond position-for-position to sorted ``classes_``.
        if P.ndim == 1:                              # defensive: a 2-class edge case
            P = np.column_stack([1.0 - P, P])
        return _bull_bear_from_proba(P, self.classes_)


class LightGBMRankRegressor:
    """L2 regression on the within-day-rank pivot label — the ``ml_ohlcv`` v2
    model (see ``analysis/pivot_target.py`` and the 2026-08 record in
    ``memory/pivot-horizon-target-2026-08.md``).

    Same native-Booster / no-sklearn constraints as ``LightGBMModel``. The
    config is the measured winner frozen, not a tunable surface: 300 trees,
    lr 0.03, 31 leaves, min_child 200, λ₂ 5, feature_fraction 0.8 — capacity
    beyond this measured ~nothing (+0.0005 IC), and every weighting scheme
    other than uniform day-equal measured ≤ base (2026-08-08 experiment), so
    ``fit`` takes the day-equal weight vector from the caller and nothing else.
    ``num_threads`` is pinned (default 6): LightGBM's determinism contract is
    per-thread-count, so a wandering thread count would break artifact
    reproducibility.

    Predictions are the label's units — centred within-day rank ∈ ~[-0.5, 0.5],
    meaningful RELATIVE to same-day scores. ``predict`` returns them raw; the
    serving layer owns any rescaling to the method-score convention.
    """

    def __init__(self, n_estimators: int = 300, learning_rate: float = 0.03,
                 num_leaves: int = 31, min_child_samples: int = 200,
                 reg_lambda: float = 5.0, colsample_bytree: float = 0.8,
                 num_threads: int = 6):
        try:
            import lightgbm  # noqa: F401
        except ImportError as e:
            raise ImportError("LightGBMRankRegressor needs lightgbm — "
                              "`pip install lightgbm`") from e
        self.n_estimators = int(n_estimators)
        self.params = dict(objective="regression", learning_rate=learning_rate,
                           num_leaves=num_leaves, min_child_samples=min_child_samples,
                           lambda_l2=reg_lambda, feature_fraction=colsample_bytree,
                           seed=0, num_threads=int(num_threads), deterministic=True,
                           force_col_wise=True, verbosity=-1)
        self._booster = None

    def fit(self, X: np.ndarray, y: np.ndarray,
            weight: Optional[np.ndarray] = None) -> "LightGBMRankRegressor":
        import lightgbm as lgb
        ds = lgb.Dataset(np.asarray(X, dtype=float),
                         label=np.asarray(y, dtype=float),
                         weight=None if weight is None else np.asarray(weight, dtype=float))
        self._booster = lgb.train(self.params, ds, num_boost_round=self.n_estimators)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self._booster.predict(np.asarray(X, dtype=float))


# ── probability calibration ──────────────────────────────────────────────────

class IsotonicCalibrator:
    """Isotonic regression (Pool Adjacent Violators) mapping a model's RAW score
    to a calibrated probability. Pure numpy — no sklearn/scipy, so the pickled
    artifact loads in the production .venv (the same constraint that forced
    ``LightGBMModel`` onto the native Booster API).

    **Why this exists.** Gradient-boosted probabilities are systematically
    distorted (Niculescu-Mizil & Caruana, ICML 2005), and the combine consumes
    the number AS a probability: ``max(0, 2p−1)`` is only a conviction if ``p``
    means what it says. Measured on the ml_buy walk-forward panel (2026-08-03,
    n=10,248), raw P(up) ranged 0.087→0.929 while the realised up-rate stayed
    flat at ~0.47 in EVERY bin — Brier skill −0.098, i.e. worse than always
    predicting the base rate. Uncalibrated, that fed conviction 0.86 into Gate 1
    and sizing on names that were no better than a coin flip.

    **Fit on OUT-OF-FOLD predictions only.** A calibrator fit on in-sample
    predictions is worthless: the model already fits its training data, so the
    in-sample curve looks calibrated while the live one is not. Callers pass
    walk-forward OOF predictions (see ``ml_stacker._fit_calibrator``).

    **The honest consequence:** for a model with no ranking power, isotonic
    correctly collapses to ~the base rate, so the conviction goes to ~0 and the
    model ABSTAINS instead of asserting. That is the intended safety property,
    not a failure — an uninformative model should say nothing.
    """

    def __init__(self) -> None:
        self.x_: Optional[np.ndarray] = None      # breakpoints (raw score)
        self.y_: Optional[np.ndarray] = None      # calibrated probability
        self.n_fit_: int = 0

    def fit(self, p: Sequence[float], y: Sequence[int]) -> "IsotonicCalibrator":
        """``p`` = raw scores, ``y`` = binary outcomes (1 = the predicted class
        happened). Both must come from data the model did NOT train on."""
        p = np.asarray(p, dtype=float)
        y = np.asarray(y, dtype=float)
        ok = np.isfinite(p) & np.isfinite(y)
        p, y = p[ok], y[ok]
        if len(p) < 20:                            # too thin to calibrate
            return self
        order = np.argsort(p, kind="mergesort")
        p, y = p[order], y[order]
        # PAVA: successively pool adjacent blocks that violate monotonicity.
        vals = y.astype(float).copy()
        wts = np.ones_like(vals)
        idx = list(range(len(vals)))               # block -> end position
        v: List[float] = []
        w: List[float] = []
        e: List[int] = []
        for i in range(len(vals)):
            v.append(vals[i]); w.append(wts[i]); e.append(i)
            while len(v) > 1 and v[-2] > v[-1]:    # violation → pool
                w_new = w[-2] + w[-1]
                v_new = (v[-2] * w[-2] + v[-1] * w[-1]) / w_new
                v.pop(); w.pop(); e_last = e.pop()
                v[-1] = v_new; w[-1] = w_new; e[-1] = e_last
        # Expand blocks back to a step function over the sorted scores.
        fitted = np.empty(len(vals), dtype=float)
        start = 0
        for bi, end in enumerate(e):
            fitted[start:end + 1] = v[bi]
            start = end + 1
        self.x_, self.y_ = p, np.clip(fitted, 0.0, 1.0)
        self.n_fit_ = int(len(p))
        return self

    def transform(self, p: Sequence[float]) -> np.ndarray:
        """Calibrated probabilities. Outside the fitted range the endpoints are
        held (no extrapolation — an unseen score gets the nearest known rate)."""
        p = np.asarray(p, dtype=float)
        if self.x_ is None or self.y_ is None or len(self.x_) == 0:
            return p                                # unfitted → identity (fail-soft)
        return np.interp(p, self.x_, self.y_, left=self.y_[0], right=self.y_[-1])

    def transform_one(self, p: float) -> float:
        return float(self.transform([p])[0])


def brier(p: Sequence[float], y: Sequence[int]) -> float:
    """Mean squared error of a probability forecast (lower is better)."""
    p = np.asarray(p, dtype=float); y = np.asarray(y, dtype=float)
    ok = np.isfinite(p) & np.isfinite(y)
    return float(np.mean((p[ok] - y[ok]) ** 2)) if ok.any() else float("nan")


def brier_skill(p: Sequence[float], y: Sequence[int]) -> float:
    """Brier skill vs the base-rate forecast. ``> 0`` beats always-predict-the-
    base-rate; ``<= 0`` means the forecast adds nothing (or hurts)."""
    y_arr = np.asarray(y, dtype=float)
    ok = np.isfinite(y_arr)
    if not ok.any():
        return float("nan")
    base = float(np.mean(y_arr[ok]))
    ref = brier(np.full(len(y_arr), base), y_arr)
    return float("nan") if not ref else 1.0 - brier(p, y_arr) / ref


def make_model_factory(name: str) -> Callable[[], Any]:
    """A zero-arg factory returning a FRESH model per walk-forward step."""
    if name == "gbm":
        return LightGBMModel
    if name == "logistic":
        return SoftmaxLogistic
    raise ValueError(f"unknown model '{name}' (logistic|gbm)")


# ── walk-forward ─────────────────────────────────────────────────────────────

def walk_forward_predict(df: pd.DataFrame, horizon: int, basis: str,
                         features: Sequence[str] = ALL_FEATURE_COLUMNS,
                         deadband: float = 0.0, min_train_days: int = 120,
                         step_days: int = 10, min_train_rows: int = 500,
                         model_factory: Optional[Callable[[], Any]] = None) -> pd.DataFrame:
    """Point-in-time OOS predictions for one (horizon, basis).

    At each step's cutoff date D: train on rows whose label END BAR printed before
    D (``end_date_<h>d < D``) — NOT merely ``signal_date < D``, because a row
    entered before D but ending after it carries a return that had not happened
    yet, the exact leak walk-forward removes. Predict rows dated in the next
    ``step_days`` block. Returns columns: signal_date, ticker, bull, bear, net, fwd.
    """
    # The PIVOT label (2026-08-12) rides the same plumbing under basis
    # "rank_pv" / "sellinv_pv": its columns are horizon-free (each row settles
    # at its OWN pivot; `end_date_pv` carries that), so the nominal ``horizon``
    # argument is ignored for column resolution.
    if str(basis).endswith("_pv"):
        ycol, ecol = f"fwd_ret_{str(basis)[:-3]}_pv", "end_date_pv"
    else:
        ycol = f"fwd_ret_{basis}_{horizon}d"
        ecol = f"end_date_{horizon}d"
    if ycol not in df.columns or ecol not in df.columns:
        return pd.DataFrame()
    feats = [f for f in features if f in df.columns]
    work = df[["signal_date", "ticker", ycol, ecol] + feats].copy()
    work[ycol] = pd.to_numeric(work[ycol], errors="coerce")
    work = work[work[ycol].notna()].reset_index(drop=True)
    if work.empty:
        return pd.DataFrame()
    work["_y"] = work[ycol].map(lambda r: label_from_return(r, deadband))
    work = work[work["_y"].notna()].reset_index(drop=True)
    work["_sig"] = work["signal_date"].map(lambda s: date.fromisoformat(str(s)[:10]))
    work["_end"] = work[ecol].map(lambda s: date.fromisoformat(str(s)[:10]))
    Xall = work[feats].to_numpy(dtype=float)
    yall = work["_y"].to_numpy(dtype=int)

    uniq = sorted(work["_sig"].unique())
    if len(uniq) <= min_train_days + 1:
        return pd.DataFrame()
    factory = model_factory or SoftmaxLogistic
    out: List[pd.DataFrame] = []
    for a in range(min_train_days, len(uniq), step_days):
        cutoff = uniq[a]
        nxt = uniq[a + step_days] if a + step_days < len(uniq) else None
        tr = (work["_end"].to_numpy() < cutoff)
        if nxt is None:
            te = (work["_sig"].to_numpy() >= cutoff)
        else:
            sig = work["_sig"].to_numpy()
            te = (sig >= cutoff) & (sig < nxt)
        if tr.sum() < min_train_rows or te.sum() == 0:
            continue
        ytr = yall[tr]
        if len(np.unique(ytr)) < 2:                     # need both directions to learn
            continue
        model = factory().fit(Xall[tr], ytr)
        bull, bear = model.bull_bear(Xall[te])
        sub = work.loc[te, ["signal_date", "ticker", ycol]].copy()
        sub["bull"] = bull
        sub["bear"] = bear
        sub["net"] = bull - bear
        sub["fwd"] = work.loc[te, ycol].to_numpy()
        out.append(sub)
    if not out:
        return pd.DataFrame()
    return pd.concat(out, ignore_index=True)


def _metrics(dates: Sequence, net: Sequence, fwd: Sequence, min_n: int = 100) -> dict:
    """IC / ICIR / directional hit% / sim-return% for a signed score vs return."""
    net = pd.to_numeric(pd.Series(net), errors="coerce")
    fwd = pd.to_numeric(pd.Series(fwd), errors="coerce")
    ok = net.notna() & fwd.notna()
    net, fwd = net[ok], fwd[ok]
    d = pd.Series(list(dates))[ok.values]
    n = int(len(net))
    res = {"n": n, "ic": None, "icir": None, "hit": None, "simret": None}
    if n < min_n:
        return res
    ic = _spearman(net, fwd)
    res["ic"] = round(ic, 4) if ic is not None else None
    _, _, icir, _ = periodic_ic_stats(list(d), list(net), list(fwd))
    res["icir"] = round(icir, 3) if icir is not None else None
    moved = fwd != 0
    if moved.any():
        res["hit"] = round(float(((net > 0) == (fwd > 0))[moved].mean() * 100.0), 2)
    res["simret"] = round(float(fwd.where(net > 0, -fwd).mean()), 4)
    return res


def condition_universe(df: pd.DataFrame, min_eff_ratio: Optional[float] = None,
                       min_dollar_vol_log: Optional[float] = None) -> pd.DataFrame:
    """Restrict to the rows ``predictability.py`` finds most forecastable —
    clean-trend (``eff_ratio``) and/or liquid (``dollar_vol_log``) names — so the
    model is trained/evaluated where an edge is plausible rather than diluted
    across chop. Both bounds are optional and causal (as-of-date features)."""
    out = df
    if min_eff_ratio is not None and "eff_ratio" in out.columns:
        out = out[pd.to_numeric(out["eff_ratio"], errors="coerce") >= min_eff_ratio]
    if min_dollar_vol_log is not None and "dollar_vol_log" in out.columns:
        out = out[pd.to_numeric(out["dollar_vol_log"], errors="coerce") >= min_dollar_vol_log]
    return out.reset_index(drop=True)


def evaluate(df: pd.DataFrame, horizons: Sequence[int] = (1, 3),
             bases: Sequence[str] = ("raw", "rel"), deadband: float = 0.0,
             baseline_features: Sequence[str] = ("ret_21", "er_signed"),
             model_name: str = "logistic", **wf) -> pd.DataFrame:
    """Walk-forward every (basis, horizon) and return the go/no-go table: the
    model's OOS IC/ICIR/hit/simret, and — for context — each baseline single
    feature's IC on the SAME out-of-sample rows (the bar the model must clear)."""
    factory = make_model_factory(model_name)
    rows: List[dict] = []
    for h in horizons:
        for basis in bases:
            preds = walk_forward_predict(df, h, basis, deadband=deadband,
                                         model_factory=factory, **wf)
            if preds.empty:
                rows.append({"model": model_name, "basis": basis, "horizon": h,
                             "n": 0, "ic": None, "icir": None, "hit": None, "simret": None})
                continue
            m = _metrics(preds["signal_date"], preds["net"], preds["fwd"])
            rows.append({"model": model_name, "basis": basis, "horizon": h, **m})
            # Baselines evaluated on the identical OOS rows (join back the feature).
            key = df[["signal_date", "ticker"] + [f for f in baseline_features if f in df.columns]]
            merged = preds.merge(key, on=["signal_date", "ticker"], how="left")
            for bf in baseline_features:
                if bf not in merged.columns:
                    continue
                bm = _metrics(merged["signal_date"], merged[bf], merged["fwd"])
                rows.append({"model": f"feat:{bf}", "basis": basis, "horizon": h, **bm})
    return pd.DataFrame(rows)


def _print_report(table: pd.DataFrame) -> None:
    import sys
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    if table is None or table.empty:
        print("No walk-forward results — dataset too small or too few dates. "
              "Try more tickers / smaller --min-train-days.")
        return
    print("\nWalk-forward OOS performance — model vs single-feature baselines")
    print("net = bull - bear; IC = Spearman(net, fwd_ret); hit = directional %; "
          "simret = mean sign(net)*fwd_ret %.")
    print("basis: raw = fwd return; rel = net of the benchmark (market-relative).\n")
    head = f"{'model':<16}{'basis':>6}{'h':>4}{'n':>9}{'IC':>9}{'ICIR':>8}{'hit%':>8}{'simret%':>9}"
    print(head); print("-" * len(head))
    def f(v, w, s):
        # Format the number FIRST, then pad — a combined spec like ">9+.4f"
        # puts width before sign and is invalid.
        return f"{format(v, s):>{w}}" if v is not None and pd.notna(v) else f"{'—':>{w}}"

    for (basis, h), g in table.groupby(["basis", "horizon"], sort=True):
        for _, r in g.iterrows():
            line = f"{r['model']:<16}{basis:>6}{int(h):>4}{int(r['n']):>9}"
            line += f(r['ic'], 9, '+.4f') + f(r['icir'], 8, '+.3f')
            line += f(r['hit'], 8, '.2f') + f(r['simret'], 9, '+.4f')
            print(line)
        print("-" * len(head))
    print("\nDEEP-CACHE walk-forward — survivorship-biased (today's universe). It answers "
          "'can an OHLCV model learn out-of-sample', not 'what it earns live'. The model "
          "must beat the single-feature baselines to be worth more than one indicator; the "
          "honest promotion check is the forward signals panel (Phase 1).")


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description="Walk-forward evaluate the OHLCV-only ML model")
    p.add_argument("--horizons", default="1,3", help="forward horizons (default 1,3)")
    p.add_argument("--limit-tickers", type=int, default=400, help="cap tickers for a quick run (default 400)")
    p.add_argument("--date-stride", type=int, default=3, help="emit every Nth session per ticker (default 3)")
    p.add_argument("--min-date", default=None, help="drop sessions before YYYY-MM-DD")
    p.add_argument("--deadband", type=float, default=0.0, help="flat-class band on the return %% (default 0)")
    p.add_argument("--min-train-days", type=int, default=120, help="warmup distinct dates before eval (default 120)")
    p.add_argument("--step-days", type=int, default=10, help="retrain/predict block size in distinct dates (default 10)")
    p.add_argument("--from-parquet", default=None, help="load a materialized dataset instead of building")
    p.add_argument("--model", default="logistic", choices=("logistic", "gbm"),
                   help="model backend (default logistic; gbm needs lightgbm)")
    p.add_argument("--min-eff-ratio", type=float, default=None,
                   help="keep only rows with Kaufman efficiency >= this (clean-trend conditioning)")
    p.add_argument("--min-dollar-vol-log", type=float, default=None,
                   help="keep only rows with log10 $-vol >= this (liquidity conditioning; ~7 = $10M)")
    a = p.parse_args(argv)
    horizons = tuple(int(h) for h in str(a.horizons).split(",") if h.strip())

    from src.db import repo
    repo.set_read_only(True)

    if a.from_parquet:
        df = load_materialized(a.from_parquet)
    else:
        df = build_dataset(horizons=horizons, limit_tickers=a.limit_tickers,
                           date_stride=a.date_stride, min_date=a.min_date)
    if df is None or df.empty:
        print("No dataset rows. Warm the OHLCV cache (python main.py --backfill) or widen the args.")
        return
    n0 = len(df)
    df = condition_universe(df, a.min_eff_ratio, a.min_dollar_vol_log)
    if a.min_eff_ratio is not None or a.min_dollar_vol_log is not None:
        print(f"universe conditioning: {n0:,} -> {len(df):,} rows "
              f"(eff>={a.min_eff_ratio}, $vol_log>={a.min_dollar_vol_log})")
    table = evaluate(df, horizons=horizons, deadband=a.deadband, model_name=a.model,
                     min_train_days=a.min_train_days, step_days=a.step_days)
    print(f"\nmodel = {a.model}  |  deadband = {a.deadband}%")
    _print_report(table)


if __name__ == "__main__":
    main()
