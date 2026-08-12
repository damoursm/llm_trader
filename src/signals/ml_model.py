"""``ml_ohlcv`` — the trained OHLCV model as a PANEL-FIRST signal method (weight 0).

Phase 1b of ``memory/ml-methods-plan-2026-07.md``, re-based 2026-08-08 onto the
SIGNED PIVOT target (``ml_ohlcv`` v2 — full record in
``memory/pivot-horizon-target-2026-08.md``). Both generations cleared the same
gate — ``analysis/ml_validate.py``, train-on-deep / judge-on-the-survivorship-
free-panel — and both wired in as panel-first (weight 0) at first. **PROMOTED
2026-08-11 (user-directed): ``ml_ohlcv`` now carries a 0.12 ``_BASE_WEIGHTS``
entry** on the strength of the forward-panel gate (IC +0.0639, t +2.31) and the
9M-row deep validation, so it counts in the combine / coherence / family votes;
the exit consensus still excludes it (``_CONSENSUS_SKIP`` — exit-side skill is
measured separately in the exit panel).

**v2 (current, ``settings.ml_ohlcv_target = "pivot_rank"``):** a
``LightGBMRankRegressor`` on the within-day rank of the signed pivot return
(``analysis/pivot_target.py``), 85 features (the 76 causal OHLCV features + the
9 leg-state features), uniform day-equal weights, trained on the FULL deep
cache — no clean-trend/liquidity conditioning, because the 2026-08 experiments
measured the unconditioned full-data model best (and every weighting scheme
≤ uniform). Score = ``clip(2·pred, −1, 1)``: the head predicts a centred
within-day rank ∈ [−0.5, 0.5], so ×2 maps its natural range onto the method-
score convention; the number is meaningful RELATIVE to same-day scores, which
is exactly how the panel consumes it.

**v1 (classifier, kept as the fallback path):** GBM ``P(up)−P(down)`` on the
10-day market-relative label, clean-trend+liquid conditioned, ``NO_VIEW``
outside that subset. An old artifact keeps serving through the v1 branch, so a
failed v2 train degrades to yesterday's model, never to a broken tick.

* ``train_and_persist`` / ``train_and_persist_pivot`` — train and pickle
  model + feature list + config to ``cache/ml/ml_ohlcv_model.pkl`` + a row in
  the ``ml_models`` registry. The artifact is fixed once written, so inference
  is reproducible from it — the same "deterministic given the stored artifact"
  contract as the real-fill costs.

* ``compute_ml_score`` — the per-ticker scorer, dispatching on the artifact's
  own ``config["target"]`` (an artifact knows what it is; serving never guesses
  from settings). Fail-soft throughout: missing artifact, absent lightgbm, or
  too-short history all yield 0.0 — the method is inactive, never a broken tick.

* ``eod_train`` — the EOD entry point, dispatching on ``settings.ml_ohlcv_target``.
  The pivot retrain is THROTTLED (``ml_pivot_retrain_days``, default 7): the
  full-universe dataset rebuild is ~20-40 min, the 63-day-refit harness showed
  staleness of days costs ~nothing, and a weekly cadence keeps the EOD window
  light. In pivot mode the ``dataset_multi`` refresh for the offline analysis
  CLIs is NOT performed here; those tools rebuild on demand.

The model's OUTPUT changes on every retrain, so stored history is not comparable
across retrains via the AST-based scorer epoch — handled by the ``ml_models``
registry (model-as-of-date). The v1→v2 SWAP is categorical (different target,
different feature set, different score semantics) and IS epoch-registered
(``METHOD_SCORER_EPOCH["ml_ohlcv"]``).
"""

from __future__ import annotations

import pickle
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
from loguru import logger

_MODEL_PATH = Path("cache/ml/ml_ohlcv_model.pkl")

# The winning config from Phase 0/1: GBM on clean-trend (Kaufman eff >= 0.4) +
# liquid (log10 $-vol >= 7 ≈ $10M) names, 10-day market-relative label.
TRAIN_CONFIG = dict(horizon=10, basis="rel", deadband=0.0,
                    min_eff_ratio=0.4, min_dollar_vol_log=7.0, model="gbm")

# v2 (2026-08-08): signed-pivot within-day-rank regressor on the FULL universe —
# unconditioned, uniform day-equal weights (both measured; see module docstring).
# The deep parquet path is v2's own (full universe, stride 1) so the v1
# dataset_multi (1500 tickers, stride 2) keeps serving the offline CLIs untouched.
TRAIN_CONFIG_PIVOT = dict(target="pivot_rank", model="gbm_rank", horizon=1,
                          basis="pivot_rank", num_threads=6)
_PIVOT_PARQUET = "cache/ml/dataset_full.parquet"

# Artifact memo, keyed on the pickle's mtime so a fresh EOD train is picked up
# automatically without a restart (mirrors cache.load_ohlcv's invalidation).
_ART_CACHE: dict = {"mtime": None, "art": None}
# Per-tick score memo, keyed on (ticker, ohlcv mtime): build_signals runs many
# times per tick, and the feature frame is the same each time within a tick.
_SCORE_CACHE: dict = {}


# ── training ─────────────────────────────────────────────────────────────────

def train_and_persist(deep_parquet: str = "cache/ml/dataset_multi.parquet",
                      limit_tickers: int = 1500, date_stride: int = 2,
                      path: Path = _MODEL_PATH) -> Optional[dict]:
    """Train the winning config on the deep cache and pickle it. Returns the
    artifact dict, or None if there was nothing to train on."""
    from src.analysis.ml_dataset import (ALL_FEATURE_COLUMNS, build_dataset,
                                         load_materialized, missing_feature_columns)
    from src.analysis.ml_train import LightGBMModel, condition_universe, label_from_return

    cfg = TRAIN_CONFIG
    h, basis = cfg["horizon"], cfg["basis"]
    df = load_materialized(deep_parquet) if deep_parquet and Path(deep_parquet).exists() else None
    # A cached parquet that predates a feature change must be REBUILT, not
    # quietly trained on: every consumer intersects the feature list with the
    # frame's columns, so a stale file trains the old model and looks fine.
    stale = missing_feature_columns(df)
    if stale:
        logger.warning(f"[ml_ohlcv] {deep_parquet} is missing {len(stale)} current "
                       f"feature columns (e.g. {stale[:3]}) — rebuilding the dataset")
    if df is None or df.empty or f"fwd_ret_{basis}_{h}d" not in df.columns or stale:
        df = build_dataset(horizons=[h], limit_tickers=limit_tickers, date_stride=date_stride)
    if df is None or df.empty:
        logger.warning("[ml_ohlcv] no deep dataset to train on")
        return None
    df = condition_universe(df, cfg["min_eff_ratio"], cfg["min_dollar_vol_log"])
    feats = [f for f in ALL_FEATURE_COLUMNS if f in df.columns]
    ycol = f"fwd_ret_{basis}_{h}d"
    y_raw = df[ycol].map(lambda r: label_from_return(r, cfg["deadband"]))
    keep = y_raw.notna()
    X = df.loc[keep, feats].to_numpy(dtype=float)
    y = y_raw[keep].to_numpy(dtype=int)
    if len(X) < 500 or len(np.unique(y)) < 2:
        logger.warning(f"[ml_ohlcv] insufficient training rows ({len(X)})")
        return None

    model = LightGBMModel().fit(X, y)
    art = {"model": model, "features": feats, "config": dict(cfg),
           "trained_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
           "n_train": int(len(X)), "train_max_date": str(df["signal_date"].max())}
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as fh:
        pickle.dump(art, fh)
    _ART_CACHE.update(mtime=None, art=None)             # force reload next score
    _SCORE_CACHE.clear()
    _record_registry(art)
    logger.info(f"[ml_ohlcv] trained on {art['n_train']:,} rows "
                f"(<= {art['train_max_date']}) -> {path}")
    return art


def train_and_persist_pivot(deep_parquet: str = _PIVOT_PARQUET,
                            path: Path = _MODEL_PATH,
                            rebuild_dataset: bool = False) -> Optional[dict]:
    """Train the v2 config (signed-pivot within-day-rank GBM, full universe,
    uniform day-equal weights) and pickle it. Every training row's pivot has
    printed by construction (``pivot_frame`` emits settled targets only), so no
    further point-in-time gate is needed at final-training time — the walk-
    forward cutoffs belong to validation, not to the shipped artifact."""
    from src.analysis.ml_dataset import (ALL_FEATURE_COLUMNS, load_materialized,
                                         materialize, missing_feature_columns)
    from src.analysis.ml_train import LightGBMRankRegressor
    from src.analysis.pivot_target import LEG_FEATURES, pivot_frame, within_day_rank

    df = load_materialized(deep_parquet) if Path(deep_parquet).exists() else None
    stale = missing_feature_columns(df)
    if stale:
        logger.warning(f"[ml_ohlcv] {deep_parquet} lacks {len(stale)} current feature "
                       f"columns (e.g. {stale[:3]}) — rebuilding")
    if rebuild_dataset or df is None or df.empty or stale:
        try:
            materialize(deep_parquet, horizons=[5, 10], date_stride=1)
            df = load_materialized(deep_parquet)
        except Exception as e:
            logger.warning(f"[ml_ohlcv] full dataset rebuild failed ({e}); "
                           f"training on the prior parquet if any")
    if df is None or df.empty:
        logger.warning("[ml_ohlcv] no deep dataset to train the pivot model on")
        return None

    pf = pivot_frame(sorted(df["ticker"].unique()))
    if pf.empty:
        logger.warning("[ml_ohlcv] pivot frame empty — nothing to train on")
        return None
    df = df.merge(pf, on=["ticker", "signal_date"], how="inner")
    df["sp_buy"] = np.asarray(df["sp_buy"], dtype=float)
    df = df[np.isfinite(df["sp_buy"])].reset_index(drop=True)
    feats = [f for f in ALL_FEATURE_COLUMNS if f in df.columns] + list(LEG_FEATURES)
    if len(df) < 50000:
        logger.warning(f"[ml_ohlcv] insufficient pivot training rows ({len(df):,})")
        return None

    day_codes, day_idx = np.unique(df["signal_date"].astype(str).to_numpy(),
                                   return_inverse=True)
    yr = within_day_rank(df["sp_buy"].to_numpy(np.float64), day_idx.astype(np.int32))
    cnt = np.bincount(day_idx)
    w = (1.0 / cnt[day_idx]).astype(np.float64)
    w /= w.mean()
    X = df[feats].to_numpy(dtype=np.float32)

    cfg = dict(TRAIN_CONFIG_PIVOT)
    model = LightGBMRankRegressor(num_threads=cfg["num_threads"]).fit(X, yr, w)
    art = {"model": model, "features": feats, "config": cfg,
           "trained_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
           "n_train": int(len(X)), "train_max_date": str(df["signal_date"].max())}
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as fh:
        pickle.dump(art, fh)
    _ART_CACHE.update(mtime=None, art=None)
    _SCORE_CACHE.clear()
    _record_registry(art)
    logger.info(f"[ml_ohlcv] pivot model trained on {art['n_train']:,} rows "
                f"/ {len(day_codes)} days (<= {art['train_max_date']}) -> {path}")
    return art


def eod_train(limit_tickers: int = 1500, date_stride: int = 2) -> Optional[dict]:
    """EOD entry point, dispatching on ``settings.ml_ohlcv_target``.

    Pivot mode is THROTTLED by ``ml_pivot_retrain_days`` — the full-universe
    rebuild is heavy and measured staleness of a few days costs ~nothing — and a
    skipped day returns None (the runner logs nothing, correctly). Classic mode
    keeps the original behaviour verbatim, including the ``dataset_multi``
    refresh the offline CLIs ride on."""
    from config.settings import settings

    if settings.ml_ohlcv_target == "pivot_rank":
        art = _load_artifact()
        if art is not None and art.get("config", {}).get("target") == "pivot_rank":
            try:
                trained = datetime.fromisoformat(str(art.get("trained_at")))
                age_days = (datetime.now(timezone.utc) - trained).total_seconds() / 86400.0
                if age_days < float(settings.ml_pivot_retrain_days):
                    logger.debug(f"[ml_ohlcv] pivot artifact {age_days:.1f}d old "
                                 f"(< {settings.ml_pivot_retrain_days}d) — retrain skipped")
                    return None
            except Exception:
                pass                                    # unreadable timestamp -> retrain
        return train_and_persist_pivot(rebuild_dataset=True)

    try:
        from src.analysis.ml_dataset import materialize
        materialize("cache/ml/dataset_multi.parquet", horizons=[1, 3, 5, 10],
                    limit_tickers=limit_tickers, date_stride=date_stride)
    except Exception as e:
        logger.warning(f"[ml_ohlcv] EOD dataset refresh failed, training on prior parquet: {e}")
    return train_and_persist(limit_tickers=limit_tickers, date_stride=date_stride)


def _record_registry(art: dict) -> None:
    """Append a row to the ``ml_models`` registry (as-of/config/metrics). Fail-soft
    — a registry hiccup must not lose the artifact that was just written."""
    try:
        import json
        from src.db.connection import connect
        cfg = art["config"]
        with connect() as con:
            con.execute(
                "INSERT INTO ml_models (trained_at, method, model_type, horizon, basis, "
                "n_train, train_max_date, features, config) VALUES (?,?,?,?,?,?,?,?,?)",
                [art["trained_at"], "ml_ohlcv", cfg["model"],
                 int(cfg["horizon"]), cfg["basis"], art["n_train"],
                 art["train_max_date"], json.dumps(art["features"]), json.dumps(cfg)])
    except Exception as e:
        logger.debug(f"[ml_ohlcv] registry write skipped: {e}")


# ── inference ────────────────────────────────────────────────────────────────

def _load_artifact() -> Optional[dict]:
    if not _MODEL_PATH.exists():
        return None
    try:
        mt = _MODEL_PATH.stat().st_mtime_ns
        if _ART_CACHE["mtime"] == mt:
            return _ART_CACHE["art"]
        with open(_MODEL_PATH, "rb") as fh:
            art = pickle.load(fh)                        # needs lightgbm importable
        _ART_CACHE.update(mtime=mt, art=art)
        _SCORE_CACHE.clear()
        return art
    except Exception as e:                               # missing lightgbm, corrupt pickle
        logger.debug(f"[ml_ohlcv] artifact load failed: {e}")
        _ART_CACHE.update(mtime=None, art=None)
        return None


def compute_ml_score(ticker: str) -> Tuple[float, str]:
    """``(net_score, label)`` for one ticker — ``net = P(up) - P(down)`` ∈ [-1, 1].

    Returns 0.0 with a reason label when the model can't or shouldn't score:
    ``NO_MODEL`` (no artifact), ``NO_DATA`` (too little history), ``NO_VIEW``
    (outside the validated clean-trend/liquid subset), ``ERROR`` (fail-soft).
    """
    art = _load_artifact()
    if art is None:
        return 0.0, "NO_MODEL"
    from src.data.cache import _ohlcv_path
    try:
        mt = _ohlcv_path(ticker, "1d").stat().st_mtime_ns
    except OSError:
        return 0.0, "NO_DATA"
    ck = (ticker, mt)
    hit = _SCORE_CACHE.get(ck)
    if hit is not None:
        return hit
    try:
        from src.analysis.ml_dataset import ticker_feature_frame
        fs = ticker_feature_frame(ticker)
        if fs is None or fs.empty:
            return _memo(ck, 0.0, "NO_DATA")
        row = fs.iloc[-1]
        cfg = art["config"]

        if cfg.get("target") == "pivot_rank":
            # v2: unconditioned by design (the winning model was full-universe;
            # tradeability is Gate 4's job, not the scorer's). The head predicts
            # a centred within-day rank in [-0.5, 0.5]; x2 maps onto the method-
            # score convention. Leg features come from the same completed-bar
            # series as the frame — None means <50 bars, i.e. genuinely no data.
            from src.analysis.pivot_target import latest_leg_features
            leg = latest_leg_features(ticker)
            if leg is None:
                return _memo(ck, 0.0, "NO_DATA")

            def _val(f):
                v = leg.get(f) if f in leg else row.get(f)
                return float(v) if v is not None and v == v else np.nan

            X = np.array([[_val(f) for f in art["features"]]], dtype=float)
            pred = float(art["model"].predict(X)[0])
            net = max(-1.0, min(1.0, 2.0 * pred))
            return _memo(ck, round(net, 4), "OK")

        eff, dv = row.get("eff_ratio"), row.get("dollar_vol_log")
        # v1: only emit a view where the model was validated (its own competence).
        if eff is None or eff != eff or float(eff) < cfg["min_eff_ratio"]:
            return _memo(ck, 0.0, "NO_VIEW")
        if dv is None or dv != dv or float(dv) < cfg["min_dollar_vol_log"]:
            return _memo(ck, 0.0, "NO_VIEW")
        X = np.array([[float(row.get(f)) if row.get(f) is not None and row.get(f) == row.get(f)
                       else np.nan for f in art["features"]]], dtype=float)
        bull, bear = art["model"].bull_bear(X)
        return _memo(ck, round(float(bull[0] - bear[0]), 4), "OK")
    except Exception as e:
        logger.debug(f"[ml_ohlcv] score failed for {ticker}: {e}")
        return _memo(ck, 0.0, "ERROR")


def _memo(key, net: float, label: str) -> Tuple[float, str]:
    _SCORE_CACHE[key] = (net, label)
    if len(_SCORE_CACHE) > 8000:                         # bound the per-run memo
        _SCORE_CACHE.clear()
        _SCORE_CACHE[key] = (net, label)
    return net, label


def reset_caches() -> None:
    """Test hook — drop the artifact + score memos."""
    _ART_CACHE.update(mtime=None, art=None)
    _SCORE_CACHE.clear()


if __name__ == "__main__":  # pragma: no cover
    from src.db import repo
    art = train_and_persist()
    print("trained" if art else "no artifact", art and {k: art[k] for k in
          ("trained_at", "n_train", "train_max_date")})
