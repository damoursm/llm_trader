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
``LightGBMRankRegressor`` on the within-day rank of the signed pivot return —
since 2026-09-16 the next H/L pivot on 30-MINUTE bars after each training
row's session close (``analysis/pivot_target.session_close_labels`` over the
deep store ``data/intraday_store``, 2021→; the daily H/L label is retired) —
85 features (the 76 causal OHLCV features + the 9 DAILY leg-state features:
daily features, intraday label), uniform day-equal weights, trained on the
FULL deep cache — no clean-trend/liquidity conditioning, because the 2026-08
experiments measured the unconditioned full-data model best (and every
weighting scheme ≤ uniform). Score = ``clip(2·pred, −1, 1)``: the head predicts a centred
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
                            rebuild_dataset: bool = False,
                            extend_store: bool = True) -> Optional[dict]:
    """Train the v2 config (within-day-rank GBM of the signed pivot target, full
    universe, uniform day-equal weights) and pickle it.

    The label (2026-09-16, the only one): each deep row is a (ticker, session)
    with DAILY features, and its target is the next H/L pivot on 30-MINUTE bars
    after that session's close — anchored at 16:00 ET with the session close as
    the price, so the search starts at the next session's first bar
    (`pivot_target.session_close_labels`). Daily features, intraday label: the
    label is the next pivot in the subsequent bars whatever bars the features
    read. The 30-minute history comes from the deep store
    (`data/intraday_store`, 2021→, extended first unless ``extend_store`` is
    False), so labelled rows start in 2021; every labelled row's pivot has
    printed by construction — no further point-in-time gate at final-training
    time (the walk-forward cutoffs belong to validation).

    Memory: features come out of DuckDB as float32 into ONE matrix and the
    labels / leg features are streamed per ticker into preallocated vectors —
    a dict-per-row frame over 9M rows was OOM-killed on this box (2026-09-14).
    ~8 GB peak; run it alone, never beside an RTH tick."""
    import gc
    import time

    import duckdb
    import pandas as pd
    from src.analysis.ml_dataset import (ALL_FEATURE_COLUMNS, materialize,
                                         missing_feature_columns)
    from src.analysis.ml_train import LightGBMRankRegressor
    from src.analysis.pivot_target import (LEG_FEATURES, _series, leg_feature_rows,
                                           pivot_basis, session_close_labels, within_day_rank)
    from src.data.intraday_store import deep_series_30m, extend_deep_30m

    t0 = time.time()
    pq = Path(deep_parquet)

    def _cols() -> list:
        con = duckdb.connect()
        try:
            return [c[0] for c in con.sql(f"DESCRIBE SELECT * FROM '{pq.as_posix()}'").fetchall()]
        finally:
            con.close()

    cols = _cols() if pq.exists() else []
    stale = missing_feature_columns(pd.DataFrame(columns=cols)) if cols else list(ALL_FEATURE_COLUMNS)
    if stale and cols:
        logger.warning(f"[ml_ohlcv] {deep_parquet} lacks {len(stale)} current feature "
                       f"columns (e.g. {stale[:3]}) — rebuilding")
    if rebuild_dataset or not cols or stale:
        try:
            materialize(deep_parquet, horizons=[5, 10], date_stride=1)
            cols = _cols()
        except Exception as e:
            logger.warning(f"[ml_ohlcv] full dataset rebuild failed ({e}); "
                           f"training on the prior parquet if any")
    if not cols:
        logger.warning("[ml_ohlcv] no deep dataset to train the pivot model on")
        return None

    feats0 = [f for f in ALL_FEATURE_COLUMNS if f in cols]
    feats = feats0 + list(LEG_FEATURES)
    sel = ", ".join(f'CAST("{f}" AS FLOAT) AS "{f}"' for f in feats0)
    con = duckdb.connect()
    try:
        tickers = [r[0] for r in con.sql(
            f"SELECT DISTINCT ticker FROM '{pq.as_posix()}' ORDER BY ticker").fetchall()]
        arr = con.sql(f"""WITH k AS (SELECT ticker, ROW_NUMBER() OVER (ORDER BY ticker) - 1 AS code
                                     FROM (SELECT DISTINCT ticker FROM '{pq.as_posix()}'))
                          SELECT k.code::INTEGER AS code, CAST(p.signal_date AS DATE) AS signal_date, {sel}
                          FROM '{pq.as_posix()}' p JOIN k USING (ticker)
                          ORDER BY k.code, signal_date""").fetchnumpy()
    finally:
        con.close()
    code = np.asarray(arr.pop("code"), dtype=np.int32)
    sd = np.asarray(arr.pop("signal_date"))
    sd_iso = np.asarray(pd.to_datetime(sd).strftime("%Y-%m-%d"))
    n = len(code)
    X = np.empty((n, len(feats)), dtype=np.float32)
    for j, f in enumerate(feats0):
        X[:, j] = np.asarray(arr.pop(f), dtype=np.float32)
    del arr
    gc.collect()
    X[:, len(feats0):] = np.nan
    y = np.full(n, np.nan, dtype=np.float64)
    logger.info(f"[ml_ohlcv] features {X.shape} float32 over {len(tickers)} tickers "
                f"loaded in {time.time() - t0:.0f}s")

    if extend_store:
        try:
            extend_deep_30m(tickers, workers=4, budget_seconds=1800.0, min_age_days=3)
        except Exception as e:
            logger.warning(f"[ml_ohlcv] deep 30-minute store extension failed ({e}) — "
                           f"training on the stored history")

    starts = np.searchsorted(code, np.arange(len(tickers)), side="left")
    ends = np.searchsorted(code, np.arange(len(tickers)), side="right")
    n_leg = n_lab = n_30 = 0
    for k, tk in enumerate(tickers):
        a, b = int(starts[k]), int(ends[k])
        if b <= a:
            continue
        s = _series(tk)
        if s is None:
            continue
        idx_d, c_d, h_d, lo_d = s
        pos = {d: i for i, d in enumerate(sd_iso[a:b])}       # ISO keys on both sides
        for rec in leg_feature_rows(c_d, h_d, lo_d, list(idx_d)):
            i = pos.get(rec["signal_date"])
            if i is None:
                continue
            for j, f in enumerate(LEG_FEATURES):
                v = rec.get(f)
                X[a + i, len(feats0) + j] = np.float32(v) if v is not None and v == v else np.nan
            n_leg += 1
        s30 = deep_series_30m(tk)
        if s30 is None:
            continue
        n_30 += 1
        sp, _end = session_close_labels(*s30, list(idx_d), c_d)
        for d, v in zip(idx_d, sp):
            if v == v:
                i = pos.get(d.isoformat())
                if i is not None:
                    y[a + i] = float(v)
                    n_lab += 1
        if (k + 1) % 500 == 0:
            logger.info(f"[ml_ohlcv] labels {k + 1}/{len(tickers)} tickers | "
                        f"{n_lab:,} settled | {time.time() - t0:.0f}s")
    logger.info(f"[ml_ohlcv] leg rows {n_leg:,} | 30-minute history for {n_30} tickers | "
                f"settled labels {n_lab:,} of {n:,} rows | {time.time() - t0:.0f}s")

    m = np.isfinite(y)
    if int(m.sum()) < 50000:
        logger.warning(f"[ml_ohlcv] insufficient pivot training rows ({int(m.sum()):,})")
        return None
    Xm = np.ascontiguousarray(X[m])
    ym = y[m]
    dm = sd_iso[m]
    del X, y
    gc.collect()
    day_codes, day_idx = np.unique(dm, return_inverse=True)
    yr = within_day_rank(ym, day_idx.astype(np.int32))
    cnt = np.bincount(day_idx)
    w = (1.0 / cnt[day_idx]).astype(np.float64)
    w /= w.mean()

    cfg = dict(TRAIN_CONFIG_PIVOT)
    cfg["pivot_basis"] = pivot_basis()        # marks + threshold + resolution; serving refuses a mismatch
    cfg["pivot_resolution"] = "30m"
    cfg["label"] = "next_30m_pivot_from_session_close"
    model = LightGBMRankRegressor(num_threads=cfg["num_threads"]).fit(Xm, yr, w)
    art = {"model": model, "features": feats, "config": cfg,
           "trained_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
           "n_train": int(len(Xm)), "n_days": int(len(day_codes)),
           "train_min_date": str(day_codes[0]), "train_max_date": str(day_codes[-1])}
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as fh:
        pickle.dump(art, fh)
    _ART_CACHE.update(mtime=None, art=None)
    _SCORE_CACHE.clear()
    _record_registry(art)
    logger.info(f"[ml_ohlcv] pivot model trained on {art['n_train']:,} rows "
                f"/ {len(day_codes)} days ({art['train_min_date']}..{art['train_max_date']}) "
                f"basis {cfg['pivot_basis']} -> {path} in {time.time() - t0:.0f}s")
    return art


def eod_train(limit_tickers: int = 1500, date_stride: int = 2,
              force: bool = False) -> Optional[dict]:
    """EOD entry point, dispatching on ``settings.ml_ohlcv_target``.

    Pivot mode is THROTTLED by ``ml_pivot_retrain_days`` — the full-universe
    rebuild is heavy and measured staleness of a few days costs ~nothing — and a
    skipped day returns None (the runner logs nothing, correctly). Classic mode
    keeps the original behaviour verbatim, including the ``dataset_multi``
    refresh the offline CLIs ride on."""
    from config.settings import settings

    if settings.ml_ohlcv_target == "pivot_rank":
        art = _load_artifact()
        if force:
            art = None                      # weekly caller: throttle bypassed
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


_BASIS_WARNED = False


def compute_ml_score(ticker: str) -> Tuple[float, str]:
    """``(net_score, label)`` for one ticker — ``net = P(up) - P(down)`` ∈ [-1, 1].

    Returns 0.0 with a reason label when the model can't or shouldn't score:
    ``NO_MODEL`` (no artifact), ``NO_DATA`` (too little history), ``NO_VIEW``
    (outside the validated clean-trend/liquid subset), ``ERROR`` (fail-soft).
    """
    art = _load_artifact()
    if art is None:
        return 0.0, "NO_MODEL"
    # 30-MINUTE ARTIFACTS (2026-09-18) take their own path. Dispatching on the
    # artifact's own stamp rather than on a setting is the same discipline as
    # the basis guard: the model decides how it must be fed. Without this a
    # 30-minute model would be served the DAILY frame and score silently wrong
    # — no error, no abstention, just a different distribution than it was fit
    # on, which is the failure mode that looks exactly like a working deploy.
    if str((art.get("config") or {}).get("feature_bars", "")).lower() == "30m":
        return _score_30m(ticker, art)
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
            #
            # BASIS GUARD (2026-08-12): the pivot definition moved to the H/L
            # basis; an artifact trained on the close basis would be fed leg
            # features it never saw. Abstain (a method with no view) until the
            # retrain writes a matching artifact — degraded, never wrong.
            from src.analysis.pivot_target import latest_leg_features, pivot_basis
            # The artifact must be stamped with the label basis IN FORCE
            # (marks + threshold + resolution, e.g. hl1@30m): a model trained
            # on a retired label — the daily H/L one, another threshold — is fed
            # leg features that still mean what it learned but was fitted to a
            # target that no longer exists here. Abstain (a method with no
            # view) until the retrain writes a matching artifact — degraded,
            # never wrong.
            _expected = pivot_basis()
            if cfg.get("pivot_basis") != _expected:
                global _BASIS_WARNED
                if not _BASIS_WARNED:
                    logger.warning(
                        f"[ml_ohlcv] artifact pivot_basis={cfg.get('pivot_basis')!r} != "
                        f"current {_expected!r} — abstaining until the retrain lands")
                    _BASIS_WARNED = True
                return _memo(ck, 0.0, "BASIS_STALE")
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


_MIN_30M_BARS = 400          # ~31 sessions; the longest causal window is ret_126


_FEAT_CACHE: dict = {}


def features_30m(ticker: str, now=None):
    """``(frame, label)`` — the 30-MINUTE feature row a 30-minute artifact is
    served on, for the last bar whose 30 minutes have ELAPSED at ``now``
    (naive UTC; default the clock). ``frame`` is None with a reason label
    (``NO_DATA`` / ``ERROR``) when there is no usable row, else a dict:

      ``features``  every `ticker_feature_frame` column at that bar, the 9 leg
                    features (computed on the series CUT at the bar) over them;
      ``bar_ts``    the bar's start (naive UTC), ``close`` its close,
      ``sday`` / ``bar_idx``  its session day and index within the session
                    (what `deep_features.serving_vector` needs), ``n_bars``.

    Shared by `_score_30m` and the live feature capture
    (`src/analysis/live_features.py`, 2026-09-25) — one computation, so the
    captured vector IS the one the model was served. Memoised per (ticker,
    half-hour slot, tick-cache mtime), like the score."""
    import pandas as pd
    if now is None:
        now = pd.Timestamp(datetime.now(timezone.utc)).tz_localize(None)
    else:
        now = pd.Timestamp(now)
        if now.tzinfo is not None:
            now = now.tz_convert("UTC").tz_localize(None)
    slot = int(now.value // (1800 * 10 ** 9))
    try:
        from src.data.cache import _ohlcv_path
        mt = _ohlcv_path(ticker, "30m").stat().st_mtime_ns
    except Exception:
        mt = 0
    ck = (ticker, slot, mt)
    hit = _FEAT_CACHE.get(ck)
    if hit is not None:
        return hit
    try:
        from src.analysis.ml_dataset import hlc_30m
        hlc = hlc_30m(ticker)
    except Exception as e:
        logger.debug(f"[ml_ohlcv] 30m bars failed for {ticker}: {e}")
        return _memo_feat(ck, None, "ERROR")
    frame, label = features_30m_from_hlc(ticker, hlc, now)
    return _memo_feat(ck, frame, label)


def features_30m_from_hlc(ticker: str, hlc, now):
    """`features_30m` on bars the CALLER supplies (``hlc`` in `ml_dataset.hlc_30m`'s
    shape; ``now`` naive UTC) — no memo. The selection-short scorer
    (`src/signals/sel_short.py`) serves ~2,000 names whose freshest bars it
    fetches itself, the tick cache holding only the names a tick touches; one
    function, so its vectors are the ones `features_30m` would compute."""
    import pandas as pd
    try:
        from src.analysis.pivot_target import LEG_FEATURES, leg_feature_rows
        from src.analysis.ml_dataset import ticker_feature_frame
        if hlc is None:
            return None, "NO_DATA"
        idx, high, low, close, volume = hlc
        # Completed bars only: a bar STARTING at t is usable from t+30m.
        n_vis = int(((idx + pd.Timedelta(minutes=30)) <= now).sum())
        if n_vis < _MIN_30M_BARS:
            return None, "NO_DATA"
        cut = (idx[:n_vis], high.iloc[:n_vis], low.iloc[:n_vis],
               close.iloc[:n_vis], volume.iloc[:n_vis])
        fs = ticker_feature_frame(ticker, hlc=cut)
        if fs is None or fs.empty:
            return None, "NO_DATA"
        row = fs.iloc[-1]

        legs = leg_feature_rows(
            close.iloc[:n_vis].to_numpy(dtype=float),
            high.iloc[:n_vis].to_numpy(dtype=float),
            low.iloc[:n_vis].to_numpy(dtype=float),
            list(idx[:n_vis]), only_last=True)
        if not legs:
            return None, "NO_DATA"
        leg = {f: float(legs[0][f]) for f in LEG_FEATURES
               if f in legs[0] and legs[0][f] == legs[0][f]}
        feats = {str(k): v for k, v in row.items()}
        feats.update(leg)
        from src.analysis import deep_features as dfe
        sdays = dfe.session_days(idx[:n_vis])
        sday = int(sdays[-1])
        frame = {"features": feats, "bar_ts": pd.Timestamp(idx[n_vis - 1]),
                 "close": float(close.iloc[n_vis - 1]), "sday": sday,
                 "bar_idx": int((sdays == sday).sum()) - 1, "n_bars": n_vis}
        return frame, "OK"
    except Exception as e:
        logger.debug(f"[ml_ohlcv] 30m features failed for {ticker}: {e}")
        return None, "ERROR"


def _memo_feat(key, frame, label: str):
    _FEAT_CACHE[key] = (frame, label)
    if len(_FEAT_CACHE) > 8000:                          # bound the per-run memo
        _FEAT_CACHE.clear()
        _FEAT_CACHE[key] = (frame, label)
    return frame, label


def _score_30m(ticker: str, art: dict) -> Tuple[float, str]:
    """Serve an artifact trained on 30-MINUTE feature rows.

    Three things differ from the daily path and all three are load-bearing:

      * the feature frame is built on 30-minute regular-hours bars
        (`ml_dataset.hlc_30m`) rather than daily sessions;
      * the row is the last bar whose 30 minutes have ELAPSED, so the score
        moves through the session instead of being frozen at the previous
        close, and the series is CUT there before the leg features are computed
        — a leg state read off the full series would see bars the tick cannot;
      * the memo is keyed on the half-hour slot as well as the cache file, since
        keying on the daily file's mtime (what the daily path does) would serve
        one stale intraday score for the rest of the session.

    The feature row itself comes from `features_30m` (shared with the live
    feature capture, so the captured vector is the served one).
    """
    import pandas as pd                      # lazy, like every other import here
    cfg = art["config"]
    now = pd.Timestamp(datetime.now(timezone.utc)).tz_localize(None)
    slot = int(now.value // (1800 * 10 ** 9))
    try:
        from src.data.cache import _ohlcv_path
        mt = _ohlcv_path(ticker, "30m").stat().st_mtime_ns
    except Exception:
        mt = 0
    ck = (ticker, "30m", slot, mt)
    hit = _SCORE_CACHE.get(ck)
    if hit is not None:
        return hit
    try:
        from src.analysis.pivot_target import pivot_basis
        expected = pivot_basis()
        if cfg.get("pivot_basis") != expected:
            global _BASIS_WARNED
            if not _BASIS_WARNED:
                logger.warning(
                    f"[ml_ohlcv] artifact pivot_basis={cfg.get('pivot_basis')!r} != "
                    f"current {expected!r} — abstaining until the retrain lands")
                _BASIS_WARNED = True
            return _memo(ck, 0.0, "BASIS_STALE")

        frame, label = features_30m(ticker, now)
        if frame is None:
            return _memo(ck, 0.0, label)
        feats = frame["features"]

        # DEEP FEATURES (2026-09-23): an artifact trained with the deep store's
        # point-in-time features (`dp_*`) is fed them from the SESSION SNAPSHOT
        # of the last completed bar's session — built at 08:30 ET by the
        # pre-open run, the same cutoff every training row of that session saw
        # — plus the bar features from this bar. A missing snapshot means the
        # pre-open run did not land: abstain (and build it in the background)
        # rather than score a model on inputs it never trained with. A missing
        # snapshot for an OLDER session means this name's tick cache stopped
        # there — abstain without building: the tick path never builds history.
        deep = {}
        if any(str(f).startswith("dp_") for f in art["features"]):
            from src.analysis import deep_features as dfe
            sday = frame["sday"]
            snapshot = dfe.load_session_snapshot(sday)
            if snapshot is None:
                if sday not in dfe.recent_session_days():
                    return _memo(ck, 0.0, "DEEP_STALE")
                dfe.trigger_snapshot_build(sday)
                return _memo(ck, 0.0, "DEEP_PENDING")
            deep = dfe.serving_vector(ticker, sday, frame["close"], frame["bar_idx"], snapshot)

        def _val(f):
            v = deep[f] if f in deep else feats.get(f)
            return float(v) if v is not None and v == v else np.nan

        X = np.array([[_val(f) for f in art["features"]]], dtype=float)
        pred = float(art["model"].predict(X)[0])
        return _memo(ck, round(max(-1.0, min(1.0, 2.0 * pred)), 4), "OK")
    except Exception as e:
        logger.debug(f"[ml_ohlcv] 30m score failed for {ticker}: {e}")
        return _memo(ck, 0.0, "ERROR")


def _memo(key, net: float, label: str) -> Tuple[float, str]:
    _SCORE_CACHE[key] = (net, label)
    if len(_SCORE_CACHE) > 8000:                         # bound the per-run memo
        _SCORE_CACHE.clear()
        _SCORE_CACHE[key] = (net, label)
    return net, label


def reset_caches() -> None:
    """Test hook — drop the artifact, score and feature memos."""
    _ART_CACHE.update(mtime=None, art=None)
    _SCORE_CACHE.clear()
    _FEAT_CACHE.clear()


if __name__ == "__main__":  # pragma: no cover
    import argparse

    p = argparse.ArgumentParser(
        description="ml_ohlcv ad-hoc retrain (see CLAUDE.md — Ad-hoc ML retrains)")
    p.add_argument("--throttled", action="store_true",
                   help="honour ml_pivot_retrain_days instead of forcing the retrain")
    a = p.parse_args()
    # Dispatch through `eod_train` so the CONFIGURED generation trains. This
    # block used to call `train_and_persist()` directly, which always trained the
    # v1 CLASSIC model regardless of `ml_ohlcv_target` — and since serving
    # dispatches on the artifact's OWN config, an ad-hoc retrain silently
    # DOWNGRADED a weighted method from v2 pivot to v1 with no error anywhere.
    # Forced by default: the throttle exists to pace the automated weekly caller,
    # and someone typing this command has already decided to pay for it.
    art = eod_train(force=not a.throttled)
    print("trained" if art else "no artifact (throttled or failed)",
          art and {k: art[k] for k in ("trained_at", "n_train", "train_max_date")})
