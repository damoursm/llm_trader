"""ml_buy — the FULL STACKER: a model whose features are ALL the per-method
scores from the signals panel, predicting the BUY outcome P(up).

Role B of the ML plan (the learned aggregator), buy-side. Where ``ml_ohlcv`` is a
NEW base signal from price/volume, this is a META-model over the EXISTING methods
— the learned counterpart to ``combined_buy_score`` — which can capture
interactions the hand-weighted linear combine cannot (momentum working only in a
clean trend, news mattering only when volume confirms, ...).

**Data: the signals PANEL only.** The method scores (news, options, sentiment,
insider, ...) are NOT replayable — they need point-in-time feeds nobody stored —
so unlike ml_ohlcv this cannot use the deep cache. It trains on the forward-
collected panel: tens of thousands of ticker-days but only WEEKS of distinct
days, so OVERFITTING to the single regime is the real risk and the panel
walk-forward is the honest judge. Nothing here is trusted until the forward IC
accrues. (Measured 2026-08-11: 20x row inflation from keeping every intraday
run adds nothing — same days, shared labels — and small-data GBM params beat
the lgb defaults at t +2.0; see STACKER_GBM_PARAMS.)

**Circularity — the load-bearing guard.** Features are the individual METHOD
scores, which do NOT depend on the weights, so training a model on them is sound.
``combined_score`` / ``combined_buy_score`` / ``confidence`` DO depend on the
weights and are EXCLUDED from the features (using them would fit the model on
values derived from the very weights it exists to inform). ``combined_score`` is
kept as a BASELINE — the bar the learned stacker must beat to be worth anything.

**Output: P(up) ∈ [0,1]** = the buy conviction (the two-sided [0,1] convention).
For panel IC-tracking it is emitted as the signed ``2*P(up)-1`` so it ranks
alongside the other methods' signed scores; when promoted it contributes P(up)
to the buy camp of the split combine.

CLI:  python -m src.analysis.ml_stacker [--horizons 1,5,10] [--model gbm]
"""

from __future__ import annotations

import argparse
from bisect import bisect_left
from datetime import date
from typing import List, Optional, Sequence

import numpy as np
import pandas as pd
from loguru import logger

from config.settings import settings
from src.analysis.ml_dataset import _benchmark_return, _benchmark_series
from src.analysis.ml_train import evaluate
from src.db.schema import SIGNAL_BASE_METHOD_COLUMNS

# The stacker's features: every individual method score. All weight-INDEPENDENT
# (combined_score/confidence are NOT here — that is the circularity guard). The
# panel persists exactly these columns, so the dataset is essentially build_panel.
# Plus `tape_score` (2026-08-22): the score-independent price/volume tape
# composite — a replay CONTEXT column, not a method column. The panel gets it
# from the signals_replay merge; serving computes it live (the same
# `compute_tape_confirmation` the confidence factor uses). Measured as the one
# robust feature ADDITION of the 27-arm redesign: +0.0175 IC/day over the same
# model without it (t +2.72, wins 72% of days, same-sign halves) — until now the
# tape's direction was only a confidence qualifier, never scored.
STACKER_FEATURES: List[str] = list(SIGNAL_BASE_METHOD_COLUMNS) + ["tape_score"]

# The LIVE feature set — 21 of the weighted methods in the aggregator's
# `method_score_map`. DELIBERATELY NOT the full post-2026-08-11 set of 27: the
# six promoted methods (mom_12_1/hi52/st_reversal/rsi2_rev/dloc_rev/ml_ohlcv)
# were measured as features on 2026-08-11 (scratchpad stacker_tune.py) and
# HURT — their panel history is mostly-NaN through the training windows (epochs
# + method age) while populated at serve time, a train/serve distribution shift
# (cur config 27f vs 21f: t −1.79). REVISIT once their panel history thickens
# (~2026-09); the extension is design-correct then. Until that re-test, the
# length-21 pin in tests/test_ml_stacker.py is the record of this decision.
STACKER_LIVE_FEATURES: List[str] = [
    "news", "sent_velocity", "tech", "massive", "insider", "put_call", "max_pain",
    "oi_skew", "vwap", "pattern", "momentum", "sector_momentum", "market_momentum",
    "money_flow", "trend_strength", "pead", "iv_rank", "iv_expr", "coint",
    "ext_gap", "broker_advisor",
    # 22nd method (2026-08-24, user directive): the deep-cache price model's own
    # score. Panel coverage is ~7% (epoch-masked before 2026-08-13), so today it
    # is near-inert (the exit model measured byte-identical predictions at the
    # same coverage) — it self-activates as post-epoch history accrues. The
    # OTHER five 2026-08-11 promoted methods stay excluded (measured harmful as
    # a six-pack at t −1.79; re-test ~2026-09-10). Shared with the exit model
    # via EXIT_METHODS' derivation.
    "ml_ohlcv",
    # 23rd feature (2026-08-22, NOT a method column — see STACKER_FEATURES note):
    "tape_score",
]

# Kept as columns for the BASELINE comparison (NOT fed to the model): the current
# hand-weighted combine is the bar a learned stacker must clear to justify itself.
# Deliberately NOT a method score (those are features) — these are the aggregate
# combine columns, which the stacker never sees but must beat.
_BASELINE_COLUMNS = ("combined_score", "combined_buy_score")


def build_stacker_dataset(horizons: Sequence[int] = (1, 5, 10), days: Optional[int] = None,
                          benchmark: Optional[str] = None,
                          dedupe: str = "last") -> pd.DataFrame:
    """One row per (ticker, signal_date) from the panel: the method-score features
    + raw/market-relative forward labels + ``end_date_<h>d`` for the walk-forward
    point-in-time split. Same schema ``ml_train.evaluate`` consumes.

    The forward return is the panel's own (authoritative) ``fwd_ret_<h>d``; the
    market-relative leg and the horizon END DATE use the benchmark's session grid
    (SPY trades every session, so it is a clean universal calendar for the cutoff).
    """
    from src.analysis.signal_panel import build_panel
    benchmark = benchmark or settings.horizon_market_benchmark
    horizons = list(horizons)
    # ``dedupe="all"`` keeps every intraday run's row (~5-8x rows with shared
    # same-day labels — callers weighting by day should day-normalise); the
    # default "last" keeps the one-row-per-(day, ticker) contract unchanged.
    panel = build_panel(horizons=horizons, days=days, dedupe=dedupe)
    if panel is None or panel.empty:
        return pd.DataFrame()

    feats = [c for c in STACKER_FEATURES if c in panel.columns]
    if "tape_score" not in feats:
        # The 22nd feature comes from the signals_replay merge; a panel without
        # it means the replay materialisation is missing/stale. Training would
        # silently fit a 21-feature model — loud, because that is invisible.
        logger.warning("[ml_stacker] panel has no tape_score column — replay "
                       "materialisation missing? Training will drop the feature.")
    base = [c for c in _BASELINE_COLUMNS if c in panel.columns]
    # combined_sell_score is kept (not a feature, not a baseline) so the swap
    # validation can form combined_score = swapped_buy - combined_sell_score.
    extra = [c for c in ("combined_sell_score",) if c in panel.columns]
    fwd_cols = [f"fwd_ret_{h}d" for h in horizons if f"fwd_ret_{h}d" in panel.columns]
    # The pivot label + its per-row settle date ride along whenever the panel
    # carries them (2026-08-12) — the rank block below turns them into
    # `fwd_ret_rank_pv`/`end_date_pv`, the stackers' default training label.
    fwd_cols += [c for c in ("fwd_ret_pivot", "end_date_pivot") if c in panel.columns]
    # dict.fromkeys dedupes while preserving order — a column that is both a
    # feature and a baseline must not be selected twice (a duplicate column makes
    # df[col] 2-D and breaks the downstream metrics).
    cols = list(dict.fromkeys(["signal_date", "ticker"] + feats + base + extra + fwd_cols))
    df = panel[cols].copy()

    b_dates, b_closes = _benchmark_series(benchmark)

    def _enrich(row) -> pd.Series:
        out = {}
        D = date.fromisoformat(str(row["signal_date"])[:10])
        i = bisect_left(b_dates, D) if b_dates else 0
        for h in horizons:
            raw = row.get(f"fwd_ret_{h}d")
            raw = float(raw) if raw is not None and pd.notna(raw) else np.nan
            out[f"fwd_ret_raw_{h}d"] = raw
            end = b_dates[i + h] if (b_dates and i < len(b_dates) and i + h < len(b_dates)) else None
            out[f"end_date_{h}d"] = end.isoformat() if end is not None else None
            bench = (_benchmark_return(b_dates, b_closes, b_dates[i], end)
                     if (end is not None and i < len(b_dates)) else None)
            out[f"fwd_ret_rel_{h}d"] = (raw - bench) if (bench is not None and raw == raw) else np.nan
        return pd.Series(out)

    df = pd.concat([df.reset_index(drop=True), df.apply(_enrich, axis=1).reset_index(drop=True)], axis=1)

    # CROSS-SECTIONAL RANK label (2026-08-04) — the per-day percentile of the raw
    # forward return, centred at 0 (so ``> 0`` = beat that day's median name).
    #
    # This is the target the LIVE combine trains on, for two measured reasons:
    #
    # 1. ``rel`` (raw − benchmark) implicitly assumes EVERY stock has beta = 1.0,
    #    so it injects noise ∝ (β−1)×market_return. The rank removes the day's
    #    common factor non-parametrically, assuming no beta at all.
    # 2. **It makes probability calibration valid.** The combine ranks WITHIN a day
    #    and takes the top N, so the quantity that matters is the per-day IC. A
    #    label carrying day-level drift (``rel``) has pooled IC ≪ its daily IC
    #    (+0.0035 vs +0.0395 measured), so a calibrator fitted on POOLED
    #    predictions maps everything to the pooled base rate and destroys exactly
    #    the within-day ranking that carries the signal (conviction >0.15 went
    #    27.6% → 0.0% of the universe). The rank label is per-day demeaned BY
    #    CONSTRUCTION, so pooled ≈ daily (+0.0252 vs +0.0263) and the pooled
    #    calibrator becomes legitimate rather than destructive.
    #
    # Causality: the label ranks forward returns that all realise at the same
    # time, so this adds no look-ahead beyond the label horizon itself.
    for h in horizons:
        raw_c = f"fwd_ret_raw_{h}d"
        if raw_c in df.columns:
            df[f"fwd_ret_rank_{h}d"] = (df.groupby("signal_date")[raw_c]
                                          .rank(pct=True, method="average") - 0.5)

    # PIVOT label (2026-08-12, user directive): the within-day centred rank of
    # the SIGNED PIVOT TARGET — the ml_ohlcv-v2 objective, now the stackers'
    # default label (``stacker_label_basis``). Settled rows only (an unsettled
    # pivot is NaN and drops out of both the rank and the training set);
    # ``end_date_pv`` carries each row's OWN settle date so the walk-forward
    # trains strictly on printed labels.
    if "fwd_ret_pivot" in df.columns:
        df["fwd_ret_rank_pv"] = (df.groupby("signal_date")["fwd_ret_pivot"]
                                   .rank(pct=True, method="average") - 0.5)
        if "end_date_pivot" in df.columns:
            df["end_date_pv"] = df["end_date_pivot"]

    logger.info(f"[ml_stacker] {len(df):,} rows over {df['ticker'].nunique()} tickers "
                f"({df['signal_date'].min()} .. {df['signal_date'].max()}), "
                f"{len(feats)} method features")
    return df


def measure(horizons: Sequence[int] = (1, 5, 10), bases: Sequence[str] = ("raw", "rel"),
            deadband: float = 0.0, model_name: str = "gbm", days: Optional[int] = None,
            min_train_days: int = 8, step_days: int = 2) -> pd.DataFrame:
    """Walk-forward the stacker over the panel and return the go/no-go table —
    the learned model's OOS IC/ICIR/hit/simret vs the BASELINES (the hand-weighted
    combined_score it must beat). Thin by construction (~34 panel days)."""
    df = build_stacker_dataset(horizons=horizons, days=days)
    if df.empty:
        return pd.DataFrame()
    return evaluate(df, horizons=horizons, bases=bases, deadband=deadband,
                    model_name=model_name, baseline_features=_BASELINE_COLUMNS,
                    features=STACKER_FEATURES, min_train_days=min_train_days,
                    step_days=step_days, min_train_rows=1000)


def buy_conviction_from_proba(p_up: float) -> float:
    """Map the stacker's P(up) to a combined_buy_score-compatible conviction in
    [0,1]: 0 at neutral (P=0.5), 1 at certain (P=1). Without this centering a raw
    P(up)~0.5 for a neutral name would swamp combined_sell_score~0 and make the
    whole universe look like a strong buy."""
    return max(0.0, min(1.0, 2.0 * float(p_up) - 1.0))


# ── live inference: the buy stacker AS combined_buy_score ─────────────────────
# The buy aggregator on the within-day PIVOT-rank label (rank_5d fallback —
# `_label_cfg`), trained on the 21 live method features
# and served at the aggregator's combine point. Native Booster (no sklearn), so
# the artifact loads in the production .venv. Fail-soft everywhere: a missing
# artifact / lightgbm returns None and the caller keeps the weighted combine.

import pickle as _pickle
from datetime import datetime as _dt, timezone as _tz
from pathlib import Path as _Path

# Small-data GBM parameters (2026-08-11 retune): the classifier's lgb defaults
# (31 leaves / min_child 200) were sized for the deep cache, not a ~20k-row
# panel. The 13-arm walk-forward retune (scratchpad stacker_tune.py, 29 OOS
# days, paired per-day diffs) has this config at rank-5d IC +0.0611 vs the
# defaults' +0.0463 (t +2.00, the pre-registered bar) with the best day-
# stability of any arm (ICIR +0.75); two sibling small-capacity arms corroborate
# the direction at t +1.5-1.7. Also measured and REJECTED in the same run: the
# 27-feature extension (the six 2026-08-11 promotions are mostly-NaN through
# the training windows -> distribution shift; revisit when their panel history
# thickens) and all-runs row inflation (20x rows of the same days, t +0.6).
# Used by BOTH stackers AND the calibrator's OOF walk, so the calibration curve
# is fit on the same model class it corrects.
STACKER_GBM_PARAMS = dict(num_leaves=15, min_child_samples=20,
                          learning_rate=0.05, n_estimators=200)


def stacker_model_class() -> str:
    """The configured model class, normalised. Unknown values fall back to
    "logistic" (the measured default) rather than erroring — the revert knob is
    for operators, and a typo must not kill training."""
    v = str(getattr(settings, "stacker_model_class", "logistic")).strip().lower()
    return v if v in ("logistic", "gbm") else "logistic"


def _stacker_model_factory():
    """One factory for BOTH stackers AND the calibrator's OOF walk (the
    calibration curve must be fit on the same model class it corrects).
    "logistic" = SoftmaxLogistic (2026-08-22 default — see the setting's note);
    "gbm" reverts to the small-data LightGBM classifier."""
    if stacker_model_class() == "gbm":
        from src.analysis.ml_train import LightGBMModel
        return LightGBMModel(**STACKER_GBM_PARAMS)
    from src.analysis.ml_train import SoftmaxLogistic
    return SoftmaxLogistic()


_BUY_MODEL_PATH = _Path("cache/ml/ml_buy_model.pkl")
BUY_TRAIN_CONFIG = dict(horizon=5, basis="rank", deadband=0.0)
_BUY_ART: dict = {"mtime": None, "art": None}


def _fit_calibrator(df: pd.DataFrame, horizon: int, basis: str, feats: Sequence[str],
                    deadband: float, min_train_days: int = 8, step_days: int = 2):
    """Fit an ``IsotonicCalibrator`` on WALK-FORWARD OUT-OF-FOLD predictions.

    The OOF requirement is load-bearing, not a nicety: a calibrator fit on the
    model's own training rows learns the curve of a model that has already
    memorised them, which looks calibrated and is not. So this re-runs the same
    point-in-time walk-forward the evaluation uses and calibrates on predictions
    the model never trained on.

    Returns ``(calibrator, diagnostics)``; ``(None, {})`` when the panel is too
    thin to produce usable OOF predictions (caller then stores no calibrator and
    inference falls back to the raw probability)."""
    from src.analysis.ml_train import (IsotonicCalibrator, brier, brier_skill,
                                       make_model_factory, walk_forward_predict)
    try:
        oof = walk_forward_predict(df, horizon, basis, features=list(feats),
                                   deadband=deadband, min_train_days=min_train_days,
                                   step_days=step_days, min_train_rows=1000,
                                   model_factory=_stacker_model_factory)
    except Exception as e:
        logger.warning(f"[ml_calib] walk-forward for calibration failed: {e}")
        return None, {}
    if oof is None or oof.empty or "bull" not in oof.columns:
        return None, {}
    oof = oof[oof["fwd"].notna()]
    if len(oof) < 200:
        logger.warning(f"[ml_calib] only {len(oof)} OOF rows — not calibrating")
        return None, {}
    raw = pd.to_numeric(oof["bull"], errors="coerce").to_numpy(dtype=float)
    # The outcome the model's "bull" class asserts: the basis return cleared the
    # deadband upward. (For the sell model the basis is already negated, so this
    # reads "the short worked" — same code, mirrored label.)
    y = (pd.to_numeric(oof["fwd"], errors="coerce").to_numpy(dtype=float) > deadband).astype(int)
    cal = IsotonicCalibrator().fit(raw, y)
    if cal.n_fit_ == 0:
        return None, {}
    p_cal = cal.transform(raw)
    diag = {"n_oof": int(len(raw)), "base_rate": round(float(y.mean()), 4),
            "brier_raw": round(brier(raw, y), 4), "brier_cal": round(brier(p_cal, y), 4),
            "skill_raw": round(brier_skill(raw, y), 4),
            "skill_cal": round(brier_skill(p_cal, y), 4),
            "raw_range": [round(float(raw.min()), 4), round(float(raw.max()), 4)],
            "cal_range": [round(float(p_cal.min()), 4), round(float(p_cal.max()), 4)]}
    logger.info(f"[ml_calib] fitted on {diag['n_oof']:,} OOF rows — Brier {diag['brier_raw']} → "
                f"{diag['brier_cal']} (skill {diag['skill_raw']:+} → {diag['skill_cal']:+}); "
                f"P range {diag['raw_range']} → {diag['cal_range']}")
    return cal, diag


def _calibrate(art: dict, p: float) -> float:
    """Apply the artifact's calibrator to a raw probability. Fail-soft: no
    calibrator (or the setting off) returns the raw value unchanged."""
    if not settings.enable_ml_probability_calibration:
        return p
    cal = art.get("calibrator")
    if cal is None:
        return p
    try:
        return cal.transform_one(p)
    except Exception:
        return p


def _label_cfg(df: pd.DataFrame, cfg: dict, tag: str):
    """Resolve the training label per ``stacker_label_basis`` (2026-08-12):
    ``pivot_rank`` uses the within-day rank of the signed pivot target whenever
    the panel carries enough settled rows; anything else (or a thin pivot
    column) keeps the legacy fixed-horizon rank label. Returns ``(cfg, ycol)``
    with ``cfg["basis"]`` switched to ``rank_pv`` when the pivot label won."""
    want_pivot = str(getattr(settings, "stacker_label_basis", "pivot_rank")).lower() == "pivot_rank"
    if want_pivot and "fwd_ret_rank_pv" in df.columns:
        n = int(pd.to_numeric(df["fwd_ret_rank_pv"], errors="coerce").notna().sum())
        if n >= 500:
            out = dict(cfg)
            out["basis"] = "rank_pv"
            logger.info(f"[{tag}] label: PIVOT rank ({n:,} settled rows)")
            return out, "fwd_ret_rank_pv"
        logger.info(f"[{tag}] pivot label too thin ({n} rows) — falling back to "
                    f"{cfg['basis']}_{cfg['horizon']}d")
    return dict(cfg), f"fwd_ret_{cfg['basis']}_{cfg['horizon']}d"


def train_and_persist_buy(days: Optional[int] = None, path=_BUY_MODEL_PATH) -> Optional[dict]:
    """Train the buy stacker on the panel (5d market-relative) over the 21 live
    method features; pickle it. Returns the artifact or None."""
    import numpy as _np
    from src.analysis.ml_train import label_from_return
    h = BUY_TRAIN_CONFIG["horizon"]
    df = build_stacker_dataset(horizons=[h], days=days)
    if df.empty:
        logger.warning("[ml_buy] no panel data to train on")
        return None
    cfg, ycol = _label_cfg(df, BUY_TRAIN_CONFIG, "ml_buy")
    basis = cfg["basis"]
    if ycol not in df.columns:
        logger.warning("[ml_buy] no panel data to train on")
        return None
    feats = [f for f in STACKER_LIVE_FEATURES if f in df.columns]
    y_raw = df[ycol].map(lambda r: label_from_return(r, cfg["deadband"]))
    keep = y_raw.notna()
    X = df.loc[keep, feats].to_numpy(dtype=float)
    y = y_raw[keep].to_numpy(dtype=int)
    if len(X) < 500 or len(_np.unique(y)) < 2:
        logger.warning(f"[ml_buy] insufficient training rows ({len(X)})")
        return None
    model = _stacker_model_factory().fit(X, y)
    # Calibrate on OUT-OF-FOLD predictions BEFORE the final all-data fit is used
    # live, so the probability the combine consumes means what it says.
    cal, cal_diag = (_fit_calibrator(df, h, cfg["basis"], feats, cfg["deadband"])
                     if settings.enable_ml_probability_calibration else (None, {}))
    art = {"model": model, "features": feats, "config": dict(cfg),
           "model_class": stacker_model_class(),
           "calibrator": cal, "calibration": cal_diag,
           "trained_at": _dt.now(_tz.utc).isoformat(timespec="seconds"),
           "n_train": int(len(X)), "train_max_date": str(df["signal_date"].max())}
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as fh:
        _pickle.dump(art, fh)
    _BUY_ART.update(mtime=None, art=None)
    _record_buy_registry(art)
    logger.info(f"[ml_buy] trained on {art['n_train']:,} rows (<= {art['train_max_date']}) -> {path}")
    return art


def _record_buy_registry(art: dict) -> None:
    try:
        import json
        from src.db.connection import connect
        with connect() as con:
            con.execute(
                "INSERT INTO ml_models (trained_at, method, model_type, horizon, basis, "
                "n_train, train_max_date, features, config) VALUES (?,?,?,?,?,?,?,?,?)",
                [art["trained_at"], "ml_buy", stacker_model_class(), int(art["config"]["horizon"]),
                 art["config"]["basis"], art["n_train"], art["train_max_date"],
                 json.dumps(art["features"]), json.dumps(art["config"])])
    except Exception as e:
        logger.debug(f"[ml_buy] registry write skipped: {e}")


def _load_buy_artifact() -> Optional[dict]:
    if not _BUY_MODEL_PATH.exists():
        return None
    try:
        mt = _BUY_MODEL_PATH.stat().st_mtime_ns
        if _BUY_ART["mtime"] == mt:
            return _BUY_ART["art"]
        with open(_BUY_MODEL_PATH, "rb") as fh:
            art = _pickle.load(fh)
        _BUY_ART.update(mtime=mt, art=art)
        return art
    except Exception as e:
        logger.debug(f"[ml_buy] artifact load failed: {e}")
        _BUY_ART.update(mtime=None, art=None)
        return None


def compute_buy_conviction(method_scores: dict) -> Optional[float]:
    """The stacker's buy conviction ∈ [0,1] for one ticker — the replacement for
    the weighted combined_buy_score. ``method_scores`` must be the DAILY method
    scores (the panel's basis the model trained on), keyed by method name. Returns
    None when the artifact/lightgbm is unavailable, so the caller keeps the
    weighted combine — an invisible degradation is impossible."""
    art = _load_buy_artifact()
    if art is None:
        return None
    try:
        import numpy as _np
        def _g(f):
            v = method_scores.get(f)
            return float(v) if v is not None and v == v else _np.nan
        x = _np.array([[_g(f) for f in art["features"]]], dtype=float)
        bull, _bear = art["model"].bull_bear(x)
        return buy_conviction_from_proba(_calibrate(art, float(bull[0])))
    except Exception as e:
        logger.debug(f"[ml_buy] conviction failed: {e}")
        return None


def eod_train_buy() -> Optional[dict]:
    """EOD entry point — retrain the buy stacker on the latest panel. Fail-soft."""
    return train_and_persist_buy()


def reset_buy_caches() -> None:
    """Test hook — drop the artifact memo."""
    _BUY_ART.update(mtime=None, art=None)


# ── the SELL stacker — symmetric to buy, AS combined_sell_score ───────────────
# Predicts P(the SHORT works) = P(the stock underperforms the benchmark at 5d),
# by training on the NEGATED market-relative return (so class "up" = short won).
# A separately-trained artifact so the two sides can diverge (per-side skill),
# even though with deadband 0 P(short works) is the complement of the buy model's
# P(up). Replaces the WEIGHTED combined_sell_score; the real question the eval
# answers is whether it beats THAT (unrelated to the buy model).

_SELL_MODEL_PATH = _Path("cache/ml/ml_sell_model.pkl")
SELL_TRAIN_CONFIG = dict(horizon=5, basis="rank", deadband=0.0)
_SELL_ART: dict = {"mtime": None, "art": None}


def train_and_persist_sell(days: Optional[int] = None, path=_SELL_MODEL_PATH) -> Optional[dict]:
    """Train the sell stacker on the panel (5d market-relative, label NEGATED so
    'up' = the short worked) over the 21 live method features; pickle it."""
    import numpy as _np
    from src.analysis.ml_train import label_from_return
    h = SELL_TRAIN_CONFIG["horizon"]
    df = build_stacker_dataset(horizons=[h], days=days)
    if df.empty:
        logger.warning("[ml_sell] no panel data to train on")
        return None
    cfg, ycol = _label_cfg(df, SELL_TRAIN_CONFIG, "ml_sell")
    basis = cfg["basis"]
    if ycol not in df.columns:
        logger.warning("[ml_sell] no panel data to train on")
        return None
    feats = [f for f in STACKER_LIVE_FEATURES if f in df.columns]
    # NEGATE the return: the short's outcome. class 2 ("up") now means the stock
    # FELL market-relative = the short worked. bull_bear's `bull` = P(short worked).
    y_raw = df[ycol].map(lambda r: label_from_return(-r, cfg["deadband"]))
    keep = y_raw.notna()
    X = df.loc[keep, feats].to_numpy(dtype=float)
    y = y_raw[keep].to_numpy(dtype=int)
    if len(X) < 500 or len(_np.unique(y)) < 2:
        logger.warning(f"[ml_sell] insufficient training rows ({len(X)})")
        return None
    model = _stacker_model_factory().fit(X, y)
    # Calibrate on OOF predictions of the SELL problem: walk_forward_predict needs
    # the negated-return basis column ("the short worked"), matching this model's label.
    cal, cal_diag = (None, {})
    if settings.enable_ml_probability_calibration:
        # The SHORT's outcome is the NEGATED basis (below the day's median on the
        # rank label = the short worked), matching this model's own label.
        if basis == "rank_pv":
            df["fwd_ret_sellinv_pv"] = -pd.to_numeric(df[ycol], errors="coerce")
            cal, cal_diag = _fit_calibrator(df, h, "sellinv_pv", feats, cfg["deadband"])
        else:
            scol = f"fwd_ret_sellinv_{h}d"
            df[scol] = -pd.to_numeric(df[ycol], errors="coerce")
            cal, cal_diag = _fit_calibrator(df, h, "sellinv", feats, cfg["deadband"])
    art = {"model": model, "features": feats, "config": dict(cfg),
           "model_class": stacker_model_class(),
           "calibrator": cal, "calibration": cal_diag,
           "trained_at": _dt.now(_tz.utc).isoformat(timespec="seconds"),
           "n_train": int(len(X)), "train_max_date": str(df["signal_date"].max())}
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as fh:
        _pickle.dump(art, fh)
    _SELL_ART.update(mtime=None, art=None)
    _record_sell_registry(art)
    logger.info(f"[ml_sell] trained on {art['n_train']:,} rows (<= {art['train_max_date']}) -> {path}")
    return art


def _record_sell_registry(art: dict) -> None:
    try:
        import json
        from src.db.connection import connect
        with connect() as con:
            con.execute(
                "INSERT INTO ml_models (trained_at, method, model_type, horizon, basis, "
                "n_train, train_max_date, features, config) VALUES (?,?,?,?,?,?,?,?,?)",
                [art["trained_at"], "ml_sell", stacker_model_class(), int(art["config"]["horizon"]),
                 art["config"]["basis"], art["n_train"], art["train_max_date"],
                 json.dumps(art["features"]), json.dumps(art["config"])])
    except Exception as e:
        logger.debug(f"[ml_sell] registry write skipped: {e}")


def _load_sell_artifact() -> Optional[dict]:
    if not _SELL_MODEL_PATH.exists():
        return None
    try:
        mt = _SELL_MODEL_PATH.stat().st_mtime_ns
        if _SELL_ART["mtime"] == mt:
            return _SELL_ART["art"]
        with open(_SELL_MODEL_PATH, "rb") as fh:
            art = _pickle.load(fh)
        _SELL_ART.update(mtime=mt, art=art)
        return art
    except Exception as e:
        logger.debug(f"[ml_sell] artifact load failed: {e}")
        _SELL_ART.update(mtime=None, art=None)
        return None


def compute_sell_conviction(method_scores: dict) -> Optional[float]:
    """The stacker's SELL conviction ∈ [0,1] for one ticker — the replacement for
    the weighted combined_sell_score. ``bull`` = P(short worked); centered the
    same way as the buy side. None when the artifact/lightgbm is unavailable, so
    the caller keeps the weighted combine."""
    art = _load_sell_artifact()
    if art is None:
        return None
    try:
        import numpy as _np
        def _g(f):
            v = method_scores.get(f)
            return float(v) if v is not None and v == v else _np.nan
        x = _np.array([[_g(f) for f in art["features"]]], dtype=float)
        bull, _bear = art["model"].bull_bear(x)           # bull = P(short worked)
        return buy_conviction_from_proba(_calibrate(art, float(bull[0])))
    except Exception as e:
        logger.debug(f"[ml_sell] conviction failed: {e}")
        return None


def eod_train_sell() -> Optional[dict]:
    """EOD entry point — retrain the sell stacker on the latest panel. Fail-soft."""
    return train_and_persist_sell()


def reset_sell_caches() -> None:
    """Test hook — drop the sell artifact memo."""
    _SELL_ART.update(mtime=None, art=None)


def validate_swap(horizon: int = 5, basis: str = "rel", days: Optional[int] = None,
                  deadband: float = 0.0, min_train_days: int = 8, step_days: int = 2,
                  features: Optional[Sequence[str]] = None) -> dict:
    """What does REPLACING combined_buy_score with the stacker do to the DECISIONS?

    Walk-forward the stacker, form ``swapped_combined = max(0, 2*P(up)-1) -
    combined_sell_score``, and compare the BUY decisions (combined > threshold) it
    would produce — count AND forward return — against the current weighted
    combine's, on the same OOS rows. The gate before letting it drive live orders:
    a sane buy count (not the whole universe, not zero) and buys that beat the
    current combine's.
    """
    from config.settings import settings as S
    from src.analysis.ml_train import make_model_factory, walk_forward_predict
    thr = float(S.buy_sell_diff_threshold)
    feats = list(features) if features is not None else STACKER_FEATURES
    df = build_stacker_dataset(horizons=[horizon], days=days)
    if df.empty:
        return {}
    preds = walk_forward_predict(df, horizon, basis, features=feats,
                                 deadband=deadband, min_train_days=min_train_days,
                                 step_days=step_days, min_train_rows=1000,
                                 model_factory=make_model_factory("gbm"))
    if preds.empty:
        return {}
    # preds already carries the rel forward return (its `fwd` column, since
    # basis="rel"); only pull fwd_raw + the combine columns from df, to avoid a
    # name collision on the rel column.
    keep = df[["signal_date", "ticker", "combined_score", "combined_sell_score",
               f"fwd_ret_raw_{horizon}d"]]
    m = preds.merge(keep, on=["signal_date", "ticker"], how="left")
    m["swapped_buy"] = (2.0 * m["bull"] - 1.0).clip(0.0, 1.0)
    sell = pd.to_numeric(m["combined_sell_score"], errors="coerce").fillna(0.0)
    m["swapped_combined"] = m["swapped_buy"] - sell
    old = pd.to_numeric(m["combined_score"], errors="coerce")
    fwd_raw = pd.to_numeric(m[f"fwd_ret_raw_{horizon}d"], errors="coerce")
    fwd_rel = pd.to_numeric(m["fwd"], errors="coerce")

    def stats(mask: pd.Series) -> dict:
        mask = mask.fillna(False)
        return {"n": int(mask.sum()), "pct": round(100.0 * float(mask.mean()), 1),
                "fwd_raw": round(float(fwd_raw[mask].mean()), 4) if mask.any() else None,
                "fwd_rel": round(float(fwd_rel[mask].mean()), 4) if mask.any() else None,
                "win_rel": round(100.0 * float((fwd_rel[mask] > 0).mean()), 1) if mask.any() else None}

    old_buy, new_buy = old > thr, m["swapped_combined"] > thr
    return {"horizon": horizon, "basis": basis, "threshold": thr, "n_rows": int(len(m)),
            "universe": stats(pd.Series(True, index=m.index)),
            "old_buys": stats(old_buy), "new_buys": stats(new_buy),
            "overlap": int((old_buy & new_buy).sum())}


def _print_swap(r: dict) -> None:
    import sys
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    if not r:
        return
    print(f"\nSWAP VALIDATION — replace combined_buy_score with the stacker (h={r['horizon']}, "
          f"basis={r['basis']}, buy threshold={r['threshold']})")
    print(f"OOS rows: {r['n_rows']:,}   |   buy = combined_score > threshold\n")
    head = f"{'decision set':<16}{'n':>8}{'% univ':>8}{'fwd_raw%':>10}{'fwd_rel%':>10}{'win_rel%':>10}"
    print(head); print("-" * len(head))
    for name, key in (("universe", "universe"), ("OLD combine buys", "old_buys"),
                      ("NEW stacker buys", "new_buys")):
        s = r[key]
        def f(v, w, fmt):
            return f"{format(v, fmt):>{w}}" if v is not None else f"{'—':>{w}}"
        print(f"{name:<16}{s['n']:>8}{s['pct']:>7}%{f(s['fwd_raw'],10,'+.3f')}"
              f"{f(s['fwd_rel'],10,'+.3f')}{f(s['win_rel'],10,'.1f')}")
    print("-" * len(head))
    print(f"overlap (both buy the same name): {r['overlap']}")
    print("\nGATE to flip live: NEW buys should be a sane count (not ~0, not the whole universe) "
          "AND beat the OLD combine's buys on fwd_rel. If it floods or its buys are worse, DON'T enable.")


def validate_sell_swap(horizon: int = 5, days: Optional[int] = None, deadband: float = 0.0,
                       min_train_days: int = 8, step_days: int = 2,
                       features: Optional[Sequence[str]] = None) -> dict:
    """What does REPLACING combined_sell_score with the sell stacker do to the SELL
    DECISIONS? Symmetric to ``validate_swap``: walk-forward the sell stacker, form
    ``swapped_combined = combined_buy - max(0, 2*P(short works)-1)``, and compare
    the SELL decisions (combined < −threshold) — count AND the SHORT's return
    (``−fwd_ret_rel``, so a sell wins when the stock underperforms) — against the
    weighted combine's. A sane sell count that beats the weighted sell combine on
    the short return is the gate to flip live."""
    from config.settings import settings as S
    from src.analysis.ml_train import make_model_factory, walk_forward_predict
    thr = float(S.buy_sell_diff_threshold)
    feats = list(features) if features is not None else STACKER_LIVE_FEATURES
    df = build_stacker_dataset(horizons=[horizon], days=days)
    if df.empty:
        return {}
    # The SHORT's outcome = the NEGATED market-relative return (a short profits
    # when the stock underperforms). Train + evaluate the stacker on it.
    scol = f"fwd_ret_sellrel_{horizon}d"
    df[scol] = -pd.to_numeric(df[f"fwd_ret_rel_{horizon}d"], errors="coerce")
    preds = walk_forward_predict(df, horizon, "sellrel", features=feats, deadband=deadband,
                                 min_train_days=min_train_days, step_days=step_days,
                                 min_train_rows=1000, model_factory=make_model_factory("gbm"))
    if preds.empty:
        return {}
    keep = df[["signal_date", "ticker", "combined_score", "combined_buy_score"]]
    m = preds.merge(keep, on=["signal_date", "ticker"], how="left")
    m["swapped_sell"] = (2.0 * m["bull"] - 1.0).clip(0.0, 1.0)     # bull = P(short works)
    buy = pd.to_numeric(m["combined_buy_score"], errors="coerce").fillna(0.0)
    m["swapped_combined"] = buy - m["swapped_sell"]
    old = pd.to_numeric(m["combined_score"], errors="coerce")
    short_ret = pd.to_numeric(m["fwd"], errors="coerce")          # the short's return (= −fwd_rel)

    def stats(mask: pd.Series) -> dict:
        mask = mask.fillna(False)
        return {"n": int(mask.sum()), "pct": round(100.0 * float(mask.mean()), 1),
                "short_ret": round(float(short_ret[mask].mean()), 4) if mask.any() else None,
                "win": round(100.0 * float((short_ret[mask] > 0).mean()), 1) if mask.any() else None}

    old_sell, new_sell = old < -thr, m["swapped_combined"] < -thr
    return {"horizon": horizon, "threshold": thr, "n_rows": int(len(m)),
            "universe": stats(pd.Series(True, index=m.index)),
            "old_sells": stats(old_sell), "new_sells": stats(new_sell),
            "overlap": int((old_sell & new_sell).sum())}


def _print_sell_swap(r: dict) -> None:
    import sys
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    if not r:
        return
    print(f"\nSELL SWAP VALIDATION — replace combined_sell_score with the sell stacker "
          f"(h={r['horizon']}, threshold={r['threshold']})")
    print(f"OOS rows: {r['n_rows']:,}   |   sell = combined_score < −threshold;  "
          f"short_ret = −fwd_rel (a sell WINS when the stock underperforms)\n")
    head = f"{'decision set':<16}{'n':>8}{'% univ':>8}{'short_ret%':>12}{'win%':>8}"
    print(head); print("-" * len(head))
    for name, key in (("universe", "universe"), ("OLD combine sells", "old_sells"),
                      ("NEW stacker sells", "new_sells")):
        s = r[key]
        def f(v, w, fmt):
            return f"{format(v, fmt):>{w}}" if v is not None else f"{'—':>{w}}"
        print(f"{name:<16}{s['n']:>8}{s['pct']:>7}%{f(s['short_ret'],12,'+.3f')}{f(s['win'],8,'.1f')}")
    print("-" * len(head))
    print(f"overlap (both sell the same name): {r['overlap']}")
    print("\nGATE: NEW sells a sane count AND beat the OLD combine's sells on short_ret. NOTE a "
          "bearish window inflates BOTH (shorts win when everything falls) — the GAP is the signal.")


def _print(table: pd.DataFrame, model_name: str) -> None:
    import sys
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    if table is None or table.empty:
        print("No stacker results — the panel likely has too few forward-labelled days yet.")
        return
    print(f"\nFULL STACKER (ml_buy) — walk-forward OOS on the signals panel, model={model_name}")
    print("Features = all method scores (combined_score EXCLUDED for circularity; shown as baseline).")
    print("The learned stacker must BEAT combined_score to be worth more than the hand-weighted combine.\n")
    head = f"{'model':<18}{'basis':>6}{'h':>4}{'n':>8}{'IC':>9}{'ICIR':>8}{'hit%':>8}{'simret%':>9}"
    print(head); print("-" * len(head))

    def f(v, w, s):
        return f"{format(v, s):>{w}}" if v is not None and pd.notna(v) else f"{'—':>{w}}"

    for (basis, h), g in table.groupby(["basis", "horizon"], sort=True):
        for _, r in g.iterrows():
            line = f"{r['model']:<18}{basis:>6}{int(h):>4}{int(r['n']):>8}"
            line += f(r['ic'], 9, '+.4f') + f(r['icir'], 8, '+.3f')
            line += f(r['hit'], 8, '.2f') + f(r['simret'], 9, '+.4f')
            print(line)
        print("-" * len(head))
    print("\nPANEL-ONLY, ~34 days of ONE regime — high overfitting risk, so this is a directional "
          "read. GO only if the stacker's rel-basis IC/ICIR clears combined_score's; otherwise the "
          "hand-weighted combine is already capturing what the methods jointly say.")


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description="Full stacker (ml_buy) — learned buy-side aggregator")
    p.add_argument("--horizons", default="1,5,10")
    p.add_argument("--bases", default="raw,rel")
    p.add_argument("--model", default="gbm", choices=("logistic", "gbm"))
    p.add_argument("--deadband", type=float, default=0.0)
    p.add_argument("--min-train-days", type=int, default=8)
    p.add_argument("--step-days", type=int, default=2)
    p.add_argument("--validate-swap", action="store_true",
                   help="measure the DECISION impact of replacing combined_buy_score with the stacker")
    p.add_argument("--swap-horizon", type=int, default=5, help="horizon the buy aggregator optimizes (default 5)")
    p.add_argument("--train", action="store_true",
                   help="retrain + persist BOTH live artifacts (ml_buy, ml_sell) instead of measuring")
    a = p.parse_args(argv)
    horizons = tuple(int(h) for h in str(a.horizons).split(",") if h.strip())
    bases = tuple(b for b in str(a.bases).split(",") if b.strip())
    from src.db import repo
    if a.train:
        # Deliberately BEFORE set_read_only: training appends to the `ml_models`
        # registry, so this branch needs the write path. Both sides are trained
        # together — they are one decision surface (buy/sell camps of the same
        # combine) and shipping a mismatched pair is never what you want.
        for _name, _fn in (("ml_buy", eod_train_buy), ("ml_sell", eod_train_sell)):
            art = _fn()
            print(f"{_name}: " + (f"{art['n_train']:,} rows (<= {art['train_max_date']}), "
                                  f"label basis={art['config']['basis']}"
                                  if art else "NO ARTIFACT (see log)"))
        return
    repo.set_read_only(True)
    if a.validate_swap:
        r = validate_swap(horizon=a.swap_horizon, basis="rel", deadband=a.deadband,
                          min_train_days=a.min_train_days, step_days=a.step_days)
        _print_swap(r)
        return
    table = measure(horizons=horizons, bases=bases, deadband=a.deadband, model_name=a.model,
                    min_train_days=a.min_train_days, step_days=a.step_days)
    _print(table, a.model)


if __name__ == "__main__":
    main()
