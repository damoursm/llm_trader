"""Phase-1 forward-panel validation — the honest promotion gate.

The deep-cache walk-forward (``ml_train``) answers "can an OHLCV model learn
out-of-sample", but on TODAY's universe — survivorship-biased, optimistic. This
module makes the promotion decision on the survivorship-free ``signals`` panel:
it TRAINS the model on the deep cache (lots of data, multi-regime) and EVALUATES
it on the panel (every ticker actually scored, including the penny/thin/soon-
delisted names the deep cache drops), which is exactly the train-deep / judge-
forward split the plan calls for.

Point-in-time is still enforced: to score a panel row dated D, the model is
trained only on deep-cache rows whose LABEL had printed before D
(``end_date < D``). So a passing number here cannot be look-ahead.

**Read it with its power in mind.** The panel is only ~6 weeks old and a 10-day
label needs ~10 more sessions, so the usable sample is a few thousand rows over
~20 days — enough for a DIRECTIONAL read (does the deep-cache edge survive, or
flip negative on the honest set?), not enough to confirm a +0.037 IC. The
panel-first live method (``ml_ohlcv``, weight 0) is what accrues the decisive
sample over the coming weeks.

Two modes:

* **classic** (default) — the v1 classifier configs (``--horizons/--bases/
  --model/--deadband`` + the clean-trend/liquidity conditioning knobs).
* **``--target pivot``** — the ``ml_ohlcv`` v2 candidate (signed-pivot
  within-day-rank GBM; ``validate_pivot_on_panel``), printed against its
  PRE-REGISTERED promotion bar. This is the gate v2 passed on 2026-08-08
  (panel IC +0.0639, t +2.31, edge +2.58pp).

CLI:  python -m src.analysis.ml_validate [--target pivot] [--from-parquet PATH]
          [--horizons 5,10] [--deadband 0] [--min-eff-ratio 0.4]
          [--min-dollar-vol-log 7] [--threads 6] [--save-preds PATH]
"""

from __future__ import annotations

import argparse
from datetime import date
from typing import Any, Callable, List, Optional, Sequence

import numpy as np
import pandas as pd
from loguru import logger

from src.analysis.ml_dataset import (ALL_FEATURE_COLUMNS, build_dataset,
                                     build_panel_dataset, load_materialized,
                                     missing_feature_columns)
from src.analysis.ml_train import (SoftmaxLogistic, _metrics, condition_universe,
                                   label_from_return, make_model_factory)


def validate_on_panel(deep_df: pd.DataFrame, panel_df: pd.DataFrame, horizon: int,
                      basis: str, features: Sequence[str] = ALL_FEATURE_COLUMNS,
                      deadband: float = 0.0, min_train_rows: int = 2000,
                      step_days: int = 5,
                      model_factory: Optional[Callable[[], Any]] = None) -> pd.DataFrame:
    """Train on ``deep_df``, predict ``panel_df``, point-in-time. Returns per-row
    OOS predictions (signal_date, ticker, net, fwd) over the panel."""
    ycol = f"fwd_ret_{basis}_{horizon}d"
    ecol = f"end_date_{horizon}d"
    if ycol not in deep_df.columns or ycol not in panel_df.columns:
        return pd.DataFrame()
    feats = [f for f in features if f in deep_df.columns and f in panel_df.columns]
    factory = model_factory or SoftmaxLogistic

    # Training pool (deep cache): labelled rows with a printed end date.
    tr = deep_df[["signal_date", ecol, ycol] + feats].copy()
    tr[ycol] = pd.to_numeric(tr[ycol], errors="coerce")
    tr = tr[tr[ycol].notna()].reset_index(drop=True)
    tr["_y"] = tr[ycol].map(lambda r: label_from_return(r, deadband))
    tr = tr[tr["_y"].notna()].reset_index(drop=True)
    tr["_end"] = tr[ecol].map(lambda s: date.fromisoformat(str(s)[:10]))
    Xtr = tr[feats].to_numpy(dtype=float)
    ytr = tr["_y"].to_numpy(dtype=int)
    end_tr = tr["_end"].to_numpy()

    # Eval set (panel): rows with a realised forward return.
    ev = panel_df[["signal_date", "ticker", ycol] + feats].copy()
    ev[ycol] = pd.to_numeric(ev[ycol], errors="coerce")
    ev = ev[ev[ycol].notna()].reset_index(drop=True)
    if ev.empty:
        return pd.DataFrame()
    ev["_sig"] = ev["signal_date"].map(lambda s: date.fromisoformat(str(s)[:10]))
    Xev = ev[feats].to_numpy(dtype=float)

    eval_dates = sorted(ev["_sig"].unique())
    out: List[pd.DataFrame] = []
    for a in range(0, len(eval_dates), step_days):
        D = eval_dates[a]
        nxt = eval_dates[a + step_days] if a + step_days < len(eval_dates) else None
        keep = end_tr < D                               # only labels printed before D
        if keep.sum() < min_train_rows:
            continue
        ytrain = ytr[keep]
        if len(np.unique(ytrain)) < 2:
            continue
        model = factory().fit(Xtr[keep], ytrain)
        sig = ev["_sig"].to_numpy()
        te = (sig >= D) if nxt is None else ((sig >= D) & (sig < nxt))
        if te.sum() == 0:
            continue
        bull, bear = model.bull_bear(Xev[te])
        sub = ev.loc[te, ["signal_date", "ticker", ycol]].copy()
        sub["net"] = bull - bear
        sub["fwd"] = ev.loc[te, ycol].to_numpy()
        out.append(sub)
    return pd.concat(out, ignore_index=True) if out else pd.DataFrame()


def run(horizons: Sequence[int] = (5, 10), bases: Sequence[str] = ("raw", "rel"),
        deadband: float = 0.0, model_name: str = "gbm",
        min_eff_ratio: Optional[float] = 0.4, min_dollar_vol_log: Optional[float] = 7.0,
        deep_parquet: Optional[str] = "cache/ml/dataset_multi.parquet",
        limit_tickers: int = 1500, date_stride: int = 2) -> pd.DataFrame:
    """Build (or load) the deep training set + the panel eval set, apply the same
    conditioning to both, and validate the winning config. Returns the table."""
    deep = load_materialized(deep_parquet) if deep_parquet else pd.DataFrame()
    # A parquet older than the current feature set would be silently intersected
    # down to the features it happens to carry, so this would validate the OLD
    # model while reporting it as the current one. Rebuild instead.
    stale = missing_feature_columns(deep)
    if stale:
        logger.warning(f"[ml_validate] {deep_parquet} lacks {len(stale)} current feature "
                       f"columns (e.g. {stale[:3]}) — rebuilding the deep set")
    if deep is None or deep.empty or stale:
        deep = build_dataset(horizons=horizons, limit_tickers=limit_tickers, date_stride=date_stride)
    panel = build_panel_dataset(horizons=horizons)
    if deep.empty or panel.empty:
        logger.warning("[ml_validate] deep or panel set empty — nothing to validate")
        return pd.DataFrame()

    n_deep0, n_panel0 = len(deep), len(panel)
    deep = condition_universe(deep, min_eff_ratio, min_dollar_vol_log)
    panel = condition_universe(panel, min_eff_ratio, min_dollar_vol_log)
    logger.info(f"[ml_validate] conditioned deep {n_deep0:,}->{len(deep):,}, "
                f"panel {n_panel0:,}->{len(panel):,}")

    factory = make_model_factory(model_name)
    rows: List[dict] = []
    for h in horizons:
        for basis in bases:
            preds = validate_on_panel(deep, panel, h, basis, deadband=deadband,
                                      model_factory=factory)
            if preds.empty:
                rows.append({"basis": basis, "horizon": h, "n": 0, "days": 0,
                             "ic": None, "icir": None, "hit": None, "simret": None})
                continue
            m = _metrics(preds["signal_date"], preds["net"], preds["fwd"], min_n=50)
            m["days"] = int(preds["signal_date"].nunique())
            rows.append({"basis": basis, "horizon": h, **m})
    return pd.DataFrame(rows)


def validate_pivot_on_panel(deep_parquet: str, step_days: int = 5,
                            threads: int = 6,
                            min_train_rows: int = 20000) -> pd.DataFrame:
    """Forward-panel validation for the ``ml_ohlcv`` v2 candidate (signed pivot
    target, 85 features, within-day-rank L2 GBM — see ``analysis/pivot_target``).

    Same honest split as ``validate_on_panel``: train on the deep cache,
    evaluate on the live ``signals`` panel, point-in-time (a training row is
    admitted only once its pivot END printed before the refit cutoff). Panel
    rows join their pivot target EXACTLY on the session date; rows whose target
    has not settled, or whose ticker lacks 50 bars, drop out — conservative,
    never mislabelled. Returns per-row OOS predictions with the realised pivot
    return, the rel-5d return (basis sanity check), and the Gate-4 columns.
    """
    from src.analysis.pivot_target import (LEG_FEATURES, pivot_frame,
                                           within_day_rank)
    from src.analysis.ml_train import LightGBMRankRegressor

    deep = load_materialized(deep_parquet)
    if deep is None or deep.empty:
        logger.warning("[ml_validate:pivot] deep parquet empty/missing")
        return pd.DataFrame()
    panel = build_panel_dataset(horizons=(5,))
    if panel.empty:
        logger.warning("[ml_validate:pivot] panel empty")
        return pd.DataFrame()

    tickers = sorted(set(deep["ticker"]) | set(panel["ticker"]))
    pf = pivot_frame(tickers)
    if pf.empty:
        return pd.DataFrame()

    feats = [f for f in ALL_FEATURE_COLUMNS if f in deep.columns and f in panel.columns]
    feats += [f for f in LEG_FEATURES]

    tr = deep.merge(pf, on=["ticker", "signal_date"], how="inner")
    tr["sp_buy"] = pd.to_numeric(tr["sp_buy"], errors="coerce")
    tr = tr[tr["sp_buy"].notna() & tr["sp_end"].notna()].reset_index(drop=True)
    Xtr = tr[feats].to_numpy(dtype=np.float32)
    ytr = tr["sp_buy"].to_numpy(dtype=np.float64)
    end_tr = tr["sp_end"].map(lambda s: date.fromisoformat(str(s)[:10])).to_numpy()
    day_tr = tr["signal_date"].astype(str).to_numpy()

    ev = panel.merge(pf, on=["ticker", "signal_date"], how="inner")
    ev["sp_buy"] = pd.to_numeric(ev["sp_buy"], errors="coerce")
    ev = ev[ev["sp_buy"].notna()].reset_index(drop=True)
    logger.info(f"[ml_validate:pivot] train {len(tr):,} rows | panel matched "
                f"{len(ev):,}/{len(panel):,} rows over "
                f"{ev['signal_date'].nunique()} days")
    if ev.empty:
        return pd.DataFrame()
    Xev = ev[feats].to_numpy(dtype=np.float32)
    sig = ev["signal_date"].map(lambda s: date.fromisoformat(str(s)[:10])).to_numpy()

    eval_dates = sorted(ev["signal_date"].map(
        lambda s: date.fromisoformat(str(s)[:10])).unique())
    out: List[pd.DataFrame] = []
    for a in range(0, len(eval_dates), step_days):
        D = eval_dates[a]
        nxt = eval_dates[a + step_days] if a + step_days < len(eval_dates) else None
        keep = end_tr < D
        if int(keep.sum()) < min_train_rows:
            continue
        dsub = day_tr[keep]
        _, day_idx = np.unique(dsub, return_inverse=True)
        yr = within_day_rank(ytr[keep], day_idx.astype(np.int32))
        cnt = np.bincount(day_idx)
        w = (1.0 / cnt[day_idx]).astype(np.float64)
        w /= w.mean()
        model = LightGBMRankRegressor(num_threads=threads).fit(Xtr[keep], yr, w)
        te = (sig >= D) if nxt is None else ((sig >= D) & (sig < nxt))
        if te.sum() == 0:
            continue
        sub = ev.loc[te, ["signal_date", "ticker", "sp_buy", "fwd_ret_rel_5d",
                          "log_price", "dollar_vol_log"]].copy()
        sub["net"] = model.predict(Xev[te])
        out.append(sub)
    return pd.concat(out, ignore_index=True) if out else pd.DataFrame()


def _print(table: pd.DataFrame, model_name: str, deadband: float) -> None:
    import sys
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    if table is None or table.empty:
        print("No panel validation results — the forward panel likely has too few "
              "realised forward returns yet. Re-run as it thickens.")
        return
    print(f"\nForward-panel validation (SURVIVORSHIP-FREE) — model={model_name}, deadband={deadband}%")
    print("Trained on the deep cache, evaluated on the live signals panel, point-in-time.\n")
    head = f"{'basis':>6}{'h':>4}{'n':>8}{'days':>6}{'IC':>9}{'ICIR':>8}{'hit%':>8}{'simret%':>9}"
    print(head); print("-" * len(head))

    def f(v, w, s):
        return f"{format(v, s):>{w}}" if v is not None and pd.notna(v) else f"{'—':>{w}}"

    for _, r in table.iterrows():
        line = f"{r['basis']:>6}{int(r['horizon']):>4}{int(r['n']):>8}{int(r['days']):>6}"
        line += f(r['ic'], 9, '+.4f') + f(r['icir'], 8, '+.3f')
        line += f(r['hit'], 8, '.2f') + f(r['simret'], 9, '+.4f')
        print(line)
    print("-" * len(head))
    print("\nPRELIMINARY — the panel is ~6 weeks old, so this is a directional read (does the "
          "deep-cache edge survive on the honest set?), not a confirmation. The panel-first live "
          "method accrues the decisive sample. A rel-basis IC that stays positive and beats the "
          "deep-cache baselines is the green light; a flip to negative says the deep edge was "
          "survivorship.")


def _print_pivot(preds: pd.DataFrame) -> None:
    """Summarise the pivot-candidate panel run against its PRE-REGISTERED bar
    (fixed 2026-08-08, before the run): (1) per-day IC t >= 2 on the full panel;
    (2) direction edge > 0; (3) Gate-4 tradeable-subset IC > 0 (sign only);
    (4) the same predictions' rel-5d per-day IC not negative."""
    import math
    import sys

    from src.analysis.signal_panel import _spearman
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    if preds is None or preds.empty:
        print("No pivot panel predictions — panel too thin or no settled pivots yet.")
        return

    def daily_ic(p, y, d):
        ics = []
        for dd in pd.unique(d):
            m = d == dd
            if m.sum() >= 8:
                r = _spearman(pd.Series(np.asarray(p)[m]), pd.Series(np.asarray(y)[m]))
                if r is not None and r == r:
                    ics.append(r)
        return np.asarray(ics)

    def block(name, sub, ycol):
        p = pd.to_numeric(sub["net"], errors="coerce")
        y = pd.to_numeric(sub[ycol], errors="coerce")
        ok = p.notna() & y.notna()
        p, y, d = p[ok].to_numpy(), y[ok].to_numpy(), sub.loc[ok, "signal_date"].to_numpy()
        ics = daily_ic(p, y, d)
        mean_ic = float(ics.mean()) if len(ics) else float("nan")
        icir = mean_ic / float(ics.std()) if len(ics) and ics.std() > 0 else float("nan")
        t = icir * math.sqrt(len(ics)) if icir == icir else float("nan")
        moved = y != 0
        hit = float(((p > 0) == (y > 0))[moved].mean() * 100) if moved.any() else float("nan")
        base = float((y[moved] > 0).mean() * 100) if moved.any() else float("nan")
        edge = hit - max(base, 100 - base)
        print(f"{name:<26}{len(p):>9,}{len(ics):>6}{mean_ic:>+9.4f}{icir:>+8.3f}"
              f"{t:>+7.2f}{edge:>+9.2f}")
        return dict(ic=mean_ic, t=t, edge=edge)

    print("\nForward-panel validation — ml_ohlcv v2 candidate (signed pivot rank GBM)")
    print(f"{'slice':<26}{'n':>9}{'days':>6}{'IC':>9}{'ICIR':>8}{'t':>7}{'edge':>9}")
    print("-" * 74)
    full = block("pivot target (ALL panel)", preds, "sp_buy")
    gate = preds[(pd.to_numeric(preds["log_price"], errors="coerce") >= np.log10(5.0))
                 & (pd.to_numeric(preds["dollar_vol_log"], errors="coerce") >= np.log10(5e6))]
    gated = block("pivot target (GATE-4)", gate, "sp_buy")
    rel = block("rel-5d sanity (same net)", preds, "fwd_ret_rel_5d")
    print("-" * 74)
    ok1 = full["t"] >= 2.0 and full["ic"] > 0
    ok2 = full["edge"] > 0
    ok3 = gated["ic"] > 0
    ok4 = not (rel["ic"] < 0 and rel["t"] <= -2.0)
    for i, (ok, txt) in enumerate([(ok1, "full-panel IC > 0 with t >= 2.0"),
                                   (ok2, "direction edge > 0"),
                                   (ok3, "Gate-4 subset IC > 0"),
                                   (ok4, "rel-5d basis not significantly negative")], 1):
        print(f"  [{'PASS' if ok else 'FAIL'}] ({i}) {txt}")
    print(f"\nVERDICT: {'PROMOTE' if all([ok1, ok2, ok3, ok4]) else 'DO NOT PROMOTE'} "
          "(bar pre-registered 2026-08-08)")


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description="Validate the OHLCV ML model on the forward signals panel")
    p.add_argument("--horizons", default="5,10")
    p.add_argument("--bases", default="raw,rel")
    p.add_argument("--model", default="gbm", choices=("logistic", "gbm"))
    p.add_argument("--deadband", type=float, default=0.0)
    p.add_argument("--min-eff-ratio", type=float, default=0.4)
    p.add_argument("--min-dollar-vol-log", type=float, default=7.0)
    p.add_argument("--from-parquet", default="cache/ml/dataset_multi.parquet",
                   help="deep training set (built on the fly if missing)")
    p.add_argument("--target", default="classic", choices=("classic", "pivot"),
                   help="'pivot' validates the ml_ohlcv v2 candidate (signed "
                        "pivot rank GBM) instead of the classifier configs")
    p.add_argument("--threads", type=int, default=6)
    p.add_argument("--save-preds", default=None,
                   help="pivot mode: also write the per-row OOS predictions here")
    a = p.parse_args(argv)

    from src.db import repo
    repo.set_read_only(True)
    if a.target == "pivot":
        preds = validate_pivot_on_panel(a.from_parquet, threads=a.threads)
        if a.save_preds and preds is not None and not preds.empty:
            preds.to_csv(a.save_preds, index=False)
        _print_pivot(preds)
        return
    horizons = tuple(int(h) for h in str(a.horizons).split(",") if h.strip())
    bases = tuple(b for b in str(a.bases).split(",") if b.strip())
    table = run(horizons=horizons, bases=bases, deadband=a.deadband, model_name=a.model,
                min_eff_ratio=a.min_eff_ratio, min_dollar_vol_log=a.min_dollar_vol_log,
                deep_parquet=a.from_parquet)
    _print(table, a.model, a.deadband)


if __name__ == "__main__":
    main()
