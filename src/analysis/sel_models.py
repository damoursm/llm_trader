"""Selection-objective models — the NEXT models (user spec 2026-09-25, memory
``next-models-spec-2026-09``).

LONG and SHORT are trained separately and chosen on
`eval_metrics.selection_objective`: per run the top (long) / bottom (short)
name, kept only when it is also a new extreme against its own last 30 trading
days of scores, one entry per name per day, each earning its next-pivot return
per day. Rows are the arrays `ml30.build` writes — intraday training rows
(``--rows random3``: three day-seeded 30-minute bars per session) plus every bar
of the evaluation window (``--eval-since``), the density the live scorer and the
own-history rule see.

Objectives (LightGBM, per side):

  lambdarank  query = run; graded relevance from the within-run percentile of
              the side's return per day, top-heavy (`GRADE_EDGES`: the run's top
              0.5% gets grade 7), NDCG@1 — "get the top pick right".
  tailreg     regression on the side's own top tail only: 0 below the run's
              80th percentile of the side's return per day, rising to 1 at the
              top, day-equal weights. Each side learns only its own tail, so the
              two models are genuinely different.
  rankreg     the live ml_ohlcv recipe — regression on the within-run centred
              rank of the pivot return, ONE model for both sides (high = long,
              low = short). The baseline the new objectives must beat on the
              same rows, windows and pool.

A short model's output is SHORT conviction (higher = a bigger expected drop);
it is scored as ``-prediction`` so the objective's bottom pick is its top one.

Every fit is judged three ways side by side (memory
``deep-features-30m-models-2026-09``): the selection objective on the pivot
label; the SAME picks' realizable returns (to the session close, the next
session's close, +5 sessions — orders can take those exits, the check against
the pivot label's hindsight); and per-run IC to the pivot and to the realizable
returns. The pool is the Gate-4 trade floor (price >= $5, 20-session dollar
volume >= $5M, both known before the row's session).

CLI::

    python -m src.analysis.sel_models --validate [--rounds 400] [--fset deep]
    python -m src.analysis.sel_models --final --objective lambdarank [--rounds N]
"""
from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from loguru import logger

from src.analysis import eval_metrics as em

TRAIN_DIR = Path("cache/ml/sel30")
EVAL_DIR = Path("cache/ml/sel30_eval")
OUT_DIR = Path("cache/ml/sel")
# model TYPE -> (training arrays, evaluation arrays, output tag prefix). The
# daily arrays hold one row per session, so they are their own evaluation set.
KINDS = {"intraday": (TRAIN_DIR, EVAL_DIR, ""),
         "daily": (Path("cache/ml/seld"), Path("cache/ml/seld"), "daily_")}
EVAL_FROM = "2026-05-01"             # the evaluation rows: out of sample for the final fits
MIN_PRICE = 5.0                      # the Gate-4 trade floor
MIN_DV = 5e6
GRADE_EDGES = (0.5, 0.8, 0.9, 0.95, 0.98, 0.99, 0.995)
TAIL_START = 0.8
MIN_QUERY = 50                       # runs thinner than this are not a cross-section to rank
LGB_PARAMS = dict(learning_rate=0.05, num_leaves=63, min_child_samples=500, feature_fraction=0.7,
                  bagging_fraction=0.7, bagging_freq=1, lambda_l2=10.0, max_bin=127, seed=0,
                  deterministic=True, force_col_wise=True, verbosity=-1)
OBJECTIVES = ("lambdarank", "tailreg", "rankreg")
EPOCH = pd.Timestamp("1970-01-01")


def dnum(s: str) -> int:
    return int((pd.Timestamp(s) - EPOCH).days)


def diso(d) -> np.ndarray:
    return np.datetime_as_string(np.asarray(d, dtype=np.int64).astype("datetime64[D]"))


def _lowprio() -> None:
    from src.analysis.ml30 import _lowprio as lp
    lp()


# ── the arrays ─────────────────────────────────────────────────────────────

class Arrays:
    """One `ml30.build` output directory: the feature matrices stay memory-
    mapped, the per-row vectors are loaded."""

    SMALL = ("y", "conf", "ba", "dn", "bar", "px", "dv20", "tk", "fsc", "f1d", "f5d")
    EXIT = ("xl", "xlb", "xs", "xsb")

    def __init__(self, d: Path):
        self.dir = Path(d)
        self.meta = json.loads((self.dir / "meta.json").read_text(encoding="utf-8"))
        self.X = np.load(self.dir / "X.npy", mmap_mode="r")
        self.D = np.load(self.dir / "D.npy", mmap_mode="r") if self.meta.get("deep") else None
        for k in self.SMALL:
            setattr(self, k, np.load(self.dir / f"{k}.npy"))
        for k in self.EXIT:                               # `ml30.add_exit_labels`, when run
            f = self.dir / f"{k}.npy"
            setattr(self, k, np.load(f) if f.exists() else None)
        self.run = self.dn.astype(np.int64) * 100 + self.bar.astype(np.int64)
        self.tickers = list(self.meta["tickers"])

    def tradeable(self) -> np.ndarray:
        return (self.px >= MIN_PRICE) & (np.nan_to_num(self.dv20) >= MIN_DV)

    def features(self, fset: str) -> Tuple[List[str], List[int], List[int]]:
        """(names, X columns, D columns). ``base``: the 85 OHLCV + leg features;
        ``deep``: + the per-ticker deep features; ``all``: + the market group
        (measured as regime timing on 2026-09-23 — an ablation, not a default)."""
        from src.analysis.deep_features import MARKET_FEATURES
        base = list(self.meta["base_features"])
        deep = list(self.meta.get("deep_features") or [])
        if fset == "base":
            dsel: List[str] = []
        elif fset == "deep":
            dsel = [f for f in deep if f not in set(MARKET_FEATURES)]
        elif fset == "all":
            dsel = deep
        else:
            raise ValueError(f"fset must be base, deep or all, got {fset!r}")
        if dsel and self.D is None:
            raise ValueError(f"{self.dir} was built without deep features")
        return base + dsel, list(range(len(base))), [deep.index(f) for f in dsel]

    def matrix(self, rows: np.ndarray, fset: str, chunk: int = 250_000) -> np.ndarray:
        names, xc, dc = self.features(fset)
        M = np.empty((len(rows), len(names)), np.float32)
        nb = len(xc)
        for a in range(0, len(rows), chunk):
            r = rows[a:a + chunk]
            M[a:a + len(r), :nb] = self.X[r][:, xc]
            if dc:
                M[a:a + len(r), nb:] = self.D[r][:, dc]
        return M


# ── targets ────────────────────────────────────────────────────────────────

def side_rpd(y: np.ndarray, ba: np.ndarray, side: str, floor: float = 1.0) -> np.ndarray:
    """The side-oriented return per day each row would earn as an entry."""
    r = em.return_per_day(y, np.where(ba > 0, ba, np.nan), floor)
    return r if side == "long" else -r


TARGETS = ("pivot", "trail")
TRAIL_EMBARGO_DAYS = 16                  # > the trailing exits' 10-session time stop, in calendar days


def side_target(arr: "Arrays", rows: np.ndarray, side: str, target: str = "pivot",
                floor: float = 1.0) -> np.ndarray:
    """What a side model learns to rank: ``"pivot"`` — the return per day to
    the next pivot (the spec's objective, read with hindsight); ``"trail"`` —
    the return per day actually captured by the pivot threshold's trailing
    stop (`ml30.trailing_exits`), its realizable twin."""
    if target == "pivot":
        return side_rpd(arr.y[rows], arr.ba[rows], side, floor)
    if target != "trail":
        raise ValueError(f"target must be one of {TARGETS}, got {target!r}")
    if arr.xl is None:
        raise ValueError(f"{arr.dir} has no exit labels — run ml30.add_exit_labels first")
    if side == "long":
        r, b = arr.xl[rows].astype(float), arr.xlb[rows].astype(float)
    else:
        r, b = -arr.xs[rows].astype(float), arr.xsb[rows].astype(float)
    r = np.where(b > 0, r, np.nan)
    return r / np.maximum(b / em.BARS_PER_DAY, floor)


def run_pct(v: np.ndarray, run: np.ndarray) -> np.ndarray:
    """Percentile of ``v`` within its run, in (0, 1], ties averaged."""
    return pd.Series(v).groupby(run).rank(pct=True, method="average").to_numpy()


def grades(pct: np.ndarray) -> np.ndarray:
    """Top-heavy relevance grades 0..7 from a within-run percentile."""
    return np.searchsorted(np.asarray(GRADE_EDGES), pct, side="right").astype(np.int32)


def tail_target(pct: np.ndarray) -> np.ndarray:
    """0 below the run's `TAIL_START` percentile, rising to 1 at the top."""
    return np.clip((pct - TAIL_START) / (1.0 - TAIL_START), 0.0, 1.0) ** 2


def day_weights(day: np.ndarray) -> np.ndarray:
    _, inv, cnt = np.unique(day, return_inverse=True, return_counts=True)
    w = 1.0 / cnt[inv]
    return w / w.mean()


# ── training ───────────────────────────────────────────────────────────────

def training_rows(arr: Arrays, cut: str, conf_cut: Optional[str] = None, target: str = "pivot") -> np.ndarray:
    """Labelled, tradeable rows of sessions <= ``cut`` whose pivot CONFIRMED by
    ``conf_cut`` (default ``cut``: no label printed after the fit's end), sorted
    by run (LambdaRank needs each query contiguous). The ``trail`` target's
    exits can run 10 sessions, so its rows stop `TRAIL_EMBARGO_DAYS` before the
    cut instead."""
    c, cc = dnum(cut), dnum(conf_cut or cut)
    ok = (np.isfinite(arr.y) & (arr.ba > 0) & (arr.conf > 0) & (arr.dn <= c) & (arr.conf <= cc)
          & arr.tradeable())
    if target == "trail":
        ok &= (np.isfinite(arr.xl) & (arr.xlb > 0) & np.isfinite(arr.xs) & (arr.xsb > 0)
               & (arr.dn <= c - TRAIL_EMBARGO_DAYS))
    rows = np.flatnonzero(ok)
    return rows[np.lexsort((arr.tk[rows], arr.run[rows]))]


def fit(arr: Arrays, rows: np.ndarray, side: str, objective: str, fset: str = "deep",
        rounds: int = 400, threads: int = 8, params: Optional[dict] = None, target: str = "pivot"):
    """Train one model on ``rows`` (sorted by run). ``side`` is ignored by
    ``rankreg`` (one model for both sides, always on the pivot label — the live
    recipe); the side objectives rank `side_target`'s ``target``."""
    import lightgbm as lgb
    t0 = time.time()
    run = arr.run[rows]
    keep = np.ones(len(rows), bool)
    if objective == "lambdarank":                       # drop thin runs: nothing to rank
        _, inv, cnt = np.unique(run, return_inverse=True, return_counts=True)
        keep = cnt[inv] >= MIN_QUERY
    rows, run = rows[keep], run[keep]
    names = arr.features(fset)[0]
    X = arr.matrix(rows, fset)
    p = dict(LGB_PARAMS, num_threads=int(threads), **(params or {}))
    if objective == "rankreg":
        from src.analysis.pivot_target import within_day_rank
        _, ri = np.unique(run, return_inverse=True)
        label = within_day_rank(arr.y[rows].astype(float), ri.astype(np.int32))
        ds = lgb.Dataset(X, label=label, weight=day_weights(arr.dn[rows]), feature_name=names,
                         free_raw_data=True)
        p.update(objective="regression")
    elif objective == "tailreg":
        label = tail_target(run_pct(side_target(arr, rows, side, target), run))
        ds = lgb.Dataset(X, label=label, weight=day_weights(arr.dn[rows]), feature_name=names,
                         free_raw_data=True)
        p.update(objective="regression")
    elif objective == "lambdarank":
        label = grades(run_pct(side_target(arr, rows, side, target), run))
        _, cnt = np.unique(run, return_counts=True)              # rows are sorted by run
        ds = lgb.Dataset(X, label=label, group=cnt, feature_name=names, free_raw_data=True)
        p.update(objective="lambdarank", metric="ndcg", eval_at=[1, 3, 5],
                 lambdarank_truncation_level=30)
    else:
        raise ValueError(f"objective must be one of {OBJECTIVES}, got {objective!r}")
    del X                          # the Dataset holds the only reference; freed once binned
    booster = lgb.train(p, ds, num_boost_round=int(rounds))
    logger.info(f"[sel] fit {objective}/{side}/{target} on {len(rows):,} rows / {len(np.unique(arr.dn[rows]))} "
                f"days ({fset}, {rounds} rounds) in {time.time() - t0:.0f}s")
    return booster


def predict(booster, arr: Arrays, rows: np.ndarray, fset: str, num_iteration: Optional[int] = None,
            chunk: int = 500_000, threads: Optional[int] = None) -> np.ndarray:
    kw = {"num_threads": int(threads)} if threads else {}
    out = np.empty(len(rows), np.float64)
    for a in range(0, len(rows), chunk):
        r = rows[a:a + chunk]
        out[a:a + len(r)] = booster.predict(arr.matrix(r, fset), num_iteration=num_iteration, **kw)
    return out


def oriented(score: np.ndarray, side: str, objective: str) -> np.ndarray:
    """What the selection rule ranks: a side model's SHORT conviction is
    negated so its top becomes the rule's bottom pick; ``rankreg`` is one
    score for both sides."""
    return -score if (side == "short" and objective != "rankreg") else score


# ── evaluation ─────────────────────────────────────────────────────────────

def frame(arr: Arrays, rows: np.ndarray, score: np.ndarray) -> pd.DataFrame:
    df = pd.DataFrame({
        "run": arr.run[rows], "signal_date": diso(arr.dn[rows]), "ticker": arr.tk[rows],
        "score": score, "fwd_ret_pivot": arr.y[rows],
        "bars_ahead": np.where(arr.ba[rows] > 0, arr.ba[rows], np.nan).astype(float),
        "fsc": arr.fsc[rows], "f1d": arr.f1d[rows], "f5d": arr.f5d[rows]})
    if arr.xl is not None:                                  # the realizable trailing exits
        for k in Arrays.EXIT:
            df[k] = getattr(arr, k)[rows]
    base = list(arr.meta["base_features"])
    if VOL_FEATURE in base:                                 # the volatility-matched control's key
        j = base.index(VOL_FEATURE)
        df["vol"] = np.concatenate([arr.X[rows[a:a + 500_000], j] for a in range(0, len(rows), 500_000)]) \
            if len(rows) else np.zeros(0, np.float32)
    return df


VOL_FEATURE = "atr_pct_14"
VOL_HALF_WIDTH = 0.025


def side_returns(d: pd.DataFrame, side: str) -> pd.DataFrame:
    """The side-oriented realizable returns of rows: next close, +5 sessions
    and the trailing exit per day."""
    sign = 1.0 if side == "long" else -1.0
    out = pd.DataFrame({"f1d": sign * d["f1d"].to_numpy(float), "f5d": sign * d["f5d"].to_numpy(float)},
                       index=d.index)
    if "xl" in d.columns:
        out["trail_rpd"] = trail_returns(d, side)[1]
    return out


def vol_control(df: pd.DataFrame, e: pd.DataFrame, side: str,
                half: float = VOL_HALF_WIDTH) -> pd.DataFrame:
    """For each entry, the mean side-oriented realizable returns of the OTHER
    names in its run whose volatility percentile is within ``half`` of the
    pick's — what a pick of the same volatility, chosen at random, earned. A
    selection whose realizable edge is only "volatile names drift" shows up as
    a zero excess over this."""
    cols = list(side_returns(df.head(0), side).columns)
    out = pd.DataFrame(np.nan, index=e.index, columns=cols)
    if "vol" not in df.columns or e.empty:
        return out
    runs = set(e["run"].unique())
    sub = df[df["run"].isin(runs)]
    vp = sub.groupby("run")["vol"].rank(pct=True)
    rets = side_returns(sub, side)
    by_run = {r: g.index for r, g in sub.groupby("run")}
    for i, row in e.iterrows():
        idx = by_run.get(row["run"])
        if idx is None or i not in vp.index:
            continue
        v0 = vp.loc[i]
        near = idx[(np.abs(vp.loc[idx].to_numpy() - v0) <= half + 1e-12) & (idx != i)]   # symmetric edges
        if len(near):
            out.loc[i] = rets.loc[near].mean().to_numpy()
    return out


def trail_returns(e: pd.DataFrame, side: str, floor: float = 1.0):
    """(return %, return per day) of each entry on the pivot threshold's
    trailing stop — the realizable twin of the pivot label — side-oriented."""
    if side == "long":
        r, b = e["xl"].to_numpy(float), e["xlb"].to_numpy(float)
    else:
        r, b = -e["xs"].to_numpy(float), e["xsb"].to_numpy(float)
    r = np.where(b > 0, r, np.nan)
    return r, r / np.maximum(b / em.BARS_PER_DAY, floor)


def run_ic(df: pd.DataFrame, label: str) -> dict:
    """Per-run Spearman IC of ``score`` with ``label``, averaged within each
    day, then the day-clustered series stats."""
    f = df[["run", "signal_date", "score", label]].dropna()
    if f.empty:
        return em._series_stats([])
    rs = f.groupby("run")["score"].rank(pct=True).to_numpy()
    rl = f.groupby("run")[label].rank(pct=True).to_numpy()
    g = pd.DataFrame({"run": f["run"].to_numpy(), "d": f["signal_date"].to_numpy(),
                      "a": rs, "b": rl, "ab": rs * rl, "aa": rs * rs, "bb": rl * rl})
    s = g.groupby("run").agg(d=("d", "first"), n=("a", "size"), a=("a", "sum"), b=("b", "sum"),
                             ab=("ab", "sum"), aa=("aa", "sum"), bb=("bb", "sum"))
    s = s[s["n"] >= em.MIN_RUN_ROWS]
    cov = s["ab"] - s["a"] * s["b"] / s["n"]
    va = s["aa"] - s["a"] ** 2 / s["n"]
    vb = s["bb"] - s["b"] ** 2 / s["n"]
    ic = cov / np.sqrt(va * vb)
    daily = ic.groupby(s["d"]).mean().sort_index()
    return em._series_stats(daily)


def report(df: pd.DataFrame, side: str, windows: Dict[str, Tuple[str, Optional[str]]]) -> dict:
    """The three views for one side on each window: the selection objective
    (pivot label), the same picks' realizable returns, and per-run IC."""
    sel = em.selection_by_windows(df, "score", side, windows, run="run")
    ent = em.selection_entries(df, "score", side, run="run")
    sign = 1.0 if side == "long" else -1.0
    out = {}
    for name, (lo, hi) in windows.items():
        w = df[(df["signal_date"] >= lo) & ((df["signal_date"] <= hi) if hi else True)]
        e = ent[(ent["_d"] >= lo) & ((ent["_d"] <= hi) if hi else True)]
        realized = {k: em._series_stats((sign * e[k]).groupby(e["_d"]).mean().sort_index())
                    for k in ("fsc", "f1d", "f5d")}
        if "xl" in e.columns:
            tr, trpd = trail_returns(e, side)
            realized["trail"] = em._series_stats(pd.Series(tr, index=e.index).groupby(e["_d"]).mean().sort_index())
            realized["trail_rpd"] = em._series_stats(
                pd.Series(trpd, index=e.index).groupby(e["_d"]).mean().sort_index())
        ctl = vol_control(df, e, side)
        mine = side_returns(e, side)
        excess = {}
        for k in ctl.columns:
            d_ctl = ctl[k].groupby(e["_d"]).mean().sort_index()
            d_exc = (mine[k] - ctl[k]).groupby(e["_d"]).mean().sort_index()
            excess[k] = {"control": em._series_stats(d_ctl), "excess": em._series_stats(d_exc)}
        ics = {k: run_ic(w, k) for k in ("fwd_ret_pivot", "fsc", "f1d", "f5d")}
        out[name] = {"selection": sel[name], "picks_realized": realized, "vol_matched": excess, "run_ic": ics}
    return out


def _brief(r: dict) -> str:
    s = r["selection"]
    ob = s.get("objective", {})
    rz = r["picks_realized"]
    tr = rz.get("trail_rpd")
    trail = f" | trail/day {tr['mean']:+.3f} (t {tr['t']:+.2f})" if tr else ""
    vm = r.get("vol_matched", {})
    if vm.get("f1d"):
        x1, xt = vm["f1d"]["excess"], vm.get("trail_rpd", {}).get("excess", {})
        trail += (f" | vs vol-matched: 1d {x1['mean']:+.3f} (t {x1['t']:+.2f})"
                  + (f" trail/day {xt['mean']:+.3f} (t {xt['t']:+.2f})" if xt else ""))
    return (f"rpd {ob.get('mean', float('nan')):+.3f} (t {ob.get('t', float('nan')):+.2f}, "
            f"halves {ob.get('half1', float('nan')):+.3f}/{ob.get('half2', float('nan')):+.3f}) "
            f"n/day {s.get('entries_per_day', float('nan')):.2f} | realized sc "
            f"{rz['fsc']['mean']:+.3f} 1d {rz['f1d']['mean']:+.3f} (t {rz['f1d']['t']:+.2f}) "
            f"5d {rz['f5d']['mean']:+.3f}{trail}")


# ── experiments ────────────────────────────────────────────────────────────

def validate(objectives: Sequence[str] = OBJECTIVES, sides: Sequence[str] = ("long", "short"),
             fset: str = "deep", cut: str = "2025-12-31", val: Tuple[str, str] = ("2026-01-02", "2026-04-30"),
             rounds: int = 400, checkpoints: Iterable[int] = (50, 100, 200, 300, 400), threads: int = 8,
             tag: str = "val", params: Optional[dict] = None, kind: str = "intraday",
             target: str = "pivot") -> dict:
    """Fit on sessions <= ``cut`` (labels confirmed by then), then score the
    training arrays' rows of the ``val`` window at each checkpoint — the
    choice of objective, feature set and tree count, made before any test-set
    row is touched."""
    _lowprio()
    train_dir, _eval_dir, prefix = KINDS[kind]
    arr = Arrays(train_dir)
    tr = training_rows(arr, cut, target=target)
    vr = np.flatnonzero((arr.dn >= dnum(val[0])) & (arr.dn <= dnum(val[1])) & arr.tradeable())
    windows = {"val": val}
    tag = prefix + tag
    out_dir = OUT_DIR / tag
    out_dir.mkdir(parents=True, exist_ok=True)
    res: dict = {"kind": kind, "target": target, "cut": cut, "val": list(val), "fset": fset, "rounds": rounds,
                 "params": dict(LGB_PARAMS, **(params or {})),
                 "n_train": int(len(tr)), "n_val": int(len(vr)), "runs": {}}
    for obj in objectives:
        for side in (("both",) if obj == "rankreg" else sides):
            b = fit(arr, tr, side, obj, fset, rounds, threads, params, target=target)
            b.save_model(str(out_dir / f"{obj}_{side}.txt"))
            for k in [k for k in checkpoints if k <= rounds]:
                pred = predict(b, arr, vr, fset, num_iteration=k)
                for s in (sides if side == "both" else (side,)):
                    df = frame(arr, vr, oriented(pred, s, obj))
                    r = report(df, s, windows)["val"]
                    res["runs"][f"{obj}|{s}|{k}"] = r
                    logger.info(f"[sel] {tag} {obj:10s} {s:5s} @{k:4d}: {_brief(r)}")
            (out_dir / "report.json").write_text(json.dumps(res, default=float), encoding="utf-8")
    return res


FINAL_WINDOWS = {"warmup_oos": ("2026-05-01", "2026-06-16"),
                 "set1_history": em.TEST_SETS["set1_history"],
                 "set2_live_all_source": em.TEST_SETS["set2_live_all_source"]}
LIVE_ARTIFACT = Path("cache/ml/ml_ohlcv_model.pkl")


def final(arms: Sequence[Tuple[str, str, int]], fset: str = "deep", cut: str = "2026-04-30",
          threads: int = 8, tag: str = "final", params: Optional[dict] = None,
          live_baseline: bool = True, kind: str = "intraday", target: str = "pivot") -> dict:
    """Fit each ``(objective, side, rounds)`` arm on sessions <= ``cut`` and
    score EVERY bar of the evaluation arrays (from 2026-05-01: out of sample for
    these fits, and the own-history warm-up for set 1), reported on
    `FINAL_WINDOWS`. The live ml_ohlcv artifact is scored on the same rows as a
    reference — it trained through 2026-06-01, so its ``warmup_oos`` window is
    IN-sample and only its set 1 is a fair comparison."""
    _lowprio()
    train_dir, eval_dir, prefix = KINDS[kind]
    arr = Arrays(train_dir)
    ev = arr if eval_dir == train_dir else Arrays(eval_dir)
    tr = training_rows(arr, cut, target=target)
    er = np.flatnonzero(ev.tradeable() & (ev.dn >= dnum(EVAL_FROM)))
    tag = prefix + tag
    out_dir = OUT_DIR / tag
    out_dir.mkdir(parents=True, exist_ok=True)
    res: dict = {"kind": kind, "target": target, "cut": cut, "fset": fset,
                 "params": dict(LGB_PARAMS, **(params or {})),
                 "n_train": int(len(tr)),
                 "n_eval": int(len(er)), "windows": FINAL_WINDOWS, "runs": {},
                 "built_at": ev.meta.get("built_at")}

    def score(name: str, pred: np.ndarray, obj: str, sides: Sequence[str]) -> None:
        np.save(out_dir / f"pred_{name}.npy", pred.astype(np.float32))
        for s in sides:
            r = report(frame(ev, er, oriented(pred, s, obj)), s, FINAL_WINDOWS)
            res["runs"][f"{name}|{s}"] = r
            for w in ("warmup_oos", "set1_history"):
                logger.info(f"[sel] {tag} {name:22s} {s:5s} {w:12s}: {_brief(r[w])}")
        (out_dir / "report.json").write_text(json.dumps(res, default=float), encoding="utf-8")

    np.save(out_dir / "eval_rows.npy", er)
    if live_baseline and kind == "intraday" and LIVE_ARTIFACT.exists():
        import pickle
        art = pickle.load(open(LIVE_ARTIFACT, "rb"))
        if list(art["features"]) != list(ev.meta["base_features"]):
            raise SystemExit("live artifact features differ from the arrays' base features")
        score("live_ml_ohlcv", predict(art["model"]._booster, ev, er, "base"), "rankreg", ("long", "short"))
    for obj, side, rounds in arms:
        b = fit(arr, tr, side, obj, fset, rounds, threads, params, target=target)
        name = f"{obj}_{side}_{rounds}"
        b.save_model(str(out_dir / f"{name}.txt"))
        score(name, predict(b, ev, er, fset), obj, ("long", "short") if side == "both" else (side,))
    return res


def score_saved(tag: str, name: str, side: str, phase: str,
                val: Tuple[str, str] = ("2026-01-02", "2026-04-30"), kind: str = "intraday") -> dict:
    """Score predictions another process saved — the MLP arm, which runs in the
    torch venv (`sel_deep`) — on the full objective. Its output is the side's
    CONVICTION, oriented like a side model. ``phase="val"``: the training
    arrays' validation rows (``val_rows.npy``) on the ``val`` window;
    ``"final"``: the evaluation arrays' rows (``rows_<name>.npy``) on
    `FINAL_WINDOWS`."""
    train_dir, eval_dir, prefix = KINDS[kind]
    out_dir = OUT_DIR / (prefix + tag)
    pred = np.load(out_dir / f"pred_{name}.npy").astype(float)
    if phase == "val":
        arr, rows, windows = Arrays(train_dir), np.load(out_dir / "val_rows.npy"), {"val": val}
    else:
        arr, rows, windows = Arrays(eval_dir), np.load(out_dir / f"rows_{name}.npy"), FINAL_WINDOWS
    rep = report(frame(arr, rows, oriented(pred, side, "side")), side, windows)
    path = out_dir / "report_saved.json"
    allrep = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    allrep[f"{name}|{side}"] = rep
    path.write_text(json.dumps(allrep, default=float), encoding="utf-8")
    for w in windows:
        logger.info(f"[sel] {tag} {name:22s} {side:5s} {w:12s}: {_brief(rep[w])}")
    return rep


def rescore(tag: str, kind: str = "intraday", phase: str = "val", fset: str = "deep",
            checkpoints: Iterable[int] = (50, 100, 200, 300, 400),
            val: Tuple[str, str] = ("2026-01-02", "2026-04-30"), threads: int = 2) -> dict:
    """Re-run `report` on models already trained — after a metric is added
    (the trailing exits), without refitting. ``phase="val"``: every saved
    booster at each checkpoint on the validation rows; ``"final"``: every saved
    prediction (``pred_<name>.npy`` on ``eval_rows.npy``) on `FINAL_WINDOWS`."""
    import lightgbm as lgb
    train_dir, eval_dir, prefix = KINDS[kind]
    out_dir = OUT_DIR / (prefix + tag)
    res: dict = {}
    if phase == "val":
        arr = Arrays(train_dir)
        rows = np.flatnonzero((arr.dn >= dnum(val[0])) & (arr.dn <= dnum(val[1])) & arr.tradeable())
        for mf in sorted(out_dir.glob("*.txt")):
            obj, side = mf.stem.split("_")[:2]
            b = lgb.Booster(model_file=str(mf))
            for k in [k for k in checkpoints if k <= b.current_iteration()]:
                pred = predict(b, arr, rows, fset, num_iteration=k, threads=threads)
                for s in (("long", "short") if side == "both" else (side,)):
                    r = report(frame(arr, rows, oriented(pred, s, obj)), s, {"val": val})["val"]
                    res[f"{obj}|{s}|{k}"] = r
                    logger.info(f"[sel] rescore {tag} {obj:10s} {s:5s} @{k:4d}: {_brief(r)}")
    else:
        arr = Arrays(eval_dir)
        rows = np.load(out_dir / "eval_rows.npy")
        for pf in sorted(out_dir.glob("pred_*.npy")):
            name = pf.stem[len("pred_"):]
            if name.startswith("mlp_"):
                continue                                    # the MLP arm is scored by `score_saved`
            obj, side = name.split("_")[0], name.split("_")[1]
            if name == "live_ml_ohlcv":
                obj, side = "rankreg", "both"
            pred = np.load(pf).astype(float)
            for s in (("long", "short") if side == "both" else (side,)):
                r = report(frame(arr, rows, oriented(pred, s, obj)), s, FINAL_WINDOWS)
                res[f"{name}|{s}"] = r
                for w in ("warmup_oos", "set1_history"):
                    logger.info(f"[sel] rescore {tag} {name:22s} {s:5s} {w:12s}: {_brief(r[w])}")
    (out_dir / "report_rescored.json").write_text(json.dumps(res, default=float), encoding="utf-8")
    return res


def daily_series(tag: str, name: str, side: str, metric: str = "rpd",
                 kind: str = "intraday") -> pd.Series:
    """One arm's per-day series on a final run's saved predictions: the mean
    entry ``rpd`` (the objective), side-oriented ``f1d`` (next close) or
    ``trail`` (trailing exit per day), indexed by signal date."""
    train_dir, eval_dir, prefix = KINDS[kind]
    out_dir = OUT_DIR / (prefix + tag)
    arr = Arrays(eval_dir)
    rows = np.load(out_dir / (f"rows_{name}.npy" if name.startswith("mlp_") else "eval_rows.npy"))
    pred = np.load(out_dir / f"pred_{name}.npy").astype(float)
    obj = "rankreg" if name == "live_ml_ohlcv" or name.startswith("rankreg") else "side"
    e = em.selection_entries(frame(arr, rows, oriented(pred, side, obj)), "score", side, run="run")
    if metric == "rpd":
        v = e["_rpd"]
    elif metric == "f1d":
        v = (1.0 if side == "long" else -1.0) * e["f1d"]
    elif metric == "trail":
        v = pd.Series(trail_returns(e, side)[1], index=e.index)
    else:
        raise ValueError(metric)
    return v.groupby(e["_d"]).mean().sort_index()


def paired(tag: str, a: str, b: str, side: str, metric: str = "rpd",
           window: Tuple[str, Optional[str]] = em.TEST_SETS["set1_history"], kind: str = "intraday") -> dict:
    """The house bar's paired contrast: per-day ``a - b`` on the days both
    arms took an entry in ``window`` — mean, day-clustered t, halves."""
    sa, sb = daily_series(tag, a, side, metric, kind), daily_series(tag, b, side, metric, kind)
    lo, hi = window
    d = (sa - sb).dropna()
    d = d[(d.index >= lo) & ((d.index <= hi) if hi else True)]
    return em._series_stats(d)


def summarize(tag: str, window: Optional[str] = None) -> pd.DataFrame:
    """One row per (arm, side[, checkpoint], window) from a run's report files:
    the objective (mean return per day, t, halves), entries per day, the picks'
    realizable returns and the per-run ICs."""
    d = OUT_DIR / tag
    rows = []
    for fn in ("report.json", "report_saved.json", "report_rescored.json"):
        p = d / fn
        if not p.exists():
            continue
        rep = json.loads(p.read_text(encoding="utf-8"))
        runs = rep.get("runs", rep) if fn == "report.json" else rep
        for key, r in runs.items():
            wins = {"val": r} if "selection" in r else r
            for w, x in wins.items():
                if window and w != window:
                    continue
                sel, rz, ic = x["selection"], x["picks_realized"], x.get("run_ic", {})
                vm = x.get("vol_matched", {})
                ob = sel.get("objective", {})
                rows.append({"arm": key, "window": w, "days": sel.get("days"),
                             "n/day": sel.get("entries_per_day"), "rpd": ob.get("mean"), "t": ob.get("t"),
                             "h1": ob.get("half1"), "h2": ob.get("half2"),
                             "trail/d": rz.get("trail_rpd", {}).get("mean"), "t_tr": rz.get("trail_rpd", {}).get("t"),
                             "sc": rz["fsc"]["mean"], "1d": rz["f1d"]["mean"], "t1d": rz["f1d"]["t"],
                             "5d": rz["f5d"]["mean"], "t5d": rz["f5d"]["t"],
                             "x1d": vm.get("f1d", {}).get("excess", {}).get("mean"),
                             "tx1d": vm.get("f1d", {}).get("excess", {}).get("t"),
                             "xtr/d": vm.get("trail_rpd", {}).get("excess", {}).get("mean"),
                             "txtr": vm.get("trail_rpd", {}).get("excess", {}).get("t"),
                             "ic_pv": ic.get("fwd_ret_pivot", {}).get("mean"),
                             "ic_1d": ic.get("f1d", {}).get("mean"), "ic_5d": ic.get("f5d", {}).get("mean")})
    out = pd.DataFrame(rows)
    # later files supersede earlier ones for the same arm (a rescore after a metric fix)
    return out.drop_duplicates(["arm", "window"], keep="last").reset_index(drop=True) if len(out) else out


def main(argv=None) -> None:
    import argparse
    ap = argparse.ArgumentParser(description="selection-objective models (long/short)")
    ap.add_argument("--summary", default="", help="print the summary table of a run tag")
    ap.add_argument("--rescore", default="", help="'tag:phase' — re-report trained models (phase val|final)")
    ap.add_argument("--score-saved", default="", help="'name:side:phase,...' predictions saved by sel_deep")
    ap.add_argument("--validate", action="store_true")
    ap.add_argument("--final", default="", help="arms 'objective:side:rounds,...', e.g. lambdarank:long:300")
    ap.add_argument("--objectives", default=",".join(OBJECTIVES))
    ap.add_argument("--sides", default="long,short")
    ap.add_argument("--fset", default="deep", choices=("base", "deep", "all"))
    ap.add_argument("--rounds", type=int, default=400)
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--tag", default="val")
    ap.add_argument("--kind", default="intraday", choices=tuple(KINDS))
    ap.add_argument("--target", default="pivot", choices=TARGETS)
    a = ap.parse_args(argv)
    logger.add("logs/sel_models.log", rotation="1 day", retention="30 days", level="INFO", enqueue=True)
    if a.validate:
        validate(objectives=[o for o in a.objectives.split(",") if o],
                 sides=[s for s in a.sides.split(",") if s], fset=a.fset, rounds=a.rounds,
                 threads=a.threads, tag=a.tag, kind=a.kind, target=a.target)
    if a.final:
        arms = [(o, s, int(r)) for o, s, r in (x.split(":") for x in a.final.split(",") if x)]
        final(arms, fset=a.fset, threads=a.threads, tag=a.tag if a.tag != "val" else "final", kind=a.kind,
              target=a.target)
    if a.rescore:
        tg, ph = a.rescore.split(":")
        rescore(tg, kind=a.kind, phase=ph, fset=a.fset, threads=min(a.threads, 2))
    if a.summary:
        with pd.option_context("display.width", 250, "display.max_rows", 500, "display.float_format", "{:+.3f}".format):
            print(summarize(a.summary).to_string(index=False))
    if a.score_saved:
        for name, side, phase in (x.split(":") for x in a.score_saved.split(",") if x):
            score_saved(a.tag if phase == "val" else ("final" if a.tag == "val" else a.tag), name, side, phase,
                        kind=a.kind)


if __name__ == "__main__":
    main()
