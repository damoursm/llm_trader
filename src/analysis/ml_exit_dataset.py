"""ML EXIT model — Phase 0 dataset + go/no-go harness.

The entry stackers decide direction/conviction; this asks the complementary
question — for a position ALREADY held, should we close it now? An exit model is
arguably a better fit for ML than entry (see memory/ml-methods-plan-2026-07.md):
its features are richer and exit-specific (MFE / MAE / days held / the combine's
conviction DEGRADATION since entry / the method-horizon elapsed ratio), and its
label is a clean counterfactual — since 2026-08-13 the ORIENTED REMAINING MOVE
to the next H/L pivot from the held day (``+`` = the leg still runs our way,
``−`` = the next turn is against us; ``ml_exit_label_basis``, fail-soft to the
fixed held-return label ``fwd_ret_pos_<h>d`` below 500 settled rows).

**The dataset is SIMULATED held positions over the signals panel.** For every
scored ticker-day with a directional view, a hypothetical position is opened in
that direction and walked forward day by day; at each held day it emits the
exit-state features + the method scores re-scored on that day (oriented to the
position) + the labels (the panel's own settled pivot target — the training
label — and the fixed forward return, both oriented; each with its own
point-in-time settle date). So it needs nothing the panel + build_panel forward-return join don't
already provide, and it is causal by construction (state uses prices ≤ held day;
label uses days > held day; the walk-forward's ``end_date`` guard removes the
rest).

**The baseline to beat is ``exit_conviction.exit_method_consensus``** — the
current hand-built exit decision (a plain mean of the oriented signal methods).
The ML model's edge is (a) a LEARNED, non-linear combination of the same methods
and (b) the position-state features the consensus throws away. It reuses the
entry harness (``ml_train.walk_forward_predict``/``evaluate``) unchanged — an exit
model predicting the sign of the oriented held return is structurally identical
to an entry model predicting the sign of the forward return.

CLI:  python -m src.analysis.ml_exit_dataset [--basis pv|fixed] [--horizon 3] [--max-hold 12] [--train]
"""

from __future__ import annotations

import argparse
from bisect import bisect_left
from datetime import date as _date
from typing import List, Optional, Sequence

import numpy as np
import pandas as pd
from loguru import logger

from config.settings import settings
from src.analysis.exit_conviction import exit_method_consensus
from src.analysis.exit_methods import method_horizon_days
from src.analysis.ml_dataset import _benchmark_series
from src.analysis.ml_stacker import STACKER_LIVE_FEATURES

# The method scores the exit model sees (oriented to the position) — the same 21
# weighted methods the consensus uses, so the comparison is apples-to-apples on
# the methods and the ML model's win must come from LEARNING + the state features.
EXIT_METHODS: List[str] = list(STACKER_LIVE_FEATURES)

# Position-state features — the exit-specific signal the consensus cannot see.
EXIT_STATE_FEATURES: List[str] = [
    "days_held", "ex_ret", "ex_mfe", "ex_mae", "ex_giveback", "ex_from_mae",
    "ex_combine", "ex_combine_delta", "ex_elapsed_ratio",
]
EXIT_FEATURE_COLUMNS: List[str] = EXIT_STATE_FEATURES + [f"ex_{m}" for m in EXIT_METHODS]


def _dir_sign(direction) -> float:
    d = str(direction or "").upper()
    return 1.0 if "BULL" in d else (-1.0 if "BEAR" in d else 0.0)


def build_exit_dataset(horizon: int = 3, days: Optional[int] = None,
                       max_hold: int = 12, entry_stride: int = 1) -> pd.DataFrame:
    """Simulate held positions over the panel; one row per (position, held-day).

    ``horizon`` = the label look-ahead (sessions held from the current day).
    ``max_hold`` caps how long a hypothetical position is walked; ``entry_stride``
    subsamples entry days per ticker. Returns an ``ml_train``-compatible frame:
    ``EXIT_FEATURE_COLUMNS`` + ``ex_consensus`` (baseline) + ``fwd_ret_pos_<h>d``
    (the oriented held-return label) + ``end_date_<h>d`` (point-in-time guard).
    """
    from src.analysis.signal_panel import build_panel
    panel = build_panel(horizons=[horizon], days=days)
    if panel is None or panel.empty:
        return pd.DataFrame()
    fwdcol = f"fwd_ret_{horizon}d"
    mcols = [m for m in EXIT_METHODS if m in panel.columns]
    # Benchmark session grid for the point-in-time end date (the ticker's own
    # panel days are sparse — not every session — so counting horizons on them
    # would be wrong; SPY trades every session).
    b_dates, _ = _benchmark_series(settings.horizon_market_benchmark)
    rows: List[dict] = []

    for tk, g in panel.groupby("ticker"):
        g = g.sort_values("signal_date").reset_index(drop=True)
        n = len(g)
        if n < 2:
            continue
        dates = g["signal_date"].astype(str).tolist()
        px = pd.to_numeric(g.get("price"), errors="coerce").tolist()
        dirs = g["direction"].tolist()
        # ex_combine standardizes on the ABSOLUTE combine (2026-08-14): the
        # shadow column where present (rank-era rows), else combined_score
        # (pre-shadow rows, which ARE absolute-basis) — so the feature series
        # is basis-invariant across the 2026-08-14 rank switch and any future
        # shape/TTL drift. The live twin in `live_exit_features` mirrors this.
        _abs = (pd.to_numeric(g.get("combined_score_abs"), errors="coerce")
                if "combined_score_abs" in g else None)
        _liv = pd.to_numeric(g.get("combined_score"), errors="coerce")
        comb = (_abs.where(_abs.notna(), _liv) if _abs is not None else _liv).tolist()
        fwd = pd.to_numeric(g.get(fwdcol), errors="coerce").tolist() if fwdcol in g else [np.nan] * n
        # The pivot label (2026-08-13 standardization): the panel's own settled
        # H/L pivot target per held day + its per-row settle date — oriented
        # below, it reads "the remaining move to the next turn": the user's
        # rank-degradation exit thesis as a label.
        pv = (pd.to_numeric(g.get("fwd_ret_pivot"), errors="coerce").tolist()
              if "fwd_ret_pivot" in g else [np.nan] * n)
        pv_end = (g["end_date_pivot"].astype(str).tolist()
                  if "end_date_pivot" in g else [None] * n)
        ms = {m: pd.to_numeric(g[m], errors="coerce").tolist() for m in mcols}

        for ei in range(0, n - 1, entry_stride):
            ds = _dir_sign(dirs[ei])
            if ds == 0.0:
                continue
            ep, ec = px[ei], comb[ei]
            if not ep or ep <= 0 or ep != ep:
                continue
            entry_scores = {m: ms[m][ei] for m in mcols if ms[m][ei] == ms[m][ei]}
            mh = method_horizon_days({"method_scores": entry_scores}) or 0.0
            mfe = mae = 0.0
            for k in range(1, max_hold + 1):
                hi = ei + k
                if hi >= n:
                    break
                hp = px[hi]
                if not hp or hp <= 0 or hp != hp:
                    break
                oret = ds * (hp / ep - 1.0) * 100.0
                mfe = max(mfe, oret)
                mae = min(mae, oret)
                hc = comb[hi]
                rec = {
                    "ticker": tk, "signal_date": dates[hi], "days_held": float(k),
                    # Position identity (ticker + entry day) — lets a SEQUENTIAL
                    # policy simulator group a position's held-days and walk them
                    # in order. Not a feature.
                    "entry_date": dates[ei],
                    "ex_ret": oret, "ex_mfe": mfe, "ex_mae": mae,
                    "ex_giveback": mfe - oret, "ex_from_mae": oret - mae,
                    "ex_combine": (ds * hc if hc == hc else np.nan),
                    "ex_combine_delta": (ds * (hc - ec) if hc == hc and ec == ec else np.nan),
                    "ex_elapsed_ratio": (k / mh if mh > 0 else np.nan),
                }
                for m in mcols:
                    v = ms[m][hi]
                    rec[f"ex_{m}"] = (ds * v if v == v else np.nan)
                # Baseline: the exact hand-built consensus (oriented methods +
                # the method-horizon pressure), the number the live exit uses.
                mh_press = 0.0 if (mh <= 0 or k < mh) else -min(1.0, k / mh - 1.0)
                oriented = {m: rec[f"ex_{m}"] for m in mcols if rec[f"ex_{m}"] == rec[f"ex_{m}"]}
                cons = exit_method_consensus({**oriented, "method_horizon": mh_press})
                rec["ex_consensus"] = cons if cons is not None else 0.0
                # Label: the oriented forward return from THIS held day (+ = keep
                # was right). Absolute (P&L-relevant for an exit), oriented by dir.
                f = fwd[hi]
                rec[f"fwd_ret_pos_{horizon}d"] = (ds * f if f == f else np.nan)
                # PIVOT label twin: the oriented remaining move to the next H/L
                # pivot from this held day (+ = the leg still runs our way,
                # − = the next turn is against us) with ITS OWN settle date.
                pvv = pv[hi]
                rec["fwd_ret_pos_pv"] = (ds * pvv if pvv == pvv else np.nan)
                rec["end_date_pv"] = (pv_end[hi] if pvv == pvv else None)
                # end date = held_day + horizon SESSIONS on the benchmark grid.
                hd = _date.fromisoformat(dates[hi])
                bi = bisect_left(b_dates, hd) if b_dates else 0
                rec[f"end_date_{horizon}d"] = (b_dates[bi + horizon].isoformat()
                                               if b_dates and bi + horizon < len(b_dates) else None)
                rows.append(rec)

    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    # Drop rows with no point-in-time end date (unusable for the walk-forward
    # split) — mirrors the entry dataset, and keeps None out of the harness.
    df = df[df[f"end_date_{horizon}d"].notna()].sort_values(["signal_date", "ticker"]).reset_index(drop=True)
    logger.info(f"[ml_exit] built {len(df):,} held-position-days over {df['ticker'].nunique()} "
                f"tickers ({df['signal_date'].min()} .. {df['signal_date'].max()})")
    return df


def measure_exit(horizon: int = 3, days: Optional[int] = None, deadband: float = 0.0,
                 max_hold: int = 12, model_name: str = "gbm",
                 min_train_days: int = 8, step_days: int = 2,
                 min_train_rows: int = 500, basis: str = "fixed") -> pd.DataFrame:
    """Walk-forward the ML exit model vs the hand-built ``exit_method_consensus``
    on the oriented held return. The go/no-go: does the learned 'keep-holding'
    score predict the held outcome (IC/ICIR/hit) better than the consensus?

    A dedicated loop (not ``ml_train.evaluate``) because the exit frame has MANY
    rows per (signal_date, ticker) — one per held-day — so the entry harness's
    baseline join on (date, ticker) would explode. Here both scores are read off
    the SAME out-of-sample rows, so the comparison is exact."""
    from datetime import date as _d
    from src.analysis.ml_train import _metrics, label_from_return, make_model_factory
    df = build_exit_dataset(horizon=horizon, days=days, max_hold=max_hold)
    if df.empty:
        return pd.DataFrame()
    # ``basis="pv"`` measures on the oriented pivot label + its own settle
    # date (what the production trainer now prefers); "fixed" keeps the
    # held-return label at ``horizon``. Metrics/IC read the SAME ycol either
    # way, so the two runs are directly comparable.
    if basis == "pv" and "fwd_ret_pos_pv" in df.columns:
        ycol, ecol = "fwd_ret_pos_pv", "end_date_pv"
    else:
        ycol, ecol = f"fwd_ret_pos_{horizon}d", f"end_date_{horizon}d"
    feats = [f for f in EXIT_FEATURE_COLUMNS if f in df.columns]
    work = df[["signal_date", "ex_consensus", ycol, ecol] + feats].copy()
    work[ycol] = pd.to_numeric(work[ycol], errors="coerce")
    work = work[work[ycol].notna()].reset_index(drop=True)
    work["_y"] = work[ycol].map(lambda r: label_from_return(r, deadband))
    work = work[work["_y"].notna()].reset_index(drop=True)
    if work.empty:
        return pd.DataFrame()
    work = work[work[ecol].notna()].reset_index(drop=True)
    if work.empty:
        return pd.DataFrame()
    work["_sig"] = work["signal_date"].map(lambda s: _d.fromisoformat(str(s)[:10]))
    work["_end"] = work[ecol].map(lambda s: _d.fromisoformat(str(s)[:10]))
    X = work[feats].to_numpy(dtype=float)
    y = work["_y"].to_numpy(dtype=int)
    uniq = sorted(work["_sig"].unique())
    if len(uniq) <= min_train_days + 1:
        return pd.DataFrame()
    factory = make_model_factory(model_name)
    out: List[pd.DataFrame] = []
    for a in range(min_train_days, len(uniq), step_days):
        cutoff = uniq[a]
        nxt = uniq[a + step_days] if a + step_days < len(uniq) else None
        tr = work["_end"].to_numpy() < cutoff              # label printed before cutoff
        sig = work["_sig"].to_numpy()
        te = (sig >= cutoff) if nxt is None else ((sig >= cutoff) & (sig < nxt))
        if tr.sum() < min_train_rows or te.sum() == 0 or len(np.unique(y[tr])) < 2:
            continue
        model = factory().fit(X[tr], y[tr])
        bull, bear = model.bull_bear(X[te])
        sub = work.loc[te, ["signal_date", ycol, "ex_consensus"]].copy()
        sub["net"] = bull - bear
        out.append(sub)
    if not out:
        return pd.DataFrame()
    allp = pd.concat(out, ignore_index=True)
    return pd.DataFrame([
        {"model": f"{model_name} (state+methods)",
         **_metrics(allp["signal_date"], allp["net"], allp[ycol])},
        {"model": "ex_consensus (baseline)",
         **_metrics(allp["signal_date"], allp["ex_consensus"], allp[ycol])},
    ])


# ── Phase 1: live inference + the persisted model ─────────────────────────────
# The learned exit-timer served on live held positions. Native Booster (no
# sklearn), so the artifact loads in the production .venv. Fail-soft everywhere:
# a missing artifact / lightgbm returns None and the caller keeps the hand-built
# exit machinery — an invisible degradation to a broken model is impossible.
#
# Coupling to the ENTRY arm (2026-08-02): this model DRIVES exits only for trades
# stamped ``ml_arm`` (opened while the ml_buy/ml_sell arm was active), so
# the ML entry and ML exit travel together as one A/B bundle. For every OTHER
# held position it is still computed and persisted to the exit_signals panel
# (build_exit_scores), so its exit-side IC accrues live before it acts anywhere.

import pickle as _pickle
from datetime import datetime as _dt, timezone as _tz
from pathlib import Path as _Path

_EXIT_MODEL_PATH = _Path("cache/ml/ml_exit_model.pkl")
# horizon overridden from settings.ml_exit_horizon_days at train time; 5d is the
# measured-stronger label (see memory/ml-methods-plan-2026-07.md — h=5 hit 53.4%,
# ICIR +0.30, vs the anti-predictive ex_consensus at both 3d and 5d).
EXIT_TRAIN_CONFIG = dict(horizon=5, deadband=0.0, max_hold=15)
_EXIT_ART: dict = {"mtime": None, "art": None}


def train_and_persist_exit(days: Optional[int] = None, path=_EXIT_MODEL_PATH) -> Optional[dict]:
    """Train the exit model on the simulated held-position dataset (oriented
    held-return label at the configured horizon) over EXIT_FEATURE_COLUMNS; pickle
    it. Returns the artifact or None (fail-soft on thin data / no lightgbm)."""
    import numpy as _np
    from src.analysis.ml_train import LightGBMModel, label_from_return
    cfg = dict(EXIT_TRAIN_CONFIG)
    cfg["horizon"] = int(settings.ml_exit_horizon_days)
    h = cfg["horizon"]
    df = build_exit_dataset(horizon=h, days=days, max_hold=cfg["max_hold"])
    ycol = f"fwd_ret_pos_{h}d"
    if df.empty or ycol not in df.columns:
        logger.warning("[ml_exit] no exit data to train on")
        return None
    # Label basis (2026-08-13 standardization directive, mirroring the
    # stackers' `_label_cfg`): the oriented remaining-move-to-the-next-pivot
    # label when enough rows have SETTLED, else the fixed held-return label.
    # `ml_exit_label_basis="fixed"` pins the old behaviour. The basis is
    # stamped in the artifact config either way.
    cfg["combine_basis"] = "absolute"      # ex_combine reads the abs shadow (2026-08-14)
    want_pv = str(getattr(settings, "ml_exit_label_basis", "pv")).lower() == "pv"
    if want_pv and "fwd_ret_pos_pv" in df.columns:
        n_pv = int(pd.to_numeric(df["fwd_ret_pos_pv"], errors="coerce").notna().sum())
        if n_pv >= 500:
            ycol, cfg["label_basis"] = "fwd_ret_pos_pv", "pv"
            logger.info(f"[ml_exit] label: oriented PIVOT target ({n_pv:,} settled rows)")
        else:
            cfg["label_basis"] = f"pos_{h}d"
            logger.info(f"[ml_exit] pivot label too thin ({n_pv}) — keeping pos_{h}d")
    else:
        cfg["label_basis"] = f"pos_{h}d"
    feats = [f for f in EXIT_FEATURE_COLUMNS if f in df.columns]
    y_raw = pd.to_numeric(df[ycol], errors="coerce").map(lambda r: label_from_return(r, cfg["deadband"]))
    keep = y_raw.notna()
    X = df.loc[keep, feats].to_numpy(dtype=float)
    y = y_raw[keep].to_numpy(dtype=int)
    if len(X) < 500 or len(_np.unique(y)) < 2:
        logger.warning(f"[ml_exit] insufficient training rows ({len(X)})")
        return None
    model = LightGBMModel().fit(X, y)
    art = {"model": model, "features": feats, "config": dict(cfg),
           "trained_at": _dt.now(_tz.utc).isoformat(timespec="seconds"),
           "n_train": int(len(X)), "train_max_date": str(df["signal_date"].max())}
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as fh:
        _pickle.dump(art, fh)
    _EXIT_ART.update(mtime=None, art=None)
    _record_exit_registry(art)
    logger.info(f"[ml_exit] trained on {art['n_train']:,} rows (<= {art['train_max_date']}) -> {path}")
    return art


def _record_exit_registry(art: dict) -> None:
    try:
        import json
        from src.db.connection import connect
        with connect() as con:
            con.execute(
                "INSERT INTO ml_models (trained_at, method, model_type, horizon, basis, "
                "n_train, train_max_date, features, config) VALUES (?,?,?,?,?,?,?,?,?)",
                [art["trained_at"], "ml_exit", "gbm", int(art["config"]["horizon"]),
                 "held_return", art["n_train"], art["train_max_date"],
                 json.dumps(art["features"]), json.dumps(art["config"])])
    except Exception as e:
        logger.debug(f"[ml_exit] registry write skipped: {e}")


def _load_exit_artifact() -> Optional[dict]:
    if not _EXIT_MODEL_PATH.exists():
        return None
    try:
        mt = _EXIT_MODEL_PATH.stat().st_mtime_ns
        if _EXIT_ART["mtime"] == mt:
            return _EXIT_ART["art"]
        with open(_EXIT_MODEL_PATH, "rb") as fh:
            art = _pickle.load(fh)
        _EXIT_ART.update(mtime=mt, art=art)
        return art
    except Exception as e:
        logger.debug(f"[ml_exit] artifact load failed: {e}")
        _EXIT_ART.update(mtime=None, art=None)
        return None


def _oriented_close_path(ticker: str, ds: float, entry_price: float, entry_date: str) -> List[float]:
    """Oriented daily-close returns from entry (exclusive) to today — the live
    analog of the dataset's per-day price walk (panel price ≈ daily close), so the
    served MFE/MAE match what the model trained on. Fail-soft [] on no cache."""
    try:
        from src.data.cache import load_ohlcv
        from src.performance.daily_nav import _session_date
        df = load_ohlcv(ticker)
        if df is None or df.empty or "Close" not in df.columns:
            return []
        ed = str(entry_date or "")[:10]
        out: List[float] = []
        closes = pd.to_numeric(df["Close"], errors="coerce")
        for ts, c in zip(df.index, closes):
            sd = _session_date(ts)
            if sd is None or (ed and str(sd) <= ed):
                continue
            if c is not None and c == c and c > 0:
                out.append(ds * (float(c) / entry_price - 1.0) * 100.0)
        return out
    except Exception:
        return []


def live_exit_features(trade: dict, signals_by_ticker: Optional[dict], today_signal) -> Optional[dict]:
    """Assemble the exit model's feature vector for one live held position, in
    parity with ``build_exit_dataset``: the 9 position-state features (from the
    trade + its OHLCV close path + the current mark) and the 21 oriented method
    scores (the current tick's TickerSignal). None when the position is
    undirected / unpriced."""
    from src.performance.tracker import _method_scores_from_signal, _trading_days_held
    action = (trade.get("action") or "").upper()
    ds = 1.0 if action == "BUY" else (-1.0 if action == "SELL" else 0.0)
    if ds == 0.0:
        return None
    try:
        ep = float(trade.get("entry_price"))
        cur = float(trade.get("current_price"))
    except (TypeError, ValueError):
        return None
    if ep <= 0 or cur <= 0:
        return None

    ex_ret = ds * (cur / ep - 1.0) * 100.0                       # oriented, raw (matches dataset)
    path = _oriented_close_path(trade.get("ticker"), ds, ep, trade.get("entry_date"))
    path.append(ex_ret)
    mfe = max([0.0] + path)                                      # clamp at 0 (entry), like the walk
    mae = min([0.0] + path)
    try:
        days_held = float(_trading_days_held(trade["entry_date"]))
    except Exception:
        days_held = float(len(path))
    # Basis-invariant ex_combine (2026-08-14): prefer the ABSOLUTE twin —
    # matches the dataset build above, so the trained model's two combine
    # features never shift scale when the live combine basis (or a shaped
    # curve under its 6h TTL) moves.
    _abs_v = getattr(today_signal, "combined_score_abs", None)
    today_comb = (float(_abs_v) if _abs_v is not None
                  else float(getattr(today_signal, "combined_score", 0.0) or 0.0))
    _sae = trade.get("signal_at_entry") or {}
    entry_comb = _sae.get("combined_score_abs")
    if entry_comb is None:
        entry_comb = _sae.get("combined_score")
    try:
        ex_combine_delta = ds * (today_comb - float(entry_comb)) if entry_comb is not None else np.nan
    except (TypeError, ValueError):
        ex_combine_delta = np.nan
    mh = method_horizon_days(trade) or 0.0
    feats = {
        "days_held": days_held, "ex_ret": ex_ret, "ex_mfe": mfe, "ex_mae": mae,
        "ex_giveback": mfe - ex_ret, "ex_from_mae": ex_ret - mae,
        "ex_combine": ds * today_comb, "ex_combine_delta": ex_combine_delta,
        "ex_elapsed_ratio": (days_held / mh) if mh > 0 else np.nan,
    }
    oriented = _method_scores_from_signal(trade.get("ticker"), trade.get("direction"), signals_by_ticker)
    for m in EXIT_METHODS:
        v = oriented.get(m)
        try:
            feats[f"ex_{m}"] = ds * float(v) if v is not None and v == v else np.nan
        except (TypeError, ValueError):
            feats[f"ex_{m}"] = np.nan
    return feats


def compute_exit_model_score(trade: dict, signals_by_ticker: Optional[dict], today_signal) -> Optional[float]:
    """Signed HOLD-conviction ∈ [−1, +1] for one held position: ``+`` = keep
    running, ``−`` = exit (the exit-panel sign convention). ``2·P(keep good) − 1``,
    where the model's ``bull`` = P(the oriented held return is positive). None when
    the artifact/lightgbm/features are unavailable, so the caller keeps the
    hand-built exit — an invisible degradation is impossible."""
    art = _load_exit_artifact()
    if art is None:
        return None
    try:
        import numpy as _np
        feats = live_exit_features(trade, signals_by_ticker, today_signal)
        if feats is None:
            return None
        x = _np.array([[feats.get(f, _np.nan) for f in art["features"]]], dtype=float)
        keep, _exit = art["model"].bull_bear(x)                 # keep = P(held return > 0)
        return max(-1.0, min(1.0, 2.0 * float(keep[0]) - 1.0))
    except Exception as e:
        logger.debug(f"[ml_exit] score failed for {trade.get('ticker')}: {e}")
        return None


def eod_train_exit() -> Optional[dict]:
    """EOD entry point — retrain the exit model on the latest panel. Fail-soft."""
    return train_and_persist_exit()


def reset_exit_caches() -> None:
    """Test hook — drop the artifact memo."""
    _EXIT_ART.update(mtime=None, art=None)


def _print(table: pd.DataFrame, horizon: int) -> None:
    import sys
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    if table is None or table.empty:
        print("No exit-model results — too few held-position-days / panel too thin.")
        return
    print(f"\nML EXIT MODEL — walk-forward OOS on simulated held positions (label horizon {horizon}d)")
    print("net = P(keep good) − P(exit); label = oriented held return (+ = holding was right).")
    print("IC = Spearman(net, held return); the model must BEAT feat:ex_consensus (the hand-built exit).\n")
    head = f"{'model':<18}{'n':>9}{'IC':>9}{'ICIR':>8}{'hit%':>8}{'simret%':>9}"
    print(head); print("-" * len(head))
    for _, r in table.iterrows():
        def f(v, w, s):
            return f"{format(v, s):>{w}}" if v is not None and pd.notna(v) else f"{'—':>{w}}"
        line = f"{r['model']:<18}{int(r['n']):>9}"
        line += f(r['ic'], 9, '+.4f') + f(r['icir'], 8, '+.3f')
        line += f(r['hit'], 8, '.2f') + f(r['simret'], 9, '+.4f')
        print(line)
    print("-" * len(head))
    print("\nGO if the gbm exit model's IC/ICIR clears ex_consensus on the held return — then it is a "
          "better exit-timer and worth wiring panel-first (measure its exit-IC live before it closes trades).")


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description="ML exit model — Phase 0 go/no-go vs exit_method_consensus")
    p.add_argument("--horizon", type=int, default=3, help="label look-ahead in sessions (default 3)")
    p.add_argument("--max-hold", type=int, default=12, help="max simulated hold in sessions (default 12)")
    p.add_argument("--deadband", type=float, default=0.0)
    p.add_argument("--min-train-days", type=int, default=8)
    p.add_argument("--step-days", type=int, default=2)
    p.add_argument("--model", default="gbm", choices=("logistic", "gbm"))
    p.add_argument("--train", action="store_true",
                   help="retrain + persist the live ml_exit artifact instead of measuring")
    p.add_argument("--basis", default="fixed", choices=("fixed", "pv"),
                   help="measure on the fixed held-return label or the oriented pivot label")
    a = p.parse_args(argv)
    from src.db import repo
    if a.train:
        # Before set_read_only — the train appends to the `ml_models` registry.
        art = eod_train_exit()
        print("ml_exit: " + (f"{art['n_train']:,} rows (<= {art['train_max_date']})"
                             if art else "NO ARTIFACT (see log)"))
        return
    repo.set_read_only(True)
    table = measure_exit(horizon=a.horizon, deadband=a.deadband, max_hold=a.max_hold,
                         model_name=a.model, min_train_days=a.min_train_days,
                         step_days=a.step_days, basis=a.basis)
    _print(table, a.horizon)


if __name__ == "__main__":
    main()
