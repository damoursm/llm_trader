"""THE DIP BOOK'S META-MODEL FILTER — PREREG96's RP33, built 2026-10-14 and OFF until the user deploys it
(`enable_dip_long_filter`). User 2026-10-14: "I'm ok with deploying PREREG96's dip meta-model filters RP33 and CP33 if
they gain in total on all three sets" — RP33's add-rule gains: set A +2.39, set B -1.06, set A0 +1.13 log points a
year (total +2.46; CP33 +1.88 / -2.69 / +1.10, +0.29 — the two filter the same orders and RP33 dominates on every set).
The study: `llm_trader_research/studies/prereg96/` (PREREG96.md, data96.py, model96.py; the production parity in
`prod96/`), `memory/arm-reliability-program-2026-10.md`.

THE MODEL (`dip_long_filter_model`: `cache/ml/dip_long/filter/model.txt` + the `model.json` beside it): PREREG96's "R"
— a LightGBM regressor (the native Booster API, L2) of a dip signal's NET return under the live rule (the 1/2-ATR limit
order alive D .. D+4, its first fill; sold at the open after the first close above the 5-session average since the
fill, or after 10 sessions; the simulator's dip costs), on the 17 inputs below, model96's parameters exactly (300
trees, 31 leaves, learning rate 0.03, feature and bagging fraction 0.8 every iteration, 50 rows a leaf, seed 0, 2
threads), fit on every FILLED signal of PREREG96's table (`signals96.pkl`) whose EXIT falls before the model's cut
date — the walk-forward's own construction (a signal still open at the cut is not a training row). `fit` rebuilds it
(`python -m src.signals.dip_filter --fit --table <signals96.pkl> --cut <date>`).

THE POLICY P33: a signal is KEPT when its score is at least the CUT — the 33rd percentile (numpy's nanquantile at 1/3)
of the TRAINING rows' out-of-fold scores, the folds = the signal years in five groups (``years[i::5]``), each fold
scored by a model of model96's parameters fit on the other years' rows (a fold with fewer than 100 such rows is left
unscored); nothing of a scored day sets it. A dropped signal never becomes an order.

THE INPUTS, all known at the close of D-1 (data96's construction: `feature_arrays` / `features_at`, verbatim;
`tests/test_no_lookahead.py` pins them):
* the name's daily bars through D-1 (`dip_long.daily_bars` of its deep-store series — the signal's own): ``rsi2``,
  ``dist_sma200_pct`` (close vs its 200-session average), ``dist_hi20_pct`` / ``dist_hi20c_pct`` (close vs the
  20-session highest high / highest close), ``atr14_pct``, ``ret1_pct`` / ``ret5_pct`` / ``ret20_pct``,
  ``rvol20_pct`` (the 20-session standard deviation of daily log returns), ``log_dv20`` (log10 of the 20-session mean
  of close x volume) and ``limit_dist_pct`` (the order's limit, `dip_long.limit_price` at the model's 1/2 ATR, against
  the close);
* ``breadth`` — the day's signal count (every qualifying name, before the filter);
* ``days_since_prev`` — calendar days since the name's previous signal: the rule's last firing in its stored series
  before D-1 (`rule_mask`, the backtest's construction, 220 bars), else the training table's last signal of the name
  from before the store's history (``prior_signals`` in the JSON: data96's daily-cache rows, through 2021-09-30), else
  missing (as for a name's first signal in training);
* SPY at D-1's close: ``spy_dist_sma200_pct``, ``spy_ret5_pct``, ``spy_ret20_pct``, ``spy_rvol20_pct`` — from the
  daily OHLCV cache (`cache/ohlcv/SPY.json`, Polygon's daily bars: the OFFICIAL close, the training table's own
  source), refreshed after every close by the EOD chain; when it does not hold D-1, the deep store's SPY
  (`dip_long.daily_bars`: the last regular-hours trade, a few basis points off the official close — measured: 17 of
  2,370 walk-forward keep decisions 2021-26 differ), journaled ``spy_source``.

FAIL-OPEN: a model that cannot be loaded (missing, its checksum not the JSON's) or no SPY close for D-1 in either
source keeps EVERY signal — the book as deployed without the filter — and the signal file's ``filter.error`` raises a
scorer-banner problem (`dip_long.health`); a signal whose own inputs raise is kept unscored (``filter_error``).

Journal (`dip_long.compute_signals`, only with the switch on): ``signals/<day>.json`` carries each signal's
``filter_score`` / ``filter_keep`` / ``filter_inputs``, the kept ones in ``signals`` and the dropped ones in
``filtered``, and a ``filter`` block (the model, its cut, the breadth, SPY's source and session, the counts);
`tracker._place_dip_orders` writes each dropped signal into ``orders/<day>.json`` with status ``filtered`` (never
placed, never sent to IBKR) and stamps the kept orders' ``filter_score``.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from loguru import logger

from config import settings

MODEL_NAME = "PREREG96 RP33"
FEATURES = ("rsi2", "dist_sma200_pct", "dist_hi20_pct", "dist_hi20c_pct", "atr14_pct", "ret1_pct", "ret5_pct",
            "ret20_pct", "rvol20_pct", "log_dv20", "limit_dist_pct", "breadth", "days_since_prev",
            "spy_dist_sma200_pct", "spy_ret5_pct", "spy_ret20_pct", "spy_rvol20_pct")
SPY_FEATURES = ("spy_dist_sma200_pct", "spy_ret5_pct", "spy_ret20_pct", "spy_rvol20_pct")
NAME_FEATURES = tuple(f for f in FEATURES if f not in SPY_FEATURES + ("breadth", "days_since_prev"))
SPY = "SPY"
# model96's parameters (BASE + the R objective), its rounds and its P33 policy — never tuned here
PARAMS = {"num_leaves": 31, "learning_rate": 0.03, "feature_fraction": 0.8, "bagging_fraction": 0.8, "bagging_freq": 1,
          "min_data_in_leaf": 50, "verbose": -1, "seed": 0, "num_threads": 2, "objective": "regression"}
ROUNDS = 300
POLICY, POLICY_QUANTILE = "P33", 1 / 3
FOLDS, MIN_FOLD_TRAIN, MIN_TRAIN = 5, 100, 300
# the construction the model was trained on (data96 / strategies.dip.LIVE): fixed with the model, not the live knobs
TREND, MIN_HISTORY, RSI_MAX, MIN_DV, LIMIT_ATR = 200, 220, 10.0, 1e9, 0.5
LABEL = "net_ret_pct"


def enabled() -> bool:
    return bool(getattr(settings, "enable_dip_long_filter", False))


def model_path() -> Path:
    return Path(str(settings.dip_long_filter_model))


def meta_path(p: Optional[Path] = None) -> Path:
    return (p or model_path()).with_suffix(".json")


# ── the inputs (data96's construction, verbatim) ─────────────────────────────

def _limit(close: float, atr: float) -> Optional[float]:
    from src.signals.dip_long import limit_price
    return limit_price(close, atr, LIMIT_ATR)


def feature_arrays(d: pd.DataFrame) -> dict:
    """Every input at every session of the daily frame ``d`` (``o h l c v``), from TRAILING windows only (row i
    reads rows <= i) — data96's `feature_arrays`."""
    from src.signals.dip_long import rsi2
    h, l, c, v = (d[k].to_numpy(float) for k in ("h", "l", "c", "v"))
    s = pd.Series(c)
    pc = np.r_[np.nan, c[:-1]]
    with np.errstate(invalid="ignore", divide="ignore"):
        tr = np.nanmax(np.c_[h - l, np.abs(h - pc), np.abs(l - pc)], axis=1)
        lr = np.log(c / pc)

    def back(k):
        return np.r_[np.full(k, np.nan), c[:-k]]
    with np.errstate(invalid="ignore", divide="ignore"):
        return {"close": c, "sma200": s.rolling(TREND).mean().to_numpy(), "rsi2": rsi2(c),
                "atr14": pd.Series(tr).rolling(14).mean().to_numpy(),
                "dv20": pd.Series(c * v).rolling(20).mean().to_numpy(),
                "hi20": pd.Series(h).rolling(20).max().to_numpy(), "hi20c": s.rolling(20).max().to_numpy(),
                "ret1": c / pc - 1.0, "ret5": c / back(5) - 1.0, "ret20": c / back(20) - 1.0,
                "rvol20": pd.Series(lr).rolling(20).std(ddof=1).to_numpy()}


def features_at(F: dict, i: int) -> dict:
    """The name's inputs at row ``i`` — data96's `features_at` (the limit floored to the cent, as IBKR rests it)."""
    c = float(F["close"][i])
    lim = _limit(c, F["atr14"][i])
    dv = float(F["dv20"][i])
    return {"rsi2": float(F["rsi2"][i]), "dist_sma200_pct": 100.0 * (c / F["sma200"][i] - 1.0),
            "dist_hi20_pct": 100.0 * (c / F["hi20"][i] - 1.0), "dist_hi20c_pct": 100.0 * (c / F["hi20c"][i] - 1.0),
            "atr14_pct": 100.0 * F["atr14"][i] / c, "ret1_pct": 100.0 * F["ret1"][i], "ret5_pct": 100.0 * F["ret5"][i],
            "ret20_pct": 100.0 * F["ret20"][i], "rvol20_pct": 100.0 * F["rvol20"][i],
            "log_dv20": math.log10(dv) if dv > 0 else float("nan"),
            "limit_dist_pct": float("nan") if lim is None else 100.0 * (lim / c - 1.0)}


def name_features(daily: Optional[pd.DataFrame], through: date) -> Optional[dict]:
    """The name's inputs at the close of the last session on or before ``through`` (nothing later is read), with that
    ``session``; None without a bar."""
    if daily is None or daily.empty:
        return None
    d = daily[daily.index <= pd.Timestamp(through)]
    if d.empty:
        return None
    out = features_at(feature_arrays(d), len(d) - 1)
    out["session"] = pd.Timestamp(d.index[-1]).date()
    return out


def rule_mask(daily: pd.DataFrame) -> np.ndarray:
    """The dip rule at every session of ``daily`` (the backtest's `strategies.dip.signals` construction: close above
    its 200-session average, 2-session RSI under 10, 20-session mean close x volume >= $1B, 220 bars through the
    session)."""
    from src.signals.dip_long import rsi2
    c = daily["c"].to_numpy(float)
    sma = pd.Series(c).rolling(TREND).mean().to_numpy()
    dv20 = (daily["c"] * daily["v"]).rolling(20).mean().to_numpy()
    with np.errstate(invalid="ignore"):
        m = (c > sma) & (rsi2(c) < RSI_MAX) & (dv20 >= MIN_DV)
    m[:MIN_HISTORY - 1] = False
    return m


def prev_signal(daily: Optional[pd.DataFrame], before: date) -> Optional[date]:
    """The name's last signal session strictly before ``before`` in its stored series (None: none)."""
    if daily is None or daily.empty:
        return None
    d = daily[daily.index < pd.Timestamp(before)]
    if len(d) < MIN_HISTORY:
        return None
    hit = np.flatnonzero(rule_mask(d))
    return pd.Timestamp(d.index[hit[-1]]).date() if len(hit) else None


def days_since_prev(daily: Optional[pd.DataFrame], session: date, prior: Optional[str]) -> Tuple[float, Optional[str]]:
    """(calendar days since the name's previous signal, that signal's date): the store's rule before ``session``,
    else ``prior`` (the training table's last signal of the name before the store's history), else (NaN, None)."""
    cands = []
    p = prev_signal(daily, session)
    if p is not None:
        cands.append(p)
    if prior:
        try:
            q = date.fromisoformat(str(prior)[:10])
            if q < session:
                cands.append(q)
        except ValueError:
            pass
    if not cands:
        return float("nan"), None
    last = max(cands)
    return float((session - last).days), last.isoformat()


# ── SPY ──────────────────────────────────────────────────────────────────────

def cache_daily(ticker: str) -> Optional[pd.DataFrame]:
    """A name's daily bars from the daily OHLCV cache (`cache/ohlcv/<TICKER>.json`) as ``o h l c v`` on a naive date
    index — data96's `cache_frame` parse."""
    from src.data.cache import _ohlcv_path
    p = _ohlcv_path(ticker)
    if not p.exists():
        return None
    try:
        js = json.loads(p.read_text(encoding="utf-8"))
        df = pd.DataFrame(js["data"], index=pd.to_datetime(js["index"]), columns=js["columns"])
        df = df.rename(columns={"Open": "o", "High": "h", "Low": "l", "Close": "c", "Volume": "v"})[["o", "h", "l", "c", "v"]]
        df = df.astype(float)
    except Exception as e:                                      # noqa: BLE001
        logger.warning(f"[dip_filter] {p} unreadable ({e})")
        return None
    if df.index.tz is not None:
        df.index = df.index.tz_localize(None)
    df.index = pd.DatetimeIndex(df.index).normalize()
    df = df[np.isfinite(df["c"]) & (df["c"] > 0)]
    return df if len(df) else None


def spy_frame_features(d: pd.DataFrame) -> pd.DataFrame:
    """SPY's inputs at every session of its daily frame — data96's `spy_features`."""
    c = d["c"]
    with np.errstate(invalid="ignore", divide="ignore"):
        lr = np.log(c / c.shift(1))
    return pd.DataFrame({"spy_dist_sma200_pct": 100.0 * (c / c.rolling(TREND).mean() - 1.0),
                         "spy_ret5_pct": 100.0 * (c / c.shift(5) - 1.0), "spy_ret20_pct": 100.0 * (c / c.shift(20) - 1.0),
                         "spy_rvol20_pct": 100.0 * lr.rolling(20).std(ddof=1)}, index=d.index)


def spy_features_at(d: Optional[pd.DataFrame], through: date) -> Optional[dict]:
    """SPY's inputs at the close of ``through`` (its frame cut there); None unless the frame holds that session."""
    if d is None or d.empty:
        return None
    d = d[d.index <= pd.Timestamp(through)]
    if d.empty or pd.Timestamp(d.index[-1]).date() != through:
        return None
    row = spy_frame_features(d).iloc[-1]
    return {k: float(row[k]) for k in SPY_FEATURES}


def spy_inputs(through: date) -> Tuple[Optional[dict], str]:
    """(SPY's inputs at the close of ``through``, their source): the daily cache (the training table's source) when it
    holds that close, else the deep store's SPY (extended once when behind); ``(None, "missing")`` when neither does."""
    got = spy_features_at(cache_daily(SPY), through)
    if got is not None:
        return got, "cache"
    from src.signals import dip_long
    d = dip_long._daily(SPY)
    got = spy_features_at(d, through)
    if got is None:
        dip_long._extend([SPY], through)
        got = spy_features_at(dip_long._daily(SPY), through)
    if got is not None:
        logger.warning(f"[dip_filter] the daily cache has no SPY close for {through} — the deep store's SPY scores "
                       f"today's dip signals")
        return got, "deep_store"
    return None, "missing"


# ── the model ────────────────────────────────────────────────────────────────

def _sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


_LOADED: Dict[tuple, tuple] = {}


def load_model(path: Optional[Path] = None):
    """(the Booster, its JSON) — the model file must carry the checksum its JSON records, its features model96's in
    order and its construction this module's; raises otherwise."""
    p = Path(path) if path is not None else model_path()
    jp = meta_path(p)
    st, sj = p.stat(), jp.stat()
    key = (str(p.resolve()), st.st_mtime_ns, st.st_size, sj.st_mtime_ns, sj.st_size)
    if key in _LOADED:
        return _LOADED[key]
    meta = json.loads(jp.read_text(encoding="utf-8"))
    want = (meta.get("checksums") or {}).get("model_sha256")
    got = _sha256(p)
    if not want or got != want:
        raise ValueError(f"{p.name}: sha256 {got[:12]} is not the JSON's {str(want)[:12]}")
    if tuple(meta.get("features") or ()) != FEATURES:
        raise ValueError(f"{jp.name}: features {meta.get('features')} are not model96's")
    con = meta.get("construction") or {}
    if (con.get("trend"), con.get("min_history"), con.get("rsi_max"), con.get("min_dv"), con.get("limit_atr")) != \
            (TREND, MIN_HISTORY, RSI_MAX, MIN_DV, LIMIT_ATR):
        raise ValueError(f"{jp.name}: construction {con} is not this module's")
    import lightgbm as lgb
    booster = lgb.Booster(model_file=str(p))
    if list(booster.feature_name()) != list(FEATURES):
        raise ValueError(f"{p.name}: the booster's features {booster.feature_name()} are not model96's")
    _LOADED.clear()
    _LOADED[key] = (booster, meta)
    return booster, meta


def matrix(rows: Sequence[dict]) -> np.ndarray:
    """The model's input matrix (FEATURES order; a missing input is NaN, as LightGBM was trained to read it)."""
    def num(v):
        try:
            x = float(v)
        except (TypeError, ValueError):
            return float("nan")
        return x if math.isfinite(x) else float("nan")
    return np.array([[num(r.get(k)) for k in FEATURES] for r in rows], dtype=float).reshape(len(rows), len(FEATURES))


def predict(rows: Sequence[dict], path: Optional[Path] = None) -> np.ndarray:
    booster, _ = load_model(path)
    return booster.predict(matrix(rows)) if len(rows) else np.zeros(0)


def keep(score: float, cut: float) -> bool:
    """P33: kept when the score is at least the cut (model96: ``score >= cut``)."""
    return bool(float(score) >= float(cut))


def _clean(v):
    return None if v is None or (isinstance(v, float) and not math.isfinite(v)) else v


# ── the day's signals ────────────────────────────────────────────────────────

def signal_inputs(ticker: str, session: date, breadth: int, spy: dict, prior: Dict[str, str],
                  daily: Optional[pd.DataFrame] = None) -> dict:
    """The 17 inputs of ``ticker``'s signal at the close of ``session`` (D-1), its stored series read through that
    close only. Raises when the store has no bar for ``session``."""
    from src.signals import dip_long
    d = dip_long._daily(ticker) if daily is None else daily
    f = name_features(d, session)
    if f is None or f.pop("session") != session:
        raise ValueError(f"{ticker}: no stored close for {session}")
    dsp, prev = days_since_prev(d, session, prior.get(ticker))
    out = dict(f, breadth=int(breadth), days_since_prev=dsp, **spy)
    out["_prev_signal"] = prev
    return out


def apply(signals: List[dict], d: date, prev: date, path: Optional[Path] = None) -> Tuple[List[dict], dict]:
    """Score the signals of session ``d`` (decided at the close of ``prev``) and drop those under the cut: returns
    (the kept signals, deepest dip first as given; ``{"info": ..., "filtered": [the dropped signals]}``). Every signal
    is stamped ``filter_score`` / ``filter_keep`` / ``filter_inputs``. Fail-open: no model, or no SPY close for
    ``prev``, keeps every signal (``info["error"]``)."""
    info: dict = {"model": MODEL_NAME, "policy": POLICY, "breadth": len(signals), "scored": 0, "kept": len(signals),
                  "filtered": 0, "error": None}
    if not signals:
        return signals, {"info": info, "filtered": []}
    try:
        booster, meta = load_model(path)
    except Exception as e:                                      # noqa: BLE001
        info["error"] = f"model unavailable ({type(e).__name__}: {e}) — every signal kept"
        logger.error(f"[dip_filter] {d}: {info['error']}")
        return signals, {"info": info, "filtered": []}
    cut = float(meta["policy"]["cut"])
    info.update(cut=cut, fit_cut=(meta.get("fit") or {}).get("cut_date"),
                model_sha256=str((meta.get("checksums") or {}).get("model_sha256"))[:16])
    try:
        spy, src = spy_inputs(prev)
    except Exception as e:                                      # noqa: BLE001
        spy, src = None, f"error: {type(e).__name__}: {e}"
    info["spy_source"] = src
    if spy is None:
        info["error"] = f"no SPY close for {prev} ({src}) — every signal kept"
        logger.error(f"[dip_filter] {d}: {info['error']}")
        return signals, {"info": info, "filtered": []}
    info["spy_session"] = prev.isoformat()
    prior = meta.get("prior_signals") or {}
    rows, idx = [], []
    for k, s in enumerate(signals):
        try:
            x = signal_inputs(str(s["ticker"]), prev, len(signals), spy, prior)
        except Exception as e:                                  # noqa: BLE001
            s["filter_error"] = f"{type(e).__name__}: {e}"
            s["filter_keep"] = True
            logger.warning(f"[dip_filter] {s.get('ticker')}: not scored ({s['filter_error']}) — kept")
            continue
        s["filter_prev_signal"] = x.pop("_prev_signal")
        s["filter_inputs"] = {k2: _clean(float(x[k2])) for k2 in FEATURES}
        rows.append(x)
        idx.append(k)
    scores = booster.predict(matrix(rows)) if rows else np.zeros(0)
    for k, sc in zip(idx, scores):
        signals[k]["filter_score"] = float(sc)
        signals[k]["filter_keep"] = keep(sc, cut)
    kept = [s for s in signals if s.get("filter_keep", True)]
    dropped = [s for s in signals if not s.get("filter_keep", True)]
    info.update(scored=len(rows), kept=len(kept), filtered=len(dropped))
    logger.info(f"[dip_filter] {d}: {len(rows)} of {len(signals)} signal(s) scored (cut {cut:+.4f}, SPY from the "
                f"{src}): {len(kept)} kept, {len(dropped)} filtered"
                + (": " + ", ".join(f"{s['ticker']} {s['filter_score']:+.3f}" for s in dropped) if dropped else ""))
    return kept, {"info": info, "filtered": dropped}


# ── the fit (the production artifact; model96's construction) ────────────────

def training_rows(table: pd.DataFrame, cut: date) -> pd.DataFrame:
    """model96's training rows for a model used from ``cut``: the FILLED signals whose exit falls before it."""
    df = table[table["filled"].astype(bool)].copy()
    df["year"] = pd.to_datetime(df["sig"]).dt.year
    df["exit_dt"] = pd.to_datetime(df["exit_day"])
    return df[df["exit_dt"] < pd.Timestamp(cut)]


def _train(X: pd.DataFrame, y: pd.Series):
    import lightgbm as lgb
    return lgb.train(dict(PARAMS), lgb.Dataset(X, label=y), num_boost_round=ROUNDS)


def oof_scores(tr: pd.DataFrame) -> pd.Series:
    """model96's out-of-fold scores of the training rows: folds = the signal years in five groups (``years[i::5]``)."""
    years = sorted(tr["year"].unique())
    folds = [set(years[i::FOLDS]) for i in range(FOLDS)]
    out = pd.Series(np.nan, index=tr.index)
    y = tr[LABEL]
    for f in folds:
        te = tr["year"].isin(f)
        if te.sum() == 0 or (~te).sum() < MIN_FOLD_TRAIN:
            continue
        m = _train(tr.loc[~te, list(FEATURES)], y[~te])
        out[te] = m.predict(tr.loc[te, list(FEATURES)])
    return out


def policy_cut(oof: pd.Series) -> float:
    """P33's cut: the 33rd percentile of the out-of-fold scores (model96: ``np.nanquantile(oof, 1 / 3)``)."""
    return float(np.nanquantile(oof, POLICY_QUANTILE))


def fit(table: pd.DataFrame, cut: date):
    """(the Booster, its cut, the training rows, their out-of-fold scores) — model96's R model and P33 cut for a model
    used from ``cut``."""
    tr = training_rows(table, cut)
    if len(tr) < MIN_TRAIN:
        raise ValueError(f"{len(tr)} training rows before {cut} (model96 needs {MIN_TRAIN})")
    booster = _train(tr[list(FEATURES)], tr[LABEL])
    oof = oof_scores(tr)
    return booster, policy_cut(oof), tr, oof


def prior_signals(table: pd.DataFrame) -> Dict[str, str]:
    """{name: its last signal in the training table before the deep store's history} — data96's daily-cache rows
    (survivors only, through 2021-09-30): the ``days_since_prev`` of a name whose previous signal predates the store."""
    old = table[table["source"] == "daily_cache"]
    if old.empty:
        return {}
    last = old.groupby("ticker")["sig"].max()
    return {str(t): pd.Timestamp(s).date().isoformat() for t, s in sorted(last.items())}


def write_model(table_path: Path, cut: date, out: Path) -> dict:
    """Fit on the table at ``table_path`` for use from ``cut`` and write ``out`` (the Booster's text) + its JSON."""
    import lightgbm as lgb
    table = pd.read_pickle(table_path)
    booster, cutv, tr, oof = fit(table, cut)
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_name(out.name + f".{os.getpid()}.tmp")
    booster.save_model(str(tmp))
    reloaded = lgb.Booster(model_file=str(tmp))
    X = tr[list(FEATURES)]
    a, b = booster.predict(X), reloaded.predict(X.to_numpy(float))
    diff = float(np.max(np.abs(a - b))) if len(a) else 0.0
    if diff != 0.0:
        raise ValueError(f"the saved model does not reproduce the fit's scores (max |diff| {diff})")
    os.replace(tmp, out)
    rows_id = hashlib.sha256("\n".join(f"{t}|{pd.Timestamp(s).date()}" for t, s in
                                       sorted(zip(tr["tk"], tr["sig"]))).encode()).hexdigest()
    ins = a
    meta = {
        "model": MODEL_NAME, "study": "PREREG96 (llm_trader_research/studies/prereg96: model96.py's R model, P33)",
        "kind": "R", "objective": "regression (L2)", "label": f"{LABEL}: the signal's net return under the live rule "
        "(the 1/2-ATR limit order's first fill, the live exit, the simulator's dip costs at $1,000), %",
        "features": list(FEATURES), "params": dict(PARAMS), "rounds": ROUNDS,
        "construction": {"trend": TREND, "min_history": MIN_HISTORY, "rsi_max": RSI_MAX, "min_dv": MIN_DV,
                         "limit_atr": LIMIT_ATR,
                         "spy": "the daily OHLCV cache's SPY (Polygon daily bars, the official close); the deep "
                                "store's SPY when the cache lacks D-1"},
        "policy": {"name": POLICY, "quantile": POLICY_QUANTILE, "cut": cutv, "keep": "score >= cut",
                   "how": "np.nanquantile of the training rows' out-of-fold scores at 1/3; folds = the signal years "
                          "in five groups (years[i::5]), each scored by a model of the same parameters fit on the "
                          "other years' rows (skipped when fewer than 100) — model96.oof_scores / POLICIES['P33']",
                   "oof_scored": int(oof.notna().sum()), "oof_quantiles": {q: float(np.nanquantile(oof, q))
                                                                          for q in (0.1, 0.2, 1 / 3, 0.5, 0.8, 0.9)},
                   "in_sample_share_kept": float(np.mean(ins >= cutv))},
        "fit": {"cut_date": cut.isoformat(), "rows": "the table's FILLED signals whose exit_day < cut_date (model96's "
                "walk-forward construction; a signal still open at the cut is not a training row)",
                "n_train": int(len(tr)), "first_signal": str(pd.Timestamp(tr["sig"].min()).date()),
                "last_signal": str(pd.Timestamp(tr["sig"].max()).date()),
                "last_exit": str(pd.Timestamp(tr["exit_dt"].max()).date()),
                "by_source": {str(k): int(v) for k, v in tr["source"].value_counts().items()},
                "by_year": {str(k): int(v) for k, v in tr["year"].value_counts().sort_index().items()},
                "label_mean_pct": float(tr[LABEL].mean())},
        "table": {"path": str(table_path), "rows": int(len(table)), "filled": int(table["filled"].astype(bool).sum()),
                  "last_signal": str(pd.Timestamp(table["sig"].max()).date())},
        "prior_signals": prior_signals(table),
        "checksums": {"model_sha256": _sha256(out), "table_sha256": _sha256(Path(table_path)),
                      "train_rows_sha256": rows_id},
        "reload_check": {"rows": int(len(tr)), "max_abs_diff": diff},
        "lightgbm": lgb.__version__,
        "built_at": datetime.now(timezone.utc).isoformat(timespec="seconds")}
    jp = meta_path(out)
    tmpj = jp.with_name(jp.name + f".{os.getpid()}.tmp")
    tmpj.write_text(json.dumps(meta, indent=1), encoding="utf-8")
    os.replace(tmpj, jp)
    return meta


# ── CLI ──────────────────────────────────────────────────────────────────────

def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--fit", action="store_true", help="fit the production model on --table for use from --cut")
    ap.add_argument("--table", default=None, help="--fit: PREREG96's signal table (signals96.pkl)")
    ap.add_argument("--cut", default=None, help="--fit: the model's cut date (training rows exit before it)")
    ap.add_argument("--out", default=None, help="--fit: the model file (default: dip_long_filter_model)")
    ap.add_argument("--score", action="store_true", help="score a day's dip signals (prints; writes nothing)")
    ap.add_argument("--day", default=None, help="--score: the session the signals trade at (default: today)")
    ap.add_argument("--status", action="store_true", help="the model's identity and cut")
    a = ap.parse_args(argv)
    if a.fit:
        if not (a.table and a.cut):
            ap.error("--fit needs --table and --cut")
        meta = write_model(Path(a.table), date.fromisoformat(a.cut), Path(a.out) if a.out else model_path())
        print(json.dumps({k: meta[k] for k in ("model", "fit", "policy", "checksums", "reload_check")}, indent=1))
        return 0
    if a.status:
        _, meta = load_model()
        print(json.dumps({k: meta.get(k) for k in ("model", "fit", "policy", "checksums", "lightgbm", "built_at")},
                         indent=1, default=str))
        print(f"enable_dip_long_filter = {enabled()}")
        return 0
    if a.score:
        from src.signals import dip_long
        d = date.fromisoformat(a.day) if a.day else dip_long.today_et()
        orig = dip_long._write_json
        dip_long._write_json = lambda p, obj: None                  # a dry run writes nothing (nor extends the store)
        try:
            day = dip_long.compute_signals(d, extend=False)
        finally:
            dip_long._write_json = orig
        if "filter" in day:                                         # the switch is on: already scored
            sig, info = list(day.get("signals") or []) + list(day.get("filtered") or []), day["filter"]
        else:                                                       # off: score them here (apply stamps each dict)
            sig = [dict(s) for s in day.get("signals") or []]
            info = apply(sig, d, dip_long.prev_session(d))[1]["info"]
        print(json.dumps(info, indent=1, default=str))
        for s in sorted(sig, key=lambda r: (r.get("filter_score") is None, -(r.get("filter_score") or 0.0))):
            sc = s.get("filter_score")
            print(f"  {s['ticker']:<8} RSI2 {float(s['rsi2']):5.2f}  score {'%+.4f' % sc if sc is not None else '   n/a '}  "
                  f"{'KEEP' if s.get('filter_keep', True) else 'DROP'}")
        return 0
    ap.print_help()
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
