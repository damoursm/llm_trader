"""SEL_SHORT — the production entry strategy from 2026-09-28 (user directive
2026-09-26: "Deploy the model with the half give-back with max hold of 15 days
with all borrowable. Use it to make all trades and put all other models as
shadow.").

WHAT IT TRADES. The intraday tail-regression LONG selection model
(`src/analysis/sel_models.py`: LightGBM `tailreg`, 400 rounds, fit on sessions
<= 2026-04-30, the 85 base + per-ticker deep features) ranks every liquid name
(price >= $5, 20-session mean regular-hours dollar volume >= $5M) on every
regular-hours 30-minute bar. Per bar the TOP-1 is the candidate; the live
freshness rule keeps it only when its score is a new high against the name's
own scores over the previous 30 trading days (or the name has fewer than 10 of
them), and only as the name's first such pick of the day
(`eval_metrics._selection`, the rule the model was evaluated with). The pick is
SHORTED — only when the stock ROSE over the 5 sessions before the pick (its
close vs the close at the same bar of day 5 sessions earlier), its short
interest is under one day of volume (THE SHORT-INTEREST FILTER, below) and IBKR
can lend it (any fee: "all borrowable"). Exit: cover when the stock has given back HALF
of that 5-session run-up (target = close - 0.5 x (close - close 5 sessions before); the VOL
arm the WHOLE run-up, `give_back`, from 2026-10-04 evening), judged at every tick in
every session (user directive 2026-09-27); otherwise cover at the first tick at/after
the same bar 15 sessions later. No stop.

THE EVIDENCE (2026-09-25/26; scratchpad `short_eval.py`, `short_exit_search.py`,
`short_exit_holds.py`; `memory/selection-model-exits-2026-09.md`): 388
out-of-sample picks, Jan-Sep 2026; on the risers that traded >= $5 and were
borrowable, give back half / max 15 sessions measured +11.3%/trade over 7.7
days, +1.40%/day, net of the real quoted spread at entry and exit, IBKR
commissions and today's borrow fee (156 trades with 50+ sessions after entry).
Checking the target at 30-minute closes (as here) measured +11.0%/trade against
+8.9% for a resting limit at the level. On the record: the edge is a
May-September result (January to mid-May ~0 for every exit); the borrow fees
were today's, not the ones at each spike; ~200 trades. The LIVE-FAITHFUL re-test
(2026-09-27: every bar, the $5 floor on the traded price, one position per
ticker, 24-hour days) puts the model arm at 0.11 %/day on 139 trades (1.41 with
the short-interest filter, all of it May-September) and the vol arm at 1.71
(2.41 with the filter) — the numbers the docs carry now.

HOW IT RUNS. `pipeline.run_pipeline` calls `launch` at the start of every tick:
on a market day it starts THIS module as a subprocess (`--prepare --day D` once a
day from `sel_short_prepare_after_et`; `--run --day D --bars k1,k2,...` for every
completed regular-hours bar of the day not yet run, oldest first), in parallel
with the tick, and
`wait`s for it before the trade step. A run fetches the day's completed bars for
the universe from Polygon (the tick cache holds only the names a tick touches;
from the deep store's last session when it is behind), rebuilds each series
(deep 30-minute store + those bars), computes the exact training features
(`ml_model.features_30m_from_hlc` + `deep_features.serving_vector` from the
pre-open session snapshot — `ensure_snapshot` first rebuilds a missing or
DEFECTIVE one), scores, appends every score to the day's score file, selects and
journals the decision. `tracker.record_sel_short_trades` opens the journal's
shorts (skipping one already at its target); `tracker.monitor_sel_short_positions`
covers them. The subprocess holds `busy.lock`, touched every 30 s, so a tick
never launches a second scorer beside a live one and a dead one's lock goes
stale in 180 s. Runbook: `docs/SEL_SHORT.md`.

THE VOLATILITY ARM (user directive 2026-09-26: "Deploy 'Short the most volatile
name' ... alongside the current short model ... we'll evaluate which one is the
best"; `enable_sel_short_vol`). The same run also ranks every name by its
30-minute ATR% (`sel_short_vol_feature`, a base feature the scorer computes
anyway) and applies the SAME rule to that score: top-1, fresh against the name's
own ATR% history over `sel_short_vol_own_window_days` (20; user directive
2026-09-27) sessions (`scores_vol/`), its first fresh pick of the day, shorted only
when it rose, the same deadline and borrow check. Its target gives back the WHOLE run-up
(`sel_short_vol_give_back` 1.0; user directive 2026-10-04 evening, "Vol arm only" once the
per-arm numbers were in; the model and ETF arms keep half). Its decision is journaled beside
the model's with ``arm="vol"``. Measured with live-faithful
mechanics (scratchpad `rank_params.py`: every bar, the $5 floor on the traded
price, one position per ticker, NBBO costs; return per day = the average trade's
return over its average 24-hour holding period): window 20 / top-1 72 trades Jan-Sep
2026, +17.9%/trade over 9.7 days, 1.71 %/day (95% 0.88-2.62), against the model's
139 at +1.1%, 0.11 %/day — the model's median trade is +9.4%, its average dragged
down by a few squeezes.
ONE TRADE PER ARM (user directive 2026-10-04, with the vol arm's quarter give-back:
"Two trades, one per arm"): a name two arms pick on the same bar opens a trade for
EACH arm, each with its own target, netted into one position at IBKR (until then the
arms shared one trade stamped ``sel_arm="model+vol"``).

THE SHORT-INTEREST FILTER (user directive 2026-09-27: "Add in live production
the 'Under one day of volume' filter"; `enable_sel_short_dtc_filter`, both
arms). A pick whose FINRA days to cover (`DTC_FEATURE`, from the session
snapshot the model reads — known at the 08:30 ET cutoff) exceeds
`sel_short_max_days_to_cover` (1.0, FINRA's floor: short interest under one day
of volume) is journaled ``decision="crowded"`` with its target and deadline, and
never handed to the ledger; it still counts as the name's pick of the day. An
unknown value passes (warned). Measured on the same live-faithful trades
(scratchpad `gross_net_dtc.py`, applied at the trade step): vol 72 -> 53 trades,
1.71 -> 2.41 %/day net; model 139 -> 56, 0.11 -> 1.41 — chosen after seeing those
trades, the model's gain entirely May-September, the vol gain's 95% range
touching zero; losses beyond -50% came at the same rate in both groups.

THE RELATIVE-VOLUME FILTER (user directive 2026-10-01: "implement relative volume
filter to prod"; `enable_sel_short_vol_rvol_filter`, VOL ARM ONLY). A vol pick whose
bar's volume is under `sel_short_vol_min_rvol` (1.58) x the name's mean 30-minute bar
volume over its previous 260 regular-hours bars (`rvol_at`; 20 sessions, >= 20 bars
needed) is journaled ``decision="low_rvol"`` with its target and deadline, never handed
to the ledger; it still counts as the name's pick of the day; an unknown value passes.
The cut is the 20th percentile of the live rule's trades 2021-26 (no outcome used).
Measured Feb 2021 - Sep 2026 (scratchpad `rvol_table.py`): with the names delisted since
2021, 596 -> 477 trades, 2.37 -> 2.63 %/day (+0.26, 95% +0.02..+0.55); today's names
alone 342 -> 264, 2.09 -> 2.34 (+0.26, -0.06..+0.66) — the feature was found by
searching (post hoc); judge it on the live picks. Every pick of both arms journals
``rvol``; the model arm is not filtered (never tested).

Files under `settings.sel_short_dir`: model.txt + model.json (`--install`),
universe/<day>.json, scores/<day>.pkl (+ scores_vol/<day>.pkl), picks/<day>.jsonl
(+ .consumed.json), runs/<day>_<bar>.json (+ .failed.json), launches/<day>.jsonl,
snapshot/<day>.json, prepare/<day>.json, busy.lock. `health()` reads them for the
email digest's scorer banner.
"""
from __future__ import annotations

import argparse
import bisect
import json
import os
import shutil
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
from loguru import logger

from config.settings import settings

ET = ZoneInfo("America/New_York")
EPOCH = pd.Timestamp("1970-01-01")
BAR = pd.Timedelta(minutes=30)
BARS_PER_SESSION = 13
SOURCE_BOOSTER = Path("cache/ml/sel/final/tailreg_long_400.txt")
SOURCE_ARRAYS = Path("cache/ml/sel30_eval")
SOURCE_PRED = Path("cache/ml/sel/final/pred_tailreg_long_400.npy")
SOURCE_ROWS = Path("cache/ml/sel/final/eval_rows.npy")
NUM_ITERATION = 400
MECHANISM = "sel_short"
# the model's picks; the most-volatile-name rule's; the same rule on ETFs; the same rule on the THIN stocks (the
# common stocks trading $1-5M a day, ranked among themselves, on free capital — 2026-10-07); VOL2, the vol rule
# without the crowding filter, the relative-volume filter and the volatility exit (2026-10-07, PREREG37)
ARMS = ("model", "vol", "etf", "thin", "vol2")
_SCORE_DIRS = {"model": "scores", "vol": "scores_vol", "etf": "scores_etf", "thin": "scores_thin",
               "vol2": "scores_vol2"}
VOL_RULE_ARMS = ("vol", "etf", "thin", "vol2")  # the arms that rank the `vol` column (30-min ATR%)
# VOL2 (user directive 2026-10-07: "Deploy it as another version that will also make real trades in the paper
# account"): the vol arm's ranking, universe, freshness (its own copy of the vol history, `scores_vol2/`), first
# fresh pick of the day, riser, same-bar fallback, whole give-back target, 15-session limit and 6x cover — WITHOUT
# the crowding filter, the relative-volume filter and the volatility exit. PREREG37 (2021-02..2024-06, one $10,000
# account, every restriction): +20.7 vs +11.7 %/yr (+7.7 pp, 95% -5.7..+20.4, both halves and every start
# positive), return per day 0.36 vs 1.44 %, 6 vs 1 margin calls over the 2021-26 context; not significant.
UNFILTERED_ARMS = ("vol2",)    # no crowding filter, no relative-volume filter
NO_VOLNORM_ARMS = ("vol2",)    # no volatility exit


def volnorm_exit(arm) -> bool:
    """The arm's trades take the volatility-normalised exit (`sel_volnorm`) — every arm but vol2's."""
    return str(arm or "model") not in NO_VOLNORM_ARMS
DTC_FEATURE = "dp_si_dtc"              # FINRA days to cover in the session snapshot (the short-interest filter)
LOCK_HEARTBEAT_SECONDS = 30.0
LOCK_STALE_SECONDS = 180.0
_ACTIVE: dict = {}                    # the subprocess this process launched last


def root() -> Path:
    return Path(settings.sel_short_dir)


# ── calendar ─────────────────────────────────────────────────────────────────

def is_session(d: date) -> bool:
    from src.performance.market_calendar import is_market_day
    return is_market_day(d)


def sessions_before(d: date, n: int) -> List[date]:
    """The ``n`` sessions strictly before ``d``, oldest first."""
    out: List[date] = []
    p = d
    while len(out) < n:
        p -= timedelta(days=1)
        if is_session(p):
            out.append(p)
    return out[::-1]


def session_after(d: date, n: int) -> date:
    """The ``n``-th session after ``d``."""
    p, k = d, 0
    while k < n:
        p += timedelta(days=1)
        if is_session(p):
            k += 1
    return p


def dnum(d: date) -> int:
    return int((pd.Timestamp(d) - EPOCH).days)


def bar_end_et(day: date, bar_of_day: int) -> datetime:
    """The END of regular-hours bar ``bar_of_day`` (0 = 09:30-10:00) on ``day``."""
    return datetime(day.year, day.month, day.day, 9, 30, tzinfo=ET) + timedelta(minutes=30 * (bar_of_day + 1))


def latest_bar(now: datetime) -> Optional[Tuple[date, int]]:
    """``(day, bar_of_day)`` of the latest regular-hours 30-minute bar COMPLETED
    at ``now`` on a market day, or None before today's first bar ends."""
    et = now.astimezone(ET)
    day = et.date()
    if not is_session(day):
        return None
    mins = et.hour * 60 + et.minute - 570
    k = mins // 30 - 1                        # 10:00 -> 0, 10:29 -> 0, 10:30 -> 1
    if k < 0:
        return None
    return day, int(min(k, BARS_PER_SESSION - 1))


def _iso(d: date) -> str:
    return d.isoformat()


# ── the model ────────────────────────────────────────────────────────────────

def install(booster: Path = SOURCE_BOOSTER, arrays: Path = SOURCE_ARRAYS) -> dict:
    """Copy the research artifact into the production directory with its feature
    order, and seed the score history from the evaluation arrays' predictions
    (every tradeable bar from 2026-05-01: the own-history rule's warm-up)."""
    import lightgbm as lgb
    from src.analysis.sel_models import Arrays
    arr = Arrays(arrays)
    names, _, _ = arr.features("deep")
    out = root()
    out.mkdir(parents=True, exist_ok=True)
    shutil.copy2(booster, out / "model.txt")
    b = lgb.Booster(model_file=str(out / "model.txt"))
    if b.num_feature() != len(names):
        raise SystemExit(f"booster has {b.num_feature()} features, the arrays' deep set {len(names)}")
    meta = {"features": names, "num_iteration": NUM_ITERATION, "objective": "tailreg", "side": "long",
            "fset": "deep", "cut": "2026-04-30", "source": str(booster), "arrays": str(arrays),
            "arrays_built_at": arr.meta.get("built_at"), "thr": arr.meta.get("thr"),
            "tickers": sorted(set(arr.meta["tickers"])),
            "installed_at": datetime.now(timezone.utc).isoformat(timespec="seconds")}
    (out / "model.json").write_text(json.dumps(meta), encoding="utf-8")
    seeded = seed_scores(arr)
    seeded_vol = seed_vol_scores(arr)
    logger.info(f"[sel_short] installed {booster} ({len(names)} features, {len(meta['tickers'])} tickers); "
                f"seeded {seeded} score days (+ {seeded_vol} volatility-arm days)")
    return meta


def install_v2(booster: Path, cut: str = "2026-04-30") -> dict:
    """Install a V2 model (`sel_v2`: the v1 inputs + `sel_v2.EXTRA`, `sel_v2.MASKED` blanked) in place of
    the installed one. The installed model is kept as model_v1.txt / model_v1.json and its score history
    moved to scores_v1/ (both kept for a rollback: copy them back). The new history is NOT seeded here —
    `backfill_days(..., arms=("model",))` writes it with the new model, through the same code as live."""
    import lightgbm as lgb
    from src.signals import sel_v2
    out = root()
    b = lgb.Booster(model_file=str(booster))
    feats = list(b.feature_name())
    old = json.loads((out / "model.json").read_text(encoding="utf-8"))
    n_old = len(old["features"])
    if feats[:n_old] != list(old["features"]) or feats[n_old:] != list(sel_v2.EXTRA):
        raise SystemExit("the booster's inputs are not the installed model's followed by sel_v2.EXTRA")
    if sel_v2.is_v2(old):
        raise SystemExit("a V2 model is already installed")
    for src, dst in (("model.txt", "model_v1.txt"), ("model.json", "model_v1.json")):
        if not (out / dst).exists():
            shutil.copy2(out / src, out / dst)
    hist, keep = out / _SCORE_DIRS["model"], out / "scores_v1"
    if hist.exists():
        if keep.exists():
            raise SystemExit(f"{keep} exists — refusing to overwrite an archived history")
        os.replace(hist, keep)
    shutil.copy2(booster, out / "model.txt")
    meta = {"features": feats, "num_iteration": NUM_ITERATION, "objective": "tailreg", "side": "long",
            "fset": "deepx", "variant": "v2", "cut": cut, "source": str(booster),
            "extra_features": list(sel_v2.EXTRA), "masked": list(sel_v2.MASKED), "thr": old.get("thr"),
            "tickers": list(old["tickers"]), "replaced": {"source": old.get("source"), "cut": old.get("cut")},
            "installed_at": datetime.now(timezone.utc).isoformat(timespec="seconds")}
    (out / "model.json").write_text(json.dumps(meta), encoding="utf-8")
    _MODEL.clear()
    logger.info(f"[sel_short] installed V2 {booster} ({len(feats)} inputs, {len(meta['masked'])} blanked); "
                f"v1 kept as model_v1.*, its history as scores_v1/")
    return meta


_MODEL: dict = {}


def load_model():
    """``(booster, meta)``, memoised per process."""
    if "b" not in _MODEL:
        import lightgbm as lgb
        meta = json.loads((root() / "model.json").read_text(encoding="utf-8"))
        _MODEL.update(b=lgb.Booster(model_file=str(root() / "model.txt")), meta=meta)
    return _MODEL["b"], _MODEL["meta"]


# ── score history (the own-history rule's memory) ────────────────────────────

def scores_path(d: date, arm: str = "model") -> Path:
    return root() / _SCORE_DIRS[arm] / f"{_iso(d)}.pkl"


def _write_pickle(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    df.to_pickle(tmp)
    os.replace(tmp, path)


def append_scores(d: date, rows: pd.DataFrame, arm: str = "model") -> None:
    """Add one run's scores (columns bar, ticker, score) to the day's file of
    ``arm``, replacing an earlier write of the same bar (a re-run)."""
    p = scores_path(d, arm)
    old = pd.read_pickle(p) if p.exists() else None
    if old is not None and len(old):
        old = old[~old["bar"].isin(rows["bar"].unique())]
        rows = pd.concat([old, rows], ignore_index=True)
    _write_pickle(rows[["bar", "ticker", "score"]].reset_index(drop=True), p)


def seed_scores(arr=None) -> int:
    """Day files from the evaluation arrays' saved predictions (the same model,
    every tradeable bar from 2026-05-01). Existing day files are kept."""
    from src.analysis.sel_models import Arrays
    arr = arr or Arrays(SOURCE_ARRAYS)
    rows = np.load(SOURCE_ROWS)
    pred = np.load(SOURCE_PRED).astype(float)
    df = pd.DataFrame({"dn": arr.dn[rows].astype(np.int64), "bar": arr.bar[rows].astype(int),
                       "ticker": [arr.tickers[i] for i in arr.tk[rows]], "score": pred})
    n = 0
    for dn, g in df.groupby("dn"):
        d = (EPOCH + pd.Timedelta(days=int(dn))).date()
        p = scores_path(d)
        if p.exists():
            continue
        _write_pickle(g[["bar", "ticker", "score"]].reset_index(drop=True), p)
        n += 1
    return n


def seed_vol_scores(arr=None, rows=None) -> int:
    """The volatility arm's day files from the evaluation arrays' own feature
    column (`sel_short_vol_feature` on every tradeable bar from 2026-05-01 —
    the rows the model's history was seeded from). Existing day files are kept;
    the arrays' last sessions are thin, so `backfill_days(..., arms=("vol",))`
    overwrites those."""
    from src.analysis.sel_models import Arrays
    arr = arr or Arrays(SOURCE_ARRAYS)
    rows = np.load(SOURCE_ROWS) if rows is None else np.asarray(rows)
    j = list(arr.meta["base_features"]).index(str(settings.sel_short_vol_feature))
    vol = np.concatenate([np.asarray(arr.X[rows[a:a + 500_000], j], float)
                          for a in range(0, len(rows), 500_000)]) if len(rows) else np.zeros(0)
    df = pd.DataFrame({"dn": arr.dn[rows].astype(np.int64), "bar": arr.bar[rows].astype(int),
                       "ticker": [arr.tickers[i] for i in arr.tk[rows]], "score": vol})
    df = df[np.isfinite(df["score"])]
    n = 0
    for dn, g in df.groupby("dn"):
        d = (EPOCH + pd.Timedelta(days=int(dn))).date()
        p = scores_path(d, "vol")
        if p.exists():
            continue
        _write_pickle(g[["bar", "ticker", "score"]].reset_index(drop=True), p)
        n += 1
    return n


def own_window(arm: str = "model") -> int:
    """The freshness window of ``arm`` in sessions: the model's
    `sel_short_own_window_days` (30), the vol arm's `sel_short_vol_own_window_days`
    (20), the ETF arm's `sel_short_etf_own_window_days` (20)."""
    if arm == "model":
        return int(settings.sel_short_own_window_days)
    if arm == "etf":
        return int(settings.sel_short_etf_own_window_days)
    return int(settings.sel_short_vol_own_window_days)


ETP_TYPES = ("ETF", "ETN", "ETV", "ETS")
_TYPES: dict = {}


def etf_names(tickers: Sequence[str], model_names: Optional[Sequence[str]] = None) -> List[str]:
    """The exchange-traded products among ``tickers`` — the ETF arm's universe:
    Polygon's security type (the bulk sweep `cache/security_types.json`) in
    ETP_TYPES, plus any name outside the model's training set (the deep store
    adds those for this arm only) that is not one of the vol arm's added stocks
    (common stocks, `vol_listing_names`)."""
    if "t" not in _TYPES:
        try:
            _TYPES["t"] = json.loads(Path("cache/security_types.json").read_text(encoding="utf-8")).get("types") or {}
        except Exception as e:                                 # noqa: BLE001
            logger.warning(f"[sel_short] security types unreadable ({e}) — the ETF arm sees only the added names")
            _TYPES["t"] = {}
    types = _TYPES["t"]
    extra = (set(tickers) - set(model_names) - set(vol_listing_names())) if model_names else set()

    def kind(t):
        return types.get(t, types.get(t.replace("-", ".")))
    # an added name counts unless Polygon types it as something else than a product (a common
    # stock that reached the deep store outside `add_listings` must never become an ETF pick)
    return sorted(t for t in tickers
                  if kind(t) in ETP_TYPES or (t in extra and kind(t) is None))


_UNDERLYING: dict = {}


def etf_underlying(ticker: str) -> Optional[dict]:
    """A leveraged / inverse single-stock fund's underlying (``{"under", "side"}``)
    from `cache/ml/sel_short/etf_underlying.json` (parsed from the funds' names),
    else None — journal-only: the ETF arm's picks carry the underlying's days to
    cover for a blind read (a 2x long fund doubles its underlying's squeeze while
    its own FINRA days to cover sits at the 1.0 floor; HIMZ -238%, 2025)."""
    if "m" not in _UNDERLYING:
        try:
            _UNDERLYING["m"] = json.loads((root() / "etf_underlying.json").read_text(encoding="utf-8"))
        except Exception:                                      # noqa: BLE001
            _UNDERLYING["m"] = {}
    return _UNDERLYING["m"].get(ticker)


def standing(d: date, tickers: Sequence[str], arm: str = "model") -> pd.DataFrame:
    """Per ticker: the number of scores, their max and min over the arm's
    freshness window (`own_window`) of sessions strictly before ``d`` — every
    run of those days, never ``d`` itself (`eval_metrics.own_history_standing`)
    — in ``arm``'s own score history."""
    w = own_window(arm)
    frames = []
    for s in sessions_before(d, w):
        p = scores_path(s, arm)
        if p.exists():
            frames.append(pd.read_pickle(p)[["ticker", "score"]])
    idx = pd.Index(list(dict.fromkeys(tickers)), name="ticker")
    if not frames:
        return pd.DataFrame({"n_prior": 0.0, "prior_max": np.nan, "prior_min": np.nan}, index=idx)
    h = pd.concat(frames, ignore_index=True)
    h = h[np.isfinite(h["score"])]
    g = h.groupby("ticker")["score"].agg(["count", "max", "min"])
    out = g.reindex(idx)
    return pd.DataFrame({"n_prior": out["count"].fillna(0.0).to_numpy(float),
                         "prior_max": out["max"].to_numpy(float),
                         "prior_min": out["min"].to_numpy(float)}, index=idx)


# ── the decision journal ─────────────────────────────────────────────────────

def picks_path(d: date) -> Path:
    return root() / "picks" / f"{_iso(d)}.jsonl"


def read_picks(d: date) -> List[dict]:
    p = picks_path(d)
    if not p.exists():
        return []
    out = []
    for line in p.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return out


def _journal(rec: dict) -> None:
    p = picks_path(date.fromisoformat(rec["day"]))
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(rec, default=str) + "\n")


def _consumed_path(d: date) -> Path:
    return root() / "picks" / f"{_iso(d)}.consumed.json"


def consumed(d: date) -> Dict[str, str]:
    p = _consumed_path(d)
    try:
        return json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}
    except Exception:
        return {}


def mark_consumed(d: date, key: str, outcome: str) -> None:
    c = consumed(d)
    c[key] = outcome
    p = _consumed_path(d)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(".tmp")
    tmp.write_text(json.dumps(c), encoding="utf-8")
    os.replace(tmp, p)


def pick_key(rec: dict) -> str:
    """The journal key a pick is consumed under — the model's keys unchanged
    since 2026-09-28, the other arms suffixed, so both arms can pick one name on
    one bar without consuming each other."""
    arm = rec.get("arm") or "model"
    base = f"{rec['day']}|{rec['bar_of_day']}|{rec['ticker']}"
    return base if arm == "model" else f"{base}|{arm}"


def pending_entries(now: Optional[datetime] = None) -> List[dict]:
    """Today's journaled shorts not yet handed to the ledger and still fresh
    enough to take (`sel_short_entry_max_age_minutes` after their bar's end)."""
    now = now or datetime.now(timezone.utc)
    et = now.astimezone(ET)
    d = et.date()
    done = consumed(d)
    max_age = timedelta(minutes=float(settings.sel_short_entry_max_age_minutes))
    out = []
    for rec in read_picks(d):
        if rec.get("decision") != "short" or pick_key(rec) in done:
            continue
        end = datetime.fromisoformat(rec["bar_end"])
        if now - end > max_age:
            mark_consumed(d, pick_key(rec), "expired")
            journal_entry(rec, "expired")
            continue
        out.append(rec)
    return out


# ── the entry step's journal and the trade log (user directive 2026-10-05: log, for every
# vol trade, the short-sale restriction, the borrow at entry and the broker's actual fill) ──

def entries_path(d: date) -> Path:
    return root() / "entries" / f"{_iso(d)}.jsonl"


def journal_entry(rec: dict, outcome: str, price: Optional[float] = None, borrow=None,
                  borrow_checked: Optional[bool] = None, recommendation_id: Optional[str] = None,
                  ibkr_margin: Optional[dict] = None) -> None:
    """One line per pick the ledger's entry step settled (`tracker.record_sel_short_trades`,
    `pending_entries`): the outcome (opened, no_borrow, borrow_fee, target_reached,
    already_open, ibkr_refused, the account's reasons, expired), the live price it was judged
    at, the pick's short-sale restriction, what IBKR's borrow file showed then and IBKR's
    what-if margin answer (``ibkr_margin``: its rates, a refusal or an error; None = not asked)
    — for the picks NOT taken too, so the backtest's borrow assumption (every pick borrowable)
    can be checked on the live picks. ``borrow`` is the `ibkr_borrow.Borrow` row (None: not in
    the file, or not checked — ``borrow_checked`` tells which). Fail-soft: a journal write
    never blocks an entry."""
    try:
        d = date.fromisoformat(str(rec["day"]))
        ts = getattr(borrow, "file_ts", None) if borrow is not None else None
        line = {"key": pick_key(rec), "day": rec.get("day"), "bar_of_day": rec.get("bar_of_day"),
                "arm": rec.get("arm") or "model", "ticker": rec.get("ticker"), "outcome": outcome,
                "at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                "price": _finite_or_none(price), "target": _finite_or_none(rec.get("target")),
                "pick_close": _finite_or_none(rec.get("px")), "ssr": rec.get("ssr"),
                "recommendation_id": recommendation_id, "borrow_checked": borrow_checked,
                "borrow_listed": (borrow is not None) if borrow_checked else None,
                "borrow_available": getattr(borrow, "available", None) if borrow is not None else None,
                "borrow_fee_pct": getattr(borrow, "fee_pct", None) if borrow is not None else None,
                "borrow_file_ts": ts.isoformat() if ts is not None else None,
                "ibkr_margin": ibkr_margin}
        p = entries_path(d)
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(line, default=str) + "\n")
    except Exception as e:                                     # noqa: BLE001
        logger.warning(f"[sel_short] entry journal failed for {rec.get('ticker')} ({e})")


def _read_jsonl(p: Path) -> List[dict]:
    if not p.exists():
        return []
    out = []
    for line in p.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return out


def trade_id(key: str) -> str:
    """The ledger's ``recommendation_id`` (and IBKR order ref) of the trade a pick opens."""
    import hashlib
    return hashlib.sha1(f"sel|{key}".encode("utf-8")).hexdigest()[:16]


TRADELOG_SESSIONS = 25                 # the prepare rewrites this many recent days (15-session holds + slack)


def tradelog_path(d: date) -> Path:
    return root() / "tradelog" / f"{_iso(d)}.jsonl"


_ORDER_COLS = ("event", "intent", "side", "requested_qty", "filled_qty", "model_price", "limit_price", "fill_price",
               "commission", "status", "ok", "error", "client_ref", "submitted_at", "bid_at_submit", "ask_at_submit")
_SUBMITS = ("SUBMIT", "SUBMIT_REFUSED", "SUBMIT_FAILED", "SETTLE_REANCHOR")
_FILLS = ("SETTLE_FILL", "FILL_REFRESH")


def _broker_rows(ids: Sequence[str]) -> Dict[str, List[dict]]:
    """Every `broker_orders` event of these trades (client refs ``<id>``, ``<id>-rN``,
    ``<id>-exit``, ``<id>-exit-rN``), read-only, oldest first, by trade id."""
    out: Dict[str, List[dict]] = {i: [] for i in ids}
    if not ids:
        return out
    from src.db.connection import connect
    q = (f"SELECT {', '.join(_ORDER_COLS)} FROM broker_orders WHERE "
         + " OR ".join("starts_with(client_ref, ?)" for _ in ids) + " ORDER BY submitted_at")
    with connect(read_only=True) as conn:
        rows = conn.execute(q, list(ids)).fetchall()
    for row in rows:
        r = dict(zip(_ORDER_COLS, row))
        ref = str(r.get("client_ref") or "")
        for i in ids:
            if ref.startswith(i):
                out[i].append(r)
                break
    return out


def _secs(a: Optional[str], b: Optional[str]) -> Optional[float]:
    try:
        return round((datetime.fromisoformat(str(b)) - datetime.fromisoformat(str(a))).total_seconds(), 1)
    except (TypeError, ValueError):
        return None


def _leg_log(t: dict, prefix: str, rows: List[dict], ledger_px: Optional[float], sell: bool) -> dict:
    """One order leg as the broker executed it: the trade's ``broker_*`` fields plus its
    order events (attempts, refusals, the book at the first submit, the time to fill,
    slippage against the ledger's price — positive = worse than the ledger)."""
    subs = [r for r in rows if r.get("event") in _SUBMITS]
    fills = [r for r in rows if r.get("event") in _FILLS and (r.get("filled_qty") or 0) > 0]
    first = subs[0] if subs else {}
    fill_px = t.get(f"{prefix}fill_price")
    slip = None
    if fill_px and ledger_px:
        slip = round(((float(ledger_px) - float(fill_px)) if sell else (float(fill_px) - float(ledger_px)))
                     / float(ledger_px) * 1e4, 1)
    errors = list(dict.fromkeys(str(r["error"]) for r in rows if r.get("error")))[:3]
    return {"status": t.get(f"{prefix}status"), "requested_qty": t.get(f"{prefix}requested_qty"),
            "filled_qty": t.get(f"{prefix}fill_qty"), "fill_price": fill_px,
            "commission": t.get(f"{prefix}commission"), "ledger_price": ledger_px, "slippage_bps": slip,
            "attempts": len(subs), "refused": sum(1 for r in rows if r.get("event") == "SUBMIT_REFUSED"),
            "killed_unfilled": sum(1 for r in rows if r.get("event") == "SETTLE_KILL"),
            "refused_reason": t.get(f"{prefix}refused_reason"), "errors": errors,
            "first_submit_at": first.get("submitted_at"), "model_price_at_submit": first.get("model_price"),
            "limit_at_submit": first.get("limit_price"), "bid_at_submit": first.get("bid_at_submit"),
            "ask_at_submit": first.get("ask_at_submit"),
            "first_fill_at": fills[0].get("submitted_at") if fills else None,
            "seconds_to_fill": _secs(first.get("submitted_at"), fills[0].get("submitted_at")) if fills and first else None}


def trade_log(d: date) -> List[dict]:
    """Every short the scorer handed to the ledger on ``d`` (all arms), joined end to end:
    the pick (short-sale restriction, days to cover, relative volume, target), the entry step
    (outcome, live price, IBKR's borrow then), the ledger trade (entry, exit, reason, return)
    and the broker's actual legs (`_leg_log`), plus the return the broker's fills realized
    (gross of commissions and borrow) when both legs filled in full."""
    picks = [r for r in read_picks(d) if r.get("decision") == "short"]
    if not picks:
        return []
    entries = {e["key"]: e for e in _read_jsonl(entries_path(d)) if e.get("key")}
    done = consumed(d)
    ids = [trade_id(pick_key(r)) for r in picks]
    from src.db.connection import connect
    with connect(read_only=True) as conn:
        trades = [json.loads(x[0]) for x in conn.execute("SELECT data FROM trades").fetchall()]
    by_id = {t.get("recommendation_id"): t for t in trades if t.get("entry_mechanism") == MECHANISM}
    orders = _broker_rows([i for i in ids if i in by_id])
    out = []
    for rec, tid in zip(picks, ids):
        key = pick_key(rec)
        e = entries.get(key, {})
        row = {"key": key, "day": rec.get("day"), "bar_of_day": rec.get("bar_of_day"), "bar_end": rec.get("bar_end"),
               "arm": rec.get("arm") or "model", "ticker": rec.get("ticker"), "pick_close": rec.get("px"),
               "pre5": rec.get("pre5"), "runup_pct": rec.get("runup_pct"), "target": rec.get("target"),
               "deadline": rec.get("deadline"), "days_to_cover": rec.get("days_to_cover"), "rvol": rec.get("rvol"),
               "atr_pct": rec.get("atr_pct"), "confidence": rec.get("confidence"),
               **{k: rec.get(k) for k in ("ssr", "ssr_prev_close", "ssr_day_low", "ssr_prev_low", "ssr_prev2_close")},
               "entry_outcome": e.get("outcome") or done.get(key) or "pending", "entry_step_at": e.get("at"),
               "entry_step_price": e.get("price"), "borrow_checked": e.get("borrow_checked"),
               "borrow_listed": e.get("borrow_listed"), "borrow_available": e.get("borrow_available"),
               "borrow_fee_pct": e.get("borrow_fee_pct"), "borrow_file_ts": e.get("borrow_file_ts"),
               "ibkr_margin": e.get("ibkr_margin")}
        t = by_id.get(tid)
        if t is not None:
            rows = orders.get(tid, [])
            ent = [r for r in rows if "-exit" not in str(r.get("client_ref"))]
            ext = [r for r in rows if "-exit" in str(r.get("client_ref"))]
            row.update({"recommendation_id": tid, "status": t.get("status"), "entry_datetime": t.get("entry_datetime"),
                        "entry_price": t.get("entry_price"), "entry_session": t.get("entry_session"),
                        "exit_datetime": t.get("exit_datetime"), "exit_price": t.get("exit_price"),
                        "exit_reason": t.get("exit_reason"), "exit_session": t.get("exit_session"),
                        "return_pct": t.get("return_pct"), "sel_stack_n": t.get("sel_stack_n"),
                        **{k: t.get(k) for k in ("sel_account_shares", "sel_account_equity", "sel_house_maint",
                                                 "sel_house_init", "sel_house_source")},
                        "broker_entry": _leg_log(t, "broker_", ent, t.get("entry_price"), sell=True),
                        "broker_exit": _leg_log(t, "broker_exit_", ext, t.get("exit_price"), sell=False)})
            be, bx = row["broker_entry"], row["broker_exit"]
            full = (be["filled_qty"] and bx["filled_qty"] and be["fill_price"] and bx["fill_price"]
                    and int(bx["filled_qty"]) >= int(be["filled_qty"]))
            row["broker_return_pct"] = (round((float(be["fill_price"]) - float(bx["fill_price"]))
                                              / float(be["fill_price"]) * 100.0, 3) if full else None)
        out.append(row)
    return out


def write_trade_logs(days: Sequence[date]) -> Dict[str, int]:
    """Rewrite ``tradelog/<day>.jsonl`` for each day that journaled a short (a held trade's
    exit lands days later, so the prepare rewrites the recent days every morning)."""
    res = {}
    for d in days:
        rows = trade_log(d)
        if not rows:
            continue
        p = tradelog_path(d)
        p.parent.mkdir(parents=True, exist_ok=True)
        tmp = p.with_suffix(".tmp")
        tmp.write_text("".join(json.dumps(r, default=str) + "\n" for r in rows), encoding="utf-8")
        os.replace(tmp, p)
        res[_iso(d)] = len(rows)
    return res


# ── bars and features ────────────────────────────────────────────────────────

def _fetch_today(tk: str, d: date, since: Optional[date] = None) -> Optional[pd.DataFrame]:
    """Regular-hours 30-minute bars from Polygon for ``since``..``d`` (``since``
    defaults to ``d``: the session's own bars; earlier when the deep store is
    behind, so a stale store can never hand the features a hole). None on a
    failure."""
    try:
        from src.data.intraday_store import _fetch_range, _rth_only
        df = _fetch_range(tk, _iso(since or d), _iso(d))
        return _rth_only(df) if df is not None and not df.empty else pd.DataFrame()
    except Exception as e:                                     # noqa: BLE001
        logger.debug(f"[sel_short] {tk}: bars failed ({e})")
        return None


def _deep_and_recent(tk: str, d: date, fetch: bool = True):
    """(deep store frame, bars after it up to ``d``) for one name."""
    from src.data.intraday_store import load_deep_30m
    deep = load_deep_30m(tk)
    if not fetch:
        return deep, pd.DataFrame()
    since = d
    if deep is not None and not deep.empty:
        last = pd.Timestamp(deep.index.max()).tz_localize("UTC").tz_convert(ET).date()
        since = min(d, last + timedelta(days=1))
    return deep, _fetch_today(tk, d, since)


def series(tk: str, today: Optional[pd.DataFrame], now_naive: pd.Timestamp, deep=None):
    """`hlc_30m`'s tuple for ``tk``: the deep store + the newer COMPLETED bars in
    ``today`` (bars ending at or before ``now_naive``), the tick cache not
    consulted."""
    from src.analysis.ml_dataset import hlc_from_frames
    if deep is None:
        from src.data.intraday_store import load_deep_30m
        deep = load_deep_30m(tk)
    frames = [deep] if deep is not None else []
    if today is not None and not today.empty:
        t = today[(pd.DatetimeIndex(today.index) + BAR) <= now_naive]
        if frames:
            t = t[t.index > frames[0].index.max()]
        frames.append(t)
    return hlc_from_frames(frames)


def _close_at(idx: pd.DatetimeIndex, close, day: date, bar_of_day: int) -> float:
    """The last close at or before bar ``bar_of_day`` of ``day`` (time-based, as
    the evaluation read the run-up)."""
    start = pd.Timestamp(bar_end_et(day, bar_of_day) - timedelta(minutes=30)).tz_convert("UTC").tz_localize(None)
    j = int(np.searchsorted(idx.values, np.datetime64(start), side="right")) - 1
    return float(close.iloc[j]) if j >= 0 else float("nan")


# A run-up base close older than one session before its day means the stored history
# has a GAP there. JOURNAL-ONLY (`pre5_stale`): the guard that refused such picks
# (`stale_history`, 2026-09-28) was removed 2026-09-30 (user: "Remove the Stale-history
# guard from live") — the run-up is measured from that last close whatever its age, as
# the evaluation always did; removing the guard added 9 trades over 2021-26 at +0.02
# %/day (memory/vol-arm-anatomy-2026-09.md).
RUNUP_MAX_LAG_SESSIONS = 1


def _stale_base(idx: Optional[pd.DatetimeIndex], day: date, bar_of_day: int) -> bool:
    """True when the last bar at or before bar ``bar_of_day`` of ``day`` is from more
    than `RUNUP_MAX_LAG_SESSIONS` sessions before ``day`` — the history has a gap.
    A REUSED ticker stitches two securities into one stored history: the bankrupt
    Akoustis at $0.04 read as a +63,878% run-up of the new AKTS at $23.80; IPOs on
    reused symbols (FIG, CRCL, STRC) and long halts do the same — 7-9% of the
    backtest's trades (memory/sel-short-backtest-audit-2026-09.md). A flag for the
    journal only: nothing decides on it."""
    if idx is None or not len(idx):
        return False
    start = pd.Timestamp(bar_end_et(day, bar_of_day) - timedelta(minutes=30)).tz_convert("UTC").tz_localize(None)
    j = int(np.searchsorted(idx.values, np.datetime64(start), side="right")) - 1
    if j < 0:
        return False                           # no bar at all: the base is NaN anyway
    bar_day = pd.Timestamp(idx[j]).tz_localize("UTC").tz_convert(ET).date()
    return bar_day < sessions_before(day, RUNUP_MAX_LAG_SESSIONS)[0]


# THE RELATIVE-VOLUME FILTER's input (user directive 2026-10-01: "implement relative
# volume filter to prod"): the pick bar's volume over the mean volume of the name's
# previous `RVOL_LOOKBACK_BARS` regular-hours 30-minute bars (20 sessions; at least
# `RVOL_MIN_PRIOR_BARS` needed) — the evaluation's ``vratio`` (60 of 60 backtest picks
# reproduced to 4e-8). The bar is located by its START, so bars after it (a past-session
# run reads the whole deep store) never enter the value (tests/test_no_lookahead.py).
RVOL_LOOKBACK_BARS = 260
RVOL_MIN_PRIOR_BARS = 20


def rvol_at(idx: Optional[pd.DatetimeIndex], vol, bar_start: pd.Timestamp) -> float:
    """The volume of the bar starting at ``bar_start`` over the mean volume of the
    up-to-260 bars before it in ``idx`` — NaN when the bar is not in the series, fewer
    than 20 bars precede it, or their mean is zero."""
    if idx is None or vol is None or not len(idx):
        return float("nan")
    j = int(np.searchsorted(idx.values, np.datetime64(pd.Timestamp(bar_start))))
    if j >= len(idx) or pd.Timestamp(idx[j]) != pd.Timestamp(bar_start):
        return float("nan")
    lo = max(0, j - RVOL_LOOKBACK_BARS)
    if j - lo < RVOL_MIN_PRIOR_BARS:
        return float("nan")
    v = np.asarray(vol, dtype=float)
    m = float(np.mean(v[lo:j]))
    return float(v[j]) / m if np.isfinite(m) and m > 0 else float("nan")


# THE SHORT-SALE RESTRICTION at the pick — JOURNAL-ONLY (user directive 2026-10-05: log the
# restriction state of every vol trade; 61% of the vol book's backtest entries happened under it,
# filled at the bid as if the restriction did not exist).
SSR_TRIGGER_DROP = 0.10                # Rule 201: a 10% decline from the previous session's close
SSR_WINDOW_BARS = 260                  # 20 sessions of stored bars searched for the two earlier sessions


def ssr_state(idx: Optional[pd.DatetimeIndex], low, close, day: date, bar_start: pd.Timestamp) -> dict:
    """Rule 201's short-sale price test at the pick bar, read off the stored regular-hours
    30-minute bars exactly as the backtest's flag (`study7.ssr_flag_cache`): in force when the
    low of ``day`` through the pick bar is at least 10% under the previous session's close, or
    the previous session's low was at least 10% under the close of the session before (the test
    runs through the following day). Two known gaps to the rule itself: a pre-market trigger is
    not seen (regular-hours bars), and the backtest read the day's low through its ENTRY bar, one
    bar later. Returns ``ssr`` (None when the bar or two earlier sessions are missing) and its
    inputs. Bars after ``bar_start`` never enter (tests/test_no_lookahead.py)."""
    out = {"ssr": None, "ssr_prev_close": None, "ssr_day_low": None, "ssr_prev_low": None,
           "ssr_prev2_close": None}
    if idx is None or low is None or close is None or not len(idx):
        return out
    j = int(np.searchsorted(idx.values, np.datetime64(pd.Timestamp(bar_start))))
    if j >= len(idx) or pd.Timestamp(idx[j]) != pd.Timestamp(bar_start):
        return out
    lo = max(0, j - SSR_WINDOW_BARS)
    w = pd.DatetimeIndex(idx[lo:j + 1])
    sd = np.asarray((w.tz_localize("UTC").tz_convert(ET).normalize().tz_localize(None) - EPOCH).days, np.int64)
    lw = np.asarray(low, dtype=float)[lo:j + 1]
    cw = np.asarray(close, dtype=float)[lo:j + 1]
    today = sd == dnum(day)
    prev = np.unique(sd[sd < dnum(day)])
    if len(prev) < 2 or not today.any():
        return out
    c1 = float(cw[np.flatnonzero(sd == prev[-1])[-1]])
    c2 = float(cw[np.flatnonzero(sd == prev[-2])[-1]])
    day_low = float(np.nanmin(lw[today]))
    prev_low = float(np.nanmin(lw[sd == prev[-1]]))
    k = 1.0 - SSR_TRIGGER_DROP
    out.update({"ssr": bool(day_low <= k * c1 or prev_low <= k * c2), "ssr_prev_close": c1,
                "ssr_day_low": day_low, "ssr_prev_low": prev_low, "ssr_prev2_close": c2})
    return out


# ── corporate-action gaps (user 2026-10-05: "Spin-offs aren't in the split data, so a CTVA-style gap can
# block the slot again" — fix it). Polygon adjusts its bars for splits, never for spin-offs or special
# distributions, and neither the split nor the dividend data records them: CTVA's 2026-10-01 spin-off
# (77.65 -> 14.44 at the open) read as a 30-minute ATR% near 40% decaying over days, the most volatile
# name of every bar, never fresh and never a riser — the vol arm took no trade for three sessions. A
# session that OPENS at least `CA_GAP_DROP` under the previous close while the name trades calmly around
# it is a level shift, not volatility: within `CA_GAP_SESSIONS` sessions of it (the run-up window, so
# such a name is no riser anyway) and while the ATR% computed WITHOUT session-opening gaps is under
# `CA_GAP_ATR_SHARE` of the full one, the vol and ETF arms neither rank the name nor add the bar to its
# freshness history (`arm_rows`). A real crash keeps trading wildly after the gap and stays ranked.
CA_GAP_DROP = 0.40
CA_GAP_SESSIONS = 5
CA_GAP_ATR_SHARE = 0.40
CA_GAP_WINDOW_BARS = 520           # 40 sessions: the ATR's EWM has forgotten everything older


def corporate_gap_flags(sday, high, low, close) -> np.ndarray:
    """Per bar of one name's regular-hours 30-minute series (oldest first): True when the bar's
    ATR% is mostly a corporate-action gap — a session within the last `CA_GAP_SESSIONS` (the bar's
    own included) whose first bar's HIGH is at most (1 - `CA_GAP_DROP`) x the previous session's last
    close (no trade at or above it: a gap down of at least 40%), and the 14-bar ATR without
    session-opening gaps under `CA_GAP_ATR_SHARE` of the full ATR (both the EWM the vol arm ranks
    on). A bar's flag reads only that bar and earlier ones."""
    c = np.asarray(close, dtype=float)
    n = len(c)
    out = np.zeros(n, dtype=bool)
    if n < 2:
        return out
    h, lo = np.asarray(high, dtype=float), np.asarray(low, dtype=float)
    sd = np.asarray(sday, dtype=np.int64)
    first = np.r_[True, sd[1:] != sd[:-1]]
    prev_c = np.r_[np.nan, c[:-1]]
    rng = h - lo
    with np.errstate(invalid="ignore"):
        tr_full = np.where(np.isfinite(prev_c), np.fmax(rng, np.fmax(np.abs(h - prev_c), np.abs(lo - prev_c))), rng)
    tr_free = np.where(first, rng, tr_full)
    atr_full = pd.Series(tr_full).ewm(alpha=1.0 / 14, adjust=False).mean().to_numpy()
    atr_free = pd.Series(tr_free).ewm(alpha=1.0 / 14, adjust=False).mean().to_numpy()
    sess = np.cumsum(first) - 1
    gap = first & np.isfinite(prev_c) & (h <= (1.0 - CA_GAP_DROP) * prev_c)
    last_gap = pd.Series(np.where(gap, sess, np.nan)).ffill().to_numpy()
    recent = np.isfinite(last_gap) & ((sess - np.nan_to_num(last_gap, nan=-1e9)) < CA_GAP_SESSIONS)
    with np.errstate(invalid="ignore", divide="ignore"):
        out = recent & (atr_free < CA_GAP_ATR_SHARE * atr_full)
    return out


def ca_gap_state(idx: Optional[pd.DatetimeIndex], high, low, close, bar_start: pd.Timestamp) -> dict:
    """`corporate_gap_flags` at the bar starting ``bar_start``, read on the `CA_GAP_WINDOW_BARS` bars
    through it (never a later one), with the gap's size. ``ca_gap`` is None when the bar is missing."""
    out = {"ca_gap": None, "ca_gap_pct": None}
    if idx is None or high is None or low is None or close is None or not len(idx):
        return out
    j = int(np.searchsorted(idx.values, np.datetime64(pd.Timestamp(bar_start))))
    if j >= len(idx) or pd.Timestamp(idx[j]) != pd.Timestamp(bar_start):
        return out
    lo_i = max(0, j - CA_GAP_WINDOW_BARS + 1)
    w = pd.DatetimeIndex(idx[lo_i:j + 1])
    sd = np.asarray((w.tz_localize("UTC").tz_convert(ET).normalize().tz_localize(None) - EPOCH).days, np.int64)
    hh = np.asarray(high, dtype=float)[lo_i:j + 1]
    ll = np.asarray(low, dtype=float)[lo_i:j + 1]
    cc = np.asarray(close, dtype=float)[lo_i:j + 1]
    flags = corporate_gap_flags(sd, hh, ll, cc)
    out["ca_gap"] = bool(flags[-1])
    if out["ca_gap"]:                      # the gap's size: its first bar's high against the previous close
        first = np.flatnonzero(np.r_[True, sd[1:] != sd[:-1]])
        ks = [int(k) for k in first if k > 0 and hh[k] <= (1.0 - CA_GAP_DROP) * cc[k - 1]]
        if ks:
            out["ca_gap_pct"] = round((hh[ks[-1]] / cc[ks[-1] - 1] - 1.0) * 100.0, 2)
    return out


def day_extras_path(d: date) -> Path:
    """The V2 model's per-session inputs of ``d`` (`sel_v2.E_SESS`, known at 08:30 ET) for the day's names."""
    return root() / "extras" / f"{_iso(d)}.json"


def ensure_day_extras(d: date, names: Sequence[str], force: bool = False) -> int:
    """Compute and store the V2 per-session inputs of ``d`` for ``names`` when the day's file is missing
    or misses names (one deep-store part read per name, threads). Returns how many names were computed."""
    from src.signals import sel_v2
    p = day_extras_path(d)
    have = {} if force else (sel_v2.read_day_extras(p) or {})
    todo = [t for t in names if t not in have]
    if not todo:
        return 0
    have.update(sel_v2.day_extras(todo, dnum(d)))
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(".tmp")
    tmp.write_text(json.dumps(have), encoding="utf-8")
    os.replace(tmp, p)
    return len(todo)


def _score_chunk(args) -> List[dict]:
    """One worker: fetch today's bars for its tickers (threads), then compute each
    one's feature vector at the target bar and score the chunk in one predict.
    A V2 model (`sel_v2.is_v2`) is NOT scored here: its cross-sectional inputs need
    every name of the bar, so the worker returns each row's vector (``_vec``) and
    the cross-section's sources (``_x_*``) and `score_bar` scores the bar."""
    tickers, day_iso, bar_of_day, dv20 = args
    os.environ.setdefault("PYTHONIOENCODING", "utf-8")
    from loguru import logger as _lg
    _lg.remove()
    from src.analysis import deep_features as dfe
    from src.signals import sel_v2
    from src.signals.ml_model import features_30m_from_hlc
    booster, meta = load_model()
    feats = list(meta["features"])
    v2 = sel_v2.is_v2(meta)
    model_set = set(meta["tickers"]) if meta.get("tickers") else None      # None: every name is the model's
    d = date.fromisoformat(day_iso)
    day_ext = (sel_v2.read_day_extras(day_extras_path(d)) or {}) if v2 else {}
    end = bar_end_et(d, bar_of_day)
    now_naive = pd.Timestamp(end).tz_convert("UTC").tz_localize(None)
    target_start = now_naive - BAR
    pre_day = sessions_before(d, int(settings.sel_short_runup_sessions))[0]
    fetch = not _NO_FETCH
    with ThreadPoolExecutor(max(1, int(settings.sel_short_fetch_threads))) as ex:
        loaded = dict(zip(tickers, ex.map(lambda t: _deep_and_recent(t, d, fetch), tickers)))
    snap = dfe.load_session_snapshot(dnum(d))
    vol_feature = str(settings.sel_short_vol_feature)
    rows, vecs = [], []
    for tk in tickers:
        rec = {"ticker": tk, "dv20": dv20.get(tk, float("nan")), "score": float("nan"),
               "vol": float("nan"), "dtc": float("nan"), "px": float("nan"), "pre5": float("nan"),
               "pre5_stale": False, "rvol": float("nan"), "ca_gap": None, "ca_gap_pct": None, "status": ""}
        deep, today = loaded.get(tk, (None, None))
        if today is None:
            rec["status"] = "FETCH_FAILED"
            rows.append(rec)
            continue
        hlc = series(tk, today, now_naive, deep=deep)
        frame, label = (features_30m_from_hlc(tk, hlc, now_naive, with_series=True) if v2
                        else features_30m_from_hlc(tk, hlc, now_naive))
        if frame is None:
            rec["status"] = label
            rows.append(rec)
            continue
        if pd.Timestamp(frame["bar_ts"]) != target_start:
            rec["status"] = "NO_BAR"                       # no trade in the target bar
            rows.append(rec)
            continue
        rec["px"] = float(frame["close"])
        rec["pre5"] = _close_at(hlc[0], hlc[3], pre_day, bar_of_day)
        rec["pre5_stale"] = _stale_base(hlc[0], pre_day, bar_of_day)   # journal-only: a gap before the base
        rec["rvol"] = rvol_at(hlc[0], hlc[4], target_start)             # the vol arm's relative-volume filter
        rec.update(ssr_state(hlc[0], hlc[2], hlc[3], d, target_start))  # journal-only: the short-sale restriction
        rec.update(ca_gap_state(hlc[0], hlc[1], hlc[2], hlc[3], target_start))  # a corporate-action gap: unranked
        try:                                   # the volatility arm's score: a base feature, no snapshot needed
            # float32, as the history (the arrays / backfill) stores it: the live
            # value then equals the history's for the same bar, so the freshness
            # comparison against the prior max is like for like (2026-09-26 dry
            # run: 1,973 of 1,982 names equal, the other 9 the clock-vs-position bars)
            rec["vol"] = float(np.float32(frame["features"].get(vol_feature, float("nan"))))
        except (TypeError, ValueError):
            rec["vol"] = float("nan")
        if snap is None:
            rec["status"] = "NO_SNAPSHOT"
            rows.append(rec)
            continue
        deep = dfe.serving_vector(tk, frame["sday"], frame["close"], frame["bar_idx"], snap)
        rec["dtc"] = float(deep.get(DTC_FEATURE, float("nan")))     # the short-interest filter's input
        if (model_set is not None and tk not in model_set) or not (
                rec["dv20"] >= float(settings.sel_short_min_dollar_volume)):
            # the vol arm's extra names, and the THIN stocks under the vol floor (2026-10-07; never a model
            # input, never in the V2 cross-section): ATR% + days to cover
            rec["status"] = "VOL_ONLY"
            rows.append(rec)
            continue
        base = frame["features"]
        if v2:
            # the V2 inputs of this name at this bar: its own series' (the visible bars, as the
            # base features read them) and the session's (the day's file, else read now)
            n_vis = int(frame["n_vis"])
            cut = (hlc[0][:n_vis], hlc[1].iloc[:n_vis], hlc[2].iloc[:n_vis], hlc[3].iloc[:n_vis],
                   hlc[4].iloc[:n_vis])
            atr = np.asarray(frame.get("atr_series"), dtype=float)
            if len(atr) != n_vis:
                atr = np.full(n_vis, np.nan)
            ext = dict(zip(sel_v2.E_BAR, sel_v2.bar_extras(cut, atr)[-1]))
            es = day_ext.get(tk)
            if es is None:
                es = sel_v2.session_extras(tk, [int(frame["sday"])])[0]
            ext.update(zip(sel_v2.E_SESS, (float(x) for x in es)))
            rec["_vec"] = np.array([base[f] if f in base else ext[f] if f in ext else deep.get(f, np.nan)
                                    for f in feats], dtype=np.float64)
            for s in sel_v2.XS_SOURCES:
                rec[f"_x_{s}"] = float(ext[s]) if s in ext else float(base.get(s, np.nan))
            rec["status"] = "OK"
            rows.append(rec)
            continue
        v = np.array([base[f] if f in base else deep.get(f, np.nan) for f in feats], dtype=np.float64)
        rec["status"] = "OK"
        rows.append(rec)
        vecs.append((len(rows) - 1, v))
    if vecs:
        M = np.vstack([v for _, v in vecs]).astype(np.float32)
        pred = booster.predict(M, num_iteration=int(meta.get("num_iteration") or NUM_ITERATION))
        for (i, _), s in zip(vecs, pred):
            rows[i]["score"] = float(s)
    return rows


def _dv20(tk: str, d: date) -> float:
    """Mean regular-hours dollar volume of the 20 sessions BEFORE ``d`` (>= 10
    needed), from the deep store — `ml30.ticker_rows`' ``dv20``."""
    from src.data.intraday_store import load_deep_30m
    df = load_deep_30m(tk)
    if df is None or df.empty:
        return float("nan")
    idx = pd.DatetimeIndex(df.index)
    et = idx.tz_localize("UTC").tz_convert("America/New_York")
    sd = et.normalize().tz_localize(None)
    c = pd.to_numeric(df["Close"], errors="coerce").to_numpy(float)
    v = pd.to_numeric(df.get("Volume", pd.Series(0.0, index=df.index)), errors="coerce").fillna(0).to_numpy(float)
    s = pd.Series(c * v, index=sd).groupby(level=0).sum()
    s = s[s.index < pd.Timestamp(d)].tail(20)
    return float(s.mean()) if len(s) >= 10 else float("nan")


# ── the volatility-normalised exit's input (user directive 2026-09-28) ───────
# A held short is covered IN PROFIT once its 30-minute ATR% has halved since the
# pick (tracker.monitor_sel_short_positions). The current value is built exactly
# as the scorer builds a pick's — for any held name, inside the day's universe or
# not (a short that worked may have fallen under the $5 floor and out of it).

_ATR_MEMO: Dict[Tuple[str, str], float] = {}


def last_completed_bar(now: datetime) -> Tuple[date, int]:
    """The latest regular-hours bar completed at ``now``: the previous session's
    last bar before today's first one ends, and on weekends and holidays."""
    lb = latest_bar(now)
    if lb is not None:
        return lb
    return sessions_before(now.astimezone(ET).date(), 1)[0], BARS_PER_SESSION - 1


def bar_from_end(bar_end_iso: str) -> Tuple[date, int]:
    """``(day, bar_of_day)`` of a journaled ``bar_end`` (ET ISO)."""
    t = datetime.fromisoformat(bar_end_iso).astimezone(ET)
    return t.date(), int((t.hour * 60 + t.minute - 570) // 30 - 1)


def atr_at(tk: str, d: date, bar_of_day: int) -> Optional[float]:
    """``tk``'s 30-minute ATR% (`sel_short_vol_feature`) at bar ``bar_of_day`` of
    ``d``: the deep store + that session's Polygon bars, `features_30m_from_hlc`,
    float32 — the scorer's construction. A bar Polygon has not published yet
    serves the session's newest bar (not memoised); a series whose newest bar is
    from an EARLIER session (a failed fetch) gives None, never an old value."""
    end = bar_end_et(d, bar_of_day)
    key = (tk, end.isoformat())
    if key in _ATR_MEMO:
        return _ATR_MEMO[key]
    try:
        from src.signals.ml_model import features_30m_from_hlc
        now_naive = pd.Timestamp(end).tz_convert("UTC").tz_localize(None)
        deep, today = _deep_and_recent(tk, d, fetch=not _NO_FETCH)
        frame, _label = features_30m_from_hlc(tk, series(tk, today, now_naive, deep=deep), now_naive)
        if frame is None:
            return None
        bar_ts = pd.Timestamp(frame["bar_ts"])
        if bar_ts.tz_localize("UTC").tz_convert(ET).date() != d:
            return None
        v = float(np.float32(frame["features"].get(str(settings.sel_short_vol_feature), float("nan"))))
    except Exception as e:                                     # noqa: BLE001
        logger.debug(f"[sel_short] {tk}: ATR% at {end:%Y-%m-%d %H:%M} failed ({e})")
        return None
    if not (np.isfinite(v) and v > 0):
        return None
    if bar_ts == now_naive - BAR:                  # the exact bar: it never changes again
        if len(_ATR_MEMO) > 5000:
            _ATR_MEMO.clear()
        _ATR_MEMO[key] = v
    return v


def live_atr(tickers: Sequence[str], now: Optional[datetime] = None) -> Dict[str, dict]:
    """Each name's ATR% at the latest completed regular-hours bar —
    ``{ticker: {"atr_pct", "bar_end"}}``; a name that cannot be computed is left out."""
    now = now or datetime.now(timezone.utc)
    d, k = last_completed_bar(now)
    end = bar_end_et(d, k).isoformat()
    out: Dict[str, dict] = {}
    for tk in dict.fromkeys(tickers):
        v = atr_at(tk, d, k)
        if v is not None:
            out[tk] = {"atr_pct": v, "bar_end": end}
    return out


# ── prepare (once per market day) ────────────────────────────────────────────

def universe_path(d: date) -> Path:
    return root() / "universe" / f"{_iso(d)}.json"


def live_arms() -> List[str]:
    """The arms the scorer runs: the model always, the vol and ETF arms when on, the vol arm's thin stocks
    (`enable_sel_short_thin`) and vol2 (`enable_sel_short_vol2`) with the vol arm."""
    vol_on = bool(getattr(settings, "enable_sel_short_vol", False))
    return (["model"] + (["vol"] if vol_on else [])
            + (["etf"] if getattr(settings, "enable_sel_short_etf", False) else [])
            + (["thin"] if vol_on and getattr(settings, "enable_sel_short_thin", False) else [])
            + (["vol2"] if vol_on and getattr(settings, "enable_sel_short_vol2", False) else []))


# ── the vol arm's THIN STOCKS (user directive 2026-10-07: "Add the thin stocks to the live vol arm") ──
# Common stocks / ADRs at $5+ whose 20-session mean regular-hours dollar volume sits between
# `sel_short_thin_min_dollar_volume` ($1M) and `sel_short_min_dollar_volume` ($5M) at the prepare —
# the names the vol arm's $5M floor leaves out. The prepare writes them to `universe_thin/<day>.json`
# (never the model's or the vol arm's universe); the run scores them beside the universe and flags
# their rows (``thin``); the "thin" arm ranks ONLY those rows with the vol rule (its own freshness
# history `scores_thin/`), so a thin name never takes a liquid pick's slot and never enters the vol
# arm's history. Each prepare also screens the whole market for thin common stocks outside the deep
# store (`screen_listings(band="thin")`) and fetches their history (`add_listings(arm="thin")`). The
# entry step funds a thin short from FREE CAPITAL only (`sim_account.size(reserve_slices=...)`, after
# every other arm's picks). PREREG32 (2021-02..2024-06, one account): final growth +27.2 vs +10.7
# %/yr, drawdown 46% vs 12%, not significant on the time-weighted bar; deployed on the user's order.
def thin_universe_path(d: date) -> Path:
    return root() / "universe_thin" / f"{_iso(d)}.json"


def load_thin_universe(d: date) -> Optional[Dict[str, float]]:
    p = thin_universe_path(d)
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


def vol_extra_names(model_names: Sequence[str]) -> List[str]:
    """The deep store's names outside the model's training set that are not the
    vol arm's added stocks — the exchange-traded products only the ETF arm ranks
    (`enable_sel_short_etf`)."""
    from src.data.deep import deep_universe
    have = set(model_names) | set(vol_listing_names())
    return sorted(t for t in deep_universe() if t not in have)


# ── the vol arm's ADDED STOCKS ("listings": every listed common stock outside the store) ──
# The model's name list is frozen at its install (`model.json` "tickers", 3,430 names) and
# `prepare` ranked only those (+ the ETF arm's products), while ~1,000 liquid common stocks sat
# outside the deep store, and outside the backtest (HKD, SMX, TOP: the most volatile eligible
# name on 31% of 2021-26 bars). User directives 2026-10-05: "New listings + test the rest" (the
# names listed within a year, added that morning), then, once the backtest with every common
# stock had run (`memory/vol-arm-every-stock-2026-10.md`), "Add them to live and backtest the
# results are good". Each prepare screens Polygon's whole-market daily bars for COMMON STOCKS /
# ADRs outside the deep store that averaged >= `sel_short_min_dollar_volume` over >= 10 of the
# 20 sessions before the day, whatever their listing date (`screen_listings`), fetches their
# 30-minute history into the deep store and records them (`add_listings` -> `vol_listings.json`).
# The vol arm ranks them beside the model's names; the model arm never (no model score) and the
# ETF arm never (not a product). Their ATR% still waits for 400 bars (`ml_model._MIN_30M_BARS`),
# as for every live name.
LISTING_TYPES = ("CS", "ADRC")
LISTING_MIN_SESSIONS = 10              # as `_dv20`: the mean of at least 10 of the last 20 sessions
LISTING_SESSIONS = 20
DETAILS_RETRY_HOURS = 20.0             # a symbol Polygon had no record of is asked again next day


def listings_path() -> Path:
    return root() / "vol_listings.json"


def read_listings() -> Dict[str, dict]:
    """``{ticker: {"added", "type", "list_date", "name"}}`` of the vol arm's added stocks."""
    p = listings_path()
    try:
        return json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}
    except Exception as e:                                     # noqa: BLE001
        logger.error(f"[sel_short] {p} unreadable ({e}) — the vol arm ranks no added listing")
        return {}


def vol_listing_names() -> List[str]:
    """The common stocks added to the vol arm's universe (`add_listings`)."""
    return sorted(read_listings())


def _internal_symbol(ticker: str) -> str:
    """Polygon's class-share symbol (``BRK.B``) in the project's form (``BRK-B``) — the
    inverse of `polygon_client.to_polygon_symbol`."""
    import re
    return re.sub(r"\.([A-Z])$", r"-\1", str(ticker).strip().upper())


def grouped_day(s: date) -> Optional[pd.DataFrame]:
    """Polygon's whole-market daily bars of the COMPLETED session ``s`` (ticker in Polygon's
    form, close, volume), cached under ``grouped/<day>.pkl``. None when Polygon returns
    nothing — not cached, so the next prepare asks again."""
    p = root() / "grouped" / f"{_iso(s)}.pkl"
    if p.exists():
        return pd.read_pickle(p)
    from src.data import polygon_client as pc
    rows = pc.get_grouped_daily(_iso(s))
    if not rows:
        return None
    raw = pd.DataFrame(rows)
    if not {"T", "c", "v"} <= set(raw.columns):
        return None
    df = pd.DataFrame({"ticker": raw["T"].astype(str), "close": pd.to_numeric(raw["c"], errors="coerce"),
                       "volume": pd.to_numeric(raw["v"], errors="coerce")})
    _write_pickle(df, p)
    return df


def ticker_details(tickers: Sequence[str]) -> Dict[str, dict]:
    """Polygon's reference record per ticker (``type``, ``list_date``, ``active``, ``name``),
    cached in ``ticker_details.json``; a symbol Polygon has no record of (an exchange test
    symbol such as ZVZZT) is cached empty and asked again after `DETAILS_RETRY_HOURS`."""
    p = root() / "ticker_details.json"
    try:
        cache = json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}
    except Exception:                                          # noqa: BLE001
        cache = {}
    now = datetime.now(timezone.utc)
    todo = []
    for t in dict.fromkeys(tickers):
        c = cache.get(t)
        if c is None:
            todo.append(t)
            continue
        if not c.get("type"):
            try:
                age_h = (now - datetime.fromisoformat(str(c.get("at")))).total_seconds() / 3600.0
            except (TypeError, ValueError):
                age_h = float("inf")
            if age_h >= DETAILS_RETRY_HOURS:
                todo.append(t)
    if todo:
        from src.data import polygon_client as pc
        with ThreadPoolExecutor(8) as ex:
            got = list(ex.map(pc.get_ticker_details, todo))
        stamp = now.isoformat(timespec="seconds")
        for t, r in zip(todo, got):
            r = r or {}
            cache[t] = {"type": r.get("type"), "list_date": r.get("list_date"), "active": r.get("active"),
                        "name": r.get("name"), "at": stamp}
        p.parent.mkdir(parents=True, exist_ok=True)
        tmp = p.with_suffix(".tmp")
        tmp.write_text(json.dumps(cache), encoding="utf-8")
        os.replace(tmp, p)
    return {t: dict(cache.get(t) or {}) for t in tickers}


def screen_listings(d: date, band: str = "core") -> List[dict]:
    """The common stocks / ADRs the vol arm adds on day ``d``: outside the deep store, the
    model's names and the recorded ones; a mean daily dollar volume (Polygon close x volume)
    >= `sel_short_min_dollar_volume` over >= `LISTING_MIN_SESSIONS` of the `LISTING_SESSIONS`
    sessions BEFORE ``d`` (no bar of ``d`` or later is read); Polygon type CS / ADRC and
    active, whatever the listing date. The live universe's floor (`_dv20`, regular hours from
    the deep store) still applies at the prepare. ``band="thin"``: the thin stocks instead —
    a mean from `sel_short_thin_min_dollar_volume` up to (not including) the vol floor."""
    from src.data.deep import deep_universe
    frames = {}
    for s in sessions_before(d, LISTING_SESSIONS):
        g = grouped_day(s)
        if g is not None and len(g):
            frames[s] = g.drop_duplicates("ticker").set_index("ticker")
    if len(frames) < LISTING_MIN_SESSIONS:
        logger.warning(f"[sel_short] listings screen {d}: {len(frames)} sessions of whole-market bars — skipped")
        return []
    dv = pd.DataFrame({s: g["close"] * g["volume"] for s, g in frames.items()})
    n = dv.notna().sum(axis=1)
    mean = dv.mean(axis=1, skipna=True)
    have = set(deep_universe()) | set(read_listings())
    try:
        _, meta = load_model()
        have |= set(meta.get("tickers") or [])
    except Exception:                                          # noqa: BLE001 — no installed model
        pass
    floor = float(settings.sel_short_min_dollar_volume)
    if band == "thin":
        sel = (n >= LISTING_MIN_SESSIONS) & (mean >= float(settings.sel_short_thin_min_dollar_volume)) & (mean < floor)
    else:
        sel = (n >= LISTING_MIN_SESSIONS) & (mean >= floor)
    cand: Dict[str, float] = {}
    for t, v in mean[sel].items():
        it = _internal_symbol(t)
        if it not in have and it.replace("-", "").isalnum():
            cand[it] = float(v)
    if not cand:
        return []
    det = ticker_details(sorted(cand))
    out = []
    for t in sorted(cand):
        r = det.get(t) or {}
        if r.get("type") in LISTING_TYPES and r.get("active") is not False:
            listed = str(r.get("list_date"))[:10] if r.get("list_date") else None
            out.append({"ticker": t, "type": r.get("type"), "list_date": listed, "name": r.get("name"),
                        "dollar_volume": round(cand[t], 1)})
    return out


def add_listings(d: date, found: Sequence[dict], backfill: bool = True, arm: str = "vol") -> dict:
    """Bring screened stocks into the vol arm: fetch each one's 30-minute history into the
    deep store (from 2021, through the session before ``d``), refresh the deep universe (the
    nightly and pre-open refreshes then carry its short interest, the session snapshot its days
    to cover), record it in `vol_listings.json` and seed its vol score history over the arm's
    freshness window with the bars the live scorer would have scored — 400 bars visible, the
    price and dollar-volume floors — added to the day files that exist, every other row kept.
    Without that history a name would read as fresh on its first live bar whatever its past."""
    out: dict = {"added": []}
    if not found:
        return out
    from src.data.deep import deep_universe
    from src.data.intraday_store import extend_deep_30m, load_deep_30m
    from src.signals.ml_model import _MIN_30M_BARS
    prev = sessions_before(d, 1)[0]
    names = [str(r["ticker"]) for r in found]
    out["extend"] = extend_deep_30m(names, workers=8, budget_seconds=900.0, min_age_days=0, today=prev)
    ok = []
    for r in found:
        df = load_deep_30m(str(r["ticker"]))
        if df is not None and len(df):
            ok.append(r)
    if not ok:
        return out
    deep_universe(refresh=True)
    cur = read_listings()
    for r in ok:
        cur[str(r["ticker"])] = {"added": _iso(d), "type": r.get("type"), "list_date": r.get("list_date"),
                                 "name": r.get("name"), **({"band": "thin"} if arm == "thin" else {})}
    p = listings_path()
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(".tmp")
    tmp.write_text(json.dumps(cur, indent=1), encoding="utf-8")
    os.replace(tmp, p)
    out["added"] = [str(r["ticker"]) for r in ok]
    if backfill and out["added"]:
        try:
            arms_ = (arm,) + (("vol2",) if arm == "vol" and "vol2" in live_arms() else ())   # vol2 = vol's history
            out["history"] = backfill_days(sessions_before(d, own_window(arm)), tickers=out["added"],
                                           arms=arms_, merge=True, existing_only=True,
                                           min_visible_bars=int(_MIN_30M_BARS))
        except Exception as e:                                 # noqa: BLE001 — names stay; history seeds later
            logger.error(f"[sel_short] listings {d}: vol history seed failed ({e}) — the added names read as "
                         f"fresh until 10 bar scores accrue")
            out["history_error"] = f"{type(e).__name__}: {e}"[:200]
    logger.info(f"[sel_short] listings {d}: added {len(out['added'])} to the vol arm — {out['added']}")
    return out


def model_universe(uni, meta: Optional[dict] = None) -> List[str]:
    """The day's universe restricted to the model's names: what the snapshot's
    coverage is judged on (a vol-only name missing from the snapshot must never
    take the model arm off the bar)."""
    if meta is None:
        try:
            _, meta = load_model()
        except Exception:                                      # noqa: BLE001 — no installed model
            return sorted(uni)
    if not meta.get("tickers"):
        return sorted(uni)
    have = set(meta["tickers"])
    return sorted(t for t in uni if t in have)


def load_universe(d: date) -> Optional[Dict[str, float]]:
    p = universe_path(d)
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


def prepare(d: date, extend: bool = True, workers: int = 16) -> dict:
    """Extend the deep 30-minute store through the previous session, fix the day's
    universe (dv20 >= the floor), make sure the session snapshot exists, and fill
    missing score days of the own-history window."""
    t0 = time.time()
    _, meta = load_model()
    model_names = list(meta.get("tickers") or [])
    out: dict = {"day": _iso(d)}
    vol_on = bool(getattr(settings, "enable_sel_short_vol", False))
    if vol_on and getattr(settings, "enable_sel_short_vol_added_stocks", False):
        t1 = time.time()
        try:                                  # the vol arm's added stocks (2026-10-05), before the universe
            found = screen_listings(d)
            out["listings_found"] = len(found)
            if found:
                out["listings"] = add_listings(d, found)
        except Exception as e:                                 # noqa: BLE001 — the day runs on the recorded names
            logger.error(f"[sel_short] prepare {d}: added-stocks screen failed ({e})")
            out["listings_error"] = f"{type(e).__name__}: {e}"[:200]
        out["listings_seconds"] = round(time.time() - t1, 1)
    thin_on = vol_on and bool(getattr(settings, "enable_sel_short_thin", False))
    if thin_on:
        t1 = time.time()
        try:                                  # the thin stocks outside the store (2026-10-07), before the universe
            found = screen_listings(d, band="thin")
            out["thin_listings_found"] = len(found)
            if found:
                out["thin_listings"] = add_listings(d, found, arm="thin")
        except Exception as e:                                 # noqa: BLE001 — the day runs on the recorded names
            logger.error(f"[sel_short] prepare {d}: thin-stock screen failed ({e})")
            out["thin_listings_error"] = f"{type(e).__name__}: {e}"[:200]
        out["thin_listings_seconds"] = round(time.time() - t1, 1)
    have = set(model_names)
    listings = [t for t in vol_listing_names() if t not in have] if vol_on else []
    tickers = (model_names + (vol_extra_names(model_names) if getattr(settings, "enable_sel_short_etf", False)
                              else []) + listings)
    prev = sessions_before(d, 1)[0]
    out.update({"tickers": len(tickers), "vol_only_tickers": len(tickers) - len(model_names),
                "vol_listings": len(listings)})
    if extend:
        from src.data.intraday_store import extend_deep_30m
        out["extend"] = extend_deep_30m(tickers, workers=workers, budget_seconds=2400.0, min_age_days=0,
                                        today=prev)
    with ThreadPoolExecutor(8) as ex:
        dv = dict(zip(tickers, ex.map(lambda t: _dv20(t, d), tickers)))
    floor = float(settings.sel_short_min_dollar_volume)
    uni = {t: round(v, 1) for t, v in dv.items() if np.isfinite(v) and v >= floor}
    # a split effective up to today resets the name's history before any
    # inference on it (user directive 2026-09-27)
    from src.data.intraday_store import reset_split_tickers
    out["split_resets"] = reset_split_tickers(sorted(uni), d)
    p = universe_path(d)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(uni), encoding="utf-8")
    out["universe"] = len(uni)
    if thin_on:
        # the thin stocks: common stocks (never a product) between the thin floor and the vol floor
        products = set(etf_names(tickers, model_names))
        tfloor = float(settings.sel_short_thin_min_dollar_volume)
        thin = {t: round(v, 1) for t, v in dv.items() if t not in products and t not in uni
                and np.isfinite(v) and tfloor <= v < floor}
        out["thin_split_resets"] = reset_split_tickers(sorted(thin), d)
        tp = thin_universe_path(d)
        tp.parent.mkdir(parents=True, exist_ok=True)
        tp.write_text(json.dumps(thin), encoding="utf-8")
        out["thin_universe"] = len(thin)
    from src.signals import sel_v2
    if sel_v2.is_v2(meta):                    # the V2 model's per-session inputs, at their 08:30 ET cutoff
        t1 = time.time()
        try:
            out["v2_session_inputs"] = ensure_day_extras(d, model_universe(uni, meta), force=True)
        except Exception as e:                                 # noqa: BLE001 — the first run retries
            logger.error(f"[sel_short] prepare {d}: V2 session inputs failed ({e})")
            out["v2_session_inputs_error"] = f"{type(e).__name__}: {e}"[:200]
        out["v2_session_seconds"] = round(time.time() - t1, 1)
    # The session snapshot is the 08:30 pre-open run's job (`run` builds it if it
    # is still missing at the first bar) - building a MISSING one here would race
    # that run. One that exists but was built on a store that was behind is
    # rebuilt now that the store is extended (`ensure_snapshot`).
    from src.analysis import deep_features as dfe
    if extend and dfe.snapshot_path(dnum(d)).exists():
        out["snapshot_coverage"] = round(ensure_snapshot(d, model_universe(uni, meta)), 3)
    for arm in live_arms():
        missing = [s for s in sessions_before(d, own_window(arm)) if not scores_path(s, arm).exists()]
        out["missing_score_days" if arm == "model" else f"missing_score_days_{arm}"] = [_iso(s) for s in missing]
        if missing:                                  # ~20 min per pass: never inside the pre-open prepare
            logger.warning(f"[sel_short] {arm} score history missing for {[_iso(s) for s in missing]} — the "
                           f"freshness rule reads fewer days; fill with `python -m src.signals.sel_short "
                           f"--backfill --arms {arm} --day {_iso(missing[0])} --until {_iso(missing[-1])}`")
    try:                                       # the trade log of the recent sessions (exits land days later)
        out["trade_logs"] = write_trade_logs(sessions_before(d, TRADELOG_SESSIONS))
    except Exception as e:                                     # noqa: BLE001 — a log never blocks the day
        logger.error(f"[sel_short] prepare {d}: trade log failed ({e})")
        out["trade_log_error"] = f"{type(e).__name__}: {e}"[:200]
    out["seconds"] = round(time.time() - t0, 1)
    pp = root() / "prepare" / f"{_iso(d)}.json"
    pp.parent.mkdir(parents=True, exist_ok=True)
    pp.write_text(json.dumps(out, default=str), encoding="utf-8")
    logger.info(f"[sel_short] prepare {d}: {out}")
    return out


SNAPSHOT_MIN_COVERAGE = 0.8


def snapshot_coverage(d: date, names: Sequence[str]) -> float:
    """Share of ``names`` the session snapshot of ``d`` carries WITH a previous
    close — the input every price-dependent deep feature needs. A snapshot built
    on a 30-minute store that was behind reads ~0.1 here (the live 09-25 file)."""
    from src.analysis import deep_features as dfe
    snap = dfe.load_session_snapshot(dnum(d))
    if not snap:
        return 0.0
    have = [snap[t] for t in names if t in snap]
    if not have:
        return 0.0

    def _ok(v) -> bool:
        try:
            return bool(np.isfinite(float(v)))
        except (TypeError, ValueError):
            return False
    return sum(_ok(r.get("_prev_close")) for r in have) / len(have)


def snapshot_status_path(d: date) -> Path:
    return root() / "snapshot" / f"{_iso(d)}.json"


def read_snapshot_status(d: date) -> Optional[dict]:
    try:
        return json.loads(snapshot_status_path(d).read_text(encoding="utf-8"))
    except Exception:
        return None


def _write_snapshot_status(d: date, **rec) -> None:
    """What the day's first check found — read by the email digest: a pre-open
    snapshot that was missing or defective (and whether the rebuild fixed it)."""
    p = snapshot_status_path(d)
    p.parent.mkdir(parents=True, exist_ok=True)
    old = read_snapshot_status(d) or {}
    old.update(rec, day=_iso(d), at=datetime.now(timezone.utc).isoformat(timespec="seconds"))
    p.write_text(json.dumps(old, default=str), encoding="utf-8")


def ensure_snapshot(d: date, names: Sequence[str]) -> float:
    """Build the session snapshot of ``d`` when it is missing or DEFECTIVE
    (coverage below `SNAPSHOT_MIN_COVERAGE`) — the scorer's features must be the
    training rows' features. Returns the coverage it ends with. Records what it
    found (`snapshot_status_path`); a rebuild that RAISES is recorded and
    returns the old coverage — the caller keeps the model arm off that bar, the
    vol arm (no snapshot needed) still runs."""
    from src.analysis import deep_features as dfe
    cov = snapshot_coverage(d, names)
    st = read_snapshot_status(d)
    if st is None:
        _write_snapshot_status(d, found=round(cov, 3), exists=dfe.snapshot_path(dnum(d)).exists(),
                               coverage=round(cov, 3), rebuilt=False)
    if cov >= SNAPSHOT_MIN_COVERAGE:
        return cov
    exists = dfe.snapshot_path(dnum(d)).exists()
    logger.warning(f"[sel_short] session snapshot {d} {'covers only %.0f%% of the universe' % (100 * cov) if exists else 'missing'}"
                   f" — building it (~3 min)")
    try:
        dfe.build_session_snapshot(dnum(d), workers=6)
    except Exception as e:                                     # noqa: BLE001
        logger.error(f"[sel_short] session snapshot {d} rebuild FAILED ({type(e).__name__}: {e})")
        _write_snapshot_status(d, coverage=round(cov, 3), rebuilt=False,
                               error=f"{type(e).__name__}: {e}"[:300])
        return cov
    cov = snapshot_coverage(d, names)
    _write_snapshot_status(d, coverage=round(cov, 3), rebuilt=True, error=None)
    if cov < SNAPSHOT_MIN_COVERAGE:
        logger.warning(f"[sel_short] session snapshot {d} still covers {100 * cov:.0f}% — the 30-minute store "
                       f"is behind the previous session for most names")
    return cov


def _backfill_one(args):
    """One name's rows for the backfill days, built exactly as the evaluation
    arrays were (`ml30.ticker_rows` with the whole-market slices, as `ml30.build`
    passes them: base + legs on the full series, deep features from the store's
    point-in-time snapshots)."""
    tk, since_dn, days, thr, slices = args[:5]
    v2 = bool(args[5]) if len(args) > 5 else False
    deep = bool(args[6]) if len(args) > 6 else True
    min_vis = int(args[7]) if len(args) > 7 and args[7] else 0
    from loguru import logger as _lg
    _lg.remove()
    from src.analysis import ml30
    try:
        r = ml30.ticker_rows(tk, float(thr), deep=deep, slices=slices, rows="fml", eval_since=int(since_dn))
    except Exception:
        return None
    if not r or "eval" not in r:
        return None
    e = r["eval"]
    keep = np.isin(e["dn"], np.asarray(days, np.int64))
    # The eval rows are the series' TAIL from `since_dn` (ml30): row k is bar number
    # len(series) - len(rows) + k + 1.
    from src.analysis import ml_dataset as md
    h = md.hlc_30m(tk)
    n_all = len(h[0]) if h is not None else 0
    if min_vis:
        # the live scorer computes no feature before `min_vis` bars are visible (NO_DATA): keep the
        # rows it would have scored
        keep &= (n_all - len(e["dn"]) + np.arange(len(e["dn"])) + 1) >= min_vis
    if not keep.any():
        return None
    out = {"tk": tk, "dn": e["dn"][keep], "bar": e["bar"][keep], "px": e["px"][keep],
           "dv20": e["dv20"][keep], "X": e["X"][keep],
           "D": e["D"][keep] if "D" in e else np.zeros((int(keep.sum()), 0), np.float32),
           "ca_gap": _ca_gap_tail(h, len(e["dn"]))[keep]}
    if v2:
        try:
            out["E"] = _v2_extra_rows(tk, out["dn"], out["bar"])
        except Exception:                                      # noqa: BLE001 — the name drops out, as a failed row
            return None
    return out


def _ca_gap_tail(hlc, n_rows: int) -> np.ndarray:
    """`corporate_gap_flags` of the LAST ``n_rows`` bars of an `hlc_30m` tuple (the backfill's eval
    rows), computed on the whole series as the live scorer's window computes it. An incomplete series
    flags nothing: a broken flag never drops a name."""
    if hlc is None or not len(hlc[0]) or any(x is None for x in hlc[1:4]):
        return np.zeros(n_rows, dtype=bool)
    try:
        idx = pd.DatetimeIndex(hlc[0])
        sd = np.asarray((idx.tz_localize("UTC").tz_convert(ET).normalize().tz_localize(None) - EPOCH).days, np.int64)
        fl = corporate_gap_flags(sd, hlc[1], hlc[2], hlc[3])
    except Exception:                                          # noqa: BLE001 — no flag, the name stays
        return np.zeros(n_rows, dtype=bool)
    if len(fl) >= n_rows:
        return fl[len(fl) - n_rows:]
    return np.r_[np.zeros(n_rows - len(fl), dtype=bool), fl]


def _v2_extra_rows(tk: str, dn: np.ndarray, bar: np.ndarray) -> np.ndarray:
    """The V2 per-bar and per-session inputs (`sel_v2.E_BAR` + `E_SESS`) of one name's rows (session day,
    bar POSITION in the session — `ml30.ticker_rows`' eval rows), from its whole series."""
    from src.analysis import deep_features as dfe
    from src.analysis import ml_dataset as md
    from src.signals import sel_v2
    E = np.full((len(dn), len(sel_v2.E_BAR) + len(sel_v2.E_SESS)), np.nan, np.float32)
    hlc = md.hlc_30m(tk)
    if hlc is None or not len(dn):
        return E
    n = len(hlc[0])
    fs = md.ticker_feature_frame(tk, hlc=hlc)
    atr = fs["atr_pct_14"].to_numpy(float) if fs is not None and len(fs) == n and "atr_pct_14" in fs.columns \
        else np.full(n, np.nan)
    eb = sel_v2.bar_extras(hlc, atr)
    sday = dfe.session_days(pd.DatetimeIndex(hlc[0]))
    first = np.r_[0, np.flatnonzero(np.diff(sday) != 0) + 1]
    uniq = np.unique(sday)
    k = np.searchsorted(uniq, dn)
    on = (k < len(uniq)) & (uniq[np.minimum(k, len(uniq) - 1)] == dn)
    bi = np.where(on, first[np.minimum(k, len(first) - 1)] + np.asarray(bar, np.int64), -1)
    okb = on & (bi >= 0) & (bi < n)
    E[okb, :len(sel_v2.E_BAR)] = eb[bi[okb]]
    ud = np.unique(dn)
    E[:, len(sel_v2.E_BAR):] = sel_v2.session_extras(tk, ud)[np.searchsorted(ud, dn)]
    return E


def _v2_backfill_scores(parts: List[dict], booster, meta: dict) -> pd.DataFrame:
    """A V2 model's scores for the backfill's tradeable rows: the vectors in the model's order, the
    cross-sectional inputs per bar (session x bar position) over every name's rows, the mask, one predict."""
    from src.analysis import deep_features as dfe
    from src.analysis import ml30
    from src.signals import sel_v2
    base, deepf = ml30.base_features(), list(dfe.DEEP_FEATURES)
    ecols = sel_v2.E_BAR + sel_v2.E_SESS
    feats = list(meta["features"])
    X = np.vstack([p["X"] for p in parts])
    D = np.vstack([p["D"] for p in parts])
    E = np.vstack([p["E"] for p in parts])
    dn = np.concatenate([p["dn"] for p in parts]).astype(np.int64)
    bar = np.concatenate([p["bar"] for p in parts]).astype(np.int64)
    tks = np.concatenate([np.full(len(p["dn"]), p["tk"], dtype=object) for p in parts])
    M = np.full((len(dn), len(feats)), np.nan, np.float64)
    for i, f in enumerate(feats):
        if f in base:
            M[:, i] = X[:, base.index(f)]
        elif f in ecols:
            M[:, i] = E[:, ecols.index(f)]
        elif f in deepf:
            M[:, i] = D[:, deepf.index(f)]
    sic = sel_v2.sic2_map()
    df = pd.DataFrame({"run": dn * 100 + bar, "tradeable": True})
    for s in sel_v2.XS_SOURCES:
        df[s] = E[:, ecols.index(s)].astype(float) if s in ecols else X[:, base.index(s)].astype(float)
    df["sic2"] = [sic.get(t, -1) for t in tks]
    xs = sel_v2.cross_section(df)
    pos = {f: i for i, f in enumerate(feats)}
    for c in sel_v2.CROSS:
        if c in pos:
            M[:, pos[c]] = xs[c].to_numpy(dtype=float)
    M = sel_v2.apply_mask(M, feats, meta.get("masked"))
    s = booster.predict(M.astype(np.float32), num_iteration=int(meta.get("num_iteration") or NUM_ITERATION))
    return pd.DataFrame({"dn": dn, "bar": bar.astype(int), "ticker": tks, "score": s.astype(float)})


def backfill_days(days: Sequence[date], tickers: Optional[Sequence[str]] = None, workers: int = 6,
                  arms: Sequence[str] = ARMS, merge: bool = False, existing_only: bool = False,
                  min_visible_bars: int = 0) -> dict:
    """Score every bar of past sessions ``days`` for the model's names and write
    the score files of ``arms`` (overwriting) — the evaluation arrays' own
    construction, so the history is what the arrays would hold (the vol arm's is
    the arrays' own ATR% column). The deep store must hold the days.
    ``merge`` adds the names' rows to each day file and keeps every other row;
    ``existing_only`` (with ``merge``) touches only day files that exist, so a
    missing day stays missing instead of becoming a file of a few names;
    ``min_visible_bars`` keeps only bars the live scorer would have scored (its
    400-bar floor — the vol arm's added stocks). Without the model arm the deep
    features are skipped (the vol and ETF histories read the ATR% alone)."""
    from src.analysis import deep_features as dfe
    from src.analysis import ml30
    from src.signals import sel_v2
    booster, meta = load_model()
    v2 = sel_v2.is_v2(meta) and "model" in arms
    deep = "model" in arms
    if tickers is None and "thin" in arms:          # the thin stocks: the model's names + the added stocks
        tickers = list(dict.fromkeys(list(meta["tickers"]) + vol_listing_names()))
    tickers = list(tickers or meta["tickers"])
    dns = sorted(dnum(d) for d in days)
    base, deepf = ml30.base_features(), list(dfe.DEEP_FEATURES)
    cols = (None if (v2 or "model" not in arms)       # the model's columns only when its arm is backfilled
            else [("X", base.index(f)) if f in base else ("D", deepf.index(f)) for f in meta["features"]])
    jv = base.index(str(settings.sel_short_vol_feature))
    out_rows, vol_rows, thin_rows, v2_parts = [], [], [], []
    t0 = time.time()
    # the whole-market families (insider, 13F, fails, dividends, listing day):
    # without them those features are missing on every row (2026-09-26)
    tables = dfe.MarketTables(tickers) if deep else None
    jobs = [(tk, dns[0], dns, meta.get("thr") or 1.0, tables.slices(tk) if tables is not None else {}, v2, deep,
             int(min_visible_bars or 0)) for tk in tickers]
    with ProcessPoolExecutor(max(1, workers)) as ex:
        for r in ex.map(_backfill_one, jobs, chunksize=8):
            if not r:
                continue
            ok = (r["px"] >= float(settings.sel_short_min_price)) & \
                 (np.nan_to_num(r["dv20"]) >= float(settings.sel_short_min_dollar_volume))
            if "model" in arms and v2:
                # V2's cross-sectional inputs need every name of a bar: score after the loop
                if ok.any():
                    v2_parts.append({"tk": r["tk"], **{k: r[k][ok] for k in ("dn", "bar", "X", "D", "E")}})
            elif "model" in arms:
                M = np.column_stack([r[src][:, j] for src, j in cols]).astype(np.float32)
                s = booster.predict(M, num_iteration=int(meta.get("num_iteration") or NUM_ITERATION))
                out_rows.append(pd.DataFrame({"dn": r["dn"][ok], "bar": r["bar"][ok].astype(int),
                                              "ticker": r["tk"], "score": s[ok]}))
            if "vol" in arms or "etf" in arms:
                v = np.asarray(r["X"][:, jv], float)
                k = ok & np.isfinite(v) & ~np.asarray(r.get("ca_gap", np.zeros(len(v), bool)), dtype=bool)
                vol_rows.append(pd.DataFrame({"dn": r["dn"][k], "bar": r["bar"][k].astype(int),
                                              "ticker": r["tk"], "score": v[k]}))
            if "thin" in arms:                       # the thin stocks' history: the band below the vol floor
                v = np.asarray(r["X"][:, jv], float)
                dvb = np.nan_to_num(r["dv20"])
                k = ((r["px"] >= float(settings.sel_short_min_price))
                     & (dvb >= float(settings.sel_short_thin_min_dollar_volume))
                     & (dvb < float(settings.sel_short_min_dollar_volume))
                     & np.isfinite(v) & ~np.asarray(r.get("ca_gap", np.zeros(len(v), bool)), dtype=bool))
                thin_rows.append(pd.DataFrame({"dn": r["dn"][k], "bar": r["bar"][k].astype(int),
                                               "ticker": r["tk"], "score": v[k]}))
    if v2 and v2_parts:
        out_rows.append(_v2_backfill_scores(v2_parts, booster, meta))
    res = {}
    etf_set = (set(etf_names(tickers, list(meta.get("tickers") or [])))
               if ("etf" in arms or "thin" in arms) else set())
    for arm, parts in (("model", out_rows), ("vol", vol_rows), ("vol2", vol_rows), ("etf", vol_rows),
                       ("thin", thin_rows)):
        if arm not in arms:
            continue
        df = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(columns=["dn", "bar", "ticker", "score"])
        if arm == "etf":                                   # the ETF arm's history: the products' ATR% only
            df = df[df["ticker"].isin(etf_set)]
        elif arm == "thin":                                # the thin stocks: common stocks only, never a product
            df = df[~df["ticker"].isin(etf_set)]
        elif arm in ("vol", "vol2") and meta.get("tickers"):   # the vol arm's: the model's names + its added stocks
            df = df[df["ticker"].isin(set(meta["tickers"]) | set(vol_listing_names()))]
        for dn in dns:
            g = df[df["dn"] == dn]
            d = (EPOCH + pd.Timedelta(days=int(dn))).date()
            new = g[["bar", "ticker", "score"]].reset_index(drop=True)
            p = scores_path(d, arm)
            if merge and existing_only and not p.exists():
                continue                                   # a missing day stays missing (prepare warns of it)
            if merge and p.exists():
                # ADD these names' rows to the day's history; every other name's rows stay
                # (a backfill for newly added names — the ETF arm's products, 2026-10-02)
                old = pd.read_pickle(p)
                key = set(zip(new["bar"].astype(int), new["ticker"]))
                old = old[[k not in key for k in zip(old["bar"].astype(int), old["ticker"])]]
                new = pd.concat([old, new], ignore_index=True)
            _write_pickle(new, p)
            res[f"{arm}:{_iso(d)}"] = int(len(g))
    logger.info(f"[sel_short] backfilled {res} in {time.time() - t0:.0f}s")
    return res


# ── one run ──────────────────────────────────────────────────────────────────

def score_bar(d: date, bar_of_day: int, uni: Dict[str, float], fetch: bool = True) -> pd.DataFrame:
    """Every universe name's score at bar ``bar_of_day`` of ``d`` (process pool).
    ``fetch=False`` scores from the stores alone (a past session already in the
    deep store)."""
    tickers = sorted(uni)
    n = max(1, int(settings.sel_short_workers))
    chunks = [tickers[i::n] for i in range(n)]
    args = [(c, _iso(d), int(bar_of_day), {t: uni[t] for t in c}) for c in chunks if c]
    rows: List[dict] = []
    fn = _score_chunk if fetch else _score_chunk_nofetch
    with ProcessPoolExecutor(n) as ex:
        for part in ex.map(fn, args):
            rows.extend(part)
    return score_v2(pd.DataFrame(rows))


def score_v2(res: pd.DataFrame) -> pd.DataFrame:
    """A V2 model's scores for one bar's rows (`_score_chunk`'s ``_vec`` / ``_x_*``): the cross-sectional
    inputs over the bar's TRADEABLE rows (status OK, price >= `sel_short_min_price`, 20-session dollar
    volume >= `sel_short_min_dollar_volume` — the training rows' rule), the masked inputs blanked, one
    predict. Returns ``res`` without the private columns; a v1 model's rows pass through unchanged."""
    from src.signals import sel_v2
    private = [c for c in res.columns if str(c).startswith("_")]
    if "_vec" not in res.columns:
        return res
    booster, meta = load_model()
    feats = list(meta["features"])
    ok = (res["status"] == "OK") & res["_vec"].map(lambda v: isinstance(v, np.ndarray))
    if ok.any():
        sub = res[ok]
        df = pd.DataFrame({"run": 0, "tradeable": (sub["px"] >= float(settings.sel_short_min_price))
                           & (pd.to_numeric(sub["dv20"], errors="coerce").fillna(0.0)
                              >= float(settings.sel_short_min_dollar_volume))}, index=sub.index)
        for s in sel_v2.XS_SOURCES:
            df[s] = pd.to_numeric(sub[f"_x_{s}"], errors="coerce")
        sic = sel_v2.sic2_map()
        df["sic2"] = [sic.get(t, -1) for t in sub["ticker"]]
        xs = sel_v2.cross_section(df)
        M = np.vstack(sub["_vec"].to_list()).astype(np.float64)
        pos = {f: i for i, f in enumerate(feats)}
        for c in sel_v2.CROSS:
            if c in pos:
                M[:, pos[c]] = xs[c].to_numpy(dtype=float)
        M = sel_v2.apply_mask(M, feats, meta.get("masked"))
        pred = booster.predict(M.astype(np.float32), num_iteration=int(meta.get("num_iteration") or NUM_ITERATION))
        res = res.copy()
        res.loc[sub.index, "score"] = pred.astype(float)
    return res.drop(columns=private)


_NO_FETCH = False


def _score_chunk_nofetch(args) -> List[dict]:
    """`_score_chunk` for a past session: its bars are already in the deep store."""
    import src.signals.sel_short as me
    me._NO_FETCH = True
    return me._score_chunk(args)


def _finite_or_none(v) -> Optional[float]:
    try:
        v = float(v)
    except (TypeError, ValueError):
        return None
    return v if np.isfinite(v) else None


def crowded(days_to_cover: Optional[float]) -> bool:
    """The short-interest filter (user directive 2026-09-27: "Add in live
    production the 'Under one day of volume' filter"): a short is crowded when its
    short interest exceeds `sel_short_max_days_to_cover` days of volume (FINRA's
    days to cover, floored at 1.00 — so the default keeps only names whose short
    interest is under one day of volume). Unknown is not crowded, as evaluated
    (scratchpad `gross_net_dtc.py`: a NaN passes)."""
    if not getattr(settings, "enable_sel_short_dtc_filter", False) or days_to_cover is None:
        return False
    return float(days_to_cover) > float(settings.sel_short_max_days_to_cover) + 1e-9


def low_rvol(rvol: Optional[float], arm: str) -> bool:
    """The VOL arm's relative-volume filter (user directive 2026-10-01: "implement
    relative volume filter to prod"): a vol pick whose bar traded less than
    `sel_short_vol_min_rvol` x the name's usual 30-minute volume (`rvol_at`) is not
    shorted. Unknown passes; the model arm is never filtered (its picks journal it)."""
    if arm not in VOL_RULE_ARMS or rvol is None or not getattr(settings, "enable_sel_short_vol_rvol_filter", False):
        return False
    return float(rvol) < float(settings.sel_short_vol_min_rvol)


def _ssr_journal(top: pd.Series) -> dict:
    """The pick row's short-sale-restriction fields (`ssr_state`) as plain JSON values."""
    v = top.get("ssr") if "ssr" in top.index else None
    out = {"ssr": bool(v) if v is not None and not (isinstance(v, float) and not np.isfinite(v)) else None}
    for k in ("ssr_prev_close", "ssr_day_low", "ssr_prev_low", "ssr_prev2_close"):
        out[k] = _finite_or_none(top.get(k)) if k in top.index else None
    return out


def give_back(arm: str) -> float:
    """The share of the 5-session run-up an arm's target gives back: the WHOLE run-up for the vol
    arm (`sel_short_vol_give_back` 1.0; user directive 2026-10-04 evening, "Vol arm only"), HALF
    for the model and ETF arms (`sel_short_give_back` 0.5; at 100% the model arm fell to 0.30 %/day
    and the ETF arm's growth did not change)."""
    return float(settings.sel_short_vol_give_back if arm in ("vol", "thin", "vol2")
                 else settings.sel_short_give_back)


def _is_fresh(score: float, tk: str, stand: pd.DataFrame) -> bool:
    """Fresh = fewer than `sel_short_own_min_history` prior scores, or above the name's own prior max."""
    st = stand.reindex([tk]).iloc[0] if len(stand) else pd.Series({"n_prior": 0.0, "prior_max": np.nan})
    n_prior = float(st.get("n_prior", np.nan))
    n_prior = n_prior if np.isfinite(n_prior) else 0.0
    pmax = float(st.get("prior_max", np.nan))
    return bool(n_prior < float(settings.sel_short_own_min_history) or (np.isfinite(pmax) and score > pmax))


def _vol_backups(ok: pd.DataFrame, stand: pd.DataFrame, earlier: Sequence[dict], g: float, d: date,
                 bar_of_day: int, arm: str = "vol") -> dict:
    """PREREG16's same-bar fallback (`enable_sel_short_vol_fallback`): the bar's top-N names by ATR%
    (N = `sel_short_vol_fallback_ranks`, 5 from 2026-10-07) that are FRESH are journaled on every vol record
    (``fresh_pool``; ``top3_fresh`` in the records journaled at depth 3); ranks 2..N that pass the whole rule are the pick's ``backups``, in rank order,
    each with its own target and deadline. As in the evaluation, a candidate counts only on its FIRST
    fresh top-N bar of the day — judged before every other filter, against the earlier records' fresh
    top-N names (and, for records journaled before them, their fresh top pick). The entry step tries
    them only when IBKR cannot lend the pick."""
    n = max(1, int(settings.sel_short_vol_fallback_ranks))
    seen = set()
    for e in earlier:
        seen.update(e.get("fresh_pool") or e.get("top3_fresh") or [])
        if e.get("fresh") and e.get("ticker"):
            seen.add(str(e["ticker"]))
    order = ok.sort_values(["score", "ticker"], ascending=[False, True], kind="mergesort").head(n)
    fresh_names: List[str] = []
    backups: List[dict] = []
    hold = int(settings.sel_short_max_hold_sessions)
    for rank, (_, row) in enumerate(order.iterrows(), start=1):
        tk = str(row["ticker"])
        fresh = _is_fresh(float(row["score"]), tk, stand)
        if fresh:
            fresh_names.append(tk)
        if rank == 1 or not fresh or tk in seen:
            continue
        px, pre5 = float(row["px"]), float(row["pre5"])
        runup = (px / pre5 - 1.0) * 100.0 if np.isfinite(pre5) and pre5 > 0 else float("nan")
        if not (np.isfinite(runup) and runup > 0):
            continue
        dtc = _finite_or_none(row["dtc"]) if "dtc" in row.index else None
        rvol = _finite_or_none(row["rvol"]) if "rvol" in row.index else None
        if arm not in UNFILTERED_ARMS and (crowded(dtc) or low_rvol(rvol, "vol")):
            continue
        b = {"ticker": tk, "rank": rank, "score": float(row["score"]), "px": px, "pre5": pre5, "runup_pct": runup,
             "target": px - g * (px - pre5),
             "deadline": bar_end_et(session_after(d, hold), bar_of_day).isoformat(),
             "days_to_cover": dtc, "rvol": rvol,
             "atr_pct": _finite_or_none(row["vol"]) if "vol" in row.index else None,
             "dv20": _finite_or_none(row["dv20"]) if "dv20" in row.index else None,
             "pre5_stale": bool(row["pre5_stale"]) if "pre5_stale" in row.index and pd.notna(row["pre5_stale"])
             else False}
        b.update(_ssr_journal(row))
        backups.append(b)
    return {"fresh_pool": fresh_names, "backups": backups}


def select(res: pd.DataFrame, d: date, bar_of_day: int, stand: pd.DataFrame,
           earlier: Sequence[dict], arm: str = "model") -> dict:
    """The rule on one run's scores: tradeable rows (price and dollar-volume
    floors), a cross-section of at least `sel_short_min_run_rows`, the top score
    (ties: ticker ascending), fresh (no standing, or above its prior max), the
    name's first fresh top pick of the day, the run-up filter, then the
    short-interest filter (`crowded`: journaled with its target and deadline so
    the untraded pick can be followed, never handed to the ledger) and, for the
    vol arm, the relative-volume filter (`low_rvol`, the same way). A crowded or
    low-volume pick is still the bar's pick — it counts as the name's fresh pick
    of the day, as in the evaluation, where the filters acted at the trade step.

    ``arm="model"`` ranks the model's ``score`` (rows with the full feature
    vector); ``arm="vol"`` ranks the ``vol`` column (every row that reached the
    bar — the ATR% needs no deep snapshot), with ``stand`` from the vol history
    and ``earlier`` holding that arm's own decisions."""
    min_px = float(settings.sel_short_min_price)
    min_dv = float(settings.sel_short_min_dollar_volume)
    # both arms' views of every row, kept for the journal's cross-arm scores
    nan = pd.Series(np.nan, index=res.index)
    model_s = res["score"] if "score" in res.columns else nan
    vol_s = res["vol"] if "vol" in res.columns else nan
    floors = (res["px"] >= min_px) & (res["dv20"] >= min_dv)
    model_ok = floors & (res["status"] == "OK") & np.isfinite(model_s)
    vol_ok = floors & np.isfinite(vol_s)
    if arm == "model":
        base_ok = (res["status"] == "OK") & np.isfinite(res["score"])
    else:
        base_ok = np.isfinite(vol_s)
        res = res.assign(score=vol_s)
    if arm == "thin":                          # the thin stocks: between the thin floor and the vol floor
        band = (res["dv20"] >= float(settings.sel_short_thin_min_dollar_volume)) & (res["dv20"] < min_dv)
    else:
        band = res["dv20"] >= min_dv
    ok = res[base_ok & (res["px"] >= min_px) & band]
    rec = {"day": _iso(d), "bar_of_day": int(bar_of_day), "bar_end": bar_end_et(d, bar_of_day).isoformat(),
           "arm": arm, "n_scored": int(len(ok)), "n_universe": int(len(res)), "decision": "none"}
    if len(ok) < int(settings.sel_short_min_run_rows):
        rec["decision"] = "thin_run"
        return rec
    earlier = [e for e in earlier if (e.get("arm") or "model") == arm]
    top = ok.sort_values(["score", "ticker"], ascending=[False, True], kind="mergesort").iloc[0]
    tk = str(top["ticker"])
    st = stand.reindex([tk]).iloc[0] if len(stand) else pd.Series({"n_prior": 0.0, "prior_max": np.nan})
    n_prior = float(st.get("n_prior", np.nan))
    n_prior = n_prior if np.isfinite(n_prior) else 0.0          # never scored = no standing = fresh
    pmax = float(st.get("prior_max", np.nan))
    fresh = bool(n_prior < float(settings.sel_short_own_min_history) or (np.isfinite(pmax) and top["score"] > pmax))
    first = not any(e.get("ticker") == tk and e.get("fresh") for e in earlier)
    px, pre5 = float(top["px"]), float(top["pre5"])
    stale = bool(top["pre5_stale"]) if "pre5_stale" in top.index and pd.notna(top["pre5_stale"]) else False
    runup = (px / pre5 - 1.0) * 100.0 if np.isfinite(pre5) and pre5 > 0 else float("nan")
    rose = bool(np.isfinite(runup) and runup > 0)
    g = give_back(arm)
    dtc = _finite_or_none(top["dtc"]) if "dtc" in top.index else None
    rvol = _finite_or_none(top["rvol"]) if "rvol" in top.index else None
    rec.update({"ticker": tk, "score": float(top["score"]), "px": px, "pre5": pre5,
                "runup_pct": runup, "n_prior": n_prior, "prior_max": pmax,
                "fresh": fresh, "first_today": first, "rose": rose, "pre5_stale": stale, "days_to_cover": dtc,
                "rvol": rvol,
                # the pick bar's 30-min ATR% (both arms; the vol arm's score): the
                # volatility-normalised exit's reference
                "atr_pct": _finite_or_none(top["vol"]) if "vol" in top.index else None})
    rec.update(_ssr_journal(top))                              # journal-only: the short-sale restriction
    rec.update(_pick_scores(ok, top, n_prior, pmax, arm, model_s[model_ok], vol_s[vol_ok],
                            res.loc[model_ok, "ticker"], res.loc[vol_ok, "ticker"]))
    if arm in ("vol", "vol2"):
        rec["confidence"] = vol_confidence(rec)                # journal-only: nothing decides on it
    if arm in ("vol", "thin", "vol2") and getattr(settings, "enable_sel_short_vol_fallback", False):
        rec.update(_vol_backups(ok, stand, earlier, g, d, bar_of_day, arm))
    if not fresh:
        rec["decision"] = "not_fresh"
    elif not first:
        rec["decision"] = "not_first_today"
    elif not rose:
        rec["decision"] = "not_a_riser"
    else:
        rec["target"] = px - g * (px - pre5)
        dl = session_after(d, int(settings.sel_short_max_hold_sessions))
        rec["deadline"] = bar_end_et(dl, bar_of_day).isoformat()
        filtered = arm not in UNFILTERED_ARMS                 # vol2: no crowding, no relative-volume filter
        rec["decision"] = "crowded" if (filtered and crowded(dtc)) else "short"
        if rec["decision"] == "short" and filtered and low_rvol(rvol, arm):
            rec["decision"] = "low_rvol"
        if (rec["decision"] == "short" and filtered and dtc is None
                and getattr(settings, "enable_sel_short_dtc_filter", False)):
            logger.warning(f"[sel_short] {arm} pick {tk}: days to cover unknown — the short-interest filter "
                           f"passes it")
    return rec


# ── the VOL arm's confidence — JOURNAL-ONLY ──
# A frozen rule fitted on the 2025 backtest trades of the vol arm (later bar, lower ATR%,
# smaller freshness margin, smaller margin over the runner-up — each a percentile on its
# 2025 ruler). Its bottom-third FILTER (live 2026-09-29) was removed 2026-09-30 (user:
# "Remove the confidence filter in live"): on 2021-24, years no design choice saw, the
# trades it removed earned MORE per day than those it kept (live with the filter minus
# without it -0.32 %/day, 95% -0.66..+0.05; 2021-26 2.07 vs 2.07 %/day on 236 vs 333
# trades). The score is still journaled on every vol pick (and stamped `sel_confidence`)
# for a blind read on live picks; nothing decides on it. `sel_short_conf_vol_v1.json`
# holds the rule; memory/sel-short-confidence-test-2026-09.md and
# memory/vol-arm-anatomy-2026-09.md the evidence.
_CONF_VOL_PATH = Path(__file__).with_name("sel_short_conf_vol_v1.json")
_CONF_VOL: Optional[dict] = None


def conf_rule_vol() -> Optional[dict]:
    """The frozen vol-arm rule, or None when it cannot be read (picks then journal no confidence)."""
    global _CONF_VOL
    if _CONF_VOL is None:
        try:
            _CONF_VOL = json.loads(_CONF_VOL_PATH.read_text(encoding="utf-8"))
        except Exception as e:                                 # noqa: BLE001
            logger.error(f"[sel_short] vol confidence rule unreadable ({e}) — picks journal no confidence")
            _CONF_VOL = {}
    return _CONF_VOL or None


def vol_confidence(rec: dict) -> Optional[float]:
    """The mean, over the inputs the record carries, of each input's percentile on its
    2025 ruler (1 - percentile for a negative sign) — exactly the evaluation's
    construction. None when the rule or every input is missing."""
    rule = conf_rule_vol()
    if not rule:
        return None
    vals = []
    for inp in rule["inputs"]:
        v = rec.get(inp["field"])
        try:
            v = float(v)
        except (TypeError, ValueError):
            continue
        if not np.isfinite(v):
            continue
        ruler = inp["ruler"]
        p = bisect.bisect_right(ruler, v) / len(ruler)
        vals.append(p if inp["sign"] > 0 else 1.0 - p)
    return float(np.mean(vals)) if vals else None


def _pick_scores(ok: pd.DataFrame, top: pd.Series, n_prior: float, pmax: float, arm: str,
                 model_s: pd.Series, vol_s: pd.Series, model_tk: pd.Series, vol_tk: pd.Series) -> dict:
    """Candidate CONFIDENCE components of the bar's pick, journaled on every decision
    record (user 2026-09-29: "Add a few different scores to the pick journal") so a
    confidence score can be tested BLIND on live picks. Nothing decides on them.
    The pre-registered 2025 -> 2026 test of an entry-time composite failed (a higher
    score picked better TYPICAL trades but held the squeezes — see the memory note
    sel-short-confidence-test-2026-09).

      run_mean / run_std   the arm's scores over the bar's tradeable cross-section
      z_in_run             (score - run_mean) / run_std
      runner_up(_score)    the bar's second name and its score; gap2_z = the margin / run_std
      own_margin_z         (score - its own prior max) / run_std — the freshness margin;
                           None under `sel_short_own_min_history` priors (fresh by default)
      other_arm_score      the OTHER arm's view of the same name (a model pick's ATR%, a
      other_arm_rank / _n  vol pick's model score) and its rank there (1 = that arm's top)
      dv20                 the pick's 20-session mean regular-hours dollar volume"""
    s = pd.to_numeric(ok["score"], errors="coerce")
    sd = float(s.std()) if len(s) > 1 else float("nan")
    sd = sd if np.isfinite(sd) and sd > 0 else float("nan")
    order = ok.assign(_s=s).sort_values(["_s", "ticker"], ascending=[False, True], kind="mergesort")
    score = float(top["score"])
    out = {"run_mean": _finite_or_none(float(s.mean())), "run_std": _finite_or_none(sd),
           "z_in_run": _finite_or_none((score - float(s.mean())) / sd)}
    if len(order) > 1:
        second = order.iloc[1]
        out.update({"runner_up": str(second["ticker"]), "runner_up_score": _finite_or_none(float(second["_s"])),
                    "gap2_z": _finite_or_none((score - float(second["_s"])) / sd)})
    has_standing = n_prior >= float(settings.sel_short_own_min_history) and np.isfinite(pmax)
    out["own_margin_z"] = _finite_or_none((score - pmax) / sd) if has_standing else None
    other_s, other_tk = (vol_s, vol_tk) if arm == "model" else (model_s, model_tk)
    tk = str(top["ticker"])
    mine = other_s[other_tk == tk]
    if len(mine) and np.isfinite(float(mine.iloc[0])):
        v = float(mine.iloc[0])
        out.update({"other_arm_score": v, "other_arm_rank": int((other_s > v).sum()) + 1,
                    "other_arm_n": int(len(other_s))})
    else:
        out.update({"other_arm_score": None, "other_arm_rank": None, "other_arm_n": int(len(other_s))})
    out["dv20"] = _finite_or_none(float(top["dv20"])) if "dv20" in top.index else None
    return out


def run(d: date, bar_of_day: int) -> dict:
    t0 = time.time()
    uni = load_universe(d)
    if uni is None:
        # never the ~40-minute store extension inside a run: the scorer fetches the
        # store's gap itself, and the next day's prepare extends through it
        prepare(d, extend=False)
        uni = load_universe(d) or {}
    # A split that took effect after the store was last adjusted (normally caught
    # by the pre-open run): reset those histories BEFORE scoring on them, and
    # rebuild the day's snapshot, whose anchor prices were read off the old scale.
    thin = (load_thin_universe(d) or {}) if "thin" in live_arms() else {}
    try:
        from src.data.intraday_store import reset_split_tickers
        resets = reset_split_tickers(sorted(set(uni) | set(thin)), d)
        if any(v == "reset" for v in resets.values()):
            from src.analysis import deep_features as dfe
            logger.warning(f"[sel_short] {d} bar {bar_of_day}: split reset {sorted(resets)} — rebuilding the snapshot")
            dfe.build_session_snapshot(dnum(d), workers=6)
    except Exception as e:                                     # noqa: BLE001
        logger.error(f"[sel_short] {d} bar {bar_of_day}: split check failed ({e})")
    try:
        cov = ensure_snapshot(d, model_universe(uni))
    except Exception as e:                                     # noqa: BLE001
        logger.error(f"[sel_short] {d} bar {bar_of_day}: snapshot check failed ({e}) — model arm off this bar")
        cov = 0.0
    try:                                       # a V2 model's per-session inputs (normally the prepare's)
        from src.signals import sel_v2
        _, meta = load_model()
        if sel_v2.is_v2(meta):
            n_ext = ensure_day_extras(d, model_universe(uni, meta))
            if n_ext:
                logger.info(f"[sel_short] {d} bar {bar_of_day}: V2 session inputs computed for {n_ext} names")
    except Exception as e:                                     # noqa: BLE001 — the workers read them per name
        logger.error(f"[sel_short] {d} bar {bar_of_day}: V2 session inputs failed ({e}) — read per name")
    # let the bar's last trades reach Polygon's aggregate before reading it
    settle = bar_end_et(d, bar_of_day) + timedelta(seconds=float(settings.sel_short_bar_settle_seconds))
    lag = (settle - datetime.now(timezone.utc)).total_seconds()
    if 0 < lag < 120:
        time.sleep(lag)
    res = score_bar(d, bar_of_day, {**thin, **uni}, fetch=True)
    if len(res) and "ticker" in res.columns:
        # the thin stocks' rows (by name, from the day's thin universe): the "thin" arm alone ranks them
        res["thin"] = res["ticker"].isin(set(thin) - set(uni))
    if cov < SNAPSHOT_MIN_COVERAGE and len(res):
        # still DEFECTIVE after the rebuild (or no snapshot): the model would
        # score on missing deep features, trade on them, and push those scores
        # into its freshness history — its rows are taken out of this bar; the
        # vol arm (ATR%, no snapshot needed) is unaffected.
        ok = res["status"] == "OK"
        res.loc[ok, "status"] = "SNAPSHOT_DEFECTIVE"
        res.loc[ok, "score"] = float("nan")
    return decide(res, d, bar_of_day, t0, extra={"snapshot_coverage": round(float(cov), 3)})


def arm_rows(res: pd.DataFrame, arm: str, ca_gaps: bool = False) -> pd.DataFrame:
    """The rows ``arm`` ranks, by NAME: the model arm the model's own names; the vol
    arm those plus its added stocks (`vol_listing_names`); the ETF arm the
    exchange-traded products (`etf_names`), the ones already in the model's universe
    included. Never by status: a bar without the session snapshot leaves the added
    products at NO_SNAPSHOT instead of VOL_ONLY, which once let them into the vol arm.
    The vol and ETF arms (ranked on ATR%) leave out a name whose bar is a corporate-action
    gap (``ca_gap``, `corporate_gap_flags`); ``ca_gaps`` returns those left-out rows instead."""
    try:
        _, meta = load_model()
        model_names = list(meta.get("tickers") or [])
    except Exception:                                          # noqa: BLE001
        model_names = []
    # the thin stocks' rows (flagged by `run` from the day's thin universe, by name) belong to the "thin"
    # arm ALONE: the other arms never rank them nor take them into their histories
    thin_rows = res["thin"].fillna(False).astype(bool) if "thin" in res.columns else None
    if arm == "thin":
        out = res[thin_rows] if thin_rows is not None else res.iloc[0:0]
    else:
        if thin_rows is not None:
            res = res[~thin_rows]
        if arm == "etf":
            names = set(etf_names(list(res["ticker"]), model_names))
            out = res[res["ticker"].isin(names)]
        elif not model_names or "ticker" not in res.columns:
            out = res                                      # no name list: every name is the model's
        else:
            keep = set(model_names) | (set(vol_listing_names()) if arm in ("vol", "vol2") else set())
            out = res[res["ticker"].isin(keep)]
    if arm in ("vol", "etf", "thin", "vol2") and "ca_gap" in out.columns:
        gapped = out["ca_gap"].eq(True)
        return out[gapped] if ca_gaps else out[~gapped]
    return out.iloc[0:0] if ca_gaps else out


def _underlying_journal(ticker: str, d: date) -> dict:
    """Journal-only, ETF arm: a single-stock fund's underlying and its FINRA days to
    cover in the session snapshot (nothing decides on them — a blind read)."""
    u = etf_underlying(ticker)
    if not u:
        return {}
    out = {"underlying": u.get("under"), "underlying_side": u.get("side"), "underlying_dtc": None}
    try:
        from src.analysis import deep_features as dfe
        row = (dfe.load_session_snapshot(dnum(d)) or {}).get(str(u.get("under")))
        if row is not None:
            out["underlying_dtc"] = _finite_or_none(row.get(DTC_FEATURE))
    except Exception:                                          # noqa: BLE001
        pass
    return out


def decide(res: pd.DataFrame, d: date, bar_of_day: int, t0: Optional[float] = None,
           extra: Optional[dict] = None) -> dict:
    """One run's decisions from its scores: each arm's scores appended to its
    history, its standing read, its rule applied, its decision journaled. The
    run file (the run-done marker) holds the model's record with the other
    arms' under ``arms``."""
    t0 = time.time() if t0 is None else t0
    min_px = float(settings.sel_short_min_price)
    earlier = [e for e in read_picks(d) if int(e.get("bar_of_day", -1)) < bar_of_day]
    arms = live_arms()
    recs = {}
    for arm in arms:
        sub = arm_rows(res, arm)
        if arm == "model":
            ok = sub[(sub["status"] == "OK") & np.isfinite(sub["score"]) & (sub["px"] >= min_px)]
            sc = ok["score"]
        else:
            vol = sub["vol"] if "vol" in sub.columns else pd.Series(np.nan, index=sub.index)
            ok = sub[np.isfinite(vol) & (sub["px"] >= min_px)]
            sc = vol[ok.index]
        append_scores(d, pd.DataFrame({"bar": bar_of_day, "ticker": ok["ticker"], "score": sc}), arm)
        stand = standing(d, list(ok["ticker"]), arm)
        rec = select(sub, d, bar_of_day, stand, earlier, arm=arm)
        if arm == "etf" and rec.get("ticker"):
            rec.update(_underlying_journal(rec["ticker"], d))
        gx = arm_rows(res, arm, ca_gaps=True)
        if len(gx):                                    # journal-only: who the corporate-action gap rule left out
            gx = gx.sort_values("vol", ascending=False).head(3)
            rec["ca_gap_excluded"] = [{"ticker": str(t), "vol": _finite_or_none(float(v)),
                                       "gap_pct": _finite_or_none(float(g)) if g is not None else None}
                                      for t, v, g in zip(gx["ticker"], gx["vol"], gx["ca_gap_pct"])]
        rec["seconds"] = round(time.time() - t0, 1)
        rec["created_at"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
        recs[arm] = rec
    recs["model"]["status_counts"] = res["status"].value_counts().to_dict() if len(res) else {}
    # the short-interest filter's input must reach the run: without it every pick
    # passes unjudged, which looks like normal operation
    n_dtc = int(np.isfinite(pd.to_numeric(res["dtc"], errors="coerce")).sum()) if "dtc" in res.columns else 0
    recs["model"]["n_days_to_cover"] = n_dtc
    recs["model"].update(extra or {})
    if getattr(settings, "enable_sel_short_dtc_filter", False) and len(res) and n_dtc == 0:
        logger.warning(f"[sel_short] {d} bar {bar_of_day}: no scored name carries days to cover — the "
                       f"short-interest filter cannot judge this run (session snapshot missing?)")
    # the relative-volume filter's input, the same way: none = every vol pick passes unjudged
    n_rvol = int(np.isfinite(pd.to_numeric(res["rvol"], errors="coerce")).sum()) if "rvol" in res.columns else 0
    recs["model"]["n_rvol"] = n_rvol
    if getattr(settings, "enable_sel_short_vol_rvol_filter", False) and len(res) and n_rvol == 0:
        logger.warning(f"[sel_short] {d} bar {bar_of_day}: no scored name carries a relative volume — the "
                       f"vol arm's relative-volume filter cannot judge this run")
    for arm in arms:
        _journal(recs[arm])
    run_rec = dict(recs["model"], arms={a: r for a, r in recs.items() if a != "model"})
    rp = root() / "runs" / f"{_iso(d)}_{bar_of_day:02d}.json"
    rp.parent.mkdir(parents=True, exist_ok=True)
    rp.write_text(json.dumps(run_rec, default=str), encoding="utf-8")
    logger.info(f"[sel_short] {d} bar {bar_of_day}: " + " | ".join(
        f"{a} {r.get('decision')} {r.get('ticker', '')}"
        + (f" (days to cover {r['days_to_cover']:.2f})" if r.get("decision") == "crowded" else "")
        + (f" (relative volume {r['rvol']:.2f})" if r.get("decision") == "low_rvol" else "")
        for a, r in recs.items())
        + f" (scored {recs['model']['n_scored']}/{recs['model']['n_universe']}, {recs['model']['seconds']}s)")
    return run_rec


def run_done(d: date, bar_of_day: int) -> bool:
    return (root() / "runs" / f"{_iso(d)}_{bar_of_day:02d}.json").exists()


RUN_MAX_FAILURES = 2                   # a bar whose run crashed this often is skipped


def _failure_path(d: date, bar_of_day: int) -> Path:
    return root() / "runs" / f"{_iso(d)}_{bar_of_day:02d}.failed.json"


def run_failures(d: date, bar_of_day: int) -> int:
    try:
        return int(json.loads(_failure_path(d, bar_of_day).read_text(encoding="utf-8")).get("n", 0))
    except Exception:
        return 0


def note_failure(d: date, bar_of_day: int, err: BaseException) -> None:
    p = _failure_path(d, bar_of_day)
    p.parent.mkdir(parents=True, exist_ok=True)
    rec = {"n": run_failures(d, bar_of_day) + 1, "error": f"{type(err).__name__}: {err}"[:500],
           "at": datetime.now(timezone.utc).isoformat(timespec="seconds")}
    p.write_text(json.dumps(rec), encoding="utf-8")


def pending_bars(now: datetime) -> List[Tuple[date, int]]:
    """Today's completed regular-hours bars with no run yet, OLDEST first. Every
    one of them is scored, not just the latest: a tick that ran past the next
    bar used to leave the bar in between unscored — its pick lost, and a hole in
    both freshness histories (ticks took 20-45 min after 2026-09-25). Oldest
    first, so each bar's first-pick-of-the-day rule sees the bars before it. A
    bar whose run failed `RUN_MAX_FAILURES` times is skipped."""
    lb = latest_bar(now)
    if lb is None:
        return []
    d, last = lb
    return [(d, b) for b in range(last + 1)
            if not run_done(d, b) and run_failures(d, b) < RUN_MAX_FAILURES]


def prepare_done(d: date) -> bool:
    return (root() / "prepare" / f"{_iso(d)}.json").exists()


# ── the tick's side of it ────────────────────────────────────────────────────

def launch(now: Optional[datetime] = None) -> Optional[dict]:
    """Start this tick's subprocess (non-blocking): ``--run`` for every completed
    bar of the day not yet run (`pending_bars`, oldest first), else ``--prepare``
    when due. None when there is nothing to do. The run prepares inline when the
    day was never prepared."""
    if not getattr(settings, "enable_sel_short", False):
        return None
    now = now or datetime.now(timezone.utc)
    et = now.astimezone(ET)
    if not is_session(et.date()) or not (root() / "model.json").exists():
        return None
    args = None
    todo = pending_bars(now) if et.hour < 17 else []
    if todo:
        args = ["--run", "--day", _iso(todo[0][0]), "--bars", ",".join(str(b) for _, b in todo)]
    elif not prepare_done(et.date()) and et.strftime("%H:%M") >= str(settings.sel_short_prepare_after_et):
        args = ["--prepare", "--day", _iso(et.date())]
    if args is None:
        return None
    prev = _ACTIVE.get("proc")
    if prev is not None and prev.poll() is None:
        logger.warning(f"[sel_short] previous subprocess (pid {prev.pid}) still running — not launching {args}")
        return None
    lock = root() / "busy.lock"
    try:
        busy = lock.exists() and time.time() - lock.stat().st_mtime < LOCK_STALE_SECONDS
    except OSError:
        busy = False
    if busy:
        logger.warning(f"[sel_short] another scorer holds {lock} — not launching {args}")
        return None
    root().mkdir(parents=True, exist_ok=True)
    log = open(Path("logs") / "sel_short.log", "a", encoding="utf-8")
    env = dict(os.environ, PYTHONIOENCODING="utf-8")
    proc = subprocess.Popen([sys.executable, "-m", "src.signals.sel_short", *args],
                            stdout=log, stderr=subprocess.STDOUT, env=env, cwd=os.getcwd())
    _ACTIVE["proc"] = proc
    logger.info(f"[sel_short] launched {' '.join(args)} (pid {proc.pid})")
    return {"proc": proc, "args": args, "started": time.time(), "log": log}


def wait(handle: Optional[dict], timeout: Optional[float] = None) -> Optional[int]:
    """Block until the tick's subprocess ends (bounded); its return code or None.
    Called twice a tick (the live path at the start, the end-of-tick pass): a
    run already waited out returns its code at once; a ``--prepare`` is never
    waited for — no pick comes from it, and it can run 20+ minutes."""
    if not handle:
        return None
    if handle.get("done"):
        return handle.get("rc")
    try:
        handle["log"].close()                     # our copy only: the child keeps its own handle
    except Exception:
        pass
    if (handle.get("args") or [""])[0] == "--prepare":
        return None
    timeout = float(settings.sel_short_wait_seconds if timeout is None else timeout)
    left = max(0.0, timeout - (time.time() - handle["started"]))
    try:
        rc = handle["proc"].wait(timeout=left)
    except subprocess.TimeoutExpired:
        logger.warning(f"[sel_short] {' '.join(handle['args'])} still running after {timeout:.0f}s — "
                       "its pick will be taken by the next tick if still fresh")
        rc = None
    if rc is None:
        return None                               # still running: a later wait may still see it end
    handle["done"], handle["rc"] = True, rc
    _note_launch(handle, rc)
    if rc != 0:
        logger.error(f"[sel_short] {' '.join(handle['args'])} exited {rc} (see logs/sel_short.log)")
    return rc


def _note_launch(handle: dict, rc: Optional[int]) -> None:
    """Journal each launch's outcome (read by `health` for the email digest)."""
    try:
        d = datetime.now(ET).date()
        p = root() / "launches" / f"{_iso(d)}.jsonl"
        p.parent.mkdir(parents=True, exist_ok=True)
        rec = {"at": datetime.now(timezone.utc).isoformat(timespec="seconds"), "args": handle.get("args"),
               "rc": rc, "seconds": round(time.time() - float(handle.get("started", time.time())), 1)}
        with open(p, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(rec) + "\n")
    except Exception as e:                                     # noqa: BLE001
        logger.debug(f"[sel_short] launch journal failed: {e}")


def health(now: Optional[datetime] = None) -> dict:
    """Today's scorer health for the email digest (user directive 2026-09-27:
    scorer failures and failed pre-open snapshots must not pass as a quiet day).
    ``problems`` lists, in plain words: completed bars still unscored 10 min
    after they ended, bars whose run failed, model bars too thin to rank
    (``thin_run`` — a Polygon outage reads this way), scorer exits != 0, and a
    pre-open session snapshot that was missing or defective (and what the
    rebuild did)."""
    now = now or datetime.now(timezone.utc)
    d = now.astimezone(ET).date()
    out: dict = {"day": _iso(d), "problems": [], "notes": [], "runs": 0, "bars_due": 0}
    if not getattr(settings, "enable_sel_short", False) or not is_session(d):
        return out
    lb = latest_bar(now)
    due = [] if lb is None else list(range(lb[1] + 1))
    out["bars_due"] = len(due)
    done = [b for b in due if run_done(d, b)]
    out["runs"] = len(done)
    late = [b for b in due if not run_done(d, b)
            and (now - bar_end_et(d, b)).total_seconds() > 600]
    failed = [b for b in due if not run_done(d, b) and run_failures(d, b) > 0]
    if late:
        out["problems"].append(f"{len(late)} completed bar(s) never scored: "
                               + ", ".join(bar_end_et(d, b).strftime("%H:%M") for b in late)
                               + (f" ({len(failed)} crashed — logs/sel_short.log)" if failed else ""))
    thin = []
    for b in done:
        try:
            rec = json.loads((root() / "runs" / f"{_iso(d)}_{b:02d}.json").read_text(encoding="utf-8"))
        except Exception:
            continue
        if rec.get("decision") == "thin_run":
            thin.append(bar_end_et(d, b).strftime("%H:%M"))
    if thin:
        out["problems"].append(f"{len(thin)} bar(s) too thin to rank (<{settings.sel_short_min_run_rows} "
                               f"names scored — data outage?): {', '.join(thin)}")
    try:
        lines = (root() / "launches" / f"{_iso(d)}.jsonl").read_text(encoding="utf-8").splitlines()
        bad = [json.loads(x) for x in lines if x.strip()]
        bad = [x for x in bad if x.get("rc") not in (0, None)]
    except Exception:
        bad = []
    if bad:
        out["problems"].append(f"scorer exited with an error {len(bad)} time(s) today "
                               f"(last: {' '.join(bad[-1].get('args') or [])} → {bad[-1].get('rc')})")
    snap = read_snapshot_status(d)
    et_hm = now.astimezone(ET).strftime("%H:%M")
    if snap is None and et_hm >= "09:35":
        from src.analysis import deep_features as dfe
        if not dfe.snapshot_path(dnum(d)).exists():
            out["problems"].append("pre-open snapshot MISSING (the 08:30 pre-open run did not build it) — "
                                   "the first run will try to build it")
        else:
            uni = load_universe(d) or {}
            cov = snapshot_coverage(d, sorted(uni)) if uni else 1.0
            if cov < SNAPSHOT_MIN_COVERAGE:
                out["problems"].append(f"pre-open snapshot DEFECTIVE: {100 * cov:.0f}% of the universe "
                                       "priced — the first run will rebuild it")
    elif snap is not None and float(snap.get("found") or 0.0) < SNAPSHOT_MIN_COVERAGE:
        what = ("missing" if not snap.get("exists") else f"defective ({100 * float(snap.get('found') or 0):.0f}% priced)")
        if snap.get("error"):
            out["problems"].append(f"pre-open snapshot was {what}: rebuild FAILED ({snap['error']}) — the model "
                                   "arm is off until it is fixed")
        elif float(snap.get("coverage") or 0.0) >= SNAPSHOT_MIN_COVERAGE:
            # resolved: still reported in the digest, without the alarm
            out["notes"].append(f"pre-open snapshot was {what}: rebuilt by the scorer "
                                f"({100 * float(snap['coverage']):.0f}%)")
        else:
            out["problems"].append(f"pre-open snapshot was {what}: still {100 * float(snap.get('coverage') or 0):.0f}% "
                                   "after the rebuild — the model arm is off (vol arm unaffected)")
    out["snapshot"] = snap
    return out


def _hold_lock() -> Path:
    """Write ``busy.lock`` and keep touching it while this process lives: a
    scorer that dies leaves a lock that goes stale in `LOCK_STALE_SECONDS`, and
    one that runs long (a prepare's store extension) never looks dead."""
    import threading
    lock = root() / "busy.lock"
    lock.parent.mkdir(parents=True, exist_ok=True)
    lock.write_text(str(os.getpid()), encoding="utf-8")

    def _beat():
        while True:
            time.sleep(LOCK_HEARTBEAT_SECONDS)
            try:
                if not lock.exists():
                    return
                os.utime(lock, None)
            except OSError:
                return

    threading.Thread(target=_beat, name="sel-short-lock", daemon=True).start()
    return lock


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="The selection-short strategy's scorer.")
    ap.add_argument("--install", action="store_true", help="install the research artifact + seed scores")
    ap.add_argument("--install-v2", default=None, metavar="BOOSTER",
                    help="install a V2 model (the v1 kept as model_v1.*, its history moved to scores_v1/)")
    ap.add_argument("--cut", default="2026-04-30", help="with --install-v2: the booster's training cut")
    ap.add_argument("--prepare", action="store_true")
    ap.add_argument("--run", action="store_true")
    ap.add_argument("--backfill", action="store_true", help="score every bar of --day into its score file")
    ap.add_argument("--day", default=None)
    ap.add_argument("--bar", type=int, default=None)
    ap.add_argument("--bars", default=None, help="with --run: several bars, comma-separated, scored in order")
    ap.add_argument("--until", default=None, help="with --backfill: the last session (default --day)")
    ap.add_argument("--arms", default=",".join(ARMS), help="with --backfill: the score histories to write")
    ap.add_argument("--tickers", default=None, help="with --backfill: these names only (comma-separated)")
    ap.add_argument("--merge", action="store_true",
                    help="with --backfill: add the names' rows to the day files instead of rewriting them")
    ap.add_argument("--seed-vol", action="store_true", help="seed the volatility arm's history from the arrays")
    ap.add_argument("--no-extend", action="store_true")
    ap.add_argument("--listings", action="store_true",
                    help="screen and add the vol arm's added stocks for --day (what the prepare does first)")
    ap.add_argument("--dry", action="store_true", help="with --listings: print the screen, add nothing")
    ap.add_argument("--thin-listings", action="store_true",
                    help="screen and add the thin stocks outside the store for --day (what the prepare does)")
    ap.add_argument("--trade-log", action="store_true",
                    help="write tradelog/<day>.jsonl for --day (..--until): picks, entry step, ledger, broker fills")
    a = ap.parse_args(argv)
    logger.remove()
    logger.add(sys.stderr, level="INFO")
    if a.install:
        install()
        return 0
    if a.install_v2:
        print(json.dumps(install_v2(Path(a.install_v2), cut=a.cut), default=str)[:400])
        return 0
    if a.seed_vol:
        print(json.dumps({"seeded_vol_days": seed_vol_scores()}))
        return 0
    d = date.fromisoformat(a.day) if a.day else datetime.now(ET).date()
    if a.trade_log:
        until = date.fromisoformat(a.until) if a.until else d
        days, p = [], d
        while p <= until:
            if is_session(p):
                days.append(p)
            p += timedelta(days=1)
        print(json.dumps(write_trade_logs(days)))
        return 0
    lock = _hold_lock()
    try:
        if a.listings:
            found = screen_listings(d)
            print(json.dumps({"found": len(found), "names": [r["ticker"] for r in found]}))
            if not a.dry:
                print(json.dumps(add_listings(d, found), default=str)[:2000])
        if a.thin_listings:
            found = screen_listings(d, band="thin")
            print(json.dumps({"found": len(found), "names": [r["ticker"] for r in found]}))
            if not a.dry:
                print(json.dumps(add_listings(d, found, arm="thin"), default=str)[:2000])
        if a.prepare:
            prepare(d, extend=not a.no_extend)
        if a.backfill:
            until = date.fromisoformat(a.until) if a.until else d
            days, p = [], d
            while p <= until:
                if is_session(p):
                    days.append(p)
                p += timedelta(days=1)
            arms = tuple(a_ for a_ in a.arms.split(",") if a_ in ARMS)
            tks = [t.strip().upper() for t in a.tickers.split(",") if t.strip()] if a.tickers else None
            if tks is not None and not a.merge:
                raise SystemExit("--tickers rewrites whole day files without --merge: every other name's "
                                 "history would be lost")
            print(json.dumps(backfill_days(days, tickers=tks, arms=arms, merge=a.merge)))
        if a.run:
            if a.bars:
                bars = [int(x) for x in str(a.bars).split(",") if x.strip()]
            elif a.bar is not None:
                bars = [a.bar]
            else:
                lb = latest_bar(datetime.now(timezone.utc))
                if lb is None:
                    print("no completed bar today")
                    return 0
                d, bars = lb[0], [lb[1]]
            rc = 0
            for bar in bars:
                # one bar's crash must not cost the bars after it
                try:
                    print(json.dumps(run(d, bar), default=str))
                except Exception as e:                          # noqa: BLE001
                    note_failure(d, bar, e)
                    logger.exception(f"[sel_short] {d} bar {bar}: run failed ({run_failures(d, bar)} "
                                     f"time(s)) — {e}")
                    rc = 1
            return rc
    finally:
        try:
            lock.unlink()
        except OSError:
            pass
    return 0


if __name__ == "__main__":
    sys.exit(main())
