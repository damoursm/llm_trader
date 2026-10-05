"""``ml_ohlcv`` on 30-MINUTE rows — the in-tree dataset builder and trainer.

The live ``ml_ohlcv`` artifact (installed 2026-09-19, ``feature_bars="30m"``)
was built by out-of-tree scripts in a session scratchpad (mk30.py /
thr_arrays.py / train_thr.py); the repo's own trainer
(``ml_model.train_and_persist_pivot``) builds DAILY rows and could not
reproduce it. This module is that recipe, in-tree, with the deep store's
point-in-time features (``deep_features``) as an option (2026-09-23).

Rows      the FIRST, MIDDLE and LAST regular-hours 30-minute bar of every session
          (``ml_dataset.hlc_30m``: the deep 30-minute store 2021→ + the tick
          cache's tail). The last bar is the session-close anchor; three per
          session keeps the dataset at ~12M rows while the model still sees the
          morning, midday and close geometry it is served on.
Features  ``ml_dataset.ticker_feature_frame`` on the 30-minute series (76 incl.
          the 6 cross-sectional ranks, NaN here AND at serving) + the 9 leg-state
          features on the 30-minute zigzag at the label threshold + — with
          ``deep`` — ``deep_features.DEEP_FEATURES`` (the 08:30 ET session
          snapshot and the bar features at the row's bar).
Label     the next H/L pivot at the threshold strictly after the bar,
          ``extreme / bar close - 1`` in %, with the CONFIRMING bar's session day
          for the training cut (a row is usable once its pivot has confirmed).
Fit       LightGBM regression on the within-day rank of the label, day-equal
          weights, 300 trees / lr 0.03 / 31 leaves / min_child 200 / l2 5 /
          feature_fraction 0.8, seed 0 (the live artifact's hyperparameters).

CLI::

    python -m src.analysis.ml30 --build [--no-deep] [--workers 8]   # arrays -> cache/ml/ml30/
    python -m src.analysis.ml30 --build --rows random3 --eval-since 2026-05-01 --dir cache/ml/sel30
                                               # + every bar since 05-01 -> cache/ml/sel30_eval/
    python -m src.analysis.ml30 --train [--cut 2026-06-01] [--no-deep] [--out PATH]
    python -m src.analysis.ml30 --install PATH                       # backup + swap the live artifact
"""
from __future__ import annotations

import gc
import json
import os
import pickle
import shutil
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd
from loguru import logger

OUT_DIR = Path("cache/ml/ml30")
HYPER = dict(n=300, learning_rate=0.03, num_leaves=31, min_child_samples=200, lambda_l2=5.0, feature_fraction=0.8)
MIN_BARS = 400                               # the serving floor (`ml_model._MIN_30M_BARS`)
EPOCH = pd.Timestamp("1970-01-01")


def base_features() -> List[str]:
    from src.analysis.ml_dataset import ALL_FEATURE_COLUMNS
    from src.analysis.pivot_target import LEG_FEATURES
    return list(ALL_FEATURE_COLUMNS) + list(LEG_FEATURES)


def _set_threshold(thr: float):
    """The leg features read `settings.pivot_min_move_pct`; set it for this
    PROCESS (the builder runs in worker processes, one threshold each)."""
    from config.settings import settings
    settings.pivot_min_move_pct = float(thr)


def session_positions(sday: np.ndarray):
    """(bar index, session day, position 0/1/2) of the first / middle / last
    bar of every session — EXACTLY the recipe that built the live artifact: a
    position per BAR, the last assignment winning, so a one-bar session is a
    single row at position 2 and a middle bar that is also the first or last
    is not a row of its own."""
    sday = np.asarray(sday)
    n = len(sday)
    first = np.r_[0, np.flatnonzero(np.diff(sday) != 0) + 1] if n else np.zeros(0, int)
    end = np.r_[first[1:], n] if n else np.zeros(0, int)
    pos = np.full(n, -1, np.int8)
    for a, b in zip(first, end):
        pos[a] = 0
        pos[b - 1] = 2
        mid = a + (b - a) // 2
        if pos[mid] == -1:
            pos[mid] = 1
    keep = np.flatnonzero(pos >= 0)
    return keep, sday[keep].astype(np.int64), pos[keep]


def random_positions(sday: np.ndarray, k: int = 3):
    """(bar index, session day, slot) of ``k`` bars per session at positions
    drawn per session DAY — seeded on the day and the session's bar count, so
    every ticker with a full session gets the SAME bar times (full
    cross-sections per run) while the dataset covers every time of day the
    model is served at, not the three the live recipe fixes."""
    sday = np.asarray(sday)
    n = len(sday)
    if not n:
        return np.zeros(0, np.int64), np.zeros(0, np.int64), np.zeros(0, np.int8)
    first = np.r_[0, np.flatnonzero(np.diff(sday) != 0) + 1]
    end = np.r_[first[1:], n]
    keep, slot = [], []
    for a, b in zip(first, end):
        nb = int(b - a)
        rng = np.random.default_rng(int(sday[a]) * 7919 + nb)
        pick = np.sort(rng.choice(nb, size=min(int(k), nb), replace=False))
        keep.extend((a + pick).tolist())
        slot.extend(range(len(pick)))
    keep = np.asarray(keep, np.int64)
    return keep, sday[keep].astype(np.int64), np.asarray(slot, np.int8)


def ticker_rows(tk: str, thr: float, deep: bool = True, slices: Optional[dict] = None,
                rows: str = "fml", eval_since: Optional[int] = None) -> Optional[dict]:
    """One ticker's rows: X (base + legs), D (deep, or absent), y, conf, ba, dn,
    pos, bar, px, dv20, liq, fsc, f1d, f5d. None when the ticker has fewer than
    MIN_BARS bars.

    ``rows`` — ``"fml"``: the first, middle and last bar of each session (the
    live artifact's recipe, the default); ``"random3"``: three bars per session
    at day-seeded positions (`random_positions`); ``"daily"``: ONE row per
    session at its FIRST 30-minute bar (entry at 10:00 ET, a daily model's
    first tick after the open) whose X is the DAILY feature frame + daily leg
    state of the PREVIOUS session (what a model served once a day, pre-market,
    knows) — labels, returns and deep features are the bar's, as in the other
    modes. ``eval_since`` (a session day number) adds ``out["eval"]``: the same
    fields for EVERY bar from that session on — the density the live scorer and
    the own-history rule see (not used with ``daily``).

    ``ba`` is the label's ``bars_ahead`` (bars after the row's bar through the
    pivot's extreme, -1 without a label) — what the selection objective divides
    the return by. ``bar`` is the bar's index within its session, ``px`` its
    close. ``dv20`` is the mean regular-hours dollar volume of the 20 sessions
    BEFORE the row's session (point-in-time, like Gate 4's completed bars) and
    ``liq`` = dv20 >= $5M. ``fsc`` / ``f1d`` / ``f5d`` are REALIZABLE forward
    returns in % from the bar's close: to the close of the first session that
    ends after it (this session's, or the next one's from a session's last bar),
    of the next session, and of the fifth session after — exits an order can
    actually take, the check against the pivot label's hindsight."""
    from src.analysis import ml_dataset as md
    from src.analysis.pivot_target import LEG_FEATURES, leg_feature_rows, _resolved_pivots, MAX_PIVOT_BARS_30M
    from src.analysis import deep_features as dfe
    hlc = md.hlc_30m(tk)
    if hlc is None:
        return None
    idx, high, low, close, vol = hlc
    n = len(idx)
    if n < MIN_BARS:
        return None
    c = close.to_numpy(float); h = high.to_numpy(float); lo = low.to_numpy(float); v = vol.to_numpy(float)
    sday = dfe.session_days(idx)
    _set_threshold(thr)

    def base_matrix(hlc_, cc, hh, ll, stamps):
        fs_ = md.ticker_feature_frame(tk, hlc=hlc_)
        if fs_ is None or fs_.empty:
            return None
        F_ = fs_.reindex(columns=md.ALL_FEATURE_COLUMNS).to_numpy(np.float32)
        L_ = np.full((len(cc), len(LEG_FEATURES)), np.nan, np.float32)
        for i, rec in enumerate(leg_feature_rows(cc, hh, ll, list(stamps))):
            for j, f in enumerate(LEG_FEATURES):
                val = rec.get(f)
                L_[i, j] = np.float32(val) if val is not None and val == val else np.nan
        return np.hstack([F_, L_])

    if rows == "daily":
        from src.analysis.predictability import _hlc_by_session
        dh = _hlc_by_session(tk)
        if dh is None or len(dh[0]) < 60:
            return None
        XB = base_matrix(dh, dh[3].to_numpy(float), dh[1].to_numpy(float), dh[2].to_numpy(float), dh[0])
        dkeys = np.asarray([(pd.Timestamp(d) - EPOCH).days for d in dh[0]], np.int64)
    else:
        XB = base_matrix(hlc, c, h, lo, idx)
    if XB is None:
        return None
    P, PP, _FL, CF = _resolved_pivots(c, h, lo, thr)
    y = np.full(n, np.nan); conf = np.full(n, -1, np.int64); ba = np.full(n, -1, np.int32)
    if len(P):
        k = np.searchsorted(P, np.arange(n), side="right"); okk = k < len(P)
        ii = np.flatnonzero(okk); kk = k[ii]
        within = (P[kk] - ii) <= MAX_PIVOT_BARS_30M
        ii, kk = ii[within], kk[within]
        y[ii] = (PP[kk] / c[ii] - 1.0) * 100.0
        conf[ii] = sday[CF[kk]]
        ba[ii] = P[kk] - ii                      # = the live label's bars_ahead
    # sessions: first/last bar, ordinal per bar, point-in-time liquidity
    first = np.r_[0, np.flatnonzero(np.diff(sday) != 0) + 1]
    last = np.r_[first[1:] - 1, n - 1]
    ns = len(first)
    sess_of_bar = np.searchsorted(np.unique(sday), sday)
    sdv = np.add.reduceat(c * v, first)
    dv20 = pd.Series(sdv).rolling(20, min_periods=10).mean().shift(1).to_numpy()
    close_s = c[last]
    is_last = np.zeros(n, bool)
    is_last[last] = True

    def fwd(k_arr):
        tgt = sess_of_bar + k_arr
        out = np.full(n, np.nan, np.float32)
        ok = tgt < ns
        out[ok] = (close_s[tgt[ok]] / c[ok] - 1.0) * 100.0
        return out

    fsc = fwd(is_last.astype(np.int64))
    f1d = fwd(np.ones(n, np.int64))
    f5d = fwd(np.full(n, 5, np.int64))
    rth = dfe.RTH(idx, c, v) if deep else None
    snap = dfe.ticker_snapshots(tk, rth=rth, slices=slices) if deep else None

    def block(bi, dn, pos, X=None):
        so = sess_of_bar[bi]
        blk = dict(dn=dn, pos=pos, bar=(bi - first[so]).astype(np.int8), X=XB[bi] if X is None else X,
                   y=y[bi].astype(np.float32), conf=conf[bi], ba=ba[bi], px=c[bi].astype(np.float32),
                   dv20=dv20[so].astype(np.float32), liq=np.nan_to_num(dv20[so]) >= 5e6,
                   fsc=fsc[bi], f1d=f1d[bi], f5d=f5d[bi])
        if deep:
            si = np.searchsorted(rth.sessions, dn)
            srows = snap.iloc[si]
            bar_in_sess = bi - rth.first[si]
            D = np.full((len(bi), len(dfe.DEEP_FEATURES)), np.nan, np.float32)
            ci = {f: i for i, f in enumerate(dfe.DEEP_FEATURES)}
            D[:, [ci[f] for f in dfe.SNAPSHOT_FEATURES]] = srows[dfe.SNAPSHOT_FEATURES].to_numpy(np.float32)
            for f, val in dfe.bar_features(srows, c[bi], bar_in_sess).items():
                D[:, ci[f]] = val
            blk["D"] = D
        return blk

    if rows == "daily":
        bi = first
        dn = sday[bi].astype(np.int64)
        prev = np.searchsorted(dkeys, dn, side="left") - 1          # the last daily bar BEFORE the session
        Xd = np.full((len(bi), XB.shape[1]), np.nan, np.float32)
        ok = prev >= 0
        Xd[ok] = XB[prev[ok]]
        return dict(tk=tk, **block(bi, dn, np.zeros(len(bi), np.int8), X=Xd))
    if rows == "fml":
        bi, dn, pos = session_positions(sday)
    elif rows == "random3":
        bi, dn, pos = random_positions(sday, 3)
    else:
        raise ValueError(f"rows must be fml, random3 or daily, got {rows!r}")
    out = dict(tk=tk, **block(bi, dn, pos))
    if eval_since is not None:
        ei = np.flatnonzero(sday >= int(eval_since))
        out["eval"] = block(ei, sday[ei].astype(np.int64), np.full(len(ei), -1, np.int8))
    return out


MAX_EXIT_BARS = 130                          # 10 sessions: the trailing exits' time stop


def trailing_exits(c: np.ndarray, h: np.ndarray, lo: np.ndarray, rows: np.ndarray, thr: float,
                   side: str, max_bars: int = MAX_EXIT_BARS):
    """The REALIZABLE twin of the pivot label: enter at the close of bar ``i``
    and exit on a ``thr``% trailing stop from the best price since entry —
    exactly where the zigzag would CONFIRM the next pivot in the trade's favour
    — with a time stop at bar ``i + max_bars``'s close. The entry price counts
    as the first best, so a path that goes the wrong way first exits at about
    -``thr``%. A bar that gaps through the stop fills at its own high (long) /
    low (short) instead. Returns ``(ret_pct, bars_held)`` per row (NaN / -1
    where the series ends before either exit)."""
    rows = np.asarray(rows, np.int64)
    n, m = len(c), len(rows)
    ret = np.full(m, np.nan)
    held = np.full(m, -1, np.int32)
    if not m:
        return ret, held
    f = float(thr) / 100.0
    entry = c[rows]
    best = entry.copy()
    live = np.ones(m, bool)
    for k in range(1, int(max_bars) + 1):
        j = rows + k
        alive = live & (j < n)
        if not alive.any():
            break
        a = np.flatnonzero(alive)
        ja = j[a]
        if side == "long":
            stop = best[a] * (1.0 - f)
            hit = lo[ja] <= stop
            px = np.minimum(stop, h[ja])
            best[a] = np.where(hit, best[a], np.maximum(best[a], h[ja]))
        else:
            stop = best[a] * (1.0 + f)
            hit = h[ja] >= stop
            px = np.maximum(stop, lo[ja])
            best[a] = np.where(hit, best[a], np.minimum(best[a], lo[ja]))
        done = a[hit]
        ret[done] = (px[hit] / entry[done] - 1.0) * 100.0
        held[done] = k
        live[done] = False
        if k == max_bars:                                            # time stop at the close
            t = a[~hit]
            ret[t] = (c[rows[t] + k] / entry[t] - 1.0) * 100.0
            held[t] = k
            live[t] = False
    return ret, held


def _exit_work(args):
    tk, dn, bar, thr = args
    try:
        import multiprocessing as _mp
        if _mp.current_process().name != "MainProcess":
            from loguru import logger as _lg
            _lg.remove()
        from src.analysis import ml_dataset as md
        from src.analysis import deep_features as dfe
        hlc = md.hlc_30m(tk)
        if hlc is None:
            return tk, None
        idx, high, low, close, _vol = hlc
        c = close.to_numpy(float); h = high.to_numpy(float); lo = low.to_numpy(float)
        sday = dfe.session_days(idx)
        days, first = np.unique(sday, return_index=True)
        si = np.searchsorted(days, dn)
        ok = (si < len(days)) & (days[np.minimum(si, len(days) - 1)] == dn)
        i = np.where(ok, first[np.minimum(si, len(days) - 1)] + bar, -1)
        ok &= (i >= 0) & (i < len(c))
        ok[ok] &= sday[i[ok]] == dn[ok]
        out = {}
        for side in ("long", "short"):
            r = np.full(len(dn), np.nan); b = np.full(len(dn), -1, np.int32)
            rr, bb = trailing_exits(c, h, lo, i[ok], thr, side)
            r[ok], b[ok] = rr, bb
            out[side] = (r, b)
        return tk, out
    except Exception as e:                                            # noqa: BLE001
        return tk, repr(e)


def add_exit_labels(d: Path, thr: Optional[float] = None, workers: int = 6) -> dict:
    """Add the trailing-exit labels (`trailing_exits`) to an existing array
    directory without rebuilding its features: ``xl.npy`` / ``xlb.npy`` (long
    exit return % / bars held) and ``xs.npy`` / ``xsb.npy`` (short), aligned to
    its rows through (ticker, session day, bar in session)."""
    d = Path(d)
    meta = json.loads((d / "meta.json").read_text(encoding="utf-8"))
    thr = float(meta["thr"] if thr is None else thr)
    tk = np.load(d / "tk.npy"); dn = np.load(d / "dn.npy"); bar = np.load(d / "bar.npy").astype(np.int64)
    order = np.argsort(tk, kind="stable")
    bounds = np.flatnonzero(np.r_[True, tk[order][1:] != tk[order][:-1], True])
    groups = {int(tk[order[a]]): order[a:b] for a, b in zip(bounds[:-1], bounds[1:])}
    names = meta["tickers"]
    jobs = [(names[code], dn[g], bar[g], thr) for code, g in groups.items()]
    xl = np.full(len(tk), np.nan, np.float32); xlb = np.full(len(tk), -1, np.int32)
    xs = np.full(len(tk), np.nan, np.float32); xsb = np.full(len(tk), -1, np.int32)
    code_of = {t: i for i, t in enumerate(names)}
    errs = 0
    t0 = time.time()
    _lowprio()

    def take(res):
        nonlocal errs
        name, out = res
        if not isinstance(out, dict):
            errs += int(out is not None)
            return
        g = groups[code_of[name]]
        xl[g], xlb[g] = out["long"]
        xs[g], xsb[g] = out["short"]

    if int(workers) <= 1:
        for j in jobs:
            take(_exit_work(j))
    else:
        from multiprocessing import Pool
        with Pool(int(workers), initializer=_lowprio) as pool:
            for res in pool.imap_unordered(_exit_work, jobs, chunksize=4):
                take(res)
    for name, arr in (("xl", xl), ("xlb", xlb), ("xs", xs), ("xsb", xsb)):
        np.save(d / f"{name}.npy", arr)
    meta.setdefault("exit_labels", {})
    meta["exit_labels"] = dict(thr=thr, max_bars=MAX_EXIT_BARS, errors=errs,
                               added_at=datetime.now(timezone.utc).isoformat(timespec="seconds"))
    (d / "meta.json").write_text(json.dumps(meta), encoding="utf-8")
    logger.info(f"[ml30] exit labels for {len(tk):,} rows ({errs} errors) in {time.time() - t0:.0f}s -> {d}")
    return meta["exit_labels"]


def _lowprio():
    try:
        import ctypes
        k = ctypes.windll.kernel32
        k.GetCurrentProcess.restype = ctypes.c_void_p
        k.SetPriorityClass.argtypes = [ctypes.c_void_p, ctypes.c_uint32]
        k.SetPriorityClass(k.GetCurrentProcess(), 0x4000)          # BELOW_NORMAL
    except Exception:                                               # noqa: BLE001
        pass


def _work(args):
    tk, thr, deep, slices, rows, eval_since = args
    try:
        import multiprocessing as _mp
        if _mp.current_process().name != "MainProcess":           # a pool worker: silence it
            from loguru import logger as _lg
            _lg.remove()
        return ticker_rows(tk, thr, deep, slices, rows=rows, eval_since=eval_since)
    except Exception as e:                                          # noqa: BLE001
        return {"tk": tk, "err": repr(e)}


class _Sink:
    """Appends each ticker's rows to raw per-array files, so the parent never
    holds the whole dataset (the live stack leaves ~20 GB free); ``finish``
    turns them into .npy arrays in chunks."""

    SPEC = {"X": np.float32, "D": np.float32, "y": np.float32, "conf": np.int64, "ba": np.int32,
            "dn": np.int64, "pos": np.int8, "bar": np.int8, "px": np.float32, "dv20": np.float32,
            "liq": np.bool_, "tk": np.int32, "fsc": np.float32, "f1d": np.float32, "f5d": np.float32}

    def __init__(self, out_dir: Path, deep: bool, nb: int, nd: int):
        self.dir = Path(out_dir)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.names = [k for k in self.SPEC if deep or k != "D"]
        self.width = {"X": nb, "D": nd}
        self.fh = {k: open(self.dir / f"{k}.bin", "wb") for k in self.names}
        self.n = 0

    def add(self, blk: dict, code: int) -> None:
        m = len(blk["y"])
        if not m:
            return
        blk = dict(blk, tk=np.full(m, code, np.int32))
        for k in self.names:
            self.fh[k].write(np.ascontiguousarray(blk[k], dtype=self.SPEC[k]).tobytes())
        self.n += m

    def finish(self, step: int = 1_000_000) -> int:
        for f in self.fh.values():
            f.close()
        for k in self.names:
            shape = (self.n, self.width[k]) if k in self.width else (self.n,)
            per = shape[1] if len(shape) == 2 else 1
            dst = np.lib.format.open_memmap(self.dir / f"{k}.npy", mode="w+", dtype=self.SPEC[k], shape=shape)
            with open(self.dir / f"{k}.bin", "rb") as src:
                a = 0
                while a < self.n:
                    m = min(step, self.n - a)
                    dst[a:a + m] = np.fromfile(src, dtype=self.SPEC[k], count=m * per).reshape((m,) + shape[1:])
                    a += m
            dst.flush()
            del dst
            os.remove(self.dir / f"{k}.bin")
        return self.n


def _fill_market(out_dir: Path, step: int = 1_000_000) -> None:
    """Market context is one row per session day for everyone — filled in the
    parent after the per-ticker pass, in chunks."""
    from src.analysis import deep_features as dfe
    dn = np.load(out_dir / "dn.npy", mmap_mode="r")
    if not len(dn):
        return
    D = np.load(out_dir / "D.npy", mmap_mode="r+")
    mc = [dfe.DEEP_FEATURES.index(f) for f in dfe.MARKET_FEATURES]
    mk = dfe.market_context(np.unique(np.asarray(dn)))
    keys = mk.index.to_numpy(np.int64)
    vals = mk[dfe.MARKET_FEATURES].to_numpy(np.float32)
    for a in range(0, len(dn), step):
        k = np.searchsorted(keys, np.asarray(dn[a:a + step]))
        D[a:a + step, mc] = vals[k]
    D.flush()


def build(tickers: Optional[Sequence[str]] = None, thr: Optional[float] = None, deep: bool = True,
          workers: int = 8, out_dir: Path = OUT_DIR, rows: str = "fml",
          eval_since: Optional[str] = None, eval_dir: Optional[Path] = None) -> dict:
    """Build the arrays for every ticker (the deep universe by default) into
    ``out_dir``: X.npy (base+legs), D.npy (deep), y, conf, ba, dn, pos, bar, px,
    dv20, liq, tk, fsc, f1d, f5d + meta.json — ``rows`` picks the training bars
    (`ticker_rows`). With ``eval_since`` (ISO date) the same pass also writes
    EVERY bar from that session on into ``eval_dir`` (default
    ``<out_dir>_eval``). ``workers <= 1`` runs in-process."""
    from config.settings import settings
    from src.data import deep as deep_store
    from src.analysis import deep_features as dfe
    t0 = time.time()
    thr = float(settings.pivot_min_move_pct if thr is None else thr)
    tickers = sorted(tickers or deep_store.deep_universe())
    tables = dfe.MarketTables(tickers) if deep else None
    ev_day = int((pd.Timestamp(eval_since) - EPOCH).days) if eval_since else None
    jobs = [(tk, thr, deep, tables.slices(tk) if tables else None, rows, ev_day) for tk in tickers]
    nb, nd = len(base_features()), len(dfe.DEEP_FEATURES)
    out_dir = Path(out_dir)
    ev_dir = Path(eval_dir) if eval_dir else out_dir.with_name(out_dir.name + "_eval")
    sink = _Sink(out_dir, deep, nb, nd)
    ev_sink = _Sink(ev_dir, deep, nb, nd) if ev_day is not None else None
    code_of = {t: i for i, t in enumerate(tickers)}
    state = {"errs": 0, "done": 0}
    _lowprio()

    def take(r):
        state["done"] += 1
        if r is None:
            return
        if "err" in r:
            state["errs"] += 1
            if state["errs"] <= 5:
                logger.warning(f"[ml30] {r['tk']}: {r['err'][:200]}")
            return
        ev = r.pop("eval", None)
        sink.add(r, code_of[r["tk"]])
        if ev_sink is not None and ev is not None:
            ev_sink.add(ev, code_of[r["tk"]])
        if state["done"] % 250 == 0:
            logger.info(f"[ml30] build {state['done']}/{len(jobs)} | {sink.n:,} rows | {time.time() - t0:.0f}s")

    if int(workers) <= 1:
        for j in jobs:
            take(_work(j))
    else:
        from multiprocessing import Pool
        with Pool(int(workers), initializer=_lowprio) as pool:
            for r in pool.imap_unordered(_work, jobs, chunksize=2):
                take(r)
    metas = {}
    for sk, d, mode in ((sink, out_dir, rows), (ev_sink, ev_dir, "all_bars")):
        if sk is None:
            continue
        n = sk.finish()
        if deep:
            _fill_market(d)
        meta = dict(built_at=datetime.now(timezone.utc).isoformat(timespec="seconds"), n_rows=int(n),
                    rows=mode, eval_since=eval_since if mode == "all_bars" else None,
                    tickers=tickers, thr=thr, deep=bool(deep), base_features=base_features(),
                    deep_features=list(dfe.DEEP_FEATURES) if deep else [],
                    deep_cutoff_et=f"{dfe.CUTOFF_ET_MIN // 60:02d}:{dfe.CUTOFF_ET_MIN % 60:02d}",
                    lag_days=dict(dfe.LAG_DAYS), errors=state["errs"])
        (d / "meta.json").write_text(json.dumps(meta), encoding="utf-8")
        metas[mode] = meta
        logger.info(f"[ml30] built {n:,} {mode} rows from {len(code_of)} tickers ({state['errs']} errors) "
                    f"in {time.time() - t0:.0f}s -> {d}")
    return metas[rows]


def fit(X: np.ndarray, y: np.ndarray, day: np.ndarray, threads: int = 6):
    """The live recipe: regression on the within-day rank, day-equal weights."""
    import lightgbm as lgb
    from src.analysis.pivot_target import within_day_rank
    codes, day_idx = np.unique(day, return_inverse=True)
    yr = within_day_rank(y.astype(float), day_idx.astype(np.int32))
    cnt = np.bincount(day_idx); w = (1.0 / cnt[day_idx]).astype(np.float64); w /= w.mean()
    p = dict(objective="regression", learning_rate=HYPER["learning_rate"], num_leaves=HYPER["num_leaves"],
             min_child_samples=HYPER["min_child_samples"], lambda_l2=HYPER["lambda_l2"],
             feature_fraction=HYPER["feature_fraction"], seed=0, num_threads=int(threads), deterministic=True,
             force_col_wise=True, verbosity=-1)
    ds = lgb.Dataset(X, label=yr, weight=w, free_raw_data=True)
    return lgb.train(p, ds, num_boost_round=HYPER["n"]), p, len(codes)


def train(cut: Optional[str] = None, conf_cut: Optional[str] = None, deep: bool = True, threads: int = 6,
          in_dir: Path = OUT_DIR, out: Optional[Path] = None) -> dict:
    """Fit on the built arrays: rows whose session is <= ``cut`` and whose
    pivot CONFIRMED by ``conf_cut`` (default: everything confirmed). Writes
    the artifact to ``out`` (default in_dir/model.pkl) and returns it."""
    from src.analysis.ml_train import LightGBMRankRegressor
    from src.analysis import deep_features as dfe
    from src.signals.ml_model import TRAIN_CONFIG_PIVOT
    from src.analysis.pivot_target import pivot_basis
    _lowprio()
    t0 = time.time()
    in_dir = Path(in_dir)
    meta = json.loads((in_dir / "meta.json").read_text(encoding="utf-8"))
    if deep and not meta.get("deep"):
        raise SystemExit("arrays were built without --deep")
    y = np.load(in_dir / "y.npy"); conf = np.load(in_dir / "conf.npy"); dn = np.load(in_dir / "dn.npy")
    X = np.load(in_dir / "X.npy", mmap_mode="r")
    dnum = lambda s: int((pd.Timestamp(s) - EPOCH).days)                     # noqa: E731
    cut_d = dnum(cut) if cut else int(dn.max())
    ccut_d = dnum(conf_cut) if conf_cut else int(conf.max())
    rows = np.flatnonzero(np.isfinite(y) & (conf > 0) & (dn <= cut_d) & (conf <= ccut_d))
    feats = list(meta["base_features"]) + (list(meta["deep_features"]) if deep else [])
    M = np.empty((len(rows), len(feats)), np.float32)
    nb = len(meta["base_features"])
    D = np.load(in_dir / "D.npy", mmap_mode="r") if deep else None
    for a in range(0, len(rows), 500_000):
        r = rows[a:a + 500_000]
        M[a:a + len(r), :nb] = X[r]
        if deep:
            M[a:a + len(r), nb:] = D[r]
    booster, params, n_days = fit(M, y[rows], dn[rows], threads=threads)
    del M; gc.collect()
    m = LightGBMRankRegressor(num_threads=threads); m._booster = booster; m.params = dict(params)
    m.n_estimators = HYPER["n"]
    thr = float(meta["thr"])
    config = dict(TRAIN_CONFIG_PIVOT)
    config.update(pivot_basis=pivot_basis(), pivot_resolution="30m", pivot_label_basis="hl",
                  pivot_min_move_pct=thr, feature_bars="30m", leg_threshold_pct=thr, hyper=dict(HYPER),
                  num_threads=threads, label=f"next_30m_hl{thr:g}_pivot_from_30m_bar", arm="ml30",
                  deep_features=bool(deep), deep_cutoff_et=meta.get("deep_cutoff_et"),
                  deep_lag_days=meta.get("lag_days"))
    last_day = pd.Timestamp(np.datetime64(int(dn[rows].max()), "D")).date().isoformat()
    art = {"model": m, "features": feats, "config": config, "n_train": int(len(rows)), "n_days": int(n_days),
           "train_max_date": last_day, "trained_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
           "built_at": meta.get("built_at")}
    out = Path(out) if out else in_dir / ("model_deep.pkl" if deep else "model_base.pkl")
    tmp = out.with_suffix(".tmp")
    with open(tmp, "wb") as fh:
        pickle.dump(art, fh)
    os.replace(tmp, out)
    logger.info(f"[ml30] trained {'DEEP' if deep else 'BASE'} on {len(rows):,} rows / {n_days} days "
                f"(<= {last_day}) in {time.time() - t0:.0f}s -> {out}")
    return art


def install(path: Path) -> Path:
    """Swap ``path`` in as the live ml_ohlcv artifact, keeping the current one
    as a dated backup. Refuses an artifact whose pivot basis is not the one in
    force (serving would abstain on it anyway)."""
    from src.signals.ml_model import _MODEL_PATH
    from src.analysis.pivot_target import pivot_basis
    art = pickle.load(open(path, "rb"))
    if art.get("config", {}).get("pivot_basis") != pivot_basis():
        raise SystemExit(f"artifact basis {art.get('config', {}).get('pivot_basis')!r} != {pivot_basis()!r}")
    if _MODEL_PATH.exists():
        stamp = datetime.now().strftime("%Y-%m-%d_%H%M")
        bak = _MODEL_PATH.with_name(f"ml_ohlcv_model.bak.{stamp}.pkl")
        shutil.copy2(_MODEL_PATH, bak)
        logger.info(f"[ml30] previous artifact kept as {bak}")
    tmp = _MODEL_PATH.with_suffix(".tmp")
    shutil.copy2(path, tmp)
    os.replace(tmp, _MODEL_PATH)
    logger.info(f"[ml30] installed {path} as {_MODEL_PATH}")
    return _MODEL_PATH


def main(argv=None) -> None:
    import argparse
    ap = argparse.ArgumentParser(description="ml_ohlcv on 30-minute rows: build / train / install")
    ap.add_argument("--build", action="store_true")
    ap.add_argument("--train", action="store_true")
    ap.add_argument("--install", default="")
    ap.add_argument("--no-deep", action="store_true", help="the pre-2026-09-23 recipe, without the deep features")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--threads", type=int, default=6)
    ap.add_argument("--cut", default="", help="last training session (default: everything)")
    ap.add_argument("--conf-cut", default="", help="last pivot-confirmation day (default: everything)")
    ap.add_argument("--out", default="")
    ap.add_argument("--dir", default=str(OUT_DIR))
    ap.add_argument("--rows", default="fml", choices=("fml", "random3", "daily"),
                    help="first/middle/last (the live recipe), 3 day-seeded bars per session, or one "
                         "daily row per session (previous session's daily features, entry at 10:00)")
    ap.add_argument("--eval-since", default="", help="also write EVERY bar from this session on (ISO date)")
    ap.add_argument("--eval-dir", default="")
    a = ap.parse_args(argv)
    logger.add("logs/ml30.log", rotation="1 day", retention="30 days", level="INFO", enqueue=True)
    deep = not a.no_deep
    if a.build:
        build(deep=deep, workers=a.workers, out_dir=Path(a.dir), rows=a.rows,
              eval_since=a.eval_since or None, eval_dir=Path(a.eval_dir) if a.eval_dir else None)
    if a.train:
        train(cut=a.cut or None, conf_cut=a.conf_cut or None, deep=deep, threads=a.threads,
              in_dir=Path(a.dir), out=Path(a.out) if a.out else None)
    if a.install:
        install(Path(a.install))


if __name__ == "__main__":
    main()
