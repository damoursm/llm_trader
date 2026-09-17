"""The DEEP 30-minute bar store — the training-side history of the pivot label.

Since 2026-09-16 the ONLY pivot label in the system is the next H/L pivot on
30-minute regular-hours bars (`analysis/pivot_target`). The tick cache
(`cache/ohlcv_30m/`) holds the newest `intraday_30m_max_bars` (260 sessions) per
name and only for names a tick touches, which is what the panel, the sim panels
and the exit dataset need. Training `ml_ohlcv` needs the label on every row of
`cache/ml/dataset_full.parquet` — ~3,400 tickers back to 2007 for the features,
so back to 2021 for the label (Polygon's 30-minute aggregates paginate that far)
— and that history lives HERE: one pickle per ticker, ``Open/High/Low/Close/
Volume`` on a naive-UTC ``DatetimeIndex`` of bar starts, RTH only (09:30–15:30
ET starts), single-source (Polygon, adjusted).

* ``load_deep_30m(tk)`` — the stored frame, or None.
* ``deep_series_30m(tk)`` — ``(idx, c, h, lo)`` for the label scan: the deep
  frame extended by whatever the TICK cache holds after its last bar (the tick
  cache is refreshed every tick for live names, the deep store only on demand),
  falling back to the tick cache alone for a name the store has never seen.
  NaN / non-positive bars are dropped.
* ``extend_deep_30m(tickers, ...)`` — fetch and append the bars each ticker is
  missing (from its last stored session to today; from ``DEEP_FROM`` for a new
  name), with a wall-clock budget and a worker pool. Idempotent; a killed run
  costs only its in-flight names. Never run beside an RTH tick — a few thousand
  Polygon calls share the tick's rate budget (`memory/refactor-rewalk-kill-loop`).

CLI: ``python -m src.data.intraday_store --extend [--workers 4] [--budget-seconds N]
[--min-age-days 3] [--tickers A,B]`` and ``--stats``.
"""
from __future__ import annotations

import os
import pickle
import time
from datetime import date, timedelta
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from loguru import logger

DEEP_DIR = Path("cache/ml/bars30m_deep")
DEEP_FROM = "2021-01-01"                 # Polygon 30-minute aggregates: measured back to at least 2021
_NY = "America/New_York"


def _path(tk: str) -> Path:
    return DEEP_DIR / f"{tk.upper()}.pkl"


def _naive_utc_index(df: pd.DataFrame) -> pd.DataFrame:
    idx = pd.DatetimeIndex(df.index)
    if idx.tz is not None:
        idx = idx.tz_convert("UTC").tz_localize(None)
    out = df.copy()
    out.index = idx
    return out.sort_index()


def _rth_only(df: pd.DataFrame) -> pd.DataFrame:
    """Keep bars starting 09:30..15:30 ET (DST-correct)."""
    if df.empty:
        return df
    et = pd.DatetimeIndex(df.index).tz_localize("UTC").tz_convert(_NY)
    mins = et.hour * 60 + et.minute
    return df[(mins >= 570) & (mins <= 930)]


def load_deep_30m(tk: str) -> Optional[pd.DataFrame]:
    p = _path(tk)
    if not p.exists():
        return None
    try:
        with open(p, "rb") as fh:
            df = pickle.load(fh)
        if df is None or df.empty or "Close" not in df.columns:
            return None
        return _naive_utc_index(df)
    except Exception as e:                                   # noqa: BLE001
        logger.debug(f"[intraday_store] {tk}: unreadable deep frame ({e})")
        return None


def save_deep_30m(tk: str, df: pd.DataFrame) -> None:
    DEEP_DIR.mkdir(parents=True, exist_ok=True)
    df = _rth_only(_naive_utc_index(df))
    df = df[~df.index.duplicated(keep="last")]
    tmp = _path(tk).with_suffix(".pkl.tmp")
    with open(tmp, "wb") as fh:
        pickle.dump(df, fh)
    os.replace(tmp, _path(tk))


def _to_arrays(df: pd.DataFrame):
    idx = pd.DatetimeIndex(df.index)
    c = pd.to_numeric(df["Close"], errors="coerce").to_numpy(dtype=float)
    h = pd.to_numeric(df["High"], errors="coerce").to_numpy(dtype=float) if "High" in df.columns else c
    lo = pd.to_numeric(df["Low"], errors="coerce").to_numpy(dtype=float) if "Low" in df.columns else c
    ok = np.isfinite(c) & np.isfinite(h) & np.isfinite(lo) & (c > 0)
    return idx[ok], c[ok], h[ok], lo[ok]


def deep_series_30m(tk: str, min_bars: int = 50):
    """``(idx, c, h, lo)`` over the deep store + the tick cache's newer tail, or
    None below ``min_bars``. The tick cache alone serves a name the deep store
    has never seen (a year of history — enough for the panel's labels, not for
    the deep training rows, which is why the store exists)."""
    from src.analysis.pivot_target import _series_30m
    deep = load_deep_30m(tk)
    tail = _series_30m(tk)
    if deep is None:
        return tail
    if tail is not None:
        t_idx, t_c, t_h, t_lo = tail
        last = deep.index.max()
        m = t_idx > last
        if m.any():
            extra = pd.DataFrame({"Close": t_c[m], "High": t_h[m], "Low": t_lo[m]}, index=t_idx[m])
            deep = pd.concat([deep[["Close", "High", "Low"]], extra]).sort_index()
            deep = deep[~deep.index.duplicated(keep="last")]
    idx, c, h, lo = _to_arrays(deep)
    if len(c) < min_bars:
        return None
    return idx, c, h, lo


def deep_last_session(tk: str) -> Optional[date]:
    df = load_deep_30m(tk)
    if df is None:
        return None
    return pd.Timestamp(df.index.max()).tz_localize("UTC").tz_convert(_NY).date()


def _fetch_range(tk: str, from_date: str, to_date: str) -> pd.DataFrame:
    from src.data.polygon_client import get_intraday_bars_range
    df = get_intraday_bars_range(tk, from_date, to_date)
    if df is None or df.empty:
        return pd.DataFrame()
    return _naive_utc_index(df)


def extend_deep_30m(tickers: Iterable[str], *, workers: int = 4, budget_seconds: float = 1800.0,
                    min_age_days: int = 3, today: Optional[date] = None) -> Dict[str, int]:
    """Append the missing bars for every ticker whose stored history ends more
    than ``min_age_days`` ago (or that has no store). Returns counters."""
    from concurrent.futures import ThreadPoolExecutor, as_completed
    today = today or date.today()
    todo: List[Tuple[str, str]] = []
    for tk in tickers:
        last = deep_last_session(tk)
        if last is None:
            todo.append((tk, DEEP_FROM))
        elif (today - last).days > min_age_days:
            todo.append((tk, (last + timedelta(days=1)).isoformat()))
    stats = dict(considered=0, fresh=0, extended=0, new=0, empty=0, failed=0, skipped_budget=0)
    stats["considered"] = len(todo)
    if not todo:
        return stats
    t0 = time.time()
    to_iso = today.isoformat()

    def _one(item):
        tk, frm = item
        try:
            fresh = _fetch_range(tk, frm, to_iso)
            if fresh.empty:
                return tk, "empty"
            old = load_deep_30m(tk)
            if old is None:
                save_deep_30m(tk, fresh)
                return tk, "new"
            merged = pd.concat([old, fresh[fresh.index > old.index.max()]]).sort_index()
            save_deep_30m(tk, merged)
            return tk, "extended"
        except Exception as e:                               # noqa: BLE001
            logger.debug(f"[intraday_store] {tk}: extend failed ({e})")
            return tk, "failed"

    with ThreadPoolExecutor(max_workers=max(1, workers)) as ex:
        pending = {}
        it = iter(todo)
        # bounded submission so a budget stop leaves nothing queued
        for item in it:
            if time.time() - t0 > budget_seconds:
                stats["skipped_budget"] += 1
                continue
            fut = ex.submit(_one, item)
            pending[fut] = item
            if len(pending) >= workers * 2:
                done = next(as_completed(list(pending)))
                _tk, res = done.result()
                stats[res] = stats.get(res, 0) + 1
                del pending[done]
        for fut in as_completed(list(pending)):
            _tk, res = fut.result()
            stats[res] = stats.get(res, 0) + 1
    logger.info(f"[intraday_store] extend: {stats} in {time.time() - t0:.0f}s")
    return stats


def stats() -> dict:
    files = sorted(DEEP_DIR.glob("*.pkl")) if DEEP_DIR.exists() else []
    ends: List[date] = []
    for p in files[:400]:
        d = deep_last_session(p.stem)
        if d:
            ends.append(d)
    return dict(tickers=len(files), sampled=len(ends),
                last_session_min=min(ends).isoformat() if ends else None,
                last_session_max=max(ends).isoformat() if ends else None,
                bytes=sum(p.stat().st_size for p in files))


if __name__ == "__main__":  # pragma: no cover
    import argparse
    import sys
    ap = argparse.ArgumentParser(description="Deep 30-minute bar store (training-side pivot label history)")
    ap.add_argument("--extend", action="store_true", help="fetch the bars each ticker is missing")
    ap.add_argument("--tickers", default="", help="comma list; default = the deep-dataset universe")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--budget-seconds", type=float, default=1800.0)
    ap.add_argument("--min-age-days", type=int, default=3)
    ap.add_argument("--stats", action="store_true")
    a = ap.parse_args()
    if a.stats:
        print(stats())
    if a.extend:
        if a.tickers:
            tks = [t.strip().upper() for t in a.tickers.split(",") if t.strip()]
        else:
            import duckdb
            tks = [r[0] for r in duckdb.connect().sql(
                "SELECT DISTINCT ticker FROM 'cache/ml/dataset_full.parquet' ORDER BY 1").fetchall()]
        print(extend_deep_30m(tks, workers=a.workers, budget_seconds=a.budget_seconds,
                              min_age_days=a.min_age_days))
    if not (a.stats or a.extend):
        ap.print_help(); sys.exit(1)
