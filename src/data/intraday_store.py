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
  cache is refreshed every tick for live names; the deep store is extended through
  the previous session every market day — by the 08:30 ET pre-open run
  (`deep.refresh.extend_bars_30m`) and the selection short's `--prepare` — and on
  demand),
  falling back to the tick cache alone for a name the store has never seen.
  NaN / non-positive bars are dropped.
* ``extend_deep_30m(tickers, ...)`` — fetch and append the bars each ticker is
  missing (from its last stored session, re-read as an overlap, to today; from
  ``DEEP_FROM`` for a new name), with a wall-clock budget and a worker pool. A
  rescaled overlap only prompts the split-data check below. Idempotent; a killed run
  costs only its in-flight names. A full catch-up is a few thousand Polygon calls
  that share the tick's rate budget — never beside an RTH tick
  (`memory/refactor-rewalk-kill-loop`); the daily one-session extension (one call
  per name, before the open) is the scheduled exception. Writers use a per-process
  temp file, so the two daily extenders cannot interleave one pickle.
* ``reset_split_tickers(tickers)`` — SPLITS (user directives 2026-09-27/28): the
  deep store's split data decides. A name whose split took effect after its
  history was last adjusted (``_adjusted_asof.json``) has its WHOLE history
  refetched before any inference — by the pre-open run, the selection short's
  prepare and each of its runs. Never on a price change alone (the block comment
  above ``SPLIT_SCALE_TOLERANCE``).

Beyond `ml_ohlcv`'s labels, this store is the price grid of every deep-feature
session snapshot (`analysis/deep_features.RTH`) and the history the selection
short's series is rebuilt from.

CLI: ``python -m src.data.intraday_store --extend [--workers 4] [--budget-seconds N]
[--min-age-days 3] [--tickers A,B]`` and ``--stats``.
"""
from __future__ import annotations

import os
import pickle
import threading
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
    # a per-PROCESS temp name: the pre-open refresh and the selection short's
    # prepare can extend the same name at the same moment, and two writers
    # sharing one temp file can install an interleaved pickle
    tmp = _path(tk).with_suffix(f".pkl.{os.getpid()}.tmp")
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


# ── splits: reset a ticker's history before any inference on it ─────────────
# Polygon serves bars ADJUSTED as of the fetch day, and the store only ever
# APPENDED: after a split the stored pre-split bars sat on the old scale beside
# post-split ones (MGN 1-for-30, 2026-09-17: $0.18 -> $4.89 between two bars) —
# a fake run-up for the selection short's riser rule, a fake ATR% for its vol
# arm, garbage features for the model and the snapshots. User directives
# 2026-09-27/28: "When seeing a stock split in the data ingestion we should
# automatically reset the historical data before doing any kind of inference on
# this ticker" — and "verify from your split data if it's possible instead of
# assuming depending on the price change". So the SPLIT DATA decides (the deep
# store's `splits` family: announced splits with their execution date):
#   * `reset_split_tickers` resets every name whose split took effect after its
#     history was last adjusted (the `_adjusted_asof.json` stamp) — in the
#     pre-open run, the selection short's prepare and each of its runs, i.e.
#     before any inference, including a split effective TODAY;
#   * an extension re-reads the last stored session; a changed scale only
#     PROMPTS the split-data check — confirmed, the name is reset; unconfirmed,
#     it is logged (a data correction, or a split the data does not hold yet)
#     and appended as before, never reset on the price change alone.
SPLIT_SCALE_TOLERANCE = 0.02               # |fresh / stored - 1| beyond this = look at the split data
SPLITS_PATH = Path("cache/ml/deep/splits.parquet")
# The store was built in September 2026 and only appended since; a name with no
# stamp is taken as adjusted as of this date, so every split since is re-checked.
_ADJ_DEFAULT = "2026-08-01"


def _adj_path() -> Path:
    return DEEP_DIR / "_adjusted_asof.json"


def _adjusted_asof() -> Dict[str, str]:
    try:
        import json
        return json.loads(_adj_path().read_text(encoding="utf-8"))
    except Exception:
        return {}


_STAMP_LOCK = threading.Lock()


def _stamp_adjusted(tk: str, day: date) -> None:
    """One read-modify-write at a time: `extend_deep_30m` stamps NEW names from its worker
    threads, which shared one temp file and raced (2026-10-05, 103 new listings: 7 'failed'
    after their bars were saved, and stamps lost)."""
    import json
    with _STAMP_LOCK:
        stamps = _adjusted_asof()
        stamps[tk.upper()] = day.isoformat()
        DEEP_DIR.mkdir(parents=True, exist_ok=True)
        tmp = _adj_path().with_suffix(f".json.{os.getpid()}.tmp")
        tmp.write_text(json.dumps(stamps, sort_keys=True), encoding="utf-8")
        os.replace(tmp, _adj_path())


def scale_change(old: pd.DataFrame, fresh: pd.DataFrame) -> Optional[float]:
    """The median fresh/stored close ratio over the bars both hold, when it is
    off 1 by more than `SPLIT_SCALE_TOLERANCE` (the adjustment basis changed),
    else None."""
    common = pd.DatetimeIndex(old.index).intersection(pd.DatetimeIndex(fresh.index))
    if len(common) == 0:
        return None
    a = pd.to_numeric(old.loc[common, "Close"], errors="coerce").to_numpy(float)
    b = pd.to_numeric(fresh.loc[common, "Close"], errors="coerce").to_numpy(float)
    ok = np.isfinite(a) & np.isfinite(b) & (a > 0) & (b > 0)
    if not ok.any():
        return None
    med = float(np.median(b[ok] / a[ok]))
    return med if abs(med - 1.0) > SPLIT_SCALE_TOLERANCE else None


def reset_deep_30m(tk: str, today: Optional[date] = None, why: str = "") -> str:
    """Refetch the ticker's WHOLE history (adjusted as of now) and replace the
    stored frame. Returns 'reset', 'empty' or 'failed'."""
    today = today or date.today()
    try:
        fresh = _fetch_range(tk, DEEP_FROM, today.isoformat())
    except Exception as e:                                   # noqa: BLE001
        logger.warning(f"[intraday_store] {tk}: history reset failed ({e})")
        return "failed"
    if fresh.empty:
        return "empty"
    save_deep_30m(tk, fresh)
    _stamp_adjusted(tk, today)
    logger.warning(f"[intraday_store] {tk}: history RESET ({why or 'split'}) — {len(fresh)} bars refetched")
    return "reset"


_SPLIT_CACHE: Dict[str, tuple] = {}


def latest_splits(today: date) -> Optional[Dict[str, date]]:
    """``{ticker: latest split execution date <= today}`` from the split data
    (read once per file version), or None when it cannot be read."""
    p = Path(SPLITS_PATH)
    try:
        key = (p.as_posix(), p.stat().st_mtime, today.isoformat())
    except OSError:
        return None
    hit = _SPLIT_CACHE.get("v")
    if hit is not None and hit[0] == key:
        return hit[1]
    try:
        import duckdb
        df = duckdb.connect().execute(
            f"SELECT ticker, max(execution_date) AS d FROM read_parquet('{p.as_posix()}') "
            "WHERE split_from > 0 AND split_to > 0 AND split_from <> split_to "
            f"AND execution_date <= '{today.isoformat()}' GROUP BY ticker").fetchdf()
    except Exception as e:                                   # noqa: BLE001
        logger.warning(f"[intraday_store] split data unreadable ({e})")
        return None
    out = {str(t).upper(): d for t, d in zip(df["ticker"], pd.to_datetime(df["d"]).dt.date)}
    _SPLIT_CACHE["v"] = (key, out)
    return out


def split_rows() -> Optional[Dict[str, List[Tuple[date, float, float]]]]:
    """``{ticker: [(execution date, split_from, split_to), ...]}`` from the split data —
    every split it records, future-dated ones included (the caller cuts by date) — read
    once per file version, or None when it cannot be read."""
    p = Path(SPLITS_PATH)
    try:
        key = (p.as_posix(), p.stat().st_mtime)
    except OSError:
        return None
    hit = _SPLIT_CACHE.get("rows")
    if hit is not None and hit[0] == key:
        return hit[1]
    try:
        import duckdb
        df = duckdb.connect().execute(
            f"SELECT DISTINCT ticker, execution_date, split_from, split_to FROM read_parquet('{p.as_posix()}') "
            "WHERE split_from > 0 AND split_to > 0 AND split_from <> split_to").fetchdf()
    except Exception as e:                                   # noqa: BLE001
        logger.warning(f"[intraday_store] split data unreadable ({e})")
        return None
    out: Dict[str, List[Tuple[date, float, float]]] = {}
    for t, d, a, b in zip(df["ticker"], pd.to_datetime(df["execution_date"]).dt.date, df["split_from"], df["split_to"]):
        out.setdefault(str(t).upper(), []).append((d, float(a), float(b)))
    _SPLIT_CACHE["rows"] = (key, out)
    return out


def split_factor_between(tk: str, after: date, through: date,
                         rows: Optional[Dict[str, List[Tuple[date, float, float]]]] = None) -> Optional[float]:
    """What a price quoted on ``after`` is multiplied by to compare with one quoted on
    ``through``: the product of split_from / split_to over the splits executed in
    (``after``, ``through``] (a 1-for-10 reverse split: x10; a 2-for-1 split: x0.5). A split
    dated after ``through`` never counts. 1.0 when none; None when the split data cannot be read."""
    rows = split_rows() if rows is None else rows
    if rows is None:
        return None
    f = 1.0
    for d, a, b in rows.get(tk.upper(), []):
        if after < d <= through:
            f *= a / b
    return f


def split_to_apply(tk: str, today: date, splits: Optional[Dict[str, date]] = None) -> Optional[date]:
    """The execution date of a split the split data records for ``tk`` AFTER its
    stored history was last adjusted (and on/before ``today``), else None."""
    splits = latest_splits(today) if splits is None else splits
    d = (splits or {}).get(tk.upper())
    if d is None:
        return None
    asof = date.fromisoformat(_adjusted_asof().get(tk.upper(), _ADJ_DEFAULT))
    return d if d > asof else None


def reset_split_tickers(tickers: Iterable[str], today: Optional[date] = None) -> Dict[str, str]:
    """Reset every ticker in ``tickers`` whose split (the split data) took
    effect after its stored history was last adjusted and on/before ``today``.
    Returns ``{ticker: outcome}`` for the names it touched. Fail-soft: unreadable
    split data resets nothing (logged)."""
    today = today or date.today()
    names = {t.upper() for t in tickers}
    splits = latest_splits(today) if names else None
    if not splits:
        return {}
    out: Dict[str, str] = {}
    for tk in sorted(names & set(splits)):
        if not _path(tk).exists():
            continue
        d = split_to_apply(tk, today, splits)
        if d is not None:
            out[tk] = reset_deep_30m(tk, today, why=f"split effective {d}")
    return out


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
            # from the last stored session itself: the overlap is re-read and
            # compared, so a split shows up as a scale change (`scale_change`)
            todo.append((tk, last.isoformat()))
    stats = dict(considered=0, fresh=0, extended=0, new=0, empty=0, failed=0, skipped_budget=0, reset=0)
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
                _stamp_adjusted(tk, date.today())
                return tk, "new"
            ratio = scale_change(old, fresh)
            if ratio is not None:
                # the price change only PROMPTS a look at the split data
                d = split_to_apply(tk, date.today())
                if d is not None:
                    return tk, reset_deep_30m(tk, date.today(),
                                             why=f"split effective {d}, stored bars rescaled x{ratio:.4g}")
                logger.warning(f"[intraday_store] {tk}: stored bars rescaled x{ratio:.4g} but the split data "
                               f"records no split since the history was adjusted — NOT reset (a data "
                               f"correction, or a split not in the data yet); appended as before")
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
