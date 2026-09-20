"""DEEP HISTORY STORE — point-in-time external data for the deep training tier.

Why this exists (2026-09-19, user directive "ingest all of the historical data you
could find, starting from the most orthogonal"). `ml_ohlcv` trains on 30-minute
rows from 2021 (the 30-minute pivot label's store starts there) and sees ONLY
price/volume; the stackers see everything else but train on the ~weeks of
`signals` panel that the news-family epoch leaves unmasked. Every family stored
here is external data with history back to 2021 or further, fetched from ONE
source by ONE code path, so the same generator can serve the live tick — the
rule `memory/news-backfill-fidelity-2026-09` earned the hard way (a replayed
feature at 0.40x live magnitude is a different feature).

Layout — one directory per FAMILY under ``cache/ml/deep/``::

    <family>/manifest.json      resumability: which keys are done / failed
    <family>/parts/<KEY>.parquet one part per ticker / quarter / file, written as fetched
    <family>.parquet            the consolidated table (``consolidate``), what training reads
    universe.json               the ticker list every per-ticker family iterates

Point-in-time keys are carried on every row and named for what they ARE, never
massaged into a "known at" date here — that is the feature builder's decision:
``acceptance`` (SEC, UTC instant), ``filed`` (XBRL facts, a date), ``filing_date``
(Form 4), ``settlement_date`` (short interest — published ~9 business days later),
``declaration_date`` / ``ex_dividend_date``, ``realtime_start`` (ALFRED vintages),
``event_ts`` (earnings), ``grade_date`` (analyst actions), ``date`` (daily series).

Everything is a CACHE (never the DuckDB database), parquet written through duckdb
(pyarrow is not installed in the production venv), atomic replace on every file,
and every per-key fetch is idempotent so a killed run costs only its in-flight
keys. CLI: ``python -m src.data.deep [family ...] [--workers N] [--budget-seconds S]
[--limit N] [--tickers A,B] [--status] [--consolidate-only]``.
"""
from __future__ import annotations

import json
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional, Sequence

import pandas as pd
from loguru import logger

DEEP_DIR = Path("cache/ml/deep")

# The SEC asks for a descriptive User-Agent on every request; this is the string
# the repo's other EDGAR clients (`eight_k`, `sec_filings`) already send.
SEC_HEADERS = {"User-Agent": "llm-trader research@example.com",
               "Accept-Encoding": "gzip, deflate"}
GENERIC_HEADERS = {"User-Agent": "llm-trader/1.0 (research)"}


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def family_dir(family: str) -> Path:
    d = DEEP_DIR / family
    (d / "parts").mkdir(parents=True, exist_ok=True)
    return d


# ── parquet through duckdb ───────────────────────────────────────────────────

def write_parquet(df: pd.DataFrame, path: Path) -> int:
    """Write ``df`` to ``path`` atomically (tmp + replace). Returns the row count.

    tz-aware datetimes are converted to naive UTC first — parquet keeps them
    either way, but every consumer in this repo works on naive UTC instants."""
    import duckdb
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    out = df.copy()
    for c in out.columns:
        s = out[c]
        if isinstance(s.dtype, pd.DatetimeTZDtype):
            out[c] = s.dt.tz_convert("UTC").dt.tz_localize(None)
        elif s.dtype == object:
            # duckdb infers VARCHAR from object columns; make sure no exotic
            # python objects (Decimal, dict) sneak through.
            out[c] = s.map(lambda v: v if (v is None or isinstance(v, (str, float, int, bool))) else str(v))
    tmp = path.with_name(path.stem + ".tmp.parquet")
    con = duckdb.connect()
    try:
        con.register("df_", out)
        con.execute(f"COPY (SELECT * FROM df_) TO '{tmp.as_posix()}' (FORMAT PARQUET, COMPRESSION ZSTD)")
    finally:
        con.close()
    os.replace(tmp, path)
    return int(len(out))


def read_parquet(path: Path, columns: Optional[Sequence[str]] = None) -> pd.DataFrame:
    import duckdb
    p = Path(path)
    if not p.exists():
        return pd.DataFrame()
    cols = ", ".join(f'"{c}"' for c in columns) if columns else "*"
    con = duckdb.connect()
    try:
        return con.sql(f"SELECT {cols} FROM '{p.as_posix()}'").df()
    finally:
        con.close()


def consolidate(family: str, out_name: Optional[str] = None) -> int:
    """Union every ``parts/*.parquet`` of a family into ``<family>.parquet``
    (schema differences across parts are unioned by name). Returns rows."""
    import duckdb
    d = family_dir(family)
    parts = sorted((d / "parts").glob("*.parquet"))
    out = DEEP_DIR / f"{out_name or family}.parquet"
    if not parts:
        logger.warning(f"[deep] {family}: no parts to consolidate")
        return 0
    tmp = out.with_name(out.stem + ".tmp.parquet")
    con = duckdb.connect()
    try:
        glob = (d / "parts" / "*.parquet").as_posix()
        con.execute(f"COPY (SELECT * FROM read_parquet('{glob}', union_by_name=true)) "
                    f"TO '{tmp.as_posix()}' (FORMAT PARQUET, COMPRESSION ZSTD)")
        n = con.sql(f"SELECT count(*) FROM '{tmp.as_posix()}'").fetchone()[0]
    finally:
        con.close()
    os.replace(tmp, out)
    logger.info(f"[deep] {family}: consolidated {len(parts)} parts -> {out} ({n:,} rows)")
    return int(n)


# ── resumability ─────────────────────────────────────────────────────────────

class Manifest:
    """``done`` / ``failed`` keys of one family, JSON on disk, atomic replace.

    ``pending(keys)`` is the whole point: a killed or budget-stopped run picks
    up where it left off, and a key that failed is retried on the next run
    (``retry_failed=False`` skips them, for a run that should not spend its
    budget on the same dead symbols again)."""

    def __init__(self, family: str):
        self.family = family
        self.path = family_dir(family) / "manifest.json"
        self._lock = threading.Lock()
        self._dirty = 0
        self.done: Dict[str, dict] = {}
        self.failed: Dict[str, dict] = {}
        if self.path.exists():
            try:
                j = json.loads(self.path.read_text(encoding="utf-8"))
                self.done = dict(j.get("done") or {})
                self.failed = dict(j.get("failed") or {})
            except Exception as e:                       # noqa: BLE001
                logger.warning(f"[deep] {family}: unreadable manifest ({e}) — starting empty")

    def pending(self, keys: Iterable[str], retry_failed: bool = True) -> List[str]:
        out = []
        for k in keys:
            if k in self.done:
                continue
            if not retry_failed and k in self.failed:
                continue
            out.append(k)
        return out

    def mark_done(self, key: str, rows: int, note: str = "") -> None:
        with self._lock:
            self.done[key] = {"at": _now_iso(), "rows": int(rows), "note": note}
            self.failed.pop(key, None)
            self._dirty += 1
            if self._dirty >= 25:
                self._save_locked()

    def mark_failed(self, key: str, err: str) -> None:
        with self._lock:
            self.failed[key] = {"at": _now_iso(), "err": str(err)[:200]}
            self._dirty += 1
            if self._dirty >= 25:
                self._save_locked()

    def save(self) -> None:
        with self._lock:
            self._save_locked()

    def _save_locked(self) -> None:
        tmp = self.path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps({"family": self.family, "saved_at": _now_iso(),
                                   "done": self.done, "failed": self.failed}, indent=0),
                       encoding="utf-8")
        os.replace(tmp, self.path)
        self._dirty = 0

    def stats(self) -> dict:
        rows = sum(int(v.get("rows", 0)) for v in self.done.values())
        return {"done": len(self.done), "failed": len(self.failed), "rows": rows}


# ── throttling and HTTP ──────────────────────────────────────────────────────

class RateLimiter:
    """Minimum spacing between calls, shared across threads."""

    def __init__(self, min_interval: float):
        self.min_interval = float(min_interval)
        self._lock = threading.Lock()
        self._next = 0.0

    def wait(self) -> None:
        with self._lock:
            now = time.monotonic()
            if now < self._next:
                time.sleep(self._next - now)
                now = time.monotonic()
            self._next = now + self.min_interval


def http_get(url: str, params: Optional[dict] = None, headers: Optional[dict] = None,
             timeout: float = 60.0, retries: int = 3, limiter: Optional[RateLimiter] = None):
    """GET with retries on transport errors / 5xx / 429 (honouring Retry-After,
    capped at 90 s). Returns the Response, or None after the last attempt. A 404
    or 403 is returned as-is (the caller decides — never retried)."""
    import requests
    last = None
    for attempt in range(retries + 1):
        if limiter is not None:
            limiter.wait()
        try:
            r = requests.get(url, params=params, headers=headers, timeout=timeout)
        except Exception as e:                           # noqa: BLE001
            last = e
            time.sleep(min(30.0, 2.0 * (attempt + 1)))
            continue
        if r.status_code == 429 or r.status_code >= 500:
            ra = r.headers.get("Retry-After")
            try:
                wait = min(90.0, float(ra)) if ra else min(60.0, 5.0 * (attempt + 1))
            except ValueError:
                wait = min(60.0, 5.0 * (attempt + 1))
            last = RuntimeError(f"HTTP {r.status_code}")
            time.sleep(wait)
            continue
        return r
    logger.debug(f"[deep] GET {url}: giving up ({last})")
    return None


# ── the universe ─────────────────────────────────────────────────────────────

def deep_universe(refresh: bool = False) -> List[str]:
    """Tickers with a deep 30-minute store OR a row in the deep daily parquet —
    the names `ml_ohlcv` trains on. Cached in ``universe.json``; ``refresh``
    rebuilds it. Sorted, upper-case, de-duplicated."""
    DEEP_DIR.mkdir(parents=True, exist_ok=True)
    p = DEEP_DIR / "universe.json"
    if p.exists() and not refresh:
        try:
            return list(json.loads(p.read_text(encoding="utf-8"))["tickers"])
        except Exception:                                # noqa: BLE001
            pass
    names = set()
    try:
        from src.data.intraday_store import DEEP_DIR as BARS_DIR
        names.update(q.stem.upper() for q in Path(BARS_DIR).glob("*.pkl"))
    except Exception:                                    # noqa: BLE001
        pass
    pq = Path("cache/ml/dataset_full.parquet")
    if pq.exists():
        try:
            import duckdb
            con = duckdb.connect()
            try:
                names.update(r[0].upper() for r in
                             con.sql(f"SELECT DISTINCT ticker FROM '{pq.as_posix()}'").fetchall())
            finally:
                con.close()
        except Exception as e:                           # noqa: BLE001
            logger.warning(f"[deep] universe: could not read {pq} ({e})")
    tickers = sorted(n for n in names if n and n.replace("-", "").replace(".", "").isalnum())
    p.write_text(json.dumps({"built_at": _now_iso(), "n": len(tickers), "tickers": tickers}),
                 encoding="utf-8")
    logger.info(f"[deep] universe: {len(tickers)} tickers -> {p}")
    return tickers


# ── the per-key runner ───────────────────────────────────────────────────────

def run_keys(family: str, keys: Sequence[str], fn: Callable[[str], Optional[pd.DataFrame]],
             workers: int = 4, budget_seconds: float = 0.0, retry_failed: bool = True,
             part_name: Optional[Callable[[str], str]] = None,
             empty_is_done: bool = True) -> dict:
    """Fetch every pending key on a thread pool, writing ``parts/<key>.parquet``
    and the manifest as results land. ``fn`` returns a DataFrame (possibly
    empty — recorded as done with 0 rows when ``empty_is_done``) or raises.
    ``budget_seconds`` > 0 stops SUBMITTING new keys once exceeded; in-flight
    keys finish. Returns a summary dict."""
    man = Manifest(family)
    todo = man.pending(keys, retry_failed=retry_failed)
    d = family_dir(family)
    t0 = time.time()
    n_ok = n_fail = n_rows = 0
    logger.info(f"[deep] {family}: {len(todo)} pending of {len(keys)} "
                f"({len(man.done)} done, {len(man.failed)} failed before) | workers {workers}")
    if not todo:
        return {"family": family, "pending": 0, "ok": 0, "failed": 0, "rows": 0, "seconds": 0.0}

    def _one(key: str):
        df = fn(key)
        n = 0
        if df is not None and len(df):
            name = part_name(key) if part_name else key
            n = write_parquet(df, d / "parts" / f"{name}.parquet")
        return key, n, df is not None

    stopped = False
    with ThreadPoolExecutor(max_workers=max(1, int(workers))) as ex:
        futs = {}
        it = iter(todo)
        # keep at most 2x workers in flight so a budget stop is prompt
        def _submit_more():
            nonlocal stopped
            while len(futs) < 2 * max(1, int(workers)):
                if budget_seconds and (time.time() - t0) > budget_seconds:
                    stopped = True
                    return
                try:
                    k = next(it)
                except StopIteration:
                    return
                futs[ex.submit(_one, k)] = k
        _submit_more()
        while futs:
            done_any = False
            for fut in as_completed(list(futs.keys())):
                key = futs.pop(fut)
                done_any = True
                try:
                    k, n, ok = fut.result()
                    if ok or empty_is_done:
                        man.mark_done(k, n)
                        n_ok += 1
                        n_rows += n
                    else:
                        man.mark_failed(k, "empty")
                        n_fail += 1
                except Exception as e:                   # noqa: BLE001
                    man.mark_failed(key, repr(e))
                    n_fail += 1
                if (n_ok + n_fail) % 100 == 0:
                    el = time.time() - t0
                    logger.info(f"[deep] {family}: {n_ok + n_fail}/{len(todo)} "
                                f"({n_fail} failed, {n_rows:,} rows) {el:.0f}s")
                break
            if done_any:
                _submit_more()
    man.save()
    el = time.time() - t0
    logger.info(f"[deep] {family}: finished {n_ok} ok / {n_fail} failed / {n_rows:,} rows "
                f"in {el:.0f}s{' (BUDGET STOP)' if stopped else ''}")
    return {"family": family, "pending": len(todo), "ok": n_ok, "failed": n_fail,
            "rows": n_rows, "seconds": el, "budget_stop": stopped}


def status() -> pd.DataFrame:
    """One row per family: manifest counts + consolidated parquet rows."""
    import duckdb
    rows = []
    if not DEEP_DIR.exists():
        return pd.DataFrame()
    for d in sorted(p for p in DEEP_DIR.iterdir() if p.is_dir()):
        man = d / "manifest.json"
        st = {"done": 0, "failed": 0, "rows": 0}
        if man.exists():
            try:
                st = Manifest(d.name).stats()
            except Exception:                            # noqa: BLE001
                pass
        parts = len(list((d / "parts").glob("*.parquet"))) if (d / "parts").exists() else 0
        out = DEEP_DIR / f"{d.name}.parquet"
        n_out = None
        if out.exists():
            con = duckdb.connect()
            try:
                n_out = con.sql(f"SELECT count(*) FROM '{out.as_posix()}'").fetchone()[0]
            finally:
                con.close()
        rows.append({"family": d.name, "done": st["done"], "failed": st["failed"],
                     "part_rows": st["rows"], "parts": parts, "consolidated_rows": n_out})
    return pd.DataFrame(rows)
