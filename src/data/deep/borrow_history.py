"""BORROW HISTORY FOR BACKTESTS — IBKR's borrow fee, lendable shares and rebate per name, as far back as they exist,
in one place (user directive 2026-10-06: "continue building in our database all the borrow fees, and now the
availability of lending. When doing future backtests we should be able to get any information useful to fees,
availability, rules, etc."; and: every name in the universe gets its history, delisted ones included).

Three sources, all IBKR's own figures:

1. OUR ARCHIVE of IBKR's short-stock file — every tick from 2026-09-25 (`src/data/ibkr_borrow.py`; the daily
   summary `borrow_daily.parquet`, `borrow.py`): fee, available, rebate of every name IBKR lists, point in time.
2. IBKR'S API — `reqHistoricalData(whatToShow="FEE_RATE")` daily bars, the fee of every LISTED name back to
   2016-10 (`fetch_missing` -> `cache/ml/ibkr_fee_rate/parts/<T>.csv`, consolidated by `build_fee_api` into
   `borrow_history/fee_api_daily.parquet`, open / high / low / close in %/yr). It matched our archive within
   0.004 %/yr. No availability, and no delisted name (IBKR has no contract for one).
3. A COMMUNITY ARCHIVE of IBKR's daily file (potential-investments.com/ib_shorting.zip, kept in
   `cache/ml/ib_shorting_archive/`): 2,362 files 2017-10-09 .. 2024-06-24 in IBKR's own format, each with IBKR's own
   timestamp (2,261 of them 06:00-08:59 ET, before the open) -> `borrow_history/archive_snapshots.parquet` (every row
   of every file: delisted names too) + `archive_files.parquet` (one row per file). 97.1% of 145,354 daily fees in it
   matched IBKR's API (inside the API day's range or equal to the previous close).

GAP: shares available between 2024-06-25 and 2026-09-24 exist in no free source (QuantRocket and iBorrowDesk sell
them); the fee of a listed name is there (the API).

Live trading does not read this module: the borrow gate reads IBKR's current file, and the deep FEATURES read our
archive only (`borrow_daily.parquet`, user directive 2026-10-02).

Readers, point in time (an instant: aware, or naive = UTC):
* `snapshot_at(ticker, when)` — IBKR's file in force at the instant (the latest stamped at or before it, at most
  ``max_age_days`` old): status ``listed`` (with available / fee / rebate) | ``not_listed`` (IBKR had nothing to lend
  — it drops such names from the file) | ``unknown`` (no file in force).
* `lendable_at(ticker, when, price)` — the live borrow gate's rule (`ibkr_borrow.short_block`: listed with shares
  worth at least `short_borrow_min_available_usd`, any fee): True / False / None.
* `fee_daily(ticker, start, end)` — the daily fee and its source: our archive's close, else IBKR's API close, else
  the community archive's morning snapshot.
A delisted name's symbol may have been reused: pass ``window=(first, last)`` (its own trading dates) so only its
rows match.

    python -m src.data.deep.borrow_history --build-archive | --build-api | --fetch-missing [--budget S] | --coverage
"""
from __future__ import annotations

import csv
import glob
import io
import json
import os
import time
import zipfile
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from loguru import logger

from src.data import deep as _deep
from src.data.deep import family_dir

FAMILY = "borrow_history"
ARCHIVE_ZIP = Path("cache/ml/ib_shorting_archive/ib_shorting.zip")
API_DIR = Path("cache/ml/ibkr_fee_rate")
ARCHIVE_FILE = "archive_snapshots.parquet"
FILES_FILE = "archive_files.parquet"
API_FILE = "fee_api_daily.parquet"
API_STATUS_FILE = "fee_api_status.parquet"
ET = "America/New_York"
CAP = 10_000_000                       # as borrow.py: IBKR prints ">10000000" past this
OWN_START = pd.Timestamp("2026-09-25", tz=ET)
MAX_AGE_DAYS = 4.0
API_CLIENT_OFFSET = 63                 # ibkr_client_id + 63: never the scheduler's session (11) nor the sweep (+50)
API_PACE = 2.0
_DONE = {"ok", "no_contract", "empty"}


def our_symbol(sym: str) -> str:
    from src.data.deep.borrow import our_symbol as _o
    return _o(sym)


def variants(ticker: str) -> List[str]:
    """IBKR's possible spellings of one of our tickers (class shares: `BRK-B` -> `BRK B` / `BRK.B`), as the live
    gate's `ibkr_borrow.lookup` tries them."""
    t = str(ticker).split("@")[0].strip().upper()
    return list(dict.fromkeys([t, t.replace("-", " "), t.replace("-", "."), t.replace(".", " ")]))


def _utc_naive(when) -> pd.Timestamp:
    t = pd.Timestamp(when)
    if t.tzinfo is None:
        return t
    return t.tz_convert("UTC").tz_localize(None)


# ── 3. the community archive (frozen: built once) ─────────────────────────────

def _parse_archive_file(name: str, raw: str) -> Tuple[Optional[pd.Timestamp], pd.DataFrame]:
    """One archived file: (IBKR's #BOF instant, aware ET; its rows)."""
    lines = raw.splitlines()
    ts, header, body = None, None, []
    for ln in lines:
        if ln.startswith("#BOF"):
            p = ln.split("\t")
            try:
                ts = pd.Timestamp(f"{p[1].replace('.', '-')} {p[2]}", tz=ET)
            except Exception:                                   # noqa: BLE001
                ts = None
        elif ln.startswith("#SYM"):
            header = [h.strip().lstrip("#").upper() for h in ln.split("\t") if h.strip()]
        elif ln and not ln.startswith("#"):
            body.append(ln)
    if ts is None or header is None or not body:
        return ts, pd.DataFrame()
    width = max(len(header), max(ln.count("\t") for ln in body) + 1)     # a row may carry extra fields (FIGI)
    df = pd.read_csv(io.StringIO("\n".join(body)), sep="\t", header=None, dtype=str, quoting=csv.QUOTE_NONE,
                     names=list(range(width)), on_bad_lines="skip", engine="c")
    df = df.iloc[:, :len(header)].set_axis(header, axis=1)
    av = df.get("AVAILABLE", pd.Series(index=df.index, dtype=str)).fillna("").str.strip()
    capped = av.str.startswith(">")
    num = pd.to_numeric(av.str.lstrip(">"), errors="coerce")
    out = pd.DataFrame({
        "ts": ts.tz_convert("UTC").tz_localize(None), "date_et": ts.date(), "file": name,
        "sym": df["SYM"].str.strip().str.upper(),
        "conid": pd.to_numeric(df.get("CON"), errors="coerce").round().astype("Int64"),
        "isin": df.get("ISIN"), "name": df.get("NAME"),
        "rebate": pd.to_numeric(df.get("REBATERATE"), errors="coerce"),
        "fee": pd.to_numeric(df.get("FEERATE"), errors="coerce"),
        "available": np.minimum(num.to_numpy(float), CAP), "available_capped": capped.to_numpy(bool)})
    out["ticker"] = out["sym"].map(our_symbol)
    return ts, out


def build_archive(zip_path: Path = ARCHIVE_ZIP, out_dir: Optional[Path] = None) -> dict:
    """Every row of every file of the community archive -> ``archive_snapshots.parquet`` (sorted by ticker, ts) and
    one row per file -> ``archive_files.parquet``. Streams through duckdb (the archive is ~32M rows)."""
    import duckdb
    out_dir = Path(out_dir or family_dir(FAMILY))
    out_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    con = duckdb.connect()
    con.execute("SET preserve_insertion_order = false")
    con.execute("SET memory_limit = '3GB'")
    con.execute(f"SET temp_directory = '{(out_dir / '.duckdb_tmp').as_posix()}'")
    con.execute("CREATE TABLE snaps (ts TIMESTAMP, date_et DATE, file VARCHAR, sym VARCHAR, conid BIGINT, "
                "isin VARCHAR, name VARCHAR, rebate DOUBLE, fee DOUBLE, available DOUBLE, available_capped BOOLEAN, "
                "ticker VARCHAR)")
    meta, batch, seen = [], [], set()
    z = zipfile.ZipFile(zip_path)
    infos = sorted((i for i in z.infolist() if i.filename.lower().endswith((".tsv", ".txt"))), key=lambda i: i.filename)

    def flush():
        if batch:
            df = pd.concat(batch, ignore_index=True)
            con.register("b_", df)
            con.execute("INSERT INTO snaps SELECT ts, date_et, file, sym, conid, isin, name, rebate, fee, available, "
                        "available_capped, ticker FROM b_")
            con.unregister("b_")
            batch.clear()
    for k, info in enumerate(infos):
        with z.open(info) as fh:
            raw = fh.read().decode("utf-8", "replace")
        ts, df = _parse_archive_file(info.filename, raw)
        if ts is not None and ts in seen:                      # a weekend re-download of the same IBKR file
            meta.append({"file": info.filename, "ts": None, "date_et": None, "rows": 0, "duplicate": True})
            continue
        if ts is not None:
            seen.add(ts)
        meta.append({"file": info.filename, "ts": None if ts is None else ts.tz_convert("UTC").tz_localize(None),
                     "date_et": None if ts is None else ts.date(), "rows": int(len(df)), "duplicate": False})
        if len(df):
            batch.append(df)
        if len(batch) >= 40:
            flush()
        if (k + 1) % 250 == 0:
            logger.info(f"[deep.borrow_history] archive: {k + 1}/{len(infos)} files ({time.time() - t0:.0f}s)")
    flush()
    tmp = out_dir / (ARCHIVE_FILE + ".tmp")
    con.execute(f"COPY (SELECT * FROM snaps ORDER BY ticker, ts) TO '{tmp.as_posix()}' (FORMAT PARQUET, COMPRESSION ZSTD)")
    n = con.execute("SELECT count(*) FROM snaps").fetchone()[0]
    con.close()
    os.replace(tmp, out_dir / ARCHIVE_FILE)
    allm = pd.DataFrame(meta)
    files = allm.dropna(subset=["ts"]).drop(columns=["duplicate"]).sort_values("ts")
    _deep.write_parquet(files.reset_index(drop=True), out_dir / FILES_FILE)
    _note(out_dir, "archive", {"zip": str(zip_path), "files": int(len(files)), "rows": int(n),
                               "duplicate_files_dropped": int(allm["duplicate"].sum()),
                               "first": str(files["date_et"].min()), "last": str(files["date_et"].max())})
    logger.info(f"[deep.borrow_history] {ARCHIVE_FILE}: {n:,} rows from {len(files):,} files "
                f"{files['date_et'].min()} .. {files['date_et'].max()} ({time.time() - t0:.0f}s)")
    return {"rows": int(n), "files": int(len(files))}


def _note(out_dir: Path, key: str, info: dict) -> None:
    """The family's manifest: what each table holds and when it was built."""
    p = Path(out_dir) / "manifest.json"
    try:
        m = json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}
    except Exception:                                           # noqa: BLE001
        m = {}
    m[key] = dict(info, built_at=datetime.now(timezone.utc).isoformat(timespec="seconds"))
    tmp = p.with_suffix(".tmp")
    tmp.write_text(json.dumps(m, indent=1, default=str), encoding="utf-8")
    os.replace(tmp, p)


# ── 2. IBKR's API (listed names) ──────────────────────────────────────────────

def _api_status(api_dir: Path = API_DIR) -> Dict[str, dict]:
    st: Dict[str, dict] = {}
    for f in sorted(glob.glob(str(Path(api_dir) / "status*.json"))):
        try:
            st.update(json.load(open(f, encoding="utf-8")))
        except Exception as e:                                  # noqa: BLE001
            logger.warning(f"[deep.borrow_history] unreadable {f}: {e}")
    return st


def build_fee_api(api_dir: Path = API_DIR, out_dir: Optional[Path] = None) -> dict:
    """IBKR's API fee history, one CSV per name -> ``fee_api_daily.parquet`` (ticker, date, open_fee, high_fee,
    low_fee, fee) + ``fee_api_status.parquet`` (every name asked: ok / empty / no_contract / error, IBKR's
    contract id, first and last bar)."""
    import duckdb
    out_dir = Path(out_dir or family_dir(FAMILY))
    out_dir.mkdir(parents=True, exist_ok=True)
    parts = Path(api_dir) / "parts"
    n = 0
    if parts.exists() and any(parts.glob("*.csv")):
        tmp = out_dir / (API_FILE + ".tmp")
        con = duckdb.connect()
        try:
            con.execute(
                f"COPY (SELECT regexp_extract(filename, '([^/\\\\]+)\\.csv$', 1) AS ticker, "
                f"CAST(date AS DATE) AS date, CAST(open_fee AS DOUBLE) AS open_fee, CAST(high_fee AS DOUBLE) AS high_fee, "
                f"CAST(low_fee AS DOUBLE) AS low_fee, CAST(fee AS DOUBLE) AS fee "
                f"FROM read_csv('{(parts / '*.csv').as_posix()}', filename=true, header=true, union_by_name=true, "
                f"columns={{'date': 'VARCHAR', 'open_fee': 'VARCHAR', 'high_fee': 'VARCHAR', 'low_fee': 'VARCHAR', 'fee': 'VARCHAR'}}) "
                f"ORDER BY ticker, date) TO '{tmp.as_posix()}' (FORMAT PARQUET, COMPRESSION ZSTD)")
            n = con.execute(f"SELECT count(*) FROM read_parquet('{tmp.as_posix()}')").fetchone()[0]
        finally:
            con.close()
        os.replace(tmp, out_dir / API_FILE)
    st = _api_status(api_dir)
    S = pd.DataFrame([dict(ticker=k, **{c: v.get(c) for c in ("status", "conid", "primary", "first", "last", "bars",
                                                               "fetched_at", "error")}) for k, v in st.items()])
    if len(S):
        S["conid"] = pd.to_numeric(S["conid"], errors="coerce")
        S["bars"] = pd.to_numeric(S["bars"], errors="coerce")
        _deep.write_parquet(S.sort_values("ticker").reset_index(drop=True), out_dir / API_STATUS_FILE)
    counts = S["status"].value_counts().to_dict() if len(S) else {}
    _note(out_dir, "api", {"rows": int(n), "names": int(len(S)), "status": counts})
    logger.info(f"[deep.borrow_history] {API_FILE}: {n:,} rows; names asked {len(S):,} {counts}")
    return {"rows": int(n), "names": int(len(S)), "status": counts}


def _in_ibc_restart(now: Optional[datetime] = None) -> bool:
    n = (now or datetime.now(timezone.utc)).astimezone(pd.Timestamp.now(tz=ET).tz)
    hm = n.hour * 60 + n.minute
    return 23 * 60 + 48 <= hm <= 23 * 60 + 58


def missing_names(api_dir: Path = API_DIR, names: Optional[Iterable[str]] = None) -> List[str]:
    """Universe names (or ``names``) IBKR's API has not been asked about yet (or whose last try failed)."""
    names = list(dict.fromkeys(str(n).upper() for n in names)) if names is not None else _deep.deep_universe()
    st = _api_status(api_dir)
    saved = {p.stem for p in (Path(api_dir) / "parts").glob("*.csv")} if (Path(api_dir) / "parts").exists() else set()
    return [t for t in names if t not in saved and st.get(t, {}).get("status") not in _DONE]


def fetch_missing(budget_seconds: float = 1800.0, max_names: int = 0, api_dir: Path = API_DIR,
                  names: Optional[Iterable[str]] = None) -> dict:
    """IBKR's API fee history for the universe names that have none yet — the names a later universe adds. One
    request every API_PACE seconds on its own client id, day or night (user 2026-10-06: "we can continue during the
    day"), never through IBC's daily restart; stops at the budget. Skips entirely while another pull is writing parts
    (a file modified in the last 10 minutes). Saves each name as it comes (resumable)."""
    out = {"fetched": 0, "with_history": 0, "skipped": None}
    parts = Path(api_dir) / "parts"
    parts.mkdir(parents=True, exist_ok=True)
    recent = [p for p in parts.glob("*.csv") if time.time() - p.stat().st_mtime < 600]
    if recent:
        out["skipped"] = "another pull is writing"
        return out
    todo = missing_names(api_dir, names)
    if max_names:
        todo = todo[:max_names]
    out["missing"] = len(todo)
    if not todo:
        return out
    return _pull(todo, "10 Y", False, budget_seconds, api_dir, out)


def held_short_names(days: Optional[int] = 7) -> List[str]:
    """The names the ledger is short now or covered in the last ``days`` days (None: every short it ever held) — the
    ones whose every day of IBKR's rate the ledger's borrow schedule needs (src/performance/borrow_fees.py).
    Read-only."""
    from src.db import repo
    cut = "" if days is None else (datetime.now(timezone.utc) - pd.Timedelta(days=days)).isoformat()
    D = repo.fetch_df("SELECT ticker, status, json_extract_string(data, '$.exit_datetime') AS exit_dt "
                      "FROM trades WHERE upper(action) = 'SELL'", read_only=True)
    return sorted({str(t).upper() for t, st, x in zip(D["ticker"], D["status"], D["exit_dt"])
                   if t and (str(st).upper() == "OPEN" or str(x or "") >= cut)})


def refresh_names(names: Iterable[str], budget_seconds: float = 600.0, api_dir: Path = API_DIR) -> dict:
    """The last month of IBKR's API fee bars for ``names``, merged into their parts (newer bars replace a day, older
    history kept) — so a held short has IBKR's rate for every day, a day IBKR's file stopped listing it included."""
    names = [n for n in dict.fromkeys(str(x).upper() for x in names) if n]
    out = {"fetched": 0, "with_history": 0, "skipped": None, "names": len(names)}
    if not names:
        return out
    (Path(api_dir) / "parts").mkdir(parents=True, exist_ok=True)
    return _pull(names, "1 M", True, budget_seconds, api_dir, out)


def _merge_part(path: Path, rows: List[str]) -> None:
    """Write a name's daily bars into its part: the new rows replace their days, older days are kept."""
    keep: Dict[str, str] = {}
    if path.exists():
        for ln in path.read_text(encoding="utf-8").splitlines()[1:]:
            if ln.strip():
                keep[ln.split(",", 1)[0]] = ln
    for ln in rows:
        keep[ln.split(",", 1)[0]] = ln
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text("date,open_fee,high_fee,low_fee,fee\n" + "".join(keep[k] + "\n" for k in sorted(keep)),
                   encoding="utf-8")
    os.replace(tmp, path)


def _pull(todo: List[str], duration: str, merge: bool, budget_seconds: float, api_dir: Path, out: dict) -> dict:
    """One FEE_RATE request per name (daily bars over ``duration``), paced, on its own client id; each name saved as
    it comes (replacing its part, or merged into it)."""
    from config.settings import settings
    from ib_async import IB, Stock
    from src.broker.ibkr import to_ib_symbol
    parts = Path(api_dir) / "parts"
    t0 = time.time()
    status_path = Path(api_dir) / "status_refresh.json"
    mine = json.load(open(status_path, encoding="utf-8")) if status_path.exists() else {}
    ib = IB()
    errs: List[Tuple[int, str]] = []
    ib.errorEvent += lambda req, code, text, contract=None, *a: errs.append((int(code), str(text)[:200]))
    try:
        while _in_ibc_restart():
            if time.time() - t0 > budget_seconds:
                out["skipped"] = "IBC restart window"
                return out
            time.sleep(30)
        ib.connect(settings.ibkr_host, settings.ibkr_port, clientId=int(settings.ibkr_client_id) + API_CLIENT_OFFSET,
                   timeout=20)
        ib.RequestTimeout = 60
        for tk in todo:
            if time.time() - t0 > budget_seconds or _in_ibc_restart():
                break
            if not ib.isConnected():
                break
            rec = {"fetched_at": datetime.now().isoformat(timespec="seconds"), "tries": mine.get(tk, {}).get("tries", 0) + 1}
            errs.clear()
            try:
                c = Stock(to_ib_symbol(tk), "SMART", "USD")
                q = ib.qualifyContracts(c)
                if not q or not getattr(c, "conId", 0):
                    rec.update(status="no_contract")
                else:
                    bars = ib.reqHistoricalData(c, endDateTime="", durationStr=duration, barSizeSetting="1 day",
                                                whatToShow="FEE_RATE", useRTH=False, formatDate=1)
                    if any(a == 162 and "pacing" in b.lower() for a, b in errs):
                        rec.update(status="error", error="pacing violation")
                        mine[tk] = rec
                        break
                    rec.update(conid=int(c.conId), primary=getattr(c, "primaryExchange", None), bars=len(bars))
                    if bars:
                        lines = [f"{str(b.date)[:10]},{b.open * 100:.6g},{b.high * 100:.6g},{b.low * 100:.6g},"
                                 f"{b.close * 100:.6g}" for b in bars]
                        if merge:
                            _merge_part(parts / f"{tk}.csv", lines)
                        else:
                            tmp = parts / f"{tk}.csv.tmp"
                            tmp.write_text("date,open_fee,high_fee,low_fee,fee\n" + "".join(ln + "\n" for ln in lines),
                                           encoding="utf-8")
                            os.replace(tmp, parts / f"{tk}.csv")
                        rec.update(status="ok", first=str(bars[0].date)[:10], last=str(bars[-1].date)[:10])
                        out["with_history"] += 1
                    else:
                        rec.update(status="empty", error="; ".join(f"{a}: {b}" for a, b in errs
                                                                    if a not in (2104, 2106, 2158))[:200] or None)
            except ConnectionError:
                break
            except Exception as e:                              # noqa: BLE001
                rec.update(status="error", error=f"{type(e).__name__}: {e}"[:200])
            if not merge or rec.get("status") == "ok":           # a month's refresh never downgrades a name's status
                mine[tk] = rec
            out["fetched"] += 1
            time.sleep(API_PACE)
    finally:
        try:
            ib.disconnect()
        except Exception:                                       # noqa: BLE001
            pass
        tmp = status_path.with_suffix(".tmp")
        tmp.write_text(json.dumps(mine, indent=0), encoding="utf-8")
        os.replace(tmp, status_path)
    out["seconds"] = round(time.time() - t0)
    logger.info(f"[deep.borrow_history] IBKR API fee history ({duration}): {out}")
    return out


# ── readers (point in time) ───────────────────────────────────────────────────

_CACHE: Dict[str, object] = {}


def _reset_cache() -> None:
    _CACHE.clear()


def _path(name: str) -> Path:
    return family_dir(FAMILY) / name


def _files() -> pd.DataFrame:
    """The community archive's files (ts naive UTC), cached."""
    key = "files"
    p = _path(FILES_FILE)
    if key not in _CACHE and p.exists():
        f = _deep.read_parquet(p)
        f["ts"] = pd.to_datetime(f["ts"])
        _CACHE[key] = f.sort_values("ts").reset_index(drop=True)
    return _CACHE.get(key, pd.DataFrame(columns=["file", "ts", "date_et", "rows"]))


def archive_rows(tickers: Iterable[str], start=None, end=None, window: Optional[Tuple] = None,
                 conids: Optional[Sequence[int]] = None) -> pd.DataFrame:
    """The community archive's rows of ``tickers`` (any IBKR spelling) or of IBKR contract ids ``conids``, between
    ``start`` and ``end`` (instants), within ``window`` (a name's own trading dates) when given."""
    import duckdb
    p = _path(ARCHIVE_FILE)
    if not p.exists():
        return pd.DataFrame()
    syms = sorted({v for t in tickers for v in variants(t)})
    cond = [f"sym IN ({','.join(repr(s) for s in syms)})"] if syms else []
    if conids:
        cond.append(f"conid IN ({','.join(str(int(c)) for c in conids)})")
    if not cond:
        return pd.DataFrame()
    where = "(" + " OR ".join(cond) + ")"
    for bound, op in ((start, ">="), (end, "<=")):
        if bound is not None:
            where += f" AND ts {op} TIMESTAMP '{_utc_naive(bound)}'"
    if window is not None:
        lo, hi = window
        where += f" AND date_et >= DATE '{pd.Timestamp(lo).date()}' AND date_et <= DATE '{pd.Timestamp(hi).date()}'"
    con = duckdb.connect()
    try:
        return con.execute(f"SELECT * FROM read_parquet('{p.as_posix()}') WHERE {where} ORDER BY ts").fetchdf()
    finally:
        con.close()


def snapshot_at(ticker: str, when, max_age_days: float = MAX_AGE_DAYS, window: Optional[Tuple] = None,
                conids: Optional[Sequence[int]] = None) -> dict:
    """IBKR's file in force at ``when`` for ``ticker``: ``{"status": listed | not_listed | unknown, "available",
    "fee", "rebate", "file_ts" (naive UTC), "source": own_archive | community_archive | None}``."""
    t = _utc_naive(when)
    out = {"status": "unknown", "available": None, "fee": None, "rebate": None, "file_ts": None, "source": None}
    if t >= _utc_naive(OWN_START):
        from src.data import ibkr_borrow as ib
        got = ib.snapshot_at(t.to_pydatetime().replace(tzinfo=timezone.utc))
        if got is None:
            return out
        ts, table = got
        ts_n = _utc_naive(ts)
        if (t - ts_n).total_seconds() > max_age_days * 86400:
            return out
        b = ib.lookup(table, ticker)
        out.update(file_ts=ts_n, source="own_archive")
        if b is None:
            out["status"] = "not_listed"
        else:
            out.update(status="listed", available=None if b.available is None else min(float(b.available), CAP),
                       fee=b.fee_pct, rebate=b.rebate_pct)
        return out
    F = _files()
    if not len(F):
        return out
    j = int(np.searchsorted(F["ts"].to_numpy(), np.datetime64(t), side="right")) - 1
    if j < 0:
        return out
    f = F.iloc[j]
    if (t - pd.Timestamp(f["ts"])).total_seconds() > max_age_days * 86400:
        return out
    rows = archive_rows([ticker], start=f["ts"], end=f["ts"], window=window, conids=conids)
    rows = rows[rows["file"] == f["file"]] if len(rows) else rows
    out.update(file_ts=pd.Timestamp(f["ts"]), source="community_archive")
    if not len(rows):
        out["status"] = "not_listed"
    else:
        r = rows.sort_values("available", ascending=False).iloc[0]
        out.update(status="listed", available=float(r["available"]) if pd.notna(r["available"]) else None,
                   fee=float(r["fee"]) if pd.notna(r["fee"]) else None,
                   rebate=float(r["rebate"]) if pd.notna(r["rebate"]) else None)
    return out


def lendable_at(ticker: str, when, price: float, min_usd: Optional[float] = None, **kw) -> Tuple[Optional[bool], dict]:
    """The live borrow gate's rule at an instant: IBKR's file in force lists the name with shares worth at least
    ``min_usd`` (default `short_borrow_min_available_usd`) at ``price``. None when no file was in force."""
    if min_usd is None:
        from config.settings import settings
        min_usd = float(getattr(settings, "short_borrow_min_available_usd", 10_000.0) or 0.0)
    s = snapshot_at(ticker, when, **kw)
    if s["status"] == "unknown":
        return None, s
    if s["status"] == "not_listed" or not s["available"]:
        return False, s
    return bool(float(s["available"]) * float(price) >= float(min_usd)), s


def fee_daily(ticker: str, start=None, end=None, window: Optional[Tuple] = None) -> pd.DataFrame:
    """The daily fee (%/yr) of ``ticker`` with its source, one row per date: our archive's close
    (``own_archive``) > IBKR's API close (``ibkr_api``) > the community archive's morning snapshot
    (``community_archive``)."""
    import duckdb
    frames = []
    own = _deep.DEEP_DIR / "borrow_daily.parquet"
    con = duckdb.connect()
    try:
        syms = [ticker, *variants(ticker), *(our_symbol(v) for v in variants(ticker))]
        lst = ",".join(repr(s) for s in dict.fromkeys(syms))
        if own.exists():
            frames.append(con.execute(f"SELECT CAST(date AS DATE) AS date, fee, open_fee, high_fee, low_fee, "
                                      f"'own_archive' AS source FROM read_parquet('{own.as_posix()}') "
                                      f"WHERE ticker IN ({lst})").fetchdf())
        api = _path(API_FILE)
        if api.exists():
            frames.append(con.execute(f"SELECT date, fee, open_fee, high_fee, low_fee, 'ibkr_api' AS source "
                                      f"FROM read_parquet('{api.as_posix()}') WHERE ticker IN ({lst})").fetchdf())
    finally:
        con.close()
    A = archive_rows([ticker], window=window)
    if len(A):
        A = A.sort_values("ts").drop_duplicates("date_et", keep="last")
        frames.append(pd.DataFrame({"date": pd.to_datetime(A["date_et"]).dt.date, "fee": A["fee"], "open_fee": np.nan,
                                    "high_fee": np.nan, "low_fee": np.nan, "source": "community_archive"}))
    if not frames:
        return pd.DataFrame(columns=["date", "fee", "open_fee", "high_fee", "low_fee", "source"])
    D = pd.concat([f for f in frames if len(f)], ignore_index=True) if any(len(f) for f in frames) else frames[0]
    if not len(D):
        return D
    D["date"] = pd.to_datetime(D["date"]).dt.date
    rank = {"own_archive": 0, "ibkr_api": 1, "community_archive": 2}
    D = D.assign(_r=D["source"].map(rank)).sort_values(["date", "_r"]).drop_duplicates("date", keep="first")
    if start is not None:
        D = D[D["date"] >= pd.Timestamp(start).date()]
    if end is not None:
        D = D[D["date"] <= pd.Timestamp(end).date()]
    return D.drop(columns="_r").reset_index(drop=True)


def coverage() -> dict:
    """What each source holds — for reports and the refresh log."""
    out = {}
    m = _path("manifest.json")
    if m.exists():
        out["manifest"] = json.loads(m.read_text(encoding="utf-8"))
    own = _deep.DEEP_DIR / "borrow_daily.parquet"
    if own.exists():
        import duckdb
        con = duckdb.connect()
        try:
            r = con.execute(f"SELECT min(date), max(date), count(DISTINCT ticker), count(*) FROM read_parquet('{own.as_posix()}')").fetchone()
        finally:
            con.close()
        out["own_archive"] = {"first": str(r[0]), "last": str(r[1]), "names": int(r[2]), "rows": int(r[3])}
    out["gap"] = "shares available 2024-06-25 .. 2026-09-24: no free source (fees: IBKR's API, listed names)"
    return out


if __name__ == "__main__":                              # pragma: no cover
    import argparse
    ap = argparse.ArgumentParser(description="borrow history for backtests")
    ap.add_argument("--build-archive", action="store_true")
    ap.add_argument("--build-api", action="store_true")
    ap.add_argument("--fetch-missing", action="store_true")
    ap.add_argument("--budget", type=float, default=1800.0)
    ap.add_argument("--coverage", action="store_true")
    a = ap.parse_args()
    logger.add("logs/deep_ingest_borrow_history.log", rotation="1 day", retention="14 days", level="INFO", enqueue=True)
    if a.build_archive:
        build_archive()
    if a.fetch_missing:
        fetch_missing(budget_seconds=a.budget)
    if a.build_api:
        build_fee_api()
    if a.coverage:
        print(json.dumps(coverage(), indent=1, default=str))
