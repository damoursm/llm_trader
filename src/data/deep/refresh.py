"""Nightly INCREMENTAL refresh of the deep history store (2026-09-23, user
directive: every data source in the historical store, and the scheduler keeps
ingesting them).

The manifest makes a family RESUMABLE, not REFRESHABLE: ``run_keys`` skips every
key already marked done, so re-running a per-ticker family after the initial
ingest fetches nothing and the data silently ages (measured 2026-09-23: every
family frozen at its 2026-09-19/20 build). This module extends each family's
TAIL instead:

* per-ticker families — from the part's own newest point-in-time key, re-read
  from the part (never trusted from the manifest), fetched with a small overlap
  and merged on the row's identity (bar ``ts``, ``article_id``, ``accession``,
  ``date`` …);
* file-keyed families (FTD, 13F, bulk Form 345) — the keys the SEC index page
  lists that no part covers yet, i.e. ``run_keys`` as it stands;
* whole-table families (context) — an overwrite where the fetch is one call,
  and an ALFRED real-time WINDOW where it is not;
* refetch-and-overwrite families (yfinance, Quiver per-ticker history, ticker
  details) — ``run_keys(force=True)``, keys ordered by staleness so a budget
  stop CONTINUES next time instead of restarting.

Cadence is per family and self-paced through ``refresh_state.json`` (the last
successful run per family): one nightly invocation runs whatever is DUE, so the
weekly families land on the first night they are due and a cut-short night
rolls the remainder forward. A family stopped by the budget keeps every part
it wrote and is not marked, so it is due again the next night.

Two point-in-time rules exist only because this runs every night:

* ``form345_live`` never fetches TODAY. The EDGAR daily index for a day is
  final only once the day is over, and a day part, once written, is never
  re-fetched — a partial day would freeze. Insider TRANSACTIONS therefore run
  one day behind in the store; the same filings' ACCEPTANCE instants land the
  same night through ``sec_filings`` (the submissions JSON is real-time).
* ALFRED is asked only for the real-time window since the last refresh. That
  window clamps every older value's ``realtime_start`` to its own start, so
  the merge keeps the EARLIEST ``realtime_start`` per (date, value) — the
  stored row carries the true first print — and takes the NEWEST
  ``realtime_end`` (only the newer fetch knows a value was superseded).

Scheduled by ``runner._run_deep_refresh`` as a SUBPROCESS at
``deep_refresh_time`` ET under ``deep_refresh_budget_seconds``. CLI::

    python -m src.data.deep.refresh [--families a,b] [--force] [--budget-seconds N]
                                    [--sec-rate S] [--workers N] [--status]
"""
from __future__ import annotations

import json
import os
import sys
import threading
import time
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import pandas as pd
from loguru import logger

from src.data import deep

# ── cadence ──────────────────────────────────────────────────────────────────
# Hours a family may age before it is due again. Nightly families use 20 h so
# a slot that drifts by an hour still fires every night; 3-day and weekly
# families are the slow publishers (FINRA short interest twice a month, SEC
# file drops) and the refetch-everything sweeps that cost an hour each.
DAILY, EVERY_3D, WEEKLY = 20.0, 68.0, 160.0
# sec_filings: the pre-open run re-reads only the companies EDGAR's live feed
# names (an incremental pass, stamped ``sec_filings_incremental``), so the
# NIGHTLY run must take the full pass every night — the anchor the next
# pre-open's feed has to reach back to (~9 h, inside its ~1-day depth). 12 h
# keeps it due at 23:45 whether the last full pass was the previous night or
# a pre-open that fell back to one (2026-09-29).
HALF_DAILY = 12.0

# (family, cadence) in RUN ORDER — measured on the first full pass,
# 2026-09-23: the universe first (new names become pending keys for
# everything after it); the nightly tails next, ~35 min together (SEC
# filings 11 min at 0.2 s spacing, bars 7, news 3.6, short volume 3.5,
# context 1.4, Quiver live 1); then the families the MODEL FEATURES need
# daily (`src.analysis.deep_features.LAG_DAYS` assumes these cadences:
# yfinance 47 min, Quiver DPI 54 min); the slow publishers; the weekly
# refetches (ticker details 3 min, Quiver event history ~1.4 h); and LAST
# Wikipedia (2.4 h, ~6 s per article-call), which the budget cuts most nights
# and which converges stalest-first over two — its feature lag (4 days) is
# sized for exactly that. On market days the 08:30 ET `preopen` run has
# already refreshed PREOPEN_FAMILIES, so the nightly finds them not due.
FAMILIES: List[Tuple[str, float]] = [
    ("universe", WEEKLY),
    ("form345_live", DAILY),
    ("sec_filings", HALF_DAILY),
    ("companyfacts", DAILY),
    ("bars30m_full", DAILY),
    ("polygon_news", DAILY),
    ("short_volume", DAILY),
    ("short_interest", DAILY),
    ("dividends", DAILY),
    ("splits", DAILY),
    ("context", DAILY),
    ("regsho", DAILY),
    ("borrow", DAILY),
    ("quiver_live", DAILY),
    ("yf", DAILY),
    ("quiver_dpi", DAILY),
    ("ftd", EVERY_3D),
    ("form13f", EVERY_3D),
    ("form345", EVERY_3D),
    ("ticker_details", WEEKLY),
    ("ipos", WEEKLY),
    ("delisted", WEEKLY),
    ("quiver_history", WEEKLY),
    ("wiki", DAILY),
]
FAMILY_NAMES = [f for f, _ in FAMILIES]

# The families a session's features need as of 08:30 ET that same morning
# (`deep_features`: acceptance / publication instants, the pre-market bars,
# yesterday's EDGAR daily index, FINRA files, market closes). Re-fetched by
# the `preopen` profile on market days; then the 30-minute store is extended
# through the previous session (`extend_bars_30m` — the snapshot's price grid)
# and the session snapshot is built.
PREOPEN_FAMILIES: List[str] = ["form345_live", "sec_filings", "bars30m_full", "polygon_news",
                               "short_volume", "short_interest", "context", "regsho", "borrow"]

# The pre-open families share neither a table nor a rate limiter across these
# LANES, so the pre-open run takes them side by side: a lane runs its families
# in order, the lanes run at once (2026-09-29). EDGAR's fair-access spacing is
# ONE lane (both families wait on the same limiter); Polygon is split in two
# lanes of ~14 min at 6 workers each. 2026-09-28 the one-after-another run took
# 47 min (sec_filings alone 16: 3,239 submission files at the 0.3 s spacing) and
# the scheduler's kill caught the snapshot build; the lanes take ~the longest
# lane (~16 min). A family missing here runs in a lane of its own.
PREOPEN_LANES: Dict[str, str] = {
    "form345_live": "edgar", "sec_filings": "edgar",
    "bars30m_full": "polygon_a", "short_volume": "polygon_a",
    "polygon_news": "polygon_b", "short_interest": "polygon_b",
    "context": "context", "regsho": "regsho", "borrow": "borrow",
}

# The NIGHTLY run's lanes (2026-09-29): one per rate-limited provider, run
# side by side, each in FAMILIES order (companyfacts after sec_filings; the
# delisted bars after the delisted list). ``universe`` runs FIRST, alone ("" —
# new names become pending keys for every family after it). One after another
# the nightly took 13,658 of its 14,400 s on 09-28 (Wikipedia alone 8,773), so
# the full sec_filings pass could not be added; in lanes it takes ~the
# Wikipedia lane. A family missing here runs in a lane of its own.
NIGHTLY_LANES: Dict[str, str] = {
    "universe": "",
    "form345_live": "edgar", "sec_filings": "edgar", "companyfacts": "edgar", "ftd": "edgar",
    "form13f": "edgar", "form345": "edgar",
    "bars30m_full": "polygon", "polygon_news": "polygon", "short_volume": "polygon",
    "short_interest": "polygon", "dividends": "polygon", "splits": "polygon",
    "ticker_details": "polygon", "ipos": "polygon", "delisted": "polygon",
    "quiver_live": "quiver", "quiver_dpi": "quiver", "quiver_history": "quiver",
    "yf": "yf", "context": "context", "wiki": "wiki", "regsho": "regsho", "borrow": "borrow",
}

# The pre-open's two per-ticker Polygon families (no market-wide endpoint: bars
# and news are per ticker) get more workers — the FINRA families no longer
# spend the Polygon limiter's budget (one market-wide sweep each).
PREOPEN_WORKERS: Dict[str, int] = {"bars30m_full": 10, "polygon_news": 10}

# Forms whose XBRL instance changes companyfacts. ``is_xbrl`` on the filing row
# is the authoritative flag; this list only names what that flag usually is.
XBRL_FORMS = ("10-K", "10-Q", "20-F", "40-F", "6-K", "10-KT", "10-QT")


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def state_path() -> Path:
    return deep.DEEP_DIR / "refresh_state.json"


def load_state() -> Dict[str, dict]:
    p = state_path()
    if not p.exists():
        return {}
    try:
        return dict(json.loads(p.read_text(encoding="utf-8")).get("families") or {})
    except Exception as e:                               # noqa: BLE001
        logger.warning(f"[deep.refresh] unreadable state file ({e}) — starting empty")
        return {}


def save_state(state: Dict[str, dict]) -> None:
    p = state_path()
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(".json.tmp")
    tmp.write_text(json.dumps({"saved_at": _now_iso(), "families": state}, indent=1),
                   encoding="utf-8")
    os.replace(tmp, p)


def due(family: str, cadence_hours: float, state: Dict[str, dict],
        now: Optional[datetime] = None) -> bool:
    """A family is due when it never completed or completed longer ago than
    its cadence. ``at`` is the COMPLETION instant of the last full run."""
    at = (state.get(family) or {}).get("at")
    if not at:
        return True
    try:
        last = datetime.fromisoformat(str(at))
    except ValueError:
        return True
    if last.tzinfo is None:
        last = last.replace(tzinfo=timezone.utc)
    now = now or datetime.now(timezone.utc)
    return (now - last).total_seconds() >= cadence_hours * 3600.0


# ── part helpers ─────────────────────────────────────────────────────────────

def part_max(family: str, key_col: str) -> Dict[str, str]:
    """``{part key: newest value of key_col as a string}`` over every part of a
    per-ticker family, in ONE duckdb scan (the manifest is not consulted —
    the part is the truth about what the store holds)."""
    import duckdb
    d = deep.family_dir(family) / "parts"
    if not any(d.glob("*.parquet")):
        return {}
    con = duckdb.connect()
    try:
        rows = con.execute(
            f'SELECT filename, max(CAST("{key_col}" AS VARCHAR)) FROM '
            f"read_parquet('{(d / '*.parquet').as_posix()}', union_by_name=true, filename=true) "
            f"GROUP BY filename").fetchall()
    finally:
        con.close()
    out = {}
    for fn, mx in rows:
        key = Path(str(fn)).stem
        if mx is not None:
            out[key] = str(mx)
    return out


def _row_key(df: pd.DataFrame, cols: Sequence[str]) -> pd.Series:
    """One string per row over ``cols`` — a dtype-proof identity (a value that
    arrives as 5000 in one fetch and 5000.0 in the next still collides)."""
    cols = [c for c in cols if c in df.columns]
    if not cols:
        return pd.Series([""] * len(df), index=df.index)
    parts = [df[c].map(lambda v: "" if v is None or (isinstance(v, float) and pd.isna(v)) else str(v))
             for c in cols]
    out = parts[0]
    for p in parts[1:]:
        out = out + "|" + p
    return out


def merge_frame(path: Path, new_df: Optional[pd.DataFrame], dedupe_on: Sequence[str],
                sort_by=None, mode: str = "append") -> Tuple[pd.DataFrame, int]:
    """Union ``new_df`` into the parquet at ``path`` on the row identity
    ``dedupe_on``. Returns ``(rows whose identity the table did not hold,
    total rows now stored)``. THE STORE NEVER LOSES A ROW IT HOLDS:

    * ``append`` — rows whose identity the table already holds are IGNORED;
      stored rows are never rewritten or de-duplicated. For families whose
      payload legitimately repeats an identity: the first refresh
      (2026-09-23) rebuilt each touched part as ``old ∪ new`` de-duplicated,
      and the Quiver tables SHRANK (contracts 1,027,080 → 966,395) — the
      historical endpoint returns identical awards as separate rows, the
      ingest stores its payload verbatim, and a merge that cleans only the
      parts a night touches leaves the store inconsistent between tickers.
    * ``upsert`` — rows whose identity the table holds are REPLACED by the
      new rows (a true primary key, a refetched full series brings revised
      values), and rows the new fetch no longer carries are KEPT. Never an
      overwrite: Quiver's DPI endpoint dropped its pre-2021 rows between
      August and September 2026 and the first refresh's overwrite lost 73k
      stored rows, and a refetch of yfinance's 100-event earnings window
      walks off its oldest event every time a new one is scheduled.

    A table that does not exist yet stores ``new_df`` verbatim, exactly as
    the ingest would have."""
    if mode not in ("append", "upsert"):
        raise ValueError(f"mode must be append|upsert, got {mode!r}")
    path = Path(path)
    old = deep.read_parquet(path) if path.exists() else pd.DataFrame()
    if new_df is None or not len(new_df):
        return pd.DataFrame(), int(len(old))
    new_df = new_df.copy()
    sort_cols = [sort_by] if isinstance(sort_by, str) else list(sort_by or [])

    def _finish(df: pd.DataFrame) -> pd.DataFrame:
        cols = [c for c in sort_cols if c in df.columns]
        if cols:
            df = df.sort_values(cols, kind="stable")
        df = df.reset_index(drop=True)
        deep.write_parquet(df, path)
        return df

    if not len(old):
        merged = _finish(new_df)
        return merged, int(len(merged))
    # keep the table's dtypes stable across refreshes (best effort)
    for c in new_df.columns:
        if c in old.columns and old[c].dtype != new_df[c].dtype:
            try:
                new_df[c] = new_df[c].astype(old[c].dtype)
            except (TypeError, ValueError):
                pass
    old_key = _row_key(old, dedupe_on)
    new_key = _row_key(new_df, dedupe_on)
    added = new_df[~new_key.isin(set(old_key))]
    if mode == "append":
        if not len(added):
            return pd.DataFrame(columns=new_df.columns), int(len(old))
        merged = _finish(pd.concat([old, added], ignore_index=True))
    else:
        kept = old[~old_key.isin(set(new_key))]
        merged = _finish(pd.concat([kept, new_df], ignore_index=True))
    return added.reset_index(drop=True), int(len(merged))


def merge_part(family: str, key: str, new_df: Optional[pd.DataFrame], dedupe_on: Sequence[str],
               sort_by=None, mode: str = "append") -> Tuple[pd.DataFrame, int]:
    """``merge_frame`` on ``parts/<key>.parquet`` of ``family``."""
    return merge_frame(deep.family_dir(family) / "parts" / f"{key}.parquet", new_df, dedupe_on,
                       sort_by, mode)


def _since(mx: Optional[str], overlap_days: int, default: str) -> str:
    """ISO date to fetch from: the part's newest date minus ``overlap_days``,
    or ``default`` for a part that does not exist yet."""
    if not mx:
        return default
    try:
        d = date.fromisoformat(str(mx)[:10])
    except ValueError:
        return default
    return (d - timedelta(days=max(0, int(overlap_days)))).isoformat()


def run_tails(family: str, keys: Sequence[str], fetch_tail: Callable[[str, str], Optional[pd.DataFrame]],
              *, key_col: str, dedupe_on: Sequence[str], sort_by, start_default: str,
              overlap_days: int = 1, workers: int = 4, deadline: Optional[float] = None,
              on_added: Optional[Callable[[str, pd.DataFrame], None]] = None,
              mode: str = "append",
              bulk: Optional[Callable[[Dict[str, str]], Dict[str, pd.DataFrame]]] = None) -> dict:
    """Extend every key's part from its own newest ``key_col`` (stalest first).
    ``fetch_tail(key, since_iso)`` returns the rows from ``since`` on (a
    refetch-everything family simply ignores ``since``). A key with no part
    is fetched from ``start_default``. Rows land through ``merge_frame`` in
    ``mode``. Stops SUBMITTING at ``deadline`` (a unix time); in-flight keys
    finish.

    ``bulk(since_by_key)`` (optional) returns ``{key: frame}`` for the keys ONE
    market-wide fetch serves — each frame exactly what ``fetch_tail(key,
    since)`` would return (the parity is tested per family). Every key it
    leaves out, and every key when it raises, is fetched on its own as before."""
    from concurrent.futures import ThreadPoolExecutor, as_completed
    man = deep.Manifest(family)
    mx = part_max(family, key_col)
    order = sorted(keys, key=lambda k: mx.get(k, ""))       # never-fetched first, then stalest
    since_by = {k: _since(mx.get(k), overlap_days, start_default) for k in order}
    pre: Dict[str, pd.DataFrame] = {}
    if bulk is not None and order:
        tb = time.time()
        try:
            pre = dict(bulk(since_by) or {})
        except Exception as e:                           # noqa: BLE001
            logger.warning(f"[deep.refresh] {family}: market-wide fetch failed ({e}) — every key on its own")
            pre = {}
        logger.info(f"[deep.refresh] {family}: {len(pre)}/{len(order)} keys from one market-wide fetch "
                    f"({time.time() - tb:.0f}s); {len(order) - len(pre)} on their own")
    t0 = time.time()
    n_ok = n_fail = n_new = 0
    stopped = False

    def _one(k: str):
        since = since_by[k]
        df = pre[k] if k in pre else fetch_tail(k, since)
        added, total = merge_part(family, k, df, dedupe_on, sort_by, mode=mode)
        return k, added, total

    logger.info(f"[deep.refresh] {family}: extending {len(order)} keys "
                f"({sum(1 for k in order if k not in mx)} without a part) | workers {workers}")
    with ThreadPoolExecutor(max_workers=max(1, int(workers))) as ex:
        futs = {}
        it = iter(order)

        def _submit_more():
            nonlocal stopped
            while len(futs) < 2 * max(1, int(workers)):
                if deadline is not None and time.time() > deadline:
                    stopped = True
                    return
                try:
                    k = next(it)
                except StopIteration:
                    return
                futs[ex.submit(_one, k)] = k
        _submit_more()
        while futs:
            for fut in as_completed(list(futs.keys())):
                key = futs.pop(fut)
                try:
                    k, added, total = fut.result()
                    man.mark_done(k, total, note=f"refreshed {_now_iso()[:10]}")
                    n_ok += 1
                    n_new += len(added)
                    if on_added is not None and len(added):
                        try:
                            on_added(k, added)
                        except Exception as e:           # noqa: BLE001
                            logger.debug(f"[deep.refresh] {family}/{k}: on_added failed ({e})")
                except Exception as e:                   # noqa: BLE001
                    man.mark_failed(key, repr(e))
                    n_fail += 1
                if (n_ok + n_fail) % 250 == 0:
                    logger.info(f"[deep.refresh] {family}: {n_ok + n_fail}/{len(order)} "
                                f"({n_fail} failed, {n_new:,} new rows) {time.time() - t0:.0f}s")
                break
            _submit_more()
    man.save()
    el = time.time() - t0
    logger.info(f"[deep.refresh] {family}: {n_ok} ok / {n_fail} failed / {n_new:,} new rows in "
                f"{el:.0f}s{' (BUDGET STOP)' if stopped else ''}")
    return {"family": family, "keys": len(order), "ok": n_ok, "failed": n_fail, "new_rows": n_new,
            "seconds": el, "budget_stop": stopped}


def _budget(deadline: Optional[float]) -> float:
    """Seconds left for a ``run_keys`` call (0 = unbounded)."""
    if deadline is None:
        return 0.0
    return max(1.0, deadline - time.time())


# ── families ─────────────────────────────────────────────────────────────────

def refresh_universe(**_) -> dict:
    tickers = deep.deep_universe(refresh=True)
    try:
        from src.data.deep import wiki
        wiki.mapping(refresh=True)
    except Exception as e:                               # noqa: BLE001
        logger.warning(f"[deep.refresh] wikidata mapping refresh failed: {e}")
    return {"family": "universe", "tickers": len(tickers)}


def _bulk_form345_last() -> Optional[date]:
    """Newest filing_date in the BULK insider table (the seam the live tail
    starts after), or None when the bulk table is absent."""
    import duckdb
    p = deep.DEEP_DIR / "form345.parquet"
    if not p.exists():
        return None
    con = duckdb.connect()
    try:
        last = con.execute(f"SELECT max(CAST(filing_date AS VARCHAR)) FROM '{p.as_posix()}' "
                           "WHERE quarter <> 'live'").fetchone()[0]
    finally:
        con.close()
    return date.fromisoformat(str(last)[:10]) if last else None


def live_form345_start(today: Optional[date] = None, recheck_days: int = 5) -> Optional[date]:
    """First day the live tail should ask EDGAR for: the day after the newest
    live part, pulled back up to ``recheck_days`` so a weekday whose index
    was unavailable when first asked (an EDGAR outage, not a holiday) is
    re-asked rather than skipped for good — an existing part is never
    re-fetched, so the recheck costs one request per missing day. Never
    earlier than the day after the bulk table's seam."""
    today = today or date.today()
    parts = sorted(p.stem for p in (deep.family_dir("form345_live") / "parts").glob("*.parquet"))
    bulk = _bulk_form345_last()
    floor = (bulk + timedelta(days=1)) if bulk else None
    if parts:
        try:
            last = date.fromisoformat(parts[-1][:10])
        except ValueError:
            last = None
    else:
        last = None
    if last is None and floor is None:
        return None
    start = (last + timedelta(days=1)) if last else floor
    recheck = today - timedelta(days=max(0, int(recheck_days)))
    start = min(start, recheck)
    if floor is not None:
        start = max(start, floor)
    return start


def refresh_form345_live(deadline: Optional[float] = None, workers: int = 4, **_) -> dict:
    """Every weekday from the seam to YESTERDAY, full market (the history is
    full-market; a universe-only tail would put a coverage discontinuity in
    one table). Today is deliberately excluded — see the module docstring."""
    from src.data.deep import form4_live
    today = date.today()
    start = live_form345_start(today)
    if start is None:
        return {"family": "form345_live", "skipped": "no bulk table and no live parts"}
    end = today - timedelta(days=1)
    if start > end:
        return {"family": "form345_live", "days": 0, "transactions": 0}
    r = form4_live.run_days(start, end, workers=workers, deadline=deadline)
    r["family"] = "form345_live"
    return r


def _cik_keys(tickers: Sequence[str]) -> List[str]:
    from src.data.deep import sec
    m = sec.cik_map()
    return [t for t in tickers if t.upper() in m]


def full_pass_anchor(family: str, margin_minutes: float = 15.0) -> Optional[datetime]:
    """When the last FULL pass of ``family`` STARTED (its completion stamp less
    its duration), less ``margin_minutes`` — how far back an incremental pass
    must reach so a filing accepted while that pass ran is not lost. None when
    the state holds no full pass."""
    st = load_state().get(family) or {}
    try:
        at = datetime.fromisoformat(str(st["at"]))
    except (KeyError, TypeError, ValueError):
        return None
    if at.tzinfo is None:
        at = at.replace(tzinfo=timezone.utc)
    return at - timedelta(seconds=float(st.get("seconds") or 0.0) + 60.0 * float(margin_minutes))


def refresh_sec_filings(deadline: Optional[float] = None, workers: int = 4,
                        tickers: Optional[Sequence[str]] = None, profile: Optional[str] = None,
                        **_) -> dict:
    """The RECENT block of every registrant's submissions JSON (real-time,
    up to 1,001 filings — more than any registrant files between two nights),
    merged on accession. Returns the tickers whose new filings carry XBRL so
    ``companyfacts`` refetches only those.

    The PRE-OPEN run re-reads only the companies with a filing accepted since
    the last full pass started (EDGAR's live feed, ``sec.current_filer_ciks``):
    every other company's recent block holds no accession the store lacks —
    each company re-read is the very call the full pass makes (2026-09-29: the
    3,239-call pass took 16.8 min of the 25-min pre-open). When the feed cannot
    prove it reaches back that far it is the full pass, as before. The result
    carries ``incremental`` so the run stamps ``sec_filings_incremental`` and the
    nightly still takes the full pass."""
    from src.data.deep import sec
    keys = _cik_keys(tickers or deep.deep_universe())
    incremental = None
    if profile == "preopen" and tickers is None:
        anchor = full_pass_anchor("sec_filings")
        ciks = sec.current_filer_ciks(anchor) if anchor is not None else None
        if ciks is None:
            logger.warning(f"[deep.refresh] sec_filings: EDGAR's live feed cannot prove coverage back to the "
                           f"last full pass ({anchor}) — FULL pass")
        else:
            cm = sec.cik_map()
            keys = [k for k in keys if str(cm.get(k.upper(), "")).zfill(10) in ciks]
            incremental = {"since": anchor.isoformat(), "filers": len(ciks), "keys": len(keys)}
            logger.info(f"[deep.refresh] sec_filings: {len(ciks)} filers on EDGAR's live feed since the last "
                        f"full pass ({anchor:%Y-%m-%d %H:%M} UTC) — {len(keys)} universe companies re-read")
    xbrl: set = set()

    def _tail(tk: str, _since: str) -> pd.DataFrame:
        cik = sec.cik_map().get(tk.upper())
        return sec.fetch_filings(cik, tk, recent_only=True) if cik else pd.DataFrame()

    def _added(tk: str, added: pd.DataFrame) -> None:
        if "is_xbrl" in added.columns and bool((pd.to_numeric(added["is_xbrl"], errors="coerce")
                                                  .fillna(0) > 0).any()):
            xbrl.add(tk.upper())

    r = run_tails("sec_filings", keys, _tail, key_col="filing_date", dedupe_on=["accession"],
                  sort_by="filing_date", start_default="1993-01-01", overlap_days=0,
                  workers=workers, deadline=deadline, on_added=_added)
    if r["new_rows"]:
        deep.consolidate("sec_filings")
    r["xbrl_tickers"] = sorted(xbrl)
    if incremental is not None:
        r["incremental"] = incremental
    return r


def companyfacts_stale_tickers() -> List[str]:
    """Tickers whose newest FINANCIAL XBRL filing in their ``sec_filings``
    part (10-K/10-Q/20-F/40-F/6-K and the transition forms, ``is_xbrl``) is
    dated after the newest ``filed`` in their ``companyfacts`` part — the only
    names a refetch can change. Derived from the PARTS, not from what ran
    tonight, so it is right whether or not ``sec_filings`` refreshed in the
    same invocation. 8-Ks are excluded on purpose: their inline-XBRL cover
    page flags ``is_xbrl`` without adding a fact, and would refetch every
    company with an 8-K since its last 10-Q, every night."""
    import duckdb
    sf = deep.family_dir("sec_filings") / "parts"
    cf = deep.family_dir("companyfacts") / "parts"
    if not any(sf.glob("*.parquet")):
        return []
    forms = "|".join(XBRL_FORMS).replace("-", "\\-")
    con = duckdb.connect()
    try:
        newest_xbrl = dict(con.execute(
            "SELECT filename, max(CAST(filing_date AS VARCHAR)) FROM "
            f"read_parquet('{(sf / '*.parquet').as_posix()}', union_by_name=true, filename=true) "
            f"WHERE COALESCE(is_xbrl, 0) > 0 AND regexp_matches(CAST(form AS VARCHAR), '^({forms})') "
            "GROUP BY filename").fetchall())
        newest_fact: Dict[str, str] = {}
        if any(cf.glob("*.parquet")):
            newest_fact = dict(con.execute(
                "SELECT filename, max(CAST(filed AS VARCHAR)) FROM "
                f"read_parquet('{(cf / '*.parquet').as_posix()}', union_by_name=true, filename=true) "
                "GROUP BY filename").fetchall())
    finally:
        con.close()
    facts = {Path(str(k)).stem: (str(v)[:10] if v else "") for k, v in newest_fact.items()}
    out = []
    for fn, mx in newest_xbrl.items():
        tk = Path(str(fn)).stem
        if mx and str(mx)[:10] > facts.get(tk, ""):
            out.append(tk)
    return sorted(out)


def refresh_companyfacts(deadline: Optional[float] = None, workers: int = 4,
                         tickers: Optional[Sequence[str]] = None, **_) -> dict:
    """Full refetch (the facts JSON IS the whole history, restatements as
    later rows) for the tickers whose filings outran their facts
    (``companyfacts_stale_tickers``) plus any name the family has never
    seen. A 404 (no XBRL: funds, some ADRs) stays done-with-0."""
    from src.data.deep import sec
    man = deep.Manifest("companyfacts")
    universe = deep.deep_universe()
    keys = set(t.upper() for t in (tickers if tickers is not None else companyfacts_stale_tickers()))
    keys |= {t for t in universe if t not in man.done}
    keys = sorted(_cik_keys(sorted(keys)))
    if not keys:
        return {"family": "companyfacts", "keys": 0, "ok": 0, "failed": 0}

    def _guarded(tk: str):
        """The facts JSON is authoritative for the whole history, so the part
        is overwritten — unless the refetch came back materially SMALLER than
        what is stored (an API hiccup, a taxonomy the API dropped), in which
        case the stored part is kept and the shortfall logged."""
        df = sec.facts_for_ticker(tk)
        part = deep.family_dir("companyfacts") / "parts" / f"{tk}.parquet"
        if df is not None and part.exists():
            old = deep.read_parquet(part)
            if len(old) and len(df) < 0.9 * len(old):
                logger.warning(f"[deep.refresh] companyfacts/{tk}: refetch returned {len(df):,} rows "
                               f"against {len(old):,} stored — keeping the stored part")
                return old
        return df

    r = deep.run_keys("companyfacts", keys, _guarded, workers=min(workers, 4),
                      budget_seconds=_budget(deadline), force=True)
    if r["ok"]:
        deep.consolidate("companyfacts")
    r["family"] = "companyfacts"
    return r


def refresh_bars30m_full(deadline: Optional[float] = None, workers: int = 6, **_) -> dict:
    from src.data.deep import polygon_deep
    return run_tails("bars30m_full", deep.deep_universe(),
                     lambda tk, since: polygon_deep.bars30m_full(tk, start=since),
                     key_col="ts", dedupe_on=["ts"], sort_by="ts", start_default=polygon_deep.BARS_FROM,
                     overlap_days=1, workers=workers, deadline=deadline)


def refresh_polygon_news(deadline: Optional[float] = None, workers: int = 6, **_) -> dict:
    from src.data.deep import polygon_deep
    r = run_tails("polygon_news", deep.deep_universe(),
                  lambda tk, since: polygon_deep.polygon_news(tk, start=since),
                  key_col="published_utc", dedupe_on=["article_id"], sort_by="published_utc",
                  start_default=polygon_deep.NEWS_FROM, overlap_days=3, workers=workers, deadline=deadline)
    if r["new_rows"]:
        deep.consolidate("polygon_news")
    return r


def refresh_short_volume(deadline: Optional[float] = None, workers: int = 6, **_) -> dict:
    from src.data.deep import polygon_deep
    r = run_tails("short_volume", deep.deep_universe(),
                  lambda tk, since: polygon_deep.short_volume(tk, start=since),
                  key_col="date", dedupe_on=["date"], sort_by="date", start_default="2024-01-01",
                  overlap_days=0, workers=workers, deadline=deadline,
                  bulk=polygon_deep.short_volume_bulk)
    if r["new_rows"]:
        deep.consolidate("short_volume")
    return r


def refresh_short_interest(deadline: Optional[float] = None, workers: int = 6, **_) -> dict:
    from src.data.deep import polygon_deep
    r = run_tails("short_interest", deep.deep_universe(),
                  lambda tk, since: polygon_deep.short_interest(tk, start=since),
                  key_col="settlement_date", dedupe_on=["settlement_date"], sort_by="settlement_date",
                  start_default="2017-01-01", overlap_days=0, workers=workers, deadline=deadline,
                  bulk=polygon_deep.short_interest_bulk)
    if r["new_rows"]:
        deep.consolidate("short_interest")
    return r


def _recent_month_windows(today: Optional[date] = None, back: int = 1, ahead: int = 2) -> List[str]:
    """Month windows from ``back`` months ago through ``ahead`` months ahead —
    dividends are DECLARED weeks before their ex-date, and the family is keyed
    by ex-date, so the forward windows are where a fresh declaration lands."""
    from src.data.deep import polygon_deep
    today = today or date.today()
    first = today.replace(day=1)
    for _ in range(back):
        first = (first - timedelta(days=1)).replace(day=1)
    last = today.replace(day=1)
    for _ in range(ahead + 1):
        last = (last.replace(day=28) + timedelta(days=4)).replace(day=1)
    last = last - timedelta(days=1)
    return [f"{a}..{b}" for a, b in polygon_deep._month_windows(first.isoformat(), last.isoformat())]


def _refresh_month_family(family: str, fn: Callable[[str], pd.DataFrame], deadline: Optional[float]) -> dict:
    keys = _recent_month_windows()
    r = deep.run_keys(family, keys, fn, workers=2, budget_seconds=_budget(deadline),
                      part_name=lambda k: k.replace("..", "_"), force=True)
    if r["ok"]:
        deep.consolidate(family)
    r["family"] = family
    return r


def refresh_dividends(deadline: Optional[float] = None, **_) -> dict:
    from src.data.deep import polygon_deep
    return _refresh_month_family("dividends", polygon_deep.dividends_month, deadline)


def refresh_splits(deadline: Optional[float] = None, **_) -> dict:
    from src.data.deep import polygon_deep
    return _refresh_month_family("splits", polygon_deep.splits_month, deadline)


def _table_max(name: str, col: str) -> Optional[str]:
    import duckdb
    p = deep.DEEP_DIR / f"{name}.parquet"
    if not p.exists():
        return None
    con = duckdb.connect()
    try:
        v = con.execute(f'SELECT max(CAST("{col}" AS VARCHAR)) FROM \'{p.as_posix()}\'').fetchone()[0]
    finally:
        con.close()
    return str(v) if v is not None else None


def refresh_regsho(deadline: Optional[float] = None, **_) -> dict:
    """The Reg SHO threshold lists of every completed day not yet held — normally
    yesterday's five (`src/data/deep/regsho.py`; Cboe posts by ~03:05 ET, so the
    08:30 pre-open run carries them into the session snapshot). A host that
    rate-limits fails its keys at once (retried next run) instead of holding the
    pre-open run (`regsho.MAX_WAIT_S`)."""
    from src.data.deep import regsho
    regsho.MAX_WAIT_S = 120.0
    # the last three weeks only: a gap further back is the backfill's job
    # (`python -m src.data.deep regsho`) — at NYSE's 2 s pace it would hold the lane
    since = max(regsho.START, date.today() - timedelta(days=21))
    r = regsho.run(since=since, workers=3, budget_seconds=_budget(deadline))
    r["family"] = "regsho"
    return r


def refresh_borrow(**_) -> dict:
    """IBKR's borrow file as one row per name per day (`src/data/deep/borrow.py`):
    every finished day of our own archive summarised once, then
    ``borrow_daily.parquet`` rebuilt from those summaries."""
    from src.data.deep import borrow
    return {"family": "borrow", "rows": int(borrow.build_daily())}


def refresh_context(deadline: Optional[float] = None, **_) -> dict:
    """market_daily / fama_french / dix: one call each, overwritten. cot_tff:
    the current year's file (and January re-reads December's year). FRED:
    the ALFRED real-time window since the stored newest first print."""
    from src.data.deep import context
    out = {"family": "context"}
    # full-series refetches UPSERTED on their primary key: revised values
    # (an adjusted close after a dividend) replace, rows a partial response
    # lacks (a symbol yfinance skipped tonight) are kept
    for name, fn, ident, sort in (("market_daily", context.market_daily, ["symbol", "date"], ["symbol", "date"]),
                                  ("fama_french", context.fama_french, ["date"], ["date"]),
                                  ("dix", context.dix, ["date"], ["date"])):
        try:
            df = fn()
            if not len(df):
                logger.warning(f"[deep.refresh] context/{name}: fetch returned nothing — previous table kept")
                out[name] = 0
                continue
            added, total = merge_frame(deep.DEEP_DIR / f"{name}.parquet", df, ident, sort, mode="upsert")
            out[name] = total
            logger.info(f"[deep.refresh] context/{name}: {len(df):,} fetched, {len(added):,} new -> {total:,} stored")
        except Exception as e:                           # noqa: BLE001
            logger.warning(f"[deep.refresh] context/{name} failed: {e}")
            out[name] = f"failed: {e}"
    # CFTC: this year's file changes weekly; a year boundary needs last year's final file too
    today = date.today()
    years = [str(today.year)] + ([str(today.year - 1)] if today.month == 1 else [])
    try:
        rc = deep.run_keys("cot_tff", years, context.cot_tff_year, workers=1,
                           budget_seconds=_budget(deadline), force=True)
        if rc["ok"]:
            deep.consolidate("cot_tff")
        out["cot_tff"] = rc["rows"]
    except Exception as e:                               # noqa: BLE001
        logger.warning(f"[deep.refresh] context/cot_tff failed: {e}")
        out["cot_tff"] = f"failed: {e}"
    try:
        out["fred_vintages"] = refresh_fred()
    except Exception as e:                               # noqa: BLE001
        logger.warning(f"[deep.refresh] context/fred_vintages failed: {e}")
        out["fred_vintages"] = f"failed: {e}"
    return out


def merge_vintage_update(existing: pd.DataFrame, new: pd.DataFrame) -> pd.DataFrame:
    """Fold a real-time-window fetch into the stored vintages. Per
    (series_id, date, value): ``realtime_start`` = the EARLIEST seen (the
    window clamps older values to its own start; the stored row has the true
    first print), ``realtime_end`` = the NEWER fetch's (only it knows a value
    was superseded), ``vintage`` = the STORED row's flag when the row existed
    (a series backfilled from plain FRED keeps its ``False`` — the window
    reports it as a vintage while its ``realtime_start`` is still the plain
    observation date), the newer flag for a row seen for the first time. Pure."""
    cols = ["series_id", "date", "realtime_start", "realtime_end", "value", "vintage"]
    if new is None or not len(new):
        return existing.reset_index(drop=True) if len(existing) else pd.DataFrame(columns=cols)
    if existing is None or not len(existing):
        return new[cols].sort_values(["series_id", "date", "realtime_start"]).reset_index(drop=True)
    e = existing[cols].copy()
    n = new[cols].copy()
    e["_src"], n["_src"] = 0, 1
    both = pd.concat([e, n], ignore_index=True)
    for c in ("date", "realtime_start", "realtime_end"):
        both[c] = both[c].astype(str)
    both["value"] = pd.to_numeric(both["value"], errors="coerce")
    grp = ["series_id", "date", "value"]
    agg = both.groupby(grp, dropna=False).agg(rs_min=("realtime_start", "min"),
                                              vint_first=("vintage", "first"))
    last = (both.sort_values("_src", kind="stable")
                .drop_duplicates(grp, keep="last")
                .merge(agg, left_on=grp, right_index=True, how="left"))
    last["realtime_start"] = last["rs_min"]
    last["vintage"] = last["vint_first"].astype(bool)   # the stored flag when the row existed
    out = last.drop(columns=["_src", "rs_min", "vint_first"])
    return out[cols].sort_values(["series_id", "date", "realtime_start"]).reset_index(drop=True)


def refresh_fred(overlap_days: int = 3) -> int:
    """New ALFRED vintages since the stored newest first print, merged in."""
    from config.settings import settings
    from src.data.deep import context
    key = settings.fred_api_key
    if not key:
        logger.warning("[deep.refresh] no FRED key configured")
        return 0
    p = deep.DEEP_DIR / "fred_vintages.parquet"
    existing = deep.read_parquet(p) if p.exists() else pd.DataFrame()
    if not len(existing):
        df = context.fred_vintages()
        return deep.write_parquet(df, p) if len(df) else 0
    mx = _table_max("fred_vintages", "realtime_start") or "2000-01-01"
    since = (date.fromisoformat(mx[:10]) - timedelta(days=overlap_days)).isoformat()
    # three at a time (FRED allows 120 requests a minute; one after another the
    # series took 195 s of the pre-open's context lane), results in series order
    from concurrent.futures import ThreadPoolExecutor

    def _upd(sid):
        try:
            return context.fred_series_update(sid, key, since)
        except Exception as e:                           # noqa: BLE001
            logger.warning(f"[deep.refresh] FRED {sid} update failed: {e}")
            return None

    with ThreadPoolExecutor(max_workers=3) as ex:
        frames = list(ex.map(_upd, list(context.FRED_SERIES)))
    new = pd.concat([f for f in frames if f is not None and len(f)], ignore_index=True) if any(
        f is not None and len(f) for f in frames) else pd.DataFrame()
    merged = merge_vintage_update(existing, new)
    n = deep.write_parquet(merged, p)
    logger.info(f"[deep.refresh] fred_vintages: {len(new):,} window rows since {since} -> {n:,} stored")
    return int(len(new))


# Row identity per Quiver family: the columns that name the EVENT. The congress
# rows also carry ExcessReturn / PriceChange / SPYChange, which are returns
# since the trade and change every day — a full-row identity would never
# collapse a repeat sighting, so those are excluded and the newer row wins.
QUIVER_IDENTITY = {
    "quiver_congress": ["ticker", "Representative", "BioGuideID", "ReportDate", "TransactionDate",
                        "Transaction", "Range", "House"],
    "quiver_lobbying": ["ticker", "Date", "Client", "Registrant", "Amount", "Issue", "Specific_Issue"],
    "quiver_contracts": ["ticker", "Date", "Description", "Agency", "Amount", "action_date"],
}
QUIVER_LIVE = {
    "quiver_congress": "/live/congresstrading",
    "quiver_lobbying": "/live/lobbying",
    "quiver_contracts": "/live/govcontractsall",
}
QUIVER_DATE = {"quiver_congress": "ReportDate", "quiver_lobbying": "Date", "quiver_contracts": "Date"}


def split_live_rows(rows: List[dict], universe: Iterable[str]) -> Dict[str, pd.DataFrame]:
    """A market-wide live payload -> ``{ticker: rows}`` for the universe,
    shaped exactly like ``quiver_deep.fetch`` shapes the historical rows."""
    if not rows:
        return {}
    df = pd.DataFrame(rows)
    df.columns = [str(c).strip().replace(" ", "_") for c in df.columns]
    if "Ticker" not in df.columns:
        return {}
    df["ticker"] = df["Ticker"].astype(str).str.strip().str.upper()
    df = df.drop(columns=["Ticker"])
    keep = {t.upper() for t in universe}
    out = {}
    for tk, g in df[df["ticker"].isin(keep)].groupby("ticker"):
        g = g.copy()
        cols = ["ticker"] + [c for c in g.columns if c != "ticker"]
        g = g[cols]
        for c in g.columns:
            if c != "ticker" and g[c].dtype == object:
                g[c] = g[c].map(lambda v: v if (v is None or isinstance(v, (str, int, float, bool))) else str(v))
        out[str(tk)] = g.reset_index(drop=True)
    return out


def refresh_quiver_live(**_) -> dict:
    """One market-wide call per family (the same endpoints the live tick
    reads), split by ticker into the per-ticker parts. Field-for-field the
    live payloads match the stored columns (verified 2026-09-23 against the
    tick's daily cache)."""
    from src.data.deep import quiver_deep
    out = {"family": "quiver_live"}
    universe = deep.deep_universe()
    for fam, path in QUIVER_LIVE.items():
        try:
            r = deep.http_get(quiver_deep._BASE + path, headers=quiver_deep._headers(), timeout=120,
                              limiter=quiver_deep._LIMITER)
            if r is None or r.status_code != 200:
                out[fam] = f"HTTP {getattr(r, 'status_code', None)}"
                logger.warning(f"[deep.refresh] {fam}: live endpoint {out[fam]} — nothing merged")
                continue
            rows = r.json()
            if not isinstance(rows, list):
                out[fam] = "unexpected payload"
                continue
            per = split_live_rows(rows, universe)
            n_new = 0
            for tk, g in per.items():
                added, _ = merge_part(fam, tk, g, QUIVER_IDENTITY[fam], QUIVER_DATE[fam])
                n_new += len(added)
            if n_new:
                deep.consolidate(fam)
            out[fam] = {"live_rows": len(rows), "tickers": len(per), "new_rows": n_new}
            logger.info(f"[deep.refresh] {fam}: {len(rows):,} live rows, {len(per)} universe tickers, "
                        f"{n_new:,} new")
        except Exception as e:                           # noqa: BLE001
            logger.warning(f"[deep.refresh] {fam} live top-up failed: {e}")
            out[fam] = f"failed: {e}"
    return out


def refresh_quiver_dpi(deadline: Optional[float] = None, workers: int = 2, **_) -> dict:
    """Per-ticker refetch of the whole DPI series, UPSERTED on the date — the
    endpoint's history is not stable (its pre-2021 rows vanished between
    August and September 2026), so a stored day is never dropped."""
    from src.data.deep import quiver_deep
    r = run_tails("quiver_dpi", deep.deep_universe(), lambda tk, _since: quiver_deep.fetch("quiver_dpi", tk),
                  key_col="Date", dedupe_on=["Date"], sort_by="Date", start_default="", overlap_days=0,
                  workers=min(workers, 2), deadline=deadline, mode="upsert")
    if r["ok"]:
        deep.consolidate("quiver_dpi")
    return r


def refresh_quiver_history(deadline: Optional[float] = None, workers: int = 2, **_) -> dict:
    """Weekly per-ticker refetch of the three event families — the safety net
    under the nightly live top-up (a disclosure filed weeks late can scroll
    out of the live window's most-recent rows before a night sees it).
    APPENDED on the event identity: the payload repeats identities on purpose
    and the stored rows are never rewritten."""
    from src.data.deep import quiver_deep
    out = {"family": "quiver_history"}
    for fam in ("quiver_congress", "quiver_lobbying", "quiver_contracts"):
        r = run_tails(fam, deep.deep_universe(), lambda tk, _since, f=fam: quiver_deep.fetch(f, tk),
                      key_col=QUIVER_DATE[fam], dedupe_on=QUIVER_IDENTITY[fam], sort_by=QUIVER_DATE[fam],
                      start_default="", overlap_days=0, workers=min(workers, 2), deadline=deadline,
                      mode="append")
        if r["new_rows"]:
            deep.consolidate(fam)
        out[fam] = r
        if r.get("budget_stop"):
            break
    return out


def _refresh_file_family(family: str, keys: List[str], fn, workers: int, deadline: Optional[float]) -> dict:
    r = deep.run_keys(family, keys, fn, workers=workers, budget_seconds=_budget(deadline))
    if r["ok"]:
        deep.consolidate(family)
    r["family"] = family
    return r


def refresh_ftd(deadline: Optional[float] = None, **_) -> dict:
    from src.data.deep import ftd
    ftd._links = {}
    return _refresh_file_family("ftd", sorted(ftd.file_links().keys()), ftd.fetch_file, 3, deadline)


def refresh_form13f(deadline: Optional[float] = None, **_) -> dict:
    from src.data.deep import form13f
    form13f._links = {}
    return _refresh_file_family("form13f", sorted(form13f.file_links().keys()), form13f.fetch_quarter, 2, deadline)


def prune_live_parts_covered_by_bulk() -> int:
    """Once a bulk quarter publishes, the live day parts inside it are
    redundant: drop every live part dated at or before the bulk seam and
    re-consolidate, so the union of the two tables stays duplicate-free."""
    seam = _bulk_form345_last()
    if seam is None:
        return 0
    d = deep.family_dir("form345_live") / "parts"
    dropped = 0
    for p in sorted(d.glob("*.parquet")):
        try:
            day = date.fromisoformat(p.stem[:10])
        except ValueError:
            continue
        if day <= seam:
            p.unlink()
            dropped += 1
    if dropped:
        logger.info(f"[deep.refresh] form345_live: dropped {dropped} day parts now covered by the "
                    f"bulk table (seam {seam})")
        if any(d.glob("*.parquet")):
            deep.consolidate("form345_live")
    return dropped


def refresh_form345(deadline: Optional[float] = None, **_) -> dict:
    """Bulk quarters the SEC has published since the last check (a 404 is the
    unpublished current quarter — recorded failed, retried next time)."""
    from src.data.deep import form345
    r = deep.run_keys("form345", form345.quarter_keys(), form345.fetch_quarter, workers=1,
                      budget_seconds=_budget(deadline))
    r["family"] = "form345"
    if r["ok"]:
        deep.consolidate("form345")
        r["live_parts_pruned"] = prune_live_parts_covered_by_bulk()
    return r


def refresh_wiki(deadline: Optional[float] = None, **_) -> dict:
    from src.data.deep import wiki
    m = wiki.mapping()
    keys = [t for t in deep.deep_universe() if t in m]
    r = run_tails("wiki", keys, lambda tk, since: wiki.pageviews_for_ticker(tk, start=since.replace("-", "")),
                  key_col="date", dedupe_on=["date"], sort_by="date", start_default="2015-07-01",
                  overlap_days=1, workers=1, deadline=deadline)
    if r["new_rows"]:
        deep.consolidate("wiki")
    return r


def refresh_ticker_details(deadline: Optional[float] = None, workers: int = 6, **_) -> dict:
    from src.data.deep import polygon_deep
    r = deep.run_keys("ticker_details", deep.deep_universe(), polygon_deep.ticker_details, workers=workers,
                      budget_seconds=_budget(deadline), force=True)
    if r["ok"]:
        deep.consolidate("ticker_details")
    r["family"] = "ticker_details"
    return r


def refresh_ipos(**_) -> dict:
    from src.data.deep import polygon_deep
    df = polygon_deep.ipos()
    n = deep.write_parquet(df, deep.DEEP_DIR / "ipos.parquet") if len(df) else 0
    if not n:
        logger.warning("[deep.refresh] ipos: fetch returned nothing — previous table kept")
    return {"family": "ipos", "rows": n}


def refresh_delisted(deadline: Optional[float] = None, workers: int = 6, since: str = "2021-01-01", **_) -> dict:
    from src.data.deep import polygon_deep
    lst = polygon_deep.delisted_tickers()
    out = {"family": "delisted", "rows": 0}
    if not len(lst):
        logger.warning("[deep.refresh] delisted: list fetch returned nothing — previous table kept")
        return out
    out["rows"] = deep.write_parquet(lst, deep.DEEP_DIR / "delisted.parquet")
    recent = lst[(lst["delisted_utc"].fillna("") >= since) & (lst["type"].isin(["CS", "ADRC"]))]
    keys = sorted(set(recent["ticker"].astype(str)))
    out["bars1d"] = deep.run_keys("bars1d_delisted", keys, polygon_deep.bars1d, workers=workers,
                                  budget_seconds=_budget(deadline))
    out["bars30m"] = deep.run_keys("bars30m_delisted", keys, polygon_deep.bars30m_full, workers=workers,
                                   budget_seconds=_budget(deadline))
    return out


YF_IDENTITY = {"yf_earnings": ["event_ts"],
               "yf_analyst": ["grade_date", "firm", "to_grade", "from_grade", "action"],
               "yf_shares": ["date"]}
# earnings and shares have a true key per event / day (a refetch brings the
# restated consensus and the corrected share count); analyst actions can
# legitimately repeat an identity, so they are appended
YF_MODE = {"yf_earnings": "upsert", "yf_analyst": "append", "yf_shares": "upsert"}
YF_SORT = {"yf_earnings": "event_ts", "yf_analyst": "grade_date", "yf_shares": "date"}


def refresh_yf(deadline: Optional[float] = None, **_) -> dict:
    """Weekly per-ticker refetch of the three yfinance tables, one worker with
    the repo's 429 backoff, MERGED into the parts (``YF_MODE``): the
    100-event earnings window walks off its oldest event every time a new
    one is scheduled, so an overwrite would erode the history. Stalest first
    by the manifest's stamp; a budget or rate-limit stop leaves the family
    due, and the next run continues where this one stopped."""
    from src.data.deep import yf_deep
    man = deep.Manifest("yf")
    order = sorted(deep.deep_universe(), key=lambda t: str((man.done.get(t) or {}).get("at", "")))
    t0 = time.time()
    n_ok = n_fail = n_new = strikes = 0
    stopped = False
    logger.info(f"[deep.refresh] yf: refetching {len(order)} tickers, stalest first")
    for tk in order:
        if deadline is not None and time.time() > deadline:
            stopped = True
            break
        try:
            res = yf_deep.fetch_ticker(tk)
            strikes = 0
        except Exception as e:                           # noqa: BLE001
            if yf_deep._is_rate_limit(e):
                wait = yf_deep._BACKOFF[min(strikes, len(yf_deep._BACKOFF) - 1)]
                strikes += 1
                logger.warning(f"[deep.refresh] yf: rate limited on {tk}; sleeping {wait}s (strike {strikes})")
                time.sleep(wait)
                if strikes >= len(yf_deep._BACKOFF):
                    logger.error("[deep.refresh] yf: three consecutive rate limits — stopping this run")
                    stopped = True
                    break
                continue
            man.mark_failed(tk, repr(e))
            n_fail += 1
            continue
        rows = 0
        for fam, df in res.items():
            added, total = merge_part(fam, tk, df, YF_IDENTITY[fam], YF_SORT[fam], mode=YF_MODE[fam])
            n_new += len(added)
            rows += total
        man.mark_done(tk, rows, note=",".join(sorted(res)))
        n_ok += 1
        if n_ok % 250 == 0:
            logger.info(f"[deep.refresh] yf: {n_ok}/{len(order)} ({n_fail} failed, {n_new:,} new rows) "
                        f"{time.time() - t0:.0f}s")
    man.save()
    if n_ok:
        for sub in ("yf_earnings", "yf_analyst", "yf_shares"):
            deep.consolidate(sub)
    el = time.time() - t0
    logger.info(f"[deep.refresh] yf: {n_ok} ok / {n_fail} failed / {n_new:,} new rows in {el:.0f}s"
                f"{' (BUDGET STOP)' if stopped else ''}")
    return {"family": "yf", "keys": len(order), "ok": n_ok, "failed": n_fail, "new_rows": n_new,
            "seconds": el, "budget_stop": stopped}


REFRESHERS: Dict[str, Callable[..., dict]] = {
    "universe": refresh_universe,
    "form345_live": refresh_form345_live,
    "sec_filings": refresh_sec_filings,
    "companyfacts": refresh_companyfacts,
    "bars30m_full": refresh_bars30m_full,
    "polygon_news": refresh_polygon_news,
    "short_volume": refresh_short_volume,
    "dividends": refresh_dividends,
    "splits": refresh_splits,
    "context": refresh_context,
    "regsho": refresh_regsho,
    "borrow": refresh_borrow,
    "quiver_live": refresh_quiver_live,
    "short_interest": refresh_short_interest,
    "quiver_dpi": refresh_quiver_dpi,
    "ftd": refresh_ftd,
    "form13f": refresh_form13f,
    "form345": refresh_form345,
    "wiki": refresh_wiki,
    "ticker_details": refresh_ticker_details,
    "ipos": refresh_ipos,
    "delisted": refresh_delisted,
    "yf": refresh_yf,
    "quiver_history": refresh_quiver_history,
}


def _set_sec_rate(seconds: float) -> None:
    """One spacing for every EDGAR client this process runs — the live
    pipeline makes its own EDGAR calls under the same 10 req/s fair-access
    ceiling, so the refresh leaves it headroom."""
    from src.data.deep import form4_live, sec
    for mod in (sec, form4_live):
        lim = getattr(mod, "_LIMITER", None)
        if lim is not None:
            lim.min_interval = float(seconds)


LOCK_NAME = "refresh.lock"


def _try_lock(path: Path):
    """An OS-level EXCLUSIVE lock on ``path``, held until the returned handle
    is released or the process dies — the OS drops it either way, so a killed
    run never leaves a stale lock and no PID bookkeeping is needed. Returns
    the open handle, or None when another process (or another handle in this
    one) holds it."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fh = open(path, "a+b")
    try:
        if os.name == "nt":
            import msvcrt
            fh.seek(0)
            msvcrt.locking(fh.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            import fcntl
            fcntl.flock(fh.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        fh.close()
        return None
    return fh


def _unlock(fh) -> None:
    try:
        if os.name == "nt":
            import msvcrt
            fh.seek(0)
            msvcrt.locking(fh.fileno(), msvcrt.LK_UNLCK, 1)
    except OSError:
        pass
    finally:
        fh.close()


def run(families: Optional[Sequence[str]] = None, *, force: bool = False, budget_seconds: float = 14400.0,
        workers: int = 6, sec_rate: float = 0.2, now: Optional[datetime] = None,
        lanes: Optional[Dict[str, str]] = None, profile: Optional[str] = None,
        family_workers: Optional[Dict[str, int]] = None) -> dict:
    """Refresh every DUE family in ``FAMILIES`` order under one wall-clock
    budget. A family that completes is stamped in the state file; one that
    fails or is cut by the budget is not, so it is due again next time.

    SINGLE INSTANCE across processes (``refresh.lock``): the broker watchdog
    force-exits the scheduler inside the refresh window on most nights
    (09-17 01:21, 09-18 01:44, 09-21 01:26, 09-23 02:23 — the recurring
    IBKR gateway wedge), which ORPHANS the running refresh subprocess while
    the supervisor relaunches the scheduler; a manual run can also meet the
    nightly slot. Two refreshes merging the same parts would race, so the
    second one returns at once with ``skipped``."""
    lock = _try_lock(deep.DEEP_DIR / LOCK_NAME)
    if lock is None:
        logger.warning(f"[deep.refresh] another refresh holds {deep.DEEP_DIR / LOCK_NAME} — "
                       "not starting a second one")
        return {"seconds": 0.0, "ran": [], "completed": [], "failed": [], "not_finished": [],
                "not_due": [], "results": {}, "skipped": "another refresh is running"}
    try:
        return _run_locked(families, force=force, budget_seconds=budget_seconds, workers=workers,
                           sec_rate=sec_rate, now=now, lanes=lanes, profile=profile,
                           family_workers=family_workers)
    finally:
        _unlock(lock)


def _run_locked(families: Optional[Sequence[str]], *, force: bool, budget_seconds: float,
                workers: int, sec_rate: float, now: Optional[datetime],
                lanes: Optional[Dict[str, str]] = None, profile: Optional[str] = None,
                family_workers: Optional[Dict[str, int]] = None) -> dict:
    """``lanes`` (family -> lane name): the lanes run concurrently, each one's
    families in ``FAMILIES`` order; a family whose lane is "" runs BEFORE the
    lanes start; None = every family one after another. ``profile`` reaches every
    refresher (sec_filings reads it); ``family_workers`` overrides ``workers``
    per family. A result carrying ``incremental`` is stamped ``<family>_incremental``
    — the family's own stamp is its last FULL run."""
    state = load_state()
    now = now or datetime.now(timezone.utc)
    t0 = time.time()
    deadline = t0 + float(budget_seconds) if budget_seconds and budget_seconds > 0 else None
    _set_sec_rate(sec_rate)
    wanted = [f.strip() for f in families if f.strip()] if families else FAMILY_NAMES
    unknown = [f for f in wanted if f not in REFRESHERS]
    if unknown:
        raise SystemExit(f"unknown families {unknown}; choose from {FAMILY_NAMES}")
    plan = [(f, c) for f, c in FAMILIES if f in wanted and (force or due(f, c, state, now))]
    skipped_not_due = [f for f in wanted if f not in {p for p, _ in plan}]
    logger.info(f"[deep.refresh] due: {[f for f, _ in plan]} | not due: {skipped_not_due} | "
                f"budget {budget_seconds:.0f}s")
    results: Dict[str, dict] = {}
    remaining: List[str] = []
    guard = threading.Lock()            # the lanes share results / remaining / the state file

    def _one(fam: str) -> None:
        if deadline is not None and time.time() > deadline:
            with guard:
                remaining.append(fam)
            return
        t1 = time.time()
        try:
            r = REFRESHERS[fam](deadline=deadline, workers=(family_workers or {}).get(fam, workers),
                                profile=profile)
            cut = bool(r.get("budget_stop")) or any(
                isinstance(v, dict) and v.get("budget_stop") for v in r.values())
            with guard:
                results[fam] = r
                if cut:
                    remaining.append(fam)
                else:
                    stamp = f"{fam}_incremental" if r.get("incremental") else fam
                    state[stamp] = {"at": _now_iso(), "seconds": round(time.time() - t1, 1),
                                  "summary": {k: v for k, v in r.items()
                                              if isinstance(v, (int, float, str)) and k != "family"}}
                    save_state(state)
            if cut:
                logger.warning(f"[deep.refresh] {fam}: budget stop — will continue next run")
            logger.info(f"[deep.refresh] {fam}: done in {time.time() - t1:.0f}s")
        except Exception as e:                           # noqa: BLE001
            logger.exception(f"[deep.refresh] {fam} FAILED: {e}")
            with guard:
                results[fam] = {"family": fam, "error": str(e)}

    if lanes:
        groups: Dict[str, List[str]] = {}
        for fam, _c in plan:
            groups.setdefault(lanes.get(fam, fam), []).append(fam)
        for fam in groups.pop("", []):                 # the "first, alone" families
            _one(fam)
        logger.info(f"[deep.refresh] lanes (concurrent): {groups}")
        threads = [threading.Thread(target=lambda fams=fams: [_one(f) for f in fams],
                                    name=f"deep-lane-{name}", daemon=True)
                   for name, fams in groups.items()]
        for th in threads:
            th.start()
        for th in threads:
            th.join()
    else:
        for fam, _c in plan:
            _one(fam)
    el = time.time() - t0
    # a family FAILED when its refresher raised ("error"); the per-key
    # "failed" COUNT every runner reports is ordinary bookkeeping, not a
    # verdict — the first run reported 15 healthy families as failed on it
    summary = {"seconds": round(el, 1), "ran": [f for f, _ in plan if f in results],
               "completed": [f for f, _ in plan if f in results and f not in remaining
                             and "error" not in results[f]],
               "failed": [f for f, r in results.items() if "error" in r],
               "not_finished": remaining, "not_due": skipped_not_due, "results": results}
    logger.info(f"[deep.refresh] finished in {el:.0f}s: completed {summary['completed']} | "
                f"failed {summary['failed']} | not finished {remaining}")
    return summary


# ── status ───────────────────────────────────────────────────────────────────

NEWEST_KEY = {
    "sec_filings": "acceptance", "companyfacts": "filed", "form345": "filing_date",
    "form345_live": "acceptance", "form13f": "filing_date", "ftd": "settlement_date",
    "polygon_news": "published_utc", "short_interest": "settlement_date", "short_volume": "date",
    "dividends": "declaration_date", "splits": "execution_date", "ipos": "listing_date",
    "market_daily": "date", "fred_vintages": "realtime_start", "fama_french": "date", "dix": "date",
    "cot_tff": "report_date", "wiki": "date", "quiver_congress": "ReportDate", "quiver_lobbying": "Date",
    "quiver_contracts": "Date", "quiver_dpi": "Date", "yf_earnings": "event_ts", "yf_analyst": "grade_date",
    "yf_shares": "date", "delisted": "delisted_utc",
}


def newest_dates(as_of: Optional[str] = None) -> Dict[str, Optional[str]]:
    """Newest point-in-time key per consolidated table, ignoring rows dated
    after ``as_of`` (scheduled future events: earnings dates, IPO listings)."""
    as_of = as_of or date.today().isoformat()
    import duckdb
    out: Dict[str, Optional[str]] = {}
    con = duckdb.connect()
    try:
        for fam, col in NEWEST_KEY.items():
            p = deep.DEEP_DIR / f"{fam}.parquet"
            if not p.exists():
                out[fam] = None
                continue
            try:
                v = con.execute(f'SELECT max(CAST("{col}" AS VARCHAR)) FROM \'{p.as_posix()}\' '
                                f'WHERE CAST("{col}" AS VARCHAR) <= \'{as_of}T99\'').fetchone()[0]
                out[fam] = str(v)[:19] if v is not None else None
            except Exception as e:                       # noqa: BLE001
                out[fam] = f"? ({type(e).__name__})"
    finally:
        con.close()
    # bar families stay per ticker: sample the newest bar of a few liquid names
    for fam in ("bars30m_full",):
        try:
            mx = part_max(fam, "ts")
            vals = [mx[k] for k in ("AAPL", "MSFT", "SPY", "JPM") if k in mx]
            out[fam] = max(vals)[:19] if vals else (max(mx.values())[:19] if mx else None)
        except Exception as e:                           # noqa: BLE001
            out[fam] = f"? ({type(e).__name__})"
    return out


def status_table() -> pd.DataFrame:
    state = load_state()
    newest = newest_dates()
    rows = []
    for fam, cadence in FAMILIES:
        st = state.get(fam) or {}
        rows.append({"family": fam, "cadence_h": cadence, "last_refresh_utc": (st.get("at") or "")[:19],
                     "due": due(fam, cadence, state), "newest": newest.get(fam, "")})
    for fam, v in newest.items():
        if fam not in FAMILY_NAMES:
            rows.append({"family": f"  {fam}", "cadence_h": "", "last_refresh_utc": "", "due": "", "newest": v})
    return pd.DataFrame(rows)


def _below_normal_priority() -> None:
    """The refresh shares the box with the live ticks; on Windows, drop this
    process (and the pools it spawns, which inherit it) to BELOW_NORMAL."""
    try:
        import ctypes
        k = ctypes.windll.kernel32
        k.GetCurrentProcess.restype = ctypes.c_void_p            # a 64-bit HANDLE, not a C int
        k.SetPriorityClass.argtypes = [ctypes.c_void_p, ctypes.c_uint32]
        k.SetPriorityClass(k.GetCurrentProcess(), 0x4000)
    except Exception:                                    # noqa: BLE001 — not Windows / no kernel32
        pass


def today_session_day() -> int:
    """Today's ET calendar date as days since epoch (the session a pre-open run serves).

    numpy is imported HERE: the module never imported it, and every pre-open run
    from 2026-09-23 to 09-25 fetched its families and then died on this line
    (NameError), so no session snapshot was ever built — the tests patch this
    function out, which is how it shipped. `test_today_session_day_runs_unpatched`
    now calls it for real."""
    import numpy as _np
    import pandas as _pd
    return int(_pd.Timestamp.now(tz="America/New_York").tz_localize(None).normalize()
               .to_datetime64().astype("datetime64[D]").astype(_np.int64))


def extend_bars_30m(day: int, budget_seconds: float = 1200.0, workers: int = 12) -> dict:
    """The 30-minute regular-hours store (`intraday_store`, the grid every
    price-dependent snapshot feature is read on) through the session BEFORE
    ``day``. A snapshot built on a grid that is behind leaves those features
    missing (`deep_features.RTH.with_session` refuses to stretch a stale close
    over the gap) — and nothing else extends that store before the pre-open
    build: the tick cache covers only the names a tick touches."""
    from src.analysis import deep_features as dfe
    from src.data import deep as _deep
    from src.data.intraday_store import extend_deep_30m, reset_split_tickers
    prev = date(1970, 1, 1) + timedelta(days=int(dfe.previous_session_day(day)))
    out = extend_deep_30m(_deep.deep_universe(), workers=workers, budget_seconds=budget_seconds,
                          min_age_days=0, today=prev)
    # a split effective up to today resets the name's history BEFORE the
    # snapshot is built on it (user directive 2026-09-27)
    today = date(1970, 1, 1) + timedelta(days=int(day))
    out["split_resets"] = reset_split_tickers(_deep.deep_universe(), today)
    return out


def run_preopen(budget_seconds: float = 2700.0, workers: int = 6, sec_rate: float = 0.3,
                snapshot: bool = True, snapshot_workers: int = 10) -> dict:
    """The market-day 08:30 ET run: re-fetch PREOPEN_FAMILIES (forced — the
    snapshot needs them as of THIS morning whatever ran overnight), extend the
    30-minute store through the previous session (`extend_bars_30m`), then build
    today's session snapshot (and the previous SESSION's if it is missing: the
    pre-market ticks score its last bar — Friday's on a Monday, never Sunday's).
    The feature cutoff is 08:30 ET, so a fetch that starts at/after it holds
    everything the cutoff admits."""
    # the 30-minute store's extension reads none of the families: its own lane,
    # side by side with them (it took 3 min AFTER them on 2026-09-29)
    ext: dict = {}
    day = today_session_day() if snapshot else None

    def _extend() -> None:
        try:
            ext["r"] = extend_bars_30m(day)
        except Exception as e:                           # noqa: BLE001
            logger.exception(f"[deep.refresh] 30-minute store extension FAILED: {e}")
            ext["failed"] = True

    th = threading.Thread(target=_extend, name="deep-lane-extend_30m", daemon=True) if snapshot else None
    if th is not None:
        th.start()
    s = run(PREOPEN_FAMILIES, force=True, budget_seconds=budget_seconds, workers=workers, sec_rate=sec_rate,
            lanes=PREOPEN_LANES, profile="preopen", family_workers=PREOPEN_WORKERS)
    if th is not None:
        th.join()
    if snapshot and not s.get("skipped"):
        from src.analysis import deep_features as dfe
        prev = dfe.previous_session_day(day)
        if "r" in ext:
            s["extend_30m"] = ext["r"]
        if ext.get("failed"):
            s.setdefault("failed", []).append("extend_30m")
        todo = [day] + ([prev] if not dfe.snapshot_path(prev).exists() else [])
        for d in todo:
            try:
                s.setdefault("snapshots", {})[str(d)] = dfe.build_session_snapshot(d, workers=snapshot_workers)
            except Exception as e:                       # noqa: BLE001
                logger.exception(f"[deep.refresh] session snapshot {d} FAILED: {e}")
                s.setdefault("failed", []).append(f"snapshot:{d}")
    return s


def main(argv=None) -> None:
    import argparse
    ap = argparse.ArgumentParser(description="deep history store — incremental refresh")
    ap.add_argument("--families", default="", help=f"comma list; default = all of {FAMILY_NAMES}")
    ap.add_argument("--profile", default="nightly", choices=["nightly", "preopen"],
                    help="preopen = the 08:30 ET run: PREOPEN_FAMILIES forced + today's session snapshot")
    ap.add_argument("--force", action="store_true", help="run the named families even if not due")
    ap.add_argument("--budget-seconds", type=float, default=0.0,
                    help="0 = the profile's default (nightly 14400, preopen 2700)")
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--sec-rate", type=float, default=0.0,
                    help="seconds between EDGAR requests (0.12 = the 10/s ceiling). Default: 0.2 "
                         "nightly, 0.3 pre-open — the 08:30 run overlaps the 08:30 tick, whose own "
                         "EDGAR calls share the SEC's 10 req/s")
    ap.add_argument("--status", action="store_true")
    a = ap.parse_args(argv)
    logger.add("logs/deep_refresh.log", rotation="1 day", retention="30 days", level="INFO", enqueue=True)
    if a.status:
        df = status_table()
        print(df.to_string(index=False) if len(df) else "(empty)")
        return
    _below_normal_priority()
    if a.profile == "preopen":
        s = run_preopen(budget_seconds=a.budget_seconds or 2700.0, workers=a.workers, sec_rate=a.sec_rate or 0.3)
    else:
        fams = [f for f in a.families.split(",") if f.strip()] if a.families else None
        from config.settings import settings as _settings
        s = run(fams, force=a.force, budget_seconds=a.budget_seconds or 14400.0, workers=a.workers,
                sec_rate=a.sec_rate or 0.2, profile="nightly",
                lanes=NIGHTLY_LANES if getattr(_settings, "deep_refresh_lanes", True) else None)
    # ASCII only: the scheduler logs this line from a cp1252 console
    if s.get("skipped"):
        print(f"deep refresh: skipped - {s['skipped']}")
        return
    print(f"deep refresh: {s['seconds']:.0f}s | completed {s['completed']} | failed {s['failed']} | "
          f"not finished {s['not_finished']} | not due {s['not_due']}")
    if s["failed"]:
        sys.exit(2)


if __name__ == "__main__":
    main()
