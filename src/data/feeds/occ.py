"""The OCC's daily options volume by underlying (the data feeds, `src.data.feeds`).

The OCC's volume query (``feeds_occ_url``) answers, for one session, every underlying's contracts by account type (C
customer, F firm, M market maker), put / call and exchange — and only for its last 2 years
(``feeds_occ_history_days``): every completed session is archived RAW, once, as ``feeds_occ_dir``/<day>.csv.gz (never
delete: the OCC keeps no older history), with its aggregate per (underlying, account type, put / call) beside it
(``agg/<day>.csv.gz``). A session the OCC has not published yet ("No record(s) found" on a market day, or "Report
date cannot be greater than" its latest session — its answer after the close of 2026-10-26) is asked again at later
runs, at most every ``feeds_occ_retry_minutes``; a near-empty file (a market closure: 2025-01-09 held 3
underlyings) is archived but never read as a session (`sessions`).

Point in time: a session's file is published after its close, so a decision on day D reads the sessions before D.
"""
from __future__ import annotations

import csv
import gzip
import io
import os
from collections import defaultdict
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Callable, Dict, List, Optional

from loguru import logger

from config import settings
from src.data import feeds as FD
from src.performance import market_calendar as MC

HEADER = "quantity,underlying,symbol,actype,porc,exchange,actdate"
NONE_TEXT = "No record(s) found"
LATER_TEXT = "Report date cannot be greater than"     # a session after the OCC's latest published one
MIN_ROWS = 1000                      # a normal session holds 100,000+ rows; fewer is a closure or a broken download


def root() -> Path:
    return Path(settings.feeds_occ_dir)


def path_for(d: date) -> Path:
    return root() / f"{d.isoformat()}.csv.gz"


def agg_path(d: date) -> Path:
    return root() / "agg" / f"{d.isoformat()}.csv.gz"


def check(text: str) -> int:
    """The number of rows of a download that looks like the OCC's volume file; raises ValueError otherwise."""
    if not text.startswith(HEADER):
        raise ValueError(f"not the OCC's volume file (starts {text[:60]!r})")
    return max(0, text.count("\n") - 1)


def aggregate(text: str) -> Dict[tuple, int]:
    out: Dict[tuple, int] = defaultdict(int)
    rd = csv.DictReader(io.StringIO(text))
    for r in rd:
        try:
            q = int(float(r.get("quantity") or 0))
        except ValueError:
            continue
        out[(str(r.get("underlying") or "").strip().upper(), str(r.get("actype") or ""), str(r.get("porc") or ""))] += q
    return out


def _write(d: date, text: str) -> int:
    n = check(text)
    p = path_for(d)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(f".{os.getpid()}.tmp")
    with gzip.open(tmp, "wt", encoding="utf-8") as fh:
        fh.write(text)
    os.replace(tmp, p)
    a = aggregate(text)
    q = agg_path(d)
    q.parent.mkdir(parents=True, exist_ok=True)
    tmp = q.with_suffix(f".{os.getpid()}.tmp")
    with gzip.open(tmp, "wt", encoding="utf-8") as fh:
        fh.write("underlying,actype,porc,quantity\n")
        for (u, ac, pc), v in sorted(a.items()):
            fh.write(f"{u},{ac},{pc},{v}\n")
    os.replace(tmp, q)
    return n


def _default_get():
    import httpx
    client = httpx.Client(timeout=180.0, headers=FD.UA)
    return client.get


def fetch_text(d: date, http_get: Callable) -> str:
    r = http_get(str(settings.feeds_occ_url), params={
        "reportDate": d.strftime("%Y%m%d"), "format": "csv", "volumeQueryType": "O", "symbolType": "ALL", "symbol": "",
        "reportType": "D", "accountType": "ALL", "productKind": "ALL", "porc": "BOTH"})
    if r.status_code != 200:
        raise RuntimeError(f"http {r.status_code}")
    return r.text


def last_completed(now: datetime) -> date:
    """The latest session whose close has passed (16:00 ET), whatever the OCC has published."""
    from zoneinfo import ZoneInfo
    et = now.astimezone(ZoneInfo("America/New_York"))
    d = et.date()
    if not (MC.is_market_day(d) and et.hour >= 16):
        d = MC.previous_market_day(d)
    return d


def due(now: datetime, st: dict) -> List[date]:
    """The completed sessions inside the OCC's window not archived yet, oldest first (a session asked within
    ``feeds_occ_retry_minutes`` waits)."""
    end = last_completed(now)
    start = (now.date() - timedelta(days=int(settings.feeds_occ_history_days)))
    retry = st.get("retry") or {}
    wait = timedelta(minutes=float(settings.feeds_occ_retry_minutes))
    out, d = [], start
    while d <= end:
        if MC.is_market_day(d) and not path_for(d).exists():
            last = retry.get(d.isoformat())
            if not last or now - datetime.fromisoformat(last) >= wait:
                out.append(d)
        d += timedelta(days=1)
    return out


def run(st: dict, now: Optional[datetime] = None, http_get: Optional[Callable] = None,
        deadline: Optional[float] = None) -> str:
    """Archive every due session (``st`` updated in place). Returns a summary."""
    import time
    now = now or FD.now_utc()
    get = http_get or _default_get()
    got, waiting = [], []
    retry = st.setdefault("retry", {})
    for d in due(now, st):
        if deadline is not None and time.time() > deadline:
            break
        text = fetch_text(d, get)
        if text.strip().startswith((NONE_TEXT, LATER_TEXT)):
            retry[d.isoformat()] = now.isoformat()            # not published yet: asked again later
            waiting.append(d.isoformat())
        else:
            n = _write(d, text)
            retry.pop(d.isoformat(), None)
            got.append((d.isoformat(), n))
        FD.pace()
    for k in [k for k in retry if date.fromisoformat(k) < now.date() - timedelta(days=int(settings.feeds_occ_history_days))]:
        retry.pop(k, None)
    st["last_sessions"] = [g[0] for g in got][-5:]
    return f"{len(got)} session(s) archived{(' ' + got[-1][0]) if got else ''}; {len(waiting)} not published yet"


def missing_recent(now: datetime, sessions: int = 3) -> List[str]:
    """The completed sessions of the last ``sessions`` (after a day's grace) without a file — the health note's."""
    d = last_completed(now - timedelta(days=1))
    out = []
    for _ in range(sessions):
        if not path_for(d).exists():
            out.append(d.isoformat())
        d = MC.previous_market_day(d)
    return sorted(out)


def import_dir(src: Path) -> int:
    """Copy a directory of raw <day>.csv.gz files (PREREG208's research download) into the store; returns the days."""
    n = 0
    for f in sorted(Path(src).glob("????-??-??.csv.gz")):
        d = date.fromisoformat(f.name[:10])
        if path_for(d).exists() or not MC.is_market_day(d):        # a closure's file is never a session
            continue
        with gzip.open(f, "rt", encoding="utf-8") as fh:
            text = fh.read()
        try:
            _write(d, text)
            n += 1
        except ValueError as e:
            logger.warning(f"[feeds] occ import {f.name}: {e}")
    return n


def read_agg(d: date) -> Dict[tuple, int]:
    q = agg_path(d)
    if not q.exists():
        return {}
    out = {}
    with gzip.open(q, "rt", encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            out[(r["underlying"], r["actype"], r["porc"])] = int(r["quantity"])
    return out


def sessions(before: date, n: int) -> List[date]:
    """The ``n`` latest archived sessions before ``before`` (a closure's near-empty file is not a session)."""
    out, d = [], before - timedelta(days=1)
    floor = before - timedelta(days=int(settings.feeds_occ_history_days) + 7)
    while len(out) < n and d >= floor:
        q = agg_path(d)
        if q.exists() and sum(read_agg(d).values()) >= MIN_ROWS:
            out.append(d)
        d -= timedelta(days=1)
    return sorted(out)


def volume(underlying: str, days: List[date], actype: str = "C", porc: str = "C") -> List[int]:
    """The underlying's contracts of ``actype`` / ``porc`` on each of ``days`` (0 on a session without a row)."""
    u = str(underlying or "").upper().replace(".", "").replace("-", "")
    return [read_agg(d).get((u, actype, porc), 0) for d in days]
