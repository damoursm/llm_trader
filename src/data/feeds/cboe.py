"""Cboe's daily options statistics (the data feeds, `src.data.feeds`).

Cboe posts each session's options market statistics as JSON (``feeds_cboe_url`` with the session's date): every
product's call / put / total volume and open interest (all products, index options, exchange-traded products, equity
options, VIX, SPX + SPXW, ...) and Cboe's put/call ratios. Each completed session of the last
``feeds_cboe_history_days`` not archived yet is fetched once — ``feeds_cboe_dir``/daily/<day>.json.gz raw, its ratios
and volumes in ``ratios.csv`` / ``volumes.csv`` (long form) —; a session Cboe has not posted yet (it answers 403) is
asked again after ``feeds_cboe_retry_minutes``. History: `backfill` (the daily JSON from 2019-10-07) and
`import_history` (Cboe's CSVs of 2006-11-01 – 2019-10-04: the total, index and equity put/call, ``history.csv``).

Point in time: a session's statistics are posted in the evening after its close, so a decision on day D reads the
sessions before D. No live book reads this feed yet (the free-data census of 2026-10-26: the crisis books' context).
"""
from __future__ import annotations

import csv
import gzip
import json
import os
import re
import time
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Callable, Dict, List, Optional

from config import settings
from src.data import feeds as FD
from src.performance import market_calendar as MC

HISTORY_CSVS = {"total": "https://cdn.cboe.com/resources/options/volume_and_call_put_ratios/totalpc.csv",
                "index": "https://cdn.cboe.com/resources/options/volume_and_call_put_ratios/indexpc.csv",
                "equity": "https://cdn.cboe.com/resources/options/volume_and_call_put_ratios/equitypc.csv"}
_RX_ROW = re.compile(r"^\s*(\d{1,2})/(\d{1,2})/(\d{4})\s*,(.*)$")


def root() -> Path:
    return Path(settings.feeds_cboe_dir)


def path_for(d: date) -> Path:
    return root() / "daily" / f"{d.isoformat()}.json.gz"


def _default_get():
    import httpx
    client = httpx.Client(timeout=60.0, headers=FD.UA, follow_redirects=True)
    return client.get


def parse(js: dict) -> Dict[str, list]:
    """The ratios (name, value) and the volumes (product, metric, call, put, total) of one session's JSON."""
    if not isinstance(js, dict) or not isinstance(js.get("ratios"), list):
        raise ValueError("not Cboe's daily options statistics (no ratios)")
    ratios = []
    for r in js["ratios"]:
        try:
            ratios.append((str(r.get("name") or "").strip(), float(r.get("value"))))
        except (TypeError, ValueError):
            continue
    vols = []
    for product, rows in js.items():
        if product == "ratios" or not isinstance(rows, list):
            continue
        for r in rows:
            if isinstance(r, dict) and "call" in r:
                vols.append((str(product), str(r.get("name") or "").strip().lower(), r.get("call"), r.get("put"),
                             r.get("total")))
    if not ratios:
        raise ValueError("Cboe's daily statistics hold no ratio")
    return {"ratios": ratios, "volumes": vols}


def _rewrite(p: Path, header: List[str], rows: Dict[tuple, list]) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(f".{os.getpid()}.tmp")
    with open(tmp, "w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(header)
        for k in sorted(rows):
            w.writerow(rows[k])
    os.replace(tmp, p)


def _read(p: Path, nkey: int) -> Dict[tuple, list]:
    if not p.exists():
        return {}
    with open(p, encoding="utf-8", newline="") as fh:
        rd = csv.reader(fh)
        next(rd, None)
        return {tuple(r[:nkey]): r for r in rd if len(r) > nkey}


def write_session(d: date, raw: bytes) -> int:
    """Archive one session's raw JSON and merge its ratios and volumes into the CSVs; returns the ratios kept."""
    got = parse(json.loads(raw.decode("utf-8")))
    p = path_for(d)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(f".{os.getpid()}.tmp")
    tmp.write_bytes(gzip.compress(raw))
    os.replace(tmp, p)
    day = d.isoformat()
    rat = _read(root() / "ratios.csv", 2)
    for name, v in got["ratios"]:
        rat[(day, name)] = [day, name, v]
    _rewrite(root() / "ratios.csv", ["day", "name", "value"], rat)
    vol = _read(root() / "volumes.csv", 3)
    for prod, metric, c, pu, t in got["volumes"]:
        vol[(day, prod, metric)] = [day, prod, metric, c, pu, t]
    _rewrite(root() / "volumes.csv", ["day", "product", "metric", "call", "put", "total"], vol)
    return len(got["ratios"])


def last_completed(now: datetime) -> date:
    """The latest session whose close has passed (16:00 ET)."""
    from src.utils import ET
    et = now.astimezone(ET)
    d = et.date()
    if not (MC.is_market_day(d) and et.hour >= 16):
        d = MC.previous_market_day(d)
    return d


def due(now: datetime, st: dict, start: Optional[date] = None) -> List[date]:
    """The completed sessions from ``start`` (else ``feeds_cboe_history_days`` back) not archived, oldest first; a
    session asked within ``feeds_cboe_retry_minutes`` waits."""
    end = last_completed(now)
    d = start or (now.date() - timedelta(days=int(settings.feeds_cboe_history_days)))
    retry = st.get("retry") or {}
    wait = timedelta(minutes=float(settings.feeds_cboe_retry_minutes))
    out = []
    while d <= end:
        if MC.is_market_day(d) and not path_for(d).exists():
            last = retry.get(d.isoformat())
            if not last or now - datetime.fromisoformat(last) >= wait:
                out.append(d)
        d += timedelta(days=1)
    return out


def run(st: dict, now: Optional[datetime] = None, http_get: Optional[Callable] = None,
        deadline: Optional[float] = None, start: Optional[date] = None) -> str:
    """Archive every due session (``st`` updated in place). Returns a summary."""
    now = now or FD.now_utc()
    get = http_get or _default_get()
    retry = st.setdefault("retry", {})
    got, waiting = [], []
    for d in due(now, st, start):
        if deadline is not None and time.time() > deadline:
            break
        r = get(str(settings.feeds_cboe_url).format(day=d.isoformat()))
        if r.status_code in (403, 404):
            retry[d.isoformat()] = now.isoformat()             # not posted yet: asked again later
            waiting.append(d.isoformat())
        elif r.status_code != 200:
            raise RuntimeError(f"http {r.status_code} for {d}")
        else:
            write_session(d, r.content)
            retry.pop(d.isoformat(), None)
            got.append(d.isoformat())
        FD.pace()
    old = (now.date() - timedelta(days=int(settings.feeds_cboe_history_days) + 7)).isoformat()
    for k in [k for k in retry if k < old]:
        retry.pop(k, None)
    return f"{len(got)} session(s) archived{(' ' + got[-1]) if got else ''}; {len(waiting)} not posted yet"


def backfill(start: date, http_get: Optional[Callable] = None) -> str:
    """Archive every session from ``start`` (the daily JSON begins 2019-10-07) not held yet."""
    st: dict = {}
    return run(st, now=FD.now_utc(), http_get=http_get, start=start)


def parse_history(text: str) -> List[tuple]:
    """Rows (day, call, put, total, ratio) of one of Cboe's history CSVs (its disclaimer and header lines skipped)."""
    out = []
    for line in text.splitlines():
        m = _RX_ROW.match(line)
        if not m:
            continue
        vals = [x.strip() for x in m.group(4).split(",")]
        try:
            nums = [float(x) for x in vals[:4]]
        except ValueError:
            continue
        if len(nums) < 4:
            continue
        d = date(int(m.group(3)), int(m.group(1)), int(m.group(2)))
        out.append((d.isoformat(), nums[0], nums[1], nums[2], nums[3]))
    return out


def import_history(http_get: Optional[Callable] = None) -> int:
    """Download Cboe's history CSVs (2006-11-01 – 2019-10-04) into ``history.csv`` (series, day, call, put, total,
    ratio); returns the rows."""
    get = http_get or _default_get()
    rows: Dict[tuple, list] = {}
    for series, url in HISTORY_CSVS.items():
        r = get(url)
        if r.status_code != 200:
            raise RuntimeError(f"http {r.status_code} for {series}")
        for day, c, p, t, ratio in parse_history(r.text):
            rows[(series, day)] = [series, day, c, p, t, ratio]
        FD.pace()
    _rewrite(root() / "history.csv", ["series", "day", "call", "put", "total", "ratio"], rows)
    return len(rows)


def ratios(name: str = "TOTAL PUT/CALL RATIO", before: Optional[date] = None, n: int = 20) -> List[tuple]:
    """The last ``n`` sessions' (day, value) of one ratio, sessions before ``before`` only (point in time)."""
    rows = sorted((r[0], float(r[2])) for r in _read(root() / "ratios.csv", 2).values()
                  if r[1] == name and (before is None or r[0] < before.isoformat()))
    return rows[-n:]
