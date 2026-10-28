"""GDELT 1.0's daily Global Knowledge Graph, aggregated by theme (the data feeds, `src.data.feeds`).

GDELT tags every news article it monitors with themes (its GKG taxonomy: ECON_STOCKMARKET, ECON_INFLATION,
TAX_DISEASE_*, CRISISLEX_*, ...) and a tone. Its 1.0 daily file (``feeds_gdelt_url``, one zip a UTC day, posted about
07:00 UTC the next day, 17-65 MB) holds one row per distinct combination with its article count (NUMARTS). For each
day from ``feeds_gdelt_start`` not held yet, at most ``feeds_gdelt_max_days`` a run, the file is checked against
GDELT's MD5 list (``feeds_gdelt_md5_url``) and reduced to every theme's article count and tone sum
(``feeds_gdelt_dir``/agg/<day>.csv.gz; the day's totals in ``days.csv``); the file itself is not kept (GDELT keeps
it). A day GDELT never posted (18 days of 2025-06-14 – 07-01 are missing) is recorded missing. History before
``feeds_gdelt_start`` (2015-02-18 on, about 156 GB to stream) waits for the user's go-ahead: `backfill` (the CLI).

Point in time: day D's file is posted before 03:00 ET of D+1, so a decision at D+1's open reads the days through D.
A theme's share of a day = its articles / the day's articles; a level means little across years (the pandemic themes
held ~10% in mid-2022): read each against its own trailing baseline. No live book reads this feed yet (the free-data
census of 2026-10-26: the crisis books' context).
"""
from __future__ import annotations

import csv
import gzip
import hashlib
import io
import os
import time
import zipfile
from collections import Counter
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

from config import settings
from src.data import feeds as FD

POSTED_UTC_HOUR = 8                # a day's file is asked for from 08:00 UTC the next day (GDELT posts ~07:00)
MISSING_AFTER_DAYS = 2             # a file still absent this long after its posting hour is recorded missing
DAYS_COLS = ["day", "rows", "articles", "tone_sum", "themes", "md5", "bytes", "fetched_utc"]


def root() -> Path:
    return Path(settings.feeds_gdelt_dir)


def agg_path(d: date) -> Path:
    return root() / "agg" / f"{d.isoformat()}.csv.gz"


def _default_get():
    import httpx
    client = httpx.Client(timeout=300.0, headers=FD.UA, follow_redirects=True)
    return client.get


def aggregate(raw_zip: bytes) -> Tuple[dict, Dict[str, List[float]]]:
    """The day's totals and each theme's [articles, tone sum] of one GKG 1.0 daily zip (NUMARTS-weighted)."""
    csv.field_size_limit(1 << 30)
    z = zipfile.ZipFile(io.BytesIO(raw_zip))
    members = z.namelist()
    if len(members) != 1 or not members[0].endswith(".csv"):
        raise ValueError(f"not a GKG daily zip ({members[:3]})")
    arts, tone_sum, rows = 0, 0.0, 0
    th_n: Counter = Counter()
    th_t: Counter = Counter()
    with z.open(members[0]) as fh:
        rd = csv.reader(io.TextIOWrapper(fh, encoding="utf-8", errors="replace"), delimiter="\t",
                        quoting=csv.QUOTE_NONE)
        for i, r in enumerate(rd):
            if i == 0:
                if not r or r[0].upper() != "DATE" or len(r) < 8 or r[1].upper() != "NUMARTS" \
                        or r[3].upper() != "THEMES" or r[7].upper() != "TONE":
                    raise ValueError(f"not the GKG 1.0 header ({r[:8]})")
                continue
            if len(r) < 8:
                continue
            try:
                n = int(r[1] or 0)
                tone = float((r[7] or "0").split(",")[0] or 0)
            except ValueError:
                continue
            rows += 1
            arts += n
            tone_sum += tone * n
            for th in set(r[3].split(";")) - {""}:
                th_n[th] += n
                th_t[th] += tone * n
    totals = {"rows": rows, "articles": arts, "tone_sum": round(tone_sum, 3), "themes": len(th_n)}
    return totals, {th: [th_n[th], round(th_t[th], 3)] for th in th_n}


def md5s(get: Callable) -> Dict[str, str]:
    r = get(str(settings.feeds_gdelt_md5_url))
    if r.status_code != 200:
        raise RuntimeError(f"GDELT's MD5 list answered http {r.status_code}")
    out = {}
    for line in r.text.splitlines():
        parts = line.split()
        if len(parts) == 2:
            out[parts[1]] = parts[0].lower()
    return out


def _read_days() -> Dict[str, list]:
    p = root() / "days.csv"
    if not p.exists():
        return {}
    with open(p, encoding="utf-8", newline="") as fh:
        rd = csv.reader(fh)
        next(rd, None)
        return {r[0]: r for r in rd if r}


def _write_days(rows: Dict[str, list]) -> None:
    p = root() / "days.csv"
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(f".{os.getpid()}.tmp")
    with open(tmp, "w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(DAYS_COLS)
        for k in sorted(rows):
            w.writerow(rows[k])
    os.replace(tmp, p)


def write_day(d: date, raw_zip: bytes, md5: str, now: datetime) -> dict:
    totals, themes = aggregate(raw_zip)
    p = agg_path(d)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(f".{os.getpid()}.tmp")
    with gzip.open(tmp, "wt", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["theme", "articles", "tone_sum"])
        for th, (n, t) in sorted(themes.items(), key=lambda kv: -kv[1][0]):
            w.writerow([th, n, t])
    os.replace(tmp, p)
    days = _read_days()
    days[d.isoformat()] = [d.isoformat(), totals["rows"], totals["articles"], totals["tone_sum"], totals["themes"], md5,
                           len(raw_zip), now.isoformat()]
    _write_days(days)
    return totals


def due(now: datetime, st: dict, start: Optional[date] = None, end: Optional[date] = None) -> List[date]:
    """The UTC days from ``start`` (else ``feeds_gdelt_start``) to the latest posted one, not held nor missing."""
    s = start or date.fromisoformat(str(settings.feeds_gdelt_start))
    u = now.astimezone(timezone.utc)
    latest = (u - timedelta(days=1)).date() if u.hour >= POSTED_UTC_HOUR else (u - timedelta(days=2)).date()
    e = min(end, latest) if end else latest
    missing = set(st.get("missing") or [])
    out, d = [], s
    while d <= e:
        if not agg_path(d).exists() and d.isoformat() not in missing:
            out.append(d)
        d += timedelta(days=1)
    return out


def run(st: dict, now: Optional[datetime] = None, http_get: Optional[Callable] = None,
        deadline: Optional[float] = None, start: Optional[date] = None, end: Optional[date] = None,
        max_days: Optional[int] = None) -> str:
    """Aggregate the due days, oldest first, at most ``feeds_gdelt_max_days`` (``st`` updated in place)."""
    now = now or FD.now_utc()
    get = http_get or _default_get()
    todo = due(now, st, start, end)[: int(max_days if max_days is not None else settings.feeds_gdelt_max_days)]
    if not todo:
        return "nothing due"
    sums = md5s(get)
    got, gone = [], []
    for d in todo:
        if deadline is not None and time.time() > deadline:
            break
        name = f"{d.strftime('%Y%m%d')}.gkg.csv.zip"
        age_h = (now - datetime(d.year, d.month, d.day, tzinfo=timezone.utc)).total_seconds() / 3600.0 - 24.0
        if name not in sums:
            if age_h > 24.0 * MISSING_AFTER_DAYS:
                st.setdefault("missing", []).append(d.isoformat())
                gone.append(d.isoformat())
            continue                                          # not posted yet
        FD.pace()
        r = get(str(settings.feeds_gdelt_url).format(day=d.strftime("%Y%m%d")))
        if r.status_code == 404 and age_h > 24.0 * MISSING_AFTER_DAYS:
            st.setdefault("missing", []).append(d.isoformat())
            gone.append(d.isoformat())
            continue
        if r.status_code != 200:
            raise RuntimeError(f"http {r.status_code} for {name}")
        h = hashlib.md5(r.content).hexdigest()
        if h != sums[name]:
            raise ValueError(f"{name}: MD5 {h} is not GDELT's {sums[name]}")
        t = write_day(d, r.content, h, now)
        got.append(f"{d.isoformat()} ({t['articles']:,} articles)")
    return (f"{len(got)} day(s) aggregated" + (f": {', '.join(got)}" if got else "")
            + (f"; missing {', '.join(gone)}" if gone else ""))


def backfill(start: date, end: date, http_get: Optional[Callable] = None) -> str:
    """Aggregate every day of [start, end] not held (the CLI; the history waits for the user's go-ahead)."""
    st = FD.load_state().get("gdelt") or {}
    msg = run(st, now=FD.now_utc(), http_get=http_get, start=start, end=end, max_days=10 ** 6)
    full = FD.load_state()
    full.setdefault("gdelt", {})["missing"] = sorted(set(st.get("missing") or []))
    FD.save_state(full)
    return msg


def shares(d: date, themes: List[str]) -> Optional[Dict[str, float]]:
    """Each theme's share of day ``d``'s articles (None when the day is not held)."""
    p = agg_path(d)
    days = _read_days()
    if not p.exists() or d.isoformat() not in days:
        return None
    total = float(days[d.isoformat()][2]) or 1.0
    want = set(themes)
    out = {th: 0.0 for th in themes}
    with gzip.open(p, "rt", encoding="utf-8", newline="") as fh:
        rd = csv.reader(fh)
        next(rd, None)
        for r in rd:
            if r and r[0] in want:
                out[r[0]] = int(r[1]) / total
    return out
