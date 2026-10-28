"""The FDA's Drugs@FDA export (the data feeds, `src.data.feeds`).

openFDA's download index (``feeds_fda_index_url``) names the Drugs@FDA file and its export date; every
``feeds_fda_check_hours`` the index is read and a NEW export downloaded (``feeds_fda_dir``/drugsfda_<export>.json.zip,
the newest ``KEEP_EXPORTS`` kept: each repeats the whole history) and its approvals extracted to ``approvals.csv`` (an original or supplement approval: the application, its
type NDA / BLA / ANDA, the sponsor, the date, the submission class and review priority). Point in time: an approval
is public on its date (the FDA's own announcement); the export can add it a week later.
"""
from __future__ import annotations

import csv
import io
import json
import os
import zipfile
from datetime import datetime
from pathlib import Path
from typing import Callable, List, Optional

from config import settings
from src.data import feeds as FD

COLS = ("application", "app_type", "sponsor", "submission_type", "submission_number", "status_date",
        "class_code", "class_description", "review_priority")
KEEP_EXPORTS = 4


def root() -> Path:
    return Path(settings.feeds_fda_dir)


def _default_get():
    import httpx
    client = httpx.Client(timeout=300.0, headers=FD.UA, follow_redirects=True)
    return client.get


def index(http_get: Callable) -> dict:
    r = http_get(str(settings.feeds_fda_index_url))
    if r.status_code != 200:
        raise RuntimeError(f"http {r.status_code}")
    d = (r.json().get("results") or {}).get("drug", {}).get("drugsfda") or {}
    parts = d.get("partitions") or []
    if not parts or not d.get("export_date"):
        raise ValueError("openFDA's index names no Drugs@FDA file")
    return {"export_date": str(d["export_date"]), "url": str(parts[0]["file"])}


def approvals(raw_zip: bytes) -> List[dict]:
    z = zipfile.ZipFile(io.BytesIO(raw_zip))
    res = json.loads(z.read(z.namelist()[0])).get("results")
    if not isinstance(res, list) or not res:
        raise ValueError("not the Drugs@FDA export (no results)")
    out = []
    for a in res:
        an = str(a.get("application_number") or "")
        ty = "ANDA" if an.startswith("ANDA") else an[:3]
        for s in a.get("submissions") or []:
            if s.get("submission_status") != "AP":
                continue
            out.append({"application": an, "app_type": ty, "sponsor": a.get("sponsor_name") or "",
                        "submission_type": s.get("submission_type") or "",
                        "submission_number": s.get("submission_number") or "",
                        "status_date": str(s.get("submission_status_date") or ""),
                        "class_code": s.get("submission_class_code") or "",
                        "class_description": s.get("submission_class_code_description") or "",
                        "review_priority": s.get("review_priority") or ""})
    return out


def _write(export_date: str, raw_zip: bytes) -> int:
    rows = approvals(raw_zip)
    root().mkdir(parents=True, exist_ok=True)
    z = root() / f"drugsfda_{export_date}.json.zip"
    tmp = z.with_suffix(f".{os.getpid()}.tmp")
    tmp.write_bytes(raw_zip)
    os.replace(tmp, z)
    c = root() / "approvals.csv"
    tmp = c.with_suffix(f".{os.getpid()}.tmp")
    with open(tmp, "w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=COLS)
        w.writeheader()
        for r in sorted(rows, key=lambda x: (x["status_date"], x["application"])):
            w.writerow(r)
    os.replace(tmp, c)
    # every export repeats the whole history: the newest KEEP_EXPORTS are kept (about 9 MB each)
    old = sorted(root().glob("drugsfda_*.json.zip"))[:-KEEP_EXPORTS]
    for f in old:
        try:
            f.unlink()
        except OSError:
            pass
    return len(rows)


def run(st: dict, now: Optional[datetime] = None, http_get: Optional[Callable] = None,
        deadline: Optional[float] = None) -> str:
    """Download a new export when openFDA's index names one (``st`` updated in place). Returns a summary."""
    get = http_get or _default_get()
    ix = index(get)
    if ix["export_date"] == st.get("export_date") and (root() / "approvals.csv").exists():
        return f"export {ix['export_date']} already held"
    FD.pace()
    r = get(ix["url"])
    if r.status_code != 200:
        raise RuntimeError(f"http {r.status_code} on the export")
    n = _write(ix["export_date"], r.content)
    st["export_date"] = ix["export_date"]
    return f"export {ix['export_date']}: {n} approvals"


def import_zip(path: Path, export_date: str) -> int:
    return _write(export_date, Path(path).read_bytes())


def read(start: Optional[str] = None, end: Optional[str] = None, kinds=("NDA", "BLA"),
         original_only: bool = True) -> List[dict]:
    """The approvals dated in [start, end] (YYYYMMDD strings), originals of ``kinds`` by default."""
    c = root() / "approvals.csv"
    if not c.exists():
        return []
    out = []
    with open(c, encoding="utf-8", newline="") as fh:
        for r in csv.DictReader(fh):
            if kinds and r["app_type"] not in kinds:
                continue
            if original_only and r["submission_type"] != "ORIG":
                continue
            if (start and r["status_date"] < start) or (end and r["status_date"] > end):
                continue
            out.append(r)
    return out
