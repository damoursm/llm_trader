"""EDGAR's full-text search: saved queries archived (the data feeds, `src.data.feeds`).

The SEC's full-text search (``feeds_efts_url``; filings from 2001) answers a phrase with every matching document — its
accession and file name, form, file date, the filer's CIKs, display names (they carry the ticker), SIC codes and an
8-K's items. Each saved query of ``QUERIES`` is asked every ``feeds_efts_check_hours`` over the days from its last
window's end minus ``feeds_efts_overlap_days`` to today, 100 hits a page; every hit is kept once
(``feeds_efts_dir``/<query>/hits/<YYYY-MM>.jsonl, by file month) with ``seen_utc`` — when this feed first saw it, the
point-in-time key production can rely on (a file date is only a date, and EDGAR's acceptance instants drift: memory
`edgar-acceptance-drift`). A query with ``docs`` keeps each matched document once
(``<query>/docs/<adsh>__<file>.gz``, at most 3 MB) and the dates its text gives after the query's anchor word
(``<query>/dates.jsonl``: exact dates, month-years and quarters with their context) — for "PDUFA", the FDA target
action dates companies announce in 8-K press releases (the free-data census of 2026-10-26; PREREG210 studies them).
The red-flag queries (a going-concern doubt, an Item 4.02 non-reliance, a Wells notice; PREREG214) keep the hits only.

SEC requests go through `sec.default_get` (the deep store's headers, paced) and none is made inside production's own
SEC lanes (`sec.quiet`). History: `import_dir` merges a research download (PREREG210's `hits/<year>.json` + `docs/`).
No live book reads this feed yet.
"""
from __future__ import annotations

import gzip
import html
import json
import os
import re
import shutil
import time
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional

from config import settings
from src.data import feeds as FD

QUERIES: Dict[str, dict] = {
    # 8-Ks naming the FDA's target action date (their press-release exhibits included)
    "pdufa": {"q": '"PDUFA"', "forms": "8-K", "docs": True, "anchor": "PDUFA"},
    # red flags, the hits only
    "going_concern": {"q": '"substantial doubt about its ability to continue as a going concern"',
                      "forms": "10-K,10-Q,8-K", "docs": False},
    "non_reliance": {"q": '"Non-Reliance on Previously Issued Financial Statements"', "forms": "8-K", "docs": False},
    "wells_notice": {"q": '"Wells notice"', "forms": "8-K,10-K,10-Q", "docs": False},
}
PAGE = 100
CAP = 10_000                      # the search serves at most 10,000 hits a query
MAX_DOCS = 60                     # documents fetched a run; a longer backlog continues at the next run
MAX_DOC_BYTES = 3_000_000
DOC_MONTHS = 2                    # the months of hits whose missing documents a run fetches
MONTHS = "January|February|March|April|May|June|July|August|September|October|November|December"
RX_DATE = re.compile(rf"\b(?:{MONTHS})\.? \d{{1,2}}, \d{{4}}")
RX_MONTH = re.compile(rf"\b(?:{MONTHS}) (?:of )?\d{{4}}")
RX_QTR = re.compile(r"\b(?:first|second|third|fourth) quarter of \d{4}|\b(?:Q[1-4]|[1-4]Q) ?\d{4}", re.I)


def root() -> Path:
    return Path(settings.feeds_efts_dir)


def _hits_path(q: str, month: str) -> Path:
    return root() / q / "hits" / f"{month}.jsonl"


def doc_path(q: str, adsh: str, fname: str) -> Path:
    return root() / q / "docs" / f"{adsh}__{re.sub(r'[^A-Za-z0-9._-]', '_', fname)}.gz"


def _read_jsonl(p: Path) -> List[dict]:
    if not p.exists():
        return []
    out = []
    with open(p, encoding="utf-8") as fh:
        for line in fh:
            try:
                out.append(json.loads(line))
            except ValueError:
                continue
    return out


def _write_jsonl(p: Path, rows: Iterable[dict]) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(f".{os.getpid()}.tmp")
    with open(tmp, "w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r, default=str) + "\n")
    os.replace(tmp, p)


def hit_row(h: dict, seen: Optional[str]) -> dict:
    s = h.get("_source") or {}
    adsh, fname = (str(h.get("_id") or ":").split(":", 1) + [""])[:2]
    return {"id": h.get("_id"), "adsh": s.get("adsh") or adsh, "file": fname, "form": s.get("form"),
            "file_date": s.get("file_date"), "ciks": s.get("ciks") or [], "display_names": s.get("display_names") or [],
            "sics": s.get("sics") or [], "items": s.get("items") or [], "period_ending": s.get("period_ending"),
            "file_type": s.get("file_type"), "file_description": s.get("file_description"), "seen_utc": seen}


def store_hits(q: str, rows: List[dict]) -> List[dict]:
    """Merge ``rows`` into the query's monthly files by id (a hit's first ``seen_utc`` kept); returns the new ones."""
    by_month: Dict[str, List[dict]] = {}
    for r in rows:
        if r.get("id"):
            by_month.setdefault(str(r.get("file_date") or "0000-00")[:7], []).append(r)
    new = []
    for m, rs in by_month.items():
        p = _hits_path(q, m)
        have = {r["id"]: r for r in _read_jsonl(p)}
        add = [r for r in rs if r["id"] not in have]
        if not add:
            continue
        for r in add:
            have[r["id"]] = r
        new += add
        _write_jsonl(p, sorted(have.values(), key=lambda r: (str(r.get("file_date")), str(r["id"]))))
    return new


def hits(q: str, months: Optional[Iterable[str]] = None) -> List[dict]:
    """The stored hits of query ``q`` (the given YYYY-MM months, else all)."""
    d = root() / q / "hits"
    files = [d / f"{m}.jsonl" for m in months] if months is not None else sorted(d.glob("*.jsonl"))
    return [r for f in files for r in _read_jsonl(f)]


def search(q: dict, start: date, end: date, get: Callable) -> List[dict]:
    """Every hit of query ``q`` filed in [start, end] (pages of 100, the search's 10,000 cap)."""
    out, frm = [], 0
    while frm < CAP:
        r = get(str(settings.feeds_efts_url), params={"q": q["q"], "forms": q["forms"], "dateRange": "custom",
                                                       "startdt": start.isoformat(), "enddt": end.isoformat(),
                                                       "from": frm})
        if r.status_code != 200:
            raise RuntimeError(f"full-text search http {r.status_code}")
        js = r.json() or {}
        page = (js.get("hits") or {}).get("hits") or []
        out += page
        total = int(((js.get("hits") or {}).get("total") or {}).get("value") or 0)
        frm += PAGE
        if not page or frm >= total:
            break
    return out


def text_of(raw: bytes) -> str:
    t = raw.decode("utf-8", errors="replace")
    t = re.sub(r"(?is)<(script|style)[^>]*>.*?</\1>", " ", t)
    t = re.sub(r"<[^>]+>", " ", t)
    t = html.unescape(t).replace("\xa0", " ")
    return re.sub(r"\s+", " ", t)


def dates_in(t: str, anchor: str) -> List[dict]:
    """Each mention of ``anchor`` with the dates in its next ~220 characters (exact dates, month-years, quarters)."""
    out = []
    if not anchor:
        return out
    for m in re.finditer(re.escape(anchor), t):
        ctx = t[m.start(): m.start() + 220]
        out.append({"exact": RX_DATE.findall(ctx), "month": RX_MONTH.findall(ctx), "quarter": RX_QTR.findall(ctx),
                    "context": t[max(0, m.start() - 80): m.start() + 220]})
    return out


def date_row(r: dict, raw: bytes, anchor: str) -> dict:
    found = dates_in(text_of(raw), anchor)
    return {"id": r["id"], "adsh": r["adsh"], "file": r["file"], "file_date": r.get("file_date"),
            "ciks": r.get("ciks"), "display_names": r.get("display_names"), "sics": r.get("sics"),
            "seen_utc": r.get("seen_utc"), "mentions": len(found),
            "exact": sorted({d for f in found for d in f["exact"]}),
            "month": sorted({d for f in found for d in f["month"]}),
            "quarter": sorted({d for f in found for d in f["quarter"]}),
            "contexts": [f["context"] for f in found][:6]}


def _merge_dates(q: str, rows: List[dict]) -> None:
    if not rows:
        return
    p = root() / q / "dates.jsonl"
    have = {d["id"]: d for d in _read_jsonl(p)}
    for d in rows:
        have[d["id"]] = d
    _write_jsonl(p, sorted(have.values(), key=lambda d: (str(d.get("file_date")), str(d["id"]))))


def fetch_docs(q: str, rows: List[dict], get: Callable, deadline: Optional[float]) -> int:
    """Fetch the documents of ``rows`` not held yet (at most MAX_DOCS) and extract their dates; returns the fetched."""
    spec = QUERIES.get(q) or {}
    n, dates = 0, []
    for r in rows:
        if n >= MAX_DOCS or (deadline is not None and time.time() > deadline):
            break
        p = doc_path(q, r["adsh"], r["file"])
        if p.exists() or not r.get("file"):
            continue
        cik = str(int((r.get("ciks") or ["0"])[0]))
        resp = get(f"https://www.sec.gov/Archives/edgar/data/{cik}/{str(r['adsh']).replace('-', '')}/{r['file']}")
        n += 1
        if resp.status_code != 200:
            continue                                   # no file kept: asked again at the next run
        raw = resp.content[:MAX_DOC_BYTES]
        p.parent.mkdir(parents=True, exist_ok=True)
        tmp = p.with_suffix(f".{os.getpid()}.tmp")
        tmp.write_bytes(gzip.compress(raw))
        os.replace(tmp, p)
        dates.append(date_row(r, raw, spec.get("anchor") or ""))
    _merge_dates(q, dates)
    return n


def skip_reason(st: dict, now: datetime) -> Optional[str]:
    from src.data.feeds import sec
    if sec.quiet(now):
        return "inside production's SEC lanes"
    a = FD.age_hours(st.get("last_ok"), now)
    if a is not None and a < float(settings.feeds_efts_check_hours):
        return "checked recently"
    return None


def run(st: dict, now: Optional[datetime] = None, http_get: Optional[Callable] = None,
        deadline: Optional[float] = None) -> str:
    """Ask every saved query over its window and fetch the documents due (``st`` updated in place)."""
    from src.data.feeds import sec
    from src.utils import ET
    now = now or FD.now_utc()
    get = http_get or sec.default_get()
    end = now.astimezone(ET).date()
    seen = now.isoformat()
    windows = st.setdefault("windows", {})
    parts = []
    for name, q in QUERIES.items():
        if deadline is not None and time.time() > deadline:
            parts.append(f"{name} cut by the budget")
            continue
        last = windows.get(name)
        start = (date.fromisoformat(last) if last else end) - timedelta(days=int(settings.feeds_efts_overlap_days))
        found = search(q, start, end, get)
        new = store_hits(name, [hit_row(h, seen) for h in found])
        windows[name] = end.isoformat()
        msg = f"{name} {len(found)} hit(s), {len(new)} new"
        if q.get("docs"):
            months = sorted({(end.replace(day=1) - timedelta(days=31 * k)).strftime("%Y-%m")
                             for k in range(DOC_MONTHS)})
            k = fetch_docs(name, hits(name, months), get, deadline)
            if k:
                msg += f", {k} document(s)"
        parts.append(msg)
    return "; ".join(parts)


def import_dir(q: str, src: Path) -> int:
    """Merge a research download of query ``q`` (``hits/<year>.json`` lists of raw hits, ``docs/*.gz``) into the
    store — the hits with no ``seen_utc`` (``imported``), their documents copied and their dates extracted."""
    src = Path(src)
    spec = QUERIES[q]
    rows = []
    for f in sorted((src / "hits").glob("*.json")):
        for h in json.loads(f.read_text(encoding="utf-8")):
            r = hit_row({"_id": h.get("id"), "_source": h}, None)
            r["imported"] = True
            rows.append(r)
    store_hits(q, rows)
    dates = []
    for r in rows:
        s = src / "docs" / doc_path(q, r["adsh"], r["file"]).name
        p = doc_path(q, r["adsh"], r["file"])
        if not s.exists():
            continue
        if not p.exists():
            p.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(s, p)
        if spec.get("docs"):
            dates.append(date_row(r, gzip.decompress(p.read_bytes()), spec.get("anchor") or ""))
    _merge_dates(q, dates)
    return len(rows)


def read_dates(q: str = "pdufa") -> List[dict]:
    """The extracted dates of query ``q``'s documents (one row a document)."""
    return _read_jsonl(root() / q / "dates.jsonl")
