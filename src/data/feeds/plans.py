"""Rule 10b5-1 trading-arrangement disclosures from the SEC's Financial Statement and Notes data sets (the data feeds,
`src.data.feeds`).

Since 2023-Q3 every 10-Q / 10-K reports (Reg S-K Item 408(a), the `ecd` taxonomy) whether an officer or director
adopted or terminated a Rule 10b5-1 or non-Rule trading arrangement in the quarter and, for each one, the person's name
and title, the dates, the duration and the shares the plan may sell. The SEC's notes data sets carry those facts with
each filing's acceptance instant (sub.tsv ``accepted``, the point-in-time key; not in companyfacts or frames).

Every ``feeds_plans_check_hours`` the list page (``feeds_plans_list_url``) is read; each listed file from
``feeds_plans_start`` on (quarterly ``YYYYqN`` to 2025-Q3, then monthly ``YYYY_MM``) not held yet is downloaded once,
one a run (``feeds_plans_zip_url``; ``feeds_plans_dir``/raw/<name>_notes.zip, a monthly file about 60 MB) and its
trading-arrangement facts, their dimension rows (the person: "Individual=<name>;") and their filings' sub rows
extracted (``parsed/<name>.{facts,dims,subs}.jsonl.gz``). Older files: `import_dir` parses the free-data census's
research downloads in place. SEC requests through `sec.default_get`, none inside production's SEC lanes.
No live book reads this feed yet (PREREG213 studies it).
"""
from __future__ import annotations

import gzip
import io
import json
import os
import re
import zipfile
from datetime import datetime
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

from config import settings
from src.data import feeds as FD

PREFIXES = ("TrdArr", "Rule10b51", "NonRule10b51", "MtrlTermsOfTrdArr")
KINDS = ("facts", "dims", "subs")
MAX_VALUE = 4000                      # a text fact's value kept to this many characters


def root() -> Path:
    return Path(settings.feeds_plans_dir)


def name_key(n: str) -> Optional[Tuple[int, int]]:
    """(year, last month) of a file name: 2023q3 -> (2023, 9), 2025_10 -> (2025, 10); None for anything else."""
    m = re.fullmatch(r"(\d{4})q([1-4])", n)
    if m:
        return int(m.group(1)), 3 * int(m.group(2))
    m = re.fullmatch(r"(\d{4})_(\d{2})", n)
    if m:
        return int(m.group(1)), int(m.group(2))
    return None


def listed(get: Callable) -> List[str]:
    r = get(str(settings.feeds_plans_list_url))
    if r.status_code != 200:
        raise RuntimeError(f"the notes list page answered http {r.status_code}")
    names = set(re.findall(r"financial-statement-notes-data-sets/([0-9a-z_]+)_notes\.zip", r.text))
    names = sorted((n for n in names if name_key(n)), key=name_key)
    if not names:
        raise ValueError("the notes list page names no file")
    return names


def held() -> List[str]:
    return sorted({p.name.split(".")[0] for p in (root() / "parsed").glob("*.facts.jsonl.gz")}, key=lambda n: name_key(n) or (0, 0))


def due_names(names: List[str]) -> List[str]:
    start = name_key(str(settings.feeds_plans_start)) or (0, 0)
    have = set(held())
    return [n for n in names if (name_key(n) or (0, 0)) >= start and n not in have]


def _rows(z: zipfile.ZipFile, member: str, keep):
    with z.open(member) as fh:
        tf = io.TextIOWrapper(fh, encoding="utf-8", errors="replace", newline="\n")
        hdr = tf.readline().rstrip("\n").split("\t")
        for line in tf:
            if keep(line):
                yield dict(zip(hdr, line.rstrip("\n").split("\t")))


def parse_zip(z: zipfile.ZipFile) -> Dict[str, List[dict]]:
    """The trading-arrangement facts of one notes file (txt.tsv and num.tsv), their dim.tsv rows and sub.tsv rows."""
    names = set(z.namelist())
    if "sub.tsv" not in names:
        raise ValueError("not a notes data set (no sub.tsv)")
    probe = tuple("\t" + p for p in PREFIXES)
    facts = []
    for member, kind in (("txt.tsv", "txt"), ("num.tsv", "num")):
        if member not in names:
            continue
        for d in _rows(z, member, lambda line: any(p in line for p in probe)):
            if not str(d.get("tag", "")).startswith(PREFIXES):
                continue
            d["kind"] = kind
            if d.get("value") and len(d["value"]) > MAX_VALUE:
                d["value"] = d["value"][:MAX_VALUE]
            facts.append(d)
    adsh = {f.get("adsh") for f in facts}
    dimh = {f.get("dimh") for f in facts} - {"", None, "0x00000000"}
    dims = list(_rows(z, "dim.tsv", lambda line: line.split("\t", 1)[0] in dimh)) if "dim.tsv" in names and dimh else []
    subs = list(_rows(z, "sub.tsv", lambda line: line.split("\t", 1)[0] in adsh)) if adsh else []
    return {"facts": facts, "dims": dims, "subs": subs}


def write_parsed(name: str, got: Dict[str, List[dict]]) -> None:
    d = root() / "parsed"
    d.mkdir(parents=True, exist_ok=True)
    for kind in KINDS:
        p = d / f"{name}.{kind}.jsonl.gz"
        tmp = p.with_suffix(f".{os.getpid()}.tmp")
        with gzip.open(tmp, "wt", encoding="utf-8") as fh:
            for r in got[kind]:
                fh.write(json.dumps(r) + "\n")
        os.replace(tmp, p)


def skip_reason(st: dict, now: datetime) -> Optional[str]:
    from src.data.feeds import sec
    if sec.quiet(now):
        return "inside production's SEC lanes"
    a = FD.age_hours(st.get("last_ok"), now)
    if a is not None and a < float(settings.feeds_plans_check_hours):
        return "checked recently"
    return None


def run(st: dict, now: Optional[datetime] = None, http_get: Optional[Callable] = None,
        deadline: Optional[float] = None) -> str:
    """Download and parse the oldest listed file not held yet (one a run). Returns a summary."""
    from src.data.feeds import sec
    get = http_get or sec.default_get()
    names = listed(get)
    due = due_names(names)
    st["listed_newest"] = names[-1]
    if not due:
        return f"nothing new (newest listed {names[-1]})"
    n = due[0]
    r = get(str(settings.feeds_plans_zip_url).format(name=n))
    if r.status_code != 200:
        raise RuntimeError(f"http {r.status_code} for {n}")
    z = zipfile.ZipFile(io.BytesIO(r.content))
    got = parse_zip(z)
    raw = root() / "raw" / f"{n}_notes.zip"
    raw.parent.mkdir(parents=True, exist_ok=True)
    tmp = raw.with_suffix(f".{os.getpid()}.tmp")
    tmp.write_bytes(r.content)
    os.replace(tmp, raw)
    write_parsed(n, got)
    return (f"{n}: {len(got['facts'])} trading-arrangement facts in {len(got['subs'])} filings"
            + (f"; {len(due) - 1} more due" if len(due) > 1 else ""))


def import_dir(src: Path) -> int:
    """Parse every ``<name>_notes.zip`` of a research download in place (the raw zips are not copied); returns the
    files parsed."""
    k = 0
    for f in sorted(Path(src).glob("*_notes.zip")):
        n = f.name[: -len("_notes.zip")]
        if not name_key(n) or n in set(held()):
            continue
        with zipfile.ZipFile(f) as z:
            write_parsed(n, parse_zip(z))
        k += 1
    return k


def read(kind: str = "facts") -> List[dict]:
    """Every parsed row of one kind ("facts", "dims" or "subs") across the files held, each with its ``file``."""
    out = []
    for p in sorted((root() / "parsed").glob(f"*.{kind}.jsonl.gz")):
        with gzip.open(p, "rt", encoding="utf-8") as fh:
            for line in fh:
                try:
                    r = json.loads(line)
                except ValueError:
                    continue
                r["file"] = p.name.split(".")[0]
                out.append(r)
    return out
