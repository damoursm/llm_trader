"""The pre-registration checker: did a study's results exist before its plan was registered?

Every study under `studies/preregNN/` opens with a registration instant — "Registered 2026-10-12 11:38 ET (Monday),
before any number below was computed." in the recent files, "(stamped 2026-10-10 22:48 ET, before any run)" in the
earlier ones — and may add amendments, "## Amendment 1 (registered 2026-10-12 11:41 ET, before F4 is computed)".
The house bar (CLAUDE.md, Standing rules) is pre-registered: a result that was computed BEFORE the plan was written
is not a test of the plan. This module makes that mechanical: it reads the instants out of every ``PREREG*.md`` in
a study directory and compares them with the modification times of the directory's result files (``*.json``,
``*.pkl``, ``*.npy``, ``*.csv``, ``*.txt``, ``*.log``; every subdirectory but ``__pycache__``).

* A VIOLATION is a result file modified before the directory's FIRST registration (the earliest registration
  instant over its PREREG files: `prereg44/` holds PREREG44 and PREREG52-56, whose results legitimately predate the
  later plans).
* Amendments are informational: for each, the result files modified between the first registration and the
  amendment are listed (what already existed when the plan was amended), never a violation.
* A directory without a parsable registration cannot be verified, so it is NOT ok (`problems` says why) — an
  unverifiable study must not read as a clean one.

Times: the instants are ET (America/New_York, through zoneinfo); a file's ``st_mtime`` is UTC epoch seconds,
converted to ET before the comparison. A registration is at the start of its minute, so a file written later in the
same minute is after it. Modification times are what the filesystem says — a copied or re-saved file carries its
copy's time, so a clean verdict is evidence, not proof.

    python -m src.backtest.prereg studies/prereg94 [studies/prereg95 ...]   -> one line per study, exit 1 on any
                                                                             violation or unverifiable study
"""
from __future__ import annotations

import argparse
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional
from zoneinfo import ZoneInfo

ET = ZoneInfo("America/New_York")
RESULT_SUFFIXES = (".json", ".pkl", ".npy", ".csv", ".txt", ".log")
SKIP_DIRS = ("__pycache__",)

# "2026-10-12 11:38 ET"; a "~" before the time (PREREG44, 87: "2026-10-11 ~14:50 ET") marks an approximate instant.
_STAMP = r"(?P<d>\d{4}-\d{2}-\d{2}) (?P<approx>~)?(?P<hh>\d{2}):(?P<mm>\d{2}) ET"
# "Registered 2026-10-12 11:38 ET" (capital R: the plan's own line, PREREG88 on); the earlier forms "(stamped
# 2026-10-10 22:48 ET" (PREREG64-86) and "Saved 2026-10-08 23:35 ET (clock read)" (PREREG44-63).
_REGISTRATION = re.compile(r"(?:\bRegistered|\bSaved|\(stamped) " + _STAMP)
# "## Amendment 1 (registered 2026-10-12 11:41 ET" — also "(2026-10-09 00:47 ET", "(stamped ..." and a dash-titled
# "## Amendment 1 — the user's correction (2026-10-09 14:49 ET": the earlier forms; "## Amendments (..." too.
_AMENDMENT = re.compile(r"^#+\s*(?P<word>Amendments?)\s*(?P<label>[^\s(]*)[^(\n]*\((?:registered |stamped )?" + _STAMP)


def _instant(m: "re.Match[str]") -> datetime:
    return datetime.strptime(f"{m['d']} {m['hh']}:{m['mm']}", "%Y-%m-%d %H:%M").replace(tzinfo=ET)


def _iso(dt: datetime) -> str:
    return dt.astimezone(ET).isoformat(timespec="seconds")


def parse_registrations(md_path: Path) -> List[Dict[str, object]]:
    """The registration and amendment instants of one PREREG file, in file order: dicts with ``kind``
    (``registration`` / ``amendment``), ``label``, ``at`` (an aware ET datetime), ``approximate`` (the instant
    was written with a "~"), ``line`` and ``prereg``."""
    out: List[Dict[str, object]] = []
    for n, line in enumerate(md_path.read_text(encoding="utf-8", errors="replace").splitlines(), 1):
        m = _AMENDMENT.search(line)
        if m:
            label = " ".join(x for x in (m.group("word"), m.group("label")) if x)
            out.append({"kind": "amendment", "label": label, "at": _instant(m), "approximate": bool(m["approx"]),
                        "line": n, "prereg": md_path.name})
            continue
        m = _REGISTRATION.search(line)
        if m:
            out.append({"kind": "registration", "label": "Registered", "at": _instant(m),
                        "approximate": bool(m["approx"]), "line": n, "prereg": md_path.name})
    return out


def result_files(study_dir: Path) -> List[Dict[str, object]]:
    """The directory's result files (recursive, ``__pycache__`` skipped) with their modification instants in ET,
    oldest first (then by name)."""
    rows = []
    for p in study_dir.rglob("*"):
        if not p.is_file() or p.suffix.lower() not in RESULT_SUFFIXES:
            continue
        if any(part in SKIP_DIRS for part in p.relative_to(study_dir).parts[:-1]):
            continue
        st = p.stat()
        mtime = datetime.fromtimestamp(st.st_mtime, tz=timezone.utc).astimezone(ET)
        rows.append({"name": p.relative_to(study_dir).as_posix(), "mtime": mtime, "size": int(st.st_size)})
    return sorted(rows, key=lambda r: (r["mtime"], r["name"]))


def check(study_dir) -> Dict[str, object]:
    """Check one study directory. Returns ``{"study", "registrations", "files", "violations", "problems", "ok"}``:

    * ``registrations`` — every registration / amendment found, in file order, each with ``files_before`` (the
      result files modified before it) and, for an amendment, ``files_since_registration`` (modified between the
      first registration and the amendment);
    * ``files`` — the result files with their ET modification times;
    * ``violations`` — the result files modified before the FIRST registration;
    * ``problems`` — why the study cannot be verified (no PREREG file, no registration instant);
    * ``ok`` — no violation and no problem.

    Every instant is an ISO-8601 ET string; amendments never create a violation.
    """
    study_dir = Path(study_dir)
    problems: List[str] = []
    if not study_dir.is_dir():
        return {"study": study_dir.name, "registrations": [], "files": [], "violations": [],
                "problems": [f"not a directory: {study_dir}"], "ok": False}
    md_files = sorted(p for p in study_dir.glob("PREREG*.md") if p.is_file())
    regs: List[Dict[str, object]] = []
    for md in md_files:
        found = parse_registrations(md)
        if not any(r["kind"] == "registration" for r in found):
            problems.append(f"{md.name}: no registration instant ('Registered YYYY-MM-DD HH:MM ET')")
        regs.extend(found)
    if not md_files:
        problems.append("no PREREG*.md in the directory")
    files = result_files(study_dir)
    first: Optional[datetime] = min((r["at"] for r in regs if r["kind"] == "registration"), default=None)

    registrations = []
    for r in regs:
        row = {"kind": r["kind"], "label": r["label"], "prereg": r["prereg"], "line": r["line"], "at": _iso(r["at"]),
               "approximate": r["approximate"], "files_before": [f["name"] for f in files if f["mtime"] < r["at"]]}
        if r["kind"] == "amendment" and first is not None:
            row["files_since_registration"] = [f["name"] for f in files if first <= f["mtime"] < r["at"]]
        registrations.append(row)
    violations = []
    if first is not None:
        violations = [{"file": f["name"], "mtime": _iso(f["mtime"]), "registered_at": _iso(first)}
                      for f in files if f["mtime"] < first]
    first_approx = any(r["approximate"] for r in regs if r["kind"] == "registration" and r["at"] == first)
    return {
        "study": study_dir.name,
        "first_registration": _iso(first) if first is not None else None,
        "first_registration_approximate": first_approx,
        "registrations": registrations,
        "files": [{"name": f["name"], "mtime": _iso(f["mtime"]), "size": f["size"]} for f in files],
        "violations": violations,
        "problems": problems,
        "ok": not violations and not problems,
    }


def verdict_line(result: Dict[str, object]) -> str:
    """One line per study for the CLI (plain ASCII: the Windows console is cp1252)."""
    study = result["study"]
    if result["problems"]:
        return f"{study}: UNVERIFIED - " + "; ".join(result["problems"])
    n_amend = sum(1 for r in result["registrations"] if r["kind"] == "amendment")
    amend = f", {n_amend} amendment{'s' if n_amend != 1 else ''}" if n_amend else ""
    approx = "~" if result.get("first_registration_approximate") else ""
    head = f"registered {approx}{result['first_registration']}{amend}; {len(result['files'])} result files"
    if result["violations"]:
        names = ", ".join(f"{v['file']} ({v['mtime']})" for v in result["violations"][:5])
        more = f" (+{len(result['violations']) - 5} more)" if len(result["violations"]) > 5 else ""
        return f"{study}: VIOLATION - {len(result['violations'])} predate the registration: {names}{more}; {head}"
    return f"{study}: OK - {head}, none before the registration"


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Check that a study's result files postdate its pre-registration.")
    ap.add_argument("study_dir", nargs="+", help="study directories (each holding PREREG*.md and its results)")
    ap.add_argument("--amendments", action="store_true",
                    help="also list, per amendment, the result files that existed when it was registered")
    args = ap.parse_args(argv)
    worst = 0
    for d in args.study_dir:
        res = check(d)
        print(verdict_line(res))
        if args.amendments:
            for r in res["registrations"]:
                if r["kind"] == "amendment":
                    since = r.get("files_since_registration", [])
                    print(f"    {r['label']} ({r['at']}): {len(since)} result files since the registration"
                          + (": " + ", ".join(since) if since else ""))
        if not res["ok"]:
            worst = 1
    return worst


if __name__ == "__main__":
    sys.exit(main())
