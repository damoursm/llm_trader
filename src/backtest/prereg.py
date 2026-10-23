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

THREE MORE CHECKS (2026-10-21; the user's "Fix all" after the audit of 2026-10-20, which found these slips recurring
because only memory guarded them — a stamp written without reading the clock, PREREG171 repeating PREREG111 unnoticed,
PREREG169's registered "reported beside" overlap left out of its results):

* STAMPS (`stamp_findings`): every stamped section — the registration, each amendment, each RESULTS heading, each
  ``NOTE (...)`` — must carry an exact instant (no "~", no "21:4x"), follow the previous section's instant in file
  order, not postdate the PREREG file's own last write (a stamp later than the file was written is not the clock's),
  not lie in the future, and a RESULTS stamp must not precede the last write of a result file its heading names (the
  results changed after they were reported, or the stamp is not the clock's). ``--now`` prints the clock's stamp in
  the house form to paste.
* PRIOR STUDIES (`prior_studies`): the registration's text is matched against every study row of the research log
  (`docs/RESEARCH.md`, one line per study; ``--also`` adds the memory index) — rare words shared weigh most — and the
  closest earlier studies are listed (``--prior`` before registering); the registration must hold a
  ``Prior studies:`` line naming what it repeats or how it differs.
* PROMISED OUTPUTS (`promised`): every output a registration or an amendment promises beside its verdict is tagged
  ``[R1]``, ``[R2]`` ... where it is promised, and each tag must reappear in a later RESULTS section or NOTE (reported,
  or ``[R2] dropped: why``); a registration that promises outputs ("reported ...") without tags fails.

These three gate the registrations from `CONVENTION_FROM` (a study registered before it is listed with its findings,
never failed by them: its files kept the earlier conventions).

    python -m src.backtest.prereg studies/prereg94 [studies/prereg95 ...]   -> one line per study, exit 1 on any
                                                                             violation or unverifiable study
    python -m src.backtest.prereg --prior studies/prereg172                 -> the closest earlier studies
    python -m src.backtest.prereg --now                                     -> "2026-10-21 02:15 ET (Wednesday)"
"""
from __future__ import annotations

import argparse
import fnmatch
import math
import re
import sys
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple
from zoneinfo import ZoneInfo

ET = ZoneInfo("America/New_York")
RESULT_SUFFIXES = (".json", ".pkl", ".npy", ".csv", ".txt", ".log")
SKIP_DIRS = ("__pycache__",)
REPO = Path(__file__).resolve().parents[2]
RESEARCH_LOG = REPO / "docs" / "RESEARCH.md"
CONVENTION_FROM = date(2026, 10, 21)    # the stamp, prior-studies and promised-output checks gate registrations from here
PRIOR_TOP = 5                           # the closest earlier studies listed

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


# ── the sections of a PREREG file: stamps, prior studies, promised outputs (2026-10-21) ─────────────────────────

# a stamp as written anywhere: an exact one, or an approximate / malformed one ("~21:45", "21:4x", "around 21:45")
_ANY_STAMP = re.compile(r"(?P<d>\d{4}-\d{2}-\d{2}),?\s+(?P<approx>~\s*|about\s+|around\s+|ca?\.\s*)?"
                        r"(?P<hh>\d{1,2}):(?P<mm>[0-9xX?]{1,2})\s*ET")
_RESULTS_HEAD = re.compile(r"^#+\s*RESULTS?\b", re.IGNORECASE)
_AMEND_HEAD = re.compile(r"^#+\s*Amendments?\b")
_NOTE_HEAD = re.compile(r"^NOTE\b")
_TAG = re.compile(r"\[R\d+[a-z]?\]")
_PROMISE = re.compile(r"\breported\b|\breport(?:s|ed)? beside\b|\balso report", re.IGNORECASE)
_PRIOR_LINE = re.compile(r"^\W*prior stud(?:y|ies)\s*:", re.IGNORECASE | re.MULTILINE)
_NAMED_FILE = re.compile(r"[\w./*-]+\.(?:json|pkl|csv|npy|txt|log)\b")


def sections(md_path: Path) -> List[Dict[str, object]]:
    """A PREREG file's sections in file order: ``kind`` — "registration" (the text before the first amendment,
    RESULTS heading or NOTE), "amendment", "results" or "note" (a heading of another kind continues the section it
    sits in) —, ``line`` (its first line), ``head`` (that line), ``text`` (its lines) and its stamp: ``at`` (an aware
    ET datetime or None), ``raw`` (the stamp as written), ``exact`` (an exact HH:MM, no approximation)."""
    lines = md_path.read_text(encoding="utf-8", errors="replace").splitlines()
    out: List[Dict[str, object]] = []
    cur: Dict[str, object] = {"kind": "registration", "line": 1, "head": lines[0] if lines else "", "lines": []}
    for n, line in enumerate(lines, 1):
        kind = ("amendment" if _AMEND_HEAD.match(line) else "results" if _RESULTS_HEAD.match(line)
                else "note" if _NOTE_HEAD.match(line) else None)
        if kind is not None:
            out.append(cur)
            cur = {"kind": kind, "line": n, "head": line, "lines": []}
        cur["lines"].append(line)
    out.append(cur)
    for s in out:
        s["text"] = "\n".join(s.pop("lines"))
        src = s["head"] if s["kind"] != "registration" else next(
            (ln for ln in s["text"].splitlines() if _REGISTRATION.search(ln) or _ANY_STAMP.search(ln)), "")
        m = _ANY_STAMP.search(src)
        s["raw"] = m.group(0) if m else None
        s["exact"] = bool(m) and not m["approx"] and m["mm"].isdigit() and len(m["mm"]) == 2
        s["at"] = (datetime.strptime(f"{m['d']} {int(m['hh']):02d}:{m['mm']}", "%Y-%m-%d %H:%M").replace(tzinfo=ET)
                   if s["exact"] else None)
    return out


def stamp_findings(md_path: Path, study_dir: Optional[Path] = None, now: Optional[datetime] = None) -> List[str]:
    """What is wrong with a PREREG file's stamps (the module docstring): an approximate or malformed stamp, a section
    stamped before the one above it, a stamp after the file's own last write or in the future, a RESULTS stamp
    before the last write of a result file its heading names. One line per finding."""
    now = now or datetime.now(ET)
    written = datetime.fromtimestamp(md_path.stat().st_mtime, tz=timezone.utc).astimezone(ET)
    files = {f["name"]: f["mtime"] for f in result_files(study_dir)} if study_dir is not None else {}
    out: List[str] = []
    prev: Optional[Tuple[datetime, int]] = None
    for s in sections(md_path):
        where = f"{md_path.name}:{s['line']} ({s['kind']})"
        if s["raw"] is None:
            if s["kind"] != "registration":
                out.append(f"{where}: no stamp ('YYYY-MM-DD HH:MM ET')")
            continue
        if not s["exact"]:
            out.append(f"{where}: not an exact clock reading: '{s['raw']}'")
            continue
        at: datetime = s["at"]
        if prev is not None and at < prev[0]:
            out.append(f"{where}: stamped {at:%Y-%m-%d %H:%M}, before the section above it (line {prev[1]}, "
                       f"{prev[0]:%H:%M})")
        if at > written:
            out.append(f"{where}: stamped {at:%Y-%m-%d %H:%M}, after the file was last written ({written:%H:%M:%S})")
        if at > now:
            out.append(f"{where}: stamped in the future ({at:%Y-%m-%d %H:%M})")
        if s["kind"] == "results" and files:
            for pat in sorted(set(_NAMED_FILE.findall(str(s["head"])))):
                base = pat.split("/")[-1]
                for name, mt in files.items():
                    if fnmatch.fnmatch(name.split("/")[-1], base) and mt >= at + timedelta(minutes=1):
                        out.append(f"{where}: stamped {at:%H:%M} but its result {name} was written later "
                                   f"({mt:%Y-%m-%d %H:%M:%S})")
        prev = (at, int(s["line"]))
    return out


def promised(md_path: Path) -> Dict[str, object]:
    """The outputs a PREREG file promises (tags ``[R1]``... in its registration and amendments) and whether each was
    delivered (the tag again in a later RESULTS section or NOTE): {"declared": [tags], "undelivered": [tags],
    "untagged": a registration / amendment that promises outputs ("reported ...") without a tag (its line), or None,
    "reported": any RESULTS section exists}."""
    secs = sections(md_path)
    declared: Dict[str, int] = {}
    untagged = None
    for i, s in enumerate(secs):
        if s["kind"] not in ("registration", "amendment"):
            continue
        tags = _TAG.findall(str(s["text"]))
        for t in tags:
            declared.setdefault(t, i)
        if not tags and untagged is None and _PROMISE.search(str(s["text"])):
            untagged = int(s["line"])
    later = lambda i: " ".join(str(s["text"]) for s in secs[i + 1:] if s["kind"] in ("results", "note"))  # noqa: E731
    undelivered = [t for t, i in declared.items() if t not in later(i)]
    return {"declared": sorted(declared), "undelivered": sorted(undelivered), "untagged": untagged,
            "reported": any(s["kind"] == "results" for s in secs)}


_STOP = set("""a an and are as at be been before being beside both but by can each every for from has have how if in
into is it its no not of on one or our over per than that the their then there these this those through to under
until was were what when where whether which while who why with within without would after again all also any only
other same so such them they too very new now fixed once read more most first last two three four five six seven
eight nine ten registered registration amendment results result note stage study studies prereg rule rules test
tests trade trades traded book books arm arms model models account live day days session sessions year years return
returns per pooled history set sets pp lp draw draws bar bars long short shorts entry exit hold held price prices
user data signal signals candidate candidates verdict whole none""".split())


def _tokens(text: str) -> List[str]:
    t = re.sub(r"(?<=[A-Za-z])-(?=[A-Za-z])", "", str(text).lower())        # lock-up -> lockup, pre-FOMC -> prefomc
    out = []
    for w in re.findall(r"[a-z][a-z0-9]+", t):
        if len(w) > 4 and w.endswith("s") and not w.endswith("ss"):
            w = w[:-1]                                                        # shorts -> short, lockups -> lockup
        if len(w) >= 3 and w not in _STOP:
            out.append(w)
    return out


def research_rows(paths: Sequence[Path]) -> List[Tuple[str, str]]:
    """(label, text) of every study row of the research log (``| 111 / 112 | 10-14 | ... |``) and every line of an
    added index (a memory index's ``- [title](file.md) — hook``)."""
    out: List[Tuple[str, str]] = []
    for p in paths:
        try:
            lines = Path(p).read_text(encoding="utf-8", errors="replace").splitlines()
        except OSError:
            continue
        for ln in lines:
            m = re.match(r"^\|\s*([^|]+?)\s*\|\s*(\d{2}-\d{2})\s*\|(.*)$", ln)       # | id | MM-DD | area | ... |
            if m:
                ids = m.group(1)
                lab = ("PREREG" + "/".join(re.findall(r"\d+", ids))) if re.fullmatch(r"[0-9 /,]+", ids) else ids
                out.append((lab, m.group(2) + " |" + m.group(3)))
            elif ln.startswith("- [") and Path(p).name.upper() != "RESEARCH.MD":
                out.append((re.sub(r"^- \[([^\]]+)\].*$", r"\1", ln)[:60], ln))
    return out


TITLE_WEIGHT = 3                         # the registration's title line counts this many times in the search


def prior_studies(text: str, rows: Sequence[Tuple[str, str]], exclude: Sequence[str] = (),
                  top: int = PRIOR_TOP) -> List[Dict[str, object]]:
    """The rows of ``rows`` closest to ``text`` (a registration, its first line its title): the cosine of their
    word-count x rarity vectors (rarity = log(N / the rows holding the word): rare shared words weigh most, the
    vectors' lengths are normalized so a long row does not win on common words; the title counts `TITLE_WEIGHT`
    times); rows whose label holds a number in ``exclude`` (the study itself) are left out. The ``top`` rows sharing
    at least two words: [{"label", "score" (the cosine, 0..1), "shared", "text"}]."""
    lines = str(text).strip().splitlines()
    qt = _tokens(lines[0] if lines else "") * TITLE_WEIGHT + _tokens(text)
    docs = [(lab, txt, _tokens(txt)) for lab, txt in rows]
    docs = [d for d in docs if not any(x and re.search(rf"(?<!\d){re.escape(str(x))}(?!\d)", d[0]) for x in exclude)]
    if not qt or not docs:
        return []
    df: Dict[str, int] = {}
    for _, _, ws in docs:
        for w in set(ws):
            df[w] = df.get(w, 0) + 1
    N = len(docs)
    idf = lambda w: math.log((N + 1) / (df.get(w, 0) + 1)) + 1.0      # noqa: E731 — a word no row holds weighs most

    def vec(ws):
        v: Dict[str, float] = {}
        for w in ws:
            v[w] = v.get(w, 0.0) + idf(w)
        n = math.sqrt(sum(x * x for x in v.values()))
        return {w: x / n for w, x in v.items()} if n else {}
    q = vec(qt)
    scored = []
    for lab, txt, ws in docs:
        d = vec(ws)
        shared = set(q) & set(d)
        if len(shared) < 2:
            continue
        score = sum(q[w] * d[w] for w in shared)
        scored.append({"label": lab, "score": round(score, 3),
                       "shared": sorted(shared, key=lambda w: -q[w] * d[w])[:8], "text": txt.strip()[:160]})
    return sorted(scored, key=lambda r: -r["score"])[:top]


def _registration_text(md_path: Path) -> str:
    return str(next((s["text"] for s in sections(md_path) if s["kind"] == "registration"), ""))


def _own_numbers(study_dir: Path, md_files: Sequence[Path]) -> List[str]:
    nums = set(re.findall(r"\d+", study_dir.name))
    for md in md_files:
        nums.update(re.findall(r"\d+", md.stem))
    return sorted(nums)


def check(study_dir, research_log: Optional[Sequence[Path]] = None, now: Optional[datetime] = None) -> Dict[str, object]:
    """Check one study directory. Returns ``{"study", "registrations", "files", "violations", "problems", "ok",
    "stamps", "prior", "promised", "gated"}``:

    * ``registrations`` — every registration / amendment found, in file order, each with ``files_before`` (the
      result files modified before it) and, for an amendment, ``files_since_registration`` (modified between the
      first registration and the amendment);
    * ``files`` — the result files with their ET modification times;
    * ``violations`` — the result files modified before the FIRST registration;
    * ``stamps`` — `stamp_findings` of every PREREG file; ``prior`` — the closest earlier studies of the research log
      (`prior_studies`, ``research_log`` = the files searched, default `RESEARCH_LOG`) and whether the registration
      holds its ``Prior studies:`` line; ``promised`` — `promised` per PREREG file;
    * ``gated`` — the first registration is on or after `CONVENTION_FROM`: the stamp findings, a missing ``Prior
      studies:`` line and undelivered or untagged promised outputs (once any RESULTS section exists) are problems;
      before it they are listed only;
    * ``findings`` — those findings, gated or not; ``failed`` — the same when gated (else empty);
    * ``problems`` — why the study cannot be verified (no PREREG file, no registration instant);
    * ``ok`` — no violation, no problem and nothing failed.

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
    gated = first is not None and first.date() >= CONVENTION_FROM
    stamps: List[str] = []
    prom: Dict[str, dict] = {}
    reg_text = ""
    for md in md_files:
        stamps.extend(stamp_findings(md, study_dir, now))
        prom[md.name] = promised(md)
        reg_text += "\n" + _registration_text(md)
    rows = research_rows(research_log if research_log is not None else [RESEARCH_LOG])
    prior = {"line": bool(_PRIOR_LINE.search(reg_text)),
             "candidates": prior_studies(reg_text, rows, _own_numbers(study_dir, md_files)) if reg_text.strip() else []}
    findings: List[str] = list(stamps)
    if md_files and not prior["line"]:
        findings.append("no 'Prior studies:' line in the registration (search: python -m src.backtest.prereg --prior "
                        f"{study_dir.as_posix()})")
    for name, p in prom.items():
        if p["untagged"] is not None:
            findings.append(f"{name}:{p['untagged']}: outputs promised without [R1]-style tags")
        if p["reported"] and p["undelivered"]:
            findings.append(f"{name}: promised outputs never reported: {', '.join(p['undelivered'])}")
    failed = list(findings) if gated else []
    return {
        "study": study_dir.name,
        "first_registration": _iso(first) if first is not None else None,
        "first_registration_approximate": first_approx,
        "registrations": registrations,
        "files": [{"name": f["name"], "mtime": _iso(f["mtime"]), "size": f["size"]} for f in files],
        "violations": violations,
        "stamps": stamps,
        "prior": prior,
        "promised": prom,
        "gated": gated,
        "findings": findings,
        "failed": failed,
        "problems": problems,
        "ok": not violations and not problems and not failed,
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
    if result.get("failed"):
        return f"{study}: FAILED - {len(result['failed'])} check(s) of its registration; {head}"
    return f"{study}: OK - {head}, none before the registration"


def now_stamp(now: Optional[datetime] = None) -> str:
    """The clock's stamp in the house form: "2026-10-21 02:15 ET (Wednesday)"."""
    t = (now or datetime.now(ET)).astimezone(ET)
    return f"{t:%Y-%m-%d %H:%M} ET ({t:%A})"


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Check that a study's result files postdate its pre-registration.")
    ap.add_argument("study_dir", nargs="*", help="study directories (each holding PREREG*.md and its results)")
    ap.add_argument("--amendments", action="store_true",
                    help="also list, per amendment, the result files that existed when it was registered")
    ap.add_argument("--prior", action="store_true",
                    help="only list the closest earlier studies of each study's registration (before registering)")
    ap.add_argument("--text", help="with --prior: search for this text instead of a registration")
    ap.add_argument("--also", action="append", default=[],
                    help="another index to search beside docs/RESEARCH.md (e.g. the memory index)")
    ap.add_argument("--now", action="store_true", help="print the clock's stamp in the house form and exit")
    args = ap.parse_args(argv)
    if args.now:
        print(now_stamp())
        return 0
    logs = [RESEARCH_LOG] + [Path(x) for x in args.also]
    if args.prior:
        rows = research_rows(logs)
        jobs = [("text", args.text, [])] if args.text else []
        for d in args.study_dir:
            mds = sorted(Path(d).glob("PREREG*.md"))
            jobs.append((Path(d).name, "\n".join(_registration_text(m) for m in mds), _own_numbers(Path(d), mds)))
        for name, text, own in jobs:
            print(f"{name}: the closest earlier studies")
            for c in prior_studies(text or "", rows, own):
                print(f"    {c['label']}  score {c['score']:.2f}  shared: {', '.join(c['shared'])}")
                print(f"        {c['text'][:150]}")
        return 0
    if not args.study_dir:
        ap.error("a study directory is needed (or --now / --prior --text)")
    worst = 0
    for d in args.study_dir:
        res = check(d, logs)
        print(verdict_line(res))
        for f in res.get("findings") or []:
            print(f"    {'FAILED' if res.get('gated') else 'note'}: {f}")
        if res.get("prior", {}).get("candidates"):
            print("    closest earlier studies: " + "; ".join(f"{c['label']} ({c['score']:.2f})"
                                                       for c in res["prior"]["candidates"][:3]))
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
