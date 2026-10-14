"""THE TRIAL LEDGER (PREREG106 item 1, registered 2026-10-14 13:09 ET; user 2026-10-14: "can you think about a way to
implement some guardrails against over fitting? I want us to be able to evaluate over fitting").

Every candidate a study evaluates is recorded ONCE: the study, the MODEL it is a variant of (an arm or book: vol,
etf, fall, dip, crash, ...; "combination" for the live account's own configuration — its sizing, its mix), the
variant, the sets it was judged on, its metrics, the decision and its KIND ("rule", "sizing", "mechanism", ...). The
rows go to an append-only JSON-lines file, `cache/ml/trials/ledger.jsonl`, anchored at the repository whatever the
caller's working directory (a study run from its own folder writes the same ledger). The ledger answers how many
trials each model has had (`count`), how many times each set has been looked at and which decisions it carried
(`set_uses`).

THE SEED. What came before the ledger is PREREG103's reconstruction of every PREREG file (`global103.json`,
``trials.counts``). Per model it gives a LOW count (the candidates the files name) and a HIGH one (every distinct
configuration evaluated, Optuna trials and grids included). `seed_from_prereg103` writes them as rows of kind "seed"
(``source`` "seed:PREREG103") — the starting point of `count`, never a recorded trial. Seed rows name no set: the
files mostly predate the sets' definition. `set_uses` therefore counts recorded trials only.

COUNTING keeps PREREG103's convention. A recorded trial adds one to its model's low and high counts. The
combination's HIGH count also takes every recorded arm trial (an arm's variant is judged inside the account); its LOW
count takes only the trials recorded under "combination". The counts are NOMINAL: correlated variants of one rule
are far fewer independent trials (PREREG103's caveat), so a deflation by them is conservative.

APPEND-ONLY, IDEMPOTENT per (study, variant, kind). A row is never rewritten or removed. Calling `record` twice with
the same payload appends nothing ("unchanged"). A new payload for a known key appends a REVISION — say a decision
that moved from "more draws" to "add" at 40 draws: the key still counts once, and its latest row is its current
state. Writes hold an OS-level exclusive lock beside the file, since studies run in parallel. Nothing here decides
anything: `add_rule.verdict(model=...)` reads the counts for its deflated report, and `audit` reads the counts and
the set uses.

    python -m src.backtest.trials --seed            # write PREREG103's seed (idempotent)
    python -m src.backtest.trials --status          # every model's count and every set's uses
"""
from __future__ import annotations

import json
import math
import os
import threading
import time
from collections import Counter
from contextlib import contextmanager
from datetime import date, datetime
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Mapping, Optional, Sequence, Tuple

from loguru import logger

REPO = Path(__file__).resolve().parents[2]
LEDGER = REPO / "cache" / "ml" / "trials" / "ledger.jsonl"
ET = "America/New_York"
VERSION = 1
# the sets a trial may name: the borrow-data test sets (`periods`: A0, A, B, C, D) and PREREG103's LONG histories
# (the dip book's 2007-2020 daily cache, the crash book's QQQ 1988-2020) — one list, so `set_uses` reads cleanly
SETS = ("A0", "A", "B", "C", "D", "LONG")
SEED_KIND = "seed"                       # reserved: PREREG103's counts, written by `seed_from_prereg103` only
SEED_STUDY = "PREREG103"
COMBINATION = "combination"
LOCK_TIMEOUT_S = 60.0
_THREAD_LOCK = threading.Lock()          # the OS lock is per process; threads of one study take this one first


# ── the file ─────────────────────────────────────────────────────────────────

def ledger_path(path=None) -> Path:
    """The ledger file: ``path`` when given, else `LEDGER`."""
    return Path(path) if path is not None else LEDGER



@contextmanager
def _locked(path: Path) -> Iterator[None]:
    """An OS-level EXCLUSIVE lock on ``<ledger>.lock`` (dropped by the OS if the process dies), waited for up to
    `LOCK_TIMEOUT_S`: the read-check-append of `record` is atomic across processes and threads."""
    lock = path.with_name(path.name + ".lock")
    lock.parent.mkdir(parents=True, exist_ok=True)
    with _THREAD_LOCK:
        fh = open(lock, "a+b")
        try:
            deadline = time.monotonic() + LOCK_TIMEOUT_S
            while True:
                try:
                    if os.name == "nt":
                        import msvcrt
                        fh.seek(0)
                        msvcrt.locking(fh.fileno(), msvcrt.LK_NBLCK, 1)
                    else:
                        import fcntl
                        fcntl.flock(fh.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                    break
                except OSError:
                    if time.monotonic() > deadline:
                        raise TimeoutError(f"the trial ledger's lock {lock} is held by another process")
                    time.sleep(0.05)
            try:
                yield
            finally:
                try:
                    if os.name == "nt":
                        import msvcrt
                        fh.seek(0)
                        msvcrt.locking(fh.fileno(), msvcrt.LK_UNLCK, 1)
                    else:
                        import fcntl
                        fcntl.flock(fh.fileno(), fcntl.LOCK_UN)
                except OSError:
                    pass
        finally:
            fh.close()


def _clean(o):
    """JSON-ready: numpy scalars to Python, non-finite floats to None, dates to ISO strings, tuples to lists."""
    if isinstance(o, Mapping):
        return {str(k): _clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple, set, frozenset)):
        seq = sorted(o, key=str) if isinstance(o, (set, frozenset)) else o
        return [_clean(v) for v in seq]
    if hasattr(o, "item") and callable(getattr(o, "item")) and not isinstance(o, (str, bytes)):
        try:
            o = o.item()                                       # numpy scalar
        except (TypeError, ValueError):
            return str(o)
    if isinstance(o, float):
        return o if math.isfinite(o) else None
    if isinstance(o, (datetime, date)):
        return o.isoformat()
    if o is None or isinstance(o, (bool, int, str)):
        return o
    return str(o)


def _now_iso() -> str:
    from zoneinfo import ZoneInfo
    return datetime.now(ZoneInfo(ET)).isoformat(timespec="seconds")


def rows(path=None) -> List[dict]:
    """Every row of the ledger in file order (a blank or malformed line is skipped with a warning: a reader never
    fails on one bad write)."""
    p = ledger_path(path)
    if not p.exists():
        return []
    out = []
    with open(p, encoding="utf-8") as fh:
        for i, ln in enumerate(fh, 1):
            if not ln.strip():
                continue
            try:
                r = json.loads(ln)
            except ValueError:
                logger.warning(f"[trials] {p}: line {i} is not JSON — skipped")
                continue
            if not isinstance(r, dict) or not all(k in r for k in ("study", "model", "variant", "kind")):
                logger.warning(f"[trials] {p}: line {i} is not a ledger row — skipped")
                continue
            out.append(r)
    return out


def key_of(r: Mapping) -> Tuple[str, str, str]:
    """A trial's identity: (study, variant, kind)."""
    return str(r["study"]), str(r["variant"]), str(r["kind"])


def current(path=None) -> Dict[Tuple[str, str, str], dict]:
    """The latest row of every key (a revision replaces its key's earlier rows), in first-recorded order."""
    out: Dict[Tuple[str, str, str], dict] = {}
    for r in rows(path):
        out[key_of(r)] = r
    return out


def _payload(r: Mapping) -> str:
    """What makes a row's content: a revision is a row whose payload differs from its key's latest."""
    return json.dumps({k: r.get(k) for k in ("model", "sets", "metrics", "decision", "note")}, sort_keys=True)


def _append(row: dict, path: Path) -> dict:
    """Under the lock: append ``row`` unless its key's latest row holds the same payload. Returns {"status":
    "recorded" | "unchanged" | "revised", "row": the row now current}."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with _locked(path):
        prev = current(path).get(key_of(row))
        if prev is not None and _payload(prev) == _payload(row):
            return {"status": "unchanged", "row": prev}
        if prev is not None:
            row = dict(row, revision=int(prev.get("revision", 0)) + 1)
        line = json.dumps(row, sort_keys=True, allow_nan=False)
        with open(path, "a", encoding="utf-8", newline="\n") as fh:
            fh.write(line + "\n")
            fh.flush()
            os.fsync(fh.fileno())
    return {"status": "revised" if prev is not None else "recorded", "row": row}


def _text(name: str, v) -> str:
    s = "" if v is None else str(v).strip()
    if not s:
        raise ValueError(f"a trial needs a non-empty {name}")
    return s


def _sets(sets) -> List[str]:
    if isinstance(sets, str):
        sets = [x.strip() for x in sets.split(",") if x.strip()]
    out = sorted({str(s) for s in (sets or ())}, key=lambda s: (SETS.index(s) if s in SETS else len(SETS), s))
    bad = [s for s in out if s not in SETS]
    if bad:
        raise ValueError(f"unknown set(s) {bad}: a trial names the sets it was judged on, among {list(SETS)}")
    return out


# ── the API ──────────────────────────────────────────────────────────────────

def record(study: str, model: str, variant: str, sets, metrics: Optional[Mapping] = None,
           decision: Optional[str] = None, kind: str = "rule", *, note: Optional[str] = None,
           at: Optional[str] = None, path=None) -> dict:
    """Record one evaluated candidate: ``study`` (its PREREG, e.g. "PREREG104"), ``model`` (the arm / book it varies,
    or "combination"), ``variant`` (its name inside the study, e.g. "F3: no volatility exit"), ``sets`` (the sets its
    numbers were computed on, among `SETS`), ``metrics`` (a JSON-ready dict — the add rule's gains, return per day,
    ...), ``decision`` (e.g. "add", "reject", "more draws", "exploratory"; None while undecided), ``kind`` ("rule",
    "sizing", "mechanism", ... — "seed" is reserved). ``at``: the instant it was evaluated (ISO; default now, ET), for a
    study recorded after the fact.

    Idempotent per (study, variant, kind): the same payload again returns {"status": "unchanged"}; another payload
    appends a revision ({"status": "revised"}; the key still counts once). Returns {"status", "row"}."""
    if str(kind).strip() == SEED_KIND:
        raise ValueError("kind 'seed' is reserved for PREREG103's counts (seed_from_prereg103)")
    row = {"v": VERSION, "study": _text("study", study), "model": _text("model", model),
           "variant": _text("variant", variant), "kind": _text("kind", kind), "sets": _sets(sets),
           "metrics": _clean(dict(metrics or {})), "decision": None if decision is None else str(decision),
           "note": None if note is None else str(note), "at": at or _now_iso(), "source": "recorded"}
    return _append(row, ledger_path(path))


def seed_rows(path=None) -> List[dict]:
    """The current seed rows (PREREG103's per-model counts)."""
    return [r for r in current(path).values() if r["kind"] == SEED_KIND]


def count(model: str, path=None) -> dict:
    """How many trials ``model`` has had: the seed (PREREG103's low / high) plus the recorded trials. The
    combination's high count also takes every recorded trial of every model (PREREG103's convention); its low count
    only those recorded under "combination". Returns {"model", "low", "high", "seed": {"low", "high", "source"} or
    None, "recorded", "by_kind", "studies"} (low = high = 0 for a model the ledger has never seen)."""
    model = _text("model", model)
    cur = list(current(path).values())
    seed = [r for r in cur if r["kind"] == SEED_KIND and r["model"] == model]
    rec = [r for r in cur if r["kind"] != SEED_KIND]
    own = [r for r in rec if r["model"] == model]
    s_low = sum(int((r.get("metrics") or {}).get("low") or 0) for r in seed)
    s_high = sum(int((r.get("metrics") or {}).get("high") or 0) for r in seed)
    high_add = len(rec) if model == COMBINATION else len(own)
    out = {"model": model, "low": s_low + len(own), "high": s_high + high_add,
           "seed": ({"low": s_low, "high": s_high, "source": seed[0].get("source")} if seed else None),
           "recorded": len(own), "by_kind": dict(Counter(r["kind"] for r in own)),
           "studies": sorted({r["study"] for r in own})}
    if model == COMBINATION:
        out["recorded_all_models"] = len(rec)
    return out


def counts(path=None) -> Dict[str, dict]:
    """`count` of every model the ledger knows (seeded or recorded)."""
    models = sorted({r["model"] for r in current(path).values()})
    return {m: count(m, path) for m in models}


def set_uses(path=None) -> Dict[str, dict]:
    """How many recorded trials each set carried (each key's current row; seed rows name no set): per set the trials,
    the decisions by outcome, the models and studies, the first and last instant. Every set in `SETS` is listed."""
    out = {s: {"trials": 0, "decisions": {}, "models": {}, "studies": [], "first": None, "last": None} for s in SETS}
    studies: Dict[str, set] = {s: set() for s in SETS}
    for r in current(path).values():
        if r["kind"] == SEED_KIND:
            continue
        for s in r.get("sets") or []:
            if s not in out:
                continue
            u = out[s]
            u["trials"] += 1
            d = "undecided" if r.get("decision") is None else str(r["decision"])
            u["decisions"][d] = u["decisions"].get(d, 0) + 1
            u["models"][r["model"]] = u["models"].get(r["model"], 0) + 1
            studies[s].add(r["study"])
            t = _instant(r.get("at"))
            if t is not None:
                if u["first"] is None or t < _instant(u["first"]):
                    u["first"] = r["at"]
                if u["last"] is None or t > _instant(u["last"]):
                    u["last"] = r["at"]
    for s in SETS:
        out[s]["studies"] = sorted(studies[s])
    return out


def _instant(at) -> Optional[datetime]:
    """An ISO instant as an aware datetime (a naive one read as UTC), None when unreadable — ISO strings with
    different UTC offsets (either side of a DST change) do not order as text."""
    if not at:
        return None
    try:
        t = datetime.fromisoformat(str(at))
    except ValueError:
        return None
    if t.tzinfo is None:
        from datetime import timezone
        t = t.replace(tzinfo=timezone.utc)
    return t


def prereg103_json() -> Path:
    """PREREG103's output (the research repository, `settings.backtest_research_dir`)."""
    from config.settings import settings
    return Path(settings.backtest_research_dir) / "studies" / "prereg103" / "global103.json"


def seed_from_prereg103(source=None, path=None) -> dict:
    """Write PREREG103's per-model trial counts (``trials.counts`` of `global103.json`: low / high and how they were
    counted) as the ledger's seed — one row per model, kind "seed", ``source`` "seed:PREREG103", never a recorded
    trial. Idempotent (a changed count appends a revision). Returns {"recorded", "unchanged", "revised": [models],
    "source"}."""
    src = Path(source) if source is not None else prereg103_json()
    d = json.loads(Path(src).read_text(encoding="utf-8"))
    tr = (d.get("trials") or {})
    cnt = tr.get("counts") or {}
    if not cnt:
        raise ValueError(f"{src} holds no trials.counts section")
    caveats = (tr.get("source") or {}).get("caveats") or []
    built = d.get("built_at")
    out = {"recorded": [], "unchanged": [], "revised": [], "source": str(src)}
    p = ledger_path(path)
    for model, c in cnt.items():
        low, high = int(c["low"]), int(c["high"])
        if not (1 <= low <= high):
            raise ValueError(f"PREREG103's count for {model!r} is not 1 <= low <= high: {low} / {high}")
        row = {"v": VERSION, "study": SEED_STUDY, "model": str(model), "variant": f"seed:{model}", "kind": SEED_KIND,
               "sets": [], "metrics": {"low": low, "high": high, "how": c.get("how")}, "decision": None,
               "note": "PREREG103's reconstruction of the trials before the ledger (every PREREG file read in full); "
                       "nominal counts: " + "; ".join(caveats[:1]),
               "at": built or _now_iso(), "source": "seed:PREREG103"}
        res = _append(row, p)
        out[res["status"]].append(str(model))
    return out


# ── the command line ─────────────────────────────────────────────────────────

def main(argv: Optional[Sequence[str]] = None) -> int:
    import argparse
    ap = argparse.ArgumentParser(description="The trial ledger (PREREG106): every candidate a study evaluates.")
    ap.add_argument("--seed", action="store_true", help="write PREREG103's per-model counts as the seed")
    ap.add_argument("--seed-source", default=None, help="global103.json (default: the research repository's)")
    ap.add_argument("--status", action="store_true", help="every model's count and every set's uses")
    ap.add_argument("--count", default=None, help="one model's count")
    ap.add_argument("--ledger", default=None, help=f"the ledger file (default {LEDGER})")
    a = ap.parse_args(argv)
    if a.seed:
        print(json.dumps(seed_from_prereg103(a.seed_source, a.ledger), indent=1))
    if a.count:
        print(json.dumps(count(a.count, a.ledger), indent=1))
    if a.status or not (a.seed or a.count):
        print(json.dumps({"ledger": str(ledger_path(a.ledger)), "counts": counts(a.ledger), "set_uses": set_uses(a.ledger)},
                         indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
