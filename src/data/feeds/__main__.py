"""The data feeds' runner (`src.data.feeds`).

    python -m src.data.feeds --run [--budget-seconds N] [--only wsb,occ,...]   the due sources, under the feeds lock
    python -m src.data.feeds --status                                          each source's state and the health notes
    python -m src.data.feeds --import-wsb DIR | --import-occ DIR | --import-fda ZIP EXPORT_DATE   seed a store
    python -m src.data.feeds --import-efts QUERY DIR | --import-plans DIR | --import-cboe-history
    python -m src.data.feeds --backfill-cboe START | --backfill-gdelt START END     history (GDELT's: the user's go-ahead)

A source that raises is recorded (``last_error``) and the others still run; the state is saved after each source. A
source module's ``skip_reason(state, now)`` (a cadence, the SEC's quiet windows) skips it for the run.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import date
from pathlib import Path

from config import settings
from src.data import feeds as FD

SOURCES = ("wsb", "occ", "fda", "efts", "cboe", "plans", "gdelt")


def _module(src: str):
    from src.data.feeds import cboe, efts, fda, gdelt, occ, plans, wsb
    return {"wsb": wsb, "occ": occ, "fda": fda, "efts": efts, "cboe": cboe, "plans": plans, "gdelt": gdelt}[src]


def _enabled(src: str) -> bool:
    return bool({"wsb": settings.enable_feeds_wsb, "occ": settings.enable_feeds_occ,
                 "fda": settings.enable_feeds_fda, "efts": settings.enable_feeds_efts,
                 "cboe": settings.enable_feeds_cboe, "plans": settings.enable_feeds_plans,
                 "gdelt": settings.enable_feeds_gdelt}[src])


def run(only=None, budget_seconds: float = None, now=None) -> dict:
    budget = float(budget_seconds if budget_seconds is not None else settings.feeds_budget_seconds)
    out = {"ran": [], "skipped": [], "failed": []}
    with FD.run_lock() as got:
        if not got:
            out["skipped"].append("locked")
            return out
        st = FD.load_state()
        deadline = time.time() + budget
        t_now = now or FD.now_utc()
        for src in SOURCES:
            if only and src not in only:
                continue
            if not _enabled(src):
                out["skipped"].append(f"{src}: off")
                continue
            s = st.setdefault(src, {})
            if src == "fda":
                a = FD.age_hours(s.get("last_try"), t_now)
                if a is not None and a < float(settings.feeds_fda_check_hours):
                    out["skipped"].append("fda: checked recently")
                    continue
            skip = getattr(_module(src), "skip_reason", None)
            reason = skip(s, t_now) if skip else None
            if reason:
                out["skipped"].append(f"{src}: {reason}")
                continue
            s["last_try"] = t_now.isoformat()
            try:
                msg = _module(src).run(s, now=t_now, deadline=deadline)
                s.update(last_ok=t_now.isoformat(), last_error=None, last_msg=msg)
                out["ran"].append(f"{src} {msg}")
            except Exception as e:  # noqa: BLE001 — recorded; the next source still runs
                s["last_error"] = f"{type(e).__name__}: {str(e)[:160]}"
                out["failed"].append(f"{src} {s['last_error']}")
            FD.save_state(st)
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python -m src.data.feeds")
    ap.add_argument("--run", action="store_true")
    ap.add_argument("--status", action="store_true")
    ap.add_argument("--only", default="")
    ap.add_argument("--budget-seconds", type=float, default=None)
    ap.add_argument("--import-wsb", default=None)
    ap.add_argument("--import-occ", default=None)
    ap.add_argument("--import-fda", nargs=2, default=None, metavar=("ZIP", "EXPORT_DATE"))
    ap.add_argument("--import-efts", nargs=2, default=None, metavar=("QUERY", "DIR"))
    ap.add_argument("--import-plans", default=None, metavar="DIR")
    ap.add_argument("--import-cboe-history", action="store_true")
    ap.add_argument("--backfill-cboe", default=None, metavar="START")
    ap.add_argument("--backfill-gdelt", nargs=2, default=None, metavar=("START", "END"))
    a = ap.parse_args(argv)
    if a.import_efts or a.import_plans or a.import_cboe_history or a.backfill_cboe or a.backfill_gdelt:
        from src.data.feeds import cboe, efts, gdelt, plans
        with FD.run_lock() as got:          # the timed runs write the same stores: never both at once
            if not got:
                print("feeds: a run holds the feeds lock; try again after it")
                return 1
            if a.import_efts:
                n = efts.import_dir(a.import_efts[0], Path(a.import_efts[1]))
                print(f"feeds: efts {a.import_efts[0]} imported {n} hit(s)")
            if a.import_plans:
                print(f"feeds: plans imported {plans.import_dir(Path(a.import_plans))} file(s)")
            if a.import_cboe_history:
                print(f"feeds: cboe history {cboe.import_history()} row(s)")
            if a.backfill_cboe:
                print(f"feeds: cboe backfill {cboe.backfill(date.fromisoformat(a.backfill_cboe))}")
            if a.backfill_gdelt:
                s, e = (date.fromisoformat(x) for x in a.backfill_gdelt)
                print(f"feeds: gdelt backfill {gdelt.backfill(s, e)}")
        return 0
    if a.import_wsb:
        from src.data.feeds import wsb
        print(f"feeds: wsb imported {wsb.import_dir(Path(a.import_wsb))} day(s)")
        return 0
    if a.import_occ:
        from src.data.feeds import occ
        print(f"feeds: occ imported {occ.import_dir(Path(a.import_occ))} session(s)")
        return 0
    if a.import_fda:
        from src.data.feeds import fda
        n = fda.import_zip(Path(a.import_fda[0]), a.import_fda[1])
        st = FD.load_state()
        st.setdefault("fda", {})["export_date"] = a.import_fda[1]
        FD.save_state(st)
        print(f"feeds: fda imported export {a.import_fda[1]}: {n} approvals")
        return 0
    if a.status:
        print(json.dumps({"state": FD.load_state(), "health": FD.health()}, indent=1, default=str))
        return 0
    if a.run:
        if not bool(settings.enable_data_feeds):
            print("feeds: off (enable_data_feeds)")
            return 0
        only = [x.strip() for x in a.only.split(",") if x.strip()] or None
        r = run(only=only, budget_seconds=a.budget_seconds)
        print("feeds: " + " | ".join(r["ran"] + [f"FAILED {x}" for x in r["failed"]] + r["skipped"]))
        return 0
    ap.print_help()
    return 2


if __name__ == "__main__":
    sys.exit(main())
