"""Automatic database refactor — keep stored history consistent with the code.

When an implementation changes, every value the old code produced stops
describing the current system. Until now the response was manual: notice the
change, hand-edit `METHOD_SCORER_EPOCH`, remember to re-run the replay, remember
that the weights were fitted on the old values. Each step correct only while
someone remembered.

This is that process, mechanised. `run_refactor()` detects what changed
(`code_version`) and repairs the database in **dependency order**, which is the
part that cannot be got wrong:

    1. DATA     — regenerate replayable method scores + market conditions
                  (`replay`). Everything downstream reads these.
    2. EPOCHS   — for what cannot be regenerated, derive the mask instant from
                  the fingerprint's `first_seen_at` (no hand-edited registry).
    3. WEIGHTS  — re-walk the point-in-time calibration (`walkforward`). Weights
                  are fitted ON the data, so they are invalid until step 1 has
                  run; doing this first would calibrate on stale scores.
    4. DERIVED  — recompute combined_score/confidence (`backtest`), which
                  depends on both the data and the weights.

Reversing any pair silently produces a database that looks refreshed and is
internally inconsistent — weights fitted on scores that no longer exist, or a
backtest run under weights that predate the data it scores.

**Replayable vs not is the whole asymmetry.** A replayable method is
REGENERATED, so a false positive from the change detector costs only CPU. A
non-replayable one can only be MASKED, which withholds real history — so those
are recorded and reported, and masking is applied only when
`refactor_auto_epoch` is on (default OFF). A cosmetic edit to a sentiment module
should not silently blank 78% of the panel.

CLI:  python -m src.analysis.refactor --check     # what changed, no writes
      python -m src.analysis.refactor --apply
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Dict, List, Optional

from loguru import logger

from config.settings import settings


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def plan(changes: Optional[Dict[str, dict]] = None) -> Dict[str, object]:
    """What a refactor would do, without doing it.

    Splits the changed set by whether it can be regenerated, because the two
    halves have completely different costs and remedies.
    """
    from src.analysis.code_version import detect_changes, unmapped_methods
    from src.db.schema import REPLAYABLE_METHOD_COLUMNS

    ch = detect_changes() if changes is None else changes
    moved = {n: d for n, d in ch.items() if not d["is_new"]}
    first_seen = {n: d for n, d in ch.items() if d["is_new"]}

    replayable = sorted(n for n in moved if n in REPLAYABLE_METHOD_COLUMNS)
    derived = sorted(n for n in moved if n in ("confidence", "combined_score"))
    maskable = sorted(set(moved) - set(replayable) - set(derived))

    return {
        "changed": sorted(moved),
        "first_seen": sorted(first_seen),
        "regenerate": replayable,
        "mask_candidates": maskable,
        "derived_changed": derived,
        "unmapped": unmapped_methods(),
        # Weights are fitted on the data, so ANY data change invalidates them.
        # A derived-formula change does not touch the scores but does change
        # what a backtest means, so it re-runs step 4 only.
        "rewalk_weights": bool(replayable or maskable),
        "rerun_backtest": bool(moved),
    }


def run_refactor(apply: bool = False,
                 changes: Optional[Dict[str, dict]] = None) -> Dict[str, object]:
    """Detect implementation changes and repair the database.

    ``apply=False`` reports the plan and writes nothing. Each step is fail-soft
    and recorded: a failure in one must not leave the caller believing the whole
    refactor succeeded, so the audit row carries a per-step status and the
    overall ``ok`` is the AND of them.
    """
    started = _now()
    p = plan(changes)
    steps: List[dict] = []

    if p["unmapped"]:
        logger.warning(f"[refactor] {len(p['unmapped'])} method(s) have no source "
                       f"mapping and are invisible to change detection: {p['unmapped']}")

    if not p["changed"]:
        logger.info("[refactor] no implementation changes detected")
        if apply and p["first_seen"]:
            from src.analysis.code_version import record
            n = record(p["first_seen"])
            logger.info(f"[refactor] recorded {n} baseline fingerprint(s)")
            steps.append({"step": "baseline", "status": "ok", "detail": f"{n} recorded"})
            _audit(started, p, steps, True)
        return {"plan": p, "steps": steps, "ok": True}

    logger.info(f"[refactor] changed: {p['changed']}")
    if not apply:
        return {"plan": p, "steps": steps, "ok": True}

    ok = True

    # ── 1. DATA ───────────────────────────────────────────────────────────
    if p["regenerate"]:
        try:
            from src.analysis.replay import materialize as replay_materialize
            n = replay_materialize(days=None)
            steps.append({"step": "replay", "status": "ok",
                          "detail": f"{n} ticker-days regenerated for {p['regenerate']}"})
        except Exception as e:
            ok = False
            steps.append({"step": "replay", "status": "failed", "detail": str(e)[:200]})
            logger.warning(f"[refactor] replay failed: {e}")

    # ── 2. EPOCHS (masking — deliberately opt-in) ─────────────────────────
    if p["mask_candidates"]:
        if settings.refactor_auto_epoch:
            steps.append({"step": "auto_epoch", "status": "ok",
                          "detail": f"masking from code_versions.first_seen_at "
                                    f"for {p['mask_candidates']}"})
        else:
            steps.append({"step": "auto_epoch", "status": "skipped",
                          "detail": f"NOT masked (refactor_auto_epoch=false): "
                                    f"{p['mask_candidates']}"})
            logger.warning(
                f"[refactor] {p['mask_candidates']} changed but cannot be "
                f"regenerated. Their stored values are now stale. Masking them "
                f"would withhold real history, so it is opt-in: set "
                f"REFACTOR_AUTO_EPOCH=true, or confirm the change was cosmetic.")

    # ── 3. WEIGHTS (must follow the data) ─────────────────────────────────
    if p["rewalk_weights"]:
        try:
            from src.analysis.walkforward import materialize as wf_materialize
            n = wf_materialize(step_days=1)
            steps.append({"step": "walkforward", "status": "ok",
                          "detail": f"{n} calibration steps"})
        except Exception as e:
            ok = False
            steps.append({"step": "walkforward", "status": "failed", "detail": str(e)[:200]})
            logger.warning(f"[refactor] walk-forward failed: {e}")

    # ── 4. DERIVED (must follow data AND weights) ─────────────────────────
    if p["rerun_backtest"]:
        try:
            from src.analysis.backtest import materialize as bt_materialize
            n = bt_materialize(days=None)
            steps.append({"step": "backtest", "status": "ok", "detail": f"{n} rows"})
        except Exception as e:
            ok = False
            steps.append({"step": "backtest", "status": "failed", "detail": str(e)[:200]})
            logger.warning(f"[refactor] backtest failed: {e}")

    # Fingerprints are recorded LAST and only on full success. Recording them
    # after a partial failure would mark the database as consistent with code it
    # was never refactored against — the next run would see no changes and the
    # staleness would be permanent and invisible.
    if ok:
        from src.analysis.code_version import record
        n = record(list(p["changed"]) + list(p["first_seen"]))
        steps.append({"step": "record", "status": "ok", "detail": f"{n} fingerprints"})
    else:
        steps.append({"step": "record", "status": "skipped",
                      "detail": "a step failed — fingerprints NOT advanced so the "
                                "refactor is retried rather than silently forgotten"})

    _audit(started, p, steps, ok)
    return {"plan": p, "steps": steps, "ok": ok}


def _audit(started: str, p: dict, steps: List[dict], ok: bool) -> None:
    from src.db.connection import connect
    try:
        import pandas as pd
        df = pd.DataFrame([{
            "started_at": started, "finished_at": _now(),
            "trigger": json.dumps(p["changed"]),
            "steps": json.dumps(steps), "ok": bool(ok)}])
        with connect() as con:
            con.register("_rf_df", df)
            con.execute("INSERT INTO refactor_runs (started_at, finished_at, "
                        "trigger, steps, ok) SELECT started_at, finished_at, "
                        "trigger, steps, ok FROM _rf_df")
            con.unregister("_rf_df")
    except Exception as e:
        logger.warning(f"[refactor] audit row failed: {e}")


def auto_epochs() -> Dict[str, str]:
    """`{method: first_seen_at}` for the CURRENT fingerprint of each method.

    This is the automatic replacement for the hand-edited `METHOD_SCORER_EPOCH`:
    the instant a method's current implementation was first observed IS its
    epoch. Consulted by `method_epochs` only when `refactor_auto_epoch` is on,
    so the manual registry stays authoritative by default.
    """
    if not settings.refactor_auto_epoch:
        return {}
    from src.analysis.code_version import load_stored
    from src.db.schema import REPLAYABLE_METHOD_COLUMNS
    out = {}
    for name, rec in load_stored().items():
        # A regenerated method is NOT masked — that is the whole point of the
        # replay: its history has been rewritten by the current scorer.
        if name in REPLAYABLE_METHOD_COLUMNS:
            continue
        out[name] = rec["first_seen_at"]
    return out


if __name__ == "__main__":  # pragma: no cover
    import argparse

    ap = argparse.ArgumentParser(description="Automatic database refactor")
    ap.add_argument("--apply", action="store_true", help="perform the refactor")
    ap.add_argument("--check", action="store_true", help="report only (default)")
    a = ap.parse_args()

    res = run_refactor(apply=a.apply and not a.check)
    p = res["plan"]
    print(f"\nchanged            : {p['changed'] or '(none)'}")
    print(f"  regenerate       : {p['regenerate'] or '(none)'}")
    print(f"  mask candidates  : {p['mask_candidates'] or '(none)'}")
    print(f"  derived changed  : {p['derived_changed'] or '(none)'}")
    print(f"first seen         : {len(p['first_seen'])} name(s)")
    print(f"unmapped (blind)   : {p['unmapped'] or '(none)'}")
    if res["steps"]:
        print("\nsteps:")
        for s in res["steps"]:
            print(f"  {s['step']:<12}{s['status']:<9}{s['detail']}")
    if not a.apply:
        print("\n(dry run — pass --apply to perform it)")
