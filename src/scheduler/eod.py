"""End-of-day maintenance — its OWN PROCESS (2026-10-07).

User 2026-10-07 ("yes please" to moving the chain out of the scheduler). It ran as a
thread inside the scheduler, where its replay (~400k ticker-days, 5-13 h) competed with
every evening and overnight tick for the GIL: a tick that also refreshed its 6-hour
calibrations crossed the 45-minute watchdog (kills 10-01 x3, 10-06 06:45 and 22:15), and
each relaunch restarted the whole chain from scratch. Its walk-forward and shape-history
steps also flushed the scheduler's calibration caches at every as-of step. Here the chain:

* runs below normal priority and survives a scheduler kill (a file log and an OS lock,
  never a pipe to the scheduler);
* records every step in ``cache/eod/state/<day>.json``: a relaunch resumes at the first
  step without an outcome — an interrupted step runs again, a failed one does not;
* opens the database only between ticks (`db_fence`).

    python -m src.scheduler.eod --day 2026-10-06              # what the scheduler launches
    python -m src.scheduler.eod --day 2026-10-06 --status     # the day's record
    python -m src.scheduler.eod --day 2026-10-06 --steps replay,backtest    # rerun these

The steps, in order, each fail-soft and under its own setting: the IBKR Flex borrow fees
(in their own thread: IBKR can take hours to generate a statement), the forward-return
cache warm, retention, the replay (``signals_replay``), the walk-forward
(``weight_history``), the shape history, the tier-2 backtest tail (``signals_backtest``)
and the IBKR BID_ASK spread sweep (a subprocess: ib_async needs its own event loop and
clientId). Neither the model retrains (weekly, `runner._run_weekly_ml_work`) nor the
automatic refactor (its own nightly slot) belong here.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import threading
import time
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Iterable, Optional, Tuple

from loguru import logger

from config import settings
from src.scheduler import db_fence

DIR = Path("cache/eod")
LOCK_NAME = "eod.lock"
CURRENT = "current.json"
CONSOLE = Path("logs/eod_console.log")
LOG_DIR = Path("logs")
LOG_KEEP_DAYS = 14
FLEX = "flex"
STEPS = ("cache_warm", "retention", "replay", "walkforward", "shape_history", "backtest", "spread_sweep")
FINAL = ("ok", "failed", "off")
FLEX_JOIN_SECONDS = 2400.0   # the Flex fetch gives IBKR ~30 min, then keeps the reference for the next run

_STATE_LOCK = threading.Lock()


def _day(day) -> date:
    return day if isinstance(day, date) else date.fromisoformat(str(day)[:10])


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


# ── the day's record ─────────────────────────────────────────────────────────

def state_path(day) -> Path:
    return Path(DIR) / "state" / f"{_day(day).isoformat()}.json"


def load_state(day) -> dict:
    try:
        st = json.loads(state_path(day).read_text(encoding="utf-8"))
        if isinstance(st, dict):
            st.setdefault("steps", {})
            st.setdefault("runs", [])
            st.setdefault("finished", None)
            return st
    except (OSError, ValueError):
        pass
    return {"day": _day(day).isoformat(), "steps": {}, "runs": [], "finished": None}


def _save(st: dict) -> None:
    p = state_path(st["day"])
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_name(f"{p.name}.{os.getpid()}.tmp")
    tmp.write_text(json.dumps(st, indent=1, default=str), encoding="utf-8")
    db_fence.retry_os(lambda: os.replace(tmp, p))


def record(day, step: str, status: str, summary: str = "", **extra) -> dict:
    """Write one step's outcome. The chain is FINISHED once every step has a final one."""
    with _STATE_LOCK:
        st = load_state(day)
        prev = st["steps"].get(step) or {}
        rec = {"started": prev.get("started")} if status in FINAL else {}
        rec.update(status=status, summary=summary, **extra)
        st["steps"][step] = rec
        if st.get("finished") is None and all(
                (st["steps"].get(s) or {}).get("status") in FINAL for s in (FLEX,) + STEPS):
            st["finished"] = _now_iso()
        _save(st)
    return rec


def summary_line(st: dict) -> str:
    return ", ".join(f"{s}={(st['steps'].get(s) or {}).get('status', '-')}" for s in (FLEX,) + STEPS)


# ── single instance ──────────────────────────────────────────────────────────

def _try_lock(path: Path):
    """An OS-level EXCLUSIVE lock, dropped by the OS when the process dies — a killed
    run never leaves a stale lock. The open handle, or None when someone holds it."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fh = open(path, "a+b")
    try:
        if os.name == "nt":
            import msvcrt
            fh.seek(0)
            msvcrt.locking(fh.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            import fcntl
            fcntl.flock(fh.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        fh.close()
        return None
    return fh


def _unlock(fh) -> None:
    try:
        if os.name == "nt":
            import msvcrt
            fh.seek(0)
            msvcrt.locking(fh.fileno(), msvcrt.LK_UNLCK, 1)
    except OSError:
        pass
    finally:
        fh.close()


def running() -> bool:
    """True while an EOD process holds the lock."""
    fh = _try_lock(Path(DIR) / LOCK_NAME)
    if fh is None:
        return True
    _unlock(fh)
    return False


def current() -> Optional[dict]:
    """The running EOD process's ``{pid, day, started}``, or None."""
    try:
        rec = json.loads((Path(DIR) / CURRENT).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return rec if isinstance(rec, dict) and db_fence.pid_alive(rec.get("pid")) else None


# ── the steps ────────────────────────────────────────────────────────────────

def _flex(day) -> Tuple[object, str]:
    from src.broker.flex import fetch_borrow_fees
    res = fetch_borrow_fees()
    return res, str(res)


def _cache_warm(day) -> Tuple[object, str]:
    from src.data.cache_warm import warm_forward_return_cache
    n = warm_forward_return_cache(days=int(settings.eod_cache_warm_days),
                                  max_tickers=(settings.eod_cache_warm_max_tickers or None))
    return n, f"{n:,} panel ticker(s) warmed"


def _retention(day) -> Tuple[object, str]:
    from src.db.retention import run_retention
    res = run_retention()
    return res, str(res)


def _replay(day) -> Tuple[object, str]:
    # Rescore history through the CURRENT scorers so the calibrations fit values today's
    # code produced; after the cache warm, which supplies the bars a replay reads. A
    # stale replay table only means the epoch mask blanks more, never a wrong value.
    from src.analysis.replay import materialize
    n = materialize(days=(int(settings.eod_replay_refresh_days) or None))
    return n, f"{n:,} ticker-days rescored"


def _walkforward(day) -> Tuple[object, str]:
    # The day's point-in-time calibration into `weight_history`; only the tail is
    # rewalked (each step ~60 s), after the replay that feeds the panel it reads.
    from src.analysis.walkforward import materialize as wf_materialize
    start = (_day(day) - timedelta(days=max(1, int(settings.walkforward_eod_days)))).isoformat()
    n = wf_materialize(start=start, step_days=1)
    return n, f"{n} calibration step(s) stored"


def _shape_history(day) -> Tuple[object, str]:
    # The day's as-of rank-shaping curve (`weight_history`'s sibling), before the
    # backtest tail that consumes it. Pinned to the MARKET day: a chain that runs past
    # midnight must not write tomorrow's row early (the step skips a date it holds).
    from src.signals.rank_shaping import materialize_shape_history
    d = _day(day).isoformat()
    n = materialize_shape_history(start=d, end=d, step_days=1)
    return n, f"{n} row(s) appended"


def _backtest(day) -> Tuple[object, str]:
    # Keep `signals_backtest` — what the CURRENT entry architecture would have decided —
    # current over the span the replay refreshed. A firewalled analysis table.
    from src.analysis.backtest import materialize as bt_materialize
    n = bt_materialize(days=(int(settings.eod_replay_refresh_days) or None))
    return n, f"{n:,} ticker-days rescored under the current entry architecture"


def _spread_sweep(day) -> Tuple[object, str]:
    # Each Gate-4 name's time-averaged quoted half-spread into cache/ibkr_spread.json —
    # the liquidity forecast's structural layer. Own clientId (ibkr_client_id+50),
    # budget-capped; its rotation makes a cut-short run resume the next night.
    import subprocess as sp
    res = sp.run([sys.executable, "-m", "src.performance.spread_sweep"],
                 capture_output=True, text=True, timeout=float(settings.spread_sweep_budget_seconds) + 300,
                 encoding="utf-8", errors="replace")
    tail = " ".join((res.stdout or "").strip().splitlines()[-8:])
    if res.returncode != 0:
        raise RuntimeError(f"rc={res.returncode}: {(res.stderr or '').strip()[-400:]}")
    return {"rc": res.returncode}, tail or "rc=0"


STEP_FNS = {FLEX: _flex, "cache_warm": _cache_warm, "retention": _retention, "replay": _replay,
            "walkforward": _walkforward, "shape_history": _shape_history, "backtest": _backtest,
            "spread_sweep": _spread_sweep}


def _enabled(step: str) -> bool:
    return bool({
        "replay": settings.enable_eod_replay_refresh,
        "walkforward": settings.enable_eod_walkforward,
        "shape_history": settings.enable_eod_shape_history,
        "backtest": settings.enable_eod_backtest_refresh,
        "spread_sweep": (settings.enable_eod_spread_sweep
                         and str(settings.broker_mode or "off").startswith("ibkr")),
    }.get(step, True))


def _jsonable(x):
    try:
        return json.loads(json.dumps(x, default=str))
    except (TypeError, ValueError):
        return str(x)


def _run_step(day, step: str, force: bool = False) -> None:
    if not force and (load_state(day)["steps"].get(step) or {}).get("status") in FINAL:
        return
    if not _enabled(step):
        record(day, step, "off", "disabled")
        return
    record(day, step, "running", started=_now_iso(), pid=os.getpid())
    t0 = time.monotonic()
    try:
        result, summary = STEP_FNS[step](day)
    except Exception as exc:                              # noqa: BLE001 — every step fail-soft
        secs = time.monotonic() - t0
        logger.warning(f"[eod] {step} failed after {secs:.0f}s: {exc}")
        record(day, step, "failed", str(exc)[:600], seconds=round(secs, 1), finished=_now_iso())
        return
    secs = time.monotonic() - t0
    logger.info(f"[eod] {step}: {summary} ({secs:.0f}s)")
    record(day, step, "ok", summary, seconds=round(secs, 1), finished=_now_iso(), result=_jsonable(result))


def run(day, only: Optional[Iterable[str]] = None) -> dict:
    """Run the day's chain: every step without an outcome (or, with ``only``, those
    steps whatever their outcome). Returns the day's record."""
    day = _day(day)
    names = list(only) if only else None
    force = names is not None
    with _STATE_LOCK:
        st = load_state(day)
        st["runs"].append({"pid": os.getpid(), "started": _now_iso(), "ended": None, "only": names})
        _save(st)
    cur = Path(DIR) / CURRENT
    cur.parent.mkdir(parents=True, exist_ok=True)
    cur.write_text(json.dumps({"pid": os.getpid(), "day": day.isoformat(), "started": _now_iso()}),
                   encoding="utf-8")
    try:
        flex = None
        if names is None or FLEX in names:
            flex = threading.Thread(target=_run_step, args=(day, FLEX, force), name="eod-flex", daemon=True)
            flex.start()
        for step in STEPS:
            if names is None or step in names:
                _run_step(day, step, force)
        if flex is not None:
            flex.join(timeout=FLEX_JOIN_SECONDS)
            if flex.is_alive():
                logger.warning("[eod] the Flex fetch is still waiting for IBKR — ending without it")
        with _STATE_LOCK:
            st = load_state(day)
            st["runs"][-1]["ended"] = _now_iso()
            _save(st)
    finally:
        db_fence.retry_os(lambda: cur.unlink())
    return load_state(day)


# ── the process ──────────────────────────────────────────────────────────────

def _setup_logging(day: date) -> None:
    from src.log_redaction import redaction_filter
    logger.remove()
    logger.add(sys.stderr, format="{time:YYYY-MM-DD HH:mm:ss} | {level:<8} | {message}", level="INFO",
               filter=redaction_filter)
    logger.add(str(Path(LOG_DIR) / f"eod_{day.isoformat()}.log"), level="INFO", filter=redaction_filter)
    cutoff = time.time() - LOG_KEEP_DAYS * 86400
    for p in Path(LOG_DIR).glob("eod_20*.log"):
        try:
            if p.stat().st_mtime < cutoff:
                p.unlink()
        except OSError:
            pass


def _lower_priority() -> None:
    """Below normal, also when run by hand: the ticks come first."""
    try:
        if os.name == "nt":
            import ctypes
            from ctypes import wintypes
            k32 = ctypes.WinDLL("kernel32", use_last_error=True)
            k32.GetCurrentProcess.restype = wintypes.HANDLE
            k32.SetPriorityClass.argtypes = [wintypes.HANDLE, wintypes.DWORD]
            k32.SetPriorityClass(k32.GetCurrentProcess(), 0x00004000)     # BELOW_NORMAL_PRIORITY_CLASS
        else:
            os.nice(5)
    except Exception:                                     # noqa: BLE001
        pass


def _deadline(seconds: float) -> None:
    if seconds <= 0:
        return

    def _expire() -> None:
        try:
            logger.critical(f"[eod] still running after {seconds:.0f}s — exiting; the step in progress "
                            "runs again at the next launch")
        finally:
            os._exit(2)

    t = threading.Timer(seconds, _expire)
    t.daemon = True
    t.start()


def main(argv=None) -> int:
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError, OSError):
        pass
    ap = argparse.ArgumentParser(description="End-of-day maintenance, as its own process")
    ap.add_argument("--day", required=True, help="the market day, YYYY-MM-DD")
    ap.add_argument("--steps", default="",
                    help="comma list to (re)run whatever their recorded outcome: " + ", ".join((FLEX,) + STEPS))
    ap.add_argument("--status", action="store_true", help="print the day's record and exit")
    args = ap.parse_args(argv)
    day = _day(args.day)
    if args.status:
        print(json.dumps(load_state(day), indent=1, default=str))
        return 0
    only = [s.strip() for s in args.steps.split(",") if s.strip()] or None
    bad = [s for s in (only or []) if s not in STEP_FNS]
    if bad:
        ap.error(f"unknown step(s): {', '.join(bad)}")
    _setup_logging(day)
    lock = _try_lock(Path(DIR) / LOCK_NAME)
    if lock is None:
        logger.warning("[eod] another EOD process holds the lock — not starting a second one")
        return 3
    try:
        _lower_priority()
        db_fence.install()
        _deadline(float(settings.eod_timeout_seconds))
        logger.info(f"[eod] maintenance for {day} STARTING (pid {os.getpid()}"
                    + (f", steps {', '.join(only)}" if only else "") + ")")
        st = run(day, only=only)
        logger.info(f"[eod] maintenance for {day}: {summary_line(st)}"
                    + (" — FINISHED" if st.get("finished") else ""))
        return 0
    finally:
        _unlock(lock)


if __name__ == "__main__":
    sys.exit(main())
