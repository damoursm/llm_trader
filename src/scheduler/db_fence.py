"""The database turn between the scheduler's ticks and the EOD process (2026-10-07).

DuckDB lets ONE process open the database read-write, and the scheduler opens every
connection read-write (reads included), so the EOD chain — moved out of the scheduler
into its own process (`src.scheduler.eod`) so it never competes with a tick for the GIL —
must not hold a connection while a tick needs the database: the tick's connect retries
for ~11 s, then fails. Two flag files under ``DIR``:

* ``tick_fence.json`` — the SCHEDULER's: raised `eod_tick_fence_lead_seconds` before each
  slot and held for the whole tick. While it stands and its process lives, the EOD
  process opens no connection.
* ``db_busy.json`` — the EOD process's: present while it holds a connection. A tick that
  starts waits for it to clear (at most `eod_db_wait_seconds`), so an EOD query already
  running finishes before the tick's first read instead of failing it.

Each side writes its own flag and THEN reads the other's, so the two can never both go
ahead. A flag whose process is gone is ignored: a watchdog ``os._exit`` leaves its fence
behind, a killed EOD process its busy flag. `install()` — the EOD process only — routes
every `connection.connect` of that process through `db_turn`.
"""
from __future__ import annotations

import json
import os
import threading
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Optional

from loguru import logger

DIR = Path("cache/eod")
FENCE = "tick_fence.json"
BUSY = "db_busy.json"
MAX_AGE_SECONDS = 4 * 3600.0       # a flag this old is stale whatever its pid says (pid reuse)
POLL_SECONDS = 1.0


def pid_alive(pid) -> bool:
    """True while process ``pid`` runs. Never ``os.kill(pid, 0)`` on Windows: there it
    TERMINATES the process."""
    try:
        pid = int(pid)
    except (TypeError, ValueError):
        return False
    if pid <= 0:
        return False
    if os.name == "nt":
        import ctypes
        from ctypes import wintypes
        k32 = ctypes.WinDLL("kernel32", use_last_error=True)       # private function objects
        k32.OpenProcess.restype = wintypes.HANDLE
        k32.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
        k32.GetExitCodeProcess.argtypes = [wintypes.HANDLE, ctypes.POINTER(wintypes.DWORD)]
        k32.CloseHandle.argtypes = [wintypes.HANDLE]
        h = k32.OpenProcess(0x1000, False, pid)                     # PROCESS_QUERY_LIMITED_INFORMATION
        if not h:
            return False
        try:
            code = wintypes.DWORD()
            return bool(k32.GetExitCodeProcess(h, ctypes.byref(code))) and code.value == 259   # STILL_ACTIVE
        finally:
            k32.CloseHandle(h)
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _path(name: str) -> Path:
    return Path(DIR) / name


def _read(name: str) -> Optional[dict]:
    try:
        rec = json.loads(_path(name).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return rec if isinstance(rec, dict) else None


def retry_os(fn, attempts: int = 20) -> None:
    """Windows refuses to replace or delete a file another process has open for a
    moment (the other side reading the flag): retry briefly."""
    for i in range(attempts):
        try:
            fn()
            return
        except FileNotFoundError:
            return
        except PermissionError:
            if i == attempts - 1:
                raise
            time.sleep(0.05)


def _write(name: str, rec: dict) -> None:
    p = _path(name)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_name(f"{p.name}.{os.getpid()}.{threading.get_ident()}.tmp")
    tmp.write_text(json.dumps(rec), encoding="utf-8")
    retry_os(lambda: os.replace(tmp, p))


def _remove(name: str, pid: Optional[int] = None) -> None:
    """Delete the flag — only when it is ``pid``'s, if given."""
    if pid is not None:
        rec = _read(name)
        if rec is None or rec.get("pid") != pid:
            return
    retry_os(lambda: _path(name).unlink())


def _held(name: str) -> Optional[dict]:
    """The flag's record while its process lives and it is not stale, else None."""
    rec = _read(name)
    if not rec or time.time() - float(rec.get("at", 0) or 0) > MAX_AGE_SECONDS:
        return None
    return rec if pid_alive(rec.get("pid")) else None


# ── the scheduler's side ─────────────────────────────────────────────────────

def raise_fence(reason: str = "tick") -> None:
    """Hold the database for this process's tick (re-stamped on every call)."""
    _write(FENCE, {"pid": os.getpid(), "at": time.time(), "reason": reason})


def drop_fence() -> None:
    _remove(FENCE, pid=os.getpid())


def fence_is_ours() -> bool:
    rec = _read(FENCE)
    return rec is not None and rec.get("pid") == os.getpid()


def wait_db_free(timeout: float) -> float:
    """Block while the EOD process holds a connection, at most ``timeout`` seconds.
    Returns the seconds waited."""
    t0 = time.time()
    rec = _held(BUSY)
    while rec is not None:
        if time.time() - t0 >= float(timeout):
            logger.warning(f"[db-fence] the EOD process (pid {rec.get('pid')}) still holds the database "
                           f"after {timeout:.0f}s — the tick goes ahead")
            break
        time.sleep(0.25)
        rec = _held(BUSY)
    return time.time() - t0


# ── the EOD process's side ───────────────────────────────────────────────────

def fence_up() -> bool:
    return _held(FENCE) is not None


_cv = threading.Condition()
_holders = 0                        # connections this process holds under the turn
_local = threading.local()          # .depth: nesting in the calling thread


def _acquire() -> None:
    """Wait out a standing fence, then announce ``db_busy`` and re-check the fence
    (the scheduler may have raised it meanwhile: then step back and wait again)."""
    t0, said = time.time(), False
    while True:
        if fence_up():
            if not said:
                logger.info("[db-fence] a tick holds the database — waiting")
                said = True
            time.sleep(POLL_SECONDS)
            continue
        _write(BUSY, {"pid": os.getpid(), "at": time.time()})
        if fence_up():
            _remove(BUSY, pid=os.getpid())
            time.sleep(POLL_SECONDS)
            continue
        if said:
            logger.info(f"[db-fence] database free again after {time.time() - t0:.0f}s")
        return


@contextmanager
def db_turn():
    """Hold the database turn for one connection. Re-entrant in a thread (a nested
    connection never waits on its own outer one); a NEW connection from another thread
    joins the holders unless a tick is waiting, in which case it waits for them to drain
    and then for the tick."""
    global _holders
    depth = getattr(_local, "depth", 0)
    if depth:
        _local.depth = depth + 1
        try:
            yield
        finally:
            _local.depth -= 1
        return
    with _cv:
        while True:
            if _holders == 0:
                _acquire()
                break
            if not fence_up():
                break
            _cv.wait(timeout=POLL_SECONDS)
        _holders += 1
    _local.depth = 1
    try:
        yield
    finally:
        _local.depth = 0
        with _cv:
            _holders -= 1
            if _holders == 0:
                _remove(BUSY, pid=os.getpid())
                _cv.notify_all()


def install(lock_retries: int = 30) -> None:
    """Route every DuckDB connection of THIS process through `db_turn` and give its
    open a ~2-minute lock retry (the dashboard's read handles). The EOD process only."""
    from src.db import connection
    connection.set_gate(db_turn, lock_retries=lock_retries)
