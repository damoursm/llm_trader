"""External auto-restart supervisor for the scheduled runner.

On 2026-06-17 the scheduler process vanished with NO Python traceback — i.e. the
process itself died (a native crash in a C extension such as kaleido/Chromium, an
OS/user kill, or a hang that was killed). An in-process ``try/except`` like the
dashboard's cannot catch that: there is no exception to catch and no live thread
left to recover. So this supervisor runs the scheduler as a CHILD process and
relaunches it whenever it exits, so a silent death self-heals instead of leaving
paper trading stopped until someone notices.

    python main.py --supervise        # production: supervised scheduler

The child is the unchanged ``python main.py --schedule`` process. Stopping this
supervisor (Ctrl-C / SIGTERM) stops the child too. A child that exits faster than
``_MIN_HEALTHY_UPTIME_S`` is treated as a crash loop and backed off exponentially
so a hard-failing scheduler doesn't spin the CPU relaunching forever.
"""
from __future__ import annotations

import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Optional

from loguru import logger

_ROOT = Path(__file__).resolve().parents[2]      # project root (where main.py lives)

_MIN_HEALTHY_UPTIME_S = 60        # a child gone sooner than this = crash loop → back off
_BACKOFF_START_S = 5
_BACKOFF_MAX_S = 300


def run_supervised() -> None:
    """Run ``main.py --schedule`` as a child, relaunching it on any exit."""
    cmd = [sys.executable, "main.py", "--schedule"]
    backoff = _BACKOFF_START_S
    child: Optional[subprocess.Popen] = None

    def _terminate_child() -> None:
        if child is not None and child.poll() is None:
            logger.info("[supervisor] stopping scheduler child …")
            child.terminate()
            try:
                child.wait(timeout=15)
            except subprocess.TimeoutExpired:
                child.kill()

    # Forward SIGTERM so an orderly kill of the supervisor stops the child too.
    # (A hard kill / Stop-Process -Force can't be intercepted — operators must
    # then also stop the orphaned `--schedule` child; see CLAUDE.md.)
    try:
        signal.signal(signal.SIGTERM, lambda *_: (_terminate_child(), sys.exit(0)))
    except (ValueError, OSError):       # not the main thread / unsupported platform
        pass

    logger.info("[supervisor] starting supervised scheduler (Ctrl-C to stop).")
    restarts = 0
    try:
        while True:
            started = time.time()
            try:
                logger.info(f"[supervisor] launching: {' '.join(cmd)}")
                child = subprocess.Popen(cmd, cwd=str(_ROOT))
                code = child.wait()                     # blocks until the scheduler exits
            except KeyboardInterrupt:
                raise                                   # clean shutdown via the outer handler
            except Exception as exc:
                # A spawn/wait failure (OSError, transient OS error, etc.) must NOT
                # propagate — that would kill the supervisor itself, the silent death
                # it exists to prevent (observed: the scheduler was dead with no
                # supervisor alive to relaunch it). Log, back off, and retry.
                logger.exception(
                    f"[supervisor] launch/wait failed ({exc}) — backing off {backoff}s and retrying."
                )
                time.sleep(backoff)
                backoff = min(backoff * 2, _BACKOFF_MAX_S)
                continue
            uptime = time.time() - started
            restarts += 1
            logger.critical(
                f"[supervisor] scheduler EXITED (code={code}) after {uptime:.0f}s "
                f"— restart #{restarts}. The runner does not exit on its own, so this "
                "is a crash or an external kill; relaunching."
            )
            if uptime >= _MIN_HEALTHY_UPTIME_S:
                backoff = _BACKOFF_START_S              # it ran healthily → reset backoff
            else:
                logger.warning(
                    f"[supervisor] scheduler died within {uptime:.0f}s — possible crash "
                    f"loop; backing off {backoff}s."
                )
            logger.info(f"[supervisor] relaunching in {backoff}s …")
            time.sleep(backoff)
            if uptime < _MIN_HEALTHY_UPTIME_S:
                backoff = min(backoff * 2, _BACKOFF_MAX_S)
    except KeyboardInterrupt:
        logger.info("[supervisor] interrupted — shutting down scheduler.")
    finally:
        _terminate_child()


if __name__ == "__main__":
    run_supervised()
