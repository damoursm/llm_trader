"""Last-resort IB Gateway recovery — kill a dead/wedged gateway and let the IBC
scheduled task relaunch + auto-login it.

The app-side self-healing (auto-reconnect on drop, wedge detection + forced
client recycle, request timeouts) can only fix the APP's side of the session.
When the GATEWAY itself is the corpse — alive-but-wedged (process up, port 4002
open, API backend dead: observed 2026-07-06 and 2026-07-13) or fully down —
every redial hits a dead socket, and IBC's watchdog can't help (it only checks
the PROCESS is alive). Until now the recovery was a documented manual ops
procedure (memory: ibkr-gateway-wedge-recovery): kill the gateway java process
owning the port, then trigger the 'IBC Gateway' scheduled task, which relaunches
and auto-logs-in within ~60s. This module automates exactly that procedure.

Safety posture:
  * PAPER-MODE ONLY — ``ibkr_live`` downgrades to a CRITICAL log advising a
    manual restart (a live gateway may need 2FA to re-login; bouncing it is a
    human decision — same stance as drift auto-flatten refusing live).
  * Cooldown-guarded (``broker_gateway_restart_cooldown_minutes``) so a
    persistently-broken gateway can't be kill-looped.
  * Every step is fail-soft: a recovery failure logs and returns False — it
    never breaks the pipeline run (the sim is unaffected regardless).
"""

import subprocess
import time
from typing import Optional

from loguru import logger

from config import settings

# Monotonic timestamp of the last restart attempt (module-level = process-lived,
# matching the singleton broker). Tests reset it via _reset_for_tests().
_last_restart_mono: float = 0.0

_SUBPROCESS_TIMEOUT = 20  # seconds per external command — generous, never hangs the tick


def _reset_for_tests() -> None:
    global _last_restart_mono
    _last_restart_mono = 0.0


def _pid_listening_on(port: int) -> Optional[int]:
    """PID of the process LISTENING on ``port`` (Windows ``netstat -ano``), or
    None. This is the gateway java process — IBC INLINE mode shares it."""
    try:
        out = subprocess.run(
            ["netstat", "-ano", "-p", "tcp"],
            capture_output=True, text=True, timeout=_SUBPROCESS_TIMEOUT,
        ).stdout or ""
        suffix = f":{port}"
        for line in out.splitlines():
            parts = line.split()
            # TCP  0.0.0.0:4002  0.0.0.0:0  LISTENING  1234
            if len(parts) >= 5 and parts[3] == "LISTENING" and parts[1].endswith(suffix):
                return int(parts[4])
    except Exception as e:
        logger.warning(f"[broker] gateway recovery: netstat probe failed ({e})")
    return None


def maybe_restart_gateway(reason: str, wait: bool = True) -> bool:
    """Kill the gateway owning ``ibkr_port`` and fire the IBC relaunch task.

    Returns True when a restart was triggered AND (``wait=True``) the fresh
    gateway's port is listening again — i.e. the caller should dial once more
    now. ``wait=False`` (the wedge path, called from broker touchpoints) fires
    and returns immediately; the next touchpoint's auto-reconnect picks the
    fresh gateway up. False = nothing was done (gated off / cooldown / failed).
    """
    if settings.broker_mode != "ibkr_paper":
        if settings.broker_mode == "ibkr_live":
            logger.critical(
                f"[broker] GATEWAY appears dead/wedged ({reason}) — auto-restart is "
                "REFUSED in ibkr_live (re-login may need 2FA). Restart IB Gateway "
                "manually / check IBC."
            )
        return False
    if not settings.broker_gateway_auto_restart:
        logger.debug(f"[broker] gateway auto-restart disabled — not acting on: {reason}")
        return False

    global _last_restart_mono
    now = time.monotonic()
    cooldown = 60.0 * max(1, int(settings.broker_gateway_restart_cooldown_minutes))
    if _last_restart_mono and now - _last_restart_mono < cooldown:
        logger.info(
            f"[broker] gateway restart wanted ({reason}) but cooldown active "
            f"({(now - _last_restart_mono):.0f}s since last) — skipping"
        )
        return False
    _last_restart_mono = now

    port = int(settings.ibkr_port)
    task = settings.broker_gateway_task_name
    pid = _pid_listening_on(port)
    logger.critical(
        f"[broker] GATEWAY RECOVERY — {reason}. Killing gateway on port {port} "
        f"(pid {pid if pid else 'not found'}) and triggering scheduled task '{task}' "
        "(IBC relaunches + auto-logs-in; paper login needs no 2FA)."
    )
    try:
        if pid:
            subprocess.run(["taskkill", "/PID", str(pid), "/F"],
                           capture_output=True, text=True, timeout=_SUBPROCESS_TIMEOUT)
        r = subprocess.run(["schtasks", "/Run", "/TN", task],
                           capture_output=True, text=True, timeout=_SUBPROCESS_TIMEOUT)
        if r.returncode != 0:
            logger.warning(
                f"[broker] gateway recovery: schtasks /Run '{task}' failed "
                f"(rc={r.returncode}): {(r.stderr or r.stdout or '').strip()[:200]}"
            )
            return False
    except Exception as e:
        logger.warning(f"[broker] gateway recovery failed ({type(e).__name__}: {e})")
        return False

    if not wait:
        return True

    # Wait for the fresh gateway's API port, then a short grace for login/API
    # readiness (port up ≠ login done; the caller's connect timeout covers the rest).
    deadline = now + max(10, int(settings.broker_gateway_restart_wait_seconds))
    while time.monotonic() < deadline:
        time.sleep(3)
        if _pid_listening_on(port):
            time.sleep(5)
            logger.info(f"[broker] gateway recovery: port {port} is back — redialing")
            return True
    logger.warning(
        f"[broker] gateway recovery: port {port} not listening after "
        f"{settings.broker_gateway_restart_wait_seconds}s — will reconnect on a later touchpoint"
    )
    return False
