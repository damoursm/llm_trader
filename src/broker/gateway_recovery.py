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
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import List, Optional, Tuple
from zoneinfo import ZoneInfo

from loguru import logger

from config import settings

_ET = ZoneInfo("America/New_York")

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


# Command-line markers that identify an IB Gateway java process. `ibcalpha.ibc.`
# is IBC's launcher main class (IbcGateway / IbcTws) and `\ibgateway\` is the
# IB install path on the classpath — both are specific enough that an unrelated
# java application can never match. Verified against a live gateway 2026-07-26.
_GATEWAY_CMDLINE_MARKERS = ("ibcalpha.ibc.", r"\ibgateway" + "\\", "/ibgateway/")

_PS_LIST_JAVA = (
    "Get-CimInstance Win32_Process -Filter \"Name='java.exe' OR Name='javaw.exe'\" | "
    "ForEach-Object { $_.ProcessId.ToString() + '~~~' + [string]$_.CommandLine }"
)


def _gateway_pids_by_signature() -> List[int]:
    """PIDs of IB Gateway java processes, identified by command-line signature.

    The port probe (``_pid_listening_on``) finds nothing when the gateway never
    BOUND its port — which is precisely the failure observed 2026-07-26: a
    gateway wedged for ~23 hours, process alive, port 4002 never opened, so
    recovery logged "pid not found", killed nothing, and fired the relaunch task
    on top of a surviving corpse. IBC's own watchdog can't see it either (it
    only checks the process is alive, and it was).

    So when the port is unbound we identify the gateway by what it IS rather
    than by what it is serving. Fail-soft: any error returns [] and recovery
    proceeds to the relaunch step exactly as before.
    """
    try:
        out = subprocess.run(
            ["powershell", "-NoProfile", "-NonInteractive", "-Command", _PS_LIST_JAVA],
            capture_output=True, text=True, timeout=_SUBPROCESS_TIMEOUT,
        ).stdout or ""
    except Exception as e:
        logger.warning(f"[broker] gateway recovery: process scan failed ({e})")
        return []
    pids: List[int] = []
    for line in out.splitlines():
        pid_s, _, cmd = line.partition("~~~")
        if not cmd:
            continue
        low = cmd.lower()
        if any(mark.lower() in low for mark in _GATEWAY_CMDLINE_MARKERS):
            try:
                pids.append(int(pid_s.strip()))
            except ValueError:
                continue
    return pids


def _process_ages(pids: List[int]) -> dict:
    """Seconds since each of ``pids`` started ({pid: age}); a pid it cannot read is
    left out. Fail-soft: any error returns {}."""
    if not pids:
        return {}
    ids = ",".join(str(int(p)) for p in pids)
    cmd = (f"Get-Process -Id {ids} -ErrorAction SilentlyContinue | ForEach-Object {{ "
           "$_.Id.ToString() + '~~~' + ((Get-Date) - $_.StartTime).TotalSeconds"
           ".ToString([Globalization.CultureInfo]::InvariantCulture) }")
    try:
        out = subprocess.run(
            ["powershell", "-NoProfile", "-NonInteractive", "-Command", cmd],
            capture_output=True, text=True, timeout=_SUBPROCESS_TIMEOUT,
        ).stdout or ""
    except Exception as e:
        logger.warning(f"[broker] gateway recovery: process age read failed ({e})")
        return {}
    ages = {}
    for line in out.splitlines():
        pid_s, _, age_s = line.partition("~~~")
        try:
            ages[int(pid_s.strip())] = float(age_s.strip())
        except ValueError:
            continue
    return ages


def _wait_for_booting_gateway(port: int, reason: str, wait: bool) -> Optional[bool]:
    """A gateway process that has not bound ``port`` yet because it is still BOOTING
    (IBC's relaunch opens it ~25-45 s after start) must not be killed — that restarts
    the clock (2026-09-29 03:02:20: the sync's recovery killed a gateway started 22 s
    before and relaunched it). None = no booting gateway (recover as usual); True =
    it opened the port (dial again); False = still booting (``wait=False``: the next
    touchpoint redials). A gateway still without a port after
    ``broker_gateway_boot_grace_seconds`` is stuck: None, and it is restarted."""
    grace = float(getattr(settings, "broker_gateway_boot_grace_seconds", 0) or 0)
    if grace <= 0 or _pid_listening_on(port):
        return None
    ages = _process_ages(_gateway_pids_by_signature())
    young = [a for a in ages.values() if 0 <= a < grace]
    if not young:
        return None
    left = grace - min(young)
    logger.info(f"[broker] gateway recovery wanted ({reason}) but the gateway started {min(young):.0f}s "
                f"ago and is still booting — {'waiting up to %.0fs for' % left if wait else 'leaving it to open'} "
                f"port {port}")
    if not wait:
        return False
    deadline = time.monotonic() + left
    while time.monotonic() < deadline:
        time.sleep(3)
        if _pid_listening_on(port):
            time.sleep(5)
            logger.info(f"[broker] gateway recovery: the booting gateway opened port {port} — redialing")
            return True
    logger.warning(f"[broker] gateway recovery: no port {port} {grace:.0f}s after the gateway started — "
                   f"it is stuck, restarting it")
    return None


# ── IBC's scheduled daily restart (AutoRestartTime) ──────────────────────────
# IBC restarts the gateway every day at its AutoRestartTime (11:50 PM ET here):
# the API socket closes and the gateway is back ~15-60 s later. That minute is
# not a dead gateway — the recovery below would kill the gateway IBC is
# restarting and launch a second one — so the broker client WAITS for it
# (`IBKRBroker.connect`) and the recovery stands down (2026-09-29).
_RESTART_CACHE: dict = {}


def _parse_clock(raw: str) -> Optional[Tuple[int, int]]:
    """``"11:50 PM"`` / ``"23:50"`` -> (23, 50); None when unparseable."""
    raw = (raw or "").strip().upper()
    for fmt in ("%I:%M %p", "%I:%M%p", "%H:%M"):
        try:
            t = datetime.strptime(raw, fmt)
            return t.hour, t.minute
        except ValueError:
            continue
    return None


def scheduled_restart_et() -> Optional[Tuple[int, int]]:
    """(hour, minute) ET of the gateway's scheduled daily restart:
    ``broker_gateway_restart_et`` when set, else IBC's ``AutoRestartTime`` read
    from ``ibc_config_path`` (ONLY that line is parsed — the file holds the login);
    None when neither is known. The file is re-read when it changes."""
    raw = str(getattr(settings, "broker_gateway_restart_et", "") or "").strip()
    if raw:
        return _parse_clock(raw)
    p = Path(str(getattr(settings, "ibc_config_path", "") or ""))
    try:
        key = (str(p), p.stat().st_mtime)
    except OSError:
        return None
    if _RESTART_CACHE.get("key") == key:
        return _RESTART_CACHE.get("value")
    value = None
    try:
        for line in p.read_text(encoding="utf-8", errors="replace").splitlines():
            s = line.strip()
            if s.startswith("AutoRestartTime="):
                value = _parse_clock(s.split("=", 1)[1])
                break
    except OSError:
        value = None
    _RESTART_CACHE.update(key=key, value=value)
    return value


def in_scheduled_restart_window(now: Optional[datetime] = None) -> Optional[datetime]:
    """The window's END (aware) when ``now`` is inside the gateway's scheduled daily
    restart — from one minute before it to ``broker_gateway_restart_window_minutes``
    after (across midnight too) — else None."""
    hm = scheduled_restart_et()
    if hm is None:
        return None
    et = (now or datetime.now(timezone.utc)).astimezone(_ET)
    span = timedelta(minutes=max(1.0, float(getattr(settings, "broker_gateway_restart_window_minutes", 5.0))))
    for back in (0, 1):                               # a window that began yesterday
        base = (et - timedelta(days=back)).replace(hour=hm[0], minute=hm[1], second=0, microsecond=0)
        if base - timedelta(minutes=1) <= et <= base + span:
            return base + span
    return None


def wait_for_gateway(until: datetime, poll_seconds: float = 5.0, grace_seconds: float = 10.0) -> bool:
    """Block until the gateway's API port listens again (then a short grace for
    its login), at most until ``until`` (aware). True when it came back."""
    port = int(settings.ibkr_port)
    while datetime.now(timezone.utc) < until:
        if _pid_listening_on(port):
            time.sleep(grace_seconds)
            return True
        time.sleep(poll_seconds)
    return bool(_pid_listening_on(port))


_RELAUNCH_WAIT_SECONDS = 20.0


def _task_running(task: str) -> Optional[bool]:
    """Whether the scheduled task has an instance running (``schtasks /Query``: one CSV
    row per trigger, the status last), None when it can't be told."""
    try:
        r = subprocess.run(["schtasks", "/Query", "/TN", task, "/FO", "CSV", "/NH"],
                           capture_output=True, text=True, timeout=_SUBPROCESS_TIMEOUT)
    except Exception:                                            # noqa: BLE001
        return None
    rows = [ln.strip() for ln in (getattr(r, "stdout", "") or "").splitlines() if ln.strip()]
    if getattr(r, "returncode", 1) != 0 or not rows:
        return None
    return any(ln.rsplit(",", 1)[-1].strip().strip('"').lower() == "running" for ln in rows)


def _relaunch(task: str, killed: List[int]):
    """``schtasks /Run`` once the killed gateway's task instance has ENDED.

    The IBC task runs with MultipleInstances=IgnoreNew, and its instance lives as long as
    StartGateway.bat: killing the gateway's java leaves the batch file's epilogue running
    for about a second, and a /Run inside that second is accepted (rc 0) but ignored — 4
    of the 5 recoveries 2026-09-29..10-05 came back only at the task's 10-minute
    keep-alive trigger while the sync waited out its 300 s. So: wait for the instance to
    end (<= 20 s); one still running then is ended first — unless it is a gateway the
    keep-alive already relaunched (a gateway process we did not kill), which is kept."""
    deadline = time.monotonic() + _RELAUNCH_WAIT_SECONDS
    running = _task_running(task)
    while running and time.monotonic() < deadline:
        time.sleep(1.0)
        running = _task_running(task)
    if running:
        fresh = [p for p in _gateway_pids_by_signature() if p not in set(killed)]
        if fresh:
            logger.info(f"[broker] gateway recovery: task '{task}' is already running a new gateway "
                        f"(pid {fresh}) — not relaunching")
            return subprocess.CompletedProcess(["schtasks", "/Run", "/TN", task], 0, "", "")
        logger.warning(f"[broker] gateway recovery: task '{task}' still running "
                       f"{_RELAUNCH_WAIT_SECONDS:.0f}s after the kill — ending it before the relaunch")
        subprocess.run(["schtasks", "/End", "/TN", task],
                       capture_output=True, text=True, timeout=_SUBPROCESS_TIMEOUT)
        time.sleep(2.0)
    return subprocess.run(["schtasks", "/Run", "/TN", task],
                          capture_output=True, text=True, timeout=_SUBPROCESS_TIMEOUT)


def maybe_restart_gateway(reason: str, wait: bool = True) -> bool:
    """Kill the gateway owning ``ibkr_port`` and fire the IBC relaunch task.

    Returns True when a restart was triggered AND (``wait=True``) the fresh
    gateway's port is listening again — i.e. the caller should dial once more
    now. ``wait=False`` (the wedge path, called from broker touchpoints) fires
    and returns immediately; the next touchpoint's auto-reconnect picks the
    fresh gateway up. False = nothing was done (gated off / cooldown / failed /
    IBC's own scheduled daily restart in progress).
    """
    end = in_scheduled_restart_window()
    if end is not None:
        logger.info(f"[broker] gateway recovery wanted ({reason}) but IBC's scheduled daily restart is in "
                    f"progress (until {end.astimezone(_ET):%H:%M} ET) — not intervening")
        return False
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
    port = int(settings.ibkr_port)
    booting = _wait_for_booting_gateway(port, reason, wait)
    if booting is not None:
        return booting                 # no restart: the cooldown is not consumed
    now = time.monotonic()
    _last_restart_mono = now

    task = settings.broker_gateway_task_name
    pid = _pid_listening_on(port)
    # A gateway that never BOUND the port owns no listener, so the port probe
    # returns nothing and the corpse would survive the relaunch (observed
    # 2026-07-26: wedged ~23h). Fall back to identifying it by signature.
    by_sig: List[int] = [] if pid else _gateway_pids_by_signature()
    targets = [pid] if pid else by_sig
    how = (f"pid {pid} on port {port}" if pid else
           (f"pid(s) {by_sig} by signature — nothing was listening on {port}"
            if by_sig else f"nothing found (no listener on {port}, no gateway process)"))
    logger.critical(
        f"[broker] GATEWAY RECOVERY — {reason}. Killing {how} and triggering "
        f"scheduled task '{task}' (IBC relaunches + auto-logs-in; paper login "
        "needs no 2FA)."
    )
    try:
        for _t in targets:
            subprocess.run(["taskkill", "/PID", str(_t), "/F"],
                           capture_output=True, text=True, timeout=_SUBPROCESS_TIMEOUT)
        r = _relaunch(task, targets)
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


# ── is the gateway logged in to IBKR? (IBC's own log) ────────────────────────
# A gateway whose IBKR login failed keeps its API port open, so connects succeed
# and the sync reads healthy (2026-09-28 10:59: a manual login to the paper
# account elsewhere → "Existing session detected" → re-login refused with
# "Unrecognized Username or Password"; nothing alerted for 68 minutes). IBC logs
# every login outcome; the LAST definitive one is the gateway's state.
_IBC_LOGIN_OK = "Login has completed"
_IBC_LOGIN_FAILED = "Unrecognized Username or Password; event=Opened"
_IBC_EXISTING = "Existing session detected"      # IBKR: this account logged in elsewhere


def ibc_login_state(log_dir: Optional[str] = None, tail_bytes: int = 400_000) -> Optional[dict]:
    """``{"logged_in": bool, "at": "YYYY-MM-DD HH:MM:SS", "reason": str,
    "existing_session": bool}`` from the newest IBC log's last login outcome, or
    None when the log can't be read or holds no outcome. ``existing_session``: the
    refusal came ≤ 10 min after "Existing session detected" (a login elsewhere).
    Fail-soft."""
    from pathlib import Path
    try:
        d = Path(log_dir or settings.ibc_log_dir)
        files = sorted(d.glob("IBC-*.txt"), key=lambda p: p.stat().st_mtime)
        if not files:
            return None
        with open(files[-1], "rb") as fh:
            fh.seek(0, 2)
            size = fh.tell()
            fh.seek(max(0, size - int(tail_bytes)))
            text = fh.read().decode("utf-8", errors="replace")
    except Exception as e:                                         # noqa: BLE001
        logger.debug(f"[broker] IBC log unreadable: {e}")
        return None
    last = None
    existing_at = None                 # the last "Existing session detected" (a login elsewhere)
    existing_before = False
    for line in text.splitlines():
        if _IBC_EXISTING in line:
            existing_at = _log_ts(line)
        elif _IBC_LOGIN_OK in line:
            last = (True, line)
            existing_before = False
        elif _IBC_LOGIN_FAILED in line:
            last = (False, line)
            ts = _log_ts(line)
            existing_before = bool(existing_at and ts and 0 <= (ts - existing_at).total_seconds() <= 600)
    if last is None:
        return None
    ok, line = last
    return {"logged_in": ok, "at": line[:19],
            "reason": "login completed" if ok else "login refused: Unrecognized Username or Password",
            # a refusal right after "Existing session detected": someone logged in to
            # this account elsewhere — never restarted automatically (it would kick them out)
            "existing_session": bool(not ok and existing_before)}


def _log_ts(line: str):
    """The instant an IBC log line was written (its local-time stamp), or None."""
    try:
        return datetime.strptime(line[:19], "%Y-%m-%d %H:%M:%S")
    except ValueError:
        return None


def wait_for_login(after: datetime, timeout_seconds: float = 90.0, poll_seconds: float = 5.0) -> bool:
    """Block until IBC logs a completed login stamped at/after ``after`` (naive
    local time, as IBC writes it), at most ``timeout_seconds``. True when it did."""
    deadline = time.monotonic() + timeout_seconds
    while True:
        st = ibc_login_state()
        if st and st.get("logged_in"):
            ts = _log_ts(str(st.get("at") or ""))
            if ts is None or ts >= after.replace(microsecond=0):
                return True
        if time.monotonic() >= deadline:
            return False
        time.sleep(poll_seconds)

