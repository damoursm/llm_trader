"""Poll-loop runner — runs the pipeline every 30 min during market hours.

The pipeline operates on live prices and completed-only daily bars, so it is run
intraday on a 30-minute cadence (09:30–16:00 ET, Mon-Fri) rather than once
pre-market. Each 30-min boundary in the session window is a "slot"; the runner
fires one tick per slot and sends the daily email on the closing (16:00) slot only
(or on every slot when `scheduler_email_every_tick` is on).

Why a poll loop instead of APScheduler/cron: this box exposes only S0 "Connected
Standby" (no S1/S2/S3), so it suspends this process whenever the display sleeps —
and on battery Windows ignores keep-awake requests, so it suspends often. A blocking
cron scheduler computes one long sleep until the next fire; the suspend/resume cycles
skew that timer so the fire is pushed out and silently missed (observed: a slot did
not run even while the machine was briefly awake across it, and no misfire logged).
A short poll loop instead re-reads the wall clock every `scheduler_poll_seconds` and
decides from the *actual* time, so it self-corrects the moment the machine resumes:
a slot whose boundary passed during a brief suspension still runs (within the misfire
grace) as soon as the process wakes. Keep-awake is still requested (it helps on AC).
"""

import sys
import time as _time_module
from datetime import datetime, time as _time, timedelta

from loguru import logger

from config import settings
from src.pipeline import run_pipeline
from src.utils import now_et

# Windows SetThreadExecutionState flags (winbase.h).
_ES_CONTINUOUS = 0x80000000
_ES_SYSTEM_REQUIRED = 0x00000001


def _keep_system_awake(enable: bool) -> None:
    """Ask Windows not to idle into sleep/Modern-Standby while the scheduler runs.

    Issues a continuous ES_SYSTEM_REQUIRED power request (the mechanism media players
    use). Honored on AC; Windows may ignore it on battery. No-op off-Windows.
    """
    if not enable or sys.platform != "win32":
        return
    try:
        import ctypes

        if ctypes.windll.kernel32.SetThreadExecutionState(_ES_CONTINUOUS | _ES_SYSTEM_REQUIRED) == 0:
            logger.warning("[scheduler] keep-awake request was not honored by the OS")
        else:
            logger.info("[scheduler] keep-awake requested (helps on AC; battery may override)")
    except Exception as exc:  # pragma: no cover - platform/edge guard
        logger.warning(f"[scheduler] could not enable keep-awake: {exc}")


def _release_keep_awake() -> None:
    if sys.platform != "win32":
        return
    try:
        import ctypes

        ctypes.windll.kernel32.SetThreadExecutionState(_ES_CONTINUOUS)
    except Exception:
        pass


def _parse_hhmm(s: str, default: _time) -> _time:
    try:
        h, m = str(s).split(":")
        return _time(int(h), int(m))
    except Exception:
        return default


def _session_slots() -> tuple[list[_time], _time]:
    """The 30-min slot times in [session_start, session_end], plus the end time."""
    start = _parse_hhmm(settings.intraday_session_start, _time(9, 30))
    end = _parse_hhmm(settings.intraday_session_end, _time(16, 0))
    slots: list[_time] = []
    cur = datetime(2000, 1, 1, start.hour, start.minute)
    end_dt = datetime(2000, 1, 1, end.hour, end.minute)
    while cur <= end_dt:
        slots.append(cur.time())
        cur += timedelta(minutes=30)
    return slots, end


def _current_slot(now_naive: datetime, slot_times: list[_time]) -> datetime | None:
    """Latest slot boundary at/before `now` on a weekday, or None outside the session."""
    if now_naive.weekday() > 4:  # Sat/Sun
        return None
    today = now_naive.date()
    candidate: datetime | None = None
    for t in slot_times:
        slot_dt = datetime.combine(today, t)
        if slot_dt <= now_naive:
            candidate = slot_dt
        else:
            break
    return candidate


def start_scheduler() -> None:
    """Run the intraday poll loop: one tick per 30-min slot, 09:30–16:00 ET, Mon-Fri."""
    _keep_system_awake(settings.scheduler_keep_awake)

    poll = max(5, int(settings.scheduler_poll_seconds))
    grace = max(1, int(settings.scheduler_misfire_grace_sec))
    slot_times, end_t = _session_slots()

    logger.info(
        f"Scheduler started (poll loop). Slots: {slot_times[0].strftime('%H:%M')}–"
        f"{slot_times[-1].strftime('%H:%M')} ET every 30 min, Mon-Fri."
    )
    logger.info(
        f"Poll: {poll}s; misfire grace: {grace}s; "
        f"email_every_tick: {settings.scheduler_email_every_tick}; keep-awake: {settings.scheduler_keep_awake}."
    )
    logger.info("Press Ctrl+C to stop.")

    last_run_slot: datetime | None = None
    try:
        while True:
            now_naive = now_et().replace(tzinfo=None)
            slot_dt = _current_slot(now_naive, slot_times)

            if slot_dt is not None and slot_dt != last_run_slot:
                lateness = (now_naive - slot_dt).total_seconds()
                if lateness <= grace:
                    send_email = settings.scheduler_email_every_tick or (slot_dt.time() >= end_t)
                    logger.info(
                        f"[scheduler] tick for {slot_dt.strftime('%H:%M')} ET "
                        f"(email={send_email}, {lateness:.0f}s after slot)"
                    )
                    try:
                        run_pipeline(send_email=send_email)
                    except Exception as exc:  # never let one tick kill the loop
                        logger.exception(f"[scheduler] tick raised: {exc}")
                else:
                    logger.warning(
                        f"[scheduler] slot {slot_dt.strftime('%H:%M')} ET missed by "
                        f"{lateness / 60:.1f} min (> {grace / 60:.0f} min grace) — skipping "
                        "(machine was suspended too long). Keep it plugged in / awake."
                    )
                last_run_slot = slot_dt  # mark even when skipped, so we don't retry this slot

            _time_module.sleep(poll)
    except (KeyboardInterrupt, SystemExit):
        logger.info("Scheduler stopped.")
    finally:
        _release_keep_awake()
