"""Poll-loop runner — runs the pipeline every 30 min during market hours.

The pipeline operates on live prices and completed-only daily bars, so it is run
intraday on a 30-minute cadence (09:30–16:00 ET, Mon-Fri) rather than once
pre-market. Each 30-min boundary in the session window is a "slot"; the runner
fires one tick per slot and sends the daily email on the closing (16:00) slot only
(or on every slot when `scheduler_email_every_tick` is on).

When `extended_hours_mode` != "off", additional tagged slots cover the
extended sessions (`extended_windows`, default 04:00–09:30 pre-market and
16:00–20:00 after-hours ET, with per-window cadence — the liquid shoulders
tick every 30 min, the thin dead zones hourly via the "@60" suffix). What an
extended tick DOES depends on the mode: "observe" (Phase 0) runs the full
pipeline with persistence but no ledger/broker mutation
(`run_pipeline(observe_only=True)`); "trade" (Phase 1, default) runs it as a
FULL trading tick — entries/exits/marks and broker paper orders happen
off-hours too, with session-aware costs and sizing applied by the tracker.
The daily email fires only on the closing RTH slot in every mode — an
after-hours trading tick past 16:00 must not re-send the daily report.
NYSE holidays are skipped entirely (no session, regular or extended, on a
closed market).

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
from src.performance.market_calendar import is_market_day
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


def _parse_windows(spec: str) -> list[tuple[_time, _time, int | None]]:
    """Parse ``'HH:MM-HH:MM[@MM],…'`` → [(start, end, step_minutes | None), …].

    The optional ``@MM`` suffix overrides the tick cadence for that window
    (None = use ``extended_tick_minutes``) so thin dead zones can tick less
    often than the liquid shoulders. Bad tokens are skipped with a warning.
    """
    out: list[tuple[_time, _time, int | None]] = []
    for tok in str(spec or "").split(","):
        tok = tok.strip()
        if not tok:
            continue
        step: int | None = None
        rng = tok
        if "@" in tok:
            rng, _, step_s = tok.partition("@")
            try:
                step = max(5, int(step_s))
            except ValueError:
                logger.warning(f"[scheduler] ignoring bad extended window: {tok!r}")
                continue
        start = end = None
        if rng.count("-") == 1:
            a, b = rng.split("-")
            start, end = _parse_hhmm(a, None), _parse_hhmm(b, None)
        if start is None or end is None or start >= end:
            logger.warning(f"[scheduler] ignoring bad extended window: {tok!r}")
            continue
        out.append((start, end, step))
    return out


def _session_slots() -> tuple[list[tuple[_time, str]], _time]:
    """All tick slots as time-ordered ``(time, kind)`` pairs, plus the RTH end time.

    kind = "rth": the regular 30-min session slots (email fires on the closing
    one). kind = "extended": observation slots from ``extended_windows``,
    generated only when ``extended_hours_mode`` != "off"; each window ticks at
    its own ``@MM`` cadence (default ``extended_tick_minutes``). A slot
    colliding with an already-emitted slot time is dropped — the RTH tick owns
    its boundaries (e.g. a 07:00–09:30 window yields 07:00…09:00 and cedes
    09:30 to RTH), and adjacent windows sharing an endpoint fire it once.
    """
    start = _parse_hhmm(settings.intraday_session_start, _time(9, 30))
    end = _parse_hhmm(settings.intraday_session_end, _time(16, 0))
    slots: list[tuple[_time, str]] = []
    cur = datetime(2000, 1, 1, start.hour, start.minute)
    end_dt = datetime(2000, 1, 1, end.hour, end.minute)
    while cur <= end_dt:
        slots.append((cur.time(), "rth"))
        cur += timedelta(minutes=30)
    if (settings.extended_hours_mode or "off").lower() != "off":
        seen = {t for t, _ in slots}
        default_step = max(5, int(settings.extended_tick_minutes))
        for w_start, w_end, w_step in _parse_windows(settings.extended_windows):
            step = w_step or default_step
            cur = datetime(2000, 1, 1, w_start.hour, w_start.minute)
            w_end_dt = datetime(2000, 1, 1, w_end.hour, w_end.minute)
            while cur <= w_end_dt:
                if cur.time() not in seen:
                    slots.append((cur.time(), "extended"))
                    seen.add(cur.time())
                cur += timedelta(minutes=step)
    slots.sort(key=lambda p: (p[0].hour, p[0].minute))
    return slots, end


def _tick_plan(kind: str, slot_t: _time, end_t: _time) -> tuple[bool, bool]:
    """Per-slot decisions as ``(observe, send_email)``.

    observe    — True only for extended slots while NOT in "trade" mode
                 (Phase 0 observation; "trade" runs them as full ticks).
    send_email — True only for RTH slots (the closing one, or every RTH tick
                 when scheduler_email_every_tick). Extended slots never email
                 in any mode: after-hours slots are past 16:00, so a plain
                 time>=close check would re-send the daily report every
                 extended trading tick.
    """
    mode = (settings.extended_hours_mode or "off").lower()
    observe = kind == "extended" and mode != "trade"
    send_email = kind == "rth" and (
        settings.scheduler_email_every_tick or (slot_t >= end_t)
    )
    return observe, send_email


def _current_slot(now_naive: datetime, slots: list[tuple[_time, str]]) -> tuple[datetime, str] | None:
    """Latest ``(slot boundary, kind)`` at/before `now` on a market day, or None.

    Weekends AND NYSE holidays yield None — there is no session (regular or
    extended) on a closed market, so ticking would only burn LLM calls
    marking stale prices.
    """
    if not is_market_day(now_naive.date()):
        return None
    today = now_naive.date()
    candidate: tuple[datetime, str] | None = None
    for t, kind in slots:
        slot_dt = datetime.combine(today, t)
        if slot_dt <= now_naive:
            candidate = (slot_dt, kind)
        else:
            break
    return candidate


def start_scheduler() -> None:
    """Run the intraday poll loop: one tick per slot, NYSE market days only.

    RTH slots every 30 min 09:30–16:00 ET; extended observation slots per
    ``extended_windows`` (default full 04:00–20:00 coverage with per-window
    cadence). Weekends and NYSE holidays are skipped.
    """
    _keep_system_awake(settings.scheduler_keep_awake)

    poll = max(5, int(settings.scheduler_poll_seconds))
    grace = max(1, int(settings.scheduler_misfire_grace_sec))
    slots, end_t = _session_slots()
    rth_slots = [t for t, k in slots if k == "rth"]
    ext_slots = [t for t, k in slots if k == "extended"]

    logger.info(
        f"Scheduler started (poll loop). RTH slots: {rth_slots[0].strftime('%H:%M')}–"
        f"{rth_slots[-1].strftime('%H:%M')} ET every 30 min, NYSE market days "
        "(weekends + holidays skipped)."
    )
    if ext_slots:
        _mode = (settings.extended_hours_mode or "off").lower()
        _what = (
            "FULL TRADING ticks — ledger + broker mutations, session-aware costs/sizing"
            if _mode == "trade"
            else "observation — full pipeline + persistence, no ledger/broker mutations"
        )
        logger.info(
            f"Extended slots ({_mode}, "
            f"{ext_slots[0].strftime('%H:%M')}–{ext_slots[-1].strftime('%H:%M')} ET): "
            f"{', '.join(t.strftime('%H:%M') for t in ext_slots)} — {_what}."
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
            current = _current_slot(now_naive, slots)

            if current is not None and current[0] != last_run_slot:
                slot_dt, kind = current
                lateness = (now_naive - slot_dt).total_seconds()
                if lateness <= grace:
                    observe, send_email = _tick_plan(kind, slot_dt.time(), end_t)
                    tick_label = ""
                    if kind == "extended":
                        tick_label = ", OBSERVATION" if observe else ", EXTENDED-TRADE"
                    logger.info(
                        f"[scheduler] tick for {slot_dt.strftime('%H:%M')} ET "
                        f"({kind}{tick_label}, "
                        f"email={send_email}, {lateness:.0f}s after slot)"
                    )
                    try:
                        # email_if_configured=False: the per-slot send_email
                        # decision is authoritative for scheduled ticks —
                        # extended trading ticks never re-send the report.
                        run_pipeline(send_email=send_email, observe_only=observe,
                                     email_if_configured=False)
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
