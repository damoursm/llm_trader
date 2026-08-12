"""Poll-loop runner — runs the pipeline every 30 min during market hours.

The pipeline operates on live prices and completed-only daily bars, so it is run
intraday on a 30-minute cadence (09:30–16:00 ET, Mon-Fri) rather than once
pre-market. Each 30-min boundary in the session window is a "slot"; the runner
fires one tick per slot and emails the report at the slots listed in
`scheduler_email_times` (default 04:00, 09:30, 16:00, 19:50 ET — pre-market open,
RTH open, RTH close, last after-hours tick), or on every slot when
`scheduler_email_every_tick` is on, or — if `scheduler_email_times` is cleared —
only on the closing (16:00) slot.

When `extended_hours_mode` != "off", additional tagged slots cover the
extended sessions (`extended_windows`, default 04:00–09:30 pre-market and
16:00–20:00 after-hours ET, with per-window cadence — the liquid shoulders
tick every 30 min, the thin dead zones hourly via the "@60" suffix). What an
extended tick DOES depends on the mode: "observe" (Phase 0) runs the full
pipeline with persistence but no ledger/broker mutation
(`run_pipeline(observe_only=True)`); "trade" (Phase 1, default) runs it as a
FULL trading tick — entries/exits/marks and broker paper orders happen
off-hours too, with session-aware costs and sizing applied by the tracker.
The report emails at the `scheduler_email_times` slots regardless of session
(so an after-hours slot like 19:50 can send one) — the time-set gate replaces the
old "16:00 close only" rule, so a non-listed after-hours tick no longer emails.
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
import threading
import time as _time_module
from datetime import datetime, time as _time, timedelta

from loguru import logger

from config import settings
from src.performance.market_calendar import current_session, is_market_day, is_overnight_session_open
from src.pipeline import run_pipeline
from src.utils import now_et

# Windows SetThreadExecutionState flags (winbase.h).
_ES_CONTINUOUS = 0x80000000
_ES_SYSTEM_REQUIRED = 0x00000001
_ES_DISPLAY_REQUIRED = 0x00000002


def _keep_system_awake(enable: bool) -> None:
    """Ask Windows not to idle into sleep / Modern Standby while the scheduler runs.

    Issues a continuous power request (the mechanism media players use). Honored on
    AC; Windows may ignore it on battery. No-op off-Windows.

    CRITICAL on Modern Standby (S0) machines (2026-07-14): ``ES_SYSTEM_REQUIRED``
    alone blocks classic S3 sleep but does NOT keep an S0 machine out of *connected
    standby* — which FREEZES this process (the 6 h dark-loop / "wall clock jumped"
    incidents: the OS enters Modern Standby when the DISPLAY turns off, regardless of
    ES_SYSTEM_REQUIRED). Adding ``ES_DISPLAY_REQUIRED`` holds the display on, and an
    on display keeps the box in the working state so background ticks keep firing.
    Complementary to the OS power settings (display-off=never on AC, lid=do-nothing);
    this in-app request is the portable belt that survives a settings reset.
    """
    if not enable or sys.platform != "win32":
        return
    try:
        import ctypes

        flags = _ES_CONTINUOUS | _ES_SYSTEM_REQUIRED | _ES_DISPLAY_REQUIRED
        if ctypes.windll.kernel32.SetThreadExecutionState(flags) == 0:
            logger.warning("[scheduler] keep-awake request was not honored by the OS")
        else:
            logger.info("[scheduler] keep-awake requested — system + display held on "
                        "(prevents Modern Standby freezing ticks; AC only, battery may override)")
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


def _parse_hhmm(s: str, default: _time | None) -> _time | None:
    try:
        h, m = str(s).split(":")
        return _time(int(h), int(m))
    except Exception:
        return default


def _parse_windows(spec: str) -> list[tuple[_time, _time, int | None]]:
    """Parse ``'HH:MM-HH:MM[@MM],…'`` → [(start, end, step_minutes | None), …].

    The optional ``@MM`` suffix overrides the tick cadence for that window
    (None = use ``extended_tick_minutes``) so thin dead zones can tick less
    often than the liquid shoulders. A window with equal endpoints
    (``19:50-19:50``) is a SINGLE slot at exactly that time — used to place
    the last after-hours tick early enough that its orders can still fill
    before the 20:00 session close (it leaves a ~10-min buffer so the
    every-tick hold-review's extra latency still clears the close). Bad tokens
    are skipped with a warning.
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
        if start is None or end is None or start > end:
            logger.warning(f"[scheduler] ignoring bad extended window: {tok!r}")
            continue
        out.append((start, end, step))
    return out


def _session_slots() -> tuple[list[tuple[_time, str]], _time]:
    """All tick slots as time-ordered ``(time, kind)`` pairs, plus the RTH end time.

    kind = "rth": the regular 30-min session slots (email fires on the closing
    one). kind = "extended": observation slots from ``extended_windows``,
    generated only when ``extended_hours_mode`` != "off"; each window ticks at
    its own ``@MM`` cadence (default ``extended_tick_minutes``). kind =
    "overnight": slots from ``overnight_windows`` when ``overnight_hours_mode``
    != "off" — their market-day validity is decided per-slot in
    ``_slot_is_valid`` (an evening slot belongs to the NEXT day's session).
    A slot colliding with an already-emitted slot time is dropped — the RTH
    tick owns its boundaries (e.g. a 07:00–09:30 window yields 07:00…09:00 and
    cedes 09:30 to RTH), and adjacent windows sharing an endpoint fire it once.
    """
    start = _parse_hhmm(settings.intraday_session_start, _time(9, 30))
    end = _parse_hhmm(settings.intraday_session_end, _time(16, 0))
    slots: list[tuple[_time, str]] = []
    cur = datetime(2000, 1, 1, start.hour, start.minute)
    end_dt = datetime(2000, 1, 1, end.hour, end.minute)
    while cur <= end_dt:
        slots.append((cur.time(), "rth"))
        cur += timedelta(minutes=30)
    seen = {t for t, _ in slots}
    default_step = max(5, int(settings.extended_tick_minutes))

    def _add_windows(spec: str, kind: str) -> None:
        for w_start, w_end, w_step in _parse_windows(spec):
            step = w_step or default_step
            w_cur = datetime(2000, 1, 1, w_start.hour, w_start.minute)
            w_end_dt = datetime(2000, 1, 1, w_end.hour, w_end.minute)
            while w_cur <= w_end_dt:
                if w_cur.time() not in seen:
                    slots.append((w_cur.time(), kind))
                    seen.add(w_cur.time())
                w_cur += timedelta(minutes=step)

    if (settings.extended_hours_mode or "off").lower() != "off":
        _add_windows(settings.extended_windows, "extended")
    if (settings.overnight_hours_mode or "off").lower() != "off":
        _add_windows(settings.overnight_windows, "overnight")
    slots.sort(key=lambda p: (p[0].hour, p[0].minute))
    return slots, end


def _email_slot_times() -> set[_time]:
    """ET slot times at which the scheduled report emails, from
    ``scheduler_email_times`` (CSV of HH:MM). Empty/unset → empty set, so the
    caller falls back to the 16:00-close default. Each time only fires if it is an
    actual tick slot (see ``_session_slots``); ``start_scheduler`` warns otherwise."""
    out: set[_time] = set()
    for tok in str(settings.scheduler_email_times or "").split(","):
        tok = tok.strip()
        if not tok:
            continue
        t = _parse_hhmm(tok, None)
        if t is not None:
            out.add(t)
    return out


def _tick_plan(kind: str, slot_t: _time, end_t: _time) -> tuple[bool, bool]:
    """Per-slot decisions as ``(observe, send_email)``.

    observe    — True only for extended/overnight slots while their session's
                 mode is NOT "trade" (Phase 0 observation; "trade" runs them as
                 full ticks). Each session has its OWN mode knob.
    send_email — precedence: (1) ``scheduler_email_every_tick`` on ⇒ EVERY slot
                 emails (RTH + extended alike); else (2) ``scheduler_email_times``
                 set ⇒ ONLY slots whose time is in that set email (this is the
                 default — 04:00/09:30/16:00/19:50 ET — and works for extended
                 slots too, e.g. the 19:50 after-hours close report); else
                 (3) legacy: only the closing RTH slot (``time >= end`` — the
                 ``kind`` gate stops an after-hours slot past 16:00 re-sending it).
    """
    if kind == "overnight":
        observe = (settings.overnight_hours_mode or "off").lower() != "trade"
    else:
        mode = (settings.extended_hours_mode or "off").lower()
        observe = kind == "extended" and mode != "trade"
    email_times = _email_slot_times()
    if settings.scheduler_email_every_tick:
        send_email = True
    elif email_times:
        send_email = slot_t in email_times
    else:
        send_email = kind == "rth" and slot_t >= end_t
    return observe, send_email


def _should_run_eod(now_naive: datetime, last_eod_date, eod_time: _time) -> bool:
    """True on the first poll at/after ``eod_time`` ET on a market day it hasn't
    yet run for. Independent of the slot grid, so a missed close slot never
    blocks maintenance — it fires whenever the machine is next awake past the
    trigger."""
    if not settings.enable_eod_maintenance:
        return False
    return (last_eod_date != now_naive.date()
            and now_naive.time() >= eod_time
            and is_market_day(now_naive.date()))


_RESCORE_THREAD = None


def _should_run_nightly_rescore(now_naive: datetime, last_date, at: _time) -> bool:
    """True on the first poll at/after ``at`` ET on a date it hasn't run for.

    Deliberately NOT gated on `is_market_day`: the rescore repairs stored
    history, which is just as stale on a Saturday, and a weekend night is the
    quietest window it will ever get.
    """
    if not settings.enable_auto_refactor:
        return False
    return last_date != now_naive.date() and now_naive.time() >= at


def _maybe_start_nightly_rescore() -> bool:
    """Detect implementation changes and, if any, rescore in the BACKGROUND.

    Two properties matter here and both are deliberate:

    * **Detection first, cheaply.** Hashing 42 module ASTs takes well under a
      second, so an unchanged night costs essentially nothing and starts no
      thread. The expensive path only runs when something actually changed.
    * **Background, because the scheduler loop is single-threaded.** A rescore
      is ~40 minutes; running it inline at 02:00 ET would block the 02:00,
      03:00 and 03:30 OVERNIGHT ticks outright — real missed trading, to repair
      history that is in no hurry. The heavy phases hold no DB lock (they
      compute, then write briefly), and both sides retry on the write lock, so
      the overlap window is small. A read that loses the race still fails soft,
      which is why walk-forward steps carry a `degraded` flag.

    Returns True when a rescore was started.
    """
    global _RESCORE_THREAD
    if _RESCORE_THREAD is not None and _RESCORE_THREAD.is_alive():
        logger.warning("[scheduler] nightly rescore still running from a previous "
                       "night — not starting another")
        return False
    try:
        from src.analysis.refactor import plan
        p = plan()
    except Exception as exc:
        logger.warning(f"[scheduler] rescore change-detection failed: {exc}")
        return False

    if not p["changed"]:
        if p["first_seen"]:
            # Baseline only — record fingerprints so the next real change is
            # detectable. Cheap and synchronous.
            try:
                from src.analysis.code_version import record
                logger.info(f"[scheduler] nightly rescore: recorded "
                            f"{record(p['first_seen'])} baseline fingerprint(s)")
            except Exception as exc:
                logger.warning(f"[scheduler] baseline record failed: {exc}")
        else:
            logger.info("[scheduler] nightly rescore: no implementation changes")
        return False

    logger.info(f"[scheduler] nightly rescore STARTING in background — changed: "
                f"{p['changed']}")

    def _work():
        try:
            from src.analysis.refactor import run_refactor
            res = run_refactor(apply=True)
            if res["ok"]:
                logger.info(f"[scheduler] nightly rescore complete: "
                            f"{[(s['step'], s['status']) for s in res['steps']]}")
            else:
                logger.warning("[scheduler] nightly rescore INCOMPLETE")
                _alert("Automatic rescore incomplete",
                       f"An implementation change was detected "
                       f"({res['plan']['changed']}) but a repair step failed:\n\n"
                       f"{res['steps']}\n\n"
                       "Stored history is STALE for those methods. Fingerprints "
                       "were deliberately not advanced, so this retries "
                       "tomorrow night rather than silently marking the "
                       "database up to date.")
        except Exception as exc:
            logger.exception(f"[scheduler] nightly rescore raised: {exc}")
            _alert_crash("nightly rescore", exc)

    _RESCORE_THREAD = threading.Thread(target=_work, name="nightly-rescore",
                                       daemon=True)
    _RESCORE_THREAD.start()
    return True


_EOD_THREAD = None


def _run_eod_maintenance() -> None:
    """Launch the EOD maintenance in a BACKGROUND thread (2026-08-11).

    It ran inline in the scheduler loop until the 20y cache made it a >5.5h job
    (measured 2026-08-10) that starved the entire extended + overnight sessions
    — "off the time-critical path" was only true while it was short. Same
    pattern as the nightly rescore: single-flight guard, daemon thread, the
    loop keeps ticking. The heavy phases hold no DB lock (compute, then write
    briefly) and both sides retry on the write lock. `last_eod_date` is marked
    by the caller at LAUNCH, so a >24h EOD cannot re-trigger itself; a still-
    running thread at the next day's slot is warned about and skipped."""
    global _EOD_THREAD
    if _EOD_THREAD is not None and _EOD_THREAD.is_alive():
        logger.warning("[scheduler] EOD maintenance still running from a previous "
                       "day — not starting another")
        return
    import threading as _threading
    _EOD_THREAD = _threading.Thread(target=_eod_work, name="eod-maintenance",
                                    daemon=True)
    _EOD_THREAD.start()
    logger.info("[scheduler] EOD maintenance STARTING in background thread")


def _eod_work() -> None:
    """The EOD body: cache warm → retention → replay, then walk-forward ∥ ML
    retrains in parallel (the trains need the replay-refreshed panel; the
    walk-forward does not feed them). Every stage fail-soft."""
    logger.info("[scheduler] EOD maintenance: warming forward-return cache + retention…")
    # Drop the slow-moving panel calibrations so the fresh panel / replay /
    # walk-forward / retrained models produced below are picked up on the NEXT
    # tick rather than waiting out ic_weight_cache_seconds (2 h). The TTL is long
    # precisely because these move slowly WITHIN a day; EOD is when they change.
    try:
        from src.analysis.market_relative import reset_cache as _reset_market_relative
        from src.signals.aggregator import reset_winrate_filter_cache
        reset_winrate_filter_cache()
        _reset_market_relative()
    except Exception as exc:
        logger.debug(f"[scheduler] EOD calibration-cache reset skipped: {exc}")
    try:
        from src.data.cache_warm import warm_forward_return_cache
        warm_forward_return_cache(
            days=int(settings.eod_cache_warm_days),
            max_tickers=(settings.eod_cache_warm_max_tickers or None))
    except Exception as exc:
        logger.warning(f"[scheduler] EOD cache warm failed: {exc}")
    try:
        from src.db.retention import run_retention
        res = run_retention()
        if res:
            logger.info(f"[scheduler] EOD retention: {res}")
    except Exception as exc:
        logger.warning(f"[scheduler] EOD retention failed: {exc}")
    # Rescore history through the CURRENT scorers so the calibrations fit values
    # today's code produced. Runs AFTER the cache warm above, which is what
    # supplies the bars a replay reads. Fail-soft: a stale replay table only
    # means the epoch mask blanks more, never that a wrong value is served.
    if settings.enable_eod_replay_refresh:
        try:
            from src.analysis.replay import materialize
            n = materialize(days=(int(settings.eod_replay_refresh_days) or None))
            logger.info(f"[scheduler] EOD replay: {n:,} ticker-days rescored")
        except Exception as exc:
            logger.warning(f"[scheduler] EOD replay refresh failed: {exc}")
    # Append TODAY's point-in-time calibration to `weight_history`. Incremental
    # by design: each step is a fixed fact about a date once its data is in, so
    # only the tail is recomputed (`walkforward_eod_days`) rather than rewalking
    # the whole span, which costs ~60s PER STEP. Runs after the replay above,
    # which supplies the panel the calibration reads.
    def _wf_step() -> None:
        if not settings.enable_eod_walkforward:
            return
        try:
            from datetime import date as _date, timedelta as _td
            from src.analysis.walkforward import materialize as wf_materialize
            _lookback = max(1, int(settings.walkforward_eod_days))
            _start = (_date.today() - _td(days=_lookback)).isoformat()
            n = wf_materialize(start=_start, step_days=1)
            logger.info(f"[scheduler] EOD walk-forward: {n} calibration step(s) stored")
        except Exception as exc:
            logger.warning(f"[scheduler] EOD walk-forward failed: {exc}")

    def _ml_step() -> None:
        # Retrain the ml_ohlcv model (signals/ml_model.py; weighted 0.12 since
        # 2026-08-11, throttled weekly in pivot mode). A stale/failed artifact
        # means the method abstains (score 0) — degraded, never a broken tick.
        if settings.enable_eod_ml_train:
            try:
                from src.signals.ml_model import eod_train
                art = eod_train()
                if art:
                    logger.info(f"[scheduler] EOD ml_ohlcv train: {art['n_train']:,} rows "
                                f"(<= {art['train_max_date']})")
            except Exception as exc:
                logger.warning(f"[scheduler] EOD ml_ohlcv train failed: {exc}")
        # Retrain both stackers on the freshly-materialised panel (the ML combine
        # arm is LIVE at 50% since 2026-08-11, STACKER_GBM_PARAMS config). A
        # stale/failed artifact only means the arm falls back to the weighted
        # combine (combine_source records it). Fail-soft.
        if settings.enable_eod_ml_buy_train:
            try:
                from src.analysis.ml_stacker import eod_train_buy, eod_train_sell
                for _name, _fn in (("ml_buy", eod_train_buy), ("ml_sell", eod_train_sell)):
                    art = _fn()
                    if art:
                        logger.info(f"[scheduler] EOD {_name} train: {art['n_train']:,} rows "
                                    f"(<= {art['train_max_date']})")
            except Exception as exc:
                logger.warning(f"[scheduler] EOD ml_buy/ml_sell train failed: {exc}")
        # Retrain the ML EXIT model on the freshly-materialised panel. It only closes
        # arm-cohort trades and is otherwise panel-first (IC-tracked), so a stale/
        # failed artifact only means the hand-built exit machinery is kept. Fail-soft.
        if settings.enable_eod_ml_exit_train:
            try:
                from src.analysis.ml_exit_dataset import eod_train_exit
                art = eod_train_exit()
                if art:
                    logger.info(f"[scheduler] EOD ml_exit train: {art['n_train']:,} rows "
                                f"(<= {art['train_max_date']})")
            except Exception as exc:
                logger.warning(f"[scheduler] EOD ml_exit train failed: {exc}")

    # Walk-forward and the ML retrains are independent of each other (both need
    # the replay-refreshed panel above; neither feeds the other) — run them in
    # parallel. Two workers, join before returning so the thread's lifetime
    # brackets the whole job for the single-flight guard.
    from concurrent.futures import ThreadPoolExecutor as _TPE
    with _TPE(max_workers=2, thread_name_prefix="eod") as _ex:
        for _f in (_ex.submit(_wf_step), _ex.submit(_ml_step)):
            try:
                _f.result()
            except Exception as exc:
                logger.warning(f"[scheduler] EOD parallel stage raised: {exc}")

    # NOTE the automatic refactor is deliberately NOT here — it runs on its own
    # nightly slot (`_maybe_start_nightly_rescore`, 02:00 ET by default) in a
    # background thread. It is a ~40-minute job, and this function runs INLINE
    # in the scheduler loop, so hosting it here would block every tick for its
    # duration.


def _alert(subject: str, body: str) -> None:
    """Operational alert email (fail-soft, gated by ``scheduler_alert_email``).
    A suspended machine can't alert in the moment — these fire on wake, which is
    still the difference between finding out at 19:10 and finding out never
    (observed 2026-06-30: a full trading day dark, 24 open positions unmanaged,
    no notification)."""
    if not settings.scheduler_alert_email:
        return
    try:
        from src.notifications.email_sender import send_alert
        send_alert(subject, body)
    except Exception as exc:
        logger.warning(f"[scheduler] alert email failed: {exc}")


# Crash alerts were the remaining silent-failure class: a tick that RAISES was
# only `logger.exception`'d (console + file), never emailed — so a persistently
# crashing pipeline (bad deploy, dependency break, a data-shape change, a report-
# template bug) could run a whole session dark with no notification, exactly like
# the pre-2026-06-30 missed-slot gap but for crashes instead of suspends. Throttle
# per exception TYPE so a repeating crash can't storm the inbox while a NEW kind of
# crash still gets through promptly.
_CRASH_ALERT_THROTTLE_S = 1800   # 30 min between alerts of the same crash type
_last_crash_alert: dict[str, float] = {}


def _alert_crash(context: str, exc: BaseException) -> None:
    import traceback as _tb
    key = type(exc).__name__
    now = _time_module.monotonic()
    if now - _last_crash_alert.get(key, 0.0) < _CRASH_ALERT_THROTTLE_S:
        return                                    # same crash type recently alerted
    _last_crash_alert[key] = now
    detail = str(exc) or type(exc).__name__       # blank-message guard (bare TimeoutError)
    tb = "".join(_tb.format_exception(type(exc), exc, exc.__traceback__))[-2500:]
    _alert(
        f"\U0001f6d1 LLM Trader: {context} crashed — {key}: {detail}",
        f"A scheduled {context} raised an unhandled exception and was skipped. The "
        f"loop keeps running, but this work did NOT complete this tick — open "
        f"positions may be unmarked/unmanaged and no report was sent.\n\n"
        f"{key}: {detail}\n\n{tb}\n\n"
        f"(Repeat {key} crashes are throttled to one alert per "
        f"{_CRASH_ALERT_THROTTLE_S // 60} min — check the log for the full run.)"
    )


def _missed_slots_between(start: datetime, end: datetime, slots: list[tuple[_time, str]],
                          grace: int) -> list[datetime]:
    """Valid slot boundaries that fell inside ``(start, end - grace]`` —
    the ticks a suspend/outage swallowed whole (per-slot session calendar,
    so Sunday-evening overnight slots count and Friday-evening ones don't)."""
    out: list[datetime] = []
    day = start.date()
    while day <= end.date():
        for t, kind in slots:
            slot_dt = datetime.combine(day, t)
            if start < slot_dt <= end - timedelta(seconds=grace) and _slot_is_valid(day, t, kind):
                out.append(slot_dt)
        day += timedelta(days=1)
    return out


def _slot_is_valid(day, t: _time, kind: str) -> bool:
    """Does this slot belong to a live session on this calendar day?

    rth/extended slots require the day itself to be a market day. An OVERNIGHT
    slot follows the venue's Sunday-night→Thursday-night schedule: the evening
    half (≥ 20:00) is valid when the NEXT day is a market day (Sunday evening
    ticks — it leads into Monday; Friday/holiday-eve evenings don't), the
    morning half (< 04:00) when TODAY is one.
    """
    if kind == "overnight":
        if t >= _time(20, 0):
            return is_market_day(day + timedelta(days=1))
        return is_market_day(day)
    return is_market_day(day)


def _current_slot(now_naive: datetime, slots: list[tuple[_time, str]]) -> tuple[datetime, str] | None:
    """Latest valid ``(slot boundary, kind)`` at/before `now`, or None.

    Validity is per-slot (``_slot_is_valid``): weekends and NYSE holidays yield
    None for rth/extended slots — no session on a closed market, so ticking
    would only burn LLM calls marking stale prices — while overnight slots
    follow the overnight venue's own calendar (Sunday evening IS valid).
    """
    today = now_naive.date()
    candidate: tuple[datetime, str] | None = None
    for t, kind in slots:
        slot_dt = datetime.combine(today, t)
        if slot_dt > now_naive:
            break
        if _slot_is_valid(today, t, kind):
            candidate = (slot_dt, kind)
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
    on_slots = [t for t, k in slots if k == "overnight"]

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
    if on_slots:
        _omode = (settings.overnight_hours_mode or "off").lower()
        _owhat = (
            "FULL TRADING ticks — ×10 modeled spread, "
            f"×{settings.overnight_size_multiplier:g} sizing, overnight-venue broker routing"
            if _omode == "trade"
            else "observation — full pipeline + persistence, no ledger/broker mutations"
        )
        logger.info(
            f"Overnight slots ({_omode}, Sun–Thu nights per the overnight venue calendar): "
            f"{', '.join(t.strftime('%H:%M') for t in on_slots)} ET — {_owhat}."
        )
    email_times = _email_slot_times()
    if settings.scheduler_email_every_tick:
        logger.info("Email: EVERY tick (scheduler_email_every_tick on).")
    elif email_times:
        slot_times = {t for t, _ in slots}
        logger.info("Email: report sends at "
                    f"{', '.join(t.strftime('%H:%M') for t in sorted(email_times))} ET.")
        missing = sorted(t for t in email_times if t not in slot_times)
        if missing:
            logger.warning(
                "[scheduler] scheduler_email_times not matching any tick slot — these "
                f"will NEVER email: {', '.join(t.strftime('%H:%M') for t in missing)}. "
                "Add them to the RTH grid / extended_windows (or fix the time)."
            )
    else:
        logger.info("Email: 16:00 close only (scheduler_email_times empty).")
    logger.info(
        f"Poll: {poll}s; misfire grace: {grace}s; keep-awake: {settings.scheduler_keep_awake}."
    )
    logger.info("Press Ctrl+C to stop.")

    last_run_slot: datetime | None = None
    alerted_slots: set[datetime] = set()
    prev_poll: datetime | None = None
    last_eod_date = None
    eod_time = _parse_hhmm(settings.eod_maintenance_time, _time(16, 20)) or _time(16, 20)
    rescore_time = _parse_hhmm(settings.rescore_time, _time(2, 0)) or _time(2, 0)
    last_rescore_date = None
    if settings.enable_eod_maintenance:
        logger.info(f"EOD maintenance at/after {eod_time.strftime('%H:%M')} ET: "
                    "forward-return cache warm + table retention (market days).")
    try:
        while True:
            now_naive = now_et().replace(tzinfo=None)

            # Suspend/resume detector: a poll-to-poll wall-clock jump far beyond
            # the cadence means the machine slept through the gap. Alert once per
            # resume when trading slots were swallowed (the missed-slot branch
            # below only sees the LATEST slot; the gap can hide a whole day).
            if prev_poll is not None:
                gap_s = (now_naive - prev_poll).total_seconds()
                if gap_s > max(3 * poll, 300):
                    missed = [m for m in _missed_slots_between(prev_poll, now_naive, slots, grace)
                              if m != last_run_slot and m not in alerted_slots]
                    logger.warning(
                        f"[scheduler] wall clock jumped {gap_s / 60:.0f} min "
                        f"(suspend/resume) — {len(missed)} tick slot(s) fell in the gap"
                    )
                    if missed:
                        alerted_slots.update(missed)
                        _alert(
                            f"⚠️ LLM Trader scheduler: {len(missed)} tick(s) missed "
                            f"(machine suspended ~{gap_s / 3600:.1f}h)",
                            "The scheduler resumed after a wall-clock gap of "
                            f"{gap_s / 60:.0f} minutes.\n\n"
                            "Missed tick slots (ET): "
                            f"{', '.join(m.strftime('%a %m-%d %H:%M') for m in missed)}\n\n"
                            "Open positions were not marked or managed during the gap. "
                            "A catch-up tick runs automatically if a trading session "
                            "is still live; otherwise the next scheduled slot resumes "
                            "normal operation. Keep the machine plugged in / awake."
                        )
            prev_poll = now_naive

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
                        # decision is authoritative for scheduled ticks
                        # (every slot when scheduler_email_every_tick, else
                        # the 16:00 closing report only).
                        run_pipeline(send_email=send_email, observe_only=observe,
                                     email_if_configured=False)
                    except Exception as exc:  # never let one tick kill the loop
                        logger.exception(f"[scheduler] tick raised: {exc}")
                        _alert_crash("pipeline tick", exc)   # log-only was silent to the operator
                else:
                    logger.warning(
                        f"[scheduler] slot {slot_dt.strftime('%H:%M')} ET missed by "
                        f"{lateness / 60:.1f} min (> {grace / 60:.0f} min grace) — skipping "
                        "(machine was suspended too long). Keep it plugged in / awake."
                    )
                    if slot_dt not in alerted_slots:
                        # Fresh-start case (the gap detector needs two polls to see a
                        # jump): the scheduler came up and found the latest slot long
                        # past — earlier slots today may have been missed too.
                        alerted_slots.add(slot_dt)
                        _alert(
                            f"⚠️ LLM Trader scheduler: slot {slot_dt.strftime('%H:%M')} ET "
                            f"missed by {lateness / 60:.0f} min",
                            f"The scheduler found slot {slot_dt.strftime('%Y-%m-%d %H:%M')} ET "
                            f"already {lateness / 60:.0f} minutes past on startup/resume — the "
                            "machine was suspended or the scheduler was down. Earlier slots "
                            "today may have been missed as well.\n\n"
                            "Open positions were not marked or managed during the outage. "
                            "A catch-up tick runs automatically if a trading session is "
                            "still live."
                        )
                    # Catch-up: manage positions late rather than not at all — but
                    # only while a session is live (marking prices on a closed
                    # market would burn LLM calls for nothing). The overnight
                    # session counts only when overnight trading is on AND the
                    # venue is actually open (Sun–Thu nights).
                    _sess = current_session()
                    _session_live = _sess in ("rth", "extended") or (
                        _sess == "overnight"
                        and (settings.overnight_hours_mode or "off").lower() == "trade"
                        and is_overnight_session_open()
                    )
                    if settings.scheduler_catchup_tick and _session_live:
                        observe, _ = _tick_plan(kind, slot_dt.time(), end_t)
                        logger.info(
                            f"[scheduler] CATCH-UP tick for missed "
                            f"{slot_dt.strftime('%H:%M')} ET slot (email=False)"
                        )
                        try:
                            run_pipeline(send_email=False, observe_only=observe,
                                         email_if_configured=False)
                        except Exception as exc:
                            logger.exception(f"[scheduler] catch-up tick raised: {exc}")
                            _alert_crash("catch-up tick", exc)
                last_run_slot = slot_dt  # mark even when skipped, so we don't retry this slot

            # End-of-day maintenance — once per market day past the trigger,
            # off the time-critical path (runs between ticks, after the close).
            # Nightly rescore — its own slot, NOT inside EOD maintenance: it is
            # a ~40-minute job and this loop is single-threaded, so running it
            # inline would block the overnight ticks it overlaps. Change
            # detection is sub-second, so an unchanged night is free.
            if _should_run_nightly_rescore(now_naive, last_rescore_date, rescore_time):
                last_rescore_date = now_naive.date()
                try:
                    _maybe_start_nightly_rescore()
                except Exception as exc:
                    logger.exception(f"[scheduler] nightly rescore trigger raised: {exc}")

            if _should_run_eod(now_naive, last_eod_date, eod_time):
                last_eod_date = now_naive.date()
                try:
                    _run_eod_maintenance()
                except Exception as exc:
                    logger.exception(f"[scheduler] EOD maintenance raised: {exc}")
                    _alert_crash("EOD maintenance", exc)

            _time_module.sleep(poll)
    except (KeyboardInterrupt, SystemExit):
        logger.info("Scheduler stopped.")
    finally:
        _release_keep_awake()
