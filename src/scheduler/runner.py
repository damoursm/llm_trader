"""APScheduler-based runner — runs the pipeline every 30 min during market hours.

The pipeline operates on live prices and completed-only daily bars, so it is run
intraday on a 30-minute cadence (09:30–16:00 ET, Mon-Fri) rather than once
pre-market. The cron fires at :00 and :30 of hours 9–16; a window guard restricts
actual runs to the configured session window so the pre-open (09:00) and
post-close (16:30) ticks are skipped. Only the closing tick sends the daily email.
"""

from datetime import time as _time

from apscheduler.schedulers.blocking import BlockingScheduler
from apscheduler.triggers.cron import CronTrigger
from loguru import logger

from config import settings
from src.pipeline import run_pipeline
from src.utils import now_et


def _parse_hhmm(s: str, default: _time) -> _time:
    try:
        h, m = str(s).split(":")
        return _time(int(h), int(m))
    except Exception:
        return default


def _run_tick() -> None:
    """Run one intraday tick if inside the regular session window."""
    start = _parse_hhmm(settings.intraday_session_start, _time(9, 30))
    end = _parse_hhmm(settings.intraday_session_end, _time(16, 0))
    t = now_et().time()
    if t < start or t > end:
        logger.info(
            f"[scheduler] {t.strftime('%H:%M')} ET outside "
            f"{start.strftime('%H:%M')}–{end.strftime('%H:%M')} — skipping tick"
        )
        return
    send_email = t >= end  # daily wrap-up email on the closing tick only
    logger.info(f"[scheduler] Intraday tick at {t.strftime('%H:%M')} ET (email={send_email})")
    run_pipeline(send_email=send_email)


def start_scheduler() -> None:
    """Start the intraday runner: every 30 min, 09:30–16:00 ET, Mon-Fri."""
    scheduler = BlockingScheduler(timezone="America/New_York")
    scheduler.add_job(
        _run_tick,
        CronTrigger(day_of_week="mon-fri", hour="9-16", minute="0,30"),
        id="intraday",
        name="Intraday 30-min analysis (09:30–16:00 ET)",
        max_instances=1,   # never overlap — a long run coalesces instead of stacking
        coalesce=True,
    )

    logger.info("Scheduler started. Job: every 30 min, 09:30–16:00 ET (Mon-Fri).")
    logger.info("Press Ctrl+C to stop.")

    try:
        scheduler.start()
    except (KeyboardInterrupt, SystemExit):
        logger.info("Scheduler stopped.")
