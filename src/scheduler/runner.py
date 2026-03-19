"""APScheduler-based scheduler for running the pipeline during market hours."""

from apscheduler.schedulers.blocking import BlockingScheduler
from apscheduler.triggers.cron import CronTrigger
from loguru import logger
from src.pipeline import run_pipeline


def start_scheduler() -> None:
    """Start the scheduled pipeline runner."""
    scheduler = BlockingScheduler(timezone="America/New_York")

    # Pre-market: 8:00 AM ET Mon-Fri
    scheduler.add_job(
        run_pipeline,
        CronTrigger(day_of_week="mon-fri", hour=8, minute=0),
        id="premarket",
        name="Pre-market analysis",
        kwargs={"send_email": True},
    )

    # Midday check: 12:00 PM ET Mon-Fri
    scheduler.add_job(
        run_pipeline,
        CronTrigger(day_of_week="mon-fri", hour=12, minute=0),
        id="midday",
        name="Midday analysis",
        kwargs={"send_email": True},
    )

    # After-close: 4:30 PM ET Mon-Fri
    scheduler.add_job(
        run_pipeline,
        CronTrigger(day_of_week="mon-fri", hour=16, minute=30),
        id="close",
        name="After-close analysis",
        kwargs={"send_email": True},
    )

    logger.info("Scheduler started. Jobs: pre-market 8:00, midday 12:00, after-close 16:30 ET")
    logger.info("Press Ctrl+C to stop.")

    try:
        scheduler.start()
    except (KeyboardInterrupt, SystemExit):
        logger.info("Scheduler stopped.")
