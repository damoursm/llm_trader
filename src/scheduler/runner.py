"""APScheduler-based scheduler for running the pipeline during market hours."""

from apscheduler.schedulers.blocking import BlockingScheduler
from apscheduler.triggers.cron import CronTrigger
from loguru import logger
from src.pipeline import run_pipeline


def start_scheduler() -> None:
    """Start the scheduled pipeline runner."""
    scheduler = BlockingScheduler(timezone="America/New_York")

    # Daily pre-market run: 8:00 AM ET Mon-Fri
    scheduler.add_job(
        run_pipeline,
        CronTrigger(day_of_week="mon-fri", hour=8, minute=0),
        id="daily",
        name="Daily pre-market analysis",
        kwargs={"send_email": True},
    )

    logger.info("Scheduler started. Job: daily at 08:00 ET (Mon-Fri)")
    logger.info("Press Ctrl+C to stop.")

    try:
        scheduler.start()
    except (KeyboardInterrupt, SystemExit):
        logger.info("Scheduler stopped.")
