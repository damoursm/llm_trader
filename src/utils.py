"""Shared utilities."""

from datetime import datetime
from zoneinfo import ZoneInfo

ET = ZoneInfo("America/New_York")


def now_et() -> datetime:
    """Return the current datetime in US/Eastern (handles EST/EDT automatically)."""
    return datetime.now(ET)


def fmt_et(dt: datetime, include_date: bool = True) -> str:
    """
    Format a datetime in Eastern time.
    Converts from any timezone (including UTC) before formatting.
    """
    eastern = dt.astimezone(ET)
    tz_label = eastern.strftime("%Z")   # "EST" or "EDT"
    if include_date:
        return eastern.strftime(f"%Y-%m-%d %H:%M {tz_label}")
    return eastern.strftime(f"%H:%M {tz_label}")
