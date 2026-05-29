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


def fmt_iso_et(iso_str: str, include_date: bool = True) -> str:
    """Parse an ISO 8601 string and format it as Eastern time.

    Trade dicts store ``entry_datetime`` / ``exit_datetime`` /
    ``decision_datetime`` as ISO 8601 strings (so they survive JSON
    round-trips cleanly). This helper turns one into the same human-friendly
    Eastern-time format ``fmt_et`` produces for live datetimes, so the email
    can show entry/exit time alongside the date. Returns an empty string for
    None / unparseable input so the template stays simple.
    """
    if not iso_str:
        return ""
    try:
        dt = datetime.fromisoformat(iso_str)
    except (TypeError, ValueError):
        return ""
    return fmt_et(dt, include_date=include_date)
