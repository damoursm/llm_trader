"""Options expiration calendar — pure date math, zero API calls.

Documented patterns with statistical edge:
  • OpEx week (Mon–Fri of the week containing the 3rd Friday):
    Market makers actively delta-hedge toward max pain, suppressing
    directional moves and pinning price near the max pain strike.
  • OpEx day itself (3rd Friday): strongest intraday pinning; big moves
    typically resolve after the market close once hedges expire.
  • Triple Witching (March, June, September, December OpEx):
    Stock options, stock index futures, AND stock index options all expire
    simultaneously. Significantly higher volume and pinning intensity than
    a standard monthly expiry.
  • Post-OpEx window (1–5 calendar days after expiry):
    The pin is released as hedges roll or expire. Directional moves tend
    to be more impulsive in the first week after expiry as institutional
    positioning resets.
"""

from datetime import date, timedelta
from src.models import OpExContext


def _third_friday(year: int, month: int) -> date:
    """Return the 3rd Friday of a given year/month."""
    first = date(year, month, 1)
    # weekday(): 0=Mon … 4=Fri
    days_to_first_friday = (4 - first.weekday()) % 7
    first_friday = first + timedelta(days=days_to_first_friday)
    return first_friday + timedelta(weeks=2)


def compute_opex_context(today: date | None = None) -> OpExContext:
    """Compute OpEx calendar context for the given date (defaults to today).

    No network calls — pure date arithmetic.
    """
    if today is None:
        today = date.today()

    year, month = today.year, today.month
    this_month_opex = _third_friday(year, month)

    # Determine which is the "next" and which is the "previous" expiry
    if today > this_month_opex:
        next_m = month + 1 if month < 12 else 1
        next_y = year if month < 12 else year + 1
        next_opex = _third_friday(next_y, next_m)
        prev_opex = this_month_opex
    else:
        next_opex = this_month_opex
        prev_m = month - 1 if month > 1 else 12
        prev_y = year if month > 1 else year - 1
        prev_opex = _third_friday(prev_y, prev_m)

    days_to_opex = (next_opex - today).days
    days_since_prev_opex = (today - prev_opex).days

    # OpEx week = Monday through Friday of the week containing the 3rd Friday
    opex_week_monday = next_opex - timedelta(days=next_opex.weekday())
    in_opex_week = opex_week_monday <= today <= next_opex

    # Triple Witching: quarterly expiries (Mar / Jun / Sep / Dec)
    is_triple_witching = next_opex.month in (3, 6, 9, 12)

    # Post-OpEx window: 1–5 calendar days after previous expiry
    in_post_opex_window = 0 < days_since_prev_opex <= 5

    # Signal classification
    if in_opex_week and days_to_opex == 0:
        signal = "OPEX_DAY"
    elif in_opex_week and days_to_opex <= 1:
        signal = "OPEX_IMMINENT"
    elif in_opex_week and is_triple_witching:
        signal = "TRIPLE_WITCHING_WEEK"
    elif in_opex_week:
        signal = "OPEX_WEEK"
    elif in_post_opex_window:
        signal = "POST_OPEX"
    else:
        signal = "NEUTRAL"

    # Human-readable summary
    tc_label = "TRIPLE WITCHING" if is_triple_witching else "standard monthly"
    if signal in ("OPEX_DAY", "OPEX_IMMINENT"):
        summary = (
            f"OpEx is {'today' if days_to_opex == 0 else 'tomorrow'} "
            f"({next_opex.strftime('%b %d')}, {tc_label}). "
            f"Max pain gravity is at peak strength — dealers are pinning price near the max pain strike. "
            f"Expect suppressed intraday moves and a directional release after the close."
        )
    elif signal in ("OPEX_WEEK", "TRIPLE_WITCHING_WEEK"):
        intensity = "triple-witching intensity" if is_triple_witching else "standard intensity"
        summary = (
            f"OpEx week ({tc_label}, {intensity}): {next_opex.strftime('%b %d')} is {days_to_opex}d away. "
            f"Market makers are delta-hedging continuously — price tends to pin near max pain. "
            f"Max pain gravity is amplified vs. non-OpEx weeks."
        )
    elif signal == "POST_OPEX":
        summary = (
            f"Post-OpEx window: {prev_opex.strftime('%b %d')} expiry was {days_since_prev_opex}d ago. "
            f"Hedges have expired or rolled. The pin has been released — directional moves are "
            f"more impulsive in the first week after expiry as institutional positioning resets. "
            f"Max pain gravity is weak until the new cycle establishes open interest."
        )
    else:
        summary = (
            f"Next OpEx: {next_opex.strftime('%b %d, %Y')} ({days_to_opex}d, {tc_label}). "
            f"No OpEx calendar effect today — max pain gravity operates at baseline level."
        )

    return OpExContext(
        today=today,
        next_opex=next_opex,
        prev_opex=prev_opex,
        days_to_opex=days_to_opex,
        days_since_prev_opex=days_since_prev_opex,
        opex_week_monday=opex_week_monday,
        in_opex_week=in_opex_week,
        is_triple_witching=is_triple_witching,
        in_post_opex_window=in_post_opex_window,
        signal=signal,
        summary=summary,
    )
