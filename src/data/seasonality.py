"""Seasonal calendar effects — pure date math, zero API calls.

Four documented patterns with measurable statistical edge:

  1. End-of-month rebalancing (last 3 / first 3 calendar days)
     Pension funds, 401(k) plans, and balanced funds rebalance monthly.
     After equity gains → buy bonds. After equity losses → buy equities.
     Net effect: systematic equity buying pressure in the final and first days
     of each month as money flows back toward target allocations.

  2. Turn-of-quarter window dressing (last 5 / first 5 calendar days of quarter)
     Fund managers buy YTD winners and sell losers in the final days of the quarter
     to improve the look of their quarterly statements. Large-cap, broad market ETFs,
     and index leaders see the most cosmetic buying. New-quarter allocation resets
     bring fresh institutional capital into the market in the first days of Q1/Q2/Q3/Q4.

  3. January effect (first 15 days of January)
     Tax-loss harvesting selling peaks in mid-to-late December. Positions sold at a
     loss are re-bought in January, historically producing a small-cap premium (IWM >
     SPY) in the first two weeks of the year. Most pronounced in beaten-down names.

  4. Monthly seasonal bias (full-month historical averages)
     Based on ~95 years of S&P 500 data (various academic sources, Hirsch Stock Trader's
     Almanac, LPL Research, Bloomberg):
     - April: strongest month on average (+2-3%)
     - November–January: best 3-month seasonal window
     - September: worst month on average (-1% to -1.5%)
     - "Sell in May and go away": May–October underperforms November–April by ~6-7% annualised
     - October: historically volatile (crash risk) but often marks important market bottoms
"""

import calendar
from datetime import date
from typing import List
from src.models import SeasonalEffect, SeasonalityContext


# ── Monthly historical bias ───────────────────────────────────────────────────
# (direction, signal_name, description)
_MONTHLY_BIAS: dict = {
    1:  ("BULLISH", "JANUARY_STRONG",
         "January: historically positive; tax-loss harvesting rebound + new-year institutional flows. "
         "Best start to the 'best 6-month' seasonal window (Nov–Apr)."),
    2:  ("NEUTRAL", "FEBRUARY_NEUTRAL",
         "February: mixed; earnings season mid-cycle, limited macro calendar. "
         "Often gives back some January gains."),
    3:  ("BULLISH", "MARCH_STRONG",
         "March: typically ends positively driven by quarter-end window dressing and "
         "institutional rebalancing into Q1 close."),
    4:  ("BULLISH", "APRIL_STRONG",
         "April: historically the strongest month of the year for US equities (avg +2-3%). "
         "Earnings season kicks off, end of tax season, and fresh April flows."),
    5:  ("BEARISH", "SELL_IN_MAY",
         "'Sell in May and go away': May begins the historically weaker May–October period. "
         "The Nov–Apr window outperforms May–Oct by ~6-7pp annualised."),
    6:  ("NEUTRAL", "JUNE_MIXED",
         "June: mixed; mid-year portfolio rebalancing at Q2 close. Triple Witching. "
         "Often ends the period of spring strength."),
    7:  ("BULLISH", "JULY_STRONG",
         "July: historically strong; first month of Q3 brings fresh institutional capital. "
         "Earnings season (major tech) often provides positive catalysts."),
    8:  ("NEUTRAL", "AUGUST_NEUTRAL",
         "August: thin liquidity, many institutional traders on vacation. "
         "Large moves can be erratic and hard to sustain. Avoid leveraged directional bets."),
    9:  ("BEARISH", "SEPTEMBER_WEAK",
         "September: historically the weakest month for US equities going back ~95 years (avg -1% to -1.5%). "
         "End of fiscal year for many mutual funds (Oct 31) → tax-loss selling starts early."),
    10: ("NEUTRAL", "OCTOBER_VOLATILE",
         "October: 'crash month' historically (1929, 1987, 2008) but also frequently marks major market bottoms. "
         "High volatility and reversals — do not over-extrapolate trends from September."),
    11: ("BULLISH", "NOVEMBER_STRONG",
         "November: start of the seasonally strongest 6-month period (Nov–Apr). "
         "Historically the 2nd strongest month. Post-election-year strength often amplified."),
    12: ("BULLISH", "DECEMBER_STRONG",
         "December: Santa Claus rally seasonality + year-end window dressing by fund managers. "
         "Institutional buying of YTD winners to lock in performance. Typically strong mid-month."),
}

# Months that are quarter-ends
_QUARTER_END_MONTHS   = frozenset({3, 6, 9, 12})
# Months that are quarter-starts
_QUARTER_START_MONTHS = frozenset({1, 4, 7, 10})


def compute_seasonality_context(today: date | None = None) -> SeasonalityContext:
    """Compute seasonal calendar context for the given date (defaults to today).

    No network calls — pure date arithmetic.
    """
    if today is None:
        today = date.today()

    year  = today.year
    month = today.month
    day   = today.day

    quarter    = (month - 1) // 3 + 1
    month_name = today.strftime("%B")
    last_day   = calendar.monthrange(year, month)[1]

    days_until_month_end = last_day - day   # 0 = last day
    days_into_month      = day - 1          # 0 = first day

    # ── Window flags ──────────────────────────────────────────────────────────
    in_month_end_window   = days_until_month_end <= 2   # last 3 calendar days
    in_month_start_window = days_into_month      <= 2   # first 3 calendar days

    in_quarter_end_window   = month in _QUARTER_END_MONTHS   and days_until_month_end <= 4
    in_quarter_start_window = month in _QUARTER_START_MONTHS and days_into_month      <= 4

    in_january_effect = month == 1 and day <= 15

    # Fiscal-year-end months (June 30 + Dec 31): window dressing is more intense
    is_fiscal_year_end = month in (6, 12) and in_quarter_end_window

    # ── Monthly bias ─────────────────────────────────────────────────────────
    monthly_direction, monthly_signal, monthly_description = _MONTHLY_BIAS[month]

    # ── Active calendar effects ───────────────────────────────────────────────
    active_effects: List[SeasonalEffect] = []

    if in_quarter_end_window:
        fy_note = " Fiscal-year-end intensity — many fund managers close their books June 30 / Dec 31." if is_fiscal_year_end else ""
        active_effects.append(SeasonalEffect(
            name="Quarter-End Window Dressing",
            direction="BULLISH",
            assets_affected="broad market ETFs (SPY, QQQ), YTD leaders, large-cap quality",
            description=(
                f"Q{quarter} ends in {days_until_month_end + 1}d. Fund managers buy recent winners "
                f"and sell laggards to improve quarterly statements.{fy_note} "
                f"Net effect: cosmetic bid on index ETFs and YTD winners; selling pressure on YTD losers."
            ),
        ))

    if in_quarter_start_window and not in_quarter_end_window:
        active_effects.append(SeasonalEffect(
            name="New-Quarter Institutional Flows",
            direction="BULLISH",
            assets_affected="broad market ETFs, sector rotation winners",
            description=(
                f"Q{quarter} opened {days_into_month + 1}d ago. Fresh quarterly pension and "
                f"fund allocations are being deployed. Equities typically see a systematic "
                f"institutional bid in the first 3-5 days of a new quarter."
            ),
        ))

    if in_month_end_window and not in_quarter_end_window:
        active_effects.append(SeasonalEffect(
            name="Month-End Rebalancing",
            direction="BULLISH",
            assets_affected="equities broadly; especially if month had equity losses",
            description=(
                f"Month ends in {days_until_month_end + 1}d. Pension funds and balanced funds "
                f"rebalance to restore target allocations. After an equity down-month: systematic "
                f"buying of equities. After an equity up-month: selling equities, buying bonds."
            ),
        ))

    if in_month_start_window and not in_quarter_start_window:
        active_effects.append(SeasonalEffect(
            name="Month-Start Flows",
            direction="BULLISH",
            assets_affected="equities broadly",
            description=(
                f"First {days_into_month + 1}d of {month_name}. Fresh 401(k)/payroll contributions "
                f"and mutual fund inflows are deployed. Systematic buying bias, particularly for "
                f"broad market index funds."
            ),
        ))

    if in_january_effect:
        active_effects.append(SeasonalEffect(
            name="January Effect",
            direction="BULLISH",
            assets_affected="small-caps (IWM, VTWO), beaten-down names from prior year",
            description=(
                f"January effect window (day {day}/15). Tax-loss harvesting selling from December "
                f"has passed; oversold names are being re-bought. Small-cap ETFs (IWM) historically "
                f"outperform large-caps (SPY) in the first two weeks of January."
            ),
        ))

    # ── Composite signal ──────────────────────────────────────────────────────
    # Count bullish vs bearish effects
    bullish_count  = sum(1 for e in active_effects if e.direction == "BULLISH")
    bearish_count  = sum(1 for e in active_effects if e.direction == "BEARISH")
    net_effects    = bullish_count - bearish_count

    monthly_score = {"BULLISH": +1, "NEUTRAL": 0, "BEARISH": -1}[monthly_direction]
    total_score   = monthly_score + net_effects

    if total_score >= 2:
        composite_signal    = "STRONG_TAILWIND"
        composite_direction = "BULLISH"
    elif total_score == 1:
        composite_signal    = "TAILWIND"
        composite_direction = "BULLISH"
    elif total_score == 0:
        composite_signal    = "NEUTRAL"
        composite_direction = "NEUTRAL"
    elif total_score == -1:
        composite_signal    = "HEADWIND"
        composite_direction = "BEARISH"
    else:
        composite_signal    = "STRONG_HEADWIND"
        composite_direction = "BEARISH"

    # ── Summary ───────────────────────────────────────────────────────────────
    effect_str = ""
    if active_effects:
        names = " + ".join(e.name for e in active_effects)
        effect_str = f" Active window effects: {names}."

    summary = (
        f"{month_name} seasonal bias: {monthly_direction} ({monthly_signal}). "
        f"{monthly_description}"
        f"{effect_str}"
    )

    return SeasonalityContext(
        today=today,
        month=month,
        month_name=month_name,
        quarter=quarter,
        monthly_bias=monthly_direction,
        monthly_signal=monthly_signal,
        monthly_description=monthly_description,
        in_month_end_window=in_month_end_window,
        in_month_start_window=in_month_start_window,
        in_quarter_end_window=in_quarter_end_window,
        in_quarter_start_window=in_quarter_start_window,
        in_january_effect=in_january_effect,
        is_fiscal_year_end=is_fiscal_year_end,
        active_effects=active_effects,
        composite_signal=composite_signal,
        composite_direction=composite_direction,
        summary=summary,
    )
