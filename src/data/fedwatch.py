"""
Fed Rate Expectations — market-implied Fed Funds path derived from T-bill spreads.

Methodology:
  The CME FedWatch tool derives probabilities from 30-Day Fed Funds Futures (ZQ contracts)
  which are not freely accessible without authentication. This module uses the next-best
  free equivalent: U.S. Treasury bill rates (DTB3, DTB6, DTB1YR) from FRED, which are
  highly correlated with expected average Fed Funds rates over each horizon.

  The spread between the current FF target midpoint and each T-bill rate directly measures
  how many basis points of cuts (or hikes) the market has priced in for that horizon:

    implied_cuts_Nm_bp = (ff_midpoint − DTB_N) × 100

  For the NEXT FOMC meeting specifically, we solve for the post-meeting implied rate from
  the 3-month T-bill using day-weighting:
    post_meeting_rate = (tbill_3m × 90 − days_before × current_rate) / days_after
    P(cut) = max(0, min(1, (current_rate − post_meeting_rate) / 0.25))

  FOMC meeting dates are parsed from the Federal Reserve's public calendar page.

Data sources (all free, no additional API keys):
  - FRED API (uses existing FRED_API_KEY):
      DFEDTARU — FF target upper bound
      DFEDTARL — FF target lower bound
      DTB3     — 3-Month T-Bill rate (90-day)
      DTB6     — 6-Month T-Bill rate (180-day)
      DTB1YR   — 1-Year T-Bill rate (365-day)
  - Federal Reserve: https://www.federalreserve.gov/monetarypolicy/fomccalendars.htm

Signal thresholds (based on 12-month implied cuts):
  ≥  75bp  STRONGLY_DOVISH   BULLISH   (3+ cuts priced in over 12m)
  ≥  25bp  DOVISH            BULLISH   (1–3 cuts priced in)
  ≥   8bp  MILDLY_DOVISH     NEUTRAL   (partial cut priced in)
  ±   8bp  NEUTRAL           NEUTRAL   (no meaningful rate-change expectation)
  ≤  −8bp  MILDLY_HAWKISH    BEARISH   (slight hike risk)
  ≤ −25bp  HAWKISH           BEARISH   (1+ hikes priced in)
  ≤ −75bp  STRONGLY_HAWKISH  BEARISH   (3+ hikes priced in)

Rate-trend signal (5-day change in 3-month T-bill):
  DTB3 fell  > 3bp  →  DOVISH_SHIFT   (market pricing in more cuts  → bullish)
  DTB3 rose  > 3bp  →  HAWKISH_SHIFT  (market pricing in fewer cuts → bearish)
  Change    ≤ 3bp   →  NEUTRAL
"""
from __future__ import annotations

import json
import math
import re
from datetime import date, timedelta
from pathlib import Path
from typing import Optional

import httpx
from loguru import logger

from config import settings
from src.models import FedWatchContext

_TIMEOUT  = 12
CACHE_DIR = Path("cache")

_FRED_BASE = "https://api.stlouisfed.org/fred/series/observations"
_FED_CAL_URL = "https://www.federalreserve.gov/monetarypolicy/fomccalendars.htm"

# ── Signal thresholds ────────────────────────────────────────────────────────

_STRONGLY_DOVISH_BP  =  75.0
_DOVISH_BP           =  25.0
_MILDLY_DOVISH_BP    =   8.0
_MILDLY_HAWKISH_BP   =  -8.0
_HAWKISH_BP          = -25.0
_STRONGLY_HAWKISH_BP = -75.0

_TREND_SHIFT_BP      =   3.0  # 3bp move in DTB3 = meaningful shift


# ─────────────────────────────────────────────────────────────────────────────
# Cache helpers
# ─────────────────────────────────────────────────────────────────────────────

def _cache_path() -> Path:
    return CACHE_DIR / f"fedwatch_{date.today().isoformat()}.json"


def _load_cache() -> Optional[FedWatchContext]:
    path = _cache_path()
    if not path.exists():
        return None
    try:
        ctx = FedWatchContext.model_validate(
            json.loads(path.read_text(encoding="utf-8"))
        )
        logger.info(
            f"[fedwatch] Loaded from cache — {ctx.signal}  "
            f"cuts_12m={ctx.implied_cuts_12m_bp:+.1f}bp  "
            f"P(cut next)={ctx.p_cut_next:.0%}"
        )
        return ctx
    except Exception as e:
        logger.warning(f"[fedwatch] Cache load failed: {e}")
        return None


def _save_cache(ctx: FedWatchContext) -> None:
    CACHE_DIR.mkdir(exist_ok=True)
    try:
        _cache_path().write_text(
            json.dumps(ctx.model_dump(mode="json"), indent=2, default=str),
            encoding="utf-8",
        )
    except Exception as e:
        logger.warning(f"[fedwatch] Cache save failed: {e}")


# ─────────────────────────────────────────────────────────────────────────────
# FRED fetch
# ─────────────────────────────────────────────────────────────────────────────

def _fred_series(series_id: str, api_key: str, limit: int = 10) -> list[dict]:
    """Return newest-first observations from FRED, or [] on failure."""
    try:
        resp = httpx.get(
            _FRED_BASE,
            params={
                "series_id":  series_id,
                "api_key":    api_key,
                "file_type":  "json",
                "sort_order": "desc",
                "limit":      limit,
            },
            timeout=_TIMEOUT,
        )
        resp.raise_for_status()
        return resp.json().get("observations", [])
    except Exception as e:
        logger.warning(f"[fedwatch] FRED {series_id} fetch failed: {e}")
        return []


def _latest_value(obs: list[dict]) -> Optional[float]:
    """Return the most recent non-missing FRED observation value."""
    for o in obs:
        if o.get("value", ".") != ".":
            try:
                return float(o["value"])
            except ValueError:
                pass
    return None


def _latest_N(obs: list[dict], n: int) -> list[float]:
    """Return the n most recent non-missing values, newest-first."""
    out = []
    for o in obs:
        if o.get("value", ".") != ".":
            try:
                out.append(float(o["value"]))
            except ValueError:
                pass
            if len(out) >= n:
                break
    return out


# ─────────────────────────────────────────────────────────────────────────────
# FOMC calendar
# ─────────────────────────────────────────────────────────────────────────────

_MONTHS = {
    "January": 1, "February": 2, "March": 3, "April": 4,
    "May": 5, "June": 6, "July": 7, "August": 8,
    "September": 9, "October": 10, "November": 11, "December": 12,
}


def _fetch_fomc_dates() -> list[date]:
    """
    Parse upcoming FOMC meeting end-dates from the Federal Reserve's calendar page.
    Returns dates sorted ascending; falls back to an empty list on error.
    """
    try:
        resp = httpx.get(
            _FED_CAL_URL,
            headers={"User-Agent": "Mozilla/5.0"},
            timeout=_TIMEOUT,
            follow_redirects=True,
        )
        resp.raise_for_status()
        text = resp.text

        year_pat  = re.compile(r'<h4><a id="\d+">(\d{4}) FOMC Meetings</a></h4>')
        month_pat = re.compile(r'fomc-meeting__month[^>]+><strong>([A-Za-z]+)</strong>')
        date_pat  = re.compile(r'fomc-meeting__date[^>]+>(\d+[-\u2013]\d+)')

        meetings: list[date] = []
        pos = 0
        current_year: Optional[int] = None
        today = date.today()

        while pos < len(text):
            ym = year_pat.search(text, pos)
            mm = month_pat.search(text, pos)
            dm = date_pat.search(text, pos)

            if ym and (not mm or ym.start() < mm.start()):
                current_year = int(ym.group(1))
                pos = ym.end()
            elif mm and dm and current_year:
                month_name = mm.group(1)
                date_range = dm.group(1)
                # Use the last day of the date range (meeting decision day)
                parts = re.split(r"[-\u2013]", date_range)
                day = int(parts[-1]) if parts else 0
                month_num = _MONTHS.get(month_name)
                if month_num and day:
                    try:
                        d = date(current_year, month_num, day)
                        if d >= today:
                            meetings.append(d)
                    except ValueError:
                        pass
                pos = max(mm.end(), dm.end())
            else:
                break

        meetings.sort()
        logger.debug(f"[fedwatch] Parsed {len(meetings)} upcoming FOMC dates")
        return meetings
    except Exception as e:
        logger.warning(f"[fedwatch] FOMC calendar fetch failed: {e}")
        return []


# ─────────────────────────────────────────────────────────────────────────────
# Probability model
# ─────────────────────────────────────────────────────────────────────────────

def _p_cut_from_tbill(
    ff_mid: float,
    tbill_3m: float,
    days_to_meeting: int,
    window_days: int = 90,
) -> float:
    """
    Derive P(25bp cut at the next FOMC meeting) from the 3-month T-bill rate.

    The 3-month T-bill rate approximates the weighted average expected FF rate
    over the next `window_days` days:
        tbill_3m ≈ (days_before × current_rate + days_after × post_rate) / window_days

    Solving for post_rate and converting to probability:
        P(cut) = max(0, min(1, (ff_mid − post_rate) / 0.25))
    """
    if days_to_meeting <= 0:
        days_to_meeting = 1
    if days_to_meeting >= window_days:
        # Meeting beyond the T-bill window — use simpler gap-based estimate
        gap_pct = ff_mid - tbill_3m
        return max(0.0, min(1.0, gap_pct / 0.25))

    days_before = days_to_meeting
    days_after  = window_days - days_before

    # Isolate implied post-meeting rate
    post_rate = (tbill_3m * window_days - days_before * ff_mid) / days_after

    p_cut  = max(0.0, min(1.0, (ff_mid - post_rate) / 0.25))
    p_hike = max(0.0, min(1.0, (post_rate - ff_mid) / 0.25))
    return p_cut  # caller computes p_hike separately


def _p_hike_from_tbill(
    ff_mid: float,
    tbill_3m: float,
    days_to_meeting: int,
    window_days: int = 90,
) -> float:
    if days_to_meeting <= 0:
        days_to_meeting = 1
    if days_to_meeting >= window_days:
        gap_pct = tbill_3m - ff_mid
        return max(0.0, min(1.0, gap_pct / 0.25))

    days_after = window_days - days_to_meeting
    post_rate  = (tbill_3m * window_days - days_to_meeting * ff_mid) / days_after
    return max(0.0, min(1.0, (post_rate - ff_mid) / 0.25))


def _classify(implied_cuts_12m_bp: float) -> tuple[str, str]:
    """Return (signal, direction) based on 12-month implied cuts."""
    if implied_cuts_12m_bp >= _STRONGLY_DOVISH_BP:
        return "STRONGLY_DOVISH",   "BULLISH"
    if implied_cuts_12m_bp >= _DOVISH_BP:
        return "DOVISH",            "BULLISH"
    if implied_cuts_12m_bp >= _MILDLY_DOVISH_BP:
        return "MILDLY_DOVISH",     "NEUTRAL"
    if implied_cuts_12m_bp > _MILDLY_HAWKISH_BP:
        return "NEUTRAL",           "NEUTRAL"
    if implied_cuts_12m_bp > _HAWKISH_BP:
        return "MILDLY_HAWKISH",    "BEARISH"
    if implied_cuts_12m_bp > _STRONGLY_HAWKISH_BP:
        return "HAWKISH",           "BEARISH"
    return "STRONGLY_HAWKISH", "BEARISH"


def _rate_trend(tbill_3m_now: float, tbill_3m_5d: Optional[float]) -> str:
    if tbill_3m_5d is None:
        return "NEUTRAL"
    change_bp = (tbill_3m_5d - tbill_3m_now) * 100  # positive = rates fell = more cuts expected
    if change_bp >= _TREND_SHIFT_BP:
        return "DOVISH_SHIFT"
    if change_bp <= -_TREND_SHIFT_BP:
        return "HAWKISH_SHIFT"
    return "NEUTRAL"


# ─────────────────────────────────────────────────────────────────────────────
# Public entry point
# ─────────────────────────────────────────────────────────────────────────────

def fetch_fedwatch_context() -> Optional[FedWatchContext]:
    """
    Derive market-implied Fed rate expectations from FRED T-bill spreads + FOMC calendar.

    Returns FedWatchContext or None if FRED key is missing or all fetches fail.
    Cached daily.
    """
    cached = _load_cache()
    if cached is not None:
        return cached

    api_key = settings.fred_api_key
    if not api_key:
        logger.debug("[fedwatch] FRED_API_KEY not set — skipping")
        return None

    logger.info("[fedwatch] Fetching T-bill rates and FOMC calendar...")

    # ── Fetch FRED series ─────────────────────────────────────────────────────
    obs_upper  = _fred_series("DFEDTARU", api_key, 3)
    obs_lower  = _fred_series("DFEDTARL", api_key, 3)
    obs_tbill3  = _fred_series("DTB3",    api_key, 10)
    obs_tbill6  = _fred_series("DTB6",    api_key, 10)
    obs_tbill12 = _fred_series("DTB1YR",  api_key, 10)

    ff_upper = _latest_value(obs_upper)
    ff_lower = _latest_value(obs_lower)
    if ff_upper is None or ff_lower is None:
        logger.warning("[fedwatch] Could not fetch FF target range — skipping")
        return None

    ff_mid    = (ff_upper + ff_lower) / 2.0
    tbill_3m  = _latest_value(obs_tbill3)
    tbill_6m  = _latest_value(obs_tbill6)
    tbill_12m = _latest_value(obs_tbill12)

    if tbill_3m is None and tbill_12m is None:
        logger.warning("[fedwatch] No T-bill data available — skipping")
        return None

    # ── Historical T-bill for trend ───────────────────────────────────────────
    vals_3m = _latest_N(obs_tbill3, 8)
    tbill_3m_5d_ago = vals_3m[5] if len(vals_3m) > 5 else None

    # ── FOMC calendar ─────────────────────────────────────────────────────────
    fomc_dates     = _fetch_fomc_dates()
    next_meeting   = fomc_dates[0] if fomc_dates else None
    days_to_next   = (next_meeting - date.today()).days if next_meeting else None

    # ── Compute implied cuts ──────────────────────────────────────────────────
    cuts_3m  = round((ff_mid - tbill_3m)  * 100, 1) if tbill_3m  is not None else 0.0
    cuts_6m  = round((ff_mid - tbill_6m)  * 100, 1) if tbill_6m  is not None else 0.0
    cuts_12m = round((ff_mid - tbill_12m) * 100, 1) if tbill_12m is not None else 0.0

    # ── Next-meeting probabilities ────────────────────────────────────────────
    if tbill_3m is not None and days_to_next is not None:
        p_cut  = round(_p_cut_from_tbill(ff_mid, tbill_3m, days_to_next), 3)
        p_hike = round(_p_hike_from_tbill(ff_mid, tbill_3m, days_to_next), 3)
        p_hold = round(max(0.0, 1.0 - p_cut - p_hike), 3)
    else:
        p_cut, p_hold, p_hike = 0.0, 1.0, 0.0

    # ── Signal ────────────────────────────────────────────────────────────────
    # Use 12m implied cuts as the primary classifier; fall back to 3m or 6m
    primary_cuts = cuts_12m if tbill_12m is not None else (cuts_6m if tbill_6m is not None else cuts_3m)
    signal, direction = _classify(primary_cuts)
    trend = _rate_trend(tbill_3m or ff_mid, tbill_3m_5d_ago)

    # ── Summary ───────────────────────────────────────────────────────────────
    signal_desc = {
        "STRONGLY_DOVISH":   f"{primary_cuts:+.0f}bp of cuts priced in over 12m — significant easing expected.",
        "DOVISH":            f"{primary_cuts:+.0f}bp of cuts priced in over 12m — market expects 1–3 rate cuts.",
        "MILDLY_DOVISH":     f"{primary_cuts:+.0f}bp of cuts priced in — partial cut expected; mild easing bias.",
        "NEUTRAL":           "Market expects rates roughly unchanged over the next 12 months.",
        "MILDLY_HAWKISH":    f"{abs(primary_cuts):.0f}bp of hikes priced in — mild tightening bias.",
        "HAWKISH":           f"{abs(primary_cuts):.0f}bp of hikes priced in — market expects 1+ rate hike(s).",
        "STRONGLY_HAWKISH":  f"{abs(primary_cuts):.0f}bp of hikes priced in — significant tightening expected.",
    }.get(signal, "")

    trend_desc = {
        "DOVISH_SHIFT":  f" T-bills fell {abs(tbill_3m - tbill_3m_5d_ago)*100:.1f}bp this week → dovish repricing.",
        "HAWKISH_SHIFT": f" T-bills rose {abs(tbill_3m - tbill_3m_5d_ago)*100:.1f}bp this week → hawkish repricing.",
        "NEUTRAL": "",
    }.get(trend, "") if tbill_3m_5d_ago is not None and tbill_3m is not None else ""

    next_str = (
        f" Next FOMC: {next_meeting} ({days_to_next}d away) — "
        f"P(cut)={p_cut:.0%}, P(hold)={p_hold:.0%}, P(hike)={p_hike:.0%}."
        if next_meeting else ""
    )

    summary = (
        f"Fed Rate Expectations: FF target {ff_lower:.2f}–{ff_upper:.2f}%, "
        f"12m implied cuts {primary_cuts:+.0f}bp ({signal}). "
        f"{signal_desc}{trend_desc}{next_str}"
    )

    ctx = FedWatchContext(
        ff_upper=round(ff_upper, 4),
        ff_lower=round(ff_lower, 4),
        ff_midpoint=round(ff_mid, 4),
        tbill_3m=round(tbill_3m, 4) if tbill_3m else None,
        tbill_6m=round(tbill_6m, 4) if tbill_6m else None,
        tbill_12m=round(tbill_12m, 4) if tbill_12m else None,
        implied_cuts_3m_bp=cuts_3m,
        implied_cuts_6m_bp=cuts_6m,
        implied_cuts_12m_bp=cuts_12m,
        next_meeting=next_meeting,
        days_to_next_meeting=days_to_next,
        p_cut_next=p_cut,
        p_hold_next=p_hold,
        p_hike_next=p_hike,
        tbill_3m_5d_ago=round(tbill_3m_5d_ago, 4) if tbill_3m_5d_ago else None,
        rate_trend=trend,
        signal=signal,
        direction=direction,
        report_date=date.today(),
        summary=summary,
    )
    _save_cache(ctx)

    logger.info(
        f"[fedwatch] FF={ff_lower:.2f}–{ff_upper:.2f}%  "
        f"implied_cuts_12m={cuts_12m:+.1f}bp  {signal}  {direction}  "
        f"P(cut_next)={p_cut:.0%}  trend={trend}"
    )
    return ctx
