"""
Earnings calendar and EPS surprise data.

Two outputs:
1. fetch_earnings_surprises() → List[NewsArticle]
   Recent EPS beat/miss events (within lookback_days), injected into the articles list
   and scored by the DeepSeek sentiment pipeline as per-ticker catalysts.

2. fetch_earnings_context() → Optional[EarningsContext]
   Upcoming earnings dates within the next N days, passed as structured context
   to Claude so it can flag pre-earnings caution or IV-expansion opportunity.

Sources:
  - yfinance ticker.earnings_dates  (surprises + upcoming estimates, no key)
  - Alpha Vantage EARNINGS_CALENDAR (supplement for upcoming, requires API key)

Both outputs cached daily.
"""

import json
import math
import time
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import List, Optional

import yfinance as yf
from loguru import logger

from config import settings
from src.models import EarningsContext, EarningsEvent, NewsArticle

CACHE_DIR = Path("cache")
_REQUEST_DELAY = 0.35   # yfinance rate-limit buffer

_MIN_SURPRISE_PCT = 5.0       # Only generate article if |surprise| exceeds this %
_UPCOMING_WINDOW_DAYS = 14    # Default look-ahead for upcoming earnings


# ─────────────────────────────────────────────────────────────────────────────
# Cache helpers
# ─────────────────────────────────────────────────────────────────────────────

def _day_key() -> str:
    return date.today().isoformat()


def _surprises_path() -> Path:
    return CACHE_DIR / f"earnings_surprises_{_day_key()}.json"


def _calendar_path() -> Path:
    return CACHE_DIR / f"earnings_calendar_{_day_key()}.json"


def _load_surprises_cache() -> Optional[List[NewsArticle]]:
    path = _surprises_path()
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        articles = [NewsArticle.model_validate(a) for a in data]
        logger.info(f"[earnings] Loaded {len(articles)} EPS surprise articles from cache")
        return articles
    except Exception as e:
        logger.warning(f"[earnings] Surprise cache load failed: {e}")
        return None


def _save_surprises_cache(articles: List[NewsArticle]) -> None:
    CACHE_DIR.mkdir(exist_ok=True)
    try:
        _surprises_path().write_text(
            json.dumps([a.model_dump(mode="json") for a in articles], indent=2, default=str),
            encoding="utf-8",
        )
    except Exception as e:
        logger.warning(f"[earnings] Surprise cache save failed: {e}")


def _load_calendar_cache() -> Optional[EarningsContext]:
    path = _calendar_path()
    if not path.exists():
        return None
    try:
        ctx = EarningsContext.model_validate(
            json.loads(path.read_text(encoding="utf-8"))
        )
        logger.info(f"[earnings] Loaded earnings calendar from cache ({len(ctx.upcoming)} events)")
        return ctx
    except Exception as e:
        logger.warning(f"[earnings] Calendar cache load failed: {e}")
        return None


def _save_calendar_cache(ctx: EarningsContext) -> None:
    CACHE_DIR.mkdir(exist_ok=True)
    try:
        _calendar_path().write_text(
            json.dumps(ctx.model_dump(mode="json"), indent=2, default=str),
            encoding="utf-8",
        )
    except Exception as e:
        logger.warning(f"[earnings] Calendar cache save failed: {e}")


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _safe_float(val) -> Optional[float]:
    """Return float or None if val is NaN/None/non-numeric."""
    if val is None:
        return None
    try:
        f = float(val)
        return None if math.isnan(f) else f
    except (TypeError, ValueError):
        return None


def _quarter_label(d: date) -> str:
    q = (d.month - 1) // 3 + 1
    return f"Q{q} {d.year}"


def _build_surprise_article(
    ticker: str,
    report_date: date,
    actual_eps: float,
    estimated_eps: float,
    surprise_pct: float,
) -> NewsArticle:
    beat = surprise_pct > 0
    mag  = "significantly" if abs(surprise_pct) > 15 else "modestly"
    verb = "beat" if beat else "missed"
    direction = "bullish" if beat else "bearish"

    title = (
        f"{ticker} {mag} {verb} EPS estimate by {abs(surprise_pct):.1f}% "
        f"({_quarter_label(report_date)} results)"
    )
    summary = (
        f"{ticker} reported {_quarter_label(report_date)} EPS of ${actual_eps:.2f} vs "
        f"consensus estimate of ${estimated_eps:.2f} — "
        f"a {'+' if beat else ''}{surprise_pct:.1f}% surprise. "
        + (
            "Earnings beats historically drive near-term price appreciation, "
            "positive estimate revisions, and analyst upgrades. "
            if beat else
            "Earnings misses typically trigger sell-offs, downward estimate revisions, "
            "and multiple compression. "
        )
        + f"This is a {direction} catalyst for {ticker}."
    )
    return NewsArticle(
        title=title,
        summary=summary,
        url=f"https://finance.yahoo.com/quote/{ticker}/financials/",
        source="Earnings/EPS",
        published_at=datetime.now(timezone.utc),
    )


# ─────────────────────────────────────────────────────────────────────────────
# EPS surprises  (→ List[NewsArticle])
# ─────────────────────────────────────────────────────────────────────────────

def fetch_earnings_surprises(
    tickers: List[str],
    lookback_days: int = 90,
    min_surprise_pct: float = _MIN_SURPRISE_PCT,
) -> List[NewsArticle]:
    """
    Fetch recent EPS beat/miss data for each ticker and return as NewsArticle objects.
    Only generates an article when |surprise| ≥ min_surprise_pct and the report is
    within lookback_days of today.

    Cached daily.
    """
    cached = _load_surprises_cache()
    if cached is not None:
        return cached

    if not settings.enable_fetch_data:
        logger.debug("[earnings] ENABLE_FETCH_DATA=false — skipping yfinance surprises fetch")
        return []

    cutoff = date.today() - timedelta(days=lookback_days)
    articles: List[NewsArticle] = []

    for sym in tickers:
        try:
            t = yf.Ticker(sym)
            ed = t.earnings_dates   # DataFrame: index=DatetimeIndex, cols=[EPS Estimate, Reported EPS, Surprise(%)]

            if ed is None or ed.empty:
                time.sleep(_REQUEST_DELAY)
                continue

            # Filter to rows where both estimate and actual are present (= past earnings)
            reported = ed.dropna(subset=["EPS Estimate", "Reported EPS"])
            if reported.empty:
                time.sleep(_REQUEST_DELAY)
                continue

            # Filter to within lookback window
            in_window = reported[reported.index.date >= cutoff]
            if in_window.empty:
                time.sleep(_REQUEST_DELAY)
                continue

            # Most recent report (index sorted descending in yfinance)
            row         = in_window.iloc[0]
            report_date = in_window.index[0].date()
            actual      = _safe_float(row.get("Reported EPS"))
            estimate    = _safe_float(row.get("EPS Estimate"))

            if actual is None or estimate is None:
                time.sleep(_REQUEST_DELAY)
                continue

            # Use pre-computed Surprise(%) if available, else calculate
            surprise = _safe_float(row.get("Surprise(%)"))
            if surprise is None:
                if abs(estimate) < 0.01:
                    time.sleep(_REQUEST_DELAY)
                    continue
                surprise = (actual - estimate) / abs(estimate) * 100

            if abs(surprise) >= min_surprise_pct:
                article = _build_surprise_article(sym, report_date, actual, estimate, surprise)
                articles.append(article)
                logger.info(
                    f"[earnings] {sym}: EPS {'+' if surprise > 0 else ''}{surprise:.1f}% "
                    f"surprise on {report_date}"
                )

            time.sleep(_REQUEST_DELAY)

        except Exception as e:
            logger.debug(f"[earnings] {sym} surprise fetch failed: {e}")
            time.sleep(_REQUEST_DELAY)

    logger.info(f"[earnings] {len(articles)} EPS surprise article(s) from {len(tickers)} tickers")
    _save_surprises_cache(articles)
    return articles


# ─────────────────────────────────────────────────────────────────────────────
# Upcoming earnings calendar  (→ Optional[EarningsContext])
# ─────────────────────────────────────────────────────────────────────────────

def fetch_earnings_context(
    tickers: List[str],
    upcoming_days: int = _UPCOMING_WINDOW_DAYS,
    alpha_vantage_key: str = "",
) -> Optional[EarningsContext]:
    """
    Build an EarningsContext listing tickers with upcoming earnings within upcoming_days.

    Primary source: yfinance earnings_dates (future rows have EPS Estimate, no Reported EPS).
    Supplement: Alpha Vantage EARNINGS_CALENDAR CSV if api key provided.

    Cached daily.
    """
    cached = _load_calendar_cache()
    if cached is not None:
        return cached

    if not settings.enable_fetch_data:
        logger.debug("[earnings] ENABLE_FETCH_DATA=false — skipping yfinance calendar fetch")
        return None

    today  = date.today()
    cutoff = today + timedelta(days=upcoming_days)
    events: List[EarningsEvent] = []
    seen: set = set()

    for sym in tickers:
        try:
            t  = yf.Ticker(sym)
            ed = t.earnings_dates

            if ed is None or ed.empty:
                time.sleep(_REQUEST_DELAY)
                continue

            # Future rows: Reported EPS is NaN (earnings not yet reported)
            future = ed[ed.index.date > today]
            future = future[future["Reported EPS"].isna()] if "Reported EPS" in future.columns else future

            if future.empty:
                time.sleep(_REQUEST_DELAY)
                continue

            # Earliest upcoming date
            earliest = future.iloc[-1]   # yfinance sorts descending → last row = earliest future
            earnings_dt = future.index[-1].date()

            if earnings_dt <= cutoff and sym not in seen:
                est_eps = _safe_float(earliest.get("EPS Estimate"))
                events.append(EarningsEvent(
                    ticker=sym,
                    earnings_date=earnings_dt,
                    days_until=(earnings_dt - today).days,
                    estimated_eps=est_eps,
                    is_confirmed=True,
                ))
                seen.add(sym)
                logger.debug(f"[earnings] {sym}: reports {earnings_dt} ({(earnings_dt - today).days}d)")

            time.sleep(_REQUEST_DELAY)

        except Exception as e:
            logger.debug(f"[earnings] {sym} calendar fetch failed: {e}")
            time.sleep(_REQUEST_DELAY)

    # ── Alpha Vantage supplement ──────────────────────────────────────────────
    if alpha_vantage_key:
        av_events = _fetch_av_calendar(alpha_vantage_key, tickers, today, cutoff, seen)
        events.extend(av_events)

    if not events:
        logger.info("[earnings] No upcoming earnings found in window")
        return None

    events.sort(key=lambda e: e.earnings_date)

    within_week = [e for e in events if e.days_until <= 7]
    summary_parts = []
    if within_week:
        names = ", ".join(e.ticker for e in within_week[:6])
        summary_parts.append(f"{len(within_week)} ticker(s) report within 7 days: {names}.")
    summary_parts.append(
        f"{len(events)} upcoming earnings event(s) in the next {upcoming_days} days. "
        "Pre-earnings IV often expands, creating options opportunities. "
        "Avoid opening new POSITION-length longs immediately before binary events."
    )

    ctx = EarningsContext(
        upcoming=events,
        report_date=today,
        summary=" ".join(summary_parts),
    )
    _save_calendar_cache(ctx)
    logger.info(f"[earnings] Earnings context: {len(events)} upcoming event(s)")
    return ctx


def _fetch_av_calendar(
    api_key: str,
    tickers: List[str],
    today: date,
    cutoff: date,
    already_seen: set,
) -> List[EarningsEvent]:
    """Fetch upcoming earnings from Alpha Vantage EARNINGS_CALENDAR (CSV, free endpoint)."""
    import csv
    import io
    import requests

    ticker_set = {t.upper() for t in tickers}
    events: List[EarningsEvent] = []

    try:
        url = (
            "https://www.alphavantage.co/query"
            f"?function=EARNINGS_CALENDAR&horizon=3month&apikey={api_key}"
        )
        resp = requests.get(url, timeout=15)
        resp.raise_for_status()

        reader = csv.DictReader(io.StringIO(resp.text))
        for row in reader:
            sym = row.get("symbol", "").upper()
            if sym not in ticker_set or sym in already_seen:
                continue
            try:
                earnings_dt = date.fromisoformat(row["reportDate"])
                if today < earnings_dt <= cutoff:
                    est = row.get("estimate", "")
                    events.append(EarningsEvent(
                        ticker=sym,
                        earnings_date=earnings_dt,
                        days_until=(earnings_dt - today).days,
                        estimated_eps=float(est) if est else None,
                        is_confirmed=True,
                    ))
                    already_seen.add(sym)
            except (ValueError, KeyError):
                continue

        logger.info(f"[earnings] Alpha Vantage calendar: {len(events)} additional event(s)")
    except Exception as e:
        logger.warning(f"[earnings] Alpha Vantage EARNINGS_CALENDAR failed: {e}")

    return events
