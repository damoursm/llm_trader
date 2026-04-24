"""
Short interest data — squeeze setups and bearish positioning signals.

Two data sources (both free, no API key):
1. yfinance ticker.info — bi-weekly snapshot from FINRA via Yahoo Finance:
     shortPercentOfFloat, shortRatio (days-to-cover), sharesShort, sharesShortPriorMonth
2. FINRA Reg SHO daily short sale volume files — recent shorting flow:
     ShortVolume / TotalVolume ratio per ticker

Returns List[NewsArticle] for tickers with actionable readings:

  Squeeze setup (BULLISH):
    shortPercentOfFloat > 15% AND days-to-cover 2–10
    → heavily shorted + manageable cover window → explosive upside on positive catalyst

  Smart money going short (BEARISH):
    shortPercentOfFloat > 20% AND short interest increased > 20% MoM
    → institutional conviction bearish thesis building

  Short covering (BULLISH):
    short interest decreased > 20% MoM
    → forced buying as shorts exit → near-term price support

Cached daily.
"""

import csv
import io
import json
import math
import time
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, List, Optional

import requests
import yfinance as yf
from loguru import logger

from config import settings
from src.models import NewsArticle

CACHE_DIR = Path("cache")
_REQUEST_DELAY = 0.4

# ── Thresholds ───────────────────────────────────────────────────────────────
_MIN_SHORT_PCT   = 0.15   # 15% of float minimum to be noteworthy
_SQUEEZE_MIN_DTC = 2.0    # days-to-cover floor for squeeze signal
_SQUEEZE_MAX_DTC = 10.0   # days-to-cover cap (too high = covering takes too long to ignite)
_MOM_THRESHOLD   = 0.20   # ±20% month-over-month change to flag trend

# ── FINRA Reg SHO URLs (tried in order, most recent trading day) ─────────────
_FINRA_OTC_URL  = "https://cdn.finra.org/equity/regsho/daily/FNRAshvol{date}.txt"
_FINRA_NASDAQ_URL = "https://cdn.finra.org/equity/regsho/daily/FNSQshvol{date}.txt"


# ─────────────────────────────────────────────────────────────────────────────
# Cache
# ─────────────────────────────────────────────────────────────────────────────

def _cache_path() -> Path:
    return CACHE_DIR / f"short_interest_{date.today().isoformat()}.json"


def _load_cache() -> Optional[List[NewsArticle]]:
    path = _cache_path()
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        articles = [NewsArticle.model_validate(a) for a in data]
        logger.info(f"[short] Loaded {len(articles)} short interest articles from cache")
        return articles
    except Exception as e:
        logger.warning(f"[short] Cache load failed: {e}")
        return None


def _save_cache(articles: List[NewsArticle]) -> None:
    CACHE_DIR.mkdir(exist_ok=True)
    try:
        _cache_path().write_text(
            json.dumps([a.model_dump(mode="json") for a in articles], indent=2, default=str),
            encoding="utf-8",
        )
    except Exception as e:
        logger.warning(f"[short] Cache save failed: {e}")


# ─────────────────────────────────────────────────────────────────────────────
# FINRA Reg SHO — daily short sale volume
# ─────────────────────────────────────────────────────────────────────────────

def _prev_trading_day(d: date, offset: int = 1) -> date:
    """Return the Nth previous weekday from d."""
    candidate = d - timedelta(days=offset)
    while candidate.weekday() >= 5:  # skip weekends
        candidate -= timedelta(days=1)
    return candidate


def _fetch_finra_short_volume(tickers: List[str]) -> Dict[str, float]:
    """
    Fetch FINRA Reg SHO daily short volume file and return
    {ticker: short_volume_ratio} for tickers in the watchlist.

    Tries the most recent two trading days; returns empty dict on failure.
    """
    ticker_set = {t.upper() for t in tickers}
    ratios: Dict[str, float] = {}
    headers = {"User-Agent": "Mozilla/5.0 (compatible; llm_trader/1.0)"}

    for days_back in range(1, 4):
        trading_day = _prev_trading_day(date.today(), days_back)
        date_str = trading_day.strftime("%Y%m%d")

        for url_tmpl in (_FINRA_OTC_URL, _FINRA_NASDAQ_URL):
            url = url_tmpl.format(date=date_str)
            try:
                resp = requests.get(url, headers=headers, timeout=15)
                if resp.status_code != 200:
                    continue

                reader = csv.DictReader(io.StringIO(resp.text), delimiter="|")
                for row in reader:
                    sym = row.get("Symbol", "").upper()
                    if sym not in ticker_set:
                        continue
                    try:
                        short_vol = int(row.get("ShortVolume", 0))
                        total_vol = int(row.get("TotalVolume", 1))
                        if total_vol > 0:
                            # Average with any existing value (merging OTC + NASDAQ)
                            existing = ratios.get(sym)
                            ratio = short_vol / total_vol
                            ratios[sym] = (existing + ratio) / 2 if existing else ratio
                    except (ValueError, ZeroDivisionError):
                        continue

                if ratios:
                    logger.info(f"[short] FINRA Reg SHO for {date_str}: {len(ratios)} tickers matched")
                    return ratios

            except Exception as e:
                logger.debug(f"[short] FINRA fetch failed ({url}): {e}")

    logger.info("[short] FINRA Reg SHO data unavailable — proceeding with yfinance only")
    return {}


# ─────────────────────────────────────────────────────────────────────────────
# Article builders
# ─────────────────────────────────────────────────────────────────────────────

def _safe_float(val) -> Optional[float]:
    if val is None:
        return None
    try:
        f = float(val)
        return None if math.isnan(f) or math.isinf(f) else f
    except (TypeError, ValueError):
        return None


def _build_squeeze_article(
    ticker: str,
    short_pct: float,
    dtc: float,
    finra_ratio: Optional[float],
) -> NewsArticle:
    finra_line = (
        f" FINRA daily short volume ratio: {finra_ratio:.0%} of recent volume was short selling."
        if finra_ratio is not None else ""
    )
    title = (
        f"{ticker}: {short_pct:.0%} short float + {dtc:.1f}-day cover — "
        f"squeeze conditions present"
    )
    summary = (
        f"{ticker} has {short_pct:.0%} of its float sold short with a days-to-cover ratio of {dtc:.1f}. "
        f"A heavily shorted stock with a manageable cover window is vulnerable to a short squeeze — "
        f"a positive catalyst (earnings beat, analyst upgrade, news) can force shorts to cover "
        f"rapidly, creating a self-reinforcing price spike.{finra_line} "
        f"This is a bullish setup when combined with positive fundamental or technical signals."
    )
    return NewsArticle(
        title=title,
        summary=summary,
        url=f"https://finance.yahoo.com/quote/{ticker}/",
        source="Short Interest",
        published_at=datetime.now(timezone.utc),
    )


def _build_bearish_article(
    ticker: str,
    short_pct: float,
    mom_change: float,
    finra_ratio: Optional[float],
) -> NewsArticle:
    finra_line = (
        f" FINRA daily short volume confirms: {finra_ratio:.0%} of recent daily volume was short selling."
        if finra_ratio is not None and finra_ratio > 0.45 else ""
    )
    title = (
        f"{ticker}: short interest rises {mom_change:.0%} MoM to {short_pct:.0%} of float — "
        f"bearish institutional positioning"
    )
    summary = (
        f"{ticker} short interest increased {mom_change:.0%} month-over-month to {short_pct:.0%} of float. "
        f"A significant increase in short interest reflects growing institutional conviction that the stock "
        f"will decline — often driven by fundamental concerns, valuation, or sector headwinds.{finra_line} "
        f"This is a bearish signal; however, very high short interest also creates squeeze risk if a "
        f"positive catalyst emerges."
    )
    return NewsArticle(
        title=title,
        summary=summary,
        url=f"https://finance.yahoo.com/quote/{ticker}/",
        source="Short Interest",
        published_at=datetime.now(timezone.utc),
    )


def _build_covering_article(
    ticker: str,
    short_pct: float,
    mom_change: float,
) -> NewsArticle:
    title = (
        f"{ticker}: short interest falls {abs(mom_change):.0%} MoM to {short_pct:.0%} of float — "
        f"covering provides price support"
    )
    summary = (
        f"{ticker} short interest decreased {abs(mom_change):.0%} month-over-month to {short_pct:.0%} of float. "
        f"Significant short covering means shorts are buying back shares to exit positions — "
        f"this forced buying acts as a near-term price support floor and removes a major overhang. "
        f"Bullish signal, especially when paired with positive fundamental catalysts."
    )
    return NewsArticle(
        title=title,
        summary=summary,
        url=f"https://finance.yahoo.com/quote/{ticker}/",
        source="Short Interest",
        published_at=datetime.now(timezone.utc),
    )


# ─────────────────────────────────────────────────────────────────────────────
# Public entry point
# ─────────────────────────────────────────────────────────────────────────────

def fetch_short_interest(tickers: List[str]) -> List[NewsArticle]:
    """
    Fetch short interest data and return NewsArticle objects for tickers
    with notable squeeze setups, bearish positioning, or significant covering.

    Sources:
      - yfinance ticker.info (shortPercentOfFloat, shortRatio, MoM change)
      - FINRA Reg SHO daily short volume files (flow confirmation)

    Cached daily.
    """
    cached = _load_cache()
    if cached is not None:
        return cached

    # ── 1. Fetch FINRA daily short volume (one request covers all tickers) ──
    finra_ratios = _fetch_finra_short_volume(tickers)

    # ── 2. Per-ticker short interest from yfinance ───────────────────────────
    articles: List[NewsArticle] = []

    if not settings.enable_fetch_data:
        logger.debug("[short_interest] ENABLE_FETCH_DATA=false — skipping yfinance fetch")
        return articles

    for sym in tickers:
        try:
            info = yf.Ticker(sym).info

            short_pct = _safe_float(info.get("shortPercentOfFloat"))
            dtc        = _safe_float(info.get("shortRatio"))        # days to cover
            shares_short = _safe_float(info.get("sharesShort"))
            shares_prior = _safe_float(info.get("sharesShortPriorMonth"))
            finra_ratio  = finra_ratios.get(sym.upper())

            if short_pct is None:
                time.sleep(_REQUEST_DELAY)
                continue

            mom_change: Optional[float] = None
            if shares_short and shares_prior and shares_prior > 0:
                mom_change = (shares_short - shares_prior) / shares_prior

            generated = False

            # ── Squeeze setup ──────────────────────────────────────────────
            if (short_pct >= _MIN_SHORT_PCT
                    and dtc is not None
                    and _SQUEEZE_MIN_DTC <= dtc <= _SQUEEZE_MAX_DTC):
                articles.append(_build_squeeze_article(sym, short_pct, dtc, finra_ratio))
                logger.info(f"[short] {sym}: squeeze setup — SI={short_pct:.0%}, DTC={dtc:.1f}d")
                generated = True

            # ── Bearish: significant SI increase ──────────────────────────
            if (not generated
                    and short_pct >= _MIN_SHORT_PCT
                    and mom_change is not None
                    and mom_change >= _MOM_THRESHOLD):
                articles.append(_build_bearish_article(sym, short_pct, mom_change, finra_ratio))
                logger.info(f"[short] {sym}: SI up {mom_change:.0%} MoM → bearish build")
                generated = True

            # ── Bullish: significant SI decrease (short covering) ─────────
            if (not generated
                    and mom_change is not None
                    and mom_change <= -_MOM_THRESHOLD):
                articles.append(_build_covering_article(sym, short_pct, mom_change))
                logger.info(f"[short] {sym}: SI down {mom_change:.0%} MoM → covering")

            time.sleep(_REQUEST_DELAY)

        except Exception as e:
            logger.debug(f"[short] {sym} failed: {e}")
            time.sleep(_REQUEST_DELAY)

    logger.info(f"[short] {len(articles)} short interest article(s) from {len(tickers)} tickers")
    _save_cache(articles)
    return articles
