"""
Polygon.io market data — primary source for equity/ETF price snapshots and OHLCV history.

Uses direct REST calls via httpx (no SDK dependency, no version drift).
Free API key at https://polygon.io — works globally, no identity verification required.

Covers: US equities and ETFs (NYSE, NASDAQ, ARCA, etc.)
Does NOT cover (stay on yfinance for these):
  - Options chains  (options_flow, gamma_exposure, put_call)
  - Indices         (^VIX, ^VXN, ^VVIX, ^VIX3M, ^VXMT, ^MOVE, ^TICK, ^NYAD)
  - Futures         (GC=F, CL=F, HG=F, DX-Y.NYB)
  - Fundamental data (analyst ratings, earnings, market cap, etc.)
"""

from __future__ import annotations

from datetime import date, timedelta
from typing import Dict, List, Optional

import httpx
import pandas as pd
from loguru import logger

from config import settings

_BASE    = "https://api.polygon.io"
_TIMEOUT = 30.0

_PERIOD_DAYS: Dict[str, int] = {
    "5d": 10, "1mo": 35, "3mo": 95, "6mo": 185,
    "1y": 370, "2y": 740, "5y": 1830,
}


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _get(path: str, params: Optional[dict] = None) -> Optional[dict]:
    """Authenticated GET to the Polygon REST API.  Returns parsed JSON or None."""
    if not settings.polygon_api_key:
        return None
    p = dict(params or {})
    p["apiKey"] = settings.polygon_api_key
    try:
        r = httpx.get(f"{_BASE}{path}", params=p, timeout=_TIMEOUT)
        r.raise_for_status()
        return r.json()
    except httpx.HTTPStatusError as exc:
        status = exc.response.status_code
        if status == 429:
            logger.warning(f"[polygon] Rate limited on {path} — returning empty (yfinance fallback will apply)")
        elif status != 404:
            logger.warning(f"[polygon] HTTP {status} on {path}")
        return None
    except Exception as exc:
        logger.warning(f"[polygon] GET {path}: {exc}")
        return None


def _n_bdays_ago(n: int) -> str:
    """Return the calendar date that is at least *n* week-days (Mon–Fri) before today."""
    d = date.today()
    remaining = n
    while remaining > 0:
        d -= timedelta(days=1)
        if d.weekday() < 5:  # Mon–Fri
            remaining -= 1
    return d.isoformat()


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def is_available() -> bool:
    """True when POLYGON_API_KEY is configured."""
    return bool(settings.polygon_api_key)


def get_snapshots_batch(tickers: List[str]) -> Dict[str, dict]:
    """
    Fetch price snapshots for all *tickers* in two Polygon API calls:

      1. /v2/snapshot  → current price (last trade), prev_close, today's volume
      2. /v2/aggs/grouped/{date} → close from 5 trading days ago for pct_change_5d

    Returns {ticker: dict} with fields:
        price, pct_change_1d, pct_change_5d, volume

    Tickers absent from Polygon's feed are simply missing from the result;
    callers should fall back to yfinance for them.
    """
    if not is_available() or not tickers:
        return {}

    # ── Call 1: multi-ticker snapshot ─────────────────────────────────────
    snap_json = _get(
        "/v2/snapshot/locale/us/markets/stocks/tickers",
        {"tickers": ",".join(tickers)},
    )
    if not snap_json or snap_json.get("status") not in ("OK", "NotFound"):
        logger.debug(f"[polygon] Snapshot returned status: {snap_json.get('status') if snap_json else 'None'}")
        return {}

    result: Dict[str, dict] = {}
    for item in snap_json.get("tickers", []):
        ticker    = item.get("ticker")
        day       = item.get("day")       or {}
        prev      = item.get("prevDay")   or {}
        last_trade = item.get("lastTrade") or {}

        # Best available price: last trade → today's close → yesterday's close
        price      = last_trade.get("p") or day.get("c") or prev.get("c")
        prev_close = prev.get("c")

        if not price or not prev_close:
            continue

        price      = float(price)
        prev_close = float(prev_close)
        pct_1d     = (price - prev_close) / prev_close * 100

        # Volume: yesterday's completed bar is more meaningful than today's intraday running total
        volume = int(prev.get("v") or day.get("v") or 0)

        result[ticker] = {
            "price":         price,
            "pct_change_1d": round(pct_1d, 2),
            "pct_change_5d": round(pct_1d, 2),  # default = 1d change; overridden below
            "volume":        volume,
        }

    if not result:
        return {}

    # ── Call 2: grouped daily close from 5 trading days ago ───────────────
    five_ago  = _n_bdays_ago(5)
    grp_json  = _get(
        f"/v2/aggs/grouped/locale/us/market/stocks/{five_ago}",
        {"adjusted": "true"},
    )
    if grp_json and grp_json.get("results"):
        week_closes: Dict[str, float] = {
            bar["T"]: float(bar["c"]) for bar in grp_json["results"] if "T" in bar and "c" in bar
        }
        for ticker, snap in result.items():
            if ticker in week_closes and week_closes[ticker]:
                week_open = week_closes[ticker]
                snap["pct_change_5d"] = round(
                    (snap["price"] - week_open) / week_open * 100, 2
                )

    logger.debug(f"[polygon] Snapshots for {len(result)}/{len(tickers)} tickers")
    return result


def get_bars(ticker: str, period: str = "3mo") -> pd.DataFrame:
    """
    Fetch daily OHLCV bars for *ticker* covering *period*.

    Returns a DataFrame with columns (Open, High, Low, Close, Volume) and a
    naive DatetimeIndex — identical format to yfinance .history().
    Returns an empty DataFrame on failure or when Polygon is unavailable.
    """
    if not is_available():
        return pd.DataFrame()

    days      = _PERIOD_DAYS.get(period, 95)
    from_date = (date.today() - timedelta(days=days)).isoformat()
    to_date   = date.today().isoformat()

    data = _get(
        f"/v2/aggs/ticker/{ticker}/range/1/day/{from_date}/{to_date}",
        {"adjusted": "true", "sort": "asc", "limit": 50000},
    )
    if not data or not data.get("results"):
        logger.debug(f"[polygon] get_bars: no data for {ticker}")
        return pd.DataFrame()

    rows = []
    for bar in data["results"]:
        ts = pd.Timestamp(bar["t"], unit="ms").normalize()
        rows.append({
            "ts":     ts,
            "Open":   float(bar["o"]),
            "High":   float(bar["h"]),
            "Low":    float(bar["l"]),
            "Close":  float(bar["c"]),
            "Volume": int(bar["v"]),
        })

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows).set_index("ts")
    df.index.name = None
    return df
