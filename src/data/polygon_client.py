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

import re
from datetime import date, timedelta
from typing import Dict, List, Optional

import httpx
import pandas as pd
from loguru import logger

from config import settings

try:
    from zoneinfo import ZoneInfo as _ZoneInfo
except ImportError:  # pragma: no cover
    from backports.zoneinfo import ZoneInfo as _ZoneInfo  # type: ignore

_BASE    = "https://api.polygon.io"
_TIMEOUT = 30.0
_NY_TZ   = _ZoneInfo("America/New_York")

# Endpoint families already reported as 403 (entitlement limit) this process —
# warn once, then debug, so a free-tier-forbidden endpoint doesn't spam the log
# (the snapshot endpoint logged 54× in one day; yfinance fallback handles it).
_WARNED_FORBIDDEN: set = set()


def _endpoint_family(path: str) -> str:
    """Collapse a trailing ``/{SYMBOL}`` so per-ticker 403s share one key."""
    return re.sub(r"/[A-Z0-9.\-=^]+$", "", path)

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
        elif status == 403:
            # Entitlement limit: this endpoint isn't on the current Polygon plan
            # (e.g. the free tier's real-time snapshot). Not transient, not a rate
            # limit — the yfinance fallback covers it. Warn once per endpoint, then
            # debug, so it doesn't repeat every tick.
            fam = _endpoint_family(path)
            if fam not in _WARNED_FORBIDDEN:
                _WARNED_FORBIDDEN.add(fam)
                logger.warning(
                    f"[polygon] HTTP 403 on {fam} — endpoint not available on this API "
                    "plan; using fallback (further 403s on it silenced this run)."
                )
            else:
                logger.debug(f"[polygon] HTTP 403 on {path} (endpoint not on plan)")
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


def get_grouped_daily_closes(max_lookback: int = 5) -> Dict[str, float]:
    """``{ticker: close}`` for the most recent COMPLETED session via ONE
    grouped-daily aggregates call.

    This endpoint works on the free tier (unlike the per-ticker snapshot endpoint,
    which 403s), and returns every US ticker's daily bar in a single request — so
    it's the deterministic, near-100%-coverage bulk price source the snapshot
    fallback needs. Tries the last few weekdays so a weekend / holiday / pre-EOD
    'today' falls through to the last session that actually has data. ``{}`` if
    unavailable (no key / all lookbacks empty)."""
    if not is_available():
        return {}
    for n in range(max_lookback + 1):
        j = _get(f"/v2/aggs/grouped/locale/us/market/stocks/{_n_bdays_ago(n)}",
                 {"adjusted": "true"})
        results = (j or {}).get("results") or []
        if results:
            return {b["T"]: float(b["c"]) for b in results
                    if b.get("T") and b.get("c")}
    return {}

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

        # Best available price: last trade → today's close → yesterday's close.
        # lastTrade includes pre/after-market prints, so extended-session
        # snapshots (observation-mode ticks) get genuine extended prices here —
        # only the yfinance fallback path needs special prepost handling.
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


def get_last_price(ticker):
    """Last-trade price for a single ticker via Polygon — used as a live-price
    fallback when yfinance is unavailable.

    Returns ``None`` when Polygon is unavailable or has no price for the ticker
    (the caller then has no usable price and records the failure).
    """
    if not is_available() or not ticker:
        return None
    j = _get(f"/v2/snapshot/locale/us/markets/stocks/tickers/{ticker}", {})
    if not j or j.get("status") not in ("OK", "NotFound"):
        return None
    t = j.get("ticker") or {}
    last_trade = t.get("lastTrade") or {}
    day = t.get("day") or {}
    prev = t.get("prevDay") or {}
    price = last_trade.get("p") or day.get("c") or prev.get("c")
    try:
        price = float(price)
        return price if price > 0 else None
    except (TypeError, ValueError):
        return None


def get_news(limit: int = 1000) -> List[dict]:
    """Latest market-wide news with per-article sentiment ``insights``.

    One call to ``/v2/reference/news`` (Benzinga-sourced + Polygon's LLM
    sentiment layer) returns the most recent articles across all tickers, each
    carrying ``insights: [{ticker, sentiment, sentiment_reasoning}]`` — verified
    present on the free key. Returns the raw results list (empty on failure);
    the caller filters to the universe and maps to NewsArticle.
    """
    if not is_available():
        return []
    j = _get("/v2/reference/news", {"order": "desc", "limit": int(limit)})
    if not j or not j.get("results"):
        return []
    return j["results"]


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


def _rth_only_30m(df: pd.DataFrame) -> pd.DataFrame:
    """Keep only regular-session 30-min bars (09:30–16:00 ET).

    The aggregates endpoint returns extended-hours bars too, but the 30-min panel
    has always been RTH-only (the yfinance contract), and the intraday cache's
    'static off-hours' reuse assumes no bar forms after the close. Bars are labelled
    by start time, so the last kept bar starts 15:30 (covers 15:30–16:00)."""
    if df.empty:
        return df
    et = df.index.tz_convert(_NY_TZ)
    mask = [(t.hour, t.minute) >= (9, 30) and (t.hour, t.minute) < (16, 0) for t in et]
    return df[mask]


def get_intraday_bars(ticker: str, lookback_days: int = 120) -> pd.DataFrame:
    """
    Fetch 30-minute OHLCV bars for *ticker* (REAL-TIME on the Stocks Advanced plan).

    Returns a DataFrame (Open, High, Low, Close, Volume) with a tz-aware UTC
    DatetimeIndex — the format market_data's intraday merge expects — RTH-filtered
    to the regular session. Empty DataFrame on failure / when Polygon is unavailable
    (the caller then falls back to yfinance).
    """
    if not is_available():
        return pd.DataFrame()

    from_date = (date.today() - timedelta(days=lookback_days)).isoformat()
    to_date   = date.today().isoformat()

    data = _get(
        f"/v2/aggs/ticker/{ticker}/range/30/minute/{from_date}/{to_date}",
        {"adjusted": "true", "sort": "asc", "limit": 50000},
    )
    if not data or not data.get("results"):
        logger.debug(f"[polygon] get_intraday_bars: no data for {ticker}")
        return pd.DataFrame()

    rows = []
    for bar in data["results"]:
        rows.append({
            "ts":     pd.Timestamp(bar["t"], unit="ms", tz="UTC"),  # bar START, UTC
            "Open":   float(bar["o"]),
            "High":   float(bar["h"]),
            "Low":    float(bar["l"]),
            "Close":  float(bar["c"]),
            "Volume": int(bar["v"]),
        })
    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows).set_index("ts").sort_index()
    df.index.name = None
    return _rth_only_30m(df)


def get_ratios_batch(tickers: List[str], chunk: int = 100) -> Dict[str, dict]:
    """TTM financial ratios for many *tickers* via the financials/ratios endpoint.

    Requires the **Stocks Advanced** plan (or the ratios add-on); the free tier
    403s. One request per ``chunk`` symbols (``ticker.any_of``); returns
    ``{ticker: ratios_dict}`` with the latest row per ticker. Empty dict when
    unavailable / not entitled, so the caller degrades gracefully.

    Ratio fields include: ``price_to_earnings``, ``price_to_book``,
    ``price_to_sales``, ``ev_to_ebitda``, ``return_on_equity``,
    ``return_on_assets``, ``debt_to_equity``, ``dividend_yield``, ``current``,
    ``free_cash_flow``, ``market_cap``, ``enterprise_value`` (plus ``ticker``,
    ``date``)."""
    if not is_available() or not tickers:
        return {}
    out: Dict[str, dict] = {}
    uniq = list(dict.fromkeys(tickers))
    for i in range(0, len(uniq), chunk):
        group = uniq[i:i + chunk]
        # The endpoint returns the latest daily snapshot (one row per ticker). It
        # rejects sort=date (only ratio columns are sortable), so we don't sort —
        # setdefault keeps the first (current) row per ticker.
        j = _get(
            "/stocks/financials/v1/ratios",
            {"ticker.any_of": ",".join(group), "limit": 1000},
        )
        for row in (j or {}).get("results", []) or []:
            tk = row.get("ticker")
            if tk:
                out.setdefault(tk, row)
    return out


def _get_paginated(path: str, params: dict, max_pages: int = 10) -> List[dict]:
    """GET a v3 reference endpoint following ``next_url`` cursors. Returns the
    concatenated ``results`` (capped at ``max_pages`` pages). Empty on failure."""
    out: List[dict] = []
    j = _get(path, params)
    pages = 0
    while j and j.get("results"):
        out.extend(j["results"])
        nxt = j.get("next_url")
        if not nxt or pages >= max_pages:
            break
        pages += 1
        try:
            r = httpx.get(nxt, params={"apiKey": settings.polygon_api_key}, timeout=_TIMEOUT)
            r.raise_for_status()
            j = r.json()
        except Exception:
            break
    return out


def get_dividends_calendar(start: str, end: str) -> List[dict]:
    """Market-wide dividends with ``ex_dividend_date`` in [start, end] (ISO dates).

    One paginated call covers the whole market — the caller filters to its universe.
    Rows carry ``ticker``, ``ex_dividend_date``, ``cash_amount``, ``frequency``,
    ``pay_date``, ``dividend_type``. Empty list when Polygon is unavailable."""
    if not is_available():
        return []
    return _get_paginated("/v3/reference/dividends", {
        "ex_dividend_date.gte": start, "ex_dividend_date.lte": end, "limit": 1000,
    })


def get_splits_calendar(start: str, end: str) -> List[dict]:
    """Market-wide stock splits with ``execution_date`` in [start, end] (ISO dates).

    Rows carry ``ticker``, ``execution_date``, ``split_from``, ``split_to``. Splits
    are rare, so one page nearly always suffices. Empty list when unavailable."""
    if not is_available():
        return []
    return _get_paginated("/v3/reference/splits", {
        "execution_date.gte": start, "execution_date.lte": end, "limit": 1000,
    })


def get_related_companies(ticker: str) -> List[str]:
    """Massive's peer graph for *ticker* — up to ~10 related tickers (one call).

    Returns a list of peer symbols; empty when unavailable / none found."""
    if not is_available() or not ticker:
        return []
    j = _get(f"/v1/related-companies/{ticker}", {})
    return [r["ticker"] for r in (j or {}).get("results", []) or [] if r.get("ticker")]


def _indicator_values(ticker: str, indicator: str, params: dict) -> list:
    """Latest daily values for a Massive technical indicator (one call). The
    response nests them under ``results.values`` newest-first."""
    if not is_available() or not ticker:
        return []
    p = {"timespan": "day", "order": "desc", "limit": 1, **params}
    j = _get(f"/v1/indicators/{indicator}/{ticker}", p)
    return ((j or {}).get("results") or {}).get("values") or []


def get_rsi(ticker: str, window: int = 14) -> Optional[float]:
    """Latest daily RSI (0–100) for *ticker* via Massive; None when unavailable."""
    vals = _indicator_values(ticker, "rsi", {"window": window})
    try:
        return float(vals[0]["value"]) if vals else None
    except (KeyError, TypeError, ValueError):
        return None


def get_macd(ticker: str) -> Optional[dict]:
    """Latest daily MACD for *ticker* via Massive → ``{value, signal, histogram}``;
    None when unavailable."""
    vals = _indicator_values(ticker, "macd", {})
    if not vals:
        return None
    v = vals[0]
    try:
        return {"value": float(v["value"]), "signal": float(v["signal"]),
                "histogram": float(v["histogram"])}
    except (KeyError, TypeError, ValueError):
        return None
