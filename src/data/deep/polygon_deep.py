"""Polygon families — endpoints the pipeline already pays for but never asks
for history from.

* ``bars30m_full``   — every 30-minute bar 04:00–19:30 ET since 2021 (the deep
  RTH store keeps 13 of the 32 bars a session carries; pre-market and
  after-hours reaction is the ``ext_gap`` family's missing history). One part
  per ticker; ~19 pages of ~2,450 bars each.
* ``short_interest``  — FINRA bi-monthly short interest, 2017-12 →, one call per
  ticker. ``settlement_date`` is NOT the publication date (FINRA publishes ~9
  business days later) — the feature builder lags it.
* ``short_volume``    — daily short-sale volume, 2024-02 →, one call per ticker.
* ``dividends`` / ``splits`` — whole-market sweeps by month (declaration,
  ex, record, pay dates; split execution dates), 2000 →.
* ``delisted``        — every inactive stock ticker with ``delisted_utc`` (the
  survivorship fix), plus ``bars1d_delisted`` / ``bars30m_delisted`` for names
  delisted since 2021.
* ``ipos``            — listing history, 2000 →.
* ``ticker_details``  — one row per ticker: list_date, SIC, shares outstanding.

Pagination follows ``next_url`` through the module's own helpers, so the
polygon_client's key handling and 403/429 semantics are reused unchanged.
"""
from __future__ import annotations

from datetime import date, timedelta
from typing import List, Optional

import pandas as pd
from loguru import logger

from src.data.deep import RateLimiter

_LIMITER = RateLimiter(0.05)          # 20 req/s ceiling, well under the paid tiers' allowance
_NY = "America/New_York"
BARS_FROM = "2021-01-01"


def _client():
    from src.data import polygon_client as pc
    return pc


def _get(path: str, params: dict) -> Optional[dict]:
    _LIMITER.wait()
    return _client()._get(path, params)


def _paginate(path: str, params: dict, max_pages: int = 80) -> List[dict]:
    pc = _client()
    j = _get(path, params)
    out: List[dict] = []
    pages = 0
    while j and j.get("results"):
        out.extend(j["results"])
        nxt = j.get("next_url")
        if not nxt or pages >= max_pages:
            break
        pages += 1
        _LIMITER.wait()
        try:
            j = pc._follow_next_url(nxt)
        except Exception as e:                           # noqa: BLE001
            logger.debug(f"[deep.polygon] pagination stopped on {path}: {e}")
            break
    return out


# ── bars ─────────────────────────────────────────────────────────────────────

def _bars_frame(results: List[dict], ticker: str) -> pd.DataFrame:
    if not results:
        return pd.DataFrame()
    df = pd.DataFrame(results)
    df = df.rename(columns={"t": "ts", "o": "open", "h": "high", "l": "low", "c": "close",
                            "v": "volume", "vw": "vwap", "n": "trades"})
    df["ts"] = pd.to_datetime(df["ts"], unit="ms", utc=True).dt.tz_localize(None)
    for c in ("open", "high", "low", "close", "vwap"):
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce").astype("float32")
    for c in ("volume", "trades"):
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0).astype("int64")
    keep = [c for c in ("ts", "open", "high", "low", "close", "volume", "vwap", "trades") if c in df.columns]
    df = df[keep].drop_duplicates("ts").sort_values("ts").reset_index(drop=True)
    df.insert(0, "ticker", ticker.upper())
    return df


def session_label(ts_utc: pd.Series) -> pd.Series:
    """'pre' (04:00–09:29 ET bar starts), 'rth' (09:30–15:59), 'post' (16:00–19:59),
    'off' otherwise — DST-correct via the New York zone."""
    et = pd.DatetimeIndex(ts_utc).tz_localize("UTC").tz_convert(_NY)
    mins = et.hour * 60 + et.minute
    out = pd.Series("off", index=ts_utc.index, dtype=object)
    out[(mins >= 240) & (mins < 570)] = "pre"
    out[(mins >= 570) & (mins < 960)] = "rth"
    out[(mins >= 960) & (mins < 1200)] = "post"
    return out


def bars30m_full(ticker: str, start: str = BARS_FROM, end: Optional[str] = None) -> pd.DataFrame:
    """All 30-minute bars (extended hours included) for ``[start, end]``."""
    pc = _client()
    end = end or date.today().isoformat()
    res = _paginate(f"/v2/aggs/ticker/{pc.to_polygon_symbol(ticker)}/range/30/minute/{start}/{end}",
                    {"adjusted": "true", "sort": "asc", "limit": 50000}, max_pages=80)
    df = _bars_frame(res, ticker)
    if len(df):
        df["session"] = session_label(df["ts"])
        df = df[df["session"] != "off"].reset_index(drop=True)
    return df


def bars1d(ticker: str, start: str = "2005-01-01", end: Optional[str] = None) -> pd.DataFrame:
    pc = _client()
    end = end or date.today().isoformat()
    res = _paginate(f"/v2/aggs/ticker/{pc.to_polygon_symbol(ticker)}/range/1/day/{start}/{end}",
                    {"adjusted": "true", "sort": "asc", "limit": 50000}, max_pages=10)
    return _bars_frame(res, ticker)


# ── short interest / volume ──────────────────────────────────────────────────

def short_interest(ticker: str) -> pd.DataFrame:
    pc = _client()
    res = _paginate("/stocks/v1/short-interest",
                    {"ticker": pc.to_polygon_symbol(ticker), "limit": 5000, "order": "asc",
                     "sort": "settlement_date"}, max_pages=10)
    if not res:
        return pd.DataFrame()
    df = pd.DataFrame(res)
    df["ticker"] = ticker.upper()
    for c in ("short_interest", "avg_daily_volume", "days_to_cover"):
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def short_volume(ticker: str) -> pd.DataFrame:
    pc = _client()
    res = _paginate("/stocks/v1/short-volume",
                    {"ticker": pc.to_polygon_symbol(ticker), "limit": 5000, "order": "asc",
                     "sort": "date"}, max_pages=10)
    if not res:
        return pd.DataFrame()
    df = pd.DataFrame(res)
    df["ticker"] = ticker.upper()
    for c in df.columns:
        if c not in ("ticker", "date"):
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


# ── market-wide sweeps ───────────────────────────────────────────────────────

def _month_windows(start: str, end: Optional[str] = None):
    d0 = date.fromisoformat(start)
    d1 = date.fromisoformat(end) if end else date.today()
    cur = d0.replace(day=1)
    while cur <= d1:
        nxt = (cur.replace(day=28) + timedelta(days=4)).replace(day=1)
        yield cur.isoformat(), min(nxt - timedelta(days=1), d1).isoformat()
        cur = nxt


def dividends_month(window: str) -> pd.DataFrame:
    """One month of the whole market's dividends, keyed by ex-date; ``window``
    is 'YYYY-MM-DD..YYYY-MM-DD'."""
    a, b = window.split("..")
    res = _paginate("/v3/reference/dividends",
                    {"ex_dividend_date.gte": a, "ex_dividend_date.lte": b, "limit": 1000,
                     "order": "asc", "sort": "ex_dividend_date"}, max_pages=60)
    if not res:
        return pd.DataFrame()
    df = pd.DataFrame(res)
    df["cash_amount"] = pd.to_numeric(df.get("cash_amount"), errors="coerce")
    df["frequency"] = pd.to_numeric(df.get("frequency"), errors="coerce")
    return df


def splits_month(window: str) -> pd.DataFrame:
    a, b = window.split("..")
    res = _paginate("/v3/reference/splits",
                    {"execution_date.gte": a, "execution_date.lte": b, "limit": 1000,
                     "order": "asc", "sort": "execution_date"}, max_pages=20)
    if not res:
        return pd.DataFrame()
    df = pd.DataFrame(res)
    for c in ("split_from", "split_to"):
        df[c] = pd.to_numeric(df.get(c), errors="coerce")
    return df


def month_keys(start: str) -> List[str]:
    return [f"{a}..{b}" for a, b in _month_windows(start)]


def delisted_tickers() -> pd.DataFrame:
    """Every inactive ticker in the stocks market, all security types."""
    res = _paginate("/v3/reference/tickers",
                    {"active": "false", "market": "stocks", "limit": 1000, "order": "asc",
                     "sort": "ticker"}, max_pages=200)
    if not res:
        return pd.DataFrame()
    df = pd.DataFrame(res)
    keep = [c for c in ("ticker", "name", "market", "locale", "primary_exchange", "type", "active",
                        "currency_name", "cik", "composite_figi", "share_class_figi",
                        "delisted_utc", "last_updated_utc") if c in df.columns]
    return df[keep]


def ipos(start: str = "2000-01-01") -> pd.DataFrame:
    res = _paginate("/vX/reference/ipos",
                    {"listing_date.gte": start, "limit": 1000, "order": "asc",
                     "sort": "listing_date"}, max_pages=50)
    return pd.DataFrame(res) if res else pd.DataFrame()


def ticker_details(ticker: str) -> pd.DataFrame:
    pc = _client()
    j = _get(f"/v3/reference/tickers/{pc.to_polygon_symbol(ticker)}", {})
    d = (j or {}).get("results") or {}
    if not d:
        return pd.DataFrame()
    keep = ("ticker", "name", "market", "locale", "primary_exchange", "type", "active", "cik",
            "composite_figi", "share_class_figi", "list_date", "sic_code", "sic_description",
            "market_cap", "share_class_shares_outstanding", "weighted_shares_outstanding",
            "total_employees", "delisted_utc", "currency_name")
    row = {k: d.get(k) for k in keep}
    row["ticker"] = ticker.upper()
    for k in ("market_cap", "share_class_shares_outstanding", "weighted_shares_outstanding",
              "total_employees"):
        row[k] = pd.to_numeric(row.get(k), errors="coerce")
    return pd.DataFrame([row])


# ── Polygon news with the provider's own per-ticker sentiment ────────────────

NEWS_FROM = "2021-01-01"


def polygon_news(ticker: str, start: str = NEWS_FROM) -> pd.DataFrame:
    """Every Polygon news article tagged with ``ticker`` since ``start``: one
    row per article with the provider's own LLM sentiment for THIS ticker when
    the article carries ``insights`` (100% of articles from 2025-01-01, none
    before — measured 2026-09-19). Attention counts per day are derivable from
    the rows; the feed's volume changed regime in 2024-25, so counts must be
    normalised within the day and against the ticker's own baseline."""
    pc = _client()
    res = _paginate("/v2/reference/news",
                    {"ticker": pc.to_polygon_symbol(ticker), "published_utc.gte": start,
                     "order": "asc", "sort": "published_utc", "limit": 1000}, max_pages=40)
    if not res:
        return pd.DataFrame()
    rows = []
    sym = pc.to_polygon_symbol(ticker).upper()
    for a in res:
        ins = None
        for i in a.get("insights") or []:
            if str(i.get("ticker", "")).upper() == sym:
                ins = i
                break
        rows.append({
            "ticker": ticker.upper(),
            "article_id": a.get("id"),
            "published_utc": a.get("published_utc"),
            "publisher": (a.get("publisher") or {}).get("name"),
            "title": a.get("title"),
            "n_tickers": len(a.get("tickers") or []),
            "sentiment": (ins or {}).get("sentiment"),
            "sentiment_reasoning": (ins or {}).get("sentiment_reasoning"),
            "keywords": ",".join(a.get("keywords") or [])[:500],
        })
    df = pd.DataFrame(rows)
    df["published_utc"] = pd.to_datetime(df["published_utc"], errors="coerce", utc=True).dt.tz_localize(None)
    df["n_tickers"] = pd.to_numeric(df["n_tickers"], errors="coerce")
    for c in ("article_id", "publisher", "title", "sentiment", "sentiment_reasoning", "keywords"):
        df[c] = df[c].astype(object).where(df[c].notna(), None)
    return df.drop_duplicates("article_id").sort_values("published_utc").reset_index(drop=True)
