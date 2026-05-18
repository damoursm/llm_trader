"""Fetch real-time and historical market data.

Primary source: Polygon.io (batched API calls, no per-IP rate limit, works globally).
Fallback:       yfinance (per-ticker, subject to 429s).

Polygon covers all US equity / ETF tickers.  yfinance is kept as fallback for:
  - Tickers absent from Polygon's feed (very small caps, OTC)
  - Options chains, indices (^VIX, ^MOVE …), and futures (GC=F …) — these
    modules call yfinance directly and are not routed through this file.

Rate-limit handling (yfinance fallback only)
--------------------------------------------
  attempt 1 → wait BACKOFF_BASE  seconds  (60 s)
  attempt 2 → wait BACKOFF_BASE * 2       (120 s)
  attempt 3 → wait BACKOFF_BASE * 4       (240 s)
  …up to BACKOFF_MAX                      (600 s)
"""

import time
import yfinance as yf
import pandas as pd
from loguru import logger
from typing import List, Optional, Tuple
from config import settings
from src.models import TickerSnapshot
from src.data import polygon_client


# ---------------------------------------------------------------------------
# Rate-limit config (yfinance fallback)
# ---------------------------------------------------------------------------

BACKOFF_BASE = 60
BACKOFF_MAX  = 600
MAX_RL_HITS  = 3
INTER_TICKER = 1.0


# ---------------------------------------------------------------------------
# yfinance rate-limit helpers
# ---------------------------------------------------------------------------

_RATE_LIMIT_PHRASES = ("429", "too many requests", "rate limit", "ratelimit", "yfratelimiterror")


def _is_rate_limit(exc: Exception) -> bool:
    msg = str(exc).lower()
    return any(p in msg for p in _RATE_LIMIT_PHRASES)


def _backoff_wait(attempt: int) -> None:
    wait = min(BACKOFF_BASE * (2 ** attempt), BACKOFF_MAX)
    logger.warning(
        f"[market_data] Rate limit hit — backing off for {wait:.0f}s "
        f"(attempt {attempt + 1}/{MAX_RL_HITS})"
    )
    elapsed = 0
    while elapsed < wait:
        chunk = min(15, wait - elapsed)
        time.sleep(chunk)
        elapsed += chunk
        remaining = wait - elapsed
        if remaining > 0:
            logger.info(f"[market_data] Resuming in {remaining:.0f}s…")


class _RateLimitAbort(Exception):
    pass


def _fetch_ticker_yf(ticker: str) -> Tuple[Optional[yf.Ticker], Optional[pd.DataFrame]]:
    """Fetch one ticker via yfinance with exponential-backoff retry."""
    rl_hits = 0
    while True:
        try:
            t = yf.Ticker(ticker)
            hist = t.history(period="5d", interval="1d")
            if not hist.empty:
                return t, hist
            logger.debug(f"[market_data] {ticker}: empty history (skipping)")
            return None, None
        except Exception as e:
            if _is_rate_limit(e):
                if rl_hits >= MAX_RL_HITS:
                    raise _RateLimitAbort(f"Rate limit persists after {MAX_RL_HITS} retries")
                _backoff_wait(rl_hits)
                rl_hits += 1
            else:
                logger.debug(f"[market_data] {ticker}: {e}")
                return None, None


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def get_snapshots(tickers: List[str]) -> List[TickerSnapshot]:
    """
    Return latest price, 1-day and 5-day % change for each ticker.

    Tries Alpaca first (single batch call for all tickers).  Any ticker not
    returned by Alpaca is retried via yfinance.  Stops early on yfinance rate-
    limit exhaustion but always returns whatever was collected.
    """
    if not settings.enable_fetch_data:
        logger.debug("[market_data] ENABLE_FETCH_DATA=false — skipping snapshot fetch")
        return []

    snapshots: List[TickerSnapshot] = []

    # ── 1. Polygon batch (two REST calls for all tickers, no per-ticker throttle) ──
    polygon_data = polygon_client.get_snapshots_batch(tickers)
    covered: set = set()

    for ticker, data in polygon_data.items():
        if data.get("price") is not None:
            snapshots.append(TickerSnapshot(
                ticker=ticker,
                price=data["price"],
                pct_change_1d=data["pct_change_1d"],
                pct_change_5d=data["pct_change_5d"],
                volume=data["volume"],
                market_cap=None,  # Polygon free tier does not expose market cap
            ))
            covered.add(ticker)
            logger.debug(
                f"[market_data] {ticker}: ${data['price']:.2f} "
                f"({data['pct_change_1d']:+.2f}% 1d) [polygon]"
            )

    # ── 2. yfinance fallback for tickers Polygon didn't cover ────────────
    remaining = [t for t in tickers if t not in covered]
    if remaining:
        if polygon_client.is_available():
            logger.info(
                f"[market_data] yfinance fallback for {len(remaining)} ticker(s) "
                f"not in Polygon: {remaining}"
            )
        for i, ticker in enumerate(remaining):
            try:
                t, hist = _fetch_ticker_yf(ticker)
                if t is None or hist is None:
                    logger.warning(f"[market_data] No data for {ticker} — skipping")
                    continue

                info          = t.fast_info
                current_price = float(info.last_price)
                prev_close    = float(hist["Close"].iloc[-2]) if len(hist) >= 2 else current_price
                pct_change    = (current_price - prev_close) / prev_close * 100
                week_open     = float(hist["Close"].iloc[0])
                week_return   = (current_price - week_open) / week_open * 100

                snapshots.append(TickerSnapshot(
                    ticker=ticker,
                    price=current_price,
                    pct_change_1d=round(pct_change, 2),
                    pct_change_5d=round(week_return, 2),
                    volume=int(info.three_month_average_volume or 0),
                    market_cap=getattr(info, "market_cap", None),
                ))
                logger.debug(
                    f"[market_data] {ticker}: ${current_price:.2f} "
                    f"({pct_change:+.2f}% 1d) [yfinance]"
                )

                if i < len(remaining) - 1:
                    time.sleep(INTER_TICKER)

            except _RateLimitAbort as e:
                logger.error(
                    f"[market_data] {e}. Stopping yfinance fallback early — "
                    f"{len(snapshots)}/{len(tickers)} tickers fetched. "
                    "Pipeline will continue with news-only signals."
                )
                break
            except Exception as e:
                logger.warning(f"[market_data] Unexpected error for {ticker}: {e}")

    source = (
        "yfinance" if not polygon_client.is_available()
        else ("polygon" if not remaining else "polygon+yfinance")
    )
    logger.info(
        f"[market_data] Fetched snapshots for {len(snapshots)}/{len(tickers)} tickers [{source}]"
    )
    return snapshots


def _normalize_index_tz(df: Optional[pd.DataFrame]) -> Optional[pd.DataFrame]:
    """Return a copy of df with the index normalised to tz-naive UTC.

    Why: yfinance returns tz-aware timestamps (often America/New_York or, after
    a JSON round-trip through the cache, UTC at 04:00/05:00); Polygon returns
    tz-naive timestamps at midnight UTC of the session date. Mixing these in
    a single concat triggers ``TypeError: Cannot compare tz-naive and tz-aware
    timestamps`` on the subsequent sort. Both representations ultimately point
    at the same NYSE session, so collapsing to tz-naive UTC is safe and gives
    sort_index something it can actually compare.
    """
    if df is None or df.empty:
        return df
    idx = df.index
    if isinstance(idx, pd.DatetimeIndex) and idx.tz is not None:
        df = df.copy()
        df.index = idx.tz_convert("UTC").tz_localize(None)
    return df


def _merge_ohlcv(cached: Optional[pd.DataFrame], fresh: pd.DataFrame) -> pd.DataFrame:
    """Combine cached bars with a fresh fetch, preferring fresh on overlap.

    Rationale: a force-refresh fetch typically covers only the last 3 months,
    but the cache may already hold years of older history.  Naive overwrite
    would silently shrink the cache.  Merging keeps the longest possible
    history while letting the fresh bars rule within their window so any
    split / dividend rescaling is propagated forward — the cache stays
    internally consistent on the new adjustment scale for everything in the
    fresh window, and retains pre-fresh-window bars as-is.

    Old rows whose dates fall inside the fresh window are dropped (the fresh
    bars are the source of truth for that range — they reflect the current
    adjustment scale).  Rows older than the fresh window's first bar are
    kept untouched.

    Both inputs go through ``_normalize_index_tz`` first so a tz-aware cache
    + tz-naive fresh fetch (or vice versa) doesn't crash the sort.
    """
    if fresh is None or fresh.empty:
        return cached if cached is not None else pd.DataFrame()
    if cached is None or cached.empty:
        return _normalize_index_tz(fresh)

    cached = _normalize_index_tz(cached)
    fresh = _normalize_index_tz(fresh)

    fresh_dates = {ts.date() for ts in fresh.index}
    keep_mask   = [ts.date() not in fresh_dates for ts in cached.index]
    cached_kept = cached.iloc[keep_mask]
    if cached_kept.empty:
        return fresh
    combined = pd.concat([cached_kept, fresh]).sort_index()
    # Drop any accidental duplicate timestamps (defensive — shouldn't happen
    # after the keep_mask filter, but tz-aware/naive mixes can sneak through).
    combined = combined[~combined.index.duplicated(keep="last")]
    return combined


def get_history(ticker: str, period: str = "3mo", force_refresh: bool = False) -> pd.DataFrame:
    """
    Return OHLCV history for chart generation / technical analysis.

    Checks the OHLCV disk cache first (TTL 3 days).  On a cache miss:
      1. Tries Polygon.io (no per-IP rate-limit concern).
      2. Falls back to yfinance with exponential-backoff retry.
    Successful fetches are **merged** with whatever was already cached
    (`_merge_ohlcv`): fresh bars take precedence inside their window so any
    split / dividend rescaling propagates, but older history outside the
    fresh window is retained.  This prevents the force-refresh path from
    silently truncating long-tail OHLCV history.

    ``force_refresh``: bypass the TTL check and re-fetch even when the
    cached last bar is within the 3-day window. Used by the performance
    tracker to keep open-trade OHLCV fully up to date.
    """
    from src.data.cache import load_ohlcv, save_ohlcv
    from datetime import date as _date

    cached = load_ohlcv(ticker)
    if cached is not None and not cached.empty and not force_refresh:
        last_bar = cached.index[-1].date()
        if (_date.today() - last_bar).days <= 3:
            logger.debug(f"[market_data] get_history: cache hit for {ticker} (last bar {last_bar})")
            return cached

    if not settings.enable_fetch_data:
        logger.debug(f"[market_data] ENABLE_FETCH_DATA=false — skipping history fetch for {ticker}")
        return cached if (cached is not None and not cached.empty) else pd.DataFrame()

    # ── 1. Polygon ────────────────────────────────────────────────────────
    df = polygon_client.get_bars(ticker, period)
    if not df.empty:
        merged = _merge_ohlcv(cached, df)
        save_ohlcv(ticker, merged)
        logger.debug(
            f"[market_data] get_history: fetched {len(df)} bars for {ticker} [polygon] "
            f"(cache now {len(merged)} bars)"
        )
        return merged

    # ── 2. yfinance fallback ──────────────────────────────────────────────
    rl_hits = 0
    while True:
        try:
            df = yf.Ticker(ticker).history(period=period, interval="1d")
            if not df.empty:
                merged = _merge_ohlcv(cached, df)
                save_ohlcv(ticker, merged)
                time.sleep(INTER_TICKER)
                logger.debug(
                    f"[market_data] get_history: fetched {len(df)} bars for {ticker} [yfinance] "
                    f"(cache now {len(merged)} bars)"
                )
                return merged
            logger.warning(f"[market_data] get_history: empty data for {ticker}")
            return cached if (cached is not None and not cached.empty) else pd.DataFrame()

        except Exception as e:
            if _is_rate_limit(e):
                if rl_hits >= MAX_RL_HITS:
                    logger.error(f"[market_data] get_history rate-limited out for {ticker}")
                    return cached if (cached is not None and not cached.empty) else pd.DataFrame()
                _backoff_wait(rl_hits)
                rl_hits += 1
            else:
                logger.warning(f"[market_data] get_history failed for {ticker}: {e}")
                return cached if (cached is not None and not cached.empty) else pd.DataFrame()
