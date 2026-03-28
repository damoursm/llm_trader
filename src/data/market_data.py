"""Fetch real-time and historical market data via yfinance.

Rate-limit handling
-------------------
Yahoo Finance enforces a per-IP request quota.  When a 429 / rate-limit error
is detected the module enters a *backoff* state:

  attempt 1 → wait BACKOFF_BASE  seconds  (60 s)
  attempt 2 → wait BACKOFF_BASE * 2       (120 s)
  attempt 3 → wait BACKOFF_BASE * 4       (240 s)
  …up to BACKOFF_MAX                      (600 s)

If all retries are exhausted, fetching stops for the current pipeline run
and returns whatever snapshots were collected so far.  The pipeline continues
with news-only signals.
"""

import time
import yfinance as yf
import pandas as pd
from loguru import logger
from typing import List, Optional, Tuple
from src.models import TickerSnapshot


# ---------------------------------------------------------------------------
# Rate-limit config
# ---------------------------------------------------------------------------

BACKOFF_BASE  = 60    # seconds to wait after first rate-limit hit
BACKOFF_MAX   = 600   # cap (10 min)
MAX_RL_HITS   = 3     # give up fetching after this many consecutive RL events
INTER_TICKER  = 1.0   # polite pause between successful tickers (seconds)


# ---------------------------------------------------------------------------
# Rate-limit detection
# ---------------------------------------------------------------------------

_RATE_LIMIT_PHRASES = (
    "429",
    "too many requests",
    "rate limit",
    "ratelimit",
    "yfratelimiterror",
)


def _is_rate_limit(exc: Exception) -> bool:
    msg = str(exc).lower()
    return any(p in msg for p in _RATE_LIMIT_PHRASES)


def _backoff_wait(attempt: int) -> None:
    """Exponential backoff capped at BACKOFF_MAX.  Logs countdown every 15 s."""
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


# ---------------------------------------------------------------------------
# Core fetch with rate-limit awareness
# ---------------------------------------------------------------------------

def _fetch_ticker(ticker: str) -> Tuple[Optional[yf.Ticker], Optional[pd.DataFrame]]:
    """
    Fetch one ticker with up to MAX_RL_HITS exponential backoff retries.

    Returns (Ticker, hist_df) on success, or (None, None) if:
      - the ticker simply has no data (don't retry)
      - we've been rate-limited too many times (caller should abort the loop)

    Raises _RateLimitAbort when MAX_RL_HITS consecutive RL errors occur.
    """
    rl_hits = 0
    while True:
        try:
            t = yf.Ticker(ticker)
            hist = t.history(period="5d", interval="1d")
            if not hist.empty:
                return t, hist
            # Empty but no exception → ticker probably invalid / delisted
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


class _RateLimitAbort(Exception):
    """Raised when the rate limit persists beyond MAX_RL_HITS retries."""


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def get_snapshots(tickers: List[str]) -> List[TickerSnapshot]:
    """
    Return latest price, 1-day and 5-day change for each ticker.

    Stops early (returns partial results) if Yahoo Finance rate-limits us
    beyond recovery.  The pipeline continues with whatever was collected.
    """
    snapshots: List[TickerSnapshot] = []

    for i, ticker in enumerate(tickers):
        try:
            t, hist = _fetch_ticker(ticker)
            if t is None or hist is None:
                logger.warning(f"[market_data] No data for {ticker} — skipping")
                continue

            info = t.fast_info
            current_price = float(info.last_price)
            prev_close    = float(hist["Close"].iloc[-2]) if len(hist) >= 2 else current_price
            pct_change    = (current_price - prev_close) / prev_close * 100

            week_open   = float(hist["Close"].iloc[0])
            week_return = (current_price - week_open) / week_open * 100

            snapshots.append(TickerSnapshot(
                ticker=ticker,
                price=current_price,
                pct_change_1d=round(pct_change, 2),
                pct_change_5d=round(week_return, 2),
                volume=int(info.three_month_average_volume or 0),
                market_cap=getattr(info, "market_cap", None),
            ))
            logger.debug(f"[market_data] {ticker}: ${current_price:.2f}  ({pct_change:+.2f}% 1d)")

            # Polite pause between tickers to stay under the rate limit
            if i < len(tickers) - 1:
                time.sleep(INTER_TICKER)

        except _RateLimitAbort as e:
            logger.error(
                f"[market_data] {e}. "
                f"Stopping early — {len(snapshots)}/{len(tickers)} tickers fetched. "
                "Pipeline will continue with news-only signals."
            )
            break

        except Exception as e:
            logger.warning(f"[market_data] Unexpected error for {ticker}: {e}")

    logger.info(f"[market_data] Fetched snapshots for {len(snapshots)}/{len(tickers)} tickers")
    return snapshots


def get_history(ticker: str, period: str = "3mo") -> pd.DataFrame:
    """
    Return OHLCV history for chart generation / technical analysis.
    Applies the same rate-limit backoff as get_snapshots.
    """
    rl_hits = 0
    while True:
        try:
            df = yf.Ticker(ticker).history(period=period, interval="1d")
            if not df.empty:
                return df
            logger.warning(f"[market_data] get_history: empty data for {ticker}")
            return pd.DataFrame()

        except Exception as e:
            if _is_rate_limit(e):
                if rl_hits >= MAX_RL_HITS:
                    logger.error(f"[market_data] get_history rate-limited out for {ticker}")
                    return pd.DataFrame()
                _backoff_wait(rl_hits)
                rl_hits += 1
            else:
                logger.warning(f"[market_data] get_history failed for {ticker}: {e}")
                return pd.DataFrame()