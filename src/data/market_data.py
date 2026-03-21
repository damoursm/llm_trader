"""Fetch real-time and historical market data via yfinance."""

import time
import yfinance as yf
import pandas as pd
from loguru import logger
from typing import List
from src.models import TickerSnapshot


def _fetch_ticker(ticker: str, retries: int = 3, delay: float = 2.0):
    """Fetch yfinance Ticker data with retries on empty/error responses."""
    for attempt in range(retries):
        try:
            t = yf.Ticker(ticker)
            hist = t.history(period="5d", interval="1d")
            if not hist.empty:
                return t, hist
        except Exception as e:
            logger.debug(f"{ticker} attempt {attempt + 1} failed: {e}")
        if attempt < retries - 1:
            time.sleep(delay)
    return None, None


def get_snapshots(tickers: List[str]) -> List[TickerSnapshot]:
    """Return latest price, change, and basic technicals for each ticker."""
    snapshots = []
    for ticker in tickers:
        try:
            t, hist = _fetch_ticker(ticker)
            if t is None or hist is None:
                logger.warning(f"No data for {ticker} after retries")
                continue

            info = t.fast_info
            current_price = float(info.last_price)
            prev_close = float(hist["Close"].iloc[-2]) if len(hist) >= 2 else current_price
            pct_change = (current_price - prev_close) / prev_close * 100

            # 5-day return
            week_open = float(hist["Close"].iloc[0])
            week_return = (current_price - week_open) / week_open * 100

            snapshots.append(TickerSnapshot(
                ticker=ticker,
                price=current_price,
                pct_change_1d=round(pct_change, 2),
                pct_change_5d=round(week_return, 2),
                volume=int(info.three_month_average_volume or 0),
                market_cap=getattr(info, "market_cap", None),
            ))
            time.sleep(0.5)  # avoid rate limiting between tickers
        except Exception as e:
            logger.warning(f"market_data failed for {ticker}: {e}")

    logger.info(f"Fetched snapshots for {len(snapshots)}/{len(tickers)} tickers")
    return snapshots


def get_history(ticker: str, period: str = "1mo") -> pd.DataFrame:
    """Return OHLCV history for technical analysis."""
    for attempt in range(3):
        try:
            t = yf.Ticker(ticker)
            df = t.history(period=period, interval="1d")
            if not df.empty:
                return df
        except Exception as e:
            logger.debug(f"get_history {ticker} attempt {attempt + 1} failed: {e}")
        if attempt < 2:
            time.sleep(2.0)
    logger.warning(f"get_history failed for {ticker} after retries")
    return pd.DataFrame()
