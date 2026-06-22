"""
Backfill the OHLCV caches for the whole universe via Massive/Polygon.

Pre-warms ``cache/ohlcv/<TICKER>.json`` (daily) and optionally ``cache/ohlcv_30m/``
so the live pipeline reads warm caches with deep history — faster first ticks,
deeper multi-timeframe indicators, and forward returns ready for the Signal-IC
panel — instead of cold-fetching per ticker. Purely additive (merges into the
existing cache; never deletes).

Universe = the pinned core (watchlist + sector / commodity / factor ETFs + held +
hypothetical) ∪ every distinct ticker scored in the ``signals`` panel within the
last ``--signal-days`` days.

Run:
    python -m src.data.backfill                 # daily, 2y, last-90d panel tickers
    python -m src.data.backfill --days 1825      # 5y of daily history
    python -m src.data.backfill --with-30m       # also warm the 30-min cache
"""

from __future__ import annotations

import argparse
from typing import List, Set

from loguru import logger

from config import settings
from src.data.market_data import get_history, is_valid_ticker

# Day count → the nearest get_bars period token (descending).
_PERIOD_TOKENS = (("5y", 1825), ("2y", 730), ("1y", 365),
                  ("6mo", 180), ("3mo", 90), ("1mo", 30))


def _period_for_days(days: int) -> str:
    for token, d in _PERIOD_TOKENS:
        if days >= d:
            return token
    return "1mo"


def _recent_signal_tickers(days: int) -> List[str]:
    """Distinct tickers scored in the `signals` panel within the last *days*."""
    from datetime import date, timedelta
    from src.db import repo

    cutoff = (date.today() - timedelta(days=days)).isoformat()
    df = repo.fetch_df("SELECT DISTINCT ticker FROM signals WHERE signal_date >= ?", [cutoff])
    if df is None or df.empty:
        return []
    return df["ticker"].dropna().astype(str).tolist()


def gather_universe(signal_days: int = 90) -> List[str]:
    """Pinned core ∪ recently-scored signals-panel tickers (validated, sorted)."""
    tickers: Set[str] = set()
    tickers.update(settings.stocks_list)
    tickers.update(settings.sectors_list)
    tickers.update(settings.commodities_list)
    tickers.update(settings.factor_list)
    tickers.update(t for t, _ in settings.hypothetical_trades_list)

    try:
        from src.performance.tracker import get_open_trade_tickers
        tickers.update(get_open_trade_tickers())
    except Exception as e:
        logger.warning(f"[backfill] open-trade tickers unavailable: {e}")

    try:
        tickers.update(_recent_signal_tickers(signal_days))
    except Exception as e:
        logger.warning(f"[backfill] signals-panel tickers unavailable: {e}")

    return sorted(t for t in tickers if is_valid_ticker(t))


def backfill(days: int = 730, with_30m: bool = False, signal_days: int = 90,
             daily: bool = True) -> dict:
    """Warm the daily (and optionally 30-min) OHLCV cache for the whole universe.

    ``daily=False`` skips the daily pass (e.g. a 30-min-only top-up after the daily
    cache is already warm). Weekly needs no pass — it is resampled from the daily
    cache on demand. Returns ``{"total", "daily", "intraday"}`` counts."""
    if not settings.enable_fetch_data:
        logger.error("[backfill] ENABLE_FETCH_DATA=false — nothing to fetch")
        return {"total": 0, "daily": 0, "intraday": 0}

    universe = gather_universe(signal_days)
    period = _period_for_days(days)
    logger.info(f"[backfill] {len(universe)} tickers | daily={daily} (period={period}) | 30m={with_30m}")

    daily_ok = 0
    if daily:
        for i, tk in enumerate(universe, 1):
            try:
                df = get_history(tk, period=period, force_refresh=True)
                if df is not None and not df.empty:
                    daily_ok += 1
            except Exception as e:  # pragma: no cover - defensive
                logger.debug(f"[backfill] {tk} daily failed: {e}")
            if i % 50 == 0:
                logger.info(f"[backfill] daily {i}/{len(universe)} ({daily_ok} ok)")

    intraday_ok = 0
    if with_30m:
        for i, tk in enumerate(universe, 1):
            try:
                df = get_history(tk, interval="30m", force_refresh=True)
                if df is not None and not df.empty:
                    intraday_ok += 1
            except Exception as e:  # pragma: no cover - defensive
                logger.debug(f"[backfill] {tk} 30m failed: {e}")
            if i % 50 == 0:
                logger.info(f"[backfill] 30m {i}/{len(universe)} ({intraday_ok} ok)")

    logger.info(
        f"[backfill] done — daily {daily_ok}/{len(universe)}"
        + (f", 30m {intraday_ok}/{len(universe)}" if with_30m else "")
    )
    return {"total": len(universe), "daily": daily_ok, "intraday": intraday_ok}


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Backfill OHLCV caches for the whole universe via Massive/Polygon.")
    ap.add_argument("--days", type=int, default=730, help="daily history depth (default 730 = 2y)")
    ap.add_argument("--signal-days", type=int, default=90,
                    help="also include tickers scored in the signals panel within N days (default 90)")
    ap.add_argument("--with-30m", action="store_true", help="also warm the 30-min cache (heavier)")
    ap.add_argument("--skip-daily", action="store_true",
                    help="skip the daily pass (e.g. a 30-min-only top-up when daily is already warm)")
    args = ap.parse_args()
    backfill(days=args.days, with_30m=args.with_30m, signal_days=args.signal_days,
             daily=not args.skip_daily)


if __name__ == "__main__":
    main()
