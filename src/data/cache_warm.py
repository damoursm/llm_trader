"""Forward-return cache warming — the fuel for every learning surface.

The IC panels, both policy evaluators, and every evidence-shrunk calibration
read forward returns from the OHLCV file cache. That cache is otherwise only
warmed incidentally by a running pipeline (current-universe tickers, and the
last tick of the day runs pre-close so ``_drop_forming_bar`` drops today's
bar) — so ``signal_date + horizon`` is frequently missing and the learning
silently UNDER-counts evidence (observed 2026-07-03: holiday fetch failures
froze the cache at 07-01, and the exit policy eval returned nothing until a
manual warm). This decouples measurement from cache warmth: once a day, after
the close, force-warm the completed daily bar for EVERY ticker that appears in
the learning panels — not just the current universe — so forward returns are
always computable. File-cache only; never touches the DB.
"""

from __future__ import annotations

from datetime import date, timedelta
from typing import Optional

from loguru import logger


def _panel_tickers(days: int) -> list:
    """Every ticker the learning surfaces need forward returns for: the signals
    + exit-signals panels over the window, plus currently-open trades (whose
    marks/NAV walk also need fresh closes). Read-through the repo; empty on any
    DB hiccup (fail-soft)."""
    from src.db import repo
    cutoff = (date.today() - timedelta(days=days)).isoformat()
    tickers: set = set()
    for sql, params in (
        ("SELECT DISTINCT ticker FROM signals WHERE signal_date >= ?", [cutoff]),
        ("SELECT DISTINCT ticker FROM exit_signals WHERE signal_date >= ?", [cutoff]),
    ):
        try:
            df = repo.fetch_df(sql, params)
            if df is not None and not df.empty:
                tickers |= {str(t) for t in df["ticker"] if t}
        except Exception as e:
            logger.debug(f"[cache_warm] panel-ticker read failed ({e})")
    try:
        for t in repo.load_trades():
            if t.get("status") == "OPEN" and t.get("ticker"):
                tickers.add(str(t["ticker"]))
    except Exception as e:
        logger.debug(f"[cache_warm] open-trade read failed ({e})")
    return sorted(tickers)


def warm_forward_return_cache(days: int = 120, max_tickers: Optional[int] = None) -> int:
    """Force-warm the OHLCV cache for every learning-panel ticker over the last
    ``days``, so their forward-return bars exist. Bounded + fail-soft; returns
    the count successfully warmed. Heavy (one bounded fetch per ticker) — call
    once a day off the time-critical path (the EOD scheduler hook)."""
    tickers = _panel_tickers(days)
    if not tickers:
        logger.info("[cache_warm] no panel tickers to warm yet")
        return 0
    from src.analysis.signal_panel import refresh_panel_ohlcv
    warmed = refresh_panel_ohlcv(tickers, max_tickers=max_tickers)
    logger.info(f"[cache_warm] warmed forward-return cache for {warmed}/{len(tickers)} "
                f"panel ticker(s) over the last {days}d")
    return warmed
