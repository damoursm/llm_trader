"""Deterministic daily NAV engine for the performance tracker.

Replaces the geometric-interpolation approximation in tracker._compute_nav_compound
with the actual day-by-day price walk read from the on-disk OHLCV cache.

Properties:
  * No synthetic uniform-per-day returns: every daily return comes from two
    adjacent observed prices, never split or interpolated.
  * Same inputs → same output (100% deterministic). The OHLCV cache is the
    single source of truth; no live network calls are made here.
  * Path-faithful daily returns: each day's return is a direct comparison
    of adjacent marks (``r_d = sign × (mark_d − mark_{d−1}) / mark_{d−1}``)
    with ``sign = +1`` for longs and ``sign = −1`` for shorts. The compound
    is equivalent to marking the position to market each day, and for shorts
    differs slightly from the buy-and-hold ``return_pct`` because volatility
    affects daily-rebalanced shorts — by design, this is the *more accurate*
    representation of what a position actually did, day by day.
  * Bid-ask half-spread is applied only at the entry and exit anchors
    (positions are not retraded each day), so the round-trip spread cost
    falls on a single return event at each end.

Public entry point: ``compute_compound_return(trades, today=None)``.
"""

from __future__ import annotations

from datetime import date, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from loguru import logger

# Imported lazily inside functions to avoid a circular import with tracker.py
# (tracker imports daily_nav, daily_nav refers to tracker spread helper).


# ---------------------------------------------------------------------------
# OHLCV reader
# ---------------------------------------------------------------------------

def _load_close_series(ticker: str) -> Dict[date, float]:
    """Return {trading_date: close} for a ticker from the on-disk OHLCV cache.

    Returns ``{}`` when the ticker has no cached history.  Pure file read —
    no network access, no fallback.
    """
    from src.data.cache import load_ohlcv

    df = load_ohlcv(ticker)
    if df is None or df.empty or "Close" not in df.columns:
        return {}

    series: Dict[date, float] = {}
    for ts, close in zip(df.index, df["Close"].tolist()):
        try:
            d = ts.date() if hasattr(ts, "date") else date.fromisoformat(str(ts)[:10])
            series[d] = float(close)
        except Exception:
            continue
    return series


# ---------------------------------------------------------------------------
# Per-trade daily P&L walk
# ---------------------------------------------------------------------------

def _effective_entry(entry_price: float, action: str, asset_type: str) -> float:
    """Cash basis paid (BUY) or received (SELL) per share, after the bid-ask half-spread."""
    from src.performance.tracker import _dynamic_half_spread

    half = _dynamic_half_spread(entry_price, asset_type)
    if action == "BUY":
        return entry_price * (1 + half)
    return entry_price * (1 - half)


def _effective_exit(exit_price: float, action: str, asset_type: str) -> float:
    """Cash basis received (BUY exit) or paid (SELL cover) per share."""
    from src.performance.tracker import _dynamic_half_spread

    half = _dynamic_half_spread(exit_price, asset_type)
    if action == "BUY":
        return exit_price * (1 - half)
    return exit_price * (1 + half)


def _direction_sign(action: str) -> int:
    """+1 for long, −1 for short. Used to apply path-faithful daily returns."""
    return 1 if action == "BUY" else -1


def _build_marks(
    trade: dict,
    today: date,
    close_series: Dict[date, float],
) -> List[Tuple[date, float]]:
    """Build the chronological (date, mark_price) anchor list for one trade.

    First mark : entry_date with the effective entry price (post-spread).
    Middle     : every cached trading-day close strictly between entry and the
                 last anchor.
    Last mark  : exit_date with effective_exit_price (CLOSED) OR
                 today with the live current_price marked through the spread
                 (OPEN, treated as a hypothetical close so the compound matches
                 the stored M2M return_pct).
    """
    action     = trade.get("action", "BUY")
    asset_type = trade.get("type", "STOCK")
    entry_px   = trade.get("entry_price")
    if entry_px is None:
        return []
    eff_entry = _effective_entry(float(entry_px), action, asset_type)

    try:
        entry_d = date.fromisoformat(trade["entry_date"])
    except Exception:
        return []

    # Determine end anchor (date + effective price)
    if trade.get("status") == "CLOSED":
        exit_px = trade.get("exit_price")
        try:
            end_d = date.fromisoformat(trade["exit_date"])
        except Exception:
            return []
        if exit_px is None:
            return []
        end_mark = _effective_exit(float(exit_px), action, asset_type)
    else:
        # OPEN: use current_price as the live mark, also through the spread
        # so the per-trade compound reconciles with the trade's stored
        # M2M return_pct from _pct_return().
        cp = trade.get("current_price") or trade.get("entry_price")
        if cp is None:
            return []
        end_d    = today
        end_mark = _effective_exit(float(cp), action, asset_type)

    if end_d < entry_d:
        end_d = entry_d

    marks: List[Tuple[date, float]] = [(entry_d, eff_entry)]

    # Intermediate trading-day closes between entry_d (exclusive) and end_d (exclusive)
    for d in sorted(close_series.keys()):
        if d <= entry_d or d >= end_d:
            continue
        marks.append((d, float(close_series[d])))

    marks.append((end_d, end_mark))
    return marks


def _daily_returns_for_trade(
    trade: dict,
    today: date,
) -> List[Tuple[date, float, float]]:
    """Return ``[(date, daily_return, weight), ...]`` for one trade.

    For each pair of adjacent marks (mark_{i-1}, mark_i):
        r_i = sign × (mark_i − mark_{i-1}) / mark_{i-1}
    where ``sign = +1`` for BUY and ``−1`` for SELL.  Marks are the
    effective entry on day 0, every cached daily close in between, and the
    effective exit (or live current price marked through the spread) on the
    final day.  This is path-faithful: each day's return is a direct
    comparison of two real prices, no interpolation, no synthetic values.

    The weight is ``position_size_multiplier`` — the capital allocation for
    capital-weighted portfolio aggregation.
    """
    ticker = trade.get("ticker", "")
    if not ticker:
        return []

    closes = _load_close_series(ticker)
    if not closes:
        logger.debug(f"[daily_nav] No OHLCV cache for {ticker} — using anchors only")

    if trade.get("entry_price") is None:
        return []

    marks = _build_marks(trade, today, closes)
    if len(marks) < 2:
        return []

    weight = float(trade.get("position_size_multiplier", 1.0))
    sign   = _direction_sign(trade.get("action", "BUY"))

    results: List[Tuple[date, float, float]] = []
    prev_mark = marks[0][1]
    for d, mark in marks[1:]:
        if prev_mark == 0:
            results.append((d, 0.0, weight))
        else:
            results.append((d, sign * (mark - prev_mark) / prev_mark, weight))
        prev_mark = mark
    return results


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def compute_compound_return(
    trades: List[dict],
    today: Optional[date] = None,
) -> Optional[float]:
    """Capital-weighted, time-weighted compound return over *trades* (in %).

    For each trade we compute the day-by-day MTM return series using actual
    OHLCV closes (no interpolation). For every calendar day we capital-weight
    those returns across the positions active that day, then compound the
    daily portfolio returns sequentially.

    Returns ``None`` when no trade produces any daily-return event.
    Values are rounded to two decimals (consistent with the previous engine).
    """
    if not trades:
        return None
    today = today or date.today()

    # date → list of (daily_return, weight) events across all trades
    by_day: Dict[date, List[Tuple[float, float]]] = {}
    for tr in trades:
        for d, r, w in _daily_returns_for_trade(tr, today):
            by_day.setdefault(d, []).append((r, w))

    if not by_day:
        return None

    compound = 1.0
    for d in sorted(by_day.keys()):
        events = by_day[d]
        total_w = sum(w for _, w in events)
        if total_w <= 0:
            continue
        day_ret = sum(r * w for r, w in events) / total_w
        compound *= (1.0 + day_ret)

    return round((compound - 1.0) * 100, 2)


def compute_trade_compound(trade: dict, today: Optional[date] = None) -> Optional[float]:
    """Per-trade compound return (%) computed solely from the actual price walk.

    Equivalent to ``compute_compound_return([trade])`` but skips portfolio
    aggregation. Useful for auditing a single position's daily P&L breakdown.
    """
    return compute_compound_return([trade], today=today)


def daily_pnl_breakdown(
    trade: dict,
    today: Optional[date] = None,
) -> List[Tuple[date, float, float]]:
    """Return the raw (date, mark, daily_return_pct) tuples for one trade.

    Exposed for audit logging — every value is grounded in an observed OHLCV
    close or the recorded entry/exit/current price. No interpolation.
    """
    today = today or date.today()
    ticker = trade.get("ticker", "")
    closes = _load_close_series(ticker) if ticker else {}
    if trade.get("entry_price") is None:
        return []

    marks = _build_marks(trade, today, closes)
    if len(marks) < 2:
        return []

    sign = _direction_sign(trade.get("action", "BUY"))
    out: List[Tuple[date, float, float]] = []
    prev_mark = marks[0][1]
    for d, mark in marks[1:]:
        ret_pct = sign * (mark - prev_mark) / prev_mark * 100 if prev_mark else 0.0
        out.append((d, mark, ret_pct))
        prev_mark = mark
    return out
