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

from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from loguru import logger

from src.performance.spread import _dynamic_half_spread


# ---------------------------------------------------------------------------
# OHLCV reader
# ---------------------------------------------------------------------------

def _session_date(ts) -> Optional[date]:
    """Map an OHLCV index timestamp to its NYSE *session* date.

    Why this needs a dedicated helper:
      * yfinance returns tz-aware Timestamps localised to ``America/New_York``
        (midnight ET each session).  After JSON round-trip through the cache
        these come back as UTC-localised Timestamps at 04:00/05:00 UTC.
      * Polygon returns naive Timestamps at 00:00 UTC representing the
        trading date directly (per Polygon's daily-aggregate convention).
      * Mixing the two — e.g. a cache that was built with yfinance and then
        refreshed with Polygon, which is exactly what ``_refresh_open_trade_
        ohlcv`` triggers — used to give off-by-one dates depending on the
        row's provenance.  The fix is to normalise here, on read.

    Rules:
      * tz-aware → convert to ``America/New_York`` and take ``.date()``.
        This yields the session date for both ET-localised (yfinance) and
        UTC-localised (post-round-trip yfinance) timestamps.
      * tz-naive → use ``.date()`` directly: Polygon's t = midnight UTC of
        the session date, so the UTC date equals the session date.
    """
    if ts is None:
        return None
    try:
        tz = getattr(ts, "tzinfo", None) or getattr(ts, "tz", None)
        if tz is not None:
            try:
                return ts.tz_convert("America/New_York").date()
            except (TypeError, AttributeError):
                return ts.astimezone(_NY_TZ).date()
        return ts.date()
    except Exception:
        try:
            return date.fromisoformat(str(ts)[:10])
        except Exception:
            return None


# Imported once at module load — cheap, avoids re-resolving inside the loop.
try:
    from zoneinfo import ZoneInfo as _ZoneInfo
except ImportError:  # pragma: no cover
    from backports.zoneinfo import ZoneInfo as _ZoneInfo  # type: ignore
_NY_TZ = _ZoneInfo("America/New_York")


def _load_close_series(ticker: str) -> Dict[date, float]:
    """Return ``{session_date: close}`` for a ticker from the on-disk OHLCV cache.

    Returns ``{}`` when the ticker has no cached history.  Pure file read —
    no network access, no fallback.

    Two important filters:
      * Non-positive closes (corrupt rows, the rare negative-price futures
        event) are skipped — ``mark / prev_mark`` is undefined there, the
        daily walk must not produce NaN.
      * Every timestamp goes through ``_session_date`` so dict keys are
        always the NYSE trading day in ET, regardless of whether the row
        came from yfinance (tz-aware) or Polygon (tz-naive).  This matches
        the ET-localised ``date.today()`` the tracker uses on the user's
        machine — no off-by-one between trade dates and OHLCV dates.
    """
    from src.data.cache import load_ohlcv

    df = load_ohlcv(ticker)
    if df is None or df.empty or "Close" not in df.columns:
        return {}

    series: Dict[date, float] = {}
    for ts, close in zip(df.index, df["Close"].tolist()):
        try:
            d = _session_date(ts)
            if d is None:
                continue
            c = float(close)
            if c > 0:
                series[d] = c
        except Exception:
            continue
    return series


# ---------------------------------------------------------------------------
# Per-trade daily P&L walk
# ---------------------------------------------------------------------------

def _effective_entry(entry_price: float, action: str, asset_type: str) -> float:
    """Cash basis paid (BUY) or received (SELL) per share, after the bid-ask half-spread."""
    half = _dynamic_half_spread(entry_price, asset_type)
    if action == "BUY":
        return entry_price * (1 + half)
    return entry_price * (1 - half)


def _effective_exit(exit_price: float, action: str, asset_type: str) -> float:
    """Cash basis received (BUY exit) or paid (SELL cover) per share."""
    half = _dynamic_half_spread(exit_price, asset_type)
    if action == "BUY":
        return exit_price * (1 - half)
    return exit_price * (1 + half)


def _direction_sign(action: str) -> int:
    """+1 for long, −1 for short. Used to apply path-faithful daily returns."""
    return 1 if action == "BUY" else -1


def _split_adjustment_factor(
    ref_close: Optional[float],
    ref_date_str: Optional[str],
    close_series: Dict[date, float],
) -> float:
    """How much to scale a price recorded at *ref_date_str* into the current
    cache adjustment scale.

    Mechanism: the OHLCV cache is adjusted (Polygon ``adjusted=true``, yfinance
    ``auto_adjust=True``), so every retroactive split or special dividend
    rescales every prior close.  An ``entry_price`` recorded before such an
    event is on the *old* scale; the walk would otherwise show a phantom price
    jump on the ex-date.  By comparing the close we saw at trade time
    (``ref_close``) to what the cache shows for the same date now
    (``current_close_at_ref``) we recover the exact adjustment ratio and apply
    it to the recorded price.

    Returns ``1.0`` whenever we can't compute the ratio (no reference data on
    the trade — legacy entry, missing cache row, non-positive values).  This
    is the safest fallback: no adjustment is applied, equivalent to assuming
    no corporate action has occurred.
    """
    if not ref_close or not ref_date_str:
        return 1.0
    try:
        ref_d = date.fromisoformat(ref_date_str)
    except Exception:
        return 1.0
    current = close_series.get(ref_d)
    if current is None or current <= 0 or ref_close <= 0:
        return 1.0
    return current / float(ref_close)


def _open_trade_end_anchor(
    entry_d: date,
    today: date,
    close_series: Dict[date, float],
) -> Optional[Tuple[date, float]]:
    """Pick the deterministic end anchor for an OPEN trade.

    Returns ``(date, close)`` for the most recent cached close in
    ``[entry_d, today]``, or ``None`` if the cache has no close in that range
    (e.g. a trade entered today before today's bar has been written).  The
    intentional consequence: the open-trade compound depends only on which
    closes are in the cache, not on the wall-clock time the pipeline ran.

    The deliberate trade-off vs the prior ``current_price``-based anchor:
    intraday moves on day ``today`` aren't reflected until tomorrow's bar
    lands in the cache.  In exchange the compound is identical for every run
    that sees the same cache state — i.e., genuinely deterministic given
    ``(trades.json, cache/ohlcv/*.json)``.
    """
    if not close_series:
        return None
    candidates = [d for d in close_series if entry_d <= d <= today]
    if not candidates:
        return None
    end_d = max(candidates)
    return end_d, close_series[end_d]


def _et_date_of_iso(iso_str) -> Optional[date]:
    """ET session date for an ISO 8601 datetime (e.g. ``current_price_datetime``).

    The live mark is timestamped in UTC; the NAV walk keys marks by NYSE session
    date, so convert to America/New_York first. Returns ``None`` if unparseable.
    """
    if not iso_str:
        return None
    try:
        dt = datetime.fromisoformat(str(iso_str))
    except (TypeError, ValueError):
        return None
    try:
        return dt.astimezone(_NY_TZ).date() if dt.tzinfo is not None else dt.date()
    except Exception:
        return None


def _build_marks(
    trade: dict,
    today: date,
    close_series: Dict[date, float],
) -> List[Tuple[date, float]]:
    """Build the chronological (date, mark_price) anchor list for one trade.

    First mark : entry_date with the effective entry price (post-spread,
                 split-adjusted to the current cache scale).
    Middle     : every cached trading-day close strictly between entry and
                 the last anchor.
    Last mark  : exit_date with effective_exit_price (CLOSED, split-adjusted)
                 OR the most recent cached close on/before today, marked
                 through the spread (OPEN — see ``_open_trade_end_anchor``).

    All anchor prices flow through a split-adjustment factor derived from the
    reference closes recorded at trade time, so a corporate action that
    happens after the trade is opened does not produce a phantom return.
    Returns ``[]`` for trades we can't safely walk (missing/negative entry
    price, missing exit data, no eligible end mark for open trade).
    """
    action     = trade.get("action", "BUY")
    asset_type = trade.get("type", "STOCK")
    entry_px   = trade.get("entry_price")
    if entry_px is None or float(entry_px) <= 0:
        return []
    try:
        entry_d = date.fromisoformat(trade["entry_date"])
    except Exception:
        return []

    # Per-anchor split adjustments. Each one converts a price recorded on its
    # own ref date into the current cache adjustment scale.  Without these,
    # any split between trade-record-time and now puts the recorded entry/
    # exit price on a different scale than the cached closes — the walk would
    # then read a phantom 50%+ daily move on the ex-date.
    entry_adj = _split_adjustment_factor(
        trade.get("entry_ref_close"),
        trade.get("entry_ref_close_date"),
        close_series,
    )
    eff_entry = _effective_entry(float(entry_px), action, asset_type) * entry_adj

    # Determine end anchor (date + effective price)
    if trade.get("status") == "CLOSED":
        exit_px = trade.get("exit_price")
        if exit_px is None or float(exit_px) <= 0:
            return []
        try:
            end_d = date.fromisoformat(trade["exit_date"])
        except Exception:
            return []
        exit_adj = _split_adjustment_factor(
            trade.get("exit_ref_close"),
            trade.get("exit_ref_close_date"),
            close_series,
        )
        end_mark = _effective_exit(float(exit_px), action, asset_type) * exit_adj
    else:
        # OPEN: prefer the live current mark so the equity curve reflects the
        # latest intraday (30-min) price the pipeline marked the position at.
        # Falls back to the most recent completed cached close when no live mark
        # is stored (legacy trades). Both paths stay grounded in observed prices
        # — never a future bar (the live mark is clamped to <= today).
        cur_px = trade.get("current_price")
        if cur_px is not None and float(cur_px) > 0:
            end_d = _et_date_of_iso(trade.get("current_price_datetime")) or today
            if end_d > today:
                end_d = today
            if end_d < entry_d:
                end_d = entry_d
            end_mark = _effective_exit(float(cur_px), action, asset_type)
        else:
            anchor = _open_trade_end_anchor(entry_d, today, close_series)
            if anchor is None:
                # No live mark and no eligible cached close yet (e.g. entered
                # today, today's bar not in cache). Nothing to walk yet.
                return []
            end_d, end_close = anchor
            if end_close <= 0:
                return []
            end_mark = _effective_exit(float(end_close), action, asset_type)

    if end_d < entry_d:
        end_d = entry_d
    if eff_entry <= 0 or end_mark <= 0:
        return []

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

    # Walk consecutive valid marks, skipping any non-positive value WITHOUT
    # updating ``prev_mark``.  A single corrupt cache row (close = 0 / NaN /
    # negative) used to set prev_mark = 0, which then silenced every
    # subsequent r_d via the divide-by-zero guard — the trade's end-anchor
    # return was lost entirely.  Now we just skip the bad row and the next
    # valid mark compares against the last good price.
    results: List[Tuple[date, float, float]] = []
    prev_mark = marks[0][1] if marks[0][1] and marks[0][1] > 0 else None
    for d, mark in marks[1:]:
        if mark is None or mark <= 0:
            continue
        if prev_mark is None or prev_mark <= 0:
            prev_mark = mark
            continue
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
    OHLCV closes (no interpolation).  We then iterate **only the dates where
    at least one trade produced an event** (i.e. trading days when some
    position was active and the cache has a close).  On each such date the
    daily portfolio return is the capital-weighted average of every active
    position's daily return; those daily portfolio returns are compounded
    sequentially.  Calendar days with no events contribute nothing (a no-op
    in compounding), so the walk is automatically free of weekends and
    holidays — they simply don't appear in ``by_day``.

    Returns ``None`` when no trade produces any daily-return event.
    Values are rounded to two decimals.
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
    prev_mark = marks[0][1] if marks[0][1] and marks[0][1] > 0 else None
    for d, mark in marks[1:]:
        if mark is None or mark <= 0:
            continue
        if prev_mark is None or prev_mark <= 0:
            prev_mark = mark
            continue
        out.append((d, mark, sign * (mark - prev_mark) / prev_mark * 100))
        prev_mark = mark
    return out
