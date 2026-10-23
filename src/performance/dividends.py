"""Dividends in the ledger (AUDIT151, the user's go-ahead 2026-10-19: "Yes please do 1 and 2") — a real account
PAYS the dividend on a stock it is short (the payment in lieu) and RECEIVES it on a stock it holds long; the ledger
charged neither, so its equity drifted from IBKR's cash by every one. The simulator applies the same rule
(`src/backtest/dividends.py`, `Mimic(dividends=True)`).

THE ENTITLEMENT (the simulator's, IBKR's trade-date convention): a position is entitled to (or owes) an ex-date's
dividend when it was opened before 20:00 ET of the market session before the ex-date (an execution at/after 20:00 is
dated to the next session) and still held at that instant — it held through the close before the ex-date. The cash is
Polygon's record (the deep store's ``dividends`` family, refreshed nightly; USD cash amounts summed per ex-date, as
declared — in the terms of their own date), brought into the trade's current price terms by the splits executed
after the ex-date (`intraday_store.split_factor_between`).

FROZEN WHEN FIRST DUE: at the first mark (or close) past the entitlement instant the event is stored on the trade
(``dividends``: ex-date, cash per share, ``frac`` = that cash over the entry price in the same terms, the instant) and
never recomputed — a fraction of the entry, unchanged by a later split. `frac_through` sums the events due by an
instant: `tracker._pct_return` adds it to a long's return and takes it from a short's, the daily NAV books it on the
ex-date's interval, and the live account follows through the return. No look-ahead: an event counts only once its
instant has passed (`tests/test_no_lookahead.py`). Only the live books' trades (`books.is_live`).
"""
from __future__ import annotations

import os
import threading
from bisect import bisect_left, bisect_right
from datetime import date, datetime, time, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from zoneinfo import ZoneInfo

from loguru import logger

from config.settings import settings

ET = ZoneInfo("America/New_York")
DIV_PATH = Path("cache/ml/deep/dividends.parquet")      # the deep store's consolidated `dividends` family
FINAL_AFTER_DAYS = 7                 # a closed trade's events are re-checked this long after its exit (late records)
FROM = "2025-01-01"                  # the ex-dates loaded (the live books trade from 2026-09-28)
_LOCK = threading.Lock()
_CACHE: Dict[str, object] = {}


def enabled() -> bool:
    return bool(getattr(settings, "enable_ledger_dividends", False))


def reset() -> None:
    """Forget the loaded table (tests; a rebuilt file is re-read by its modification time anyway)."""
    with _LOCK:
        _CACHE.clear()


def table() -> Dict[str, Tuple[List[str], List[float]]]:
    """{Polygon symbol: ([ex-dates, ISO, sorted], [USD cash per share summed per ex-date])} — re-read when the file
    changes. Empty when the file is missing or unreadable (logged once): the ledger then charges no dividend."""
    p = DIV_PATH
    try:
        mtime = p.stat().st_mtime_ns
    except OSError:
        return {}
    with _LOCK:
        if _CACHE.get("key") == (str(p), mtime):
            return _CACHE["table"]                               # type: ignore[return-value]
    try:
        import duckdb
        con = duckdb.connect()
        rows = con.execute(
            f"SELECT ticker, ex_dividend_date, sum(cash) FROM (SELECT DISTINCT ticker, ex_dividend_date, "
            f"dividend_type, cash_amount AS cash FROM read_parquet('{p.as_posix()}') WHERE currency = 'USD' "
            f"AND cash_amount > 0 AND ex_dividend_date >= '{FROM}') GROUP BY 1, 2 ORDER BY 1, 2").fetchall()
        con.close()
    except Exception as e:                                       # noqa: BLE001
        logger.warning(f"[dividends] {p} unreadable ({e}) — no dividend charged until it is")
        return {}
    out: Dict[str, Tuple[List[str], List[float]]] = {}
    for tk, ex, cash in rows:
        exs, cs = out.setdefault(str(tk), ([], []))
        exs.append(str(ex)[:10])
        cs.append(float(cash))
    with _LOCK:
        _CACHE.update(key=(str(p), mtime), table=out)
    return out


def _symbol(ticker: str) -> str:
    from src.data.polygon_client import to_polygon_symbol
    return to_polygon_symbol(str(ticker))


def start_of(ex: date) -> datetime:
    """The entitlement instant of an ex-date: 20:00 ET of the market session before it (UTC)."""
    from src.performance.market_calendar import previous_market_day
    prev = previous_market_day(ex)
    return datetime.combine(prev, time(20, 0), tzinfo=ET).astimezone(timezone.utc)


def _instant(iso) -> Optional[datetime]:
    if not iso:
        return None
    try:
        dt = datetime.fromisoformat(str(iso).replace(" ", "T"))
    except ValueError:
        return None
    return dt if dt.tzinfo is not None else dt.replace(tzinfo=timezone.utc)


def eligible(trade: dict) -> bool:
    from src.performance.books import is_live
    return enabled() and is_live(trade) and str(trade.get("type") or "STOCK").upper() == "STOCK"


def update(trade: dict, end_iso=None, now: Optional[datetime] = None) -> int:
    """Store on ``trade`` every dividend event due by ``end_iso`` (default now; a closed trade: its exit) that it does
    not hold yet — the ex-date's entitlement instant after the entry and at or before both ``end_iso`` and the exit.
    Returns the number added. A record whose split terms cannot be read waits for the next call."""
    if not eligible(trade):
        return 0
    now = now or datetime.now(timezone.utc)
    entry = _instant(trade.get("entry_datetime"))
    if entry is None:
        return 0
    end = _instant(end_iso) or now
    exit_at = _instant(trade.get("exit_datetime")) if trade.get("status") == "CLOSED" else None
    if exit_at is not None and exit_at < end:
        end = exit_at
    if end > now:
        end = now                                                # never an instant that has not happened
    t = table().get(_symbol(trade.get("ticker") or ""))
    if not t:
        return 0
    exs, cash = t
    have = {e.get("ex") for e in trade.get("dividends") or []}
    lo = bisect_right(exs, entry.astimezone(ET).date().isoformat())   # ex-dates after the entry's ET day
    added = 0
    for i in range(lo, len(exs)):
        ex = date.fromisoformat(exs[i])
        st = start_of(ex)
        if st > end:
            break
        if st <= entry or exs[i] in have:
            continue
        price = float(trade.get("entry_price") or 0.0)
        if price <= 0:
            return added
        from src.data.intraday_store import split_factor_between
        through = (exit_at or now).astimezone(ET).date()
        k = split_factor_between(str(trade.get("ticker") or "").upper(), ex, through)
        if k is None:
            logger.warning(f"[dividends] {trade.get('ticker')} ex {exs[i]}: split data unreadable — event waits")
            return added
        c = float(cash[i]) * float(k)
        trade.setdefault("dividends", []).append({
            "ex": exs[i], "cash": round(float(cash[i]), 6), "split_k": float(k), "cash_ps": round(c, 6),
            "frac": c / price, "at": st.isoformat(timespec="seconds"), "recorded_at": now.isoformat(timespec="seconds")})
        added += 1
    if added:
        trade["dividends"].sort(key=lambda e: e["ex"])
        side = "owes" if str(trade.get("action") or "BUY").upper() == "SELL" else "receives"
        logger.info(f"[dividends] {trade.get('ticker')} {side} {added} dividend(s): "
                    + ", ".join(f"{e['ex']} ${e['cash_ps']:.4f}/sh" for e in trade["dividends"][-added:]))
    return added


def frac_through(trade: dict, end_iso=None) -> float:
    """The stored events' total fraction of the entry price due by ``end_iso`` (default: every stored event) —
    POSITIVE; the caller signs it (a long receives, a short pays)."""
    evs = trade.get("dividends") or []
    if not evs or not eligible(trade):
        return 0.0
    end = _instant(end_iso)
    tot = 0.0
    for e in evs:
        at = _instant(e.get("at"))
        if end is None or (at is not None and at <= end):
            tot += float(e.get("frac") or 0.0)
    return tot


def finalize_closed(trades: List[dict], now: Optional[datetime] = None) -> int:
    """Closed live-book trades exited in the last `FINAL_AFTER_DAYS` days: any event a late record makes due inside
    their hold is added (a record the nightly refresh brought after the exit). Returns the number added."""
    if not enabled():
        return 0
    now = now or datetime.now(timezone.utc)
    added = 0
    for t in trades:
        if t.get("status") != "CLOSED" or not eligible(t):
            continue
        x = _instant(t.get("exit_datetime"))
        if x is None or (now - x).days > FINAL_AFTER_DAYS:
            continue
        added += update(t, t.get("exit_datetime"), now=now)
    return added


def nav_events(trade: dict) -> List[Tuple[date, float]]:
    """(ex-date, fraction of the entry price) of the stored events — the daily NAV books each on its ex-date."""
    if not eligible(trade):
        return []
    return [(date.fromisoformat(e["ex"]), float(e.get("frac") or 0.0)) for e in trade.get("dividends") or []]


def main() -> None:
    import argparse
    import json
    ap = argparse.ArgumentParser(description="the ledger's dividend table")
    ap.add_argument("--ticker", help="the ex-dates and cash of one name")
    a = ap.parse_args()
    t = table()
    if a.ticker:
        exs, cash = t.get(_symbol(a.ticker), ([], []))
        print(json.dumps([{"ex": e, "cash": c} for e, c in zip(exs, cash)][-12:], indent=1))
    else:
        print(json.dumps({"names": len(t), "file": str(DIV_PATH), "exists": os.path.exists(DIV_PATH)}, indent=1))


if __name__ == "__main__":
    main()
