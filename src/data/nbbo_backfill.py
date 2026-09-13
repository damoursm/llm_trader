"""Historical NBBO backfill — the point-in-time book the system HAD but never stored.

Live NBBO capture only began 2026-08-31 (`signals.exp_halfspread_bps`,
`broker_orders.bid_at_submit`), so every liquidity question asked over a window
longer than that has had to fall back on a STRUCTURAL estimate (Corwin–Schultz,
the IBKR time-average sweep, the class table). Polygon's consolidated quote tape
is entitled on this plan (verified 2026-09-02: `/v3/quotes` returns two-sided
books back at least a year with sub-second SIP timestamps), so the real book at
each past decision instant is recoverable.

What this fetches: for a list of ``(ticker, instant)`` points, the LAST NBBO at
or before that instant — the quote a run at that moment would have seen. Each
row carries ``age_s`` (how stale that quote was at the instant) so the consumer
can refuse a stale book exactly the way ``reconcile._quote_for`` refuses one past
120 s; a quote is never silently dropped for being old, because monitoring uses
want to see it and only order-pricing uses must not.

Store: ``cache/nbbo_history.parquet``, keyed ``(ticker, ts_ns)``, RESUMABLE —
re-running skips points already present, so a killed run costs only its last
chunk. This is a CACHE, not the database: the scheduler is the sole DuckDB
writer, so nothing here touches ``data/llm_trader.db``.

NOT wired into any live path. Standalone CLI::

    python -m src.data.nbbo_backfill --trades                 # every ledger entry + exit
    python -m src.data.nbbo_backfill --panel --days 45        # each Gate-4 name at its pre-close run
    python -m src.data.nbbo_backfill --summary
"""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple

import pandas as pd
from loguru import logger

from config.settings import settings
from src.data.polygon_client import _get, is_available, to_polygon_symbol

STORE = Path("cache") / "nbbo_history.parquet"
_COLUMNS = ["ticker", "ts_ns", "bid", "ask", "mid", "hs_bps", "sip_ts_ns", "age_s", "fetched_at"]
_CHUNK = 2000
_DEFAULT_WORKERS = 16


# --------------------------------------------------------------------------- fetch
def quote_at(ticker: str, ts_ns: int) -> Optional[dict]:
    """The last two-sided NBBO at or before ``ts_ns`` (epoch nanoseconds).

    None when the ticker has no quote tape (indices, futures), the book is
    one-sided or crossed, or the request fails. ``age_s`` is how old the quote
    already was at ``ts_ns`` — the caller judges freshness.
    """
    if not is_available() or not ticker or not ts_ns:
        return None
    sym = to_polygon_symbol(ticker)
    j = _get(f"/v3/quotes/{sym}",
             {"timestamp.lte": int(ts_ns), "order": "desc", "sort": "timestamp", "limit": 1})
    rows = (j or {}).get("results") or []
    if not rows:
        return None
    q = rows[0]
    try:
        bid, ask = float(q.get("bid_price") or 0), float(q.get("ask_price") or 0)
        if not (0 < bid <= ask):
            return None
        mid = (bid + ask) / 2.0
        sip = int(q.get("sip_timestamp") or 0)
        return {"ticker": ticker, "ts_ns": int(ts_ns), "bid": bid, "ask": ask, "mid": mid,
                "hs_bps": (ask - bid) / 2.0 / mid * 1e4,
                "sip_ts_ns": sip,
                "age_s": round(max(0.0, (int(ts_ns) - sip) / 1e9), 1) if sip else None,
                "fetched_at": datetime.now(timezone.utc).isoformat(timespec="seconds")}
    except (TypeError, ValueError):
        return None


# --------------------------------------------------------------------------- store
def load_store() -> pd.DataFrame:
    if STORE.exists():
        try:
            return pd.read_parquet(STORE)
        except Exception as exc:  # a truncated write must not brick the backfill
            logger.warning(f"[nbbo_backfill] unreadable store ({exc}) — starting empty")
    return pd.DataFrame(columns=_COLUMNS)


def _write(df: pd.DataFrame) -> None:
    STORE.parent.mkdir(parents=True, exist_ok=True)
    tmp = STORE.with_suffix(".parquet.tmp")
    df.to_parquet(tmp, index=False)
    tmp.replace(STORE)


def backfill(points: Sequence[Tuple[str, int]], workers: int = _DEFAULT_WORKERS,
             label: str = "") -> pd.DataFrame:
    """Fetch every ``(ticker, ts_ns)`` not already stored; append and return the store.

    Points already present are skipped, so this is safe to re-run and safe to
    kill: each chunk is written atomically before the next is fetched.
    """
    store = load_store()
    have = set(zip(store.ticker.astype(str), store.ts_ns.astype("int64"))) if len(store) else set()
    todo = [(str(t), int(ts)) for t, ts in points if (str(t), int(ts)) not in have]
    logger.info(f"[nbbo_backfill] {label}: {len(points):,} points, {len(todo):,} missing "
                f"({len(points) - len(todo):,} already stored)")
    if not todo:
        return store
    t0 = time.time()
    got: List[dict] = []
    miss = 0
    for start in range(0, len(todo), _CHUNK):
        chunk = todo[start:start + _CHUNK]
        with cf.ThreadPoolExecutor(max_workers=workers) as ex:
            for r in ex.map(lambda p: quote_at(p[0], p[1]), chunk):
                if r:
                    got.append(r)
                else:
                    miss += 1
        if got:
            store = pd.concat([store, pd.DataFrame(got)], ignore_index=True)
            store = store.drop_duplicates(subset=["ticker", "ts_ns"], keep="last")
            _write(store)
            got = []
        done = min(start + _CHUNK, len(todo))
        rate = done / max(1e-9, time.time() - t0)
        logger.info(f"[nbbo_backfill] {label}: {done:,}/{len(todo):,} ({rate:.1f}/s, "
                    f"{miss:,} without a book) — store {len(store):,} rows")
    return store


# --------------------------------------------------------------------------- points
def _iso_to_ns(s) -> Optional[int]:
    ts = pd.to_datetime(s, utc=True, errors="coerce")
    if ts is None or ts is pd.NaT or pd.isna(ts):
        return None
    return int(ts.value)


def _is_quotable(ticker: str) -> bool:
    """Indices (^VIX) and futures (GC=F) have no consolidated equity quote tape."""
    t = str(ticker)
    return bool(t) and not t.startswith("^") and "=" not in t


def points_from_trades() -> List[Tuple[str, int]]:
    """Every ledger trade's ENTRY instant, plus its EXIT instant when closed."""
    from src.db import repo
    df = pd.DataFrame(repo.load_trades())
    if df.empty:
        return []
    pts: List[Tuple[str, int]] = []
    for _, r in df.iterrows():
        tk = str(r.get("ticker") or "")
        if not _is_quotable(tk):
            continue
        for col in ("entry_datetime", "exit_datetime"):
            ns = _iso_to_ns(r.get(col))
            if ns:
                pts.append((tk, ns))
        if not r.get("entry_datetime"):  # legacy rows carry a date only
            ns = _iso_to_ns(r.get("entry_date"))
            if ns:
                pts.append((tk, ns))
    return sorted(set(pts))


def _gate4_pool(days: int) -> pd.DataFrame:
    """Panel rows that would clear Gate 4 (price + 20d dollar volume), cache-only."""
    from src.db import repo
    from src.data.cache import load_ohlcv
    sig = repo.fetch_df(
        "SELECT signal_date, ticker, generated_at, price FROM signals "
        "WHERE signal_date >= (CURRENT_DATE - INTERVAL (?) DAY)::VARCHAR", [days])
    if sig.empty:
        return sig
    sig["signal_date"] = sig.signal_date.astype(str).str[:10]
    adv: dict = {}

    def _adv(tk: str) -> float:
        if tk not in adv:
            val = 0.0
            try:
                from src.data.liquidity import dollar_volume   # production's own Gate-4 basis
                dv = dollar_volume(load_ohlcv(tk))
                val = float(dv) if dv is not None else 0.0
            except Exception:
                pass
            adv[tk] = val
        return adv[tk]

    px = pd.to_numeric(sig.price, errors="coerce")
    keep = (px >= float(settings.trade_min_price)) & sig.ticker.map(
        lambda t: _adv(str(t)) >= float(settings.trade_min_dollar_volume))
    return sig[keep & sig.ticker.map(_is_quotable)].copy()


def points_from_panel(days: int = 45, mode: str = "preclose") -> List[Tuple[str, int]]:
    """One point per (signal_date, ticker) at a chosen run of that day.

    ``preclose`` — the last run generated before 16:00 ET (the honest-anchor
    choice: its price and its label share a basis). ``lastrun`` — the day's last
    run, whatever session it fell in. ``allruns`` — every run (expensive).
    """
    pool = _gate4_pool(days)
    if pool.empty:
        return []
    et = pd.to_datetime(pool.generated_at, utc=True, errors="coerce").dt.tz_convert("America/New_York")
    pool = pool.assign(_et=et)
    if mode == "preclose":
        pre = pool[(pool._et.dt.hour < 16) & (pool._et.dt.strftime("%Y-%m-%d") == pool.signal_date)]
        pick = pre.sort_values("generated_at").groupby(["signal_date", "ticker"], as_index=False).tail(1)
        missing = pool.merge(pick[["signal_date", "ticker"]], on=["signal_date", "ticker"],
                             how="left", indicator=True).query("_merge=='left_only'")
        fallback = missing.sort_values("generated_at").groupby(["signal_date", "ticker"], as_index=False).tail(1)
        pick = pd.concat([pick, fallback], ignore_index=True)
    elif mode == "lastrun":
        pick = pool.sort_values("generated_at").groupby(["signal_date", "ticker"], as_index=False).tail(1)
    else:
        pick = pool
    pts = [(str(t), ns) for t, ns in zip(pick.ticker, pick.generated_at.map(_iso_to_ns)) if ns]
    return sorted(set(pts))


# --------------------------------------------------------------------------- report
def summary() -> None:
    df = load_store()
    if df.empty:
        print("nbbo_history: empty")
        return
    ts = pd.to_datetime(df.ts_ns, unit="ns", utc=True)
    df = df.assign(day=ts.dt.tz_convert("America/New_York").dt.strftime("%Y-%m-%d"))
    fresh = df[df.age_s.fillna(1e9) <= 120]
    print(f"nbbo_history: {len(df):,} quotes / {df.ticker.nunique():,} tickers / "
          f"{df.day.nunique()} days ({df.day.min()} -> {df.day.max()})")
    print(f"  fresh (<=120s stale): {len(fresh):,} ({len(fresh)/len(df):.1%}); "
          f"median half-spread {df.hs_bps.median():.2f} bp (fresh {fresh.hs_bps.median():.2f} bp); "
          f"share <12bp {(df.hs_bps < 12).mean():.3f}")
    per_day = df.groupby("day").agg(n=("hs_bps", "size"), med=("hs_bps", "median"),
                                    lt12=("hs_bps", lambda s: (s < 12).mean()),
                                    med_age=("age_s", "median"))
    print(per_day.tail(15).round(3).to_string())


def main(argv: Optional[Sequence[str]] = None) -> None:
    ap = argparse.ArgumentParser(description="Backfill historical NBBO from Polygon")
    ap.add_argument("--trades", action="store_true", help="every ledger entry/exit instant")
    ap.add_argument("--panel", action="store_true", help="each Gate-4 name at one run per day")
    ap.add_argument("--days", type=int, default=45)
    ap.add_argument("--mode", choices=("preclose", "lastrun", "allruns"), default="preclose")
    ap.add_argument("--workers", type=int, default=_DEFAULT_WORKERS)
    ap.add_argument("--limit", type=int, default=0, help="cap the number of points (smoke test)")
    ap.add_argument("--summary", action="store_true")
    args = ap.parse_args(argv)

    if args.summary and not (args.trades or args.panel):
        summary()
        return
    if not is_available():
        print("Polygon is not configured — nothing to do.")
        return
    if args.trades:
        pts = points_from_trades()
        backfill(pts[:args.limit] if args.limit else pts, args.workers, "trades")
    if args.panel:
        pts = points_from_panel(args.days, args.mode)
        backfill(pts[:args.limit] if args.limit else pts, args.workers, f"panel:{args.mode}")
    summary()


if __name__ == "__main__":
    main()
