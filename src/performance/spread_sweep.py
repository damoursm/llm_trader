"""EOD IBKR BID_ASK sweep — MEASURED quoted spreads for the liquidity forecast.

``reqHistoricalData(whatToShow="BID_ASK")`` returns bars whose open/close are
the time-average bid/ask, so ``(close − open)/2/mid`` is the day's time-average
quoted HALF-spread — a direct measurement that supersedes the Corwin–Schultz
estimate wherever it exists (probed 2026-08-31: works on this account,
~1 s/ticker; AAPL 0.6 bp vs CS-forecast 10 bp — CS also UNDERSTATES wide names,
LX measured 43.3 vs CS 19.7). ``liquidity_forecast`` consumes the store as its
primary structural layer; CS/AR remain the fallback for names IBKR can't serve.

Mechanics
---------
* Population: distinct tickers from the last 7 days of ``signals`` plus every
  open trade, restricted to the Gate-4 recipe (stored price ≥ $5, cache-only
  20-bar mean dollar volume ≥ $5M) — the pool the book can actually trade.
* Rotation: never-swept names first, then oldest ``fetched_at`` — so a paced or
  budget-cut run resumes where it left off and the pool refreshes over a few
  nights regardless of budget.
* Pacing: one request per ticker (1 week of DAILY bars, RTH only) with
  ``spread_sweep_sleep_seconds`` between requests; IB's historical pacing
  (~60 req/10 min, BID_ASK counting double) is handled by aborting after
  ``_MAX_CONSECUTIVE_FAILURES`` — the rotation makes an aborted run harmless.
* Store: ``cache/ibkr_spread.json`` (atomic replace), one entry per ticker:
  ``{half_bps, date, bars, fetched_at}``. ``half_bps`` is the MEDIAN over the
  week's daily bars (robust to one weird day).
* Runs as a SUBPROCESS from EOD maintenance (``enable_eod_spread_sweep``) with
  its own clientId (``ibkr_client_id + 50``) — ib_async needs an event loop the
  EOD background thread doesn't have, and a second in-process connection would
  contend with the tick's broker session.

CLI:  python -m src.performance.spread_sweep [--budget-seconds N] [--sleep S]
          [--limit N] [--probe]
      (--probe: one-shot live get_quote health check — the RTH re-probe.)

Fail-soft everywhere: no gateway / refused connects / per-ticker errors leave
the store as it was; ``liquidity_forecast`` treats a missing or stale entry as
"no IBKR view" and falls back to CS.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np

from loguru import logger  # project configures loguru sinks only

from config.settings import settings
from src.data.cache import CACHE_DIR

SPREAD_STORE_PATH = CACHE_DIR / "ibkr_spread.json"
_MAX_CONSECUTIVE_FAILURES = 5
_SWEEP_CLIENT_ID_OFFSET = 50      # ibkr_client_id + this — never the tick's id


# ── store ───────────────────────────────────────────────────────────────────

def load_spread_store(path: Optional[Path] = None) -> Dict[str, dict]:
    """The persisted {ticker: {half_bps, date, bars, fetched_at}} map; {} when
    absent/corrupt (the forecast then simply has no IBKR layer)."""
    p = path or SPREAD_STORE_PATH
    try:
        with open(p, "r", encoding="utf-8") as fh:
            data = json.load(fh)
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def save_spread_store(store: Dict[str, dict], path: Optional[Path] = None) -> None:
    """Atomic write (temp + replace) so a concurrent reader never sees a torn
    file — the scheduler process reads this while the EOD subprocess writes."""
    p = path or SPREAD_STORE_PATH
    tmp = Path(str(p) + ".tmp")
    p.parent.mkdir(exist_ok=True)
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(store, fh)
    os.replace(tmp, p)


def half_bps_from_bars(bars: Sequence) -> Optional[float]:
    """Median time-average half-spread (bp) over BID_ASK bars.

    BID_ASK bar convention: ``open`` = time-average bid, ``close`` = time-
    average ask (``high``/``low`` are the extremes). Bars with a crossed or
    non-positive book are skipped; None when nothing usable remains."""
    vals: List[float] = []
    for bar in bars or []:
        try:
            b, a = float(bar.open), float(bar.close)
        except (TypeError, ValueError, AttributeError):
            continue
        mid = (a + b) / 2.0
        if b > 0 and a >= b and mid > 0:
            vals.append((a - b) / 2.0 / mid * 1e4)
    return float(np.median(vals)) if vals else None


# ── population ──────────────────────────────────────────────────────────────

def _sweep_population() -> List[str]:
    """Gate-4-shaped tickers worth measuring: last-7d signals ∪ open trades,
    price ≥ $5 (stored panel price) and cache-only 20-bar mean $vol ≥ $5M."""
    from src.db import repo
    tickers: Dict[str, float] = {}
    try:
        df = repo.fetch_df(
            "SELECT ticker, max(price) AS price FROM signals "
            "WHERE signal_date >= ? GROUP BY ticker",
            [(date.fromordinal(date.today().toordinal() - 7)).isoformat()])
        for r in df.itertuples():
            tk = str(r.ticker).upper()
            # Equities only: the panel also carries macro-context symbols
            # (futures "ES=F", indices "^VIX") that have no SMART stock
            # contract — first sweep burned 3 slots on them (Error 200).
            if "=" in tk or tk.startswith("^"):
                continue
            if r.price and float(r.price) >= float(settings.trade_min_price):
                tickers[tk] = float(r.price)
    except Exception as e:
        logger.warning(f"[spread_sweep] signals population unavailable: {e}")
    try:
        for t in repo.load_trades():
            if t.get("status") == "OPEN" and t.get("ticker"):
                tickers.setdefault(str(t["ticker"]).upper(), 1e9)  # always keep
    except Exception:
        pass

    min_dv = float(settings.trade_min_dollar_volume)
    out: List[str] = []
    from src.data.cache import load_ohlcv
    import pandas as pd
    for tk, px in tickers.items():
        if px >= 1e9:                      # open positions bypass the ADV cut
            out.append(tk)
            continue
        try:
            df = load_ohlcv(tk)
            if df is None or df.empty or "Volume" not in df.columns:
                continue
            c = pd.to_numeric(df["Close"], errors="coerce")
            v = pd.to_numeric(df["Volume"], errors="coerce")
            dv = (c * v).tail(20).mean()
            if dv == dv and float(dv) >= min_dv:
                out.append(tk)
        except Exception:
            continue
    return out


def _rotation_order(tickers: Sequence[str], store: Dict[str, dict]) -> List[str]:
    """Never-swept names first, then stalest ``fetched_at`` first."""
    def key(tk: str):
        e = store.get(tk) or {}
        return (0, "") if not e.get("fetched_at") else (1, str(e["fetched_at"]))
    return sorted(dict.fromkeys(t.upper() for t in tickers), key=key)


# ── the sweep ───────────────────────────────────────────────────────────────

def run_sweep(budget_seconds: Optional[float] = None,
              sleep_seconds: Optional[float] = None,
              limit: Optional[int] = None) -> dict:
    """Fetch BID_ASK daily bars for as much of the population as the budget
    allows; merge into the store. Returns a summary dict (also on failure)."""
    if not getattr(settings, "enable_liquidity_forecast", True):
        return {"skipped": "liquidity forecast disabled"}
    mode = str(getattr(settings, "broker_mode", "off") or "off")
    if not mode.startswith("ibkr"):
        return {"skipped": f"broker_mode={mode} (needs an IBKR gateway)"}

    budget = float(budget_seconds if budget_seconds is not None
                   else settings.spread_sweep_budget_seconds)
    pause = float(sleep_seconds if sleep_seconds is not None
                  else settings.spread_sweep_sleep_seconds)

    store = load_spread_store()
    todo = _rotation_order(_sweep_population(), store)
    if limit:
        todo = todo[:int(limit)]
    summary = {"population": len(todo), "fetched": 0, "empty": 0, "errors": 0,
               "elapsed_s": 0.0, "connected": False}
    if not todo:
        return summary

    from src.broker.ibkr import IBKRBroker
    b = IBKRBroker(client_id=int(settings.ibkr_client_id) + _SWEEP_CLIENT_ID_OFFSET)
    t0 = time.time()
    try:
        if not b.connect():
            summary["skipped"] = "gateway connect failed"
            return summary
        summary["connected"] = True
        consec_fail = 0
        now_iso = datetime.now(timezone.utc).isoformat(timespec="seconds")
        for i, tk in enumerate(todo):
            if (time.time() - t0) >= budget:
                summary["stopped"] = f"budget ({budget:.0f}s) at {i}/{len(todo)}"
                break
            try:
                bars = b._ib.reqHistoricalData(
                    b._qualify(tk), endDateTime="", durationStr="1 W",
                    barSizeSetting="1 day", whatToShow="BID_ASK",
                    useRTH=1, formatDate=1)
                half = half_bps_from_bars(bars)
                if half is not None:
                    store[tk] = {
                        "half_bps": round(half, 2),
                        "date": str(bars[-1].date)[:10],
                        "bars": len(bars),
                        "fetched_at": now_iso,
                    }
                    summary["fetched"] += 1
                    consec_fail = 0
                else:
                    summary["empty"] += 1
                    consec_fail += 1
            except Exception as e:
                summary["errors"] += 1
                consec_fail += 1
                logger.debug(f"[spread_sweep] {tk} failed: {e}")
            if consec_fail >= _MAX_CONSECUTIVE_FAILURES:
                # Pacing violation / feed outage: the rotation resumes tomorrow.
                summary["stopped"] = f"{consec_fail} consecutive failures at {i}/{len(todo)}"
                break
            if pause > 0:
                time.sleep(pause)
        if summary["fetched"]:
            save_spread_store(store)
    finally:
        try:
            b.disconnect()
        except Exception:
            pass
    summary["elapsed_s"] = round(time.time() - t0, 1)
    summary["store_size"] = len(store)
    return summary


# ── one-shot live-quote health probe (the RTH re-probe) ─────────────────────

def probe_quotes(tickers: Sequence[str] = ("AAPL", "NVDA", "SPY", "FUN", "SOFI"),
                 ) -> dict:
    """Connect with the sweep clientId and try ``get_quote`` on liquid names —
    answers "is the live top-of-book feed serving this session?" (it refused
    with error 10089 overnight 2026-08-31; the free feed covers ~08:00–17:00
    ET). The ib_async→loguru bridge makes any refusal visible in the log."""
    from src.broker.ibkr import IBKRBroker
    b = IBKRBroker(client_id=int(settings.ibkr_client_id) + _SWEEP_CLIENT_ID_OFFSET)
    out = {"connected": False, "quotes": {}}
    try:
        if not b.connect():
            return out
        out["connected"] = True
        for tk in tickers:
            q = None
            try:
                q = b.get_quote(tk)
            except Exception:
                pass
            out["quotes"][tk] = (
                {"bid": q.bid, "ask": q.ask,
                 "half_bps": round((q.ask - q.bid) / (q.ask + q.bid) * 1e4, 2)}
                if q else None)
    finally:
        try:
            b.disconnect()
        except Exception:
            pass
    ok = sum(1 for v in out["quotes"].values() if v)
    out["verdict"] = (f"{ok}/{len(out['quotes'])} names returned a two-sided "
                      f"book — live quote feed {'HEALTHY' if ok else 'NOT serving this session'}")
    return out


def main(argv: Optional[list] = None) -> None:
    import sys
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    ap = argparse.ArgumentParser(description="IBKR BID_ASK spread sweep / quote probe")
    ap.add_argument("--budget-seconds", type=float, default=None)
    ap.add_argument("--sleep", type=float, default=None)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--probe", action="store_true",
                    help="live get_quote health probe instead of the sweep")
    a = ap.parse_args(argv)
    if a.probe:
        res = probe_quotes()
        for tk, v in res.get("quotes", {}).items():
            print(f"  {tk:<6} {v if v else 'NO two-sided quote'}")
        print(res.get("verdict", "not connected"))
        return
    res = run_sweep(budget_seconds=a.budget_seconds, sleep_seconds=a.sleep,
                    limit=a.limit)
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
