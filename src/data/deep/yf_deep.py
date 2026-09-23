"""yfinance per-ticker histories — three tables per name, one pass.

* ``yf_earnings`` — earnings dates with EPS estimate / reported / surprise
  (up to 100 events, AAPL back to 2005). The estimate is Yahoo's CURRENT
  vintage of the past consensus — a restated-consensus caveat the feature
  builder must remember; the event instant itself is sound.
* ``yf_analyst``  — every upgrade / downgrade / price-target action
  (AAPL: 971 rows from 2012). Date-level only, no time of day.
* ``yf_shares``   — shares outstanding history (from ~2015).

yfinance is 429-prone at universe scale, so this runs on ONE worker by default
with the repo's standard backoff (60 → 120 → 240 s, stop after three in a row).
"""
from __future__ import annotations

import time
from typing import Dict, Optional

import pandas as pd
from loguru import logger

from src.data.deep import Manifest, family_dir, write_parquet

_BACKOFF = [60, 120, 240]


def _is_rate_limit(e: Exception) -> bool:
    s = f"{type(e).__name__} {e}".lower()
    return "ratelimit" in s or "too many requests" in s or "429" in s


def fetch_ticker(ticker: str) -> Dict[str, pd.DataFrame]:
    import yfinance as yf
    t = yf.Ticker(ticker)
    out: Dict[str, pd.DataFrame] = {}
    # earnings dates
    try:
        ed = t.get_earnings_dates(limit=100)
    except Exception as e:                               # noqa: BLE001
        if _is_rate_limit(e):
            raise
        ed = None
    if ed is not None and len(ed):
        d = ed.copy()
        idx = pd.DatetimeIndex(d.index)
        et = idx.tz_convert("America/New_York") if idx.tz is not None else idx.tz_localize("America/New_York")
        d = d.reset_index(drop=True)
        d.columns = [str(c) for c in d.columns]
        frame = pd.DataFrame({
            "ticker": ticker.upper(),
            "event_ts": et.tz_convert("UTC").tz_localize(None),
            "event_et": [x.strftime("%Y-%m-%d %H:%M") for x in et],
            "eps_estimate": pd.to_numeric(d.get("EPS Estimate"), errors="coerce"),
            "eps_reported": pd.to_numeric(d.get("Reported EPS"), errors="coerce"),
            "surprise_pct": pd.to_numeric(d.get("Surprise(%)"), errors="coerce"),
        })
        out["yf_earnings"] = frame.sort_values("event_ts").reset_index(drop=True)
    # analyst actions
    try:
        ud = t.upgrades_downgrades
    except Exception as e:                               # noqa: BLE001
        if _is_rate_limit(e):
            raise
        ud = None
    if ud is not None and len(ud):
        d = ud.copy()
        d.index.name = "grade_date"
        d = d.reset_index()
        frame = pd.DataFrame({
            "ticker": ticker.upper(),
            "grade_date": pd.to_datetime(d["grade_date"], errors="coerce").dt.tz_localize(None)
            if getattr(pd.to_datetime(d["grade_date"], errors="coerce").dt, "tz", None) is not None
            else pd.to_datetime(d["grade_date"], errors="coerce"),
            "firm": d.get("Firm"), "to_grade": d.get("ToGrade"), "from_grade": d.get("FromGrade"),
            "action": d.get("Action"), "pt_action": d.get("priceTargetAction"),
            "pt_current": pd.to_numeric(d.get("currentPriceTarget"), errors="coerce"),
            "pt_prior": pd.to_numeric(d.get("priorPriceTarget"), errors="coerce"),
        })
        for c in ("firm", "to_grade", "from_grade", "action", "pt_action"):
            frame[c] = frame[c].astype(object).where(frame[c].notna(), None)
        out["yf_analyst"] = frame.sort_values("grade_date").reset_index(drop=True)
    # shares outstanding
    try:
        sh = t.get_shares_full(start="2005-01-01")
    except Exception as e:                               # noqa: BLE001
        if _is_rate_limit(e):
            raise
        sh = None
    if sh is not None and len(sh):
        s = pd.Series(sh)
        idx = pd.DatetimeIndex(s.index)
        if idx.tz is not None:
            idx = idx.tz_convert("UTC").tz_localize(None)
        frame = pd.DataFrame({"ticker": ticker.upper(), "date": idx.date.astype(str),
                              "shares": pd.to_numeric(s.values, errors="coerce")})
        out["yf_shares"] = frame.drop_duplicates("date", keep="last").reset_index(drop=True)
    return out


def run(tickers, workers: int = 1, budget_seconds: float = 0.0, retry_failed: bool = True,
        force: bool = False) -> dict:
    """Sequential-by-default sweep writing three part dirs; manifest 'yf'.
    ``force`` refetches every ticker (the weekly refresh — yfinance returns
    the whole history and restates past consensus, so the part is
    overwritten), stalest first so a budget stop continues next time."""
    man = Manifest("yf")
    if force:
        todo = sorted(tickers, key=lambda t: str((man.done.get(t) or {}).get("at", "")))
    else:
        todo = man.pending(tickers, retry_failed=retry_failed)
    dirs = {k: family_dir(k) / "parts" for k in ("yf_earnings", "yf_analyst", "yf_shares")}
    t0 = time.time()
    n_ok = n_fail = 0
    strikes = 0
    stopped = False
    logger.info(f"[deep.yf] {len(todo)} pending of {len(tickers)} ({len(man.done)} done)")
    for i, tk in enumerate(todo):
        if budget_seconds and (time.time() - t0) > budget_seconds:
            logger.info("[deep.yf] budget stop")
            stopped = True
            break
        try:
            res = fetch_ticker(tk)
            strikes = 0
        except Exception as e:                           # noqa: BLE001
            if _is_rate_limit(e):
                wait = _BACKOFF[min(strikes, len(_BACKOFF) - 1)]
                strikes += 1
                logger.warning(f"[deep.yf] rate limited on {tk}; sleeping {wait}s (strike {strikes})")
                time.sleep(wait)
                if strikes >= len(_BACKOFF):
                    logger.error("[deep.yf] three consecutive rate limits — stopping this run")
                    stopped = True
                    break
                continue
            man.mark_failed(tk, repr(e))
            n_fail += 1
            continue
        rows = 0
        for fam, df in res.items():
            if df is not None and len(df):
                rows += write_parquet(df, dirs[fam] / f"{tk}.parquet")
        man.mark_done(tk, rows, note=",".join(sorted(res.keys())))
        n_ok += 1
        if (i + 1) % 50 == 0:
            logger.info(f"[deep.yf] {i + 1}/{len(todo)} ({n_fail} failed) {time.time() - t0:.0f}s")
    man.save()
    return {"family": "yf", "pending": len(todo), "ok": n_ok, "failed": n_fail,
            "seconds": time.time() - t0, "budget_stop": stopped}
