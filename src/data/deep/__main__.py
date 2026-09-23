"""CLI — ``python -m src.data.deep [family ...] [options]``.

Families run in ORTHOGONALITY order when none is named (most information
outside price first):

  sec_filings companyfacts form345 bars30m_full short_interest short_volume
  dividends splits ticker_details ipos context ftd wiki yf quiver delisted

``--status`` prints the manifest / parquet counts per family and exits.

This is the INITIAL ingest (resumable through the manifest, which skips every
key already done). Keeping the store current is ``python -m src.data.deep.refresh``
(``refresh.py``): the nightly incremental tail the scheduler runs.
"""
from __future__ import annotations

import argparse
import sys
import time
from datetime import date

from loguru import logger

from src.data.deep import (DEEP_DIR, consolidate, deep_universe, run_keys, status,
                           write_parquet)

ORDER = ["sec_filings", "companyfacts", "form345", "form13f", "bars30m_full", "short_interest",
         "short_volume", "dividends", "splits", "ticker_details", "ipos", "context", "ftd",
         "wiki", "yf", "quiver", "polygon_news", "delisted"]


def _universe(a) -> list:
    if a.tickers:
        return [t.strip().upper() for t in a.tickers.split(",") if t.strip()]
    u = deep_universe(refresh=a.refresh_universe)
    return u[: a.limit] if a.limit else u


def run_family(fam: str, a) -> dict:
    w, b, rf = a.workers, a.budget_seconds, not a.no_retry_failed
    if fam == "sec_filings":
        from src.data.deep import sec
        r = run_keys(fam, _universe(a), sec.filings_for_ticker, workers=min(w, 4), budget_seconds=b, retry_failed=rf)
        consolidate(fam)
    elif fam == "companyfacts":
        from src.data.deep import sec
        r = run_keys(fam, _universe(a), sec.facts_for_ticker, workers=min(w, 4), budget_seconds=b, retry_failed=rf)
        consolidate(fam)
    elif fam == "form345":
        from src.data.deep import form345
        keys = form345.quarter_keys()
        r = run_keys(fam, keys[: a.limit] if a.limit else keys, form345.fetch_quarter,
                     workers=min(w, 2), budget_seconds=b, retry_failed=rf)
        consolidate(fam)
    elif fam == "bars30m_full":
        from src.data.deep import polygon_deep
        r = run_keys(fam, _universe(a), polygon_deep.bars30m_full, workers=w, budget_seconds=b, retry_failed=rf)
    elif fam == "short_interest":
        from src.data.deep import polygon_deep
        r = run_keys(fam, _universe(a), polygon_deep.short_interest, workers=w, budget_seconds=b, retry_failed=rf)
        consolidate(fam)
    elif fam == "short_volume":
        from src.data.deep import polygon_deep
        r = run_keys(fam, _universe(a), polygon_deep.short_volume, workers=w, budget_seconds=b, retry_failed=rf)
        consolidate(fam)
    elif fam == "dividends":
        from src.data.deep import polygon_deep
        keys = polygon_deep.month_keys("2000-01-01")
        keys = keys[-a.limit:] if a.limit else keys
        r = run_keys(fam, keys, polygon_deep.dividends_month, workers=min(w, 4), budget_seconds=b,
                     retry_failed=rf, part_name=lambda k: k.replace("..", "_"))
        consolidate(fam)
    elif fam == "splits":
        from src.data.deep import polygon_deep
        keys = polygon_deep.month_keys("2000-01-01")
        keys = keys[-a.limit:] if a.limit else keys
        r = run_keys(fam, keys, polygon_deep.splits_month, workers=min(w, 4), budget_seconds=b,
                     retry_failed=rf, part_name=lambda k: k.replace("..", "_"))
        consolidate(fam)
    elif fam == "ticker_details":
        from src.data.deep import polygon_deep
        r = run_keys(fam, _universe(a), polygon_deep.ticker_details, workers=w, budget_seconds=b, retry_failed=rf)
        consolidate(fam)
    elif fam == "polygon_news":
        from src.data.deep import polygon_deep
        r = run_keys(fam, _universe(a), polygon_deep.polygon_news, workers=w, budget_seconds=b, retry_failed=rf)
        consolidate(fam)
    elif fam == "form13f":
        from src.data.deep import form13f
        keys = sorted(form13f.file_links().keys())
        r = run_keys(fam, keys[: a.limit] if a.limit else keys, form13f.fetch_quarter,
                     workers=min(w, 2), budget_seconds=b, retry_failed=rf)
        consolidate(fam)
    elif fam == "ipos":
        from src.data.deep import polygon_deep
        df = polygon_deep.ipos()
        n = write_parquet(df, DEEP_DIR / "ipos.parquet") if len(df) else 0
        r = {"family": fam, "rows": n}
    elif fam == "context":
        from src.data.deep import context
        r = {"family": fam, "rows": 0}
        for name, fn in (("market_daily", context.market_daily), ("fred_vintages", context.fred_vintages),
                         ("fama_french", context.fama_french), ("dix", context.dix)):
            try:
                df = fn()
                n = write_parquet(df, DEEP_DIR / f"{name}.parquet") if len(df) else 0
                logger.info(f"[deep] context/{name}: {n:,} rows")
                r["rows"] += n
            except Exception as e:                       # noqa: BLE001
                logger.warning(f"[deep] context/{name} failed: {e}")
        years = [str(y) for y in range(2010, date.today().year + 1)]
        years = years[-a.limit:] if a.limit else years
        rc = run_keys("cot_tff", years, context.cot_tff_year, workers=2, budget_seconds=b, retry_failed=rf)
        consolidate("cot_tff")
        r["cot_tff"] = rc
    elif fam == "ftd":
        from src.data.deep import ftd
        keys = sorted(ftd.file_links().keys())
        r = run_keys(fam, keys[: a.limit] if a.limit else keys, ftd.fetch_file, workers=min(w, 3),
                     budget_seconds=b, retry_failed=rf)
        consolidate(fam)
    elif fam == "wiki":
        from src.data.deep import wiki
        m = wiki.mapping(refresh=a.refresh_universe)
        keys = [t for t in _universe(a) if t in m]
        logger.info(f"[deep] wiki: {len(keys)} of the universe have a Wikipedia article")
        r = run_keys(fam, keys, wiki.pageviews_for_ticker, workers=1, budget_seconds=b, retry_failed=rf)
        consolidate(fam)
    elif fam == "yf":
        from src.data.deep import yf_deep
        r = yf_deep.run(_universe(a), workers=1, budget_seconds=b, retry_failed=rf)
        for sub in ("yf_earnings", "yf_analyst", "yf_shares"):
            consolidate(sub)
    elif fam == "quiver":
        from src.data.deep import quiver_deep
        r = {"family": fam}
        for sub in quiver_deep.ENDPOINTS:
            r[sub] = run_keys(sub, _universe(a), quiver_deep.make_fetcher(sub), workers=min(w, 2),
                              budget_seconds=b, retry_failed=rf)
            consolidate(sub)
    elif fam == "delisted":
        from src.data.deep import polygon_deep
        lst = polygon_deep.delisted_tickers()
        n = write_parquet(lst, DEEP_DIR / "delisted.parquet") if len(lst) else 0
        logger.info(f"[deep] delisted list: {n:,} tickers")
        r = {"family": fam, "rows": n}
        if len(lst):
            recent = lst[(lst["delisted_utc"].fillna("") >= a.delisted_since) &
                         (lst["type"].isin(["CS", "ADRC"]))]
            keys = sorted(set(recent["ticker"].astype(str)))
            if a.limit:
                keys = keys[: a.limit]
            logger.info(f"[deep] delisted since {a.delisted_since}, CS/ADRC: {len(keys)} names")
            r["bars1d"] = run_keys("bars1d_delisted", keys, polygon_deep.bars1d, workers=w,
                                   budget_seconds=b, retry_failed=rf)
            if a.delisted_30m:
                r["bars30m"] = run_keys("bars30m_delisted", keys, polygon_deep.bars30m_full,
                                        workers=w, budget_seconds=b, retry_failed=rf)
    else:
        raise SystemExit(f"unknown family {fam!r}; choose from {ORDER}")
    return r


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description="deep history store ingestion")
    ap.add_argument("families", nargs="*", help=f"any of {ORDER}; default = all, in that order")
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--budget-seconds", type=float, default=0.0, help="per family; 0 = unbounded")
    ap.add_argument("--limit", type=int, default=0, help="first N keys only (smoke test)")
    ap.add_argument("--tickers", default="", help="comma list overriding the universe")
    ap.add_argument("--refresh-universe", action="store_true")
    ap.add_argument("--no-retry-failed", action="store_true")
    ap.add_argument("--delisted-since", default="2021-01-01")
    ap.add_argument("--delisted-30m", action="store_true", help="also fetch 30-minute bars for delisted names")
    ap.add_argument("--status", action="store_true")
    ap.add_argument("--consolidate-only", action="store_true")
    ap.add_argument("--log-tag", default="", help="suffix for logs/deep_ingest_<tag>.log (one per process)")
    a = ap.parse_args(argv)

    tag = a.log_tag or (a.families[0] if a.families else "all")
    logger.add(f"logs/deep_ingest_{tag}.log", rotation="1 day", retention="14 days", level="INFO",
               enqueue=True)
    if a.status:
        df = status()
        print(f"universe: {len(deep_universe())} tickers")
        print(df.to_string(index=False) if len(df) else "(empty)")
        return
    fams = a.families or ORDER
    if a.consolidate_only:
        for fam in fams:
            consolidate(fam)
        return
    t0 = time.time()
    for fam in fams:
        logger.info(f"[deep] ===== {fam} =====")
        try:
            r = run_family(fam, a)
            logger.info(f"[deep] {fam}: {r}")
        except Exception as e:                           # noqa: BLE001
            logger.exception(f"[deep] {fam} FAILED: {e}")
    logger.info(f"[deep] all done in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main(sys.argv[1:])
