"""BORROW HISTORY — IBKR's lendable shares and borrow fee per name per day, from
OUR OWN archive of IBKR's public short-stock file (user directive 2026-10-02: "Can
we not reuse iBorrowDesk and just ingest our own from ibkr borrow data?") — the
availability every short backtest assumes and the live short's borrow gate reads.

Every tick downloads IBKR's file (`src/data/ibkr_borrow.py`; IBKR republishes it
about every 15 minutes, day and night) and archives it raw:
`data/ibkr_borrow/<ET date>/<HHMMSS>.txt.gz`, named by the file's own `#BOF` time
(2026-09-25 onward). `archive_daily` summarises each finished ET calendar day ONCE:
open / high / low / CLOSE of the shares available and of the fee, plus the closing
rebate — the close is the day's LAST file (the evening's last tick), the freshest
state the next session's 08:30 cutoff can read. A day is summarised only after it
ends (`summarise_archive` never touches today), so a row dated D is known from D+1
(`deep_features.LAG_DAYS["borrow"]`). The archive holds one file per tick, not
every 15-minute update: its high / low can miss a short dip — the features read
the close only.

`borrow_daily.parquet` (`build_daily`): ticker, date, source (``ibkr``), available,
fee, rebate, open_/high_/low_ available and fee.

No third-party copy is read. iBorrowDesk (which samples the same file) was tried
on 2026-10-02: its free per-ticker API banned the IP after ~100 requests, and its
daily close matched this archive's file for file on the overlap.

    python -m src.data.deep.borrow --build
"""
from __future__ import annotations

import gzip
from datetime import date
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd
from loguru import logger

from src.data import deep as _deep
from src.data.deep import family_dir, write_parquet

ARCHIVE_FAMILY = "borrow_ibkr"           # one part per summarised archive day
DAILY_FILE = "borrow_daily.parquet"
CAP = 10_000_000                         # IBKR prints ">10000000" past this; capped here
DAILY_COLUMNS = ["ticker", "date", "source", "available", "fee", "rebate", "open_available", "high_available",
                 "low_available", "open_fee", "high_fee", "low_fee"]


def our_symbol(sym: str) -> str:
    """IBKR's spelling -> ours (`BRK B` -> `BRK-B`)."""
    return str(sym).strip().upper().replace(" ", "-").replace(".", "-")


def _cap(v):
    a = pd.to_numeric(pd.Series(v), errors="coerce").to_numpy(float)
    return np.minimum(a, CAP)


def archive_files(day: date, base: Optional[Path] = None) -> List[Path]:
    """The archive's files of ET date ``day``, in time order."""
    from src.data import ibkr_borrow as ib
    d = (base or ib.ARCHIVE_DIR) / day.isoformat()
    return sorted(d.glob("*.txt.gz")) if d.exists() else []


def archive_daily(day: date, base: Optional[Path] = None, min_files: int = 3) -> pd.DataFrame:
    """One ET calendar day of the archive: open / high / low / close per symbol over
    the day's files, the close being the day's last file. Empty when the archive
    holds fewer than ``min_files`` that day."""
    from src.data import ibkr_borrow as ib
    files = archive_files(day, base)
    if len(files) < min_files:
        return pd.DataFrame(columns=DAILY_COLUMNS)
    seq: List[pd.DataFrame] = []
    for k, f in enumerate(files):
        with gzip.open(f, "rt", encoding="utf-8", errors="replace") as fh:
            _, table = ib.parse(fh.read())
        seq.append(pd.DataFrame({"sym": list(table),
                                 "available": [b.available for b in table.values()],
                                 "fee": [b.fee_pct for b in table.values()],
                                 "rebate": [b.rebate_pct for b in table.values()]}).assign(k=k))
    a = pd.concat(seq, ignore_index=True)
    a["available"] = _cap(a["available"])
    a = a.sort_values(["sym", "k"])
    g = a.groupby("sym", sort=False)
    out = pd.DataFrame({
        "available": g["available"].last(), "fee": g["fee"].last(), "rebate": g["rebate"].last(),
        "open_available": g["available"].first(), "high_available": g["available"].max(),
        "low_available": g["available"].min(), "open_fee": g["fee"].first(), "high_fee": g["fee"].max(),
        "low_fee": g["fee"].min()}).reset_index()
    out["ticker"] = out["sym"].map(our_symbol)
    out["date"] = day.isoformat()
    out["source"] = "ibkr"
    out = out.drop_duplicates("ticker", keep="first")
    return out[DAILY_COLUMNS]


def archive_days(base: Optional[Path] = None) -> List[date]:
    from src.data import ibkr_borrow as ib
    root = base or ib.ARCHIVE_DIR
    out = []
    for d in sorted(root.iterdir()) if root.exists() else []:
        try:
            out.append(date.fromisoformat(d.name))
        except ValueError:
            continue
    return out


def summarise_archive(base: Optional[Path] = None, today: Optional[date] = None) -> dict:
    """Every finished archive day not yet summarised -> ``borrow_ibkr/parts/<date>``
    (once per day: an archived file never changes). A day still in progress
    (``today``, default the ET date now) is never summarised; one with too few
    files is recorded empty."""
    from src.data.deep import run_keys
    today = today or pd.Timestamp.now(tz="America/New_York").date()
    keys = [d.isoformat() for d in archive_days(base) if d < today]
    return run_keys(ARCHIVE_FAMILY, keys, lambda k: archive_daily(date.fromisoformat(k), base), workers=2,
                    retry_failed=True, empty_is_done=True)


def build_daily(base: Optional[Path] = None, today: Optional[date] = None) -> int:
    """The summarised archive days -> ``borrow_daily.parquet``."""
    from src.data.deep import read_parquet
    import glob
    summarise_archive(base, today)
    parts = sorted(glob.glob(str(family_dir(ARCHIVE_FAMILY) / "parts" / "*.parquet")))
    days = [x for x in (read_parquet(Path(p)) for p in parts) if len(x)]
    df = pd.concat(days, ignore_index=True)[DAILY_COLUMNS] if days else pd.DataFrame(columns=DAILY_COLUMNS)
    df = df.sort_values(["ticker", "date"]).drop_duplicates(["ticker", "date"], keep="last")
    n = write_parquet(df.reset_index(drop=True), _deep.DEEP_DIR / DAILY_FILE)
    logger.info(f"[deep.borrow] {DAILY_FILE}: {n:,} rows from our archive, {df['ticker'].nunique():,} names, "
                f"{df['date'].nunique() if len(df) else 0} days")
    return n


if __name__ == "__main__":                              # pragma: no cover
    import argparse
    ap = argparse.ArgumentParser(description="borrow history: borrow_daily.parquet")
    ap.add_argument("--build", action="store_true")
    a = ap.parse_args()
    logger.add("logs/deep_ingest_borrow.log", rotation="1 day", retention="14 days", level="INFO", enqueue=True)
    if a.build:
        build_daily()
