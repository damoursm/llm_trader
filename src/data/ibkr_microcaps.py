"""IBKR'S U.S. MICROCAP LIST, ARCHIVED DAILY (from 2026-10-23; PREREG190, the user: "Yes please add the two productions
steps").

IBKR publishes the stocks it treats as U.S. microcaps (``ibkr_microcap_list_url``: #SYMBOL, MAIN_EXCHANGE,
DESCRIPTION, IB_CONTRACT_ID) — over the counter, or on Nasdaq / NYSE American with a market cap of $50-300M at $5 or
less, or a cap of $50M or less, until both exceed $300M and $5 for 30 consecutive days (its knowledge base, "U.S.
Microcap Stock Restrictions"). Its "No Opening Trades: Small Cap, Subject to Compliance Restriction" refusals of an
opening short come from that list, and IBKR keeps no history of it, so the EOD chain's first step (`archive`) keeps
one raw copy per market day: ``ibkr_microcap_dir``/<ET date>.csv.gz (never delete). The download is DATA: it is
checked for IBKR's header and a plausible size and parsed as CSV only — never evaluated.

Point in time: the file of day D is fetched after D's close, so a session's list is the latest file of a day BEFORE
it (`on_list`), at most ``ibkr_microcap_max_age_days`` old (None beyond: unknown, not "not listed").
"""
from __future__ import annotations

import csv
import gzip
import io
import os
from datetime import date, timedelta
from pathlib import Path
from typing import Dict, Optional, Set, Tuple

from loguru import logger

from config import settings

HEADER = "#SYMBOL,MAIN_EXCHANGE"
MIN_ROWS = 1000                       # IBKR's list held 9,843 symbols on 2026-10-23: far fewer is a broken download

_CACHE: Dict[str, Tuple[float, Set[str]]] = {}


def root() -> Path:
    return Path(settings.ibkr_microcap_dir)


def path_for(d: date) -> Path:
    return root() / f"{d.isoformat()}.csv.gz"


def parse(raw: bytes) -> Set[str]:
    """The symbols of one copy of IBKR's list (upper case, IBKR's form: a class share is "BRK B")."""
    text = raw.decode("utf-8", errors="replace")
    out: Set[str] = set()
    for row in csv.reader(io.StringIO(text)):
        if not row or not row[0].strip() or row[0].startswith("#"):
            continue
        out.add(row[0].strip().upper())
    return out


def check(raw: bytes) -> int:
    """The number of symbols in a download that looks like IBKR's list; raises ValueError otherwise."""
    head = raw[:200].decode("utf-8", errors="replace").lstrip("﻿")
    if not head.startswith(HEADER):
        raise ValueError(f"not IBKR's microcap list (header {head[:60]!r})")
    n = len(parse(raw))
    if n < MIN_ROWS:
        raise ValueError(f"IBKR's microcap list holds only {n} symbols (< {MIN_ROWS})")
    return n


def _download() -> bytes:
    import httpx
    r = httpx.get(str(settings.ibkr_microcap_list_url), timeout=60.0, follow_redirects=True,
                  headers={"User-Agent": "Mozilla/5.0 (llm_trader microcap archive)"})
    r.raise_for_status()
    return r.content


def archive(d: date, fetch=None) -> Tuple[Path, int]:
    """Keep day ``d``'s copy of IBKR's list (once: a day already archived is not fetched again). Returns the file and
    its symbol count; raises when the download fails or does not look like IBKR's list (nothing written)."""
    p = path_for(d)
    if p.exists():
        with gzip.open(p, "rb") as fh:
            return p, len(parse(fh.read()))
    raw = (fetch or _download)()
    n = check(raw)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(f".{os.getpid()}.tmp")
    with gzip.open(tmp, "wb") as fh:
        fh.write(raw)
    os.replace(tmp, p)
    logger.info(f"[microcaps] IBKR's U.S. microcap list archived: {n} symbols -> {p}")
    return p, n


def latest_day(before: Optional[date] = None) -> Optional[date]:
    """The latest archived day (strictly before ``before`` when given)."""
    days = []
    for f in root().glob("????-??-??.csv.gz"):
        try:
            days.append(date.fromisoformat(f.name[:10]))
        except ValueError:
            continue
    days = [x for x in days if before is None or x < before]
    return max(days) if days else None


def symbols_before(d: date) -> Optional[Tuple[date, Set[str]]]:
    """(its day, the symbols) of the list in force at session ``d``'s open: the latest file of a day before ``d``, at
    most ``ibkr_microcap_max_age_days`` old; None without one."""
    f_day = latest_day(before=d)
    if f_day is None or (d - f_day) > timedelta(days=int(settings.ibkr_microcap_max_age_days)):
        return None
    p = path_for(f_day)
    try:
        key, mt = str(p), p.stat().st_mtime
        hit = _CACHE.get(key)
        if hit is None or hit[0] != mt:
            with gzip.open(p, "rb") as fh:
                hit = (mt, parse(fh.read()))
            _CACHE[key] = hit
        return f_day, hit[1]
    except (OSError, EOFError, ValueError) as e:
        logger.warning(f"[microcaps] {p} unreadable ({e})")
        return None


def on_list(ticker: str, d: date) -> Optional[bool]:
    """Whether ``ticker`` was on IBKR's U.S. microcap list at session ``d``'s open; None when no list is in force."""
    got = symbols_before(d)
    if got is None:
        return None
    t = str(ticker or "").strip().upper()
    return t in got[1] or t.replace("-", " ").replace(".", " ") in got[1]
