"""IBKR borrow availability and fees — archived every tick, and the gate every
short entry passes (2026-09-25, user directive: "Save IBKR's file every tick, so
later evaluations use the real fee at entry. Have any short strategy skip names
with no shares or a fee above a cap.").

SOURCE
------
IBKR publishes its stock-loan book for every US symbol it can lend: shares
available to borrow and the annual borrow fee (plus the rebate rate), refreshed
~every 15 minutes, as ``usa.txt`` on its public FTP (user ``shortstock``, no
password). Measured 2026-09-25: 19,754 symbols, 1.8 MB, ~1 s to download;
``ftp2.interactivebrokers.com`` answers from this machine while ``ftp3`` (the
host most references give) times out, so the hosts are tried in order. The IB
Gateway API carries the same availability (generic tick 236) but only on the
DELAYED market-data type for this account, and never the fee — the file is the
one source with both.

ARCHIVE
-------
Each tick downloads the file and keeps the RAW text, gzipped, under
``data/ibkr_borrow/<ET date>/<HHMMSS>.txt.gz`` named by the file's own ``#BOF``
timestamp (US Eastern). ``data/``, not ``cache/``: IBKR keeps no history, so
this is the only record of what it quoted — never delete it. A file already
archived (same timestamp) is not written twice. `borrow_at(ticker, when)` reads
the newest snapshot at or before ``when``, which is what a later evaluation
charges a trade entered at ``when``.

THE GATE
--------
`short_block(ticker, price)` answers for a NEW short: ``"no_borrow"`` when IBKR
cannot lend a position this size (not in its file, no shares, or shares × price
below ``short_borrow_min_available_usd``), ``"borrow_fee"`` when the annual fee
is above ``short_borrow_max_fee_pct``, else None. Applied at the point a short
is OPENED (`tracker.record_new_trades`, `tracker.record_follow_through_trades`),
never in the shared gate cascade — a SELL there also closes a held long on a
signal reversal, and closing a long borrows nothing. A skipped name re-qualifies
any tick the file says otherwise. No current snapshot (download failing, file
older than ``ibkr_borrow_max_age_minutes``) → NOT CHECKED, logged once: the gate
judges IBKR's answer, not its absence (the Gate 4b convention); the broker still
refuses a short IBKR cannot locate.
"""
from __future__ import annotations

import bisect
import ftplib
import gzip
import io
import os
import threading
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from zoneinfo import ZoneInfo

from loguru import logger

from config.settings import settings

ARCHIVE_DIR = Path("data/ibkr_borrow")
_ET = ZoneInfo("America/New_York")
_CAPPED = 10_000_001                     # the file prints ">10000000" past that

_LOCK = threading.Lock()
_LATEST: Dict[str, object] = {"ts": None, "table": None}     # the newest snapshot in this process
_PARSED: Dict[str, Tuple[Optional[datetime], Dict[str, "Borrow"]]] = {}   # archive path -> parsed
_UNCHECKED_LOGGED = {"ts": 0.0}


@dataclass(frozen=True)
class Borrow:
    """One symbol's line of IBKR's file."""
    symbol: str
    fee_pct: Optional[float]             # annual borrow fee, % (FEERATE)
    rebate_pct: Optional[float]          # annual rebate, % (REBATERATE)
    available: Optional[int]             # shares available to borrow (capped at 10,000,001)
    file_ts: Optional[datetime]          # the snapshot's own timestamp, aware


# ── parsing ──────────────────────────────────────────────────────────────────

def _num(s: str) -> Optional[float]:
    try:
        return float(s)
    except (TypeError, ValueError):
        return None


def parse(text: str) -> Tuple[Optional[datetime], Dict[str, Borrow]]:
    """``(file timestamp, {symbol: Borrow})`` from the file's text. Header
    ``#BOF|2026.09.25|16:27:30`` (US Eastern), then
    ``SYM|CUR|NAME|CON|ISIN|REBATERATE|FEERATE|AVAILABLE|FIGI|`` rows."""
    ts: Optional[datetime] = None
    table: Dict[str, Borrow] = {}
    for line in text.splitlines():
        if line.startswith("#BOF"):
            p = line.split("|")
            try:
                ts = datetime.strptime(f"{p[1]} {p[2]}", "%Y.%m.%d %H:%M:%S").replace(tzinfo=_ET)
            except (IndexError, ValueError):
                ts = None
            continue
        if not line or line.startswith("#"):
            continue
        p = line.split("|")
        if len(p) < 8:
            continue
        sym = p[0].strip().upper()
        if not sym or sym in table:
            continue
        raw = p[7].strip()
        if raw.startswith(">"):
            avail: Optional[int] = _CAPPED
        else:
            v = _num(raw)
            avail = int(v) if v is not None else None
        table[sym] = Borrow(symbol=sym, fee_pct=_num(p[6].strip()), rebate_pct=_num(p[5].strip()),
                            available=avail, file_ts=ts)
    for b in list(table.values()):
        if b.file_ts is None and ts is not None:
            table[b.symbol] = Borrow(b.symbol, b.fee_pct, b.rebate_pct, b.available, ts)
    return ts, table


def lookup(table: Dict[str, Borrow], ticker: str) -> Optional[Borrow]:
    """A ticker's row: IBKR writes class shares with a space (``BRK B``) where
    this project uses a hyphen (``BRK-B``)."""
    tk = (ticker or "").strip().upper()
    for s in (tk, tk.replace("-", " "), tk.replace("-", ".")):
        b = table.get(s)
        if b is not None:
            return b
    return None


# ── download + archive ───────────────────────────────────────────────────────

def _hosts() -> List[str]:
    return [h.strip() for h in str(getattr(settings, "ibkr_borrow_hosts", "") or "").split(",") if h.strip()]


def _download(timeout: float = 30.0) -> bytes:
    last: Optional[Exception] = None
    for host in _hosts():
        try:
            ftp = ftplib.FTP(host, timeout=timeout)
            try:
                ftp.login("shortstock", "")
                buf = io.BytesIO()
                ftp.retrbinary("RETR usa.txt", buf.write)
            finally:
                try:
                    ftp.quit()
                except Exception:                              # noqa: BLE001
                    pass
            data = buf.getvalue()
            if data:
                return data
        except Exception as exc:                               # noqa: BLE001 — try the next host
            last = exc
            logger.debug(f"[borrow] {host}: {exc}")
    raise RuntimeError(f"IBKR borrow file unavailable from {_hosts()}: {last}")


def archive_path(ts: datetime, base: Optional[Path] = None) -> Path:
    et = ts.astimezone(_ET)
    return (base or ARCHIVE_DIR) / et.strftime("%Y-%m-%d") / f"{et:%H%M%S}.txt.gz"


def snapshot(base: Optional[Path] = None) -> Dict[str, Borrow]:
    """Download the file, archive it (once per file timestamp) and make it this
    process's current snapshot. Returns the parsed table — empty on failure, so
    the tick's source log reads it as a dark source."""
    if not getattr(settings, "enable_ibkr_borrow_snapshot", False):
        return {}
    data = _download()
    text = data.decode("utf-8", "replace")
    ts, table = parse(text)
    if ts is None or not table:
        raise RuntimeError("IBKR borrow file had no #BOF timestamp or no rows")
    path = archive_path(ts, base)
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(path.name + ".tmp")
        with gzip.open(tmp, "wb") as fh:
            fh.write(data)
        os.replace(tmp, path)
        logger.info(f"[borrow] IBKR file {ts:%Y-%m-%d %H:%M:%S} ET: {len(table):,} symbols archived -> {path}")
    with _LOCK:
        _LATEST["ts"], _LATEST["table"] = ts, table
    return table


def archive_index(base: Optional[Path] = None) -> List[Tuple[datetime, Path]]:
    """Every archived snapshot, oldest first."""
    root = base or ARCHIVE_DIR
    out: List[Tuple[datetime, Path]] = []
    if not root.exists():
        return out
    for day in root.iterdir():
        if not day.is_dir():
            continue
        for f in day.glob("*.txt.gz"):
            try:
                ts = datetime.strptime(f"{day.name} {f.name[:6]}", "%Y-%m-%d %H%M%S").replace(tzinfo=_ET)
            except ValueError:
                continue
            out.append((ts, f))
    out.sort(key=lambda x: x[0])
    return out


def _read(path: Path) -> Tuple[Optional[datetime], Dict[str, Borrow]]:
    key = str(path)
    got = _PARSED.get(key)
    if got is None:
        with gzip.open(path, "rt", encoding="utf-8", errors="replace") as fh:
            got = parse(fh.read())
        if len(_PARSED) >= 16:                         # a small LRU is enough: evaluations walk in time order
            _PARSED.pop(next(iter(_PARSED)))
        _PARSED[key] = got
    return got


def snapshot_at(when: datetime, base: Optional[Path] = None) -> Optional[Tuple[datetime, Dict[str, Borrow]]]:
    """The newest archived snapshot at or before ``when`` (aware; naive = UTC)."""
    if when.tzinfo is None:
        when = when.replace(tzinfo=timezone.utc)
    idx = archive_index(base)
    if not idx:
        return None
    i = bisect.bisect_right([t for t, _ in idx], when) - 1
    if i < 0:
        return None
    ts, table = _read(idx[i][1])
    return (ts or idx[i][0]), table


def borrow_at(ticker: str, when: datetime, base: Optional[Path] = None) -> Optional[Borrow]:
    """What IBKR quoted for ``ticker`` in the newest snapshot at or before
    ``when`` — the fee a later evaluation charges a short entered then. None
    when no snapshot that old exists or IBKR did not list the name."""
    got = snapshot_at(when, base)
    return lookup(got[1], ticker) if got else None


def latest(max_age_minutes: Optional[float] = None, base: Optional[Path] = None,
           now: Optional[datetime] = None) -> Optional[Dict[str, Borrow]]:
    """The current snapshot: this process's last download, else the newest
    archived one — None when it is older than ``max_age_minutes``."""
    age_cap = float(max_age_minutes if max_age_minutes is not None
                    else getattr(settings, "ibkr_borrow_max_age_minutes", 120.0))
    now = now or datetime.now(timezone.utc)
    with _LOCK:
        ts, table = _LATEST["ts"], _LATEST["table"]
    if table is None:
        got = snapshot_at(now, base)
        if got is None:
            return None
        ts, table = got
    if ts is None or (now - ts).total_seconds() > age_cap * 60.0:
        return None
    return table


def reset() -> None:
    """Drop the in-process state (tests)."""
    with _LOCK:
        _LATEST["ts"], _LATEST["table"] = None, None
    _PARSED.clear()
    _UNCHECKED_LOGGED["ts"] = 0.0


# ── the gate ─────────────────────────────────────────────────────────────────

def is_equity_symbol(ticker: str) -> bool:
    """Only a stock or ETF short borrows shares. The project's own convention
    (`polygon_client`, `spread_sweep`, `news_fetcher`): ``=`` marks a future or
    FX pair (CL=F, EURUSD=X), ``^`` an index; ``-USD`` is a crypto pair. The
    live ledger has shorted CL=F, YM=F and ^NSEI — IBKR's stock file lists none
    of them, and reading that as "no borrow" would block trades that need none."""
    t = (ticker or "").strip().upper()
    return bool(t) and "=" not in t and not t.startswith("^") and not t.endswith("-USD")


_SETTINGS_CAP = object()


def short_block(ticker: str, price: Optional[float], base: Optional[Path] = None,
                now: Optional[datetime] = None,
                max_fee_pct=_SETTINGS_CAP) -> Tuple[Optional[str], Optional[Borrow]]:
    """``(reason, row)`` for opening a NEW short in ``ticker`` at ``price``:
    ``"no_borrow"`` (IBKR cannot lend a position this size), ``"borrow_fee"``
    (annual fee above the cap) or None (borrowable — ``row`` is what IBKR quoted,
    stamped on the trade). Gate off or no current snapshot → ``(None, None)``.
    ``max_fee_pct``: the fee cap for THIS call — omitted, the setting
    ``short_borrow_max_fee_pct``; None, no fee cap at all (the selection-short
    strategy trades every BORROWABLE name, fee charged, by user directive
    2026-09-26); availability is checked either way."""
    if not getattr(settings, "enable_short_borrow_gate", False):
        return None, None
    if not is_equity_symbol(ticker):
        return None, None                  # a future / index / FX / crypto short borrows no stock
    table = latest(base=base, now=now)
    if table is None:
        if time.time() - _UNCHECKED_LOGGED["ts"] > 600:
            logger.warning("[borrow] no current IBKR borrow file — new shorts are NOT borrow-checked "
                           "this tick (the broker still refuses a short it cannot locate)")
            _UNCHECKED_LOGGED["ts"] = time.time()
        return None, None
    b = lookup(table, ticker)
    min_usd = float(getattr(settings, "short_borrow_min_available_usd", 0.0) or 0.0)
    if max_fee_pct is _SETTINGS_CAP:
        cap = float(getattr(settings, "short_borrow_max_fee_pct", 0.0) or 0.0)
    else:
        cap = float(max_fee_pct) if max_fee_pct is not None else 0.0
    if b is None or not b.available:
        logger.info(f"[borrow] short {ticker} skipped — IBKR has no shares to lend")
        return "no_borrow", b
    px = float(price) if price and price > 0 else None
    if px is not None and b.available * px < min_usd:
        logger.info(f"[borrow] short {ticker} skipped — IBKR lends {b.available:,} shares "
                    f"(${b.available * px:,.0f}) < ${min_usd:,.0f}")
        return "no_borrow", b
    if cap > 0 and b.fee_pct is not None and b.fee_pct > cap:
        logger.info(f"[borrow] short {ticker} skipped — borrow fee {b.fee_pct:.1f}%/yr > cap {cap:g}%")
        return "borrow_fee", b
    return None, b


def entry_stamp(b: Optional[Borrow]) -> dict:
    """The fields a short's trade record carries: the fee it will be charged
    (`spread.borrow_annual_pct` reads ``borrow_fee_pct``) and what IBKR showed."""
    if b is None:
        return {}
    return {"borrow_fee_pct": b.fee_pct, "borrow_available_at_entry": b.available,
            "borrow_file_ts": b.file_ts.isoformat() if b.file_ts else None}


if __name__ == "__main__":                                 # python -m src.data.ibkr_borrow [TICKER ...]
    import sys
    t = snapshot()
    for tk in sys.argv[1:]:
        b = lookup(t, tk)
        print(tk, b)
