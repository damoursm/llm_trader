"""yfinance's `download` serialized for the whole process (2026-10-12).

yfinance 1.2.0 keeps every `download` call's results in ONE module-wide dict (`yfinance.shared._DFS`): each call
resets it at its start, its per-ticker threads fill it, and the threaded path busy-waits until it holds every ticker
it asked for (`multi.download`: ``while len(shared._DFS) < len(tickers): sleep(0.01)``). Two downloads at once — the
tick's Steps 1-3 pool runs bond internals, breadth, credit, global macro, highs / lows and MOVE side by side — wipe
each other's results: a call returns the other's tickers, or waits forever. The 2026-10-12 01:00 overnight tick hung
in `bond_internals`' download until the tick watchdog killed the scheduler (`timeout=` bounds the HTTP requests, not
that loop). `install()` wraps `yfinance.download` in one process-wide lock, once; every caller reads
``yf.download`` at call time, so all of them go through it.
"""
from __future__ import annotations

import functools
import threading

_LOCK = threading.RLock()


def install() -> bool:
    """Serialize `yfinance.download` in this process (idempotent). Returns True when the wrapper is in place."""
    try:
        import yfinance as yf
    except Exception:                                          # noqa: BLE001 — never break an import over it
        return False
    current = yf.download
    if getattr(current, "_serialized", False):
        return True
    unsafe = current

    @functools.wraps(unsafe)
    def download(*args, **kwargs):
        with _LOCK:
            return unsafe(*args, **kwargs)

    download._serialized = True                                # type: ignore[attr-defined]
    yf.download = download
    return True
