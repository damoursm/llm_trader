"""DuckDB connection management.

Concurrency model: DuckDB allows a single read-write handle OR multiple read-only
handles across processes. The daily pipeline is the sole writer and uses
short-lived read-write connections (open → do → close) so the write lock is held
only momentarily. The dashboard connects read-only. A read-only open while the
pipeline briefly holds the write lock will raise; the dashboard retries (see
`dashboard/data.py`).

The OPEN itself also retries here, in both directions: the pipeline's write open
fails while the dashboard holds ANY read handle (and vice versa), and the two
run as separate processes — observed 2026-07-01 14:49, the pipeline's
`_load_trades` lost that race mid-tick. A short exponential backoff rides out
the other side's brief handle instead of surfacing a spurious failure.
"""

from __future__ import annotations

import time
from contextlib import contextmanager
from pathlib import Path

import duckdb

from config import settings
from src.db.schema import ensure_schema

# Lock-contention retry: ~11 s total budget, mirroring the dashboard's read
# retry. Only lock/IO collisions are retried — schema errors etc. raise at once.
_LOCK_RETRIES = 6
_LOCK_BASE_DELAY = 0.4
_LOCK_MAX_DELAY = 5.0


def db_path() -> Path:
    return Path(settings.db_path)


def _is_lock_error(exc: Exception) -> bool:
    """True for the cross-process file-lock collision (Windows: 'being used by
    another process'; POSIX: 'lock on file' / 'Conflicting lock')."""
    msg = str(exc).lower()
    return isinstance(exc, duckdb.Error) and (
        "used by another process" in msg
        or "lock" in msg
        or "resource temporarily unavailable" in msg
    )


def _connect_with_retry(path: str, read_only: bool):
    delay = _LOCK_BASE_DELAY
    for attempt in range(_LOCK_RETRIES):
        try:
            return duckdb.connect(path, read_only=read_only)
        except Exception as e:
            if attempt >= _LOCK_RETRIES - 1 or not _is_lock_error(e):
                raise
            time.sleep(delay)
            delay = min(delay * 2, _LOCK_MAX_DELAY)


@contextmanager
def connect(read_only: bool = False):
    """Yield a short-lived DuckDB connection and close it on exit.

    Read-write connections ensure the schema exists first. Read-only connections
    require the database file to already exist (run the pipeline or migration first).
    Lock collisions with the other process retry with backoff before raising.
    """
    path = db_path()

    if read_only:
        if not path.exists():
            raise FileNotFoundError(
                f"DuckDB file not found at {path}. Run the pipeline or "
                f"`python -m src.db.migrate` first."
            )
        conn = _connect_with_retry(str(path), read_only=True)
        try:
            yield conn
        finally:
            conn.close()
        return

    path.parent.mkdir(parents=True, exist_ok=True)
    conn = _connect_with_retry(str(path), read_only=False)
    try:
        ensure_schema(conn)
        yield conn
    finally:
        conn.close()
