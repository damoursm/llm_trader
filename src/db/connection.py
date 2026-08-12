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

# Paths whose schema this PROCESS has already ensured (see connect()).
_SCHEMA_READY: set = set()


def reset_schema_cache() -> None:
    """Forget which paths have had their schema ensured. For tests that recreate
    a database file at a path this process already touched."""
    _SCHEMA_READY.clear()


def db_path() -> Path:
    return Path(settings.db_path)


def _is_lock_error(exc: Exception) -> bool:
    """True for a RETRYABLE open collision.

    Two distinct races, both transient because every handle here is short-lived:

    * cross-PROCESS file lock (Windows: 'being used by another process'; POSIX:
      'lock on file' / 'Conflicting lock');
    * in-PROCESS configuration clash — DuckDB keeps ONE database instance per
      path per process, so opening read-write while a read-only handle is still
      open (or vice versa) raises "Can't open a connection to same database file
      with a different configuration than existing connections". The real fix is
      to keep one config per process (see ``repo.fetch_df``), but this is also
      worth riding out: the conflicting handle is always a context manager about
      to close, and without it a single overlapping read killed the caller.
    """
    msg = str(exc).lower()
    return isinstance(exc, duckdb.Error) and (
        "used by another process" in msg
        or "lock" in msg
        or "resource temporarily unavailable" in msg
        or "different configuration" in msg
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
        # ensure_schema is idempotent but NOT free: 18 CREATE-IF-NOT-EXISTS plus
        # 61 ADD-COLUMN-IF-NOT-EXISTS statements, measured at 14.1 ms per connect
        # against the 397 MB live DB — paid on EVERY read-write open. The schema
        # cannot change under us mid-process (this process is the only writer, and
        # a schema change means a code change means a restart), so run it once per
        # process per path. Also what makes routing reads through the read-write
        # config affordable (see repo.fetch_df).
        key = str(path.resolve())
        if key not in _SCHEMA_READY:
            ensure_schema(conn)
            _SCHEMA_READY.add(key)
        yield conn
    finally:
        conn.close()
