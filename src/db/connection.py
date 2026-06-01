"""DuckDB connection management.

Concurrency model: DuckDB allows a single read-write handle OR multiple read-only
handles across processes. The daily pipeline is the sole writer and uses
short-lived read-write connections (open → do → close) so the write lock is held
only momentarily. The dashboard connects read-only. A read-only open while the
pipeline briefly holds the write lock will raise; the dashboard retries (see
`dashboard/data.py`).
"""

from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path

import duckdb

from config import settings
from src.db.schema import ensure_schema


def db_path() -> Path:
    return Path(settings.db_path)


@contextmanager
def connect(read_only: bool = False):
    """Yield a short-lived DuckDB connection and close it on exit.

    Read-write connections ensure the schema exists first. Read-only connections
    require the database file to already exist (run the pipeline or migration first).
    """
    path = db_path()

    if read_only:
        if not path.exists():
            raise FileNotFoundError(
                f"DuckDB file not found at {path}. Run the pipeline or "
                f"`python -m src.db.migrate` first."
            )
        conn = duckdb.connect(str(path), read_only=True)
        try:
            yield conn
        finally:
            conn.close()
        return

    path.parent.mkdir(parents=True, exist_ok=True)
    conn = duckdb.connect(str(path))
    try:
        ensure_schema(conn)
        yield conn
    finally:
        conn.close()
