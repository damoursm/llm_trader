"""SQLite persistence for the recommendation feedback loop (Priority 0).

Two tables:
  * ``recommendations`` — every recommendation we generate, with the live price
    at the time of the call and the underlying sentiment/technical scores.
  * ``rec_returns``     — one row per (recommendation, horizon) once it has
    matured, holding the realized close-to-close forward return.

Joining these two tables is what lets us measure hit rate, calibrate
confidence, and report *accurate* returns in the email.
"""

import os
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from typing import Dict, Iterable, List, Optional

from loguru import logger

from config import settings
from src.models import Recommendation, TickerSignal


_SCHEMA = """
CREATE TABLE IF NOT EXISTS recommendations (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    ticker          TEXT    NOT NULL,
    direction       TEXT    NOT NULL,
    action          TEXT    NOT NULL,
    confidence      REAL    NOT NULL,
    sentiment_score REAL,
    technical_score REAL,
    rationale       TEXT,
    price_at_rec    REAL,
    generated_at    TEXT    NOT NULL   -- ISO-8601 UTC
);

CREATE INDEX IF NOT EXISTS idx_rec_ticker_time
    ON recommendations (ticker, generated_at);

CREATE TABLE IF NOT EXISTS rec_returns (
    rec_id         INTEGER NOT NULL,
    horizon_days   INTEGER NOT NULL,
    entry_date     TEXT,
    entry_price    REAL,
    exit_date      TEXT,
    exit_price     REAL,
    raw_return     REAL,
    aligned_return REAL,
    scored_at      TEXT,
    PRIMARY KEY (rec_id, horizon_days),
    FOREIGN KEY (rec_id) REFERENCES recommendations (id)
);
"""


@contextmanager
def _connect():
    """Yield a SQLite connection, creating the parent directory as needed."""
    path = settings.performance_db_path
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    try:
        conn.execute("PRAGMA foreign_keys = ON")
        yield conn
        conn.commit()
    finally:
        conn.close()


def init_db() -> None:
    """Create tables/indexes if they do not yet exist (idempotent)."""
    with _connect() as conn:
        conn.executescript(_SCHEMA)


def log_recommendations(
    recs: List[Recommendation],
    prices: Optional[Dict[str, float]] = None,
    signals: Optional[Dict[str, TickerSignal]] = None,
) -> int:
    """Persist a batch of recommendations for later grading.

    ``prices`` maps ticker -> live price at call time (recorded for reference);
    ``signals`` maps ticker -> TickerSignal so we can store the component scores
    that drove the call (needed later to calibrate the 60/40 weighting).
    """
    if not recs:
        return 0

    prices = prices or {}
    signals = signals or {}
    rows = []
    for r in recs:
        sig = signals.get(r.ticker)
        rows.append((
            r.ticker,
            r.direction,
            r.action,
            float(r.confidence),
            float(sig.sentiment_score) if sig else None,
            float(sig.technical_score) if sig else None,
            r.rationale,
            float(prices[r.ticker]) if r.ticker in prices else None,
            r.generated_at.astimezone(timezone.utc).isoformat(),
        ))

    with _connect() as conn:
        conn.executemany(
            """
            INSERT INTO recommendations
                (ticker, direction, action, confidence, sentiment_score,
                 technical_score, rationale, price_at_rec, generated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            rows,
        )
    logger.info(f"Logged {len(rows)} recommendations to performance store")
    return len(rows)


def fetch_ungraded(horizon_days: int) -> List[sqlite3.Row]:
    """Return recommendations that have no graded return yet for this horizon."""
    with _connect() as conn:
        return conn.execute(
            """
            SELECT r.id, r.ticker, r.action, r.direction, r.generated_at
            FROM recommendations r
            LEFT JOIN rec_returns rr
                   ON rr.rec_id = r.id AND rr.horizon_days = ?
            WHERE rr.rec_id IS NULL
            ORDER BY r.ticker, r.generated_at
            """,
            (horizon_days,),
        ).fetchall()


def save_returns(rows: Iterable[dict]) -> int:
    """Insert graded forward-return rows (ignores ones already present)."""
    rows = list(rows)
    if not rows:
        return 0
    with _connect() as conn:
        conn.executemany(
            """
            INSERT OR IGNORE INTO rec_returns
                (rec_id, horizon_days, entry_date, entry_price, exit_date,
                 exit_price, raw_return, aligned_return, scored_at)
            VALUES (:rec_id, :horizon_days, :entry_date, :entry_price, :exit_date,
                    :exit_price, :raw_return, :aligned_return, :scored_at)
            """,
            rows,
        )
    return len(rows)


def horizon_stats(horizon_days: int) -> dict:
    """Aggregate hit-rate / return stats for directional (BUY/SELL) calls."""
    with _connect() as conn:
        row = conn.execute(
            """
            SELECT
                COUNT(*)                              AS graded,
                SUM(CASE WHEN rr.aligned_return > 0 THEN 1 ELSE 0 END) AS hits,
                AVG(rr.aligned_return)                AS avg_aligned,
                AVG(rr.raw_return)                    AS avg_raw,
                MAX(rr.aligned_return)                AS best,
                MIN(rr.aligned_return)                AS worst
            FROM rec_returns rr
            JOIN recommendations r ON r.id = rr.rec_id
            WHERE rr.horizon_days = ?
              AND r.action IN ('BUY', 'SELL')
            """,
            (horizon_days,),
        ).fetchone()
    return dict(row) if row else {}


def recent_graded(limit: int = 10) -> List[sqlite3.Row]:
    """Most recently matured directional (BUY/SELL) graded recommendations."""
    with _connect() as conn:
        return conn.execute(
            """
            SELECT r.ticker, r.action, r.direction, r.confidence, r.generated_at,
                   rr.horizon_days, rr.entry_date, rr.entry_price,
                   rr.exit_date, rr.exit_price, rr.raw_return, rr.aligned_return
            FROM rec_returns rr
            JOIN recommendations r ON r.id = rr.rec_id
            WHERE r.action IN ('BUY', 'SELL')
            ORDER BY rr.scored_at DESC, r.generated_at DESC
            LIMIT ?
            """,
            (limit,),
        ).fetchall()


def count_recommendations() -> int:
    with _connect() as conn:
        return conn.execute("SELECT COUNT(*) FROM recommendations").fetchone()[0]


def count_graded() -> int:
    with _connect() as conn:
        return conn.execute("SELECT COUNT(*) FROM rec_returns").fetchone()[0]
