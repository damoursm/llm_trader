"""One-time migration of the legacy JSON ledgers into DuckDB.

Imports ``cache/trades.json`` and ``cache/hypothetical_trades.json`` into the
DuckDB tables. Idempotent: skips a ledger whose table already has rows, so
re-running never duplicates or clobbers live data. Pass ``force=True`` (or the
``--force`` flag) to overwrite regardless.

    python -m src.db.migrate            # safe import (skips populated tables)
    python -m src.db.migrate --force    # overwrite from JSON
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from loguru import logger

from src.db import repo
from src.db.connection import connect

TRADES_JSON = Path("cache/trades.json")
HYPOTHETICAL_JSON = Path("cache/hypothetical_trades.json")


def _read_json(path: Path) -> list:
    if not path.exists():
        logger.info(f"[migrate] {path} not found — nothing to import")
        return []
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return data if isinstance(data, list) else []
    except Exception as e:
        logger.warning(f"[migrate] Could not read {path}: {e}")
        return []


def _count(table: str) -> int:
    with connect() as conn:
        return conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]


def migrate(force: bool = False) -> None:
    if _count("trades") and not force:
        logger.info(f"[migrate] trades already has {_count('trades')} rows — skipping (use --force to overwrite)")
    else:
        trades = _read_json(TRADES_JSON)
        repo.save_trades(trades)
        logger.info(f"[migrate] Imported {len(trades)} trade(s) → DuckDB")

    if _count("hypothetical_trades") and not force:
        logger.info(f"[migrate] hypothetical_trades already has {_count('hypothetical_trades')} rows — skipping")
    else:
        hyp = _read_json(HYPOTHETICAL_JSON)
        repo.save_hypothetical(hyp)
        logger.info(f"[migrate] Imported {len(hyp)} hypothetical trade(s) → DuckDB")


if __name__ == "__main__":
    migrate(force="--force" in sys.argv)
    logger.info("[migrate] Done.")
