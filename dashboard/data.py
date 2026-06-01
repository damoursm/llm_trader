"""Read-only data access for the dashboard.

The dashboard must never take the DuckDB write lock, so the repo is put in
read-only mode and every read is wrapped in a short retry (the pipeline holds
the write lock only momentarily during its end-of-run persistence). The heavy
performance computation is cached briefly so tab switches stay responsive.
"""

from __future__ import annotations

import time
from typing import Optional

import pandas as pd
from loguru import logger

from src.db import repo

# The dashboard is a read-only consumer of the database.
repo.set_read_only(True)

_RETRY_ATTEMPTS = 4
_RETRY_DELAY = 0.5


def _retry(fn, what: str = ""):
    last = None
    for _ in range(_RETRY_ATTEMPTS):
        try:
            return fn()
        except FileNotFoundError:
            raise
        except Exception as e:  # most likely a transient DuckDB lock during a run
            last = e
            time.sleep(_RETRY_DELAY)
    logger.warning(f"[dashboard] read failed ({what}): {last}")
    raise last


_REC_COLS = (
    "generated_at, ticker, type, direction, action, confidence, time_horizon, "
    "actionable, dominant_method, llm_provider, rationale"
)


def runs_df() -> pd.DataFrame:
    return _retry(lambda: repo.fetch_df("SELECT * FROM runs ORDER BY started_at DESC"), "runs")


def latest_run_id() -> Optional[str]:
    df = runs_df()
    return None if df.empty else str(df.iloc[0]["run_id"])


def run_row(run_id: str):
    df = _retry(lambda: repo.fetch_df("SELECT * FROM runs WHERE run_id = ?", [run_id]), "run_row")
    return None if df.empty else df.iloc[0]


def recommendations_df(run_id: Optional[str] = None) -> pd.DataFrame:
    if run_id:
        return _retry(lambda: repo.fetch_df(
            f"SELECT {_REC_COLS} FROM recommendations WHERE run_id = ? "
            f"ORDER BY actionable DESC, confidence DESC", [run_id]), "recs")
    return _retry(lambda: repo.fetch_df(
        f"SELECT {_REC_COLS} FROM recommendations ORDER BY generated_at DESC LIMIT 200"), "recs")


def run_sources_df(run_id: str) -> pd.DataFrame:
    return _retry(lambda: repo.fetch_df(
        "SELECT source_label, ok, duration_s, error FROM run_sources "
        "WHERE run_id = ? ORDER BY ok, source_label", [run_id]), "sources")


# performance() is heavy (NAV walk + per-method solo simulation) — cache briefly.
_perf_cache = {"ts": 0.0, "data": None}
_PERF_TTL = 60.0


def performance(force: bool = False) -> dict:
    now = time.time()
    cached = _perf_cache["data"]
    if not force and cached is not None and (now - _perf_cache["ts"]) < _PERF_TTL:
        return cached
    from src.performance.tracker import get_performance_for_email
    data = _retry(get_performance_for_email, "performance")
    _perf_cache.update(ts=now, data=data)
    return data
