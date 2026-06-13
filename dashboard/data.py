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

_RETRY_ATTEMPTS = 6
_RETRY_BASE_DELAY = 0.4  # exponential backoff: 0.4, 0.8, 1.6, 3.2, 5.0, … (~11s total)
_RETRY_MAX_DELAY = 5.0


def _retry(fn, what: str = ""):
    last = None
    delay = _RETRY_BASE_DELAY
    for attempt in range(_RETRY_ATTEMPTS):
        try:
            return fn()
        except FileNotFoundError:
            raise
        except Exception as e:  # most likely a transient DuckDB lock during a run
            last = e
            if attempt < _RETRY_ATTEMPTS - 1:
                time.sleep(delay)
                delay = min(delay * 2, _RETRY_MAX_DELAY)
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


def latest_run_failures() -> list:
    """Failed data sources (ok = false) for the most recent run — powers the
    dashboard health banner. Returns ``[]`` when there are none or no runs."""
    rid = latest_run_id()
    if not rid:
        return []
    df = run_sources_df(rid)
    if df.empty or "ok" not in df.columns:
        return []
    failed = df[~df["ok"].astype(bool)]
    return failed.to_dict("records")


# performance() is heavy (NAV walk + per-method solo simulation) — cache briefly,
# keyed by time window so the dashboard's 1w / 1m / inception toggle stays snappy.
_perf_cache: dict = {}          # window key -> {"ts": float, "data": dict}
_PERF_TTL = 60.0


def performance(window_days: Optional[int] = None, session: Optional[str] = None,
                force: bool = False) -> dict:
    """Windowed + session-filtered performance bundle. ``window_days`` = 7 / 30
    (None = inception); ``session`` = rth / extended / overnight (None = all)."""
    key = ("all" if window_days is None else int(window_days), session or "all")
    now = time.time()
    entry = _perf_cache.get(key)
    if not force and entry is not None and (now - entry["ts"]) < _PERF_TTL:
        return entry["data"]
    from src.performance.tracker import get_performance_for_email
    result = _retry(lambda: get_performance_for_email(window_days=window_days, session=session), "performance")
    _perf_cache[key] = {"ts": now, "data": result}
    return result


def filled_lmt_legs() -> list:
    """Real LMT fills (ENTRY/EXIT, no drift flattens) from broker_orders —
    the basis for the IBKR one-way cost tile and the sim-cost calibration.
    Read-only + retry, like every dashboard accessor."""
    return _retry(lambda: repo.fetch_filled_lmt_legs(), "lmt_legs")


def broker_trades(force: bool = False) -> list:
    """The IBKR-fills projection of the ledger (real executions, real
    commissions — see ``src.performance.broker_view``), cached briefly.
    Reads through ``repo.load_trades()`` so the read-only mode set above
    applies; never touches the tracker's write paths."""
    now = time.time()
    entry = _perf_cache.get("broker_trades")
    if not force and entry is not None and (now - entry["ts"]) < _PERF_TTL:
        return entry["data"]
    from src.performance.broker_view import build_broker_trades
    trades = _retry(lambda: repo.load_trades(), "broker_trades")
    result = build_broker_trades(trades)
    _perf_cache["broker_trades"] = {"ts": now, "data": result}
    return result
