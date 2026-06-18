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


def trade_reviews_df(ticker: str) -> pd.DataFrame:
    """The opener-pinned hold-review trajectory for one ticker (fix #2), ordered
    by time. Empty DataFrame when there's no history yet or the table predates
    this feature (a read-only dashboard can't create it — the next pipeline run
    will)."""
    def _q():
        return repo.fetch_df(
            "SELECT * FROM trade_reviews WHERE ticker = ? ORDER BY reviewed_at", [ticker])
    try:
        return _retry(_q, "trade_reviews")
    except Exception:
        return pd.DataFrame()


def trades_for_ticker(ticker: str) -> list:
    """All ledger trade dicts for one ticker (entry/exit markers for the review
    timeline). Read-only via repo.load_trades()."""
    try:
        trades = _retry(lambda: repo.load_trades(), "trades_for_ticker")
    except Exception:
        return []
    return [t for t in trades if t.get("ticker") == ticker]


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


# ── Diagnostics accessors (IC · calibration · exit quality · execution) ──────

def signal_ic(days: Optional[int] = None, horizons=(1, 5, 10), min_n: int = 10) -> dict:
    """Per-method information-coefficient table over the persisted signals panel
    joined with forward returns. Cached (the OHLCV join is heavy). Returns
    ``{panel_rows, tickers, ic}`` where ``ic`` is a DataFrame (empty until the
    panel has enough forward-return history)."""
    key = ("signal_ic", days, tuple(horizons), int(min_n))
    now = time.time()
    entry = _perf_cache.get(key)
    if entry is not None and (now - entry["ts"]) < _PERF_TTL:
        return entry["data"]
    from src.analysis.signal_panel import build_panel, compute_ic

    def _q():
        panel = build_panel(horizons=horizons, days=days)
        ic = (compute_ic(panel, horizons=horizons, min_n=min_n)
              if panel is not None and not panel.empty else pd.DataFrame())
        return {
            "panel_rows": 0 if panel is None or panel.empty else int(len(panel)),
            "tickers":    0 if panel is None or panel.empty else int(panel["ticker"].nunique()),
            "ic":         ic,
        }
    result = _retry(_q, "signal_ic")
    _perf_cache[key] = {"ts": now, "data": result}
    return result


def confidence_calibration(window_days: Optional[int] = None, session: Optional[str] = None) -> dict:
    """Confidence-calibration report (buckets + slope) over the windowed/session
    perf bundle's closed + open trades — so it tracks the tab's toggles and
    reuses the cached perf computation."""
    perf = performance(window_days=window_days, session=session)
    from src.analysis.confidence_calibration import compute_calibration
    trades = (perf.get("closed_trades") or []) + (perf.get("open_trades") or [])
    return compute_calibration(trades)


def exit_quality(window_days: Optional[int] = None, session: Optional[str] = None) -> dict:
    """MFE/MAE exit-quality report over the windowed/session closed trades (the
    sim ledger carries the excursion fields)."""
    perf = performance(window_days=window_days, session=session)
    from src.analysis.exit_quality import compute_exit_quality
    return compute_exit_quality(perf.get("closed_trades") or [])


def broker_forensics() -> dict:
    """Slippage / fill-rate / drift / reject forensics over the broker tables
    (all runs — not windowed)."""
    from src.analysis.broker_forensics import (
        compute_forensics, load_broker_orders, load_broker_reconciles)
    return _retry(lambda: compute_forensics(load_broker_orders(), load_broker_reconciles()),
                  "broker_forensics")


def tracking_error() -> dict:
    """Sim-vs-broker tracking-error report over every trade with a matching
    broker fill."""
    from src.analysis.tracking_error import compute_tracking_error
    trades = _retry(lambda: repo.load_trades(), "tracking_error")
    return compute_tracking_error(trades)


def source_reliability(days: int = 14) -> list:
    """Per-source success rate + latency over the last N days (from run_sources)
    — surfaces chronically-flaky or slow data sources. Cached + retry."""
    key = ("source_reliability", int(days))
    now = time.time()
    entry = _perf_cache.get(key)
    if entry is not None and (now - entry["ts"]) < _PERF_TTL:
        return entry["data"]
    from src.analysis.data_quality import compute_source_reliability, load_source_rows
    result = _retry(lambda: compute_source_reliability(load_source_rows(days)), "source_reliability")
    _perf_cache[key] = {"ts": now, "data": result}
    return result


def method_coverage(days: int = 14) -> dict:
    """Per-method data coverage (% of tickers with a real, non-zero score) + a
    recent-vs-prior delta to flag feeds that went dark. From the signals panel.
    Cached + retry."""
    key = ("method_coverage", int(days))
    now = time.time()
    entry = _perf_cache.get(key)
    if entry is not None and (now - entry["ts"]) < _PERF_TTL:
        return entry["data"]
    from src.analysis.data_quality import compute_method_coverage, load_signal_rows
    result = _retry(lambda: compute_method_coverage(load_signal_rows(days)), "method_coverage")
    _perf_cache[key] = {"ts": now, "data": result}
    return result


def latest_gate_diag() -> dict:
    """gate_diag JSON of the most recent run (carries the price-provenance
    verdict for the banner + Execution tab). ``{}`` when unavailable."""
    rid = latest_run_id()
    if not rid:
        return {}
    df = _retry(lambda: repo.fetch_df("SELECT gate_diag FROM runs WHERE run_id = ?", [rid]),
                "gate_diag")
    if df.empty or not df.iloc[0]["gate_diag"]:
        return {}
    import json
    try:
        return json.loads(df.iloc[0]["gate_diag"])
    except Exception:
        return {}
