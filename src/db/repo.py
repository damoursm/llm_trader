"""Read/write API over DuckDB.

Trades round-trip faithfully through a JSON `data` column (the full dict, exactly
as the old cache/*.json files stored it), so every analytics function in
`src.performance.tracker` keeps operating on identical dicts. The scalar columns
alongside `data` are projections used for SQL queries and the dashboard.
"""

from __future__ import annotations

import hashlib
import json
from typing import List, Optional

import pandas as pd

from src.db.connection import connect


# When True, read paths open read-only connections. The dashboard sets this so it
# never takes the write lock; the pipeline leaves it False (default).
_READ_ONLY = False


def set_read_only(flag: bool) -> None:
    global _READ_ONLY
    _READ_ONLY = bool(flag)


# ── small coercion helpers ────────────────────────────────────────────────

def _f(x) -> Optional[float]:
    try:
        return float(x) if x is not None else None
    except (TypeError, ValueError):
        return None


def _json(x) -> Optional[str]:
    return json.dumps(x, default=str) if x is not None else None


def _trade_id(t: dict) -> str:
    key = f"{t.get('ticker', '')}|{t.get('decision_datetime') or t.get('entry_datetime') or t.get('entry_date', '')}"
    return hashlib.sha1(key.encode("utf-8")).hexdigest()[:16]


def _hyp_id(t: dict) -> str:
    key = f"{t.get('ticker', '')}|{t.get('entry_datetime') or t.get('entry_date', '')}|{t.get('action', '')}"
    return hashlib.sha1(key.encode("utf-8")).hexdigest()[:16]


# ── real trades (replaces cache/trades.json) ───────────────────────────────

_TRADE_COLS = [
    "_seq", "trade_id", "run_id", "recommendation_id", "ticker", "action",
    "direction", "confidence", "decision_datetime", "entry_datetime",
    "entry_date", "entry_price", "status", "exit_date", "exit_price",
    "return_pct", "position_size_multiplier", "sector_key", "dominant_method",
    "time_horizon", "data",
]


def _trade_row(i: int, t: dict) -> tuple:
    return (
        i,
        _trade_id(t),
        t.get("run_id"),
        t.get("recommendation_id"),
        t.get("ticker"),
        t.get("action"),
        t.get("direction"),
        _f(t.get("confidence")),
        t.get("decision_datetime"),
        t.get("entry_datetime"),
        t.get("entry_date"),
        _f(t.get("entry_price")),
        t.get("status"),
        t.get("exit_date"),
        _f(t.get("exit_price")),
        _f(t.get("return_pct")),
        _f(t.get("position_size_multiplier")),
        t.get("sector_key"),
        t.get("dominant_method"),
        t.get("time_horizon"),
        json.dumps(t, default=str),
    )


def load_trades() -> List[dict]:
    with connect(read_only=_READ_ONLY) as conn:
        rows = conn.execute("SELECT data FROM trades ORDER BY _seq").fetchall()
    return [json.loads(r[0]) for r in rows]


def save_trades(trades: List[dict]) -> None:
    """Full-replace the trades table (matches the old whole-file rewrite semantics)."""
    rows = [_trade_row(i, t) for i, t in enumerate(trades)]
    placeholders = ", ".join(["?"] * len(_TRADE_COLS))
    with connect() as conn:
        conn.execute("BEGIN TRANSACTION")
        conn.execute("DELETE FROM trades")
        if rows:
            conn.executemany(
                f"INSERT INTO trades ({', '.join(_TRADE_COLS)}) VALUES ({placeholders})",
                rows,
            )
        conn.execute("COMMIT")


# ── hypothetical trades (replaces cache/hypothetical_trades.json) ───────────

_HYP_COLS = [
    "_seq", "trade_id", "ticker", "action", "direction", "entry_date",
    "entry_datetime", "entry_price", "status", "return_pct", "data",
]


def _hyp_row(i: int, t: dict) -> tuple:
    return (
        i,
        _hyp_id(t),
        t.get("ticker"),
        t.get("action"),
        t.get("direction"),
        t.get("entry_date"),
        t.get("entry_datetime"),
        _f(t.get("entry_price")),
        t.get("status"),
        _f(t.get("return_pct")),
        json.dumps(t, default=str),
    )


def load_hypothetical() -> List[dict]:
    with connect(read_only=_READ_ONLY) as conn:
        rows = conn.execute("SELECT data FROM hypothetical_trades ORDER BY _seq").fetchall()
    return [json.loads(r[0]) for r in rows]


def save_hypothetical(trades: List[dict]) -> None:
    rows = [_hyp_row(i, t) for i, t in enumerate(trades)]
    placeholders = ", ".join(["?"] * len(_HYP_COLS))
    with connect() as conn:
        conn.execute("BEGIN TRANSACTION")
        conn.execute("DELETE FROM hypothetical_trades")
        if rows:
            conn.executemany(
                f"INSERT INTO hypothetical_trades ({', '.join(_HYP_COLS)}) VALUES ({placeholders})",
                rows,
            )
        conn.execute("COMMIT")


# ── run metadata + recommendations (write path from the pipeline) ───────────

_RUN_COLS = [
    "run_id", "started_at", "finished_at", "elapsed_s", "market_mode",
    "macro_regime", "confidence_threshold", "allow_buys", "universe_size",
    "n_recommendations", "n_actionable", "llm_synthesis_provider",
    "llm_sentiment_provider", "gate_diag",
]


def insert_run(run: dict) -> None:
    vals = (
        run.get("run_id"),
        run.get("started_at"),
        run.get("finished_at"),
        _f(run.get("elapsed_s")),
        run.get("market_mode"),
        run.get("macro_regime"),
        _f(run.get("confidence_threshold")),
        run.get("allow_buys"),
        run.get("universe_size"),
        run.get("n_recommendations"),
        run.get("n_actionable"),
        run.get("llm_synthesis_provider"),
        run.get("llm_sentiment_provider"),
        _json(run.get("gate_diag")),
    )
    placeholders = ", ".join(["?"] * len(_RUN_COLS))
    with connect() as conn:
        conn.execute(
            f"INSERT OR REPLACE INTO runs ({', '.join(_RUN_COLS)}) VALUES ({placeholders})",
            vals,
        )


def insert_run_sources(run_id: str, sources: List[dict]) -> None:
    """Persist the per-source 'APIs used' record for a run (idempotent per run)."""
    rows = [
        (run_id, s.get("label"), s.get("enabled", True), s.get("ok"),
         s.get("error"), _f(s.get("duration_s")))
        for s in (sources or [])
    ]
    with connect() as conn:
        conn.execute("DELETE FROM run_sources WHERE run_id = ?", [run_id])
        if rows:
            conn.executemany(
                "INSERT INTO run_sources (run_id, source_label, enabled, ok, error, duration_s) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                rows,
            )


_REC_COLS = [
    "rec_id", "run_id", "generated_at", "ticker", "type", "direction", "action",
    "confidence", "time_horizon", "rationale", "actionable", "dominant_method",
    "methods_agreeing", "contributing_scores", "llm_provider",
]


def insert_recommendations(recs: List[dict]) -> None:
    if not recs:
        return
    rows = []
    for r in recs:
        rid = r.get("rec_id") or hashlib.sha1(
            f"{r.get('run_id', '')}|{r.get('ticker', '')}".encode("utf-8")
        ).hexdigest()[:16]
        rows.append((
            rid,
            r.get("run_id"),
            r.get("generated_at"),
            r.get("ticker"),
            r.get("type"),
            r.get("direction"),
            r.get("action"),
            _f(r.get("confidence")),
            r.get("time_horizon"),
            r.get("rationale"),
            r.get("actionable"),
            r.get("dominant_method"),
            _json(r.get("methods_agreeing")),
            _json(r.get("contributing_scores")),
            r.get("llm_provider"),
        ))
    placeholders = ", ".join(["?"] * len(_REC_COLS))
    with connect() as conn:
        conn.executemany(
            f"INSERT OR REPLACE INTO recommendations ({', '.join(_REC_COLS)}) VALUES ({placeholders})",
            rows,
        )


# ── generic read path (used by the dashboard) ──────────────────────────────

def fetch_df(sql: str, params: Optional[list] = None, read_only: bool = True) -> pd.DataFrame:
    """Run a query and return a pandas DataFrame. Read-only by default."""
    with connect(read_only=read_only) as conn:
        return conn.execute(sql, params or []).df()
