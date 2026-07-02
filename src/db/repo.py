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
from src.db.schema import SIGNAL_METHOD_COLUMNS


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


def save_trades(trades: List[dict], allow_shrink: bool = False) -> None:
    """Full-replace the trades table (matches the old whole-file rewrite semantics).

    Wipe guard: in production the ledger only ever GROWS (closed trades stay), so
    a save that would drop more than half of an established table means the caller
    loaded a truncated/empty list (e.g. a lost lock race returning ``[]``) — the
    exact failure mode of the 2026-06-11 ledger wipe. Refuse it loudly instead of
    silently destroying history; pass ``allow_shrink=True`` for deliberate
    maintenance (manual cleanup, migrations)."""
    rows = [_trade_row(i, t) for i, t in enumerate(trades)]
    placeholders = ", ".join(["?"] * len(_TRADE_COLS))
    with connect() as conn:
        if not allow_shrink:
            existing = conn.execute("SELECT COUNT(*) FROM trades").fetchone()[0]
            if existing >= 4 and len(rows) < existing / 2:
                raise RuntimeError(
                    f"save_trades refused: would shrink the ledger {existing} → "
                    f"{len(rows)} rows. If this is deliberate maintenance, call "
                    f"with allow_shrink=True; otherwise the caller loaded a "
                    f"truncated ledger (lock race / partial read) and saving it "
                    f"would wipe history."
                )
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
         s.get("error"), _f(s.get("duration_s")), s.get("n_items"), s.get("empty"))
        for s in (sources or [])
    ]
    with connect() as conn:
        conn.execute("DELETE FROM run_sources WHERE run_id = ?", [run_id])
        if rows:
            conn.executemany(
                "INSERT INTO run_sources "
                "(run_id, source_label, enabled, ok, error, duration_s, n_items, empty) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                rows,
            )


_REC_COLS = [
    "rec_id", "run_id", "generated_at", "ticker", "type", "direction", "action",
    "confidence", "time_horizon", "rationale", "actionable", "dominant_method",
    "methods_agreeing", "contributing_scores", "llm_provider",
    "target_horizon", "horizon_net_edge_pct",
    "shadow_target_horizon", "shadow_direction", "shadow_horizon_net_edge_pct",
    "expected_move_pct", "market_aligned", "upside_score",
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
            r.get("target_horizon"),
            _f(r.get("horizon_net_edge_pct")),
            r.get("shadow_target_horizon"),
            r.get("shadow_direction"),
            _f(r.get("shadow_horizon_net_edge_pct")),
            _f(r.get("expected_move_pct")),
            r.get("market_aligned"),
            _f(r.get("upside_score")),
        ))
    placeholders = ", ".join(["?"] * len(_REC_COLS))
    with connect() as conn:
        conn.executemany(
            f"INSERT OR REPLACE INTO recommendations ({', '.join(_REC_COLS)}) VALUES ({placeholders})",
            rows,
        )


# ── signals panel (full per-ticker cross-section, every run) ───────────────

_SIGNAL_BASE_COLS = [
    "run_id", "generated_at", "signal_date", "ticker", "type", "direction",
    "combined_score", "confidence", "n_methods_agreeing", "dominant_method", "price",
]
_SIGNAL_COLS = _SIGNAL_BASE_COLS + list(SIGNAL_METHOD_COLUMNS) + ["scores"]


def insert_signals(run_id: str, generated_at: str, signal_date: str,
                   rows: List[dict]) -> None:
    """Persist the full per-ticker signal cross-section for one run.

    One row per ticker — ALL tickers the aggregator scored, not just the
    top-10 recommendations. Each row dict carries scalar fields plus a
    ``scores`` dict (method → score); known methods are projected into their
    own DOUBLE columns for direct SQL (`corr(news, fwd_ret)`), and the full
    dict is kept as JSON so a method added before its column exists is never
    lost. Idempotent per run_id.
    """
    if not rows:
        return
    out = []
    for r in rows:
        scores = r.get("scores") or {}
        out.append(tuple(
            [run_id, generated_at, signal_date,
             r.get("ticker"),
             r.get("type"),
             r.get("direction"),
             _f(r.get("combined_score")),
             _f(r.get("confidence")),
             r.get("n_methods_agreeing"),
             r.get("dominant_method"),
             _f(r.get("price"))]
            + [_f(scores.get(m)) for m in SIGNAL_METHOD_COLUMNS]
            + [_json(scores)]
        ))
    placeholders = ", ".join(["?"] * len(_SIGNAL_COLS))
    with connect() as conn:
        conn.execute("BEGIN TRANSACTION")
        conn.execute("DELETE FROM signals WHERE run_id = ?", [run_id])
        conn.executemany(
            f"INSERT INTO signals ({', '.join(_SIGNAL_COLS)}) VALUES ({placeholders})",
            out,
        )
        conn.execute("COMMIT")


# ── simulated single-method trades (long-format reshape of `signals`) ──────

_SIM_TRADE_COLS = [
    "run_id", "generated_at", "signal_date", "ticker", "method",
    "score", "direction", "entry_price",
]


def insert_simulated_trades(run_id: str, generated_at: str, signal_date: str,
                            rows: List[dict]) -> None:
    """Persist one row per (ticker, method) that had a non-zero score this run.

    The long-format counterpart to ``insert_signals`` (which stores the same
    scores WIDE, one column per method). Each row records the method's implied
    side — BUY when score>0, SELL when score<0 — at the decision-time price, so a
    single method's directional accuracy can be measured over EVERY scored ticker
    (not just the gate-selected few that became real trades). Idempotent per
    ``run_id``. Each ``rows`` dict carries ticker/method/score/direction/entry_price.
    """
    if not rows:
        return
    out = [(
        run_id, generated_at, signal_date,
        r.get("ticker"), r.get("method"),
        _f(r.get("score")), r.get("direction"), _f(r.get("entry_price")),
    ) for r in rows]
    placeholders = ", ".join(["?"] * len(_SIM_TRADE_COLS))
    with connect() as conn:
        conn.execute("BEGIN TRANSACTION")
        conn.execute("DELETE FROM simulated_trades WHERE run_id = ?", [run_id])
        conn.executemany(
            f"INSERT INTO simulated_trades ({', '.join(_SIM_TRADE_COLS)}) VALUES ({placeholders})",
            out,
        )
        conn.execute("COMMIT")


# ── per-tick exit-method scores (held positions) — the exit learning panel ──

_EXIT_SIGNAL_COLS = [
    "run_id", "reviewed_at", "signal_date", "ticker", "position_id",
    "entry_direction", "method", "score", "price",
]


def insert_exit_signals(run_id: str, rows: List[dict]) -> None:
    """Persist one row per (held position, exit method) re-scored this tick.

    The exit-side counterpart to ``insert_simulated_trades``: each row records an
    exit method's signed **hold-conviction** score (+ = the position should keep
    running, − = it should reverse/exit) on one open position at one tick, so the
    predictiveness of each individual exit method — and of the synthesized
    ``llm_review`` that actually decides — can be measured over EVERY held tick,
    not just the few positions that closed. Idempotent per ``run_id``. Each dict
    carries reviewed_at/signal_date/ticker/position_id/entry_direction/method/
    score/price."""
    if not rows:
        return
    out = [(
        run_id, r.get("reviewed_at"), r.get("signal_date"),
        r.get("ticker"), r.get("position_id"), r.get("entry_direction"),
        r.get("method"), _f(r.get("score")), _f(r.get("price")),
    ) for r in rows]
    placeholders = ", ".join(["?"] * len(_EXIT_SIGNAL_COLS))
    with connect() as conn:
        conn.execute("BEGIN TRANSACTION")
        conn.execute("DELETE FROM exit_signals WHERE run_id = ?", [run_id])
        conn.executemany(
            f"INSERT INTO exit_signals ({', '.join(_EXIT_SIGNAL_COLS)}) VALUES ({placeholders})",
            out,
        )
        conn.execute("COMMIT")


# ── broker execution record (write path from reconcile, via the pipeline) ──

_BROKER_RECONCILE_COLS = [
    "run_id", "created_at", "mode", "connected", "ok", "account_id",
    "account_equity", "account_currency",
    "pnl_daily", "pnl_unrealized", "pnl_realized",
    "entries_submitted", "exits_submitted",
    "fills_repaired", "rejects", "n_drift", "drift", "errors",
]

_BROKER_ORDER_COLS = [
    "run_id", "event", "intent", "ticker", "side", "order_type", "requested_qty",
    "filled_qty", "model_price", "limit_price", "fill_price", "slippage_bps",
    "commission", "status", "ok", "error", "order_id", "client_ref", "submitted_at",
]


# ── per-tick opener-pinned hold-review trajectory (fix #2) ─────────────────

_TRADE_REVIEW_COLS = [
    "run_id", "reviewed_at", "ticker", "position_id", "entry_datetime",
    "confidence", "action", "direction", "conf_floor", "entry_confidence",
    "entry_action", "price", "return_pct", "synthesis_model", "sentiment_model",
]


def insert_trade_reviews(rows: List[dict]) -> None:
    """Append one row per held position re-judged this tick (the opener-pinned
    hold-review). Builds the confidence-over-time trajectory the dashboard plots
    per ticker. Append-only — each tick is a fresh observation."""
    if not rows:
        return
    out = [(
        r.get("run_id"), r.get("reviewed_at"), r.get("ticker"), r.get("position_id"),
        r.get("entry_datetime"), _f(r.get("confidence")), r.get("action"), r.get("direction"),
        _f(r.get("conf_floor")), _f(r.get("entry_confidence")), r.get("entry_action"),
        _f(r.get("price")), _f(r.get("return_pct")),
        r.get("synthesis_model"), r.get("sentiment_model"),
    ) for r in rows]
    placeholders = ", ".join(["?"] * len(_TRADE_REVIEW_COLS))
    with connect() as conn:
        conn.executemany(
            f"INSERT INTO trade_reviews ({', '.join(_TRADE_REVIEW_COLS)}) VALUES ({placeholders})",
            out,
        )


def insert_broker_report(run_id: str, report: dict) -> None:
    """Persist one reconcile report: a summary row (broker_reconciles) plus one
    event row per order submission / fill repair (broker_orders).

    This is the durable record the paper phase measures from — per-order model
    vs fill price, cost-normalized slippage bps, commissions, rejects, drift.
    Idempotent per run_id (re-running a run_id replaces its rows).
    """
    if not report:
        return
    from datetime import datetime, timezone

    summary = (
        run_id,
        datetime.now(timezone.utc).isoformat(timespec="seconds"),
        report.get("mode"),
        bool(report.get("connected")),
        bool(report.get("ok")),
        report.get("account_id"),
        _f(report.get("account_equity")),
        report.get("account_currency"),
        _f(report.get("pnl_daily")),
        _f(report.get("pnl_unrealized")),
        _f(report.get("pnl_realized")),
        int(report.get("entries_submitted") or 0),
        int(report.get("exits_submitted") or 0),
        int(report.get("fills_repaired") or 0),
        int(report.get("rejects") or 0),
        len(report.get("drift") or []),
        _json(report.get("drift")),
        _json(report.get("errors")),
    )
    order_rows = [
        (
            run_id,
            o.get("event"),
            o.get("intent"),
            o.get("ticker"),
            o.get("side"),
            o.get("order_type"),
            o.get("requested_qty"),
            o.get("filled_qty"),
            _f(o.get("model_price")),
            _f(o.get("limit_price")),
            _f(o.get("fill_price")),
            _f(o.get("slippage_bps")),
            _f(o.get("commission")),
            o.get("status"),
            o.get("ok"),
            o.get("error"),
            o.get("order_id"),
            o.get("client_ref"),
            o.get("submitted_at"),
        )
        for o in (report.get("orders") or [])
    ]
    rec_ph = ", ".join(["?"] * len(_BROKER_RECONCILE_COLS))
    ord_ph = ", ".join(["?"] * len(_BROKER_ORDER_COLS))
    with connect() as conn:
        conn.execute("BEGIN TRANSACTION")
        conn.execute("DELETE FROM broker_reconciles WHERE run_id = ?", [run_id])
        conn.execute("DELETE FROM broker_orders WHERE run_id = ?", [run_id])
        conn.execute(
            f"INSERT INTO broker_reconciles ({', '.join(_BROKER_RECONCILE_COLS)}) VALUES ({rec_ph})",
            summary,
        )
        if order_rows:
            conn.executemany(
                f"INSERT INTO broker_orders ({', '.join(_BROKER_ORDER_COLS)}) VALUES ({ord_ph})",
                order_rows,
            )
        conn.execute("COMMIT")


# ── generic read path (used by the dashboard) ──────────────────────────────

def fetch_df(sql: str, params: Optional[list] = None, read_only: bool = True) -> pd.DataFrame:
    """Run a query and return a pandas DataFrame. Read-only by default."""
    with connect(read_only=read_only) as conn:
        return conn.execute(sql, params or []).df()


def fetch_filled_lmt_legs() -> list:
    """Filled **LMT** strategy legs from ``broker_orders`` — the real fills that
    represent how the system trades going forward (MKT is no longer the
    default; off-RTH always forces LMT). ENTRY/EXIT only; DRIFT_FLATTEN orphan
    cleanups are excluded (aggressive one-offs, not representative trade legs).
    Deduped to the most-complete fill per ``client_ref`` (a SUBMIT row and a
    later SETTLE_FILL/FILL_REFRESH for the same order, or duplicate twins, must
    not double-count). Each row: side, filled_qty, model_price, fill_price,
    commission. Used to calibrate the sim cost (``tracker.calibrate_sim_costs``)
    and to show the IBKR one-way cost. ``[]`` when the table/file isn't there
    yet (fresh DB, tests). MKT fills never appear by construction."""
    sql = ("SELECT client_ref, side, filled_qty, model_price, fill_price, commission "
           "FROM broker_orders "
           "WHERE upper(order_type) = 'LMT' AND filled_qty > 0 AND fill_price IS NOT NULL "
           "AND event <> 'DRIFT_FLATTEN' AND intent IN ('ENTRY', 'EXIT')")
    try:
        rows = fetch_df(sql).to_dict("records")
    except Exception:
        return []
    best: dict = {}
    for r in rows:
        ref = r.get("client_ref")
        if ref not in best or (r.get("filled_qty") or 0) > (best[ref].get("filled_qty") or 0):
            best[ref] = r
    return list(best.values())
