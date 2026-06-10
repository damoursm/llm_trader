"""DuckDB schema — idempotent CREATE TABLE IF NOT EXISTS statements.

JSON-shaped columns (full trade dict, gate diagnostics, method maps) are stored as
VARCHAR holding JSON text. This keeps the schema portable (no JSON extension
dependency) while still allowing `json_extract` at query time when needed.

Tables
------
runs                — one row per pipeline invocation (regime, thresholds, LLM provider).
run_sources         — one row per data source per run: the "APIs used" record.
recommendations     — every top-N recommendation with its rationale + attribution.
trades              — the real signal-driven ledger (replaces cache/trades.json).
hypothetical_trades — the always-open paper book (replaces cache/hypothetical_trades.json).
broker_reconciles   — one row per broker sync: connectivity, counts, drift, errors.
broker_orders       — one event row per order submission / fill repair: model vs
                      fill price, cost-normalized slippage bps, commission. The
                      durable record the paper phase measures slippage/rejects from.
signals             — the full per-ticker signal cross-section of EVERY run (not
                      just the top-10 recommendations): one row per (run, ticker)
                      with all method scores, combined score, confidence. Joined
                      against forward returns from cache/ohlcv this is the panel
                      for information-coefficient analysis and threshold tuning —
                      news/options inputs can't be reconstructed historically, so
                      forward collection here is the only path to a
                      backtest-quality dataset.
"""

from __future__ import annotations

# Per-method score columns on the `signals` table. MUST mirror
# `src.performance.tracker._ALL_METHODS` — duplicated here (instead of imported)
# because tracker depends on src.db, so importing it back would be circular.
# tests/test_db_signals.py asserts the two stay in sync; when adding a method,
# add the column here too (new columns only apply to newly created DB files —
# an existing DB needs a one-time ALTER TABLE signals ADD COLUMN <m> DOUBLE).
SIGNAL_METHOD_COLUMNS = (
    "news", "sent_velocity", "tech", "insider", "put_call", "max_pain",
    "oi_skew", "vwap", "pattern", "momentum", "sector_momentum", "money_flow",
    "trend_strength", "pead", "iv_rank", "iv_expr", "coint", "cross_sectional",
)

SCHEMA_STATEMENTS = [
    """
    CREATE TABLE IF NOT EXISTS runs (
        run_id                  VARCHAR PRIMARY KEY,
        started_at              VARCHAR,
        finished_at             VARCHAR,
        elapsed_s               DOUBLE,
        market_mode             VARCHAR,
        macro_regime            VARCHAR,
        confidence_threshold    DOUBLE,
        allow_buys              BOOLEAN,
        universe_size           INTEGER,
        n_recommendations       INTEGER,
        n_actionable            INTEGER,
        llm_synthesis_provider  VARCHAR,
        llm_sentiment_provider  VARCHAR,
        gate_diag               VARCHAR
    );
    """,
    """
    CREATE TABLE IF NOT EXISTS run_sources (
        run_id        VARCHAR,
        source_label  VARCHAR,
        enabled       BOOLEAN,
        ok            BOOLEAN,
        error         VARCHAR,
        duration_s    DOUBLE
    );
    """,
    """
    CREATE TABLE IF NOT EXISTS recommendations (
        rec_id               VARCHAR PRIMARY KEY,
        run_id               VARCHAR,
        generated_at         VARCHAR,
        ticker               VARCHAR,
        type                 VARCHAR,
        direction            VARCHAR,
        action               VARCHAR,
        confidence           DOUBLE,
        time_horizon         VARCHAR,
        rationale            VARCHAR,
        actionable           BOOLEAN,
        dominant_method      VARCHAR,
        methods_agreeing     VARCHAR,
        contributing_scores  VARCHAR,
        llm_provider         VARCHAR
    );
    """,
    """
    CREATE TABLE IF NOT EXISTS trades (
        _seq                      BIGINT,
        trade_id                  VARCHAR,
        run_id                    VARCHAR,
        recommendation_id         VARCHAR,
        ticker                    VARCHAR,
        action                    VARCHAR,
        direction                 VARCHAR,
        confidence                DOUBLE,
        decision_datetime         VARCHAR,
        entry_datetime            VARCHAR,
        entry_date                VARCHAR,
        entry_price               DOUBLE,
        status                    VARCHAR,
        exit_date                 VARCHAR,
        exit_price                DOUBLE,
        return_pct                DOUBLE,
        position_size_multiplier  DOUBLE,
        sector_key                VARCHAR,
        dominant_method           VARCHAR,
        time_horizon              VARCHAR,
        data                      VARCHAR
    );
    """,
    """
    CREATE TABLE IF NOT EXISTS hypothetical_trades (
        _seq            BIGINT,
        trade_id        VARCHAR,
        ticker          VARCHAR,
        action          VARCHAR,
        direction       VARCHAR,
        entry_date      VARCHAR,
        entry_datetime  VARCHAR,
        entry_price     DOUBLE,
        status          VARCHAR,
        return_pct      DOUBLE,
        data            VARCHAR
    );
    """,
    """
    CREATE TABLE IF NOT EXISTS broker_reconciles (
        run_id            VARCHAR,
        created_at        VARCHAR,
        mode              VARCHAR,
        connected         BOOLEAN,
        ok                BOOLEAN,
        account_id        VARCHAR,
        account_equity    DOUBLE,
        account_currency  VARCHAR,
        entries_submitted INTEGER,
        exits_submitted   INTEGER,
        fills_repaired    INTEGER,
        rejects           INTEGER,
        n_drift           INTEGER,
        drift             VARCHAR,
        errors            VARCHAR
    );
    """,
    """
    CREATE TABLE IF NOT EXISTS broker_orders (
        run_id        VARCHAR,
        event         VARCHAR,
        intent        VARCHAR,
        ticker        VARCHAR,
        side          VARCHAR,
        order_type    VARCHAR,
        requested_qty INTEGER,
        filled_qty    INTEGER,
        model_price   DOUBLE,
        limit_price   DOUBLE,
        fill_price    DOUBLE,
        slippage_bps  DOUBLE,
        commission    DOUBLE,
        status        VARCHAR,
        ok            BOOLEAN,
        error         VARCHAR,
        order_id      VARCHAR,
        client_ref    VARCHAR,
        submitted_at  VARCHAR
    );
    """,
    f"""
    CREATE TABLE IF NOT EXISTS signals (
        run_id              VARCHAR,
        generated_at        VARCHAR,
        signal_date         VARCHAR,
        ticker              VARCHAR,
        type                VARCHAR,
        direction           VARCHAR,
        combined_score      DOUBLE,
        confidence          DOUBLE,
        n_methods_agreeing  INTEGER,
        dominant_method     VARCHAR,
        price               DOUBLE,
        {", ".join(f"{m} DOUBLE" for m in SIGNAL_METHOD_COLUMNS)},
        scores              VARCHAR
    );
    """,
]


def ensure_schema(conn) -> None:
    """Create all tables if they do not yet exist (idempotent)."""
    for stmt in SCHEMA_STATEMENTS:
        conn.execute(stmt)
