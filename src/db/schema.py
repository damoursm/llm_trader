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
"""

from __future__ import annotations

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
]


def ensure_schema(conn) -> None:
    """Create all tables if they do not yet exist (idempotent)."""
    for stmt in SCHEMA_STATEMENTS:
        conn.execute(stmt)
