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
simulated_trades    — the LONG-format reshape of `signals`: one row per
                      (run, ticker, method) with a non-zero score, carrying the
                      method's implied side (BUY if score>0 else SELL) + entry
                      price. Lets every single method be evaluated as if it alone
                      decided the trade — its directional win rate over ALL scored
                      tickers, even when the synthesized recommendation went the
                      other way. Outcomes (forward returns at 30m/1d/3d/1w/2w/1m)
                      are computed on demand from cache/ohlcv (they are future
                      data, never knowable at write time).
"""

from __future__ import annotations

# Base per-method score columns on the `signals` table. MUST mirror
# `src.performance.tracker._ALL_METHODS` — duplicated here (instead of imported)
# because tracker depends on src.db, so importing it back would be circular.
# tests/test_db_signals.py asserts the two stay in sync; when adding a method,
# add the column here too (new columns only apply to newly created DB files —
# an existing DB needs a one-time ALTER TABLE signals ADD COLUMN <m> DOUBLE).
SIGNAL_BASE_METHOD_COLUMNS = (
    "news", "sent_velocity", "tech", "massive", "insider", "put_call", "max_pain",
    "oi_skew", "vwap", "pattern", "momentum", "sector_momentum", "market_momentum",
    "money_flow", "trend_strength", "pead", "iv_rank", "iv_expr", "coint", "cross_sectional",
    "ext_gap",
    # Broker-aware group (IBKR account / short-borrow). First method, 2026-06-28.
    "broker_advisor",
    # Massive fundamental + corp-action directional factors — promoted into the
    # trade-attribution set (2026-06-24) so they appear in the solo/eval Method-
    # Performance tables. Still grouped under the IC table's "Fundamentals" category
    # via SIGNAL_FUNDAMENTAL_COLUMNS (now a categorisation subset of this BASE set).
    "f_value", "f_quality", "f_growth", "f_short_squeeze", "f_split", "f_dividend",
    # Trend-predictability methods — signed Kaufman efficiency + ADX·DMI, split into
    # one-sided long/short methods (2026-07-04). Additive overlay on combined_score,
    # tracked per-method here so each side's IC is measured independently.
    "kaufman_long", "kaufman_short", "adx_long", "adx_short",
)

# Multi-timeframe technical columns — the 30-min + weekly variants of the 8
# OHLCV methods. Mirrors `src.signals.multi_timeframe.TECHNICAL_METHODS` ×
# the non-daily timeframes (the DAILY variant is the bare method column above).
# These are PANEL-ONLY (the IC dashboard); they are NOT in tracker._ALL_METHODS
# (the trade-attribution set). tests/test_db_signals.py guards the convention.
_MTF_METHODS = (
    "tech", "vwap", "momentum", "money_flow",
    "trend_strength", "iv_rank", "pattern", "sector_momentum",
)
_MTF_TIMEFRAMES = ("30m", "1w")
SIGNAL_TIMEFRAME_COLUMNS = tuple(
    f"{m}_{tf}" for tf in _MTF_TIMEFRAMES for m in _MTF_METHODS
)

# Fundamentals factor columns (Massive value/quality/growth/short-squeeze + corp-action
# split/dividend). As of 2026-06-24 they are PART OF SIGNAL_BASE_METHOD_COLUMNS (so they
# are trade-attributed in the solo/eval Method-Performance tables) AND fold into
# combined_score via the fundamental/corp-action overlays. This tuple is now a
# CATEGORISATION SUBSET of BASE — signal_panel groups them under the IC table's
# "Fundamentals" category, and the _ADD_COLUMNS loop below keeps the columns on old DBs.
SIGNAL_FUNDAMENTAL_COLUMNS = ("f_value", "f_quality", "f_growth", "f_short_squeeze",
                              "f_split", "f_dividend")

# Full set of method-score columns persisted to the `signals` panel. The fundamental
# factors now live IN the base set (trade-attributed); SIGNAL_FUNDAMENTAL_COLUMNS is a
# categorisation SUBSET of it (the IC table's "Fundamentals" grouping), no longer a
# separate appended group — so it is NOT added again here (that would duplicate columns).
SIGNAL_METHOD_COLUMNS = SIGNAL_BASE_METHOD_COLUMNS + SIGNAL_TIMEFRAME_COLUMNS

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
        duration_s    DOUBLE,
        n_items       INTEGER,
        empty         BOOLEAN
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
        llm_provider         VARCHAR,
        target_horizon       VARCHAR,
        horizon_net_edge_pct DOUBLE,
        shadow_target_horizon       VARCHAR,
        shadow_direction            VARCHAR,
        shadow_horizon_net_edge_pct DOUBLE,
        expected_move_pct           DOUBLE,
        market_aligned              VARCHAR,
        upside_score                DOUBLE
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
        pnl_daily         DOUBLE,
        pnl_unrealized    DOUBLE,
        pnl_realized      DOUBLE,
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
    """
    CREATE TABLE IF NOT EXISTS trade_reviews (
        run_id            VARCHAR,
        reviewed_at       VARCHAR,
        ticker            VARCHAR,
        position_id       VARCHAR,
        entry_datetime    VARCHAR,
        confidence        DOUBLE,
        action            VARCHAR,
        direction         VARCHAR,
        conf_floor        DOUBLE,
        entry_confidence  DOUBLE,
        entry_action      VARCHAR,
        price             DOUBLE,
        return_pct        DOUBLE,
        synthesis_model   VARCHAR,
        sentiment_model   VARCHAR
    );
    """,
    """
    CREATE TABLE IF NOT EXISTS simulated_trades (
        run_id        VARCHAR,
        generated_at  VARCHAR,
        signal_date   VARCHAR,
        ticker        VARCHAR,
        method        VARCHAR,
        score         DOUBLE,
        direction     VARCHAR,
        entry_price   DOUBLE
    );
    """,
    """
    CREATE TABLE IF NOT EXISTS exit_signals (
        run_id          VARCHAR,
        reviewed_at     VARCHAR,
        signal_date     VARCHAR,
        ticker          VARCHAR,
        position_id     VARCHAR,
        entry_direction VARCHAR,
        method          VARCHAR,
        score           DOUBLE,
        price           DOUBLE
    );
    """,
]


# One-time idempotent column additions for tables created before a field
# existed. DuckDB's ADD COLUMN IF NOT EXISTS is a no-op once the column is
# present, so this runs safely on every write connection.
_ADD_COLUMNS = (
    ("run_sources", "n_items", "INTEGER"),
    ("run_sources", "empty", "BOOLEAN"),
    # Massive server-side technical-indicator method column on an existing DB.
    ("signals", "massive", "DOUBLE"),
    # Market-relative momentum promoted from diagnostic into the weighted combine.
    ("signals", "market_momentum", "DOUBLE"),
    # Multi-timeframe technical columns on an existing signals table.
    *(("signals", col, "DOUBLE") for col in SIGNAL_TIMEFRAME_COLUMNS),
    # Fundamentals factor diagnostic columns on an existing signals table.
    *(("signals", col, "DOUBLE") for col in SIGNAL_FUNDAMENTAL_COLUMNS),
    # Broker-advisor method column on an existing signals table.
    ("signals", "broker_advisor", "DOUBLE"),
    # Trend-predictability methods (Kaufman/ADX, split long/short) on an existing DB.
    ("signals", "kaufman_long", "DOUBLE"),
    ("signals", "kaufman_short", "DOUBLE"),
    ("signals", "adx_long", "DOUBLE"),
    ("signals", "adx_short", "DOUBLE"),
    # Universe provenance (2026-07-03): which discovery source first surfaced
    # the ticker this run (watchlist / trending / screener / smart_money / …) —
    # the measurement behind per-source hit rates and, later, an adaptive
    # discovery budget. Trades carry the same stamp in their JSON.
    ("signals", "universe_source", "VARCHAR"),
    # IBKR account P&L snapshot (reqPnL) on an existing broker_reconciles table.
    ("broker_reconciles", "pnl_daily", "DOUBLE"),
    ("broker_reconciles", "pnl_unrealized", "DOUBLE"),
    ("broker_reconciles", "pnl_realized", "DOUBLE"),
    # Horizon synthesis on an existing recommendations table.
    ("recommendations", "target_horizon", "VARCHAR"),
    ("recommendations", "horizon_net_edge_pct", "DOUBLE"),
    # Direction-aware market-neutral shadow horizon.
    ("recommendations", "shadow_target_horizon", "VARCHAR"),
    ("recommendations", "shadow_direction", "VARCHAR"),
    ("recommendations", "shadow_horizon_net_edge_pct", "DOUBLE"),
    # Expected-move / market-aligned upside ranking.
    ("recommendations", "expected_move_pct", "DOUBLE"),
    ("recommendations", "market_aligned", "VARCHAR"),
    ("recommendations", "upside_score", "DOUBLE"),
)


def ensure_schema(conn) -> None:
    """Create all tables if they do not yet exist (idempotent)."""
    for stmt in SCHEMA_STATEMENTS:
        conn.execute(stmt)
    for table, col, coltype in _ADD_COLUMNS:
        conn.execute(f"ALTER TABLE {table} ADD COLUMN IF NOT EXISTS {col} {coltype}")
