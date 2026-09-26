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
sentiment_shadow    — paired sentiment verdicts: the engine that scored the run
                      and the OTHER engine's read of the SAME article digest.
news_articles       — THE FULL per-tick article pool, URL-deduped and stored
                      once per unique article. This is the archive a future
                      backfill reads: `sentiment_digests` keeps only the top-20
                      cut that reached a scorer, `cache/news_*.json` keeps only
                      the yfinance/NewsAPI leg (~600 of a ~2,433-article pool),
                      and everything else — RSS wires, FDA, per-ticker Google
                      News, Polygon/Benzinga, 8-K, Reddit, Quiver — is fetched
                      fresh every tick and was discarded. `first_seen_at` is
                      what makes a point-in-time reconstruction honest: an
                      article is visible to a replayed tick only from the run
                      that first saw it.
news_article_feeds  — which FEEDS delivered each archived article (since the
                      2026-09-25 all-source ingestion): one row per (url_hash,
                      feed) with that feed's first/last sighting and the
                      tickers it tagged, because the pool keeps only the first
                      copy of a URL — what rebuilds a single feed's pool for
                      the per-source news features.
sentiment_digests   — the article digest each sentiment verdict was scored on,
                      keyed by an ENGINE-FREE digest_id (ticker + article set),
                      so a repair or a human judgment attaches to the exact
                      text the model saw. A bounded store (retention days).
catalyst_repairs    — the catalyst-label REPAIR pass: per digest, the first-pass
                      class, why it was re-examined (trigger), the specialist's
                      votes and the resolved class + quality. The consumer
                      resolution (repair → live → backfill in
                      news_events.load_news_events, catalyst_tilt dropping the
                      unresolved events) is BUILT but held behind
                      enable_catalyst_repair_resolution until the pass clears
                      its measurement bar; until then this is pure accrual.
catalyst_judgments  — human verdicts on model catalyst labels (ok / ambiguous /
                      error + the correct class), the gold set the repair eval
                      grades against.
engine_recommendations — every synthesis engine's per-ticker decision on the
                      SAME signal cross-section, one row per (run, engine,
                      ticker); `live` marks the engine whose decision the
                      pipeline acted on. The paired dataset for the
                      local-vs-remote synthesis A/B.
"""

from __future__ import annotations

# Base per-method score columns on the `signals` table. MUST mirror
# `src.performance.tracker._ALL_METHODS` — duplicated here (instead of imported)
# because tracker depends on src.db, so importing it back would be circular.
# tests/test_db_signals.py asserts the two stay in sync; when adding a method,
# add the column here too (new columns only apply to newly created DB files —
# an existing DB needs a one-time ALTER TABLE signals ADD COLUMN <m> DOUBLE).
SIGNAL_BASE_METHOD_COLUMNS = (
    "news", "sent_velocity",
    # news_shock (2026-08-14, panel-first at weight 0 — signals/news_shock.py):
    # sign(news) × abnormal attention vs the ticker's own trailing baseline
    # (the SIGNAL_NEWS_ATTENTION_COLUMNS below are its forward-collected input).
    "news_shock",
    # news_bear_fresh (2026-08-15, panel-first at weight 0 —
    # signals/news_bear_fresh.py): bearish news × un-priced-tape guard.
    "news_bear_fresh",
    # news_bull_fresh (2026-09-10, panel-first at weight 0 —
    # signals/news_bull_fresh.py): the bull-side mirror.
    "news_bull_fresh",
    # catalyst_tilt (2026-08-15, panel-first at weight 0 —
    # signals/catalyst_tilt.py): news × learned per-(catalyst, side) orientation.
    "catalyst_tilt",
    # news_unpriced / news_unpriced_all (2026-09-07, panel-first at weight 0
    # — signals/news_priced_in.py): the unpriced share of the news move,
    # anchored on the news cluster's first article. Order mirrors
    # tracker._ALL_METHODS exactly (a drift test asserts tuple equality).
    "news_unpriced",
    "news_unpriced_all",
    # news_quiet (2026-09-09, WEIGHTED 0.10): the news read on a story whose
    # freshest cluster is >= `news_quiet_min_age_hours` old; abstains (0.0) on
    # a story still in flow.
    "news_quiet",
    "tech", "massive", "insider", "put_call", "max_pain",
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
    # Classic cross-sectional anomalies (2026-07-08) — 52-week-high proximity,
    # 12-1 skip-month momentum, short-term reversal. PANEL-FIRST at weight 0:
    # IC-measured + trade-attributed but NOT in combined_score/coherence yet
    # (signals/classic_anomalies.py).
    "hi52", "mom_12_1", "st_reversal",
    # Mean-reversion additions (2026-08-10, panel-first at weight 0): Connors
    # RSI(2) + daily candle-location reversal (signals/classic_anomalies.py) —
    # the de-correlated winners of the 20y full-history MR battery.
    "rsi2_rev", "dloc_rev",
    # Tier-2 panel-first methods (2026-07-08, weight 0): TTM squeeze
    # (ttm_squeeze.py), IV term-structure slope (iv_term_structure.py, from the
    # GEX chains), anchored VWAP (anchored_vwap.py, 52w high/low anchors).
    "squeeze", "iv_term", "avwap",
    # Tier-3 panel-first methods (2026-07-08, weight 0): residual momentum
    # (residual_momentum.py, beta-adjusted 12-1 vs SPY) and volume profile
    # (volume_profile.py, POC / 70% value area).
    "resid_mom", "vol_profile",
    # ML OHLCV model (2026-07-30, panel-first at weight 0 — signals/ml_model.py):
    # GBM on clean-trend/liquid names, 10-day market-relative. net = P(up)-P(down).
    "ml_ohlcv",
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
# News-attention inputs (2026-08-14, forward-collected): the recency-weighted
# article mass + fresh-article count persisted per (run, ticker). They are the
# BASELINE SERIES for the `news_shock` method — today's mass is judged against
# the ticker's own trailing daily median of this column — and a general
# attention covariate for panel analyses. NOT method scores (no direction).
SIGNAL_NEWS_ATTENTION_COLUMNS = ("news_article_count", "news_recency_mass")

# The age `news_quiet` judged on (hours since the freshest news cluster's first
# article). Not a score and not a `news_shock` baseline input, hence its own
# group: it is kept so the 48h threshold stays re-testable from the panel alone,
# without re-deriving clusters from a pool that may no longer be reconstructible
# (see memory/news-backfill-fidelity-2026-09.md).
SIGNAL_NEWS_QUIET_COLUMNS = ("news_quiet_age_h",)

# The logprob-derived continuous verdict (2026-09-10). NOT a method score — it
# is the same verdict as `news_raw_score` read more precisely — so it sits
# beside it rather than in SIGNAL_METHOD_COLUMNS, and nothing consumes it yet.
SIGNAL_LOGPROB_COLUMNS = ("news_expected_score", "news_argmax_score")

# Expected-liquidity forecast (2026-08-30): the run's ex-ante expected one-way
# execution deviation in bp (RTH basis) from src/performance/liquidity_forecast
# — CONTEXT, not a method score (predicts execution drift, not direction).
# NULL on rows written before the column existed or when the run had no view.
SIGNAL_LIQUIDITY_COLUMNS = ("exp_halfspread_bps",)

# Pre-combine market state (2026-09-25): ATR(14) / close, Bollinger width /
# middle band and the 5d/20d volume ratio from the tick's own
# `compute_technical_score` pass, plus the tape composite from
# `compute_tape_confirmation` — the four values the stackers read at serving
# time, under the SAME names `replay.replay_context` stores in `signals_replay`.
# CONTEXT, not method scores. Before 2026-09-25 only the EOD replay recorded
# them, so a run's rows lacked them until the replay reached that day; the tick
# now writes them itself, and where a replayed value exists
# `replay.restore_replayed` still overwrites the panel column with it (the two
# are one computation, so this only fills the gap). NULL on rows written before
# the column existed, when the technical pass is disabled, or when the tape
# check did not run. Short or missing history gives the neutral values the
# replay stores too (the technical module's EMPTY_RESULT, a NO_DATA tape at
# 0.0), not NULL — the stackers read those same values when they serve.
SIGNAL_MARKET_STATE_COLUMNS = ("atr_pct", "bb_width_pct", "vol_ratio", "tape_score")

# News-event dataset columns (2026-08-15): the sentiment LLM's dominant catalyst
# class (sentiment.NEWS_CATALYST_TYPES) + its RAW verdict before the evidence/
# diversity scalers — typed (name, sql_type) pairs because catalyst is VARCHAR.
# Joined against `fwd_ret_pivot` by `python -m src.analysis.news_events` to
# learn "what kind of news → what kind of move". Historical rows predate the
# capture (NULL); the backfill table below covers them.
SIGNAL_NEWS_EVENT_COLUMNS = (("news_catalyst", "VARCHAR"),
                             ("news_raw_score", "DOUBLE"),
                             # Engine-free id of the digest the verdict was scored
                             # on (2026-09-06) — the join key into
                             # sentiment_digests / catalyst_repairs.
                             ("news_digest_id", "VARCHAR"))

SIGNAL_FUNDAMENTAL_COLUMNS = ("f_value", "f_quality", "f_growth", "f_short_squeeze",
                              "f_split", "f_dividend")

# Full set of method-score columns persisted to the `signals` panel. The fundamental
# factors now live IN the base set (trade-attributed); SIGNAL_FUNDAMENTAL_COLUMNS is a
# categorisation SUBSET of it (the IC table's "Fundamentals" grouping), no longer a
# separate appended group — so it is NOT added again here (that would duplicate columns).
SIGNAL_METHOD_COLUMNS = SIGNAL_BASE_METHOD_COLUMNS + SIGNAL_TIMEFRAME_COLUMNS

# Methods `src/analysis/replay.py` can regenerate for a PAST date from the cached
# daily OHLCV alone — the columns of the `signals_replay` table. Membership is a
# claim about *faithfulness*, verified by `replay.validate()`, not merely about
# an OHLCV input:
#   * a method here reproduces its stored value ~92% exactly (median error
#     0.0000) whenever its scorer has not changed;
#   * `pattern` is deliberately EXCLUDED despite being OHLCV-driven — it blends a
#     live accuracy registry that grows with every trade, so replaying it today
#     uses a registry the original run never saw (measured: 44% exact);
#   * `sector_momentum` is excluded for the same class of reason: its second leg
#     is a sector ETF fetched at CURRENT time, not truncated to the signal date.
# Everything else (news, sentiment, the options family, insider, pead, massive…)
# needs a point-in-time feed nobody stored and can never be replayed.
REPLAYABLE_METHOD_COLUMNS = (
    "tech", "vwap", "momentum", "money_flow", "trend_strength", "iv_rank",
)

# Market-condition values the replay can also recover, because they are computed
# from the SAME cached OHLCV as the method scores above. `atr_pct`,
# `bb_width_pct` and `vol_ratio` come off the very `compute_technical_score`
# call that produces the `tech` score; `tape_score` is the cache-only tape
# composite. All four were being computed inside the replay and discarded.
#
# `movement_factor` is the one CONFIDENCE component that does not depend on the
# weights — it is `_movement_factor(atr_pct, bb_width_pct)` times a dealer-gamma
# modifier — so it can be restored rather than merely masked. Measured 91.8%
# exact (median error 0.0000). CAVEAT: GEX is options data and is NOT
# replayable; it enters as a 0.85/1.15 multiplier when dealer gamma is
# PINNED/AMPLIFIED and 1.0 otherwise, so a replayed value omits it. That is a
# BOUNDED, one-sided approximation (never worse than ±15% on one of six
# factors), which is why it is preferred over the alternative of NaN — but it is
# an approximation, unlike the raw inputs beside it, which are exact.
#
# The other five components all take `combined_score` (hence the weights) as an
# input, so recomputing them is a BACKTEST, not a recovery. They stay masked.
REPLAYABLE_CONTEXT_COLUMNS = (
    "movement_factor", "atr_pct", "bb_width_pct", "vol_ratio", "tape_score",
)

REPLAY_TABLE_COLUMNS = REPLAYABLE_METHOD_COLUMNS + REPLAYABLE_CONTEXT_COLUMNS

# Confidence-formula component columns (2026-07-21) — NOT method scores (so they are
# deliberately kept OUT of SIGNAL_METHOD_COLUMNS / tracker._ALL_METHODS: they are
# multiplicative confidence factors, not [-1,+1] directional views). Verbatim values
# from aggregator._score_ticker's `confidence = raw_confidence * coherence_factor *
# movement_factor * volume_factor * family_factor * tape_conf_factor` chain, persisted
# so src/analysis/confidence_components.py can isolate each factor's forward-return
# contribution without re-deriving it (and drifting from the live formula) later.
SIGNAL_CONFIDENCE_COMPONENT_COLUMNS = (
    "raw_confidence", "coherence_factor", "movement_factor",
    "volume_factor", "family_conf_factor", "tape_conf_factor",
    # 2026-08-14: the third pass's sector-alignment multiplier (1.10 aligned /
    # 0.75 contradicted). It had been applied to `confidence` since the pass was
    # written and persisted nowhere, so a sector-adjusted row could not multiply
    # back from its stored components by construction.
    "sector_conf_factor",
)

# Buy/sell split combine sides (2026-07-22) — the two camp-conviction aggregates
# whose DIFFERENCE is combined_score. NOT method scores (kept OUT of
# SIGNAL_METHOD_COLUMNS / tracker._ALL_METHODS — they are aggregates OF the
# methods, like combined_score itself); persisted per ticker so each side's
# forward IC is monitored on the dashboard (Signal IC → Buy side / Sell side).
SIGNAL_COMBINED_SIDE_COLUMNS = ("combined_buy_score", "combined_sell_score")

# Absolute-basis SHADOW combine (2026-08-14, rank directive follow-up): with
# `method_score_basis="rank"` live, the weighted combine over the RAW absolute
# scores is computed anyway (phase-2 arithmetic is free) and persisted per row,
# so rank-vs-absolute is settled by the live A/B rather than a 43-day offline
# experiment. When the basis is "absolute" these equal the live weighted combine
# (pre-ML-arm). NULL on rows written before the columns existed.
SIGNAL_ABS_SHADOW_COLUMNS = ("combined_score_abs", "combined_buy_score_abs",
                             "combined_sell_score_abs")

# Follow-through candidate columns (2026-08-25, src/signals/follow_through.py —
# panel-first accrual for the exit-as-entry mechanism): ft_score = the MOST
# NEGATIVE hypothetical-cohort exit conviction for the ticker this run; ft_dir =
# the follow-through (opposite) direction it implies (+1 long / −1 short / 0
# none); ft_selected = 1.0 when the tail+level+first-episode-day rules chose it
# as an entry candidate. NOT method scores (an entry mechanism, not a combine
# voter); NULL on rows written before the columns existed.
SIGNAL_FT_COLUMNS = ("ft_score", "ft_dir", "ft_selected")

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
        submitted_at  VARCHAR,
        -- Book at submit (2026-08-31): the two-sided quote _quote_for fetched
        -- for the LMT cap, persisted instead of discarded — lets slippage
        -- decompose into spread-crossing vs adverse drift. NULL when no fresh
        -- quote existed (feed down, spread-aware limits off, older rows).
        bid_at_submit DOUBLE,
        ask_at_submit DOUBLE
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
        {", ".join(f"{c} DOUBLE" for c in SIGNAL_CONFIDENCE_COMPONENT_COLUMNS)},
        {", ".join(f"{c} DOUBLE" for c in SIGNAL_COMBINED_SIDE_COLUMNS)},
        {", ".join(f"{c} DOUBLE" for c in SIGNAL_NEWS_ATTENTION_COLUMNS)},
        {", ".join(f"{c} DOUBLE" for c in SIGNAL_NEWS_QUIET_COLUMNS)},
        {", ".join(f"{c} DOUBLE" for c in SIGNAL_LOGPROB_COLUMNS)},
        {", ".join(f"{c} {t}" for c, t in SIGNAL_NEWS_EVENT_COLUMNS)},
        {", ".join(f"{c} DOUBLE" for c in SIGNAL_LIQUIDITY_COLUMNS)},
        {", ".join(f"{c} DOUBLE" for c in SIGNAL_MARKET_STATE_COLUMNS)},
        combine_source      VARCHAR,
        scores              VARCHAR
    );
    """,
    """
    -- Books RECOVERED after the fact for orders that predate the live
    -- bid_at_submit capture (2026-08-31), from Polygon /v3/quotes at each
    -- order's own submit instant. A SEPARATE table on purpose, mirroring
    -- news_event_backfill: `broker_orders.bid_at_submit` records what the run
    -- ACTUALLY SAW at submit, this records a later reconstruction — pooling
    -- them in one column would destroy that provenance. The spread
    -- calibration reads both (it only needs the true book); anything auditing
    -- what the order path had available must read broker_orders alone.
    CREATE TABLE IF NOT EXISTS broker_order_quotes (
        ticker        VARCHAR,
        submitted_at  VARCHAR,
        bid           DOUBLE,
        ask           DOUBLE,
        quote_age_s   DOUBLE,
        source        VARCHAR,
        recovered_at  VARCHAR
    );
    """,
    # Historical catalyst classifications for panel rows that predate the live
    # `news_catalyst` capture — written by `python -m src.analysis.news_backfill`
    # from re-fetched Polygon headlines. Keyed (ticker, signal_date); catalyst
    # NULL = attempted but no headline coverage for that day (so re-runs skip
    # it). The analysis layer COALESCEs live column first, this table second —
    # provenance stays separate: `signals` records what the run saw, this table
    # records a LATER classification of externally re-fetched headlines.
    """
    CREATE TABLE IF NOT EXISTS news_event_backfill (
        ticker              VARCHAR,
        signal_date         VARCHAR,
        catalyst            VARCHAR,
        headline_count      INTEGER,
        top_headline        VARCHAR,
        classifier_version  VARCHAR,
        classified_at       VARCHAR
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
    CREATE TABLE IF NOT EXISTS arm_recommendations (
        run_id        VARCHAR,
        generated_at  VARCHAR,
        signal_date   VARCHAR,
        arm           VARCHAR,   -- dual | blind | sighted
        live          BOOLEAN,   -- True = the arm that actually drove this run
        ticker        VARCHAR,
        action        VARCHAR,
        direction     VARCHAR,
        confidence    DOUBLE,
        snap_price    DOUBLE
    );
    """,
    f"""
    -- Historical ticker-days rescored by the CURRENT scorers over the cached
    -- OHLCV (src/analysis/replay.py). Lets a calibration fit values today's
    -- code produced instead of discarding superseded rows via the epoch mask.
    -- NEVER a substitute for `signals`, which records what actually happened;
    -- only the OHLCV-derived methods are replayable and `replayed_at` says
    -- which code version produced each row.
    CREATE TABLE IF NOT EXISTS signals_replay (
        signal_date   VARCHAR,
        ticker        VARCHAR,
        run_id        VARCHAR,
        generated_at  VARCHAR,
        replayed_at   VARCHAR,
        {", ".join(f"{m} DOUBLE" for m in REPLAY_TABLE_COLUMNS)}
    );
    """,
    """
    -- Implementation fingerprints (src/analysis/code_version.py). Append-only:
    -- one row per (name, fingerprint) the first time that fingerprint was seen.
    -- Drives the AUTOMATIC refactor — a changed fingerprint means the stored
    -- values for that method were produced by code that no longer exists, so
    -- they are regenerated (replayable) or masked (not). `first_seen_at` is what
    -- gives a non-replayable method its epoch without anyone hand-editing a
    -- registry. NOT a partition key: no analysis reads it.
    CREATE TABLE IF NOT EXISTS code_versions (
        name          VARCHAR,
        fingerprint   VARCHAR,
        first_seen_at VARCHAR
    );
    """,
    """
    -- Audit trail of automatic refactor runs (src/analysis/refactor.py).
    CREATE TABLE IF NOT EXISTS refactor_runs (
        started_at    VARCHAR,
        finished_at   VARCHAR,
        trigger       VARCHAR,   -- JSON {name: old->new} that caused the run
        steps         VARCHAR,   -- JSON [{step, status, detail}]
        ok            BOOLEAN
    );
    """,
    """
    -- Walk-forward weight calibration (src/analysis/walkforward.py). One row per
    -- calibration step: the weight state the system WOULD have had on that date,
    -- computed with a point-in-time cutoff so only strictly-earlier data was
    -- visible. This is what makes a backtest out-of-sample — weights at D cannot
    -- encode D's outcome — and is therefore the one weight source a calibration
    -- may legitimately consume, unlike the fixed-weight signals_backtest.
    CREATE TABLE IF NOT EXISTS weight_history (
        as_of         VARCHAR,
        computed_at   VARCHAR,
        n_active      INTEGER,
        degraded      BOOLEAN,   -- a fail-soft layer returned empty; not a market fact
        weights       VARCHAR,   -- JSON {method: effective weight, inversion signed}
        inverted      VARCHAR,   -- JSON [method]
        filtered      VARCHAR,   -- JSON [method] dropped by the hard filter
        buy_filtered  VARCHAR,
        sell_filtered VARCHAR,
        buy_mults     VARCHAR,   -- JSON {method: per-side weight multiplier}
        sell_mults    VARCHAR
    );
    """,
    """
    -- Walk-forward SHAPE history (src/signals/rank_shaping.py, 2026-08-21):
    -- the payoff-shaped rank curves as they would have been calibrated on each
    -- date, fitted under analysis_asof(as_of) so a curve can never see rows at
    -- or after its own date. The sibling of weight_history for the FITTED
    -- CONSUMPTION layer: without it a tier-2 backtest reads TODAY's curves
    -- over the very window they were fitted on (measured 2026-08-20: that
    -- flips the current arch's pivot IC from -0.041 to +0.049 — pure
    -- in-sample). One row per (as_of, method); a method absent on a date means
    -- "no curve existed" = identity, exactly the live fail-soft.
    CREATE TABLE IF NOT EXISTS shape_history (
        as_of        VARCHAR,
        method       VARCHAR,
        curve        VARCHAR,    -- JSON [10 floats], decile midpoints
        computed_at  VARCHAR
    );
    """,
    """
    -- Tier 2 BACKTEST (src/analysis/backtest.py): combined_score + confidence
    -- recomputed under a NAMED weight set. Deliberately a SEPARATE table from
    -- signals_replay, because these values depend on the weights and the
    -- weights are calibrated from the panel — feeding them back would fit the
    -- weights on values derived from themselves. `weight_set` is a hash of the
    -- weights used; rows under different hashes are not comparable.
    -- READ to evaluate a configuration; NEVER to fit one.
    CREATE TABLE IF NOT EXISTS signals_backtest (
        signal_date         VARCHAR,
        ticker              VARCHAR,
        generated_at        VARCHAR,
        weight_set          VARCHAR,
        computed_at         VARCHAR,
        combined_buy_score  DOUBLE,
        combined_score_abs  DOUBLE,
        combined_buy_score_abs DOUBLE,
        combined_sell_score_abs DOUBLE,
        combined_sell_score DOUBLE,
        combined_score      DOUBLE,
        raw_confidence      DOUBLE,
        coherence_factor    DOUBLE,
        movement_factor     DOUBLE,
        volume_factor       DOUBLE,
        family_conf_factor  DOUBLE,
        tape_conf_factor    DOUBLE,
        confidence          DOUBLE,
        direction           VARCHAR
    );
    """,
    """
    -- ML model registry (2026-07-30, signals/ml_model.py). One row per training
    -- of a trained-model method (e.g. ml_ohlcv). The artifact's OUTPUT changes on
    -- every retrain, so `train_max_date` is the as-of anchor for interpreting that
    -- model version's panel scores — the model-registry epoch the plan defers to
    -- promotion time. Audit trail + provenance; never read into a live decision.
    CREATE TABLE IF NOT EXISTS ml_models (
        trained_at      VARCHAR,
        method          VARCHAR,
        model_type      VARCHAR,
        horizon         INTEGER,
        basis           VARCHAR,
        n_train         INTEGER,
        train_max_date  VARCHAR,
        features        VARCHAR,   -- JSON [feature]
        config          VARCHAR    -- JSON training config
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
    CREATE TABLE IF NOT EXISTS sentiment_shadow (
        run_id           VARCHAR,
        generated_at     VARCHAR,
        ticker           VARCHAR,
        digest_hash      VARCHAR,
        n_articles       INTEGER,
        primary_engine   VARCHAR,
        primary_model    VARCHAR,
        primary_raw      DOUBLE,
        primary_score    DOUBLE,
        primary_catalyst VARCHAR,
        shadow_engine    VARCHAR,
        shadow_model     VARCHAR,
        shadow_raw       DOUBLE,
        shadow_score     DOUBLE,
        shadow_catalyst  VARCHAR,
        shadow_latency_s DOUBLE,
        digest_id        VARCHAR
    );
    """,
    """
    CREATE TABLE IF NOT EXISTS engine_recommendations (
        run_id         VARCHAR,
        generated_at   VARCHAR,
        signal_date    VARCHAR,
        engine         VARCHAR,
        model          VARCHAR,
        prompt_variant VARCHAR,
        live           BOOLEAN,
        ticker         VARCHAR,
        action         VARCHAR,
        direction      VARCHAR,
        confidence     DOUBLE,
        time_horizon   VARCHAR,
        rationale      VARCHAR,
        snap_price     DOUBLE,
        rule_filled    BOOLEAN,
        latency_s      DOUBLE,
        n_signals      INTEGER,
        n_recs         INTEGER
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
    """
    CREATE TABLE IF NOT EXISTS news_articles (
        url_hash      VARCHAR,
        url           VARCHAR,
        title         VARCHAR,
        source        VARCHAR,
        published_at  VARCHAR,
        summary       VARCHAR,
        tickers_json  VARCHAR,
        first_seen_at VARCHAR,
        first_run_id  VARCHAR,
        last_seen_at  VARCHAR,
        n_sightings   INTEGER
    );
    """,
    """
    CREATE TABLE IF NOT EXISTS news_article_feeds (
        url_hash      VARCHAR,
        feed          VARCHAR,
        tickers_json  VARCHAR,
        first_seen_at VARCHAR,
        first_run_id  VARCHAR,
        last_seen_at  VARCHAR,
        n_sightings   INTEGER
    );
    """,
    """
    CREATE TABLE IF NOT EXISTS sentiment_digests (
        digest_id     VARCHAR,
        ticker        VARCHAR,
        run_id        VARCHAR,
        generated_at  VARCHAR,
        n_articles    INTEGER,
        digest_text   VARCHAR,
        articles_json VARCHAR
    );
    """,
    """
    CREATE TABLE IF NOT EXISTS catalyst_repairs (
        digest_id          VARCHAR,
        run_id             VARCHAR,
        generated_at       VARCHAR,
        ticker             VARCHAR,
        engine             VARCHAR,
        model              VARCHAR,
        first_pass         VARCHAR,
        trigger            VARCHAR,
        specialist_model   VARCHAR,
        specialist_version VARCHAR,
        votes              VARCHAR,
        n_calls            INTEGER,
        final              VARCHAR,
        quality            VARCHAR,
        about_target       VARCHAR,
        latency_s          DOUBLE,
        error              VARCHAR,
        arm                VARCHAR,
        rationale          VARCHAR
    );
    """,
    """
    CREATE TABLE IF NOT EXISTS news_replay (
        run_id             VARCHAR,
        ticker             VARCHAR,
        signal_date        VARCHAR,
        generated_at       VARCHAR,
        replayed_at        VARCHAR,
        replay_version     VARCHAR,
        engine             VARCHAR,
        n_articles         INTEGER,
        n_relevant         INTEGER,
        n_pool             INTEGER,
        bundle_file        VARCHAR,
        pool_spec          VARCHAR,
        news               DOUBLE,
        news_raw_score     DOUBLE,
        sent_velocity      DOUBLE,
        news_shock         DOUBLE,
        news_quiet         DOUBLE,
        news_bull_fresh    DOUBLE,
        news_bear_fresh    DOUBLE,
        catalyst_tilt      DOUBLE,
        news_unpriced      DOUBLE,
        news_unpriced_all  DOUBLE,
        news_catalyst      VARCHAR,
        news_recency_mass  DOUBLE,
        news_article_count INTEGER,
        news_digest_id     VARCHAR,
        news_epoch         VARCHAR
    );
    """,
    """
    CREATE TABLE IF NOT EXISTS sentiment_cluster_arm (
        run_id          VARCHAR,
        generated_at    VARCHAR,
        ticker          VARCHAR,
        digest_id       VARCHAR,
        engine          VARCHAR,
        model           VARCHAR,
        n_articles      INTEGER,
        n_clusters      INTEGER,
        cluster_scores  VARCHAR,
        cluster_sizes   VARCHAR,
        arm_raw         DOUBLE,
        arm_score       DOUBLE,
        primary_raw     DOUBLE,
        primary_score   DOUBLE,
        latency_s       DOUBLE
    );
    """,
    """
    CREATE TABLE IF NOT EXISTS catalyst_judgments (
        judgment_id      VARCHAR,
        judged_at        VARCHAR,
        source           VARCHAR,
        engine           VARCHAR,
        model            VARCHAR,
        ticker           VARCHAR,
        run_id           VARCHAR,
        digest_id        VARCHAR,
        model_catalyst   VARCHAR,
        verdict          VARCHAR,
        correct_catalyst VARCHAR,
        entity_error     BOOLEAN,
        reason           VARCHAR,
        rationale        VARCHAR
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
    # Classic anomalies (hi52 / 12-1 momentum / short-term reversal) on an existing DB.
    ("signals", "hi52", "DOUBLE"),
    ("signals", "mom_12_1", "DOUBLE"),
    ("signals", "st_reversal", "DOUBLE"),
    # Mean-reversion additions (2026-08-10) on an existing DB.
    ("signals", "rsi2_rev", "DOUBLE"),
    ("signals", "dloc_rev", "DOUBLE"),
    # Tier-2 panel-first methods (squeeze / iv_term / avwap) on an existing DB.
    ("signals", "squeeze", "DOUBLE"),
    ("signals", "iv_term", "DOUBLE"),
    ("signals", "avwap", "DOUBLE"),
    # Tier-3 panel-first methods (resid_mom / vol_profile) on an existing DB.
    ("signals", "resid_mom", "DOUBLE"),
    ("signals", "vol_profile", "DOUBLE"),
    # ML OHLCV model (2026-07-30, panel-first at weight 0) on an existing DB.
    ("signals", "ml_ohlcv", "DOUBLE"),
    # Universe provenance (2026-07-03): which discovery source first surfaced
    # the ticker this run (watchlist / trending / screener / smart_money / …) —
    # the measurement behind per-source hit rates and, later, an adaptive
    # discovery budget. Trades carry the same stamp in their JSON.
    ("signals", "universe_source", "VARCHAR"),
    # Confidence-formula component factors (2026-07-21) on an existing signals table.
    *(("signals", col, "DOUBLE") for col in SIGNAL_CONFIDENCE_COMPONENT_COLUMNS),
    # Buy/sell split combine sides (2026-07-22) on an existing signals table.
    *(("signals", col, "DOUBLE") for col in SIGNAL_COMBINED_SIDE_COLUMNS),
    *(("signals", col, "DOUBLE") for col in SIGNAL_ABS_SHADOW_COLUMNS),
    # Which combine produced this row's buy/sell scores (2026-08-02): the ML
    # stackers or the hand-weighted camps. The A/B arm is per-RUN but the swap is
    # fail-soft PER SIDE (a missing artifact keeps that side weighted), so the
    # source is recorded per ticker per side: weighted | ml | ml_buy | ml_sell.
    # Lets every panel analysis segment ML-combine vs weighted-combine performance.
    ("signals", "combine_source", "VARCHAR"),
    # Follow-through candidate columns (2026-08-25) on an existing DB.
    *(("signals", col, "DOUBLE") for col in SIGNAL_FT_COLUMNS),
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
    # news_shock method column + its attention-input columns (2026-08-14).
    ("signals", "news_shock", "DOUBLE"),
    *(("signals", col, "DOUBLE") for col in SIGNAL_NEWS_ATTENTION_COLUMNS),
    ("signals", "news_bull_fresh", "DOUBLE"),
    # news_quiet (2026-09-09): the SCORE needs its own ALTER — the CREATE
    # expands SIGNAL_METHOD_COLUMNS but only ever runs on a NEW database.
    ("signals", "news_quiet", "DOUBLE"),
    *(("signals", col, "DOUBLE") for col in SIGNAL_NEWS_QUIET_COLUMNS),
    *(("signals", col, "DOUBLE") for col in SIGNAL_LOGPROB_COLUMNS),
    # News-event dataset columns (2026-08-15): catalyst class + raw LLM verdict.
    *(("signals", col, coltype) for col, coltype in SIGNAL_NEWS_EVENT_COLUMNS),
    # news_bear_fresh method column (2026-08-15, panel-first).
    ("signals", "news_bear_fresh", "DOUBLE"),
    # news_unpriced / news_unpriced_all method columns (2026-09-07).
    ("signals", "news_unpriced", "DOUBLE"),
    ("signals", "news_unpriced_all", "DOUBLE"),
    # catalyst_tilt method column (2026-08-15, panel-first).
    ("signals", "catalyst_tilt", "DOUBLE"),
    # Expected-liquidity forecast context column (2026-08-30) on an existing DB.
    *(("signals", col, "DOUBLE") for col in SIGNAL_LIQUIDITY_COLUMNS),
    # Pre-combine market state (2026-09-25) on an existing DB.
    *(("signals", col, "DOUBLE") for col in SIGNAL_MARKET_STATE_COLUMNS),
    # Book-at-submit columns (2026-08-31) on an existing broker_orders table.
    ("broker_orders", "bid_at_submit", "DOUBLE"),
    ("broker_orders", "ask_at_submit", "DOUBLE"),
    # Engine-free digest id on an existing sentiment_shadow table (2026-09-06).
    ("sentiment_shadow", "digest_id", "VARCHAR"),
    # Specialist arm ('live' | 'think') + the first-pass rationale the specialist read,
    # so an offline arm can replay the exact input (2026-09-07).
    # news_replay pool shape (2026-09-07): the table shipped without it, and a
    # CREATE TABLE IF NOT EXISTS cannot add a column to an existing DB.
    ("news_replay", "pool_spec", "VARCHAR"),
    ("news_replay", "n_relevant", "INTEGER"),
    # 2026-09-12: the two weighted news methods that shipped after the table did.
    ("news_replay", "news_quiet", "DOUBLE"),
    ("news_replay", "news_bull_fresh", "DOUBLE"),
    # 2026-09-23: the archive re-score's certificate (the digest the scorer read,
    # compared with the live row's) and the news epoch it was scored under — the
    # panel restores a row only while that epoch is still in force.
    ("news_replay", "news_digest_id", "VARCHAR"),
    ("news_replay", "news_epoch", "VARCHAR"),
    ("catalyst_repairs", "arm", "VARCHAR"),
    ("catalyst_repairs", "rationale", "VARCHAR"),
)


def ensure_schema(conn) -> None:
    """Create all tables if they do not yet exist (idempotent)."""
    for stmt in SCHEMA_STATEMENTS:
        conn.execute(stmt)
    for table, col, coltype in _ADD_COLUMNS:
        conn.execute(f"ALTER TABLE {table} ADD COLUMN IF NOT EXISTS {col} {coltype}")
