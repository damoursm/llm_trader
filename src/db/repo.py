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
from src.db.schema import (SIGNAL_METHOD_COLUMNS, SIGNAL_NEWS_ATTENTION_COLUMNS,
                           SIGNAL_NEWS_QUIET_COLUMNS, SIGNAL_LOGPROB_COLUMNS,
                           SIGNAL_CONFIDENCE_COMPONENT_COLUMNS,
                           SIGNAL_ABS_SHADOW_COLUMNS,
                           SIGNAL_COMBINED_SIDE_COLUMNS,
                           SIGNAL_NEWS_EVENT_COLUMNS,
                           SIGNAL_FT_COLUMNS,
                           SIGNAL_LIQUIDITY_COLUMNS,
                           SIGNAL_MARKET_STATE_COLUMNS)


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


def update_trade(trade: dict) -> bool:
    """Rewrite ONE trade row in place, matched on its stable ``trade_id``.

    ``save_trades`` full-replaces the table (~227 ms for 282 trades), which is
    the right shape for a bulk ledger save but wasteful when a single broker leg
    changed — and the reconciler persists after EVERY order submission so a
    watchdog kill can't orphan a live order (see ``reconcile._persist_legs``), so
    a 10-order tick paid ~2.3 s re-writing unchanged rows.

    Returns False (caller should fall back to ``save_trades``) when the row
    isn't found, or when the ``trade_id`` is NOT unique — that hash is
    ticker+timestamp, so duplicate ledger rows are possible in principle and a
    targeted DELETE would take out both.
    """
    tid = _trade_id(trade)
    placeholders = ", ".join(["?"] * len(_TRADE_COLS))
    with connect() as conn:
        seqs = conn.execute("SELECT _seq FROM trades WHERE trade_id = ?", [tid]).fetchall()
        if len(seqs) != 1:
            return False                     # unknown or ambiguous — not our fast path
        row = _trade_row(seqs[0][0], trade)  # keep the row's existing ordinal
        conn.execute("BEGIN TRANSACTION")
        conn.execute("DELETE FROM trades WHERE trade_id = ?", [tid])
        conn.execute(
            f"INSERT INTO trades ({', '.join(_TRADE_COLS)}) VALUES ({placeholders})", row)
        conn.execute("COMMIT")
    return True


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
    "universe_source",
]
_SIGNAL_COLS = (_SIGNAL_BASE_COLS + list(SIGNAL_METHOD_COLUMNS)
               + list(SIGNAL_CONFIDENCE_COMPONENT_COLUMNS)
               + list(SIGNAL_COMBINED_SIDE_COLUMNS)
               + list(SIGNAL_ABS_SHADOW_COLUMNS)
               + list(SIGNAL_NEWS_ATTENTION_COLUMNS)
               + list(SIGNAL_NEWS_QUIET_COLUMNS)
               + list(SIGNAL_LOGPROB_COLUMNS)
               + [c for c, _t in SIGNAL_NEWS_EVENT_COLUMNS]
               + ["combine_source"]
               + list(SIGNAL_FT_COLUMNS)
               + list(SIGNAL_LIQUIDITY_COLUMNS)
               + list(SIGNAL_MARKET_STATE_COLUMNS)
               + ["scores"])


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
             _f(r.get("price")),
             r.get("universe_source")]
            + [_f(scores.get(m)) for m in SIGNAL_METHOD_COLUMNS]
            + [_f(r.get(c)) for c in SIGNAL_CONFIDENCE_COMPONENT_COLUMNS]
            + [_f(r.get(c)) for c in SIGNAL_COMBINED_SIDE_COLUMNS]
            + [_f(r.get(c)) for c in SIGNAL_ABS_SHADOW_COLUMNS]
            + [_f(r.get(c)) for c in SIGNAL_NEWS_ATTENTION_COLUMNS]
            + [_f(r.get(c)) for c in SIGNAL_NEWS_QUIET_COLUMNS]
            + [_f(r.get(c)) for c in SIGNAL_LOGPROB_COLUMNS]
            + [r.get("news_catalyst"), _f(r.get("news_raw_score")),
               r.get("news_digest_id")]
            + [r.get("combine_source")]
            + [_f(r.get(c)) for c in SIGNAL_FT_COLUMNS]
            + [_f(r.get(c)) for c in SIGNAL_LIQUIDITY_COLUMNS]
            + [_f(r.get(c)) for c in SIGNAL_MARKET_STATE_COLUMNS]
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


_ORDER_QUOTE_COLS = ["ticker", "submitted_at", "bid", "ask", "quote_age_s",
                     "source", "recovered_at"]


def insert_order_quotes(rows: List[dict]) -> None:
    """Upsert recovered books keyed (ticker, submitted_at). Written by the
    NBBO backfill for orders that predate the live capture; the live path never
    touches this table (see the schema comment on provenance)."""
    if not rows:
        return
    vals = [tuple(r.get(c) for c in _ORDER_QUOTE_COLS) for r in rows]
    keys = [(r.get("ticker"), r.get("submitted_at")) for r in rows]
    ph = ", ".join(["?"] * len(_ORDER_QUOTE_COLS))
    with connect() as conn:
        conn.execute("BEGIN TRANSACTION")
        conn.executemany(
            "DELETE FROM broker_order_quotes WHERE ticker = ? AND submitted_at = ?", keys)
        conn.executemany(
            f"INSERT INTO broker_order_quotes ({', '.join(_ORDER_QUOTE_COLS)}) "
            f"VALUES ({ph})", vals)
        conn.execute("COMMIT")


_BACKFILL_COLS = ["ticker", "signal_date", "catalyst", "headline_count",
                  "top_headline", "classifier_version", "classified_at"]


def insert_news_event_backfill(rows: List[dict]) -> None:
    """Upsert historical catalyst classifications keyed (ticker, signal_date).

    Written by `src/analysis/news_backfill.py` in small batches while the live
    scheduler may hold the write lock — one short transaction per call, riding
    `connect()`'s lock-retry. A re-classification of the same key replaces the
    old row (DELETE + INSERT), so re-runs with a newer classifier are safe."""
    if not rows:
        return
    vals = [tuple(r.get(c) for c in _BACKFILL_COLS) for r in rows]
    keys = [(r.get("ticker"), r.get("signal_date")) for r in rows]
    placeholders = ", ".join(["?"] * len(_BACKFILL_COLS))
    with connect() as conn:
        conn.execute("BEGIN TRANSACTION")
        conn.executemany(
            "DELETE FROM news_event_backfill WHERE ticker = ? AND signal_date = ?", keys)
        conn.executemany(
            f"INSERT INTO news_event_backfill ({', '.join(_BACKFILL_COLS)}) "
            f"VALUES ({placeholders})", vals)
        conn.execute("COMMIT")


# ── per-arm synthesis recommendations (the prompt-arm bake-off panel) ──────

_ARM_REC_COLS = [
    "run_id", "generated_at", "signal_date", "arm", "live",
    "ticker", "action", "direction", "confidence", "snap_price",
]


def insert_arm_recommendations(run_id: str, generated_at: str, signal_date: str,
                               rows: List[dict]) -> None:
    """Persist every synthesis ARM's call on every ticker for one run.

    The live arm's row is the recommendation that actually drove the run; the
    other arms' rows are SHADOW calls -- the same tickers, the same tick, the
    same context, asked under a different prompt and acted on by nobody. That
    pairing is the whole point: the 2026-07-22 bake-off established that
    unpaired arm comparisons here are window artifacts (Qwen's apparent lead and
    pro-thinking's apparent collapse were both pure calendar overlap), so an arm
    is only judgeable against another arm's call on the SAME ticker-day.

    Idempotent per ``run_id``. Each dict carries arm/live/ticker/action/
    direction/confidence/snap_price.
    """
    if not rows:
        return
    out = [(
        run_id, generated_at, signal_date,
        r.get("arm"), bool(r.get("live")),
        r.get("ticker"), r.get("action"), r.get("direction"),
        _f(r.get("confidence")), _f(r.get("snap_price")),
    ) for r in rows]
    placeholders = ", ".join(["?"] * len(_ARM_REC_COLS))
    with connect() as conn:
        conn.execute("BEGIN TRANSACTION")
        conn.execute("DELETE FROM arm_recommendations WHERE run_id = ?", [run_id])
        conn.executemany(
            f"INSERT INTO arm_recommendations ({', '.join(_ARM_REC_COLS)}) "
            f"VALUES ({placeholders})",
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


_SENTIMENT_SHADOW_COLS = (
    "run_id", "generated_at", "ticker", "digest_hash", "n_articles",
    "primary_engine", "primary_model", "primary_raw", "primary_score",
    "primary_catalyst", "shadow_engine", "shadow_model", "shadow_raw",
    "shadow_score", "shadow_catalyst", "shadow_latency_s", "digest_id",
)


def insert_sentiment_shadow(rows: List[dict]) -> None:
    """Persist paired sentiment verdicts: the engine that scored the run and the
    OTHER engine's read of the SAME article digest, one row per (run, ticker).

    Unlike the other per-run writers this does NOT delete by ``run_id`` — the
    shadow pass runs in the background and can outlive its own tick, so a batch
    may carry rows from the previous run alongside this one and a run-wide
    delete would erase what an earlier drain already wrote. Idempotency is per
    (run_id, ticker) instead."""
    if not rows:
        return
    out = [(
        r.get("run_id"), r.get("generated_at"), r.get("ticker"), r.get("digest_hash"),
        int(r.get("n_articles") or 0),
        r.get("primary_engine"), r.get("primary_model"),
        _f(r.get("primary_raw")), _f(r.get("primary_score")), r.get("primary_catalyst"),
        r.get("shadow_engine"), r.get("shadow_model"),
        _f(r.get("shadow_raw")), _f(r.get("shadow_score")), r.get("shadow_catalyst"),
        _f(r.get("shadow_latency_s")), r.get("digest_id"),
    ) for r in rows]
    keys = [(r.get("run_id"), r.get("ticker")) for r in rows]
    placeholders = ", ".join(["?"] * len(_SENTIMENT_SHADOW_COLS))
    with connect() as conn:
        conn.execute("BEGIN TRANSACTION")
        conn.executemany(
            "DELETE FROM sentiment_shadow WHERE run_id IS NOT DISTINCT FROM ? "
            "AND ticker IS NOT DISTINCT FROM ?", keys)
        conn.executemany(
            f"INSERT INTO sentiment_shadow ({', '.join(_SENTIMENT_SHADOW_COLS)}) "
            f"VALUES ({placeholders})", out)
        conn.execute("COMMIT")


_SENTIMENT_DIGEST_COLS = (
    "digest_id", "ticker", "run_id", "generated_at", "n_articles",
    "digest_text", "articles_json",
)


def insert_news_articles(run_id: str, generated_at: str, articles: List) -> dict:
    """Archive the tick's FULL merged article pool, URL-deduped.

    One row per unique article for all time: a repeat sighting bumps
    ``last_seen_at`` / ``n_sightings`` and leaves ``first_seen_at`` alone, which
    is the field a point-in-time replay has to key on — an article must not be
    visible to a tick that ran before anything had fetched it.

    Why this exists: `sentiment_digests` stores only the top-20 cut that reached
    a scorer, and only for tickers that were not abstained; `cache/news_*.json`
    stores only the yfinance/NewsAPI leg. Measured 2026-09-11, that is ~600 of a
    ~2,433-article pool, so ~75% of every tick was being discarded — which is
    exactly why `memory/news-backfill-fidelity-2026-09` concluded historical news
    could not be faithfully regenerated.

    Returns ``{"new": n, "seen": n}``. Fail-soft by the caller's `_safe`.
    """
    if not articles:
        return {"new": 0, "seen": 0}
    seen: dict = {}
    for a in articles:
        url = str(getattr(a, "url", "") or "").strip()
        title = str(getattr(a, "title", "") or "").strip()
        if not url and not title:
            continue
        key = hashlib.sha1((url or title).encode("utf-8", errors="replace")).hexdigest()
        if key in seen:
            continue
        pub = getattr(a, "published_at", None)
        tks = getattr(a, "tickers", None)
        seen[key] = (key, url, title[:1000],
                     str(getattr(a, "source", "") or "")[:200],
                     pub.isoformat() if hasattr(pub, "isoformat") else None,
                     str(getattr(a, "summary", "") or "")[:4000],
                     json.dumps(list(tks)) if tks else None,
                     generated_at, str(run_id), generated_at, 1)
    if not seen:
        return {"new": 0, "seen": 0}
    with connect() as conn:
        conn.execute("CREATE TEMP TABLE _na_in AS SELECT * FROM news_articles WHERE 1=0")
        conn.executemany(
            "INSERT INTO _na_in VALUES (?,?,?,?,?,?,?,?,?,?,?)", list(seen.values()))
        n_new = conn.execute(
            "SELECT count(*) FROM _na_in i WHERE NOT EXISTS "
            "(SELECT 1 FROM news_articles a WHERE a.url_hash = i.url_hash)").fetchone()[0]
        # repeat sighting: bump the tail, never the head
        conn.execute(
            "UPDATE news_articles SET last_seen_at = ?, n_sightings = n_sightings + 1 "
            "WHERE url_hash IN (SELECT url_hash FROM _na_in)", [generated_at])
        conn.execute(
            "INSERT INTO news_articles SELECT * FROM _na_in i WHERE NOT EXISTS "
            "(SELECT 1 FROM news_articles a WHERE a.url_hash = i.url_hash)")
        conn.execute("DROP TABLE _na_in")
    return {"new": int(n_new), "seen": len(seen) - int(n_new)}


def insert_news_article_feeds(run_id: str, generated_at: str, attribution: dict) -> dict:
    """Which FEEDS delivered each archived article (2026-09-25, the all-source
    news ingestion — `src/data/news_coverage.py`).

    `news_articles` keeps ONE row per article and the tick's pool keeps the
    first copy of a URL, so the second and third feeds that carried the same
    story were invisible — and a per-source feature group (`news_history`)
    needs exactly that. One row per (url_hash, feed): the first and last tick
    THAT feed delivered it, and the tickers that feed tagged it with when it
    first did. A past run's pool for one feed = the rows with
    ``first_seen_at <= run <= last_seen_at``, joined to `news_articles` for the
    text. ``attribution`` is `news_coverage.feed_attribution`'s
    ``{url_hash: {feed: [tickers]}}``. Returns ``{"new": n, "seen": n}``.
    """
    rows = []
    for h, legs in (attribution or {}).items():
        for feed, tks in (legs or {}).items():
            rows.append((str(h), str(feed), json.dumps(list(tks)) if tks else None,
                         generated_at, str(run_id), generated_at, 1))
    if not rows:
        return {"new": 0, "seen": 0}
    with connect() as conn:
        conn.execute("CREATE TEMP TABLE _naf_in AS SELECT * FROM news_article_feeds WHERE 1=0")
        conn.executemany("INSERT INTO _naf_in VALUES (?,?,?,?,?,?,?)", rows)
        n_new = conn.execute(
            "SELECT count(*) FROM _naf_in i WHERE NOT EXISTS (SELECT 1 FROM news_article_feeds f "
            "WHERE f.url_hash = i.url_hash AND f.feed = i.feed)").fetchone()[0]
        # repeat sighting: bump the tail, never the head
        conn.execute(
            "UPDATE news_article_feeds AS f SET last_seen_at = ?, n_sightings = f.n_sightings + 1 "
            "FROM _naf_in AS i WHERE f.url_hash = i.url_hash AND f.feed = i.feed", [generated_at])
        conn.execute(
            "INSERT INTO news_article_feeds SELECT * FROM _naf_in i WHERE NOT EXISTS "
            "(SELECT 1 FROM news_article_feeds f WHERE f.url_hash = i.url_hash AND f.feed = i.feed)")
        conn.execute("DROP TABLE _naf_in")
    return {"new": int(n_new), "seen": len(rows) - int(n_new)}


def insert_sentiment_digests(rows: List[dict]) -> None:
    """Persist the article digests the sentiment scorer actually saw, keyed by
    the ENGINE-FREE ``digest_id`` (2026-09-06).

    The verdict cache stores only a hash, so a past verdict's exact input
    could never be replayed; this store keeps the text so a catalyst label can
    be re-judged later on the SAME digest the model scored. Idempotent per
    ``digest_id``: the first writer wins (the text is identical by
    construction — the id is a hash of it), so a redundant row from the
    other engine or a cache hit on a later tick never duplicates."""
    if not rows:
        return
    out = [(
        r.get("digest_id"), r.get("ticker"), r.get("run_id"), r.get("generated_at"),
        int(r.get("n_articles") or 0), r.get("digest_text"), r.get("articles_json"),
        r.get("digest_id"),
    ) for r in rows if r.get("digest_id")]
    if not out:
        return
    placeholders = ", ".join(["?"] * len(_SENTIMENT_DIGEST_COLS))
    with connect() as conn:
        conn.execute("BEGIN TRANSACTION")
        conn.executemany(
            f"INSERT INTO sentiment_digests ({', '.join(_SENTIMENT_DIGEST_COLS)}) "
            f"SELECT {placeholders} WHERE NOT EXISTS "
            f"(SELECT 1 FROM sentiment_digests WHERE digest_id = ?)", out)
        conn.execute("COMMIT")


_CATALYST_REPAIR_COLS = (
    "digest_id", "run_id", "generated_at", "ticker", "engine", "model",
    "first_pass", "trigger", "specialist_model", "specialist_version", "votes",
    "n_calls", "final", "quality", "about_target", "latency_s", "error",
    "arm", "rationale",
)


def insert_catalyst_repairs(rows: List[dict]) -> None:
    """Persist catalyst-repair outcomes (specialist re-typing of a first-pass
    catalyst label), one row per ``(digest_id, engine, arm)`` — the engine is
    the one whose FIRST-PASS label was judged, so a DeepSeek-primary digest and
    its local shadow verdict each keep their own row, and the arm is the
    specialist configuration that judged it (``live`` for the tick-time pass,
    ``think`` for the offline thinking-on arm), so an offline arm can never
    overwrite the live row it is meant to be compared against.

    Like the sentiment shadow this is drained from a background pass that can
    outlive its tick, so there is no run-wide delete; idempotency is per
    ``(digest_id, engine, arm)`` (the LAST outcome wins — a later, resolved
    repair replaces an earlier failed one). A missing ``arm`` reads as
    ``live``."""
    if not rows:
        return
    out = [(
        r.get("digest_id"), r.get("run_id"), r.get("generated_at"), r.get("ticker"),
        r.get("engine"), r.get("model"), r.get("first_pass"), r.get("trigger"),
        r.get("specialist_model"), r.get("specialist_version"),
        (r.get("votes") if isinstance(r.get("votes"), str) else _json(r.get("votes"))),
        int(r.get("n_calls") or 0), r.get("final"), r.get("quality"),
        r.get("about_target"), _f(r.get("latency_s")), r.get("error"),
        str(r.get("arm") or "live"), r.get("rationale"),
    ) for r in rows if r.get("digest_id")]
    if not out:
        return
    keys = [(row[0], row[4], row[17]) for row in out]
    placeholders = ", ".join(["?"] * len(_CATALYST_REPAIR_COLS))
    with connect() as conn:
        conn.execute("BEGIN TRANSACTION")
        conn.executemany(
            "DELETE FROM catalyst_repairs WHERE digest_id = ? "
            "AND coalesce(engine, '') = coalesce(?, '') "
            "AND coalesce(arm, 'live') = ?", keys)
        conn.executemany(
            f"INSERT INTO catalyst_repairs ({', '.join(_CATALYST_REPAIR_COLS)}) "
            f"VALUES ({placeholders})", out)
        conn.execute("COMMIT")


_CLUSTER_ARM_COLS = (
    "run_id", "generated_at", "ticker", "digest_id", "engine", "model",
    "n_articles", "n_clusters", "cluster_scores", "cluster_sizes",
    "arm_raw", "arm_score", "primary_raw", "primary_score", "latency_s",
)


def insert_sentiment_cluster_arm(rows: List[dict]) -> None:
    """Per-ticker CLUSTER-ARM verdicts paired with the live single-call one.

    Idempotent per ``(run_id, ticker)`` and deliberately NOT run-wide DELETEd:
    like the shadow rows, a call still in flight lands on a later tick's drain
    carrying its own ``run_id``.
    """
    if not rows:
        return
    out = [tuple(r.get(c) for c in _CLUSTER_ARM_COLS) for r in rows if r.get("ticker")]
    if not out:
        return
    keys = [(r[0], r[2]) for r in out]
    placeholders = ", ".join(["?"] * len(_CLUSTER_ARM_COLS))
    with connect() as conn:
        conn.execute("BEGIN TRANSACTION")
        conn.executemany(
            "DELETE FROM sentiment_cluster_arm WHERE run_id = ? AND ticker = ?", keys)
        conn.executemany(
            f"INSERT INTO sentiment_cluster_arm ({', '.join(_CLUSTER_ARM_COLS)}) "
            f"VALUES ({placeholders})", out)
        conn.execute("COMMIT")


_NEWS_REPLAY_COLS = (
    "run_id", "ticker", "signal_date", "generated_at", "replayed_at",
    "replay_version", "engine", "n_articles", "n_relevant", "n_pool",
    "bundle_file", "pool_spec",
    "news", "news_raw_score", "sent_velocity", "news_shock", "news_bear_fresh",
    "catalyst_tilt", "news_unpriced", "news_unpriced_all", "news_catalyst",
    "news_recency_mass", "news_article_count",
    # 2026-09-12: `news_quiet` and `news_bull_fresh` were added to
    # `news_replay.NEWS_REPLAY_COLUMNS` and to the SCHEMA but not here — and
    # this tuple is what the INSERT actually names, so both were computed on
    # every row and silently dropped by the writer. The symptom was 0.0%
    # coverage on two methods that read ~1% and ~18% live; nothing errored.
    # A second copy of a column list is the failure this repo tests for
    # mechanically (`tests/test_db_signals.py` pins the same relationship for
    # `SIGNAL_METHOD_COLUMNS`), so a drift test now pins this one too.
    "news_quiet", "news_bull_fresh",
    # 2026-09-23: archive re-score provenance (certificate + scoring epoch).
    "news_digest_id", "news_epoch",
)


def insert_news_replay(rows: List[dict]) -> None:
    """Per-tick regenerated news features (`src/analysis/news_replay.py`).

    Its OWN table, deliberately not `signals_replay`: these values are produced
    from a POOL that is only partly recoverable (the RSS and Google legs were
    never persisted), so they are not interchangeable with a stored score until
    the fidelity report says so. Idempotent per (run_id, ticker) so a resumed
    or re-run tick replaces its rows rather than duplicating them.
    """
    if not rows:
        return
    out = [tuple(r.get(c) for c in _NEWS_REPLAY_COLS) for r in rows if r.get("ticker")]
    if not out:
        return
    # Keyed by POOL SHAPE too, so a faithful row and an experimental one for the
    # same ticker-run coexist instead of overwriting each other — the whole
    # point is comparing them.
    spec = _NEWS_REPLAY_COLS.index("pool_spec")
    keys = [(r[0], r[1], r[spec] or "faithful") for r in out]
    placeholders = ", ".join(["?"] * len(_NEWS_REPLAY_COLS))
    with connect() as conn:
        conn.execute("BEGIN TRANSACTION")
        conn.executemany(
            "DELETE FROM news_replay WHERE run_id = ? AND ticker = ? "
            "AND coalesce(pool_spec, 'faithful') = ?", keys)
        conn.executemany(
            f"INSERT INTO news_replay ({', '.join(_NEWS_REPLAY_COLS)}) "
            f"VALUES ({placeholders})", out)
        conn.execute("COMMIT")


_CATALYST_JUDGMENT_COLS = (
    "judgment_id", "judged_at", "source", "engine", "model", "ticker", "run_id",
    "digest_id", "model_catalyst", "verdict", "correct_catalyst", "entity_error",
    "reason", "rationale",
)


def insert_catalyst_judgments(rows: List[dict]) -> None:
    """Persist human/audit judgments of catalyst labels (the gold set the
    repair evaluation and the voter scaffold train against). Idempotent per
    ``judgment_id``."""
    if not rows:
        return
    out = [(
        r.get("judgment_id"), r.get("judged_at"), r.get("source"), r.get("engine"),
        r.get("model"), r.get("ticker"), r.get("run_id"), r.get("digest_id"),
        r.get("model_catalyst"), r.get("verdict"), r.get("correct_catalyst"),
        (None if r.get("entity_error") is None else bool(r.get("entity_error"))),
        r.get("reason"), r.get("rationale"),
    ) for r in rows if r.get("judgment_id")]
    if not out:
        return
    keys = [(row[0],) for row in out]
    placeholders = ", ".join(["?"] * len(_CATALYST_JUDGMENT_COLS))
    with connect() as conn:
        conn.execute("BEGIN TRANSACTION")
        conn.executemany("DELETE FROM catalyst_judgments WHERE judgment_id = ?", keys)
        conn.executemany(
            f"INSERT INTO catalyst_judgments ({', '.join(_CATALYST_JUDGMENT_COLS)}) "
            f"VALUES ({placeholders})", out)
        conn.execute("COMMIT")


_ENGINE_REC_COLS = (
    "run_id", "generated_at", "signal_date", "engine", "model", "prompt_variant",
    "live", "ticker", "action", "direction", "confidence", "time_horizon",
    "rationale", "snap_price", "rule_filled", "latency_s", "n_signals", "n_recs",
)


def insert_engine_recommendations(rows: List[dict]) -> None:
    """Persist every synthesis engine's per-ticker decision on the same signal
    cross-section, one row per (run, engine, prompt variant, ticker); ``live``
    marks the engine whose decision the pipeline acted on this run.

    Same discipline as `insert_sentiment_shadow`: NO run-wide delete. The
    shadow synthesis runs in the background and can outlive its own tick, so
    a drain may carry the previous run's rows alongside this run's live ones,
    and a run-wide delete would erase what an earlier drain already wrote.
    Idempotency is per (run_id, engine, prompt_variant, ticker) — the variant
    is part of the key because the decomposition arm runs the SAME engine on
    the other prompt (``deepseek:compact`` beside the live ``deepseek:full``),
    and keying on the engine alone would let the later drain erase the live
    row."""
    if not rows:
        return
    out = [(
        r.get("run_id"), r.get("generated_at"), r.get("signal_date"),
        r.get("engine"), r.get("model"), r.get("prompt_variant"),
        bool(r.get("live")), r.get("ticker"), r.get("action"), r.get("direction"),
        _f(r.get("confidence")), r.get("time_horizon"), r.get("rationale"),
        _f(r.get("snap_price")), bool(r.get("rule_filled")),
        _f(r.get("latency_s")),
        int(r["n_signals"]) if r.get("n_signals") is not None else None,
        int(r["n_recs"]) if r.get("n_recs") is not None else None,
    ) for r in rows]
    keys = [(r.get("run_id"), r.get("engine"), r.get("prompt_variant"), r.get("ticker"))
            for r in rows]
    placeholders = ", ".join(["?"] * len(_ENGINE_REC_COLS))
    with connect() as conn:
        conn.execute("BEGIN TRANSACTION")
        conn.executemany(
            "DELETE FROM engine_recommendations WHERE run_id IS NOT DISTINCT FROM ? "
            "AND engine IS NOT DISTINCT FROM ? AND prompt_variant IS NOT DISTINCT FROM ? "
            "AND ticker IS NOT DISTINCT FROM ?", keys)
        conn.executemany(
            f"INSERT INTO engine_recommendations ({', '.join(_ENGINE_REC_COLS)}) "
            f"VALUES ({placeholders})", out)
        conn.execute("COMMIT")


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
    "bid_at_submit", "ask_at_submit",
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
            _f(o.get("bid_at_submit")),
            _f(o.get("ask_at_submit")),
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

def fetch_df(sql: str, params: Optional[list] = None,
             read_only: Optional[bool] = None) -> pd.DataFrame:
    """Run a query and return a pandas DataFrame.

    ``read_only`` defaults to this PROCESS's role (``set_read_only``), not to a
    hardcoded True. DuckDB keeps one database instance per path per process and
    refuses a second handle with a different configuration, so a writer process
    that opened read-only here raced its own writes:

        ConnectionException: Can't open a connection to same database file with
        a different configuration than existing connections

    The pipeline is threaded (hold-review branch, shadow arms, EOD maintenance),
    so a read overlapping a write is routine — this fired intermittently, and the
    error is not a lock error, so it used to bypass the retry entirely. Reading
    through the writer's own read-write config removes the clash at the source; a
    read-write handle serves reads identically, and the connect-time schema check
    is memoised per process so it costs ~0.5 ms, not the 14.1 ms it once did. The
    dashboard (``set_read_only(True)``) is unaffected and stays read-only
    throughout. Pass the flag explicitly only to override the process role.
    """
    with connect(read_only=_READ_ONLY if read_only is None else read_only) as conn:
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
    sql = ("SELECT client_ref, run_id, ticker, side, filled_qty, model_price, fill_price, commission, "
           "submitted_at "
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
